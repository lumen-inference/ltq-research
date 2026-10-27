// LTQ GPU decode prototype (batch 1, y = W x): is the format viable on GPU — memory-bound or decode-bound.
// Tile = 256 consecutive weights of a row; tail-biting trellis within the tile; per weight 1 code bit and K-1 non-code bits.
// Label: bit 0 = current code bit, bits 1.. = parities of the window of the 9 previous code bits by masks (as in tcq_viterbi).
// Variants: VAR 0 — joint table (10-bit window, non-code bits) -> fp16 in shared (1 access per weight, bank conflicts possible);
//           VAR 1 — label via popc + value table replicated across 32 banks (conflict-free).
// Baselines: fp16 and a 4-bit table format (16 entries, fp16 scale per 32 weights — 4.5 bits/weight).
//
//   nvcc -O3 -arch=native -o ltq_matvec_bench ltq_matvec_bench.cu && ./ltq_matvec_bench
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s: %s (%s:%d)\n", #x, cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

static __host__ __device__ inline uint32_t hash32(uint64_t i, uint32_t seed)
{
    uint64_t x = i * 0x9E3779B97F4A7C15ull + seed * 0xD1B54A32D192ED03ull;
    x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 29; x *= 0x94D049BB133111EBull; x ^= x >> 32;
    return (uint32_t)x;
}

__global__ void fill_u32(uint32_t* p, size_t n, uint32_t seed)
{
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = hash32(i, seed);
}
__global__ void fill_half(__half* p, size_t n, uint32_t seed, float lo, float hi)
{
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = __float2half(lo + (hi - lo) * (hash32(i, seed) >> 8) * (1.0f / 16777216.0f));
}

// ---- format parameters ----
// masks in window coordinates (bit 9 = current code bit, bit b < 9 = bit of position t - 9 + b)
static int rev9(int m) { int r = 0; for (int j = 0; j < 9; j++) if ((m >> j) & 1) r |= 1 << (8 - j); return r; }
struct Fmt { int K, nlb; int hm[3]; };  // nlb = number of label bits (1 + number of masks)
static Fmt fmt_of(int K)
{
    if (K == 2) return {2, 4, {359, 58, 299}};
    if (K == 3) return {3, 3, {359, 241, 0}};
    return {K, 2, {359, 0, 0}};
}
static int label_host(int win, const int* wm, int nlb)
{
    int l = (win >> 9) & 1;
    for (int k = 1; k < nlb; k++) l |= (__builtin_popcount(win & wm[k - 1]) & 1) << k;
    return l;
}

struct LtqDev {
    const uint32_t *code, *unc, *unc1, *hdr; const __half* scale;
};

template <int K>
__device__ __forceinline__ uint32_t lane_unc(const LtqDev& d, size_t tile, int l, uint32_t& u1)
{
    constexpr int UB = K - 1;
    if (UB == 1) return d.unc[tile * 8 + (l >> 2)] >> (8 * (l & 3));
    if (UB == 2) return d.unc[tile * 16 + (l >> 1)] >> (16 * (l & 1));
    if (UB == 3) { u1 = d.unc1[tile * 8 + (l >> 2)] >> (8 * (l & 3)); return d.unc[tile * 16 + (l >> 1)] >> (16 * (l & 1)); }
    return d.unc[tile * 32 + l];
}

template <int K>
__device__ __forceinline__ uint32_t unc_of(uint32_t uw, uint32_t u1, int i)
{
    constexpr int UB = K - 1;
    if (UB == 1) return (uw >> i) & 1;
    if (UB == 2) return (uw >> (2 * i)) & 3;
    if (UB == 3) return ((uw >> (2 * i)) & 3) | (((u1 >> i) & 1) << 2);
    return (uw >> (4 * i)) & 15;
}

template <int K, int VAR, int NLB, int R>
__global__ void __launch_bounds__(256) ltq_matvec(LtqDev d, const float* __restrict__ x, float* __restrict__ y, int M, int Kd,
                                                  const void* __restrict__ gtab, const float* __restrict__ gains, int wm1, int wm2, int wm3)
{
    constexpr int UB = K - 1;
    constexpr int NENT = (1 << NLB) << UB;  // entries in the value table
    extern __shared__ __align__(16) unsigned char smem[];
    float* sg = (float*)smem;  // 4 gains
    if (threadIdx.x < 4) sg[threadIdx.x] = gains[threadIdx.x];
    if (VAR == 0) {
        __half* st = (__half*)(smem + 16);
        const __half* g = (const __half*)gtab;
        for (int i = threadIdx.x; i < (1024 << UB); i += blockDim.x) st[i] = g[i];
    } else {
        float* st = (float*)(smem + 16);
        const float* g = (const float*)gtab;
        for (int i = threadIdx.x; i < NENT * 32; i += blockDim.x) st[i] = g[i >> 5];  // entry e for lane l is word e * 32 + l
    }
    __syncthreads();
    const int l = threadIdx.x & 31, o = 8 * (l & 3);
    const int warps = gridDim.x * (blockDim.x >> 5);
    const int nt = Kd >> 8;
    // R rows per warp: x is loaded once per tile, the warp has R independent chains
    for (int row0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * R; row0 < M; row0 += warps * R) {
        float accs[R];
#pragma unroll
        for (int r = 0; r < R; r++) accs[r] = 0.f;
#pragma unroll 2
        for (int j = 0; j < nt; j++) {
          const float4 xa = __ldg((const float4*)(x + (size_t)j * 256 + 8 * l));
          const float4 xb = __ldg((const float4*)(x + (size_t)j * 256 + 8 * l + 4));
#pragma unroll
          for (int r = 0; r < R; r++) {
            float& acc = accs[r];
            const size_t tile = (size_t)(row0 + r) * nt + j;
            const uint32_t myw = __ldg(d.code + tile * 8 + (l & 7));
            uint32_t u1 = 0;
            const uint32_t uw = lane_unc<K>(d, tile, l, u1);
            const uint32_t hdr = __ldg(d.hdr + tile);
            const float sc = __half2float(d.scale[tile]);
            const uint32_t cw = __shfl_sync(0xffffffffu, myw, l >> 2);
            const uint32_t pw = __shfl_sync(0xffffffffu, myw, ((l >> 2) + 7) & 7);
            const uint32_t win17 = (uint32_t)((((uint64_t)cw << 32) | pw) >> (23 + o));
            const float xs[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
            float part = 0.f;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const uint32_t w10 = (win17 >> i) & 0x3ff;
                const uint32_t u = unc_of<K>(uw, u1, i);
                float v;
                if (VAR == 0) {
                    v = __half2float(((const __half*)(smem + 16))[(w10 << UB) | u]);
                } else {
                    uint32_t lb = w10 >> 9;
                    if (NLB > 1) lb |= (__popc(w10 & wm1) & 1) << 1;
                    if (NLB > 2) lb |= (__popc(w10 & wm2) & 1) << 2;
                    if (NLB > 3) lb |= (__popc(w10 & wm3) & 1) << 3;
                    v = ((const float*)(smem + 16))[(((lb << UB) | u) << 5) | l];
                }
                part = fmaf(v, xs[i], part);
            }
            acc = fmaf(part, sg[(hdr >> (2 * (l >> 1))) & 3] * sc, acc);
          }
        }
#pragma unroll
        for (int r = 0; r < R; r++) {
            float acc = accs[r];
#pragma unroll
            for (int s = 16; s; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
            if (l == 0) y[row0 + r] = acc;
        }
    }
}

// 4-bit table format: 32 words per tile (8 nibbles per lane), fp16 scale per 32 weights
template <int R>
__global__ void __launch_bounds__(256) q4_matvec(const uint32_t* __restrict__ q, const __half* __restrict__ sc, const float* __restrict__ x,
                                                 float* __restrict__ y, int M, int Kd, const float* __restrict__ tab16)
{
    __shared__ float st[16 * 32];
    for (int i = threadIdx.x; i < 512; i += blockDim.x) st[i] = tab16[i >> 5];
    __syncthreads();
    const int l = threadIdx.x & 31, warps = gridDim.x * (blockDim.x >> 5), nt = Kd >> 8;
    for (int row0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * R; row0 < M; row0 += warps * R) {
        float accs[R];
#pragma unroll
        for (int r = 0; r < R; r++) accs[r] = 0.f;
#pragma unroll 2
        for (int j = 0; j < nt; j++) {
            const float4 xa = __ldg((const float4*)(x + (size_t)j * 256 + 8 * l));
            const float4 xb = __ldg((const float4*)(x + (size_t)j * 256 + 8 * l + 4));
            const float xs[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
            for (int r = 0; r < R; r++) {
                const size_t tile = (size_t)(row0 + r) * nt + j;
                const uint32_t w = __ldg(q + tile * 32 + l);
                const float s = __half2float(sc[tile * 8 + (l >> 2)]);
                float part = 0.f;
#pragma unroll
                for (int i = 0; i < 8; i++) part = fmaf(st[(((w >> (4 * i)) & 15) << 5) | l], xs[i], part);
                accs[r] = fmaf(part, s, accs[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < R; r++) {
            float acc = accs[r];
#pragma unroll
            for (int s = 16; s; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
            if (l == 0) y[row0 + r] = acc;
        }
    }
}

__global__ void __launch_bounds__(256) f16_matvec(const __half* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, int M, int Kd)
{
    const int l = threadIdx.x & 31, warps = gridDim.x * (blockDim.x >> 5), nt = Kd >> 8;
    for (int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); row < M; row += warps) {
        float acc = 0.f;
#pragma unroll 4
        for (int j = 0; j < nt; j++) {
            const uint4 w = __ldg((const uint4*)(W + (size_t)row * Kd + (size_t)j * 256 + 8 * l));
            const float4 xa = __ldg((const float4*)(x + (size_t)j * 256 + 8 * l));
            const float4 xb = __ldg((const float4*)(x + (size_t)j * 256 + 8 * l + 4));
            const __half2* h = (const __half2*)&w;
            float2 a = __half22float2(h[0]), b = __half22float2(h[1]), c = __half22float2(h[2]), e = __half22float2(h[3]);
            acc += a.x * xa.x + a.y * xa.y + b.x * xa.z + b.y * xa.w + c.x * xb.x + c.y * xb.y + e.x * xb.z + e.y * xb.w;
        }
#pragma unroll
        for (int s = 16; s; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
        if (l == 0) y[row] = acc;
    }
}

__global__ void read_bw(const uint4* __restrict__ p, size_t n, unsigned* out)
{
    uint32_t a = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = __ldg(p + i); a ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (a == 0x12345678u) *out = a;
}

// ---------------- host ----------------
static int g_sms;
template <typename F>
static float time_ms(F launch, int iters)
{
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int i = 0; i < 3; i++) launch();
    CK(cudaEventRecord(a));
    for (int i = 0; i < iters; i++) launch();
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); CK(cudaGetLastError());
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    return ms / iters;
}
static void fill(uint32_t* p, size_t n, uint32_t seed) { fill_u32<<<1024, 256>>>(p, n, seed); }

struct LtqBuf {
    uint32_t *code, *unc, *unc1, *hdr; __half* scale; size_t ntile; int K;
    size_t bytes() const { return ntile * (32 + 32 * (K - 1) + 4 + 2); }
};
static LtqBuf ltq_alloc(int K, size_t ntile, uint32_t seed)
{
    LtqBuf b{}; b.K = K; b.ntile = ntile;
    const int UB = K - 1, uw = UB == 3 ? 16 : 8 * UB;
    CK(cudaMalloc(&b.code, ntile * 32)); CK(cudaMalloc(&b.unc, ntile * uw * 4)); CK(cudaMalloc(&b.hdr, ntile * 4));
    CK(cudaMalloc(&b.scale, ntile * 2));
    fill(b.code, ntile * 8, seed); fill(b.unc, ntile * uw, seed + 1); fill(b.hdr, ntile, seed + 2);
    fill_half<<<1024, 256>>>(b.scale, ntile, seed + 3, 0.5f, 1.5f);
    if (UB == 3) { CK(cudaMalloc(&b.unc1, ntile * 32)); fill(b.unc1, ntile * 8, seed + 4); } else b.unc1 = nullptr;
    return b;
}
static void ltq_free(LtqBuf& b) { cudaFree(b.code); cudaFree(b.unc); cudaFree(b.hdr); cudaFree(b.scale); if (b.unc1) cudaFree(b.unc1); }

struct LtqTabs { void* comb; float* vt; float* gains; int wm[3]; int nlb; std::vector<float> h_vt, h_gains; std::vector<int> wmv; };
static LtqTabs ltq_tabs(int K)
{
    Fmt f = fmt_of(K); LtqTabs t{}; t.nlb = f.nlb;
    for (int k = 0; k < 3; k++) t.wm[k] = f.hm[k] ? rev9(f.hm[k]) : 0;
    const int UB = K - 1, nent = (1 << f.nlb) << UB;
    t.h_vt.resize(nent);
    for (int e = 0; e < nent; e++) t.h_vt[e] = __half2float(__float2half(((hash32(e, 77) >> 8) * (1.0f / 16777216.0f) - 0.5f) * 4.0f));
    std::vector<__half> comb(1024 << UB);
    for (int w = 0; w < 1024; w++)
        for (int u = 0; u < (1 << UB); u++) comb[(w << UB) | u] = __float2half(t.h_vt[(label_host(w, t.wm, f.nlb) << UB) | u]);
    t.h_gains = {0.7f, 0.9f, 1.1f, 1.3f};
    CK(cudaMalloc(&t.comb, comb.size() * 2)); CK(cudaMemcpy(t.comb, comb.data(), comb.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&t.vt, nent * 4)); CK(cudaMemcpy(t.vt, t.h_vt.data(), nent * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&t.gains, 16)); CK(cudaMemcpy(t.gains, t.h_gains.data(), 16, cudaMemcpyHostToDevice));
    return t;
}

template <int K, int VAR, int R = 1>
static void ltq_launch(const LtqBuf& b, const LtqTabs& t, const float* x, float* y, int M, int Kd, int grid)
{
    constexpr int UB = K - 1, NLB = K == 2 ? 4 : K == 3 ? 3 : 2, NENT = (1 << NLB) << UB;
    const size_t smem = 16 + (VAR == 0 ? (size_t)(1024 << UB) * 2 : (size_t)NENT * 32 * 4);
    LtqDev d{b.code, b.unc, b.unc1, b.hdr, b.scale};
    ltq_matvec<K, VAR, NLB, R><<<grid, 256, smem>>>(d, x, y, M, Kd, VAR == 0 ? t.comb : (const void*)t.vt, t.gains, t.wm[0], t.wm[1], t.wm[2]);
}
template <int K, int VAR, int R>
static int ltq_grid()
{
    constexpr int UB = K - 1, NLB = K == 2 ? 4 : K == 3 ? 3 : 2, NENT = (1 << NLB) << UB;
    const size_t smem = 16 + (VAR == 0 ? (size_t)(1024 << UB) * 2 : (size_t)NENT * 32 * 4);
    CK(cudaFuncSetAttribute(ltq_matvec<K, VAR, NLB, R>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    int nb; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, ltq_matvec<K, VAR, NLB, R>, 256, smem));
    return nb * g_sms;
}

// CPU reference for the first rows
static double ltq_ref_row(const LtqBuf& b, const LtqTabs& t, const std::vector<float>& x, int row, int Kd,
                          const std::vector<uint32_t>& code, const std::vector<uint32_t>& unc, const std::vector<uint32_t>& unc1,
                          const std::vector<uint32_t>& hdr, const std::vector<__half>& scale)
{
    const int K = b.K, UB = K - 1, nt = Kd / 256;
    double acc = 0;
    for (int j = 0; j < nt; j++) {
        const size_t tile = (size_t)row * nt + j;
        auto cbit = [&](int p) { p &= 255; return (code[tile * 8 + (p >> 5)] >> (p & 31)) & 1; };
        for (int p = 0; p < 256; p++) {
            int w = 0;
            for (int bb = 0; bb < 10; bb++) w |= cbit(p - 9 + bb) << bb;
            uint32_t u;
            if (UB == 1) u = (unc[tile * 8 + (p >> 5)] >> (p & 31)) & 1;
            else if (UB == 2) u = (unc[tile * 16 + (p >> 4)] >> (2 * (p & 15))) & 3;
            else if (UB == 3) u = ((unc[tile * 16 + (p >> 4)] >> (2 * (p & 15))) & 3) | (((unc1[tile * 8 + (p >> 5)] >> (p & 31)) & 1) << 2);
            else u = (unc[tile * 32 + (p >> 3)] >> (4 * (p & 7))) & 15;
            const float v = t.h_vt[(label_host(w, t.wm, t.nlb) << UB) | u];
            const float g = t.h_gains[(hdr[tile] >> (2 * (p >> 4))) & 3] * __half2float(scale[tile]);
            acc += (double)v * g * x[(size_t)j * 256 + p];
        }
    }
    return acc;
}

template <int K, int VAR>
static void check_ltq(const LtqTabs& t, const float* dx, const std::vector<float>& hx, int Kd)
{
    const int M = 64; LtqBuf b = ltq_alloc(K, (size_t)M * Kd / 256, 1000 + K);
    float* dy; CK(cudaMalloc(&dy, M * 4));
    ltq_launch<K, VAR, 4>(b, t, dx, dy, M, Kd, 8);  // path with R rows per warp
    CK(cudaDeviceSynchronize());
    std::vector<float> y(M); CK(cudaMemcpy(y.data(), dy, M * 4, cudaMemcpyDeviceToHost));
    const int UB = K - 1, uw = UB == 3 ? 16 : 8 * UB;
    std::vector<uint32_t> code(b.ntile * 8), unc(b.ntile * uw), unc1(UB == 3 ? b.ntile * 8 : 0), hdr(b.ntile);
    std::vector<__half> scale(b.ntile);
    CK(cudaMemcpy(code.data(), b.code, code.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(unc.data(), b.unc, unc.size() * 4, cudaMemcpyDeviceToHost));
    if (UB == 3) CK(cudaMemcpy(unc1.data(), b.unc1, unc1.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hdr.data(), b.hdr, hdr.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(scale.data(), b.scale, scale.size() * 2, cudaMemcpyDeviceToHost));
    double maxrel = 0;
    for (int r = 0; r < M; r++) {
        const double ref = ltq_ref_row(b, t, hx, r, Kd, code, unc, unc1, hdr, scale);
        maxrel = fmax(maxrel, fabs(y[r] - ref) / (fabs(ref) + 1e-3 * sqrt((double)Kd)));
    }
    printf("check K=%d VAR=%d: max rel. error %.2e %s\n", K, VAR, maxrel, maxrel < 1e-3 ? "OK" : "FAIL");
    ltq_free(b); cudaFree(dy);
}

static const int KD = 8192;
static float* g_x; static float* g_y;

template <int K, int VAR, int R>
static void bench_ltq(const LtqTabs& t, size_t target_bytes, const char* mode)
{
    const size_t bpt = 32 + 32 * (K - 1) + 6;
    const int M = (int)(target_bytes / (bpt * (KD / 256))) & ~7;
    LtqBuf b = ltq_alloc(K, (size_t)M * KD / 256, 7 + K);
    const int grid = ltq_grid<K, VAR, R>();
    const int iters = target_bytes > (256u << 20) ? 20 : 200;
    const float ms = time_ms([&] { ltq_launch<K, VAR, R>(b, t, g_x, g_y, M, KD, grid); }, iters);
    const double wts = (double)M * KD;
    printf("  %-5s R=%d LTQ K=%d %-12s %.3f bits/weight | %7.1f GB/s | %7.1f G weights/s\n", mode, R, K, VAR == 0 ? "shared table" : "popc+dup",
           bpt * 8.0 / 256, b.bytes() / ms / 1e6, wts / ms / 1e6);
    ltq_free(b);
}

template <int R>
static void bench_q4(size_t target_bytes, const char* mode, const float* tab16)
{
    const int M = (int)(target_bytes / (144 * (KD / 256))) & ~7;
    const size_t nt = (size_t)M * KD / 256;
    uint32_t* q; __half* sc; CK(cudaMalloc(&q, nt * 128)); CK(cudaMalloc(&sc, nt * 16));
    fill(q, nt * 32, 5); fill_half<<<1024, 256>>>(sc, nt * 8, 6, 0.5f, 1.5f);
    int nb; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, q4_matvec<R>, 256, 0));
    const int iters = target_bytes > (256u << 20) ? 20 : 200;
    const float ms = time_ms([&] { q4_matvec<R><<<nb * g_sms, 256>>>(q, sc, g_x, g_y, M, KD, tab16); }, iters);
    printf("  %-5s R=%d Q4 (table 16)        4.500 bits/weight | %7.1f GB/s | %7.1f G weights/s\n", mode, R, nt * 144 / ms / 1e6, (double)M * KD / ms / 1e6);
    cudaFree(q); cudaFree(sc);
}

static void bench_f16(size_t target_bytes, const char* mode)
{
    const int M = (int)(target_bytes / (2 * (size_t)KD));
    __half* W; CK(cudaMalloc(&W, (size_t)M * KD * 2));
    fill_half<<<1024, 256>>>(W, (size_t)M * KD, 9, -1.f, 1.f);
    int nb; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, f16_matvec, 256, 0));
    const int iters = target_bytes > (256u << 20) ? 20 : 200;
    const float ms = time_ms([&] { f16_matvec<<<nb * g_sms, 256>>>(W, g_x, g_y, M, KD); }, iters);
    printf("  %-5s fp16                16.000 bits/weight | %7.1f GB/s | %7.1f G weights/s\n", mode, (double)M * KD * 2 / ms / 1e6, (double)M * KD / ms / 1e6);
    cudaFree(W);
}

int main()
{
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0)); g_sms = p.multiProcessorCount;
    int clk = 0, mclk = 0, bus = 0;
    cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
    cudaDeviceGetAttribute(&mclk, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, 0);
    printf("%s: %d SM, sm_%d%d, %.0f MHz, L2 %d MB, shared/SM %zu KB\n", p.name, g_sms, p.major, p.minor, clk / 1e3,
           p.l2CacheSize >> 20, p.sharedMemPerMultiprocessor >> 10);

    std::vector<float> hx(KD);
    for (int i = 0; i < KD; i++) hx[i] = ((hash32(i, 3) >> 8) * (1.0f / 16777216.0f) - 0.5f);
    CK(cudaMalloc(&g_x, KD * 4)); CK(cudaMemcpy(g_x, hx.data(), KD * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&g_y, (64u << 20)));
    float tab16h[16]; for (int i = 0; i < 16; i++) tab16h[i] = (i - 7.5f) / 8.f;
    float* tab16; CK(cudaMalloc(&tab16, 64)); CK(cudaMemcpy(tab16, tab16h, 64, cudaMemcpyHostToDevice));

    LtqTabs t2 = ltq_tabs(2), t3 = ltq_tabs(3), t4 = ltq_tabs(4), t5 = ltq_tabs(5);
    check_ltq<2, 0>(t2, g_x, hx, KD); check_ltq<2, 1>(t2, g_x, hx, KD);
    check_ltq<3, 0>(t3, g_x, hx, KD); check_ltq<3, 1>(t3, g_x, hx, KD);
    check_ltq<4, 0>(t4, g_x, hx, KD); check_ltq<4, 1>(t4, g_x, hx, KD);
    check_ltq<5, 0>(t5, g_x, hx, KD); check_ltq<5, 1>(t5, g_x, hx, KD);

    {   // memory read ceiling
        const size_t n = (2048ull << 20) / 16; uint4* buf; unsigned* o; CK(cudaMalloc(&buf, n * 16)); CK(cudaMalloc(&o, 4));
        fill((uint32_t*)buf, n * 4, 11);
        const float ms = time_ms([&] { read_bw<<<g_sms * 8, 256>>>(buf, n, o); }, 20);
        printf("memory read: %.1f GB/s (theor. %.1f)\n", n * 16 / ms / 1e6, 2.0 * mclk * 1e3 * (bus / 8) / 1e9);
        cudaFree(buf); cudaFree(o);
    }
    const size_t DRAM = 2048ull << 20, L2 = (size_t)(p.l2CacheSize * 0.4);
    for (int pass = 0; pass < 2; pass++) {
        const size_t tb = pass == 0 ? DRAM : L2;
        const char* mode = pass == 0 ? "DRAM" : "L2";
        printf("\n== %s: weights %.0f MB ==\n", pass == 0 ? "from memory (memory-bound?)" : "in L2 (decode ceiling)", tb / 1048576.0);
        bench_f16(tb, mode);
        bench_q4<1>(tb, mode, tab16); bench_q4<2>(tb, mode, tab16); bench_q4<4>(tb, mode, tab16);
        bench_ltq<2, 0, 1>(t2, tb, mode); bench_ltq<2, 0, 2>(t2, tb, mode); bench_ltq<2, 0, 4>(t2, tb, mode);
        bench_ltq<2, 1, 2>(t2, tb, mode); bench_ltq<2, 1, 4>(t2, tb, mode);
        bench_ltq<3, 0, 2>(t3, tb, mode); bench_ltq<3, 0, 4>(t3, tb, mode);
        bench_ltq<3, 1, 2>(t3, tb, mode); bench_ltq<3, 1, 4>(t3, tb, mode);
        bench_ltq<4, 0, 2>(t4, tb, mode); bench_ltq<4, 0, 4>(t4, tb, mode);
        bench_ltq<4, 1, 2>(t4, tb, mode); bench_ltq<4, 1, 4>(t4, tb, mode);
        bench_ltq<5, 0, 2>(t5, tb, mode); bench_ltq<5, 0, 4>(t5, tb, mode);
        bench_ltq<5, 1, 2>(t5, tb, mode); bench_ltq<5, 1, 4>(t5, tb, mode);
    }
    return 0;
}
