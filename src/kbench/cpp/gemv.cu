// Kernel-language comparison: CUDA C++ vs cuda-oxide (Rust). The LTQ GEMV kernel is verbatim from src/cuda/ltq_matvec_bench.cu;
// data, tables, launch configurations and timing are the same as in ../oxide_gemv (the cuda-oxide port).
// Prints results as JSON lines, writes the outputs for cross-checking to out/y_cpp_K{K}_V{VAR}.bin.
//
//   nvcc -O3 -arch=sm_120 -Xcompiler -static-libstdc++ -o gemv_cpp gemv.cu && ./gemv_cpp
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
// masks in window coordinates (bit 9 is the current code bit, bit b < 9 is the bit at position t - 9 + b)
static int rev9(int m) { int r = 0; for (int j = 0; j < 9; j++) if ((m >> j) & 1) r |= 1 << (8 - j); return r; }
struct Fmt { int K, nlb; int hm[3]; };  // nlb is the number of label bits (1 + number of masks)
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
__global__ void read_bw(const uint4* __restrict__ p, size_t n, unsigned* out)
{
    uint32_t a = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = __ldg(p + i); a ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (a == 0x12345678u) *out = a;
}

// ---------------- host (from the prototype: buffers, tables, launch, reference) ----------------
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

// ---------------- common comparison harness (mirrored in ../oxide_gemv/src/main.rs) ----------------
#include <chrono>
#include <sys/stat.h>

__global__ void empty_kernel(int* p) { if (p && threadIdx.x == 1000) *p = 1; }

static const int KD = 8192;
static float* g_x; static float* g_y;

template <int K, int VAR>
static void check(const LtqTabs& t, const std::vector<float>& hx)
{
    const int M = 64; LtqBuf b = ltq_alloc(K, (size_t)M * KD / 256, 1000 + K);
    float* dy; CK(cudaMalloc(&dy, M * 4));
    ltq_launch<K, VAR, 4>(b, t, g_x, dy, M, KD, 8);
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
        const double ref = ltq_ref_row(b, t, hx, r, KD, code, unc, unc1, hdr, scale);
        maxrel = fmax(maxrel, fabs(y[r] - ref) / (fabs(ref) + 1e-3 * sqrt((double)KD)));
    }
    char path[64]; snprintf(path, sizeof path, "out/y_cpp_K%d_V%d.bin", K, VAR);
    FILE* f = fopen(path, "wb"); fwrite(y.data(), 4, M, f); fclose(f);
    printf("{\"lang\":\"cpp\",\"kind\":\"check\",\"K\":%d,\"VAR\":%d,\"maxrel_vs_cpu\":%.3e}\n", K, VAR, maxrel);
    ltq_free(b); cudaFree(dy);
}

template <int K, int VAR, int R>
static void bench(const LtqTabs& t, size_t target_bytes, const char* mode)
{
    constexpr int NLB = K == 2 ? 4 : K == 3 ? 3 : 2;
    const size_t bpt = 32 + 32 * (K - 1) + 6;
    const int M = (int)(target_bytes / (bpt * (KD / 256))) & ~7;
    LtqBuf b = ltq_alloc(K, (size_t)M * KD / 256, 7 + K);
    const int grid = ltq_grid<K, VAR, R>();
    cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, ltq_matvec<K, VAR, NLB, R>));
    const int iters = target_bytes > (256u << 20) ? 20 : 200;
    const float ms = time_ms([&] { ltq_launch<K, VAR, R>(b, t, g_x, g_y, M, KD, grid); }, iters);
    printf("{\"lang\":\"cpp\",\"kind\":\"gemv\",\"mode\":\"%s\",\"K\":%d,\"VAR\":%d,\"R\":%d,\"grid\":%d,\"regs\":%d,\"local_bytes\":%zu,"
           "\"ms\":%.5f,\"GBps\":%.1f,\"Gw_s\":%.1f}\n", mode, K, VAR, R, grid, fa.numRegs, fa.localSizeBytes, ms,
           b.bytes() / ms / 1e6, (double)M * KD / ms / 1e6);
    fflush(stdout);
    ltq_free(b);
}

template <int K>
static void bench_k(const LtqTabs& t, size_t tb, const char* mode)
{
    bench<K, 0, 2>(t, tb, mode); bench<K, 0, 4>(t, tb, mode); bench<K, 1, 2>(t, tb, mode); bench<K, 1, 4>(t, tb, mode);
}

int main()
{
    mkdir("out", 0755);
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0)); g_sms = p.multiProcessorCount;
    printf("{\"lang\":\"cpp\",\"kind\":\"device\",\"name\":\"%s\",\"sms\":%d,\"l2_mb\":%d}\n", p.name, g_sms, p.l2CacheSize >> 20);

    std::vector<float> hx(KD);
    for (int i = 0; i < KD; i++) hx[i] = ((hash32(i, 3) >> 8) * (1.0f / 16777216.0f) - 0.5f);
    CK(cudaMalloc(&g_x, KD * 4)); CK(cudaMemcpy(g_x, hx.data(), KD * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&g_y, (64u << 20)));
    LtqTabs t2 = ltq_tabs(2), t3 = ltq_tabs(3), t4 = ltq_tabs(4), t5 = ltq_tabs(5);

    check<2, 0>(t2, hx); check<2, 1>(t2, hx); check<3, 0>(t3, hx); check<3, 1>(t3, hx);
    check<4, 0>(t4, hx); check<4, 1>(t4, hx); check<5, 0>(t5, hx); check<5, 1>(t5, hx);

    {   // memory read ceiling
        const size_t n = (2048ull << 20) / 16; uint4* buf; unsigned* o; CK(cudaMalloc(&buf, n * 16)); CK(cudaMalloc(&o, 4));
        fill((uint32_t*)buf, n * 4, 11);
        const float ms = time_ms([&] { read_bw<<<g_sms * 8, 256>>>(buf, n, o); }, 20);
        printf("{\"lang\":\"cpp\",\"kind\":\"read_bw\",\"GBps\":%.1f}\n", n * 16 / ms / 1e6);
        cudaFree(buf); cudaFree(o);
    }
    {   // launch overhead: enqueueing and executing N empty kernels
        const int N = 20000;
        for (int i = 0; i < 100; i++) empty_kernel<<<1, 32>>>(nullptr);
        CK(cudaDeviceSynchronize());
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; i++) empty_kernel<<<1, 32>>>(nullptr);
        auto t1 = std::chrono::steady_clock::now();
        CK(cudaDeviceSynchronize());
        auto t2 = std::chrono::steady_clock::now();
        printf("{\"lang\":\"cpp\",\"kind\":\"launch\",\"enqueue_us\":%.3f,\"total_us\":%.3f}\n",
               std::chrono::duration<double, std::micro>(t1 - t0).count() / N, std::chrono::duration<double, std::micro>(t2 - t0).count() / N);
    }
    const size_t DRAM = 2048ull << 20, L2 = (size_t)(p.l2CacheSize * 0.4);
    for (int pass = 0; pass < 2; pass++) {
        const size_t tb = pass == 0 ? DRAM : L2;
        const char* mode = pass == 0 ? "DRAM" : "L2";
        bench_k<2>(t2, tb, mode); bench_k<3>(t3, tb, mode); bench_k<4>(t4, tb, mode); bench_k<5>(t5, tb, mode);
    }
    return 0;
}
