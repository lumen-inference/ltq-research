// Full-model CUDA benchmark of .liqk v2 kernels on RTX 5070 Ti (bypassing the 48 MB L2 cache by allocating all 16/28 layers)
// Tests:
//   - Fused Online suh + FWHT-256 + Q8_1 (5 intra-warp __shfl_xor_sync stages + 3 inter-warp stages)
//   - Band4 / SoA planes aligned to 16 bytes (get_int_b4 instead of unaligned get_int_b2)
//   - Warp-Register VQ Bank (__shfl_sync + __byte_perm + 8x dp4a without extra sumy dp4a)
//   - Branchless half-bit formats LIQK3_0_V2 (3.0625 bpw) and LIQK3_5_V2 (3.9375 bpw)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

struct block_q8_1 {
    half2 ds;
    int8_t qs[32];
};

__global__ __launch_bounds__(256, 1)
void quantize_q8_1_fwht256_kernel(const float * __restrict__ x, block_q8_1 * __restrict__ y, const uint32_t * __restrict__ suh_bits, int K) {
    __shared__ float smem[256];
    const int tid = threadIdx.x;
    const int blk = blockIdx.x;
    const int idx = blk * 256 + tid;
    if (idx >= K) return;

    const int lane = tid & 31;
    const int warp = tid >> 5;

    float xi = x[idx];
    uint32_t s_word = suh_bits[idx >> 5];
    if ((s_word >> lane) & 1u) xi = -xi;

    #pragma unroll
    for (int s = 1; s < 32; s <<= 1) {
        float other = __shfl_xor_sync(0xffffffff, xi, s);
        xi = (lane & s) ? (other - xi) : (xi + other);
    }

    smem[(lane << 3) | warp] = xi;
    __syncthreads();

    float v = smem[tid];
    #pragma unroll
    for (int s = 1; s < 8; s <<= 1) {
        float other = __shfl_xor_sync(0xffffffff, v, s);
        v = (lane & s) ? (other - v) : (v + other);
    }
    smem[tid] = v;
    __syncthreads();

    xi = smem[(lane << 3) | warp] * 0.0625f;

    float amax = fabsf(xi);
    float sum = xi;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }

    const float d = amax / 127.0f;
    const int8_t q = (amax == 0.0f) ? 0 : __float2int_rn(xi / d);
    const int ib = blk * 8 + warp;
    y[ib].qs[lane] = q;
    if (lane == 0) {
        y[ib].ds = __floats2half2_rn(d, sum);
    }
}

struct alignas(8) band4_liqk2_v2 { // 4 * 76 = 304 B
    uint32_t qs[4][16];
    uint8_t  scales[4][8];
    uint16_t extra[4];
    half     d[4];
};

struct alignas(8) band4_liqk3_0_v2 { // 4 * 98 = 392 B (branchless half-bit 3b/2b)
    uint32_t qs[4][16];
    uint32_t qh[4][4];
    uint8_t  scales_l[4][8];
    uint16_t scales_h[4];
    uint16_t extra[4];
    uint16_t mask_h[4];
    half     d[4];
};

struct alignas(8) band4_liqk3_k_v2 { // 4 * 110 = 440 B
    uint32_t qs[4][16];
    uint32_t qh[4][8];
    uint8_t  scales_l[4][8];
    uint16_t scales_h[4];
    uint16_t extra[4];
    half     d[4];
};

struct alignas(8) band4_liqk3_5_v2 { // 4 * 126 = 504 B (branchless half-bit 4b/3b: 64B qs + 32B qh + 16B qx)
    uint32_t qs[4][16];
    uint32_t qh[4][8];
    uint32_t qx[4][4];
    uint8_t  scales_l[4][8];
    uint16_t scales_h[4];
    uint16_t extra[4];
    half     d[4];
};

struct alignas(8) band4_liqk4_0_v2 { // 4 * 136 = 544 B
    uint32_t qs[4][32];
    uint8_t  scales[4][6];
    half     d[4];
};

struct alignas(8) band4_liqk4_k_v2 { // 4 * 144 = 576 B
    uint32_t qs[4][32];
    uint8_t  scales_l[4][8];
    uint8_t  scales_h[4][4];
    uint16_t extra[4];
    half     d[4];
};

static __device__ __forceinline__ int2 lookup_vq_bank_8(const int q_packed, const uint32_t bank_lo, const uint32_t bank_hi) {
    const uint32_t tmp0 = __byte_perm(bank_lo, bank_hi, q_packed >>  0);
    const uint32_t tmp1 = __byte_perm(bank_lo, bank_hi, q_packed >> 16);
    return make_int2(__byte_perm(tmp0, tmp1, 0x6420), __byte_perm(tmp0, tmp1, 0x7531));
}

static __device__ __forceinline__ int2 lookup_vq_bank_16(const int q4, const uint32_t b0, const uint32_t b1, const uint32_t b2, const uint32_t b3) {
    uint32_t tmp[2];
    const uint32_t sel = (0x32103210u | ((q4 & 0x88888888u) >> 1));
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t sh = 16 * i;
        const uint32_t lo = __byte_perm(b0, b1, q4 >> sh);
        const uint32_t hi = __byte_perm(b2, b3, q4 >> sh);
        tmp[i] = __byte_perm(lo, hi, sel >> sh);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

template <int FORMAT_ID>
__global__ __launch_bounds__(128, 1)
void gemv_liqk_v2_kernel(
    const void * __restrict__ vx,
    const block_q8_1 * __restrict__ y,
    float * __restrict__ dst,
    int K, int M)
{
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 4 + warp_id;
    if (row >= M) return;

    const uint32_t my_bank_lo = 0x050301ffu ^ (lane * 0x01010101u);
    const uint32_t my_bank_hi = 0x0f0b0907u ^ (lane * 0x01010101u);

    const int n_bands = K / 1024;
    const int b_sub = lane >> 3;
    const int g     = lane & 7;
    const int qs0   = 8 * (g >> 2);
    const int sh    = 2 * (g & 3);

    float sumf = 0.0f;

    for (int ib = 0; ib < n_bands; ++ib) {
        const int y_idx = ib * 32 + lane;
        const int * u_ptr = (const int *)y[y_idx].qs;
        const int u0 = u_ptr[0], u1 = u_ptr[1], u2 = u_ptr[2], u3 = u_ptr[3];
        const int u4 = u_ptr[4], u5 = u_ptr[5], u6 = u_ptr[6], u7 = u_ptr[7];
        const float dy = __low2float(y[y_idx].ds);

        if constexpr (FORMAT_ID == 20) {
            const band4_liqk2_v2 * b = (const band4_liqk2_v2 *)vx + row * n_bands + ib;
            const uint32_t * qs = b->qs[b_sub] + qs0;
            const int bank_id = (b->extra[b_sub] >> (2 * g)) & 3;
            const uint32_t tbl = __shfl_sync(0xffffffff, my_bank_lo, bank_id);
            int sumq0 = 0, sumq1 = 0;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int q0 = (qs[j] >> sh) & 0x03030303;
                const int q1 = (qs[j + 4] >> sh) & 0x03030303;
                const int2 v = lookup_vq_bank_8(q0 | (q1 << 4), tbl, tbl);
                const int uj0 = (j == 0 ? u0 : (j == 1 ? u1 : (j == 2 ? u2 : u3)));
                const int uj1 = (j == 0 ? u4 : (j == 1 ? u5 : (j == 2 ? u6 : u7)));
                sumq0 = __dp4a(v.x, uj0, sumq0);
                sumq1 = __dp4a(v.y, uj1, sumq1);
            }
            const int sc = b->scales[b_sub][g];
            sumf += __half2float(b->d[b_sub]) * dy * (((sc & 0xF) - 8) * sumq0 + ((sc >> 4) - 8) * sumq1);
        } else if constexpr (FORMAT_ID == 30) { // Branchless LIQK3_0_V2 (3.0625 bpw)
            const band4_liqk3_0_v2 * b = (const band4_liqk3_0_v2 *)vx + row * n_bands + ib;
            const uint32_t * qs = b->qs[b_sub] + qs0;
            const uint32_t * qh = b->qh[b_sub];
            const int bank_id = (b->extra[b_sub] >> (2 * g)) & 3;
            const uint32_t tbl_lo = __shfl_sync(0xffffffff, my_bank_lo, bank_id);
            const uint32_t tbl_hi = __shfl_sync(0xffffffff, my_bank_hi, bank_id);
            const int hi_mask = (g < 4) ? 0x01010101 : 0;
            const int sh_h = 2 * (g & 3);
            int sumq0 = 0, sumq1 = 0;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int lo0 = (qs[j] >> sh) & 0x03030303;
                const int lo1 = (qs[j + 4] >> sh) & 0x03030303;
                const uint32_t qhj = qh[j];
                const int hi0 = (qhj >> sh_h) & hi_mask;
                const int hi1 = (qhj >> (sh_h + 1)) & hi_mask;
                const int q0  = lo0 | (hi0 << 2);
                const int q1  = lo1 | (hi1 << 2);
                const int2 v = lookup_vq_bank_8(q0 | (q1 << 4), tbl_lo, tbl_hi);
                const int uj0 = (j == 0 ? u0 : (j == 1 ? u1 : (j == 2 ? u2 : u3)));
                const int uj1 = (j == 0 ? u4 : (j == 1 ? u5 : (j == 2 ? u6 : u7)));
                sumq0 = __dp4a(v.x, uj0, sumq0);
                sumq1 = __dp4a(v.y, uj1, sumq1);
            }
            const int sc = b->scales_l[b_sub][g];
            const int m0 = sc & 0xF, m1 = sc >> 4;
            const int ls0 = (b->scales_h[b_sub] >> (2 * g + 0)) & 1 ? -(2 * m0 + 1) : (2 * m0 + 1);
            const int ls1 = (b->scales_h[b_sub] >> (2 * g + 1)) & 1 ? -(2 * m1 + 1) : (2 * m1 + 1);
            sumf += __half2float(b->d[b_sub]) * dy * (ls0 * sumq0 + ls1 * sumq1);
        } else if constexpr (FORMAT_ID == 34) { // LIQK3_K_V2 (3.4375 bpw)
            const band4_liqk3_k_v2 * b = (const band4_liqk3_k_v2 *)vx + row * n_bands + ib;
            const uint32_t * qs = b->qs[b_sub] + qs0;
            const uint32_t * qh = b->qh[b_sub];
            const int bank_id = (b->extra[b_sub] >> (2 * g)) & 3;
            const uint32_t tbl_lo = __shfl_sync(0xffffffff, my_bank_lo, bank_id);
            const uint32_t tbl_hi = __shfl_sync(0xffffffff, my_bank_hi, bank_id);
            int sumq0 = 0, sumq1 = 0;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int lo0 = (qs[j] >> sh) & 0x03030303;
                const int hi0 = (qh[j] >> g)  & 0x01010101;
                const int q0  = lo0 | (hi0 << 2);

                const int lo1 = (qs[j + 4] >> sh) & 0x03030303;
                const int hi1 = (qh[j + 4] >> g)  & 0x01010101;
                const int q1  = lo1 | (hi1 << 2);

                const int2 v = lookup_vq_bank_8(q0 | (q1 << 4), tbl_lo, tbl_hi);
                const int uj0 = (j == 0 ? u0 : (j == 1 ? u1 : (j == 2 ? u2 : u3)));
                const int uj1 = (j == 0 ? u4 : (j == 1 ? u5 : (j == 2 ? u6 : u7)));
                sumq0 = __dp4a(v.x, uj0, sumq0);
                sumq1 = __dp4a(v.y, uj1, sumq1);
            }
            const int sc = b->scales_l[b_sub][g];
            const int m0 = sc & 0xF, m1 = sc >> 4;
            const int ls0 = (b->scales_h[b_sub] >> (2 * g + 0)) & 1 ? -(2 * m0 + 1) : (2 * m0 + 1);
            const int ls1 = (b->scales_h[b_sub] >> (2 * g + 1)) & 1 ? -(2 * m1 + 1) : (2 * m1 + 1);
            sumf += __half2float(b->d[b_sub]) * dy * (ls0 * sumq0 + ls1 * sumq1);
        } else if constexpr (FORMAT_ID == 39) { // Branchless LIQK3_5_V2 (3.9375 bpw: 3-bit base + 1-bit qx extension)
            const band4_liqk3_5_v2 * b = (const band4_liqk3_5_v2 *)vx + row * n_bands + ib;
            const uint32_t * qs = b->qs[b_sub] + qs0;
            const uint32_t * qh = b->qh[b_sub];
            const uint32_t * qx = b->qx[b_sub];
            const int bank_id = (b->extra[b_sub] >> (2 * g)) & 3;
            const uint32_t b0 = __shfl_sync(0xffffffff, my_bank_lo, bank_id);
            const uint32_t b1 = __shfl_sync(0xffffffff, my_bank_hi, bank_id);
            const uint32_t b2 = __shfl_sync(0xffffffff, my_bank_lo, bank_id + 4);
            const uint32_t b3 = __shfl_sync(0xffffffff, my_bank_hi, bank_id + 4);
            const int x_mask = (g < 4) ? 0x01010101 : 0;
            const int sh_x = 2 * (g & 3);
            int sumq0 = 0, sumq1 = 0;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int lo0 = (qs[j] >> sh) & 0x03030303;
                const int hi0 = (qh[j] >> g)  & 0x01010101;
                const int ex0 = (qx[j] >> sh_x) & x_mask;
                const int q0  = lo0 | (hi0 << 2) | (ex0 << 3);

                const int lo1 = (qs[j + 4] >> sh) & 0x03030303;
                const int hi1 = (qh[j + 4] >> g)  & 0x01010101;
                const int ex1 = (qx[j] >> (sh_x + 1)) & x_mask;
                const int q1  = lo1 | (hi1 << 2) | (ex1 << 3);

                const int2 v = lookup_vq_bank_16(q0 | (q1 << 4), b0, b1, b2, b3);
                const int uj0 = (j == 0 ? u0 : (j == 1 ? u1 : (j == 2 ? u2 : u3)));
                const int uj1 = (j == 0 ? u4 : (j == 1 ? u5 : (j == 2 ? u6 : u7)));
                sumq0 = __dp4a(v.x, uj0, sumq0);
                sumq1 = __dp4a(v.y, uj1, sumq1);
            }
            const int sc = b->scales_l[b_sub][g];
            const int m0 = sc & 0xF, m1 = sc >> 4;
            const int ls0 = (b->scales_h[b_sub] >> (2 * g + 0)) & 1 ? -(2 * m0 + 1) : (2 * m0 + 1);
            const int ls1 = (b->scales_h[b_sub] >> (2 * g + 1)) & 1 ? -(2 * m1 + 1) : (2 * m1 + 1);
            sumf += __half2float(b->d[b_sub]) * dy * (ls0 * sumq0 + ls1 * sumq1);
        } else if constexpr (FORMAT_ID == 42) {
            const band4_liqk4_0_v2 * b = (const band4_liqk4_0_v2 *)vx + row * n_bands + ib;
            const uint32_t * qs4 = b->qs[b_sub] + 4 * g;
            const uint32_t b0 = __shfl_sync(0xffffffff, my_bank_lo, 0);
            const uint32_t b1 = __shfl_sync(0xffffffff, my_bank_hi, 0);
            const uint32_t b2 = __shfl_sync(0xffffffff, my_bank_lo, 1);
            const uint32_t b3 = __shfl_sync(0xffffffff, my_bank_hi, 1);
            int sumq0 = 0, sumq1 = 0;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int2 v = lookup_vq_bank_16(qs4[j], b0, b1, b2, b3);
                const int uj0 = (j == 0 ? u0 : (j == 1 ? u1 : (j == 2 ? u2 : u3)));
                const int uj1 = (j == 0 ? u4 : (j == 1 ? u5 : (j == 2 ? u6 : u7)));
                sumq0 = __dp4a(v.x, uj0, sumq0);
                sumq1 = __dp4a(v.y, uj1, sumq1);
            }
            const int sc = b->scales[b_sub][g & 3];
            sumf += __half2float(b->d[b_sub]) * dy * (((sc & 0xF) - 8) * sumq0 + ((sc >> 4) - 8) * sumq1);
        } else if constexpr (FORMAT_ID == 45) {
            const band4_liqk4_k_v2 * b = (const band4_liqk4_k_v2 *)vx + row * n_bands + ib;
            const uint32_t * qs4 = b->qs[b_sub] + 4 * g;
            const int bank_id = (b->extra[b_sub] >> (2 * g)) & 3;
            const uint32_t b0 = __shfl_sync(0xffffffff, my_bank_lo, bank_id);
            const uint32_t b1 = __shfl_sync(0xffffffff, my_bank_hi, bank_id);
            const uint32_t b2 = __shfl_sync(0xffffffff, my_bank_lo, bank_id + 4);
            const uint32_t b3 = __shfl_sync(0xffffffff, my_bank_hi, bank_id + 4);
            int sumq0 = 0, sumq1 = 0;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int2 v = lookup_vq_bank_16(qs4[j], b0, b1, b2, b3);
                const int uj0 = (j == 0 ? u0 : (j == 1 ? u1 : (j == 2 ? u2 : u3)));
                const int uj1 = (j == 0 ? u4 : (j == 1 ? u5 : (j == 2 ? u6 : u7)));
                sumq0 = __dp4a(v.x, uj0, sumq0);
                sumq1 = __dp4a(v.y, uj1, sumq1);
            }
            const int hs = b->scales_h[b_sub][g >> 1] >> (4 * (g & 1));
            const int sc = b->scales_l[b_sub][g];
            const int ls0 = ((sc & 0xF) | (((hs >> 0) & 3) << 4)) - 32;
            const int ls1 = ((sc >> 4)  | (((hs >> 2) & 3) << 4)) - 32;
            sumf += __half2float(b->d[b_sub]) * dy * (ls0 * sumq0 + ls1 * sumq1);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumf += __shfl_xor_sync(0xffffffff, sumf, offset);
    }
    if (lane == 0) {
        dst[row] = sumf;
    }
}

template <int FORMAT_ID, typename BandT>
void bench_full_model(const char * name, double bpw, int n_layers, int d_model, int d_kv, int d_ffn, int iters = 30) {
    // Allocate ALL n_layers layers in VRAM (>500..1800 MB) to fully prevent the weights from hitting the 48 MB L2 cache!
    const int total_out_dmodel = d_model + 2 * d_kv + d_model + 2 * d_ffn;
    const size_t bytes_w1 = (size_t)total_out_dmodel * (d_model / 1024) * sizeof(BandT);
    const size_t bytes_w2 = (size_t)d_model * (d_ffn / 1024) * sizeof(BandT);
    const size_t total_layer_bytes = bytes_w1 + bytes_w2;
    const size_t total_model_bytes = total_layer_bytes * n_layers;

    std::vector<void *> d_w1(n_layers), d_w2(n_layers);
    for (int l = 0; l < n_layers; ++l) {
        CHECK_CUDA(cudaMalloc(&d_w1[l], bytes_w1));
        CHECK_CUDA(cudaMalloc(&d_w2[l], bytes_w2));
        CHECK_CUDA(cudaMemset(d_w1[l], 0x55, bytes_w1));
        CHECK_CUDA(cudaMemset(d_w2[l], 0x55, bytes_w2));
    }

    float * d_x1 = nullptr, * d_x2 = nullptr, * d_out = nullptr;
    block_q8_1 * d_q1 = nullptr, * d_q2 = nullptr;
    uint32_t * d_suh = nullptr;
    CHECK_CUDA(cudaMalloc(&d_x1, d_model * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_x2, d_ffn * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, total_out_dmodel * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_q1, (d_model / 32) * sizeof(block_q8_1)));
    CHECK_CUDA(cudaMalloc(&d_q2, (d_ffn / 32) * sizeof(block_q8_1)));
    CHECK_CUDA(cudaMalloc(&d_suh, (d_ffn / 32) * sizeof(uint32_t)));

    // Warmup over all layers
    for (int w = 0; w < 5; ++w) {
        for (int l = 0; l < n_layers; ++l) {
            quantize_q8_1_fwht256_kernel<<<d_model / 256, 256>>>(d_x1, d_q1, d_suh, d_model);
            quantize_q8_1_fwht256_kernel<<<d_model / 256, 256>>>(d_x1, d_q1, d_suh, d_model);
            quantize_q8_1_fwht256_kernel<<<d_model / 256, 256>>>(d_x1, d_q1, d_suh, d_model);
            gemv_liqk_v2_kernel<FORMAT_ID><<<(total_out_dmodel + 3) / 4, 128>>>(d_w1[l], d_q1, d_out, d_model, total_out_dmodel);
            quantize_q8_1_fwht256_kernel<<<d_ffn / 256, 256>>>(d_x2, d_q2, d_suh, d_ffn);
            gemv_liqk_v2_kernel<FORMAT_ID><<<(d_model + 3) / 4, 128>>>(d_w2[l], d_q2, d_out, d_ffn, d_model);
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t ev0, ev1;
    CHECK_CUDA(cudaEventCreate(&ev0));
    CHECK_CUDA(cudaEventCreate(&ev1));
    CHECK_CUDA(cudaEventRecord(ev0));
    for (int it = 0; it < iters; ++it) {
        for (int l = 0; l < n_layers; ++l) {
            quantize_q8_1_fwht256_kernel<<<d_model / 256, 256>>>(d_x1, d_q1, d_suh, d_model);
            quantize_q8_1_fwht256_kernel<<<d_model / 256, 256>>>(d_x1, d_q1, d_suh, d_model);
            quantize_q8_1_fwht256_kernel<<<d_model / 256, 256>>>(d_x1, d_q1, d_suh, d_model);
            gemv_liqk_v2_kernel<FORMAT_ID><<<(total_out_dmodel + 3) / 4, 128>>>(d_w1[l], d_q1, d_out, d_model, total_out_dmodel);
            quantize_q8_1_fwht256_kernel<<<d_ffn / 256, 256>>>(d_x2, d_q2, d_suh, d_ffn);
            gemv_liqk_v2_kernel<FORMAT_ID><<<(d_model + 3) / 4, 128>>>(d_w2[l], d_q2, d_out, d_ffn, d_model);
        }
    }
    CHECK_CUDA(cudaEventRecord(ev1));
    CHECK_CUDA(cudaEventSynchronize(ev1));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, ev0, ev1));

    double ms_per_tok = (double)ms / iters;
    double gbps = ((double)total_model_bytes / 1e9) / (ms_per_tok / 1000.0);
    printf("    \"%s\": {\"bpw\": %.4f, \"linear_ms_per_tok\": %.3f, \"effective_gbps\": %.2f},\n",
           name, bpw, ms_per_tok, gbps);

    for (int l = 0; l < n_layers; ++l) {
        CHECK_CUDA(cudaFree(d_w1[l]));
        CHECK_CUDA(cudaFree(d_w2[l]));
    }
    CHECK_CUDA(cudaFree(d_x1));
    CHECK_CUDA(cudaFree(d_x2));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_q1));
    CHECK_CUDA(cudaFree(d_q2));
    CHECK_CUDA(cudaFree(d_suh));
    CHECK_CUDA(cudaEventDestroy(ev0));
    CHECK_CUDA(cudaEventDestroy(ev1));
}

int main() {
    printf("{\n  \"1B_full_linear_gemv\": {\n");
    bench_full_model<20, band4_liqk2_v2>("LIQK2_K_V2", 2.3750, 16, 2048, 512, 8192);
    bench_full_model<30, band4_liqk3_0_v2>("LIQK3_0_V2", 3.0625, 16, 2048, 512, 8192);
    bench_full_model<34, band4_liqk3_k_v2>("LIQK3_K_V2", 3.4375, 16, 2048, 512, 8192);
    bench_full_model<39, band4_liqk3_5_v2>("LIQK3_5_V2", 3.9375, 16, 2048, 512, 8192);
    bench_full_model<42, band4_liqk4_0_v2>("LIQK4_0_V2", 4.2500, 16, 2048, 512, 8192);
    bench_full_model<45, band4_liqk4_k_v2>("LIQK4_K_V2", 4.5000, 16, 2048, 512, 8192);
    printf("    \"_done\": 1\n  },\n  \"3B_full_linear_gemv\": {\n");
    bench_full_model<20, band4_liqk2_v2>("LIQK2_K_V2", 2.3750, 28, 3072, 1024, 8192);
    bench_full_model<30, band4_liqk3_0_v2>("LIQK3_0_V2", 3.0625, 28, 3072, 1024, 8192);
    bench_full_model<34, band4_liqk3_k_v2>("LIQK3_K_V2", 3.4375, 28, 3072, 1024, 8192);
    bench_full_model<39, band4_liqk3_5_v2>("LIQK3_5_V2", 3.9375, 28, 3072, 1024, 8192);
    bench_full_model<42, band4_liqk4_0_v2>("LIQK4_0_V2", 4.2500, 28, 3072, 1024, 8192);
    bench_full_model<45, band4_liqk4_k_v2>("LIQK4_K_V2", 4.5000, 28, 3072, 1024, 8192);
    printf("    \"_done\": 1\n  }\n}\n");
    return 0;
}

