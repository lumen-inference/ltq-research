// Full-model CPU AVX-512 VBMI/VNNI benchmark of .liqk v2 kernels on AMD Ryzen 9 9950X (Zen 5, 16 threads)
// Tests:
//   - Fused Online suh + FWHT-256 + Q8_K in ZMM registers (__m512)
//   - 64-byte Warp-Register VQ Bank via a single _mm512_permutexvar_epi8 instruction (vpermb)
//   - Vector dot product via _mm512_dpbusd_epi32 (vpdpbusd)
//   - Aligned Band4 / SoA planes for all 6 .liqk v2 formats (including the half-bit LIQK3_0_V2 and LIQK3_5_V2)

#include <immintrin.h>
#include <omp.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>

struct alignas(64) block_q8_k_v2 {
    float d;
    uint8_t qs[256]; // offset to uint8 (+128) for direct use in vpdpbusd!
    int16_t bsums[16];
};

// Fused Online suh + FWHT-256 + Q8_K on AVX-512 (256 floats = 16 ZMM registers!)
static inline void quantize_q8_k_fwht256_avx512(const float * __restrict__ x, block_q8_k_v2 * __restrict__ y, int K) {
    const int nb = K / 256;
    float buf[256] __attribute__((aligned(64)));
    const __m512 norm_scale = _mm512_set1_ps(0.0625f);
    const __m512 sign_flip  = _mm512_set_ps(
        -1.f, 1.f, -1.f, 1.f, 1.f, -1.f, 1.f, -1.f,
        -1.f, 1.f, -1.f, 1.f, 1.f, -1.f, 1.f, -1.f);

    for (int ib = 0; ib < nb; ++ib) {
        // Butterfly stages within and across 16 ZMM registers
        for (int g = 0; g < 16; ++g) {
            __m512 v = _mm512_mul_ps(_mm512_loadu_ps(x + ib * 256 + g * 16), sign_flip);
            // intra-vector stages (1, 2, 4, 8)
            __m512 s1 = _mm512_permute_ps(v, 0xB1);
            v = _mm512_mask_sub_ps(_mm512_add_ps(v, s1), 0xAAAA, s1, _mm512_sub_ps(s1, v));
            __m512 s2 = _mm512_permute_ps(v, 0x4E);
            v = _mm512_mask_sub_ps(_mm512_add_ps(v, s2), 0xCCCC, s2, _mm512_sub_ps(s2, v));
            __m512 s4 = _mm512_shuffle_f32x4(v, v, 0xB1);
            v = _mm512_mask_sub_ps(_mm512_add_ps(v, s4), 0xF0F0, s4, _mm512_sub_ps(s4, v));
            __m512 s8 = _mm512_shuffle_f32x4(v, v, 0x4E);
            v = _mm512_mask_sub_ps(_mm512_add_ps(v, s8), 0xFF00, s8, _mm512_sub_ps(s8, v));
            _mm512_store_ps(buf + g * 16, v);
        }
        // Cross-register stages (16, 32, 64, 128)
        for (int step = 16; step < 256; step <<= 1) {
            for (int base = 0; base < 256; base += (step << 1)) {
                for (int j = 0; j < step; j += 16) {
                    __m512 a = _mm512_load_ps(buf + base + j);
                    __m512 b = _mm512_load_ps(buf + base + j + step);
                    _mm512_store_ps(buf + base + j, _mm512_add_ps(a, b));
                    _mm512_store_ps(buf + base + j + step, _mm512_sub_ps(a, b));
                }
            }
        }
        __m512 max_abs = _mm512_setzero_ps();
        for (int g = 0; g < 16; ++g) {
            __m512 v = _mm512_mul_ps(_mm512_load_ps(buf + g * 16), norm_scale);
            _mm512_store_ps(buf + g * 16, v);
            max_abs = _mm512_max_ps(max_abs, _mm512_abs_ps(v));
        }
        float amax = _mm512_reduce_max_ps(max_abs);
        float iscale = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        y[ib].d = (amax > 0.0f) ? (amax / 127.0f) : 0.0f;
        __m512 vscale = _mm512_set1_ps(iscale);
        __m512i offset128 = _mm512_set1_epi32(128);
        for (int g = 0; g < 4; ++g) {
            __m512i i0 = _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_load_ps(buf + g * 64 + 0),  vscale)), offset128);
            __m512i i1 = _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_load_ps(buf + g * 64 + 16), vscale)), offset128);
            __m512i i2 = _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_load_ps(buf + g * 64 + 32), vscale)), offset128);
            __m512i i3 = _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_load_ps(buf + g * 64 + 48), vscale)), offset128);
            __m512i p16_0 = _mm512_packs_epi32(i0, i1);
            __m512i p16_1 = _mm512_packs_epi32(i2, i3);
            __m512i u8 = _mm512_packus_epi16(p16_0, p16_1);
            _mm512_storeu_si512((__m512i *)(y[ib].qs + g * 64), u8);
        }
    }
}

// Zen 5 AVX-512 VBMI + VNNI GEMV kernel for .liqk v2:
// Uses _mm512_permutexvar_epi8 (vpermb) over the 64-byte Warp-Register VQ Bank + _mm512_dpbusd_epi32 (vpdpbusd)!
template <int BYTES_PER_BLOCK>
static void gemv_liqk_v2_avx512(
    const uint8_t * __restrict__ vx,
    const block_q8_k_v2 * __restrict__ y,
    float * __restrict__ dst,
    int K, int M)
{
    const int nb = K / 256;
    const size_t row_bytes = (size_t)nb * BYTES_PER_BLOCK;

    #pragma omp parallel for schedule(static)
    for (int row = 0; row < M; ++row) {
        const uint8_t * rx = vx + (size_t)row * row_bytes;
        const __m512i vq_bank = _mm512_set1_epi64(0x0f0b0907050301ffULL);
        const __m512i m2 = _mm512_set1_epi8(0x03);
        const __m512i mh = _mm512_set1_epi8(0x04);
        const __m512i mx = _mm512_set1_epi8(0x08);
        const __m512i m4 = _mm512_set1_epi8(0x0F);

        __m512 acc_f = _mm512_setzero_ps();

        for (int ib = 0; ib < nb; ++ib) {
            const uint8_t * blk = rx + ib * BYTES_PER_BLOCK;
            const __m512i q8_0 = _mm512_loadu_si512((const __m512i *)(y[ib].qs + 0));
            const __m512i q8_1 = _mm512_loadu_si512((const __m512i *)(y[ib].qs + 64));
            const __m512i q8_2 = _mm512_loadu_si512((const __m512i *)(y[ib].qs + 128));
            const __m512i q8_3 = _mm512_loadu_si512((const __m512i *)(y[ib].qs + 192));

            __m512i sumi = _mm512_setzero_si512();

            if constexpr (BYTES_PER_BLOCK == 76) { // LIQK2_K_V2 (2.375 bpw)
                const __m512i lbits = _mm512_loadu_si512((const __m512i *)blk);
                const __m512i bsel  = _mm512_set1_epi8((blk[72] & 3) << 3);
                const __m512i idx0 = _mm512_or_si512(_mm512_and_si512(lbits, m2), bsel);
                const __m512i idx1 = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(lbits, 2), m2), bsel);
                const __m512i idx2 = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(lbits, 4), m2), bsel);
                const __m512i idx3 = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(lbits, 6), m2), bsel);
                sumi = _mm512_dpbusd_epi32(sumi, q8_0, _mm512_permutexvar_epi8(idx0, vq_bank));
                sumi = _mm512_dpbusd_epi32(sumi, q8_1, _mm512_permutexvar_epi8(idx1, vq_bank));
                sumi = _mm512_dpbusd_epi32(sumi, q8_2, _mm512_permutexvar_epi8(idx2, vq_bank));
                sumi = _mm512_dpbusd_epi32(sumi, q8_3, _mm512_permutexvar_epi8(idx3, vq_bank));
            } else if constexpr (BYTES_PER_BLOCK == 98) { // LIQK3_0_V2 (3.0625 bpw, half-bit 3b/2b)
                const __m512i lbits = _mm512_loadu_si512((const __m512i *)blk);
                const __m512i hbits = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(blk + 64)));
                const __m512i bsel  = _mm512_set1_epi8((blk[90] & 3) << 3);
                const __m512i idx0 = _mm512_or_si512(_mm512_or_si512(_mm512_and_si512(lbits, m2), _mm512_and_si512(_mm512_slli_epi16(hbits, 2), mh)), bsel);
                const __m512i idx1 = _mm512_or_si512(_mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(lbits, 2), m2), _mm512_and_si512(_mm512_slli_epi16(hbits, 1), mh)), bsel);
                const __m512i idx2 = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(lbits, 4), m2), bsel);
                const __m512i idx3 = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(lbits, 6), m2), bsel);
                sumi = _mm512_dpbusd_epi32(sumi, q8_0, _mm512_permutexvar_epi8(idx0, vq_bank));
                sumi = _mm512_dpbusd_epi32(sumi, q8_1, _mm512_permutexvar_epi8(idx1, vq_bank));
                sumi = _mm512_dpbusd_epi32(sumi, q8_2, _mm512_permutexvar_epi8(idx2, vq_bank));
                sumi = _mm512_dpbusd_epi32(sumi, q8_3, _mm512_permutexvar_epi8(idx3, vq_bank));
            } else if constexpr (BYTES_PER_BLOCK == 110) { // LIQK3_K_V2 (3.4375 bpw)
                const __m512i lbits = _mm512_loadu_si512((const __m512i *)blk);
                const __m512i hbits = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)(blk + 64)));
                const __m512i bsel  = _mm512_set1_epi8((blk[106] & 3) << 3);
                const __m512i idx0 = _mm512_or_si512(_mm512_or_si512(_mm512_and_si512(lbits, m2), _mm512_and_si512(_mm512_slli_epi16(hbits, 2), mh)), bsel);
                const __m512i idx1 = _mm512_or_si512(_mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(lbits, 2), m2), _mm512_and_si512(_mm512_slli_epi16(hbits, 1), mh)), bsel);
                const __m512i idx2 = _mm512_or_si512(_mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(lbits, 4), m2), _mm512_and_si512(hbits, mh)), bsel);
                const __m512i idx3 = _mm512_or_si512(_mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(lbits, 6), m2), _mm512_and_si512(_mm512_srli_epi16(hbits, 1), mh)), bsel);
                sumi = _mm512_dpbusd_epi32(sumi, q8_0, _mm512_permutexvar_epi8(idx0, vq_bank));
                sumi = _mm512_dpbusd_epi32(sumi, q8_1, _mm512_permutexvar_epi8(idx1, vq_bank));
                sumi = _mm512_dpbusd_epi32(sumi, q8_2, _mm512_permutexvar_epi8(idx2, vq_bank));
                sumi = _mm512_dpbusd_epi32(sumi, q8_3, _mm512_permutexvar_epi8(idx3, vq_bank));
            } else if constexpr (BYTES_PER_BLOCK == 126) { // LIQK3_5_V2 (3.9375 bpw, half-bit 4b/3b)
                const __m512i lbits = _mm512_loadu_si512((const __m512i *)blk);
                const __m512i hbits = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)(blk + 64)));
                const __m512i xbits = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(blk + 96)));
                const __m512i bsel  = _mm512_set1_epi8((blk[122] & 3) << 4);
                const __m512i idx0 = _mm512_or_si512(_mm512_or_si512(_mm512_or_si512(_mm512_and_si512(lbits, m2), _mm512_and_si512(_mm512_slli_epi16(hbits, 2), mh)), _mm512_and_si512(_mm512_slli_epi16(xbits, 3), mx)), bsel);
                const __m512i idx1 = _mm512_or_si512(_mm512_or_si512(_mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(lbits, 2), m2), _mm512_and_si512(_mm512_slli_epi16(hbits, 1), mh)), _mm512_and_si512(_mm512_slli_epi16(xbits, 2), mx)), bsel);
                const __m512i idx2 = _mm512_or_si512(_mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(lbits, 4), m2), _mm512_and_si512(hbits, mh)), bsel);
                const __m512i idx3 = _mm512_or_si512(_mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(lbits, 6), m2), _mm512_and_si512(_mm512_srli_epi16(hbits, 1), mh)), bsel);
                sumi = _mm512_dpbusd_epi32(sumi, q8_0, _mm512_permutexvar_epi8(idx0, vq_bank));
                sumi = _mm512_dpbusd_epi32(sumi, q8_1, _mm512_permutexvar_epi8(idx1, vq_bank));
                sumi = _mm512_dpbusd_epi32(sumi, q8_2, _mm512_permutexvar_epi8(idx2, vq_bank));
                sumi = _mm512_dpbusd_epi32(sumi, q8_3, _mm512_permutexvar_epi8(idx3, vq_bank));
            } else { // LIQK4_0_V2 (136B) and LIQK4_K_V2 (144B)
                const __m512i q4_0 = _mm512_loadu_si512((const __m512i *)(blk + 0));
                const __m512i q4_1 = _mm512_loadu_si512((const __m512i *)(blk + 64));
                const __m512i bsel = _mm512_set1_epi8((blk[132] & 3) << 4);
                const __m512i idx0 = _mm512_or_si512(_mm512_and_si512(q4_0, m4), bsel);
                const __m512i idx1 = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(q4_0, 4), m4), bsel);
                const __m512i idx2 = _mm512_or_si512(_mm512_and_si512(q4_1, m4), bsel);
                const __m512i idx3 = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(q4_1, 4), m4), bsel);
                sumi = _mm512_dpbusd_epi32(sumi, q8_0, _mm512_permutexvar_epi8(idx0, vq_bank));
                sumi = _mm512_dpbusd_epi32(sumi, q8_1, _mm512_permutexvar_epi8(idx1, vq_bank));
                sumi = _mm512_dpbusd_epi32(sumi, q8_2, _mm512_permutexvar_epi8(idx2, vq_bank));
                sumi = _mm512_dpbusd_epi32(sumi, q8_3, _mm512_permutexvar_epi8(idx3, vq_bank));
            }

            acc_f = _mm512_fmadd_ps(_mm512_set1_ps(y[ib].d * 0.01f), _mm512_cvtepi32_ps(sumi), acc_f);
        }
        dst[row] = _mm512_reduce_add_ps(acc_f);
    }
}

template <int BYTES_PER_BLOCK>
void bench_cpu_model(const char * name, double bpw, int n_layers, int d_model, int d_kv, int d_ffn, int iters = 10) {
    const int total_out_dmodel = d_model + 2 * d_kv + d_model + 2 * d_ffn;
    const size_t bytes_w1 = (size_t)total_out_dmodel * (d_model / 256) * BYTES_PER_BLOCK;
    const size_t bytes_w2 = (size_t)d_model * (d_ffn / 256) * BYTES_PER_BLOCK;
    const size_t total_model_bytes = (bytes_w1 + bytes_w2) * n_layers;

    // Allocate the full volume of all layers in RAM (>500..1900 MB, more than the 64 MB L3 cache of the Ryzen 9 9950X!)
    std::vector<uint8_t *> w1(n_layers), w2(n_layers);
    for (int l = 0; l < n_layers; ++l) {
        w1[l] = (uint8_t *)aligned_alloc(64, (bytes_w1 + 63) & ~63ULL);
        w2[l] = (uint8_t *)aligned_alloc(64, (bytes_w2 + 63) & ~63ULL);
        memset(w1[l], 0x55, bytes_w1);
        memset(w2[l], 0x55, bytes_w2);
    }
    std::vector<float> x1(d_model, 0.1f), x2(d_ffn, 0.1f), out(total_out_dmodel, 0.0f);
    std::vector<block_q8_k_v2> q1(d_model / 256), q2(d_ffn / 256);

    // Warmup
    for (int l = 0; l < n_layers; ++l) {
        quantize_q8_k_fwht256_avx512(x1.data(), q1.data(), d_model);
        gemv_liqk_v2_avx512<BYTES_PER_BLOCK>(w1[l], q1.data(), out.data(), d_model, total_out_dmodel);
        quantize_q8_k_fwht256_avx512(x2.data(), q2.data(), d_ffn);
        gemv_liqk_v2_avx512<BYTES_PER_BLOCK>(w2[l], q2.data(), out.data(), d_ffn, d_model);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
        for (int l = 0; l < n_layers; ++l) {
            quantize_q8_k_fwht256_avx512(x1.data(), q1.data(), d_model);
            quantize_q8_k_fwht256_avx512(x1.data(), q1.data(), d_model);
            quantize_q8_k_fwht256_avx512(x1.data(), q1.data(), d_model);
            gemv_liqk_v2_avx512<BYTES_PER_BLOCK>(w1[l], q1.data(), out.data(), d_model, total_out_dmodel);
            quantize_q8_k_fwht256_avx512(x2.data(), q2.data(), d_ffn);
            gemv_liqk_v2_avx512<BYTES_PER_BLOCK>(w2[l], q2.data(), out.data(), d_ffn, d_model);
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_per_tok = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    double gbps = ((double)total_model_bytes / 1e9) / (ms_per_tok / 1000.0);
    double tok_s = 1000.0 / ms_per_tok;
    printf("    \"%s\": {\"bpw\": %.4f, \"linear_ms_per_tok\": %.3f, \"linear_tok_s\": %.2f, \"effective_gbps\": %.2f},\n",
           name, bpw, ms_per_tok, tok_s, gbps);

    for (int l = 0; l < n_layers; ++l) {
        free(w1[l]);
        free(w2[l]);
    }
}

int main() {
    omp_set_num_threads(16);
    printf("{\n  \"1B_cpu_avx512_t16\": {\n");
    bench_cpu_model<76>("LIQK2_K_V2", 2.3750, 16, 2048, 512, 8192);
    bench_cpu_model<98>("LIQK3_0_V2", 3.0625, 16, 2048, 512, 8192);
    bench_cpu_model<110>("LIQK3_K_V2", 3.4375, 16, 2048, 512, 8192);
    bench_cpu_model<126>("LIQK3_5_V2", 3.9375, 16, 2048, 512, 8192);
    bench_cpu_model<136>("LIQK4_0_V2", 4.2500, 16, 2048, 512, 8192);
    bench_cpu_model<144>("LIQK4_K_V2", 4.5000, 16, 2048, 512, 8192);
    printf("    \"_done\": 1\n  },\n  \"3B_cpu_avx512_t16\": {\n");
    bench_cpu_model<76>("LIQK2_K_V2", 2.3750, 28, 3072, 1024, 8192);
    bench_cpu_model<98>("LIQK3_0_V2", 3.0625, 28, 3072, 1024, 8192);
    bench_cpu_model<110>("LIQK3_K_V2", 3.4375, 28, 3072, 1024, 8192);
    bench_cpu_model<126>("LIQK3_5_V2", 3.9375, 28, 3072, 1024, 8192);
    bench_cpu_model<136>("LIQK4_0_V2", 4.2500, 28, 3072, 1024, 8192);
    bench_cpu_model<144>("LIQK4_K_V2", 4.5000, 28, 3072, 1024, 8192);
    printf("    \"_done\": 1\n  }\n}\n");
    return 0;
}
