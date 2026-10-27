// CPU decode microbenchmark (AVX2 without AVX-512): weight decode + dot product with int8 activations, single thread.
// Compares TCQ formats (tile of 16 rows x 16 columns, tail-biting, nl = 4 / 8 / 16 label subsets, header h = 2)
// against an IQ4_NL-style kernel (4-bit index, 16-entry table, one scale per 32 weights). Measures compute throughput
// (data in L2) to compare it against the memory bandwidth of the target CPUs.
//
// TCQ tile layout (K bits per weight, 16 x 16):
//   c[16]  — code bits of the tile rows (bit t = column t); label of weight (r, t): bit 0 = c_t, bits 1.. = parities of code bits
//            c_{t-1-j} by masks m, m2, m3 (window of the 9 previous bits, across the row boundary, tail-biting over the tile);
//   u[..]  — K-1 non-code bits per weight, packed as in q2_K: byte k of vector v holds the fields of 8/(K-1) row-pair groups;
//   hdr    — 2 gain bits per tile row.
// Value = tab[label * 2^(K-1) + u] * gain[hdr_r] (int8 table).
//
//   gcc -O3 -mavx2 -mfma -mno-avx512f -mtune=znver2 -o tcq_avx2_bench tcq_avx2_bench.c && ./tcq_avx2_bench
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define M_ROWS 512
#define K_COLS 4096
#define NTILE (K_COLS / 16)

typedef struct {
    uint16_t c[16];
    uint8_t u[64];  // up to 2 low bits of the non-code part per weight: 2 vectors of 32 bytes (2-bit fields, as in q2_K)
    uint8_t u1[32]; // K = 4: third non-code bit — 1-bit bit fields (like the high bit of q3_K)
    uint8_t u2[64]; // K = 5: bits 2-3 of the non-code part — second set of 2-bit fields (like u)
    uint32_t hdr;
} tile_t;

static const int MASK_M = 359, MASK_2 = 241, MASK_3 = 299;

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int ubits_of(int K) { return K - 1; }

// u of weight (r, t): row pair g = r / 2, byte k = (r % 2) * 16 + t; groups per byte gpb = 8 / ub
static int get_u(const tile_t* tl, int K, int r, int t)
{
    int ub = ubits_of(K) > 2 ? 2 : ubits_of(K), gpb = 8 / ub, g = r / 2, k = (r % 2) * 16 + t;
    int v = g / gpb, sh = ub * (g % gpb);
    int u = (tl->u[v * 32 + k] >> sh) & ((1 << ub) - 1);
    if (K == 4) u |= ((tl->u1[k] >> g) & 1) << 2;
    if (K == 5) u |= ((tl->u2[v * 32 + k] >> sh) & 3) << 2;
    return u;
}

static int parity(uint32_t x) { return __builtin_popcount(x) & 1; }

static int label_of(const tile_t* tl, int r, int t, int nl)
{
    uint32_t X = ((uint32_t)tl->c[r] << 16) | tl->c[(r + 15) & 15];
    uint32_t s1 = X >> (16 + t - 9);  // bits c_{t-9} .. c_{t-1} -> (s >> 1): bit j = c_{t-1-j}
    uint32_t w = 0;
    for (int j = 0; j < 9; j++) w |= ((s1 >> (8 - j)) & 1) << j;
    int lb = (X >> (16 + t)) & 1;
    lb |= parity(w & MASK_M) << 1;
    if (nl >= 8) lb |= parity(w & MASK_2) << 2;
    if (nl >= 16) lb |= parity(w & MASK_3) << 3;
    return lb;
}

static void ref_tcq(const tile_t* T, const int8_t* tab, const float* gains, const int8_t* a, float* y, int K, int nl)
{
    for (int b = 0; b < M_ROWS / 16; b++)
        for (int r = 0; r < 16; r++) {
            float acc = 0;
            for (int j = 0; j < NTILE; j++) {
                const tile_t* tl = &T[b * NTILE + j];
                int s = 0;
                for (int t = 0; t < 16; t++) {
                    int idx = label_of(tl, r, t, nl) * (1 << (K - 1)) + get_u(tl, K, r, t);
                    s += tab[idx] * a[j * 16 + t];
                }
                acc += s * gains[(tl->hdr >> (2 * r)) & 3];
            }
            y[b * 16 + r] = acc;
        }
}

// parities for 8 rows in 32-bit lanes: X = c_r << 16 | c_{r-1}; bit t of word P = XOR_j bit (16 + t - 1 - j) of X
static inline __m256i par_mask(__m256i X, int m)
{
    __m256i p = _mm256_setzero_si256();
    for (int j = 0; j < 9; j++)
        if ((m >> j) & 1) p = _mm256_xor_si256(p, _mm256_srli_epi32(X, 15 - j));
    return p;
}

static inline __attribute__((always_inline)) void tcq_tile(const tile_t* tl, __m256i a2, const int8_t* tab,
                                                          const float* gains, __m256* acc, const int K, const int nl)
{
    const __m256i lo16 = _mm256_set1_epi32(0xffff);
    // c of rows 0..7 and 8..15 in 32-bit lanes and their previous rows (cyclically: row 15 precedes row 0)
    __m256i C = _mm256_loadu_si256((const __m256i*)tl->c);
    __m256i lo8 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(C));
    __m256i hi8 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(C, 1));
    const __m256i rot = _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6);
    __m256i rl = _mm256_permutevar8x32_epi32(lo8, rot), rh = _mm256_permutevar8x32_epi32(hi8, rot);
    __m256i pl = _mm256_blend_epi32(rl, rh, 1), ph = _mm256_blend_epi32(rh, rl, 1);
    __m256i Xl = _mm256_or_si256(_mm256_slli_epi32(lo8, 16), pl), Xh = _mm256_or_si256(_mm256_slli_epi32(hi8, 16), ph);
    // word A = c | P1 << 16, B = P2 | P3 << 16 per row
    __m256i Al = _mm256_or_si256(lo8, _mm256_slli_epi32(par_mask(Xl, MASK_M), 16));
    __m256i Ah = _mm256_or_si256(hi8, _mm256_slli_epi32(par_mask(Xh, MASK_M), 16));
    __m256i Bl = _mm256_setzero_si256(), Bh = _mm256_setzero_si256();
    if (nl >= 8) {
        Bl = _mm256_and_si256(par_mask(Xl, MASK_2), lo16);
        Bh = _mm256_and_si256(par_mask(Xh, MASK_2), lo16);
    }
    if (nl >= 16) {
        Bl = _mm256_or_si256(Bl, _mm256_slli_epi32(par_mask(Xl, MASK_3), 16));
        Bh = _mm256_or_si256(Bh, _mm256_slli_epi32(par_mask(Xh, MASK_3), 16));
    }
    // pair rows: after unpack + permute4x64, lane 0 = row 2g, lane 1 = row 2g + 1 (low qword = pair
    // (0,1) / (2,3), high = (4,5) / (6,7))
    __m256i R[4];
    R[0] = _mm256_permute4x64_epi64(_mm256_unpacklo_epi32(Al, Bl), 0xd8);
    R[1] = _mm256_permute4x64_epi64(_mm256_unpackhi_epi32(Al, Bl), 0xd8);
    R[2] = _mm256_permute4x64_epi64(_mm256_unpacklo_epi32(Ah, Bh), 0xd8);
    R[3] = _mm256_permute4x64_epi64(_mm256_unpackhi_epi32(Ah, Bh), 0xd8);
    // row scales: gain[hdr_r]
    const __m256i sh_lo = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14), sh_hi = _mm256_setr_epi32(16, 18, 20, 22, 24, 26, 28, 30);
    __m256i hv = _mm256_set1_epi32((int)tl->hdr), three = _mm256_set1_epi32(3);
    __m256 gv = _mm256_castps128_ps256(_mm_loadu_ps(gains));
    __m256 sc_lo = _mm256_permutevar8x32_ps(gv, _mm256_and_si256(_mm256_srlv_epi32(hv, sh_lo), three));
    __m256 sc_hi = _mm256_permutevar8x32_ps(gv, _mm256_and_si256(_mm256_srlv_epi32(hv, sh_hi), three));

    const __m256i bitm = _mm256_setr_epi8(1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128,
                                          1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128);
    const int ub = K - 1, gpb = 8 / ub;
    const __m256i umask = _mm256_set1_epi8((1 << ub) - 1);
    const __m256i T0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)tab));
    __m256i T1 = T0, T2 = T0, T3 = T0;
    if (nl * (1 << ub) > 16) T1 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab + 16)));
    if (nl * (1 << ub) > 32) {
        T2 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab + 32)));
        T3 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab + 48)));
    }
    // index bits: u = low ub, then c, P1, P2 (P2, P3 beyond 4 index bits go via blend)
    const int sh_c = ub;
#pragma GCC unroll 8
    for (int g = 0; g < 8; g++) {
        // g = 0: R0 / 0 (rows 0, 1); 1: R1 / 0 (2, 3); 2: R0 / 8 (4, 5); 3: R1 / 8 (6, 7); 4..7 — same for R2, R3
        const __m256i src = R[(g >> 2) * 2 + (g & 1)];
        const int off = ((g >> 1) & 1) * 8;
        // byte i of a lane takes byte (i >> 3) of the word: c = bytes 0, 1; P1 = 2, 3; P2 = 4, 5; P3 = 6, 7
        const __m256i selc = _mm256_add_epi8(_mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                                              0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1),
                                             _mm256_set1_epi8((char)off));
        __m256i mc = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(src, selc), bitm), bitm);
        __m256i mp1 = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(src, _mm256_add_epi8(selc, _mm256_set1_epi8(2))), bitm), bitm);
        __m256i uu = _mm256_and_si256(_mm256_srli_epi16(_mm256_loadu_si256((const __m256i*)(tl->u + 32 * (g / gpb))), ub * (g % gpb)), umask);
        __m256i idx = _mm256_or_si256(uu, _mm256_and_si256(mc, _mm256_set1_epi8(1 << sh_c)));
        __m256i w;
        const int ibits = ub + 2;  // u, c, P1
        if (ibits <= 4) {
            idx = _mm256_or_si256(idx, _mm256_and_si256(mp1, _mm256_set1_epi8(1 << (sh_c + 1))));
            if (nl == 4) {
                w = _mm256_shuffle_epi8(T0, idx);
            } else {
                __m256i mp2 = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(src, _mm256_add_epi8(selc, _mm256_set1_epi8(4))), bitm), bitm);
                if (ibits == 4 || nl == 8) {
                    if (ibits + 1 <= 4) {
                        idx = _mm256_or_si256(idx, _mm256_and_si256(mp2, _mm256_set1_epi8(1 << (sh_c + 2))));
                        if (nl == 8) w = _mm256_shuffle_epi8(T0, idx);
                        else {
                            __m256i mp3 = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(src, _mm256_add_epi8(selc, _mm256_set1_epi8(6))), bitm), bitm);
                            w = _mm256_blendv_epi8(_mm256_shuffle_epi8(T0, idx), _mm256_shuffle_epi8(T1, idx), mp3);
                        }
                    } else {
                        __m256i w0 = _mm256_blendv_epi8(_mm256_shuffle_epi8(T0, idx), _mm256_shuffle_epi8(T1, idx), mp2);
                        if (nl == 8) w = w0;
                        else {
                            __m256i mp3 = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(src, _mm256_add_epi8(selc, _mm256_set1_epi8(6))), bitm), bitm);
                            __m256i w1 = _mm256_blendv_epi8(_mm256_shuffle_epi8(T2, idx), _mm256_shuffle_epi8(T3, idx), mp2);
                            w = _mm256_blendv_epi8(w0, w1, mp3);
                        }
                    }
                } else {
                    w = _mm256_shuffle_epi8(T0, idx);  // unused
                }
            }
        } else {
            w = _mm256_shuffle_epi8(T0, idx);  // unused (K > 3)
        }
        __m256i p = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(a2, w));
        __m256i s = _mm256_madd_epi16(p, _mm256_set1_epi16(1));
        // pair scale: [s_{2g} x4 | s_{2g+1} x4]
        __m256 scs = (g < 4) ? sc_lo : sc_hi;
        const int r0 = (2 * g) & 7;
        __m256 sp = _mm256_permutevar8x32_ps(scs, _mm256_setr_epi32(r0, r0, r0, r0, r0 + 1, r0 + 1, r0 + 1, r0 + 1));
        acc[g] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), sp, acc[g]);
    }
}

static inline float hsum4(__m128 v)
{
    v = _mm_add_ps(v, _mm_movehl_ps(v, v));
    v = _mm_add_ss(v, _mm_movehdup_ps(v));
    return _mm_cvtss_f32(v);
}

static inline __attribute__((always_inline)) void simd_tcq_k(const tile_t* T, const int8_t* tab, const float* gains,
                                                             const int8_t* a, float* y, const int K, const int nl)
{
    for (int b = 0; b < M_ROWS / 16; b++) {
        __m256 acc[8];
        for (int g = 0; g < 8; g++) acc[g] = _mm256_setzero_ps();
        for (int j = 0; j < NTILE; j++) {
            __m256i a2 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(a + 16 * j)));
            tcq_tile(&T[b * NTILE + j], a2, tab, gains, acc, K, nl);
        }
        for (int g = 0; g < 8; g++) {
            y[b * 16 + 2 * g] = hsum4(_mm256_castps256_ps128(acc[g]));
            y[b * 16 + 2 * g + 1] = hsum4(_mm256_extractf128_ps(acc[g], 1));
        }
    }
}


// --- labels via tables: label bytes of a tile row are linear (GF(2)) in the code-bit window X = c_r << 16 | c_{r-1}
// (bits 7..31): XOR of three 16-byte vectors over chunks of X — history (9 bits), low and high byte of the row.
// Label bits are placed directly at their index positions: c, P1, ... at positions ub..3, beyond 4 bits into bit 7, then 6 (for blendv).
static uint8_t LH[512][16], L0[256][16], L1[256][16];

static void build_luts(int K, int nl)
{
    const int ub = K - 1, lbits = nl == 4 ? 2 : nl == 8 ? 3 : 4;
    int pos[4], masks[4] = {0, MASK_M, MASK_2, MASK_3};
    for (int i = 0, p = ub, extra = 0; i < lbits; i++) pos[i] = p <= 3 ? p++ : (extra++ == 0 ? 7 : 6);
    // contribution of bit b of window X to the byte of weight t
    uint8_t contrib[32][16];
    memset(contrib, 0, sizeof(contrib));
    for (int t = 0; t < 16; t++) {
        contrib[16 + t][t] ^= 1 << pos[0];
        for (int i = 1; i < lbits; i++)
            for (int j = 0; j < 9; j++)
                if ((masks[i] >> j) & 1) contrib[16 + t - 1 - j][t] ^= 1 << pos[i];
    }
    for (int v = 0; v < 512; v++)
        for (int t = 0; t < 16; t++) {
            uint8_t h = 0;
            for (int k = 0; k < 9; k++) if ((v >> k) & 1) h ^= contrib[7 + k][t];
            LH[v][t] = h;
            if (v < 256) {
                uint8_t a = 0, b = 0;
                for (int k = 0; k < 8; k++) if ((v >> k) & 1) { a ^= contrib[16 + k][t]; b ^= contrib[24 + k][t]; }
                L0[v][t] = a;
                L1[v][t] = b;
            }
        }
}

static inline __m256i lut_pair(const uint8_t (*L)[16], int i0, int i1)
{
    return _mm256_loadu2_m128i((const __m128i*)L[i1], (const __m128i*)L[i0]);
}

static int PP_MODE = 0;  // 1: pair parity (half-integer formats) — the low bit of u of the odd weight is taken from the even one

static inline __attribute__((always_inline)) void tcq_tile_lut(const tile_t* tl, __m256i a2, const int8_t* tab,
                                                              const float* gains, __m256* acc, const int K, const int nl,
                                                              const int pp)
{
    const int ub = K - 1, ub2 = ub > 2 ? 2 : ub, gpb = 8 / ub2, lbits = nl == 4 ? 2 : nl == 8 ? 3 : 4;
    const int nsel = ub + lbits > 4 ? ub + lbits - 4 : 0;
    const __m256i umask = _mm256_set1_epi8((1 << ub2) - 1), m4 = _mm256_set1_epi8(0x0f);
    const __m256i T0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)tab));
    __m256i T1 = T0, T2 = T0, T3 = T0;
    if (nsel >= 1) T1 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab + 16)));
    if (nsel >= 2) {
        T2 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab + 32)));
        T3 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab + 48)));
    }
    const __m256i sh_lo = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14), sh_hi = _mm256_setr_epi32(16, 18, 20, 22, 24, 26, 28, 30);
    __m256i hv = _mm256_set1_epi32((int)tl->hdr), three = _mm256_set1_epi32(3);
    __m256 gv = _mm256_castps128_ps256(_mm_loadu_ps(gains));
    __m256 sc_lo = _mm256_permutevar8x32_ps(gv, _mm256_and_si256(_mm256_srlv_epi32(hv, sh_lo), three));
    __m256 sc_hi = _mm256_permutevar8x32_ps(gv, _mm256_and_si256(_mm256_srlv_epi32(hv, sh_hi), three));
    const uint16_t* c = tl->c;
#pragma GCC unroll 8
    for (int g = 0; g < 8; g++) {
        const int r0 = 2 * g, r1 = 2 * g + 1;
        __m256i v = lut_pair(LH, c[(r0 + 15) & 15] >> 7, c[r0] >> 7);
        v = _mm256_xor_si256(v, lut_pair(L0, c[r0] & 0xff, c[r1] & 0xff));
        v = _mm256_xor_si256(v, lut_pair(L1, c[r0] >> 8, c[r1] >> 8));
        __m256i uu = _mm256_and_si256(_mm256_srli_epi16(_mm256_loadu_si256((const __m256i*)(tl->u + 32 * (g / gpb))), ub2 * (g % gpb)), umask);
        if (K == 4)  // third bit: bit g of byte u1 -> bit 2 of the index
        {
            const __m256i u1 = _mm256_loadu_si256((const __m256i*)tl->u1);
            const __m256i sh = g >= 2 ? _mm256_srli_epi16(u1, g - 2) : _mm256_slli_epi16(u1, 2 - g);
            uu = _mm256_or_si256(uu, _mm256_and_si256(sh, _mm256_set1_epi8(4)));
        }
        if (K == 5)  // bits 2-3: the same 2-bit fields in the second array, shifted by 2
            uu = _mm256_or_si256(uu, _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(
                     _mm256_loadu_si256((const __m256i*)(tl->u2 + 32 * (g / gpb))), ub2 * (g % gpb)), umask), 2));
        if (pp) {  // odd weight of a pair: low bit of u = low bit of the even neighbor (byte shift within the row)
            const __m256i odd = _mm256_set1_epi16(0x0100);
            const __m256i lsb_even = _mm256_slli_si256(_mm256_and_si256(uu, _mm256_set1_epi16(0x0001)), 1);
            uu = _mm256_or_si256(_mm256_andnot_si256(odd, uu), lsb_even);
        }
        __m256i idx = _mm256_or_si256(v, uu), lo = _mm256_and_si256(idx, m4), w;
        if (nsel == 0) {
            w = _mm256_shuffle_epi8(T0, lo);
        } else if (nsel == 1) {
            w = _mm256_blendv_epi8(_mm256_shuffle_epi8(T0, lo), _mm256_shuffle_epi8(T1, lo), idx);
        } else {
            __m256i w0 = _mm256_blendv_epi8(_mm256_shuffle_epi8(T0, lo), _mm256_shuffle_epi8(T1, lo), idx);
            __m256i w1 = _mm256_blendv_epi8(_mm256_shuffle_epi8(T2, lo), _mm256_shuffle_epi8(T3, lo), idx);
            w = _mm256_blendv_epi8(w0, w1, _mm256_slli_epi16(idx, 1));
        }
        __m256i p = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(a2, w));
        __m256i s = _mm256_madd_epi16(p, _mm256_set1_epi16(1));
        __m256 scs = (g < 4) ? sc_lo : sc_hi;
        const int q = r0 & 7;
        __m256 sp = _mm256_permutevar8x32_ps(scs, _mm256_setr_epi32(q, q, q, q, q + 1, q + 1, q + 1, q + 1));
        acc[g] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), sp, acc[g]);
    }
}

static inline __attribute__((always_inline)) void simd_tcq_lut_k(const tile_t* T, const int8_t* tab, const float* gains,
                                                                 const int8_t* a, float* y, const int K, const int nl,
                                                                 const int PP)
{
    for (int b = 0; b < M_ROWS / 16; b++) {
        __m256 acc[8];
        for (int g = 0; g < 8; g++) acc[g] = _mm256_setzero_ps();
        for (int j = 0; j < NTILE; j++) {
            __m256i a2 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(a + 16 * j)));
            tcq_tile_lut(&T[b * NTILE + j], a2, tab, gains, acc, K, nl, PP);
        }
        for (int g = 0; g < 8; g++) {
            y[b * 16 + 2 * g] = hsum4(_mm256_castps256_ps128(acc[g]));
            y[b * 16 + 2 * g + 1] = hsum4(_mm256_extractf128_ps(acc[g], 1));
        }
    }
}

// --- "tables + integer gains" variant: gain * 64 in int16 goes into madd, the sum accumulates in int32 over a 256-block (16 tiles),
// float conversion and FMA happen once per block rather than for every row pair of every tile
static int16_t GAIN_I[4];

static inline __attribute__((always_inline)) void tcq_tile_lut_i(const tile_t* tl, __m256i a2, const int8_t* tab,
                                                                __m256i* acc, const int K, const int nl)
{
    const int ub = K - 1, ub2 = ub > 2 ? 2 : ub, gpb = 8 / ub2, lbits = nl == 4 ? 2 : nl == 8 ? 3 : 4;
    const int nsel = ub + lbits > 4 ? ub + lbits - 4 : 0;
    const __m256i umask = _mm256_set1_epi8((1 << ub2) - 1), m4 = _mm256_set1_epi8(0x0f);
    const __m256i T0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)tab));
    __m256i T1 = T0, T2 = T0, T3 = T0;
    if (nsel >= 1) T1 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab + 16)));
    if (nsel >= 2) {
        T2 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab + 32)));
        T3 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tab + 48)));
    }
    const uint16_t* c = tl->c;
    const uint32_t hdr = tl->hdr;
#pragma GCC unroll 8
    for (int g = 0; g < 8; g++) {
        const int r0 = 2 * g, r1 = 2 * g + 1;
        __m256i v = lut_pair(LH, c[(r0 + 15) & 15] >> 7, c[r0] >> 7);
        v = _mm256_xor_si256(v, lut_pair(L0, c[r0] & 0xff, c[r1] & 0xff));
        v = _mm256_xor_si256(v, lut_pair(L1, c[r0] >> 8, c[r1] >> 8));
        __m256i uu = _mm256_and_si256(_mm256_srli_epi16(_mm256_loadu_si256((const __m256i*)(tl->u + 32 * (g / gpb))), ub2 * (g % gpb)), umask);
        if (K == 4) {
            const __m256i u1 = _mm256_loadu_si256((const __m256i*)tl->u1);
            const __m256i sh = g >= 2 ? _mm256_srli_epi16(u1, g - 2) : _mm256_slli_epi16(u1, 2 - g);
            uu = _mm256_or_si256(uu, _mm256_and_si256(sh, _mm256_set1_epi8(4)));
        }
        if (K == 5)  // bits 2-3: the same 2-bit fields in the second array, shifted by 2
            uu = _mm256_or_si256(uu, _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(
                     _mm256_loadu_si256((const __m256i*)(tl->u2 + 32 * (g / gpb))), ub2 * (g % gpb)), umask), 2));
        __m256i idx = _mm256_or_si256(v, uu), lo = _mm256_and_si256(idx, m4), w;
        if (nsel == 0) {
            w = _mm256_shuffle_epi8(T0, lo);
        } else if (nsel == 1) {
            w = _mm256_blendv_epi8(_mm256_shuffle_epi8(T0, lo), _mm256_shuffle_epi8(T1, lo), idx);
        } else {
            __m256i w0 = _mm256_blendv_epi8(_mm256_shuffle_epi8(T0, lo), _mm256_shuffle_epi8(T1, lo), idx);
            __m256i w1 = _mm256_blendv_epi8(_mm256_shuffle_epi8(T2, lo), _mm256_shuffle_epi8(T3, lo), idx);
            w = _mm256_blendv_epi8(w0, w1, _mm256_slli_epi16(idx, 1));
        }
        // gains of the pair's rows: lane 0 = row 2g, lane 1 = 2g + 1 (int16, 8 per lane)
        const __m256i gp = _mm256_setr_m128i(_mm_set1_epi16(GAIN_I[(hdr >> (2 * r0)) & 3]), _mm_set1_epi16(GAIN_I[(hdr >> (2 * r1)) & 3]));
        __m256i p = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(a2, w));
        acc[g] = _mm256_add_epi32(acc[g], _mm256_madd_epi16(p, gp));
    }
}

static inline __attribute__((always_inline)) void simd_tcq_lut_i_k(const tile_t* T, const int8_t* tab, const float* gains,
                                                                   const int8_t* a, float* y, const int K, const int nl)
{
    (void)gains;
    const __m256 dscale = _mm256_set1_ps(1.0f / 64.0f);  // block d (here 1) * 1/64 of the integer gains
    for (int b = 0; b < M_ROWS / 16; b++) {
        __m256 accf[8];
        for (int g = 0; g < 8; g++) accf[g] = _mm256_setzero_ps();
        for (int j0 = 0; j0 < NTILE; j0 += 16) {  // 256-block: 16 tiles
            __m256i acc[8];
            for (int g = 0; g < 8; g++) acc[g] = _mm256_setzero_si256();
            for (int j = j0; j < j0 + 16; j++) {
                __m256i a2 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(a + 16 * j)));
                tcq_tile_lut_i(&T[b * NTILE + j], a2, tab, acc, K, nl);
            }
            for (int g = 0; g < 8; g++) accf[g] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[g]), dscale, accf[g]);
        }
        for (int g = 0; g < 8; g++) {
            y[b * 16 + 2 * g] = hsum4(_mm256_castps256_ps128(accf[g]));
            y[b * 16 + 2 * g + 1] = hsum4(_mm256_extractf128_ps(accf[g], 1));
        }
    }
}

// specializations: K and nl are compile-time constants (as in GGML kernels), no branches or divisions in the hot loop
#define SPEC(K, NL) \
    static void simd_tcq_##K##_##NL(const tile_t* T, const int8_t* tab, const float* g, const int8_t* a, float* y) \
    { simd_tcq_k(T, tab, g, a, y, K, NL); } \
    static void simd_tcq_lut_##K##_##NL(const tile_t* T, const int8_t* tab, const float* g, const int8_t* a, float* y) \
    { simd_tcq_lut_k(T, tab, g, a, y, K, NL, 0); } \
    static void simd_tcq_lut_i_##K##_##NL(const tile_t* T, const int8_t* tab, const float* g, const int8_t* a, float* y) \
    { simd_tcq_lut_i_k(T, tab, g, a, y, K, NL); }
SPEC(3, 4) SPEC(3, 8) SPEC(3, 16) SPEC(2, 16) SPEC(4, 4) SPEC(4, 8) SPEC(5, 4)
// half-integer: table K = 3 / 8 subsets (TCQ25X) and K = 4 / 4 (TCQ35X) with recovery of the pair's low bit
static void simd_tcq_lut_pp_3_8(const tile_t* T, const int8_t* tab, const float* g, const int8_t* a, float* y)
{ simd_tcq_lut_k(T, tab, g, a, y, 3, 8, 1); }
static void simd_tcq_lut_pp_4_4(const tile_t* T, const int8_t* tab, const float* g, const int8_t* a, float* y)
{ simd_tcq_lut_k(T, tab, g, a, y, 4, 4, 1); }
typedef void (*kern_t)(const tile_t*, const int8_t*, const float*, const int8_t*, float*);

// --- IQ4_NL-like kernel: block of 32 weights = 16 bytes of nibbles + float scale ---
typedef struct { uint8_t q[16]; float d; } iq4_t;

static void simd_iq4(const iq4_t* B, const int8_t* tab16, const int8_t* a, float* y)
{
    const __m256i T = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)tab16));
    const __m256i m4 = _mm256_set1_epi8(0x0f), one = _mm256_set1_epi16(1);
    for (int r = 0; r < M_ROWS; r++) {
        __m256 acc = _mm256_setzero_ps();
        const iq4_t* br = B + (size_t)r * (K_COLS / 32);
        for (int j = 0; j < K_COLS / 32; j++) {
            __m128i q = _mm_loadu_si128((const __m128i*)br[j].q);
            __m256i qq = _mm256_inserti128_si256(_mm256_castsi128_si256(q), _mm_srli_epi16(q, 4), 1);
            __m256i w = _mm256_shuffle_epi8(T, _mm256_and_si256(qq, m4));
            __m256i av = _mm256_loadu_si256((const __m256i*)(a + 32 * j));
            __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(av, w)), one);
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(br[j].d), acc);
        }
        __m128 v = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        y[r] = hsum4(v);
    }
}

static void fill(void* p, size_t n, unsigned seed)
{
    srand(seed);
    for (size_t i = 0; i < n; i++) ((uint8_t*)p)[i] = rand() & 0xff;
}

int main(void)
{
    const int ntiles = M_ROWS / 16 * NTILE;
    tile_t* T = aligned_alloc(64, sizeof(tile_t) * ntiles);
    fill(T, sizeof(tile_t) * ntiles, 1);
    int8_t a[K_COLS], tab[64];
    srand(2);
    for (int i = 0; i < K_COLS; i++) a[i] = (int8_t)((rand() % 255) - 127);
    for (int i = 0; i < 64; i++) tab[i] = (int8_t)((rand() % 255) - 127);
    const float gains[4] = {51 / 64.0f, 61 / 64.0f, 70 / 64.0f, 83 / 64.0f};  // exactly representable as gain * 64
    for (int i = 0; i < 4; i++) GAIN_I[i] = (int16_t)lrintf(gains[i] * 64);
    float *y = malloc(4 * M_ROWS), *yr = malloc(4 * M_ROWS);
    const int iters = 200;

    struct { const char* name; int K, nl; double bpw; kern_t f[3]; } cfg[] = {
        {"TCQ K=3 nl=4 (16 ent.)", 3, 4, 3.0 + 2.0 / 16, {simd_tcq_3_4, simd_tcq_lut_3_4, simd_tcq_lut_i_3_4}},
        {"TCQ K=3 nl=8 (32 ent.)", 3, 8, 3.0 + 2.0 / 16, {simd_tcq_3_8, simd_tcq_lut_3_8, simd_tcq_lut_i_3_8}},
        {"TCQ K=3 nl=16 (64 ent.)", 3, 16, 3.0 + 2.0 / 16, {simd_tcq_3_16, simd_tcq_lut_3_16, simd_tcq_lut_i_3_16}},
        {"TCQ K=2 nl=16 (32 ent.)", 2, 16, 2.0 + 2.0 / 16, {simd_tcq_2_16, simd_tcq_lut_2_16, simd_tcq_lut_i_2_16}},
        {"TCQ K=4 nl=4 (32 ent.)", 4, 4, 4.0 + 2.0 / 16, {simd_tcq_4_4, simd_tcq_lut_4_4, simd_tcq_lut_i_4_4}},
        {"TCQ K=4 nl=8 (64 ent.)", 4, 8, 4.0 + 2.0 / 16, {simd_tcq_4_8, simd_tcq_lut_4_8, simd_tcq_lut_i_4_8}},
        {"TCQ K=5 nl=4 (64 ent.)", 5, 4, 5.0 + 2.0 / 16, {simd_tcq_5_4, simd_tcq_lut_5_4, simd_tcq_lut_i_5_4}},
    };
    for (unsigned ci = 0; ci < sizeof(cfg) / sizeof(cfg[0]); ci++)
        for (int var = 1; var < 3; var++) {  // 1 = label tables, 2 = + integer gains (variant 0, parities, see history)  // 0 = parities + bit-plane unpacking (K <= 3), 1 = labels via tables
            const int K = cfg[ci].K, nl = cfg[ci].nl;
            build_luts(K, nl);
            ref_tcq(T, tab, gains, a, yr, K, nl);
            cfg[ci].f[var](T, tab, gains, a, y);
            double err = 0, nrm = 0;
            for (int i = 0; i < M_ROWS; i++) { err += fabs(y[i] - yr[i]); nrm += fabs(yr[i]); }
            double t0 = now();
            for (int it = 0; it < iters; it++) {
                cfg[ci].f[var](T, tab, gains, a, y);
                __asm__ volatile("" ::: "memory");
            }
            double dt = (now() - t0) / iters, gw = (double)M_ROWS * K_COLS / dt * 1e-9;
            printf("%-26s %-9s error %.1e | %.2f G weights/s | %.1f GB/s at %.3f bits/weight\n", cfg[ci].name,
                   var == 2 ? "+ int.gain" : "tables", err / nrm, gw, gw * cfg[ci].bpw / 8, cfg[ci].bpw);
        }
    // half-integer: speed only (the reference was not changed — only the cost of recovering the low bit is measured)
    struct { const char* name; kern_t f; double bpw; } ppc[] = {
        {"TCQ K=2.5 (table K=3/8)", simd_tcq_lut_pp_3_8, 2.5 + 2.0 / 16}, {"TCQ K=3.5 (table K=4/4)", simd_tcq_lut_pp_4_4, 3.5 + 2.0 / 16}};
    for (int ci = 0; ci < 2; ci++) {
        build_luts(ci ? 4 : 3, ci ? 4 : 8);
        double t0 = now();
        for (int it = 0; it < iters; it++) { ppc[ci].f(T, tab, gains, a, y); __asm__ volatile("" ::: "memory"); }
        double dt = (now() - t0) / iters, gw = (double)M_ROWS * K_COLS / dt * 1e-9;
        printf("%-26s half-integer        | %.2f G weights/s | %.1f GB/s at %.3f bits/weight\n", ppc[ci].name, gw, gw * ppc[ci].bpw / 8, ppc[ci].bpw);
    }
    iq4_t* B = aligned_alloc(64, sizeof(iq4_t) * M_ROWS * (K_COLS / 32));
    fill(B, sizeof(iq4_t) * M_ROWS * (K_COLS / 32), 3);
    for (int i = 0; i < M_ROWS * (K_COLS / 32); i++) B[i].d = 0.01f;
    double t0 = now();
    for (int it = 0; it < iters; it++) {
        simd_iq4(B, tab, a, y);
        __asm__ volatile("" ::: "memory");
    }
    double dt = (now() - t0) / iters, gw = (double)M_ROWS * K_COLS / dt * 1e-9;
    printf("%-26s                 | %.2f G weights/s | %.1f GB/s at %.3f bits/weight\n", "IQ4_NL-like", gw, gw * 4.5 / 8, 4.5);
    return 0;
}
