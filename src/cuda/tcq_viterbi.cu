// Viterbi for Ungerboeck TCQ with a gain header (like viterbi_ub in src/tcq_proto.py).
// One block per tile of T weights; BLOCK threads, each owns SPT = S / BLOCK trellis memory states (S = 2^nu) and all G
// gains. Traceback runs in the same block, one sub-block chunk at a time via shared (see the end of the kernel).
// Predecessors of state s': p_h = h * S/2 + (s' >> 1), h in {0, 1}; label = (parity((s' >> 1) & m) << 1) | (s' & 1).
// The tile's x and w live in shared; branch errors (G x 4 per step) are computed in shared in a batch per sub-block.
// S = 1024, BLOCK = 256: two steps per barrier (radix-4 butterfly), 3x fewer shared accesses.
// Traceback pointers: one bit per (step, gain, state) in global memory (__ballot_sync per warp).
// G < Gfull: on sub-block j, slot g takes gain gains[gcand[n, j, g]] — Viterbi runs only over G candidates out of Gfull,
// header and decoder are unchanged. Candidates are picked by the forward-pass prologue: the G gains with the lowest weighted
// scalar-quantization error of the sub-block using the merged table (all 4 subsets, no trellis constraints).
// Build: nvcc -O3 -arch=sm_120 -shared -Xcompiler -fPIC -o libtcq_viterbi.so tcq_viterbi.cu
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>

#define MAX_G 16  // gains in the header (G > 8 only via candidates)
#define MAX_U 128  // levels per subset (K <= 8); at most 4 * MAX_U in the whole table
#define MAX_SUBS 32  // sub-blocks per tile
#define MAX_T 512    // samples per tile (traceback keeps the path in local memory)

struct TcqArgs {
    const float* x;         // [N, T]
    const float* w;         // [N, T]
    const float* tab;       // [4 * U]
    const float* gains;     // [Gfull]
    float* rec;             // [N, T]
    uint32_t* bp;           // [N, T, G, S / 32]
    uint8_t* gsel;          // [N, T / sub, S]
    const int* init_st;     // [N] initial state of the tile (< 0 = free) or nullptr
    int* end_st;            // [N] final state of the chosen path or nullptr
    const float* tab_hi;    // [4 * U_hi] half-integer bitrate: table for positions with K + 1 bits (or = tab)
    const float* gains_hi;  // [Gfull]
    uint8_t* gcand;         // [N, T / sub, G] (prologue output) or nullptr when G == Gfull
    int* best;              // [N] g * S + s of the best path's end (from forward pass to traceback)
    int* code;              // [N, T] (gain << 16) | (label * U + u) of the chosen path or nullptr (for table training)
    int* mid_st;            // [N] path state before sample mid_t or nullptr (first tail-biting pass)
    int mid_t;
    int tail;               // 1: final state must equal init_st (tail-biting, second pass)
    int m2, m3, m4, nl;     // extended labels: nl = 4 / 8 / 16 / 32 subsets, extra parities by masks m2..m4
    int N, T, S, U, G, Gfull, m, sub, U_hi, mix;  // mix > 0: position t is "high" if (t / mix) & 1
    int ld, rot;            // row stride of x / w / rec / code and cyclic shift: tile sample t is element (t + rot) % ld
    uint32_t* pbits;        // [N, T / 2, G, S / 32] pair parity (pp mode) or nullptr
    int pp;                 // 1: half-integer rate — in a pair (2i, 2i+1) the low bits of the non-code indices are equal (D2 lattice)
};

__device__ __forceinline__ float branch_err(float xt, float gain, const float* tab_lab, int U)
{
    float e = 3.4e38f;
    for (int u = 0; u < U; u++) {
        float d = xt - gain * tab_lab[u];
        e = fminf(e, d * d);
    }
    return e;
}

// bit i -> bits 2i, 2i+1 (16 bits -> 32)
__device__ __forceinline__ uint32_t spread2(uint32_t x)
{
    x = (x | (x << 8)) & 0x00ff00ffu;
    x = (x | (x << 4)) & 0x0f0f0f0fu;
    x = (x | (x << 2)) & 0x33333333u;
    x = (x | (x << 1)) & 0x55555555u;
    return x | (x << 1);
}

// bit i -> bit 4i (8 bits -> 32)
__device__ __forceinline__ uint32_t spread4(uint32_t x)
{
    x = (x | (x << 12)) & 0x000f000fu;
    x = (x | (x << 6)) & 0x03030303u;
    return (x | (x << 3)) & 0x11111111u;
}

// same, over the levels of the subset with parity p (u = p, p + 2, ...)
__device__ __forceinline__ float branch_err_par(float xt, float gain, const float* tab_lab, int U, int p)
{
    float e = 3.4e38f;
    for (int u = p; u < U; u += 2) {
        float d = xt - gain * tab_lab[u];
        e = fminf(e, d * d);
    }
    return e;
}

__device__ __forceinline__ int tcq_label(int s, int m) { return ((__popc((s >> 1) & m) & 1) << 1) | (s & 1); }

// label with nl subsets: low 2 bits as in Ungerboeck, high bits are parities of (s >> 1) by masks m2, m3, m4
__device__ __forceinline__ int tcq_label_x(int s, const TcqArgs& a, int nl)
{
    int lb = tcq_label(s, a.m);
    if (nl >= 8) lb |= (__popc((s >> 1) & a.m2) & 1) << 2;
    if (nl >= 16) lb |= (__popc((s >> 1) & a.m3) & 1) << 3;
    if (nl >= 32) lb |= (__popc((s >> 1) & a.m4) & 1) << 4;
    return lb;
}

template <int BLOCK, int SPT, int G, int NL, int MD>
__global__ void __launch_bounds__(BLOCK) tcq_forward_kernel(const TcqArgs a)
{
    constexpr int S = BLOCK * SPT;
    constexpr int WORDS = S / 32;
    constexpr int HALF = S / 2;
    const int T = a.T, sub = a.sub, U = a.U, U_hi = a.U_hi, mix = a.mix;
    extern __shared__ float smem[];
    float* dp0 = smem;                    // [G, S]
    float* dp1 = smem + G * S;            // [G, S]
    // MD: 0 = plain, 1 = pair parity (K - 0.5), 2 = even sum of low bits in a group of 4 (D4 lattice, K - 0.25)
    // MD = 3: all low bits of a group of 4 are equal (K - 0.75; the class is the shared low bit, no separate bit b needed)
    constexpr bool PP = MD == 1, G4 = MD == 2 || MD == 3, EQ = MD == 3;
    constexpr int NP = MD ? 2 : 1;          // parity classes per label
    constexpr int NPL = 9;                  // bit planes per group (G4), see below
    float* dp2 = smem + 2 * G * S;        // G4: third buffer (classes after the first pair of the group)
    float* s_err = smem + (G4 ? 3 : 2) * G * S;  // [sub, G, NL, NP]
    float* s_x = s_err + sub * G * NL * NP;  // [T]
    float* s_w = s_x + T;                 // [T]
    __shared__ float s_tab[4 * MAX_U], s_tab_hi[4 * MAX_U], s_gain[MAX_G], s_gain_hi[MAX_G];
    __shared__ uint8_t s_gc[MAX_SUBS * G];
    __shared__ float s_ce[MAX_SUBS * MAX_G];
    __shared__ float red_v[BLOCK / 32];
    __shared__ int red_i[BLOCK / 32];

    const int n = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    uint32_t* bpn = a.bp + (size_t)n * T * G * WORDS;
    uint8_t* gsn = a.gsel + (size_t)n * (T / sub) * S;
    uint32_t* ppn = PP ? a.pbits + (size_t)n * (T / 2) * G * WORDS : nullptr;
    // G4: [T / 4, G, NPL, WORDS]: 0..1 = predecessor bit of the 1st pair's output per class, 2..3 = low bit of its first
    // sample per class, 4..5 = predecessor bit of the 2nd pair's intermediate per class, 6 = class, 7 = predecessor
    // bit of the 2nd pair's output, 8 = low bit of its first sample (for the chosen class)
    uint32_t* gpn = G4 ? a.pbits + (size_t)n * (T / 4) * G * NPL * WORDS : nullptr;
    const int nsub = T / sub, Gf = a.Gfull;
    const bool cand = a.gcand != nullptr;

    for (int i = tid; i < T; i += BLOCK) {
        const size_t j = (size_t)n * a.ld + (i + a.rot) % a.ld;
        s_x[i] = a.x[j];
        s_w[i] = a.w[j];
    }
    for (int i = tid; i < NL * U; i += BLOCK) s_tab[i] = a.tab[i];
    for (int i = tid; i < NL * U_hi; i += BLOCK) s_tab_hi[i] = a.tab_hi[i];
    if (tid < a.Gfull) { s_gain[tid] = a.gains[tid]; s_gain_hi[tid] = a.gains_hi[tid]; }

    int labs[SPT];
    #pragma unroll
    for (int k = 0; k < SPT; k++) labs[k] = tcq_label_x(tid + k * BLOCK, a, NL);
    int r4m[4], r4o[4];  // labels of the intermediate and output states of the radix-4 butterfly (SPT == 4)
    #pragma unroll
    for (int c = 0; c < 4; c++) {
        r4m[c] = tcq_label_x((c < 2 ? 0 : HALF) + 2 * tid + (c & 1), a, NL);
        r4o[c] = tcq_label_x(4 * tid + c, a, NL);
    }
    // initial state s_0 is given (carried over from the previous tile): at step 0 the allowed states are those
    // s_0 transitions to with one code bit: s = ((s_0 << 1) | c) & (S - 1)
    const int s0 = a.init_st ? a.init_st[n] : -1;
    __syncthreads();
    if (cand) {
        for (int i = tid; i < nsub * Gf; i += BLOCK) {
            const int j = i / Gf, gi = i % Gf;
            float e = 0.f;
            for (int t = j * sub; t < (j + 1) * sub; t++) {
                const bool hi = mix > 0 && ((t / mix) & 1);
                e += (hi ? branch_err(s_x[t], s_gain_hi[gi], s_tab_hi, NL * U_hi)
                         : branch_err(s_x[t], s_gain[gi], s_tab, NL * U)) * s_w[t];
            }
            s_ce[i] = e;
        }
        __syncthreads();
        if (tid < nsub) {  // G smallest (ties go to the lower index)
            uint32_t used = 0;
            for (int c = 0; c < G; c++) {
                int bg = 0;
                float be = 3.4e38f;
                for (int gi = 0; gi < Gf; gi++)
                    if (!((used >> gi) & 1) && s_ce[tid * Gf + gi] < be) { be = s_ce[tid * Gf + gi]; bg = gi; }
                used |= 1u << bg;
                s_gc[tid * G + c] = (uint8_t)bg;
                a.gcand[((size_t)n * nsub + tid) * G + c] = (uint8_t)bg;
            }
        }
        __syncthreads();
    }

    float* cur = dp0;
    float* nxt = dp1;
    for (int t0 = 0; t0 < T; t0 += sub) {
        if (t0 > 0 && G > 1) {
            // sub-block boundary: each state takes the best over gains (each thread owns its states)
            #pragma unroll
            for (int k = 0; k < SPT; k++) {
                int s = tid + k * BLOCK;
                float best = cur[s];
                int bg = 0;
                #pragma unroll
                for (int g = 1; g < G; g++) {
                    float v = cur[g * S + s];
                    if (v < best) { best = v; bg = g; }
                }
                gsn[(size_t)(t0 / sub) * S + s] = (uint8_t)bg;
                #pragma unroll
                for (int g = 0; g < G; g++) cur[g * S + s] = best;
            }
        }
        // branch errors of the sub-block; the previous sub-block no longer reads them (everyone passed the last step's barrier)
        for (int i = tid; i < sub * G * NL * NP; i += BLOCK) {
            const int t = t0 + i / (G * NL * NP), g = (i / (NL * NP)) % G, lb = (i / NP) % NL, p = i % NP;
            const int gi = cand ? (int)s_gc[(t0 / sub) * G + g] : g;
            const bool hi = mix > 0 && ((t / mix) & 1);
            float e;
            if (MD) e = branch_err_par(s_x[t], s_gain[gi], s_tab + lb * U, U, p);
            else e = hi ? branch_err(s_x[t], s_gain_hi[gi], s_tab_hi + lb * U_hi, U_hi)
                        : branch_err(s_x[t], s_gain[gi], s_tab + lb * U, U);
            s_err[i] = e * s_w[t];
        }
        __syncthreads();
        for (int t = t0; t < t0 + sub;) {
            const float* err = s_err + (t - t0) * G * NL * NP;
            int nstep = 1;
            if (G4 && t == 0) {
                // group 0, first pair (0, 1): intermediates by low bit b — dp1 (b = 0), dp2 (b = 1); outputs
                // of step 1 per class c = b0 ^ b1 — radix-2, via registers; classes stay in dp1 / dp2
                const float* err1 = err + G * NL * 2;
                #pragma unroll
                for (int k = 0; k < SPT; k++) {
                    int s = tid + k * BLOCK;
                    bool ok = s0 < 0 || (s >> 1) == (s0 & (HALF - 1));
                    #pragma unroll
                    for (int g = 0; g < G; g++) {
                        dp1[g * S + s] = ok ? err[(g * NL + labs[k]) * 2] : 3.4e38f;
                        dp2[g * S + s] = ok ? err[(g * NL + labs[k]) * 2 + 1] : 3.4e38f;
                    }
                }
                __syncthreads();
                float ov[2][SPT][G];
                #pragma unroll
                for (int k = 0; k < SPT; k++) {
                    int s = tid + k * BLOCK;
                    int q0 = s >> 1, q1 = HALF + (s >> 1);
                    #pragma unroll
                    for (int g = 0; g < G; g++)
                        #pragma unroll
                        for (int c = 0; c < 2; c++) {
                            float best = 3.4e38f;
                            bool bh = false, bb = false;
                            #pragma unroll
                            for (int b = EQ ? c : 0; b < (EQ ? c + 1 : 2); b++) {
                                const float* arr = (b ? dp2 : dp1) + g * S;
                                float va = arr[q0], vb = arr[q1];
                                bool h = vb < va;
                                float v = (h ? vb : va) + err1[(g * NL + labs[k]) * 2 + (EQ ? c : (b ^ c))];
                                if (v < best) { best = v; bh = h; bb = b; }
                            }
                            ov[c][k][g] = best;
                            uint32_t wh = __ballot_sync(0xffffffffu, bh), wb = __ballot_sync(0xffffffffu, bb);
                            if (lane == 0) {
                                gpn[((size_t)g * NPL + c) * WORDS + (s >> 5)] = wh;
                                gpn[((size_t)g * NPL + 2 + c) * WORDS + (s >> 5)] = wb;
                            }
                        }
                }
                __syncthreads();
                #pragma unroll
                for (int k = 0; k < SPT; k++)
                    #pragma unroll
                    for (int g = 0; g < G; g++) {
                        dp1[g * S + tid + k * BLOCK] = ov[0][k][g];
                        dp2[g * S + tid + k * BLOCK] = ov[1][k][g];
                    }
                nstep = 2;
            } else if (PP && t == 0) {
                // pair (0, 1): intermediates (step 0) per parity class — cur (p = 0) and nxt (p = 1); outputs of step 1 —
                // radix-2 from both classes, the class with the smaller sum (ties go to p = 0); outputs via registers
                const float* err1 = err + G * NL * NP;
                #pragma unroll
                for (int k = 0; k < SPT; k++) {
                    int s = tid + k * BLOCK;
                    bool ok = s0 < 0 || (s >> 1) == (s0 & (HALF - 1));
                    #pragma unroll
                    for (int g = 0; g < G; g++) {
                        cur[g * S + s] = ok ? err[(g * NL + labs[k]) * 2] : 3.4e38f;
                        nxt[g * S + s] = ok ? err[(g * NL + labs[k]) * 2 + 1] : 3.4e38f;
                    }
                }
                __syncthreads();
                float outv[SPT][G];
                #pragma unroll
                for (int k = 0; k < SPT; k++) {
                    int s = tid + k * BLOCK;
                    int q0 = s >> 1, q1 = HALF + (s >> 1);
                    #pragma unroll
                    for (int g = 0; g < G; g++) {
                        float best = 3.4e38f;
                        bool bh = false, bpar = false;
                        #pragma unroll
                        for (int p = 0; p < 2; p++) {
                            const float* arr = (p ? nxt : cur) + g * S;
                            float va = arr[q0], vb = arr[q1];
                            bool h = vb < va;
                            float v = (h ? vb : va) + err1[(g * NL + labs[k]) * 2 + p];
                            if (v < best) { best = v; bh = h; bpar = p; }
                        }
                        outv[k][g] = best;
                        uint32_t bits = __ballot_sync(0xffffffffu, bh), pbit = __ballot_sync(0xffffffffu, bpar);
                        if (lane == 0) {
                            bpn[((size_t)1 * G + g) * WORDS + (s >> 5)] = bits;
                            ppn[((size_t)0 * G + g) * WORDS + (s >> 5)] = pbit;
                        }
                    }
                }
                __syncthreads();
                #pragma unroll
                for (int k = 0; k < SPT; k++)
                    #pragma unroll
                    for (int g = 0; g < G; g++) cur[g * S + tid + k * BLOCK] = outv[k][g];
                nstep = 2;
            } else if (t == 0) {
                #pragma unroll
                for (int k = 0; k < SPT; k++) {
                    int s = tid + k * BLOCK;
                    bool ok = s0 < 0 || (s >> 1) == (s0 & (HALF - 1));
                    #pragma unroll
                    for (int g = 0; g < G; g++) cur[g * S + s] = ok ? err[g * NL + labs[k]] : 3.4e38f;
                }
            } else if constexpr (G4 && SPT == 4) {
                // group (t, ..., t+3): phase A (t % 4 == 0) — first pair from dp0 into classes dp1 / dp2 (class = XOR of the pair's
                // low bits); phase B (t % 4 == 2) — second pair from each class with the same XOR (sum over the group is even),
                // output is the best class, into dp0
                constexpr int Q = S / 4;
                const int j = tid, grp = t / 4;
                nstep = 2;
                const float* err1 = err + G * NL * 2;
                const bool phaseA = (t & 3) == 0;
                #pragma unroll
                for (int g = 0; g < G; g++) {
                    uint32_t* gw = gpn + ((size_t)grp * G + g) * NPL * WORDS;
                    if (phaseA) {
                        const float* c_ = dp0 + g * S;
                        float i0 = c_[j], i1 = c_[Q + j], i2 = c_[HALF + j], i3 = c_[3 * Q + j];
                        bool h02 = i2 < i0, h13 = i3 < i1;
                        float a02 = h02 ? i2 : i0, a13 = h13 ? i3 : i1;
                        uint32_t b02 = __ballot_sync(0xffffffffu, h02), b13 = __ballot_sync(0xffffffffu, h13);
                        uint32_t* bw = bpn + (size_t)t * G * WORDS + g * WORDS;
                        if (lane < 4) {
                            uint32_t src = lane < 2 ? b02 : b13;
                            bw[(lane < 2 ? 0 : HALF / 32) + 2 * warp + (lane & 1)] = spread2(lane & 1 ? src >> 16 : src & 0xffffu);
                        }
                        float mm[4][2];
                        #pragma unroll
                        for (int b = 0; b < 2; b++) {
                            mm[0][b] = a02 + err[(g * NL + r4m[0]) * 2 + b];
                            mm[1][b] = a02 + err[(g * NL + r4m[1]) * 2 + b];
                            mm[2][b] = a13 + err[(g * NL + r4m[2]) * 2 + b];
                            mm[3][b] = a13 + err[(g * NL + r4m[3]) * 2 + b];
                        }
                        float ov[2][4];
                        uint32_t bh[2][4], bb[2][4];
                        #pragma unroll
                        for (int cc = 0; cc < 4; cc++) {
                            const int A = cc < 2 ? 0 : 1, B = cc < 2 ? 2 : 3;
                            #pragma unroll
                            for (int c = 0; c < 2; c++) {
                                float best = 3.4e38f;
                                bool hh = false, bq = false;
                                #pragma unroll
                                for (int b = EQ ? c : 0; b < (EQ ? c + 1 : 2); b++) {
                                    bool h = mm[B][b] < mm[A][b];
                                    float v = (h ? mm[B][b] : mm[A][b]) + err1[(g * NL + r4o[cc]) * 2 + (EQ ? c : (b ^ c))];
                                    if (v < best) { best = v; hh = h; bq = b; }
                                }
                                ov[c][cc] = best;
                                bh[c][cc] = __ballot_sync(0xffffffffu, hh);
                                bb[c][cc] = __ballot_sync(0xffffffffu, bq);
                            }
                        }
                        *reinterpret_cast<float4*>(dp1 + g * S + 4 * j) = make_float4(ov[0][0], ov[0][1], ov[0][2], ov[0][3]);
                        *reinterpret_cast<float4*>(dp2 + g * S + 4 * j) = make_float4(ov[1][0], ov[1][1], ov[1][2], ov[1][3]);
                        if (lane < 4)
                            #pragma unroll
                            for (int c = 0; c < 2; c++) {
                                uint32_t wh = 0, wb = 0;
                                #pragma unroll
                                for (int cc = 0; cc < 4; cc++) {
                                    wh |= spread4((bh[c][cc] >> (8 * lane)) & 0xffu) << cc;
                                    wb |= spread4((bb[c][cc] >> (8 * lane)) & 0xffu) << cc;
                                }
                                gw[c * WORDS + 4 * warp + lane] = wh;
                                gw[(2 + c) * WORDS + 4 * warp + lane] = wb;
                            }
                    } else {
                        float bo[4] = {3.4e38f, 3.4e38f, 3.4e38f, 3.4e38f};
                        bool oc[4] = {false, false, false, false}, oh[4] = {false, false, false, false},
                             ob[4] = {false, false, false, false};
                        #pragma unroll
                        for (int c = 0; c < 2; c++) {
                            const float* c_ = (c ? dp2 : dp1) + g * S;
                            float i0 = c_[j], i1 = c_[Q + j], i2 = c_[HALF + j], i3 = c_[3 * Q + j];
                            bool h02 = i2 < i0, h13 = i3 < i1;
                            float a02 = h02 ? i2 : i0, a13 = h13 ? i3 : i1;
                            uint32_t b02 = __ballot_sync(0xffffffffu, h02), b13 = __ballot_sync(0xffffffffu, h13);
                            if (lane < 4) {
                                uint32_t src = lane < 2 ? b02 : b13;
                                gw[(4 + c) * WORDS + (lane < 2 ? 0 : HALF / 32) + 2 * warp + (lane & 1)] =
                                    spread2(lane & 1 ? src >> 16 : src & 0xffffu);
                            }
                            float mm[4][2];
                            #pragma unroll
                            for (int b = 0; b < 2; b++) {
                                mm[0][b] = a02 + err[(g * NL + r4m[0]) * 2 + b];
                                mm[1][b] = a02 + err[(g * NL + r4m[1]) * 2 + b];
                                mm[2][b] = a13 + err[(g * NL + r4m[2]) * 2 + b];
                                mm[3][b] = a13 + err[(g * NL + r4m[3]) * 2 + b];
                            }
                            #pragma unroll
                            for (int cc = 0; cc < 4; cc++) {
                                const int A = cc < 2 ? 0 : 1, B = cc < 2 ? 2 : 3;
                                #pragma unroll
                                for (int b = EQ ? c : 0; b < (EQ ? c + 1 : 2); b++) {
                                    bool h = mm[B][b] < mm[A][b];
                                    float v = (h ? mm[B][b] : mm[A][b]) + err1[(g * NL + r4o[cc]) * 2 + (EQ ? c : (b ^ c))];
                                    if (v < bo[cc]) { bo[cc] = v; oc[cc] = c; oh[cc] = h; ob[cc] = b; }
                                }
                            }
                        }
                        *reinterpret_cast<float4*>(dp0 + g * S + 4 * j) = make_float4(bo[0], bo[1], bo[2], bo[3]);
                        uint32_t wc[4], wh[4], wb[4];
                        #pragma unroll
                        for (int cc = 0; cc < 4; cc++) {
                            wc[cc] = __ballot_sync(0xffffffffu, oc[cc]);
                            wh[cc] = __ballot_sync(0xffffffffu, oh[cc]);
                            wb[cc] = __ballot_sync(0xffffffffu, ob[cc]);
                        }
                        if (lane < 4) {
                            uint32_t xc = 0, xh = 0, xb = 0;
                            #pragma unroll
                            for (int cc = 0; cc < 4; cc++) {
                                xc |= spread4((wc[cc] >> (8 * lane)) & 0xffu) << cc;
                                xh |= spread4((wh[cc] >> (8 * lane)) & 0xffu) << cc;
                                xb |= spread4((wb[cc] >> (8 * lane)) & 0xffu) << cc;
                            }
                            gw[6 * WORDS + 4 * warp + lane] = xc;
                            gw[7 * WORDS + 4 * warp + lane] = xh;
                            gw[8 * WORDS + 4 * warp + lane] = xb;
                        }
                    }
                }
            } else if constexpr (PP && SPT == 4) {
                // pair (t, t+1), t even: intermediates per parity class, the output takes the best class; written out are
                // the predecessor bit (for the chosen class) and the class bit
                constexpr int Q = S / 4;
                const int j = tid;
                nstep = 2;
                const float* err1 = err + G * NL * 2;
                #pragma unroll
                for (int g = 0; g < G; g++) {
                    const float* c = cur + g * S;
                    float i0 = c[j], i1 = c[Q + j], i2 = c[HALF + j], i3 = c[3 * Q + j];
                    bool h02 = i2 < i0, h13 = i3 < i1;
                    float a02 = h02 ? i2 : i0, a13 = h13 ? i3 : i1;
                    float mm[4][2];
                    #pragma unroll
                    for (int p = 0; p < 2; p++) {
                        mm[0][p] = a02 + err[(g * NL + r4m[0]) * 2 + p];
                        mm[1][p] = a02 + err[(g * NL + r4m[1]) * 2 + p];
                        mm[2][p] = a13 + err[(g * NL + r4m[2]) * 2 + p];
                        mm[3][p] = a13 + err[(g * NL + r4m[3]) * 2 + p];
                    }
                    uint32_t b02 = __ballot_sync(0xffffffffu, h02), b13 = __ballot_sync(0xffffffffu, h13);
                    uint32_t* bw = bpn + (size_t)t * G * WORDS + g * WORDS;
                    if (lane < 4) {
                        uint32_t src = lane < 2 ? b02 : b13;
                        bw[(lane < 2 ? 0 : HALF / 32) + 2 * warp + (lane & 1)] = spread2(lane & 1 ? src >> 16 : src & 0xffffu);
                    }
                    float ov[4];
                    uint32_t bh[4], bq[4];
                    #pragma unroll
                    for (int cc = 0; cc < 4; cc++) {
                        const int A = cc < 2 ? 0 : 1, B = cc < 2 ? 2 : 3;  // predecessors: 2j / S/2+2j or 2j+1 / S/2+2j+1
                        float best = 3.4e38f;
                        bool hh = false, pq = false;
                        #pragma unroll
                        for (int p = 0; p < 2; p++) {
                            bool h = mm[B][p] < mm[A][p];
                            float v = (h ? mm[B][p] : mm[A][p]) + err1[(g * NL + r4o[cc]) * 2 + p];
                            if (v < best) { best = v; hh = h; pq = p; }
                        }
                        ov[cc] = best;
                        bh[cc] = __ballot_sync(0xffffffffu, hh);
                        bq[cc] = __ballot_sync(0xffffffffu, pq);
                    }
                    *reinterpret_cast<float4*>(nxt + g * S + 4 * j) = make_float4(ov[0], ov[1], ov[2], ov[3]);
                    if (lane < 4) {
                        uint32_t wh = 0, wq = 0;
                        #pragma unroll
                        for (int cc = 0; cc < 4; cc++) {
                            wh |= spread4((bh[cc] >> (8 * lane)) & 0xffu) << cc;
                            wq |= spread4((bq[cc] >> (8 * lane)) & 0xffu) << cc;
                        }
                        bw[G * WORDS + 4 * warp + lane] = wh;
                        ppn[((size_t)(t / 2) * G + g) * WORDS + 4 * warp + lane] = wq;
                    }
                }
                float* tmp = cur; cur = nxt; nxt = tmp;
            } else if constexpr (SPT == 4) {
                // radix-4: inputs j + k * S/4 -> intermediates {2j, 2j+1, S/2+2j, S/2+2j+1} (step t)
                // -> outputs 4j..4j+3 (step t + 1); at step 1 (t odd) intermediates only
                constexpr int Q = S / 4;
                const int j = tid;
                const bool pair = (t & 1) == 0;
                nstep = pair ? 2 : 1;
                const float* err1 = err + G * NL;
                #pragma unroll
                for (int g = 0; g < G; g++) {
                    const float* c = cur + g * S;
                    float i0 = c[j], i1 = c[Q + j], i2 = c[HALF + j], i3 = c[3 * Q + j];
                    bool h02 = i2 < i0, h13 = i3 < i1;
                    float a02 = h02 ? i2 : i0, a13 = h13 ? i3 : i1;
                    float m0 = a02 + err[g * NL + r4m[0]], m1 = a02 + err[g * NL + r4m[1]];
                    float m2 = a13 + err[g * NL + r4m[2]], m3 = a13 + err[g * NL + r4m[3]];
                    uint32_t b02 = __ballot_sync(0xffffffffu, h02), b13 = __ballot_sync(0xffffffffu, h13);
                    uint32_t* bw = bpn + (size_t)t * G * WORDS + g * WORDS;
                    if (lane < 4) {
                        uint32_t src = lane < 2 ? b02 : b13;
                        bw[(lane < 2 ? 0 : HALF / 32) + 2 * warp + (lane & 1)] = spread2(lane & 1 ? src >> 16 : src & 0xffffu);
                    }
                    float* o = nxt + g * S;
                    if (!pair) {
                        *reinterpret_cast<float2*>(o + 2 * j) = make_float2(m0, m1);
                        *reinterpret_cast<float2*>(o + HALF + 2 * j) = make_float2(m2, m3);
                    } else {
                        bool ha = m2 < m0, hb = m3 < m1;
                        float aa = ha ? m2 : m0, ab = hb ? m3 : m1;
                        *reinterpret_cast<float4*>(o + 4 * j) =
                            make_float4(aa + err1[g * NL + r4o[0]], aa + err1[g * NL + r4o[1]],
                                        ab + err1[g * NL + r4o[2]], ab + err1[g * NL + r4o[3]]);
                        uint32_t ba = __ballot_sync(0xffffffffu, ha), bb = __ballot_sync(0xffffffffu, hb);
                        if (lane < 4)
                            bw[G * WORDS + 4 * warp + lane] = spread4((ba >> (8 * lane)) & 0xffu) * 3u |
                                                              spread4((bb >> (8 * lane)) & 0xffu) * 12u;
                    }
                }
                float* tmp = cur; cur = nxt; nxt = tmp;
            } else {
                #pragma unroll
                for (int k = 0; k < SPT; k++) {
                    int s = tid + k * BLOCK;
                    int p0 = s >> 1, p1 = HALF + (s >> 1);
                    #pragma unroll
                    for (int g = 0; g < G; g++) {
                        float va = cur[g * S + p0], vb = cur[g * S + p1];
                        bool h = vb < va;
                        nxt[g * S + s] = (h ? vb : va) + err[g * NL + labs[k]];
                        uint32_t bits = __ballot_sync(0xffffffffu, h);
                        if (lane == 0) bpn[((size_t)t * G + g) * WORDS + (s >> 5)] = bits;
                    }
                }
                float* tmp = cur; cur = nxt; nxt = tmp;
            }
            __syncthreads();
            t += nstep;
        }
    }

    // argmin over (g, s): first minimal index, like torch.argmin
    float bv = 3.4e38f;
    int bi = 0x7fffffff;
    const int s_end = a.tail ? s0 : -1;  // tail-biting: the path must end in the initial state
    #pragma unroll
    for (int g = 0; g < G; g++)
        #pragma unroll
        for (int k = 0; k < SPT; k++) {
            int idx = g * S + tid + k * BLOCK;
            float v = (s_end < 0 || tid + k * BLOCK == s_end) ? cur[idx] : 3.4e38f;
            if (v < bv || (v == bv && idx < bi)) { bv = v; bi = idx; }
        }
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, bv, off);
        int oi = __shfl_down_sync(0xffffffffu, bi, off);
        if (ov < bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) { red_v[warp] = bv; red_i[warp] = bi; }
    __syncthreads();
    // traceback in the same block, one sub-block chunk at a time: within a sub-block the gain is constant and known, so the block
    // fetches that gain's pointer rows into shared with one coalesced read (sub x S/32 words), and a single thread walks
    // the chain in shared — sub global-memory latencies instead of sub * (T / sub); values are computed by all threads
    __shared__ uint32_t s_bp[32 * WORDS];
    __shared__ uint8_t s_gsr[S];
    __shared__ uint16_t s_st[MAX_T];
    __shared__ uint8_t s_g[MAX_T];
    __shared__ uint8_t s_par[MD ? MAX_T : 1];
    __shared__ uint32_t s_pp[PP ? 16 * WORDS : 1];
    __shared__ uint32_t s_gp[G4 ? 8 * NPL * WORDS : 1];
    __shared__ int s_cur[3];  // state, gain slot, "first tail-biting pass done" flag
    if (tid == 0) {
        for (int k = 1; k < BLOCK / 32; k++)
            if (red_v[k] < bv || (red_v[k] == bv && red_i[k] < bi)) { bv = red_v[k]; bi = red_i[k]; }
        a.best[n] = bi;
        s_cur[0] = bi % S;
        s_cur[1] = bi / S;
        s_cur[2] = 0;
        if (a.end_st) a.end_st[n] = bi % S;
    }
    __syncthreads();
    for (int lo = T - sub; lo >= 0; lo -= sub) {
        const int gc = s_cur[1];
        for (int i = tid; i < sub * WORDS; i += BLOCK)
            s_bp[i] = bpn[((size_t)(lo + i / WORDS) * G + gc) * WORDS + (i % WORDS)];
        if (G > 1 && lo > 0)
            for (int i = tid; i < S; i += BLOCK) s_gsr[i] = gsn[(size_t)(lo / sub) * S + i];
        if (PP)  // pair parity-class bits of the chunk (lo even, sub even)
            for (int i = tid; i < (sub / 2) * WORDS; i += BLOCK)
                s_pp[i] = ppn[((size_t)(lo / 2 + i / WORDS) * G + gc) * WORDS + (i % WORDS)];
        if (G4)  // group bit planes of the chunk (lo and sub are multiples of 4)
            for (int i = tid; i < (sub / 4) * NPL * WORDS; i += BLOCK)
                s_gp[i] = gpn[((size_t)(lo / 4 + i / (NPL * WORDS)) * G + gc) * NPL * WORDS + (i % (NPL * WORDS))];
        __syncthreads();
        if (tid == 0) {
            int st = s_cur[0], g = s_cur[1], pcur = 0, cstar = 0;
            for (int t = lo + sub - 1; t >= lo; t--) {
                s_st[t] = (uint16_t)st;
                s_g[t] = (uint8_t)g;
                if (PP) {  // end of a pair (t odd): class of the output state applies to both samples of the pair
                    if (t & 1) pcur = (s_pp[((t - lo) >> 1) * WORDS + (st >> 5)] >> (st & 31)) & 1;
                    s_par[t] = (uint8_t)pcur;
                }
                if (a.mid_st && t == a.mid_t - 1) { a.mid_st[n] = st; s_cur[2] = 1; break; }
                if (G4 && (t & 3)) {  // inside a group, predecessor and low bits come from the group's planes
                    const uint32_t* gq = s_gp + ((t - lo) >> 2) * NPL * WORDS;
                    #define GBIT(pl) ((gq[(pl) * WORDS + (st >> 5)] >> (st & 31)) & 1)
                    int hh;
                    if ((t & 3) == 3) {
                        cstar = GBIT(6);
                        hh = GBIT(7);
                        if (EQ) {  // shared low bit of the whole group
                            s_par[t] = s_par[t - 1] = s_par[t - 2] = s_par[t - 3] = (uint8_t)cstar;
                        } else {
                            const int bb = GBIT(8);
                            s_par[t] = (uint8_t)(bb ^ cstar);
                            s_par[t - 1] = (uint8_t)bb;
                        }
                    } else if ((t & 3) == 2) {
                        hh = GBIT(4 + cstar);
                    } else {
                        hh = GBIT(0 + cstar);
                        if (!EQ) {
                            const int bb = GBIT(2 + cstar);
                            s_par[t] = (uint8_t)(bb ^ cstar);
                            s_par[t - 1] = (uint8_t)bb;
                        }
                    }
                    #undef GBIT
                    st = hh * HALF + (st >> 1);
                } else if (t > 0) {
                    const uint32_t word = s_bp[(t - lo) * WORDS + (st >> 5)];
                    st = ((word >> (st & 31)) & 1) * HALF + (st >> 1);
                    if (G > 1 && t == lo) g = s_gsr[st];  // sub-block boundary: gain slot of the previous sub-block
                }
            }
            s_cur[0] = st;
            s_cur[1] = g;
        }
        __syncthreads();
        if (s_cur[2]) return;  // the first tail-biting pass only needs the state before mid_t
    }
    float* rn = a.rec + (size_t)n * a.ld;
    for (int t = tid; t < T; t += BLOCK) {
        const int stt = s_st[t], gg = s_g[t];
        const int lb = tcq_label_x(stt, a, NL);
        const bool hi = mix > 0 && ((t / mix) & 1);
        const int Ut = hi ? U_hi : U;
        const int gi = cand ? (int)s_gc[(t / sub) * G + gg] : gg;
        const float gt = hi ? s_gain_hi[gi] : s_gain[gi];
        const float* tt = (hi ? s_tab_hi : s_tab) + lb * Ut;
        const float xt = s_x[t];
        float bestd = 3.4e38f, val = 0.f;
        int bu = 0;
        for (int u = MD ? s_par[t] : 0; u < Ut; u += MD ? 2 : 1) {
            float v = gt * tt[u];
            float d = xt - v;
            if (d * d < bestd) { bestd = d * d; val = v; bu = u; }
        }
        const int tp = (t + a.rot) % a.ld;
        rn[tp] = val;
        if (a.code) a.code[(size_t)n * a.ld + tp] = (gi << 16) | (lb * Ut + bu);
    }
}


template <int BLOCK, int SPT, int G, int NL, int MD = 0>
static int launch_forward(const TcqArgs& a, cudaStream_t stream)
{
    if (a.pp && MD == 0) {  // pair / group parity — radix-4 only
        if constexpr (SPT == 4) {
            if (a.pp == 1) return launch_forward<BLOCK, SPT, G, NL, 1>(a, stream);
            if (a.pp == 2) return launch_forward<BLOCK, SPT, G, NL, 2>(a, stream);
            if (a.pp == 3) return launch_forward<BLOCK, SPT, G, NL, 3>(a, stream);
        }
        return -1;
    }
    size_t smem = (((MD >= 2) ? 3 : 2) * (size_t)G * a.S + (size_t)a.sub * G * a.nl * (MD ? 2 : 1) + 2 * (size_t)a.T) * sizeof(float);
    // dynamic + static shared memory may exceed the default 48 KB (G = 8: dp is 64 KB; tables up to 4 KB)
    // dynamic shared limit is set once per instantiation and straight to the maximum: with parallel calls from different threads
    // (LIQK_PAR) with different T (tail-biting window), per-call setting raced and lowered the limit under another launch (error 701)
    static int dyn_max = -1;
    if (dyn_max < 0) {
        cudaFuncAttributes fa;
        int dev = 0, optin = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        cudaFuncGetAttributes(&fa, tcq_forward_kernel<BLOCK, SPT, G, NL, MD>);
        int want = optin - (int)fa.sharedSizeBytes;
        cudaFuncSetAttribute(tcq_forward_kernel<BLOCK, SPT, G, NL, MD>, cudaFuncAttributeMaxDynamicSharedMemorySize, want);
        dyn_max = want;
    }
    if ((int)smem > dyn_max) return -1;
    tcq_forward_kernel<BLOCK, SPT, G, NL, MD><<<a.N, BLOCK, smem, stream>>>(a);
    return 0;
}

template <int BLOCK, int SPT, int NL>
static int dispatch_g(const TcqArgs& a, cudaStream_t stream)
{
    switch (a.G) {
        case 1: return launch_forward<BLOCK, SPT, 1, NL>(a, stream);
        case 2: return launch_forward<BLOCK, SPT, 2, NL>(a, stream);
        case 3: return launch_forward<BLOCK, SPT, 3, NL>(a, stream);
        case 4: return launch_forward<BLOCK, SPT, 4, NL>(a, stream);
        case 8: return launch_forward<BLOCK, SPT, 8, NL>(a, stream);
        default: return -1;
    }
}

// number of label subsets is a template parameter (branch-error indices without division by a runtime NL)
template <int BLOCK, int SPT>
static int dispatch_nl(const TcqArgs& a, cudaStream_t stream)
{
    switch (a.nl) {
        case 4: return dispatch_g<BLOCK, SPT, 4>(a, stream);
        case 8: return dispatch_g<BLOCK, SPT, 8>(a, stream);
        case 16: return dispatch_g<BLOCK, SPT, 16>(a, stream);
        case 32: return dispatch_g<BLOCK, SPT, 32>(a, stream);
        default: return -1;
    }
}

extern "C" int tcq_viterbi(
    const float* x, const float* w, const float* tab, const float* gains, float* rec, uint32_t* bp, uint8_t* gsel,
    const int* init_st, int* end_st, const float* tab_hi, const float* gains_hi, uint8_t* gcand, int* best, int* code, int* mid_st, int mid_t, int tail,
    int N, int T, int S, int U, int G, int Gfull, int m, int sub, int U_hi, int mix, int m2, int m3, int m4, int nl, int ld, int rot, uint32_t* pbits, int pp, void* stream)
{
    if (G > MAX_G || Gfull > MAX_G || (!gcand && Gfull != G) || U > MAX_U || U_hi > MAX_U || T % sub != 0 || (sub & 1) || T / sub > MAX_SUBS || T > MAX_T || sub > 32 ||
        sub * G * nl * (pp ? 2 : 1) > 4096 || (pp && (S != 1024 || mix > 0 || (T & 1))) || (pp >= 2 && ((T & 3) || (sub & 3))) || pp > 3 || (nl != 4 && nl != 8 && nl != 16 && nl != 32) || ld < T ||
        ((S != 1024 || getenv("TCQ_BLOCK")) && nl != 4) || nl * U > 4 * MAX_U || nl * U_hi > 4 * MAX_U)
        return -1;
    TcqArgs a{x, w, tab, gains, rec, bp, gsel, init_st, end_st, tab_hi, gains_hi, gcand, best, code, mid_st, mid_t, tail,
              m2, m3, m4, nl, N, T, S, U, G, Gfull, m, sub, U_hi, mix, ld, rot, pbits, pp};
    cudaStream_t st = (cudaStream_t)stream;
    // S = 1024: block 256 x 4 with radix-4 (TCQ_BLOCK=1024 — one state per thread, radix-2)
    static const int force = getenv("TCQ_BLOCK") ? atoi(getenv("TCQ_BLOCK")) : 0;
    int rc;
    if (S == 1024)
        rc = force == 1024 ? dispatch_g<1024, 1, 4>(a, st) : dispatch_nl<256, 4>(a, st);  // radix-2 path supports only 4 labels
    else if (S == 256)
        rc = dispatch_g<256, 1, 4>(a, st);
    else
        return -1;
    if (rc != 0) return rc;
    return (int)cudaGetLastError();
}

// Diagonal v2 sub-block quantization via LUT (DP profile, `quantize_subs_lut_chunked`): one thread per sub-block of 16
// weights, 64 candidates kept in registers, table rows (64 bucket values) read from L2 one per weight —
// instead of materializing [sub-blocks, 16, 64] in torch.
__global__ void lut_subs_kernel(const float* __restrict__ xs, const float* __restrict__ dh, const float* __restrict__ lut,
                                float* __restrict__ out, int S, float inv_step, float half_b, int nb)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= S) return;
    float err[64];
    #pragma unroll
    for (int c = 0; c < 64; c++) err[c] = 0.f;
    int bj[16];
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        const float x = xs[(size_t)i * 16 + j], d = dh[(size_t)i * 16 + j];
        int b = (int)rintf(x * inv_step + half_b);
        b = b < 0 ? 0 : (b > nb - 1 ? nb - 1 : b);
        bj[j] = b;
        const float4* row = reinterpret_cast<const float4*>(lut + (size_t)b * 64);
        #pragma unroll
        for (int c4 = 0; c4 < 16; c4++) {
            const float4 v = row[c4];
            float e0 = x - v.x, e1 = x - v.y, e2 = x - v.z, e3 = x - v.w;
            err[4 * c4] += d * e0 * e0;
            err[4 * c4 + 1] += d * e1 * e1;
            err[4 * c4 + 2] += d * e2 * e2;
            err[4 * c4 + 3] += d * e3 * e3;
        }
    }
    int best = 0;
    float be = err[0];
    #pragma unroll
    for (int c = 1; c < 64; c++)
        if (err[c] < be) { be = err[c]; best = c; }
    #pragma unroll
    for (int j = 0; j < 16; j++) out[(size_t)i * 16 + j] = lut[(size_t)bj[j] * 64 + best];
}

extern "C" int lut_subs_quant(const float* xs, const float* dh, const float* lut, float* out, int S, float inv_step,
                              float half_b, int nb, void* stream)
{
    lut_subs_kernel<<<(S + 127) / 128, 128, 0, (cudaStream_t)stream>>>(xs, dh, lut, out, S, inv_step, half_b, nb);
    return (int)cudaGetLastError();
}
