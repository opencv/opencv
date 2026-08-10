// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "rvv_hal.hpp"

namespace cv { namespace rvv_hal { namespace dnn {

#if CV_HAL_RVV_1P0_ENABLED

// Blocked NCHWc general (non-depthwise) convolution, RVV, CV_32F. Mirrors the built-in
// conv32fC8 (modules/dnn/src/layers/cpu_kernels/conv2_kernels.simd.hpp) but computes only the
// tasks [task_start, task_end) of the engine's grid, and applies the same fused epilogue
//   out = min( s >= 0 ? s : s*alpha, maxval ),  s = acc*scale + bias, then optional residual add,
// with alpha = prelu_slope[k] (if provided) else default_alpha.
//
// The shape of the computation is set by the blocked layout: one output channel block is K0 == 8
// wide, so a single e32m2 group holds a whole K0 vector at any VLEN >= 128, and one weight vector
// w[c0] = W[c0][0..K0) is reused across all the output positions held in flight. This kernel keeps
// SPAT_BLOCK == 10 positions in flight (as the built-in does), so each weight load feeds 10 FMAs
// and the serial FMA latency is hidden by 10 independent accumulator chains.
//
// Native vsetvl is what makes this possible: the universal-intrinsic path needs a vector of
// exactly K0 lanes, which it cannot express on RVV (its ops derive their length from VLMAX), so
// the built-in falls back to scalar code here at every VLEN -- see the `#elif 0` branches in
// conv2_kernels.simd.hpp and the discussion in PR #29619.
//
// Positions are processed in blocks that are either entirely inside the padding-free interior
// (flat per-tap offsets from ofstab[], no checks) or general (every tap bounds-checked against
// the coordinate table, an out-of-range tap reading a zero vector so the FMA stays branch-free).
//
// A K0-lane vector leaves most of a wide register idle, so a second strategy packs P = VLMAX/K0
// *output channel blocks* into one vector (vl = P*K0). The input value x[c0] does not depend on
// the output channel, so all P blocks share one vfmacc.vf and no per-iteration gather is needed --
// unlike packing spatial positions, which would need a strided load plus a vrgather every
// iteration. The P blocks live blockstride apart in the weight tensor, so their weights are tiled
// once per group into a contiguous scratch (below) and the inner loop stays unit-stride; the P
// outputs are planesize apart, so the wide epilogue is followed by a K0-at-a-time scatter (about
// 1% of the inner loop). Both strategies produce identical results; the wide one is selected only
// where the tiling cost amortises (see use_wide).

namespace {

#define CONV_ALWAYS_INLINE inline __attribute__((always_inline))

constexpr int CONV_MAX_C0 = 8;      // engine's fixed f32 channel block
constexpr int CONV_SPAT_BLOCK = 10; // output positions in flight (matches the built-in)
constexpr int CONV_DIMS = 3;        // coordtab/pads/inner use a fixed Z,Y,X frame

// Geometry that stays constant for a whole task; passed by const ref into always-inlined code,
// so the fields are read directly and never materialised as a copy.
struct ConvGeom
{
    int Di, Hi, Wi;                 // input spatial dims (Z,Y,X)
    int D, H, W;                    // output spatial dims (Z,Y,X)
    int Sz, Sy, Sx;                 // strides
    int padZ, padY, padX;           // begin pads
    int innerZ0, innerZ1;           // padding-free interior bounds
    int innerY0, innerY1;
    int innerX0, innerX1;
    int C0, K0, C1Max, cblocks, ksize;
    int64_t iplanesize;             // elements per input channel-block plane
    const int* coordtab;            // [ksize*3] per-tap (dz,dy,dx)
    const int* ofstab;              // [ksize] per-tap flat input offset (interior)
    const float* zbuf;              // C0 zeros, read in place of an out-of-range tap
};

// Decode plane index p into output coordinates.
static CONV_ALWAYS_INLINE void convDecodePos(int p, int D, int H, int W, int& z, int& y, int& x)
{
    if (D == 1) {
        z = 0; y = p/W; x = p - y*W;
    } else {
        z = p/(H*W);
        int yx = p - z*(H*W);
        y = yx/W; x = yx - y*W;
    }
}

// RVV vector types are sizeless: accumulators cannot live in an array, so the per-position work
// is unrolled through an X-macro list, one named accumulator per output position.
#define CONV_ACC_LIST10(F) F(0) F(1) F(2) F(3) F(4) F(5) F(6) F(7) F(8) F(9)
#define CONV_ACC_LIST1(F)  F(0)

#define CONV_ACC_INIT(j)  vfloat32m2_t acc##j = __riscv_vfmv_v_f_f32m2(0.f, vl);
#define CONV_ACC_INNER(j) acc##j = __riscv_vfmacc_vf_f32m2(acc##j, xb[(int64_t)(j)*x_step + c0], w, vl);
#define CONV_ACC_GEN(j)   acc##j = __riscv_vfmacc_vf_f32m2(acc##j, xp[j][c0], w, vl);
#define CONV_ACC_STORE(j) { \
    vfloat32m2_t s = __riscv_vfmadd_vv_f32m2(acc##j, vsc, vbi, vl); \
    if (rp) \
        s = __riscv_vfadd_vv_f32m2(s, __riscv_vle32_v_f32m2(rp + (int64_t)(j)*K0, vl), vl); \
    vfloat32m2_t neg = __riscv_vfmul_vv_f32m2(s, val, vl); \
    vbool16_t m = __riscv_vmfge_vf_f32m2_b16(s, 0.f, vl); \
    s = __riscv_vmerge_vvm_f32m2(neg, s, m, vl); \
    s = __riscv_vfmin_vf_f32m2(s, maxval, vl); \
    __riscv_vse32_v_f32m2(op + (int64_t)(j)*K0, s, vl); \
}

// Compute NP consecutive output positions starting at plane index p (whose coordinates are
// cur_z/cur_y/cur_x), then apply the fused epilogue and store. op/rp point at the first of the
// NP output/residual pixels; inpbase/wbase are this task's input and weight block bases.
#define CONV_DEFINE_BLOCK(NAME, NP, ACC_LIST)                                                 \
static CONV_ALWAYS_INLINE void NAME(int p, int cur_z, int cur_y, int cur_x,                   \
                                    const float* inpbase, const float* wbase,                 \
                                    float* op, const float* rp,                               \
                                    const float* scalebuf, const float* biasbuf,              \
                                    const float* alphabuf, float maxval, const ConvGeom& G)   \
{                                                                                             \
    const int C0 = G.C0, K0 = G.K0;                                                           \
    const size_t vl = __riscv_vsetvl_e32m2((size_t)K0); /* K0 <= 8 <= VLMAX(e32m2) */         \
                                                                                              \
    ACC_LIST(CONV_ACC_INIT)                                                                   \
                                                                                              \
    const bool same_row = (cur_x + (NP) <= G.W);                                              \
    const bool all_inner = same_row &&                                                        \
        (cur_z >= G.innerZ0 && cur_z < G.innerZ1) &&                                          \
        (cur_y >= G.innerY0 && cur_y < G.innerY1) &&                                          \
        (cur_x >= G.innerX0) && (cur_x + (NP) - 1 < G.innerX1);                               \
                                                                                              \
    if (all_inner) {                                                                          \
        /* no tap of any position touches padding: address the input straight through the */  \
        /* per-tap offset table, consecutive positions being x_step apart */                  \
        const int x_step = G.Sx*C0;                                                           \
        const int64_t base_ofs = (((int64_t)(cur_z*G.Sz - G.padZ)*G.Hi +                      \
                                   (cur_y*G.Sy - G.padY))*G.Wi + (cur_x*G.Sx - G.padX))*C0;   \
        for (int i = 0; i < G.ksize; i++) {                                                   \
            const float* kw = wbase + (int64_t)i*G.C1Max*C0*K0;                               \
            const float* xb = inpbase + base_ofs + G.ofstab[i];                               \
            for (int c1 = 0; c1 < G.cblocks; c1++, kw += C0*K0, xb += G.iplanesize) {         \
                for (int c0 = 0; c0 < C0; c0++) {                                             \
                    vfloat32m2_t w = __riscv_vle32_v_f32m2(kw + c0*K0, vl);                   \
                    ACC_LIST(CONV_ACC_INNER)                                                  \
                }                                                                             \
            }                                                                                 \
        }                                                                                     \
    } else {                                                                                  \
        int pz[NP], py[NP], px[NP];                                                           \
        bool inr[NP];                                                                         \
        if (same_row) {                                                                       \
            const bool zy_inner = (cur_z >= G.innerZ0 && cur_z < G.innerZ1) &&                \
                                  (cur_y >= G.innerY0 && cur_y < G.innerY1);                  \
            for (int j = 0; j < (NP); j++) {                                                  \
                pz[j] = cur_z*G.Sz - G.padZ;                                                  \
                py[j] = cur_y*G.Sy - G.padY;                                                  \
                px[j] = (cur_x + j)*G.Sx - G.padX;                                            \
                inr[j] = zy_inner && (cur_x + j >= G.innerX0) && (cur_x + j < G.innerX1);     \
            }                                                                                 \
        } else {                                                                              \
            for (int j = 0; j < (NP); j++) {                                                  \
                int zj, yj, xj;                                                               \
                convDecodePos(p + j, G.D, G.H, G.W, zj, yj, xj);                              \
                pz[j] = zj*G.Sz - G.padZ;                                                     \
                py[j] = yj*G.Sy - G.padY;                                                     \
                px[j] = xj*G.Sx - G.padX;                                                     \
                inr[j] = (zj >= G.innerZ0 && zj < G.innerZ1) &&                               \
                         (yj >= G.innerY0 && yj < G.innerY1) &&                               \
                         (xj >= G.innerX0 && xj < G.innerX1);                                 \
            }                                                                                 \
        }                                                                                     \
                                                                                              \
        int64_t xofs[NP];                                                                     \
        bool xok[NP];                                                                         \
        for (int i = 0; i < G.ksize; i++) {                                                   \
            const int dz = G.coordtab[i*CONV_DIMS], dy = G.coordtab[i*CONV_DIMS + 1],         \
                      dx = G.coordtab[i*CONV_DIMS + 2];                                       \
            for (int j = 0; j < (NP); j++) {                                                  \
                const int zij = pz[j] + dz, yij = py[j] + dy, xij = px[j] + dx;               \
                xok[j] = inr[j] || (((unsigned)zij < (unsigned)G.Di) &                        \
                                    ((unsigned)yij < (unsigned)G.Hi) &                        \
                                    ((unsigned)xij < (unsigned)G.Wi)) != 0;                   \
                xofs[j] = (((int64_t)zij*G.Hi + yij)*G.Wi + xij)*C0;                          \
            }                                                                                 \
            const float* kw = wbase + (int64_t)i*G.C1Max*C0*K0;                               \
            for (int c1 = 0; c1 < G.cblocks; c1++, kw += C0*K0) {                             \
                const float* xp[NP];                                                          \
                for (int j = 0; j < (NP); j++)                                                \
                    xp[j] = xok[j] ? inpbase + (int64_t)c1*G.iplanesize + xofs[j] : G.zbuf;   \
                for (int c0 = 0; c0 < C0; c0++) {                                             \
                    vfloat32m2_t w = __riscv_vle32_v_f32m2(kw + c0*K0, vl);                   \
                    ACC_LIST(CONV_ACC_GEN)                                                    \
                }                                                                             \
            }                                                                                 \
        }                                                                                     \
    }                                                                                         \
                                                                                              \
    /* acc*scale + bias (+ residual), then out = min(acc >= 0 ? acc : acc*alpha, maxval) */    \
    const vfloat32m2_t vsc = __riscv_vle32_v_f32m2(scalebuf, vl);                             \
    const vfloat32m2_t vbi = __riscv_vle32_v_f32m2(biasbuf, vl);                              \
    const vfloat32m2_t val = __riscv_vle32_v_f32m2(alphabuf, vl);                             \
    ACC_LIST(CONV_ACC_STORE)                                                                  \
}

CONV_DEFINE_BLOCK(convBlock10, CONV_SPAT_BLOCK, CONV_ACC_LIST10)
CONV_DEFINE_BLOCK(convBlock1, 1, CONV_ACC_LIST1)

// --- wide path: P output channel blocks per vector -------------------------------------------
// Same traversal as above, but the weight vector comes from the tiled scratch (P blocks laid out
// contiguously) and one accumulator covers P*K0 channels. out0/res0 point at the group's first
// block; block b is planesize away.

#define CONV_ACC_WINIT(j)  vfloat32m2_t acc##j = __riscv_vfmv_v_f_f32m2(0.f, vlw);
#define CONV_ACC_WINNER(j) acc##j = __riscv_vfmacc_vf_f32m2(acc##j, xb[(int64_t)(j)*x_step + c0], w, vlw);
#define CONV_ACC_WGEN(j)   acc##j = __riscv_vfmacc_vf_f32m2(acc##j, xp[j][c0], w, vlw);
#define CONV_ACC_WSTORE(j) { \
    vfloat32m2_t s = __riscv_vfmadd_vv_f32m2(acc##j, vsc, vbi, vlw); \
    if (res0) { \
        for (int b = 0; b < P; b++) \
            __riscv_vse32_v_f32m2(scratch + (int64_t)b*K0, \
                __riscv_vle32_v_f32m2(res0 + (int64_t)b*planesize + (int64_t)(j)*K0, vk), vk); \
        s = __riscv_vfadd_vv_f32m2(s, __riscv_vle32_v_f32m2(scratch, vlw), vlw); \
    } \
    vfloat32m2_t neg = __riscv_vfmul_vv_f32m2(s, val, vlw); \
    vbool16_t m = __riscv_vmfge_vf_f32m2_b16(s, 0.f, vlw); \
    s = __riscv_vmerge_vvm_f32m2(neg, s, m, vlw); \
    s = __riscv_vfmin_vf_f32m2(s, maxval, vlw); \
    __riscv_vse32_v_f32m2(scratch, s, vlw); \
    for (int b = 0; b < P; b++) \
        __riscv_vse32_v_f32m2(out0 + (int64_t)b*planesize + (int64_t)(j)*K0, \
            __riscv_vle32_v_f32m2(scratch + (int64_t)b*K0, vk), vk); \
}

#define CONV_DEFINE_WIDE_BLOCK(NAME, NP, ACC_LIST)                                            \
static CONV_ALWAYS_INLINE void NAME(int p, int cur_z, int cur_y, int cur_x,                   \
                                    const float* inpbase, const float* tw,                    \
                                    float* out0, const float* res0, int64_t planesize, int P, \
                                    const float* scalebuf, const float* biasbuf,              \
                                    const float* alphabuf, float maxval, float* scratch,      \
                                    const ConvGeom& G)                                        \
{                                                                                             \
    const int C0 = G.C0, K0 = G.K0;                                                           \
    const size_t vk  = __riscv_vsetvl_e32m2((size_t)K0);                                      \
    const size_t vlw = __riscv_vsetvl_e32m2((size_t)(P*K0));                                  \
    const int64_t wstep_c0 = (int64_t)P*K0;                                                   \
    const int64_t wstep_c1 = (int64_t)C0*P*K0;                                                \
                                                                                              \
    ACC_LIST(CONV_ACC_WINIT)                                                                  \
                                                                                              \
    const bool same_row = (cur_x + (NP) <= G.W);                                              \
    const bool all_inner = same_row &&                                                        \
        (cur_z >= G.innerZ0 && cur_z < G.innerZ1) &&                                          \
        (cur_y >= G.innerY0 && cur_y < G.innerY1) &&                                          \
        (cur_x >= G.innerX0) && (cur_x + (NP) - 1 < G.innerX1);                               \
                                                                                              \
    if (all_inner) {                                                                          \
        const int x_step = G.Sx*C0;                                                           \
        const int64_t base_ofs = (((int64_t)(cur_z*G.Sz - G.padZ)*G.Hi +                      \
                                   (cur_y*G.Sy - G.padY))*G.Wi + (cur_x*G.Sx - G.padX))*C0;   \
        for (int i = 0; i < G.ksize; i++) {                                                   \
            const float* kw = tw + (int64_t)i*G.cblocks*wstep_c1;                             \
            const float* xb = inpbase + base_ofs + G.ofstab[i];                               \
            for (int c1 = 0; c1 < G.cblocks; c1++, kw += wstep_c1, xb += G.iplanesize) {      \
                for (int c0 = 0; c0 < C0; c0++) {                                             \
                    vfloat32m2_t w = __riscv_vle32_v_f32m2(kw + c0*wstep_c0, vlw);            \
                    ACC_LIST(CONV_ACC_WINNER)                                                 \
                }                                                                             \
            }                                                                                 \
        }                                                                                     \
    } else {                                                                                  \
        int pz[NP], py[NP], px[NP];                                                           \
        bool inr[NP];                                                                         \
        if (same_row) {                                                                       \
            const bool zy_inner = (cur_z >= G.innerZ0 && cur_z < G.innerZ1) &&                \
                                  (cur_y >= G.innerY0 && cur_y < G.innerY1);                  \
            for (int j = 0; j < (NP); j++) {                                                  \
                pz[j] = cur_z*G.Sz - G.padZ;                                                  \
                py[j] = cur_y*G.Sy - G.padY;                                                  \
                px[j] = (cur_x + j)*G.Sx - G.padX;                                            \
                inr[j] = zy_inner && (cur_x + j >= G.innerX0) && (cur_x + j < G.innerX1);     \
            }                                                                                 \
        } else {                                                                              \
            for (int j = 0; j < (NP); j++) {                                                  \
                int zj, yj, xj;                                                               \
                convDecodePos(p + j, G.D, G.H, G.W, zj, yj, xj);                              \
                pz[j] = zj*G.Sz - G.padZ;                                                     \
                py[j] = yj*G.Sy - G.padY;                                                     \
                px[j] = xj*G.Sx - G.padX;                                                     \
                inr[j] = (zj >= G.innerZ0 && zj < G.innerZ1) &&                               \
                         (yj >= G.innerY0 && yj < G.innerY1) &&                               \
                         (xj >= G.innerX0 && xj < G.innerX1);                                 \
            }                                                                                 \
        }                                                                                     \
                                                                                              \
        int64_t xofs[NP];                                                                     \
        bool xok[NP];                                                                         \
        for (int i = 0; i < G.ksize; i++) {                                                   \
            const int dz = G.coordtab[i*CONV_DIMS], dy = G.coordtab[i*CONV_DIMS + 1],         \
                      dx = G.coordtab[i*CONV_DIMS + 2];                                       \
            for (int j = 0; j < (NP); j++) {                                                  \
                const int zij = pz[j] + dz, yij = py[j] + dy, xij = px[j] + dx;               \
                xok[j] = inr[j] || (((unsigned)zij < (unsigned)G.Di) &                        \
                                    ((unsigned)yij < (unsigned)G.Hi) &                        \
                                    ((unsigned)xij < (unsigned)G.Wi)) != 0;                   \
                xofs[j] = (((int64_t)zij*G.Hi + yij)*G.Wi + xij)*C0;                          \
            }                                                                                 \
            const float* kw = tw + (int64_t)i*G.cblocks*wstep_c1;                             \
            for (int c1 = 0; c1 < G.cblocks; c1++, kw += wstep_c1) {                          \
                const float* xp[NP];                                                          \
                for (int j = 0; j < (NP); j++)                                                \
                    xp[j] = xok[j] ? inpbase + (int64_t)c1*G.iplanesize + xofs[j] : G.zbuf;   \
                for (int c0 = 0; c0 < C0; c0++) {                                             \
                    vfloat32m2_t w = __riscv_vle32_v_f32m2(kw + c0*wstep_c0, vlw);            \
                    ACC_LIST(CONV_ACC_WGEN)                                                   \
                }                                                                             \
            }                                                                                 \
        }                                                                                     \
    }                                                                                         \
                                                                                              \
    const vfloat32m2_t vsc = __riscv_vle32_v_f32m2(scalebuf, vlw);                            \
    const vfloat32m2_t vbi = __riscv_vle32_v_f32m2(biasbuf, vlw);                             \
    const vfloat32m2_t val = __riscv_vle32_v_f32m2(alphabuf, vlw);                            \
    ACC_LIST(CONV_ACC_WSTORE)                                                                 \
}

CONV_DEFINE_WIDE_BLOCK(convWideBlock10, CONV_SPAT_BLOCK, CONV_ACC_LIST10)
CONV_DEFINE_WIDE_BLOCK(convWideBlock1, 1, CONV_ACC_LIST1)

} // anonymous namespace

int conv32f(const float* inp_data, const float* residual_data,
            float* out_data, const float* weights_all,
            const float* scale_all, const float* bias_all,
            int C, int K, int C0, int ngroups, int Kblk, int C1Max,
            const int* insize, const int* outsize, const int* strides,
            const int* pads, const int* inner, const int* coordtab,
            const int* ofstab, int ksize,
            float maxval, float default_alpha, const float* prelu_slope,
            int nspat_chunks, int task_start, int task_end)
{
    const int K0 = C0;
    if (C0 != CONV_MAX_C0 || ngroups < 1 || Kblk < 1 || ksize < 1 || nspat_chunks < 1)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    const int Cg = C/ngroups, Kg = K/ngroups;
    // Only the aligned case is handled: every output channel block is full (k_count == K0 with a
    // K0-aligned k_base) and every group starts on a channel block. Anything else needs the
    // built-in's scatter/gather epilogue, so decline the whole range and let it run.
    if (K % K0 != 0 || Kg % K0 != 0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (ngroups > 1 && Cg % C0 != 0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    ConvGeom G;
    G.Di = insize[0];  G.Hi = insize[1];  G.Wi = insize[2];
    G.D  = outsize[0]; G.H  = outsize[1]; G.W  = outsize[2];
    G.Sz = strides[0]; G.Sy = strides[1]; G.Sx = strides[2];
    G.padZ = pads[0]; G.padY = pads[1]; G.padX = pads[2];
    G.innerZ0 = inner[0]; G.innerZ1 = inner[CONV_DIMS];
    G.innerY0 = inner[1]; G.innerY1 = inner[CONV_DIMS + 1];
    G.innerX0 = inner[2]; G.innerX1 = inner[CONV_DIMS + 2];
    G.C0 = C0; G.K0 = K0; G.C1Max = C1Max; G.ksize = ksize;
    G.cblocks = (Cg + C0 - 1)/C0;
    G.coordtab = coordtab;
    G.ofstab = ofstab;

    const int64_t iplanesize = (int64_t)G.Di*G.Hi*G.Wi*C0;
    const int planeblocks = G.D*G.H*G.W;
    const int64_t planesize = (int64_t)planeblocks*K0;
    const int C1 = (C + C0 - 1)/C0, K1 = (K + K0 - 1)/K0;
    G.iplanesize = iplanesize;

    // A 1x1x1 kernel with unit strides and no padding addresses the input one-to-one, so the
    // spatial dims collapse into a single run of W pixels (mirrors the built-in). The built-in
    // flattens without testing the pads, which is safe only because the engine never pads a
    // 1x1 kernel; keep the test here so a padded one would take the bounds-checked path
    // instead of running off the end of a row.
    if (ksize == 1 && G.Sz*G.Sy*G.Sx == 1 &&
        (pads[0] | pads[1] | pads[2] | pads[CONV_DIMS] | pads[CONV_DIMS+1] | pads[CONV_DIMS+2]) == 0) {
        G.W *= G.D*G.H;
        G.Wi *= G.Di*G.Hi;
        G.D = G.Di = G.H = G.Hi = 1;
        G.innerZ1 = G.innerY1 = 1;
        G.innerX1 = G.W;
    }

    const float zbuf[CONV_MAX_C0] = {};
    G.zbuf = zbuf;
    float scalebuf[CONV_MAX_C0], biasbuf[CONV_MAX_C0], alphabuf[CONV_MAX_C0];

    // Wide path selection: pack P output channel blocks into one vector. P is what a full e32m2
    // register holds (1 at VLEN=128 -- no gain, so the wide path is off there). Tiling the P
    // blocks' weights costs ~2*P vector copies per (tap, c1, c0) and is amortised over the
    // positions of a chunk, so require a chunk long enough to pay for it; everything else keeps
    // the per-block path, which means a mis-tuned guard costs speed, never correctness.
    const int Pmax = (int)(__riscv_vsetvlmax_e32m2()/(size_t)K0);
    const int P = Pmax < Kblk ? Pmax : Kblk;
    const int chunk_len = planeblocks/nspat_chunks;
    const bool use_wide = P > 1 && chunk_len >= 8*P;
    const int64_t blockstride = (int64_t)ksize*C1Max*C0*K0;
    const int64_t twsize = use_wide ? (int64_t)ksize*G.cblocks*C0*P*K0 : 0;
    cv::AutoBuffer<float> widebuf(use_wide ? (size_t)(twsize + 4*P*K0) : 0);
    float* tw      = widebuf.data();                  // tiled weights
    float* tscale  = tw + twsize;                     // tiled scale/bias/alpha, P*K0 each
    float* tbias   = tscale + (int64_t)P*K0;
    float* talpha  = tbias + (int64_t)P*K0;
    float* scratch = talpha + (int64_t)P*K0;          // epilogue gather/scatter staging

    for (int t = task_start; t < task_end; t++) {
        const int block_id = t/nspat_chunks, chunk_id = t - block_id*nspat_chunks;
        const int p0 = (int)((int64_t)chunk_id*planeblocks/nspat_chunks);
        const int p1 = (int)((int64_t)(chunk_id + 1)*planeblocks/nspat_chunks);
        if (p1 <= p0)
            continue;

        const int n = block_id/(ngroups*Kblk);
        const int rem = block_id - n*(ngroups*Kblk);
        const int g = rem/Kblk, kblk = rem - g*Kblk;
        const int k_base = g*Kg + kblk*K0;
        if (k_base >= K)
            continue;

        // Group [gstart, gstart+np) of channel blocks into one wide vector. Sibling blocks are
        // nspat_chunks apart in the task grid; only group when every sibling is inside this
        // worker's range, otherwise a block outside it belongs to another worker. Members other
        // than the group start skip themselves, so nothing is computed (or stored) twice.
        int np = 0;
        int gstart = 0;
        if (use_wide) {
            gstart = kblk - kblk % P;
            np = Kblk - gstart < P ? Kblk - gstart : P;
            const int t0 = t - (kblk - gstart)*nspat_chunks;
            const int tlast = t0 + (np - 1)*nspat_chunks;
            if (t0 < task_start || tlast >= task_end)
                np = 0;                 // group straddles the range: fall back to per-block
            else if (t != t0)
                continue;               // already computed as part of its group
        }

        for (int kk = 0; kk < K0; kk++) {
            scalebuf[kk] = scale_all ? scale_all[k_base + kk] : 1.f;
            biasbuf[kk]  = bias_all  ? bias_all[k_base + kk]  : 0.f;
            alphabuf[kk] = prelu_slope ? prelu_slope[k_base + kk] : default_alpha;
        }

        const int c1_start = (g*Cg)/C0;
        const float* inpbase = inp_data + ((int64_t)n*C1 + c1_start)*iplanesize;
        const float* wbase = weights_all + (int64_t)(g*Kblk + kblk)*ksize*C1Max*C0*K0;
        const int64_t outofs = (int64_t)n*K1*planesize + (int64_t)k_base*planeblocks;
        float* outp = out_data + outofs + (int64_t)p0*K0;
        const float* resp = residual_data ? residual_data + outofs + (int64_t)p0*K0 : nullptr;

        if (np > 1) {
            // Tile the group's weights so the inner loop reads P*K0 contiguous lanes:
            //   tw[(i*cblocks + c1)*C0*P*K0 + c0*P*K0 + b*K0 + k] = w_(gstart+b)[i][c1][c0][k]
            const size_t vk = __riscv_vsetvl_e32m2((size_t)K0);
            for (int i = 0; i < ksize; i++) {
                for (int c1 = 0; c1 < G.cblocks; c1++) {
                    float* dst = tw + ((int64_t)i*G.cblocks + c1)*C0*np*K0;
                    for (int b = 0; b < np; b++) {
                        const float* src = wbase + (int64_t)b*blockstride +
                                           (int64_t)i*C1Max*C0*K0 + (int64_t)c1*C0*K0;
                        for (int c0 = 0; c0 < C0; c0++)
                            __riscv_vse32_v_f32m2(dst + (int64_t)c0*np*K0 + (int64_t)b*K0,
                                __riscv_vle32_v_f32m2(src + (int64_t)c0*K0, vk), vk);
                    }
                }
            }
            for (int b = 0; b < np; b++) {
                const int kb = k_base + b*K0;
                for (int kk = 0; kk < K0; kk++) {
                    tscale[b*K0 + kk] = scale_all ? scale_all[kb + kk] : 1.f;
                    tbias[b*K0 + kk]  = bias_all  ? bias_all[kb + kk]  : 0.f;
                    talpha[b*K0 + kk] = prelu_slope ? prelu_slope[kb + kk] : default_alpha;
                }
            }

            float* outw = outp;
            const float* resw = resp;
            int pw = p0;
            for (; pw + CONV_SPAT_BLOCK <= p1; pw += CONV_SPAT_BLOCK, outw += CONV_SPAT_BLOCK*K0) {
                int z, y, x;
                convDecodePos(pw, G.D, G.H, G.W, z, y, x);
                convWideBlock10(pw, z, y, x, inpbase, tw, outw, resw, planesize, np,
                                tscale, tbias, talpha, maxval, scratch, G);
                if (resw) resw += CONV_SPAT_BLOCK*K0;
            }
            for (; pw < p1; pw++, outw += K0) {
                int z, y, x;
                convDecodePos(pw, G.D, G.H, G.W, z, y, x);
                convWideBlock1(pw, z, y, x, inpbase, tw, outw, resw, planesize, np,
                               tscale, tbias, talpha, maxval, scratch, G);
                if (resw) resw += K0;
            }
            continue;
        }

        int p = p0;
        for (; p + CONV_SPAT_BLOCK <= p1; p += CONV_SPAT_BLOCK, outp += CONV_SPAT_BLOCK*K0) {
            int z, y, x;
            convDecodePos(p, G.D, G.H, G.W, z, y, x);
            convBlock10(p, z, y, x, inpbase, wbase, outp, resp,
                        scalebuf, biasbuf, alphabuf, maxval, G);
            if (resp) resp += CONV_SPAT_BLOCK*K0;
        }
        for (; p < p1; p++, outp += K0) {
            int z, y, x;
            convDecodePos(p, G.D, G.H, G.W, z, y, x);
            convBlock1(p, z, y, x, inpbase, wbase, outp, resp,
                       scalebuf, biasbuf, alphabuf, maxval, G);
            if (resp) resp += K0;
        }
    }

    return CV_HAL_ERROR_OK;
}

#endif // CV_HAL_RVV_1P0_ENABLED

}}} // cv::rvv_hal::dnn
