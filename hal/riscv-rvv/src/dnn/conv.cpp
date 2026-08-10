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
// NOTE the K0-lane vector uses 8 of VLMAX lanes, so throughput does not grow past VLEN=256.
// Filling the register at wider VLEN means processing several output channel blocks at once,
// which is a separable follow-up (cf. the register-fill path of depthwise.cpp).

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
