// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../conv2_common.hpp"
#include "opencv2/core/hal/intrin.hpp"

// === dispatched calls (implemented here)

namespace cv {
namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

cv::dnn::ConvFunc getConvFunc_(int depth, int C0);

CV_CPU_OPTIMIZATION_NAMESPACE_END
}} // cv::dnn::

// === implementation

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

namespace cv {
namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

#define CONV_ENABLE_SIMD 1

#undef CONV_ADD_NO_RESIDUAL2
#define CONV_ADD_NO_RESIDUAL2(idx0, idx1) /* empty */

#if CV_NEON_AARCH64

/////////////////////////// AARH64-optimized implementation /////////////////////////////

#undef CONV_INIT_SUMS
#define CONV_INIT_SUMS() \
    float32x4_t zz = vdupq_n_f32(0.f); \
    float32x4_t s0l = zz, s0h = zz, s1l = zz, s1h = zz; \
    float32x4_t s2l = zz, s2h = zz, s3l = zz, s3h = zz; \
    float32x4_t s4l = zz, s4h = zz, s5l = zz, s5h = zz; \
    float32x4_t s6l = zz, s6h = zz, s7l = zz, s7h = zz; \
    float32x4_t s8l = zz, s8h = zz, s9l = zz, s9h = zz

#undef CONV_UPDATE_BLOCK
#define CONV_UPDATE_BLOCK(w_ofs, lane) \
    wl = vld1q_f32(wptr + w_ofs*K0 + 0); \
    wh = vld1q_f32(wptr + w_ofs*K0 + 4); \
    s0l = vfmaq_laneq_f32(s0l, wl, x0, lane); \
    s0h = vfmaq_laneq_f32(s0h, wh, x0, lane); \
    s1l = vfmaq_laneq_f32(s1l, wl, x1, lane); \
    s1h = vfmaq_laneq_f32(s1h, wh, x1, lane); \
    s2l = vfmaq_laneq_f32(s2l, wl, x2, lane); \
    s2h = vfmaq_laneq_f32(s2h, wh, x2, lane); \
    s3l = vfmaq_laneq_f32(s3l, wl, x3, lane); \
    s3h = vfmaq_laneq_f32(s3h, wh, x3, lane); \
    s4l = vfmaq_laneq_f32(s4l, wl, x4, lane); \
    s4h = vfmaq_laneq_f32(s4h, wh, x4, lane); \
    s5l = vfmaq_laneq_f32(s5l, wl, x5, lane); \
    s5h = vfmaq_laneq_f32(s5h, wh, x5, lane); \
    s6l = vfmaq_laneq_f32(s6l, wl, x6, lane); \
    s6h = vfmaq_laneq_f32(s6h, wh, x6, lane); \
    s7l = vfmaq_laneq_f32(s7l, wl, x7, lane); \
    s7h = vfmaq_laneq_f32(s7h, wh, x7, lane); \
    s8l = vfmaq_laneq_f32(s8l, wl, x8, lane); \
    s8h = vfmaq_laneq_f32(s8h, wh, x8, lane); \
    s9l = vfmaq_laneq_f32(s9l, wl, x9, lane); \
    s9h = vfmaq_laneq_f32(s9h, wh, x9, lane)

#undef CONV_UPDATE_LOOP_BODY
#define CONV_UPDATE_LOOP_BODY() \
    float32x4_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9; \
    float32x4_t wl, wh; \
    \
    x0 = vld1q_f32(inptr[0]); \
    x1 = vld1q_f32(inptr[1]); \
    x2 = vld1q_f32(inptr[2]); \
    x3 = vld1q_f32(inptr[3]); \
    x4 = vld1q_f32(inptr[4]); \
    x5 = vld1q_f32(inptr[5]); \
    x6 = vld1q_f32(inptr[6]); \
    x7 = vld1q_f32(inptr[7]); \
    x8 = vld1q_f32(inptr[8]); \
    x9 = vld1q_f32(inptr[9]); \
    \
    CONV_UPDATE_BLOCK(0, 0); \
    CONV_UPDATE_BLOCK(1, 1); \
    CONV_UPDATE_BLOCK(2, 2); \
    CONV_UPDATE_BLOCK(3, 3); \
    \
    x0 = vld1q_f32(inptr[0] + 4); \
    x1 = vld1q_f32(inptr[1] + 4); \
    x2 = vld1q_f32(inptr[2] + 4); \
    x3 = vld1q_f32(inptr[3] + 4); \
    x4 = vld1q_f32(inptr[4] + 4); \
    x5 = vld1q_f32(inptr[5] + 4); \
    x6 = vld1q_f32(inptr[6] + 4); \
    x7 = vld1q_f32(inptr[7] + 4); \
    x8 = vld1q_f32(inptr[8] + 4); \
    x9 = vld1q_f32(inptr[9] + 4); \
    \
    inptr[0] += inpstep[0]; inptr[1] += inpstep[1]; \
    inptr[2] += inpstep[2]; inptr[3] += inpstep[3]; \
    inptr[4] += inpstep[4]; inptr[5] += inpstep[5]; \
    inptr[6] += inpstep[6]; inptr[7] += inpstep[7]; \
    inptr[8] += inpstep[8]; inptr[9] += inpstep[9]; \
    \
    CONV_UPDATE_BLOCK(4, 0); \
    CONV_UPDATE_BLOCK(5, 1); \
    CONV_UPDATE_BLOCK(6, 2); \
    CONV_UPDATE_BLOCK(7, 3)

#undef CONV_START_FINALIZE_OUT
#define CONV_START_FINALIZE_OUT() \
    float32x4_t vscale_lo = vld1q_f32(scalebuf), vscale_hi = vld1q_f32(scalebuf + 4); \
    float32x4_t vbias_lo = vld1q_f32(biasbuf), vbias_hi = vld1q_f32(biasbuf + 4); \
    float32x4_t valpha = vdupq_n_f32(alpha); \
    float32x4_t vmaxval = vdupq_n_f32(maxval)

#define CONV_ADD_RESIDUAL2(idx0, idx1) \
    s##idx0##l = vaddq_f32(s##idx0##l, vld1q_f32(tmpbuf + idx0*K0)); \
    s##idx0##h = vaddq_f32(s##idx0##h, vld1q_f32(tmpbuf + idx0*K0 + 4)); \
    s##idx1##l = vaddq_f32(s##idx1##l, vld1q_f32(tmpbuf + idx1*K0)); \
    s##idx1##h = vaddq_f32(s##idx1##h, vld1q_f32(tmpbuf + idx1*K0 + 4))

#undef CONV_FINALIZE_OUT2
#define CONV_FINALIZE_OUT2(idx0, idx1, add_residual2) \
    s##idx0##l = vfmaq_f32(vbias_lo, s##idx0##l, vscale_lo); \
    s##idx0##h = vfmaq_f32(vbias_hi, s##idx0##h, vscale_hi); \
    s##idx1##l = vfmaq_f32(vbias_lo, s##idx1##l, vscale_lo); \
    s##idx1##h = vfmaq_f32(vbias_hi, s##idx1##h, vscale_hi); \
    add_residual2(idx0, idx1); \
    s##idx0##l = vbslq_f32(vcgeq_f32(s##idx0##l, zz), s##idx0##l, vmulq_f32(s##idx0##l, valpha)); \
    s##idx0##h = vbslq_f32(vcgeq_f32(s##idx0##h, zz), s##idx0##h, vmulq_f32(s##idx0##h, valpha)); \
    s##idx1##l = vbslq_f32(vcgeq_f32(s##idx1##l, zz), s##idx1##l, vmulq_f32(s##idx1##l, valpha)); \
    s##idx1##h = vbslq_f32(vcgeq_f32(s##idx1##h, zz), s##idx1##h, vmulq_f32(s##idx1##h, valpha)); \
    s##idx0##l = vminq_f32(s##idx0##l, vmaxval); \
    s##idx0##h = vminq_f32(s##idx0##h, vmaxval); \
    s##idx1##l = vminq_f32(s##idx1##l, vmaxval); \
    s##idx1##h = vminq_f32(s##idx1##h, vmaxval); \
    vst1q_f32(outbuf + idx0*K0, s##idx0##l); \
    vst1q_f32(outbuf + idx0*K0 + 4, s##idx0##h); \
    vst1q_f32(outbuf + idx1*K0, s##idx1##l); \
    vst1q_f32(outbuf + idx1*K0 + 4, s##idx1##h)

#undef CONV_FINALIZE_OUT_ALL
#define CONV_FINALIZE_OUT_ALL() \
    CONV_START_FINALIZE_OUT(); \
    if (resptr) { \
        CONV_FINALIZE_OUT2(0, 1, CONV_ADD_RESIDUAL2); \
        CONV_FINALIZE_OUT2(2, 3, CONV_ADD_RESIDUAL2); \
        CONV_FINALIZE_OUT2(4, 5, CONV_ADD_RESIDUAL2); \
        CONV_FINALIZE_OUT2(6, 7, CONV_ADD_RESIDUAL2); \
        CONV_FINALIZE_OUT2(8, 9, CONV_ADD_RESIDUAL2); \
    } else { \
        CONV_FINALIZE_OUT2(0, 1, CONV_ADD_NO_RESIDUAL2); \
        CONV_FINALIZE_OUT2(2, 3, CONV_ADD_NO_RESIDUAL2); \
        CONV_FINALIZE_OUT2(4, 5, CONV_ADD_NO_RESIDUAL2); \
        CONV_FINALIZE_OUT2(6, 7, CONV_ADD_NO_RESIDUAL2); \
        CONV_FINALIZE_OUT2(8, 9, CONV_ADD_NO_RESIDUAL2); \
    }

#elif CV_SIMD128

/////////////////////////// generic branch for arch's with 128-bit SIMD /////////////////////////////

#undef CONV_INIT_SUMS
#define CONV_INIT_SUMS() \
    v_float32x4 zz = v_setzero_f32(); \
    v_float32x4 s0l = zz, s0h = zz, s1l = zz, s1h = zz; \
    v_float32x4 s2l = zz, s2h = zz, s3l = zz, s3h = zz; \
    v_float32x4 s4l = zz, s4h = zz, s5l = zz, s5h = zz; \
    v_float32x4 s6l = zz, s6h = zz, s7l = zz, s7h = zz; \
    v_float32x4 s8l = zz, s8h = zz, s9l = zz, s9h = zz

#undef CONV_UPDATE_BLOCK
#define CONV_UPDATE_BLOCK2x8(ofs, idx0, idx1) \
    x0 = v_setall_f32(inptr[idx0][ofs]); \
    x1 = v_setall_f32(inptr[idx1][ofs]); \
    s##idx0##l = v_fma(x0, w0l, s##idx0##l); \
    s##idx0##h = v_fma(x0, w0h, s##idx0##h); \
    s##idx1##l = v_fma(x1, w0l, s##idx1##l); \
    s##idx1##h = v_fma(x1, w0h, s##idx1##h); \
    x0 = v_setall_f32(inptr[idx0][ofs+1]); \
    x1 = v_setall_f32(inptr[idx1][ofs+1]); \
    s##idx0##l = v_fma(x0, w1l, s##idx0##l); \
    s##idx0##h = v_fma(x0, w1h, s##idx0##h); \
    s##idx1##l = v_fma(x1, w1l, s##idx1##l); \
    s##idx1##h = v_fma(x1, w1h, s##idx1##h); \
    x0 = v_setall_f32(inptr[idx0][ofs+2]); \
    x1 = v_setall_f32(inptr[idx1][ofs+2]); \
    s##idx0##l = v_fma(x0, w2l, s##idx0##l); \
    s##idx0##h = v_fma(x0, w2h, s##idx0##h); \
    s##idx1##l = v_fma(x1, w2l, s##idx1##l); \
    s##idx1##h = v_fma(x1, w2h, s##idx1##h); \
    x0 = v_setall_f32(inptr[idx0][ofs+3]); \
    x1 = v_setall_f32(inptr[idx1][ofs+3]); \
    s##idx0##l = v_fma(x0, w3l, s##idx0##l); \
    s##idx0##h = v_fma(x0, w3h, s##idx0##h); \
    s##idx1##l = v_fma(x1, w3l, s##idx1##l); \
    s##idx1##h = v_fma(x1, w3h, s##idx1##h)

#undef CONV_UPDATE_LOOP_BODY
#define CONV_UPDATE_LOOP_BODY() \
    v_float32x4 x0, x1; \
    v_float32x4 w0l, w0h, w1l, w1h, w2l, w2h, w3l, w3h; \
    \
    w0l = v_load(wptr + 0*K0); w0h = v_load(wptr + 0*K0 + 4); \
    w1l = v_load(wptr + 1*K0); w1h = v_load(wptr + 1*K0 + 4); \
    w2l = v_load(wptr + 2*K0); w2h = v_load(wptr + 2*K0 + 4); \
    w3l = v_load(wptr + 3*K0); w3h = v_load(wptr + 3*K0 + 4); \
    \
    CONV_UPDATE_BLOCK2x8(0, 0, 1); \
    CONV_UPDATE_BLOCK2x8(0, 2, 3); \
    CONV_UPDATE_BLOCK2x8(0, 4, 5); \
    CONV_UPDATE_BLOCK2x8(0, 6, 7); \
    CONV_UPDATE_BLOCK2x8(0, 8, 9); \
    \
    w0l = v_load(wptr + 4*K0); w0h = v_load(wptr + 4*K0 + 4); \
    w1l = v_load(wptr + 5*K0); w1h = v_load(wptr + 5*K0 + 4); \
    w2l = v_load(wptr + 6*K0); w2h = v_load(wptr + 6*K0 + 4); \
    w3l = v_load(wptr + 7*K0); w3h = v_load(wptr + 7*K0 + 4); \
    \
    CONV_UPDATE_BLOCK2x8(4, 0, 1); \
    CONV_UPDATE_BLOCK2x8(4, 2, 3); \
    CONV_UPDATE_BLOCK2x8(4, 4, 5); \
    CONV_UPDATE_BLOCK2x8(4, 6, 7); \
    CONV_UPDATE_BLOCK2x8(4, 8, 9); \
    \
    inptr[0] += inpstep[0]; inptr[1] += inpstep[1]; \
    inptr[2] += inpstep[2]; inptr[3] += inpstep[3]; \
    inptr[4] += inpstep[4]; inptr[5] += inpstep[5]; \
    inptr[6] += inpstep[6]; inptr[7] += inpstep[7]; \
    inptr[8] += inpstep[8]; inptr[9] += inpstep[9]

#undef CONV_START_FINALIZE_OUT
#define CONV_START_FINALIZE_OUT() \
    v_float32x4 vscale_lo = v_load(scalebuf), vscale_hi = v_load(scalebuf + 4); \
    v_float32x4 vbias_lo = v_load(biasbuf), vbias_hi = v_load(biasbuf + 4); \
    v_float32x4 valpha = v_setall_f32(alpha); \
    v_float32x4 vmaxval = v_setall_f32(maxval)

#define CONV_ADD_RESIDUAL2(idx0, idx1) \
    s##idx0##l = v_add(s##idx0##l, v_load(tmpbuf + idx0*K0)); \
    s##idx0##h = v_add(s##idx0##h, v_load(tmpbuf + idx0*K0 + 4)); \
    s##idx1##l = v_add(s##idx1##l, v_load(tmpbuf + idx1*K0)); \
    s##idx1##h = v_add(s##idx1##h, v_load(tmpbuf + idx1*K0 + 4))

#undef CONV_FINALIZE_OUT2
#define CONV_FINALIZE_OUT2(idx0, idx1, add_residual2) \
    s##idx0##l = v_fma(s##idx0##l, vscale_lo, vbias_lo); \
    s##idx0##h = v_fma(s##idx0##h, vscale_hi, vbias_hi); \
    s##idx1##l = v_fma(s##idx1##l, vscale_lo, vbias_lo); \
    s##idx1##h = v_fma(s##idx1##h, vscale_hi, vbias_hi); \
    add_residual2(idx0, idx1); \
    s##idx0##l = v_select(v_ge(s##idx0##l, zz), s##idx0##l, v_mul(s##idx0##l, valpha)); \
    s##idx0##h = v_select(v_ge(s##idx0##h, zz), s##idx0##h, v_mul(s##idx0##h, valpha)); \
    s##idx1##l = v_select(v_ge(s##idx1##l, zz), s##idx1##l, v_mul(s##idx1##l, valpha)); \
    s##idx1##h = v_select(v_ge(s##idx1##h, zz), s##idx1##h, v_mul(s##idx1##h, valpha)); \
    s##idx0##l = v_min(s##idx0##l, vmaxval); \
    s##idx0##h = v_min(s##idx0##h, vmaxval); \
    s##idx1##l = v_min(s##idx1##l, vmaxval); \
    s##idx1##h = v_min(s##idx1##h, vmaxval); \
    v_store(outbuf + idx0*K0, s##idx0##l); \
    v_store(outbuf + idx0*K0 + 4, s##idx0##h); \
    v_store(outbuf + idx1*K0, s##idx1##l); \
    v_store(outbuf + idx1*K0 + 4, s##idx1##h)

#undef CONV_FINALIZE_OUT_ALL
#define CONV_FINALIZE_OUT_ALL() \
    CONV_START_FINALIZE_OUT(); \
    if (resptr) { \
        CONV_FINALIZE_OUT2(0, 1, CONV_ADD_RESIDUAL2); \
        CONV_FINALIZE_OUT2(2, 3, CONV_ADD_RESIDUAL2); \
        CONV_FINALIZE_OUT2(4, 5, CONV_ADD_RESIDUAL2); \
        CONV_FINALIZE_OUT2(6, 7, CONV_ADD_RESIDUAL2); \
        CONV_FINALIZE_OUT2(8, 9, CONV_ADD_RESIDUAL2); \
    } else { \
        CONV_FINALIZE_OUT2(0, 1, CONV_ADD_NO_RESIDUAL2); \
        CONV_FINALIZE_OUT2(2, 3, CONV_ADD_NO_RESIDUAL2); \
        CONV_FINALIZE_OUT2(4, 5, CONV_ADD_NO_RESIDUAL2); \
        CONV_FINALIZE_OUT2(6, 7, CONV_ADD_NO_RESIDUAL2); \
        CONV_FINALIZE_OUT2(8, 9, CONV_ADD_NO_RESIDUAL2); \
    }

#else

#undef CONV_ENABLE_SIMD

#endif

static void conv32fC8(const void* inp__, const void* residual__, void* out__,
                      const ConvState& cs, const void* weights__,
                      const float* scale__, const float* bias__)
{
    using FT = float;
    constexpr int C0shift = 3, K0shift = C0shift;
    constexpr int C0 = 1 << C0shift;
    constexpr int K0 = C0;
    const MatShape& inpshape = cs.inpshape;
    const MatShape& outshape = cs.outshape;

    CV_Assert_N(inpshape.layout == DATA_LAYOUT_BLOCK, outshape.layout == DATA_LAYOUT_BLOCK);
    CV_Assert_N(inpshape.back() == C0, outshape.back() == K0);

    int K_ = outshape.channels();
    int N = inpshape[0];

    const int total_blocks = N * cs.ngroups * cs.Kblk;

    if ((K_/cs.ngroups) % K0 != 0) {
        // if there could be 'padding' channels in the output,
        // clear the output before the parallel loop
        // to make sure that all the padding channels are cleared.
        size_t outtotal = outshape.total()*sizeof(FT);
        memset(out__, 0, outtotal);
    }

    parallel_for_(Range(0, total_blocks), [&](const Range& range) {
        constexpr int MAX_CONV_DIMS = ConvState::MAX_CONV_DIMS;
        const int C = inpshape.channels(), K = outshape.channels();
        const int C1 = (C + C0 - 1)/C0, K1 = (K + K0 - 1)/K0;
        const int ngroups = cs.ngroups, Kblk = cs.Kblk, C1Max = cs.C1Max;
        const int Cg = C / ngroups;
        const int Kg = K / ngroups;
        int ksize = 1;
        for (int i = 0; i < MAX_CONV_DIMS; i++)
            ksize *= cs.kshape[i];
        int ndims = inpshape.dims;
        int D = ndims >= 6 ? outshape[ndims-4] : 1;
        int H = ndims >= 5 ? outshape[ndims-3] : 1;
        int W = outshape[ndims-2];
        int Di = ndims >= 6 ? inpshape[ndims-4] : 1;
        int Hi = ndims >= 5 ? inpshape[ndims-3] : 1;
        int Wi = inpshape[ndims-2];
        const int Sz = cs.strides[0], Sy = cs.strides[1], Sx = cs.strides[2];
        const int padZ = cs.pads[0], padY = cs.pads[1], padX = cs.pads[2];
        const float* scaleptr = (const float*)scale__;
        const float* biasptr = (const float*)bias__;
        const int* ofsZYX = cs.coordtab.data();
        int planeblocks = D*H*W;
        int planesize = planeblocks*K0;
        int iplanesize = Di*Hi*Wi*C0;

    #if CONV_ENABLE_SIMD
        int innerZ0 = cs.inner[0], innerZ1 = cs.inner[MAX_CONV_DIMS];
        int innerY0 = cs.inner[1], innerY1 = cs.inner[MAX_CONV_DIMS+1];
        int innerX0 = cs.inner[2], innerX1 = cs.inner[MAX_CONV_DIMS+2];
        float zbuf[C0] = {};
    #endif

        FastActivation fastActivation = cs.fastActivation;
        const float* activParams = cs.activParams;
        activation_func_t activation = cs.activation;
        float maxval = fastActivation == FAST_ACTIV_CLIP ? activParams[1] : FLT_MAX;
        float alpha = fastActivation == FAST_ACTIV_LEAKY_RELU ? activParams[0] :
                    fastActivation == FAST_ACTIV_NONE ? 1.f : 0.f;
        float scalebuf[K0], biasbuf[K0];

        // 1x1x1 convolution with (1,1,1) strides:
        // flatten input/output tensors in this case to accelerate address computations
        if (ksize == 1 && Sz*Sy*Sx == 1) {
            W *= D*H;
            Wi *= Di*Hi;
            D = Di = H = Hi = 1;
        #if CONV_ENABLE_SIMD
            innerZ1 = innerY1 = 1;
            innerX1 = W;
        #endif
        }

        for (int t = range.start; t < range.end; t++) {
            const int n = t / (ngroups * Kblk);
            const int rem = t - n * (ngroups * Kblk);
            const int g = rem / Kblk;
            const int kblk = rem - g * Kblk;

            const int k_base = g * Kg + kblk * K0;
            if (k_base >= K) continue;

            const int k_count = std::min(std::min(K0, Kg - kblk*K0), K - k_base);
            bool aligned_k = (k_base & (K0-1)) == 0 && k_count == K0;

            const int c_start  = g * Cg;
            const int c00      = c_start & (C0-1);
            const int c1_start = c_start >> C0shift;
            const int cblocks  = (c00 + Cg + C0 - 1) >> C0shift;
            const float* inpbaseptr = (float*)inp__ + (n * C1 + c1_start) * iplanesize;
            const float* wbaseptr = (float*)weights__ + (g*Kblk + kblk)*(ksize*C1Max*C0*K0);

            {
                int kk = 0;
                for (; kk < k_count; kk++) {
                    scalebuf[kk] = scaleptr ? scaleptr[k_base + kk] : 1.f;
                    biasbuf[kk] = biasptr ? biasptr[k_base + kk] : 0.f;
                }
                for (; kk < K0; kk++) {
                    scalebuf[kk] = 0.f;
                    biasbuf[kk] = 0.f;
                }
            }

            constexpr int SPAT_BLOCK_SIZE = 10;
            float* outptr = (float*)out__ + n*(K1*planesize);
            const float* resptr = residual__ ? (float*)residual__ + n*(K1*planesize) : nullptr;
            float tmpbuf[SPAT_BLOCK_SIZE*K0] = {};
            int p = 0;

        #if CONV_ENABLE_SIMD
            for (; p < planeblocks; p += SPAT_BLOCK_SIZE,
                                    outptr += SPAT_BLOCK_SIZE*K0)
            {
                Vec3i pt[SPAT_BLOCK_SIZE];
                bool inner[SPAT_BLOCK_SIZE];

                if (p + SPAT_BLOCK_SIZE > planeblocks) {
                    if (p == 0)
                        break;
                    int p_new = planeblocks - SPAT_BLOCK_SIZE;
                    int dp = p_new - p;
                    outptr += dp*K0;
                    resptr += (resptr ? dp*K0 : 0);
                    p = p_new;
                }

                if (resptr) {
                    if (aligned_k) {
                        memcpy(tmpbuf, resptr + k_base*planeblocks, SPAT_BLOCK_SIZE*K0*sizeof(FT));
                    } else {
                        for (int kk = 0; kk < k_count; ++kk) {
                            const int k = k_base + kk;
                            int kofs = (k >> K0shift) * planesize + (k & (K0-1));
                            for (int j = 0; j < SPAT_BLOCK_SIZE; j++)
                                tmpbuf[kk + j*K0] = resptr[kofs + j*K0];
                        }
                    }
                    resptr += SPAT_BLOCK_SIZE*K0;
                }

                if ((p % W) + SPAT_BLOCK_SIZE <= W) {
                    int zj = p / (H*W);
                    int yxj = p - zj*(H*W);
                    int yj = yxj / W;
                    int x = yxj - yj*W;
                    const bool zy_inner = (zj >= innerZ0 && zj < innerZ1) && (yj >= innerY0 && yj < innerY1);
                    for (int j = 0; j < SPAT_BLOCK_SIZE; j++) {
                        int xj = x + j;
                        pt[j] = Vec3i(zj*Sz - padZ, yj*Sy - padY, xj*Sx - padX);
                        inner[j] = zy_inner && (xj >= innerX0 && xj < innerX1);
                    }
                } else {
                    for (int j = 0; j < SPAT_BLOCK_SIZE; j++) {
                        int pj = p + j;
                        int zj = pj / (H*W);
                        int yxj = pj - zj*(H*W);
                        int yj = yxj / W;
                        int xj = yxj - yj*W;
                        pt[j] = Vec3i(zj*Sz - padZ, yj*Sy - padY, xj*Sx - padX);
                        inner[j] = (zj >= innerZ0 && zj < innerZ1) &&
                                   (yj >= innerY0 && yj < innerY1) &&
                                   (xj >= innerX0 && xj < innerX1);
                    }
                }

                CONV_INIT_SUMS();

                for (int i = 0; i < ksize; i++) {
                    const float* inptr[SPAT_BLOCK_SIZE];
                    int inpstep[SPAT_BLOCK_SIZE];
                    for (int j = 0; j < SPAT_BLOCK_SIZE; j++) {
                        Vec3i ptj = pt[j];
                        int zij = ptj[0] + ofsZYX[i*3 + 0];
                        int yij = ptj[1] + ofsZYX[i*3 + 1];
                        int xij = ptj[2] + ofsZYX[i*3 + 2];
                        if (inner[j] || ((((unsigned)zij < (unsigned)Di)&
                                          ((unsigned)yij < (unsigned)Hi)&
                                          ((unsigned)xij < (unsigned)Wi)) != 0)) {
                            inptr[j] = inpbaseptr + (((zij * Hi) + yij) * Wi + xij) * C0;
                            inpstep[j] = iplanesize;
                        } else {
                            inptr[j] = zbuf;
                            inpstep[j] = 0;
                        }
                    }
                    const float* wptr = wbaseptr + i*C1Max*K0*C0;

                    for (int c1 = 0; c1 < cblocks; c1++, wptr += C0*K0) {
                        CONV_UPDATE_LOOP_BODY();
                    }
                }

                float* outbuf = aligned_k ? outptr + k_base*planeblocks : tmpbuf;
                CONV_FINALIZE_OUT_ALL();

                if (activation) {
                    activation(outbuf, outbuf, SPAT_BLOCK_SIZE*K0, activParams);
                }

                if (!aligned_k) {
                    for (int kk = 0; kk < k_count; ++kk) {
                        const int k = k_base + kk;
                        int kofs = (k >> K0shift) * planesize + (k & (K0-1));
                        for (int j = 0; j < SPAT_BLOCK_SIZE; j++)
                            outptr[kofs + j*K0] = tmpbuf[kk + j*K0];
                    }
                }
            }
        #endif

            float resbuf[K0] = {};

            for (; p < planeblocks; p++, outptr += K0, resptr += (resptr ? K0 : 0))
            {
                int zj = p / (H*W);
                int yxj = p - zj*(H*W);
                int yj = yxj / W;
                int xj = yxj - yj*W;
                int zi_base = zj*Sz - padZ;
                int yi_base = yj*Sy - padY;
                int xi_base = xj*Sx - padX;

            #if CV_SIMD128
                v_float32x4 zz = v_setzero_f32();
                v_float32x4 s0 = zz, s1 = zz;
            #else
                for (int kk = 0; kk < K0; kk++) {
                    tmpbuf[kk] = 0.f;
                }
            #endif

                if (resptr) {
                    for (int kk = 0; kk < k_count; ++kk) {
                        const int k = k_base + kk;
                        int kofs = (k >> K0shift) * planesize + (k & (K0-1));
                        resbuf[kk] = resptr[kofs];
                    }
                }

                for (int i = 0; i < ksize; i++) {
                    int zi = zi_base + ofsZYX[i*3 + 0];
                    int yi = yi_base + ofsZYX[i*3 + 1];
                    int xi = xi_base + ofsZYX[i*3 + 2];

                    if ((((unsigned)zi >= (unsigned)Di) |
                         ((unsigned)yi >= (unsigned)Hi) |
                         ((unsigned)xi >= (unsigned)Wi)) != 0)
                        continue;

                    const float* inptr = inpbaseptr + (((zi * Hi) + yi) * Wi + xi) * C0;
                    const float* wptr = wbaseptr + i*C1Max*K0*C0;

                    for (int c1 = 0; c1 < cblocks; ++c1, inptr += iplanesize, wptr += K0*C0) {
                    #if CV_SIMD128
                        v_float32x4 w0, w1, x;
                        #undef CONV_UPDATE_BLOCK1
                        #define CONV_UPDATE_BLOCK1(ofs) \
                            w0 = v_load(wptr + ofs*K0); w1 = v_load(wptr + ofs*K0 + 4); \
                            x = v_setall_f32(inptr[ofs]); \
                            s0 = v_fma(x, w0, s0); s1 = v_fma(x, w1, s1)
                        CONV_UPDATE_BLOCK1(0);
                        CONV_UPDATE_BLOCK1(1);
                        CONV_UPDATE_BLOCK1(2);
                        CONV_UPDATE_BLOCK1(3);
                        CONV_UPDATE_BLOCK1(4);
                        CONV_UPDATE_BLOCK1(5);
                        CONV_UPDATE_BLOCK1(6);
                        CONV_UPDATE_BLOCK1(7);
                    #else
                        for (int c0 = 0; c0 < C0; ++c0) {
                            const float xval = inptr[c0];
                            for (int kk = 0; kk < K0; ++kk)
                                tmpbuf[kk] += xval * wptr[c0*K0 + kk];
                        }
                    #endif
                    }
                }

                float* outbuf = aligned_k ? outptr + k_base*planeblocks : tmpbuf;

            #if CV_SIMD128
                v_float32x4 vscale_lo = v_load(scalebuf), vscale_hi = v_load(scalebuf + 4);
                v_float32x4 vbias_lo = v_load(biasbuf), vbias_hi = v_load(biasbuf + 4);
                v_float32x4 valpha = v_setall_f32(alpha);
                v_float32x4 vmaxval = v_setall_f32(maxval);

                s0 = v_fma(s0, vscale_lo, vbias_lo);
                s1 = v_fma(s1, vscale_hi, vbias_hi);
                s0 = v_add(s0, v_load(resbuf));
                s1 = v_add(s1, v_load(resbuf + 4));
                s0 = v_select(v_ge(s0, zz), s0, v_mul(s0, valpha));
                s1 = v_select(v_ge(s1, zz), s1, v_mul(s1, valpha));
                s0 = v_min(s0, vmaxval);
                s1 = v_min(s1, vmaxval);
                v_store(outbuf, s0);
                v_store(outbuf + 4, s1);
            #else
                for (int kk = 0; kk < K0; kk++) {
                    float v = tmpbuf[kk]*scalebuf[kk] + biasbuf[kk] + resbuf[kk];
                    v = std::min(v*(v >= 0 ? 1.f : alpha), maxval);
                    outbuf[kk] = v;
                }
            #endif

                if (activation) {
                    activation(outbuf, outbuf, K0, activParams);
                }

                if (!aligned_k) {
                    for (int kk = 0; kk < k_count; kk++) {
                        const int k = k_base + kk;
                        int kofs = (k >> K0shift) * planesize + (k & (K0-1));
                        outptr[kofs] = tmpbuf[kk];
                    }
                }
            }
        }
    });
}

cv::dnn::ConvFunc getConvFunc_(int depth, int C0)
{
    ConvFunc func = nullptr;
    if (depth == CV_32F && C0 == 8) {
        func = conv32fC8;
    }
    return func;
}

CV_CPU_OPTIMIZATION_NAMESPACE_END
}}
#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
