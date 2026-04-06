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

#if (defined CV_NEON_AARCH64) && CV_NEON_AARCH64

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
    float32x4_t valpha_lo = vld1q_f32(alphabuf), valpha_hi = vld1q_f32(alphabuf + 4); \
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
    s##idx0##l = vbslq_f32(vcgeq_f32(s##idx0##l, zz), s##idx0##l, vmulq_f32(s##idx0##l, valpha_lo)); \
    s##idx0##h = vbslq_f32(vcgeq_f32(s##idx0##h, zz), s##idx0##h, vmulq_f32(s##idx0##h, valpha_hi)); \
    s##idx1##l = vbslq_f32(vcgeq_f32(s##idx1##l, zz), s##idx1##l, vmulq_f32(s##idx1##l, valpha_lo)); \
    s##idx1##h = vbslq_f32(vcgeq_f32(s##idx1##h, zz), s##idx1##h, vmulq_f32(s##idx1##h, valpha_hi)); \
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

#elif CV_SIMD256

///////////// generic branch for arch's with 256-bit SIMD (however, it's tweaked for AVX2 with just 16 registers) ////////////////

#undef CONV_INIT_SUMS
#define CONV_INIT_SUMS() \
    v_float32x8 zz = v256_setzero_f32(); \
    v_float32x8 s0 = zz, s1 = zz, s2 = zz, s3 = zz, s4 = zz; \
    v_float32x8 s5 = zz, s6 = zz, s7 = zz, s8 = zz, s9 = zz

#undef CONV_UPDATE_BLOCK
#define CONV_UPDATE_BLOCK2x8(ofs, idx0, idx1) \
    x0 = v256_setall_f32(inptr[idx0][ofs]); \
    x1 = v256_setall_f32(inptr[idx1][ofs]); \
    s##idx0 = v_fma(x0, w0, s##idx0); \
    s##idx1 = v_fma(x1, w0, s##idx1); \
    x0 = v256_setall_f32(inptr[idx0][ofs+1]); \
    x1 = v256_setall_f32(inptr[idx1][ofs+1]); \
    s##idx0 = v_fma(x0, w1, s##idx0); \
    s##idx1 = v_fma(x1, w1, s##idx1); \
    x0 = v256_setall_f32(inptr[idx0][ofs+2]); \
    x1 = v256_setall_f32(inptr[idx1][ofs+2]); \
    s##idx0 = v_fma(x0, w2, s##idx0); \
    s##idx1 = v_fma(x1, w2, s##idx1); \
    x0 = v256_setall_f32(inptr[idx0][ofs+3]); \
    x1 = v256_setall_f32(inptr[idx1][ofs+3]); \
    s##idx0 = v_fma(x0, w3, s##idx0); \
    s##idx1 = v_fma(x1, w3, s##idx1)

#undef CONV_UPDATE_LOOP_BODY
#define CONV_UPDATE_LOOP_BODY() \
    v_float32x8 x0, x1; \
    v_float32x8 w0, w1, w2, w3; \
    \
    w0 = v256_load(wptr + 0*K0); \
    w1 = v256_load(wptr + 1*K0); \
    w2 = v256_load(wptr + 2*K0); \
    w3 = v256_load(wptr + 3*K0); \
    \
    CONV_UPDATE_BLOCK2x8(0, 0, 1); \
    CONV_UPDATE_BLOCK2x8(0, 2, 3); \
    CONV_UPDATE_BLOCK2x8(0, 4, 5); \
    CONV_UPDATE_BLOCK2x8(0, 6, 7); \
    CONV_UPDATE_BLOCK2x8(0, 8, 9); \
    \
    w0 = v256_load(wptr + 4*K0); \
    w1 = v256_load(wptr + 5*K0); \
    w2 = v256_load(wptr + 6*K0); \
    w3 = v256_load(wptr + 7*K0); \
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
    v_float32x8 vscale = v256_load(scalebuf); \
    v_float32x8 vbias = v256_load(biasbuf); \
    v_float32x8 valpha = v256_load(alphabuf); \
    v_float32x8 vmaxval = v256_setall_f32(maxval)

#define CONV_ADD_RESIDUAL2(idx0, idx1) \
    s##idx0 = v_add(s##idx0, v256_load(tmpbuf + idx0*K0)); \
    s##idx1 = v_add(s##idx1, v256_load(tmpbuf + idx1*K0))

#undef CONV_FINALIZE_OUT2
#define CONV_FINALIZE_OUT2(idx0, idx1, add_residual2) \
    s##idx0 = v_fma(s##idx0, vscale, vbias); \
    s##idx1 = v_fma(s##idx1, vscale, vbias); \
    add_residual2(idx0, idx1); \
    s##idx0 = v_select(v_ge(s##idx0, zz), s##idx0, v_mul(s##idx0, valpha)); \
    s##idx1 = v_select(v_ge(s##idx1, zz), s##idx1, v_mul(s##idx1, valpha)); \
    s##idx0 = v_min(s##idx0, vmaxval); \
    s##idx1 = v_min(s##idx1, vmaxval); \
    v_store(outbuf + idx0*K0, s##idx0); \
    v_store(outbuf + idx1*K0, s##idx1)

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
    v_float32x4 valpha_lo = v_load(alphabuf), valpha_hi = v_load(alphabuf + 4); \
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
    s##idx0##l = v_select(v_ge(s##idx0##l, zz), s##idx0##l, v_mul(s##idx0##l, valpha_lo)); \
    s##idx0##h = v_select(v_ge(s##idx0##h, zz), s##idx0##h, v_mul(s##idx0##h, valpha_hi)); \
    s##idx1##l = v_select(v_ge(s##idx1##l, zz), s##idx1##l, v_mul(s##idx1##l, valpha_lo)); \
    s##idx1##h = v_select(v_ge(s##idx1##h, zz), s##idx1##h, v_mul(s##idx1##h, valpha_hi)); \
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

// Compute number of spatial chunks for load balancing across threads.
static int computeSpatChunks(int total_blocks, int planeblocks, int min_per_chunk = 16) {
    int nSpatChunks = 1;
    int nthreads = cv::getNumThreads();
    int target_tasks = nthreads * 8;
    if (total_blocks < target_tasks && planeblocks > min_per_chunk) {
        nSpatChunks = (target_tasks + total_blocks - 1) / total_blocks;
        int max_chunks = planeblocks / min_per_chunk;
        nSpatChunks = std::min(nSpatChunks, std::max(1, max_chunks));
    }
    return nSpatChunks;
}

static void setupActivation(const ConvState& cs, int K,
                             FastActivation& fastActivation, const float*& activParams,
                             ActivationFunc& activation, float& maxval, float& defaultAlpha) {
    fastActivation = cs.fastActivation;
    activParams = cs.activParams.data();
    activation = cs.activation;
    maxval = FLT_MAX;
    defaultAlpha = 0.f;
    if (fastActivation == FAST_ACTIV_CLIP) {
        CV_Assert(cs.activParams.size() == 2u);
        maxval = activParams[1];
    } else if (fastActivation == FAST_ACTIV_RELU) {
        CV_Assert(!activParams);
    } else if (fastActivation == FAST_ACTIV_LEAKY_RELU) {
        CV_Assert(cs.activParams.size() == 1u);
        defaultAlpha = activParams[0];
    } else if (fastActivation == FAST_ACTIV_PRELU) {
        CV_Assert(cs.activParams.size() == size_t(K));
    } else {
        // FAST_ACTIV_NONE: activation (if any) is handled via function pointer
        defaultAlpha = 1.f;
    }
}

static void fillCoeffBufs(FastActivation fastActivation, const float* activParams, float defaultAlpha,
                           int k_count, int k_base,
                           const float* scaleptr, const float* biasptr,
                           float* scalebuf, float* biasbuf, float* alphabuf) {
    int kk = 0;
    for (; kk < k_count; kk++) {
        scalebuf[kk] = scaleptr ? scaleptr[k_base + kk] : 1.f;
        biasbuf[kk] = biasptr ? biasptr[k_base + kk] : 0.f;
        alphabuf[kk] = fastActivation == FAST_ACTIV_PRELU ? activParams[k_base + kk] : defaultAlpha;
    }
    for (; kk < 8; kk++) {
        scalebuf[kk] = 0.f;
        biasbuf[kk] = 0.f;
        alphabuf[kk] = 0.f;
    }
}

#ifdef CONV_ENABLE_SIMD
static bool simdTailAdjust(int spat_block_size, int& p, int p0, int p1, int K0,
                            float*& outptr, const float*& resptr) {
    if (p + spat_block_size > p1) {
        if (p == p0) return true;
        int _p_new = p1 - spat_block_size;
        int _dp = _p_new - p;
        outptr += _dp * K0;
        if (resptr) resptr += _dp * K0;
        p = _p_new;
    }
    return false;
}

static void copyResidualBlock(bool aligned_k, int k_base, int k_count, int K0shift,
                               int K0, int planeblocks, int planesize, int spat_block_size,
                               float* tmpbuf, const float*& resptr) {
    if (resptr) {
        if (aligned_k) {
            memcpy(tmpbuf, resptr + k_base*planeblocks, spat_block_size*K0*sizeof(float));
        } else {
            for (int _kk = 0; _kk < k_count; ++_kk) {
                const int _k = k_base + _kk;
                int _kofs = (_k >> K0shift) * planesize + (_k & (K0-1));
                for (int _j = 0; _j < spat_block_size; _j++)
                    tmpbuf[_kk + _j*K0] = resptr[_kofs + _j*K0];
            }
        }
        resptr += spat_block_size*K0;
    }
}

static void scatterOutputBlock(bool aligned_k, int k_base, int k_count, int K0shift,
                                int K0, int planeblocks, int planesize, int spat_block_size,
                                float* outptr, const float* tmpbuf) {
    if (!aligned_k) {
        for (int _kk = 0; _kk < k_count; ++_kk) {
            const int _k = k_base + _kk;
            int _kofs = (_k >> K0shift) * planesize + (_k & (K0-1));
            for (int _j = 0; _j < spat_block_size; _j++)
                outptr[_kofs + _j*K0] = tmpbuf[_kk + _j*K0];
        }
    }
}
#endif // CONV_ENABLE_SIMD

static void loadScalarResidual(const float* resptr, int k_base, int k_count, int K0shift,
                                int K0, int planesize, float* resbuf) {
    if (resptr) {
        for (int _kk = 0; _kk < k_count; ++_kk) {
            const int _k = k_base + _kk;
            int _kofs = (_k >> K0shift) * planesize + (_k & (K0-1));
            resbuf[_kk] = resptr[_kofs];
        }
    }
}

static void callActivationScalar(ActivationFunc activation, float* outbuf, int K0, const float* activParams) {
    if (activation) activation(outbuf, outbuf, K0, activParams);
}

static void scatterScalarOut(bool aligned_k, int k_base, int k_count, int K0shift,
                              int K0, int planesize, float* outptr, const float* outbuf) {
    if (!aligned_k) {
        for (int _kk = 0; _kk < k_count; _kk++) {
            const int _k = k_base + _kk;
            int _kofs = (_k >> K0shift) * planesize + (_k & (K0-1));
            outptr[_kofs] = outbuf[_kk];
        }
    }
}

// Architecture-specific: initialize scalar accumulators for a single output point.
#if CV_SIMD256
#define CONV_INIT_SCALAR_SUMS() \
    v_float32x8 zz = v256_setzero_f32(); \
    v_float32x8 s0 = zz
#elif CV_SIMD128
#define CONV_INIT_SCALAR_SUMS() \
    v_float32x4 zz = v_setzero_f32(); \
    v_float32x4 s0 = zz, s1 = zz
#else
#define CONV_INIT_SCALAR_SUMS() \
    for (int _ks = 0; _ks < K0; _ks++) tmpbuf[_ks] = 0.f
#endif

#if CV_SIMD256
#define CONV_FINALIZE_SCALAR_OUT(outbuf) \
    { \
        v_float32x8 _vsc = v256_load(scalebuf); \
        v_float32x8 _vbi = v256_load(biasbuf); \
        v_float32x8 _val = v256_load(alphabuf); \
        v_float32x8 _vmx = v256_setall_f32(maxval); \
        s0 = v_fma(s0, _vsc, _vbi); \
        s0 = v_add(s0, v256_load(resbuf)); \
        s0 = v_select(v_ge(s0, zz), s0, v_mul(s0, _val)); \
        s0 = v_min(s0, _vmx); \
        v_store((outbuf), s0); \
    }
#elif CV_SIMD128
#define CONV_FINALIZE_SCALAR_OUT(outbuf) \
    { \
        v_float32x4 _vsc_lo = v_load(scalebuf), _vsc_hi = v_load(scalebuf + 4); \
        v_float32x4 _vbi_lo = v_load(biasbuf),  _vbi_hi = v_load(biasbuf + 4); \
        v_float32x4 _val_lo = v_load(alphabuf),  _val_hi = v_load(alphabuf + 4); \
        v_float32x4 _vmx = v_setall_f32(maxval); \
        s0 = v_fma(s0, _vsc_lo, _vbi_lo); \
        s1 = v_fma(s1, _vsc_hi, _vbi_hi); \
        s0 = v_add(s0, v_load(resbuf)); \
        s1 = v_add(s1, v_load(resbuf + 4)); \
        s0 = v_select(v_ge(s0, zz), s0, v_mul(s0, _val_lo)); \
        s1 = v_select(v_ge(s1, zz), s1, v_mul(s1, _val_hi)); \
        s0 = v_min(s0, _vmx); s1 = v_min(s1, _vmx); \
        v_store((outbuf), s0); v_store((outbuf) + 4, s1); \
    }
#else
#define CONV_FINALIZE_SCALAR_OUT(outbuf) \
    for (int _kk = 0; _kk < K0; _kk++) { \
        float _v = tmpbuf[_kk]*scalebuf[_kk] + biasbuf[_kk] + resbuf[_kk]; \
        _v = std::min(_v*(_v >= 0 ? 1.f : alphabuf[_kk]), maxval); \
        (outbuf)[_kk] = _v; \
    }
#endif

// Specialized 1x1 convolution kernel with stride=1.
static void conv32fC8_1x1(const void* inp__, const void* residual__, void* out__,
                           const ConvState& cs, const void* weights__,
                           const float* scale__, const float* bias__)
{
    const MatShape& inpshape = cs.inpshape;
    const MatShape& outshape = cs.outshape;

    CV_Assert_N(inpshape.layout == DATA_LAYOUT_BLOCK, outshape.layout == DATA_LAYOUT_BLOCK);

    int K_ = outshape.channels();
    int ndims_ = outshape.dims;
    int N = outshape[0];
    int D_ = ndims_ >= 6 ? outshape[ndims_ - 4] : 1;
    int H_ = ndims_ >= 5 ? outshape[ndims_ - 3] : 1;
    int W_ = outshape[ndims_-2];
    int planeblocks_ = D_*H_*W_;
    size_t outtotal = outshape.total();

    int Kblk_ = cs.wshape[1];
    int C1Max_ = cs.wshape[3];
    int total_blocks = N * cs.ngroups * Kblk_;

    if ((K_/cs.ngroups) % inpshape.back() != 0) {
        memset(out__, 0, outtotal*sizeof(float));
    }

    int nSpatChunks_ = computeSpatChunks(total_blocks, planeblocks_);
    int total_tasks = total_blocks * nSpatChunks_;

    parallel_for_(Range(0, total_tasks), [&](const Range& range) {
        constexpr int SPAT_BLOCK_SIZE = 10;
        constexpr int C0shift = 3, K0shift = C0shift;
        constexpr int C0 = 1 << C0shift, K0 = C0;

        CV_Assert_N(inpshape.back() == C0, outshape.back() == K0);

        const int C = inpshape.channels(), K = outshape.channels();
        const int C1 = (C + C0 - 1)/C0, K1 = (K + K0 - 1)/K0;
        const int ngroups = cs.ngroups, Kblk = Kblk_, C1Max = C1Max_;
        const int Cg = C / ngroups;
        const int Kg = K / ngroups;
        const int nSpatChunks = nSpatChunks_;

        int planeblocks = planeblocks_;
        int planesize = planeblocks*K0;
        int ndims = ndims_;
        int Di = ndims >= 6 ? inpshape[ndims-4] : 1;
        int Hi = ndims >= 5 ? inpshape[ndims-3] : 1;
        int Wi = inpshape[ndims-2];
        int iplanesize = Di*Hi*Wi*C0;

        const float* scaleptr = (const float*)scale__;
        const float* biasptr = (const float*)bias__;

        FastActivation fastActivation;
        const float* activParams;
        ActivationFunc activation;
        float maxval, defaultAlpha;
        float scalebuf[K0], biasbuf[K0], alphabuf[K0];
        setupActivation(cs, K, fastActivation, activParams, activation, maxval, defaultAlpha);

        for (int t = range.start; t < range.end; t++) {
            const int block_id = t / nSpatChunks;
            const int chunk_id = t % nSpatChunks;
            const int p0 = chunk_id * planeblocks / nSpatChunks;
            const int p1 = (chunk_id + 1) * planeblocks / nSpatChunks;
            const int n = block_id / (ngroups * Kblk);
            const int rem = block_id - n * (ngroups * Kblk);
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
            const float* wbaseptr = (float*)weights__ + (g*Kblk + kblk)*(1*C1Max*C0*K0);

            fillCoeffBufs(fastActivation, activParams, defaultAlpha, k_count, k_base, scaleptr, biasptr, scalebuf, biasbuf, alphabuf);

            float* outptr = (float*)out__ + n*(K1*planesize) + p0*K0;
            const float* resptr = residual__ ? (float*)residual__ + n*(K1*planesize) + p0*K0 : nullptr;
            float tmpbuf[SPAT_BLOCK_SIZE*K0] = {};
            int p = p0;

        #ifdef CONV_ENABLE_SIMD
            for (; p < p1; p += SPAT_BLOCK_SIZE,
                           outptr += SPAT_BLOCK_SIZE*K0)
            {
                if (simdTailAdjust(SPAT_BLOCK_SIZE, p, p0, p1, K0, outptr, resptr)) break;
                copyResidualBlock(aligned_k, k_base, k_count, K0shift, K0, planeblocks, planesize, SPAT_BLOCK_SIZE, tmpbuf, resptr);
                CONV_INIT_SUMS();

                const float* inptr[SPAT_BLOCK_SIZE];
                int inpstep[SPAT_BLOCK_SIZE];
                for (int j = 0; j < SPAT_BLOCK_SIZE; j++) {
                    inptr[j] = inpbaseptr + (size_t)(p + j) * C0;
                    inpstep[j] = iplanesize;
                }
                const float* wptr = wbaseptr;
                for (int c1 = 0; c1 < cblocks; c1++, wptr += C0*K0) {
                    CONV_UPDATE_LOOP_BODY();
                }

                float* outbuf = aligned_k ? outptr + k_base*planeblocks : tmpbuf;
                CONV_FINALIZE_OUT_ALL();
                if (activation) { activation(outbuf, outbuf, SPAT_BLOCK_SIZE*K0, activParams); }
                scatterOutputBlock(aligned_k, k_base, k_count, K0shift, K0, planeblocks, planesize, SPAT_BLOCK_SIZE, outptr, tmpbuf);
            }
        #endif

            float resbuf[K0] = {};

            for (; p < p1; p++, outptr += K0, resptr += (resptr ? K0 : 0))
            {
                CONV_INIT_SCALAR_SUMS();
                loadScalarResidual(resptr, k_base, k_count, K0shift, K0, planesize, resbuf);

                const float* inptr = inpbaseptr + (size_t)p * C0;
                const float* wptr = wbaseptr;

                for (int c1 = 0; c1 < cblocks; ++c1, inptr += iplanesize, wptr += K0*C0) {
                #if CV_SIMD256
                    v_float32x8 w, x;
                    #undef CONV_UPDATE_BLOCK1
                    #define CONV_UPDATE_BLOCK1(ofs) \
                        w = v256_load(wptr + ofs*K0); \
                        x = v256_setall_f32(inptr[ofs]); \
                        s0 = v_fma(x, w, s0)
                    CONV_UPDATE_BLOCK1(0); CONV_UPDATE_BLOCK1(1);
                    CONV_UPDATE_BLOCK1(2); CONV_UPDATE_BLOCK1(3);
                    CONV_UPDATE_BLOCK1(4); CONV_UPDATE_BLOCK1(5);
                    CONV_UPDATE_BLOCK1(6); CONV_UPDATE_BLOCK1(7);
                #elif CV_SIMD128
                    v_float32x4 w0, w1, x;
                    #undef CONV_UPDATE_BLOCK1
                    #define CONV_UPDATE_BLOCK1(ofs) \
                        w0 = v_load(wptr + ofs*K0); w1 = v_load(wptr + ofs*K0 + 4); \
                        x = v_setall_f32(inptr[ofs]); \
                        s0 = v_fma(x, w0, s0); s1 = v_fma(x, w1, s1)
                    CONV_UPDATE_BLOCK1(0); CONV_UPDATE_BLOCK1(1);
                    CONV_UPDATE_BLOCK1(2); CONV_UPDATE_BLOCK1(3);
                    CONV_UPDATE_BLOCK1(4); CONV_UPDATE_BLOCK1(5);
                    CONV_UPDATE_BLOCK1(6); CONV_UPDATE_BLOCK1(7);
                #else
                    for (int c0 = 0; c0 < C0; ++c0) {
                        const float xval = inptr[c0];
                        for (int kk = 0; kk < K0; ++kk)
                            tmpbuf[kk] += xval * wptr[c0*K0 + kk];
                    }
                #endif
                }

                float* outbuf = aligned_k ? outptr + k_base*planeblocks : tmpbuf;
                CONV_FINALIZE_SCALAR_OUT(outbuf);
                callActivationScalar(activation, outbuf, K0, activParams);
                scatterScalarOut(aligned_k, k_base, k_count, K0shift, K0, planesize, outptr, outbuf);
            }
        }
    });
}

// Specialized 2D 3x3 convolution kernel with stride=1, dilation=1.
static void conv32fC8_3x3s1(const void* inp__, const void* residual__, void* out__,
                              const ConvState& cs, const void* weights__,
                              const float* scale__, const float* bias__)
{
    const MatShape& inpshape = cs.inpshape;
    const MatShape& outshape = cs.outshape;

    CV_Assert_N(inpshape.layout == DATA_LAYOUT_BLOCK, outshape.layout == DATA_LAYOUT_BLOCK);

    int K_ = outshape.channels();
    int ndims_ = outshape.dims;
    int N = outshape[0];
    int H_ = ndims_ >= 5 ? outshape[ndims_ - 3] : 1;
    int W_ = outshape[ndims_-2];
    int planeblocks_ = H_*W_;
    size_t outtotal = outshape.total();

    int Kblk_ = cs.wshape[1];
    int C1Max_ = cs.wshape[3];
    int total_blocks = N * cs.ngroups * Kblk_;

    if ((K_/cs.ngroups) % inpshape.back() != 0) {
        memset(out__, 0, outtotal*sizeof(float));
    }

    int nSpatChunks_ = computeSpatChunks(total_blocks, planeblocks_);
    int total_tasks = total_blocks * nSpatChunks_;

    parallel_for_(Range(0, total_tasks), [&](const Range& range) {
        constexpr int SPAT_BLOCK_SIZE = 10;
        constexpr int C0shift = 3, K0shift = C0shift;
        constexpr int C0 = 1 << C0shift, K0 = C0;

        CV_Assert_N(inpshape.back() == C0, outshape.back() == K0);

        const int C = inpshape.channels(), K = outshape.channels();
        const int C1 = (C + C0 - 1)/C0, K1 = (K + K0 - 1)/K0;
        const int ngroups = cs.ngroups, Kblk = Kblk_, C1Max = C1Max_;
        const int Cg = C / ngroups;
        const int Kg = K / ngroups;
        const int nSpatChunks = nSpatChunks_;
        int W = W_;
        int Hi = ndims_ >= 5 ? inpshape[ndims_-3] : 1;
        int Wi = inpshape[ndims_-2];
        const int padY = cs.pads[1], padX = cs.pads[2];
        const float* scaleptr = (const float*)scale__;
        const float* biasptr = (const float*)bias__;
        int planeblocks = planeblocks_;
        int planesize = planeblocks*K0;
        int iplanesize = Hi*Wi*C0;

    #ifdef CONV_ENABLE_SIMD
        constexpr int MAX_CONV_DIMS = ConvState::MAX_CONV_DIMS;
        int innerY0 = cs.inner[1], innerY1 = cs.inner[MAX_CONV_DIMS+1];
        int innerX0 = cs.inner[2], innerX1 = cs.inner[MAX_CONV_DIMS+2];
    #endif

        FastActivation fastActivation;
        const float* activParams;
        ActivationFunc activation;
        float maxval, defaultAlpha;
        float scalebuf[K0], biasbuf[K0], alphabuf[K0];
        setupActivation(cs, K, fastActivation, activParams, activation, maxval, defaultAlpha);

        for (int t = range.start; t < range.end; t++) {
            const int block_id = t / nSpatChunks;
            const int chunk_id = t % nSpatChunks;
            const int p0 = chunk_id * planeblocks / nSpatChunks;
            const int p1 = (chunk_id + 1) * planeblocks / nSpatChunks;
            const int n = block_id / (ngroups * Kblk);
            const int rem = block_id - n * (ngroups * Kblk);
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
            const float* wbaseptr = (float*)weights__ + (g*Kblk + kblk)*(9*C1Max*C0*K0);

            fillCoeffBufs(fastActivation, activParams, defaultAlpha, k_count, k_base, scaleptr, biasptr, scalebuf, biasbuf, alphabuf);

            float* outptr = (float*)out__ + n*(K1*planesize) + p0*K0;
            const float* resptr = residual__ ? (float*)residual__ + n*(K1*planesize) + p0*K0 : nullptr;
            float tmpbuf[SPAT_BLOCK_SIZE*K0] = {};
            int p = p0;

        #ifdef CONV_ENABLE_SIMD
            int inp_ofs[9];
            for (int ky = 0; ky < 3; ky++)
                for (int kx = 0; kx < 3; kx++)
                    inp_ofs[ky*3 + kx] = (ky * Wi + kx) * C0;
            float zbuf[C0] = {};
            for (; p < p1; p += SPAT_BLOCK_SIZE,
                           outptr += SPAT_BLOCK_SIZE*K0)
            {
                if (simdTailAdjust(SPAT_BLOCK_SIZE, p, p0, p1, K0, outptr, resptr)) break;
                copyResidualBlock(aligned_k, k_base, k_count, K0shift, K0, planeblocks, planesize, SPAT_BLOCK_SIZE, tmpbuf, resptr);

                CONV_INIT_SUMS();

                bool all_inner = false;
                int yi_base, xi_base;

                if ((p % W) + SPAT_BLOCK_SIZE <= W) {
                    int yj = p / W;
                    int xj = p - yj * W;
                    yi_base = yj - padY;
                    xi_base = xj - padX;
                    bool y_inner = (yj >= innerY0 && yj < innerY1);
                    all_inner = y_inner && (xj >= innerX0) &&
                                (xj + SPAT_BLOCK_SIZE - 1 < innerX1);
                } else {
                    int yj = p / W;
                    int xj = p - yj * W;
                    yi_base = yj - padY;
                    xi_base = xj - padX;
                }

                if (all_inner) {
                    const float* inp_yx_base0 = inpbaseptr + (yi_base * Wi + xi_base) * C0;

                    const float* inp_pos[SPAT_BLOCK_SIZE];
                    for (int j = 0; j < SPAT_BLOCK_SIZE; j++)
                        inp_pos[j] = inp_yx_base0 + j * C0;

                    for (int kpos = 0; kpos < 9; kpos++) {
                        const float* kwptr = wbaseptr + kpos * C1Max * C0 * K0;
                        const int kofs = inp_ofs[kpos];
                        for (int c1 = 0; c1 < cblocks; c1++) {
                            const int c1_ofs = c1 * iplanesize;
                            const float* inptr[SPAT_BLOCK_SIZE];
                            int inpstep[SPAT_BLOCK_SIZE];
                            for (int j = 0; j < SPAT_BLOCK_SIZE; j++) {
                                inptr[j] = inp_pos[j] + kofs + c1_ofs;
                                inpstep[j] = 0;
                            }
                            const float* wptr = kwptr + c1 * C0 * K0;
                            CONV_UPDATE_LOOP_BODY();
                        }
                    }
                } else {
                    int yi_arr[SPAT_BLOCK_SIZE], xi_arr[SPAT_BLOCK_SIZE];
                    bool inner_arr[SPAT_BLOCK_SIZE];
                    bool same_row = ((p % W) + SPAT_BLOCK_SIZE <= W);

                    if (same_row) {
                        int yj = p / W;
                        int xj = p - yj * W;
                        bool y_inner = (yj >= innerY0 && yj < innerY1);
                        for (int j = 0; j < SPAT_BLOCK_SIZE; j++) {
                            yi_arr[j] = yj - padY;
                            xi_arr[j] = xj + j - padX;
                            inner_arr[j] = y_inner && ((xj + j) >= innerX0 && (xj + j) < innerX1);
                        }
                    } else {
                        for (int j = 0; j < SPAT_BLOCK_SIZE; j++) {
                            int pj = p + j;
                            int yj = pj / W;
                            int xj = pj - yj * W;
                            yi_arr[j] = yj - padY;
                            xi_arr[j] = xj - padX;
                            inner_arr[j] = (yj >= innerY0 && yj < innerY1) &&
                                        (xj >= innerX0 && xj < innerX1);
                        }
                    }
                    int inp_spatial_ofs[9][SPAT_BLOCK_SIZE];
                    for (int kpos = 0; kpos < 9; kpos++) {
                        int ky = kpos / 3, kx = kpos % 3;
                        for (int j = 0; j < SPAT_BLOCK_SIZE; j++) {
                            int yij = yi_arr[j] + ky;
                            int xij = xi_arr[j] + kx;
                            if (inner_arr[j] ||
                                (((unsigned)yij < (unsigned)Hi) &
                                ((unsigned)xij < (unsigned)Wi))) {
                                inp_spatial_ofs[kpos][j] = (yij * Wi + xij) * C0;
                            } else {
                                inp_spatial_ofs[kpos][j] = -1;
                            }
                        }
                    }

                    for (int kpos = 0; kpos < 9; kpos++) {
                        const float* kwptr = wbaseptr + kpos * C1Max * C0 * K0;
                        for (int c1 = 0; c1 < cblocks; c1++) {
                            const float* inpbase_c1 = inpbaseptr + c1 * iplanesize;
                            const float* inptr[SPAT_BLOCK_SIZE];
                            int inpstep[SPAT_BLOCK_SIZE];
                            for (int j = 0; j < SPAT_BLOCK_SIZE; j++) {
                                int ofs = inp_spatial_ofs[kpos][j];
                                inptr[j] = (ofs >= 0) ? inpbase_c1 + ofs : zbuf;
                                inpstep[j] = 0;
                            }
                            const float* wptr = kwptr + c1 * C0 * K0;
                            CONV_UPDATE_LOOP_BODY();
                        }
                    }
                }

                float* outbuf = aligned_k ? outptr + k_base*planeblocks : tmpbuf;
                CONV_FINALIZE_OUT_ALL();
                if (activation) { activation(outbuf, outbuf, SPAT_BLOCK_SIZE*K0, activParams); }
                scatterOutputBlock(aligned_k, k_base, k_count, K0shift, K0, planeblocks, planesize, SPAT_BLOCK_SIZE, outptr, tmpbuf);
            }
        #endif // CONV_ENABLE_SIMD

            float resbuf[K0] = {};

            for (; p < p1; p++, outptr += K0, resptr += (resptr ? K0 : 0))
            {
                int yj = p / W;
                int xj = p - yj * W;
                int yi_s = yj - padY;
                int xi_s = xj - padX;

                CONV_INIT_SCALAR_SUMS();
                loadScalarResidual(resptr, k_base, k_count, K0shift, K0, planesize, resbuf);

                for (int ky = 0; ky < 3; ky++) {
                    int yi = yi_s + ky;
                    if ((unsigned)yi >= (unsigned)Hi) continue;
                    for (int kx = 0; kx < 3; kx++) {
                        int xi = xi_s + kx;
                        if ((unsigned)xi >= (unsigned)Wi) continue;
                        int kpos = ky*3 + kx;
                        const float* kwptr = wbaseptr + kpos * C1Max * C0 * K0;
                        const float* inptr_s = inpbaseptr;
                        for (int c1 = 0; c1 < cblocks; ++c1, inptr_s += iplanesize) {
                            const float* inptr = inptr_s + (yi*Wi + xi)*C0;
                            const float* wptr = kwptr + c1 * C0 * K0;
                        #if CV_SIMD256
                            v_float32x8 w, xv;
                            #undef CONV_UPDATE_BLOCK1
                            #define CONV_UPDATE_BLOCK1(ofs) \
                                w = v256_load(wptr + ofs*K0); \
                                xv = v256_setall_f32(inptr[ofs]); \
                                s0 = v_fma(xv, w, s0)
                            CONV_UPDATE_BLOCK1(0); CONV_UPDATE_BLOCK1(1);
                            CONV_UPDATE_BLOCK1(2); CONV_UPDATE_BLOCK1(3);
                            CONV_UPDATE_BLOCK1(4); CONV_UPDATE_BLOCK1(5);
                            CONV_UPDATE_BLOCK1(6); CONV_UPDATE_BLOCK1(7);
                        #elif CV_SIMD128
                            v_float32x4 w0, w1, xv;
                            #undef CONV_UPDATE_BLOCK1
                            #define CONV_UPDATE_BLOCK1(ofs) \
                                w0 = v_load(wptr + ofs*K0); w1 = v_load(wptr + ofs*K0 + 4); \
                                xv = v_setall_f32(inptr[ofs]); \
                                s0 = v_fma(xv, w0, s0); s1 = v_fma(xv, w1, s1)
                            CONV_UPDATE_BLOCK1(0); CONV_UPDATE_BLOCK1(1);
                            CONV_UPDATE_BLOCK1(2); CONV_UPDATE_BLOCK1(3);
                            CONV_UPDATE_BLOCK1(4); CONV_UPDATE_BLOCK1(5);
                            CONV_UPDATE_BLOCK1(6); CONV_UPDATE_BLOCK1(7);
                        #else
                            for (int c0 = 0; c0 < C0; ++c0) {
                                const float xval = inptr[c0];
                                for (int kk = 0; kk < K0; ++kk)
                                    tmpbuf[kk] += xval * wptr[c0*K0 + kk];
                            }
                        #endif
                        }
                    }
                }

                float* outbuf = aligned_k ? outptr + k_base*planeblocks : tmpbuf;
                CONV_FINALIZE_SCALAR_OUT(outbuf);
                callActivationScalar(activation, outbuf, K0, activParams);
                scatterScalarOut(aligned_k, k_base, k_count, K0shift, K0, planesize, outptr, outbuf);
            }
        }
    });
}

static void conv32fC8(const void* inp__, const void* residual__, void* out__,
                      const ConvState& cs, const void* weights__,
                      const float* scale__, const float* bias__)
{
    int ksize = cs.wshape[2];
    if (ksize == 1 && cs.strides[0]*cs.strides[1]*cs.strides[2] == 1) {
        return conv32fC8_1x1(inp__, residual__, out__, cs, weights__, scale__, bias__);
    }
    if (ksize == 9 && cs.strides[1] == 1 && cs.strides[2] == 1 &&
        cs.dilations[1] == 1 && cs.dilations[2] == 1 &&
        cs.outshape.dims <= 5) {
        return conv32fC8_3x3s1(inp__, residual__, out__, cs, weights__, scale__, bias__);
    }

    const MatShape& inpshape = cs.inpshape;
    const MatShape& outshape = cs.outshape;

    CV_Assert_N(inpshape.layout == DATA_LAYOUT_BLOCK, outshape.layout == DATA_LAYOUT_BLOCK);

    int K_ = outshape.channels();
    int ndims_ = outshape.dims;
    int N = outshape[0];
    int D_ = ndims_ >= 6 ? outshape[ndims_ - 4] : 1;
    int H_ = ndims_ >= 5 ? outshape[ndims_ - 3] : 1;
    int W_ = outshape[ndims_-2];
    int planeblocks_ = D_*H_*W_;
    size_t outtotal = outshape.total();

    int Kblk_ = cs.wshape[1];
    int C1Max_ = cs.wshape[3];
    int total_blocks = N * cs.ngroups * Kblk_;

    if ((K_/cs.ngroups) % inpshape.back() != 0) {
        memset(out__, 0, outtotal*sizeof(float));
    }

    int nSpatChunksGen_ = computeSpatChunks(total_blocks, planeblocks_);
    int total_tasks_gen = total_blocks * nSpatChunksGen_;

    parallel_for_(Range(0, total_tasks_gen), [&](const Range& range) {
        constexpr int SPAT_BLOCK_SIZE = 10;
        constexpr int C0shift = 3, K0shift = C0shift;
        constexpr int C0 = 1 << C0shift, K0 = C0;

        CV_Assert_N(inpshape.back() == C0, outshape.back() == K0);

        const int C = inpshape.channels(), K = outshape.channels();
        const int C1 = (C + C0 - 1)/C0, K1 = (K + K0 - 1)/K0;
        const int ngroups = cs.ngroups, Kblk = Kblk_, C1Max = C1Max_;
        const int Cg = C / ngroups;
        const int Kg = K / ngroups;
        const int nSpatChunks = nSpatChunksGen_;
        int ksize = cs.wshape[2];
        int ndims = ndims_;
        int D = D_, H = H_, W = W_;
        int Di = ndims >= 6 ? inpshape[ndims-4] : 1;
        int Hi = ndims >= 5 ? inpshape[ndims-3] : 1;
        int Wi = inpshape[ndims-2];
        const int Sz = cs.strides[0], Sy = cs.strides[1], Sx = cs.strides[2];
        const int padZ = cs.pads[0], padY = cs.pads[1], padX = cs.pads[2];
        const float* scaleptr = (const float*)scale__;
        const float* biasptr = (const float*)bias__;
        const int* ofsZYX = cs.coordtab.data();
        int planeblocks = planeblocks_;
        int planesize = planeblocks*K0;
        int iplanesize = Di*Hi*Wi*C0;

    #ifdef CONV_ENABLE_SIMD
        constexpr int MAX_CONV_DIMS = ConvState::MAX_CONV_DIMS;
        int innerZ0 = cs.inner[0], innerZ1 = cs.inner[MAX_CONV_DIMS];
        int innerY0 = cs.inner[1], innerY1 = cs.inner[MAX_CONV_DIMS+1];
        int innerX0 = cs.inner[2], innerX1 = cs.inner[MAX_CONV_DIMS+2];
        float zbuf[C0] = {};
    #endif

        FastActivation fastActivation;
        const float* activParams;
        ActivationFunc activation;
        float maxval, defaultAlpha;
        float scalebuf[K0], biasbuf[K0], alphabuf[K0];
        setupActivation(cs, K, fastActivation, activParams, activation, maxval, defaultAlpha);

        // 1x1x1 convolution with (1,1,1) strides:
        bool is_1x1s1 = (ksize == 1 && Sz*Sy*Sx == 1);
        if (is_1x1s1) {
            W *= D*H;
            Wi *= Di*Hi;
            D = Di = H = Hi = 1;
        #ifdef CONV_ENABLE_SIMD
            innerZ1 = innerY1 = 1;
            innerX1 = W;
        #endif
        }

        for (int t = range.start; t < range.end; t++) {
            const int block_id = t / nSpatChunks;
            const int chunk_id = t % nSpatChunks;
            const int p0 = chunk_id * planeblocks / nSpatChunks;
            const int p1 = (chunk_id + 1) * planeblocks / nSpatChunks;
            const int n = block_id / (ngroups * Kblk);
            const int rem = block_id - n * (ngroups * Kblk);
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

            fillCoeffBufs(fastActivation, activParams, defaultAlpha, k_count, k_base, scaleptr, biasptr, scalebuf, biasbuf, alphabuf);

            float* outptr = (float*)out__ + n*(K1*planesize) + p0*K0;
            const float* resptr = residual__ ? (float*)residual__ + n*(K1*planesize) + p0*K0 : nullptr;
            float tmpbuf[SPAT_BLOCK_SIZE*K0] = {};
            int p = p0;

        #ifdef CONV_ENABLE_SIMD
            if (is_1x1s1) {
            for (; p < p1; p += SPAT_BLOCK_SIZE,
                           outptr += SPAT_BLOCK_SIZE*K0)
            {
                if (simdTailAdjust(SPAT_BLOCK_SIZE, p, p0, p1, K0, outptr, resptr)) break;
                copyResidualBlock(aligned_k, k_base, k_count, K0shift, K0, planeblocks, planesize, SPAT_BLOCK_SIZE, tmpbuf, resptr);
                CONV_INIT_SUMS();

                const float* inptr[SPAT_BLOCK_SIZE];
                int inpstep[SPAT_BLOCK_SIZE];
                for (int j = 0; j < SPAT_BLOCK_SIZE; j++) {
                    inptr[j] = inpbaseptr + (size_t)(p + j) * C0;
                    inpstep[j] = iplanesize;
                }
                const float* wptr = wbaseptr;
                for (int c1 = 0; c1 < cblocks; c1++, wptr += C0*K0) {
                    CONV_UPDATE_LOOP_BODY();
                }

                float* outbuf = aligned_k ? outptr + k_base*planeblocks : tmpbuf;
                CONV_FINALIZE_OUT_ALL();
                if (activation) { activation(outbuf, outbuf, SPAT_BLOCK_SIZE*K0, activParams); }
                scatterOutputBlock(aligned_k, k_base, k_count, K0shift, K0, planeblocks, planesize, SPAT_BLOCK_SIZE, outptr, tmpbuf);
            }
            } else {
            cv::AutoBuffer<int, 64> kofs_tab(ksize);
            for (int i = 0; i < ksize; i++) {
                kofs_tab[i] = ((ofsZYX[i*3] * Hi + ofsZYX[i*3+1]) * Wi + ofsZYX[i*3+2]) * C0;
            }
            int x_step = Sx * C0;

            // Precompute initial (z, y, x) from p to avoid repeated division
            int cur_z, cur_y, cur_x;
            if (D == 1) {
                cur_z = 0; cur_y = p / W; cur_x = p - cur_y * W;
            } else {
                cur_z = p / (H * W);
                int yxj = p - cur_z * (H * W);
                cur_y = yxj / W; cur_x = yxj - cur_y * W;
            }

            for (; p < p1; p += SPAT_BLOCK_SIZE,
                           outptr += SPAT_BLOCK_SIZE*K0)
            {
                if (p + SPAT_BLOCK_SIZE > p1) {
                    if (p == p0)
                        break;
                    int p_new = p1 - SPAT_BLOCK_SIZE;
                    int dp = p_new - p;
                    outptr += dp*K0;
                    resptr += (resptr ? dp*K0 : 0);
                    p = p_new;
                    // Recompute coordinates for the adjusted position
                    if (D == 1) {
                        cur_z = 0; cur_y = p / W; cur_x = p - cur_y * W;
                    } else {
                        cur_z = p / (H * W);
                        int yxj = p - cur_z * (H * W);
                        cur_y = yxj / W; cur_x = yxj - cur_y * W;
                    }
                }

                copyResidualBlock(aligned_k, k_base, k_count, K0shift, K0, planeblocks, planesize, SPAT_BLOCK_SIZE, tmpbuf, resptr);
                CONV_INIT_SUMS();

                bool same_row = (cur_x + SPAT_BLOCK_SIZE <= W);
                bool all_inner = same_row &&
                    (cur_z >= innerZ0 && cur_z < innerZ1) &&
                    (cur_y >= innerY0 && cur_y < innerY1) &&
                    (cur_x >= innerX0) && (cur_x + SPAT_BLOCK_SIZE - 1 < innerX1);

                if (all_inner) {
                    int zi_base = cur_z*Sz - padZ;
                    int yi_base = cur_y*Sy - padY;
                    int xi_base = cur_x*Sx - padX;
                    int base_ofs = (zi_base * Hi + yi_base) * Wi * C0 + xi_base * C0;

                    // ksize-outer, c1-inner: matches weight layout [Kblk][ksize][C1Max][C0*K0]
                    for (int i = 0; i < ksize; i++) {
                        const float* kwptr = wbaseptr + i * C1Max * C0 * K0;
                        for (int c1 = 0; c1 < cblocks; c1++) {
                            const float* inptr[SPAT_BLOCK_SIZE];
                            int inpstep[SPAT_BLOCK_SIZE];
                            const float* inp_ki = inpbaseptr + c1 * iplanesize + base_ofs + kofs_tab[i];
                            for (int j = 0; j < SPAT_BLOCK_SIZE; j++) {
                                inptr[j] = inp_ki + j * x_step;
                                inpstep[j] = 0;
                            }
                            const float* wptr = kwptr + c1 * C0 * K0;
                            CONV_UPDATE_LOOP_BODY();
                        }
                    }
                } else {
                    Vec3i pt[SPAT_BLOCK_SIZE];
                    bool inner[SPAT_BLOCK_SIZE];

                    if (same_row) {
                        const bool zy_inner = (cur_z >= innerZ0 && cur_z < innerZ1) && (cur_y >= innerY0 && cur_y < innerY1);
                        for (int j = 0; j < SPAT_BLOCK_SIZE; j++) {
                            pt[j] = Vec3i(cur_z*Sz - padZ, cur_y*Sy - padY, (cur_x+j)*Sx - padX);
                            inner[j] = zy_inner && ((cur_x+j) >= innerX0 && (cur_x+j) < innerX1);
                        }
                    } else {
                        for (int j = 0; j < SPAT_BLOCK_SIZE; j++) {
                            int pj = p + j;
                            int zj, yj, xj;
                            if (D == 1) {
                                zj = 0; yj = pj / W; xj = pj - yj * W;
                            } else {
                                zj = pj / (H*W);
                                int yxj = pj - zj*(H*W);
                                yj = yxj / W; xj = yxj - yj*W;
                            }
                            pt[j] = Vec3i(zj*Sz - padZ, yj*Sy - padY, xj*Sx - padX);
                            inner[j] = (zj >= innerZ0 && zj < innerZ1) &&
                                       (yj >= innerY0 && yj < innerY1) &&
                                       (xj >= innerX0 && xj < innerX1);
                        }
                    }

                    // ksize-outer, c1-inner: matches weight layout [Kblk][ksize][C1Max][C0*K0]
                    for (int i = 0; i < ksize; i++) {
                        const float* kwptr = wbaseptr + i * C1Max * C0 * K0;
                        for (int c1 = 0; c1 < cblocks; c1++) {
                            const float* inpbase_c1 = inpbaseptr + c1 * iplanesize;
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
                                    inptr[j] = inpbase_c1 + (((zij * Hi) + yij) * Wi + xij) * C0;
                                    inpstep[j] = 0;
                                } else {
                                    inptr[j] = zbuf;
                                    inpstep[j] = 0;
                                }
                            }
                            const float* wptr = kwptr + c1 * C0 * K0;
                            CONV_UPDATE_LOOP_BODY();
                        }
                    }
                }

                float* outbuf = aligned_k ? outptr + k_base*planeblocks : tmpbuf;
                CONV_FINALIZE_OUT_ALL();
                if (activation) { activation(outbuf, outbuf, SPAT_BLOCK_SIZE*K0, activParams); }
                scatterOutputBlock(aligned_k, k_base, k_count, K0shift, K0, planeblocks, planesize, SPAT_BLOCK_SIZE, outptr, tmpbuf);

                cur_x += SPAT_BLOCK_SIZE;
                if (cur_x >= W) {
                    int pn = p + SPAT_BLOCK_SIZE;
                    if (D == 1) {
                        cur_z = 0; cur_y = pn / W; cur_x = pn - cur_y * W;
                    } else {
                        cur_z = pn / (H*W);
                        int yxj = pn - cur_z*(H*W);
                        cur_y = yxj / W; cur_x = yxj - cur_y * W;
                    }
                }
            }
            }
        #endif

            float resbuf[K0] = {};

            for (; p < p1; p++, outptr += K0, resptr += (resptr ? K0 : 0))
            {
                int zj = p / (H*W);
                int yxj = p - zj*(H*W);
                int yj = yxj / W;
                int xj = yxj - yj*W;
                int zi_base = zj*Sz - padZ;
                int yi_base = yj*Sy - padY;
                int xi_base = xj*Sx - padX;

                CONV_INIT_SCALAR_SUMS();
                loadScalarResidual(resptr, k_base, k_count, K0shift, K0, planesize, resbuf);

                // ksize-outer, c1-inner (matching weight layout [Kblk][ksize][C1Max][C0*K0])
                for (int i = 0; i < ksize; i++) {
                    int zi = zi_base + ofsZYX[i*3 + 0];
                    int yi = yi_base + ofsZYX[i*3 + 1];
                    int xi = xi_base + ofsZYX[i*3 + 2];

                    if ((((unsigned)zi >= (unsigned)Di) |
                         ((unsigned)yi >= (unsigned)Hi) |
                         ((unsigned)xi >= (unsigned)Wi)) != 0)
                        continue;

                    const float* kwptr = wbaseptr + i * C1Max * K0 * C0;
                    for (int c1 = 0; c1 < cblocks; ++c1) {
                        const float* inptr = inpbaseptr + c1 * iplanesize + (((zi * Hi) + yi) * Wi + xi) * C0;
                        const float* wptr = kwptr + c1 * K0 * C0;
                    #if CV_SIMD256
                        v_float32x8 w, x;
                        #undef CONV_UPDATE_BLOCK1
                        #define CONV_UPDATE_BLOCK1(ofs) \
                            w = v256_load(wptr + ofs*K0); \
                            x = v256_setall_f32(inptr[ofs]); \
                            s0 = v_fma(x, w, s0)
                        CONV_UPDATE_BLOCK1(0);
                        CONV_UPDATE_BLOCK1(1);
                        CONV_UPDATE_BLOCK1(2);
                        CONV_UPDATE_BLOCK1(3);
                        CONV_UPDATE_BLOCK1(4);
                        CONV_UPDATE_BLOCK1(5);
                        CONV_UPDATE_BLOCK1(6);
                        CONV_UPDATE_BLOCK1(7);
                    #elif CV_SIMD128
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
                CONV_FINALIZE_SCALAR_OUT(outbuf);
                callActivationScalar(activation, outbuf, K0, activParams);
                scatterScalarOut(aligned_k, k_base, k_count, K0shift, K0, planesize, outptr, outbuf);
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
