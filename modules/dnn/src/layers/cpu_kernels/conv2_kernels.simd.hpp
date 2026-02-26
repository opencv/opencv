// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../conv2_common.hpp"
#include "opencv2/core/hal/intrin.hpp"

// === dispatched calls (implemented here)

namespace cv {
namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

cv::dnn::ConvFunc getConvFunc_(int depth, int C0, ConvKind convkind);

CV_CPU_OPTIMIZATION_NAMESPACE_END
}} // cv::dnn::

// === implementation

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

namespace cv {
namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

static void conv32fC8(const void* inp__, const void* residual__, void* out__,
                      const ConvState& cs, const void* weights__,
                      const float* scale__, const float* bias__,
                      const int32_t* inpofs__, const int32_t* ofsofs__)
{
    int nlanes_ = VTraits<v_float32>::vlanes();
    int C0_ = cs.inpshape.back();
    int NK1 = cs.outshape[0]*cs.outshape[1];

    CV_Assert(C0_ == 8);
    CV_Assert(C0_ == nlanes_ || C0_ == nlanes_*2 || C0_ % (nlanes_*4) == 0);
    CV_Assert(cs.activation == nullptr || cs.fastActivation == FAST_ACTIV_NONE);
    CV_Assert(0 <= cs.nspatialdims && cs.nspatialdims <= ConvState::MAX_CONV_DIMS);

    parallel_for_(Range(0, NK1), [&](const Range& range) {
        const float* scale_ = scale__;
        const float* bias_ = bias__;
        const int32_t* inpofs_ = inpofs__;
        const int32_t* ofsofs_ = ofsofs__;
        constexpr int BLOCK_SIZE = 8;
        int nk0 = range.start, nk1 = range.end;
        constexpr int C0 = 8, K0 = C0;
        int planesize = 1, iplanesize = 1, ksize = 1;
        int nspatialdims = cs.nspatialdims;
        for (int i = 0; i < nspatialdims; i++) {
            planesize *= cs.outshape[i + 2];
            iplanesize *= cs.inpshape[i + 2];
            ksize *= cs.kshape[ConvState::MAX_CONV_DIMS - nspatialdims + i];
        }
        int C1 = cs.inpshape[1], K1 = cs.outshape[1];
        int ngroups = cs.ngroups, K1g = K1/ngroups, C1g = C1/ngroups;
        int nC = C1g*ksize*C0*K0;
        AutoBuffer<float> sumbuf(BLOCK_SIZE*K0*3);
        float* sum = sumbuf.data();
        float* scale = sum + BLOCK_SIZE*K0;
        float* bias = sum + BLOCK_SIZE*K0*2;
        const float* inptrs[BLOCK_SIZE];
        const int32_t* ofsptrs[BLOCK_SIZE];
        FastActivation fastActivation = cs.fastActivation;
        const float* activParams = cs.activParams;
        activation_func_t activation = cs.activation;
        float maxval = fastActivation == FAST_ACTIV_CLIP ? activParams[1] : FLT_MAX;
        float alpha = fastActivation == FAST_ACTIV_LEAKY_RELU ? activParams[0] :
                    fastActivation == FAST_ACTIV_NONE ? 1.f : 0.f;
        int nlanes = nlanes_;

        for (int j = 0; j < BLOCK_SIZE*K0; j++) {
            scale[j] = 1.f;
            bias[j] = 0.f;
        }

        for (int nk = nk0; nk < nk1; nk++) {
            int n = nk/K1, k1 = nk - n*K1;
            int g = k1/K1g;
            float* out = (float*)out__ + nk*planesize*K0;
            const float* inp0 = (const float*)inp__ + (n*C1 + g*C1g)*iplanesize*C0;
            const float* resptr = residual__ ? (const float*)residual__ + nk*planesize*K0 : nullptr;
            const float* wptr = (const float*)weights__ + k1*nC;

            if (scale_) {
                for (int b = 0; b < BLOCK_SIZE; b++)
                    for (int k = 0; k < K0; k++)
                        scale[b*K0 + k] = scale_[k1*K0 + k];
            }

            if (bias_) {
                for (int b = 0; b < BLOCK_SIZE; b++)
                    for (int k = 0; k < K0; k++)
                        bias[b*K0 + k] = bias_[k1*K0 + k];
            }

            for (int xy0 = 0; xy0 < planesize; xy0 += BLOCK_SIZE, out += K0*BLOCK_SIZE,
                 resptr += (resptr ? K0*BLOCK_SIZE : 0)) {
                int j = 0, blocksize = std::min(planesize - xy0, BLOCK_SIZE);

                for (; j < blocksize; j++) {
                    int jj = (xy0 + j)*2;
                    inptrs[j] = inp0 + ofsofs_[jj];
                    ofsptrs[j] = inpofs_ + ofsofs_[jj+1];
                }

                if (j < BLOCK_SIZE) {
                    const float* last_inptr = inptrs[blocksize-1];
                    const int32_t* last_ofsptr = ofsptrs[blocksize-1];
                    for (; j < BLOCK_SIZE; j++) {
                        inptrs[j] = last_inptr;
                        ofsptrs[j] = last_ofsptr;
                    }
                }

#ifdef __ARM_NEON
                float32x4_t z = vdupq_n_f32(0.f);
                float32x4_t s00 = z, s01 = z, s10 = z, s11 = z;
                float32x4_t s20 = z, s21 = z, s30 = z, s31 = z;
                float32x4_t s40 = z, s41 = z, s50 = z, s51 = z;
                float32x4_t s60 = z, s61 = z, s70 = z, s71 = z;
                const int32_t *ofs0 = ofsptrs[0], *ofs1 = ofsptrs[1];
                const int32_t *ofs2 = ofsptrs[2], *ofs3 = ofsptrs[3];
                const int32_t *ofs4 = ofsptrs[4], *ofs5 = ofsptrs[5];
                const int32_t *ofs6 = ofsptrs[6], *ofs7 = ofsptrs[7];

                const float *inptr0 = inptrs[0], *inptr1 = inptrs[1];
                const float *inptr2 = inptrs[2], *inptr3 = inptrs[3];
                const float *inptr4 = inptrs[4], *inptr5 = inptrs[5];
                const float *inptr6 = inptrs[6], *inptr7 = inptrs[7];

                for (int c1 = 0, i = 0; c1 < nC; c1 += K0*C0, i++) {
                    float32x4_t w00 = vld1q_f32(wptr + c1 + 8*0), w01 = vld1q_f32(wptr + c1 + 8*0 + 4);
                    float32x4_t w10 = vld1q_f32(wptr + c1 + 8*1), w11 = vld1q_f32(wptr + c1 + 8*1 + 4);
                    float32x4_t w20 = vld1q_f32(wptr + c1 + 8*2), w21 = vld1q_f32(wptr + c1 + 8*2 + 4);
                    float32x4_t w30 = vld1q_f32(wptr + c1 + 8*3), w31 = vld1q_f32(wptr + c1 + 8*3 + 4);
                    float32x4_t w40 = vld1q_f32(wptr + c1 + 8*4), w41 = vld1q_f32(wptr + c1 + 8*4 + 4);
                    float32x4_t w50 = vld1q_f32(wptr + c1 + 8*5), w51 = vld1q_f32(wptr + c1 + 8*5 + 4);
                    float32x4_t w60 = vld1q_f32(wptr + c1 + 8*6), w61 = vld1q_f32(wptr + c1 + 8*6 + 4);
                    float32x4_t w70 = vld1q_f32(wptr + c1 + 8*7), w71 = vld1q_f32(wptr + c1 + 8*7 + 4);

                    int32_t ofs0i, ofs1i, ofs0p, ofs1p;
                    float m0, m1;
                    float32x4_t x0, x1;

#undef UPDATE_SUM
#define UPDATE_SUM(a, b) \
    ofs0i = ofs##a[i]; ofs1i = ofs##b[i]; \
    m0 = float(ofs0i >= 0); m1 = float(ofs1i >= 0); \
    ofs0p = std::max(ofs0i, 0); ofs1p = std::max(ofs1i, 0); \
    x0 = vld1q_f32(inptr##a + ofs0p); \
    x1 = vld1q_f32(inptr##b + ofs1p); \
    x0 = vmulq_n_f32(x0, m0); \
    x1 = vmulq_n_f32(x1, m1); \
    s##a##0 = vfmaq_laneq_f32(s##a##0, w00, x0, 0); \
    s##a##1 = vfmaq_laneq_f32(s##a##1, w01, x0, 0); \
    s##b##0 = vfmaq_laneq_f32(s##b##0, w00, x1, 0); \
    s##b##1 = vfmaq_laneq_f32(s##b##1, w01, x1, 0); \
    s##a##0 = vfmaq_laneq_f32(s##a##0, w10, x0, 1); \
    s##a##1 = vfmaq_laneq_f32(s##a##1, w11, x0, 1); \
    s##b##0 = vfmaq_laneq_f32(s##b##0, w10, x1, 1); \
    s##b##1 = vfmaq_laneq_f32(s##b##1, w11, x1, 1); \
    s##a##0 = vfmaq_laneq_f32(s##a##0, w20, x0, 2); \
    s##a##1 = vfmaq_laneq_f32(s##a##1, w21, x0, 2); \
    s##b##0 = vfmaq_laneq_f32(s##b##0, w20, x1, 2); \
    s##b##1 = vfmaq_laneq_f32(s##b##1, w21, x1, 2); \
    s##a##0 = vfmaq_laneq_f32(s##a##0, w30, x0, 3); \
    s##a##1 = vfmaq_laneq_f32(s##a##1, w31, x0, 3); \
    s##b##0 = vfmaq_laneq_f32(s##b##0, w30, x1, 3); \
    s##b##1 = vfmaq_laneq_f32(s##b##1, w31, x1, 3); \
    x0 = vld1q_f32(inptr##a + ofs0p + 4); \
    x1 = vld1q_f32(inptr##b + ofs1p + 4); \
    x0 = vmulq_n_f32(x0, m0); \
    x1 = vmulq_n_f32(x1, m1); \
    s##a##0 = vfmaq_laneq_f32(s##a##0, w40, x0, 0); \
    s##a##1 = vfmaq_laneq_f32(s##a##1, w41, x0, 0); \
    s##b##0 = vfmaq_laneq_f32(s##b##0, w40, x1, 0); \
    s##b##1 = vfmaq_laneq_f32(s##b##1, w41, x1, 0); \
    s##a##0 = vfmaq_laneq_f32(s##a##0, w50, x0, 1); \
    s##a##1 = vfmaq_laneq_f32(s##a##1, w51, x0, 1); \
    s##b##0 = vfmaq_laneq_f32(s##b##0, w50, x1, 1); \
    s##b##1 = vfmaq_laneq_f32(s##b##1, w51, x1, 1); \
    s##a##0 = vfmaq_laneq_f32(s##a##0, w60, x0, 2); \
    s##a##1 = vfmaq_laneq_f32(s##a##1, w61, x0, 2); \
    s##b##0 = vfmaq_laneq_f32(s##b##0, w60, x1, 2); \
    s##b##1 = vfmaq_laneq_f32(s##b##1, w61, x1, 2); \
    s##a##0 = vfmaq_laneq_f32(s##a##0, w70, x0, 3); \
    s##a##1 = vfmaq_laneq_f32(s##a##1, w71, x0, 3); \
    s##b##0 = vfmaq_laneq_f32(s##b##0, w70, x1, 3); \
    s##b##1 = vfmaq_laneq_f32(s##b##1, w71, x1, 3)

                    UPDATE_SUM(0, 1);
                    UPDATE_SUM(2, 3);
                    UPDATE_SUM(4, 5);
                    UPDATE_SUM(6, 7);
                }

                vst1q_f32(sum + 8*0, s00); vst1q_f32(sum + 8*0 + 4, s01);
                vst1q_f32(sum + 8*1, s10); vst1q_f32(sum + 8*1 + 4, s11);
                vst1q_f32(sum + 8*2, s20); vst1q_f32(sum + 8*2 + 4, s21);
                vst1q_f32(sum + 8*3, s30); vst1q_f32(sum + 8*3 + 4, s31);
                vst1q_f32(sum + 8*4, s40); vst1q_f32(sum + 8*4 + 4, s41);
                vst1q_f32(sum + 8*5, s50); vst1q_f32(sum + 8*5 + 4, s51);
                vst1q_f32(sum + 8*6, s60); vst1q_f32(sum + 8*6 + 4, s61);
                vst1q_f32(sum + 8*7, s70); vst1q_f32(sum + 8*7 + 4, s71);
#elif CV_SIMD128
                v_float32 z = vx_setzero_f32();
                v_float32 s00 = z, s01 = z, s10 = z, s11 = z;
                v_float32 s20 = z, s21 = z, s30 = z, s31 = z;
                v_float32 s40 = z, s41 = z, s50 = z, s51 = z;
                v_float32 s60 = z, s61 = z, s70 = z, s71 = z;
                const int32_t *ofs0 = ofsptrs[0], *ofs1 = ofsptrs[1];
                const int32_t *ofs2 = ofsptrs[2], *ofs3 = ofsptrs[3];
                const int32_t *ofs4 = ofsptrs[4], *ofs5 = ofsptrs[5];
                const int32_t *ofs6 = ofsptrs[6], *ofs7 = ofsptrs[7];

                const float *inptr0 = inptrs[0], *inptr1 = inptrs[1];
                const float *inptr2 = inptrs[2], *inptr3 = inptrs[3];
                const float *inptr4 = inptrs[4], *inptr5 = inptrs[5];
                const float *inptr6 = inptrs[6], *inptr7 = inptrs[7];

                for (int c1 = 0, i = 0; c1 < nC; c1 += K0*C0, i++) {
                    v_float32 w00 = vx_load(wptr + c1 + 8*0), w01 = vx_load(wptr + c1 + 8*0 + 4);
                    v_float32 w10 = vx_load(wptr + c1 + 8*1), w11 = vx_load(wptr + c1 + 8*1 + 4);
                    v_float32 w20 = vx_load(wptr + c1 + 8*2), w21 = vx_load(wptr + c1 + 8*2 + 4);
                    v_float32 w30 = vx_load(wptr + c1 + 8*3), w31 = vx_load(wptr + c1 + 8*3 + 4);
                    v_float32 w40 = vx_load(wptr + c1 + 8*4), w41 = vx_load(wptr + c1 + 8*4 + 4);
                    v_float32 w50 = vx_load(wptr + c1 + 8*5), w51 = vx_load(wptr + c1 + 8*5 + 4);
                    v_float32 w60 = vx_load(wptr + c1 + 8*6), w61 = vx_load(wptr + c1 + 8*6 + 4);
                    v_float32 w70 = vx_load(wptr + c1 + 8*7), w71 = vx_load(wptr + c1 + 8*7 + 4);

                    int32_t ofs0i, ofs1i, ofs0p, ofs1p;
                    float m0, m1;
                    v_float32 x0, x1;

#undef UPDATE_SUM
#define UPDATE_SUM(a, b) \
    ofs0i = ofs##a[i]; ofs1i = ofs##b[i]; \
    m0 = float(ofs0i >= 0); m1 = float(ofs1i >= 0); \
    ofs0p = std::max(ofs0i, 0); ofs1p = std::max(ofs1i, 0); \
    x0 = vx_setall_f32(inptr##a[ofs0p]*m0); \
    x1 = vx_setall_f32(inptr##b[ofs1p]*m1); \
    s##a##0 = v_fma(w00, x0, s##a##0); \
    s##a##1 = v_fma(w01, x0, s##a##1); \
    s##b##0 = v_fma(w00, x1, s##b##0); \
    s##b##1 = v_fma(w01, x1, s##b##1); \
    x0 = vx_setall_f32(inptr##a[ofs0p+1]*m0); \
    x1 = vx_setall_f32(inptr##b[ofs1p+1]*m1); \
    s##a##0 = v_fma(w10, x0, s##a##0); \
    s##a##1 = v_fma(w11, x0, s##a##1); \
    s##b##0 = v_fma(w10, x1, s##b##0); \
    s##b##1 = v_fma(w11, x1, s##b##1); \
    x0 = vx_setall_f32(inptr##a[ofs0p+2]*m0); \
    x1 = vx_setall_f32(inptr##b[ofs1p+2]*m1); \
    s##a##0 = v_fma(w20, x0, s##a##0); \
    s##a##1 = v_fma(w21, x0, s##a##1); \
    s##b##0 = v_fma(w20, x1, s##b##0); \
    s##b##1 = v_fma(w21, x1, s##b##1); \
    x0 = vx_setall_f32(inptr##a[ofs0p+3]*m0); \
    x1 = vx_setall_f32(inptr##b[ofs1p+3]*m1); \
    s##a##0 = v_fma(w30, x0, s##a##0); \
    s##a##1 = v_fma(w31, x0, s##a##1); \
    s##b##0 = v_fma(w30, x1, s##b##0); \
    s##b##1 = v_fma(w31, x1, s##b##1); \
    x0 = vx_setall_f32(inptr##a[ofs0p+4]*m0); \
    x1 = vx_setall_f32(inptr##b[ofs1p+4]*m1); \
    s##a##0 = v_fma(w40, x0, s##a##0); \
    s##a##1 = v_fma(w41, x0, s##a##1); \
    s##b##0 = v_fma(w40, x1, s##b##0); \
    s##b##1 = v_fma(w41, x1, s##b##1); \
    x0 = vx_setall_f32(inptr##a[ofs0p+5]*m0); \
    x1 = vx_setall_f32(inptr##b[ofs1p+5]*m1); \
    s##a##0 = v_fma(w50, x0, s##a##0); \
    s##a##1 = v_fma(w51, x0, s##a##1); \
    s##b##0 = v_fma(w50, x1, s##b##0); \
    s##b##1 = v_fma(w51, x1, s##b##1); \
    x0 = vx_setall_f32(inptr##a[ofs0p+6]*m0); \
    x1 = vx_setall_f32(inptr##b[ofs1p+6]*m1); \
    s##a##0 = v_fma(w60, x0, s##a##0); \
    s##a##1 = v_fma(w61, x0, s##a##1); \
    s##b##0 = v_fma(w60, x1, s##b##0); \
    s##b##1 = v_fma(w61, x1, s##b##1); \
    x0 = vx_setall_f32(inptr##a[ofs0p+7]*m0); \
    x1 = vx_setall_f32(inptr##b[ofs1p+7]*m1); \
    s##a##0 = v_fma(w70, x0, s##a##0); \
    s##a##1 = v_fma(w71, x0, s##a##1); \
    s##b##0 = v_fma(w70, x1, s##b##0); \
    s##b##1 = v_fma(w71, x1, s##b##1)

                    UPDATE_SUM(0, 1);
                    UPDATE_SUM(2, 3);
                    UPDATE_SUM(4, 5);
                    UPDATE_SUM(6, 7);
                }

                vx_store(sum + 8*0, s00); vx_store(sum + 8*0 + 4, s01);
                vx_store(sum + 8*1, s10); vx_store(sum + 8*1 + 4, s11);
                vx_store(sum + 8*2, s20); vx_store(sum + 8*2 + 4, s21);
                vx_store(sum + 8*3, s30); vx_store(sum + 8*3 + 4, s31);
                vx_store(sum + 8*4, s40); vx_store(sum + 8*4 + 4, s41);
                vx_store(sum + 8*5, s50); vx_store(sum + 8*5 + 4, s51);
                vx_store(sum + 8*6, s60); vx_store(sum + 8*6 + 4, s61);
                vx_store(sum + 8*7, s70); vx_store(sum + 8*7 + 4, s71);
#else
                for (int i = 0; i < BLOCK_SIZE*K0; i++)
                    sum[i] = 0.f;

                for (int c1 = 0, i = 0; c1 < nC; c1 += K0*C0, i++) {
                    for (j = 0; j < BLOCK_SIZE; j++) {
                        int32_t ofs_ij = ofsptrs[j][i];
                        const float* x = &inptrs[j][std::max(ofs_ij, 0)];
                        float mij = (float)(ofs_ij >= 0);
                        for (int c0 = 0; c0 < C0; c0++) {
                            float xc = x[c0]*mij;
                            for (int k = 0; k < K0; k++) {
                                float w = wptr[c1 + c0*K0 + k];
                                sum[K0*j + k] += xc*w;
                            }
                        }
                    }
                }
#endif

                if (activation) {
                    if (resptr) {
                        j = 0;
                    #if CV_SIMD || CV_SIMD_SCALABLE
                        for (; j <= blocksize*K0 - nlanes*2; j += nlanes*2) {
                            v_float32 v0 = vx_load(sum + j);
                            v_float32 v1 = vx_load(sum + j + nlanes);
                            v_float32 scale0 = vx_load(scale + j);
                            v_float32 scale1 = vx_load(scale + j + nlanes);
                            v_float32 bias0 = vx_load(bias + j);
                            v_float32 bias1 = vx_load(bias + j + nlanes);
                            v_float32 res0 = vx_load(resptr + j);
                            v_float32 res1 = vx_load(resptr + j + nlanes);
                            v0 = v_fma(v0, scale0, v_add(bias0, res0));
                            v1 = v_fma(v1, scale1, v_add(bias1, res1));
                            vx_store(sum + j, v0);
                            vx_store(sum + j + nlanes, v1);
                        }
                    #endif
                        for (; j < blocksize*K0; j++) {
                            float v = sum[j]*scale[j] + bias[j] + resptr[j];
                            sum[j] = v;
                        }
                    } else {
                        j = 0;
                    #if CV_SIMD || CV_SIMD_SCALABLE
                        for (; j <= blocksize*K0 - nlanes*2; j += nlanes*2) {
                            v_float32 v0 = vx_load(sum + j);
                            v_float32 v1 = vx_load(sum + j + nlanes);
                            v_float32 scale0 = vx_load(scale + j);
                            v_float32 scale1 = vx_load(scale + j + nlanes);
                            v_float32 bias0 = vx_load(bias + j);
                            v_float32 bias1 = vx_load(bias + j + nlanes);
                            v0 = v_fma(v0, scale0, bias0);
                            v1 = v_fma(v1, scale1, bias1);
                            vx_store(sum + j, v0);
                            vx_store(sum + j + nlanes, v1);
                        }
                    #endif
                        for (; j < blocksize*K0; j++) {
                            float v = sum[j]*scale[j] + bias[j];
                            sum[j] = v;
                        }
                    }
                    activation(sum, out, blocksize*K0, activParams);
                } else {
                    if (resptr) {
                        j = 0;
                    #if CV_SIMD || CV_SIMD_SCALABLE
                        v_float32 valpha = v_setall_f32(alpha);
                        v_float32 vmaxval = v_setall_f32(maxval);
                        v_float32 vzero = v_setzero_f32();
                        
                        for (; j < blocksize*K0; j += nlanes*2) {
                            if (j + nlanes*2 > blocksize*K0) {
                                if (j == 0)
                                    break;
                                j = blocksize*K0 - nlanes*2;
                            }
                            v_float32 v0 = vx_load(sum + j);
                            v_float32 v1 = vx_load(sum + j + nlanes);
                            v_float32 scale0 = vx_load(scale + j);
                            v_float32 scale1 = vx_load(scale + j + nlanes);
                            v_float32 bias0 = vx_load(bias + j);
                            v_float32 bias1 = vx_load(bias + j + nlanes);
                            v_float32 res0 = vx_load(resptr + j);
                            v_float32 res1 = vx_load(resptr + j + nlanes);
                            v0 = v_fma(v0, scale0, v_add(bias0, res0));
                            v1 = v_fma(v1, scale1, v_add(bias1, res1));
                            v0 = v_min(v_select(v_ge(v0, vzero), v0, v_mul(v0, valpha)), vmaxval);
                            v1 = v_min(v_select(v_ge(v1, vzero), v1, v_mul(v1, valpha)), vmaxval);
                            vx_store(out + j, v0);
                            vx_store(out + j + nlanes, v1);
                        }
                    #endif
                        for (; j < blocksize*K0; j++) {
                            float v = sum[j]*scale[j] + bias[j] + resptr[j];
                            v = std::min(v*(v >= 0 ? 1.f : alpha), maxval);
                            out[j] = v;
                        }
                    } else {
                        j = 0;
                    #if CV_SIMD || CV_SIMD_SCALABLE
                        v_float32 valpha = v_setall_f32(alpha);
                        v_float32 vmaxval = v_setall_f32(maxval);
                        v_float32 vzero = v_setzero_f32();
                        
                        for (; j < blocksize*K0; j += nlanes*2) {
                            if (j + nlanes*2 > blocksize*K0) {
                                if (j == 0)
                                    break;
                                j = blocksize*K0 - nlanes*2;
                            }
                            v_float32 v0 = vx_load(sum + j);
                            v_float32 v1 = vx_load(sum + j + nlanes);
                            v_float32 scale0 = vx_load(scale + j);
                            v_float32 scale1 = vx_load(scale + j + nlanes);
                            v_float32 bias0 = vx_load(bias + j);
                            v_float32 bias1 = vx_load(bias + j + nlanes);
                            v0 = v_fma(v0, scale0, bias0);
                            v1 = v_fma(v1, scale1, bias1);
                            v0 = v_min(v_select(v_ge(v0, vzero), v0, v_mul(v0, valpha)), vmaxval);
                            v1 = v_min(v_select(v_ge(v1, vzero), v1, v_mul(v1, valpha)), vmaxval);
                            vx_store(out + j, v0);
                            vx_store(out + j + nlanes, v1);
                        }
                    #endif
                        for (; j < blocksize*K0; j++) {
                            float v = sum[j]*scale[j] + bias[j];
                            v = std::min(v*(v >= 0 ? 1.f : alpha), maxval);
                            out[j] = v;
                        }
                    }
                }
            }
        }
    });
}

#undef __ARM_NEON

static void convAlt32fC8(const void* inp__, const void* residual__, void* out__,
                         const ConvState& cs, const void* weights__,
                         const float* scale__, const float* bias__,
                         const int32_t* inpofs__, const int32_t* ofsofs__)
{
    using FT = float;
    constexpr int C0shift = 3, K0shift = C0shift;
    constexpr int C0 = 1 << C0shift;
    constexpr int K0 = C0;
    const MatShape& inpshape = cs.inpshape;
    const MatShape& outshape = cs.outshape;
    
    CV_Assert_N(inpshape.layout == DATA_LAYOUT_BLOCK, outshape.layout == DATA_LAYOUT_BLOCK);
    CV_Assert_N(inpshape.back() == C0, outshape.back() == K0);
    CV_Assert(!scale__ && !residual__ && cs.fastActivation == FAST_ACTIV_NONE && !cs.activation);
    
    int N = inpshape[0];
    
    int ksize_ = 1;
    for (int i = 0; i < ConvState::MAX_CONV_DIMS; i++)
        ksize_ *= cs.kshape[i];
    
    int Dk_ = cs.kshape[0], Hk_ = cs.kshape[1], Wk_ = cs.kshape[2];

    // precompute (oz,oy,ox) for each i
    AutoBuffer<int> ofsZYXbuf(ksize_ * 3);
    for (int zk = 0, i3 = 0; zk < Dk_; zk++) {
        int oz = zk * cs.dilations[0] - cs.pads[0];
        for (int yk = 0; yk < Hk_; yk++) {
            int oy = yk * cs.dilations[1] - cs.pads[1];
            for (int xk = 0; xk < Wk_; xk++, i3 += 3) {
                int ox = xk * cs.dilations[2] - cs.pads[2];
                ofsZYXbuf[i3] = oz;
                ofsZYXbuf[i3 + 1] = oy;
                ofsZYXbuf[i3 + 2] = ox;
            }
        }
    }

    const int total_blocks = N * cs.ngroups * cs.Kblk;
    memset(out__, 0, outshape.total()*sizeof(FT));

    parallel_for_(Range(0, total_blocks), [&](const Range& range) {
        constexpr int MAX_CONV_DIMS = ConvState::MAX_CONV_DIMS;
        const int C = inpshape.channels(), K = outshape.channels();
        const int C1 = (C + C0 - 1)/C0, K1 = (K + K0 - 1)/K0;
        const int ngroups = cs.ngroups, Kblk = cs.Kblk, C1Max = cs.C1Max;
        const int Cg = C / ngroups;
        const int Kg = K / ngroups;
        int ksize = ksize_;
        int ndims = inpshape.dims;
        const int D = ndims >= 6 ? outshape[ndims-4] : 1;
        const int H = ndims >= 5 ? outshape[ndims-3] : 1;
        const int W = outshape[ndims-2];
        const int Di = ndims >= 6 ? inpshape[ndims-4] : 1;
        const int Hi = ndims >= 5 ? inpshape[ndims-3] : 1;
        const int Wi = inpshape[ndims-2];
        //const int Dk = cs.kshape[0], Hk = cs.kshape[1], Wk = cs.kshape[2];
        const int Sz = cs.strides[0], Sy = cs.strides[1], Sx = cs.strides[2];
        //const int Dz = cs.dilations[0], Dy = cs.dilations[1], Dx = cs.dilations[2];
        //const int padZ = cs.pads[0], padY = cs.pads[1], padX = cs.pads[2];
        const float* biasptr = (const float*)bias__;
        const int* ofsZYX = ofsZYXbuf.data();
        int planesize = D*H*W*K0;
        int iplanesize = Di*Hi*Wi*C0;
        
        int innerZ0 = cs.inner[0], innerZ1 = cs.inner[MAX_CONV_DIMS];
        int innerY0 = cs.inner[1], innerY1 = cs.inner[MAX_CONV_DIMS+1];
        int innerX0 = cs.inner[2], innerX1 = cs.inner[MAX_CONV_DIMS+2];
        
        for (int t = range.start; t < range.end; ++t) {
            const int n = t / (ngroups * Kblk);
            const int rem = t - n * (ngroups * Kblk);
            const int g = rem / Kblk;
            const int kblk = rem - g * Kblk;

            const int k_base = g * Kg + kblk * K0;
            if (k_base >= K) continue;

            const int k_count = std::min(std::min(K0, Kg - kblk*K0), K - k_base);
            //printf("(%d,%d). t=%d, n=%d, g=%d, kblk=%d, k_base=%d, k_count=%d\n", begin, end, t, n, g, kblk, k_base, k_count);

            const int c_start  = g * Cg;
            const int c00      = c_start & (C0-1);
            const int c1_start = c_start >> C0shift;
            const int cblocks  = (c00 + Cg + C0 - 1) >> C0shift;
            const float* inptr0 = (float*)inp__ + (n * C1 + c1_start) * iplanesize;
            const float* wptr0 = (float*)weights__ + (g*Kblk + kblk)*(ksize*C1Max*C0*K0);
            const int b_count = biasptr ? k_count : 0;

#ifdef __ARM_NEON
            float32x4_t vbias_lo = {
                (b_count > 0) ? biasptr[k_base + 0] : 0.0f,
                (b_count > 1) ? biasptr[k_base + 1] : 0.0f,
                (b_count > 2) ? biasptr[k_base + 2] : 0.0f,
                (b_count > 3) ? biasptr[k_base + 3] : 0.0f
            };
            float32x4_t vbias_hi = {
                (b_count > 4) ? biasptr[k_base + 4] : 0.0f,
                (b_count > 5) ? biasptr[k_base + 5] : 0.0f,
                (b_count > 6) ? biasptr[k_base + 6] : 0.0f,
                (b_count > 7) ? biasptr[k_base + 7] : 0.0f
            };
            const int xi_step = Sx * C0;
#endif

            for (int z = 0; z < D; z++) {
                const int zi_base = z * Sz;
                const bool z_inner = (z >= innerZ0 && z < innerZ1);

                for (int y = 0; y < H; y++) {
                    const int yi_base = y * Sy;
                    const bool zy_inner = z_inner && (y >= innerY0 && y < innerY1);

                    int dx = 1;
                    float* outptr0 = (float*)out__ + n*(K1*planesize) + (z*H + y)*(W*K0);

                    for (int x = 0; x < W; x += dx) {
                        float* outptr = outptr0 + x*K0;
                        const int xi_base = x * Sx;

#ifdef __ARM_NEON
                        const int SPAT_BLOCK_SIZE = 6;
                        if (zy_inner &&
                            innerX0 <= x &&
                            x + SPAT_BLOCK_SIZE <= innerX1) {
                            dx = SPAT_BLOCK_SIZE;

                            float32x4_t acc0_lo = vbias_lo, acc0_hi = vbias_hi;
                            float32x4_t acc1_lo = vbias_lo, acc1_hi = vbias_hi;
                            float32x4_t acc2_lo = vbias_lo, acc2_hi = vbias_hi;
                            float32x4_t acc3_lo = vbias_lo, acc3_hi = vbias_hi;
                            float32x4_t acc4_lo = vbias_lo, acc4_hi = vbias_hi;
                            float32x4_t acc5_lo = vbias_lo, acc5_hi = vbias_hi;

                            for (int i = 0; i < ksize; ++i) {
                                const int zi = zi_base + ofsZYX[i * 3 + 0];
                                const int yi = yi_base + ofsZYX[i * 3 + 1];
                                const int xi = xi_base + ofsZYX[i * 3 + 2];

                                const float* inptr = inptr0 + (((zi * Hi) + yi) * Wi + xi) * C0;
                                const float* wptr = wptr0 + i*C1Max*K0*C0;

                                for (int c1 = 0; c1 < cblocks; ++c1, wptr += C0*K0, inptr += iplanesize) {
                                    float32x4_t x0_lo = vld1q_f32(inptr + 0);
                                    float32x4_t x0_hi = vld1q_f32(inptr + 4);
                                    float32x4_t x1_lo = vld1q_f32(inptr + xi_step + 0);
                                    float32x4_t x1_hi = vld1q_f32(inptr + xi_step + 4);
                                    float32x4_t x2_lo = vld1q_f32(inptr + xi_step*2 + 0);
                                    float32x4_t x2_hi = vld1q_f32(inptr + xi_step*2 + 4);
                                    float32x4_t x3_lo = vld1q_f32(inptr + xi_step*3 + 0);
                                    float32x4_t x3_hi = vld1q_f32(inptr + xi_step*3 + 4);
                                    float32x4_t x4_lo = vld1q_f32(inptr + xi_step*4 + 0);
                                    float32x4_t x4_hi = vld1q_f32(inptr + xi_step*4 + 4);
                                    float32x4_t x5_lo = vld1q_f32(inptr + xi_step*5 + 0);
                                    float32x4_t x5_hi = vld1q_f32(inptr + xi_step*5 + 4);
                                    float32x4_t w_lo, w_hi;

#undef ACCROW6
#define ACCROW6(w_ofs, suffix, lane) \
    w_lo = vld1q_f32(wptr + w_ofs*K0 + 0); \
    w_hi = vld1q_f32(wptr + w_ofs*K0 + 4); \
    acc0_lo = vfmaq_laneq_f32(acc0_lo, w_lo, x0_##suffix, lane); \
    acc0_hi = vfmaq_laneq_f32(acc0_hi, w_hi, x0_##suffix, lane); \
    acc1_lo = vfmaq_laneq_f32(acc1_lo, w_lo, x1_##suffix, lane); \
    acc1_hi = vfmaq_laneq_f32(acc1_hi, w_hi, x1_##suffix, lane); \
    acc2_lo = vfmaq_laneq_f32(acc2_lo, w_lo, x2_##suffix, lane); \
    acc2_hi = vfmaq_laneq_f32(acc2_hi, w_hi, x2_##suffix, lane); \
    acc3_lo = vfmaq_laneq_f32(acc3_lo, w_lo, x3_##suffix, lane); \
    acc3_hi = vfmaq_laneq_f32(acc3_hi, w_hi, x3_##suffix, lane); \
    acc4_lo = vfmaq_laneq_f32(acc4_lo, w_lo, x4_##suffix, lane); \
    acc4_hi = vfmaq_laneq_f32(acc4_hi, w_hi, x4_##suffix, lane); \
    acc5_lo = vfmaq_laneq_f32(acc5_lo, w_lo, x5_##suffix, lane); \
    acc5_hi = vfmaq_laneq_f32(acc5_hi, w_hi, x5_##suffix, lane); \

                                    ACCROW6(0, lo, 0);
                                    ACCROW6(1, lo, 1);
                                    ACCROW6(2, lo, 2);
                                    ACCROW6(3, lo, 3);

                                    ACCROW6(4, hi, 0);
                                    ACCROW6(5, hi, 1);
                                    ACCROW6(6, hi, 2);
                                    ACCROW6(7, hi, 3);
                                }
                            }

                            float tmp[SPAT_BLOCK_SIZE*C0];
                            vst1q_f32(tmp + 0, acc0_lo); vst1q_f32(tmp + 4, acc0_hi);
                            vst1q_f32(tmp + 8, acc1_lo); vst1q_f32(tmp + 8 + 4, acc1_hi);
                            vst1q_f32(tmp + 8*2, acc2_lo); vst1q_f32(tmp + 8*2 + 4, acc2_hi);
                            vst1q_f32(tmp + 8*3, acc3_lo); vst1q_f32(tmp + 8*3 + 4, acc3_hi);
                            vst1q_f32(tmp + 8*4, acc4_lo); vst1q_f32(tmp + 8*4 + 4, acc4_hi);
                            vst1q_f32(tmp + 8*5, acc5_lo); vst1q_f32(tmp + 8*5 + 4, acc5_hi);

                            for (int kk = 0; kk < k_count; ++kk) {
                                const int k = k_base + kk;
                                int kofs = (k >> K0shift) * planesize + (k & (K0-1));
                                outptr[kofs + 0 * 8] = tmp[kk];
                                outptr[kofs + 1 * 8] = tmp[kk+8];
                                outptr[kofs + 2 * 8] = tmp[kk+8*2];
                                outptr[kofs + 3 * 8] = tmp[kk+8*3];
                                outptr[kofs + 4 * 8] = tmp[kk+8*4];
                                outptr[kofs + 5 * 8] = tmp[kk+8*5];
                            }

                            continue;
                        }
#endif
                        dx = 1;

                        float accs[8];

#ifdef __ARM_NEON
                        float32x4_t acc0 = vbias_lo, acc1 = vbias_hi;
#else
                        for (int kk = 0; kk < K0; ++kk)
                            accs[kk] = (kk < b_count) ? biasptr[k_base + kk] : 0.0f;
#endif

                        for (int i = 0; i < ksize; ++i) {
                            size_t i3 = size_t(i)*3u;
                            const int zi = zi_base + ofsZYX[i3 + 0];
                            const int yi = yi_base + ofsZYX[i3 + 1];
                            const int xi = xi_base + ofsZYX[i3 + 2];
                            if ((((unsigned)zi >= (unsigned)Di) |
                                ((unsigned)yi >= (unsigned)Hi) |
                                ((unsigned)xi >= (unsigned)Wi)) != 0)
                                continue;

                            const float* inptr = inptr0 + (((zi * Hi) + yi) * Wi + xi) * C0;
                            const float* wptr = wptr0 + i*C1Max*K0*C0;

#ifdef __ARM_NEON
                            for (int c1 = 0; c1 < cblocks; ++c1, inptr += iplanesize, wptr += K0*C0) {
                                float32x4_t x0 = vld1q_f32(inptr), x1 = vld1q_f32(inptr + 4);
                                float32x4_t w0, w1;
#undef ACCROW1
#define ACCROW1(w_ofs, idx, lane) \
    w0 = vld1q_f32(wptr + w_ofs*K0); \
    w1 = vld1q_f32(wptr + w_ofs*K0 + 4); \
    acc0 = vfmaq_laneq_f32(acc0, w0, x##idx, lane); \
    acc1 = vfmaq_laneq_f32(acc1, w1, x##idx, lane)

                                ACCROW1(0, 0, 0);
                                ACCROW1(1, 0, 1);
                                ACCROW1(2, 0, 2);
                                ACCROW1(3, 0, 3);
                                ACCROW1(4, 1, 0);
                                ACCROW1(5, 1, 1);
                                ACCROW1(6, 1, 2);
                                ACCROW1(7, 1, 3);
                            }
#else
                            for (int c1 = 0; c1 < cblocks; ++c1, inptr += iplanesize, wptr += K0*C0) {
                                for (int c0 = 0; c0 < C0; ++c0) {
                                    const float xval = inptr[c0];
                                    for (int kk = 0; kk < K0; ++kk)
                                        accs[kk] += xval * wptr[c0*K0 + kk];
                                }
                            }
#endif
                        }
#ifdef __ARM_NEON
                        vst1q_f32(accs, acc0);
                        vst1q_f32(accs + 4, acc1);
#endif
                        for (int kk = 0; kk < k_count; ++kk) {
                            const int k = k_base + kk;
                            int kofs = (k >> K0shift) * planesize + (k & (K0-1));
                            outptr[kofs] = accs[kk];
                        }
                    }
                }
            }
        }
    });
}

cv::dnn::ConvFunc getConvFunc_(int depth, int C0, ConvKind convkind)
{
    ConvFunc func = nullptr;
    if (depth == CV_32F && C0 == 8) {
        func = convkind == CONV_KIND_MAIN ? conv32fC8 : convAlt32fC8;
    }
    return func;
}

CV_CPU_OPTIMIZATION_NAMESPACE_END
}}
#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
