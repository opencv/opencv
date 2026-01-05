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
                        for (j = 0; j < blocksize*K0; j++) {
                            float v = sum[j]*scale[j] + bias[j] + resptr[j];
                            sum[j] = v;
                        }
                    } else {
                        for (j = 0; j < blocksize*K0; j++) {
                            float v = sum[j]*scale[j] + bias[j];
                            sum[j] = v;
                        }
                    }
                    activation(sum, out, blocksize*K0, activParams);
                } else {
                    if (resptr) {
                        for (j = 0; j < blocksize*K0; j++) {
                            float v = sum[j]*scale[j] + bias[j] + resptr[j];
                            v = std::min(v*(v >= 0 ? 1.f : alpha), maxval);
                            out[j] = v;
                        }
                    } else {
                        for (j = 0; j < blocksize*K0; j++) {
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

cv::dnn::ConvFunc getConvFunc_(int depth, int C0)
{
    ConvFunc func = depth == CV_32F && C0 == 8 ? conv32fC8 : nullptr;
    return func;
}

CV_CPU_OPTIMIZATION_NAMESPACE_END
}}
#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
