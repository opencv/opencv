// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "../net_impl.hpp"
#include "../layers/conv2_common.hpp"
#include "opencv2/core/hal/intrin.hpp"

namespace cv {
namespace dnn {

static MatShape getWpackShapeInt8(const MatShape& wshape, int ngroups, int C0)
{
    CV_Assert(wshape.dims >= 3);
    int K = wshape[0], Cg = wshape[1];
    int ksize = (int)(wshape.total()) / (K * Cg);
    CV_Assert(K % ngroups == 0);
    int Kg = K / ngroups, K0 = C0;
    int Kblk = (Kg + K0 - 1) / K0;
    int C1Max = 0;
    for (int g = 0; g < ngroups; ++g) {
        int c_start = g * Cg;
        int c00 = c_start & (C0 - 1);
        int cblocks = (c00 + Cg + C0 - 1) / C0;
        C1Max = std::max(C1Max, cblocks);
    }
    return MatShape({ngroups, Kblk, ksize, C1Max, C0 * K0}, DATA_LAYOUT_UNKNOWN);
}

static void repackConvWeightsInt8(const Mat& weights, Mat& Wpack, int ngroups, int C0_)
{
    CV_Assert(weights.isContinuous());
    CV_Assert(weights.type() == CV_8SC1);
    CV_Assert(ngroups > 0);
    CV_Assert((C0_ & (C0_ - 1)) == 0 && C0_ >= 4);

    MatShape wshape = weights.shape();
    CV_Assert(wshape.dims >= 3);

    int K = wshape[0];
    CV_Assert(K % ngroups == 0);

    if (!Wpack.isContinuous())
        Wpack.release();

    MatShape wpackShape = getWpackShapeInt8(wshape, ngroups, C0_);
    Wpack.create(wpackShape, CV_8SC1);
    Wpack.setZero();

    parallel_for_(Range(0, K), [&](const Range& range) {
        int Cg = wshape[1], Kg = K / ngroups;
        int ksize = wpackShape[2], C1Max = wpackShape[3];
        int C0 = C0_, K0 = C0;
        const int8_t* wdata = weights.ptr<int8_t>();
        int8_t* Wpackdata = Wpack.ptr<int8_t>();

        for (int k = range.start; k < range.end; ++k) {
            int g = k / Kg;
            int kin = k - g * Kg;
            int kblk = kin / K0;
            int k0 = kin & (K0 - 1);

            int c_start = g * Cg;
            int c00 = c_start & (C0 - 1);

            for (int c = 0; c < Cg; ++c) {
                int ch = c00 + c;
                int c1 = ch / C0;
                int c0 = ch & (C0 - 1);

                const int8_t* wptr = wdata + ((k * Cg + c) * ksize);
                int8_t* wpackptr = Wpackdata +
                    (((g * (int64_t)wpackShape[1] + kblk) * ksize * C1Max + c1) * C0 + c0) * K0 + k0;
                for (int i = 0; i < ksize; ++i) {
                    wpackptr[i * (C1Max * C0 * K0)] = wptr[i];
                }
            }
        }
    });
}

static void repackWeightsForVNNI(const Mat& wpack, int ngroups, int Kg, int Cg,
                                  const Mat& biasInt32,
                                  Mat& wpackVNNI, Mat& biasVNNI)
{
    constexpr int C0 = 8, K0 = 8;
    MatShape ws = wpack.shape();
    int Kblk = ws[1], ksize = ws[2], C1Max = ws[3];

    wpackVNNI.create(ws, CV_8SC1);
    const int8_t* src = wpack.ptr<int8_t>();
    int8_t* dst = wpackVNNI.ptr<int8_t>();

    int blockSize = C0 * K0;
    int totalBlocks = (int)(ws.total() / blockSize);

    parallel_for_(Range(0, totalBlocks), [&](const Range& range) {
        for (int b = range.start; b < range.end; b++) {
            const int8_t* s = src + (size_t)b * blockSize;
            int8_t* d = dst + (size_t)b * blockSize;
            for (int k = 0; k < K0; k++) {
                d[k*4 + 0] = s[0*K0 + k];
                d[k*4 + 1] = s[1*K0 + k];
                d[k*4 + 2] = s[2*K0 + k];
                d[k*4 + 3] = s[3*K0 + k];
                d[32 + k*4 + 0] = s[4*K0 + k];
                d[32 + k*4 + 1] = s[5*K0 + k];
                d[32 + k*4 + 2] = s[6*K0 + k];
                d[32 + k*4 + 3] = s[7*K0 + k];
            }
        }
    });

    int K = ngroups * Kg;
    biasVNNI.create({K}, CV_32SC1);
    const int32_t* bsrc = biasInt32.ptr<int32_t>();
    int32_t* bdst = biasVNNI.ptr<int32_t>();

    parallel_for_(Range(0, K), [&](const Range& range) {
        for (int k = range.start; k < range.end; k++) {
            int g = k / Kg;
            int kin = k - g * Kg;
            int kblk = kin / K0;
            int k0 = kin & (K0 - 1);
            int c_start = g * Cg;
            int c00 = c_start & (C0 - 1);
            int cblocks = (c00 + Cg + C0 - 1) / C0;

            const int8_t* wbase = src + (size_t)((g * Kblk + kblk) * ksize * C1Max) * C0 * K0;
            int32_t wsum = 0;
            for (int ki = 0; ki < ksize; ki++)
                for (int c1 = 0; c1 < cblocks; c1++)
                    for (int c0 = 0; c0 < C0; c0++)
                        wsum += (int)wbase[((size_t)ki * C1Max + c1) * C0 * K0 + c0 * K0 + k0];
            bdst[k] = bsrc[k] - 128 * wsum;
        }
    });
}

#if !defined(CV_AVXVNNI_AVAILABLE)
#if (CV_TRY_AVX2 || CV_AVX2) && \
    ((defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 11) || \
     (defined(__clang__) && !defined(__apple_build_version__) && __clang_major__ >= 12))
#define CV_AVXVNNI_AVAILABLE 1
#else
#define CV_AVXVNNI_AVAILABLE 0
#endif
#endif

#if CV_AVXVNNI_AVAILABLE
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#pragma GCC push_options
#pragma GCC target("avxvnni")
static void convInt8BlockVNNI(const void* inp_, const void* residual_,
                              void* out_, const ConvState& cs,
                              const void* weightsVNNI_,
                              const int* biasVNNI, const float* multiplier,
                              int /*inp_zp*/, int out_zp,
                              const int8_t* activLUT)
{
    constexpr int C0 = 8, K0 = 8;
    constexpr int C0shift = 3;
    constexpr int SPAT_BLOCK_SIZE = 8;
    constexpr int MAX_CONV_DIMS = ConvState::MAX_CONV_DIMS;

    CV_Assert(cs.inpshape.layout == DATA_LAYOUT_BLOCK);
    CV_Assert(cs.outshape.layout == DATA_LAYOUT_BLOCK);
    CV_Assert(cs.inpshape.back() == C0);

    const MatShape& inpshape = cs.inpshape;
    const MatShape& outshape = cs.outshape;

    int N = outshape[0];
    int ndims = outshape.dims;
    int K = outshape.channels();
    int C = inpshape.channels();
    int K1 = (K + K0 - 1) / K0;
    int C1 = (C + C0 - 1) / C0;

    int D_ = ndims >= 6 ? outshape[ndims-4] : 1;
    int H_ = ndims >= 5 ? outshape[ndims-3] : 1;
    int W_ = outshape[ndims-2];
    int planeblocks_ = D_ * H_ * W_;

    int ngroups = cs.ngroups;
    int Kg = K / ngroups;
    int Cg = C / ngroups;
    int Kblk = cs.wshape[1];
    int ksize_ = cs.wshape[2];
    int C1Max = cs.wshape[3];

    int innerZ0 = cs.inner[0], innerZ1 = cs.inner[MAX_CONV_DIMS];
    int innerY0 = cs.inner[1], innerY1 = cs.inner[MAX_CONV_DIMS+1];
    int innerX0 = cs.inner[2], innerX1 = cs.inner[MAX_CONV_DIMS+2];

    const int8_t* inp = (const int8_t*)inp_;
    const int8_t* residual = (const int8_t*)residual_;
    int8_t* out = (int8_t*)out_;
    const int8_t* wdata = (const int8_t*)weightsVNNI_;
    const int* ofsZYX = cs.coordtab.data();

    size_t outtotal = outshape.total();
    if ((Kg % K0) != 0) memset(out, 0, outtotal);

    int total_blocks = N * ngroups * Kblk;
    const __m256i v_xor = _mm256_set1_epi8((char)0x80);

    parallel_for_(Range(0, total_blocks), [&](const Range& range) {
        int D = D_, H = H_, W = W_;
        int Di = ndims >= 6 ? inpshape[ndims-4] : 1;
        int Hi = ndims >= 5 ? inpshape[ndims-3] : 1;
        int Wi = inpshape[ndims-2];
        int planeblocks = planeblocks_;
        int iplanesize = Di * Hi * Wi * C0;
        int planesize = planeblocks * K0;

        int Sz = cs.strides[0], Sy = cs.strides[1], Sx = cs.strides[2];
        int padZ = cs.pads[0], padY = cs.pads[1], padX = cs.pads[2];
        int ksize = ksize_;
        int8_t zbuf[C0] = {};

        for (int t = range.start; t < range.end; t++) {
            int n = t / (ngroups * Kblk);
            int rem = t - n * (ngroups * Kblk);
            int g = rem / Kblk;
            int kblk = rem - g * Kblk;

            int k_base = g * Kg + kblk * K0;
            if (k_base >= K) continue;

            int c_start = g * Cg;
            int c1_start = c_start >> C0shift;
            int c00 = c_start & (C0 - 1);
            int cblocks = (c00 + Cg + C0 - 1) >> C0shift;

            const int8_t* inpbaseptr = inp + (size_t)(n * C1 + c1_start) * iplanesize;
            const int8_t* wbaseptr = wdata + (size_t)(g * Kblk + kblk) * ksize * C1Max * C0 * K0;

            int k1 = k_base >> C0shift;
            int8_t* outptr = out + (size_t)(n * K1 + k1) * planesize;
            const int8_t* resptr = residual ? residual + (size_t)(n * K1 + k1) * planesize : nullptr;

            alignas(32) int32_t biasbuf[K0];
            memcpy(biasbuf, biasVNNI + k_base, K0 * sizeof(int32_t));

            alignas(32) float multbuf[K0];
            memcpy(multbuf, multiplier + k_base, K0 * sizeof(float));

            int D_l = D, H_l = H, W_l = W;
            int Di_l = Di, Hi_l = Hi, Wi_l = Wi;
            int iplanesize_l = iplanesize;
            int planeblocks_l = planeblocks;
            int ksize_l = ksize;

            if (ksize == 1 && Sx == 1 && Sy == 1 && Sz == 1) {
                W_l *= D_l * H_l;
                Wi_l *= Di_l * Hi_l;
                D_l = Di_l = H_l = Hi_l = 1;
                iplanesize_l = Wi_l * C0;
                planeblocks_l = W_l;
            }

            int p = 0;
            for (; p < planeblocks_l; p += SPAT_BLOCK_SIZE) {
                if (p + SPAT_BLOCK_SIZE > planeblocks_l) {
                    if (p == 0) break;
                    int p_new = planeblocks_l - SPAT_BLOCK_SIZE;
                    outptr += (p_new - p) * K0;
                    if (resptr) resptr += (p_new - p) * K0;
                    p = p_new;
                }

                Vec3i pt[SPAT_BLOCK_SIZE];
                bool inner[SPAT_BLOCK_SIZE];
                bool all_inner = true;

                if ((p % W_l) + SPAT_BLOCK_SIZE <= W_l) {
                    int zj = p / (H_l * W_l);
                    int yxj = p - zj * H_l * W_l;
                    int yj = yxj / W_l;
                    int xj0 = yxj - yj * W_l;
                    bool zy_inner = (zj >= innerZ0 && zj < innerZ1) &&
                                    (yj >= innerY0 && yj < innerY1);
                    for (int j = 0; j < SPAT_BLOCK_SIZE; j++) {
                        int xj = xj0 + j;
                        pt[j] = Vec3i(zj * Sz - padZ, yj * Sy - padY, xj * Sx - padX);
                        inner[j] = zy_inner && (xj >= innerX0 && xj < innerX1);
                        all_inner &= inner[j];
                    }
                } else {
                    for (int j = 0; j < SPAT_BLOCK_SIZE; j++) {
                        int pj = p + j;
                        int zj = pj / (H_l * W_l);
                        int yxj = pj - zj * H_l * W_l;
                        int yj = yxj / W_l;
                        int xj = yxj - yj * W_l;
                        pt[j] = Vec3i(zj * Sz - padZ, yj * Sy - padY, xj * Sx - padX);
                        inner[j] = (zj >= innerZ0 && zj < innerZ1) &&
                                   (yj >= innerY0 && yj < innerY1) &&
                                   (xj >= innerX0 && xj < innerX1);
                        all_inner &= inner[j];
                    }
                }

                __m256i vbias = _mm256_load_si256((const __m256i*)biasbuf);
                __m256i s0 = vbias, s1 = vbias, s2 = vbias, s3 = vbias;
                __m256i s4 = vbias, s5 = vbias, s6 = vbias, s7 = vbias;

                #define CONV_VNNI_MAC(j) { \
                    __m256i x0 = _mm256_xor_si256(_mm256_set1_epi32(*(const int32_t*)&inptr[(j)][0]), v_xor); \
                    __m256i x1 = _mm256_xor_si256(_mm256_set1_epi32(*(const int32_t*)&inptr[(j)][4]), v_xor); \
                    s##j = _mm256_dpbusd_epi32(s##j, x0, wg0); \
                    s##j = _mm256_dpbusd_epi32(s##j, x1, wg1); \
                }

                if (all_inner) {
                    for (int i = 0; i < ksize_l; i++) {
                        const int8_t* inptr[SPAT_BLOCK_SIZE];
                        for (int j = 0; j < SPAT_BLOCK_SIZE; j++) {
                            int zij = pt[j][0] + ofsZYX[i * 3];
                            int yij = pt[j][1] + ofsZYX[i * 3 + 1];
                            int xij = pt[j][2] + ofsZYX[i * 3 + 2];
                            inptr[j] = inpbaseptr + (((zij * Hi_l) + yij) * Wi_l + xij) * C0;
                        }

                        const int8_t* wptr = wbaseptr + (size_t)i * C1Max * K0 * C0;

                        for (int c1 = 0; c1 < cblocks; c1++, wptr += C0 * K0) {
                            __m256i wg0 = _mm256_load_si256((const __m256i*)(wptr));
                            __m256i wg1 = _mm256_load_si256((const __m256i*)(wptr + 32));

                            CONV_VNNI_MAC(0); CONV_VNNI_MAC(1);
                            CONV_VNNI_MAC(2); CONV_VNNI_MAC(3);
                            CONV_VNNI_MAC(4); CONV_VNNI_MAC(5);
                            CONV_VNNI_MAC(6); CONV_VNNI_MAC(7);

                            for (int j = 0; j < SPAT_BLOCK_SIZE; j++)
                                inptr[j] += iplanesize_l;
                        }
                    }
                } else {
                    for (int i = 0; i < ksize_l; i++) {
                        const int8_t* inptr[SPAT_BLOCK_SIZE];
                        int inpstep[SPAT_BLOCK_SIZE];

                        for (int j = 0; j < SPAT_BLOCK_SIZE; j++) {
                            int zij = pt[j][0] + ofsZYX[i * 3];
                            int yij = pt[j][1] + ofsZYX[i * 3 + 1];
                            int xij = pt[j][2] + ofsZYX[i * 3 + 2];
                            if (inner[j] || ((((unsigned)zij < (unsigned)Di_l) &
                                 ((unsigned)yij < (unsigned)Hi_l) &
                                 ((unsigned)xij < (unsigned)Wi_l)) != 0)) {
                                inptr[j] = inpbaseptr + (((zij * Hi_l) + yij) * Wi_l + xij) * C0;
                                inpstep[j] = iplanesize_l;
                            } else {
                                inptr[j] = zbuf;
                                inpstep[j] = 0;
                            }
                        }

                        const int8_t* wptr = wbaseptr + (size_t)i * C1Max * K0 * C0;

                        for (int c1 = 0; c1 < cblocks; c1++, wptr += C0 * K0) {
                            __m256i wg0 = _mm256_load_si256((const __m256i*)(wptr));
                            __m256i wg1 = _mm256_load_si256((const __m256i*)(wptr + 32));

                            CONV_VNNI_MAC(0); CONV_VNNI_MAC(1);
                            CONV_VNNI_MAC(2); CONV_VNNI_MAC(3);
                            CONV_VNNI_MAC(4); CONV_VNNI_MAC(5);
                            CONV_VNNI_MAC(6); CONV_VNNI_MAC(7);

                            for (int j = 0; j < SPAT_BLOCK_SIZE; j++)
                                inptr[j] += inpstep[j];
                        }
                    }
                }
                #undef CONV_VNNI_MAC

                __m256 vmult = _mm256_load_ps(multbuf);
                __m256 vzp = _mm256_set1_ps((float)out_zp);

                #define CONV_VNNI_STORE(j) { \
                    __m256 facc = _mm256_add_ps( \
                        _mm256_mul_ps(_mm256_cvtepi32_ps(s##j), vmult), vzp); \
                    if (resptr) { \
                        __m256i res32 = _mm256_cvtepi8_epi32( \
                            _mm_loadl_epi64((const __m128i*)(resptr + (p + (j)) * K0))); \
                        facc = _mm256_add_ps(facc, \
                            _mm256_sub_ps(_mm256_cvtepi32_ps(res32), vzp)); \
                    } \
                    __m256i ival = _mm256_cvtps_epi32(facc); \
                    __m128i lo = _mm256_castsi256_si128(ival); \
                    __m128i hi = _mm256_extracti128_si256(ival, 1); \
                    __m128i packed16 = _mm_packs_epi32(lo, hi); \
                    __m128i packed8 = _mm_packs_epi16(packed16, packed16); \
                    int8_t* optr = outptr + (p + (j)) * K0; \
                    if (activLUT) { \
                        alignas(16) int8_t tmp8[16]; \
                        _mm_store_si128((__m128i*)tmp8, packed8); \
                        for (int k = 0; k < K0; k++) \
                            optr[k] = (int8_t)activLUT[(int)tmp8[k] + 128]; \
                    } else { \
                        _mm_storel_epi64((__m128i*)optr, packed8); \
                    } \
                }
                CONV_VNNI_STORE(0); CONV_VNNI_STORE(1);
                CONV_VNNI_STORE(2); CONV_VNNI_STORE(3);
                CONV_VNNI_STORE(4); CONV_VNNI_STORE(5);
                CONV_VNNI_STORE(6); CONV_VNNI_STORE(7);
                #undef CONV_VNNI_STORE
            }

            // Scalar tail
            for (; p < planeblocks_l; p++) {
                alignas(32) int32_t acc[K0];
                memcpy(acc, biasbuf, K0 * sizeof(int32_t));

                int zj = p / (H_l * W_l);
                int yxj = p - zj * H_l * W_l;
                int yj = yxj / W_l;
                int xj = yxj - yj * W_l;
                int zi_base = zj * Sz - padZ, yi_base = yj * Sy - padY, xi_base = xj * Sx - padX;

                for (int i = 0; i < ksize_l; i++) {
                    int zi = zi_base + ofsZYX[i * 3];
                    int yi = yi_base + ofsZYX[i * 3 + 1];
                    int xi = xi_base + ofsZYX[i * 3 + 2];
                    if ((((unsigned)zi >= (unsigned)Di_l) | ((unsigned)yi >= (unsigned)Hi_l) |
                         ((unsigned)xi >= (unsigned)Wi_l)) != 0) continue;
                    const int8_t* inptr = inpbaseptr + (((zi * Hi_l) + yi) * Wi_l + xi) * C0;
                    const int8_t* wptr = wbaseptr + (size_t)i * C1Max * K0 * C0;
                    for (int c1i = 0; c1i < cblocks; c1i++, inptr += iplanesize_l, wptr += C0 * K0) {
                        for (int c0 = 0; c0 < C0; c0++) {
                            int ival = (int)(uint8_t)((uint8_t)inptr[c0] ^ 0x80u);
                            const int8_t* wp = wptr + c0 * K0;
                            for (int k0 = 0; k0 < K0; k0++)
                                acc[k0] += ival * (int)wp[k0];
                        }
                    }
                }
                int8_t* optr = outptr + p * K0;
                const int8_t* rptr = resptr ? resptr + p * K0 : nullptr;
                for (int k0 = 0; k0 < K0; k0++) {
                    float val = (float)acc[k0] * multbuf[k0] + (float)out_zp;
                    if (rptr) val += (float)((int)rptr[k0] - out_zp);
                    int ival = cvRound(val);
                    ival = std::max(-128, std::min(127, ival));
                    if (activLUT) ival = (int)activLUT[ival + 128];
                    optr[k0] = (int8_t)ival;
                }
            }
        }
    });
}
#pragma GCC pop_options
#endif // CV_AVXVNNI_AVAILABLE

static void convInt8Block(const void* inp_, const void* residual_,
                          void* out_, const ConvState& cs,
                          const void* weights_,
                          const void* weightsVNNI_,
                          const int* bias, const int* biasVNNI_,
                          const float* multiplier,
                          int /*inp_zp*/, int out_zp,
                          const int8_t* activLUT)
{
#if CV_AVXVNNI_AVAILABLE
    if (cv::checkHardwareSupport(CV_CPU_AVX_VNNI) && weightsVNNI_ && biasVNNI_) {
        convInt8BlockVNNI(inp_, residual_, out_, cs, weightsVNNI_,
                          biasVNNI_, multiplier, 0, out_zp, activLUT);
        return;
    }
#else
    (void)weightsVNNI_;
    (void)biasVNNI_;
#endif

    constexpr int C0 = 8, K0 = 8;
    constexpr int C0shift = 3;

    CV_Assert(cs.inpshape.layout == DATA_LAYOUT_BLOCK);
    CV_Assert(cs.outshape.layout == DATA_LAYOUT_BLOCK);
    CV_Assert(cs.inpshape.back() == C0);

    const MatShape& inpshape = cs.inpshape;
    const MatShape& outshape = cs.outshape;

    int N = outshape[0];
    int ndims = outshape.dims;
    int K = outshape.channels();
    int C = inpshape.channels();
    int K1 = (K + K0 - 1) / K0;
    int C1 = (C + C0 - 1) / C0;

    int D_ = ndims >= 6 ? outshape[ndims-4] : 1;
    int H_ = ndims >= 5 ? outshape[ndims-3] : 1;
    int W_ = outshape[ndims-2];
    int planeblocks_ = D_ * H_ * W_;

    int ngroups = cs.ngroups;
    int Kg = K / ngroups;
    int Cg = C / ngroups;
    int Kblk = cs.wshape[1];
    int ksize_ = cs.wshape[2];
    int C1Max = cs.wshape[3];

#if CV_AVX2
    constexpr int MAX_CONV_DIMS = ConvState::MAX_CONV_DIMS;
    int innerZ0 = cs.inner[0], innerZ1 = cs.inner[MAX_CONV_DIMS];
    int innerY0 = cs.inner[1], innerY1 = cs.inner[MAX_CONV_DIMS+1];
    int innerX0 = cs.inner[2], innerX1 = cs.inner[MAX_CONV_DIMS+2];
#endif

    const int8_t* inp = (const int8_t*)inp_;
    const int8_t* residual = (const int8_t*)residual_;
    int8_t* out = (int8_t*)out_;
    const int8_t* wdata = (const int8_t*)weights_;
    const int* ofsZYX = cs.coordtab.data();

    size_t outtotal = outshape.total();
    if ((Kg % K0) != 0) {
        memset(out, 0, outtotal);
    }

    int total_blocks = N * ngroups * Kblk;

    parallel_for_(Range(0, total_blocks), [&](const Range& range) {
#if CV_AVX2
        constexpr int SPAT_BLOCK_SIZE = 8;
        alignas(32) int32_t sumbuf[SPAT_BLOCK_SIZE * K0];
#endif

        int D = D_, H = H_, W = W_;
        int Di = ndims >= 6 ? inpshape[ndims-4] : 1;
        int Hi = ndims >= 5 ? inpshape[ndims-3] : 1;
        int Wi = inpshape[ndims-2];
        int planeblocks = planeblocks_;
        int iplanesize = Di * Hi * Wi * C0;
        int planesize = planeblocks * K0;

        int Sz = cs.strides[0], Sy = cs.strides[1], Sx = cs.strides[2];
        int padZ = cs.pads[0], padY = cs.pads[1], padX = cs.pads[2];
        int ksize = ksize_;
#if CV_AVX2
        int8_t zbuf[C0] = {};
#endif

        for (int t = range.start; t < range.end; t++) {
            int n = t / (ngroups * Kblk);
            int rem = t - n * (ngroups * Kblk);
            int g = rem / Kblk;
            int kblk = rem - g * Kblk;

            int k_base = g * Kg + kblk * K0;
            if (k_base >= K) continue;

            int c_start = g * Cg;
            int c1_start = c_start >> C0shift;
            int c00 = c_start & (C0 - 1);
            int cblocks = (c00 + Cg + C0 - 1) >> C0shift;

            const int8_t* inpbaseptr = inp + (size_t)(n * C1 + c1_start) * iplanesize;
            const int8_t* wbaseptr = wdata + (size_t)(g * Kblk + kblk) * ksize * C1Max * C0 * K0;

            int k1 = k_base >> C0shift;
            int8_t* outptr = out + (size_t)(n * K1 + k1) * planesize;
            const int8_t* resptr = residual ? residual + (size_t)(n * K1 + k1) * planesize : nullptr;

            int D_l = D, H_l = H, W_l = W;
            int Di_l = Di, Hi_l = Hi, Wi_l = Wi;
            int iplanesize_l = iplanesize;
            int planeblocks_l = planeblocks;
            int ksize_l = ksize;

            if (ksize == 1 && Sx == 1 && Sy == 1 && Sz == 1) {
                W_l *= D_l * H_l;
                Wi_l *= Di_l * Hi_l;
                D_l = Di_l = H_l = Hi_l = 1;
                iplanesize_l = Wi_l * C0;
                planeblocks_l = W_l;
            }

            alignas(32) int32_t biasbuf[K0];
            memcpy(biasbuf, bias + k_base, K0 * sizeof(int32_t));

            alignas(32) float multbuf[K0];
            memcpy(multbuf, multiplier + k_base, K0 * sizeof(float));

        #if CV_AVX2
            int p = 0;
            for (; p < planeblocks_l; p += SPAT_BLOCK_SIZE) {
                if (p + SPAT_BLOCK_SIZE > planeblocks_l) {
                    if (p == 0) break;
                    int p_new = planeblocks_l - SPAT_BLOCK_SIZE;
                    outptr += (p_new - p) * K0;
                    if (resptr) resptr += (p_new - p) * K0;
                    p = p_new;
                }

                Vec3i pt[SPAT_BLOCK_SIZE];
                bool inner[SPAT_BLOCK_SIZE];
                bool all_inner = true;

                if ((p % W_l) + SPAT_BLOCK_SIZE <= W_l) {
                    int zj = p / (H_l * W_l);
                    int yxj = p - zj * H_l * W_l;
                    int yj = yxj / W_l;
                    int xj0 = yxj - yj * W_l;
                    bool zy_inner = (zj >= innerZ0 && zj < innerZ1) &&
                                    (yj >= innerY0 && yj < innerY1);
                    for (int j = 0; j < SPAT_BLOCK_SIZE; j++) {
                        int xj = xj0 + j;
                        pt[j] = Vec3i(zj * Sz - padZ, yj * Sy - padY, xj * Sx - padX);
                        inner[j] = zy_inner && (xj >= innerX0 && xj < innerX1);
                        all_inner &= inner[j];
                    }
                } else {
                    for (int j = 0; j < SPAT_BLOCK_SIZE; j++) {
                        int pj = p + j;
                        int zj = pj / (H_l * W_l);
                        int yxj = pj - zj * H_l * W_l;
                        int yj = yxj / W_l;
                        int xj = yxj - yj * W_l;
                        pt[j] = Vec3i(zj * Sz - padZ, yj * Sy - padY, xj * Sx - padX);
                        inner[j] = (zj >= innerZ0 && zj < innerZ1) &&
                                   (yj >= innerY0 && yj < innerY1) &&
                                   (xj >= innerX0 && xj < innerX1);
                        all_inner &= inner[j];
                    }
                }

                __m256i vbias = _mm256_load_si256((const __m256i*)biasbuf);
                __m256i s0 = vbias, s1 = vbias, s2 = vbias, s3 = vbias;
                __m256i s4 = vbias, s5 = vbias, s6 = vbias, s7 = vbias;

                for (int i = 0; i < ksize_l; i++) {
                    const int8_t* inptr[SPAT_BLOCK_SIZE];
                    int inpstep[SPAT_BLOCK_SIZE];

                    for (int j = 0; j < SPAT_BLOCK_SIZE; j++) {
                        int zij = pt[j][0] + ofsZYX[i * 3];
                        int yij = pt[j][1] + ofsZYX[i * 3 + 1];
                        int xij = pt[j][2] + ofsZYX[i * 3 + 2];

                        if (inner[j] || ((((unsigned)zij < (unsigned)Di_l) &
                             ((unsigned)yij < (unsigned)Hi_l) &
                             ((unsigned)xij < (unsigned)Wi_l)) != 0)) {
                            inptr[j] = inpbaseptr + (((zij * Hi_l) + yij) * Wi_l + xij) * C0;
                            inpstep[j] = iplanesize_l;
                        } else {
                            inptr[j] = zbuf;
                            inpstep[j] = 0;
                        }
                    }

                    const int8_t* wptr = wbaseptr + (size_t)i * C1Max * K0 * C0;

                    for (int c1 = 0; c1 < cblocks; c1++, wptr += C0 * K0) {
                        #define CONV_INT8_MAC_PAIR(c0) { \
                            __m128i wraw = _mm_loadu_si128((const __m128i*)(wptr + (c0) * K0)); \
                            __m128i w_intlv = _mm_unpacklo_epi8(wraw, _mm_srli_si128(wraw, 8)); \
                            __m256i w16 = _mm256_cvtepi8_epi16(w_intlv); \
                            __m256i x0 = _mm256_set1_epi32((uint16_t)(int16_t)(int8_t)inptr[0][(c0)] | ((uint32_t)(uint16_t)(int16_t)(int8_t)inptr[0][(c0)+1] << 16)); \
                            __m256i x1 = _mm256_set1_epi32((uint16_t)(int16_t)(int8_t)inptr[1][(c0)] | ((uint32_t)(uint16_t)(int16_t)(int8_t)inptr[1][(c0)+1] << 16)); \
                            __m256i x2 = _mm256_set1_epi32((uint16_t)(int16_t)(int8_t)inptr[2][(c0)] | ((uint32_t)(uint16_t)(int16_t)(int8_t)inptr[2][(c0)+1] << 16)); \
                            __m256i x3 = _mm256_set1_epi32((uint16_t)(int16_t)(int8_t)inptr[3][(c0)] | ((uint32_t)(uint16_t)(int16_t)(int8_t)inptr[3][(c0)+1] << 16)); \
                            __m256i x4 = _mm256_set1_epi32((uint16_t)(int16_t)(int8_t)inptr[4][(c0)] | ((uint32_t)(uint16_t)(int16_t)(int8_t)inptr[4][(c0)+1] << 16)); \
                            __m256i x5 = _mm256_set1_epi32((uint16_t)(int16_t)(int8_t)inptr[5][(c0)] | ((uint32_t)(uint16_t)(int16_t)(int8_t)inptr[5][(c0)+1] << 16)); \
                            __m256i x6 = _mm256_set1_epi32((uint16_t)(int16_t)(int8_t)inptr[6][(c0)] | ((uint32_t)(uint16_t)(int16_t)(int8_t)inptr[6][(c0)+1] << 16)); \
                            __m256i x7 = _mm256_set1_epi32((uint16_t)(int16_t)(int8_t)inptr[7][(c0)] | ((uint32_t)(uint16_t)(int16_t)(int8_t)inptr[7][(c0)+1] << 16)); \
                            s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(x0, w16)); \
                            s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(x1, w16)); \
                            s2 = _mm256_add_epi32(s2, _mm256_madd_epi16(x2, w16)); \
                            s3 = _mm256_add_epi32(s3, _mm256_madd_epi16(x3, w16)); \
                            s4 = _mm256_add_epi32(s4, _mm256_madd_epi16(x4, w16)); \
                            s5 = _mm256_add_epi32(s5, _mm256_madd_epi16(x5, w16)); \
                            s6 = _mm256_add_epi32(s6, _mm256_madd_epi16(x6, w16)); \
                            s7 = _mm256_add_epi32(s7, _mm256_madd_epi16(x7, w16)); \
                        }
                        CONV_INT8_MAC_PAIR(0);
                        CONV_INT8_MAC_PAIR(2);
                        CONV_INT8_MAC_PAIR(4);
                        CONV_INT8_MAC_PAIR(6);
                        #undef CONV_INT8_MAC_PAIR

                        for (int j = 0; j < SPAT_BLOCK_SIZE; j++)
                            inptr[j] += inpstep[j];
                    }
                }

                // Store accumulators
                _mm256_store_si256((__m256i*)(sumbuf + 0*K0), s0);
                _mm256_store_si256((__m256i*)(sumbuf + 1*K0), s1);
                _mm256_store_si256((__m256i*)(sumbuf + 2*K0), s2);
                _mm256_store_si256((__m256i*)(sumbuf + 3*K0), s3);
                _mm256_store_si256((__m256i*)(sumbuf + 4*K0), s4);
                _mm256_store_si256((__m256i*)(sumbuf + 5*K0), s5);
                _mm256_store_si256((__m256i*)(sumbuf + 6*K0), s6);
                _mm256_store_si256((__m256i*)(sumbuf + 7*K0), s7);

                __m256 vmult = _mm256_load_ps(multbuf);
                __m256 vzp = _mm256_set1_ps((float)out_zp);

                for (int j = 0; j < SPAT_BLOCK_SIZE; j++) {
                    __m256i acc = _mm256_load_si256((const __m256i*)(sumbuf + j * K0));
                    __m256 facc = _mm256_cvtepi32_ps(acc);
                    facc = _mm256_add_ps(_mm256_mul_ps(facc, vmult), vzp);

                    if (resptr) {
                        const int8_t* rptr = resptr + (p + j) * K0;
                        __m256i res32 = _mm256_cvtepi8_epi32(
                            _mm_loadl_epi64((const __m128i*)rptr));
                        facc = _mm256_add_ps(facc,
                            _mm256_sub_ps(_mm256_cvtepi32_ps(res32), vzp));
                    }

                    __m256i ival = _mm256_cvtps_epi32(facc);
                    __m128i lo = _mm256_castsi256_si128(ival);
                    __m128i hi = _mm256_extracti128_si256(ival, 1);
                    __m128i packed16 = _mm_packs_epi32(lo, hi);
                    __m128i packed8 = _mm_packs_epi16(packed16, packed16);

                    int8_t* optr = outptr + (p + j) * K0;
                    if (activLUT) {
                        alignas(16) int8_t tmp8[16];
                        _mm_store_si128((__m128i*)tmp8, packed8);
                        for (int k = 0; k < K0; k++)
                            optr[k] = (int8_t)activLUT[(int)tmp8[k] + 128];
                    } else {
                        _mm_storel_epi64((__m128i*)optr, packed8);
                    }
                }
            }

            for (; p < planeblocks_l; p++)
        #else
            for (int p = 0; p < planeblocks_l; p++)
        #endif
            {
                alignas(32) int32_t acc[K0];
                memcpy(acc, biasbuf, K0 * sizeof(int32_t));

                int zj = p / (H_l * W_l);
                int yxj = p - zj * H_l * W_l;
                int yj = yxj / W_l;
                int xj = yxj - yj * W_l;
                int zi_base = zj * Sz - padZ;
                int yi_base = yj * Sy - padY;
                int xi_base = xj * Sx - padX;

                for (int i = 0; i < ksize_l; i++) {
                    int zi = zi_base + ofsZYX[i * 3];
                    int yi = yi_base + ofsZYX[i * 3 + 1];
                    int xi = xi_base + ofsZYX[i * 3 + 2];

                    if ((((unsigned)zi >= (unsigned)Di_l) |
                         ((unsigned)yi >= (unsigned)Hi_l) |
                         ((unsigned)xi >= (unsigned)Wi_l)) != 0)
                        continue;

                    const int8_t* inptr = inpbaseptr + (((zi * Hi_l) + yi) * Wi_l + xi) * C0;
                    const int8_t* wptr = wbaseptr + (size_t)i * C1Max * K0 * C0;

                    for (int c1 = 0; c1 < cblocks; c1++, inptr += iplanesize_l, wptr += C0 * K0) {
                        for (int c0 = 0; c0 < C0; c0++) {
                            int ival = (int)inptr[c0];
                            const int8_t* wp = wptr + c0 * K0;
                            for (int k0 = 0; k0 < K0; k0++)
                                acc[k0] += ival * (int)wp[k0];
                        }
                    }
                }

                int8_t* optr = outptr + p * K0;
                const int8_t* rptr = resptr ? resptr + p * K0 : nullptr;
                for (int k0 = 0; k0 < K0; k0++) {
                    float val = (float)acc[k0] * multbuf[k0] + (float)out_zp;
                    if (rptr)
                        val += (float)((int)rptr[k0] - out_zp);
                    int ival = cvRound(val);
                    ival = std::max(-128, std::min(127, ival));
                    if (activLUT)
                        ival = (int)activLUT[ival + 128];
                    optr[k0] = (int8_t)ival;
                }
            }
        }
    });
}

class Conv2Int8LayerImpl CV_FINAL : public Conv2Int8Layer
{
public:
    Mat weights;       // repacked int8 weights in block format
    Mat weightsVNNI;   // VNNI-transposed weights (pre-computed)
    Mat biasInt32;     // int32 fused bias
    Mat biasVNNI;      // int32 VNNI-adjusted bias (pre-computed)
    Mat outMultiplier; // float32 per-channel output multiplier
    MatShape wshape0;  // original weight shape (K x Cg x kH x kW)
    MatShape prevInpshape;
    ConvState cs;

    Mat activationLUT;
    Ptr<ActivationLayerInt8> activ;

    bool addResidual;

    Conv2Int8LayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        auto_pad = getAutoPadding(params);
        ceil_mode = params.get<bool>("ceil_mode", false);
        strides = params.getVector<int>("stride");
        dilations = params.getVector<int>("dilation");
        pads = params.getVector<int>("pad");
        ngroups = params.get<int>("group", 1);

        input_sc = params.get<float>("input_scale", 1.f);
        input_zp = params.get<int>("input_zeropoint", 0);
        output_sc = params.get<float>("scales", 1.f);
        output_zp = params.get<int>("zeropoints", 0);
        per_channel = params.get<bool>("per_channel", true);

        addResidual = false;

        if (!blobs.empty()) {
            CV_Assert(blobs.size() >= 3);
            wshape0 = blobs[0].shape();
            biasInt32 = blobs[1];
            outMultiplier = blobs[2];
        }
    }

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs, const int,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& internals) const CV_OVERRIDE
    {
        // Input can be CV_8SC1 or CV_8UC1; output matches input type
        int outtype = !inputs.empty() ? inputs[0] : CV_8SC1;
        outputs.assign(requiredOutputs, outtype);
        internals.clear();
    }

    bool getMemoryShapes(const std::vector<MatShape>& inpshapes,
                         const int,
                         std::vector<MatShape>& outshapes,
                         std::vector<MatShape>& tempshapes) const CV_OVERRIDE
    {
        size_t ninputs = inpshapes.size();
        CV_Assert(ninputs >= 1);

        std::vector<int> emptyKernelShape;
        outshapes.assign(1, convInferShape(inpshapes[0], wshape0, emptyKernelShape,
                                           ngroups, strides, dilations,
                                           pads, auto_pad, ceil_mode));
        tempshapes.clear();
        return true;
    }

    int getLayouts(const std::vector<DataLayout>& actualInputs,
                   std::vector<DataLayout>& desiredInputs,
                   const int requiredOutputs,
                   std::vector<DataLayout>& outputs) const CV_OVERRIDE
    {
        size_t ninputs = actualInputs.size();
        CV_Assert(ninputs >= 1u && requiredOutputs == 1u);
        desiredInputs = actualInputs;
        desiredInputs[0] = DATA_LAYOUT_BLOCK;
        for (size_t i = 1; i < ninputs; i++)
            desiredInputs[i] = DATA_LAYOUT_UNKNOWN;
        outputs.assign(requiredOutputs, DATA_LAYOUT_BLOCK);
        return getNetImpl(this)->defaultC0;
    }

    void setWeightsInt8(const Mat& w_q, const Mat& bias, const Mat& multiplier, int C0)
    {
        CV_Assert(w_q.type() == CV_8SC1);
        wshape0 = w_q.shape();
        biasInt32 = bias;
        outMultiplier = multiplier;

        repackConvWeightsInt8(w_q, weights, ngroups, C0);
    }

    bool fuseAddResidual(Arg /*residual*/)
    {
        addResidual = true;
        return true;
    }

    bool setActivation(const Ptr<ActivationLayer>& layer) CV_OVERRIDE
    {
        Ptr<ActivationLayerInt8> activ_int8 = layer.dynamicCast<ActivationLayerInt8>();
        if (!activ_int8.empty()) {
            activ = activ_int8;
            if (!activ_int8->blobs.empty())
                activ_int8->blobs[0].convertTo(activationLUT, CV_8S);
            return true;
        }
        return false;
    }

    void forward(InputArrayOfArrays input_arrs,
                 OutputArrayOfArrays output_arrs,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        int ninputs = (int)input_arrs.total();
        CV_Assert(ninputs >= 1);

        const Mat& inp = input_arrs.getMat(0);
        MatShape inpshape = inp.shape();
        CV_Assert(inpshape.layout == DATA_LAYOUT_BLOCK);
        CV_Assert(inp.type() == CV_8SC1 || inp.type() == CV_8UC1);

        int C0 = inpshape.back();

        if (weights.empty() && !blobs.empty()) {
            repackConvWeightsInt8(blobs[0], weights, ngroups, C0);
        }

        if (weightsVNNI.empty() && !weights.empty()) {
            int K = wshape0[0];
            int Cg = wshape0[1];
            int Kg = K / ngroups;
            repackWeightsForVNNI(weights, ngroups, Kg, Cg,
                                  biasInt32, weightsVNNI, biasVNNI);
        }

        Mat residual;
        const void* resptr = nullptr;
        if (addResidual) {
            residual = input_arrs.getMat(ninputs - 1);
            resptr = residual.data;
            ninputs--;
        }

        std::vector<int> emptyKernelShape;
        MatShape outshape = convInferShape(inpshape, wshape0, emptyKernelShape,
                                           ngroups, strides, dilations,
                                           pads, auto_pad, ceil_mode);
        int outtype = inp.type();

        int outkind = output_arrs.kind();
        Mat out;

        if (outkind == _InputArray::STD_VECTOR_MAT) {
            std::vector<Mat>& outs = output_arrs.getMatVecRef();
            outs.resize(1);
            outs[0].fit(outshape, outtype);
            out = outs[0];
        } else {
            std::vector<UMat>& outs = output_arrs.getUMatVecRef();
            outs.resize(1);
            outs[0].fit(outshape, outtype);
            out.fit(outshape, outtype);
        }

        if (inpshape != prevInpshape) {
            cs.initConv(inpshape, wshape0, outshape, ngroups,
                        strides, dilations, pads, auto_pad, ceil_mode,
                        FAST_ACTIV_NONE, {});
            prevInpshape = inpshape;
        }

        const int8_t* lutptr = !activationLUT.empty() ? activationLUT.ptr<int8_t>() : nullptr;

        convInt8Block(inp.data, resptr, out.data, cs,
                      weights.data,
                      weightsVNNI.empty() ? nullptr : weightsVNNI.data,
                      biasInt32.ptr<int>(),
                      biasVNNI.empty() ? nullptr : biasVNNI.ptr<int>(),
                      outMultiplier.ptr<float>(),
                      input_zp, output_zp,
                      lutptr);

        if (outkind == _InputArray::STD_VECTOR_UMAT) {
            std::vector<UMat>& outs = output_arrs.getUMatVecRef();
            out.copyTo(outs[0]);
        }
    }
};

Ptr<Conv2Int8Layer> Conv2Int8Layer::create(const LayerParams& params)
{
    return Ptr<Conv2Int8Layer>(new Conv2Int8LayerImpl(params));
}

} // namespace dnn
} // namespace cv
