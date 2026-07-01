// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "opencv2/core/hal/intrin.hpp"
#include "../layers/conv2_common.hpp"
#if CV_AVX2 && (defined(__x86_64__) || defined(_M_X64))
#include <immintrin.h>
#endif

#if !defined(CV_AVXVNNI_AVAILABLE)
#if (CV_TRY_AVX2 || CV_AVX2) && \
    ((defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 11) || \
     (defined(__clang__) && !defined(__apple_build_version__) && __clang_major__ >= 12))
#define CV_AVXVNNI_AVAILABLE 1
#else
#define CV_AVXVNNI_AVAILABLE 0
#endif
#endif

namespace cv {
namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void convInt8Block(const void* inp_, const void* residual_,
                   void* out_, const ConvState& cs,
                   const void* weights_,
                   const void* weightsVNNI_,
                   const int* bias, const int* biasVNNI_,
                   const float* multiplier,
                   int inp_zp, int out_zp,
                   const int8_t* activLUT,
                   bool inputIsU8);

#if !defined(CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY)

#if CV_AVXVNNI_AVAILABLE
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#ifdef __clang__
#pragma clang attribute push(__attribute__((target("avx2,avxvnni"))), apply_to = function)
#else
#pragma GCC push_options
#pragma GCC target("avxvnni")
#endif
static void convInt8BlockVNNI(const void* inp_, const void* residual_,
                              void* out_, const ConvState& cs,
                              const void* weightsVNNI_,
                              const int* biasVNNI, const float* multiplier,
                              int inp_zp, int out_zp,
                              const int8_t* activLUT,
                              bool inputIsU8)
{
    constexpr int C0 = 8, K0 = 8;
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

    int nkb = N * ngroups * Kblk;
    int nthreads = std::max(1, cv::getNumThreads());
    int min_ntiles = std::max(1, (nthreads * 4 + nkb - 1) / nkb);
    int TILE = std::max(SPAT_BLOCK_SIZE,
        ((planeblocks_ + min_ntiles - 1) / min_ntiles + SPAT_BLOCK_SIZE - 1) & ~(SPAT_BLOCK_SIZE - 1));
    int ntiles = (planeblocks_ + TILE - 1) / TILE;
    int total_tasks = nkb * ntiles;
    const __m256i v_xor = inputIsU8 ? _mm256_setzero_si256() : _mm256_set1_epi8((char)0x80);

    parallel_for_(Range(0, total_tasks), [&](const Range& range) {
        constexpr int C0 = 8, K0 = 8;
        constexpr int C0shift = 3;
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
        int8_t zbuf[C0];
        memset(zbuf, (uint8_t)inp_zp, C0);

        for (int t = range.start; t < range.end; t++) {
            int tile = t % ntiles;
            int kt   = t / ntiles;
            int n    = kt / (ngroups * Kblk);
            int rem  = kt - n * (ngroups * Kblk);
            int g    = rem / Kblk;
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

            const int k0_off_dw = k_base & (K0-1);
            if (cs.depthwise) {
                biasbuf[k0_off_dw] = biasbuf[0];
                multbuf[k0_off_dw] = multbuf[0];
            }

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

            int p_start = tile * TILE;
            int p_end   = std::min(p_start + TILE, planeblocks_l);
            int p = p_start;
            for (; p < p_end; p += SPAT_BLOCK_SIZE) {
                if (p + SPAT_BLOCK_SIZE > p_end) {
                    if (p == p_start) break;
                    p = p_end - SPAT_BLOCK_SIZE;
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
                        __m256i res32 = inputIsU8 \
                            ? _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(resptr + (p + (j)) * K0))) \
                            : _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(resptr + (p + (j)) * K0))); \
                        facc = _mm256_add_ps(facc, \
                            _mm256_sub_ps(_mm256_cvtepi32_ps(res32), vzp)); \
                    } \
                    __m256i ival = _mm256_cvtps_epi32(facc); \
                    __m128i lo = _mm256_castsi256_si128(ival); \
                    __m128i hi = _mm256_extracti128_si256(ival, 1); \
                    __m128i packed16 = _mm_packs_epi32(lo, hi); \
                    __m128i packed8 = inputIsU8 ? _mm_packus_epi16(packed16, packed16) \
                                                : _mm_packs_epi16(packed16, packed16); \
                    int8_t* optr = outptr + (p + (j)) * K0; \
                    alignas(16) int8_t tmp8[16]; \
                    _mm_store_si128((__m128i*)tmp8, packed8); \
                    if (!cs.depthwise) { \
                        if (activLUT) { \
                            for (int k = 0; k < K0; k++) { \
                                int lut_idx = inputIsU8 ? (int)(uint8_t)tmp8[k] : ((int)tmp8[k] + 128); \
                                optr[k] = (int8_t)activLUT[lut_idx]; \
                            } \
                        } else { \
                            _mm_storel_epi64((__m128i*)optr, packed8); \
                        } \
                    } else { \
                        if (activLUT) { \
                            int lut_idx = inputIsU8 ? (int)(uint8_t)tmp8[k0_off_dw] : ((int)tmp8[k0_off_dw] + 128); \
                            optr[k0_off_dw] = (int8_t)activLUT[lut_idx]; \
                        } else { \
                            optr[k0_off_dw] = tmp8[k0_off_dw]; \
                        } \
                    } \
                }
                CONV_VNNI_STORE(0); CONV_VNNI_STORE(1);
                CONV_VNNI_STORE(2); CONV_VNNI_STORE(3);
                CONV_VNNI_STORE(4); CONV_VNNI_STORE(5);
                CONV_VNNI_STORE(6); CONV_VNNI_STORE(7);
                #undef CONV_VNNI_STORE
            }

            // Scalar tail
            for (; p < p_end; p++) {
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
                            int ival = inputIsU8 ? (int)(uint8_t)inptr[c0]
                                                 : (int)(uint8_t)((uint8_t)inptr[c0] ^ 0x80u);
                            // VNNI-repacked weights: weight(c0,k0) = wptr[k0*4 + c0] (c0<4)
                            // else wptr[32 + k0*4 + (c0-4)]
                            if (c0 < 4) {
                                for (int k0 = 0; k0 < K0; k0++)
                                    acc[k0] += ival * (int)wptr[k0 * 4 + c0];
                            } else {
                                for (int k0 = 0; k0 < K0; k0++)
                                    acc[k0] += ival * (int)wptr[32 + k0 * 4 + (c0 - 4)];
                            }
                        }
                    }
                }
                int8_t* optr = outptr + p * K0;
                const int8_t* rptr = resptr ? resptr + p * K0 : nullptr;
                int k0_lo = cs.depthwise ? k0_off_dw : 0;
                int k0_hi = cs.depthwise ? k0_off_dw + 1 : K0;
                for (int k0 = k0_lo; k0 < k0_hi; k0++) {
                    float val = (float)acc[k0] * multbuf[k0] + (float)out_zp;
                    if (rptr) val += (float)((int)rptr[k0] - out_zp);
                    int ival = cvRound(val);
                    if (inputIsU8) {
                        ival = std::max(0, std::min(255, ival));
                        if (activLUT) ival = (int)(uint8_t)activLUT[ival];
                        ((uint8_t*)optr)[k0] = (uint8_t)ival;
                    } else {
                        ival = std::max(-128, std::min(127, ival));
                        if (activLUT) ival = (int)activLUT[ival + 128];
                        optr[k0] = (int8_t)ival;
                    }
                }
            }
        }
    });
}

static void convInt8Block1x1(const void* inp_, const void* residual_,
                              void* out_, const ConvState& cs,
                              const void* weightsVNNI_,
                              const int* biasVNNI, const float* multiplier,
                              int inp_zp, int out_zp,
                              const int8_t* activLUT,
                              bool inputIsU8)
{
    constexpr int C0 = 8, K0 = 8;
    constexpr int SPAT_BLOCK_SIZE = 8;

    const MatShape& inpshape = cs.inpshape;
    const MatShape& outshape = cs.outshape;

    int N     = outshape[0];
    int ndims = outshape.dims;
    int K     = outshape.channels();
    int C     = inpshape.channels();
    int K1    = (K + K0 - 1) / K0;
    int C1    = (C + C0 - 1) / C0;

    int D_  = ndims >= 6 ? outshape[ndims-4] : 1;
    int H_  = ndims >= 5 ? outshape[ndims-3] : 1;
    int W_  = outshape[ndims-2];
    int P   = D_ * H_ * W_;

    int Di_ = ndims >= 6 ? inpshape[ndims-4] : 1;
    int Hi_ = ndims >= 5 ? inpshape[ndims-3] : 1;
    int Wi_ = inpshape[ndims-2];
    int Pi  = Di_ * Hi_ * Wi_;

    int ngroups = cs.ngroups;
    int Kg    = K / ngroups;
    int Cg    = C / ngroups;
    int Kblk  = cs.wshape[1];
    int C1Max = cs.wshape[3];

    const int8_t* inp      = (const int8_t*)inp_;
    const int8_t* residual = (const int8_t*)residual_;
    int8_t*       out      = (int8_t*)out_;
    const int8_t* wdata    = (const int8_t*)weightsVNNI_;

    int iplanesize = Pi * C0;
    int planesize  = P * K0;

    if ((Kg % K0) != 0)
        memset(out, 0, outshape.total());

    constexpr int SB_TILE = 8;
    int nthreads    = std::max(1, cv::getNumThreads());
    int nkb         = N * ngroups * Kblk;
    int min_ntiles  = std::max(1, (nthreads * 4 + nkb - 1) / nkb);
    int TILE        = std::max(SB_TILE,
        ((P + min_ntiles - 1) / min_ntiles + SB_TILE - 1) & ~(SB_TILE - 1));
    int ntiles      = (P + TILE - 1) / TILE;
    int total_tasks = nkb * ntiles;

    const __m256i v_xor = inputIsU8 ? _mm256_setzero_si256() : _mm256_set1_epi8((char)0x80);

    parallel_for_(Range(0, total_tasks), [&](const Range& range) {
        constexpr int C0 = 8, K0 = 8;
        constexpr int C0shift = 3;

        for (int t = range.start; t < range.end; t++) {
            int tile = t % ntiles;
            int kt   = t / ntiles;
            int n    = kt / (ngroups * Kblk);
            int rem  = kt - n * (ngroups * Kblk);
            int g    = rem / Kblk;
            int kblk = rem - g * Kblk;

            int k_base = g * Kg + kblk * K0;
            if (k_base >= K) continue;

            int c_start  = g * Cg;
            int c1_start = c_start >> C0shift;
            int c00      = c_start & (C0 - 1);
            int cblocks  = (c00 + Cg + C0 - 1) >> C0shift;

            const int8_t* inpbaseptr = inp + (size_t)(n * C1 + c1_start) * iplanesize;
            const int8_t* wbaseptr   = wdata + (size_t)(g * Kblk + kblk) * C1Max * C0 * K0;

            int k1 = k_base >> C0shift;
            int8_t*       outptr = out      + (size_t)(n * K1 + k1) * planesize;
            const int8_t* resptr = residual ? residual + (size_t)(n * K1 + k1) * planesize : nullptr;

            alignas(32) int32_t biasbuf[K0];
            alignas(32) float   multbuf[K0];
            memcpy(biasbuf, biasVNNI   + k_base, K0 * sizeof(int32_t));
            memcpy(multbuf, multiplier + k_base, K0 * sizeof(float));

            int p_start = tile * TILE;
            int p_end   = std::min(p_start + TILE, P);
            int p = p_start;

            for (; p + SPAT_BLOCK_SIZE <= p_end; p += SPAT_BLOCK_SIZE) {
                __m256i vbias = _mm256_load_si256((const __m256i*)biasbuf);
                __m256i s0 = vbias, s1 = vbias, s2 = vbias, s3 = vbias;
                __m256i s4 = vbias, s5 = vbias, s6 = vbias, s7 = vbias;

                const int8_t* inptr[SPAT_BLOCK_SIZE];
                for (int j = 0; j < SPAT_BLOCK_SIZE; j++)
                    inptr[j] = inpbaseptr + (p + j) * C0;

                const int8_t* wptr = wbaseptr;
                for (int c1 = 0; c1 < cblocks; c1++, wptr += C0 * K0) {
                    __m256i wg0 = _mm256_load_si256((const __m256i*)(wptr));
                    __m256i wg1 = _mm256_load_si256((const __m256i*)(wptr + 32));

                    #define CONV_1x1_MAC(j) { \
                        __m256i x0 = _mm256_xor_si256(_mm256_set1_epi32(*(const int32_t*)(inptr[(j)]    )), v_xor); \
                        __m256i x1 = _mm256_xor_si256(_mm256_set1_epi32(*(const int32_t*)(inptr[(j)] + 4)), v_xor); \
                        s##j = _mm256_dpbusd_epi32(s##j, x0, wg0); \
                        s##j = _mm256_dpbusd_epi32(s##j, x1, wg1); \
                    }
                    CONV_1x1_MAC(0); CONV_1x1_MAC(1);
                    CONV_1x1_MAC(2); CONV_1x1_MAC(3);
                    CONV_1x1_MAC(4); CONV_1x1_MAC(5);
                    CONV_1x1_MAC(6); CONV_1x1_MAC(7);
                    #undef CONV_1x1_MAC

                    for (int j = 0; j < SPAT_BLOCK_SIZE; j++)
                        inptr[j] += iplanesize;
                }

                __m256 vmult = _mm256_load_ps(multbuf);
                __m256 vzp   = _mm256_set1_ps((float)out_zp);

                #define CONV_1x1_STORE(j) { \
                    __m256 facc = _mm256_add_ps( \
                        _mm256_mul_ps(_mm256_cvtepi32_ps(s##j), vmult), vzp); \
                    if (resptr) { \
                        __m256i res32 = inputIsU8 \
                            ? _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(resptr + (p + (j)) * K0))) \
                            : _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(resptr + (p + (j)) * K0))); \
                        facc = _mm256_add_ps(facc, \
                            _mm256_sub_ps(_mm256_cvtepi32_ps(res32), vzp)); \
                    } \
                    __m256i ival = _mm256_cvtps_epi32(facc); \
                    __m128i lo = _mm256_castsi256_si128(ival); \
                    __m128i hi = _mm256_extracti128_si256(ival, 1); \
                    __m128i packed16 = _mm_packs_epi32(lo, hi); \
                    __m128i packed8  = inputIsU8 ? _mm_packus_epi16(packed16, packed16) \
                                                 : _mm_packs_epi16(packed16, packed16); \
                    int8_t* optr = outptr + (p + (j)) * K0; \
                    if (activLUT) { \
                        alignas(16) int8_t tmp8[16]; \
                        _mm_store_si128((__m128i*)tmp8, packed8); \
                        for (int k = 0; k < K0; k++) { \
                            int lut_idx = inputIsU8 ? (int)(uint8_t)tmp8[k] : ((int)tmp8[k] + 128); \
                            optr[k] = (int8_t)activLUT[lut_idx]; \
                        } \
                    } else { \
                        _mm_storel_epi64((__m128i*)optr, packed8); \
                    } \
                }
                CONV_1x1_STORE(0); CONV_1x1_STORE(1);
                CONV_1x1_STORE(2); CONV_1x1_STORE(3);
                CONV_1x1_STORE(4); CONV_1x1_STORE(5);
                CONV_1x1_STORE(6); CONV_1x1_STORE(7);
                #undef CONV_1x1_STORE
            }

            for (; p < p_end; p++) {
                alignas(32) int32_t acc[K0];
                memcpy(acc, biasbuf, K0 * sizeof(int32_t));

                const int8_t* inptr_s = inpbaseptr + (size_t)p * C0;
                const int8_t* wptr    = wbaseptr;
                for (int c1 = 0; c1 < cblocks; c1++, inptr_s += iplanesize, wptr += C0 * K0) {
                    for (int c0 = 0; c0 < C0; c0++) {
                        int iv = inputIsU8 ? (int)(uint8_t)inptr_s[c0]
                                           : (int)(uint8_t)((uint8_t)inptr_s[c0] ^ 0x80u);
                        const int8_t* wp = wptr + c0 * K0;
                        for (int k0 = 0; k0 < K0; k0++)
                            acc[k0] += iv * (int)wp[k0];
                    }
                }

                int8_t*       optr = outptr + (size_t)p * K0;
                const int8_t* rptr = resptr ? resptr + (size_t)p * K0 : nullptr;
                for (int k0 = 0; k0 < K0; k0++) {
                    float val = (float)acc[k0] * multbuf[k0] + (float)out_zp;
                    if (rptr) {
                        val += inputIsU8 ? (float)(int)(uint8_t)rptr[k0] - (float)out_zp
                                         : (float)(int)rptr[k0] - (float)out_zp;
                    }
                    int iv = cvRound(val);
                    if (inputIsU8) {
                        iv = std::max(0, std::min(255, iv));
                        if (activLUT) iv = (int)(uint8_t)activLUT[iv];
                        ((uint8_t*)optr)[k0] = (uint8_t)iv;
                    } else {
                        iv = std::max(-128, std::min(127, iv));
                        if (activLUT) iv = (int)activLUT[iv + 128];
                        optr[k0] = (int8_t)iv;
                    }
                }
            }
        }
    });
}

#ifdef __clang__
#pragma clang attribute pop
#else
#pragma GCC pop_options
#endif
#endif // CV_AVXVNNI_AVAILABLE

#if CV_NEON && defined(CV_NEON_DOT) && CV_NEON_DOT
#include <arm_neon.h>

#define CONV_NEON_STORE(j) { \
    float32x4_t facc_lo = vfmaq_f32(vzp, vcvtq_f32_s32(s##j##_lo), vmult_lo); \
    float32x4_t facc_hi = vfmaq_f32(vzp, vcvtq_f32_s32(s##j##_hi), vmult_hi); \
    if (resptr) { \
        const int8_t* rp = resptr + (p + (j)) * K0; \
        if (inputIsU8) { \
            uint16x8_t r16u = vmovl_u8(vld1_u8((const uint8_t*)rp)); \
            facc_lo = vaddq_f32(facc_lo, vsubq_f32( \
                vcvtq_f32_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(r16u)))), vzp)); \
            facc_hi = vaddq_f32(facc_hi, vsubq_f32( \
                vcvtq_f32_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(r16u)))), vzp)); \
        } else { \
            int16x8_t r16 = vmovl_s8(vld1_s8(rp)); \
            facc_lo = vaddq_f32(facc_lo, vsubq_f32( \
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(r16))), vzp)); \
            facc_hi = vaddq_f32(facc_hi, vsubq_f32( \
                vcvtq_f32_s32(vmovl_s16(vget_high_s16(r16))), vzp)); \
        } \
    } \
    int32x4_t ival_lo = vcvtnq_s32_f32(facc_lo); \
    int32x4_t ival_hi = vcvtnq_s32_f32(facc_hi); \
    int16x4_t p16_lo = vqmovn_s32(ival_lo); \
    int16x4_t p16_hi = vqmovn_s32(ival_hi); \
    int16x8_t p16 = vcombine_s16(p16_lo, p16_hi); \
    int8_t* optr = outptr + (p + (j)) * K0; \
    if (inputIsU8) { \
        uint8x8_t p8u = vqmovun_s16(p16); \
        if (activLUT) { \
            uint8_t tmp8[8]; \
            vst1_u8(tmp8, p8u); \
            for (int k = 0; k < K0; k++) \
                ((uint8_t*)optr)[k] = (uint8_t)activLUT[tmp8[k]]; \
        } else { \
            vst1_u8((uint8_t*)optr, p8u); \
        } \
    } else { \
        int8x8_t p8 = vqmovn_s16(p16); \
        if (activLUT) { \
            int8_t tmp8[8]; \
            vst1_s8(tmp8, p8); \
            for (int k = 0; k < K0; k++) \
                optr[k] = (int8_t)activLUT[(int)tmp8[k] + 128]; \
        } else { \
            vst1_s8(optr, p8); \
        } \
    } \
}

static void convInt8BlockNEON(const void* inp_, const void* residual_,
                              void* out_, const ConvState& cs,
                              const void* weightsVNNI_,
                              const int* bias, const float* multiplier,
                              int inp_zp, int out_zp,
                              const int8_t* activLUT,
                              bool inputIsU8)
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

    const int8_t* inp = (const int8_t*)inp_;
    const int8_t* residual = (const int8_t*)residual_;
    int8_t* out = (int8_t*)out_;
    const int8_t* wdata = (const int8_t*)weightsVNNI_;
    const int* ofsZYX = cs.coordtab.data();

    size_t outtotal = outshape.total();
    if ((Kg % K0) != 0) memset(out, 0, outtotal);

    const int8x16_t v_xor = inputIsU8 ? vdupq_n_s8((int8_t)0x80) : vdupq_n_s8(0);

    int Sz = cs.strides[0], Sy = cs.strides[1], Sx = cs.strides[2];

    if (ksize_ == 1 && Sx == 1 && Sy == 1 && Sz == 1) {
        int total_spatial = planeblocks_;
        int iplanesize = total_spatial * C0;
        int cblocks_g = (Cg + C0 - 1) / C0;

        int cblk_bytes = C0 * cblocks_g;
        int SPAT_TILE = std::max(SPAT_BLOCK_SIZE,
            ((48 * 1024) / std::max(1, cblk_bytes)) & ~(SPAT_BLOCK_SIZE - 1));
        SPAT_TILE = std::min(SPAT_TILE, total_spatial);
        SPAT_TILE = std::max(SPAT_TILE, SPAT_BLOCK_SIZE);

        int wblk_bytes = C0 * K0 * cblocks_g;
        int KG = std::min(Kblk, std::max(1, (48 * 1024) / std::max(1, wblk_bytes)));

        int ntiles = (total_spatial + SPAT_TILE - 1) / SPAT_TILE;
        int nktiles = (Kblk + KG - 1) / KG;

        while (N * ngroups * ntiles * nktiles < 16 && KG > 1) {
            KG = std::max(1, KG / 2);
            nktiles = (Kblk + KG - 1) / KG;
        }

        int total_tasks = N * ngroups * ntiles * nktiles;

        parallel_for_(Range(0, total_tasks), [&](const Range& range) {
        for (int task = range.start; task < range.end; task++) {
            int temp = task;
            int n = temp / (ngroups * ntiles * nktiles);
            temp -= n * (ngroups * ntiles * nktiles);
            int g = temp / (ntiles * nktiles);
            temp -= g * (ntiles * nktiles);
            int tile_idx = temp / nktiles;
            int kgroup_idx = temp - tile_idx * nktiles;

            int p_start = tile_idx * SPAT_TILE;
            int p_end = std::min(p_start + SPAT_TILE, total_spatial);

            int kblk_start = kgroup_idx * KG;
            int kblk_end = std::min(kblk_start + KG, Kblk);

            int c_start = g * Cg;
            int c1_start = c_start >> C0shift;
            int c00 = c_start & (C0 - 1);
            int cblocks = (c00 + Cg + C0 - 1) >> C0shift;

            const int8_t* inpbase = inp + (size_t)(n * C1 + c1_start) * iplanesize;
            int planesize = total_spatial * K0;

            for (int kblk = kblk_start; kblk < kblk_end; kblk++) {
                int k_base = g * Kg + kblk * K0;
                if (k_base >= K) continue;

                int k1 = k_base >> C0shift;
                int8_t* outptr = out + (size_t)(n * K1 + k1) * planesize;
                const int8_t* resptr = residual ? residual + (size_t)(n * K1 + k1) * planesize : nullptr;
                const int8_t* wbaseptr = wdata + (size_t)(g * Kblk + kblk) * C1Max * C0 * K0;

                int32x4_t vbias_lo = vld1q_s32(bias + k_base);
                int32x4_t vbias_hi = vld1q_s32(bias + k_base + 4);
                float32x4_t vmult_lo = vld1q_f32(multiplier + k_base);
                float32x4_t vmult_hi = vld1q_f32(multiplier + k_base + 4);
                float32x4_t vzp = vdupq_n_f32((float)out_zp);

                int p = p_start;
                for (; p <= p_end - SPAT_BLOCK_SIZE; p += SPAT_BLOCK_SIZE) {
                    int32x4_t s0_lo = vbias_lo, s0_hi = vbias_hi;
                    int32x4_t s1_lo = vbias_lo, s1_hi = vbias_hi;
                    int32x4_t s2_lo = vbias_lo, s2_hi = vbias_hi;
                    int32x4_t s3_lo = vbias_lo, s3_hi = vbias_hi;
                    int32x4_t s4_lo = vbias_lo, s4_hi = vbias_hi;
                    int32x4_t s5_lo = vbias_lo, s5_hi = vbias_hi;
                    int32x4_t s6_lo = vbias_lo, s6_hi = vbias_hi;
                    int32x4_t s7_lo = vbias_lo, s7_hi = vbias_hi;

                    const int8_t* inpptr = inpbase + (size_t)p * C0;
                    const int8_t* wptr = wbaseptr;

                    for (int c1 = 0; c1 < cblocks; c1++, wptr += C0 * K0) {
                        int8x16_t wg0 = vld1q_s8(wptr);
                        int8x16_t wg1 = vld1q_s8(wptr + 16);
                        int8x16_t wg2 = vld1q_s8(wptr + 32);
                        int8x16_t wg3 = vld1q_s8(wptr + 48);

                        const int8_t* ip = inpptr;
                        #define CONV_NEON_MAC_1x1(j) { \
                            int8x16_t x_lo = veorq_s8(vreinterpretq_s8_s32( \
                                vdupq_n_s32(*(const int32_t*)&ip[0])), v_xor); \
                            int8x16_t x_hi = veorq_s8(vreinterpretq_s8_s32( \
                                vdupq_n_s32(*(const int32_t*)&ip[4])), v_xor); \
                            s##j##_lo = vdotq_s32(s##j##_lo, x_lo, wg0); \
                            s##j##_hi = vdotq_s32(s##j##_hi, x_lo, wg1); \
                            s##j##_lo = vdotq_s32(s##j##_lo, x_hi, wg2); \
                            s##j##_hi = vdotq_s32(s##j##_hi, x_hi, wg3); \
                            ip += C0; \
                        }
                        CONV_NEON_MAC_1x1(0); CONV_NEON_MAC_1x1(1);
                        CONV_NEON_MAC_1x1(2); CONV_NEON_MAC_1x1(3);
                        CONV_NEON_MAC_1x1(4); CONV_NEON_MAC_1x1(5);
                        CONV_NEON_MAC_1x1(6); CONV_NEON_MAC_1x1(7);
                        #undef CONV_NEON_MAC_1x1

                        inpptr += iplanesize;
                    }

                    CONV_NEON_STORE(0); CONV_NEON_STORE(1);
                    CONV_NEON_STORE(2); CONV_NEON_STORE(3);
                    CONV_NEON_STORE(4); CONV_NEON_STORE(5);
                    CONV_NEON_STORE(6); CONV_NEON_STORE(7);
                }

                if (p < p_end && p_end >= SPAT_BLOCK_SIZE) {
                    p = p_end - SPAT_BLOCK_SIZE;
                    int32x4_t s0_lo = vbias_lo, s0_hi = vbias_hi;
                    int32x4_t s1_lo = vbias_lo, s1_hi = vbias_hi;
                    int32x4_t s2_lo = vbias_lo, s2_hi = vbias_hi;
                    int32x4_t s3_lo = vbias_lo, s3_hi = vbias_hi;
                    int32x4_t s4_lo = vbias_lo, s4_hi = vbias_hi;
                    int32x4_t s5_lo = vbias_lo, s5_hi = vbias_hi;
                    int32x4_t s6_lo = vbias_lo, s6_hi = vbias_hi;
                    int32x4_t s7_lo = vbias_lo, s7_hi = vbias_hi;

                    const int8_t* inpptr = inpbase + (size_t)p * C0;
                    const int8_t* wptr = wbaseptr;
                    for (int c1 = 0; c1 < cblocks; c1++, wptr += C0 * K0) {
                        int8x16_t wg0 = vld1q_s8(wptr);
                        int8x16_t wg1 = vld1q_s8(wptr + 16);
                        int8x16_t wg2 = vld1q_s8(wptr + 32);
                        int8x16_t wg3 = vld1q_s8(wptr + 48);
                        const int8_t* ip = inpptr;
                        #define CONV_NEON_MAC_T(j) { \
                            int8x16_t x_lo = veorq_s8(vreinterpretq_s8_s32( \
                                vdupq_n_s32(*(const int32_t*)&ip[0])), v_xor); \
                            int8x16_t x_hi = veorq_s8(vreinterpretq_s8_s32( \
                                vdupq_n_s32(*(const int32_t*)&ip[4])), v_xor); \
                            s##j##_lo = vdotq_s32(s##j##_lo, x_lo, wg0); \
                            s##j##_hi = vdotq_s32(s##j##_hi, x_lo, wg1); \
                            s##j##_lo = vdotq_s32(s##j##_lo, x_hi, wg2); \
                            s##j##_hi = vdotq_s32(s##j##_hi, x_hi, wg3); \
                            ip += C0; \
                        }
                        CONV_NEON_MAC_T(0); CONV_NEON_MAC_T(1);
                        CONV_NEON_MAC_T(2); CONV_NEON_MAC_T(3);
                        CONV_NEON_MAC_T(4); CONV_NEON_MAC_T(5);
                        CONV_NEON_MAC_T(6); CONV_NEON_MAC_T(7);
                        #undef CONV_NEON_MAC_T
                        inpptr += iplanesize;
                    }
                    CONV_NEON_STORE(0); CONV_NEON_STORE(1);
                    CONV_NEON_STORE(2); CONV_NEON_STORE(3);
                    CONV_NEON_STORE(4); CONV_NEON_STORE(5);
                    CONV_NEON_STORE(6); CONV_NEON_STORE(7);
                }
            }
        }
        });
        return;
    }

    int innerZ0 = cs.inner[0], innerZ1 = cs.inner[MAX_CONV_DIMS];
    int innerY0 = cs.inner[1], innerY1 = cs.inner[MAX_CONV_DIMS+1];
    int innerX0 = cs.inner[2], innerX1 = cs.inner[MAX_CONV_DIMS+2];

    int total_blocks = N * ngroups * Kblk;

    parallel_for_(Range(0, total_blocks), [&](const Range& range) {
        int H = H_, W = W_;
        int Di = ndims >= 6 ? inpshape[ndims-4] : 1;
        int Hi = ndims >= 5 ? inpshape[ndims-3] : 1;
        int Wi = inpshape[ndims-2];
        int planeblocks = planeblocks_;
        int iplanesize = Di * Hi * Wi * C0;
        int planesize = planeblocks * K0;

        int padZ = cs.pads[0], padY = cs.pads[1], padX = cs.pads[2];
        int ksize = ksize_;
        int8_t zbuf[C0];
        memset(zbuf, (uint8_t)inp_zp, C0);

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

            int32x4_t vbias_lo = vld1q_s32(bias + k_base);
            int32x4_t vbias_hi = vld1q_s32(bias + k_base + 4);
            float32x4_t vmult_lo = vld1q_f32(multiplier + k_base);
            float32x4_t vmult_hi = vld1q_f32(multiplier + k_base + 4);
            float32x4_t vzp = vdupq_n_f32((float)out_zp);

            int H_l = H, W_l = W;
            int Di_l = Di, Hi_l = Hi, Wi_l = Wi;
            int iplanesize_l = iplanesize;
            int planeblocks_l = planeblocks;
            int ksize_l = ksize;

            int p = 0;
            for (; p < planeblocks_l; p += SPAT_BLOCK_SIZE) {
                if (p + SPAT_BLOCK_SIZE > planeblocks_l) {
                    if (p == 0) break;
                    p = planeblocks_l - SPAT_BLOCK_SIZE;
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

                int32x4_t s0_lo = vbias_lo, s0_hi = vbias_hi;
                int32x4_t s1_lo = vbias_lo, s1_hi = vbias_hi;
                int32x4_t s2_lo = vbias_lo, s2_hi = vbias_hi;
                int32x4_t s3_lo = vbias_lo, s3_hi = vbias_hi;
                int32x4_t s4_lo = vbias_lo, s4_hi = vbias_hi;
                int32x4_t s5_lo = vbias_lo, s5_hi = vbias_hi;
                int32x4_t s6_lo = vbias_lo, s6_hi = vbias_hi;
                int32x4_t s7_lo = vbias_lo, s7_hi = vbias_hi;

                #define CONV_NEON_MAC(j) { \
                    int8x16_t x_lo = veorq_s8(vreinterpretq_s8_s32( \
                        vdupq_n_s32(*(const int32_t*)&inptr[(j)][0])), v_xor); \
                    int8x16_t x_hi = veorq_s8(vreinterpretq_s8_s32( \
                        vdupq_n_s32(*(const int32_t*)&inptr[(j)][4])), v_xor); \
                    s##j##_lo = vdotq_s32(s##j##_lo, x_lo, wg0); \
                    s##j##_hi = vdotq_s32(s##j##_hi, x_lo, wg1); \
                    s##j##_lo = vdotq_s32(s##j##_lo, x_hi, wg2); \
                    s##j##_hi = vdotq_s32(s##j##_hi, x_hi, wg3); \
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
                            int8x16_t wg0 = vld1q_s8(wptr);
                            int8x16_t wg1 = vld1q_s8(wptr + 16);
                            int8x16_t wg2 = vld1q_s8(wptr + 32);
                            int8x16_t wg3 = vld1q_s8(wptr + 48);

                            CONV_NEON_MAC(0); CONV_NEON_MAC(1);
                            CONV_NEON_MAC(2); CONV_NEON_MAC(3);
                            CONV_NEON_MAC(4); CONV_NEON_MAC(5);
                            CONV_NEON_MAC(6); CONV_NEON_MAC(7);

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
                            int8x16_t wg0 = vld1q_s8(wptr);
                            int8x16_t wg1 = vld1q_s8(wptr + 16);
                            int8x16_t wg2 = vld1q_s8(wptr + 32);
                            int8x16_t wg3 = vld1q_s8(wptr + 48);

                            CONV_NEON_MAC(0); CONV_NEON_MAC(1);
                            CONV_NEON_MAC(2); CONV_NEON_MAC(3);
                            CONV_NEON_MAC(4); CONV_NEON_MAC(5);
                            CONV_NEON_MAC(6); CONV_NEON_MAC(7);

                            for (int j = 0; j < SPAT_BLOCK_SIZE; j++)
                                inptr[j] += inpstep[j];
                        }
                    }
                }
                #undef CONV_NEON_MAC

                CONV_NEON_STORE(0); CONV_NEON_STORE(1);
                CONV_NEON_STORE(2); CONV_NEON_STORE(3);
                CONV_NEON_STORE(4); CONV_NEON_STORE(5);
                CONV_NEON_STORE(6); CONV_NEON_STORE(7);
            }

            for (; p < planeblocks_l; p++) {
                alignas(16) int32_t acc[K0];
                memcpy(acc, bias + k_base, K0 * sizeof(int32_t));

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
                    const int8_t* wptr_vnni = wbaseptr + (size_t)i * C1Max * K0 * C0;
                    for (int c1i = 0; c1i < cblocks; c1i++, inptr += iplanesize_l) {
                        const int8_t* wp = wptr_vnni + (size_t)c1i * C0 * K0;
                        for (int c0 = 0; c0 < C0; c0++) {
                            int ival = inputIsU8 ? ((int)(uint8_t)inptr[c0] - 128)
                                                 : (int)inptr[c0];
                            int base = (c0 < 4) ? 0 : 32;
                            int c_in_group = c0 & 3;
                            for (int k0 = 0; k0 < K0; k0++)
                                acc[k0] += ival * (int)wp[base + k0 * 4 + c_in_group];
                        }
                    }
                }
                int8_t* optr = outptr + p * K0;
                const int8_t* rptr = resptr ? resptr + p * K0 : nullptr;
                for (int k0 = 0; k0 < K0; k0++) {
                    float val = (float)acc[k0] * multiplier[k_base + k0] + (float)out_zp;
                    if (rptr) {
                        int rval = inputIsU8 ? (int)(uint8_t)rptr[k0] : (int)rptr[k0];
                        val += (float)(rval - out_zp);
                    }
                    int ival = cvRound(val);
                    if (inputIsU8) {
                        ival = std::max(0, std::min(255, ival));
                        if (activLUT) ival = (int)(uint8_t)activLUT[ival];
                        ((uint8_t*)optr)[k0] = (uint8_t)ival;
                    } else {
                        ival = std::max(-128, std::min(127, ival));
                        if (activLUT) ival = (int)activLUT[ival + 128];
                        optr[k0] = (int8_t)ival;
                    }
                }
            }
        }
    });
}
#undef CONV_NEON_STORE
#endif // CV_NEON && CV_NEON_DOT

#if CV_RVV
#include <riscv_vector.h>

// The int8 conv kernel below is instantiated for two register layouts via
// these traits. Both variants keep i32 accumulators over the K0=8 output
// channels (vl=8) and use i16 widening multiply-accumulate (vwmacc):
//  - M1: accumulators in LMUL=1; needs vlmax_e32m1 >= 8, i.e. VLEN >= 256
//    (checked at runtime; e.g. SpacemiT K1).
//  - M2: accumulators in LMUL=2; works on any RVV 1.0 core with VLEN >= 128.
struct ConvInt8RVV_M1
{
    typedef vint32m1_t vi32;
    typedef vint16mf2_t vi16;
    typedef vfloat32m1_t vf32;
    static inline vi32 vle32(const int32_t* p, size_t vl) { return __riscv_vle32_v_i32m1(p, vl); }
    static inline vf32 vle32f(const float* p, size_t vl) { return __riscv_vle32_v_f32m1(p, vl); }
    static inline vf32 vfmv(float x, size_t vl) { return __riscv_vfmv_v_f_f32m1(x, vl); }
    static inline vi16 loadw16(const int8_t* p, size_t vl)
    { return __riscv_vsext_vf2_i16mf2(__riscv_vle8_v_i8mf4(p, vl), vl); }
    static inline vi32 vwmacc(vi32 acc, int16_t x, vi16 w, size_t vl)
    { return __riscv_vwmacc_vx_i32m1(acc, x, w, vl); }
    static inline vf32 tofloat(vi32 x, size_t vl) { return __riscv_vfcvt_f_x_v_f32m1(x, vl); }
    static inline vf32 fmadd(vf32 a, vf32 b, vf32 c, size_t vl) { return __riscv_vfmadd_vv_f32m1(a, b, c, vl); }
    static inline vf32 fadd(vf32 a, vf32 b, size_t vl) { return __riscv_vfadd_vv_f32m1(a, b, vl); }
    static inline vf32 fsub(vf32 a, vf32 b, size_t vl) { return __riscv_vfsub_vv_f32m1(a, b, vl); }
    static inline vi32 res32(const int8_t* p, bool isU8, size_t vl)
    {
        if (isU8)
            return __riscv_vreinterpret_v_u32m1_i32m1(
                __riscv_vzext_vf4_u32m1(__riscv_vle8_v_u8mf4((const uint8_t*)p, vl), vl));
        return __riscv_vsext_vf4_i32m1(__riscv_vle8_v_i8mf4(p, vl), vl);
    }
    static inline vi32 round32(vf32 x, size_t vl) { return __riscv_vfcvt_x_f_v_i32m1(x, vl); }
    static inline void clampstore(int8_t* dst, vi32 x, int lo, int hi, size_t vl)
    {
        x = __riscv_vmax_vx_i32m1(x, lo, vl);
        x = __riscv_vmin_vx_i32m1(x, hi, vl);
        vint16mf2_t x16 = __riscv_vncvt_x_x_w_i16mf2(x, vl);
        __riscv_vse8_v_i8mf4(dst, __riscv_vncvt_x_x_w_i8mf4(x16, vl), vl);
    }
};

struct ConvInt8RVV_M2
{
    typedef vint32m2_t vi32;
    typedef vint16m1_t vi16;
    typedef vfloat32m2_t vf32;
    static inline vi32 vle32(const int32_t* p, size_t vl) { return __riscv_vle32_v_i32m2(p, vl); }
    static inline vf32 vle32f(const float* p, size_t vl) { return __riscv_vle32_v_f32m2(p, vl); }
    static inline vf32 vfmv(float x, size_t vl) { return __riscv_vfmv_v_f_f32m2(x, vl); }
    static inline vi16 loadw16(const int8_t* p, size_t vl)
    { return __riscv_vsext_vf2_i16m1(__riscv_vle8_v_i8mf2(p, vl), vl); }
    static inline vi32 vwmacc(vi32 acc, int16_t x, vi16 w, size_t vl)
    { return __riscv_vwmacc_vx_i32m2(acc, x, w, vl); }
    static inline vf32 tofloat(vi32 x, size_t vl) { return __riscv_vfcvt_f_x_v_f32m2(x, vl); }
    static inline vf32 fmadd(vf32 a, vf32 b, vf32 c, size_t vl) { return __riscv_vfmadd_vv_f32m2(a, b, c, vl); }
    static inline vf32 fadd(vf32 a, vf32 b, size_t vl) { return __riscv_vfadd_vv_f32m2(a, b, vl); }
    static inline vf32 fsub(vf32 a, vf32 b, size_t vl) { return __riscv_vfsub_vv_f32m2(a, b, vl); }
    static inline vi32 res32(const int8_t* p, bool isU8, size_t vl)
    {
        if (isU8)
            return __riscv_vreinterpret_v_u32m2_i32m2(
                __riscv_vzext_vf4_u32m2(__riscv_vle8_v_u8mf2((const uint8_t*)p, vl), vl));
        return __riscv_vsext_vf4_i32m2(__riscv_vle8_v_i8mf2(p, vl), vl);
    }
    static inline vi32 round32(vf32 x, size_t vl) { return __riscv_vfcvt_x_f_v_i32m2(x, vl); }
    static inline void clampstore(int8_t* dst, vi32 x, int lo, int hi, size_t vl)
    {
        x = __riscv_vmax_vx_i32m2(x, lo, vl);
        x = __riscv_vmin_vx_i32m2(x, hi, vl);
        vint16m1_t x16 = __riscv_vncvt_x_x_w_i16m1(x, vl);
        __riscv_vse8_v_i8mf2(dst, __riscv_vncvt_x_x_w_i8mf2(x16, vl), vl);
    }
};

// vfcvt.x.f.v uses the current FP rounding mode; Linux default is RNE,
// which matches cvRound() used by the scalar reference path.
#define CONV_RVV_MAC(j) { \
    const int8_t* ipj = inptr[(j)]; \
    acc##j = RVV::vwmacc(acc##j, (int16_t)(int8_t)((uint8_t)ipj[0] ^ xor_val), w0, vl); \
    acc##j = RVV::vwmacc(acc##j, (int16_t)(int8_t)((uint8_t)ipj[1] ^ xor_val), w1, vl); \
    acc##j = RVV::vwmacc(acc##j, (int16_t)(int8_t)((uint8_t)ipj[2] ^ xor_val), w2, vl); \
    acc##j = RVV::vwmacc(acc##j, (int16_t)(int8_t)((uint8_t)ipj[3] ^ xor_val), w3, vl); \
    acc##j = RVV::vwmacc(acc##j, (int16_t)(int8_t)((uint8_t)ipj[4] ^ xor_val), w4, vl); \
    acc##j = RVV::vwmacc(acc##j, (int16_t)(int8_t)((uint8_t)ipj[5] ^ xor_val), w5, vl); \
    acc##j = RVV::vwmacc(acc##j, (int16_t)(int8_t)((uint8_t)ipj[6] ^ xor_val), w6, vl); \
    acc##j = RVV::vwmacc(acc##j, (int16_t)(int8_t)((uint8_t)ipj[7] ^ xor_val), w7, vl); \
}

#define CONV_RVV_STORE(j) { \
    typename RVV::vf32 facc = RVV::fmadd(RVV::tofloat(acc##j, vl), vmult, vzp, vl); \
    int8_t* optrj = outptr + (size_t)(p + (j)) * K0; \
    if (resptr) { \
        const int8_t* rp = resptr + (size_t)(p + (j)) * K0; \
        facc = RVV::fadd(facc, \
            RVV::fsub(RVV::tofloat(RVV::res32(rp, inputIsU8, vl), vl), vzp, vl), vl); \
    } \
    typename RVV::vi32 ival = RVV::round32(facc, vl); \
    if (inputIsU8) { \
        if (activLUT) { \
            int8_t tmp8[K0]; \
            RVV::clampstore(tmp8, ival, 0, 255, vl); \
            for (int k = 0; k < K0; k++) \
                ((uint8_t*)optrj)[k] = (uint8_t)activLUT[(int)(uint8_t)tmp8[k]]; \
        } else { \
            RVV::clampstore(optrj, ival, 0, 255, vl); \
        } \
    } else { \
        if (activLUT) { \
            int8_t tmp8[K0]; \
            RVV::clampstore(tmp8, ival, -128, 127, vl); \
            for (int k = 0; k < K0; k++) \
                optrj[k] = (int8_t)activLUT[(int)tmp8[k] + 128]; \
        } else { \
            RVV::clampstore(optrj, ival, -128, 127, vl); \
        } \
    } \
}

template<typename RVV>
static void convInt8BlockRVV(const void* inp_, const void* residual_,
                             void* out_, const ConvState& cs,
                             const void* weights_,
                             const int* bias, const float* multiplier,
                             int inp_zp, int out_zp,
                             const int8_t* activLUT,
                             bool inputIsU8)
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
    const int8_t* wdata = (const int8_t*)weights_;
    const int* ofsZYX = cs.coordtab.data();

    size_t outtotal = outshape.total();
    if ((Kg % K0) != 0)
        memset(out, 0, outtotal);

    int total_blocks = N * ngroups * Kblk;

    parallel_for_(Range(0, total_blocks), [&](const Range& range) {
        const size_t vl = K0;   // 8 lanes over output channels; gated by vlmax check at dispatch
        const uint8_t xor_val = inputIsU8 ? 0x80 : 0;

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
        int8_t zbuf[C0];
        memset(zbuf, (uint8_t)inp_zp, C0);

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
            alignas(32) float multbuf[K0];
            {
                int k_valid = std::min(K0, K - k_base);
                memcpy(biasbuf, bias + k_base, k_valid * sizeof(int32_t));
                memset(biasbuf + k_valid, 0, (K0 - k_valid) * sizeof(int32_t));
                memcpy(multbuf, multiplier + k_base, k_valid * sizeof(float));
                memset(multbuf + k_valid, 0, (K0 - k_valid) * sizeof(float));
            }

            typename RVV::vi32 vbias = RVV::vle32(biasbuf, vl);
            typename RVV::vf32 vmult = RVV::vle32f(multbuf, vl);
            typename RVV::vf32 vzp = RVV::vfmv((float)out_zp, vl);

            int p = 0;
            for (; p < planeblocks_l; p += SPAT_BLOCK_SIZE) {
                if (p + SPAT_BLOCK_SIZE > planeblocks_l) {
                    if (p == 0) break;
                    p = planeblocks_l - SPAT_BLOCK_SIZE;
                }

                Vec3i pt[SPAT_BLOCK_SIZE];
                bool inner[SPAT_BLOCK_SIZE];

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
                    }
                }

                typename RVV::vi32 acc0 = vbias, acc1 = vbias, acc2 = vbias, acc3 = vbias;
                typename RVV::vi32 acc4 = vbias, acc5 = vbias, acc6 = vbias, acc7 = vbias;

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
                        typename RVV::vi16 w0 = RVV::loadw16(wptr, vl);
                        typename RVV::vi16 w1 = RVV::loadw16(wptr + K0, vl);
                        typename RVV::vi16 w2 = RVV::loadw16(wptr + 2 * K0, vl);
                        typename RVV::vi16 w3 = RVV::loadw16(wptr + 3 * K0, vl);
                        typename RVV::vi16 w4 = RVV::loadw16(wptr + 4 * K0, vl);
                        typename RVV::vi16 w5 = RVV::loadw16(wptr + 5 * K0, vl);
                        typename RVV::vi16 w6 = RVV::loadw16(wptr + 6 * K0, vl);
                        typename RVV::vi16 w7 = RVV::loadw16(wptr + 7 * K0, vl);

                        CONV_RVV_MAC(0); CONV_RVV_MAC(1);
                        CONV_RVV_MAC(2); CONV_RVV_MAC(3);
                        CONV_RVV_MAC(4); CONV_RVV_MAC(5);
                        CONV_RVV_MAC(6); CONV_RVV_MAC(7);

                        for (int j = 0; j < SPAT_BLOCK_SIZE; j++)
                            inptr[j] += inpstep[j];
                    }
                }

                CONV_RVV_STORE(0); CONV_RVV_STORE(1);
                CONV_RVV_STORE(2); CONV_RVV_STORE(3);
                CONV_RVV_STORE(4); CONV_RVV_STORE(5);
                CONV_RVV_STORE(6); CONV_RVV_STORE(7);
            }

            for (; p < planeblocks_l; p++) {
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

                    bool out_of_bounds = (((unsigned)zi >= (unsigned)Di_l) |
                         ((unsigned)yi >= (unsigned)Hi_l) |
                         ((unsigned)xi >= (unsigned)Wi_l)) != 0;

                    const int8_t* inptr = out_of_bounds ? zbuf :
                        inpbaseptr + (((zi * Hi_l) + yi) * Wi_l + xi) * C0;
                    int inpstep = out_of_bounds ? 0 : iplanesize_l;
                    const int8_t* wptr = wbaseptr + (size_t)i * C1Max * K0 * C0;

                    for (int c1 = 0; c1 < cblocks; c1++, inptr += inpstep, wptr += C0 * K0) {
                        for (int c0 = 0; c0 < C0; c0++) {
                            int ival = (int)(int8_t)((uint8_t)inptr[c0] ^ xor_val);
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
                    if (rptr) {
                        int rval = inputIsU8 ? (int)(uint8_t)rptr[k0] : (int)rptr[k0];
                        val += (float)(rval - out_zp);
                    }
                    int ival = cvRound(val);
                    if (inputIsU8) {
                        ival = std::max(0, std::min(255, ival));
                        if (activLUT) ival = (int)(uint8_t)activLUT[ival];
                        ((uint8_t*)optr)[k0] = (uint8_t)ival;
                    } else {
                        ival = std::max(-128, std::min(127, ival));
                        if (activLUT) ival = (int)activLUT[ival + 128];
                        optr[k0] = (int8_t)ival;
                    }
                }
            }
        }
    });
}

#undef CONV_RVV_MAC
#undef CONV_RVV_STORE
#endif // CV_RVV

// INT8 depthwise convolution kernel (ngroups==K==C, Kg==Cg==1).
static void convInt8BlockDepthwise(const void* inp_, const void* residual_,
                                   void* out_, const ConvState& cs,
                                   const void* weights_,
                                   const int* bias, const float* multiplier,
                                   int inp_zp, int out_zp,
                                   const int8_t* activLUT,
                                   bool inputIsU8)
{
    constexpr int C0 = 8, K0 = 8;
    constexpr int MAX_CONV_DIMS = ConvState::MAX_CONV_DIMS;

    const MatShape& inpshape = cs.inpshape;
    const MatShape& outshape = cs.outshape;

    int N    = outshape[0];
    int ndims = outshape.dims;
    int K    = outshape.channels();
    int C    = inpshape.channels();
    int K1   = (K + K0 - 1) / K0;
    int C1   = (C + C0 - 1) / C0;

    int D_  = ndims >= 6 ? outshape[ndims-4] : 1;
    int H_  = ndims >= 5 ? outshape[ndims-3] : 1;
    int W_  = outshape[ndims-2];
    int planeblocks_ = D_ * H_ * W_;

    int Di_ = ndims >= 6 ? inpshape[ndims-4] : 1;
    int Hi_ = ndims >= 5 ? inpshape[ndims-3] : 1;
    int Wi_ = inpshape[ndims-2];

    int ngroups = cs.ngroups;
    int ksize_  = cs.wshape[2];
    int C1Max   = cs.wshape[3];

    int Sz = cs.strides[0], Sy = cs.strides[1], Sx = cs.strides[2];
    int padZ = cs.pads[0], padY = cs.pads[1], padX = cs.pads[2];

    int innerZ0 = cs.inner[0], innerZ1 = cs.inner[MAX_CONV_DIMS];
    int innerY0 = cs.inner[1], innerY1 = cs.inner[MAX_CONV_DIMS+1];
    int innerX0 = cs.inner[2], innerX1 = cs.inner[MAX_CONV_DIMS+2];

    const int8_t* inp      = (const int8_t*)inp_;
    const int8_t* residual = (const int8_t*)residual_;
    int8_t*       out      = (int8_t*)out_;
    const int8_t* wdata    = (const int8_t*)weights_;
    const int*    ofsZYX   = cs.coordtab.data();

    int iplanesize_ = Di_ * Hi_ * Wi_ * C0;
    int planesize_  = planeblocks_ * K0;

    // Spatial tiling: split planeblocks into ntiles tiles so the total task count
    // is N*K1*ntiles, giving all threads work even when K is small (e.g. K=16 → K1=2).
    constexpr int SB_TILE = 8;
    int nthreads = std::max(1, cv::getNumThreads());
    int TILE = std::max(SB_TILE,
        ((planeblocks_ + nthreads * 4 - 1) / (nthreads * 4) + SB_TILE - 1) & ~(SB_TILE - 1));
    int ntiles = (planeblocks_ + TILE - 1) / TILE;
    int total_tasks = N * K1 * ntiles;
#if CV_AVX2 && (defined(__x86_64__) || defined(_M_X64))
    static constexpr int WBUF_SIZE = 128 * 8;  // 128 * K0, K0=8; static for MSVC 19.29
#endif

    parallel_for_(Range(0, total_tasks), [&](const Range& range) {
        int H = H_, W = W_;
        int Di = Di_, Hi = Hi_, Wi = Wi_;
        int planeblocks = planeblocks_;
        int iplanesize = iplanesize_;
        int planesize  = planesize_;
        int ksize      = ksize_;

        for (int task = range.start; task < range.end; task++) {
            int nk   = task / ntiles;
            int tile = task - nk * ntiles;
            int n    = nk / K1;
            int k1   = nk - n * K1;

            int g_start      = k1 * K0;
            int g_end        = std::min(g_start + K0, ngroups);
            int p_task_start = tile * TILE;
            int p_task_end   = std::min(p_task_start + TILE, planeblocks);

            const int8_t* inpbaseptr = inp      + (size_t)(n * C1 + k1) * iplanesize;
            int8_t*       outbase    = out      + (size_t)(n * K1 + k1) * planesize;
            const int8_t* resbase    = residual ? residual + (size_t)(n * K1 + k1) * planesize
                                                : nullptr;


#if CV_AVX2 && (defined(__x86_64__) || defined(_M_X64))
            if (ksize <= 128) {
                int n_g = g_end - g_start;
                constexpr int SB = 8;
                // Pre-gather weights for this group block into wbuf[ksize * K0].
                alignas(32) int8_t wbuf[WBUF_SIZE];
                for (int i = 0; i < ksize; i++) {
                    for (int j = 0; j < n_g; j++) {
                        int g  = g_start + j;
                        int c0 = g % C0;
                        wbuf[i * K0 + j] = wdata[(size_t)g * ksize * C1Max * C0 * K0
                                                + (size_t)i * C1Max * C0 * K0
                                                + (size_t)c0 * K0];
                    }
                    for (int j = n_g; j < K0; j++)
                        wbuf[i * K0 + j] = 0;
                }

                __m256i vbias = _mm256_loadu_si256((const __m256i*)(bias + g_start));
                __m256  vmult = _mm256_loadu_ps(multiplier + g_start);
                __m256  vzp   = _mm256_set1_ps((float)out_zp);
                __m128i v_xor = inputIsU8 ? _mm_setzero_si128()
                                          : _mm_set1_epi8((char)0x80);

                int p = p_task_start;
                for (; p < p_task_end; p += SB) {
                    if (p + SB > p_task_end) {
                        if (p == p_task_start) break;
                        p = p_task_end - SB;
                    }

                    Vec3i pt[SB];
                    bool  inner_pix[SB];
                    bool  all_inner = true;
                    bool  same_row  = false;

                    if ((p % W) + SB <= W) {
                        same_row = true;
                        int zj  = p / (H * W);
                        int yxj = p - zj * H * W;
                        int yj  = yxj / W;
                        int xj0 = yxj - yj * W;
                        bool zy_inner = (zj >= innerZ0 && zj < innerZ1) &&
                                        (yj >= innerY0 && yj < innerY1);
                        for (int j = 0; j < SB; j++) {
                            int xj = xj0 + j;
                            pt[j] = Vec3i(zj*Sz - padZ, yj*Sy - padY, xj*Sx - padX);
                            inner_pix[j] = zy_inner && (xj >= innerX0 && xj < innerX1);
                            all_inner   &= inner_pix[j];
                        }
                    } else {
                        for (int j = 0; j < SB; j++) {
                            int pj  = p + j;
                            int zj  = pj / (H * W);
                            int yxj = pj - zj * H * W;
                            int yj  = yxj / W;
                            int xj  = yxj - yj * W;
                            pt[j] = Vec3i(zj*Sz - padZ, yj*Sy - padY, xj*Sx - padX);
                            inner_pix[j] = (zj >= innerZ0 && zj < innerZ1) &&
                                           (yj >= innerY0 && yj < innerY1) &&
                                           (xj >= innerX0 && xj < innerX1);
                            all_inner &= inner_pix[j];
                        }
                    }

                    __m256i s0 = vbias, s1 = vbias, s2 = vbias, s3 = vbias;
                    __m256i s4 = vbias, s5 = vbias, s6 = vbias, s7 = vbias;

                    if (all_inner && same_row) {
                        const int step = Sx * C0;
                        for (int i = 0; i < ksize; i++) {
                            __m128i w8  = _mm_loadl_epi64((const __m128i*)(wbuf + i * K0));
                            __m256i w32 = _mm256_cvtepi8_epi32(w8);
                            int zi_eff = pt[0][0] + ofsZYX[i*3];
                            int yi_eff = pt[0][1] + ofsZYX[i*3+1];
                            int xi_eff = pt[0][2] + ofsZYX[i*3+2];
                            const int8_t* ip = inpbaseptr +
                                (((zi_eff * Hi) + yi_eff) * Wi + xi_eff) * C0;
                            #define DW_HOT(j) { \
                                __m256i inp32 = _mm256_cvtepu8_epi32(_mm_xor_si128( \
                                    _mm_loadl_epi64((const __m128i*)(ip+(j)*step)), v_xor)); \
                                s##j = _mm256_add_epi32(s##j, \
                                    _mm256_mullo_epi32(inp32, w32)); \
                            }
                            DW_HOT(0); DW_HOT(1); DW_HOT(2); DW_HOT(3);
                            DW_HOT(4); DW_HOT(5); DW_HOT(6); DW_HOT(7);
                            #undef DW_HOT
                        }
                    } else if (all_inner) {
                        for (int i = 0; i < ksize; i++) {
                            __m128i w8  = _mm_loadl_epi64((const __m128i*)(wbuf + i * K0));
                            __m256i w32 = _mm256_cvtepi8_epi32(w8);
                            #define DW_INNER(j) { \
                                int zi = pt[j][0] + ofsZYX[i*3]; \
                                int yi = pt[j][1] + ofsZYX[i*3+1]; \
                                int xi = pt[j][2] + ofsZYX[i*3+2]; \
                                const int8_t* ip = inpbaseptr + \
                                    (((zi * Hi) + yi) * Wi + xi) * C0; \
                                __m256i inp32 = _mm256_cvtepu8_epi32(_mm_xor_si128( \
                                    _mm_loadl_epi64((const __m128i*)ip), v_xor)); \
                                s##j = _mm256_add_epi32(s##j, \
                                    _mm256_mullo_epi32(inp32, w32)); \
                            }
                            DW_INNER(0); DW_INNER(1); DW_INNER(2); DW_INNER(3);
                            DW_INNER(4); DW_INNER(5); DW_INNER(6); DW_INNER(7);
                            #undef DW_INNER
                        }
                    } else {
                        for (int i = 0; i < ksize; i++) {
                            __m128i w8  = _mm_loadl_epi64((const __m128i*)(wbuf + i * K0));
                            __m256i w32 = _mm256_cvtepi8_epi32(w8);
                            #define DW_BORDER(j) { \
                                int zi = pt[j][0] + ofsZYX[i*3]; \
                                int yi = pt[j][1] + ofsZYX[i*3+1]; \
                                int xi = pt[j][2] + ofsZYX[i*3+2]; \
                                __m256i inp32; \
                                if (!((unsigned)zi >= (unsigned)Di || \
                                      (unsigned)yi >= (unsigned)Hi || \
                                      (unsigned)xi >= (unsigned)Wi)) { \
                                    const int8_t* ip = inpbaseptr + \
                                        (((zi * Hi) + yi) * Wi + xi) * C0; \
                                    inp32 = _mm256_cvtepu8_epi32(_mm_xor_si128( \
                                        _mm_loadl_epi64((const __m128i*)ip), v_xor)); \
                                } else { \
                                    inp32 = inputIsU8 \
                                        ? _mm256_set1_epi32(inp_zp) \
                                        : _mm256_set1_epi32((int)(uint8_t)((uint8_t)inp_zp ^ 0x80u)); \
                                } \
                                s##j = _mm256_add_epi32(s##j, _mm256_mullo_epi32(inp32, w32)); \
                            }
                            DW_BORDER(0); DW_BORDER(1); DW_BORDER(2); DW_BORDER(3);
                            DW_BORDER(4); DW_BORDER(5); DW_BORDER(6); DW_BORDER(7);
                            #undef DW_BORDER
                        }
                    }

                    #define DW_STORE(j) { \
                        __m256 facc = _mm256_add_ps( \
                            _mm256_mul_ps(_mm256_cvtepi32_ps(s##j), vmult), vzp); \
                        int8_t*       optr = outbase + (p + (j)) * K0; \
                        const int8_t* rptr = resbase \
                            ? resbase + (p + (j)) * K0 : nullptr; \
                        if (rptr) { \
                            __m256i r32 = _mm256_cvtepi8_epi32( \
                                _mm_loadl_epi64((const __m128i*)rptr)); \
                            facc = _mm256_add_ps(facc, _mm256_sub_ps( \
                                _mm256_cvtepi32_ps(r32), vzp)); \
                        } \
                        __m256i iv  = _mm256_cvtps_epi32(facc); \
                        __m128i lo  = _mm256_castsi256_si128(iv); \
                        __m128i hi  = _mm256_extracti128_si256(iv, 1); \
                        __m128i p16 = _mm_packs_epi32(lo, hi); \
                        __m128i p8  = inputIsU8 ? _mm_packus_epi16(p16, p16) \
                                                : _mm_packs_epi16(p16, p16); \
                        if (activLUT) { \
                            alignas(16) int8_t tmp8[16]; \
                            _mm_store_si128((__m128i*)tmp8, p8); \
                            if (inputIsU8) { \
                                for (int jj = 0; jj < n_g; jj++) \
                                    ((uint8_t*)optr)[jj] = \
                                        (uint8_t)activLUT[(int)(uint8_t)tmp8[jj]]; \
                            } else { \
                                for (int jj = 0; jj < n_g; jj++) \
                                    optr[jj] = (int8_t)activLUT[(int)tmp8[jj]+128]; \
                            } \
                        } else if (n_g == K0) { \
                            _mm_storel_epi64((__m128i*)optr, p8); \
                        } else { \
                            alignas(16) int8_t tmp8[16]; \
                            _mm_store_si128((__m128i*)tmp8, p8); \
                            memcpy(optr, tmp8, n_g); \
                        } \
                    }
                    DW_STORE(0); DW_STORE(1); DW_STORE(2); DW_STORE(3);
                    DW_STORE(4); DW_STORE(5); DW_STORE(6); DW_STORE(7);
                    #undef DW_STORE
                }
                for (; p < p_task_end; p++) {
                    int zj  = p / (H * W);
                    int yxj = p - zj * H * W;
                    int yj  = yxj / W;
                    int xj  = yxj - yj * W;
                    int zi_base = zj * Sz - padZ, yi_base = yj * Sy - padY,
                        xi_base = xj * Sx - padX;
                    __m256i acc = vbias;
                    for (int i = 0; i < ksize; i++) {
                        int zi = zi_base + ofsZYX[i*3], yi = yi_base + ofsZYX[i*3+1],
                            xi = xi_base + ofsZYX[i*3+2];
                        if ((unsigned)zi>=(unsigned)Di || (unsigned)yi>=(unsigned)Hi ||
                            (unsigned)xi>=(unsigned)Wi) continue;
                        const int8_t* ip = inpbaseptr + (((zi*Hi)+yi)*Wi+xi)*C0;
                        __m256i inp32 = _mm256_cvtepu8_epi32(
                            _mm_xor_si128(_mm_loadl_epi64((const __m128i*)ip), v_xor));
                        __m256i w32 = _mm256_cvtepi8_epi32(
                            _mm_loadl_epi64((const __m128i*)(wbuf + i*K0)));
                        acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(inp32, w32));
                    }
                    __m256 facc = _mm256_add_ps(
                        _mm256_mul_ps(_mm256_cvtepi32_ps(acc), vmult), vzp);
                    int8_t* optr = outbase + p * K0;
                    const int8_t* rptr = resbase ? resbase + p * K0 : nullptr;
                    if (rptr) {
                        facc = _mm256_add_ps(facc, _mm256_sub_ps(_mm256_cvtepi32_ps(
                            _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)rptr))), vzp));
                    }
                    __m256i iv = _mm256_cvtps_epi32(facc);
                    __m128i p16 = _mm_packs_epi32(_mm256_castsi256_si128(iv),
                                                  _mm256_extracti128_si256(iv, 1));
                    __m128i p8  = inputIsU8 ? _mm_packus_epi16(p16, p16)
                                            : _mm_packs_epi16(p16, p16);
                    if (activLUT) {
                        alignas(16) int8_t tmp8[16];
                        _mm_store_si128((__m128i*)tmp8, p8);
                        if (inputIsU8)
                            for (int jj=0; jj<n_g; jj++)
                                ((uint8_t*)optr)[jj]=(uint8_t)activLUT[(int)(uint8_t)tmp8[jj]];
                        else
                            for (int jj=0; jj<n_g; jj++)
                                optr[jj]=(int8_t)activLUT[(int)tmp8[jj]+128];
                    } else if (n_g == K0) {
                        _mm_storel_epi64((__m128i*)optr, p8);
                    } else {
                        alignas(16) int8_t tmp8[16];
                        _mm_store_si128((__m128i*)tmp8, p8);
                        memcpy(optr, tmp8, n_g);
                    }
                }
                continue;
            }
#endif // CV_AVX2

            for (int p = p_task_start; p < p_task_end; p++) {
                int zj  = p / (H * W);
                int yxj = p - zj * H * W;
                int yj  = yxj / W;
                int xj  = yxj - yj * W;

                bool is_inner = (zj >= innerZ0 && zj < innerZ1) &&
                                (yj >= innerY0 && yj < innerY1) &&
                                (xj >= innerX0 && xj < innerX1);

                int zi_base = zj * Sz - padZ;
                int yi_base = yj * Sy - padY;
                int xi_base = xj * Sx - padX;

                int8_t*       optr = outbase + p * K0;
                const int8_t* rptr = resbase ? resbase + p * K0 : nullptr;

                for (int g = g_start; g < g_end; g++) {
                    int j  = g - g_start;
                    // For Kg=1 depthwise: kin=0, k0=0, c0=g%C0
                    int c0 = g % C0;
                    const int8_t* wbase_g = wdata + (size_t)g * ksize * C1Max * C0 * K0;
                    int32_t acc = bias[g];

                    if (is_inner) {
                        for (int i = 0; i < ksize; i++) {
                            int zi = zi_base + ofsZYX[i * 3];
                            int yi = yi_base + ofsZYX[i * 3 + 1];
                            int xi = xi_base + ofsZYX[i * 3 + 2];
                            const int8_t* ip = inpbaseptr + (((zi * Hi) + yi) * Wi + xi) * C0;
                            int ival = inputIsU8 ? (int)(uint8_t)ip[c0]
                                                 : (int)(uint8_t)((uint8_t)ip[c0] ^ 0x80u);
                            acc += ival * (int)wbase_g[(size_t)i * C1Max * C0 * K0 + c0 * K0];
                        }
                    } else {
                        for (int i = 0; i < ksize; i++) {
                            int zi = zi_base + ofsZYX[i * 3];
                            int yi = yi_base + ofsZYX[i * 3 + 1];
                            int xi = xi_base + ofsZYX[i * 3 + 2];
                            int ival;
                            if ((unsigned)zi >= (unsigned)Di ||
                                (unsigned)yi >= (unsigned)Hi ||
                                (unsigned)xi >= (unsigned)Wi) {
                                ival = inputIsU8 ? inp_zp
                                                 : (int)(uint8_t)((uint8_t)inp_zp ^ 0x80u);
                            } else {
                                const int8_t* ip = inpbaseptr + (((zi * Hi) + yi) * Wi + xi) * C0;
                                ival = inputIsU8 ? (int)(uint8_t)ip[c0]
                                                 : (int)(uint8_t)((uint8_t)ip[c0] ^ 0x80u);
                            }
                            acc += ival * (int)wbase_g[(size_t)i * C1Max * C0 * K0 + c0 * K0];
                        }
                    }

                    float val = (float)acc * multiplier[g] + (float)out_zp;
                    if (rptr) val += (float)((int)rptr[j] - out_zp);
                    int ival = cvRound(val);
                    if (inputIsU8) {
                        ival = std::max(0, std::min(255, ival));
                        if (activLUT) ival = (int)(uint8_t)activLUT[ival];
                        ((uint8_t*)optr)[j] = (uint8_t)ival;
                    } else {
                        ival = std::max(-128, std::min(127, ival));
                        if (activLUT) ival = (int)activLUT[ival + 128];
                        optr[j] = (int8_t)ival;
                    }
                }
            }
        }
    });
}

void convInt8Block(const void* inp_, const void* residual_,
                   void* out_, const ConvState& cs,
                   const void* weights_,
                   const void* weightsVNNI_,
                   const int* bias, const int* biasVNNI_,
                   const float* multiplier,
                   int inp_zp, int out_zp,
                   const int8_t* activLUT,
                   bool inputIsU8)
{
    // Dispatch to dedicated depthwise kernel before the general VNNI path.
    if (cs.depthwise) {
        const int* dw_bias = biasVNNI_ ? biasVNNI_ : bias;
        convInt8BlockDepthwise(inp_, residual_, out_, cs, weights_,
                               dw_bias, multiplier, inp_zp, out_zp, activLUT, inputIsU8);
        return;
    }
#if CV_AVXVNNI_AVAILABLE
    if (cv::checkHardwareSupport(CV_CPU_AVX_VNNI) && weightsVNNI_ && biasVNNI_) {
        if (cs.wshape[2] == 1 &&
            cs.strides[0] == 1 && cs.strides[1] == 1 && cs.strides[2] == 1) {
            convInt8Block1x1(inp_, residual_, out_, cs, weightsVNNI_,
                             biasVNNI_, multiplier, inp_zp, out_zp, activLUT, inputIsU8);
            return;
        }
        convInt8BlockVNNI(inp_, residual_, out_, cs, weightsVNNI_,
                          biasVNNI_, multiplier, inp_zp, out_zp, activLUT, inputIsU8);
        return;
    }
#endif
#if CV_NEON && defined(CV_NEON_DOT) && CV_NEON_DOT
    if (weightsVNNI_) {
        convInt8BlockNEON(inp_, residual_, out_, cs, weightsVNNI_,
                          bias, multiplier, inp_zp, out_zp, activLUT, inputIsU8);
        return;
    }
#endif
#if CV_RVV
    if (weights_) {
        // vl=8 over the K0 output channels; pick the narrowest register group
        // that provides 8 lanes at the runtime VLEN (no VLEN hardcoded).
        if (__riscv_vsetvlmax_e32m1() >= 8) {
            convInt8BlockRVV<ConvInt8RVV_M1>(inp_, residual_, out_, cs, weights_,
                                             bias, multiplier, inp_zp, out_zp, activLUT, inputIsU8);
            return;
        }
        if (__riscv_vsetvlmax_e32m2() >= 8) {
            convInt8BlockRVV<ConvInt8RVV_M2>(inp_, residual_, out_, cs, weights_,
                                             bias, multiplier, inp_zp, out_zp, activLUT, inputIsU8);
            return;
        }
    }
#endif
#if !CV_AVXVNNI_AVAILABLE && !(CV_NEON && defined(CV_NEON_DOT) && CV_NEON_DOT)
    (void)weightsVNNI_;
    (void)biasVNNI_;
#else
    (void)biasVNNI_;
#endif

    constexpr int C0 = 8, K0 = 8;

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
#if CV_AVX2
    static constexpr int SUMBUF_SIZE = 8 * 8;  // SPAT_BLOCK_SIZE * K0, both=8; static for MSVC 19.29
#endif

    parallel_for_(Range(0, total_blocks), [&](const Range& range) {
        constexpr int C0 = 8, K0 = 8;
        constexpr int C0shift = 3;
#if CV_AVX2
        constexpr int SPAT_BLOCK_SIZE = 8;
        alignas(32) int32_t sumbuf[SUMBUF_SIZE];
#endif
        const uint8_t xor_val = inputIsU8 ? 0x80 : 0;

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
        int8_t zbuf[C0];
        memset(zbuf, (uint8_t)inp_zp, C0);

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
            alignas(32) float multbuf[K0];
            {
                int k_valid = std::min(K0, K - k_base);
                memcpy(biasbuf, bias + k_base, k_valid * sizeof(int32_t));
                memset(biasbuf + k_valid, 0, (K0 - k_valid) * sizeof(int32_t));
                memcpy(multbuf, multiplier + k_base, k_valid * sizeof(float));
                memset(multbuf + k_valid, 0, (K0 - k_valid) * sizeof(float));
            }

        #if CV_AVX2
            int p = 0;
            for (; p < planeblocks_l; p += SPAT_BLOCK_SIZE) {
                if (p + SPAT_BLOCK_SIZE > planeblocks_l) {
                    if (p == 0) break;
                    p = planeblocks_l - SPAT_BLOCK_SIZE;
                }

                Vec3i pt[SPAT_BLOCK_SIZE];
                bool inner[SPAT_BLOCK_SIZE];

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
                        #define CONV_INT8_LOAD(j, c0) \
                            _mm256_set1_epi32( \
                                (uint16_t)(int16_t)(int8_t)((uint8_t)inptr[j][(c0)] ^ xor_val) | \
                                ((uint32_t)(uint16_t)(int16_t)(int8_t)((uint8_t)inptr[j][(c0)+1] ^ xor_val) << 16))
                        #define CONV_INT8_MAC_PAIR(c0) { \
                            __m128i wraw = _mm_loadu_si128((const __m128i*)(wptr + (c0) * K0)); \
                            __m128i w_intlv = _mm_unpacklo_epi8(wraw, _mm_srli_si128(wraw, 8)); \
                            __m256i w16 = _mm256_cvtepi8_epi16(w_intlv); \
                            __m256i x0 = CONV_INT8_LOAD(0, c0); \
                            __m256i x1 = CONV_INT8_LOAD(1, c0); \
                            __m256i x2 = CONV_INT8_LOAD(2, c0); \
                            __m256i x3 = CONV_INT8_LOAD(3, c0); \
                            __m256i x4 = CONV_INT8_LOAD(4, c0); \
                            __m256i x5 = CONV_INT8_LOAD(5, c0); \
                            __m256i x6 = CONV_INT8_LOAD(6, c0); \
                            __m256i x7 = CONV_INT8_LOAD(7, c0); \
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
                        #undef CONV_INT8_LOAD

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
                        __m256i res32 = inputIsU8 ?
                            _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)rptr)) :
                            _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)rptr));
                        facc = _mm256_add_ps(facc,
                            _mm256_sub_ps(_mm256_cvtepi32_ps(res32), vzp));
                    }

                    __m256i ival = _mm256_cvtps_epi32(facc);
                    __m128i lo = _mm256_castsi256_si128(ival);
                    __m128i hi = _mm256_extracti128_si256(ival, 1);
                    __m128i packed16 = _mm_packs_epi32(lo, hi);
                    __m128i packed8 = inputIsU8 ? _mm_packus_epi16(packed16, packed16)
                                                : _mm_packs_epi16(packed16, packed16);

                    int8_t* optr = outptr + (p + j) * K0;
                    if (activLUT) {
                        alignas(16) int8_t tmp8[16];
                        _mm_store_si128((__m128i*)tmp8, packed8);
                        for (int k = 0; k < K0; k++) {
                            int lut_idx = inputIsU8 ? (int)(uint8_t)tmp8[k] : ((int)tmp8[k] + 128);
                            optr[k] = (int8_t)activLUT[lut_idx];
                        }
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

                    bool out_of_bounds = (((unsigned)zi >= (unsigned)Di_l) |
                         ((unsigned)yi >= (unsigned)Hi_l) |
                         ((unsigned)xi >= (unsigned)Wi_l)) != 0;

                    const int8_t* inptr = out_of_bounds ? zbuf :
                        inpbaseptr + (((zi * Hi_l) + yi) * Wi_l + xi) * C0;
                    int inpstep = out_of_bounds ? 0 : iplanesize_l;
                    const int8_t* wptr = wbaseptr + (size_t)i * C1Max * K0 * C0;

                    for (int c1 = 0; c1 < cblocks; c1++, inptr += inpstep, wptr += C0 * K0) {
                        for (int c0 = 0; c0 < C0; c0++) {
                            int ival = (int)(int8_t)((uint8_t)inptr[c0] ^ xor_val);
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
                    if (rptr) {
                        int rval = inputIsU8 ? (int)(uint8_t)rptr[k0] : (int)rptr[k0];
                        val += (float)(rval - out_zp);
                    }
                    int ival = cvRound(val);
                    if (inputIsU8) {
                        ival = std::max(0, std::min(255, ival));
                        if (activLUT) ival = (int)(uint8_t)activLUT[ival];
                        ((uint8_t*)optr)[k0] = (uint8_t)ival;
                    } else {
                        ival = std::max(-128, std::min(127, ival));
                        if (activLUT) ival = (int)activLUT[ival + 128];
                        optr[k0] = (int8_t)ival;
                    }
                }
            }
        }
    });
}

#endif // !CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END
}} // namespace cv::dnn
