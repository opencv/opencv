// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "opencv2/core/hal/intrin.hpp"
#include "../layers/conv2_common.hpp"

namespace cv {
namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

// AVX-VNNI block convolution for the int8 engine. VPDPBUSD multiplies unsigned
// activations by signed weights, so this kernel consumes the uint8 input directly
// and needs the weights/bias reordered for VNNI (weightsVNNI_/biasVNNI).
void convInt8BlockVNNI(const void* inp_, const void* residual_,
                       void* out_, const ConvState& cs,
                       const void* weightsVNNI_,
                       const int* biasVNNI, const float* multiplier,
                       int inp_zp, int out_zp,
                       const int8_t* activLUT,
                       bool inputIsU8);

#if !defined(CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY) && CV_AVX_VNNI

static void convInt8BlockVNNI_kxk(const void* inp_, const void* residual_,
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
        bool padded = cs.hasPadding();
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

            const int k_valid = std::min(K0, K - k_base);

            alignas(32) int32_t biasbuf[K0];
            memcpy(biasbuf, biasVNNI + k_base, k_valid * sizeof(int32_t));
            memset(biasbuf + k_valid, 0, (K0 - k_valid) * sizeof(int32_t));

            alignas(32) float multbuf[K0];
            memcpy(multbuf, multiplier + k_base, k_valid * sizeof(float));
            memset(multbuf + k_valid, 0, (K0 - k_valid) * sizeof(float));

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

            if (ksize == 1 && Sx == 1 && Sy == 1 && Sz == 1 && !padded) {
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

static void convInt8BlockVNNI_1x1(const void* inp_, const void* residual_,
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

            const int k_valid = std::min(K0, K - k_base);

            alignas(32) int32_t biasbuf[K0];
            alignas(32) float   multbuf[K0];
            memcpy(biasbuf, biasVNNI   + k_base, k_valid * sizeof(int32_t));
            memset(biasbuf + k_valid, 0, (K0 - k_valid) * sizeof(int32_t));
            memcpy(multbuf, multiplier + k_base, k_valid * sizeof(float));
            memset(multbuf + k_valid, 0, (K0 - k_valid) * sizeof(float));

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

void convInt8BlockVNNI(const void* inp_, const void* residual_,
                       void* out_, const ConvState& cs,
                       const void* weightsVNNI_,
                       const int* biasVNNI, const float* multiplier,
                       int inp_zp, int out_zp,
                       const int8_t* activLUT,
                       bool inputIsU8)
{
    if (cs.wshape[2] == 1 && !cs.hasPadding() &&
        cs.strides[0] == 1 && cs.strides[1] == 1 && cs.strides[2] == 1) {
        convInt8BlockVNNI_1x1(inp_, residual_, out_, cs, weightsVNNI_,
                              biasVNNI, multiplier, inp_zp, out_zp, activLUT, inputIsU8);
        return;
    }
    convInt8BlockVNNI_kxk(inp_, residual_, out_, cs, weightsVNNI_,
                          biasVNNI, multiplier, inp_zp, out_zp, activLUT, inputIsU8);
}

#endif // !CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY && CV_AVX_VNNI

CV_CPU_OPTIMIZATION_NAMESPACE_END
}} // namespace cv::dnn
