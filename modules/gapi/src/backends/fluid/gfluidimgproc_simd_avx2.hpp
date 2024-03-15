// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2022 Intel Corporation

#if !defined(GAPI_STANDALONE)

#include "opencv2/gapi/own/saturate.hpp"

#include <immintrin.h>

#include "opencv2/core.hpp"

#include <opencv2/core/hal/intrin.hpp>

#include <cstdint>
#include <cstring>

#include <algorithm>
#include <limits>
#include <vector>

namespace cv {
namespace gapi {
namespace fluid {
namespace avx2 {

CV_ALWAYS_INLINE void v_gather_pairs(const float src[], const int* mapsx,
                                     v_float32x8& low, v_float32x8& high)
{
    low.val = _mm256_castsi256_ps(_mm256_setr_epi64x(*reinterpret_cast<const int64_t*>(&src[mapsx[0]]),
                                                     *reinterpret_cast<const int64_t*>(&src[mapsx[1]]),
                                                     *reinterpret_cast<const int64_t*>(&src[mapsx[2]]),
                                                     *reinterpret_cast<const int64_t*>(&src[mapsx[3]])));
    high.val = _mm256_castsi256_ps(_mm256_setr_epi64x(*reinterpret_cast<const int64_t*>(&src[mapsx[4]]),
                                                      *reinterpret_cast<const int64_t*>(&src[mapsx[5]]),
                                                      *reinterpret_cast<const int64_t*>(&src[mapsx[6]]),
                                                      *reinterpret_cast<const int64_t*>(&src[mapsx[7]])));
}

CV_ALWAYS_INLINE void v_deinterleave(const v_float32x8& low, const v_float32x8& high,
                                     v_float32x8& even,      v_float32x8& odd)
{
    __m256 tmp0 = _mm256_unpacklo_ps(low.val, high.val);
    __m256 tmp1 = _mm256_unpackhi_ps(low.val, high.val);
    __m256 tmp2 = _mm256_unpacklo_ps(tmp0, tmp1);
    __m256 tmp3 = _mm256_unpackhi_ps(tmp0, tmp1);
    even.val = _mm256_castsi256_ps(_mm256_permute4x64_epi64(_mm256_castps_si256(tmp2), 216 /*11011000*/));
    odd.val = _mm256_castsi256_ps(_mm256_permute4x64_epi64(_mm256_castps_si256(tmp3), 216 /*11011000*/));
}

// Resize (bi-linear, 32FC1)
CV_ALWAYS_INLINE void calcRowLinear32FC1Impl(float *dst[],
                                             const float *src0[],
                                             const float *src1[],
                                             const float  alpha[],
                                             const int    mapsx[],
                                             const float  beta[],
                                             const Size& inSz,
                                             const Size& outSz,
                                             const int   lpi)
{
    bool xRatioEq1 = inSz.width == outSz.width;
    bool yRatioEq1 = inSz.height == outSz.height;

    const int nlanes = VTraits<v_float32x8>::vlanes();

    if (!xRatioEq1 && !yRatioEq1)
    {
        for (int line = 0; line < lpi; ++line) {
            float beta0 = beta[line];
            float beta1 = 1 - beta0;
            v_float32x8 v_beta0 = v256_setall_f32(beta0);
            int x = 0;

            v_float32x8 low1, high1, s00, s01;
            v_float32x8 low2, high2, s10, s11;
            for (; x <= outSz.width - nlanes; x += nlanes)
            {
                v_float32x8 alpha0 = v256_load(&alpha[x]);
                //  v_float32 alpha1 = 1.f - alpha0;

                v_gather_pairs(src0[line], &mapsx[x], low1, high1);
                v_deinterleave(low1, high1, s00, s01);

                //  v_float32 res0 = s00*alpha0 + s01*alpha1;
                v_float32x8 res0 = v_fma(s00 - s01, alpha0, s01);

                v_gather_pairs(src1[line], &mapsx[x], low2, high2);
                v_deinterleave(low2, high2, s10, s11);

                //  v_float32 res1 = s10*alpha0 + s11*alpha1;
                v_float32x8 res1 = v_fma(s10 - s11, alpha0, s11);
                //  v_float32 d = res0*beta0 + res1*beta1;
                v_float32x8 d = v_fma(res0 - res1, v_beta0, res1);

                v_store(&dst[line][x], d);
            }

            for (; x < outSz.width; ++x)
            {
                float alpha0 = alpha[x];
                float alpha1 = 1 - alpha0;
                int   sx0 = mapsx[x];
                int   sx1 = sx0 + 1;
                float res0 = src0[line][sx0] * alpha0 + src0[line][sx1] * alpha1;
                float res1 = src1[line][sx0] * alpha0 + src1[line][sx1] * alpha1;
                dst[line][x] = beta0 * res0 + beta1 * res1;
            }
        }
    }
    else if (!xRatioEq1)
    {

        for (int line = 0; line < lpi; ++line) {
            int x = 0;

            v_float32x8 low, high, s00, s01;
            for (; x <= outSz.width - nlanes; x += nlanes)
            {
                v_float32x8 alpha0 = v256_load(&alpha[x]);
                //  v_float32 alpha1 = 1.f - alpha0;

                v_gather_pairs(src0[line], &mapsx[x], low, high);
                v_deinterleave(low, high, s00, s01);

                //  v_float32 d = s00*alpha0 + s01*alpha1;
                v_float32x8 d = v_fma(s00 - s01, alpha0, s01);

                v_store(&dst[line][x], d);
            }

            for (; x < outSz.width; ++x) {
                float alpha0 = alpha[x];
                float alpha1 = 1 - alpha0;
                int   sx0 = mapsx[x];
                int   sx1 = sx0 + 1;
                dst[line][x] = src0[line][sx0] * alpha0 + src0[line][sx1] * alpha1;
            }
        }

    }
    else if (!yRatioEq1)
    {
        int length = inSz.width;  // == outSz.width

        for (int line = 0; line < lpi; ++line) {
            float beta0 = beta[line];
            float beta1 = 1 - beta0;
            v_float32x8 v_beta0 = v256_setall_f32(beta0);
            int x = 0;

            for (; x <= length - nlanes; x += nlanes)
            {
                v_float32x8 s0 = v256_load(&src0[line][x]);
                v_float32x8 s1 = v256_load(&src1[line][x]);

                //  v_float32 d = s0*beta0 + s1*beta1;
                v_float32x8 d = v_fma(s0 - s1, v_beta0, s1);

                v_store(&dst[line][x], d);
            }

            for (; x < length; ++x) {
                dst[line][x] = beta0 * src0[line][x] + beta1 * src1[line][x];
            }
        }

    }
    else
    {
        int length = inSz.width;  // == outSz.width
        memcpy(dst[0], src0[0], length * sizeof(float)*lpi);
    }
}

CV_ALWAYS_INLINE void resize_horizontal_anyLPI(uint8_t* dst,
                                               const uchar* src, const short mapsx[],
                                               const short alpha[], const int width)
{
    constexpr int nlanes = 16;
    constexpr int chanNum = 3;
    __m128i zero = _mm_setzero_si128();

    for (int x = 0; width >= nlanes;)
    {
        for (; x <= width - nlanes; x += nlanes)
        {
            __m128i a012 = _mm_setr_epi16(alpha[x], alpha[x], alpha[x], alpha[x + 1],
                                          alpha[x + 1], alpha[x + 1], alpha[x + 2], alpha[x + 2]);
            __m128i a2345 = _mm_setr_epi16(alpha[x + 2], alpha[x + 3], alpha[x + 3], alpha[x + 3],
                                           alpha[x + 4], alpha[x + 4], alpha[x + 4], alpha[x + 5]);

            __m128i a567 = _mm_setr_epi16(alpha[x + 5], alpha[x + 5], alpha[x + 6], alpha[x + 6],
                                          alpha[x + 6], alpha[x + 7], alpha[x + 7], alpha[x + 7]);
            __m128i a8910 = _mm_setr_epi16(alpha[x + 8], alpha[x + 8], alpha[x + 8], alpha[x + 9],
                                           alpha[x + 9], alpha[x + 9], alpha[x + 10], alpha[x + 10]);

            __m128i a10111213 = _mm_setr_epi16(alpha[x + 10], alpha[x + 11], alpha[x + 11], alpha[x + 11],
                                               alpha[x + 12], alpha[x + 12], alpha[x + 12], alpha[x + 13]);
            __m128i a131415 = _mm_setr_epi16(alpha[x + 13], alpha[x + 13], alpha[x + 14], alpha[x + 14],
                                             alpha[x + 14], alpha[x + 15], alpha[x + 15], alpha[x + 15]);

            __m128i a1 = _mm_setr_epi8(src[chanNum * (mapsx[x] + 0)],     src[chanNum * (mapsx[x] + 0) + 1],     src[chanNum * (mapsx[x] + 0) + 2],
                                       src[chanNum * (mapsx[x + 1] + 0)], src[chanNum * (mapsx[x + 1] + 0) + 1], src[chanNum * (mapsx[x + 1] + 0) + 2],
                                       src[chanNum * (mapsx[x + 2] + 0)], src[chanNum * (mapsx[x + 2] + 0) + 1], src[chanNum * (mapsx[x + 2] + 0) + 2],
                                       src[chanNum * (mapsx[x + 3] + 0)], src[chanNum * (mapsx[x + 3] + 0) + 1], src[chanNum * (mapsx[x + 3] + 0) + 2],
                                       src[chanNum * (mapsx[x + 4] + 0)], src[chanNum * (mapsx[x + 4] + 0) + 1], src[chanNum * (mapsx[x + 4] + 0) + 2],
                                       src[chanNum * (mapsx[x + 5] + 0)]);
            __m128i b1 = _mm_setr_epi8(src[chanNum * (mapsx[x] + 1)],     src[chanNum * (mapsx[x] + 1) + 1],     src[chanNum * (mapsx[x] + 1) + 2],
                                       src[chanNum * (mapsx[x + 1] + 1)], src[chanNum * (mapsx[x + 1] + 1) + 1], src[chanNum * (mapsx[x + 1] + 1) + 2],
                                       src[chanNum * (mapsx[x + 2] + 1)], src[chanNum * (mapsx[x + 2] + 1) + 1], src[chanNum * (mapsx[x + 2] + 1) + 2],
                                       src[chanNum * (mapsx[x + 3] + 1)], src[chanNum * (mapsx[x + 3] + 1) + 1], src[chanNum * (mapsx[x + 3] + 1) + 2],
                                       src[chanNum * (mapsx[x + 4] + 1)], src[chanNum * (mapsx[x + 4] + 1) + 1], src[chanNum * (mapsx[x + 4] + 1) + 2],
                                       src[chanNum * (mapsx[x + 5] + 1)]);

            __m128i a2 = _mm_setr_epi8(src[chanNum * (mapsx[x + 5] + 0) + 1], src[chanNum * (mapsx[x + 5] + 0) + 2], src[chanNum * (mapsx[x + 6] + 0)],
                                       src[chanNum * (mapsx[x + 6] + 0) + 1], src[chanNum * (mapsx[x + 6] + 0) + 2], src[chanNum * (mapsx[x + 7] + 0)],
                                       src[chanNum * (mapsx[x + 7] + 0) + 1], src[chanNum * (mapsx[x + 7] + 0) + 2], src[chanNum * (mapsx[x + 8] + 0)],
                                       src[chanNum * (mapsx[x + 8] + 0) + 1], src[chanNum * (mapsx[x + 8] + 0) + 2], src[chanNum * (mapsx[x + 9] + 0)],
                                       src[chanNum * (mapsx[x + 9] + 0) + 1], src[chanNum * (mapsx[x + 9] + 0) + 2], src[chanNum * (mapsx[x + 10] + 0)],
                                       src[chanNum * (mapsx[x + 10] + 0) + 1]);

            __m128i b2 = _mm_setr_epi8(src[chanNum * (mapsx[x + 5] + 1) + 1], src[chanNum * (mapsx[x + 5] + 1) + 2], src[chanNum * (mapsx[x + 6] + 1)],
                                       src[chanNum * (mapsx[x + 6] + 1) + 1], src[chanNum * (mapsx[x + 6] + 1) + 2], src[chanNum * (mapsx[x + 7] + 1)],
                                       src[chanNum * (mapsx[x + 7] + 1) + 1], src[chanNum * (mapsx[x + 7] + 1) + 2], src[chanNum * (mapsx[x + 8] + 1)],
                                       src[chanNum * (mapsx[x + 8] + 1) + 1], src[chanNum * (mapsx[x + 8] + 1) + 2], src[chanNum * (mapsx[x + 9] + 1)],
                                       src[chanNum * (mapsx[x + 9] + 1) + 1], src[chanNum * (mapsx[x + 9] + 1) + 2], src[chanNum * (mapsx[x + 10] + 1)],
                                       src[chanNum * (mapsx[x + 10] + 1) + 1]);

            __m128i a3 = _mm_setr_epi8(src[chanNum * (mapsx[x + 10] + 0) + 2], src[chanNum * (mapsx[x + 11] + 0)], src[chanNum * (mapsx[x + 11] + 0) + 1],
                                       src[chanNum * (mapsx[x + 11] + 0) + 2], src[chanNum * (mapsx[x + 12] + 0)], src[chanNum * (mapsx[x + 12] + 0) + 1],
                                       src[chanNum * (mapsx[x + 12] + 0) + 2], src[chanNum * (mapsx[x + 13] + 0)], src[chanNum * (mapsx[x + 13] + 0) + 1],
                                       src[chanNum * (mapsx[x + 13] + 0) + 2], src[chanNum * (mapsx[x + 14] + 0)], src[chanNum * (mapsx[x + 14] + 0) + 1],
                                       src[chanNum * (mapsx[x + 14] + 0) + 2], src[chanNum * (mapsx[x + 15] + 0)], src[chanNum * (mapsx[x + 15] + 0) + 1],
                                       src[chanNum * (mapsx[x + 15] + 0) + 2]);

            __m128i b3 = _mm_setr_epi8(src[chanNum * (mapsx[x + 10] + 1) + 2], src[chanNum * (mapsx[x + 11] + 1)], src[chanNum * (mapsx[x + 11] + 1) + 1],
                                       src[chanNum * (mapsx[x + 11] + 1) + 2], src[chanNum * (mapsx[x + 12] + 1)], src[chanNum * (mapsx[x + 12] + 1) + 1],
                                       src[chanNum * (mapsx[x + 12] + 1) + 2], src[chanNum * (mapsx[x + 13] + 1)], src[chanNum * (mapsx[x + 13] + 1) + 1],
                                       src[chanNum * (mapsx[x + 13] + 1) + 2], src[chanNum * (mapsx[x + 14] + 1)], src[chanNum * (mapsx[x + 14] + 1) + 1],
                                       src[chanNum * (mapsx[x + 14] + 1) + 2], src[chanNum * (mapsx[x + 15] + 1)], src[chanNum * (mapsx[x + 15] + 1) + 1],
                                       src[chanNum * (mapsx[x + 15] + 1) + 2]);

            __m128i a11 = _mm_unpacklo_epi8(a1, zero);
            __m128i a12 = _mm_unpackhi_epi8(a1, zero);
            __m128i a21 = _mm_unpacklo_epi8(a2, zero);
            __m128i a22 = _mm_unpackhi_epi8(a2, zero);
            __m128i a31 = _mm_unpacklo_epi8(a3, zero);
            __m128i a32 = _mm_unpackhi_epi8(a3, zero);
            __m128i b11 = _mm_unpacklo_epi8(b1, zero);
            __m128i b12 = _mm_unpackhi_epi8(b1, zero);
            __m128i b21 = _mm_unpacklo_epi8(b2, zero);
            __m128i b22 = _mm_unpackhi_epi8(b2, zero);
            __m128i b31 = _mm_unpacklo_epi8(b3, zero);
            __m128i b32 = _mm_unpackhi_epi8(b3, zero);

            __m128i r1 = _mm_mulhrs_epi16(_mm_sub_epi16(a11, b11), a012);
            __m128i r2 = _mm_mulhrs_epi16(_mm_sub_epi16(a12, b12), a2345);
            __m128i r3 = _mm_mulhrs_epi16(_mm_sub_epi16(a21, b21), a567);
            __m128i r4 = _mm_mulhrs_epi16(_mm_sub_epi16(a22, b22), a8910);
            __m128i r5 = _mm_mulhrs_epi16(_mm_sub_epi16(a31, b31), a10111213);
            __m128i r6 = _mm_mulhrs_epi16(_mm_sub_epi16(a32, b32), a131415);

            __m128i r_1 = _mm_add_epi16(b11, r1);
            __m128i r_2 = _mm_add_epi16(b12, r2);
            __m128i r_3 = _mm_add_epi16(b21, r3);
            __m128i r_4 = _mm_add_epi16(b22, r4);
            __m128i r_5 = _mm_add_epi16(b31, r5);
            __m128i r_6 = _mm_add_epi16(b32, r6);

            __m128i res1 = _mm_packus_epi16(r_1, r_2);
            __m128i res2 = _mm_packus_epi16(r_3, r_4);
            __m128i res3 = _mm_packus_epi16(r_5, r_6);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[chanNum * x]), res1);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[chanNum * x + 16]), res2);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[chanNum * x + 32]), res3);
        }
        if (x < width) {
            x = width - nlanes;
            continue;
        }
        break;
    }
}

CV_ALWAYS_INLINE void verticalPass_lpi4_8U(const uint8_t* src0[], const uint8_t* src1[],
                                           uint8_t tmp[], const short beta[],
                                           const int& length)
{
    constexpr int half_nlanes = (v_uint8::nlanes / 2);
    GAPI_DbgAssert(length >= half_nlanes);

    __m256i b0 = _mm256_set1_epi16(beta[0]);
    __m256i b1 = _mm256_set1_epi16(beta[1]);
    __m256i b2 = _mm256_set1_epi16(beta[2]);
    __m256i b3 = _mm256_set1_epi16(beta[3]);

    __m256i shuf_mask = _mm256_setr_epi8(0, 8, 4, 12, 1, 9, 5, 13,
                                         2, 10, 6, 14, 3, 11, 7, 15,
                                         0, 8, 4, 12, 1, 9, 5, 13,
                                         2, 10, 6, 14, 3, 11, 7, 15);
    for (int w = 0; w < length; ) {
        for (; w <= length - half_nlanes; w += half_nlanes)
        {
            __m256i val0_0 = _mm256_cvtepu8_epi16(_mm_lddqu_si128(reinterpret_cast<const __m128i*>(&src0[0][w])));
            __m256i val0_1 = _mm256_cvtepu8_epi16(_mm_lddqu_si128(reinterpret_cast<const __m128i*>(&src0[1][w])));
            __m256i val0_2 = _mm256_cvtepu8_epi16(_mm_lddqu_si128(reinterpret_cast<const __m128i*>(&src0[2][w])));
            __m256i val0_3 = _mm256_cvtepu8_epi16(_mm_lddqu_si128(reinterpret_cast<const __m128i*>(&src0[3][w])));

            __m256i val1_0 = _mm256_cvtepu8_epi16(_mm_lddqu_si128(reinterpret_cast<const __m128i*>(&src1[0][w])));
            __m256i val1_1 = _mm256_cvtepu8_epi16(_mm_lddqu_si128(reinterpret_cast<const __m128i*>(&src1[1][w])));
            __m256i val1_2 = _mm256_cvtepu8_epi16(_mm_lddqu_si128(reinterpret_cast<const __m128i*>(&src1[2][w])));
            __m256i val1_3 = _mm256_cvtepu8_epi16(_mm_lddqu_si128(reinterpret_cast<const __m128i*>(&src1[3][w])));

            __m256i t0 = _mm256_mulhrs_epi16(_mm256_sub_epi16(val0_0, val1_0), b0);
            __m256i t1 = _mm256_mulhrs_epi16(_mm256_sub_epi16(val0_1, val1_1), b1);
            __m256i t2 = _mm256_mulhrs_epi16(_mm256_sub_epi16(val0_2, val1_2), b2);
            __m256i t3 = _mm256_mulhrs_epi16(_mm256_sub_epi16(val0_3, val1_3), b3);

            __m256i r0 = _mm256_add_epi16(val1_0, t0);
            __m256i r1 = _mm256_add_epi16(val1_1, t1);
            __m256i r2 = _mm256_add_epi16(val1_2, t2);
            __m256i r3 = _mm256_add_epi16(val1_3, t3);

            __m256i q0 = _mm256_packus_epi16(r0, r1);
            __m256i q1 = _mm256_packus_epi16(r2, r3);

            __m256i q2 = _mm256_blend_epi16(q0, _mm256_slli_si256(q1, 4), 0xCC /*0b11001100*/);
            __m256i q3 = _mm256_blend_epi16(_mm256_srli_si256(q0, 4), q1, 0xCC /*0b11001100*/);

            __m256i q4 = _mm256_shuffle_epi8(q2, shuf_mask);
            __m256i q5 = _mm256_shuffle_epi8(q3, shuf_mask);

            __m256i q6 = _mm256_permute2x128_si256(q4, q5, 0x20);
            __m256i q7 = _mm256_permute2x128_si256(q4, q5, 0x31);

            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&tmp[4 * w + 0]), q6);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&tmp[4 * w + 2 * half_nlanes]), q7);
        }

        if (w < length)
        {
            w = length - half_nlanes;
        }
    }
}

CV_ALWAYS_INLINE void verticalPass_anylpi_8U(const uint8_t* src0[], const uint8_t* src1[],
                                             uint8_t tmp[], const int& beta0,
                                             const int l, const int length1, const int length2) {
    constexpr int half_nlanes = (v_uint8::nlanes / 2);
    GAPI_DbgAssert(length1 >= half_nlanes);

    for (int w = 0; w < length2; ) {
        for (; w <= length1 - half_nlanes; w += half_nlanes) {
            v_int16x16 s0 = v_reinterpret_as_s16(vx_load_expand(&src0[l][w]));
            v_int16x16 s1 = v_reinterpret_as_s16(vx_load_expand(&src1[l][w]));
            v_int16x16 t = v_mulhrs(s0 - s1, beta0) + s1;
            v_pack_u_store(tmp + w, t);
        }

        if (w < length1) {
            w = length1 - half_nlanes;
        }
    }
}

CV_ALWAYS_INLINE v_int16x16 v_gather_chan(const uchar src[], const v_int16x16& index, int channel, int pos)
{
    constexpr int chanNum = 3;
    v_int16x16 r;
    r.val = _mm256_insert_epi16(r.val, *reinterpret_cast<const uchar*>(&src[chanNum * (_mm256_extract_epi16(index.val, 0) + pos) + channel]), 0);
    r.val = _mm256_insert_epi16(r.val, *reinterpret_cast<const uchar*>(&src[chanNum * (_mm256_extract_epi16(index.val, 1) + pos) + channel]), 1);
    r.val = _mm256_insert_epi16(r.val, *reinterpret_cast<const uchar*>(&src[chanNum * (_mm256_extract_epi16(index.val, 2) + pos) + channel]), 2);
    r.val = _mm256_insert_epi16(r.val, *reinterpret_cast<const uchar*>(&src[chanNum * (_mm256_extract_epi16(index.val, 3) + pos) + channel]), 3);
    r.val = _mm256_insert_epi16(r.val, *reinterpret_cast<const uchar*>(&src[chanNum * (_mm256_extract_epi16(index.val, 4) + pos) + channel]), 4);
    r.val = _mm256_insert_epi16(r.val, *reinterpret_cast<const uchar*>(&src[chanNum * (_mm256_extract_epi16(index.val, 5) + pos) + channel]), 5);
    r.val = _mm256_insert_epi16(r.val, *reinterpret_cast<const uchar*>(&src[chanNum * (_mm256_extract_epi16(index.val, 6) + pos) + channel]), 6);
    r.val = _mm256_insert_epi16(r.val, *reinterpret_cast<const uchar*>(&src[chanNum * (_mm256_extract_epi16(index.val, 7) + pos) + channel]), 7);
    r.val = _mm256_insert_epi16(r.val, *reinterpret_cast<const uchar*>(&src[chanNum * (_mm256_extract_epi16(index.val, 8) + pos) + channel]), 8);
    r.val = _mm256_insert_epi16(r.val, *reinterpret_cast<const uchar*>(&src[chanNum * (_mm256_extract_epi16(index.val, 9) + pos) + channel]), 9);
    r.val = _mm256_insert_epi16(r.val, *reinterpret_cast<const uchar*>(&src[chanNum * (_mm256_extract_epi16(index.val, 10) + pos) + channel]), 10);
    r.val = _mm256_insert_epi16(r.val, *reinterpret_cast<const uchar*>(&src[chanNum * (_mm256_extract_epi16(index.val, 11) + pos) + channel]), 11);
    r.val = _mm256_insert_epi16(r.val, *reinterpret_cast<const uchar*>(&src[chanNum * (_mm256_extract_epi16(index.val, 12) + pos) + channel]), 12);
    r.val = _mm256_insert_epi16(r.val, *reinterpret_cast<const uchar*>(&src[chanNum * (_mm256_extract_epi16(index.val, 13) + pos) + channel]), 13);
    r.val = _mm256_insert_epi16(r.val, *reinterpret_cast<const uchar*>(&src[chanNum * (_mm256_extract_epi16(index.val, 14) + pos) + channel]), 14);
    r.val = _mm256_insert_epi16(r.val, *reinterpret_cast<const uchar*>(&src[chanNum * (_mm256_extract_epi16(index.val, 15) + pos) + channel]), 15);
    return r;
}

CV_ALWAYS_INLINE void v_gather_channel(v_uint8x32& vec, const uint8_t tmp[], const short mapsx[],
                                       int chanNum, int c, int x, int shift)
{
    vec.val = _mm256_insert_epi32(vec.val, *reinterpret_cast<const int*>(&tmp[4 * (chanNum *  mapsx[x + shift + 0] + c)]), 0);
    vec.val = _mm256_insert_epi32(vec.val, *reinterpret_cast<const int*>(&tmp[4 * (chanNum *  mapsx[x + shift + 1] + c)]), 1);
    vec.val = _mm256_insert_epi32(vec.val, *reinterpret_cast<const int*>(&tmp[4 * (chanNum *  mapsx[x + shift + 2] + c)]), 2);
    vec.val = _mm256_insert_epi32(vec.val, *reinterpret_cast<const int*>(&tmp[4 * (chanNum *  mapsx[x + shift + 3] + c)]), 3);

    vec.val = _mm256_insert_epi32(vec.val, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * (mapsx[x + shift + 0] + 1) + c)]), 4);
    vec.val = _mm256_insert_epi32(vec.val, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * (mapsx[x + shift + 1] + 1) + c)]), 5);
    vec.val = _mm256_insert_epi32(vec.val, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * (mapsx[x + shift + 2] + 1) + c)]), 6);
    vec.val = _mm256_insert_epi32(vec.val, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * (mapsx[x + shift + 3] + 1) + c)]), 7);
}

CV_ALWAYS_INLINE v_int16x16 v_mulhrs(const v_int16x16& a, const v_int16x16& b)
{
    return v_int16x16(_mm256_mulhrs_epi16(a.val, b.val));
}

static inline v_int16x16 v_mulhrs(const v_int16x16& a, short b)
{
    return v_mulhrs(a, v256_setall_s16(b));
}

CV_ALWAYS_INLINE void calcRowLinear_8UC_Impl(uint8_t* dst[],
                                             const uint8_t* src0[],
                                             const uint8_t* src1[],
                                             const short    alpha[],
                                             const short*   clone,  // 4 clones of alpha
                                             const short    mapsx[],
                                             const short    beta[],
                                                 uint8_t    tmp[],
                                             const Size&    inSz,
                                             const Size&    outSz,
                                             const int      lpi)
{
    constexpr int nlanes = 32; // number of 8-bit integers that fit into a 256-bit SIMD vector.
    constexpr int half_nlanes = nlanes / 2;
    constexpr int chanNum = 3;

    if ((inSz.width * chanNum < half_nlanes) || (outSz.width < half_nlanes))
        return;

    const int shift = (half_nlanes / 4);

    if (4 == lpi)
    {
        verticalPass_lpi4_8U(src0, src1, tmp, beta,
                             inSz.width*chanNum);

        // horizontal pass
        __m256i shuff_mask = _mm256_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
                                              0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
        __m256i perm_mask = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

        constexpr int nproc_pixels = 5;
        for (int x = 0;;)
        {
            for (; x <= outSz.width - (nproc_pixels + 1); x += nproc_pixels)
            {
                v_int16x16 a10 = vx_load(&clone[4 * x]);
                v_int16x16 a32 = vx_load(&clone[4 * (x + 4)]);
                v_int16x16 a54 = vx_load(&clone[4 * (x + 8)]);
                v_int16x16 a76 = vx_load(&clone[4 * (x + 12)]);

                __m128i pix1 = _mm_setzero_si128();
                pix1 = _mm_insert_epi64(pix1, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * mapsx[x])    ]), 0);
                pix1 = _mm_insert_epi32(pix1, *reinterpret_cast<const int*>(    &tmp[4 * (chanNum * mapsx[x]) + 8]), 2);

                __m128i pix2 = _mm_setzero_si128();
                pix2 = _mm_insert_epi64(pix2, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * (mapsx[x] + 1))    ]), 0);
                pix2 = _mm_insert_epi32(pix2, *reinterpret_cast<const int*>(    &tmp[4 * (chanNum * (mapsx[x] + 1)) + 8]), 2);

                __m128i pix3 = _mm_setzero_si128();
                pix3 = _mm_insert_epi64(pix3, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * mapsx[x + 1])    ]), 0);
                pix3 = _mm_insert_epi32(pix3, *reinterpret_cast<const int*>(    &tmp[4 * (chanNum * mapsx[x + 1]) + 8]), 2);

                __m128i pix4 = _mm_setzero_si128();
                pix4 = _mm_insert_epi64(pix4, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * (mapsx[x + 1] + 1))    ]), 0);
                pix4 = _mm_insert_epi32(pix4, *reinterpret_cast<const int*>(    &tmp[4 * (chanNum * (mapsx[x + 1] + 1)) + 8]), 2);

                __m256i ext_pix1 = _mm256_cvtepi8_epi16(pix1);
                __m256i ext_pix2 = _mm256_cvtepi8_epi16(pix2);
                __m256i ext_pix3 = _mm256_cvtepi8_epi16(pix3);
                __m256i ext_pix4 = _mm256_cvtepi8_epi16(pix4);

                __m256i t0_0 = _mm256_mulhrs_epi16(_mm256_sub_epi16(ext_pix1, ext_pix2), a00);
                __m256i t1_0 = _mm256_mulhrs_epi16(_mm256_sub_epi16(ext_pix3, ext_pix4), a11);

                __m256i r0_0 = _mm256_add_epi16(ext_pix2, t0_0);
                __m256i r1_0 = _mm256_add_epi16(ext_pix4, t1_0);

                __m256i q0_0 = _mm256_packus_epi16(r0_0, r1_0);

                __m256i perm64 = _mm256_permute4x64_epi64(q0_0, 216 /*11011000*/);
                __m256i shuf = _mm256_shuffle_epi8(perm64, shuff_mask);

                __m256i res1 = _mm256_permutevar8x32_epi32(shuf, perm_mask);

                pix1 = _mm_insert_epi64(pix1, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * mapsx[x + 2])    ]), 0);
                pix1 = _mm_insert_epi32(pix1, *reinterpret_cast<const int*>(    &tmp[4 * (chanNum * mapsx[x + 2]) + 8]), 2);

                pix2 = _mm_insert_epi64(pix2, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * (mapsx[x + 2] + 1))    ]), 0);
                pix2 = _mm_insert_epi32(pix2, *reinterpret_cast<const int*>(    &tmp[4 * (chanNum * (mapsx[x + 2] + 1)) + 8]), 2);

                pix3 = _mm_insert_epi64(pix3, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * mapsx[x + 3])    ]), 0);
                pix3 = _mm_insert_epi32(pix3, *reinterpret_cast<const int*>(    &tmp[4 * (chanNum * mapsx[x + 3]) + 8]), 2);

                pix4 = _mm_insert_epi64(pix4, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * (mapsx[x + 3] + 1))    ]), 0);
                pix4 = _mm_insert_epi32(pix4, *reinterpret_cast<const int*>(    &tmp[4 * (chanNum * (mapsx[x + 3] + 1)) + 8]), 2);

                ext_pix1 = _mm256_cvtepi8_epi16(pix1);
                ext_pix2 = _mm256_cvtepi8_epi16(pix2);
                ext_pix3 = _mm256_cvtepi8_epi16(pix3);
                ext_pix4 = _mm256_cvtepi8_epi16(pix4);

                t0_0 = _mm256_mulhrs_epi16(_mm256_sub_epi16(ext_pix1, ext_pix2), a22);
                t1_0 = _mm256_mulhrs_epi16(_mm256_sub_epi16(ext_pix3, ext_pix4), a33);

                r0_0 = _mm256_add_epi16(ext_pix2, t0_0);
                r1_0 = _mm256_add_epi16(ext_pix4, t1_0);

                q0_0 = _mm256_packus_epi16(r0_0, r1_0);

                perm64 = _mm256_permute4x64_epi64(q0_0, 216 /*11011000*/);
                shuf = _mm256_shuffle_epi8(perm64, shuff_mask);

                __m256i res2 = _mm256_permutevar8x32_epi32(shuf, perm_mask);

                pix1 = _mm_insert_epi64(pix1, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * mapsx[x + 4])]), 0);
                pix1 = _mm_insert_epi32(pix1, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * mapsx[x + 4]) + 8]), 2);

                pix2 = _mm_insert_epi64(pix2, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * (mapsx[x + 4] + 1))]), 0);
                pix2 = _mm_insert_epi32(pix2, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * (mapsx[x + 4] + 1)) + 8]), 2);

                pix3 = _mm_insert_epi64(pix3, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * mapsx[x + 5])]), 0);
                pix3 = _mm_insert_epi32(pix3, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * mapsx[x + 5]) + 8]), 2);

                pix4 = _mm_insert_epi64(pix4, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * (mapsx[x + 5] + 1))]), 0);
                pix4 = _mm_insert_epi32(pix4, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * (mapsx[x + 5] + 1)) + 8]), 2);

                ext_pix1 = _mm256_cvtepi8_epi16(pix1);
                ext_pix2 = _mm256_cvtepi8_epi16(pix2);
                ext_pix3 = _mm256_cvtepi8_epi16(pix3);
                ext_pix4 = _mm256_cvtepi8_epi16(pix4);

                t0_0 = _mm256_mulhrs_epi16(_mm256_sub_epi16(ext_pix1, ext_pix2), a44);
                t1_0 = _mm256_mulhrs_epi16(_mm256_sub_epi16(ext_pix3, ext_pix4), a55);

                r0_0 = _mm256_add_epi16(ext_pix2, t0_0);
                r1_0 = _mm256_add_epi16(ext_pix4, t1_0);

                q0_0 = _mm256_packus_epi16(r0_0, r1_0);

                perm64 = _mm256_permute4x64_epi64(q0_0, 216 /*11011000*/);
                shuf = _mm256_shuffle_epi8(perm64, shuff_mask);

                __m256i res3 = _mm256_permutevar8x32_epi32(shuf, perm_mask);

                pix1 = _mm_insert_epi64(pix1, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * mapsx[x + 6])]), 0);
                pix1 = _mm_insert_epi32(pix1, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * mapsx[x + 6]) + 8]), 2);

                pix2 = _mm_insert_epi64(pix2, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * (mapsx[x + 6] + 1))]), 0);
                pix2 = _mm_insert_epi32(pix2, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * (mapsx[x + 6] + 1)) + 8]), 2);

                pix3 = _mm_insert_epi64(pix3, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * mapsx[x + 7])]), 0);
                pix3 = _mm_insert_epi32(pix3, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * mapsx[x + 7]) + 8]), 2);

                pix4 = _mm_insert_epi64(pix4, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * (mapsx[x + 7] + 1))]), 0);
                pix4 = _mm_insert_epi32(pix4, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * (mapsx[x + 7] + 1)) + 8]), 2);

                ext_pix1 = _mm256_cvtepi8_epi16(pix1);
                ext_pix2 = _mm256_cvtepi8_epi16(pix2);
                ext_pix3 = _mm256_cvtepi8_epi16(pix3);
                ext_pix4 = _mm256_cvtepi8_epi16(pix4);

                t0_0 = _mm256_mulhrs_epi16(_mm256_sub_epi16(ext_pix1, ext_pix2), a66);
                t1_0 = _mm256_mulhrs_epi16(_mm256_sub_epi16(ext_pix3, ext_pix4), a77);

                r0_0 = _mm256_add_epi16(ext_pix2, t0_0);
                r1_0 = _mm256_add_epi16(ext_pix4, t1_0);

                q0_0 = _mm256_packus_epi16(r0_0, r1_0);

                perm64 = _mm256_permute4x64_epi64(q0_0, 216 /*11011000*/);
                shuf = _mm256_shuffle_epi8(perm64, shuff_mask);

                __m256i res4 = _mm256_permutevar8x32_epi32(shuf, perm_mask);

                pix1 = _mm_insert_epi64(pix1, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * mapsx[x + 8])]), 0);
                pix1 = _mm_insert_epi32(pix1, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * mapsx[x + 8]) + 8]), 2);

                pix2 = _mm_insert_epi64(pix2, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * (mapsx[x + 8] + 1))]), 0);
                pix2 = _mm_insert_epi32(pix2, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * (mapsx[x + 8] + 1)) + 8]), 2);

                pix3 = _mm_insert_epi64(pix3, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * mapsx[x + 9])]), 0);
                pix3 = _mm_insert_epi32(pix3, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * mapsx[x + 9]) + 8]), 2);

                pix4 = _mm_insert_epi64(pix4, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * (mapsx[x + 9] + 1))]), 0);
                pix4 = _mm_insert_epi32(pix4, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * (mapsx[x + 9] + 1)) + 8]), 2);

                ext_pix1 = _mm256_cvtepi8_epi16(pix1);
                ext_pix2 = _mm256_cvtepi8_epi16(pix2);
                ext_pix3 = _mm256_cvtepi8_epi16(pix3);
                ext_pix4 = _mm256_cvtepi8_epi16(pix4);

                t0_0 = _mm256_mulhrs_epi16(_mm256_sub_epi16(ext_pix1, ext_pix2), a88);
                t1_0 = _mm256_mulhrs_epi16(_mm256_sub_epi16(ext_pix3, ext_pix4), a99);

                r0_0 = _mm256_add_epi16(ext_pix2, t0_0);
                r1_0 = _mm256_add_epi16(ext_pix4, t1_0);

                q0_0 = _mm256_packus_epi16(r0_0, r1_0);

                perm64 = _mm256_permute4x64_epi64(q0_0, 216 /*11011000*/);
                shuf = _mm256_shuffle_epi8(perm64, shuff_mask);

                __m256i res5 = _mm256_permutevar8x32_epi32(shuf, perm_mask);

                __m256i bl1 = _mm256_blend_epi16(res1, _mm256_slli_si256(res2, 8), 240 /*0b11110000*/);
                __m256i bl2 = _mm256_blend_epi16(_mm256_srli_si256(res1, 8), res2, 240 /*0b11110000*/);

                __m256i bl3 = _mm256_blend_epi16(res3, _mm256_slli_si256(res4, 8), 240 /*0b11110000*/);
                __m256i bl4 = _mm256_blend_epi16(_mm256_srli_si256(res3, 8), res4, 240 /*0b11110000*/);

                __m256i perm1 = _mm256_permute2x128_si256(bl1, bl3, 32);
                __m256i perm1 = _mm256_permute2x128_si256(bl2, bl4, 32);

                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dst[0][chanNum * x]), bl1);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dst[1][chanNum * x]), bl2);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dst[2][chanNum * x]), bl3);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dst[3][chanNum * x]), bl4);
            }

            for (; x < outSz.width; ++x)
                {
                    constexpr static const int ONE = 1 << 15;
                    constexpr static const int half = 1 << 14;
                    auto alpha0 = alpha[x];
                    auto alpha1 = saturate_cast<short>(ONE - alpha[x]);

                    for (int c = 0; c < chanNum; ++c)
                    {
                        dst[0][chanNum * x + c] = (tmp[4 * (chanNum *  mapsx[x]      + c)    ] * alpha0 +
                                                   tmp[4 * (chanNum * (mapsx[x] + 1) + c)    ] * alpha1 + half) >> 15;
                        dst[1][chanNum * x + c] = (tmp[4 * (chanNum *  mapsx[x]      + c) + 1] * alpha0 +
                                                   tmp[4 * (chanNum * (mapsx[x] + 1) + c) + 1] * alpha1 + half) >> 15;
                        dst[2][chanNum * x + c] = (tmp[4 * (chanNum *  mapsx[x]      + c) + 2] * alpha0 +
                                                   tmp[4 * (chanNum * (mapsx[x] + 1) + c) + 2] * alpha1 + half) >> 15;
                        dst[3][chanNum * x + c] = (tmp[4 * (chanNum *  mapsx[x]      + c) + 3] * alpha0 +
                                                   tmp[4 * (chanNum * (mapsx[x] + 1) + c) + 3] * alpha1 + half) >> 15;
                    }
                }

                break;
        }
    }
    else
    {  // if any lpi
        for (int l = 0; l < lpi; ++l) {
            short beta0 = beta[l];

            // vertical pass
            verticalPass_anylpi_8U(src0, src1, tmp, beta0, l,
                                   inSz.width*chanNum, inSz.width*chanNum);

            // horizontal pass
            for (int x = 0; x < outSz.width; ) {
                for (; x <= outSz.width - half_nlanes && x >= 0; x += half_nlanes) {
                    for (int c = 0; c < chanNum; ++c) {
                        v_int16x16 a0 = v256_load(&alpha[x]);        // as signed Q1.1.14
                        v_int16x16 sx = v256_load(&mapsx[x]);        // as integer (int16)
                        v_int16x16 t0 = v_gather_chan(tmp, sx, c, 0);
                        v_int16x16 t1 = v_gather_chan(tmp, sx, c, 1);
                        v_int16x16 d = v_mulhrs(t0 - t1, a0) + t1;
                        v_pack_u_store(&dst[l][x], d);
                    }
                }

                if (x < outSz.width) {
                    x = outSz.width - half_nlanes;
                }
            }
        }
    }
    return;
}

} // namespace avx2
} // namespace fliud
} // namespace gapi
} // namespace cv
#endif // !defined(GAPI_STANDALONE)
