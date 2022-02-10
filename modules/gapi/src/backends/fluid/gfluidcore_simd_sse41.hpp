// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#if !defined(GAPI_STANDALONE)

#include "opencv2/gapi/own/saturate.hpp"

#include <smmintrin.h>

#include "opencv2/core.hpp"

#include <opencv2/core/hal/intrin.hpp>

#include <cstdint>
#include <cstring>

#include <algorithm>
#include <limits>
#include <vector>

#if defined __GNUC__
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wstrict-overflow"
#endif
namespace cv {
namespace gapi {
namespace fluid {
namespace sse42 {

CV_ALWAYS_INLINE void v_gather_pixel_map(v_uint8x16& vec, const uchar src[], const short* index, const int pos)
{
    const int chanNum = 4;

    // pixel_1 (rgbx)
    vec.val = _mm_insert_epi32(vec.val, *reinterpret_cast<const int*>(&src[chanNum * (*index + pos)]), 0);
    // pixel_2 (rgbx)
    vec.val = _mm_insert_epi32(vec.val, *reinterpret_cast<const int*>(&src[chanNum * (*(index + 1) + pos)]), 1);
    // pixel_3
    vec.val = _mm_insert_epi32(vec.val, *reinterpret_cast<const int*>(&src[chanNum * (*(index + 2) + pos)]), 2);
    // pixel_4
    vec.val = _mm_insert_epi32(vec.val, *reinterpret_cast<const int*>(&src[chanNum * (*(index + 3) + pos)]), 3);
}

CV_ALWAYS_INLINE void resize_vertical_anyLPI(const uchar* src0, const uchar* src1,
                                             uchar* dst, const int inLength,
                                             const short beta) {
    constexpr int nlanes = 16;
    __m128i zero = _mm_setzero_si128();
    __m128i b = _mm_set1_epi16(beta);

    for (int w = 0; inLength >= nlanes;)
    {
        for (; w <= inLength - nlanes; w += nlanes)
        {
            __m128i s0 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(&src0[w]));
            __m128i s1 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(&src1[w]));
            __m128i a1 = _mm_unpacklo_epi8(s0, zero);
            __m128i b1 = _mm_unpacklo_epi8(s1, zero);
            __m128i a2 = _mm_unpackhi_epi8(s0, zero);
            __m128i b2 = _mm_unpackhi_epi8(s1, zero);
            __m128i r1 = _mm_mulhrs_epi16(_mm_sub_epi16(a1, b1), b);
            __m128i r2 = _mm_mulhrs_epi16(_mm_sub_epi16(a2, b2), b);
            __m128i res1 = _mm_add_epi16(r1, b1);
            __m128i res2 = _mm_add_epi16(r2, b2);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + w), _mm_packus_epi16(res1, res2));
        }

        if (w < inLength) {
            w = inLength - nlanes;
            continue;
        }
        break;
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

template<int chanNum>
CV_ALWAYS_INLINE void calcRowLinear_8UC_Impl_(uint8_t**,
                                              const uint8_t**,
                                              const uint8_t**,
                                              const short* ,
                                              const short* ,
                                              const short*,
                                              const short* ,
                                                  uint8_t*,
                                              const Size& ,
                                              const Size& ,
                                              const int )
{
    static_assert(chanNum != 3, "Unsupported number of channel");
}
template<>
CV_ALWAYS_INLINE void calcRowLinear_8UC_Impl_<3>(uint8_t* dst[],
                                              const uint8_t* src0[],
                                              const uint8_t* src1[],
                                              const short    alpha[],
                                              const short*   clone,  // 4 clones of alpha
                                              const short    mapsx[],
                                              const short    beta[],
                                                  uint8_t    tmp[],
                                              const Size&    inSz,
                                              const Size&    outSz,
                                              const int      lpi) {
    bool xRatioEq = inSz.width == outSz.width;
    bool yRatioEq = inSz.height == outSz.height;
    constexpr int nlanes = 16;
    constexpr int half_nlanes = 16 / 2;
    constexpr int chanNum = 3;

    if (!xRatioEq && !yRatioEq) {
        int inLength = inSz.width * chanNum;

        if (lpi == 4)
        {
            // vertical pass
            __m128i b0 = _mm_set1_epi16(beta[0]);
            __m128i b1 = _mm_set1_epi16(beta[1]);
            __m128i b2 = _mm_set1_epi16(beta[2]);
            __m128i b3 = _mm_set1_epi16(beta[3]);
            __m128i zero = _mm_setzero_si128();
            __m128i vertical_shuf_mask = _mm_setr_epi8(0, 8, 4, 12, 1, 9, 5, 13, 2, 10, 6, 14, 3, 11, 7, 15);

            for (int w = 0; w < inSz.width * chanNum; ) {
                for (; w <= inSz.width * chanNum - half_nlanes && w >= 0; w += half_nlanes) {
#ifdef __i386__
                    __m128i val0lo = _mm_castpd_si128(_mm_loadh_pd(
                                                      _mm_load_sd(reinterpret_cast<const double*>(&src0[0][w])),
                                                                  reinterpret_cast<const double*>(&src0[1][w])));
                    __m128i val0hi = _mm_castpd_si128(_mm_loadh_pd(
                                                      _mm_load_sd(reinterpret_cast<const double*>(&src0[2][w])),
                                                                  reinterpret_cast<const double*>(&src0[3][w])));
                    __m128i val1lo = _mm_castpd_si128(_mm_loadh_pd(
                                                      _mm_load_sd(reinterpret_cast<const double*>(&src1[0][w])),
                                                                  reinterpret_cast<const double*>(&src1[1][w])));
                    __m128i val1hi = _mm_castpd_si128(_mm_loadh_pd(
                                                      _mm_load_sd(reinterpret_cast<const double*>(&src1[2][w])),
                                                                  reinterpret_cast<const double*>(&src1[3][w])));
#else
                    __m128i val0lo = _mm_insert_epi64(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(&src0[0][w])),
                                                      *reinterpret_cast<const int64_t*>(&src0[1][w]), 1);
                    __m128i val0hi = _mm_insert_epi64(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(&src0[2][w])),
                                                      *reinterpret_cast<const int64_t*>(&src0[3][w]), 1);
                    __m128i val1lo = _mm_insert_epi64(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(&src1[0][w])),
                                                      *reinterpret_cast<const int64_t*>(&src1[1][w]), 1);
                    __m128i val1hi = _mm_insert_epi64(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(&src1[2][w])),
                                                      *reinterpret_cast<const int64_t*>(&src1[3][w]), 1);
#endif
                    __m128i val0_0 = _mm_cvtepu8_epi16(val0lo);
                    __m128i val0_2 = _mm_cvtepu8_epi16(val0hi);
                    __m128i val1_0 = _mm_cvtepu8_epi16(val1lo);
                    __m128i val1_2 = _mm_cvtepu8_epi16(val1hi);

                    __m128i val0_1 = _mm_unpackhi_epi8(val0lo, zero);
                    __m128i val0_3 = _mm_unpackhi_epi8(val0hi, zero);
                    __m128i val1_1 = _mm_unpackhi_epi8(val1lo, zero);
                    __m128i val1_3 = _mm_unpackhi_epi8(val1hi, zero);

                    __m128i t0 = _mm_mulhrs_epi16(_mm_sub_epi16(val0_0, val1_0), b0);
                    __m128i t1 = _mm_mulhrs_epi16(_mm_sub_epi16(val0_1, val1_1), b1);
                    __m128i t2 = _mm_mulhrs_epi16(_mm_sub_epi16(val0_2, val1_2), b2);
                    __m128i t3 = _mm_mulhrs_epi16(_mm_sub_epi16(val0_3, val1_3), b3);

                    __m128i r0 = _mm_add_epi16(val1_0, t0);
                    __m128i r1 = _mm_add_epi16(val1_1, t1);
                    __m128i r2 = _mm_add_epi16(val1_2, t2);
                    __m128i r3 = _mm_add_epi16(val1_3, t3);

                    __m128i q0 = _mm_packus_epi16(r0, r1);
                    __m128i q1 = _mm_packus_epi16(r2, r3);

                    __m128i q2 = _mm_blend_epi16(q0, _mm_slli_si128(q1, 4), 0xCC /*0b11001100*/);
                    __m128i q3 = _mm_blend_epi16(_mm_srli_si128(q0, 4), q1, 0xCC /*0b11001100*/);

                    __m128i q4 = _mm_shuffle_epi8(q2, vertical_shuf_mask);
                    __m128i q5 = _mm_shuffle_epi8(q3, vertical_shuf_mask);

                    _mm_storeu_si128(reinterpret_cast<__m128i*>(&tmp[4 * w + 0]), q4);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(&tmp[4 * w + 16]), q5);
                }

                if (w < inSz.width * chanNum) {
                    w = inSz.width * chanNum - half_nlanes;
                }
            }

            // horizontal pass
            __m128i horizontal_shuf_mask = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);

            for (int x = 0; outSz.width >= nlanes; )
            {
                for (; x <= outSz.width - nlanes; x += nlanes)
                {
#ifdef _WIN64
                    __m128i a00 = _mm_setr_epi64x(*reinterpret_cast<const int64_t*>(&clone[4 * x]), *reinterpret_cast<const int64_t*>(&clone[4 * x]));
                    __m128i a01 = _mm_setr_epi64x(*reinterpret_cast<const int64_t*>(&clone[4 * x]), *reinterpret_cast<const int64_t*>(&clone[4 * (x + 1)]));
                    __m128i a11 = _mm_setr_epi64x(*reinterpret_cast<const int64_t*>(&clone[4 * (x + 1)]), *reinterpret_cast<const int64_t*>(&clone[4 * (x + 1)]));
                    __m128i a22 = _mm_setr_epi64x(*reinterpret_cast<const int64_t*>(&clone[4 * (x + 2)]), *reinterpret_cast<const int64_t*>(&clone[4 * (x + 2)]));
                    __m128i a23 = _mm_setr_epi64x(*reinterpret_cast<const int64_t*>(&clone[4 * (x + 2)]), *reinterpret_cast<const int64_t*>(&clone[4 * (x + 3)]));
                    __m128i a33 = _mm_setr_epi64x(*reinterpret_cast<const int64_t*>(&clone[4 * (x + 3)]), *reinterpret_cast<const int64_t*>(&clone[4 * (x + 3)]));
                    __m128i a44 = _mm_setr_epi64x(*reinterpret_cast<const int64_t*>(&clone[4 * (x + 4)]), *reinterpret_cast<const int64_t*>(&clone[4 * (x + 4)]));
                    __m128i a45 = _mm_setr_epi64x(*reinterpret_cast<const int64_t*>(&clone[4 * (x + 4)]), *reinterpret_cast<const int64_t*>(&clone[4 * (x + 5)]));
                    __m128i a55 = _mm_setr_epi64x(*reinterpret_cast<const int64_t*>(&clone[4 * (x + 5)]), *reinterpret_cast<const int64_t*>(&clone[4 * (x + 5)]));
                    __m128i a66 = _mm_setr_epi64x(*reinterpret_cast<const int64_t*>(&clone[4 * (x + 6)]), *reinterpret_cast<const int64_t*>(&clone[4 * (x + 6)]));
                    __m128i a67 = _mm_setr_epi64x(*reinterpret_cast<const int64_t*>(&clone[4 * (x + 6)]), *reinterpret_cast<const int64_t*>(&clone[4 * (x + 7)]));
                    __m128i a77 = _mm_setr_epi64x(*reinterpret_cast<const int64_t*>(&clone[4 * (x + 7)]), *reinterpret_cast<const int64_t*>(&clone[4 * (x + 7)]));
                    __m128i a88 = _mm_setr_epi64x(*reinterpret_cast<const int64_t*>(&clone[4 * (x + 8)]), *reinterpret_cast<const int64_t*>(&clone[4 * (x + 8)]));
                    __m128i a89 = _mm_setr_epi64x(*reinterpret_cast<const int64_t*>(&clone[4 * (x + 8)]), *reinterpret_cast<const int64_t*>(&clone[4 * (x + 9)]));
                    __m128i a99 = _mm_setr_epi64x(*reinterpret_cast<const int64_t*>(&clone[4 * (x + 9)]), *reinterpret_cast<const int64_t*>(&clone[4 * (x + 9)]));
                    __m128i a1010 = _mm_setr_epi64x(*reinterpret_cast<const int64_t*>(&clone[4 * (x + 10)]), *reinterpret_cast<const int64_t*>(&clone[4 * (x + 10)]));
                    __m128i a1011 = _mm_setr_epi64x(*reinterpret_cast<const int64_t*>(&clone[4 * (x + 10)]), *reinterpret_cast<const int64_t*>(&clone[4 * (x + 11)]));
                    __m128i a1111 = _mm_setr_epi64x(*reinterpret_cast<const int64_t*>(&clone[4 * (x + 11)]), *reinterpret_cast<const int64_t*>(&clone[4 * (x + 11)]));
                    __m128i a1212 = _mm_setr_epi64x(*reinterpret_cast<const int64_t*>(&clone[4 * (x + 12)]), *reinterpret_cast<const int64_t*>(&clone[4 * (x + 12)]));
                    __m128i a1213 = _mm_setr_epi64x(*reinterpret_cast<const int64_t*>(&clone[4 * (x + 12)]), *reinterpret_cast<const int64_t*>(&clone[4 * (x + 13)]));
                    __m128i a1313 = _mm_setr_epi64x(*reinterpret_cast<const int64_t*>(&clone[4 * (x + 13)]), *reinterpret_cast<const int64_t*>(&clone[4 * (x + 13)]));
                    __m128i a1414 = _mm_setr_epi64x(*reinterpret_cast<const int64_t*>(&clone[4 * (x + 14)]), *reinterpret_cast<const int64_t*>(&clone[4 * (x + 14)]));
                    __m128i a1415 = _mm_setr_epi64x(*reinterpret_cast<const int64_t*>(&clone[4 * (x + 14)]), *reinterpret_cast<const int64_t*>(&clone[4 * (x + 15)]));
                    __m128i a1515 = _mm_setr_epi64x(*reinterpret_cast<const int64_t*>(&clone[4 * (x + 15)]), *reinterpret_cast<const int64_t*>(&clone[4 * (x + 15)]));
#else
                    __m128i a00 = _mm_setr_epi64(*reinterpret_cast<const __m64*>(&clone[4 * x]), *reinterpret_cast<const __m64*>(&clone[4 * x]));
                    __m128i a01 = _mm_setr_epi64(*reinterpret_cast<const __m64*>(&clone[4 * x]), *reinterpret_cast<const __m64*>(&clone[4 * (x + 1)]));
                    __m128i a11 = _mm_setr_epi64(*reinterpret_cast<const __m64*>(&clone[4 * (x + 1)]), *reinterpret_cast<const __m64*>(&clone[4 * (x + 1)]));
                    __m128i a22 = _mm_setr_epi64(*reinterpret_cast<const __m64*>(&clone[4 * (x + 2)]), *reinterpret_cast<const __m64*>(&clone[4 * (x + 2)]));
                    __m128i a23 = _mm_setr_epi64(*reinterpret_cast<const __m64*>(&clone[4 * (x + 2)]), *reinterpret_cast<const __m64*>(&clone[4 * (x + 3)]));
                    __m128i a33 = _mm_setr_epi64(*reinterpret_cast<const __m64*>(&clone[4 * (x + 3)]), *reinterpret_cast<const __m64*>(&clone[4 * (x + 3)]));
                    __m128i a44 = _mm_setr_epi64(*reinterpret_cast<const __m64*>(&clone[4 * (x + 4)]), *reinterpret_cast<const __m64*>(&clone[4 * (x + 4)]));
                    __m128i a45 = _mm_setr_epi64(*reinterpret_cast<const __m64*>(&clone[4 * (x + 4)]), *reinterpret_cast<const __m64*>(&clone[4 * (x + 5)]));
                    __m128i a55 = _mm_setr_epi64(*reinterpret_cast<const __m64*>(&clone[4 * (x + 5)]), *reinterpret_cast<const __m64*>(&clone[4 * (x + 5)]));
                    __m128i a66 = _mm_setr_epi64(*reinterpret_cast<const __m64*>(&clone[4 * (x + 6)]), *reinterpret_cast<const __m64*>(&clone[4 * (x + 6)]));
                    __m128i a67 = _mm_setr_epi64(*reinterpret_cast<const __m64*>(&clone[4 * (x + 6)]), *reinterpret_cast<const __m64*>(&clone[4 * (x + 7)]));
                    __m128i a77 = _mm_setr_epi64(*reinterpret_cast<const __m64*>(&clone[4 * (x + 7)]), *reinterpret_cast<const __m64*>(&clone[4 * (x + 7)]));
                    __m128i a88 = _mm_setr_epi64(*reinterpret_cast<const __m64*>(&clone[4 * (x + 8)]), *reinterpret_cast<const __m64*>(&clone[4 * (x + 8)]));
                    __m128i a89 = _mm_setr_epi64(*reinterpret_cast<const __m64*>(&clone[4 * (x + 8)]), *reinterpret_cast<const __m64*>(&clone[4 * (x + 9)]));
                    __m128i a99 = _mm_setr_epi64(*reinterpret_cast<const __m64*>(&clone[4 * (x + 9)]), *reinterpret_cast<const __m64*>(&clone[4 * (x + 9)]));
                    __m128i a1010 = _mm_setr_epi64(*reinterpret_cast<const __m64*>(&clone[4 * (x + 10)]), *reinterpret_cast<const __m64*>(&clone[4 * (x + 10)]));
                    __m128i a1011 = _mm_setr_epi64(*reinterpret_cast<const __m64*>(&clone[4 * (x + 10)]), *reinterpret_cast<const __m64*>(&clone[4 * (x + 11)]));
                    __m128i a1111 = _mm_setr_epi64(*reinterpret_cast<const __m64*>(&clone[4 * (x + 11)]), *reinterpret_cast<const __m64*>(&clone[4 * (x + 11)]));
                    __m128i a1212 = _mm_setr_epi64(*reinterpret_cast<const __m64*>(&clone[4 * (x + 12)]), *reinterpret_cast<const __m64*>(&clone[4 * (x + 12)]));
                    __m128i a1213 = _mm_setr_epi64(*reinterpret_cast<const __m64*>(&clone[4 * (x + 12)]), *reinterpret_cast<const __m64*>(&clone[4 * (x + 13)]));
                    __m128i a1313 = _mm_setr_epi64(*reinterpret_cast<const __m64*>(&clone[4 * (x + 13)]), *reinterpret_cast<const __m64*>(&clone[4 * (x + 13)]));
                    __m128i a1414 = _mm_setr_epi64(*reinterpret_cast<const __m64*>(&clone[4 * (x + 14)]), *reinterpret_cast<const __m64*>(&clone[4 * (x + 14)]));
                    __m128i a1415 = _mm_setr_epi64(*reinterpret_cast<const __m64*>(&clone[4 * (x + 14)]), *reinterpret_cast<const __m64*>(&clone[4 * (x + 15)]));
                    __m128i a1515 = _mm_setr_epi64(*reinterpret_cast<const __m64*>(&clone[4 * (x + 15)]), *reinterpret_cast<const __m64*>(&clone[4 * (x + 15)]));
#endif

                    // load 3 channels of first pixel from first pair of 4-couple scope
                    __m128i pix1 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(&tmp[4 * (chanNum * mapsx[x])]));
                    // insert first channel from next couple of pixels to completely fill the simd vector
                    pix1 = _mm_insert_epi32(pix1, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * mapsx[x + 1])]), 3);

                    // load 3 channels of neighbor pixel from first pair of 4-couple scope
                    __m128i pix2 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(&tmp[4 * (chanNum * (mapsx[x] + 1))]));
                    // insert first channel from next couple of pixels to completely fill the simd vector
                    pix2 = _mm_insert_epi32(pix2, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * (mapsx[x + 1] + 1))]), 3);

                    // expand 8-bit data to 16-bit
                    __m128i val_0 = _mm_unpacklo_epi8(pix1, zero);
                    __m128i val_1 = _mm_unpacklo_epi8(pix2, zero);

                    // expand 8-bit data to 16-bit
                    __m128i val_2 = _mm_unpackhi_epi8(pix1, zero);
                    __m128i val_3 = _mm_unpackhi_epi8(pix2, zero);

                    // the main calculations
                    __m128i t0_0 = _mm_mulhrs_epi16(_mm_sub_epi16(val_0, val_1), a00);
                    __m128i t1_0 = _mm_mulhrs_epi16(_mm_sub_epi16(val_2, val_3), a01);
                    __m128i r0_0 = _mm_add_epi16(val_1, t0_0);
                    __m128i r1_0 = _mm_add_epi16(val_3, t1_0);

                    // pack 16-bit data to 8-bit
                    __m128i q0_0 = _mm_packus_epi16(r0_0, r1_0);
                    // gather data from the same lines together
                    __m128i res1 = _mm_shuffle_epi8(q0_0, horizontal_shuf_mask);

                    val_0 = _mm_unpacklo_epi8(_mm_insert_epi64(val_0, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * mapsx[x + 1] + 1)]), 0), zero);
                    val_1 = _mm_unpacklo_epi8(_mm_insert_epi64(val_1, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * (mapsx[x + 1] + 1) + 1)]), 0), zero);

                    val_2 = _mm_insert_epi64(val_2, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * mapsx[x + 2])]), 0);
                    val_3 = _mm_insert_epi64(val_3, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * (mapsx[x + 2] + 1))]), 0);

                    val_2 = _mm_unpacklo_epi8(val_2, zero);
                    val_3 = _mm_unpacklo_epi8(val_3, zero);

                    __m128i t0_1 = _mm_mulhrs_epi16(_mm_sub_epi16(val_0, val_1), a11);
                    __m128i t1_1 = _mm_mulhrs_epi16(_mm_sub_epi16(val_2, val_3), a22);
                    __m128i r0_1 = _mm_add_epi16(val_1, t0_1);
                    __m128i r1_1 = _mm_add_epi16(val_3, t1_1);

                    __m128i q0_1 = _mm_packus_epi16(r0_1, r1_1);
                    __m128i res2 = _mm_shuffle_epi8(q0_1, horizontal_shuf_mask);

                    __m128i pix7 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(&tmp[4 * (chanNum * (mapsx[x + 3] - 1) + 2)]));
                    pix7 = _mm_insert_epi32(pix7, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * mapsx[x + 2] + 2)]), 0);

                    __m128i pix8 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(&tmp[4 * (chanNum * mapsx[x + 3] + 2)]));
                    pix8 = _mm_insert_epi32(pix8, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * (mapsx[x + 2] + 1) + 2)]), 0);

                    val_0 = _mm_unpacklo_epi8(pix7, zero);
                    val_1 = _mm_unpacklo_epi8(pix8, zero);

                    val_2 = _mm_unpackhi_epi8(pix7, zero);
                    val_3 = _mm_unpackhi_epi8(pix8, zero);

                    // the main calculations
                    __m128i t0_2 = _mm_mulhrs_epi16(_mm_sub_epi16(val_0, val_1), a23);
                    __m128i t1_2 = _mm_mulhrs_epi16(_mm_sub_epi16(val_2, val_3), a33);
                    __m128i r0_2 = _mm_add_epi16(val_1, t0_2);
                    __m128i r1_2 = _mm_add_epi16(val_3, t1_2);

                    // pack 16-bit data to 8-bit
                    __m128i q0_2 = _mm_packus_epi16(r0_2, r1_2);
                    __m128i res3 = _mm_shuffle_epi8(q0_2, horizontal_shuf_mask);

                    __m128i pix9 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(&tmp[4 * (chanNum * mapsx[x + 4])]));
                    // insert first channel from next couple of pixels to completely fill the simd vector
                    pix9 = _mm_insert_epi32(pix9, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * mapsx[x + 5])]), 3);

                    // load 3 channels of neighbor pixel from first pair of 4-couple scope
                    __m128i pix10 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(&tmp[4 * (chanNum * (mapsx[x + 4] + 1))]));
                    // insert first channel from next couple of pixels to completely fill the simd vector
                    pix10 = _mm_insert_epi32(pix10, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * (mapsx[x + 5] + 1))]), 3);

                    // expand 8-bit data to 16-bit
                    val_0 = _mm_unpacklo_epi8(pix9, zero);
                    val_1 = _mm_unpacklo_epi8(pix10, zero);

                    // expand 8-bit data to 16-bit
                    val_2 = _mm_unpackhi_epi8(pix9, zero);
                    val_3 = _mm_unpackhi_epi8(pix10, zero);

                    // the main calculations
                    __m128i t0_3 = _mm_mulhrs_epi16(_mm_sub_epi16(val_0, val_1), a44);
                    __m128i t1_3 = _mm_mulhrs_epi16(_mm_sub_epi16(val_2, val_3), a45);
                    __m128i r0_3 = _mm_add_epi16(val_1, t0_3);
                    __m128i r1_3 = _mm_add_epi16(val_3, t1_3);

                    // pack 16-bit data to 8-bit
                    __m128i q0_3 = _mm_packus_epi16(r0_3, r1_3);
                    // gather data from the same lines together
                    __m128i res4 = _mm_shuffle_epi8(q0_3, horizontal_shuf_mask);

                    val_0 = _mm_unpacklo_epi8(_mm_insert_epi64(val_0, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum *  mapsx[x + 5]      + 1)]), 0), zero);
                    val_1 = _mm_unpacklo_epi8(_mm_insert_epi64(val_1, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * (mapsx[x + 5] + 1) + 1)]), 0), zero);

                    val_2 = _mm_insert_epi64(val_2, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * mapsx[x + 6])]), 0);
                    val_3 = _mm_insert_epi64(val_3, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * (mapsx[x + 6] + 1))]), 0);

                    val_2 = _mm_unpacklo_epi8(val_2, zero);
                    val_3 = _mm_unpacklo_epi8(val_3, zero);

                    __m128i t0_4 = _mm_mulhrs_epi16(_mm_sub_epi16(val_0, val_1), a55);
                    __m128i t1_4 = _mm_mulhrs_epi16(_mm_sub_epi16(val_2, val_3), a66);
                    __m128i r0_4 = _mm_add_epi16(val_1, t0_4);
                    __m128i r1_4 = _mm_add_epi16(val_3, t1_4);

                    __m128i q0_4 = _mm_packus_epi16(r0_4, r1_4);
                    __m128i res5 = _mm_shuffle_epi8(q0_4, horizontal_shuf_mask);

                    __m128i pix15 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(&tmp[4 * (chanNum * (mapsx[x + 7] - 1) + 2)]));
                    pix15 = _mm_insert_epi32(pix15, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * mapsx[x + 6] + 2)]), 0);

                    __m128i pix16 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(&tmp[4 * (chanNum * mapsx[x + 7]   + 2)]));
                    pix16 = _mm_insert_epi32(pix16, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * (mapsx[x + 6] + 1) + 2)]), 0);

                    val_0 = _mm_unpacklo_epi8(pix15, zero);
                    val_1 = _mm_unpacklo_epi8(pix16, zero);

                    val_2 = _mm_unpackhi_epi8(pix15, zero);
                    val_3 = _mm_unpackhi_epi8(pix16, zero);

                    // the main calculations
                    __m128i t0_5 = _mm_mulhrs_epi16(_mm_sub_epi16(val_0, val_1), a67);
                    __m128i t1_5 = _mm_mulhrs_epi16(_mm_sub_epi16(val_2, val_3), a77);
                    __m128i r0_5 = _mm_add_epi16(val_1, t0_5);
                    __m128i r1_5 = _mm_add_epi16(val_3, t1_5);

                    // pack 16-bit data to 8-bit
                    __m128i q0_5 = _mm_packus_epi16(r0_5, r1_5);
                    __m128i res6 = _mm_shuffle_epi8(q0_5, horizontal_shuf_mask);

                    __m128i bl1 = _mm_blend_epi16(res1, _mm_slli_si128(res2, 4), 0xCC /*0b11001100*/);
                    __m128i bl2 = _mm_blend_epi16(_mm_srli_si128(res1, 4), res2, 0xCC /*0b11001100*/);

                    __m128i bl3 = _mm_blend_epi16(res3, _mm_slli_si128(res4, 4), 0xCC /*0b11001100*/);
                    __m128i bl4 = _mm_blend_epi16(_mm_srli_si128(res3, 4), res4, 0xCC /*0b11001100*/);

                    __m128i bl5 = _mm_blend_epi16(res5, _mm_slli_si128(res6, 4), 0xCC /*0b11001100*/);
                    __m128i bl6 = _mm_blend_epi16(_mm_srli_si128(res5, 4), res6, 0xCC /*0b11001100*/);

                    __m128i bl13 = _mm_blend_epi16(bl1, _mm_slli_si128(bl3, 8), 0xF0 /*0b11110000*/);
                    __m128i bl31 = _mm_blend_epi16(_mm_srli_si128(bl1, 8), bl3, 0xF0 /*0b11110000*/);

                    __m128i bl24 = _mm_blend_epi16(bl2, _mm_slli_si128(bl4, 8), 0xF0 /*0b11110000*/);
                    __m128i bl42 = _mm_blend_epi16(_mm_srli_si128(bl2, 8), bl4, 0xF0 /*0b11110000*/);

                    // load 3 channels of first pixel from first pair of 4-couple scope
                    __m128i pix17 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(&tmp[4 * (chanNum * mapsx[x + 8])]));
                    // insert first channel from next couple of pixels to completely fill the simd vector
                    pix17 = _mm_insert_epi32(pix17, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * mapsx[x + 9])]), 3);

                    // load 3 channels of neighbor pixel from first pair of 4-couple scope
                    __m128i pix18 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(&tmp[4 * (chanNum * (mapsx[x + 8] + 1))]));
                    // insert first channel from next couple of pixels to completely fill the simd vector
                    pix18 = _mm_insert_epi32(pix18, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * (mapsx[x + 9] + 1))]), 3);

                    // expand 8-bit data to 16-bit
                    val_0 = _mm_unpacklo_epi8(pix17, zero);
                    val_1 = _mm_unpacklo_epi8(pix18, zero);

                    // expand 8-bit data to 16-bit
                    val_2 = _mm_unpackhi_epi8(pix17, zero);
                    val_3 = _mm_unpackhi_epi8(pix18, zero);

                    // the main calculations
                    __m128i t0_6 = _mm_mulhrs_epi16(_mm_sub_epi16(val_0, val_1), a88);
                    __m128i t1_6 = _mm_mulhrs_epi16(_mm_sub_epi16(val_2, val_3), a89);
                    __m128i r0_6 = _mm_add_epi16(val_1, t0_6);
                    __m128i r1_6 = _mm_add_epi16(val_3, t1_6);

                    // pack 16-bit data to 8-bit
                    __m128i q0_6 = _mm_packus_epi16(r0_6, r1_6);
                    // gather data from the same lines together
                    __m128i res7 = _mm_shuffle_epi8(q0_6, horizontal_shuf_mask);

                    val_0 = _mm_unpacklo_epi8(_mm_insert_epi64(val_0, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * mapsx[x + 9] + 1)]), 0), zero);
                    val_1 = _mm_unpacklo_epi8(_mm_insert_epi64(val_1, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * (mapsx[x + 9] + 1) + 1)]), 0), zero);

                    val_2 = _mm_insert_epi64(val_2, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * mapsx[x + 10])]), 0);
                    val_3 = _mm_insert_epi64(val_3, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * (mapsx[x + 10] + 1))]), 0);

                    val_2 = _mm_unpacklo_epi8(val_2, zero);
                    val_3 = _mm_unpacklo_epi8(val_3, zero);

                    __m128i t0_7 = _mm_mulhrs_epi16(_mm_sub_epi16(val_0, val_1), a99);
                    __m128i t1_7 = _mm_mulhrs_epi16(_mm_sub_epi16(val_2, val_3), a1010);
                    __m128i r0_7 = _mm_add_epi16(val_1, t0_7);
                    __m128i r1_7 = _mm_add_epi16(val_3, t1_7);

                    __m128i q0_7 = _mm_packus_epi16(r0_7, r1_7);
                    __m128i res8 = _mm_shuffle_epi8(q0_7, horizontal_shuf_mask);

                    __m128i pix21 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(&tmp[4 * (chanNum * (mapsx[x + 11] - 1) + 2)]));
                    pix21 = _mm_insert_epi32(pix21, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * mapsx[x + 10] + 2)]), 0);

                    __m128i pix22 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(&tmp[4 * (chanNum * mapsx[x + 11] + 2)]));
                    pix22 = _mm_insert_epi32(pix22, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * (mapsx[x + 10] + 1) + 2)]), 0);

                    val_0 = _mm_unpacklo_epi8(pix21, zero);
                    val_1 = _mm_unpacklo_epi8(pix22, zero);

                    val_2 = _mm_unpackhi_epi8(pix21, zero);
                    val_3 = _mm_unpackhi_epi8(pix22, zero);

                    // the main calculations
                    __m128i t0_8 = _mm_mulhrs_epi16(_mm_sub_epi16(val_0, val_1), a1011);
                    __m128i t1_8 = _mm_mulhrs_epi16(_mm_sub_epi16(val_2, val_3), a1111);
                    __m128i r0_8 = _mm_add_epi16(val_1, t0_8);
                    __m128i r1_8 = _mm_add_epi16(val_3, t1_8);

                    // pack 16-bit data to 8-bit
                    __m128i q0_8 = _mm_packus_epi16(r0_8, r1_8);
                    __m128i res9 = _mm_shuffle_epi8(q0_8, horizontal_shuf_mask);

                    __m128i pix23 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(&tmp[4 * (chanNum * mapsx[x + 12])]));
                    // insert first channel from next couple of pixels to completely fill the simd vector
                    pix23 = _mm_insert_epi32(pix23, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * mapsx[x + 13])]), 3);

                    // load 3 channels of neighbor pixel from first pair of 4-couple scope
                    __m128i pix24 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(&tmp[4 * (chanNum * (mapsx[x + 12] + 1))]));
                    // insert first channel from next couple of pixels to completely fill the simd vector
                    pix24 = _mm_insert_epi32(pix24, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * (mapsx[x + 13] + 1))]), 3);

                    // expand 8-bit data to 16-bit
                    val_0 = _mm_unpacklo_epi8(pix23, zero);
                    val_1 = _mm_unpacklo_epi8(pix24, zero);

                    // expand 8-bit data to 16-bit
                    val_2 = _mm_unpackhi_epi8(pix23, zero);
                    val_3 = _mm_unpackhi_epi8(pix24, zero);

                    // the main calculations
                    __m128i t0_9 = _mm_mulhrs_epi16(_mm_sub_epi16(val_0, val_1), a1212);
                    __m128i t1_9 = _mm_mulhrs_epi16(_mm_sub_epi16(val_2, val_3), a1213);
                    __m128i r0_9 = _mm_add_epi16(val_1, t0_9);
                    __m128i r1_9 = _mm_add_epi16(val_3, t1_9);

                    // pack 16-bit data to 8-bit
                    __m128i q0_9 = _mm_packus_epi16(r0_9, r1_9);
                    // gather data from the same lines together
                    __m128i res10 = _mm_shuffle_epi8(q0_9, horizontal_shuf_mask);

                    val_0 = _mm_unpacklo_epi8(_mm_insert_epi64(val_0, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * mapsx[x + 13] + 1)]), 0), zero);
                    val_1 = _mm_unpacklo_epi8(_mm_insert_epi64(val_1, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * (mapsx[x + 13] + 1) + 1)]), 0), zero);

                    val_2 = _mm_insert_epi64(val_2, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * mapsx[x + 14])]), 0);
                    val_3 = _mm_insert_epi64(val_3, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * (mapsx[x + 14] + 1))]), 0);

                    val_2 = _mm_unpacklo_epi8(val_2, zero);
                    val_3 = _mm_unpacklo_epi8(val_3, zero);

                    __m128i t0_10 = _mm_mulhrs_epi16(_mm_sub_epi16(val_0, val_1), a1313);
                    __m128i t1_10 = _mm_mulhrs_epi16(_mm_sub_epi16(val_2, val_3), a1414);
                    __m128i r0_10 = _mm_add_epi16(val_1, t0_10);
                    __m128i r1_10 = _mm_add_epi16(val_3, t1_10);

                    __m128i q0_10 = _mm_packus_epi16(r0_10, r1_10);
                    __m128i res11 = _mm_shuffle_epi8(q0_10, horizontal_shuf_mask);

                    __m128i pix27 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(&tmp[4 * (chanNum * (mapsx[x + 15] - 1) + 2)]));
                    pix27 = _mm_insert_epi32(pix27, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * mapsx[x + 14] + 2)]), 0);

                    __m128i pix28 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(&tmp[4 * (chanNum * mapsx[x + 15] + 2)]));
                    pix28 = _mm_insert_epi32(pix28, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * (mapsx[x + 14] + 1) + 2)]), 0);

                    val_0 = _mm_unpacklo_epi8(pix27, zero);
                    val_1 = _mm_unpacklo_epi8(pix28, zero);

                    val_2 = _mm_unpackhi_epi8(pix27, zero);
                    val_3 = _mm_unpackhi_epi8(pix28, zero);

                    // the main calculations
                    __m128i t0_11 = _mm_mulhrs_epi16(_mm_sub_epi16(val_0, val_1), a1415);
                    __m128i t1_11 = _mm_mulhrs_epi16(_mm_sub_epi16(val_2, val_3), a1515);
                    __m128i r0_11 = _mm_add_epi16(val_1, t0_11);
                    __m128i r1_11 = _mm_add_epi16(val_3, t1_11);

                    // pack 16-bit data to 8-bit
                    __m128i q0_11 = _mm_packus_epi16(r0_11, r1_11);
                    __m128i res12 = _mm_shuffle_epi8(q0_11, horizontal_shuf_mask);

                    __m128i bl7 = _mm_blend_epi16(res7, _mm_slli_si128(res8, 4), 0xCC /*0b11001100*/);
                    __m128i bl8 = _mm_blend_epi16(_mm_srli_si128(res7, 4), res8, 0xCC /*0b11001100*/);

                    __m128i bl9 = _mm_blend_epi16(res9, _mm_slli_si128(res10, 4), 0xCC /*0b11001100*/);
                    __m128i bl10 = _mm_blend_epi16(_mm_srli_si128(res9, 4), res10, 0xCC /*0b11001100*/);

                    __m128i bl11 = _mm_blend_epi16(res11, _mm_slli_si128(res12, 4), 0xCC /*0b11001100*/);
                    __m128i bl12 = _mm_blend_epi16(_mm_srli_si128(res11, 4), res12, 0xCC /*0b11001100*/);

                    __m128i bl57 = _mm_blend_epi16(bl5, _mm_slli_si128(bl7, 8), 0xF0 /*0b11110000*/);
                    __m128i bl75 = _mm_blend_epi16(_mm_srli_si128(bl5, 8), bl7, 0xF0 /*0b11110000*/);

                    __m128i bl68 = _mm_blend_epi16(bl6, _mm_slli_si128(bl8, 8), 0xF0 /*0b11110000*/);
                    __m128i bl86 = _mm_blend_epi16(_mm_srli_si128(bl6, 8), bl8, 0xF0 /*0b11110000*/);

                    __m128i bl911 = _mm_blend_epi16(bl9, _mm_slli_si128(bl11, 8), 0xF0 /*0b11110000*/);
                    __m128i bl119 = _mm_blend_epi16(_mm_srli_si128(bl9, 8), bl11, 0xF0 /*0b11110000*/);

                    __m128i bl1012 = _mm_blend_epi16(bl10, _mm_slli_si128(bl12, 8), 0xF0 /*0b11110000*/);
                    __m128i bl1210 = _mm_blend_epi16(_mm_srli_si128(bl10, 8), bl12, 0xF0 /*0b11110000*/);

                    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[0][3 * x]), bl13);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[1][3 * x]), bl24);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[2][3 * x]), bl31);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[3][3 * x]), bl42);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[0][3 * x + 16]), bl57);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[1][3 * x + 16]), bl68);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[2][3 * x + 16]), bl75);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[3][3 * x + 16]), bl86);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[0][3 * x + 32]), bl911);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[1][3 * x + 32]), bl1012);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[2][3 * x + 32]), bl119);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[3][3 * x + 32]), bl1210);
                }

                if (x < outSz.width) {
                    x = outSz.width - nlanes;
                    continue;
                }
                break;
            }
        }
        else
        {  // if any lpi
            for (int l = 0; l < lpi; ++l) {
                short beta0 = beta[l];
                const uchar* s0 = src0[l];
                const uchar* s1 = src1[l];

                // vertical pass
                resize_vertical_anyLPI(s0, s1, tmp, inLength, beta0);

                // horizontal pass
                resize_horizontal_anyLPI(dst[l], tmp, mapsx, alpha, outSz.width);
            }
        }
    } else if (!xRatioEq) {
        GAPI_DbgAssert(yRatioEq);

        for (int l = 0; l < lpi; ++l) {
            const uchar* src = src0[l];

            // horizontal pass
            resize_horizontal_anyLPI(dst[l], src, mapsx, alpha, outSz.width);
        }
    } else if (!yRatioEq) {
        GAPI_DbgAssert(xRatioEq);
        int inLength = inSz.width*chanNum;  // == outSz.width

        for (int l = 0; l < lpi; ++l) {
            short beta0 = beta[l];
            const uchar* s0 = src0[l];
            const uchar* s1 = src1[l];

            // vertical pass
            resize_vertical_anyLPI(s0, s1, dst[l], inLength, beta0);
        }
    } else {
        GAPI_DbgAssert(xRatioEq && yRatioEq);
        int length = inSz.width *chanNum;

        for (int l = 0; l < lpi; ++l) {
            memcpy(dst[l], src0[l], length);
        }
    }
}
} // namespace sse42
} // namespace fliud
} // namespace gapi
} // namespace cv
#endif // !defined(GAPI_STANDALONE)
