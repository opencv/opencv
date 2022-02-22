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
namespace sse41 {

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
    constexpr int nlanes = 16; // number of 8-bit integers that fit into a 128-bit SIMD vector.
    constexpr int half_nlanes = nlanes / 2;
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
#if defined(__i386__) || defined(_M_IX86)
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
            __m128i horizontal_shuf_mask1 = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 3, 7, 11, 15);
            constexpr int nproc_pixels = 5;
            for (int x = 0; ; )
            {
                for (; x <= outSz.width - (nproc_pixels + 1); x += nproc_pixels)
                {
#ifdef _MSC_VER
                    __m128i a00 = _mm_setr_epi64x(*reinterpret_cast<const int64_t*>(&clone[4 * x]), *reinterpret_cast<const int64_t*>(&clone[4 * x]));
#else
                    __m128i a00 = _mm_setr_epi64(*reinterpret_cast<const __m64*>(&clone[4 * x]), *reinterpret_cast<const __m64*>(&clone[4 * x]));
#endif
                    __m128i pix1 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(&tmp[4 * (chanNum * mapsx[x])]));
                    __m128i pix2 = _mm_setzero_si128();
#if defined(__i386__) || defined(_M_IX86)
                    pix2 = _mm_castpd_si128(_mm_load_sd(reinterpret_cast<const double*>(&tmp[4 * (chanNum * (mapsx[x] + 1))])));
#else
                    pix2 = _mm_insert_epi64(pix2, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * (mapsx[x] + 1))]), 0);
#endif

                    pix2 = _mm_insert_epi32(pix2, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * (mapsx[x] + 1)) + 8]), 2);

                    // expand 8-bit data to 16-bit
                    __m128i val_0 = _mm_unpacklo_epi8(pix1, zero);
                    __m128i val_1 = _mm_unpacklo_epi8(pix2, zero);
                    __m128i val_2 = _mm_unpackhi_epi8(pix1, zero);
                    __m128i val_3 = _mm_unpackhi_epi8(pix2, zero);

                    // the main calculations
                    __m128i t0_0 = _mm_mulhrs_epi16(_mm_sub_epi16(val_0, val_1), a00);
                    __m128i t1_0 = _mm_mulhrs_epi16(_mm_sub_epi16(val_2, val_3), a00);
                    __m128i r0_0 = _mm_add_epi16(val_1, t0_0);
                    __m128i r1_0 = _mm_add_epi16(val_3, t1_0);

                    // pack 16-bit data to 8-bit
                    __m128i q0_0 = _mm_packus_epi16(r0_0, r1_0);
                    // gather data from the same lines together
                    __m128i res1 = _mm_shuffle_epi8(q0_0, horizontal_shuf_mask);

#ifdef _MSC_VER
                    __m128i a11 = _mm_setr_epi64x(*reinterpret_cast<const int64_t*>(&clone[4 * (x + 1)]), *reinterpret_cast<const int64_t*>(&clone[4 * (x + 1)]));
#else
                    __m128i a11 = _mm_setr_epi64(*reinterpret_cast<const __m64*>(&clone[4 * (x + 1)]), *reinterpret_cast<const __m64*>(&clone[4 * (x + 1)]));
#endif

                    pix1 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(&tmp[4 * (chanNum * mapsx[x + 1])]));
#if defined(__i386__) || defined(_M_IX86)
                    pix2 = _mm_castpd_si128(_mm_load_sd(reinterpret_cast<const double*>(&tmp[4 * (chanNum * (mapsx[x + 1] + 1))])));
#else
                    pix2 = _mm_insert_epi64(pix2, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * (mapsx[x + 1] + 1))]), 0);
#endif
                    pix2 = _mm_insert_epi32(pix2, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * (mapsx[x + 1] + 1)) + 8]), 2);

                    // expand 8-bit data to 16-bit
                    val_0 = _mm_unpacklo_epi8(pix1, zero);
                    val_1 = _mm_unpacklo_epi8(pix2, zero);
                    val_2 = _mm_unpackhi_epi8(pix1, zero);
                    val_3 = _mm_unpackhi_epi8(pix2, zero);

                    // the main calculations
                    t0_0 = _mm_mulhrs_epi16(_mm_sub_epi16(val_0, val_1), a11);
                    t1_0 = _mm_mulhrs_epi16(_mm_sub_epi16(val_2, val_3), a11);
                    r0_0 = _mm_add_epi16(val_1, t0_0);
                    r1_0 = _mm_add_epi16(val_3, t1_0);

                    // pack 16-bit data to 8-bit
                    q0_0 = _mm_packus_epi16(r0_0, r1_0);
                    // gather data from the same lines together
                    __m128i res2 = _mm_shuffle_epi8(q0_0, horizontal_shuf_mask);

#ifdef _MSC_VER
                    __m128i a22 = _mm_setr_epi64x(*reinterpret_cast<const int64_t*>(&clone[4 * (x + 2)]), *reinterpret_cast<const int64_t*>(&clone[4 * (x + 2)]));
#else
                    __m128i a22 = _mm_setr_epi64(*reinterpret_cast<const __m64*>(&clone[4 * (x + 2)]), *reinterpret_cast<const __m64*>(&clone[4 * (x + 2)]));
#endif

                    pix1 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(&tmp[4 * (chanNum * mapsx[x + 2])]));
#if defined(__i386__) || defined(_M_IX86)
                    pix2 = _mm_castpd_si128(_mm_load_sd(reinterpret_cast<const double*>(&tmp[4 * (chanNum * (mapsx[x + 2] + 1))])));
#else
                    pix2 = _mm_insert_epi64(pix2, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * (mapsx[x + 2] + 1))]), 0);
#endif
                    pix2 = _mm_insert_epi32(pix2, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * (mapsx[x + 2] + 1)) + 8]), 2);

                    // expand 8-bit data to 16-bit
                    val_0 = _mm_unpacklo_epi8(pix1, zero);
                    val_1 = _mm_unpacklo_epi8(pix2, zero);
                    val_2 = _mm_unpackhi_epi8(pix1, zero);
                    val_3 = _mm_unpackhi_epi8(pix2, zero);

                    // the main calculations
                    t0_0 = _mm_mulhrs_epi16(_mm_sub_epi16(val_0, val_1), a22);
                    t1_0 = _mm_mulhrs_epi16(_mm_sub_epi16(val_2, val_3), a22);
                    r0_0 = _mm_add_epi16(val_1, t0_0);
                    r1_0 = _mm_add_epi16(val_3, t1_0);

                    // pack 16-bit data to 8-bit
                    q0_0 = _mm_packus_epi16(r0_0, r1_0);
                    // gather data from the same lines together
                    __m128i res3 = _mm_shuffle_epi8(q0_0, horizontal_shuf_mask);

#ifdef _MSC_VER
                    __m128i a33 = _mm_setr_epi64x(*reinterpret_cast<const int64_t*>(&clone[4 * (x + 3)]), *reinterpret_cast<const int64_t*>(&clone[4 * (x + 3)]));
#else
                    __m128i a33 = _mm_setr_epi64(*reinterpret_cast<const __m64*>(&clone[4 * (x + 3)]), *reinterpret_cast<const __m64*>(&clone[4 * (x + 3)]));
#endif

                    pix1 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(&tmp[4 * (chanNum * mapsx[x + 3])]));
#if defined(__i386__) || defined(_M_IX86)
                    pix2 = _mm_castpd_si128(_mm_load_sd(reinterpret_cast<const double*>(&tmp[4 * (chanNum * (mapsx[x + 3] + 1))])));
#else
                    pix2 = _mm_insert_epi64(pix2, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * (mapsx[x + 3] + 1))]), 0);
#endif
                    pix2 = _mm_insert_epi32(pix2, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * (mapsx[x + 3] + 1)) + 8]), 2);

                    // expand 8-bit data to 16-bit
                    val_0 = _mm_unpacklo_epi8(pix1, zero);
                    val_1 = _mm_unpacklo_epi8(pix2, zero);
                    val_2 = _mm_unpackhi_epi8(pix1, zero);
                    val_3 = _mm_unpackhi_epi8(pix2, zero);

                    // the main calculations
                    t0_0 = _mm_mulhrs_epi16(_mm_sub_epi16(val_0, val_1), a33);
                    t1_0 = _mm_mulhrs_epi16(_mm_sub_epi16(val_2, val_3), a33);
                    r0_0 = _mm_add_epi16(val_1, t0_0);
                    r1_0 = _mm_add_epi16(val_3, t1_0);

                    // pack 16-bit data to 8-bit
                    q0_0 = _mm_packus_epi16(r0_0, r1_0);
                    // gather data from the same lines together
                    __m128i res4 = _mm_shuffle_epi8(q0_0, horizontal_shuf_mask);

#ifdef _MSC_VER
                    __m128i a44 = _mm_setr_epi64x(*reinterpret_cast<const int64_t*>(&clone[4 * (x + 4)]), *reinterpret_cast<const int64_t*>(&clone[4 * (x + 4)]));
#else
                    __m128i a44 = _mm_setr_epi64(*reinterpret_cast<const __m64*>(&clone[4 * (x + 4)]), *reinterpret_cast<const __m64*>(&clone[4 * (x + 4)]));
#endif

                    pix1 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(&tmp[4 * (chanNum * mapsx[x + 4])]));
#if defined(__i386__) || defined(_M_IX86)
                    pix2 = _mm_castpd_si128(_mm_load_sd(reinterpret_cast<const double*>(&tmp[4 * (chanNum * (mapsx[x + 4] + 1))])));
#else
                    pix2 = _mm_insert_epi64(pix2, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * (mapsx[x + 4] + 1))]), 0);
#endif
                    pix2 = _mm_insert_epi32(pix2, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * (mapsx[x + 4] + 1)) + 8]), 2);

                    // expand 8-bit data to 16-bit
                    val_0 = _mm_unpacklo_epi8(pix1, zero);
                    val_1 = _mm_unpacklo_epi8(pix2, zero);
                    val_2 = _mm_unpackhi_epi8(pix1, zero);
                    val_3 = _mm_unpackhi_epi8(pix2, zero);

                    // the main calculations
                    t0_0 = _mm_mulhrs_epi16(_mm_sub_epi16(val_0, val_1), a44);
                    t1_0 = _mm_mulhrs_epi16(_mm_sub_epi16(val_2, val_3), a44);
                    r0_0 = _mm_add_epi16(val_1, t0_0);
                    r1_0 = _mm_add_epi16(val_3, t1_0);

                    // pack 16-bit data to 8-bit
                    q0_0 = _mm_packus_epi16(r0_0, r1_0);
                    // gather data from the same lines together
                    __m128i res5 = _mm_shuffle_epi8(q0_0, horizontal_shuf_mask);

                    __m128i bl1 = _mm_blend_epi16(res1, _mm_slli_si128(res2, 4), 0xCC /*0b11001100*/);
                    __m128i bl2 = _mm_blend_epi16(_mm_srli_si128(res1, 4), res2, 0xCC /*0b11001100*/);

                    __m128i bl3 = _mm_blend_epi16(res3, _mm_slli_si128(res4, 4), 0xCC /*0b11001100*/);
                    __m128i bl4 = _mm_blend_epi16(_mm_srli_si128(res3, 4), res4, 0xCC /*0b11001100*/);

                    __m128i bl13 = _mm_blend_epi16(bl1, _mm_slli_si128(bl3, 8), 0xF0 /*0b11110000*/);
                    __m128i bl31 = _mm_blend_epi16(_mm_srli_si128(bl1, 8), bl3, 0xF0 /*0b11110000*/);

                    __m128i bl24 = _mm_blend_epi16(bl2, _mm_slli_si128(bl4, 8), 0xF0 /*0b11110000*/);
                    __m128i bl42 = _mm_blend_epi16(_mm_srli_si128(bl2, 8), bl4, 0xF0 /*0b11110000*/);

                    bl1 = _mm_blend_epi16(_mm_shuffle_epi8(bl13, horizontal_shuf_mask1),
                                          _mm_slli_si128(res5, 12), 192 /*0b11000000*/);
                    bl2 = _mm_blend_epi16(_mm_shuffle_epi8(bl24, horizontal_shuf_mask1),
                                          _mm_slli_si128(res5, 8), 192 /*0b11000000*/);
                    bl3 = _mm_blend_epi16(_mm_shuffle_epi8(bl31, horizontal_shuf_mask1),
                                          _mm_slli_si128(res5, 4), 192 /*0b11000000*/);
                    bl4 = _mm_blend_epi16(_mm_shuffle_epi8(bl42, horizontal_shuf_mask1),
                                          res5, 192 /*0b11000000*/);

                    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[0][chanNum * x]), bl1);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[1][chanNum * x]), bl2);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[2][chanNum * x]), bl3);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[3][chanNum * x]), bl4);
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
