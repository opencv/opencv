// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#if !defined(GAPI_STANDALONE)

#include "opencv2/gapi/own/saturate.hpp"

#include "nmmintrin.h"

#include "opencv2/core.hpp"

#ifdef CV_AVX2
#undef CV_AVX2
#endif
#define CV_AVX2 0

#ifdef CV_SSE4_2
#undef CV_SSE4_2
#undef CV_SSE4_1
#undef CV_SSSE3
#undef CV_SSE3
#undef CV_SSE2
#undef CV_SSE
#endif
#define CV_SSE4_2 1
#define CV_SSE4_1 1
#define CV_SSSE3  1
#define CV_SSE3   1
#define CV_SSE2   1
#define CV_SSE    1
#define CV_CPU_HAS_SUPPORT_SSE2 1

#ifdef CV_SIMD128
#undef CV_SIMD128
#endif

#ifdef CV_SIMD256
#undef CV_SIMD256
#endif
#define CV_SIMD256 0

#define CV_SIMD128 1
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
CV_ALWAYS_INLINE v_int16x8 v_mulhrs(const v_int16x8& a, const v_int16x8& b)
{
    return v_int16x8(_mm_mulhrs_epi16(a.val, b.val));
}

CV_ALWAYS_INLINE v_int16x8 v_mulhrs(const v_int16x8& a, short b) {
    return v_mulhrs(a, v_setall_s16(b));
}

namespace {
    template<int chanNum>
    CV_ALWAYS_INLINE void v_gather_pixel_map(v_uint8x16&, const uchar*, const short*, const int)
    {
        CV_Assert("Unsupported number of channel");
    }

    template<>
    CV_ALWAYS_INLINE void v_gather_pixel_map<3>(v_uint8x16& vec, const uchar src[], const short* index, const int pos)
    {
        const int chanNum = 3;
        // pixel_1 (rgb)
        vec.val = _mm_insert_epi16(vec.val, *reinterpret_cast<const ushort*>(&src[chanNum * (*index + pos)]), 0);
        vec.val = _mm_insert_epi8(vec.val, *reinterpret_cast<const uchar*>(&src[chanNum * (*index + pos) + 2]), 2);
        // pixel_2 (rgb)
        vec.val = _mm_insert_epi8(vec.val, *reinterpret_cast<const uchar*>(&src[chanNum * (*(index + 1) + pos)]), 3);
        vec.val = _mm_insert_epi16(vec.val, *reinterpret_cast<const ushort*>(&src[chanNum * (*(index + 1) + pos) + 1]), 2);
        // pixel_3
        vec.val = _mm_insert_epi16(vec.val, *reinterpret_cast<const ushort*>(&src[chanNum * (*(index + 2) + pos)]), 3);
        vec.val = _mm_insert_epi8(vec.val, *reinterpret_cast<const uchar*>(&src[chanNum * (*(index + 2) + pos) + 2]), 8);
        // pixel_4
        vec.val = _mm_insert_epi8(vec.val, *reinterpret_cast<const uchar*>(&src[chanNum * (*(index + 3) + pos)]), 9);
        vec.val = _mm_insert_epi16(vec.val, *reinterpret_cast<const ushort*>(&src[chanNum * (*(index + 3) + pos) + 1]), 5);
        // pixel_5
        vec.val = _mm_insert_epi16(vec.val, *reinterpret_cast<const ushort*>(&src[chanNum * (*(index + 4) + pos)]), 6);
        vec.val = _mm_insert_epi8(vec.val, *reinterpret_cast<const uchar*>(&src[chanNum * (*(index + 4) + pos) + 2]), 14);
    }

    template<>
    CV_ALWAYS_INLINE void v_gather_pixel_map<4>(v_uint8x16& vec, const uchar src[], const short* index, const int pos)
    {
        int chanNum = 4;

        // pixel_1 (rgbx)
        vec.val = _mm_insert_epi32(vec.val, *reinterpret_cast<const int*>(&src[chanNum * (*index + pos)]), 0);
        // pixel_2 (rgbx)
        vec.val = _mm_insert_epi32(vec.val, *reinterpret_cast<const int*>(&src[chanNum * (*(index + 1) + pos)]), 1);
        // pixel_3
        vec.val = _mm_insert_epi32(vec.val, *reinterpret_cast<const int*>(&src[chanNum * (*(index + 2) + pos)]), 2);
        // pixel_4
        vec.val = _mm_insert_epi32(vec.val, *reinterpret_cast<const int*>(&src[chanNum * (*(index + 3) + pos)]), 3);
    }
}  // namespace

CV_ALWAYS_INLINE void v_set_alpha(const short* alpha, v_int16x8& a1, v_int16x8& a2)
{
    a1.val = _mm_setr_epi16(*alpha, *alpha, *alpha, *(alpha + 1), *(alpha + 1), *(alpha + 1),
                            *(alpha + 2), *(alpha + 2));
    a2.val = _mm_setr_epi16(*(alpha + 2), *(alpha + 3), *(alpha + 3), *(alpha + 3),
                            *(alpha + 4), *(alpha + 4), *(alpha + 4), 0);
}

CV_ALWAYS_INLINE void vertical_anyLPI(const uchar* src0, const uchar* src1,
                                      uchar* tmp, const int inLength,
                                      const short beta) {
    constexpr int nlanes = static_cast<int>(v_uint8::nlanes);

    const int half_nlanes = nlanes/2;
    int w = 0;
    for (;;) {
        for (; w <= inLength - nlanes; w += nlanes) {
            v_int16 s0 = v_reinterpret_as_s16(vx_load_expand(&src0[w]));
            v_int16 s1 = v_reinterpret_as_s16(vx_load_expand(&src1[w]));
            v_int16 s2 = v_reinterpret_as_s16(vx_load_expand(&src0[w + half_nlanes]));
            v_int16 s3 = v_reinterpret_as_s16(vx_load_expand(&src1[w + half_nlanes]));
            v_int16 res1 = v_mulhrs(s0 - s1, beta) + s1;
            v_int16 res2 = v_mulhrs(s2 - s3, beta) + s3;

            vx_store(tmp + w, v_pack_u(res1, res2));
        }

        if (w < inLength) {
            w = inLength - nlanes;
            continue;
        }
        break;
    }
}

template<int chanNum>
CV_ALWAYS_INLINE void horizontal_anyLPI(uint8_t* dst,
                                        const uchar* src, const short mapsx[],
                                        const short alpha[], const int width)
{
    constexpr int nlanes = static_cast<int>(v_uint8::nlanes);
    constexpr int pixels = nlanes / 3;

    int x = 0;
    v_uint16 a1, a2, b1, b2;
    v_uint8 a, b;
    v_int16 a00, a01;
    for (; width >= pixels;) {
        for (; x <= width - pixels; x += pixels) {
            v_set_alpha(&alpha[x], a00, a01);
            v_gather_pixel_map<chanNum>(a, src, &mapsx[x], 0);
            v_gather_pixel_map<chanNum>(b, src, &mapsx[x], 1);

            v_expand(a, a1, a2);
            v_expand(b, b1, b2);
            v_int16 a_1 = v_reinterpret_as_s16(a1);
            v_int16 a_2 = v_reinterpret_as_s16(a2);
            v_int16 b_1 = v_reinterpret_as_s16(b1);
            v_int16 b_2 = v_reinterpret_as_s16(b2);

            v_int16 r_1 = v_mulhrs(a_1 - b_1, a00) + b_1;
            v_int16 r_2 = v_mulhrs(a_2 - b_2, a01) + b_2;

            v_uint8 res = v_pack_u(r_1, r_2);
            vx_store(&dst[chanNum * x], res);
        }
        if (x < width) {
            x = width - pixels;
            continue;
        }
        break;
    }
}

template<int chanNum>
CV_ALWAYS_INLINE void calcRowLinear_8UC_Impl_(uint8_t* dst[],
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
    constexpr int half_nlanes = v_uint8::nlanes / 2;

    if (!xRatioEq && !yRatioEq) {
        int inLength = inSz.width * chanNum;

        if (lpi == 4)
        {
            // vertical pass
            GAPI_DbgAssert(inSz.width >= half_nlanes);

            __m128i b0 = _mm_set1_epi16(beta[0]);
            __m128i b1 = _mm_set1_epi16(beta[1]);
            __m128i b2 = _mm_set1_epi16(beta[2]);
            __m128i b3 = _mm_set1_epi16(beta[3]);
            __m128i zero = _mm_setzero_si128();
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

                    __m128i q4 = _mm_shuffle_epi8(q2, _mm_setr_epi8(0, 8, 4, 12, 1, 9, 5, 13, 2, 10, 6, 14, 3, 11, 7, 15));
                    __m128i q5 = _mm_shuffle_epi8(q3, _mm_setr_epi8(0, 8, 4, 12, 1, 9, 5, 13, 2, 10, 6, 14, 3, 11, 7, 15));

                    _mm_storeu_si128(reinterpret_cast<__m128i*>(&tmp[4 * w + 0]), q4);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(&tmp[4 * w + 16]), q5);
                }

                if (w < inSz.width * chanNum) {
                    w = inSz.width * chanNum - half_nlanes;
                }
            }

            // horizontal pass
            __m128i shuf_mask = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);

            for (int x = 0; outSz.width >= half_nlanes; )
            {
                for (; x <= outSz.width - half_nlanes; x += half_nlanes)
                {
                    __m128i a10 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&clone[4 * x]));
                    //__m128i a32 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&clone[4 * (x + 2)]));

                    // load 3 channels of first pixel from first pair of 4-couple scope
                    __m128i pix1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&tmp[4 * (chanNum * mapsx[x])]));
                    // insert first channel from next couple of pixels to completely fill the simd vector
                    pix1 = _mm_insert_epi32(pix1, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * mapsx[x + 1])]), 3);

                    // load 3 channels of neighbor pixel from first pair of 4-couple scope
                    __m128i pix2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&tmp[4 * (chanNum * (mapsx[x] + 1))]));
                    // insert first channel from next couple of pixels to completely fill the simd vector
                    pix2 = _mm_insert_epi32(pix2, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * (mapsx[x + 1] + 1))]), 3);

                    // expand 8-bit data to 16-bit
                    __m128i val_0 = _mm_unpacklo_epi8(pix1, zero);
                    __m128i val_1 = _mm_unpacklo_epi8(pix2, zero);

                    // expand 8-bit data to 16-bit
                    __m128i val_2 = _mm_unpackhi_epi8(pix1, zero);
                    __m128i val_3 = _mm_unpackhi_epi8(pix2, zero);

                    // the main calculations
                    __m128i t0_0 = _mm_mulhrs_epi16(_mm_sub_epi16(val_0, val_1), a10);
                    __m128i t1_0 = _mm_mulhrs_epi16(_mm_sub_epi16(val_2, val_3), a10);
                    __m128i r0_0 = _mm_add_epi16(val_1, t0_0);
                    __m128i r1_0 = _mm_add_epi16(val_3, t1_0);

                    // pack 16-bit data to 8-bit
                    __m128i q0_0 = _mm_packus_epi16(r0_0, r1_0);
                    // gather data from the same lines together
                    __m128i res1 = _mm_shuffle_epi8(q0_0, shuf_mask);

                    val_0 = _mm_unpacklo_epi8(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(&tmp[4 * (chanNum * mapsx[x + 1] + 1)])), zero);
                    val_1 = _mm_unpacklo_epi8(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(&tmp[4 * (chanNum * (mapsx[x + 1] + 1) + 1)])), zero);

                    val_2 = _mm_insert_epi64(val_2, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * mapsx[x + 2])]), 0);
                    val_3 = _mm_insert_epi64(val_3, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * (mapsx[x + 2] + 1))]), 0);

                    val_2 = _mm_unpacklo_epi8(val_2, zero);
                    val_3 = _mm_unpacklo_epi8(val_3, zero);

                    __m128i t0_1 = _mm_mulhrs_epi16(_mm_sub_epi16(val_0, val_1), a10);
                    __m128i t1_1 = _mm_mulhrs_epi16(_mm_sub_epi16(val_2, val_3), a10);
                    __m128i r0_1 = _mm_add_epi16(val_1, t0_1);
                    __m128i r1_1 = _mm_add_epi16(val_3, t1_1);

                    __m128i q0_1 = _mm_packus_epi16(r0_1, r1_1);
                    __m128i res2 = _mm_shuffle_epi8(q0_1, shuf_mask);

                    __m128i pix7 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&tmp[4 * (chanNum * (mapsx[x + 3] - 1) + 2)]));
                    pix7 = _mm_insert_epi32(pix7, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * mapsx[x + 2] + 2)]), 0);

                    __m128i pix8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&tmp[4 * (chanNum * mapsx[x + 3] + 2)]));
                    pix8 = _mm_insert_epi32(pix8, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * (mapsx[x + 2] + 1) + 2)]), 0);

                    val_0 = _mm_unpacklo_epi8(pix7, zero);
                    val_1 = _mm_unpacklo_epi8(pix8, zero);

                    val_2 = _mm_unpackhi_epi8(pix7, zero);
                    val_3 = _mm_unpackhi_epi8(pix8, zero);

                    // the main calculations
                    __m128i t0_2 = _mm_mulhrs_epi16(_mm_sub_epi16(val_0, val_1), a10);
                    __m128i t1_2 = _mm_mulhrs_epi16(_mm_sub_epi16(val_2, val_3), a10);
                    __m128i r0_2 = _mm_add_epi16(val_1, t0_2);
                    __m128i r1_2 = _mm_add_epi16(val_3, t1_2);

                    // pack 16-bit data to 8-bit
                    __m128i q0_2 = _mm_packus_epi16(r0_2, r1_2);
                    __m128i res3 = _mm_shuffle_epi8(q0_2, shuf_mask);

                    __m128i pix9 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&tmp[4 * (chanNum * mapsx[x + 4])]));
                    // insert first channel from next couple of pixels to completely fill the simd vector
                    pix9 = _mm_insert_epi32(pix9, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * mapsx[x + 5])]), 3);

                    // load 3 channels of neighbor pixel from first pair of 4-couple scope
                    __m128i pix10 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&tmp[4 * (chanNum * (mapsx[x + 4] + 1))]));
                    // insert first channel from next couple of pixels to completely fill the simd vector
                    pix10 = _mm_insert_epi32(pix10, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * (mapsx[x + 5] + 1))]), 3);

                    // expand 8-bit data to 16-bit
                    val_0 = _mm_unpacklo_epi8(pix9, zero);
                    val_1 = _mm_unpacklo_epi8(pix10, zero);

                    // expand 8-bit data to 16-bit
                    val_2 = _mm_unpackhi_epi8(pix9, zero);
                    val_3 = _mm_unpackhi_epi8(pix10, zero);

                    // the main calculations
                    __m128i t0_3 = _mm_mulhrs_epi16(_mm_sub_epi16(val_0, val_1), a10);
                    __m128i t1_3 = _mm_mulhrs_epi16(_mm_sub_epi16(val_2, val_3), a10);
                    __m128i r0_3 = _mm_add_epi16(val_1, t0_3);
                    __m128i r1_3 = _mm_add_epi16(val_3, t1_3);

                    // pack 16-bit data to 8-bit
                    __m128i q0_3 = _mm_packus_epi16(r0_3, r1_3);
                    // gather data from the same lines together
                    __m128i res4 = _mm_shuffle_epi8(q0_3, shuf_mask);

                    val_0 = _mm_unpacklo_epi8(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(&tmp[4 * (chanNum *  mapsx[x + 5]      + 1)])), zero);
                    val_1 = _mm_unpacklo_epi8(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(&tmp[4 * (chanNum * (mapsx[x + 5] + 1) + 1)])), zero);

                    val_2 = _mm_insert_epi64(val_2, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * mapsx[x + 6])]), 0);
                    val_3 = _mm_insert_epi64(val_3, *reinterpret_cast<const int64_t*>(&tmp[4 * (chanNum * (mapsx[x + 6] + 1))]), 0);

                    val_2 = _mm_unpacklo_epi8(val_2, zero);
                    val_3 = _mm_unpacklo_epi8(val_3, zero);

                    __m128i t0_4 = _mm_mulhrs_epi16(_mm_sub_epi16(val_0, val_1), a10);
                    __m128i t1_4 = _mm_mulhrs_epi16(_mm_sub_epi16(val_2, val_3), a10);
                    __m128i r0_4 = _mm_add_epi16(val_1, t0_4);
                    __m128i r1_4 = _mm_add_epi16(val_3, t1_4);

                    __m128i q0_4 = _mm_packus_epi16(r0_4, r1_4);
                    __m128i res5 = _mm_shuffle_epi8(q0_4, shuf_mask);

                    __m128i pix15 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&tmp[4 * (chanNum * (mapsx[x + 7] - 1) + 2)]));
                    pix15 = _mm_insert_epi32(pix15, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * mapsx[x + 6] + 2)]), 0);

                    __m128i pix16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&tmp[4 * (chanNum * mapsx[x + 7]   + 2)]));
                    pix16 = _mm_insert_epi32(pix16, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * (mapsx[x + 6] + 1) + 2)]), 0);

                    val_0 = _mm_unpacklo_epi8(pix15, zero);
                    val_1 = _mm_unpacklo_epi8(pix16, zero);

                    val_2 = _mm_unpackhi_epi8(pix15, zero);
                    val_3 = _mm_unpackhi_epi8(pix16, zero);

                    // the main calculations
                    __m128i t0_5 = _mm_mulhrs_epi16(_mm_sub_epi16(val_0, val_1), a10);
                    __m128i t1_5 = _mm_mulhrs_epi16(_mm_sub_epi16(val_2, val_3), a10);
                    __m128i r0_5 = _mm_add_epi16(val_1, t0_5);
                    __m128i r1_5 = _mm_add_epi16(val_3, t1_5);

                    // pack 16-bit data to 8-bit
                    __m128i q0_5 = _mm_packus_epi16(r0_5, r1_5);
                    __m128i res6 = _mm_shuffle_epi8(q0_5, shuf_mask);

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

                    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[0][3 * x]), bl13);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[1][3 * x]), bl24);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[2][3 * x]), bl31);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[3][3 * x]), bl42);
                    _mm_storel_epi64(reinterpret_cast<__m128i*>(&dst[0][3 * x + 16]), bl5);
                    _mm_storel_epi64(reinterpret_cast<__m128i*>(&dst[1][3 * x + 16]), bl6);
                    _mm_storel_epi64(reinterpret_cast<__m128i*>(&dst[2][3 * x + 16]), _mm_srli_si128(bl5, 8));
                    _mm_storel_epi64(reinterpret_cast<__m128i*>(&dst[3][3 * x + 16]), _mm_srli_si128(bl6, 8));
                }

                if (x < outSz.width) {
                    x = outSz.width - half_nlanes;
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
                vertical_anyLPI(s0, s1, tmp, inLength, beta0);

                // horizontal pass
                horizontal_anyLPI<chanNum>(dst[l], tmp, mapsx, alpha, outSz.width);
            }
        }
    } else if (!xRatioEq) {
        GAPI_DbgAssert(yRatioEq);

        for (int l = 0; l < lpi; ++l) {
            const uchar* src = src0[l];

            // horizontal pass
            horizontal_anyLPI<chanNum>(dst[l], src, mapsx, alpha, outSz.width);
        }
    } else if (!yRatioEq) {
        GAPI_DbgAssert(xRatioEq);
        int inLength = inSz.width*chanNum;  // == outSz.width

        for (int l = 0; l < lpi; ++l) {
            short beta0 = beta[l];
            const uchar* s0 = src0[l];
            const uchar* s1 = src1[l];

            // vertical pass
            vertical_anyLPI(s0, s1, dst[l], inLength, beta0);
        }
    } else {
        GAPI_DbgAssert(xRatioEq && yRatioEq);
        int length = inSz.width;  // == outSz.width

        for (int l = 0; l < lpi; ++l) {
            memcpy(dst[l], src0[l], length);
        }
    }
}
} // namespace fliud
} // namespace gapi
} // namespace cv
#endif // !defined(GAPI_STANDALONE)
