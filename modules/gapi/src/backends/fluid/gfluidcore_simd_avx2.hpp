// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#if !defined(GAPI_STANDALONE)

#include "opencv2/gapi/own/saturate.hpp"

#include "opencv2/core.hpp"

#include <immintrin.h>

#ifdef CV_AVX2
#undef CV_AVX2
#endif

#define CV_AVX2 1

#define CV_CPU_HAS_SUPPORT_SSE2 1

#ifdef CV_SIMD256
#undef CV_SIMD256
#endif

#define CV_SIMD256 1
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
namespace avx {
CV_ALWAYS_INLINE v_int16x16 v_mulhrs(const v_int16x16& a, const v_int16x16& b)
{
    return v_int16x16(_mm256_mulhrs_epi16(a.val, b.val));
}

CV_ALWAYS_INLINE v_int16x16 v_mulhrs(const v_int16x16& a, short b)
{
    return v_mulhrs(a, v256_setall_s16(b));
}

namespace {
    template<int chanNum>
    CV_ALWAYS_INLINE void v_gather_pixel_map(v_uint8x32&, const uchar*, const short*, const int)
    {
        CV_Assert("Unsupported number of channel");
    }

    template<>
    CV_ALWAYS_INLINE void v_gather_pixel_map<3>(v_uint8x32& vec, const uchar src[], const short* index, const int pos)
    {
        int chanNum = 3;
        //v_setzero_u8(vec);
        // pixel_1 (rgb)
        vec.val = _mm256_insert_epi16(vec.val, *reinterpret_cast<const ushort*>(&src[chanNum * (*index + pos)]), 0);
        vec.val = _mm256_insert_epi8(vec.val, *reinterpret_cast<const uchar*>(&src[chanNum * (*index + pos) + 2]), 2);
        // pixel_2 (rgb)
        vec.val = _mm256_insert_epi8(vec.val, *reinterpret_cast<const uchar*>(&src[chanNum * (*(index + 1) + pos)]), 3);
        vec.val = _mm256_insert_epi16(vec.val, *reinterpret_cast<const ushort*>(&src[chanNum * (*(index + 1) + pos) + 1]), 2);
        // pixel_3
        vec.val = _mm256_insert_epi16(vec.val, *reinterpret_cast<const ushort*>(&src[chanNum * (*(index + 2) + pos)]), 3);
        vec.val = _mm256_insert_epi8(vec.val, *reinterpret_cast<const uchar*>(&src[chanNum * (*(index + 2) + pos) + 2]), 8);
        // pixel_4
        vec.val = _mm256_insert_epi8(vec.val, *reinterpret_cast<const uchar*>(&src[chanNum * (*(index + 3) + pos)]), 9);
        vec.val = _mm256_insert_epi16(vec.val, *reinterpret_cast<const ushort*>(&src[chanNum * (*(index + 3) + pos) + 1]), 5);
        // pixel_5
        vec.val = _mm256_insert_epi16(vec.val, *reinterpret_cast<const ushort*>(&src[chanNum * (*(index + 4) + pos)]), 6);
        vec.val = _mm256_insert_epi8(vec.val, *reinterpret_cast<const uchar*>(&src[chanNum * (*(index + 4) + pos) + 2]), 14);
        // pixel_6
        vec.val = _mm256_insert_epi8(vec.val, *reinterpret_cast<const uchar*>(&src[chanNum * (*(index + 5) + pos)]), 15);
        vec.val = _mm256_insert_epi16(vec.val, *reinterpret_cast<const ushort*>(&src[chanNum * (*(index + 5) + pos) + 1]), 8);
        // pixel_7
        vec.val = _mm256_insert_epi16(vec.val, *reinterpret_cast<const ushort*>(&src[chanNum * (*(index + 6) + pos)]), 9);
        vec.val = _mm256_insert_epi8(vec.val, *reinterpret_cast<const uchar*>(&src[chanNum * (*(index + 6) + pos) + 2]), 20);
        // pixel_8
        vec.val = _mm256_insert_epi8(vec.val, *reinterpret_cast<const uchar*>(&src[chanNum * (*(index + 7) + pos)]), 21);
        vec.val = _mm256_insert_epi16(vec.val, *reinterpret_cast<const ushort*>(&src[chanNum * (*(index + 7) + pos) + 1]), 11);
        // pixel_9
        vec.val = _mm256_insert_epi16(vec.val, *reinterpret_cast<const ushort*>(&src[chanNum * (*(index + 8) + pos)]), 12);
        vec.val = _mm256_insert_epi8(vec.val, *reinterpret_cast<const uchar*>(&src[chanNum * (*(index + 8) + pos) + 2]), 26);
        // pixel_10
        vec.val = _mm256_insert_epi8(vec.val, *reinterpret_cast<const uchar*>(&src[chanNum * (*(index + 9) + pos)]), 27);
        vec.val = _mm256_insert_epi16(vec.val, *reinterpret_cast<const ushort*>(&src[chanNum * (*(index + 9) + pos) + 1]), 14);
    }

    template<>
    CV_ALWAYS_INLINE void v_gather_pixel_map<4>(v_uint8x32& vec, const uchar src[], const short* index, const int pos)
    {
        int chanNum = 4;

        // pixel_1 (rgbx)
        vec.val = _mm256_insert_epi32(vec.val, *reinterpret_cast<const int*>(&src[chanNum * (*index + pos)]), 0);
        // pixel_2 (rgbx)
        vec.val = _mm256_insert_epi32(vec.val, *reinterpret_cast<const int*>(&src[chanNum * (*(index + 1) + pos)]), 1);
        // pixel_3
        vec.val = _mm256_insert_epi32(vec.val, *reinterpret_cast<const int*>(&src[chanNum * (*(index + 2) + pos)]), 2);
        // pixel_4
        vec.val = _mm256_insert_epi32(vec.val, *reinterpret_cast<const int*>(&src[chanNum * (*(index + 3) + pos)]), 3);
        // pixel_5
        vec.val = _mm256_insert_epi32(vec.val, *reinterpret_cast<const int*>(&src[chanNum * (*(index + 4) + pos)]), 4);
        // pixel_6
        vec.val = _mm256_insert_epi32(vec.val, *reinterpret_cast<const int*>(&src[chanNum * (*(index + 5) + pos)]), 5);
        // pixel_7
        vec.val = _mm256_insert_epi32(vec.val, *reinterpret_cast<const int*>(&src[chanNum * (*(index + 6) + pos)]), 6);
        // pixel_8
        vec.val = _mm256_insert_epi32(vec.val, *reinterpret_cast<const int*>(&src[chanNum * (*(index + 7) + pos)]), 7);
    }
}  // namespace

CV_ALWAYS_INLINE void v_set_alpha(const short* alpha, v_int16x16& a1, v_int16x16& a2)
{
    a1.val = _mm256_setr_epi16(*alpha,       *alpha,       *alpha,       *(alpha + 1), *(alpha + 1), *(alpha + 1),
                               *(alpha + 2), *(alpha + 2), *(alpha + 2), *(alpha + 3), *(alpha + 3), *(alpha + 3),
                               *(alpha + 4), *(alpha + 4), *(alpha + 4), *(alpha + 5));
    a2.val = _mm256_setr_epi16(*(alpha + 5), *(alpha + 5), *(alpha + 6), *(alpha + 6), *(alpha + 6), *(alpha + 7),
                               *(alpha + 7), *(alpha + 7), *(alpha + 8), *(alpha + 8), *(alpha + 8), *(alpha + 9),
                               *(alpha + 9), *(alpha + 9), 0, 0);
}
} // namespace avx
} // namespace fliud
} // namespace gapi
} // namespace cv
#endif // !defined(GAPI_STANDALONE)
