/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include "convert.hpp"

namespace cv
{
namespace opt_SSE4_1
{

int cvtScale_SIMD_u8u16f32_SSE41(const uchar * src, ushort * dst, int width, float scale, float shift)
{
    int x = 0;

    __m128i v_zero = _mm_setzero_si128();
    __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

    for ( ; x <= width - 8; x += 8)
    {
        __m128i v_src = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i const *)(src + x)), v_zero);
        __m128 v_src_f = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src, v_zero));
        __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

        v_src_f = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src, v_zero));
        __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

        __m128i v_dst = _mm_packus_epi32(_mm_cvtps_epi32(v_dst_0),
                                            _mm_cvtps_epi32(v_dst_1));
        _mm_storeu_si128((__m128i *)(dst + x), v_dst);
    }

    return x;
}

int cvtScale_SIMD_s8u16f32_SSE41(const schar * src, ushort * dst, int width, float scale, float shift)
{
    int x = 0;

    __m128i v_zero = _mm_setzero_si128();
    __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

    for ( ; x <= width - 8; x += 8)
    {
        __m128i v_src = _mm_srai_epi16(_mm_unpacklo_epi8(v_zero, _mm_loadl_epi64((__m128i const *)(src + x))), 8);
        __m128 v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src), 16));
        __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

        v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src), 16));
        __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

        __m128i v_dst = _mm_packus_epi32(_mm_cvtps_epi32(v_dst_0),
                                            _mm_cvtps_epi32(v_dst_1));
        _mm_storeu_si128((__m128i *)(dst + x), v_dst);
    }

    return x;
}

int cvtScale_SIMD_u16u16f32_SSE41(const ushort * src, ushort * dst, int width, float scale, float shift)
{
    int x = 0;

    __m128i v_zero = _mm_setzero_si128();
    __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

    for ( ; x <= width - 8; x += 8)
    {
        __m128i v_src = _mm_loadu_si128((__m128i const *)(src + x));
        __m128 v_src_f = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src, v_zero));
        __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

        v_src_f = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src, v_zero));
        __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

        __m128i v_dst = _mm_packus_epi32(_mm_cvtps_epi32(v_dst_0),
                                            _mm_cvtps_epi32(v_dst_1));
        _mm_storeu_si128((__m128i *)(dst + x), v_dst);
    }

    return x;
}

int cvtScale_SIMD_s16u16f32_SSE41(const short * src, ushort * dst, int width, float scale, float shift)
{
    int x = 0;

    __m128i v_zero = _mm_setzero_si128();
    __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

    for ( ; x <= width - 8; x += 8)
    {
        __m128i v_src = _mm_loadu_si128((__m128i const *)(src + x));
        __m128 v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src), 16));
        __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

        v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src), 16));
        __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

        __m128i v_dst = _mm_packus_epi32(_mm_cvtps_epi32(v_dst_0),
                                            _mm_cvtps_epi32(v_dst_1));
        _mm_storeu_si128((__m128i *)(dst + x), v_dst);
    }

    return x;
}

int cvtScale_SIMD_s32u16f32_SSE41(const int * src, ushort * dst, int width, float scale, float shift)
{
    int x = 0;

    __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

    for ( ; x <= width - 8; x += 8)
    {
        __m128i v_src = _mm_loadu_si128((__m128i const *)(src + x));
        __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(v_src), v_scale), v_shift);

        v_src = _mm_loadu_si128((__m128i const *)(src + x + 4));
        __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(v_src), v_scale), v_shift);

        __m128i v_dst = _mm_packus_epi32(_mm_cvtps_epi32(v_dst_0),
                                            _mm_cvtps_epi32(v_dst_1));
        _mm_storeu_si128((__m128i *)(dst + x), v_dst);
    }

    return x;
}

int cvtScale_SIMD_f32u16f32_SSE41(const float * src, ushort * dst, int width, float scale, float shift)
{
    int x = 0;

    __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

    for ( ; x <= width - 8; x += 8)
    {
        __m128 v_src = _mm_loadu_ps(src + x);
        __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

        v_src = _mm_loadu_ps(src + x + 4);
        __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

        __m128i v_dst = _mm_packus_epi32(_mm_cvtps_epi32(v_dst_0),
                                            _mm_cvtps_epi32(v_dst_1));
        _mm_storeu_si128((__m128i *)(dst + x), v_dst);
    }

    return x;
}

int cvtScale_SIMD_f64u16f32_SSE41(const double * src, ushort * dst, int width, float scale, float shift)
{
    int x = 0;

    __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

    for ( ; x <= width - 8; x += 8)
    {
        __m128 v_src = _mm_movelh_ps(_mm_cvtpd_ps(_mm_loadu_pd(src + x)),
                                        _mm_cvtpd_ps(_mm_loadu_pd(src + x + 2)));
        __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

        v_src = _mm_movelh_ps(_mm_cvtpd_ps(_mm_loadu_pd(src + x + 4)),
                                _mm_cvtpd_ps(_mm_loadu_pd(src + x + 6)));
        __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

        __m128i v_dst = _mm_packus_epi32(_mm_cvtps_epi32(v_dst_0),
                                            _mm_cvtps_epi32(v_dst_1));
        _mm_storeu_si128((__m128i *)(dst + x), v_dst);
    }

    return x;
}

int Cvt_SIMD_f64u16_SSE41(const double * src, ushort * dst, int width)
{
    int x = 0;

    for ( ; x <= width - 8; x += 8)
    {
        __m128 v_src0 = _mm_cvtpd_ps(_mm_loadu_pd(src + x));
        __m128 v_src1 = _mm_cvtpd_ps(_mm_loadu_pd(src + x + 2));
        __m128 v_src2 = _mm_cvtpd_ps(_mm_loadu_pd(src + x + 4));
        __m128 v_src3 = _mm_cvtpd_ps(_mm_loadu_pd(src + x + 6));

        v_src0 = _mm_movelh_ps(v_src0, v_src1);
        v_src1 = _mm_movelh_ps(v_src2, v_src3);

        __m128i v_dst = _mm_packus_epi32(_mm_cvtps_epi32(v_src0),
                                            _mm_cvtps_epi32(v_src1));
        _mm_storeu_si128((__m128i *)(dst + x), v_dst);
    }

    return x;
}

}
}

/* End of file. */
