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
namespace opt_FP16
{
#if !defined(CV_NEON) || !CV_NEON
const static int cVectorWidth = 8;

void cvtScaleHalf_SIMD32f16f( const float* src, size_t sstep, short* dst, size_t dstep, cv::Size size )
{
    CV_INSTRUMENT_REGION()

    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( ; size.height--; src += sstep, dst += dstep )
    {
        int x = 0;
        for ( ; x <= size.width - cVectorWidth ; x += cVectorWidth )
        {
            __m256 v_src = _mm256_loadu_ps(src + x);

            // round to nearest even
            __m128i v_dst = _mm256_cvtps_ph(v_src, 0);

            _mm_storeu_si128((__m128i*)(dst + x), v_dst);
        }

        for ( ; x < size.width; x++ )
        {
            dst[x] = convertFp16SW(src[x]);
        }
    }
}

void cvtScaleHalf_SIMD16f32f( const short* src, size_t sstep, float* dst, size_t dstep, cv::Size size )
{
    CV_INSTRUMENT_REGION()

    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( ; size.height--; src += sstep, dst += dstep )
    {
        int x = 0;
        for ( ; x <= size.width - cVectorWidth ; x += cVectorWidth )
        {
            __m128i v_src = _mm_loadu_si128((__m128i*)(src + x));

            __m256 v_dst = _mm256_cvtph_ps(v_src);

            _mm256_storeu_ps(dst + x, v_dst);
        }

        for ( ; x < size.width; x++ )
        {
            dst[x] = convertFp16SW(src[x]);
        }
    }
}
#elif CV_NEON
const static int cVectorWidth = 4;

void cvtScaleHalf_SIMD32f16f( const float* src, size_t sstep, short* dst, size_t dstep, cv::Size size )
{
    CV_INSTRUMENT_REGION()

    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( ; size.height--; src += sstep, dst += dstep )
    {
        int x = 0;
        for ( ; x <= size.width - cVectorWidth ; x += cVectorWidth)
        {
            float32x4_t v_src = vld1q_f32(src + x);

            float16x4_t v_dst = vcvt_f16_f32(v_src);

            cv_vst1_f16((__fp16*)dst + x, v_dst);
        }

        for ( ; x < size.width; x++ )
        {
            dst[x] = convertFp16SW(src[x]);
        }
    }
}

void cvtScaleHalf_SIMD16f32f( const short* src, size_t sstep, float* dst, size_t dstep, cv::Size size )
{
    CV_INSTRUMENT_REGION()

    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( ; size.height--; src += sstep, dst += dstep )
    {
        int x = 0;
        for ( ; x <= size.width - cVectorWidth ; x += cVectorWidth )
        {
            float16x4_t v_src = cv_vld1_f16((__fp16*)src + x);

            float32x4_t v_dst = vcvt_f32_f16(v_src);

            vst1q_f32(dst + x, v_dst);
        }

        for ( ; x < size.width; x++ )
        {
            dst[x] = convertFp16SW(src[x]);
        }
    }
}
#else
#error "Unsupported build configuration"
#endif
}
}
/* End of file. */
