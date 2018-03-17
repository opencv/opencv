// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


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

} // cv::
