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
// Copyright (C) 2000-2020 Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2014, Itseez Inc., all rights reserved.
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

#include "opencv2/core/hal/intrin.hpp"

#if CV_AVX512_SKX
#include "sumpixels.avx512_skx.hpp"
#endif

namespace cv { namespace hal {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

// forward declarations
bool integral_SIMD(
        int depth, int sdepth, int sqdepth,
        const uchar* src, size_t srcstep,
        uchar* sum, size_t sumstep,
        uchar* sqsum, size_t sqsumstep,
        uchar* tilted, size_t tstep,
        int width, int height, int cn);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
namespace {

template <typename T, typename ST, typename QT>
struct Integral_SIMD
{
    bool operator()(const T *, size_t,
                    ST *, size_t,
                    QT *, size_t,
                    ST *, size_t,
                    int, int, int) const
    {
        return false;
    }
};

#if CV_AVX512_SKX
template <>
struct Integral_SIMD<uchar, double, double> {
    Integral_SIMD() {};


    bool operator()(const uchar *src, size_t _srcstep,
                    double *sum,      size_t _sumstep,
                    double *sqsum,    size_t _sqsumstep,
                    double *tilted,   size_t _tiltedstep,
                    int width, int height, int cn) const
    {
        CV_UNUSED(_tiltedstep);
        // TODO:  Add support for 1 channel input (WIP)
        if (!tilted && (cn <= 4))
        {
            calculate_integral_avx512(src, _srcstep, sum, _sumstep,
                                      sqsum, _sqsumstep, width, height, cn);
            return true;
        }
        return false;
    }

};
#endif

#if CV_SIMD && CV_SIMD_WIDTH <= 64

template <>
struct Integral_SIMD<uchar, int, double>
{
    Integral_SIMD() {}

    bool operator()(const uchar * src, size_t _srcstep,
                    int * sum, size_t _sumstep,
                    double * sqsum, size_t,
                    int * tilted, size_t,
                    int width, int height, int cn) const
    {
        if (sqsum || tilted || cn != 1)
            return false;

        // the first iteration
        memset(sum, 0, (width + 1) * sizeof(int));

        // the others
        for (int i = 0; i < height; ++i)
        {
            const uchar * src_row = src + _srcstep * i;
            int * prev_sum_row = (int *)((uchar *)sum + _sumstep * i) + 1;
            int * sum_row = (int *)((uchar *)sum + _sumstep * (i + 1)) + 1;

            sum_row[-1] = 0;

            v_int32 prev = vx_setzero_s32();
            int j = 0;
            for ( ; j + v_uint16::nlanes <= width; j += v_uint16::nlanes)
            {
                v_int16 el8 = v_reinterpret_as_s16(vx_load_expand(src_row + j));
                v_int32 el4l, el4h;
#if CV_AVX2 && CV_SIMD_WIDTH == 32
                __m256i vsum = _mm256_add_epi16(el8.val, _mm256_slli_si256(el8.val, 2));
                vsum = _mm256_add_epi16(vsum, _mm256_slli_si256(vsum, 4));
                vsum = _mm256_add_epi16(vsum, _mm256_slli_si256(vsum, 8));
                __m256i shmask = _mm256_set1_epi32(7);
                el4l.val = _mm256_add_epi32(_mm256_cvtepi16_epi32(_v256_extract_low(vsum)), prev.val);
                el4h.val = _mm256_add_epi32(_mm256_cvtepi16_epi32(_v256_extract_high(vsum)), _mm256_permutevar8x32_epi32(el4l.val, shmask));
                prev.val = _mm256_permutevar8x32_epi32(el4h.val, shmask);
#else
                el8 += v_rotate_left<1>(el8);
                el8 += v_rotate_left<2>(el8);
#if CV_SIMD_WIDTH >= 32
                el8 += v_rotate_left<4>(el8);
#if CV_SIMD_WIDTH == 64
                el8 += v_rotate_left<8>(el8);
#endif
#endif
                v_expand(el8, el4l, el4h);
                el4l += prev;
                el4h += el4l;

                prev = v_broadcast_element<v_int32::nlanes - 1>(el4h);
#endif
                v_store(sum_row + j                  , el4l + vx_load(prev_sum_row + j                  ));
                v_store(sum_row + j + v_int32::nlanes, el4h + vx_load(prev_sum_row + j + v_int32::nlanes));
            }

            for (int v = sum_row[j - 1] - prev_sum_row[j - 1]; j < width; ++j)
                sum_row[j] = (v += src_row[j]) + prev_sum_row[j];
        }
        return true;
    }
};

template <>
struct Integral_SIMD<uchar, float, double>
{
    Integral_SIMD() {}

    bool operator()(const uchar * src, size_t _srcstep,
        float * sum, size_t _sumstep,
        double * sqsum, size_t,
        float * tilted, size_t,
        int width, int height, int cn) const
    {
        if (sqsum || tilted || cn != 1)
            return false;

        // the first iteration
        memset(sum, 0, (width + 1) * sizeof(int));

        // the others
        for (int i = 0; i < height; ++i)
        {
            const uchar * src_row = src + _srcstep * i;
            float * prev_sum_row = (float *)((uchar *)sum + _sumstep * i) + 1;
            float * sum_row = (float *)((uchar *)sum + _sumstep * (i + 1)) + 1;

            sum_row[-1] = 0;

            v_float32 prev = vx_setzero_f32();
            int j = 0;
            for (; j + v_uint16::nlanes <= width; j += v_uint16::nlanes)
            {
                v_int16 el8 = v_reinterpret_as_s16(vx_load_expand(src_row + j));
                v_float32 el4l, el4h;
#if CV_AVX2 && CV_SIMD_WIDTH == 32
                __m256i vsum = _mm256_add_epi16(el8.val, _mm256_slli_si256(el8.val, 2));
                vsum = _mm256_add_epi16(vsum, _mm256_slli_si256(vsum, 4));
                vsum = _mm256_add_epi16(vsum, _mm256_slli_si256(vsum, 8));
                __m256i shmask = _mm256_set1_epi32(7);
                el4l.val = _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_v256_extract_low(vsum))), prev.val);
                el4h.val = _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_v256_extract_high(vsum))), _mm256_permutevar8x32_ps(el4l.val, shmask));
                prev.val = _mm256_permutevar8x32_ps(el4h.val, shmask);
#else
                el8 += v_rotate_left<1>(el8);
                el8 += v_rotate_left<2>(el8);
#if CV_SIMD_WIDTH >= 32
                el8 += v_rotate_left<4>(el8);
#if CV_SIMD_WIDTH == 64
                el8 += v_rotate_left<8>(el8);
#endif
#endif
                v_int32 el4li, el4hi;
                v_expand(el8, el4li, el4hi);
                el4l = v_cvt_f32(el4li) + prev;
                el4h = v_cvt_f32(el4hi) + el4l;

                prev = v_broadcast_element<v_float32::nlanes - 1>(el4h);
#endif
                v_store(sum_row + j                    , el4l + vx_load(prev_sum_row + j                    ));
                v_store(sum_row + j + v_float32::nlanes, el4h + vx_load(prev_sum_row + j + v_float32::nlanes));
            }

            for (float v = sum_row[j - 1] - prev_sum_row[j - 1]; j < width; ++j)
                sum_row[j] = (v += src_row[j]) + prev_sum_row[j];
        }
        return true;
    }
};

#endif

} // namespace anon

bool integral_SIMD(
        int depth, int sdepth, int sqdepth,
        const uchar* src, size_t srcstep,
        uchar* sum, size_t sumstep,
        uchar* sqsum, size_t sqsumstep,
        uchar* tilted, size_t tstep,
        int width, int height, int cn)
{
    CV_INSTRUMENT_REGION();

#define ONE_CALL(T, ST, QT) \
    return Integral_SIMD<T, ST, QT>()((const T*)src, srcstep, (ST*)sum, sumstep, (QT*)sqsum, sqsumstep, (ST*)tilted, tstep, width, height, cn)

    if( depth == CV_8U && sdepth == CV_32S && sqdepth == CV_64F )
        ONE_CALL(uchar, int, double);
    else if( depth == CV_8U && sdepth == CV_32S && sqdepth == CV_32F )
        ONE_CALL(uchar, int, float);
    else if( depth == CV_8U && sdepth == CV_32S && sqdepth == CV_32S )
        ONE_CALL(uchar, int, int);
    else if( depth == CV_8U && sdepth == CV_32F && sqdepth == CV_64F )
        ONE_CALL(uchar, float, double);
    else if( depth == CV_8U && sdepth == CV_32F && sqdepth == CV_32F )
        ONE_CALL(uchar, float, float);
    else if( depth == CV_8U && sdepth == CV_64F && sqdepth == CV_64F )
        ONE_CALL(uchar, double, double);
    else if( depth == CV_16U && sdepth == CV_64F && sqdepth == CV_64F )
        ONE_CALL(ushort, double, double);
    else if( depth == CV_16S && sdepth == CV_64F && sqdepth == CV_64F )
        ONE_CALL(short, double, double);
    else if( depth == CV_32F && sdepth == CV_32F && sqdepth == CV_64F )
        ONE_CALL(float, float, double);
    else if( depth == CV_32F && sdepth == CV_32F && sqdepth == CV_32F )
        ONE_CALL(float, float, float);
    else if( depth == CV_32F && sdepth == CV_64F && sqdepth == CV_64F )
        ONE_CALL(float, double, double);
    else if( depth == CV_64F && sdepth == CV_64F && sqdepth == CV_64F )
        ONE_CALL(double, double, double);
    else
        return false;

#undef ONE_CALL
}

#endif
CV_CPU_OPTIMIZATION_NAMESPACE_END
}} // cv::hal::
