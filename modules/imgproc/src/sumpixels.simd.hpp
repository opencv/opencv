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
        if (sqsum || tilted || cn > 4)
            return false;
#if !CV_SSE4_1 && CV_SSE2
        // 3 channel code is slower for SSE2 & SSE3
        if (cn == 3)
            return false;
#endif

        width *= cn;

        // the first iteration
        memset(sum, 0, (width + cn) * sizeof(int));

        if (cn == 1)
        {
            // the others
            for (int i = 0; i < height; ++i)
            {
                const uchar * src_row = src + _srcstep * i;
                int * prev_sum_row = (int *)((uchar *)sum + _sumstep * i) + 1;
                int * sum_row = (int *)((uchar *)sum + _sumstep * (i + 1)) + 1;

                sum_row[-1] = 0;

                v_int32 prev = vx_setzero_s32();
                int j = 0;
                for ( ; j + VTraits<v_uint16>::vlanes() <= width; j += VTraits<v_uint16>::vlanes())
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
                    el8 = v_add(el8, v_rotate_left<1>(el8));
                    el8 = v_add(el8, v_rotate_left<2>(el8));
#if CV_SIMD_WIDTH >= 32
                    el8 += v_rotate_left<4>(el8);
#if CV_SIMD_WIDTH == 64
                    el8 += v_rotate_left<8>(el8);
#endif
#endif
                    v_expand(el8, el4l, el4h);
                    el4l = v_add(el4l, prev);
                    el4h = v_add(el4h, el4l);
                    prev = v_broadcast_highest(el4h);
#endif
                    v_store(sum_row + j                  , v_add(el4l, vx_load(prev_sum_row + j)));
                    v_store(sum_row + j + VTraits<v_int32>::vlanes(), v_add(el4h, vx_load(prev_sum_row + j + VTraits<v_int32>::vlanes())));
                }

                for (int v = sum_row[j - 1] - prev_sum_row[j - 1]; j < width; ++j)
                    sum_row[j] = (v += src_row[j]) + prev_sum_row[j];
            }
        }
        else if (cn == 2)
        {
            // the others
            v_int16 mask = vx_setall_s16((short)0xff);
            for (int i = 0; i < height; ++i)
            {
                const uchar * src_row = src + _srcstep * i;
                int * prev_sum_row = (int *)((uchar *)sum + _sumstep * i) + cn;
                int * sum_row = (int *)((uchar *)sum + _sumstep * (i + 1)) + cn;

                sum_row[-1] = sum_row[-2] = 0;

                v_int32 prev_1 = vx_setzero_s32(), prev_2 = vx_setzero_s32();
                int j = 0;
                for ( ; j + VTraits<v_uint16>::vlanes() * cn <= width; j += VTraits<v_uint16>::vlanes() * cn)
                {
                    v_int16 v_src_row = v_reinterpret_as_s16(vx_load(src_row + j));
                    v_int16 el8_1 = v_and(v_src_row, mask);
                    v_int16 el8_2 = v_reinterpret_as_s16(v_shr<8>(v_reinterpret_as_u16(v_src_row)));
                    v_int32 el4l_1, el4h_1, el4l_2, el4h_2;
#if CV_AVX2 && CV_SIMD_WIDTH == 32
                    __m256i vsum_1 = _mm256_add_epi16(el8_1.val, _mm256_slli_si256(el8_1.val, 2));
                    __m256i vsum_2 = _mm256_add_epi16(el8_2.val, _mm256_slli_si256(el8_2.val, 2));
                    vsum_1 = _mm256_add_epi16(vsum_1, _mm256_slli_si256(vsum_1, 4));
                    vsum_2 = _mm256_add_epi16(vsum_2, _mm256_slli_si256(vsum_2, 4));
                    vsum_1 = _mm256_add_epi16(vsum_1, _mm256_slli_si256(vsum_1, 8));
                    vsum_2 = _mm256_add_epi16(vsum_2, _mm256_slli_si256(vsum_2, 8));
                    __m256i shmask = _mm256_set1_epi32(7);
                    el4l_1.val = _mm256_add_epi32(_mm256_cvtepi16_epi32(_v256_extract_low(vsum_1)), prev_1.val);
                    el4l_2.val = _mm256_add_epi32(_mm256_cvtepi16_epi32(_v256_extract_low(vsum_2)), prev_2.val);
                    el4h_1.val = _mm256_add_epi32(_mm256_cvtepi16_epi32(_v256_extract_high(vsum_1)), _mm256_permutevar8x32_epi32(el4l_1.val, shmask));
                    el4h_2.val = _mm256_add_epi32(_mm256_cvtepi16_epi32(_v256_extract_high(vsum_2)), _mm256_permutevar8x32_epi32(el4l_2.val, shmask));
                    prev_1.val = _mm256_permutevar8x32_epi32(el4h_1.val, shmask);
                    prev_2.val = _mm256_permutevar8x32_epi32(el4h_2.val, shmask);
#else
                    el8_1 = v_add(el8_1, v_rotate_left<1>(el8_1));
                    el8_2 = v_add(el8_2, v_rotate_left<1>(el8_2));
                    el8_1 = v_add(el8_1, v_rotate_left<2>(el8_1));
                    el8_2 = v_add(el8_2, v_rotate_left<2>(el8_2));
#if CV_SIMD_WIDTH >= 32
                    el8_1 += v_rotate_left<4>(el8_1);
                    el8_2 += v_rotate_left<4>(el8_2);
#if CV_SIMD_WIDTH == 64
                    el8_1 += v_rotate_left<8>(el8_1);
                    el8_2 += v_rotate_left<8>(el8_2);
#endif
#endif
                    v_expand(el8_1, el4l_1, el4h_1);
                    v_expand(el8_2, el4l_2, el4h_2);
                    el4l_1 = v_add(el4l_1, prev_1);
                    el4l_2 = v_add(el4l_2, prev_2);
                    el4h_1 = v_add(el4h_1, el4l_1);
                    el4h_2 = v_add(el4h_2, el4l_2);
                    prev_1 = v_broadcast_highest(el4h_1);
                    prev_2 = v_broadcast_highest(el4h_2);
#endif
                    v_int32 el4_1, el4_2, el4_3, el4_4;
                    v_zip(el4l_1, el4l_2, el4_1, el4_2);
                    v_zip(el4h_1, el4h_2, el4_3, el4_4);
                    v_store(sum_row + j                      , v_add(el4_1, vx_load(prev_sum_row + j)));
                    v_store(sum_row + j + VTraits<v_int32>::vlanes()    , v_add(el4_2, vx_load(prev_sum_row + j + VTraits<v_int32>::vlanes())));
                    v_store(sum_row + j + VTraits<v_int32>::vlanes() * 2, v_add(el4_3, vx_load(prev_sum_row + j + VTraits<v_int32>::vlanes() * 2)));
                    v_store(sum_row + j + VTraits<v_int32>::vlanes() * 3, v_add(el4_4, vx_load(prev_sum_row + j + VTraits<v_int32>::vlanes() * 3)));
                }

                for (int v2 = sum_row[j - 1] - prev_sum_row[j - 1],
                         v1 = sum_row[j - 2] - prev_sum_row[j - 2]; j < width; j += 2)
                {
                    sum_row[j]     = (v1 += src_row[j])     + prev_sum_row[j];
                    sum_row[j + 1] = (v2 += src_row[j + 1]) + prev_sum_row[j + 1];
                }
            }
        }
#if CV_SSE4_1 || !CV_SSE2
        else if (cn == 3)
        {
            // the others
            for (int i = 0; i < height; ++i)
            {
                const uchar * src_row = src + _srcstep * i;
                int * prev_sum_row = (int *)((uchar *)sum + _sumstep * i) + cn;
                int * sum_row = (int *)((uchar *)sum + _sumstep * (i + 1)) + cn;
                int row_cache[VTraits<v_int32>::max_nlanes * 6];

                sum_row[-1] = sum_row[-2] = sum_row[-3] = 0;

                v_int32 prev_1 = vx_setzero_s32(), prev_2 = vx_setzero_s32(),
                        prev_3 = vx_setzero_s32();
                int j = 0;
                const int j_max =
                        ((_srcstep * i + (width - VTraits<v_uint16>::vlanes() * cn + VTraits<v_uint8>::vlanes() * cn)) >= _srcstep * height)
                        ? width - VTraits<v_uint8>::vlanes() * cn    // uint8 in v_load_deinterleave()
                        : width - VTraits<v_uint16>::vlanes() * cn;  // v_expand_low
                for ( ; j <= j_max; j += VTraits<v_uint16>::vlanes() * cn)
                {
                    v_uint8 v_src_row_1, v_src_row_2, v_src_row_3;
                    v_load_deinterleave(src_row + j, v_src_row_1, v_src_row_2, v_src_row_3);
                    v_int16 el8_1 = v_reinterpret_as_s16(v_expand_low(v_src_row_1));
                    v_int16 el8_2 = v_reinterpret_as_s16(v_expand_low(v_src_row_2));
                    v_int16 el8_3 = v_reinterpret_as_s16(v_expand_low(v_src_row_3));
                    v_int32 el4l_1, el4h_1, el4l_2, el4h_2, el4l_3, el4h_3;
#if CV_AVX2 && CV_SIMD_WIDTH == 32
                    __m256i vsum_1 = _mm256_add_epi16(el8_1.val, _mm256_slli_si256(el8_1.val, 2));
                    __m256i vsum_2 = _mm256_add_epi16(el8_2.val, _mm256_slli_si256(el8_2.val, 2));
                    __m256i vsum_3 = _mm256_add_epi16(el8_3.val, _mm256_slli_si256(el8_3.val, 2));
                    vsum_1 = _mm256_add_epi16(vsum_1, _mm256_slli_si256(vsum_1, 4));
                    vsum_2 = _mm256_add_epi16(vsum_2, _mm256_slli_si256(vsum_2, 4));
                    vsum_3 = _mm256_add_epi16(vsum_3, _mm256_slli_si256(vsum_3, 4));
                    vsum_1 = _mm256_add_epi16(vsum_1, _mm256_slli_si256(vsum_1, 8));
                    vsum_2 = _mm256_add_epi16(vsum_2, _mm256_slli_si256(vsum_2, 8));
                    vsum_3 = _mm256_add_epi16(vsum_3, _mm256_slli_si256(vsum_3, 8));
                    __m256i shmask = _mm256_set1_epi32(7);
                    el4l_1.val = _mm256_add_epi32(_mm256_cvtepi16_epi32(_v256_extract_low(vsum_1)), prev_1.val);
                    el4l_2.val = _mm256_add_epi32(_mm256_cvtepi16_epi32(_v256_extract_low(vsum_2)), prev_2.val);
                    el4l_3.val = _mm256_add_epi32(_mm256_cvtepi16_epi32(_v256_extract_low(vsum_3)), prev_3.val);
                    el4h_1.val = _mm256_add_epi32(_mm256_cvtepi16_epi32(_v256_extract_high(vsum_1)), _mm256_permutevar8x32_epi32(el4l_1.val, shmask));
                    el4h_2.val = _mm256_add_epi32(_mm256_cvtepi16_epi32(_v256_extract_high(vsum_2)), _mm256_permutevar8x32_epi32(el4l_2.val, shmask));
                    el4h_3.val = _mm256_add_epi32(_mm256_cvtepi16_epi32(_v256_extract_high(vsum_3)), _mm256_permutevar8x32_epi32(el4l_3.val, shmask));
                    prev_1.val = _mm256_permutevar8x32_epi32(el4h_1.val, shmask);
                    prev_2.val = _mm256_permutevar8x32_epi32(el4h_2.val, shmask);
                    prev_3.val = _mm256_permutevar8x32_epi32(el4h_3.val, shmask);
#else
                    el8_1 = v_add(el8_1,v_rotate_left<1>(el8_1));
                    el8_2 = v_add(el8_2,v_rotate_left<1>(el8_2));
                    el8_3 = v_add(el8_3,v_rotate_left<1>(el8_3));
                    el8_1 = v_add(el8_1,v_rotate_left<2>(el8_1));
                    el8_2 = v_add(el8_2,v_rotate_left<2>(el8_2));
                    el8_3 = v_add(el8_3,v_rotate_left<2>(el8_3));
#if CV_SIMD_WIDTH >= 32
                    el8_1 = v_add(el8_1, v_rotate_left<4>(el8_1));
                    el8_2 = v_add(el8_2, v_rotate_left<4>(el8_2));
                    el8_3 = v_add(el8_3, v_rotate_left<4>(el8_3));
#if CV_SIMD_WIDTH == 64
                    el8_1 = v_add(el8_1, v_rotate_left<8>(el8_1));
                    el8_2 = v_add(el8_2, v_rotate_left<8>(el8_2));
                    el8_3 = v_add(el8_3, v_rotate_left<8>(el8_3));
#endif
#endif
                    v_expand(el8_1, el4l_1, el4h_1);
                    v_expand(el8_2, el4l_2, el4h_2);
                    v_expand(el8_3, el4l_3, el4h_3);
                    el4l_1 = v_add(el4l_1, prev_1);
                    el4l_2 = v_add(el4l_2, prev_2);
                    el4l_3 = v_add(el4l_3, prev_3);
                    el4h_1 = v_add(el4h_1, el4l_1);
                    el4h_2 = v_add(el4h_2, el4l_2);
                    el4h_3 = v_add(el4h_3, el4l_3);
                    prev_1 = v_broadcast_highest(el4h_1);
                    prev_2 = v_broadcast_highest(el4h_2);
                    prev_3 = v_broadcast_highest(el4h_3);
#endif
                    v_store_interleave(row_cache                      , el4l_1, el4l_2, el4l_3);
                    v_store_interleave(row_cache + VTraits<v_int32>::vlanes() * 3, el4h_1, el4h_2, el4h_3);
                    el4l_1 = vx_load(row_cache                      );
                    el4l_2 = vx_load(row_cache + VTraits<v_int32>::vlanes()    );
                    el4l_3 = vx_load(row_cache + VTraits<v_int32>::vlanes() * 2);
                    el4h_1 = vx_load(row_cache + VTraits<v_int32>::vlanes() * 3);
                    el4h_2 = vx_load(row_cache + VTraits<v_int32>::vlanes() * 4);
                    el4h_3 = vx_load(row_cache + VTraits<v_int32>::vlanes() * 5);
                    v_store(sum_row + j                      ,            v_add(el4l_1, vx_load(prev_sum_row + j                      )));
                    v_store(sum_row + j + VTraits<v_int32>::vlanes()    , v_add(el4l_2, vx_load(prev_sum_row + j + VTraits<v_int32>::vlanes()    )));
                    v_store(sum_row + j + VTraits<v_int32>::vlanes() * 2, v_add(el4l_3, vx_load(prev_sum_row + j + VTraits<v_int32>::vlanes() * 2)));
                    v_store(sum_row + j + VTraits<v_int32>::vlanes() * 3, v_add(el4h_1, vx_load(prev_sum_row + j + VTraits<v_int32>::vlanes() * 3)));
                    v_store(sum_row + j + VTraits<v_int32>::vlanes() * 4, v_add(el4h_2, vx_load(prev_sum_row + j + VTraits<v_int32>::vlanes() * 4)));
                    v_store(sum_row + j + VTraits<v_int32>::vlanes() * 5, v_add(el4h_3, vx_load(prev_sum_row + j + VTraits<v_int32>::vlanes() * 5)));
                }

                for (int v3 = sum_row[j - 1] - prev_sum_row[j - 1],
                         v2 = sum_row[j - 2] - prev_sum_row[j - 2],
                         v1 = sum_row[j - 3] - prev_sum_row[j - 3]; j < width; j += 3)
                {
                    sum_row[j]     = (v1 += src_row[j])     + prev_sum_row[j];
                    sum_row[j + 1] = (v2 += src_row[j + 1]) + prev_sum_row[j + 1];
                    sum_row[j + 2] = (v3 += src_row[j + 2]) + prev_sum_row[j + 2];
                }
            }
        }
#endif
        else if (cn == 4)
        {
            // the others
            for (int i = 0; i < height; ++i)
            {
                const uchar * src_row = src + _srcstep * i;
                int * prev_sum_row = (int *)((uchar *)sum + _sumstep * i) + cn;
                int * sum_row = (int *)((uchar *)sum + _sumstep * (i + 1)) + cn;

                sum_row[-1] = sum_row[-2] = sum_row[-3] = sum_row[-4] = 0;

                v_int32 prev = vx_setzero_s32();
                int j = 0;
                for ( ; j + VTraits<v_uint16>::vlanes() <= width; j += VTraits<v_uint16>::vlanes())
                {
                    v_int16 el8 = v_reinterpret_as_s16(vx_load_expand(src_row + j));
                    v_int32 el4l, el4h;
#if CV_AVX2 && CV_SIMD_WIDTH == 32
                    __m256i vsum = _mm256_add_epi16(el8.val, _mm256_slli_si256(el8.val, 8));
                    el4l.val = _mm256_add_epi32(_mm256_cvtepi16_epi32(_v256_extract_low(vsum)), prev.val);
                    el4h.val = _mm256_add_epi32(_mm256_cvtepi16_epi32(_v256_extract_high(vsum)), _mm256_permute2x128_si256(el4l.val, el4l.val, 0x31));
                    prev.val = _mm256_permute2x128_si256(el4h.val, el4h.val, 0x31);
#else
#if CV_SIMD_WIDTH >= 32
                    el8 += v_rotate_left<4>(el8);
#if CV_SIMD_WIDTH == 64
                    el8 += v_rotate_left<8>(el8);
#endif
#endif
                    v_expand(el8, el4l, el4h);
                    el4l = v_add(el4l, prev);
                    el4h = v_add(el4h, el4l);
#if CV_SIMD_WIDTH == 16
                    prev = el4h;
#elif CV_SIMD_WIDTH == 32
                    prev = v_combine_high(el4h, el4h);
#else
                    v_int32 t = v_rotate_right<12>(el4h);
                    t |= v_rotate_left<4>(t);
                    prev = v_combine_low(t, t);
#endif
#endif
                    v_store(sum_row + j                  , v_add(el4l, vx_load(prev_sum_row + j)));
                    v_store(sum_row + j + VTraits<v_int32>::vlanes(), v_add(el4h, vx_load(prev_sum_row + j + VTraits<v_int32>::vlanes())));
                }

                for (int v4 = sum_row[j - 1] - prev_sum_row[j - 1],
                         v3 = sum_row[j - 2] - prev_sum_row[j - 2],
                         v2 = sum_row[j - 3] - prev_sum_row[j - 3],
                         v1 = sum_row[j - 4] - prev_sum_row[j - 4]; j < width; j += 4)
                {
                    sum_row[j]     = (v1 += src_row[j])     + prev_sum_row[j];
                    sum_row[j + 1] = (v2 += src_row[j + 1]) + prev_sum_row[j + 1];
                    sum_row[j + 2] = (v3 += src_row[j + 2]) + prev_sum_row[j + 2];
                    sum_row[j + 3] = (v4 += src_row[j + 3]) + prev_sum_row[j + 3];
                }
            }
        }
        else
        {
            return false;
        }
        vx_cleanup();

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
        if (sqsum || tilted || cn > 4)
            return false;

        width *= cn;

        // the first iteration
        memset(sum, 0, (width + cn) * sizeof(float));

        if (cn == 1)
        {
            // the others
            for (int i = 0; i < height; ++i)
            {
                const uchar * src_row = src + _srcstep * i;
                float * prev_sum_row = (float *)((uchar *)sum + _sumstep * i) + 1;
                float * sum_row = (float *)((uchar *)sum + _sumstep * (i + 1)) + 1;

                sum_row[-1] = 0;

                v_float32 prev = vx_setzero_f32();
                int j = 0;
                for (; j + VTraits<v_uint16>::vlanes() <= width; j += VTraits<v_uint16>::vlanes())
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
                    el8 = v_add(el8, v_rotate_left<1>(el8));
                    el8 = v_add(el8, v_rotate_left<2>(el8));
#if CV_SIMD_WIDTH >= 32
                    el8 += v_rotate_left<4>(el8);
#if CV_SIMD_WIDTH == 64
                    el8 += v_rotate_left<8>(el8);
#endif
#endif
                    v_int32 el4li, el4hi;
                    v_expand(el8, el4li, el4hi);
                    el4l = v_add(v_cvt_f32(el4li), prev);
                    el4h = v_add(v_cvt_f32(el4hi), el4l);
                    prev = v_broadcast_highest(el4h);
#endif
                    v_store(sum_row + j                    , v_add(el4l, vx_load(prev_sum_row + j)));
                    v_store(sum_row + j + VTraits<v_float32>::vlanes(), v_add(el4h, vx_load(prev_sum_row + j + VTraits<v_float32>::vlanes())));
                }

                for (float v = sum_row[j - 1] - prev_sum_row[j - 1]; j < width; ++j)
                    sum_row[j] = (v += src_row[j]) + prev_sum_row[j];
            }
        }
        else if (cn == 2)
        {
            // the others
            v_int16 mask = vx_setall_s16((short)0xff);
            for (int i = 0; i < height; ++i)
            {
                const uchar * src_row = src + _srcstep * i;
                float * prev_sum_row = (float *)((uchar *)sum + _sumstep * i) + cn;
                float * sum_row = (float *)((uchar *)sum + _sumstep * (i + 1)) + cn;

                sum_row[-1] = sum_row[-2] = 0;

                v_float32 prev_1 = vx_setzero_f32(), prev_2 = vx_setzero_f32();
                int j = 0;
                for (; j + VTraits<v_uint16>::vlanes() * cn <= width; j += VTraits<v_uint16>::vlanes() * cn)
                {
                    v_int16 v_src_row = v_reinterpret_as_s16(vx_load(src_row + j));
                    v_int16 el8_1 = v_and(v_src_row, mask);
                    v_int16 el8_2 = v_reinterpret_as_s16(v_shr<8>(v_reinterpret_as_u16(v_src_row)));
                    v_float32 el4l_1, el4h_1, el4l_2, el4h_2;
#if CV_AVX2 && CV_SIMD_WIDTH == 32
                    __m256i vsum_1 = _mm256_add_epi16(el8_1.val, _mm256_slli_si256(el8_1.val, 2));
                    __m256i vsum_2 = _mm256_add_epi16(el8_2.val, _mm256_slli_si256(el8_2.val, 2));
                    vsum_1 = _mm256_add_epi16(vsum_1, _mm256_slli_si256(vsum_1, 4));
                    vsum_2 = _mm256_add_epi16(vsum_2, _mm256_slli_si256(vsum_2, 4));
                    vsum_1 = _mm256_add_epi16(vsum_1, _mm256_slli_si256(vsum_1, 8));
                    vsum_2 = _mm256_add_epi16(vsum_2, _mm256_slli_si256(vsum_2, 8));
                    __m256i shmask = _mm256_set1_epi32(7);
                    el4l_1.val = _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_v256_extract_low(vsum_1))), prev_1.val);
                    el4l_2.val = _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_v256_extract_low(vsum_2))), prev_2.val);
                    el4h_1.val = _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_v256_extract_high(vsum_1))), _mm256_permutevar8x32_ps(el4l_1.val, shmask));
                    el4h_2.val = _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_v256_extract_high(vsum_2))), _mm256_permutevar8x32_ps(el4l_2.val, shmask));
                    prev_1.val = _mm256_permutevar8x32_ps(el4h_1.val, shmask);
                    prev_2.val = _mm256_permutevar8x32_ps(el4h_2.val, shmask);
#else
                    el8_1 = v_add(el8_1, v_rotate_left<1>(el8_1));
                    el8_2 = v_add(el8_2, v_rotate_left<1>(el8_2));
                    el8_1 = v_add(el8_1, v_rotate_left<2>(el8_1));
                    el8_2 = v_add(el8_2, v_rotate_left<2>(el8_2));
#if CV_SIMD_WIDTH >= 32
                    el8_1 += v_rotate_left<4>(el8_1);
                    el8_2 += v_rotate_left<4>(el8_2);
#if CV_SIMD_WIDTH == 64
                    el8_1 += v_rotate_left<8>(el8_1);
                    el8_2 += v_rotate_left<8>(el8_2);
#endif
#endif
                    v_int32 el4li_1, el4hi_1, el4li_2, el4hi_2;
                    v_expand(el8_1, el4li_1, el4hi_1);
                    v_expand(el8_2, el4li_2, el4hi_2);
                    el4l_1 = v_add(v_cvt_f32(el4li_1), prev_1);
                    el4l_2 = v_add(v_cvt_f32(el4li_2), prev_2);
                    el4h_1 = v_add(v_cvt_f32(el4hi_1), el4l_1);
                    el4h_2 = v_add(v_cvt_f32(el4hi_2), el4l_2);
                    prev_1 = v_broadcast_highest(el4h_1);
                    prev_2 = v_broadcast_highest(el4h_2);
#endif
                    v_float32 el4_1, el4_2, el4_3, el4_4;
                    v_zip(el4l_1, el4l_2, el4_1, el4_2);
                    v_zip(el4h_1, el4h_2, el4_3, el4_4);
                    v_store(sum_row + j                        , v_add(el4_1, vx_load(prev_sum_row + j)));
                    v_store(sum_row + j + VTraits<v_float32>::vlanes()    , v_add(el4_2, vx_load(prev_sum_row + j + VTraits<v_float32>::vlanes())));
                    v_store(sum_row + j + VTraits<v_float32>::vlanes() * 2, v_add(el4_3, vx_load(prev_sum_row + j + VTraits<v_float32>::vlanes() * 2)));
                    v_store(sum_row + j + VTraits<v_float32>::vlanes() * 3, v_add(el4_4, vx_load(prev_sum_row + j + VTraits<v_float32>::vlanes() * 3)));
                }

                for (float v2 = sum_row[j - 1] - prev_sum_row[j - 1],
                           v1 = sum_row[j - 2] - prev_sum_row[j - 2]; j < width; j += 2)
                {
                    sum_row[j]     = (v1 += src_row[j])     + prev_sum_row[j];
                    sum_row[j + 1] = (v2 += src_row[j + 1]) + prev_sum_row[j + 1];
                }
            }
        }
        else if (cn == 3)
        {
            // the others
            for (int i = 0; i < height; ++i)
            {
                const uchar * src_row = src + _srcstep * i;
                float * prev_sum_row = (float *)((uchar *)sum + _sumstep * i) + cn;
                float * sum_row = (float *)((uchar *)sum + _sumstep * (i + 1)) + cn;
                float row_cache[VTraits<v_float32>::max_nlanes * 6];

                sum_row[-1] = sum_row[-2] = sum_row[-3] = 0;

                v_float32 prev_1 = vx_setzero_f32(), prev_2 = vx_setzero_f32(),
                          prev_3 = vx_setzero_f32();
                int j = 0;
                const int j_max =
                        ((_srcstep * i + (width - VTraits<v_uint16>::vlanes() * cn + VTraits<v_uint8>::vlanes() * cn)) >= _srcstep * height)
                        ? width - VTraits<v_uint8>::vlanes() * cn    // uint8 in v_load_deinterleave()
                        : width - VTraits<v_uint16>::vlanes() * cn;  // v_expand_low
                for ( ; j <= j_max; j += VTraits<v_uint16>::vlanes() * cn)
                {
                    v_uint8 v_src_row_1, v_src_row_2, v_src_row_3;
                    v_load_deinterleave(src_row + j, v_src_row_1, v_src_row_2, v_src_row_3);
                    v_int16 el8_1 = v_reinterpret_as_s16(v_expand_low(v_src_row_1));
                    v_int16 el8_2 = v_reinterpret_as_s16(v_expand_low(v_src_row_2));
                    v_int16 el8_3 = v_reinterpret_as_s16(v_expand_low(v_src_row_3));
                    v_float32 el4l_1, el4h_1, el4l_2, el4h_2, el4l_3, el4h_3;
#if CV_AVX2 && CV_SIMD_WIDTH == 32
                    __m256i vsum_1 = _mm256_add_epi16(el8_1.val, _mm256_slli_si256(el8_1.val, 2));
                    __m256i vsum_2 = _mm256_add_epi16(el8_2.val, _mm256_slli_si256(el8_2.val, 2));
                    __m256i vsum_3 = _mm256_add_epi16(el8_3.val, _mm256_slli_si256(el8_3.val, 2));
                    vsum_1 = _mm256_add_epi16(vsum_1, _mm256_slli_si256(vsum_1, 4));
                    vsum_2 = _mm256_add_epi16(vsum_2, _mm256_slli_si256(vsum_2, 4));
                    vsum_3 = _mm256_add_epi16(vsum_3, _mm256_slli_si256(vsum_3, 4));
                    vsum_1 = _mm256_add_epi16(vsum_1, _mm256_slli_si256(vsum_1, 8));
                    vsum_2 = _mm256_add_epi16(vsum_2, _mm256_slli_si256(vsum_2, 8));
                    vsum_3 = _mm256_add_epi16(vsum_3, _mm256_slli_si256(vsum_3, 8));
                    __m256i shmask = _mm256_set1_epi32(7);
                    el4l_1.val = _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_v256_extract_low(vsum_1))), prev_1.val);
                    el4l_2.val = _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_v256_extract_low(vsum_2))), prev_2.val);
                    el4l_3.val = _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_v256_extract_low(vsum_3))), prev_3.val);
                    el4h_1.val = _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_v256_extract_high(vsum_1))), _mm256_permutevar8x32_ps(el4l_1.val, shmask));
                    el4h_2.val = _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_v256_extract_high(vsum_2))), _mm256_permutevar8x32_ps(el4l_2.val, shmask));
                    el4h_3.val = _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_v256_extract_high(vsum_3))), _mm256_permutevar8x32_ps(el4l_3.val, shmask));
                    prev_1.val = _mm256_permutevar8x32_ps(el4h_1.val, shmask);
                    prev_2.val = _mm256_permutevar8x32_ps(el4h_2.val, shmask);
                    prev_3.val = _mm256_permutevar8x32_ps(el4h_3.val, shmask);
#else
                    el8_1 = v_add(el8_1, v_rotate_left<1>(el8_1));
                    el8_2 = v_add(el8_2, v_rotate_left<1>(el8_2));
                    el8_3 = v_add(el8_3, v_rotate_left<1>(el8_3));
                    el8_1 = v_add(el8_1, v_rotate_left<2>(el8_1));
                    el8_2 = v_add(el8_2, v_rotate_left<2>(el8_2));
                    el8_3 = v_add(el8_3, v_rotate_left<2>(el8_3));
#if CV_SIMD_WIDTH >= 32
                    el8_1 += v_rotate_left<4>(el8_1);
                    el8_2 += v_rotate_left<4>(el8_2);
                    el8_3 += v_rotate_left<4>(el8_3);
#if CV_SIMD_WIDTH == 64
                    el8_1 += v_rotate_left<8>(el8_1);
                    el8_2 += v_rotate_left<8>(el8_2);
                    el8_3 += v_rotate_left<8>(el8_3);
#endif
#endif
                    v_int32 el4li_1, el4hi_1, el4li_2, el4hi_2, el4li_3, el4hi_3;
                    v_expand(el8_1, el4li_1, el4hi_1);
                    v_expand(el8_2, el4li_2, el4hi_2);
                    v_expand(el8_3, el4li_3, el4hi_3);
                    el4l_1 = v_add(v_cvt_f32(el4li_1), prev_1);
                    el4l_2 = v_add(v_cvt_f32(el4li_2), prev_2);
                    el4l_3 = v_add(v_cvt_f32(el4li_3), prev_3);
                    el4h_1 = v_add(v_cvt_f32(el4hi_1), el4l_1);
                    el4h_2 = v_add(v_cvt_f32(el4hi_2), el4l_2);
                    el4h_3 = v_add(v_cvt_f32(el4hi_3), el4l_3);
                    prev_1 = v_broadcast_highest(el4h_1);
                    prev_2 = v_broadcast_highest(el4h_2);
                    prev_3 = v_broadcast_highest(el4h_3);
#endif
                    v_store_interleave(row_cache                        , el4l_1, el4l_2, el4l_3);
                    v_store_interleave(row_cache + VTraits<v_float32>::vlanes() * 3, el4h_1, el4h_2, el4h_3);
                    el4l_1 = vx_load(row_cache                        );
                    el4l_2 = vx_load(row_cache + VTraits<v_float32>::vlanes()    );
                    el4l_3 = vx_load(row_cache + VTraits<v_float32>::vlanes() * 2);
                    el4h_1 = vx_load(row_cache + VTraits<v_float32>::vlanes() * 3);
                    el4h_2 = vx_load(row_cache + VTraits<v_float32>::vlanes() * 4);
                    el4h_3 = vx_load(row_cache + VTraits<v_float32>::vlanes() * 5);
                    v_store(sum_row + j                        , v_add(el4l_1, vx_load(prev_sum_row + j)));
                    v_store(sum_row + j + VTraits<v_float32>::vlanes()    , v_add(el4l_2, vx_load(prev_sum_row + j + VTraits<v_float32>::vlanes())));
                    v_store(sum_row + j + VTraits<v_float32>::vlanes() * 2, v_add(el4l_3, vx_load(prev_sum_row + j + VTraits<v_float32>::vlanes() * 2)));
                    v_store(sum_row + j + VTraits<v_float32>::vlanes() * 3, v_add(el4h_1, vx_load(prev_sum_row + j + VTraits<v_float32>::vlanes() * 3)));
                    v_store(sum_row + j + VTraits<v_float32>::vlanes() * 4, v_add(el4h_2, vx_load(prev_sum_row + j + VTraits<v_float32>::vlanes() * 4)));
                    v_store(sum_row + j + VTraits<v_float32>::vlanes() * 5, v_add(el4h_3, vx_load(prev_sum_row + j + VTraits<v_float32>::vlanes() * 5)));
                }

                for (float v3 = sum_row[j - 1] - prev_sum_row[j - 1],
                           v2 = sum_row[j - 2] - prev_sum_row[j - 2],
                           v1 = sum_row[j - 3] - prev_sum_row[j - 3]; j < width; j += 3)
                {
                    sum_row[j]     = (v1 += src_row[j])     + prev_sum_row[j];
                    sum_row[j + 1] = (v2 += src_row[j + 1]) + prev_sum_row[j + 1];
                    sum_row[j + 2] = (v3 += src_row[j + 2]) + prev_sum_row[j + 2];
                }
            }
        }
        else if (cn == 4)
        {
            // the others
            for (int i = 0; i < height; ++i)
            {
                const uchar * src_row = src + _srcstep * i;
                float * prev_sum_row = (float *)((uchar *)sum + _sumstep * i) + cn;
                float * sum_row = (float *)((uchar *)sum + _sumstep * (i + 1)) + cn;

                sum_row[-1] = sum_row[-2] = sum_row[-3] = sum_row[-4] = 0;

                v_float32 prev = vx_setzero_f32();
                int j = 0;
                for ( ; j + VTraits<v_uint16>::vlanes() <= width; j += VTraits<v_uint16>::vlanes())
                {
                    v_int16 el8 = v_reinterpret_as_s16(vx_load_expand(src_row + j));
                    v_float32 el4l, el4h;
#if CV_AVX2 && CV_SIMD_WIDTH == 32
                    __m256i vsum = _mm256_add_epi16(el8.val, _mm256_slli_si256(el8.val, 8));
                    el4l.val = _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_v256_extract_low(vsum))), prev.val);
                    el4h.val = _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_v256_extract_high(vsum))), _mm256_permute2f128_ps(el4l.val, el4l.val, 0x31));
                    prev.val = _mm256_permute2f128_ps(el4h.val, el4h.val, 0x31);
#else
#if CV_SIMD_WIDTH >= 32
                    el8 += v_rotate_left<4>(el8);
#if CV_SIMD_WIDTH == 64
                    el8 += v_rotate_left<8>(el8);
#endif
#endif
                    v_int32 el4li, el4hi;
                    v_expand(el8, el4li, el4hi);
                    el4l = v_add(v_cvt_f32(el4li), prev);
                    el4h = v_add(v_cvt_f32(el4hi), el4l);
#if CV_SIMD_WIDTH == 16
                    prev = el4h;
#elif CV_SIMD_WIDTH == 32
                    prev = v_combine_high(el4h, el4h);
#else
                    v_float32 t = v_rotate_right<12>(el4h);
                    t |= v_rotate_left<4>(t);
                    prev = v_combine_low(t, t);
#endif
#endif
                    v_store(sum_row + j                    , v_add(el4l, vx_load(prev_sum_row + j)));
                    v_store(sum_row + j + VTraits<v_float32>::vlanes(), v_add(el4h, vx_load(prev_sum_row + j + VTraits<v_float32>::vlanes())));
                }

                for (float v4 = sum_row[j - 1] - prev_sum_row[j - 1],
                           v3 = sum_row[j - 2] - prev_sum_row[j - 2],
                           v2 = sum_row[j - 3] - prev_sum_row[j - 3],
                           v1 = sum_row[j - 4] - prev_sum_row[j - 4]; j < width; j += 4)
                {
                    sum_row[j]     = (v1 += src_row[j])     + prev_sum_row[j];
                    sum_row[j + 1] = (v2 += src_row[j + 1]) + prev_sum_row[j + 1];
                    sum_row[j + 2] = (v3 += src_row[j + 2]) + prev_sum_row[j + 2];
                    sum_row[j + 3] = (v4 += src_row[j + 3]) + prev_sum_row[j + 3];
                }
            }
        }
        else
        {
            return false;
        }
        vx_cleanup();

        return true;
    }
};

#if CV_SIMD128_64F
template <>
struct Integral_SIMD<uchar, double, double>
{
    Integral_SIMD() {}

    bool operator()(const uchar * src, size_t _srcstep,
        double * sum, size_t _sumstep,
        double * sqsum, size_t _sqsumstep,
        double * tilted, size_t,
        int width, int height, int cn) const
    {
#if CV_AVX512_SKX
        if (!tilted && cn <= 4 && (cn > 1 || sqsum))
        {
            calculate_integral_avx512(src, _srcstep, sum, _sumstep, sqsum, _sqsumstep, width, height, cn);
            return true;
        }
#else
        CV_UNUSED(_sqsumstep);
#endif
        if (sqsum || tilted || cn > 4)
            return false;

        width *= cn;

        // the first iteration
        memset(sum, 0, (width + cn) * sizeof(double));

        if (cn == 1)
        {
            // the others
            for (int i = 0; i < height; ++i)
            {
                const uchar * src_row = src + _srcstep * i;
                double * prev_sum_row = (double *)((uchar *)sum + _sumstep * i) + 1;
                double * sum_row = (double *)((uchar *)sum + _sumstep * (i + 1)) + 1;

                sum_row[-1] = 0;

                v_float64 prev = vx_setzero_f64();
                int j = 0;
                for (; j + VTraits<v_uint16>::vlanes() <= width; j += VTraits<v_uint16>::vlanes())
                {
                    v_int16 el8 = v_reinterpret_as_s16(vx_load_expand(src_row + j));
                    v_float64 el4ll, el4lh, el4hl, el4hh;
#if CV_AVX2 && CV_SIMD_WIDTH == 32
                    __m256i vsum = _mm256_add_epi16(el8.val, _mm256_slli_si256(el8.val, 2));
                    vsum = _mm256_add_epi16(vsum, _mm256_slli_si256(vsum, 4));
                    vsum = _mm256_add_epi16(vsum, _mm256_slli_si256(vsum, 8));
                    __m256i el4l_32 = _mm256_cvtepi16_epi32(_v256_extract_low(vsum));
                    __m256i el4h_32 = _mm256_cvtepi16_epi32(_v256_extract_high(vsum));
                    el4ll.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_low(el4l_32)), prev.val);
                    el4lh.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_high(el4l_32)), prev.val);
                    __m256d el4d = _mm256_permute4x64_pd(el4lh.val, 0xff);
                    el4hl.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_low(el4h_32)), el4d);
                    el4hh.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_high(el4h_32)), el4d);
                    prev.val = _mm256_permute4x64_pd(el4hh.val, 0xff);
#else
                    el8 = v_add(el8, v_rotate_left<1>(el8));
                    el8 = v_add(el8, v_rotate_left<2>(el8));
#if CV_SIMD_WIDTH >= 32
                    el8 += v_rotate_left<4>(el8);
#if CV_SIMD_WIDTH == 64
                    el8 += v_rotate_left<8>(el8);
#endif
#endif
                    v_int32 el4li, el4hi;
                    v_expand(el8, el4li, el4hi);
                    el4ll = v_add(v_cvt_f64(el4li), prev);
                    el4lh = v_add(v_cvt_f64_high(el4li), prev);
                    el4hl = v_add(v_cvt_f64(el4hi), el4ll);
                    el4hh = v_add(v_cvt_f64_high(el4hi), el4lh);
                    prev = vx_setall_f64(v_extract_highest(el4hh));
//                    prev = v_broadcast_highest(el4hh);
#endif
                    v_store(sum_row + j                        , v_add(el4ll, vx_load(prev_sum_row + j)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes()    , v_add(el4lh, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes())));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 2, v_add(el4hl, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 2)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 3, v_add(el4hh, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 3)));
                }

                for (double v = sum_row[j - 1] - prev_sum_row[j - 1]; j < width; ++j)
                    sum_row[j] = (v += src_row[j]) + prev_sum_row[j];
            }
        }
        else if (cn == 2)
        {
            // the others
            v_int16 mask = vx_setall_s16((short)0xff);
            for (int i = 0; i < height; ++i)
            {
                const uchar * src_row = src + _srcstep * i;
                double * prev_sum_row = (double *)((uchar *)sum + _sumstep * i) + cn;
                double * sum_row = (double *)((uchar *)sum + _sumstep * (i + 1)) + cn;

                sum_row[-1] = sum_row[-2] = 0;

                v_float64 prev_1 = vx_setzero_f64(), prev_2 = vx_setzero_f64();
                int j = 0;
                for (; j + VTraits<v_uint16>::vlanes() * cn <= width; j += VTraits<v_uint16>::vlanes() * cn)
                {
                    v_int16 v_src_row = v_reinterpret_as_s16(vx_load(src_row + j));
                    v_int16 el8_1 = v_and(v_src_row, mask);
                    v_int16 el8_2 = v_reinterpret_as_s16(v_shr<8>(v_reinterpret_as_u16(v_src_row)));
                    v_float64 el4ll_1, el4lh_1, el4hl_1, el4hh_1, el4ll_2, el4lh_2, el4hl_2, el4hh_2;
#if CV_AVX2 && CV_SIMD_WIDTH == 32
                    __m256i vsum_1 = _mm256_add_epi16(el8_1.val, _mm256_slli_si256(el8_1.val, 2));
                    __m256i vsum_2 = _mm256_add_epi16(el8_2.val, _mm256_slli_si256(el8_2.val, 2));
                    vsum_1 = _mm256_add_epi16(vsum_1, _mm256_slli_si256(vsum_1, 4));
                    vsum_2 = _mm256_add_epi16(vsum_2, _mm256_slli_si256(vsum_2, 4));
                    vsum_1 = _mm256_add_epi16(vsum_1, _mm256_slli_si256(vsum_1, 8));
                    vsum_2 = _mm256_add_epi16(vsum_2, _mm256_slli_si256(vsum_2, 8));
                    __m256i el4l1_32 = _mm256_cvtepi16_epi32(_v256_extract_low(vsum_1));
                    __m256i el4l2_32 = _mm256_cvtepi16_epi32(_v256_extract_low(vsum_2));
                    __m256i el4h1_32 = _mm256_cvtepi16_epi32(_v256_extract_high(vsum_1));
                    __m256i el4h2_32 = _mm256_cvtepi16_epi32(_v256_extract_high(vsum_2));
                    el4ll_1.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_low(el4l1_32)), prev_1.val);
                    el4ll_2.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_low(el4l2_32)), prev_2.val);
                    el4lh_1.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_high(el4l1_32)), prev_1.val);
                    el4lh_2.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_high(el4l2_32)), prev_2.val);
                    __m256d el4d_1 = _mm256_permute4x64_pd(el4lh_1.val, 0xff);
                    __m256d el4d_2 = _mm256_permute4x64_pd(el4lh_2.val, 0xff);
                    el4hl_1.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_low(el4h1_32)), el4d_1);
                    el4hl_2.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_low(el4h2_32)), el4d_2);
                    el4hh_1.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_high(el4h1_32)), el4d_1);
                    el4hh_2.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_high(el4h2_32)), el4d_2);
                    prev_1.val = _mm256_permute4x64_pd(el4hh_1.val, 0xff);
                    prev_2.val = _mm256_permute4x64_pd(el4hh_2.val, 0xff);
#else
                    el8_1 = v_add(el8_1, v_rotate_left<1>(el8_1));
                    el8_2 = v_add(el8_2, v_rotate_left<1>(el8_2));
                    el8_1 = v_add(el8_1, v_rotate_left<2>(el8_1));
                    el8_2 = v_add(el8_2, v_rotate_left<2>(el8_2));
#if CV_SIMD_WIDTH >= 32
                    el8_1 += v_rotate_left<4>(el8_1);
                    el8_2 += v_rotate_left<4>(el8_2);
#if CV_SIMD_WIDTH == 64
                    el8_1 += v_rotate_left<8>(el8_1);
                    el8_2 += v_rotate_left<8>(el8_2);
#endif
#endif
                    v_int32 el4li_1, el4hi_1, el4li_2, el4hi_2;
                    v_expand(el8_1, el4li_1, el4hi_1);
                    v_expand(el8_2, el4li_2, el4hi_2);
                    el4ll_1 = v_add(v_cvt_f64(el4li_1), prev_1);
                    el4ll_2 = v_add(v_cvt_f64(el4li_2), prev_2);
                    el4lh_1 = v_add(v_cvt_f64_high(el4li_1), prev_1);
                    el4lh_2 = v_add(v_cvt_f64_high(el4li_2), prev_2);
                    el4hl_1 = v_add(v_cvt_f64(el4hi_1), el4ll_1);
                    el4hl_2 = v_add(v_cvt_f64(el4hi_2), el4ll_2);
                    el4hh_1 = v_add(v_cvt_f64_high(el4hi_1), el4lh_1);
                    el4hh_2 = v_add(v_cvt_f64_high(el4hi_2), el4lh_2);
                    prev_1 = vx_setall_f64(v_extract_highest(el4hh_1));
                    prev_2 = vx_setall_f64(v_extract_highest(el4hh_2));
//                    prev_1 = v_broadcast_highest(el4hh_1);
//                    prev_2 = v_broadcast_highest(el4hh_2);
#endif
                    v_float64 el4_1, el4_2, el4_3, el4_4, el4_5, el4_6, el4_7, el4_8;
                    v_zip(el4ll_1, el4ll_2, el4_1, el4_2);
                    v_zip(el4lh_1, el4lh_2, el4_3, el4_4);
                    v_zip(el4hl_1, el4hl_2, el4_5, el4_6);
                    v_zip(el4hh_1, el4hh_2, el4_7, el4_8);
                    v_store(sum_row + j                        , v_add(el4_1, vx_load(prev_sum_row + j)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes()    , v_add(el4_2, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes())));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 2, v_add(el4_3, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 2)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 3, v_add(el4_4, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 3)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 4, v_add(el4_5, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 4)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 5, v_add(el4_6, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 5)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 6, v_add(el4_7, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 6)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 7, v_add(el4_8, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 7)));
                }

                for (double v2 = sum_row[j - 1] - prev_sum_row[j - 1],
                            v1 = sum_row[j - 2] - prev_sum_row[j - 2]; j < width; j += 2)
                {
                    sum_row[j]     = (v1 += src_row[j])     + prev_sum_row[j];
                    sum_row[j + 1] = (v2 += src_row[j + 1]) + prev_sum_row[j + 1];
                }
            }
        }
        else if (cn == 3)
        {
            // the others
            for (int i = 0; i < height; ++i)
            {
                const uchar * src_row = src + _srcstep * i;
                double * prev_sum_row = (double *)((uchar *)sum + _sumstep * i) + cn;
                double * sum_row = (double *)((uchar *)sum + _sumstep * (i + 1)) + cn;
                double row_cache[VTraits<v_float64>::max_nlanes * 12];

                sum_row[-1] = sum_row[-2] = sum_row[-3] = 0;

                v_float64 prev_1 = vx_setzero_f64(), prev_2 = vx_setzero_f64(),
                          prev_3 = vx_setzero_f64();
                int j = 0;
                const int j_max =
                        ((_srcstep * i + (width - VTraits<v_uint16>::vlanes() * cn + VTraits<v_uint8>::vlanes() * cn)) >= _srcstep * height)
                        ? width - VTraits<v_uint8>::vlanes() * cn    // uint8 in v_load_deinterleave()
                        : width - VTraits<v_uint16>::vlanes() * cn;  // v_expand_low
                for ( ; j <= j_max; j += VTraits<v_uint16>::vlanes() * cn)
                {
                    v_uint8 v_src_row_1, v_src_row_2, v_src_row_3;
                    v_load_deinterleave(src_row + j, v_src_row_1, v_src_row_2, v_src_row_3);
                    v_int16 el8_1 = v_reinterpret_as_s16(v_expand_low(v_src_row_1));
                    v_int16 el8_2 = v_reinterpret_as_s16(v_expand_low(v_src_row_2));
                    v_int16 el8_3 = v_reinterpret_as_s16(v_expand_low(v_src_row_3));
                    v_float64 el4ll_1, el4lh_1, el4hl_1, el4hh_1, el4ll_2, el4lh_2, el4hl_2, el4hh_2, el4ll_3, el4lh_3, el4hl_3, el4hh_3;
#if CV_AVX2 && CV_SIMD_WIDTH == 32
                    __m256i vsum_1 = _mm256_add_epi16(el8_1.val, _mm256_slli_si256(el8_1.val, 2));
                    __m256i vsum_2 = _mm256_add_epi16(el8_2.val, _mm256_slli_si256(el8_2.val, 2));
                    __m256i vsum_3 = _mm256_add_epi16(el8_3.val, _mm256_slli_si256(el8_3.val, 2));
                    vsum_1 = _mm256_add_epi16(vsum_1, _mm256_slli_si256(vsum_1, 4));
                    vsum_2 = _mm256_add_epi16(vsum_2, _mm256_slli_si256(vsum_2, 4));
                    vsum_3 = _mm256_add_epi16(vsum_3, _mm256_slli_si256(vsum_3, 4));
                    vsum_1 = _mm256_add_epi16(vsum_1, _mm256_slli_si256(vsum_1, 8));
                    vsum_2 = _mm256_add_epi16(vsum_2, _mm256_slli_si256(vsum_2, 8));
                    vsum_3 = _mm256_add_epi16(vsum_3, _mm256_slli_si256(vsum_3, 8));
                    __m256i el4l1_32 = _mm256_cvtepi16_epi32(_v256_extract_low(vsum_1));
                    __m256i el4l2_32 = _mm256_cvtepi16_epi32(_v256_extract_low(vsum_2));
                    __m256i el4l3_32 = _mm256_cvtepi16_epi32(_v256_extract_low(vsum_3));
                    __m256i el4h1_32 = _mm256_cvtepi16_epi32(_v256_extract_high(vsum_1));
                    __m256i el4h2_32 = _mm256_cvtepi16_epi32(_v256_extract_high(vsum_2));
                    __m256i el4h3_32 = _mm256_cvtepi16_epi32(_v256_extract_high(vsum_3));
                    el4ll_1.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_low(el4l1_32)), prev_1.val);
                    el4ll_2.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_low(el4l2_32)), prev_2.val);
                    el4ll_3.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_low(el4l3_32)), prev_3.val);
                    el4lh_1.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_high(el4l1_32)), prev_1.val);
                    el4lh_2.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_high(el4l2_32)), prev_2.val);
                    el4lh_3.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_high(el4l3_32)), prev_3.val);
                    __m256d el4d_1 = _mm256_permute4x64_pd(el4lh_1.val, 0xff);
                    __m256d el4d_2 = _mm256_permute4x64_pd(el4lh_2.val, 0xff);
                    __m256d el4d_3 = _mm256_permute4x64_pd(el4lh_3.val, 0xff);
                    el4hl_1.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_low(el4h1_32)), el4d_1);
                    el4hl_2.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_low(el4h2_32)), el4d_2);
                    el4hl_3.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_low(el4h3_32)), el4d_3);
                    el4hh_1.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_high(el4h1_32)), el4d_1);
                    el4hh_2.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_high(el4h2_32)), el4d_2);
                    el4hh_3.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_high(el4h3_32)), el4d_3);
                    prev_1.val = _mm256_permute4x64_pd(el4hh_1.val, 0xff);
                    prev_2.val = _mm256_permute4x64_pd(el4hh_2.val, 0xff);
                    prev_3.val = _mm256_permute4x64_pd(el4hh_3.val, 0xff);
#else
                    el8_1 = v_add(el8_1, v_rotate_left<1>(el8_1));
                    el8_2 = v_add(el8_2, v_rotate_left<1>(el8_2));
                    el8_3 = v_add(el8_3, v_rotate_left<1>(el8_3));
                    el8_1 = v_add(el8_1, v_rotate_left<2>(el8_1));
                    el8_2 = v_add(el8_2, v_rotate_left<2>(el8_2));
                    el8_3 = v_add(el8_3, v_rotate_left<2>(el8_3));
#if CV_SIMD_WIDTH >= 32
                    el8_1 += v_rotate_left<4>(el8_1);
                    el8_2 += v_rotate_left<4>(el8_2);
                    el8_3 += v_rotate_left<4>(el8_3);
#if CV_SIMD_WIDTH == 64
                    el8_1 += v_rotate_left<8>(el8_1);
                    el8_2 += v_rotate_left<8>(el8_2);
                    el8_3 += v_rotate_left<8>(el8_3);
#endif
#endif
                    v_int32 el4li_1, el4hi_1, el4li_2, el4hi_2, el4li_3, el4hi_3;
                    v_expand(el8_1, el4li_1, el4hi_1);
                    v_expand(el8_2, el4li_2, el4hi_2);
                    v_expand(el8_3, el4li_3, el4hi_3);
                    el4ll_1 = v_add(v_cvt_f64(el4li_1), prev_1);
                    el4ll_2 = v_add(v_cvt_f64(el4li_2), prev_2);
                    el4ll_3 = v_add(v_cvt_f64(el4li_3), prev_3);
                    el4lh_1 = v_add(v_cvt_f64_high(el4li_1), prev_1);
                    el4lh_2 = v_add(v_cvt_f64_high(el4li_2), prev_2);
                    el4lh_3 = v_add(v_cvt_f64_high(el4li_3), prev_3);
                    el4hl_1 = v_add(v_cvt_f64(el4hi_1), el4ll_1);
                    el4hl_2 = v_add(v_cvt_f64(el4hi_2), el4ll_2);
                    el4hl_3 = v_add(v_cvt_f64(el4hi_3), el4ll_3);
                    el4hh_1 = v_add(v_cvt_f64_high(el4hi_1), el4lh_1);
                    el4hh_2 = v_add(v_cvt_f64_high(el4hi_2), el4lh_2);
                    el4hh_3 = v_add(v_cvt_f64_high(el4hi_3), el4lh_3);
                    prev_1 = vx_setall_f64(v_extract_highest(el4hh_1));
                    prev_2 = vx_setall_f64(v_extract_highest(el4hh_2));
                    prev_3 = vx_setall_f64(v_extract_highest(el4hh_3));
//                    prev_1 = v_broadcast_highest(el4hh_1);
//                    prev_2 = v_broadcast_highest(el4hh_2);
//                    prev_3 = v_broadcast_highest(el4hh_3);
#endif
                    v_store_interleave(row_cache                        , el4ll_1, el4ll_2, el4ll_3);
                    v_store_interleave(row_cache + VTraits<v_float64>::vlanes() * 3, el4lh_1, el4lh_2, el4lh_3);
                    v_store_interleave(row_cache + VTraits<v_float64>::vlanes() * 6, el4hl_1, el4hl_2, el4hl_3);
                    v_store_interleave(row_cache + VTraits<v_float64>::vlanes() * 9, el4hh_1, el4hh_2, el4hh_3);
                    el4ll_1 = vx_load(row_cache                         );
                    el4ll_2 = vx_load(row_cache + VTraits<v_float64>::vlanes()     );
                    el4ll_3 = vx_load(row_cache + VTraits<v_float64>::vlanes() * 2 );
                    el4lh_1 = vx_load(row_cache + VTraits<v_float64>::vlanes() * 3 );
                    el4lh_2 = vx_load(row_cache + VTraits<v_float64>::vlanes() * 4 );
                    el4lh_3 = vx_load(row_cache + VTraits<v_float64>::vlanes() * 5 );
                    el4hl_1 = vx_load(row_cache + VTraits<v_float64>::vlanes() * 6 );
                    el4hl_2 = vx_load(row_cache + VTraits<v_float64>::vlanes() * 7 );
                    el4hl_3 = vx_load(row_cache + VTraits<v_float64>::vlanes() * 8 );
                    el4hh_1 = vx_load(row_cache + VTraits<v_float64>::vlanes() * 9 );
                    el4hh_2 = vx_load(row_cache + VTraits<v_float64>::vlanes() * 10);
                    el4hh_3 = vx_load(row_cache + VTraits<v_float64>::vlanes() * 11);
                    v_store(sum_row + j                         , v_add(el4ll_1, vx_load(prev_sum_row + j)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes()     , v_add(el4ll_2, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes())));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 2 , v_add(el4ll_3, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 2)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 3 , v_add(el4lh_1, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 3)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 4 , v_add(el4lh_2, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 4)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 5 , v_add(el4lh_3, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 5)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 6 , v_add(el4hl_1, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 6)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 7 , v_add(el4hl_2, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 7)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 8 , v_add(el4hl_3, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 8)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 9 , v_add(el4hh_1, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 9)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 10, v_add(el4hh_2, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 10)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 11, v_add(el4hh_3, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 11)));
                }

                for (double v3 = sum_row[j - 1] - prev_sum_row[j - 1],
                            v2 = sum_row[j - 2] - prev_sum_row[j - 2],
                            v1 = sum_row[j - 3] - prev_sum_row[j - 3]; j < width; j += 3)
                {
                    sum_row[j]     = (v1 += src_row[j])     + prev_sum_row[j];
                    sum_row[j + 1] = (v2 += src_row[j + 1]) + prev_sum_row[j + 1];
                    sum_row[j + 2] = (v3 += src_row[j + 2]) + prev_sum_row[j + 2];
                }
            }
        }
        else if (cn == 4)
        {
            // the others
            for (int i = 0; i < height; ++i)
            {
                const uchar * src_row = src + _srcstep * i;
                double * prev_sum_row = (double *)((uchar *)sum + _sumstep * i) + cn;
                double * sum_row = (double *)((uchar *)sum + _sumstep * (i + 1)) + cn;

                sum_row[-1] = sum_row[-2] = sum_row[-3] = sum_row[-4] = 0;

                v_float64 prev_1 = vx_setzero_f64(), prev_2 = vx_setzero_f64();
                int j = 0;
                for ( ; j + VTraits<v_uint16>::vlanes() <= width; j += VTraits<v_uint16>::vlanes())
                {
                    v_int16 el8 = v_reinterpret_as_s16(vx_load_expand(src_row + j));
                    v_float64 el4ll, el4lh, el4hl, el4hh;
#if CV_AVX2 && CV_SIMD_WIDTH == 32
                    __m256i vsum = _mm256_add_epi16(el8.val, _mm256_slli_si256(el8.val, 8));
                    __m256i el4l_32 = _mm256_cvtepi16_epi32(_v256_extract_low(vsum));
                    __m256i el4h_32 = _mm256_cvtepi16_epi32(_v256_extract_high(vsum));
                    el4ll.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_low(el4l_32)), prev_1.val);
                    el4lh.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_high(el4l_32)), prev_2.val);
                    el4hl.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_low(el4h_32)), el4lh.val);
                    el4hh.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_high(el4h_32)), el4lh.val);
                    prev_1.val = prev_2.val = el4hh.val;
#else
#if CV_SIMD_WIDTH >= 32
                    el8 += v_rotate_left<4>(el8);
#if CV_SIMD_WIDTH == 64
                    el8 += v_rotate_left<8>(el8);
#endif
#endif
                    v_int32 el4li, el4hi;
                    v_expand(el8, el4li, el4hi);
                    el4ll = v_add(v_cvt_f64(el4li), prev_1);
                    el4lh = v_add(v_cvt_f64_high(el4li), prev_2);
                    el4hl = v_add(v_cvt_f64(el4hi), el4ll);
                    el4hh = v_add(v_cvt_f64_high(el4hi), el4lh);
#if CV_SIMD_WIDTH == 16
                    prev_1 = el4hl;
                    prev_2 = el4hh;
#elif CV_SIMD_WIDTH == 32
                    prev_1 = prev_2 = el4hh;
#else
                    prev_1 = prev_2 = v_combine_high(el4hh, el4hh);
#endif
#endif
                    v_store(sum_row + j                        , v_add(el4ll, vx_load(prev_sum_row + j)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes()    , v_add(el4lh, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes())));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 2, v_add(el4hl, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 2)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 3, v_add(el4hh, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 3)));
                }

                for (double v4 = sum_row[j - 1] - prev_sum_row[j - 1],
                            v3 = sum_row[j - 2] - prev_sum_row[j - 2],
                            v2 = sum_row[j - 3] - prev_sum_row[j - 3],
                            v1 = sum_row[j - 4] - prev_sum_row[j - 4]; j < width; j += 4)
                {
                    sum_row[j]     = (v1 += src_row[j])     + prev_sum_row[j];
                    sum_row[j + 1] = (v2 += src_row[j + 1]) + prev_sum_row[j + 1];
                    sum_row[j + 2] = (v3 += src_row[j + 2]) + prev_sum_row[j + 2];
                    sum_row[j + 3] = (v4 += src_row[j + 3]) + prev_sum_row[j + 3];
                }
            }
        }
        else
        {
            return false;
        }
        vx_cleanup();

        return true;
    }
};
#endif

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
