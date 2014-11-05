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
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#include "../precomp.hpp"
#include "imgwarp_avx2.hpp"

#if CV_AVX2
const int INTER_RESIZE_COEF_BITS=11;
const int INTER_RESIZE_COEF_SCALE=1 << INTER_RESIZE_COEF_BITS;

int VResizeLinearVec_32s8u_avx2(const uchar** _src, uchar* dst, const uchar* _beta, int width )
{
    const int** src = (const int**)_src;
    const short* beta = (const short*)_beta;
    const int *S0 = src[0], *S1 = src[1];
    int x = 0;
    __m256i b0 = _mm256_set1_epi16(beta[0]), b1 = _mm256_set1_epi16(beta[1]);
    __m256i delta = _mm256_set1_epi16(2);
    const int index[8] = { 0, 4, 1, 5, 2, 6, 3, 7 };
    __m256i shuffle = _mm256_load_si256((const __m256i*)index);

    if( (((size_t)S0|(size_t)S1)&31) == 0 )
        for( ; x <= width - 32; x += 32 )
        {
            __m256i x0, x1, x2, y0, y1, y2;
            x0 = _mm256_load_si256((const __m256i*)(S0 + x));
            x1 = _mm256_load_si256((const __m256i*)(S0 + x + 8));
            y0 = _mm256_load_si256((const __m256i*)(S1 + x));
            y1 = _mm256_load_si256((const __m256i*)(S1 + x + 8));
            x0 = _mm256_packs_epi32(_mm256_srai_epi32(x0, 4), _mm256_srai_epi32(x1, 4));
            y0 = _mm256_packs_epi32(_mm256_srai_epi32(y0, 4), _mm256_srai_epi32(y1, 4));

            x1 = _mm256_load_si256((const __m256i*)(S0 + x + 16));
            x2 = _mm256_load_si256((const __m256i*)(S0 + x + 24));
            y1 = _mm256_load_si256((const __m256i*)(S1 + x + 16));
            y2 = _mm256_load_si256((const __m256i*)(S1 + x + 24));
            x1 = _mm256_packs_epi32(_mm256_srai_epi32(x1, 4), _mm256_srai_epi32(x2, 4));
            y1 = _mm256_packs_epi32(_mm256_srai_epi32(y1, 4), _mm256_srai_epi32(y2, 4));

            x0 = _mm256_adds_epi16(_mm256_mulhi_epi16(x0, b0), _mm256_mulhi_epi16(y0, b1));
            x1 = _mm256_adds_epi16(_mm256_mulhi_epi16(x1, b0), _mm256_mulhi_epi16(y1, b1));

            x0 = _mm256_srai_epi16(_mm256_adds_epi16(x0, delta), 2);
            x1 = _mm256_srai_epi16(_mm256_adds_epi16(x1, delta), 2);
            x0 = _mm256_packus_epi16(x0, x1);
            x0 = _mm256_permutevar8x32_epi32(x0, shuffle);
            _mm256_storeu_si256( (__m256i*)(dst + x), x0);
        }
    else
        for( ; x <= width - 32; x += 32 )
        {
            __m256i x0, x1, x2, y0, y1, y2;
            x0 = _mm256_loadu_si256((const __m256i*)(S0 + x));
            x1 = _mm256_loadu_si256((const __m256i*)(S0 + x + 8));
            y0 = _mm256_loadu_si256((const __m256i*)(S1 + x));
            y1 = _mm256_loadu_si256((const __m256i*)(S1 + x + 8));
            x0 = _mm256_packs_epi32(_mm256_srai_epi32(x0, 4), _mm256_srai_epi32(x1, 4));
            y0 = _mm256_packs_epi32(_mm256_srai_epi32(y0, 4), _mm256_srai_epi32(y1, 4));

            x1 = _mm256_loadu_si256((const __m256i*)(S0 + x + 16));
            x2 = _mm256_loadu_si256((const __m256i*)(S0 + x + 24));
            y1 = _mm256_loadu_si256((const __m256i*)(S1 + x + 16));
            y2 = _mm256_loadu_si256((const __m256i*)(S1 + x + 24));
            x1 = _mm256_packs_epi32(_mm256_srai_epi32(x1, 4), _mm256_srai_epi32(x2, 4));
            y1 = _mm256_packs_epi32(_mm256_srai_epi32(y1, 4), _mm256_srai_epi32(y2, 4));

            x0 = _mm256_adds_epi16(_mm256_mulhi_epi16(x0, b0), _mm256_mulhi_epi16(y0, b1));
            x1 = _mm256_adds_epi16(_mm256_mulhi_epi16(x1, b0), _mm256_mulhi_epi16(y1, b1));

            x0 = _mm256_srai_epi16(_mm256_adds_epi16(x0, delta), 2);
            x1 = _mm256_srai_epi16(_mm256_adds_epi16(x1, delta), 2);
            x0 = _mm256_packus_epi16(x0, x1);
            x0 = _mm256_permutevar8x32_epi32(x0, shuffle);
            _mm256_storeu_si256( (__m256i*)(dst + x), x0);
        }

    for( ; x < width - 8; x += 8 )
    {
        __m256i x0, y0;
        x0 = _mm256_srai_epi32(_mm256_loadu_si256((const __m256i*)(S0 + x)), 4);
        y0 = _mm256_srai_epi32(_mm256_loadu_si256((const __m256i*)(S1 + x)), 4);
        x0 = _mm256_packs_epi32(x0, x0);
        y0 = _mm256_packs_epi32(y0, y0);
        x0 = _mm256_adds_epi16(_mm256_mulhi_epi16(x0, b0), _mm256_mulhi_epi16(y0, b1));
        x0 = _mm256_srai_epi16(_mm256_adds_epi16(x0, delta), 2);
        x0 = _mm256_packus_epi16(x0, x0);
        *(int*)(dst + x) = _mm_cvtsi128_si32(_mm256_extracti128_si256(x0, 0));
        *(int*)(dst + x + 4) = _mm_cvtsi128_si32(_mm256_extracti128_si256(x0, 1));
    }

    return x;
}

template<int shiftval>
int VResizeLinearVec_32f16_avx2(const uchar** _src, uchar* _dst, const uchar* _beta, int width )
{
    const float** src = (const float**)_src;
    const float* beta = (const float*)_beta;
    const float *S0 = src[0], *S1 = src[1];
    ushort* dst = (ushort*)_dst;
    int x = 0;

    __m256 b0 = _mm256_set1_ps(beta[0]), b1 = _mm256_set1_ps(beta[1]);
    __m256i preshift = _mm256_set1_epi32(shiftval);
    __m256i postshift = _mm256_set1_epi16((short)shiftval);

    if( (((size_t)S0|(size_t)S1)&31) == 0 )
        for( ; x <= width - 32; x += 32 )
        {
            __m256 x0, x1, y0, y1;
            __m256i t0, t1, t2;
            x0 = _mm256_load_ps(S0 + x);
            x1 = _mm256_load_ps(S0 + x + 8);
            y0 = _mm256_load_ps(S1 + x);
            y1 = _mm256_load_ps(S1 + x + 8);

            x0 = _mm256_add_ps(_mm256_mul_ps(x0, b0), _mm256_mul_ps(y0, b1));
            x1 = _mm256_add_ps(_mm256_mul_ps(x1, b0), _mm256_mul_ps(y1, b1));
            t0 = _mm256_add_epi32(_mm256_cvtps_epi32(x0), preshift);
            t2 = _mm256_add_epi32(_mm256_cvtps_epi32(x1), preshift);
            t0 = _mm256_add_epi16(_mm256_packs_epi32(t0, t2), postshift);

            x0 = _mm256_load_ps(S0 + x + 16);
            x1 = _mm256_load_ps(S0 + x + 24);
            y0 = _mm256_load_ps(S1 + x + 16);
            y1 = _mm256_load_ps(S1 + x + 24);

            x0 = _mm256_add_ps(_mm256_mul_ps(x0, b0), _mm256_mul_ps(y0, b1));
            x1 = _mm256_add_ps(_mm256_mul_ps(x1, b0), _mm256_mul_ps(y1, b1));
            t1 = _mm256_add_epi32(_mm256_cvtps_epi32(x0), preshift);
            t2 = _mm256_add_epi32(_mm256_cvtps_epi32(x1), preshift);
            t1 = _mm256_add_epi16(_mm256_packs_epi32(t1, t2), postshift);

            _mm256_storeu_si256( (__m256i*)(dst + x), t0);
            _mm256_storeu_si256( (__m256i*)(dst + x + 16), t1);
        }
    else
        for( ; x <= width - 32; x += 32 )
        {
            __m256 x0, x1, y0, y1;
            __m256i t0, t1, t2;
            x0 = _mm256_loadu_ps(S0 + x);
            x1 = _mm256_loadu_ps(S0 + x + 8);
            y0 = _mm256_loadu_ps(S1 + x);
            y1 = _mm256_loadu_ps(S1 + x + 8);

            x0 = _mm256_add_ps(_mm256_mul_ps(x0, b0), _mm256_mul_ps(y0, b1));
            x1 = _mm256_add_ps(_mm256_mul_ps(x1, b0), _mm256_mul_ps(y1, b1));
            t0 = _mm256_add_epi32(_mm256_cvtps_epi32(x0), preshift);
            t2 = _mm256_add_epi32(_mm256_cvtps_epi32(x1), preshift);
            t0 = _mm256_add_epi16(_mm256_packs_epi32(t0, t2), postshift);

            x0 = _mm256_loadu_ps(S0 + x + 16);
            x1 = _mm256_loadu_ps(S0 + x + 24);
            y0 = _mm256_loadu_ps(S1 + x + 16);
            y1 = _mm256_loadu_ps(S1 + x + 24);

            x0 = _mm256_add_ps(_mm256_mul_ps(x0, b0), _mm256_mul_ps(y0, b1));
            x1 = _mm256_add_ps(_mm256_mul_ps(x1, b0), _mm256_mul_ps(y1, b1));
            t1 = _mm256_add_epi32(_mm256_cvtps_epi32(x0), preshift);
            t2 = _mm256_add_epi32(_mm256_cvtps_epi32(x1), preshift);
            t1 = _mm256_add_epi16(_mm256_packs_epi32(t1, t2), postshift);

            _mm256_storeu_si256( (__m256i*)(dst + x), t0);
            _mm256_storeu_si256( (__m256i*)(dst + x + 16), t1);
        }

    for( ; x < width - 8; x += 8 )
    {
        __m256 x0, y0;
        __m256i t0;
        x0 = _mm256_loadu_ps(S0 + x);
        y0 = _mm256_loadu_ps(S1 + x);

        x0 = _mm256_add_ps(_mm256_mul_ps(x0, b0), _mm256_mul_ps(y0, b1));
        t0 = _mm256_add_epi32(_mm256_cvtps_epi32(x0), preshift);
        t0 = _mm256_add_epi16(_mm256_packs_epi32(t0, t0), postshift);
        _mm_storel_epi64( (__m128i*)(dst + x), _mm256_extracti128_si256(t0, 0));
        _mm_storel_epi64( (__m128i*)(dst + x + 4), _mm256_extracti128_si256(t0, 1));
    }

    return x;
}

int VResizeCubicVec_32s8u_avx2(const uchar** _src, uchar* dst, const uchar* _beta, int width )
{
    const int** src = (const int**)_src;
    const short* beta = (const short*)_beta;
    const int *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3];
    int x = 0;
    float scale = 1.f/(INTER_RESIZE_COEF_SCALE*INTER_RESIZE_COEF_SCALE);
    __m256 b0 = _mm256_set1_ps(beta[0]*scale), b1 = _mm256_set1_ps(beta[1]*scale),
        b2 = _mm256_set1_ps(beta[2]*scale), b3 = _mm256_set1_ps(beta[3]*scale);
    const int shuffle = 0xd8;   // 11 | 01 | 10 | 00

    if( (((size_t)S0|(size_t)S1|(size_t)S2|(size_t)S3)&31) == 0 )
        for( ; x <= width - 16; x += 16 )
        {
            __m256i x0, x1, y0, y1;
            __m256 s0, s1, f0, f1;
            x0 = _mm256_load_si256((const __m256i*)(S0 + x));
            x1 = _mm256_load_si256((const __m256i*)(S0 + x + 8));
            y0 = _mm256_load_si256((const __m256i*)(S1 + x));
            y1 = _mm256_load_si256((const __m256i*)(S1 + x + 8));

            s0 = _mm256_mul_ps(_mm256_cvtepi32_ps(x0), b0);
            s1 = _mm256_mul_ps(_mm256_cvtepi32_ps(x1), b0);
            f0 = _mm256_mul_ps(_mm256_cvtepi32_ps(y0), b1);
            f1 = _mm256_mul_ps(_mm256_cvtepi32_ps(y1), b1);
            s0 = _mm256_add_ps(s0, f0);
            s1 = _mm256_add_ps(s1, f1);

            x0 = _mm256_load_si256((const __m256i*)(S2 + x));
            x1 = _mm256_load_si256((const __m256i*)(S2 + x + 8));
            y0 = _mm256_load_si256((const __m256i*)(S3 + x));
            y1 = _mm256_load_si256((const __m256i*)(S3 + x + 8));

            f0 = _mm256_mul_ps(_mm256_cvtepi32_ps(x0), b2);
            f1 = _mm256_mul_ps(_mm256_cvtepi32_ps(x1), b2);
            s0 = _mm256_add_ps(s0, f0);
            s1 = _mm256_add_ps(s1, f1);
            f0 = _mm256_mul_ps(_mm256_cvtepi32_ps(y0), b3);
            f1 = _mm256_mul_ps(_mm256_cvtepi32_ps(y1), b3);
            s0 = _mm256_add_ps(s0, f0);
            s1 = _mm256_add_ps(s1, f1);

            x0 = _mm256_cvtps_epi32(s0);
            x1 = _mm256_cvtps_epi32(s1);

            x0 = _mm256_packs_epi32(x0, x1);
            x0 = _mm256_permute4x64_epi64(x0, shuffle);
            x0 = _mm256_packus_epi16(x0, x0);
            _mm_storel_epi64( (__m128i*)(dst + x), _mm256_extracti128_si256(x0, 0));
            _mm_storel_epi64( (__m128i*)(dst + x + 8), _mm256_extracti128_si256(x0, 1));
        }
    else
        for( ; x <= width - 16; x += 16 )
        {
            __m256i x0, x1, y0, y1;
            __m256 s0, s1, f0, f1;
            x0 = _mm256_loadu_si256((const __m256i*)(S0 + x));
            x1 = _mm256_loadu_si256((const __m256i*)(S0 + x + 8));
            y0 = _mm256_loadu_si256((const __m256i*)(S1 + x));
            y1 = _mm256_loadu_si256((const __m256i*)(S1 + x + 8));

            s0 = _mm256_mul_ps(_mm256_cvtepi32_ps(x0), b0);
            s1 = _mm256_mul_ps(_mm256_cvtepi32_ps(x1), b0);
            f0 = _mm256_mul_ps(_mm256_cvtepi32_ps(y0), b1);
            f1 = _mm256_mul_ps(_mm256_cvtepi32_ps(y1), b1);
            s0 = _mm256_add_ps(s0, f0);
            s1 = _mm256_add_ps(s1, f1);

            x0 = _mm256_loadu_si256((const __m256i*)(S2 + x));
            x1 = _mm256_loadu_si256((const __m256i*)(S2 + x + 8));
            y0 = _mm256_loadu_si256((const __m256i*)(S3 + x));
            y1 = _mm256_loadu_si256((const __m256i*)(S3 + x + 8));

            f0 = _mm256_mul_ps(_mm256_cvtepi32_ps(x0), b2);
            f1 = _mm256_mul_ps(_mm256_cvtepi32_ps(x1), b2);
            s0 = _mm256_add_ps(s0, f0);
            s1 = _mm256_add_ps(s1, f1);
            f0 = _mm256_mul_ps(_mm256_cvtepi32_ps(y0), b3);
            f1 = _mm256_mul_ps(_mm256_cvtepi32_ps(y1), b3);
            s0 = _mm256_add_ps(s0, f0);
            s1 = _mm256_add_ps(s1, f1);

            x0 = _mm256_cvtps_epi32(s0);
            x1 = _mm256_cvtps_epi32(s1);

            x0 = _mm256_packs_epi32(x0, x1);
            x0 = _mm256_permute4x64_epi64(x0, shuffle);
            x0 = _mm256_packus_epi16(x0, x0);
            _mm_storel_epi64( (__m128i*)(dst + x), _mm256_extracti128_si256(x0, 0));
            _mm_storel_epi64( (__m128i*)(dst + x + 8), _mm256_extracti128_si256(x0, 1));
        }

    return x;
}

template<int shiftval>
int VResizeCubicVec_32f16_avx2(const uchar** _src, uchar* _dst, const uchar* _beta, int width )
{
    const float** src = (const float**)_src;
    const float* beta = (const float*)_beta;
    const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3];
    ushort* dst = (ushort*)_dst;
    int x = 0;
    __m256 b0 = _mm256_set1_ps(beta[0]), b1 = _mm256_set1_ps(beta[1]),
        b2 = _mm256_set1_ps(beta[2]), b3 = _mm256_set1_ps(beta[3]);
    __m256i preshift = _mm256_set1_epi32(shiftval);
    __m256i postshift = _mm256_set1_epi16((short)shiftval);
    const int shuffle = 0xd8;   // 11 | 01 | 10 | 00

    if( (((size_t)S0|(size_t)S1|(size_t)S2|(size_t)S3)&31) == 0 )
        for( ; x <= width - 16; x += 16 )
        {
            __m256 x0, x1, y0, y1, s0, s1;
            __m256i t0, t1;
            x0 = _mm256_load_ps(S0 + x);
            x1 = _mm256_load_ps(S0 + x + 8);
            y0 = _mm256_load_ps(S1 + x);
            y1 = _mm256_load_ps(S1 + x + 8);

            s0 = _mm256_mul_ps(x0, b0);
            s1 = _mm256_mul_ps(x1, b0);
            y0 = _mm256_mul_ps(y0, b1);
            y1 = _mm256_mul_ps(y1, b1);
            s0 = _mm256_add_ps(s0, y0);
            s1 = _mm256_add_ps(s1, y1);

            x0 = _mm256_load_ps(S2 + x);
            x1 = _mm256_load_ps(S2 + x + 8);
            y0 = _mm256_load_ps(S3 + x);
            y1 = _mm256_load_ps(S3 + x + 8);

            x0 = _mm256_mul_ps(x0, b2);
            x1 = _mm256_mul_ps(x1, b2);
            y0 = _mm256_mul_ps(y0, b3);
            y1 = _mm256_mul_ps(y1, b3);
            s0 = _mm256_add_ps(s0, x0);
            s1 = _mm256_add_ps(s1, x1);
            s0 = _mm256_add_ps(s0, y0);
            s1 = _mm256_add_ps(s1, y1);

            t0 = _mm256_add_epi32(_mm256_cvtps_epi32(s0), preshift);
            t1 = _mm256_add_epi32(_mm256_cvtps_epi32(s1), preshift);

            t0 = _mm256_add_epi16(_mm256_packs_epi32(t0, t1), postshift);
            t0 = _mm256_permute4x64_epi64(t0, shuffle);
            _mm256_storeu_si256( (__m256i*)(dst + x), t0);
        }
    else
        for( ; x <= width - 16; x += 16 )
        {
            __m256 x0, x1, y0, y1, s0, s1;
            __m256i t0, t1;
            x0 = _mm256_loadu_ps(S0 + x);
            x1 = _mm256_loadu_ps(S0 + x + 8);
            y0 = _mm256_loadu_ps(S1 + x);
            y1 = _mm256_loadu_ps(S1 + x + 8);

            s0 = _mm256_mul_ps(x0, b0);
            s1 = _mm256_mul_ps(x1, b0);
            y0 = _mm256_mul_ps(y0, b1);
            y1 = _mm256_mul_ps(y1, b1);
            s0 = _mm256_add_ps(s0, y0);
            s1 = _mm256_add_ps(s1, y1);

            x0 = _mm256_loadu_ps(S2 + x);
            x1 = _mm256_loadu_ps(S2 + x + 8);
            y0 = _mm256_loadu_ps(S3 + x);
            y1 = _mm256_loadu_ps(S3 + x + 8);

            x0 = _mm256_mul_ps(x0, b2);
            x1 = _mm256_mul_ps(x1, b2);
            y0 = _mm256_mul_ps(y0, b3);
            y1 = _mm256_mul_ps(y1, b3);
            s0 = _mm256_add_ps(s0, x0);
            s1 = _mm256_add_ps(s1, x1);
            s0 = _mm256_add_ps(s0, y0);
            s1 = _mm256_add_ps(s1, y1);

            t0 = _mm256_add_epi32(_mm256_cvtps_epi32(s0), preshift);
            t1 = _mm256_add_epi32(_mm256_cvtps_epi32(s1), preshift);

            t0 = _mm256_add_epi16(_mm256_packs_epi32(t0, t1), postshift);
            t0 = _mm256_permute4x64_epi64(t0, shuffle);
            _mm256_storeu_si256( (__m256i*)(dst + x), t0);
        }

    return x;
}
#else
int VResizeLinearVec_32s8u_avx2(const uchar**, uchar*, const uchar*, int ) { return 0; }

template<int shiftval>
int VResizeLinearVec_32f16_avx2(const uchar**, uchar*, const uchar*, int ) { return 0; }

int VResizeCubicVec_32s8u_avx2(const uchar**, uchar*, const uchar*, int ) { return 0; }

template<int shiftval>
int VResizeCubicVec_32f16_avx2(const uchar**, uchar*, const uchar*, int ) { return 0; }
#endif

// Template instantiations.
template int VResizeLinearVec_32f16_avx2<SHRT_MIN>(const uchar** _src, uchar* _dst, const uchar* _beta, int width );
template int VResizeLinearVec_32f16_avx2<0>(const uchar** _src, uchar* _dst, const uchar* _beta, int width );

template int VResizeCubicVec_32f16_avx2<SHRT_MIN>(const uchar** _src, uchar* _dst, const uchar* _beta, int width );
template int VResizeCubicVec_32f16_avx2<0>(const uchar** _src, uchar* _dst, const uchar* _beta, int width );

/* End of file. */
