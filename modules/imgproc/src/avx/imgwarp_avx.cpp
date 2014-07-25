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
#include "imgwarp_avx.hpp"

#if CV_AVX
int VResizeLinearVec_32f_avx(const uchar** _src, uchar* _dst, const uchar* _beta, int width )
{
    const float** src = (const float**)_src;
    const float* beta = (const float*)_beta;
    const float *S0 = src[0], *S1 = src[1];
    float* dst = (float*)_dst;
    int x = 0;

    __m256 b0 = _mm256_set1_ps(beta[0]), b1 = _mm256_set1_ps(beta[1]);

    if( (((size_t)S0|(size_t)S1)&31) == 0 )
        for( ; x <= width - 16; x += 16 )
        {
            __m256 x0, x1, y0, y1;
            x0 = _mm256_load_ps(S0 + x);
            x1 = _mm256_load_ps(S0 + x + 8);
            y0 = _mm256_load_ps(S1 + x);
            y1 = _mm256_load_ps(S1 + x + 8);

            x0 = _mm256_add_ps(_mm256_mul_ps(x0, b0), _mm256_mul_ps(y0, b1));
            x1 = _mm256_add_ps(_mm256_mul_ps(x1, b0), _mm256_mul_ps(y1, b1));

            _mm256_storeu_ps( dst + x, x0);
            _mm256_storeu_ps( dst + x + 8, x1);
        }
    else
        for( ; x <= width - 16; x += 16 )
        {
            __m256 x0, x1, y0, y1;
            x0 = _mm256_loadu_ps(S0 + x);
            x1 = _mm256_loadu_ps(S0 + x + 8);
            y0 = _mm256_loadu_ps(S1 + x);
            y1 = _mm256_loadu_ps(S1 + x + 8);

            x0 = _mm256_add_ps(_mm256_mul_ps(x0, b0), _mm256_mul_ps(y0, b1));
            x1 = _mm256_add_ps(_mm256_mul_ps(x1, b0), _mm256_mul_ps(y1, b1));

            _mm256_storeu_ps( dst + x, x0);
            _mm256_storeu_ps( dst + x + 8, x1);
        }

    return x;
}

int VResizeCubicVec_32f_avx(const uchar** _src, uchar* _dst, const uchar* _beta, int width )
{
    const float** src = (const float**)_src;
    const float* beta = (const float*)_beta;
    const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3];
    float* dst = (float*)_dst;
    int x = 0;
    __m256 b0 = _mm256_set1_ps(beta[0]), b1 = _mm256_set1_ps(beta[1]),
        b2 = _mm256_set1_ps(beta[2]), b3 = _mm256_set1_ps(beta[3]);

    if( (((size_t)S0|(size_t)S1|(size_t)S2|(size_t)S3)&31) == 0 )
        for( ; x <= width - 16; x += 16 )
        {
            __m256 x0, x1, y0, y1, s0, s1;
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

            _mm256_storeu_ps( dst + x, s0);
            _mm256_storeu_ps( dst + x + 8, s1);
        }
    else
        for( ; x <= width - 16; x += 16 )
        {
            __m256 x0, x1, y0, y1, s0, s1;
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

            _mm256_storeu_ps( dst + x, s0);
            _mm256_storeu_ps( dst + x + 8, s1);
        }

    return x;
}
#else
int VResizeLinearVec_32f_avx(const uchar**, uchar*, const uchar*, int ) { return 0; }

int VResizeCubicVec_32f_avx(const uchar**, uchar*, const uchar*, int ) { return 0; }
#endif

/* End of file. */
