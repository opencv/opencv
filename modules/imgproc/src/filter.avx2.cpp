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

#include "precomp.hpp"
#include "filter.hpp"

namespace cv
{

int RowVec_32f_AVX(const float* src0, const float* _kx, float* dst, int width, int cn, int _ksize)
{
    int i = 0, k;
    for (; i <= width - 8; i += 8)
    {
        const float* src = src0 + i;
        __m256 f, x0;
        __m256 s0 = _mm256_set1_ps(0.0f);
        for (k = 0; k < _ksize; k++, src += cn)
        {
            f = _mm256_set1_ps(_kx[k]);
            x0 = _mm256_loadu_ps(src);
#if CV_FMA3
            s0 = _mm256_fmadd_ps(x0, f, s0);
#else
            s0 = _mm256_add_ps(s0, _mm256_mul_ps(x0, f));
#endif
        }
        _mm256_storeu_ps(dst + i, s0);
    }
    _mm256_zeroupper();
    return i;
}

int SymmColumnVec_32f_Symm_AVX(const float** src, const float* ky, float* dst, float delta, int width, int ksize2)
{
    int i = 0, k;
    const float *S, *S2;
    const __m128 d4 = _mm_set1_ps(delta);
    const __m256 d8 = _mm256_set1_ps(delta);

    for( ; i <= width - 16; i += 16 )
    {
        __m256 f = _mm256_set1_ps(ky[0]);
        __m256 s0, s1;
        __m256 x0;
        S = src[0] + i;
        s0 = _mm256_loadu_ps(S);
#if CV_FMA3
        s0 = _mm256_fmadd_ps(s0, f, d8);
#else
        s0 = _mm256_add_ps(_mm256_mul_ps(s0, f), d8);
#endif
        s1 = _mm256_loadu_ps(S+8);
#if CV_FMA3
        s1 = _mm256_fmadd_ps(s1, f, d8);
#else
        s1 = _mm256_add_ps(_mm256_mul_ps(s1, f), d8);
#endif

        for( k = 1; k <= ksize2; k++ )
        {
            S = src[k] + i;
            S2 = src[-k] + i;
            f = _mm256_set1_ps(ky[k]);
            x0 = _mm256_add_ps(_mm256_loadu_ps(S), _mm256_loadu_ps(S2));
#if CV_FMA3
            s0 = _mm256_fmadd_ps(x0, f, s0);
#else
            s0 = _mm256_add_ps(s0, _mm256_mul_ps(x0, f));
#endif
            x0 = _mm256_add_ps(_mm256_loadu_ps(S+8), _mm256_loadu_ps(S2+8));
#if CV_FMA3
            s1 = _mm256_fmadd_ps(x0, f, s1);
#else
            s1 = _mm256_add_ps(s1, _mm256_mul_ps(x0, f));
#endif
        }

        _mm256_storeu_ps(dst + i, s0);
        _mm256_storeu_ps(dst + i + 8, s1);
    }

    for( ; i <= width - 4; i += 4 )
    {
        __m128 f = _mm_set1_ps(ky[0]);
        __m128 x0, s0 = _mm_load_ps(src[0] + i);
        s0 = _mm_add_ps(_mm_mul_ps(s0, f), d4);

        for( k = 1; k <= ksize2; k++ )
        {
            f = _mm_set1_ps(ky[k]);
            x0 = _mm_add_ps(_mm_load_ps(src[k]+i), _mm_load_ps(src[-k] + i));
            s0 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
        }

        _mm_storeu_ps(dst + i, s0);
    }

    _mm256_zeroupper();
    return i;
}

int SymmColumnVec_32f_Unsymm_AVX(const float** src, const float* ky, float* dst, float delta, int width, int ksize2)
{
    int i = 0, k;
    const float *S2;
    const __m128 d4 = _mm_set1_ps(delta);
    const __m256 d8 = _mm256_set1_ps(delta);

    for (; i <= width - 16; i += 16)
    {
        __m256 f, s0 = d8, s1 = d8;
        __m256 x0;

        for (k = 1; k <= ksize2; k++)
        {
            const float *S = src[k] + i;
            S2 = src[-k] + i;
            f = _mm256_set1_ps(ky[k]);
            x0 = _mm256_sub_ps(_mm256_loadu_ps(S), _mm256_loadu_ps(S2));
#if CV_FMA3
            s0 = _mm256_fmadd_ps(x0, f, s0);
#else
            s0 = _mm256_add_ps(s0, _mm256_mul_ps(x0, f));
#endif
            x0 = _mm256_sub_ps(_mm256_loadu_ps(S + 8), _mm256_loadu_ps(S2 + 8));
#if CV_FMA3
            s1 = _mm256_fmadd_ps(x0, f, s1);
#else
            s1 = _mm256_add_ps(s1, _mm256_mul_ps(x0, f));
#endif
        }

        _mm256_storeu_ps(dst + i, s0);
        _mm256_storeu_ps(dst + i + 8, s1);
    }

    for (; i <= width - 4; i += 4)
    {
        __m128 f, x0, s0 = d4;

        for (k = 1; k <= ksize2; k++)
        {
            f = _mm_set1_ps(ky[k]);
            x0 = _mm_sub_ps(_mm_load_ps(src[k] + i), _mm_load_ps(src[-k] + i));
            s0 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
        }

        _mm_storeu_ps(dst + i, s0);
    }

    _mm256_zeroupper();
    return i;
}

}

/* End of file. */
