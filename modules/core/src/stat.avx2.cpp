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
#include "stat.hpp"

namespace cv { namespace hal {
namespace opt_AVX2
{

static inline int _mm256_extract_epi32_(__m256i reg, const int i)
{
    CV_DECL_ALIGNED(32) int reg_data[8];
    CV_DbgAssert(0 <= i && i < 8);
    _mm256_store_si256((__m256i*)reg_data, reg);
    return reg_data[i];
}

int normHamming_AVX2(const uchar* a, int n, int& result)
{
    int i = 0;
    __m256i _r0 = _mm256_setzero_si256();
    __m256i _0 = _mm256_setzero_si256();
    __m256i _popcnt_table = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
                                                0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    __m256i _popcnt_mask = _mm256_set1_epi8(0x0F);

    for(; i <= n - 32; i+= 32)
    {
        __m256i _a0 = _mm256_loadu_si256((const __m256i*)(a + i));

        __m256i _popc0 = _mm256_shuffle_epi8(_popcnt_table, _mm256_and_si256(_a0, _popcnt_mask));
        __m256i _popc1 = _mm256_shuffle_epi8(_popcnt_table,
                            _mm256_and_si256(_mm256_srli_epi16(_a0, 4), _popcnt_mask));

        _r0 = _mm256_add_epi32(_r0, _mm256_sad_epu8(_0, _mm256_add_epi8(_popc0, _popc1)));
    }
    _r0 = _mm256_add_epi32(_r0, _mm256_shuffle_epi32(_r0, 2));
    result = _mm256_extract_epi32_(_mm256_add_epi32(_r0, _mm256_permute2x128_si256(_r0, _r0, 1)), 0);
    return i;
}

int normHamming_AVX2(const uchar* a, const uchar* b, int n, int& result)
{
    int i = 0;
    __m256i _r0 = _mm256_setzero_si256();
    __m256i _0 = _mm256_setzero_si256();
    __m256i _popcnt_table = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
                                                0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    __m256i _popcnt_mask = _mm256_set1_epi8(0x0F);

    for(; i <= n - 32; i+= 32)
    {
        __m256i _a0 = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i _b0 = _mm256_loadu_si256((const __m256i*)(b + i));

        __m256i _xor = _mm256_xor_si256(_a0, _b0);

        __m256i _popc0 = _mm256_shuffle_epi8(_popcnt_table, _mm256_and_si256(_xor, _popcnt_mask));
        __m256i _popc1 = _mm256_shuffle_epi8(_popcnt_table,
                            _mm256_and_si256(_mm256_srli_epi16(_xor, 4), _popcnt_mask));

        _r0 = _mm256_add_epi32(_r0, _mm256_sad_epu8(_0, _mm256_add_epi8(_popc0, _popc1)));
    }
    _r0 = _mm256_add_epi32(_r0, _mm256_shuffle_epi32(_r0, 2));
    result = _mm256_extract_epi32_(_mm256_add_epi32(_r0, _mm256_permute2x128_si256(_r0, _r0, 1)), 0);
    return i;
}

float normL2Sqr_AVX2(const float* a, const float* b, int n)
{
    int j = 0; float d = 0.f;
    float CV_DECL_ALIGNED(32) buf[8];
    __m256 d0 = _mm256_setzero_ps();

    for( ; j <= n - 8; j += 8 )
    {
        __m256 t0 = _mm256_sub_ps(_mm256_loadu_ps(a + j), _mm256_loadu_ps(b + j));
#ifdef CV_FMA3
        d0 = _mm256_fmadd_ps(t0, t0, d0);
#else
        d0 = _mm256_add_ps(d0, _mm256_mul_ps(t0, t0));
#endif
    }
    _mm256_store_ps(buf, d0);
    d = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];

    if(j <= n - 4)
    {
        float t0 = a[j] - b[j], t1 = a[j+1] - b[j+1], t2 = a[j+2] - b[j+2], t3 = a[j+3] - b[j+3];
        d += t0*t0 + t1*t1 + t2*t2 + t3*t3;
        j += 4;
    }

    for( ; j < n; j++ )
    {
        float t = a[j] - b[j];
        d += t*t;
    }
    return d;
}

}
}} //cv::hal
