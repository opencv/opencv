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
// Copyright (C) 2014, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
/
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

namespace cv
{

template <typename T, typename AT>
int Acc_AVX(const T *, AT *, const uchar *, int, int)
{
    return 0;
}

template <typename T, typename AT>
int AccSqr_AVX(const T *, AT *, const uchar *, int, int)
{
    return 0;
}

template <typename T, typename AT>
int AccProd_AVX(const T *, const T *, AT *, const uchar *, int, int)
{
    return 0;
}

template <typename T, typename AT>
int AccW_AVX(const T *, AT *, const uchar *, int, int, AT)
{
    return 0;
}

template <>
int Acc_AVX<float, float>(const float * src, float * dst, const uchar * mask, int len, int cn)
{
    int x = 0;
    if (!mask)
    {
        len *= cn;
        for ( ; x <= len - 8 ; x += 8)
        {
            __m256 v_src = _mm256_loadu_ps(src + x);
            __m256 v_dst = _mm256_loadu_ps(dst + x);
            v_dst = _mm256_add_ps(v_src, v_dst);
            _mm256_storeu_ps(dst + x, v_dst);
        }
    }
    return x;
}

template <>
int Acc_AVX<float, double>(const float * src, double * dst, const uchar * mask, int len, int cn)
{
    int x = 0;
    if (!mask)
    {
        len *= cn;
        for ( ; x <= len - 8 ; x += 8)
        {
            __m256 v_src = _mm256_loadu_ps(src + x);
            __m256d v_src0 = _mm256_cvtps_pd(_mm256_extractf128_ps(v_src,0));
            __m256d v_src1 = _mm256_cvtps_pd(_mm256_extractf128_ps(v_src,1));
            __m256d v_dst0 = _mm256_loadu_pd(dst + x);
            __m256d v_dst1 = _mm256_loadu_pd(dst + x + 4);
            v_dst0 = _mm256_add_pd(v_src0, v_dst0);
            v_dst1 = _mm256_add_pd(v_src1, v_dst1);
            _mm256_storeu_pd(dst + x, v_dst0);
            _mm256_storeu_pd(dst + x + 4, v_dst1);
        }
    }
    return x;
}

template <>
int Acc_AVX<double, double>(const double * src, double * dst, const uchar * mask, int len, int cn)
{
    int x = 0;

    if (!mask)
    {
        len *= cn;
        for ( ; x <= len - 4; x += 4)
        {
            __m256d v_src = _mm256_loadu_pd(src + x);
            __m256d v_dst = _mm256_loadu_pd(dst + x);

            v_dst = _mm256_add_pd(v_dst, v_src);
            _mm256_storeu_pd(dst + x, v_dst);
        }
    }
    return x;
}

template <>
int AccSqr_AVX<float, float>(const float * src, float * dst, const uchar * mask, int len, int cn)
{
    int x = 0;
    if (!mask)
    {
        len *= cn;
        for ( ; x <= len - 8 ; x += 8)
        {
            __m256 v_src = _mm256_loadu_ps(src + x);
            __m256 v_dst = _mm256_loadu_ps(dst + x);

            v_src = _mm256_mul_ps(v_src, v_src);
            v_dst = _mm256_add_ps(v_src, v_dst);
            _mm256_storeu_ps(dst + x, v_dst);
        }
    }
    return x;
}

template <>
int AccSqr_AVX<float, double>(const float * src, double * dst, const uchar * mask, int len, int cn)
{
    int x = 0;
    if (!mask)
    {
        len *= cn;
        for ( ; x <= len - 8 ; x += 8)
        {
            __m256 v_src = _mm256_loadu_ps(src + x);
            __m256d v_src0 = _mm256_cvtps_pd(_mm256_extractf128_ps(v_src,0));
            __m256d v_src1 = _mm256_cvtps_pd(_mm256_extractf128_ps(v_src,1));
            __m256d v_dst0 = _mm256_loadu_pd(dst + x);
            __m256d v_dst1 = _mm256_loadu_pd(dst + x + 4);

            v_src0 = _mm256_mul_pd(v_src0, v_src0);
            v_src1 = _mm256_mul_pd(v_src1, v_src1);
            v_dst0 = _mm256_add_pd(v_src0, v_dst0);
            v_dst1 = _mm256_add_pd(v_src1, v_dst1);
            _mm256_storeu_pd(dst + x, v_dst0);
            _mm256_storeu_pd(dst + x + 4, v_dst1);
        }
    }
    return x;
}

template <>
int AccSqr_AVX<double, double>(const double * src, double * dst, const uchar * mask, int len, int cn)
{
    int x = 0;

    if (!mask)
    {
        len *= cn;
        for ( ; x <= len - 4; x += 4)
        {
            __m256d v_src = _mm256_loadu_pd(src + x);
            __m256d v_dst = _mm256_loadu_pd(dst + x);

            v_src = _mm256_mul_pd(v_src, v_src);
            v_dst = _mm256_add_pd(v_dst, v_src);
            _mm256_storeu_pd(dst + x, v_dst);
        }
    }
    return x;
}

template <>
int AccProd_AVX<float, float>(const float * src1, const float * src2, float * dst, const uchar * mask, int len, int cn)
{
    int x = 0;

    if (!mask)
    {
        len *= cn;
        for ( ; x <= len - 8; x += 8)
        {
            __m256 v_src0 = _mm256_loadu_ps(src1 + x);
            __m256 v_src1 = _mm256_loadu_ps(src2 + x);
            __m256 v_dst = _mm256_loadu_ps(dst + x);
            __m256 v_src = _mm256_mul_ps(v_src0, v_src1);

            v_dst = _mm256_add_ps(v_src, v_dst);
            _mm256_storeu_ps(dst + x, v_dst);
        }
    }

    return x;
}

template <>
int AccProd_AVX<float, double>(const float * src1, const float * src2, double * dst, const uchar * mask, int len, int cn)
{
    int x = 0;

    if (!mask)
    {
        len *= cn;
        for ( ; x <= len - 8; x += 8)
        {
            __m256 v_1src = _mm256_loadu_ps(src1 + x);
            __m256 v_2src = _mm256_loadu_ps(src2 + x);
            __m256d v_src00 = _mm256_cvtps_pd(_mm256_extractf128_ps(v_1src,0));
            __m256d v_src01 = _mm256_cvtps_pd(_mm256_extractf128_ps(v_1src,1));
            __m256d v_src10 = _mm256_cvtps_pd(_mm256_extractf128_ps(v_2src,0));
            __m256d v_src11 = _mm256_cvtps_pd(_mm256_extractf128_ps(v_2src,1));
            __m256d v_dst0 = _mm256_loadu_pd(dst + x);
            __m256d v_dst1 = _mm256_loadu_pd(dst + x + 4);

            __m256d v_src0 = _mm256_mul_pd(v_src00, v_src10);
            __m256d v_src1 = _mm256_mul_pd(v_src01, v_src11);
            v_dst0 = _mm256_add_pd(v_src0, v_dst0);
            v_dst1 = _mm256_add_pd(v_src1, v_dst1);
            _mm256_storeu_pd(dst + x, v_dst0);
            _mm256_storeu_pd(dst + x + 4, v_dst1);
        }
    }
    return x;
}

template <>
int AccProd_AVX<double, double>(const double * src1, const double * src2, double * dst, const uchar * mask, int len, int cn)
{
    int x = 0;

    if (!mask)
    {
        len *= cn;
        for ( ; x <= len - 4; x += 4)
        {
            __m256d v_src0 = _mm256_loadu_pd(src1 + x);
            __m256d v_src1 = _mm256_loadu_pd(src2 + x);
            __m256d v_dst = _mm256_loadu_pd(dst + x);

            v_src0 = _mm256_mul_pd(v_src0, v_src1);
            v_dst = _mm256_add_pd(v_dst, v_src0);
            _mm256_storeu_pd(dst + x, v_dst);
        }
    }
    return x;
}

template <>
int AccW_AVX<float, float>(const float * src, float * dst, const uchar * mask, int len, int cn, float alpha)
{
    int x = 0;
    __m256 v_alpha = _mm256_set1_ps(alpha);
    __m256 v_beta = _mm256_set1_ps(1.0f - alpha);

    if (!mask)
    {
        len *= cn;
        for ( ; x <= len - 16; x += 16)
        {
            _mm256_storeu_ps(dst + x, _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(dst + x), v_beta), _mm256_mul_ps(_mm256_loadu_ps(src + x), v_alpha)));
            _mm256_storeu_ps(dst + x + 8, _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(dst + x + 8), v_beta), _mm256_mul_ps(_mm256_loadu_ps(src + x + 8), v_alpha)));
        }
    }

    return x;
}

template <>
int AccW_AVX<float, double>(const float * src, double * dst, const uchar * mask, int len, int cn, double alpha)
{
    int x = 0;
    __m256d v_alpha = _mm256_set1_pd(alpha);
    __m256d v_beta = _mm256_set1_pd(1.0f - alpha);

    if (!mask)
    {
        len *= cn;
        for ( ; x <= len - 16; x += 16)
        {
            __m256 v_src0 = _mm256_loadu_ps(src + x);
            __m256 v_src1 = _mm256_loadu_ps(src + x + 8);
            __m256d v_src00 = _mm256_cvtps_pd(_mm256_extractf128_ps(v_src0,0));
            __m256d v_src01 = _mm256_cvtps_pd(_mm256_extractf128_ps(v_src0,1));
            __m256d v_src10 = _mm256_cvtps_pd(_mm256_extractf128_ps(v_src1,0));
            __m256d v_src11 = _mm256_cvtps_pd(_mm256_extractf128_ps(v_src1,1));

            _mm256_storeu_pd(dst + x, _mm256_add_pd(_mm256_mul_pd(_mm256_loadu_pd(dst + x), v_beta), _mm256_mul_pd(v_src00, v_alpha)));
            _mm256_storeu_pd(dst + x + 4, _mm256_add_pd(_mm256_mul_pd(_mm256_loadu_pd(dst + x + 4), v_beta), _mm256_mul_pd(v_src01, v_alpha)));
            _mm256_storeu_pd(dst + x + 8, _mm256_add_pd(_mm256_mul_pd(_mm256_loadu_pd(dst + x + 8), v_beta), _mm256_mul_pd(v_src10, v_alpha)));
            _mm256_storeu_pd(dst + x + 12, _mm256_add_pd(_mm256_mul_pd(_mm256_loadu_pd(dst + x + 12), v_beta), _mm256_mul_pd(v_src11, v_alpha)));
        }
    }

    return x;
}

template <>
int AccW_AVX<double, double>(const double * src, double * dst, const uchar * mask, int len, int cn, double alpha)
{
    int x = 0;
    __m256d v_alpha = _mm256_set1_pd(alpha);
    __m256d v_beta = _mm256_set1_pd(1.0f - alpha);

    if (!mask)
    {
        len *= cn;
        for ( ; x <= len - 8; x += 8)
        {
            __m256d v_src0 = _mm256_loadu_pd(src + x);
            __m256d v_src1 = _mm256_loadu_pd(src + x + 4);

            _mm256_storeu_pd(dst + x, _mm256_add_pd(_mm256_mul_pd(_mm256_loadu_pd(dst + x), v_beta), _mm256_mul_pd(v_src0, v_alpha)));
            _mm256_storeu_pd(dst + x + 4, _mm256_add_pd(_mm256_mul_pd(_mm256_loadu_pd(dst + x + 4), v_beta), _mm256_mul_pd(v_src1, v_alpha)));
        }
    }

    return x;
}

#define INSTANTIATE_ACC_AVX(type, acctype)  \
template int Acc_AVX<type, acctype>(const type *, acctype *, const uchar *, int, int);     \
template int AccSqr_AVX<type, acctype>(const type *, acctype *, const uchar *, int, int); \
template int AccProd_AVX<type, acctype>(const type *, const type *, acctype *, const uchar *, int, int); \
template int AccW_AVX<type, acctype>(const type *, acctype *, const uchar *, int, int, acctype);

INSTANTIATE_ACC_AVX(uchar, float)
INSTANTIATE_ACC_AVX(uchar, double)
INSTANTIATE_ACC_AVX(ushort, float)
INSTANTIATE_ACC_AVX(ushort, double)

}

/* End of file. */
