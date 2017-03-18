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
#include "opencl_kernels_imgproc.hpp"
#include "opencv2/core/hal/intrin.hpp"

#include "opencv2/core/openvx/ovx_defs.hpp"

namespace cv
{

template <typename T, typename AT>
struct Acc_SIMD
{
    int operator() (const T *, AT *, const uchar *, int, int) const
    {
        return 0;
    }
};

template <typename T, typename AT>
struct AccSqr_SIMD
{
    int operator() (const T *, AT *, const uchar *, int, int) const
    {
        return 0;
    }
};

template <typename T, typename AT>
struct AccProd_SIMD
{
    int operator() (const T *, const T *, AT *, const uchar *, int, int) const
    {
        return 0;
    }
};

template <typename T, typename AT>
struct AccW_SIMD
{
    int operator() (const T *, AT *, const uchar *, int, int, AT) const
    {
        return 0;
    }
};

#if CV_AVX
template <>
struct Acc_SIMD<float, float>
{
    int operator() (const float * src, float * dst, const uchar * mask, int len, int cn) const
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
};

template <>
struct Acc_SIMD<float, double>
{
    int operator() (const float * src, double * dst, const uchar * mask, int len, int cn) const
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
};

template <>
struct Acc_SIMD<double, double>
{
    int operator() (const double * src, double * dst, const uchar * mask, int len, int cn) const
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
};

template <>
struct AccSqr_SIMD<float, float>
{
    int operator() (const float * src, float * dst, const uchar * mask, int len, int cn) const
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
};

template <>
struct AccSqr_SIMD<float, double>
{
    int operator() (const float * src, double * dst, const uchar * mask, int len, int cn) const
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
};

template <>
struct AccSqr_SIMD<double, double>
{
    int operator() (const double * src, double * dst, const uchar * mask, int len, int cn) const
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
};

template <>
struct AccProd_SIMD<float, float>
{
    int operator() (const float * src1, const float * src2, float * dst, const uchar * mask, int len, int cn) const
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
};

template <>
struct AccProd_SIMD<float, double>
{
    int operator() (const float * src1, const float * src2, double * dst, const uchar * mask, int len, int cn) const
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
};

template <>
struct AccProd_SIMD<double, double>
{
    int operator() (const double * src1, const double * src2, double * dst, const uchar * mask, int len, int cn) const
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
};

template <>
struct AccW_SIMD<float, float>
{
    int operator() (const float * src, float * dst, const uchar * mask, int len, int cn, float alpha) const
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
};

template <>
struct AccW_SIMD<float, double>
{
    int operator() (const float * src, double * dst, const uchar * mask, int len, int cn, double alpha) const
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
};

template <>
struct AccW_SIMD<double, double>
{
    int operator() (const double * src, double * dst, const uchar * mask, int len, int cn, double alpha) const
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
};
#elif CV_SIMD128
template <>
struct Acc_SIMD<float, float>
{
    int operator() (const float * src, float * dst, const uchar * mask, int len, int cn) const
    {
        int x = 0;

        if (!mask)
        {
            len *= cn;
            for ( ; x <= len - 8; x += 8)
            {
                v_store(dst + x, v_load(dst + x) + v_load(src + x));
                v_store(dst + x + 4, v_load(dst + x + 4) + v_load(src + x + 4));
            }
        }

        return x;
    }
};

#if CV_SIMD128_64F
template <>
struct Acc_SIMD<float, double>
{
    int operator() (const float * src, double * dst, const uchar * mask, int len, int cn) const
    {
        int x = 0;

        if (!mask)
        {
            len *= cn;
            for ( ; x <= len - 4; x += 4)
            {
                v_float32x4 v_src = v_load(src + x);
                v_float64x2 v_src0 = v_cvt_f64(v_src);
                v_float64x2 v_src1 = v_cvt_f64_high(v_src);

                v_store(dst + x, v_load(dst + x) + v_src0);
                v_store(dst + x + 2, v_load(dst + x + 2) + v_src1);
            }
        }
        return x;
    }
};

template <>
struct Acc_SIMD<double, double>
{
    int operator() (const double * src, double * dst, const uchar * mask, int len, int cn) const
    {
        int x = 0;

        if (!mask)
        {
            len *= cn;
            for ( ; x <= len - 4; x += 4)
            {
                v_float64x2 v_src0 = v_load(src + x);
                v_float64x2 v_src1 = v_load(src + x + 2);

                v_store(dst + x, v_load(dst + x) + v_src0);
                v_store(dst + x + 2, v_load(dst + x + 2) + v_src1);
            }
        }
        return x;
    }
};
#endif //CV_SIMD128_64F

template <>
struct AccSqr_SIMD<float, float>
{
    int operator() (const float * src, float * dst, const uchar * mask, int len, int cn) const
    {
        int x = 0;

        if (!mask)
        {
            len *= cn;
            for ( ; x <= len - 8; x += 8)
            {
                v_float32x4 v_src0 = v_load(src + x);
                v_float32x4 v_src1 = v_load(src + x + 4);
                v_src0 = v_src0 * v_src0;
                v_src1 = v_src1 * v_src1;

                v_store(dst + x, v_load(dst + x) + v_src0);
                v_store(dst + x + 4, v_load(dst + x + 4) + v_src1);
            }
        }

        return x;
    }
};

#if CV_SIMD128_64F
template <>
struct AccSqr_SIMD<float, double>
{
    int operator() (const float * src, double * dst, const uchar * mask, int len, int cn) const
    {
        int x = 0;

        if (!mask)
        {
            len *= cn;
            for ( ; x <= len - 4; x += 4)
            {
                v_float32x4 v_src = v_load(src + x);
                v_float64x2 v_src0 = v_cvt_f64(v_src);
                v_float64x2 v_src1 = v_cvt_f64_high(v_src);
                v_src0 = v_src0 * v_src0;
                v_src1 = v_src1 * v_src1;

                v_store(dst + x, v_load(dst + x) + v_src0);
                v_store(dst + x + 2, v_load(dst + x + 2) + v_src1);
            }
        }
        return x;
    }
};

template <>
struct AccSqr_SIMD<double, double>
{
    int operator() (const double * src, double * dst, const uchar * mask, int len, int cn) const
    {
        int x = 0;

        if (!mask)
        {
            len *= cn;
            for ( ; x <= len - 4; x += 4)
            {
                v_float64x2 v_src0 = v_load(src + x);
                v_float64x2 v_src1 = v_load(src + x + 2);
                v_src0 = v_src0 * v_src0;
                v_src1 = v_src1 * v_src1;

                v_store(dst + x, v_load(dst + x) + v_src0);
                v_store(dst + x + 2, v_load(dst + x + 2) + v_src1);
            }
        }
        return x;
    }
};
#endif //CV_SIMD128_64F

template <>
struct AccProd_SIMD<float, float>
{
    int operator() (const float * src1, const float * src2, float * dst, const uchar * mask, int len, int cn) const
    {
        int x = 0;

        if (!mask)
        {
            len *= cn;
            for ( ; x <= len - 8; x += 8)
            {
                v_store(dst + x, v_load(dst + x) + v_load(src1 + x) * v_load(src2 + x));
                v_store(dst + x + 4, v_load(dst + x + 4) + v_load(src1 + x + 4) * v_load(src2 + x + 4));
            }
        }

        return x;
    }
};

#if CV_SIMD128_64F
template <>
struct AccProd_SIMD<float, double>
{
    int operator() (const float * src1, const float * src2, double * dst, const uchar * mask, int len, int cn) const
    {
        int x = 0;

        if (!mask)
        {
            len *= cn;
            for ( ; x <= len - 4; x += 4)
            {
                v_float32x4 v_1src  = v_load(src1 + x);
                v_float32x4 v_2src  = v_load(src2 + x);

                v_float64x2 v_1src0 = v_cvt_f64(v_1src);
                v_float64x2 v_1src1 = v_cvt_f64_high(v_1src);
                v_float64x2 v_2src0 = v_cvt_f64(v_2src);
                v_float64x2 v_2src1 = v_cvt_f64_high(v_2src);

                v_store(dst + x, v_load(dst + x) + (v_1src0 * v_2src0));
                v_store(dst + x + 2, v_load(dst + x + 2) + (v_1src1 * v_2src1));
            }
        }
        return x;
    }
};

template <>
struct AccProd_SIMD<double, double>
{
    int operator() (const double * src1, const double * src2, double * dst, const uchar * mask, int len, int cn) const
    {
        int x = 0;

        if (!mask)
        {
            len *= cn;
            for ( ; x <= len - 4; x += 4)
            {
                v_float64x2 v_src00 = v_load(src1 + x);
                v_float64x2 v_src01 = v_load(src1 + x + 2);
                v_float64x2 v_src10 = v_load(src2 + x);
                v_float64x2 v_src11 = v_load(src2 + x + 2);

                v_store(dst + x, v_load(dst + x) + (v_src00 * v_src10));
                v_store(dst + x + 2, v_load(dst + x + 2) + (v_src01 * v_src11));
            }
        }
        return x;
    }
};
#endif //CV_SIMD128_64F

template <>
struct AccW_SIMD<float, float>
{
    int operator() (const float * src, float * dst, const uchar * mask, int len, int cn, float alpha) const
    {
        int x = 0;
        v_float32x4 v_alpha = v_setall_f32(alpha);
        v_float32x4 v_beta = v_setall_f32(1.0f - alpha);

        if (!mask)
        {
            len *= cn;
            for ( ; x <= len - 8; x += 8)
            {
                v_store(dst + x, ((v_load(dst + x) * v_beta) + (v_load(src + x) * v_alpha)));
                v_store(dst + x + 4, ((v_load(dst + x + 4) * v_beta) + (v_load(src + x + 4) * v_alpha)));
            }
        }

        return x;
    }
};

#if CV_SIMD128_64F
template <>
struct AccW_SIMD<float, double>
{
    int operator() (const float * src, double * dst, const uchar * mask, int len, int cn, double alpha) const
    {
        int x = 0;
        v_float64x2 v_alpha = v_setall_f64(alpha);
        v_float64x2 v_beta = v_setall_f64(1.0f - alpha);

        if (!mask)
        {
            len *= cn;
            for ( ; x <= len - 8; x += 8)
            {
                v_float32x4 v_src0 = v_load(src + x);
                v_float32x4 v_src1 = v_load(src + x + 4);
                v_float64x2 v_src00 = v_cvt_f64(v_src0);
                v_float64x2 v_src01 = v_cvt_f64_high(v_src0);
                v_float64x2 v_src10 = v_cvt_f64(v_src1);
                v_float64x2 v_src11 = v_cvt_f64_high(v_src1);

                v_store(dst + x, ((v_load(dst + x) * v_beta) + (v_src00 * v_alpha)));
                v_store(dst + x + 2, ((v_load(dst + x + 2) * v_beta) + (v_src01 * v_alpha)));
                v_store(dst + x + 4, ((v_load(dst + x + 4) * v_beta) + (v_src10 * v_alpha)));
                v_store(dst + x + 6, ((v_load(dst + x + 6) * v_beta) + (v_src11 * v_alpha)));
            }
        }

        return x;
    }
};

template <>
struct AccW_SIMD<double, double>
{
    int operator() (const double * src, double * dst, const uchar * mask, int len, int cn, double alpha) const
    {
        int x = 0;
        v_float64x2 v_alpha = v_setall_f64(alpha);
        v_float64x2 v_beta = v_setall_f64(1.0f - alpha);

        if (!mask)
        {
            len *= cn;
            for ( ; x <= len - 4; x += 4)
            {
                v_float64x2 v_src0 = v_load(src + x);
                v_float64x2 v_src1 = v_load(src + x + 2);

                v_store(dst + x, ((v_load(dst + x) * v_beta) + (v_src0 * v_alpha)));
                v_store(dst + x + 2, ((v_load(dst + x + 2) * v_beta) + (v_src1 * v_alpha)));
            }
        }

        return x;
    }
};
#endif //CV_SIMD128_64F
#endif //CV_SIMD128

#if CV_SIMD128
template <>
struct Acc_SIMD<uchar, float>
{
    int operator() (const uchar * src, float * dst, const uchar * mask, int len, int cn) const
    {
        int x = 0;

        if (!mask)
        {
            len *= cn;
            for ( ; x <= len - 16; x += 16)
            {
                v_uint8x16 v_src  = v_load(src + x);
                v_uint16x8 v_src0, v_src1;
                v_expand(v_src, v_src0, v_src1);

                v_uint32x4 v_src00, v_src01, v_src10, v_src11;
                v_expand(v_src0, v_src00, v_src01);
                v_expand(v_src1, v_src10, v_src11);

                v_store(dst + x, v_load(dst + x) + v_cvt_f32(v_reinterpret_as_s32(v_src00)));
                v_store(dst + x + 4, v_load(dst + x + 4) + v_cvt_f32(v_reinterpret_as_s32(v_src01)));
                v_store(dst + x + 8, v_load(dst + x + 8) + v_cvt_f32(v_reinterpret_as_s32(v_src10)));
                v_store(dst + x + 12, v_load(dst + x + 12) + v_cvt_f32(v_reinterpret_as_s32(v_src11)));
            }
        }
        else if (cn == 1)
        {
            v_uint8x16 v_0 = v_setall_u8(0);

            for ( ; x <= len - 16; x += 16)
            {
                v_uint8x16 v_mask = v_load(mask + x);
                v_mask = ~(v_0 == v_mask);
                v_uint8x16 v_src = v_load(src + x);
                v_src = v_src & v_mask;
                v_uint16x8 v_src0, v_src1;
                v_expand(v_src, v_src0, v_src1);

                v_uint32x4 v_src00, v_src01, v_src10, v_src11;
                v_expand(v_src0, v_src00, v_src01);
                v_expand(v_src1, v_src10, v_src11);

                v_store(dst + x, v_load(dst + x) + v_cvt_f32(v_reinterpret_as_s32(v_src00)));
                v_store(dst + x + 4, v_load(dst + x + 4) + v_cvt_f32(v_reinterpret_as_s32(v_src01)));
                v_store(dst + x + 8, v_load(dst + x + 8) + v_cvt_f32(v_reinterpret_as_s32(v_src10)));
                v_store(dst + x + 12, v_load(dst + x + 12) + v_cvt_f32(v_reinterpret_as_s32(v_src11)));
            }
        }

        return x;
    }
};

template <>
struct Acc_SIMD<ushort, float>
{
    int operator() (const ushort * src, float * dst, const uchar * mask, int len, int cn) const
    {
        int x = 0;
        if (!mask)
        {
            len *= cn;
            for ( ; x <= len - 8; x += 8)
            {
                v_uint16x8 v_src = v_load(src + x);
                v_uint32x4 v_src0, v_src1;
                v_expand(v_src, v_src0, v_src1);

                v_store(dst + x, v_load(dst + x) + v_cvt_f32(v_reinterpret_as_s32(v_src0)));
                v_store(dst + x + 4, v_load(dst + x + 4) + v_cvt_f32(v_reinterpret_as_s32(v_src1)));
            }
        }

        return x;
    }
};

#if CV_SIMD128_64F
template <>
struct Acc_SIMD<uchar, double>
{
    int operator() (const uchar * src, double * dst, const uchar * mask, int len, int cn) const
    {
        int x = 0;

        if (!mask)
        {
            len *= cn;
            for ( ; x <= len - 16; x += 16)
            {
                v_uint8x16 v_src  = v_load(src + x);
                v_uint16x8 v_int0, v_int1;
                v_expand(v_src, v_int0, v_int1);

                v_uint32x4 v_int00, v_int01, v_int10, v_int11;
                v_expand(v_int0, v_int00, v_int01);
                v_expand(v_int1, v_int10, v_int11);

                v_float64x2 v_src0 = v_cvt_f64(v_reinterpret_as_s32(v_int00));
                v_float64x2 v_src1 = v_cvt_f64_high(v_reinterpret_as_s32(v_int00));
                v_float64x2 v_src2 = v_cvt_f64(v_reinterpret_as_s32(v_int01));
                v_float64x2 v_src3 = v_cvt_f64_high(v_reinterpret_as_s32(v_int01));
                v_float64x2 v_src4 = v_cvt_f64(v_reinterpret_as_s32(v_int10));
                v_float64x2 v_src5 = v_cvt_f64_high(v_reinterpret_as_s32(v_int10));
                v_float64x2 v_src6 = v_cvt_f64(v_reinterpret_as_s32(v_int11));
                v_float64x2 v_src7 = v_cvt_f64_high(v_reinterpret_as_s32(v_int11));

                v_float64x2 v_dst0 = v_load(dst + x);
                v_float64x2 v_dst1 = v_load(dst + x + 2);
                v_float64x2 v_dst2 = v_load(dst + x + 4);
                v_float64x2 v_dst3 = v_load(dst + x + 6);
                v_float64x2 v_dst4 = v_load(dst + x + 8);
                v_float64x2 v_dst5 = v_load(dst + x + 10);
                v_float64x2 v_dst6 = v_load(dst + x + 12);
                v_float64x2 v_dst7 = v_load(dst + x + 14);

                v_dst0 = v_dst0 + v_src0;
                v_dst1 = v_dst1 + v_src1;
                v_dst2 = v_dst2 + v_src2;
                v_dst3 = v_dst3 + v_src3;
                v_dst4 = v_dst4 + v_src4;
                v_dst5 = v_dst5 + v_src5;
                v_dst6 = v_dst6 + v_src6;
                v_dst7 = v_dst7 + v_src7;

                v_store(dst + x, v_dst0);
                v_store(dst + x + 2, v_dst1);
                v_store(dst + x + 4, v_dst2);
                v_store(dst + x + 6, v_dst3);
                v_store(dst + x + 8, v_dst4);
                v_store(dst + x + 10, v_dst5);
                v_store(dst + x + 12, v_dst6);
                v_store(dst + x + 14, v_dst7);
            }
        }
        return x;
    }
};

template <>
struct Acc_SIMD<ushort, double>
{
    int operator() (const ushort * src, double * dst, const uchar * mask, int len, int cn) const
    {
        int x = 0;

        if (!mask)
        {
            len *= cn;
            for ( ; x <= len - 8; x += 8)
            {
                v_uint16x8 v_src  = v_load(src + x);
                v_uint32x4 v_int0, v_int1;
                v_expand(v_src, v_int0, v_int1);

                v_float64x2 v_src0 = v_cvt_f64(v_reinterpret_as_s32(v_int0));
                v_float64x2 v_src1 = v_cvt_f64_high(v_reinterpret_as_s32(v_int0));
                v_float64x2 v_src2 = v_cvt_f64(v_reinterpret_as_s32(v_int1));
                v_float64x2 v_src3 = v_cvt_f64_high(v_reinterpret_as_s32(v_int1));

                v_float64x2 v_dst0 = v_load(dst + x);
                v_float64x2 v_dst1 = v_load(dst + x + 2);
                v_float64x2 v_dst2 = v_load(dst + x + 4);
                v_float64x2 v_dst3 = v_load(dst + x + 6);

                v_dst0 = v_dst0 + v_src0;
                v_dst1 = v_dst1 + v_src1;
                v_dst2 = v_dst2 + v_src2;
                v_dst3 = v_dst3 + v_src3;

                v_store(dst + x, v_dst0);
                v_store(dst + x + 2, v_dst1);
                v_store(dst + x + 4, v_dst2);
                v_store(dst + x + 6, v_dst3);
            }
        }
        return x;
    }
};
#endif

template <>
struct AccSqr_SIMD<uchar, float>
{
    int operator() (const uchar * src, float * dst, const uchar * mask, int len, int cn) const
    {
        int x = 0;

        if (!mask)
        {
            len *= cn;
            for ( ; x <= len - 16; x += 16)
            {
                v_uint8x16 v_src  = v_load(src + x);
                v_uint16x8 v_src0, v_src1;
                v_expand(v_src, v_src0, v_src1);
                v_src0 = v_src0 * v_src0;
                v_src1 = v_src1 * v_src1;

                v_uint32x4 v_src00, v_src01, v_src10, v_src11;
                v_expand(v_src0, v_src00, v_src01);
                v_expand(v_src1, v_src10, v_src11);

                v_store(dst + x, v_load(dst + x) + v_cvt_f32(v_reinterpret_as_s32(v_src00)));
                v_store(dst + x + 4, v_load(dst + x + 4) + v_cvt_f32(v_reinterpret_as_s32(v_src01)));
                v_store(dst + x + 8, v_load(dst + x + 8) + v_cvt_f32(v_reinterpret_as_s32(v_src10)));
                v_store(dst + x + 12, v_load(dst + x + 12) + v_cvt_f32(v_reinterpret_as_s32(v_src11)));
            }
        }
        else if (cn == 1)
        {
            v_uint8x16 v_0 = v_setall_u8(0);
            for ( ; x <= len - 16; x += 16)
            {
                v_uint8x16 v_mask = v_load(mask + x);
                v_mask = ~(v_0 == v_mask);
                v_uint8x16 v_src = v_load(src + x);
                v_src = v_src & v_mask;
                v_uint16x8 v_src0, v_src1;
                v_expand(v_src, v_src0, v_src1);
                v_src0 = v_src0 * v_src0;
                v_src1 = v_src1 * v_src1;

                v_uint32x4 v_src00, v_src01, v_src10, v_src11;
                v_expand(v_src0, v_src00, v_src01);
                v_expand(v_src1, v_src10, v_src11);

                v_store(dst + x, v_load(dst + x) + v_cvt_f32(v_reinterpret_as_s32(v_src00)));
                v_store(dst + x + 4, v_load(dst + x + 4) + v_cvt_f32(v_reinterpret_as_s32(v_src01)));
                v_store(dst + x + 8, v_load(dst + x + 8) + v_cvt_f32(v_reinterpret_as_s32(v_src10)));
                v_store(dst + x + 12, v_load(dst + x + 12) + v_cvt_f32(v_reinterpret_as_s32(v_src11)));
            }
        }

        return x;
    }
};

template <>
struct AccSqr_SIMD<ushort, float>
{
    int operator() (const ushort * src, float * dst, const uchar * mask, int len, int cn) const
    {
        int x = 0;

        if (!mask)
        {
            len *= cn;
            for ( ; x <= len - 8; x += 8)
            {
                v_uint16x8 v_src = v_load(src + x);
                v_uint32x4 v_src0, v_src1;
                v_expand(v_src, v_src0, v_src1);

                v_float32x4 v_float0, v_float1;
                v_float0 = v_cvt_f32(v_reinterpret_as_s32(v_src0));
                v_float1 = v_cvt_f32(v_reinterpret_as_s32(v_src1));
                v_float0 = v_float0 * v_float0;
                v_float1 = v_float1 * v_float1;

                v_store(dst + x, v_load(dst + x) + v_float0);
                v_store(dst + x + 4, v_load(dst + x + 4) + v_float1);
            }
        }

        return x;
    }
};

#if CV_SIMD128_64F
template <>
struct AccSqr_SIMD<uchar, double>
{
    int operator() (const uchar * src, double * dst, const uchar * mask, int len, int cn) const
    {
        int x = 0;

        if (!mask)
        {
            len *= cn;
            for ( ; x <= len - 8; x += 8)
            {
                v_uint8x16 v_src = v_load(src + x);
                v_uint16x8 v_int, dummy;
                v_expand(v_src, v_int, dummy);

                v_uint32x4 v_int0, v_int1;
                v_expand(v_int, v_int0, v_int1);

                v_float64x2 v_src0 = v_cvt_f64(v_reinterpret_as_s32(v_int0));
                v_float64x2 v_src1 = v_cvt_f64_high(v_reinterpret_as_s32(v_int0));
                v_float64x2 v_src2 = v_cvt_f64(v_reinterpret_as_s32(v_int1));
                v_float64x2 v_src3 = v_cvt_f64_high(v_reinterpret_as_s32(v_int1));
                v_src0 = v_src0 * v_src0;
                v_src1 = v_src1 * v_src1;
                v_src2 = v_src2 * v_src2;
                v_src3 = v_src3 * v_src3;

                v_float64x2 v_dst0 = v_load(dst + x);
                v_float64x2 v_dst1 = v_load(dst + x + 2);
                v_float64x2 v_dst2 = v_load(dst + x + 4);
                v_float64x2 v_dst3 = v_load(dst + x + 6);

                v_dst0 += v_src0;
                v_dst1 += v_src1;
                v_dst2 += v_src2;
                v_dst3 += v_src3;

                v_store(dst + x, v_dst0);
                v_store(dst + x + 2, v_dst1);
                v_store(dst + x + 4, v_dst2);
                v_store(dst + x + 6, v_dst3);
            }
        }
        return x;
    }
};

template <>
struct AccSqr_SIMD<ushort, double>
{
    int operator() (const ushort * src, double * dst, const uchar * mask, int len, int cn) const
    {
        int x = 0;

        if (!mask)
        {
            len *= cn;
            for ( ; x <= len - 8; x += 8)
            {
                v_uint16x8 v_src  = v_load(src + x);
                v_uint32x4 v_int_0, v_int_1;
                v_expand(v_src, v_int_0, v_int_1);

                v_int32x4 v_int0 = v_reinterpret_as_s32(v_int_0);
                v_int32x4 v_int1 = v_reinterpret_as_s32(v_int_1);

                v_float64x2 v_src0 = v_cvt_f64(v_int0);
                v_float64x2 v_src1 = v_cvt_f64_high(v_int0);
                v_float64x2 v_src2 = v_cvt_f64(v_int1);
                v_float64x2 v_src3 = v_cvt_f64_high(v_int1);
                v_src0 = v_src0 * v_src0;
                v_src1 = v_src1 * v_src1;
                v_src2 = v_src2 * v_src2;
                v_src3 = v_src3 * v_src3;

                v_float64x2 v_dst0 = v_load(dst + x);
                v_float64x2 v_dst1 = v_load(dst + x + 2);
                v_float64x2 v_dst2 = v_load(dst + x + 4);
                v_float64x2 v_dst3 = v_load(dst + x + 6);

                v_dst0 += v_src0;
                v_dst1 += v_src1;
                v_dst2 += v_src2;
                v_dst3 += v_src3;

                v_store(dst + x, v_dst0);
                v_store(dst + x + 2, v_dst1);
                v_store(dst + x + 4, v_dst2);
                v_store(dst + x + 6, v_dst3);
            }
        }
        return x;
    }
};
#endif

template <>
struct AccProd_SIMD<uchar, float>
{
    int operator() (const uchar * src1, const uchar * src2, float * dst, const uchar * mask, int len, int cn) const
    {
        int x = 0;

        len *= cn;
        if (!mask)
        {
            for ( ; x <= len - 16; x += 16)
            {
                v_uint8x16 v_1src = v_load(src1 + x);
                v_uint8x16 v_2src = v_load(src2 + x);

                v_uint16x8 v_1src0, v_1src1, v_2src0, v_2src1;
                v_expand(v_1src, v_1src0, v_1src1);
                v_expand(v_2src, v_2src0, v_2src1);

                v_uint16x8 v_src0, v_src1;
                v_src0 = v_1src0 * v_2src0;
                v_src1 = v_1src1 * v_2src1;

                v_uint32x4 v_src00, v_src01, v_src10, v_src11;
                v_expand(v_src0, v_src00, v_src01);
                v_expand(v_src1, v_src10, v_src11);

                v_store(dst + x, v_load(dst + x) + v_cvt_f32(v_reinterpret_as_s32(v_src00)));
                v_store(dst + x + 4, v_load(dst + x + 4) + v_cvt_f32(v_reinterpret_as_s32(v_src01)));
                v_store(dst + x + 8, v_load(dst + x + 8) + v_cvt_f32(v_reinterpret_as_s32(v_src10)));
                v_store(dst + x + 12, v_load(dst + x + 12) + v_cvt_f32(v_reinterpret_as_s32(v_src11)));
            }
        }
        else if (cn == 1)
        {
            v_uint8x16 v_0 = v_setzero_u8();

            for ( ; x <= len - 16; x += 16)
            {
                v_uint8x16 v_mask = v_load(mask + x);
                v_mask = ~(v_0 == v_mask);

                v_uint8x16 v_1src = v_load(src1 + x) & v_mask;
                v_uint8x16 v_2src = v_load(src2 + x) & v_mask;

                v_uint16x8 v_1src0, v_1src1, v_2src0, v_2src1;
                v_expand(v_1src, v_1src0, v_1src1);
                v_expand(v_2src, v_2src0, v_2src1);

                v_uint16x8 v_src0, v_src1;
                v_src0 = v_1src0 * v_2src0;
                v_src1 = v_1src1 * v_2src1;

                v_uint32x4 v_src00, v_src01, v_src10, v_src11;
                v_expand(v_src0, v_src00, v_src01);
                v_expand(v_src1, v_src10, v_src11);

                v_store(dst + x, v_load(dst + x) + v_cvt_f32(v_reinterpret_as_s32(v_src00)));
                v_store(dst + x + 4, v_load(dst + x + 4) + v_cvt_f32(v_reinterpret_as_s32(v_src01)));
                v_store(dst + x + 8, v_load(dst + x + 8) + v_cvt_f32(v_reinterpret_as_s32(v_src10)));
                v_store(dst + x + 12, v_load(dst + x + 12) + v_cvt_f32(v_reinterpret_as_s32(v_src11)));
            }
        }

        return x;
    }
};

template <>
struct AccProd_SIMD<ushort, float>
{
    int operator() (const ushort * src1, const ushort * src2, float * dst, const uchar * mask, int len, int cn) const
    {
        int x = 0;

        if (!mask)
        {
            len *= cn;
            for ( ; x <= len - 8; x += 8)
            {
                v_uint16x8 v_1src = v_load(src1 + x);
                v_uint16x8 v_2src = v_load(src2 + x);

                v_uint32x4 v_1src0, v_1src1, v_2src0, v_2src1;
                v_expand(v_1src, v_1src0, v_1src1);
                v_expand(v_2src, v_2src0, v_2src1);

                v_float32x4 v_1float0 = v_cvt_f32(v_reinterpret_as_s32(v_1src0));
                v_float32x4 v_1float1 = v_cvt_f32(v_reinterpret_as_s32(v_1src1));
                v_float32x4 v_2float0 = v_cvt_f32(v_reinterpret_as_s32(v_2src0));
                v_float32x4 v_2float1 = v_cvt_f32(v_reinterpret_as_s32(v_2src1));

                v_float32x4 v_src0 = v_1float0 * v_2float0;
                v_float32x4 v_src1 = v_1float1 * v_2float1;

                v_store(dst + x, v_load(dst + x) + v_src0);
                v_store(dst + x + 4, v_load(dst + x + 4) + v_src1);
            }
        }
        else if (cn == 1)
        {
            v_uint16x8 v_0 = v_setzero_u16();

            for ( ; x <= len - 8; x += 8)
            {
                v_uint8x16 v_mask = v_load_halves(mask + x, mask + x);
                v_uint16x8 v_mask0, v_mask1;
                v_expand(v_mask, v_mask0, v_mask1);
                v_mask0 = ~(v_0 == v_mask0);

                v_uint16x8 v_1src = v_load(src1 + x) & v_mask0;
                v_uint16x8 v_2src = v_load(src2 + x) & v_mask0;

                v_uint32x4 v_1src0, v_1src1, v_2src0, v_2src1;
                v_expand(v_1src, v_1src0, v_1src1);
                v_expand(v_2src, v_2src0, v_2src1);

                v_float32x4 v_1float0 = v_cvt_f32(v_reinterpret_as_s32(v_1src0));
                v_float32x4 v_1float1 = v_cvt_f32(v_reinterpret_as_s32(v_1src1));
                v_float32x4 v_2float0 = v_cvt_f32(v_reinterpret_as_s32(v_2src0));
                v_float32x4 v_2float1 = v_cvt_f32(v_reinterpret_as_s32(v_2src1));

                v_float32x4 v_src0 = v_1float0 * v_2float0;
                v_float32x4 v_src1 = v_1float1 * v_2float1;

                v_store(dst + x, v_load(dst + x) + v_src0);
                v_store(dst + x + 4, v_load(dst + x + 4) + v_src1);
            }
        }

        return x;
    }
};

#if CV_SIMD128_64F
template <>
struct AccProd_SIMD<uchar, double>
{
    int operator() (const uchar * src1, const uchar * src2, double * dst, const uchar * mask, int len, int cn) const
    {
        int x = 0;

        if (!mask)
        {
            len *= cn;
            for ( ; x <= len - 8; x += 8)
            {
                v_uint8x16 v_1src  = v_load(src1 + x);
                v_uint8x16 v_2src  = v_load(src2 + x);

                v_uint16x8 v_1int, v_2int, dummy;
                v_expand(v_1src, v_1int, dummy);
                v_expand(v_2src, v_2int, dummy);

                v_uint32x4 v_1int_0, v_1int_1, v_2int_0, v_2int_1;
                v_expand(v_1int, v_1int_0, v_1int_1);
                v_expand(v_2int, v_2int_0, v_2int_1);

                v_int32x4 v_1int0 = v_reinterpret_as_s32(v_1int_0);
                v_int32x4 v_1int1 = v_reinterpret_as_s32(v_1int_1);
                v_int32x4 v_2int0 = v_reinterpret_as_s32(v_2int_0);
                v_int32x4 v_2int1 = v_reinterpret_as_s32(v_2int_1);

                v_float64x2 v_src0 = v_cvt_f64(v_1int0) * v_cvt_f64(v_2int0);
                v_float64x2 v_src1 = v_cvt_f64_high(v_1int0) * v_cvt_f64_high(v_2int0);
                v_float64x2 v_src2 = v_cvt_f64(v_1int1) * v_cvt_f64(v_2int1);
                v_float64x2 v_src3 = v_cvt_f64_high(v_1int1) * v_cvt_f64_high(v_2int1);

                v_float64x2 v_dst0 = v_load(dst + x);
                v_float64x2 v_dst1 = v_load(dst + x + 2);
                v_float64x2 v_dst2 = v_load(dst + x + 4);
                v_float64x2 v_dst3 = v_load(dst + x + 6);

                v_dst0 += v_src0;
                v_dst1 += v_src1;
                v_dst2 += v_src2;
                v_dst3 += v_src3;

                v_store(dst + x, v_dst0);
                v_store(dst + x + 2, v_dst1);
                v_store(dst + x + 4, v_dst2);
                v_store(dst + x + 6, v_dst3);
            }
        }
        return x;
    }
};

template <>
struct AccProd_SIMD<ushort, double>
{
    int operator() (const ushort * src1, const ushort * src2, double * dst, const uchar * mask, int len, int cn) const
    {
        int x = 0;

        if (!mask)
        {
            len *= cn;
            for ( ; x <= len - 8; x += 8)
            {
                v_uint16x8 v_1src  = v_load(src1 + x);
                v_uint16x8 v_2src  = v_load(src2 + x);

                v_uint32x4 v_1int_0, v_1int_1, v_2int_0, v_2int_1;
                v_expand(v_1src, v_1int_0, v_1int_1);
                v_expand(v_2src, v_2int_0, v_2int_1);

                v_int32x4 v_1int0 = v_reinterpret_as_s32(v_1int_0);
                v_int32x4 v_1int1 = v_reinterpret_as_s32(v_1int_1);
                v_int32x4 v_2int0 = v_reinterpret_as_s32(v_2int_0);
                v_int32x4 v_2int1 = v_reinterpret_as_s32(v_2int_1);

                v_float64x2 v_src0 = v_cvt_f64(v_1int0) * v_cvt_f64(v_2int0);
                v_float64x2 v_src1 = v_cvt_f64_high(v_1int0) * v_cvt_f64_high(v_2int0);
                v_float64x2 v_src2 = v_cvt_f64(v_1int1) * v_cvt_f64(v_2int1);
                v_float64x2 v_src3 = v_cvt_f64_high(v_1int1) * v_cvt_f64_high(v_2int1);

                v_float64x2 v_dst0 = v_load(dst + x);
                v_float64x2 v_dst1 = v_load(dst + x + 2);
                v_float64x2 v_dst2 = v_load(dst + x + 4);
                v_float64x2 v_dst3 = v_load(dst + x + 6);

                v_dst0 = v_dst0 + v_src0;
                v_dst1 = v_dst1 + v_src1;
                v_dst2 = v_dst2 + v_src2;
                v_dst3 = v_dst3 + v_src3;

                v_store(dst + x, v_dst0);
                v_store(dst + x + 2, v_dst1);
                v_store(dst + x + 4, v_dst2);
                v_store(dst + x + 6, v_dst3);
            }
        }
        return x;
    }
};
#endif

template <>
struct AccW_SIMD<uchar, float>
{
    int operator() (const uchar * src, float * dst, const uchar * mask, int len, int cn, float alpha) const
    {
        int x = 0;
        v_float32x4 v_alpha = v_setall_f32(alpha);
        v_float32x4 v_beta = v_setall_f32(1.0f - alpha);

        if (!mask)
        {
            len *= cn;
            for ( ; x <= len - 16; x += 16)
            {
                v_uint8x16 v_src = v_load(src + x);

                v_uint16x8 v_src0, v_src1;
                v_expand(v_src, v_src0, v_src1);

                v_uint32x4 v_src00, v_src01, v_src10, v_src11;
                v_expand(v_src0, v_src00, v_src01);
                v_expand(v_src1, v_src10, v_src11);

                v_float32x4 v_dst00 = v_load(dst + x);
                v_float32x4 v_dst01 = v_load(dst + x + 4);
                v_float32x4 v_dst10 = v_load(dst + x + 8);
                v_float32x4 v_dst11 = v_load(dst + x + 12);

                v_dst00 = (v_dst00 * v_beta) + (v_cvt_f32(v_reinterpret_as_s32(v_src00)) * v_alpha);
                v_dst01 = (v_dst01 * v_beta) + (v_cvt_f32(v_reinterpret_as_s32(v_src01)) * v_alpha);
                v_dst10 = (v_dst10 * v_beta) + (v_cvt_f32(v_reinterpret_as_s32(v_src10)) * v_alpha);
                v_dst11 = (v_dst11 * v_beta) + (v_cvt_f32(v_reinterpret_as_s32(v_src11)) * v_alpha);

                v_store(dst + x, v_dst00);
                v_store(dst + x + 4, v_dst01);
                v_store(dst + x + 8, v_dst10);
                v_store(dst + x + 12, v_dst11);
            }
        }

        return x;
    }
};

template <>
struct AccW_SIMD<ushort, float>
{
    int operator() (const ushort * src, float * dst, const uchar * mask, int len, int cn, float alpha) const
    {
        int x = 0;
        v_float32x4 v_alpha = v_setall_f32(alpha);
        v_float32x4 v_beta = v_setall_f32(1.0f - alpha);

        if (!mask)
        {
            len *= cn;
            for ( ; x <= len - 8; x += 8)
            {
                v_uint16x8 v_src = v_load(src + x);
                v_uint32x4 v_int0, v_int1;
                v_expand(v_src, v_int0, v_int1);

                v_float32x4 v_src0 = v_cvt_f32(v_reinterpret_as_s32(v_int0));
                v_float32x4 v_src1 = v_cvt_f32(v_reinterpret_as_s32(v_int1));
                v_src0 = v_src0 * v_alpha;
                v_src1 = v_src1 * v_alpha;

                v_float32x4 v_dst0 = v_load(dst + x) * v_beta;
                v_float32x4 v_dst1 = v_load(dst + x + 4) * v_beta;

                v_store(dst + x, v_dst0 + v_src0);
                v_store(dst + x + 4, v_dst1 + v_src1);
            }
        }

        return x;
    }
};

#if CV_SIMD128_64F
template <>
struct AccW_SIMD<uchar, double>
{
    int operator() (const uchar * src, double * dst, const uchar * mask, int len, int cn, double alpha) const
    {
        int x = 0;
        v_float64x2 v_alpha = v_setall_f64(alpha);
        v_float64x2 v_beta = v_setall_f64(1.0f - alpha);

        if (!mask)
        {
            len *= cn;
            for ( ; x <= len - 8; x += 8)
            {
                v_uint8x16 v_src = v_load(src + x);
                v_uint16x8 v_int, dummy;
                v_expand(v_src, v_int, dummy);

                v_uint32x4 v_int_0, v_int_1;
                v_expand(v_int, v_int_0, v_int_1);

                v_int32x4 v_int0 = v_reinterpret_as_s32(v_int_0);
                v_int32x4 v_int1 = v_reinterpret_as_s32(v_int_1);

                v_float64x2 v_src0 = v_cvt_f64(v_int0);
                v_float64x2 v_src1 = v_cvt_f64_high(v_int0);
                v_float64x2 v_src2 = v_cvt_f64(v_int1);
                v_float64x2 v_src3 = v_cvt_f64_high(v_int1);

                v_float64x2 v_dst0 = v_load(dst + x);
                v_float64x2 v_dst1 = v_load(dst + x + 2);
                v_float64x2 v_dst2 = v_load(dst + x + 4);
                v_float64x2 v_dst3 = v_load(dst + x + 6);

                v_dst0 = (v_dst0 * v_beta) + (v_src0 * v_alpha);
                v_dst1 = (v_dst1 * v_beta) + (v_src1 * v_alpha);
                v_dst2 = (v_dst2 * v_beta) + (v_src2 * v_alpha);
                v_dst3 = (v_dst3 * v_beta) + (v_src3 * v_alpha);

                v_store(dst + x, v_dst0);
                v_store(dst + x + 2, v_dst1);
                v_store(dst + x + 4, v_dst2);
                v_store(dst + x + 6, v_dst3);
            }
        }

        return x;
    }
};

template <>
struct AccW_SIMD<ushort, double>
{
    int operator() (const ushort * src, double * dst, const uchar * mask, int len, int cn, double alpha) const
    {
        int x = 0;
        v_float64x2 v_alpha = v_setall_f64(alpha);
        v_float64x2 v_beta = v_setall_f64(1.0f - alpha);

        if (!mask)
        {
            len *= cn;
            for ( ; x <= len - 8; x += 8)
            {
                v_uint16x8 v_src = v_load(src + x);
                v_uint32x4 v_int_0, v_int_1;
                v_expand(v_src, v_int_0, v_int_1);

                v_int32x4 v_int0 = v_reinterpret_as_s32(v_int_0);
                v_int32x4 v_int1 = v_reinterpret_as_s32(v_int_1);

                v_float64x2 v_src00 = v_cvt_f64(v_int0);
                v_float64x2 v_src01 = v_cvt_f64_high(v_int0);
                v_float64x2 v_src10 = v_cvt_f64(v_int1);
                v_float64x2 v_src11 = v_cvt_f64_high(v_int1);

                v_float64x2 v_dst00 = v_load(dst + x);
                v_float64x2 v_dst01 = v_load(dst + x + 2);
                v_float64x2 v_dst10 = v_load(dst + x + 4);
                v_float64x2 v_dst11 = v_load(dst + x + 6);

                v_dst00 = (v_dst00 * v_beta) + (v_src00 * v_alpha);
                v_dst01 = (v_dst01 * v_beta) + (v_src01 * v_alpha);
                v_dst10 = (v_dst10 * v_beta) + (v_src10 * v_alpha);
                v_dst11 = (v_dst11 * v_beta) + (v_src11 * v_alpha);

                v_store(dst + x, v_dst00);
                v_store(dst + x + 2, v_dst01);
                v_store(dst + x + 4, v_dst10);
                v_store(dst + x + 6, v_dst11);
            }
        }

        return x;
    }
};
#endif //CV_SIMD128_64F
#endif //CV_SIMD128

template<typename T, typename AT> void
acc_( const T* src, AT* dst, const uchar* mask, int len, int cn )
{
    int i = Acc_SIMD<T, AT>()(src, dst, mask, len, cn);

    if( !mask )
    {
        len *= cn;
        #if CV_ENABLE_UNROLLED
        for( ; i <= len - 4; i += 4 )
        {
            AT t0, t1;
            t0 = src[i] + dst[i];
            t1 = src[i+1] + dst[i+1];
            dst[i] = t0; dst[i+1] = t1;

            t0 = src[i+2] + dst[i+2];
            t1 = src[i+3] + dst[i+3];
            dst[i+2] = t0; dst[i+3] = t1;
        }
        #endif
        for( ; i < len; i++ )
            dst[i] += src[i];
    }
    else if( cn == 1 )
    {
        for( ; i < len; i++ )
        {
            if( mask[i] )
                dst[i] += src[i];
        }
    }
    else if( cn == 3 )
    {
        for( ; i < len; i++, src += 3, dst += 3 )
        {
            if( mask[i] )
            {
                AT t0 = src[0] + dst[0];
                AT t1 = src[1] + dst[1];
                AT t2 = src[2] + dst[2];

                dst[0] = t0; dst[1] = t1; dst[2] = t2;
            }
        }
    }
    else
    {
        for( ; i < len; i++, src += cn, dst += cn )
            if( mask[i] )
            {
                for( int k = 0; k < cn; k++ )
                    dst[k] += src[k];
            }
    }
}


template<typename T, typename AT> void
accSqr_( const T* src, AT* dst, const uchar* mask, int len, int cn )
{
    int i = AccSqr_SIMD<T, AT>()(src, dst, mask, len, cn);

    if( !mask )
    {
        len *= cn;
         #if CV_ENABLE_UNROLLED
        for( ; i <= len - 4; i += 4 )
        {
            AT t0, t1;
            t0 = (AT)src[i]*src[i] + dst[i];
            t1 = (AT)src[i+1]*src[i+1] + dst[i+1];
            dst[i] = t0; dst[i+1] = t1;

            t0 = (AT)src[i+2]*src[i+2] + dst[i+2];
            t1 = (AT)src[i+3]*src[i+3] + dst[i+3];
            dst[i+2] = t0; dst[i+3] = t1;
        }
        #endif
        for( ; i < len; i++ )
            dst[i] += (AT)src[i]*src[i];
    }
    else if( cn == 1 )
    {
        for( ; i < len; i++ )
        {
            if( mask[i] )
                dst[i] += (AT)src[i]*src[i];
        }
    }
    else if( cn == 3 )
    {
        for( ; i < len; i++, src += 3, dst += 3 )
        {
            if( mask[i] )
            {
                AT t0 = (AT)src[0]*src[0] + dst[0];
                AT t1 = (AT)src[1]*src[1] + dst[1];
                AT t2 = (AT)src[2]*src[2] + dst[2];

                dst[0] = t0; dst[1] = t1; dst[2] = t2;
            }
        }
    }
    else
    {
        for( ; i < len; i++, src += cn, dst += cn )
            if( mask[i] )
            {
                for( int k = 0; k < cn; k++ )
                    dst[k] += (AT)src[k]*src[k];
            }
    }
}


template<typename T, typename AT> void
accProd_( const T* src1, const T* src2, AT* dst, const uchar* mask, int len, int cn )
{
    int i = AccProd_SIMD<T, AT>()(src1, src2, dst, mask, len, cn);

    if( !mask )
    {
        len *= cn;
        #if CV_ENABLE_UNROLLED
        for( ; i <= len - 4; i += 4 )
        {
            AT t0, t1;
            t0 = (AT)src1[i]*src2[i] + dst[i];
            t1 = (AT)src1[i+1]*src2[i+1] + dst[i+1];
            dst[i] = t0; dst[i+1] = t1;

            t0 = (AT)src1[i+2]*src2[i+2] + dst[i+2];
            t1 = (AT)src1[i+3]*src2[i+3] + dst[i+3];
            dst[i+2] = t0; dst[i+3] = t1;
        }
        #endif
        for( ; i < len; i++ )
            dst[i] += (AT)src1[i]*src2[i];
    }
    else if( cn == 1 )
    {
        for( ; i < len; i++ )
        {
            if( mask[i] )
                dst[i] += (AT)src1[i]*src2[i];
        }
    }
    else if( cn == 3 )
    {
        for( ; i < len; i++, src1 += 3, src2 += 3, dst += 3 )
        {
            if( mask[i] )
            {
                AT t0 = (AT)src1[0]*src2[0] + dst[0];
                AT t1 = (AT)src1[1]*src2[1] + dst[1];
                AT t2 = (AT)src1[2]*src2[2] + dst[2];

                dst[0] = t0; dst[1] = t1; dst[2] = t2;
            }
        }
    }
    else
    {
        for( ; i < len; i++, src1 += cn, src2 += cn, dst += cn )
            if( mask[i] )
            {
                for( int k = 0; k < cn; k++ )
                    dst[k] += (AT)src1[k]*src2[k];
            }
    }
}


template<typename T, typename AT> void
accW_( const T* src, AT* dst, const uchar* mask, int len, int cn, double alpha )
{
    AT a = (AT)alpha, b = 1 - a;
    int i = AccW_SIMD<T, AT>()(src, dst, mask, len, cn, a);

    if( !mask )
    {
        len *= cn;
        #if CV_ENABLE_UNROLLED
        for( ; i <= len - 4; i += 4 )
        {
            AT t0, t1;
            t0 = src[i]*a + dst[i]*b;
            t1 = src[i+1]*a + dst[i+1]*b;
            dst[i] = t0; dst[i+1] = t1;

            t0 = src[i+2]*a + dst[i+2]*b;
            t1 = src[i+3]*a + dst[i+3]*b;
            dst[i+2] = t0; dst[i+3] = t1;
        }
        #endif
        for( ; i < len; i++ )
            dst[i] = src[i]*a + dst[i]*b;
    }
    else if( cn == 1 )
    {
        for( ; i < len; i++ )
        {
            if( mask[i] )
                dst[i] = src[i]*a + dst[i]*b;
        }
    }
    else if( cn == 3 )
    {
        for( ; i < len; i++, src += 3, dst += 3 )
        {
            if( mask[i] )
            {
                AT t0 = src[0]*a + dst[0]*b;
                AT t1 = src[1]*a + dst[1]*b;
                AT t2 = src[2]*a + dst[2]*b;

                dst[0] = t0; dst[1] = t1; dst[2] = t2;
            }
        }
    }
    else
    {
        for( ; i < len; i++, src += cn, dst += cn )
            if( mask[i] )
            {
                for( int k = 0; k < cn; k++ )
                    dst[k] = src[k]*a + dst[k]*b;
            }
    }
}


#define DEF_ACC_FUNCS(suffix, type, acctype) \
static void acc_##suffix(const type* src, acctype* dst, \
                         const uchar* mask, int len, int cn) \
{ acc_(src, dst, mask, len, cn); } \
\
static void accSqr_##suffix(const type* src, acctype* dst, \
                            const uchar* mask, int len, int cn) \
{ accSqr_(src, dst, mask, len, cn); } \
\
static void accProd_##suffix(const type* src1, const type* src2, \
                             acctype* dst, const uchar* mask, int len, int cn) \
{ accProd_(src1, src2, dst, mask, len, cn); } \
\
static void accW_##suffix(const type* src, acctype* dst, \
                          const uchar* mask, int len, int cn, double alpha) \
{ accW_(src, dst, mask, len, cn, alpha); }


DEF_ACC_FUNCS(8u32f, uchar, float)
DEF_ACC_FUNCS(8u64f, uchar, double)
DEF_ACC_FUNCS(16u32f, ushort, float)
DEF_ACC_FUNCS(16u64f, ushort, double)
DEF_ACC_FUNCS(32f, float, float)
DEF_ACC_FUNCS(32f64f, float, double)
DEF_ACC_FUNCS(64f, double, double)


typedef void (*AccFunc)(const uchar*, uchar*, const uchar*, int, int);
typedef void (*AccProdFunc)(const uchar*, const uchar*, uchar*, const uchar*, int, int);
typedef void (*AccWFunc)(const uchar*, uchar*, const uchar*, int, int, double);

static AccFunc accTab[] =
{
    (AccFunc)acc_8u32f, (AccFunc)acc_8u64f,
    (AccFunc)acc_16u32f, (AccFunc)acc_16u64f,
    (AccFunc)acc_32f, (AccFunc)acc_32f64f,
    (AccFunc)acc_64f
};

static AccFunc accSqrTab[] =
{
    (AccFunc)accSqr_8u32f, (AccFunc)accSqr_8u64f,
    (AccFunc)accSqr_16u32f, (AccFunc)accSqr_16u64f,
    (AccFunc)accSqr_32f, (AccFunc)accSqr_32f64f,
    (AccFunc)accSqr_64f
};

static AccProdFunc accProdTab[] =
{
    (AccProdFunc)accProd_8u32f, (AccProdFunc)accProd_8u64f,
    (AccProdFunc)accProd_16u32f, (AccProdFunc)accProd_16u64f,
    (AccProdFunc)accProd_32f, (AccProdFunc)accProd_32f64f,
    (AccProdFunc)accProd_64f
};

static AccWFunc accWTab[] =
{
    (AccWFunc)accW_8u32f, (AccWFunc)accW_8u64f,
    (AccWFunc)accW_16u32f, (AccWFunc)accW_16u64f,
    (AccWFunc)accW_32f, (AccWFunc)accW_32f64f,
    (AccWFunc)accW_64f
};

inline int getAccTabIdx(int sdepth, int ddepth)
{
    return sdepth == CV_8U && ddepth == CV_32F ? 0 :
           sdepth == CV_8U && ddepth == CV_64F ? 1 :
           sdepth == CV_16U && ddepth == CV_32F ? 2 :
           sdepth == CV_16U && ddepth == CV_64F ? 3 :
           sdepth == CV_32F && ddepth == CV_32F ? 4 :
           sdepth == CV_32F && ddepth == CV_64F ? 5 :
           sdepth == CV_64F && ddepth == CV_64F ? 6 : -1;
}

#ifdef HAVE_OPENCL

enum
{
    ACCUMULATE = 0,
    ACCUMULATE_SQUARE = 1,
    ACCUMULATE_PRODUCT = 2,
    ACCUMULATE_WEIGHTED = 3
};

static bool ocl_accumulate( InputArray _src, InputArray _src2, InputOutputArray _dst, double alpha,
                            InputArray _mask, int op_type )
{
    CV_Assert(op_type == ACCUMULATE || op_type == ACCUMULATE_SQUARE ||
              op_type == ACCUMULATE_PRODUCT || op_type == ACCUMULATE_WEIGHTED);

    const ocl::Device & dev = ocl::Device::getDefault();
    bool haveMask = !_mask.empty(), doubleSupport = dev.doubleFPConfig() > 0;
    int stype = _src.type(), sdepth = CV_MAT_DEPTH(stype), cn = CV_MAT_CN(stype), ddepth = _dst.depth();
    int kercn = haveMask ? cn : ocl::predictOptimalVectorWidthMax(_src, _src2, _dst), rowsPerWI = dev.isIntel() ? 4 : 1;

    if (!doubleSupport && (sdepth == CV_64F || ddepth == CV_64F))
        return false;

    const char * const opMap[4] = { "ACCUMULATE", "ACCUMULATE_SQUARE", "ACCUMULATE_PRODUCT",
                                   "ACCUMULATE_WEIGHTED" };

    char cvt[40];
    ocl::Kernel k("accumulate", ocl::imgproc::accumulate_oclsrc,
                  format("-D %s%s -D srcT1=%s -D cn=%d -D dstT1=%s%s -D rowsPerWI=%d -D convertToDT=%s",
                         opMap[op_type], haveMask ? " -D HAVE_MASK" : "",
                         ocl::typeToStr(sdepth), kercn, ocl::typeToStr(ddepth),
                         doubleSupport ? " -D DOUBLE_SUPPORT" : "", rowsPerWI,
                         ocl::convertTypeStr(sdepth, ddepth, 1, cvt)));
    if (k.empty())
        return false;

    UMat src = _src.getUMat(), src2 = _src2.getUMat(), dst = _dst.getUMat(), mask = _mask.getUMat();

    ocl::KernelArg srcarg = ocl::KernelArg::ReadOnlyNoSize(src),
            src2arg = ocl::KernelArg::ReadOnlyNoSize(src2),
            dstarg = ocl::KernelArg::ReadWrite(dst, cn, kercn),
            maskarg = ocl::KernelArg::ReadOnlyNoSize(mask);

    int argidx = k.set(0, srcarg);
    if (op_type == ACCUMULATE_PRODUCT)
        argidx = k.set(argidx, src2arg);
    argidx = k.set(argidx, dstarg);
    if (op_type == ACCUMULATE_WEIGHTED)
    {
        if (ddepth == CV_32F)
            argidx = k.set(argidx, (float)alpha);
        else
            argidx = k.set(argidx, alpha);
    }
    if (haveMask)
        k.set(argidx, maskarg);

    size_t globalsize[2] = { (size_t)src.cols * cn / kercn, ((size_t)src.rows + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, NULL, false);
}

#endif

}

#if defined(HAVE_IPP)
namespace cv
{
static bool ipp_accumulate(InputArray _src, InputOutputArray _dst, InputArray _mask)
{
    CV_INSTRUMENT_REGION_IPP()

    int stype = _src.type(), sdepth = CV_MAT_DEPTH(stype), scn = CV_MAT_CN(stype);
    int dtype = _dst.type(), ddepth = CV_MAT_DEPTH(dtype);

    Mat src = _src.getMat(), dst = _dst.getMat(), mask = _mask.getMat();

    if (src.dims <= 2 || (src.isContinuous() && dst.isContinuous() && (mask.empty() || mask.isContinuous())))
    {
        typedef IppStatus (CV_STDCALL * IppiAdd)(const void * pSrc, int srcStep, Ipp32f * pSrcDst, int srcdstStep, IppiSize roiSize);
        typedef IppStatus (CV_STDCALL * IppiAddMask)(const void * pSrc, int srcStep, const Ipp8u * pMask, int maskStep, Ipp32f * pSrcDst,
                                                    int srcDstStep, IppiSize roiSize);
        IppiAdd ippiAdd_I = 0;
        IppiAddMask ippiAdd_IM = 0;

        if (mask.empty())
        {
            CV_SUPPRESS_DEPRECATED_START
            ippiAdd_I = sdepth == CV_8U && ddepth == CV_32F ? (IppiAdd)ippiAdd_8u32f_C1IR :
                sdepth == CV_16U && ddepth == CV_32F ? (IppiAdd)ippiAdd_16u32f_C1IR :
                sdepth == CV_32F && ddepth == CV_32F ? (IppiAdd)ippiAdd_32f_C1IR : 0;
            CV_SUPPRESS_DEPRECATED_END
        }
        else if (scn == 1)
        {
            ippiAdd_IM = sdepth == CV_8U && ddepth == CV_32F ? (IppiAddMask)ippiAdd_8u32f_C1IMR :
                sdepth == CV_16U && ddepth == CV_32F ? (IppiAddMask)ippiAdd_16u32f_C1IMR :
                sdepth == CV_32F && ddepth == CV_32F ? (IppiAddMask)ippiAdd_32f_C1IMR : 0;
        }

        if (ippiAdd_I || ippiAdd_IM)
        {
            IppStatus status = ippStsErr;

            Size size = src.size();
            int srcstep = (int)src.step, dststep = (int)dst.step, maskstep = (int)mask.step;
            if (src.isContinuous() && dst.isContinuous() && mask.isContinuous())
            {
                srcstep = static_cast<int>(src.total() * src.elemSize());
                dststep = static_cast<int>(dst.total() * dst.elemSize());
                maskstep = static_cast<int>(mask.total() * mask.elemSize());
                size.width = static_cast<int>(src.total());
                size.height = 1;
            }
            size.width *= scn;

            if (ippiAdd_I)
                status = CV_INSTRUMENT_FUN_IPP(ippiAdd_I, src.ptr(), srcstep, dst.ptr<Ipp32f>(), dststep, ippiSize(size.width, size.height));
            else if (ippiAdd_IM)
                status = CV_INSTRUMENT_FUN_IPP(ippiAdd_IM, src.ptr(), srcstep, mask.ptr<Ipp8u>(), maskstep,
                    dst.ptr<Ipp32f>(), dststep, ippiSize(size.width, size.height));

            if (status >= 0)
                return true;
        }
    }
    return false;
}
}
#endif

#ifdef HAVE_OPENVX
namespace cv
{
enum
{
    VX_ACCUMULATE_OP = 0,
    VX_ACCUMULATE_SQUARE_OP = 1,
    VX_ACCUMULATE_WEIGHTED_OP = 2
};

static bool openvx_accumulate(InputArray _src, InputOutputArray _dst, InputArray _mask, double _weight, int opType)
{
    Mat srcMat = _src.getMat(), dstMat = _dst.getMat();
    if(!_mask.empty() ||
       (opType == VX_ACCUMULATE_WEIGHTED_OP && dstMat.type() != CV_8UC1  ) ||
       (opType != VX_ACCUMULATE_WEIGHTED_OP && dstMat.type() != CV_16SC1 ) ||
       srcMat.type() != CV_8UC1)
    {
        return false;
    }
    //TODO: handle different number of channels (channel extract && channel combine)
    //TODO: handle mask (threshold mask to 0xff && bitwise AND with src)
    //(both things can be done by creating a graph)

    try
    {
        ivx::Context context = ovx::getOpenVXContext();
        ivx::Image srcImage = ivx::Image::createFromHandle(context, ivx::Image::matTypeToFormat(srcMat.type()),
                                                           ivx::Image::createAddressing(srcMat), srcMat.data);
        ivx::Image dstImage = ivx::Image::createFromHandle(context, ivx::Image::matTypeToFormat(dstMat.type()),
                                                           ivx::Image::createAddressing(dstMat), dstMat.data);
        ivx::Scalar shift = ivx::Scalar::create<VX_TYPE_UINT32>(context, 0);
        ivx::Scalar alpha = ivx::Scalar::create<VX_TYPE_FLOAT32>(context, _weight);

        switch (opType)
        {
        case VX_ACCUMULATE_OP:
            ivx::IVX_CHECK_STATUS(vxuAccumulateImage(context, srcImage, dstImage));
            break;
        case VX_ACCUMULATE_SQUARE_OP:
            ivx::IVX_CHECK_STATUS(vxuAccumulateSquareImage(context, srcImage, shift, dstImage));
            break;
        case VX_ACCUMULATE_WEIGHTED_OP:
            ivx::IVX_CHECK_STATUS(vxuAccumulateWeightedImage(context, srcImage, alpha, dstImage));
            break;
        default:
            break;
        }

#ifdef VX_VERSION_1_1
        //we should take user memory back before release
        //(it's not done automatically according to standard)
        srcImage.swapHandle(); dstImage.swapHandle();
#endif
    }
    catch (ivx::RuntimeError & e)
    {
        VX_DbgThrow(e.what());
    }
    catch (ivx::WrapperError & e)
    {
        VX_DbgThrow(e.what());
    }

    return true;
}
}
#endif

void cv::accumulate( InputArray _src, InputOutputArray _dst, InputArray _mask )
{
    CV_INSTRUMENT_REGION()

    int stype = _src.type(), sdepth = CV_MAT_DEPTH(stype), scn = CV_MAT_CN(stype);
    int dtype = _dst.type(), ddepth = CV_MAT_DEPTH(dtype), dcn = CV_MAT_CN(dtype);

    CV_Assert( _src.sameSize(_dst) && dcn == scn );
    CV_Assert( _mask.empty() || (_src.sameSize(_mask) && _mask.type() == CV_8U) );

    CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat(),
               ocl_accumulate(_src, noArray(), _dst, 0.0, _mask, ACCUMULATE))

    CV_IPP_RUN((_src.dims() <= 2 || (_src.isContinuous() && _dst.isContinuous() && (_mask.empty() || _mask.isContinuous()))),
        ipp_accumulate(_src, _dst, _mask));

    CV_OVX_RUN(_src.dims() <= 2,
               openvx_accumulate(_src, _dst, _mask, 0.0, VX_ACCUMULATE_OP))

    Mat src = _src.getMat(), dst = _dst.getMat(), mask = _mask.getMat();


    int fidx = getAccTabIdx(sdepth, ddepth);
    AccFunc func = fidx >= 0 ? accTab[fidx] : 0;
    CV_Assert( func != 0 );

    const Mat* arrays[] = {&src, &dst, &mask, 0};
    uchar* ptrs[3];
    NAryMatIterator it(arrays, ptrs);
    int len = (int)it.size;

    for( size_t i = 0; i < it.nplanes; i++, ++it )
        func(ptrs[0], ptrs[1], ptrs[2], len, scn);
}

#if defined(HAVE_IPP)
namespace cv
{
static bool ipp_accumulate_square(InputArray _src, InputOutputArray _dst, InputArray _mask)
{
    CV_INSTRUMENT_REGION_IPP()

    int stype = _src.type(), sdepth = CV_MAT_DEPTH(stype), scn = CV_MAT_CN(stype);
    int dtype = _dst.type(), ddepth = CV_MAT_DEPTH(dtype);

    Mat src = _src.getMat(), dst = _dst.getMat(), mask = _mask.getMat();

    if (src.dims <= 2 || (src.isContinuous() && dst.isContinuous() && (mask.empty() || mask.isContinuous())))
    {
        typedef IppStatus (CV_STDCALL * ippiAddSquare)(const void * pSrc, int srcStep, Ipp32f * pSrcDst, int srcdstStep, IppiSize roiSize);
        typedef IppStatus (CV_STDCALL * ippiAddSquareMask)(const void * pSrc, int srcStep, const Ipp8u * pMask, int maskStep, Ipp32f * pSrcDst,
                                                            int srcDstStep, IppiSize roiSize);
        ippiAddSquare ippiAddSquare_I = 0;
        ippiAddSquareMask ippiAddSquare_IM = 0;

        if (mask.empty())
        {
            ippiAddSquare_I = sdepth == CV_8U && ddepth == CV_32F ? (ippiAddSquare)ippiAddSquare_8u32f_C1IR :
                sdepth == CV_16U && ddepth == CV_32F ? (ippiAddSquare)ippiAddSquare_16u32f_C1IR :
                sdepth == CV_32F && ddepth == CV_32F ? (ippiAddSquare)ippiAddSquare_32f_C1IR : 0;
        }
        else if (scn == 1)
        {
            ippiAddSquare_IM = sdepth == CV_8U && ddepth == CV_32F ? (ippiAddSquareMask)ippiAddSquare_8u32f_C1IMR :
                sdepth == CV_16U && ddepth == CV_32F ? (ippiAddSquareMask)ippiAddSquare_16u32f_C1IMR :
                sdepth == CV_32F && ddepth == CV_32F ? (ippiAddSquareMask)ippiAddSquare_32f_C1IMR : 0;
        }

        if (ippiAddSquare_I || ippiAddSquare_IM)
        {
            IppStatus status = ippStsErr;

            Size size = src.size();
            int srcstep = (int)src.step, dststep = (int)dst.step, maskstep = (int)mask.step;
            if (src.isContinuous() && dst.isContinuous() && mask.isContinuous())
            {
                srcstep = static_cast<int>(src.total() * src.elemSize());
                dststep = static_cast<int>(dst.total() * dst.elemSize());
                maskstep = static_cast<int>(mask.total() * mask.elemSize());
                size.width = static_cast<int>(src.total());
                size.height = 1;
            }
            size.width *= scn;

            if (ippiAddSquare_I)
                status = CV_INSTRUMENT_FUN_IPP(ippiAddSquare_I, src.ptr(), srcstep, dst.ptr<Ipp32f>(), dststep, ippiSize(size.width, size.height));
            else if (ippiAddSquare_IM)
                status = CV_INSTRUMENT_FUN_IPP(ippiAddSquare_IM, src.ptr(), srcstep, mask.ptr<Ipp8u>(), maskstep,
                    dst.ptr<Ipp32f>(), dststep, ippiSize(size.width, size.height));

            if (status >= 0)
                return true;
        }
    }
    return false;
}
}
#endif

void cv::accumulateSquare( InputArray _src, InputOutputArray _dst, InputArray _mask )
{
    CV_INSTRUMENT_REGION()

    int stype = _src.type(), sdepth = CV_MAT_DEPTH(stype), scn = CV_MAT_CN(stype);
    int dtype = _dst.type(), ddepth = CV_MAT_DEPTH(dtype), dcn = CV_MAT_CN(dtype);

    CV_Assert( _src.sameSize(_dst) && dcn == scn );
    CV_Assert( _mask.empty() || (_src.sameSize(_mask) && _mask.type() == CV_8U) );

    CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat(),
               ocl_accumulate(_src, noArray(), _dst, 0.0, _mask, ACCUMULATE_SQUARE))

    CV_IPP_RUN((_src.dims() <= 2 || (_src.isContinuous() && _dst.isContinuous() && (_mask.empty() || _mask.isContinuous()))),
        ipp_accumulate_square(_src, _dst, _mask));

    CV_OVX_RUN(_src.dims() <= 2,
               openvx_accumulate(_src, _dst, _mask, 0.0, VX_ACCUMULATE_SQUARE_OP))

    Mat src = _src.getMat(), dst = _dst.getMat(), mask = _mask.getMat();

    int fidx = getAccTabIdx(sdepth, ddepth);
    AccFunc func = fidx >= 0 ? accSqrTab[fidx] : 0;
    CV_Assert( func != 0 );

    const Mat* arrays[] = {&src, &dst, &mask, 0};
    uchar* ptrs[3];
    NAryMatIterator it(arrays, ptrs);
    int len = (int)it.size;

    for( size_t i = 0; i < it.nplanes; i++, ++it )
        func(ptrs[0], ptrs[1], ptrs[2], len, scn);
}

#if defined(HAVE_IPP)
namespace cv
{
static bool ipp_accumulate_product(InputArray _src1, InputArray _src2,
                            InputOutputArray _dst, InputArray _mask)
{
    CV_INSTRUMENT_REGION_IPP()

    int stype = _src1.type(), sdepth = CV_MAT_DEPTH(stype), scn = CV_MAT_CN(stype);
    int dtype = _dst.type(), ddepth = CV_MAT_DEPTH(dtype);

    Mat src1 = _src1.getMat(), src2 = _src2.getMat(), dst = _dst.getMat(), mask = _mask.getMat();

    if (src1.dims <= 2 || (src1.isContinuous() && src2.isContinuous() && dst.isContinuous()))
    {
        typedef IppStatus (CV_STDCALL * ippiAddProduct)(const void * pSrc1, int src1Step, const void * pSrc2,
                                                        int src2Step, Ipp32f * pSrcDst, int srcDstStep, IppiSize roiSize);
        typedef IppStatus (CV_STDCALL * ippiAddProductMask)(const void * pSrc1, int src1Step, const void * pSrc2, int src2Step,
                                                            const Ipp8u * pMask, int maskStep, Ipp32f * pSrcDst, int srcDstStep, IppiSize roiSize);
        ippiAddProduct ippiAddProduct_I = 0;
        ippiAddProductMask ippiAddProduct_IM = 0;

        if (mask.empty())
        {
            ippiAddProduct_I = sdepth == CV_8U && ddepth == CV_32F ? (ippiAddProduct)ippiAddProduct_8u32f_C1IR :
                sdepth == CV_16U && ddepth == CV_32F ? (ippiAddProduct)ippiAddProduct_16u32f_C1IR :
                sdepth == CV_32F && ddepth == CV_32F ? (ippiAddProduct)ippiAddProduct_32f_C1IR : 0;
        }
        else if (scn == 1)
        {
            ippiAddProduct_IM = sdepth == CV_8U && ddepth == CV_32F ? (ippiAddProductMask)ippiAddProduct_8u32f_C1IMR :
                sdepth == CV_16U && ddepth == CV_32F ? (ippiAddProductMask)ippiAddProduct_16u32f_C1IMR :
                sdepth == CV_32F && ddepth == CV_32F ? (ippiAddProductMask)ippiAddProduct_32f_C1IMR : 0;
        }

        if (ippiAddProduct_I || ippiAddProduct_IM)
        {
            IppStatus status = ippStsErr;

            Size size = src1.size();
            int src1step = (int)src1.step, src2step = (int)src2.step, dststep = (int)dst.step, maskstep = (int)mask.step;
            if (src1.isContinuous() && src2.isContinuous() && dst.isContinuous() && mask.isContinuous())
            {
                src1step = static_cast<int>(src1.total() * src1.elemSize());
                src2step = static_cast<int>(src2.total() * src2.elemSize());
                dststep = static_cast<int>(dst.total() * dst.elemSize());
                maskstep = static_cast<int>(mask.total() * mask.elemSize());
                size.width = static_cast<int>(src1.total());
                size.height = 1;
            }
            size.width *= scn;

            if (ippiAddProduct_I)
                status = CV_INSTRUMENT_FUN_IPP(ippiAddProduct_I, src1.ptr(), src1step, src2.ptr(), src2step, dst.ptr<Ipp32f>(),
                    dststep, ippiSize(size.width, size.height));
            else if (ippiAddProduct_IM)
                status = CV_INSTRUMENT_FUN_IPP(ippiAddProduct_IM, src1.ptr(), src1step, src2.ptr(), src2step, mask.ptr<Ipp8u>(), maskstep,
                    dst.ptr<Ipp32f>(), dststep, ippiSize(size.width, size.height));

            if (status >= 0)
                return true;
        }
    }
    return false;
}
}
#endif



void cv::accumulateProduct( InputArray _src1, InputArray _src2,
                            InputOutputArray _dst, InputArray _mask )
{
    CV_INSTRUMENT_REGION()

    int stype = _src1.type(), sdepth = CV_MAT_DEPTH(stype), scn = CV_MAT_CN(stype);
    int dtype = _dst.type(), ddepth = CV_MAT_DEPTH(dtype), dcn = CV_MAT_CN(dtype);

    CV_Assert( _src1.sameSize(_src2) && stype == _src2.type() );
    CV_Assert( _src1.sameSize(_dst) && dcn == scn );
    CV_Assert( _mask.empty() || (_src1.sameSize(_mask) && _mask.type() == CV_8U) );

    CV_OCL_RUN(_src1.dims() <= 2 && _dst.isUMat(),
               ocl_accumulate(_src1, _src2, _dst, 0.0, _mask, ACCUMULATE_PRODUCT))

    CV_IPP_RUN( (_src1.dims() <= 2 || (_src1.isContinuous() && _src2.isContinuous() && _dst.isContinuous())),
        ipp_accumulate_product(_src1, _src2, _dst, _mask));

    Mat src1 = _src1.getMat(), src2 = _src2.getMat(), dst = _dst.getMat(), mask = _mask.getMat();

    int fidx = getAccTabIdx(sdepth, ddepth);
    AccProdFunc func = fidx >= 0 ? accProdTab[fidx] : 0;
    CV_Assert( func != 0 );

    const Mat* arrays[] = {&src1, &src2, &dst, &mask, 0};
    uchar* ptrs[4];
    NAryMatIterator it(arrays, ptrs);
    int len = (int)it.size;

    for( size_t i = 0; i < it.nplanes; i++, ++it )
        func(ptrs[0], ptrs[1], ptrs[2], ptrs[3], len, scn);
}

#if defined(HAVE_IPP)
namespace cv
{
static bool ipp_accumulate_weighted( InputArray _src, InputOutputArray _dst,
                             double alpha, InputArray _mask )
{
    CV_INSTRUMENT_REGION_IPP()

    int stype = _src.type(), sdepth = CV_MAT_DEPTH(stype), scn = CV_MAT_CN(stype);
    int dtype = _dst.type(), ddepth = CV_MAT_DEPTH(dtype);

    Mat src = _src.getMat(), dst = _dst.getMat(), mask = _mask.getMat();

    if (src.dims <= 2 || (src.isContinuous() && dst.isContinuous() && mask.isContinuous()))
    {
        typedef IppStatus (CV_STDCALL * ippiAddWeighted)(const void * pSrc, int srcStep, Ipp32f * pSrcDst, int srcdstStep,
                                                            IppiSize roiSize, Ipp32f alpha);
        typedef IppStatus (CV_STDCALL * ippiAddWeightedMask)(const void * pSrc, int srcStep, const Ipp8u * pMask,
                                                                int maskStep, Ipp32f * pSrcDst,
                                                                int srcDstStep, IppiSize roiSize, Ipp32f alpha);
        ippiAddWeighted ippiAddWeighted_I = 0;
        ippiAddWeightedMask ippiAddWeighted_IM = 0;

        if (mask.empty())
        {
            ippiAddWeighted_I = sdepth == CV_8U && ddepth == CV_32F ? (ippiAddWeighted)ippiAddWeighted_8u32f_C1IR :
                sdepth == CV_16U && ddepth == CV_32F ? (ippiAddWeighted)ippiAddWeighted_16u32f_C1IR :
                sdepth == CV_32F && ddepth == CV_32F ? (ippiAddWeighted)ippiAddWeighted_32f_C1IR : 0;
        }
        else if (scn == 1)
        {
            ippiAddWeighted_IM = sdepth == CV_8U && ddepth == CV_32F ? (ippiAddWeightedMask)ippiAddWeighted_8u32f_C1IMR :
                sdepth == CV_16U && ddepth == CV_32F ? (ippiAddWeightedMask)ippiAddWeighted_16u32f_C1IMR :
                sdepth == CV_32F && ddepth == CV_32F ? (ippiAddWeightedMask)ippiAddWeighted_32f_C1IMR : 0;
        }

        if (ippiAddWeighted_I || ippiAddWeighted_IM)
        {
            IppStatus status = ippStsErr;

            Size size = src.size();
            int srcstep = (int)src.step, dststep = (int)dst.step, maskstep = (int)mask.step;
            if (src.isContinuous() && dst.isContinuous() && mask.isContinuous())
            {
                srcstep = static_cast<int>(src.total() * src.elemSize());
                dststep = static_cast<int>(dst.total() * dst.elemSize());
                maskstep = static_cast<int>(mask.total() * mask.elemSize());
                size.width = static_cast<int>((int)src.total());
                size.height = 1;
            }
            size.width *= scn;

            if (ippiAddWeighted_I)
                status = CV_INSTRUMENT_FUN_IPP(ippiAddWeighted_I, src.ptr(), srcstep, dst.ptr<Ipp32f>(), dststep, ippiSize(size.width, size.height), (Ipp32f)alpha);
            else if (ippiAddWeighted_IM)
                status = CV_INSTRUMENT_FUN_IPP(ippiAddWeighted_IM, src.ptr(), srcstep, mask.ptr<Ipp8u>(), maskstep,
                    dst.ptr<Ipp32f>(), dststep, ippiSize(size.width, size.height), (Ipp32f)alpha);

            if (status >= 0)
                return true;
        }
    }
    return false;
}
}
#endif

void cv::accumulateWeighted( InputArray _src, InputOutputArray _dst,
                             double alpha, InputArray _mask )
{
    CV_INSTRUMENT_REGION()

    int stype = _src.type(), sdepth = CV_MAT_DEPTH(stype), scn = CV_MAT_CN(stype);
    int dtype = _dst.type(), ddepth = CV_MAT_DEPTH(dtype), dcn = CV_MAT_CN(dtype);

    CV_Assert( _src.sameSize(_dst) && dcn == scn );
    CV_Assert( _mask.empty() || (_src.sameSize(_mask) && _mask.type() == CV_8U) );

    CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat(),
               ocl_accumulate(_src, noArray(), _dst, alpha, _mask, ACCUMULATE_WEIGHTED))

    CV_IPP_RUN((_src.dims() <= 2 || (_src.isContinuous() && _dst.isContinuous() && _mask.isContinuous())), ipp_accumulate_weighted(_src, _dst, alpha, _mask));

    CV_OVX_RUN(_src.dims() <= 2,
               openvx_accumulate(_src, _dst, _mask, alpha, VX_ACCUMULATE_WEIGHTED_OP))

    Mat src = _src.getMat(), dst = _dst.getMat(), mask = _mask.getMat();


    int fidx = getAccTabIdx(sdepth, ddepth);
    AccWFunc func = fidx >= 0 ? accWTab[fidx] : 0;
    CV_Assert( func != 0 );

    const Mat* arrays[] = {&src, &dst, &mask, 0};
    uchar* ptrs[3];
    NAryMatIterator it(arrays, ptrs);
    int len = (int)it.size;

    for( size_t i = 0; i < it.nplanes; i++, ++it )
        func(ptrs[0], ptrs[1], ptrs[2], len, scn, alpha);
}


CV_IMPL void
cvAcc( const void* arr, void* sumarr, const void* maskarr )
{
    cv::Mat src = cv::cvarrToMat(arr), dst = cv::cvarrToMat(sumarr), mask;
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::accumulate( src, dst, mask );
}

CV_IMPL void
cvSquareAcc( const void* arr, void* sumarr, const void* maskarr )
{
    cv::Mat src = cv::cvarrToMat(arr), dst = cv::cvarrToMat(sumarr), mask;
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::accumulateSquare( src, dst, mask );
}

CV_IMPL void
cvMultiplyAcc( const void* arr1, const void* arr2,
               void* sumarr, const void* maskarr )
{
    cv::Mat src1 = cv::cvarrToMat(arr1), src2 = cv::cvarrToMat(arr2);
    cv::Mat dst = cv::cvarrToMat(sumarr), mask;
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::accumulateProduct( src1, src2, dst, mask );
}

CV_IMPL void
cvRunningAvg( const void* arr, void* sumarr, double alpha, const void* maskarr )
{
    cv::Mat src = cv::cvarrToMat(arr), dst = cv::cvarrToMat(sumarr), mask;
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::accumulateWeighted( src, dst, alpha, mask );
}

/* End of file. */
