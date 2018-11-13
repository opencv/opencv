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

#include <vector>

#include "opencv2/core/hal/intrin.hpp"
#include "opencl_kernels_imgproc.hpp"

#include "opencv2/core/openvx/ovx_defs.hpp"

#include "filter.hpp"

#include "fixedpoint.inl.hpp"

/****************************************************************************************\
                                     Gaussian Blur
\****************************************************************************************/

cv::Mat cv::getGaussianKernel( int n, double sigma, int ktype )
{
    CV_Assert(n > 0);
    const int SMALL_GAUSSIAN_SIZE = 7;
    static const float small_gaussian_tab[][SMALL_GAUSSIAN_SIZE] =
    {
        {1.f},
        {0.25f, 0.5f, 0.25f},
        {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f},
        {0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f}
    };

    const float* fixed_kernel = n % 2 == 1 && n <= SMALL_GAUSSIAN_SIZE && sigma <= 0 ?
        small_gaussian_tab[n>>1] : 0;

    CV_Assert( ktype == CV_32F || ktype == CV_64F );
    Mat kernel(n, 1, ktype);
    float* cf = kernel.ptr<float>();
    double* cd = kernel.ptr<double>();

    double sigmaX = sigma > 0 ? sigma : ((n-1)*0.5 - 1)*0.3 + 0.8;
    double scale2X = -0.5/(sigmaX*sigmaX);
    double sum = 0;

    int i;
    for( i = 0; i < n; i++ )
    {
        double x = i - (n-1)*0.5;
        double t = fixed_kernel ? (double)fixed_kernel[i] : std::exp(scale2X*x*x);
        if( ktype == CV_32F )
        {
            cf[i] = (float)t;
            sum += cf[i];
        }
        else
        {
            cd[i] = t;
            sum += cd[i];
        }
    }

    CV_DbgAssert(fabs(sum) > 0);
    sum = 1./sum;
    for( i = 0; i < n; i++ )
    {
        if( ktype == CV_32F )
            cf[i] = (float)(cf[i]*sum);
        else
            cd[i] *= sum;
    }

    return kernel;
}

namespace cv {

template <typename T>
static std::vector<T> getFixedpointGaussianKernel( int n, double sigma )
{
    if (sigma <= 0)
    {
        if(n == 1)
            return std::vector<T>(1, softdouble(1.0));
        else if(n == 3)
        {
            T v3[] = { softdouble(0.25), softdouble(0.5), softdouble(0.25) };
            return std::vector<T>(v3, v3 + 3);
        }
        else if(n == 5)
        {
            T v5[] = { softdouble(0.0625), softdouble(0.25), softdouble(0.375), softdouble(0.25), softdouble(0.0625) };
            return std::vector<T>(v5, v5 + 5);
        }
        else if(n == 7)
        {
            T v7[] = { softdouble(0.03125), softdouble(0.109375), softdouble(0.21875), softdouble(0.28125), softdouble(0.21875), softdouble(0.109375), softdouble(0.03125) };
            return std::vector<T>(v7, v7 + 7);
        }
    }


    softdouble sigmaX = sigma > 0 ? softdouble(sigma) : mulAdd(softdouble(n),softdouble(0.15),softdouble(0.35));// softdouble(((n-1)*0.5 - 1)*0.3 + 0.8)
    softdouble scale2X = softdouble(-0.5*0.25)/(sigmaX*sigmaX);
    std::vector<softdouble> values(n);
    softdouble sum(0.);
    for(int i = 0, x = 1 - n; i < n; i++, x+=2 )
    {
        // x = i - (n - 1)*0.5
        // t = std::exp(scale2X*x*x)
        values[i] = exp(softdouble(x*x)*scale2X);
        sum += values[i];
    }
    sum = softdouble::one()/sum;

    std::vector<T> kernel(n);
    for(int i = 0; i < n; i++ )
    {
        kernel[i] = values[i] * sum;
    }

    return kernel;
};

template <typename ET, typename FT>
void hlineSmooth1N(const ET* src, int cn, const FT* m, int, FT* dst, int len, int)
{
    for (int i = 0; i < len*cn; i++, src++, dst++)
        *dst = (*m) * (*src);
}
template <>
void hlineSmooth1N<uint8_t, ufixedpoint16>(const uint8_t* src, int cn, const ufixedpoint16* m, int, ufixedpoint16* dst, int len, int)
{
    int lencn = len*cn;
    int i = 0;
#if CV_SIMD
    const int VECSZ = v_uint16::nlanes;
    v_uint16 v_mul = vx_setall_u16(*((uint16_t*)m));
    for (; i <= lencn - VECSZ; i += VECSZ)
        v_store((uint16_t*)dst + i, v_mul_wrap(v_mul, vx_load_expand(src + i)));
#endif
    for (; i < lencn; i++)
        dst[i] = m[0] * src[i];
}
template <typename ET, typename FT>
void hlineSmooth1N1(const ET* src, int cn, const FT*, int, FT* dst, int len, int)
{
    for (int i = 0; i < len*cn; i++, src++, dst++)
        *dst = *src;
}
template <>
void hlineSmooth1N1<uint8_t, ufixedpoint16>(const uint8_t* src, int cn, const ufixedpoint16*, int, ufixedpoint16* dst, int len, int)
{
    int lencn = len*cn;
    int i = 0;
#if CV_SIMD
    const int VECSZ = v_uint16::nlanes;
    for (; i <= lencn - VECSZ; i += VECSZ)
        v_store((uint16_t*)dst + i, v_shl<8>(vx_load_expand(src + i)));
#endif
    for (; i < lencn; i++)
        dst[i] = src[i];
}
template <typename ET, typename FT>
void hlineSmooth3N(const ET* src, int cn, const FT* m, int, FT* dst, int len, int borderType)
{
    if (len == 1)
    {
        FT msum = borderType != BORDER_CONSTANT ? m[0] + m[1] + m[2] : m[1];
        for (int k = 0; k < cn; k++)
            dst[k] = msum * src[k];
    }
    else
    {
        // Point that fall left from border
        for (int k = 0; k < cn; k++)
            dst[k] = m[1] * src[k] + m[2] * src[cn + k];
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int src_idx = borderInterpolate(-1, len, borderType);
            for (int k = 0; k < cn; k++)
                dst[k] = dst[k] + m[0] * src[src_idx*cn + k];
        }

        src += cn; dst += cn;
        for (int i = cn; i < (len - 1)*cn; i++, src++, dst++)
            *dst = m[0] * src[-cn] + m[1] * src[0] + m[2] * src[cn];

        // Point that fall right from border
        for (int k = 0; k < cn; k++)
            dst[k] = m[0] * src[k - cn] + m[1] * src[k];
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int src_idx = (borderInterpolate(len, len, borderType) - (len - 1))*cn;
            for (int k = 0; k < cn; k++)
                dst[k] = dst[k] + m[2] * src[src_idx + k];
        }
    }
}
template <>
void hlineSmooth3N<uint8_t, ufixedpoint16>(const uint8_t* src, int cn, const ufixedpoint16* m, int, ufixedpoint16* dst, int len, int borderType)
{
    if (len == 1)
    {
        ufixedpoint16 msum = borderType != BORDER_CONSTANT ? m[0] + m[1] + m[2] : m[1];
        for (int k = 0; k < cn; k++)
            dst[k] = msum * src[k];
    }
    else
    {
        // Point that fall left from border
        for (int k = 0; k < cn; k++)
            dst[k] = m[1] * src[k] + m[2] * src[cn + k];
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int src_idx = borderInterpolate(-1, len, borderType);
            for (int k = 0; k < cn; k++)
                dst[k] = dst[k] + m[0] * src[src_idx*cn + k];
        }

        src += cn; dst += cn;
        int i = cn, lencn = (len - 1)*cn;
#if CV_SIMD
        const uint16_t* _m = (const uint16_t*)m;
        const int VECSZ = v_uint16::nlanes;
        v_uint16 v_mul0 = vx_setall_u16(_m[0]);
        v_uint16 v_mul1 = vx_setall_u16(_m[1]);
        v_uint16 v_mul2 = vx_setall_u16(_m[2]);
        for (; i <= lencn - VECSZ; i += VECSZ, src += VECSZ, dst += VECSZ)
            v_store((uint16_t*)dst, v_mul_wrap(vx_load_expand(src - cn), v_mul0) +
                                    v_mul_wrap(vx_load_expand(src), v_mul1) +
                                    v_mul_wrap(vx_load_expand(src + cn), v_mul2));
#endif
        for (; i < lencn; i++, src++, dst++)
            *dst = m[0] * src[-cn] + m[1] * src[0] + m[2] * src[cn];

        // Point that fall right from border
        for (int k = 0; k < cn; k++)
            dst[k] = m[0] * src[k - cn] + m[1] * src[k];
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int src_idx = (borderInterpolate(len, len, borderType) - (len - 1))*cn;
            for (int k = 0; k < cn; k++)
                dst[k] = dst[k] + m[2] * src[src_idx + k];
        }
    }
}
template <typename ET, typename FT>
void hlineSmooth3N121(const ET* src, int cn, const FT*, int, FT* dst, int len, int borderType)
{
    if (len == 1)
    {
        if(borderType != BORDER_CONSTANT)
            for (int k = 0; k < cn; k++)
                dst[k] = FT(src[k]);
        else
            for (int k = 0; k < cn; k++)
                dst[k] = FT(src[k])>>1;
    }
    else
    {
        // Point that fall left from border
        for (int k = 0; k < cn; k++)
            dst[k] = (FT(src[k])>>1) + (FT(src[cn + k])>>2);
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int src_idx = borderInterpolate(-1, len, borderType);
            for (int k = 0; k < cn; k++)
                dst[k] = dst[k] + (FT(src[src_idx*cn + k])>>2);
        }

        src += cn; dst += cn;
        for (int i = cn; i < (len - 1)*cn; i++, src++, dst++)
            *dst = (FT(src[-cn])>>2) + (FT(src[cn])>>2) + (FT(src[0])>>1);

        // Point that fall right from border
        for (int k = 0; k < cn; k++)
            dst[k] = (FT(src[k - cn])>>2) + (FT(src[k])>>1);
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int src_idx = (borderInterpolate(len, len, borderType) - (len - 1))*cn;
            for (int k = 0; k < cn; k++)
                dst[k] = dst[k] + (FT(src[src_idx + k])>>2);
        }
    }
}
template <>
void hlineSmooth3N121<uint8_t, ufixedpoint16>(const uint8_t* src, int cn, const ufixedpoint16*, int, ufixedpoint16* dst, int len, int borderType)
{
    if (len == 1)
    {
        if (borderType != BORDER_CONSTANT)
            for (int k = 0; k < cn; k++)
                dst[k] = ufixedpoint16(src[k]);
        else
            for (int k = 0; k < cn; k++)
                dst[k] = ufixedpoint16(src[k]) >> 1;
    }
    else
    {
        // Point that fall left from border
        for (int k = 0; k < cn; k++)
            dst[k] = (ufixedpoint16(src[k])>>1) + (ufixedpoint16(src[cn + k])>>2);
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int src_idx = borderInterpolate(-1, len, borderType);
            for (int k = 0; k < cn; k++)
                dst[k] = dst[k] + (ufixedpoint16(src[src_idx*cn + k])>>2);
        }

        src += cn; dst += cn;
        int i = cn, lencn = (len - 1)*cn;
#if CV_SIMD
        const int VECSZ = v_uint16::nlanes;
        for (; i <= lencn - VECSZ; i += VECSZ, src += VECSZ, dst += VECSZ)
            v_store((uint16_t*)dst, (vx_load_expand(src - cn) + vx_load_expand(src + cn) + (vx_load_expand(src) << 1)) << 6);
#endif
        for (; i < lencn; i++, src++, dst++)
            *((uint16_t*)dst) = (uint16_t(src[-cn]) + uint16_t(src[cn]) + (uint16_t(src[0]) << 1)) << 6;

        // Point that fall right from border
        for (int k = 0; k < cn; k++)
            dst[k] = (ufixedpoint16(src[k - cn])>>2) + (ufixedpoint16(src[k])>>1);
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int src_idx = (borderInterpolate(len, len, borderType) - (len - 1))*cn;
            for (int k = 0; k < cn; k++)
                dst[k] = dst[k] + (ufixedpoint16(src[src_idx + k])>>2);
        }
    }
}
template <typename ET, typename FT>
void hlineSmooth3Naba(const ET* src, int cn, const FT* m, int, FT* dst, int len, int borderType)
{
    if (len == 1)
    {
        FT msum = borderType != BORDER_CONSTANT ? (m[0]<<1) + m[1] : m[1];
        for (int k = 0; k < cn; k++)
            dst[k] = msum * src[k];
    }
    else
    {
        // Point that fall left from border
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int src_idx = borderInterpolate(-1, len, borderType);
            for (int k = 0; k < cn; k++)
                dst[k] = m[1] * src[k] + m[0] * src[cn + k] + m[0] * src[src_idx*cn + k];
        }
        else
        {
            for (int k = 0; k < cn; k++)
                dst[k] = m[1] * src[k] + m[0] * src[cn + k];
        }

        src += cn; dst += cn;
        for (int i = cn; i < (len - 1)*cn; i++, src++, dst++)
            *dst = m[1] * src[0] + m[0] * src[-cn] + m[0] * src[cn];

        // Point that fall right from border
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int src_idx = (borderInterpolate(len, len, borderType) - (len - 1))*cn;
            for (int k = 0; k < cn; k++)
                dst[k] = m[1] * src[k] + m[0] * src[k - cn] + m[0] * src[src_idx + k];
        }
        else
        {
            for (int k = 0; k < cn; k++)
                dst[k] = m[0] * src[k - cn] + m[1] * src[k];
        }
    }
}
template <>
void hlineSmooth3Naba<uint8_t, ufixedpoint16>(const uint8_t* src, int cn, const ufixedpoint16* m, int, ufixedpoint16* dst, int len, int borderType)
{
    if (len == 1)
    {
        ufixedpoint16 msum = borderType != BORDER_CONSTANT ? (m[0]<<1) + m[1] : m[1];
        for (int k = 0; k < cn; k++)
            dst[k] = msum * src[k];
    }
    else
    {
        // Point that fall left from border
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int src_idx = borderInterpolate(-1, len, borderType);
            for (int k = 0; k < cn; k++)
                ((uint16_t*)dst)[k] = ((uint16_t*)m)[1] * src[k] + ((uint16_t*)m)[0] * ((uint16_t)(src[cn + k]) + (uint16_t)(src[src_idx*cn + k]));
        }
        else
        {
            for (int k = 0; k < cn; k++)
                dst[k] = m[1] * src[k] + m[0] * src[cn + k];
        }

        src += cn; dst += cn;
        int i = cn, lencn = (len - 1)*cn;
#if CV_SIMD
        const uint16_t* _m = (const uint16_t*)m;
        const int VECSZ = v_uint16::nlanes;
        v_uint16 v_mul0 = vx_setall_u16(_m[0]);
        v_uint16 v_mul1 = vx_setall_u16(_m[1]);
        for (; i <= lencn - VECSZ; i += VECSZ, src += VECSZ, dst += VECSZ)
            v_store((uint16_t*)dst, v_mul_wrap(vx_load_expand(src - cn) + vx_load_expand(src + cn), v_mul0) +
                                    v_mul_wrap(vx_load_expand(src), v_mul1));
#endif
        for (; i < lencn; i++, src++, dst++)
            *((uint16_t*)dst) = ((uint16_t*)m)[1] * src[0] + ((uint16_t*)m)[0] * ((uint16_t)(src[-cn]) + (uint16_t)(src[cn]));

        // Point that fall right from border
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int src_idx = (borderInterpolate(len, len, borderType) - (len - 1))*cn;
            for (int k = 0; k < cn; k++)
                ((uint16_t*)dst)[k] = ((uint16_t*)m)[1] * src[k] + ((uint16_t*)m)[0] * ((uint16_t)(src[k - cn]) + (uint16_t)(src[src_idx + k]));
        }
        else
        {
            for (int k = 0; k < cn; k++)
                dst[k] = m[0] * src[k - cn] + m[1] * src[k];
        }
    }
}
template <typename ET, typename FT>
void hlineSmooth5N(const ET* src, int cn, const FT* m, int, FT* dst, int len, int borderType)
{
    if (len == 1)
    {
        FT msum = borderType != BORDER_CONSTANT ? m[0] + m[1] + m[2] + m[3] + m[4] : m[2];
        for (int k = 0; k < cn; k++)
            dst[k] = msum * src[k];
    }
    else if (len == 2)
    {
        if (borderType == BORDER_CONSTANT)
            for (int k = 0; k < cn; k++)
            {
                dst[k   ] = m[2] * src[k] + m[3] * src[k+cn];
                dst[k+cn] = m[1] * src[k] + m[2] * src[k+cn];
            }
        else
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            int idxp1 = borderInterpolate(2, len, borderType)*cn;
            int idxp2 = borderInterpolate(3, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k     ] = m[1] * src[k + idxm1] + m[2] * src[k] + m[3] * src[k + cn] + m[4] * src[k + idxp1] + m[0] * src[k + idxm2];
                dst[k + cn] = m[0] * src[k + idxm1] + m[1] * src[k] + m[2] * src[k + cn] + m[3] * src[k + idxp1] + m[4] * src[k + idxp2];
            }
        }
    }
    else if (len == 3)
    {
        if (borderType == BORDER_CONSTANT)
            for (int k = 0; k < cn; k++)
            {
                dst[k       ] = m[2] * src[k] + m[3] * src[k + cn] + m[4] * src[k + 2*cn];
                dst[k +   cn] = m[1] * src[k] + m[2] * src[k + cn] + m[3] * src[k + 2*cn];
                dst[k + 2*cn] = m[0] * src[k] + m[1] * src[k + cn] + m[2] * src[k + 2*cn];
            }
        else
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            int idxp1 = borderInterpolate(3, len, borderType)*cn;
            int idxp2 = borderInterpolate(4, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k       ] = m[2] * src[k] + m[3] * src[k + cn] + m[4] * src[k + 2*cn] + m[0] * src[k + idxm2] + m[1] * src[k + idxm1];
                dst[k +   cn] = m[1] * src[k] + m[2] * src[k + cn] + m[3] * src[k + 2*cn] + m[0] * src[k + idxm1] + m[4] * src[k + idxp1];
                dst[k + 2*cn] = m[0] * src[k] + m[1] * src[k + cn] + m[2] * src[k + 2*cn] + m[3] * src[k + idxp1] + m[4] * src[k + idxp2];
            }
        }
    }
    else
    {
        // Points that fall left from border
        for (int k = 0; k < cn; k++)
        {
            dst[k] = m[2] * src[k] + m[3] * src[cn + k] + m[4] * src[2*cn + k];
            dst[k + cn] = m[1] * src[k] + m[2] * src[cn + k] + m[3] * src[2*cn + k] + m[4] * src[3*cn + k];
        }
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = dst[k] + m[0] * src[idxm2 + k] + m[1] * src[idxm1 + k];
                dst[k + cn] = dst[k + cn] + m[0] * src[idxm1 + k];
            }
        }

        src += 2*cn; dst += 2*cn;
        for (int i = 2*cn; i < (len - 2)*cn; i++, src++, dst++)
            *dst = m[0] * src[-2*cn] + m[1] * src[-cn] + m[2] * src[0] + m[3] * src[cn] + m[4] * src[2*cn];

        // Points that fall right from border
        for (int k = 0; k < cn; k++)
        {
            dst[k] = m[0] * src[k - 2*cn] + m[1] * src[k - cn] + m[2] * src[k] + m[3] * src[k + cn];
            dst[k + cn] = m[0] * src[k - cn] + m[1] * src[k] + m[2] * src[k + cn];
        }
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int idxp1 = (borderInterpolate(len, len, borderType) - (len - 2))*cn;
            int idxp2 = (borderInterpolate(len+1, len, borderType) - (len - 2))*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = dst[k] + m[4] * src[idxp1 + k];
                dst[k + cn] = dst[k + cn] + m[3] * src[idxp1 + k] + m[4] * src[idxp2 + k];
            }
        }
    }
}
template <>
void hlineSmooth5N<uint8_t, ufixedpoint16>(const uint8_t* src, int cn, const ufixedpoint16* m, int, ufixedpoint16* dst, int len, int borderType)
{
    if (len == 1)
    {
        ufixedpoint16 msum = borderType != BORDER_CONSTANT ? m[0] + m[1] + m[2] + m[3] + m[4] : m[2];
        for (int k = 0; k < cn; k++)
            dst[k] = msum * src[k];
    }
    else if (len == 2)
    {
        if (borderType == BORDER_CONSTANT)
            for (int k = 0; k < cn; k++)
            {
                dst[k] = m[2] * src[k] + m[3] * src[k + cn];
                dst[k + cn] = m[1] * src[k] + m[2] * src[k + cn];
            }
        else
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            int idxp1 = borderInterpolate(2, len, borderType)*cn;
            int idxp2 = borderInterpolate(3, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = m[1] * src[k + idxm1] + m[2] * src[k] + m[3] * src[k + cn] + m[4] * src[k + idxp1] + m[0] * src[k + idxm2];
                dst[k + cn] = m[0] * src[k + idxm1] + m[1] * src[k] + m[2] * src[k + cn] + m[3] * src[k + idxp1] + m[4] * src[k + idxp2];
            }
        }
    }
    else if (len == 3)
    {
        if (borderType == BORDER_CONSTANT)
            for (int k = 0; k < cn; k++)
            {
                dst[k] = m[2] * src[k] + m[3] * src[k + cn] + m[4] * src[k + 2 * cn];
                dst[k + cn] = m[1] * src[k] + m[2] * src[k + cn] + m[3] * src[k + 2 * cn];
                dst[k + 2 * cn] = m[0] * src[k] + m[1] * src[k + cn] + m[2] * src[k + 2 * cn];
            }
        else
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            int idxp1 = borderInterpolate(3, len, borderType)*cn;
            int idxp2 = borderInterpolate(4, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = m[2] * src[k] + m[3] * src[k + cn] + m[4] * src[k + 2 * cn] + m[0] * src[k + idxm2] + m[1] * src[k + idxm1];
                dst[k + cn] = m[1] * src[k] + m[2] * src[k + cn] + m[3] * src[k + 2 * cn] + m[0] * src[k + idxm1] + m[4] * src[k + idxp1];
                dst[k + 2 * cn] = m[0] * src[k] + m[1] * src[k + cn] + m[2] * src[k + 2 * cn] + m[3] * src[k + idxp1] + m[4] * src[k + idxp2];
            }
        }
    }
    else
    {
        // Points that fall left from border
        for (int k = 0; k < cn; k++)
        {
            dst[k] = m[2] * src[k] + m[3] * src[cn + k] + m[4] * src[2 * cn + k];
            dst[k + cn] = m[1] * src[k] + m[2] * src[cn + k] + m[3] * src[2 * cn + k] + m[4] * src[3 * cn + k];
        }
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = dst[k] + m[0] * src[idxm2 + k] + m[1] * src[idxm1 + k];
                dst[k + cn] = dst[k + cn] + m[0] * src[idxm1 + k];
            }
        }

        src += 2 * cn; dst += 2 * cn;
        int i = 2*cn, lencn = (len - 2)*cn;
#if CV_SIMD
        const uint16_t* _m = (const uint16_t*)m;
        const int VECSZ = v_uint16::nlanes;
        v_uint16 v_mul0 = vx_setall_u16(_m[0]);
        v_uint16 v_mul1 = vx_setall_u16(_m[1]);
        v_uint16 v_mul2 = vx_setall_u16(_m[2]);
        v_uint16 v_mul3 = vx_setall_u16(_m[3]);
        v_uint16 v_mul4 = vx_setall_u16(_m[4]);
        for (; i <= lencn - VECSZ; i += VECSZ, src += VECSZ, dst += VECSZ)
            v_store((uint16_t*)dst, v_mul_wrap(vx_load_expand(src - 2 * cn), v_mul0) +
                                    v_mul_wrap(vx_load_expand(src - cn), v_mul1) +
                                    v_mul_wrap(vx_load_expand(src), v_mul2) +
                                    v_mul_wrap(vx_load_expand(src + cn), v_mul3) +
                                    v_mul_wrap(vx_load_expand(src + 2 * cn), v_mul4));
#endif
        for (; i < lencn; i++, src++, dst++)
            *dst = m[0] * src[-2*cn] + m[1] * src[-cn] + m[2] * src[0] + m[3] * src[cn] + m[4] * src[2*cn];

        // Points that fall right from border
        for (int k = 0; k < cn; k++)
        {
            dst[k] = m[0] * src[k - 2 * cn] + m[1] * src[k - cn] + m[2] * src[k] + m[3] * src[k + cn];
            dst[k + cn] = m[0] * src[k - cn] + m[1] * src[k] + m[2] * src[k + cn];
        }
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int idxp1 = (borderInterpolate(len, len, borderType) - (len - 2))*cn;
            int idxp2 = (borderInterpolate(len + 1, len, borderType) - (len - 2))*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = dst[k] + m[4] * src[idxp1 + k];
                dst[k + cn] = dst[k + cn] + m[3] * src[idxp1 + k] + m[4] * src[idxp2 + k];
            }
        }
    }
}
template <typename ET, typename FT>
void hlineSmooth5N14641(const ET* src, int cn, const FT*, int, FT* dst, int len, int borderType)
{
    if (len == 1)
    {
        if (borderType == BORDER_CONSTANT)
            for (int k = 0; k < cn; k++)
                dst[k] = (FT(src[k])>>3)*(uint8_t)3;
        else
            for (int k = 0; k < cn; k++)
                dst[k] = src[k];
    }
    else if (len == 2)
    {
        if (borderType == BORDER_CONSTANT)
            for (int k = 0; k < cn; k++)
            {
                dst[k] = (FT(src[k])>>4)*(uint8_t)6 + (FT(src[k + cn])>>2);
                dst[k + cn] = (FT(src[k]) >> 2) + (FT(src[k + cn])>>4)*(uint8_t)6;
            }
        else
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            int idxp1 = borderInterpolate(2, len, borderType)*cn;
            int idxp2 = borderInterpolate(3, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = (FT(src[k])>>4)*(uint8_t)6 + (FT(src[k + idxm1])>>2) + (FT(src[k + cn])>>2) + (FT(src[k + idxp1])>>4) + (FT(src[k + idxm2])>>4);
                dst[k + cn] = (FT(src[k + cn])>>4)*(uint8_t)6 + (FT(src[k])>>2) + (FT(src[k + idxp1])>>2) + (FT(src[k + idxm1])>>4) + (FT(src[k + idxp2])>>4);
            }
        }
    }
    else if (len == 3)
    {
        if (borderType == BORDER_CONSTANT)
            for (int k = 0; k < cn; k++)
            {
                dst[k] = (FT(src[k])>>4)*(uint8_t)6 + (FT(src[k + cn])>>2) + (FT(src[k + 2 * cn])>>4);
                dst[k + cn] = (FT(src[k + cn])>>4)*(uint8_t)6 + (FT(src[k])>>2) + (FT(src[k + 2 * cn])>>2);
                dst[k + 2 * cn] = (FT(src[k + 2 * cn])>>4)*(uint8_t)6 + (FT(src[k + cn])>>2) + (FT(src[k])>>4);
            }
        else
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            int idxp1 = borderInterpolate(3, len, borderType)*cn;
            int idxp2 = borderInterpolate(4, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = (FT(src[k])>>4)*(uint8_t)6 + (FT(src[k + cn])>>2) + (FT(src[k + idxm1])>>2) + (FT(src[k + 2 * cn])>>4) + (FT(src[k + idxm2])>>4);
                dst[k + cn] = (FT(src[k + cn])>>4)*(uint8_t)6 + (FT(src[k])>>2) + (FT(src[k + 2 * cn])>>2) + (FT(src[k + idxm1])>>4) + (FT(src[k + idxp1])>>4);
                dst[k + 2 * cn] = (FT(src[k + 2 * cn])>>4)*(uint8_t)6 + (FT(src[k + cn])>>2) + (FT(src[k + idxp1])>>2) + (FT(src[k])>>4) + (FT(src[k + idxp2])>>4);
            }
        }
    }
    else
    {
        // Points that fall left from border
        for (int k = 0; k < cn; k++)
        {
            dst[k] = (FT(src[k])>>4)*(uint8_t)6 + (FT(src[cn + k])>>2) + (FT(src[2 * cn + k])>>4);
            dst[k + cn] = (FT(src[cn + k])>>4)*(uint8_t)6 + (FT(src[k])>>2) + (FT(src[2 * cn + k])>>2) + (FT(src[3 * cn + k])>>4);
        }
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = dst[k] + (FT(src[idxm2 + k])>>4) + (FT(src[idxm1 + k])>>2);
                dst[k + cn] = dst[k + cn] + (FT(src[idxm1 + k])>>4);
            }
        }

        src += 2 * cn; dst += 2 * cn;
        for (int i = 2 * cn; i < (len - 2)*cn; i++, src++, dst++)
            *dst = (FT(src[0])>>4)*(uint8_t)6 + (FT(src[-cn])>>2) + (FT(src[cn])>>2) + (FT(src[-2 * cn])>>4) + (FT(src[2 * cn])>>4);

        // Points that fall right from border
        for (int k = 0; k < cn; k++)
        {
            dst[k] = (FT(src[k])>>4)*(uint8_t)6 + (FT(src[k - cn])>>2) + (FT(src[k + cn])>>2) + (FT(src[k - 2 * cn])>>4);
            dst[k + cn] = (FT(src[k + cn])>>4)*(uint8_t)6 + (FT(src[k])>>2) + (FT(src[k - cn])>>4);
        }
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int idxp1 = (borderInterpolate(len, len, borderType) - (len - 2))*cn;
            int idxp2 = (borderInterpolate(len + 1, len, borderType) - (len - 2))*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = dst[k] + (FT(src[idxp1 + k])>>4);
                dst[k + cn] = dst[k + cn] + (FT(src[idxp1 + k])>>2) + (FT(src[idxp2 + k])>>4);
            }
        }
    }
}
template <>
void hlineSmooth5N14641<uint8_t, ufixedpoint16>(const uint8_t* src, int cn, const ufixedpoint16*, int, ufixedpoint16* dst, int len, int borderType)
{
    if (len == 1)
    {
        if (borderType == BORDER_CONSTANT)
            for (int k = 0; k < cn; k++)
                dst[k] = (ufixedpoint16(src[k])>>3) * (uint8_t)3;
        else
        {
            for (int k = 0; k < cn; k++)
                dst[k] = src[k];
        }
    }
    else if (len == 2)
    {
        if (borderType == BORDER_CONSTANT)
            for (int k = 0; k < cn; k++)
            {
                dst[k] = (ufixedpoint16(src[k]) >> 4) * (uint8_t)6 + (ufixedpoint16(src[k + cn]) >> 2);
                dst[k + cn] = (ufixedpoint16(src[k]) >> 2) + (ufixedpoint16(src[k + cn]) >> 4) * (uint8_t)6;
            }
        else
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            int idxp1 = borderInterpolate(2, len, borderType)*cn;
            int idxp2 = borderInterpolate(3, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = (ufixedpoint16(src[k]) >> 4) * (uint8_t)6 + (ufixedpoint16(src[k + idxm1]) >> 2) + (ufixedpoint16(src[k + cn]) >> 2) + (ufixedpoint16(src[k + idxp1]) >> 4) + (ufixedpoint16(src[k + idxm2]) >> 4);
                dst[k + cn] = (ufixedpoint16(src[k + cn]) >> 4) * (uint8_t)6 + (ufixedpoint16(src[k]) >> 2) + (ufixedpoint16(src[k + idxp1]) >> 2) + (ufixedpoint16(src[k + idxm1]) >> 4) + (ufixedpoint16(src[k + idxp2]) >> 4);
            }
        }
    }
    else if (len == 3)
    {
        if (borderType == BORDER_CONSTANT)
            for (int k = 0; k < cn; k++)
            {
                dst[k] = (ufixedpoint16(src[k]) >> 4) * (uint8_t)6 + (ufixedpoint16(src[k + cn]) >> 2) + (ufixedpoint16(src[k + 2 * cn]) >> 4);
                dst[k + cn] = (ufixedpoint16(src[k + cn]) >> 4) * (uint8_t)6 + (ufixedpoint16(src[k]) >> 2) + (ufixedpoint16(src[k + 2 * cn]) >> 2);
                dst[k + 2 * cn] = (ufixedpoint16(src[k + 2 * cn]) >> 4) * (uint8_t)6 + (ufixedpoint16(src[k + cn]) >> 2) + (ufixedpoint16(src[k]) >> 4);
            }
        else
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            int idxp1 = borderInterpolate(3, len, borderType)*cn;
            int idxp2 = borderInterpolate(4, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = (ufixedpoint16(src[k]) >> 4) * (uint8_t)6 + (ufixedpoint16(src[k + cn]) >> 2) + (ufixedpoint16(src[k + idxm1]) >> 2) + (ufixedpoint16(src[k + 2 * cn]) >> 4) + (ufixedpoint16(src[k + idxm2]) >> 4);
                dst[k + cn] = (ufixedpoint16(src[k + cn]) >> 4) * (uint8_t)6 + (ufixedpoint16(src[k]) >> 2) + (ufixedpoint16(src[k + 2 * cn]) >> 2) + (ufixedpoint16(src[k + idxm1]) >> 4) + (ufixedpoint16(src[k + idxp1]) >> 4);
                dst[k + 2 * cn] = (ufixedpoint16(src[k + 2 * cn]) >> 4) * (uint8_t)6 + (ufixedpoint16(src[k + cn]) >> 2) + (ufixedpoint16(src[k + idxp1]) >> 2) + (ufixedpoint16(src[k]) >> 4) + (ufixedpoint16(src[k + idxp2]) >> 4);
            }
        }
    }
    else
    {
        // Points that fall left from border
        for (int k = 0; k < cn; k++)
        {
            dst[k] = (ufixedpoint16(src[k]) >> 4) * (uint8_t)6 + (ufixedpoint16(src[cn + k]) >> 2) + (ufixedpoint16(src[2 * cn + k]) >> 4);
            dst[k + cn] = (ufixedpoint16(src[cn + k]) >> 4) * (uint8_t)6 + (ufixedpoint16(src[k]) >> 2) + (ufixedpoint16(src[2 * cn + k]) >> 2) + (ufixedpoint16(src[3 * cn + k]) >> 4);
        }
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = dst[k] + (ufixedpoint16(src[idxm2 + k]) >> 4) + (ufixedpoint16(src[idxm1 + k]) >> 2);
                dst[k + cn] = dst[k + cn] + (ufixedpoint16(src[idxm1 + k]) >> 4);
            }
        }

        src += 2 * cn; dst += 2 * cn;
        int i = 2 * cn, lencn = (len - 2)*cn;
#if CV_SIMD
        const int VECSZ = v_uint16::nlanes;
        v_uint16 v_6 = vx_setall_u16(6);
        for (; i <= lencn - VECSZ; i += VECSZ, src += VECSZ, dst += VECSZ)
            v_store((uint16_t*)dst, (v_mul_wrap(vx_load_expand(src), v_6) + ((vx_load_expand(src - cn) + vx_load_expand(src + cn)) << 2) + vx_load_expand(src - 2 * cn) + vx_load_expand(src + 2 * cn)) << 4);
#endif
        for (; i < lencn; i++, src++, dst++)
            *((uint16_t*)dst) = (uint16_t(src[0]) * 6 + ((uint16_t(src[-cn]) + uint16_t(src[cn])) << 2) + uint16_t(src[-2 * cn]) + uint16_t(src[2 * cn])) << 4;

        // Points that fall right from border
        for (int k = 0; k < cn; k++)
        {
            dst[k] = (ufixedpoint16(src[k]) >> 4) * (uint8_t)6 + (ufixedpoint16(src[k - cn]) >> 2) + (ufixedpoint16(src[k + cn]) >> 2) + (ufixedpoint16(src[k - 2 * cn]) >> 4);
            dst[k + cn] = (ufixedpoint16(src[k + cn]) >> 4) * (uint8_t)6 + (ufixedpoint16(src[k]) >> 2) + (ufixedpoint16(src[k - cn]) >> 4);
        }
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int idxp1 = (borderInterpolate(len, len, borderType) - (len - 2))*cn;
            int idxp2 = (borderInterpolate(len + 1, len, borderType) - (len - 2))*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = dst[k] + (ufixedpoint16(src[idxp1 + k]) >> 4);
                dst[k + cn] = dst[k + cn] + (ufixedpoint16(src[idxp1 + k]) >> 2) + (ufixedpoint16(src[idxp2 + k]) >> 4);
            }
        }
    }
}
template <typename ET, typename FT>
void hlineSmooth5Nabcba(const ET* src, int cn, const FT* m, int, FT* dst, int len, int borderType)
{
    if (len == 1)
    {
        FT msum = borderType != BORDER_CONSTANT ? ((m[0] + m[1])<<1) + m[2] : m[2];
        for (int k = 0; k < cn; k++)
            dst[k] = msum * src[k];
    }
    else if (len == 2)
    {
        if (borderType == BORDER_CONSTANT)
            for (int k = 0; k < cn; k++)
            {
                dst[k] = m[2] * src[k] + m[1] * src[k + cn];
                dst[k + cn] = m[1] * src[k] + m[2] * src[k + cn];
            }
        else
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            int idxp1 = borderInterpolate(2, len, borderType)*cn;
            int idxp2 = borderInterpolate(3, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = m[1] * src[k + idxm1] + m[2] * src[k] + m[1] * src[k + cn] + m[0] * src[k + idxp1] + m[0] * src[k + idxm2];
                dst[k + cn] = m[0] * src[k + idxm1] + m[1] * src[k] + m[2] * src[k + cn] + m[1] * src[k + idxp1] + m[0] * src[k + idxp2];
            }
        }
    }
    else if (len == 3)
    {
        if (borderType == BORDER_CONSTANT)
            for (int k = 0; k < cn; k++)
            {
                dst[k] = m[2] * src[k] + m[1] * src[k + cn] + m[0] * src[k + 2 * cn];
                dst[k + cn] = m[1] * src[k] + m[2] * src[k + cn] + m[1] * src[k + 2 * cn];
                dst[k + 2 * cn] = m[0] * src[k] + m[1] * src[k + cn] + m[2] * src[k + 2 * cn];
            }
        else
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            int idxp1 = borderInterpolate(3, len, borderType)*cn;
            int idxp2 = borderInterpolate(4, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = m[2] * src[k] + m[1] * src[k + cn] + m[0] * src[k + 2 * cn] + m[0] * src[k + idxm2] + m[1] * src[k + idxm1];
                dst[k + cn] = m[1] * src[k] + m[2] * src[k + cn] + m[1] * src[k + 2 * cn] + m[0] * src[k + idxm1] + m[0] * src[k + idxp1];
                dst[k + 2 * cn] = m[0] * src[k] + m[1] * src[k + cn] + m[2] * src[k + 2 * cn] + m[1] * src[k + idxp1] + m[0] * src[k + idxp2];
            }
        }
    }
    else
    {
        // Points that fall left from border
        for (int k = 0; k < cn; k++)
        {
            dst[k] = m[2] * src[k] + m[1] * src[cn + k] + m[0] * src[2 * cn + k];
            dst[k + cn] = m[1] * src[k] + m[2] * src[cn + k] + m[1] * src[2 * cn + k] + m[0] * src[3 * cn + k];
        }
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = dst[k] + m[0] * src[idxm2 + k] + m[1] * src[idxm1 + k];
                dst[k + cn] = dst[k + cn] + m[0] * src[idxm1 + k];
            }
        }

        src += 2 * cn; dst += 2 * cn;
        for (int i = 2 * cn; i < (len - 2)*cn; i++, src++, dst++)
            *dst = m[0] * src[-2 * cn] + m[1] * src[-cn] + m[2] * src[0] + m[3] * src[cn] + m[4] * src[2 * cn];

        // Points that fall right from border
        for (int k = 0; k < cn; k++)
        {
            dst[k] = m[0] * src[k - 2 * cn] + m[1] * src[k - cn] + m[2] * src[k] + m[3] * src[k + cn];
            dst[k + cn] = m[0] * src[k - cn] + m[1] * src[k] + m[2] * src[k + cn];
        }
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int idxp1 = (borderInterpolate(len, len, borderType) - (len - 2))*cn;
            int idxp2 = (borderInterpolate(len + 1, len, borderType) - (len - 2))*cn;
            for (int k = 0; k < cn; k++)
            {
                dst[k] = dst[k] + m[0] * src[idxp1 + k];
                dst[k + cn] = dst[k + cn] + m[1] * src[idxp1 + k] + m[0] * src[idxp2 + k];
            }
        }
    }
}
template <>
void hlineSmooth5Nabcba<uint8_t, ufixedpoint16>(const uint8_t* src, int cn, const ufixedpoint16* m, int, ufixedpoint16* dst, int len, int borderType)
{
    if (len == 1)
    {
        ufixedpoint16 msum = borderType != BORDER_CONSTANT ? ((m[0] + m[1]) << 1) + m[2] : m[2];
        for (int k = 0; k < cn; k++)
            dst[k] = msum * src[k];
    }
    else if (len == 2)
    {
        if (borderType == BORDER_CONSTANT)
            for (int k = 0; k < cn; k++)
            {
                dst[k] = m[2] * src[k] + m[1] * src[k + cn];
                dst[k + cn] = m[1] * src[k] + m[2] * src[k + cn];
            }
        else
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            int idxp1 = borderInterpolate(2, len, borderType)*cn;
            int idxp2 = borderInterpolate(3, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                ((uint16_t*)dst)[k] = ((uint16_t*)m)[1] * ((uint16_t)(src[k + idxm1]) + (uint16_t)(src[k + cn])) + ((uint16_t*)m)[2] * src[k] + ((uint16_t*)m)[0] * ((uint16_t)(src[k + idxp1]) + (uint16_t)(src[k + idxm2]));
                ((uint16_t*)dst)[k + cn] = ((uint16_t*)m)[0] * ((uint16_t)(src[k + idxm1]) + (uint16_t)(src[k + idxp2])) + ((uint16_t*)m)[1] * ((uint16_t)(src[k]) + (uint16_t)(src[k + idxp1])) + ((uint16_t*)m)[2] * src[k + cn];
            }
        }
    }
    else if (len == 3)
    {
        if (borderType == BORDER_CONSTANT)
            for (int k = 0; k < cn; k++)
            {
                dst[k] = m[2] * src[k] + m[1] * src[k + cn] + m[0] * src[k + 2 * cn];
                ((uint16_t*)dst)[k + cn] = ((uint16_t*)m)[1] * ((uint16_t)(src[k]) + (uint16_t)(src[k + 2 * cn])) + ((uint16_t*)m)[2] * src[k + cn];
                dst[k + 2 * cn] = m[0] * src[k] + m[1] * src[k + cn] + m[2] * src[k + 2 * cn];
            }
        else
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            int idxp1 = borderInterpolate(3, len, borderType)*cn;
            int idxp2 = borderInterpolate(4, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                ((uint16_t*)dst)[k] = ((uint16_t*)m)[2] * src[k] + ((uint16_t*)m)[1] * ((uint16_t)(src[k + cn]) + (uint16_t)(src[k + idxm1])) + ((uint16_t*)m)[0] * ((uint16_t)(src[k + 2 * cn]) + (uint16_t)(src[k + idxm2]));
                ((uint16_t*)dst)[k + cn] = ((uint16_t*)m)[2] * src[k + cn] + ((uint16_t*)m)[1] * ((uint16_t)(src[k]) + (uint16_t)(src[k + 2 * cn])) + ((uint16_t*)m)[0] * ((uint16_t)(src[k + idxm1]) + (uint16_t)(src[k + idxp1]));
                ((uint16_t*)dst)[k + 2 * cn] = ((uint16_t*)m)[0] * ((uint16_t)(src[k]) + (uint16_t)(src[k + idxp2])) + ((uint16_t*)m)[1] * ((uint16_t)(src[k + cn]) + (uint16_t)(src[k + idxp1])) + ((uint16_t*)m)[2] * src[k + 2 * cn];
            }
        }
    }
    else
    {
        // Points that fall left from border
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int idxm2 = borderInterpolate(-2, len, borderType)*cn;
            int idxm1 = borderInterpolate(-1, len, borderType)*cn;
            for (int k = 0; k < cn; k++)
            {
                ((uint16_t*)dst)[k] = ((uint16_t*)m)[2] * src[k] + ((uint16_t*)m)[1] * ((uint16_t)(src[cn + k]) + (uint16_t)(src[idxm1 + k])) + ((uint16_t*)m)[0] * ((uint16_t)(src[2 * cn + k]) + (uint16_t)(src[idxm2 + k]));
                ((uint16_t*)dst)[k + cn] = ((uint16_t*)m)[1] * ((uint16_t)(src[k]) + (uint16_t)(src[2 * cn + k])) + ((uint16_t*)m)[2] * src[cn + k] + ((uint16_t*)m)[0] * ((uint16_t)(src[3 * cn + k]) + (uint16_t)(src[idxm1 + k]));
            }
        }
        else
        {
            for (int k = 0; k < cn; k++)
            {
                dst[k] = m[2] * src[k] + m[1] * src[cn + k] + m[0] * src[2 * cn + k];
                ((uint16_t*)dst)[k + cn] = ((uint16_t*)m)[1] * ((uint16_t)(src[k]) + (uint16_t)(src[2 * cn + k])) + ((uint16_t*)m)[2] * src[cn + k] + ((uint16_t*)m)[0] * src[3 * cn + k];
            }
        }

        src += 2 * cn; dst += 2 * cn;
        int i = 2 * cn, lencn = (len - 2)*cn;
#if CV_SIMD
        const uint16_t* _m = (const uint16_t*)m;
        const int VECSZ = v_uint16::nlanes;
        v_uint16 v_mul0 = vx_setall_u16(_m[0]);
        v_uint16 v_mul1 = vx_setall_u16(_m[1]);
        v_uint16 v_mul2 = vx_setall_u16(_m[2]);
        for (; i <= lencn - VECSZ; i += VECSZ, src += VECSZ, dst += VECSZ)
            v_store((uint16_t*)dst, v_mul_wrap(vx_load_expand(src - 2 * cn) + vx_load_expand(src + 2 * cn), v_mul0) +
                                    v_mul_wrap(vx_load_expand(src - cn) + vx_load_expand(src + cn), v_mul1) +
                                    v_mul_wrap(vx_load_expand(src), v_mul2));
#endif
        for (; i < lencn; i++, src++, dst++)
            *((uint16_t*)dst) = ((uint16_t*)m)[0] * ((uint16_t)(src[-2 * cn]) + (uint16_t)(src[2 * cn])) + ((uint16_t*)m)[1] * ((uint16_t)(src[-cn]) + (uint16_t)(src[cn])) + ((uint16_t*)m)[2] * src[0];

        // Points that fall right from border
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int idxp1 = (borderInterpolate(len, len, borderType) - (len - 2))*cn;
            int idxp2 = (borderInterpolate(len + 1, len, borderType) - (len - 2))*cn;
            for (int k = 0; k < cn; k++)
            {
                ((uint16_t*)dst)[k] = ((uint16_t*)m)[0] * ((uint16_t)(src[k - 2 * cn]) + (uint16_t)(src[idxp1 + k])) + ((uint16_t*)m)[1] * ((uint16_t)(src[k - cn]) + (uint16_t)(src[k + cn])) + ((uint16_t*)m)[2] * src[k];
                ((uint16_t*)dst)[k + cn] = ((uint16_t*)m)[0] * ((uint16_t)(src[k - cn]) + (uint16_t)(src[idxp2 + k])) + ((uint16_t*)m)[1] * ((uint16_t)(src[k]) + (uint16_t)(src[idxp1 + k])) + ((uint16_t*)m)[2] * src[k + cn];
            }
        }
        else
        {
            for (int k = 0; k < cn; k++)
            {
                ((uint16_t*)dst)[k] = ((uint16_t*)m)[0] * src[k - 2 * cn] + ((uint16_t*)m)[1] * ((uint16_t)(src[k - cn]) + (uint16_t)(src[k + cn])) + ((uint16_t*)m)[2] * src[k];
                dst[k + cn] = m[0] * src[k - cn] + m[1] * src[k] + m[2] * src[k + cn];
            }
        }
    }
}
template <typename ET, typename FT>
void hlineSmooth(const ET* src, int cn, const FT* m, int n, FT* dst, int len, int borderType)
{
    int pre_shift = n / 2;
    int post_shift = n - pre_shift;
    int i = 0;
    for (; i < min(pre_shift, len); i++, dst += cn) // Points that fall left from border
    {
        for (int k = 0; k < cn; k++)
            dst[k] = m[pre_shift-i] * src[k];
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
            for (int j = i - pre_shift, mid = 0; j < 0; j++, mid++)
            {
                int src_idx = borderInterpolate(j, len, borderType);
                for (int k = 0; k < cn; k++)
                    dst[k] = dst[k] + m[mid] * src[src_idx*cn + k];
            }
        int j, mid;
        for (j = 1, mid = pre_shift - i + 1; j < min(i + post_shift, len); j++, mid++)
            for (int k = 0; k < cn; k++)
                dst[k] = dst[k] + m[mid] * src[j*cn + k];
        if (borderType != BORDER_CONSTANT)
            for (; j < i + post_shift; j++, mid++)
            {
                int src_idx = borderInterpolate(j, len, borderType);
                for (int k = 0; k < cn; k++)
                    dst[k] = dst[k] + m[mid] * src[src_idx*cn + k];
            }
    }
    i *= cn;
    for (; i < (len - post_shift + 1)*cn; i++, src++, dst++)
    {
        *dst = m[0] * src[0];
        for (int j = 1; j < n; j++)
            *dst = *dst + m[j] * src[j*cn];
    }
    i /= cn;
    for (i -= pre_shift; i < len - pre_shift; i++, src += cn, dst += cn) // Points that fall right from border
    {
        for (int k = 0; k < cn; k++)
            dst[k] = m[0] * src[k];
        int j = 1;
        for (; j < len - i; j++)
            for (int k = 0; k < cn; k++)
                dst[k] = dst[k] + m[j] * src[j*cn + k];
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
            for (; j < n; j++)
            {
                int src_idx = borderInterpolate(i + j, len, borderType) - i;
                for (int k = 0; k < cn; k++)
                    dst[k] = dst[k] + m[j] * src[src_idx*cn + k];
            }
    }
}
template <>
void hlineSmooth<uint8_t, ufixedpoint16>(const uint8_t* src, int cn, const ufixedpoint16* m, int n, ufixedpoint16* dst, int len, int borderType)
{
    int pre_shift = n / 2;
    int post_shift = n - pre_shift;
    int i = 0;
    for (; i < min(pre_shift, len); i++, dst += cn) // Points that fall left from border
    {
        for (int k = 0; k < cn; k++)
            dst[k] = m[pre_shift - i] * src[k];
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
            for (int j = i - pre_shift, mid = 0; j < 0; j++, mid++)
            {
                int src_idx = borderInterpolate(j, len, borderType);
                for (int k = 0; k < cn; k++)
                    dst[k] = dst[k] + m[mid] * src[src_idx*cn + k];
            }
        int j, mid;
        for (j = 1, mid = pre_shift - i + 1; j < min(i + post_shift, len); j++, mid++)
            for (int k = 0; k < cn; k++)
                dst[k] = dst[k] + m[mid] * src[j*cn + k];
        if (borderType != BORDER_CONSTANT)
            for (; j < i + post_shift; j++, mid++)
            {
                int src_idx = borderInterpolate(j, len, borderType);
                for (int k = 0; k < cn; k++)
                    dst[k] = dst[k] + m[mid] * src[src_idx*cn + k];
            }
    }
    i *= cn;
    int lencn = (len - post_shift + 1)*cn;
#if CV_SIMD
    const int VECSZ = v_uint16::nlanes;
    for (; i <= lencn - VECSZ; i+=VECSZ, src+=VECSZ, dst+=VECSZ)
    {
        v_uint16 v_res0 = v_mul_wrap(vx_load_expand(src), vx_setall_u16(*((uint16_t*)m)));
        for (int j = 1; j < n; j++)
            v_res0 += v_mul_wrap(vx_load_expand(src + j * cn), vx_setall_u16(*((uint16_t*)(m + j))));
        v_store((uint16_t*)dst, v_res0);
    }
#endif
    for (; i < lencn; i++, src++, dst++)
    {
            *dst = m[0] * src[0];
            for (int j = 1; j < n; j++)
                *dst = *dst + m[j] * src[j*cn];
    }
    i /= cn;
    for (i -= pre_shift; i < len - pre_shift; i++, src += cn, dst += cn) // Points that fall right from border
    {
        for (int k = 0; k < cn; k++)
            dst[k] = m[0] * src[k];
        int j = 1;
        for (; j < len - i; j++)
            for (int k = 0; k < cn; k++)
                dst[k] = dst[k] + m[j] * src[j*cn + k];
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
            for (; j < n; j++)
            {
                int src_idx = borderInterpolate(i + j, len, borderType) - i;
                for (int k = 0; k < cn; k++)
                    dst[k] = dst[k] + m[j] * src[src_idx*cn + k];
            }
    }
}
template <typename ET, typename FT>
void hlineSmoothONa_yzy_a(const ET* src, int cn, const FT* m, int n, FT* dst, int len, int borderType)
{
    int pre_shift = n / 2;
    int post_shift = n - pre_shift;
    int i = 0;
    for (; i < min(pre_shift, len); i++, dst += cn) // Points that fall left from border
    {
        for (int k = 0; k < cn; k++)
            dst[k] = m[pre_shift - i] * src[k];
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
            for (int j = i - pre_shift, mid = 0; j < 0; j++, mid++)
            {
                int src_idx = borderInterpolate(j, len, borderType);
                for (int k = 0; k < cn; k++)
                    dst[k] = dst[k] + m[mid] * src[src_idx*cn + k];
            }
        int j, mid;
        for (j = 1, mid = pre_shift - i + 1; j < min(i + post_shift, len); j++, mid++)
            for (int k = 0; k < cn; k++)
                dst[k] = dst[k] + m[mid] * src[j*cn + k];
        if (borderType != BORDER_CONSTANT)
            for (; j < i + post_shift; j++, mid++)
            {
                int src_idx = borderInterpolate(j, len, borderType);
                for (int k = 0; k < cn; k++)
                    dst[k] = dst[k] + m[mid] * src[src_idx*cn + k];
            }
    }
    i *= cn;
    for (; i < (len - post_shift + 1)*cn; i++, src++, dst++)
    {
        *dst = m[pre_shift] * src[pre_shift*cn];
        for (int j = 0; j < pre_shift; j++)
            *dst = *dst + m[j] * src[j*cn] + m[j] * src[(n-1-j)*cn];
    }
    i /= cn;
    for (i -= pre_shift; i < len - pre_shift; i++, src += cn, dst += cn) // Points that fall right from border
    {
        for (int k = 0; k < cn; k++)
            dst[k] = m[0] * src[k];
        int j = 1;
        for (; j < len - i; j++)
            for (int k = 0; k < cn; k++)
                dst[k] = dst[k] + m[j] * src[j*cn + k];
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
            for (; j < n; j++)
            {
                int src_idx = borderInterpolate(i + j, len, borderType) - i;
                for (int k = 0; k < cn; k++)
                    dst[k] = dst[k] + m[j] * src[src_idx*cn + k];
            }
    }
}
template <>
void hlineSmoothONa_yzy_a<uint8_t, ufixedpoint16>(const uint8_t* src, int cn, const ufixedpoint16* m, int n, ufixedpoint16* dst, int len, int borderType)
{
    int pre_shift = n / 2;
    int post_shift = n - pre_shift;
    int i = 0;
    for (; i < min(pre_shift, len); i++, dst += cn) // Points that fall left from border
    {
        for (int k = 0; k < cn; k++)
            dst[k] = m[pre_shift - i] * src[k];
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
            for (int j = i - pre_shift, mid = 0; j < 0; j++, mid++)
            {
                int src_idx = borderInterpolate(j, len, borderType);
                for (int k = 0; k < cn; k++)
                    dst[k] = dst[k] + m[mid] * src[src_idx*cn + k];
            }
        int j, mid;
        for (j = 1, mid = pre_shift - i + 1; j < min(i + post_shift, len); j++, mid++)
            for (int k = 0; k < cn; k++)
                dst[k] = dst[k] + m[mid] * src[j*cn + k];
        if (borderType != BORDER_CONSTANT)
            for (; j < i + post_shift; j++, mid++)
            {
                int src_idx = borderInterpolate(j, len, borderType);
                for (int k = 0; k < cn; k++)
                    dst[k] = dst[k] + m[mid] * src[src_idx*cn + k];
            }
    }
    i *= cn;
    int lencn = (len - post_shift + 1)*cn;
#if CV_SIMD
    const int VECSZ = v_uint16::nlanes;
    for (; i <= lencn - VECSZ; i += VECSZ, src += VECSZ, dst += VECSZ)
    {
        v_uint16 v_res0 = v_mul_wrap(vx_load_expand(src + pre_shift * cn), vx_setall_u16(*((uint16_t*)(m + pre_shift))));
        for (int j = 0; j < pre_shift; j ++)
            v_res0 += v_mul_wrap(vx_load_expand(src + j * cn) + vx_load_expand(src + (n - 1 - j)*cn), vx_setall_u16(*((uint16_t*)(m + j))));
        v_store((uint16_t*)dst, v_res0);
    }
#endif
    for (; i < lencn; i++, src++, dst++)
    {
        *dst = m[pre_shift] * src[pre_shift*cn];
        for (int j = 0; j < pre_shift; j++)
            *dst = *dst + m[j] * src[j*cn] + m[j] * src[(n - 1 - j)*cn];
    }
    i /= cn;
    for (i -= pre_shift; i < len - pre_shift; i++, src += cn, dst += cn) // Points that fall right from border
    {
        for (int k = 0; k < cn; k++)
            dst[k] = m[0] * src[k];
        int j = 1;
        for (; j < len - i; j++)
            for (int k = 0; k < cn; k++)
                dst[k] = dst[k] + m[j] * src[j*cn + k];
        if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
            for (; j < n; j++)
            {
                int src_idx = borderInterpolate(i + j, len, borderType) - i;
                for (int k = 0; k < cn; k++)
                    dst[k] = dst[k] + m[j] * src[src_idx*cn + k];
            }
    }
}
template <typename ET, typename FT>
void vlineSmooth1N(const FT* const * src, const FT* m, int, ET* dst, int len)
{
    const FT* src0 = src[0];
    for (int i = 0; i < len; i++)
        dst[i] = *m * src0[i];
}
template <>
void vlineSmooth1N<uint8_t, ufixedpoint16>(const ufixedpoint16* const * src, const ufixedpoint16* m, int, uint8_t* dst, int len)
{
    const ufixedpoint16* src0 = src[0];
    int i = 0;
#if CV_SIMD
    const int VECSZ = v_uint16::nlanes;
    v_uint16 v_mul = vx_setall_u16(*((uint16_t*)m)<<1);
    for (; i <= len - VECSZ; i += VECSZ)
        v_rshr_pack_store<1>(dst + i, v_mul_hi(vx_load((uint16_t*)src0 + i), v_mul));
#endif
    for (; i < len; i++)
        dst[i] = m[0] * src0[i];
}
template <typename ET, typename FT>
void vlineSmooth1N1(const FT* const * src, const FT*, int, ET* dst, int len)
{
    const FT* src0 = src[0];
    for (int i = 0; i < len; i++)
        dst[i] = src0[i];
}
template <>
void vlineSmooth1N1<uint8_t, ufixedpoint16>(const ufixedpoint16* const * src, const ufixedpoint16*, int, uint8_t* dst, int len)
{
    const ufixedpoint16* src0 = src[0];
    int i = 0;
#if CV_SIMD
    const int VECSZ = v_uint16::nlanes;
    for (; i <= len - VECSZ; i += VECSZ)
        v_rshr_pack_store<8>(dst + i, vx_load((uint16_t*)(src0 + i)));
#endif
    for (; i < len; i++)
        dst[i] = src0[i];
}
template <typename ET, typename FT>
void vlineSmooth3N(const FT* const * src, const FT* m, int, ET* dst, int len)
{
    for (int i = 0; i < len; i++)
        dst[i] = m[0] * src[0][i] + m[1] * src[1][i] + m[2] * src[2][i];
}
template <>
void vlineSmooth3N<uint8_t, ufixedpoint16>(const ufixedpoint16* const * src, const ufixedpoint16* m, int, uint8_t* dst, int len)
{
    int i = 0;
#if CV_SIMD
    static const v_int16 v_128 = v_reinterpret_as_s16(vx_setall_u16((uint16_t)1 << 15));
    v_int32 v_128_4 = vx_setall_s32(128 << 16);
    const int VECSZ = v_uint16::nlanes;
    if (len >= VECSZ)
    {
        ufixedpoint32 val[] = { (m[0] + m[1] + m[2]) * ufixedpoint16((uint8_t)128) };
        v_128_4 = vx_setall_s32(*((int32_t*)val));
    }
    v_int16 v_mul01 = v_reinterpret_as_s16(vx_setall_u32(*((uint32_t*)m)));
    v_int16 v_mul2 = v_reinterpret_as_s16(vx_setall_u16(*((uint16_t*)(m + 2))));
    for (; i <= len - 4*VECSZ; i += 4*VECSZ)
    {
        v_int16 v_src00, v_src10, v_src01, v_src11, v_src02, v_src12, v_src03, v_src13;
        v_int16 v_tmp0, v_tmp1;

        const int16_t* src0 = (const int16_t*)src[0] + i;
        const int16_t* src1 = (const int16_t*)src[1] + i;
        v_src00 = vx_load(src0);
        v_src01 = vx_load(src0 + VECSZ);
        v_src02 = vx_load(src0 + 2*VECSZ);
        v_src03 = vx_load(src0 + 3*VECSZ);
        v_src10 = vx_load(src1);
        v_src11 = vx_load(src1 + VECSZ);
        v_src12 = vx_load(src1 + 2*VECSZ);
        v_src13 = vx_load(src1 + 3*VECSZ);
        v_zip(v_add_wrap(v_src00, v_128), v_add_wrap(v_src10, v_128), v_tmp0, v_tmp1);
        v_int32 v_res0 = v_dotprod(v_tmp0, v_mul01);
        v_int32 v_res1 = v_dotprod(v_tmp1, v_mul01);
        v_zip(v_add_wrap(v_src01, v_128), v_add_wrap(v_src11, v_128), v_tmp0, v_tmp1);
        v_int32 v_res2 = v_dotprod(v_tmp0, v_mul01);
        v_int32 v_res3 = v_dotprod(v_tmp1, v_mul01);
        v_zip(v_add_wrap(v_src02, v_128), v_add_wrap(v_src12, v_128), v_tmp0, v_tmp1);
        v_int32 v_res4 = v_dotprod(v_tmp0, v_mul01);
        v_int32 v_res5 = v_dotprod(v_tmp1, v_mul01);
        v_zip(v_add_wrap(v_src03, v_128), v_add_wrap(v_src13, v_128), v_tmp0, v_tmp1);
        v_int32 v_res6 = v_dotprod(v_tmp0, v_mul01);
        v_int32 v_res7 = v_dotprod(v_tmp1, v_mul01);

        v_int32 v_resj0, v_resj1;
        const int16_t* src2 = (const int16_t*)src[2] + i;
        v_src00 = vx_load(src2);
        v_src01 = vx_load(src2 + VECSZ);
        v_src02 = vx_load(src2 + 2*VECSZ);
        v_src03 = vx_load(src2 + 3*VECSZ);
        v_mul_expand(v_add_wrap(v_src00, v_128), v_mul2, v_resj0, v_resj1);
        v_res0 += v_resj0;
        v_res1 += v_resj1;
        v_mul_expand(v_add_wrap(v_src01, v_128), v_mul2, v_resj0, v_resj1);
        v_res2 += v_resj0;
        v_res3 += v_resj1;
        v_mul_expand(v_add_wrap(v_src02, v_128), v_mul2, v_resj0, v_resj1);
        v_res4 += v_resj0;
        v_res5 += v_resj1;
        v_mul_expand(v_add_wrap(v_src03, v_128), v_mul2, v_resj0, v_resj1);
        v_res6 += v_resj0;
        v_res7 += v_resj1;

        v_res0 += v_128_4;
        v_res1 += v_128_4;
        v_res2 += v_128_4;
        v_res3 += v_128_4;
        v_res4 += v_128_4;
        v_res5 += v_128_4;
        v_res6 += v_128_4;
        v_res7 += v_128_4;

        v_store(dst + i          , v_pack(v_reinterpret_as_u16(v_rshr_pack<16>(v_res0, v_res1)),
                                          v_reinterpret_as_u16(v_rshr_pack<16>(v_res2, v_res3))));
        v_store(dst + i + 2*VECSZ, v_pack(v_reinterpret_as_u16(v_rshr_pack<16>(v_res4, v_res5)),
                                          v_reinterpret_as_u16(v_rshr_pack<16>(v_res6, v_res7))));
    }
#endif
    for (; i < len; i++)
        dst[i] = m[0] * src[0][i] + m[1] * src[1][i] + m[2] * src[2][i];
}
template <typename ET, typename FT>
void vlineSmooth3N121(const FT* const * src, const FT*, int, ET* dst, int len)
{
    for (int i = 0; i < len; i++)
        dst[i] = (FT::WT(src[0][i]) >> 2) + (FT::WT(src[2][i]) >> 2) + (FT::WT(src[1][i]) >> 1);
}
template <>
void vlineSmooth3N121<uint8_t, ufixedpoint16>(const ufixedpoint16* const * src, const ufixedpoint16*, int, uint8_t* dst, int len)
{
    int i = 0;
#if CV_SIMD
    const int VECSZ = v_uint16::nlanes;
    for (; i <= len - 2*VECSZ; i += 2*VECSZ)
    {
        v_uint32 v_src00, v_src01, v_src02, v_src03, v_src10, v_src11, v_src12, v_src13, v_src20, v_src21, v_src22, v_src23;
        v_expand(vx_load((uint16_t*)(src[0]) + i), v_src00, v_src01);
        v_expand(vx_load((uint16_t*)(src[0]) + i + VECSZ), v_src02, v_src03);
        v_expand(vx_load((uint16_t*)(src[1]) + i), v_src10, v_src11);
        v_expand(vx_load((uint16_t*)(src[1]) + i + VECSZ), v_src12, v_src13);
        v_expand(vx_load((uint16_t*)(src[2]) + i), v_src20, v_src21);
        v_expand(vx_load((uint16_t*)(src[2]) + i + VECSZ), v_src22, v_src23);
        v_store(dst + i, v_pack(v_rshr_pack<10>(v_src00 + v_src20 + (v_src10 + v_src10), v_src01 + v_src21 + (v_src11 + v_src11)),
                                v_rshr_pack<10>(v_src02 + v_src22 + (v_src12 + v_src12), v_src03 + v_src23 + (v_src13 + v_src13))));
    }
#endif
    for (; i < len; i++)
        dst[i] = (((uint32_t)(((uint16_t*)(src[0]))[i]) + (uint32_t)(((uint16_t*)(src[2]))[i]) + ((uint32_t)(((uint16_t*)(src[1]))[i]) << 1)) + (1 << 9)) >> 10;
}
template <typename ET, typename FT>
void vlineSmooth5N(const FT* const * src, const FT* m, int, ET* dst, int len)
{
    for (int i = 0; i < len; i++)
        dst[i] = m[0] * src[0][i] + m[1] * src[1][i] + m[2] * src[2][i] + m[3] * src[3][i] + m[4] * src[4][i];
}
template <>
void vlineSmooth5N<uint8_t, ufixedpoint16>(const ufixedpoint16* const * src, const ufixedpoint16* m, int, uint8_t* dst, int len)
{
    int i = 0;
#if CV_SIMD
    const int VECSZ = v_uint16::nlanes;
    if (len >= 4 * VECSZ)
    {
        ufixedpoint32 val[] = { (m[0] + m[1] + m[2] + m[3] + m[4]) * ufixedpoint16((uint8_t)128) };
        v_int32 v_128_4 = vx_setall_s32(*((int32_t*)val));
        static const v_int16 v_128 = v_reinterpret_as_s16(vx_setall_u16((uint16_t)1 << 15));
        v_int16 v_mul01 = v_reinterpret_as_s16(vx_setall_u32(*((uint32_t*)m)));
        v_int16 v_mul23 = v_reinterpret_as_s16(vx_setall_u32(*((uint32_t*)(m + 2))));
        v_int16 v_mul4 = v_reinterpret_as_s16(vx_setall_u16(*((uint16_t*)(m + 4))));
        for (; i <= len - 4*VECSZ; i += 4*VECSZ)
        {
            v_int16 v_src00, v_src10, v_src01, v_src11, v_src02, v_src12, v_src03, v_src13;
            v_int16 v_tmp0, v_tmp1;

            const int16_t* src0 = (const int16_t*)src[0] + i;
            const int16_t* src1 = (const int16_t*)src[1] + i;
            v_src00 = vx_load(src0);
            v_src01 = vx_load(src0 + VECSZ);
            v_src02 = vx_load(src0 + 2*VECSZ);
            v_src03 = vx_load(src0 + 3*VECSZ);
            v_src10 = vx_load(src1);
            v_src11 = vx_load(src1 + VECSZ);
            v_src12 = vx_load(src1 + 2*VECSZ);
            v_src13 = vx_load(src1 + 3*VECSZ);
            v_zip(v_add_wrap(v_src00, v_128), v_add_wrap(v_src10, v_128), v_tmp0, v_tmp1);
            v_int32 v_res0 = v_dotprod(v_tmp0, v_mul01);
            v_int32 v_res1 = v_dotprod(v_tmp1, v_mul01);
            v_zip(v_add_wrap(v_src01, v_128), v_add_wrap(v_src11, v_128), v_tmp0, v_tmp1);
            v_int32 v_res2 = v_dotprod(v_tmp0, v_mul01);
            v_int32 v_res3 = v_dotprod(v_tmp1, v_mul01);
            v_zip(v_add_wrap(v_src02, v_128), v_add_wrap(v_src12, v_128), v_tmp0, v_tmp1);
            v_int32 v_res4 = v_dotprod(v_tmp0, v_mul01);
            v_int32 v_res5 = v_dotprod(v_tmp1, v_mul01);
            v_zip(v_add_wrap(v_src03, v_128), v_add_wrap(v_src13, v_128), v_tmp0, v_tmp1);
            v_int32 v_res6 = v_dotprod(v_tmp0, v_mul01);
            v_int32 v_res7 = v_dotprod(v_tmp1, v_mul01);

            const int16_t* src2 = (const int16_t*)src[2] + i;
            const int16_t* src3 = (const int16_t*)src[3] + i;
            v_src00 = vx_load(src2);
            v_src01 = vx_load(src2 + VECSZ);
            v_src02 = vx_load(src2 + 2*VECSZ);
            v_src03 = vx_load(src2 + 3*VECSZ);
            v_src10 = vx_load(src3);
            v_src11 = vx_load(src3 + VECSZ);
            v_src12 = vx_load(src3 + 2*VECSZ);
            v_src13 = vx_load(src3 + 3*VECSZ);
            v_zip(v_add_wrap(v_src00, v_128), v_add_wrap(v_src10, v_128), v_tmp0, v_tmp1);
            v_res0 += v_dotprod(v_tmp0, v_mul23);
            v_res1 += v_dotprod(v_tmp1, v_mul23);
            v_zip(v_add_wrap(v_src01, v_128), v_add_wrap(v_src11, v_128), v_tmp0, v_tmp1);
            v_res2 += v_dotprod(v_tmp0, v_mul23);
            v_res3 += v_dotprod(v_tmp1, v_mul23);
            v_zip(v_add_wrap(v_src02, v_128), v_add_wrap(v_src12, v_128), v_tmp0, v_tmp1);
            v_res4 += v_dotprod(v_tmp0, v_mul23);
            v_res5 += v_dotprod(v_tmp1, v_mul23);
            v_zip(v_add_wrap(v_src03, v_128), v_add_wrap(v_src13, v_128), v_tmp0, v_tmp1);
            v_res6 += v_dotprod(v_tmp0, v_mul23);
            v_res7 += v_dotprod(v_tmp1, v_mul23);

            v_int32 v_resj0, v_resj1;
            const int16_t* src4 = (const int16_t*)src[4] + i;
            v_src00 = vx_load(src4);
            v_src01 = vx_load(src4 + VECSZ);
            v_src02 = vx_load(src4 + 2*VECSZ);
            v_src03 = vx_load(src4 + 3*VECSZ);
            v_mul_expand(v_add_wrap(v_src00, v_128), v_mul4, v_resj0, v_resj1);
            v_res0 += v_resj0;
            v_res1 += v_resj1;
            v_mul_expand(v_add_wrap(v_src01, v_128), v_mul4, v_resj0, v_resj1);
            v_res2 += v_resj0;
            v_res3 += v_resj1;
            v_mul_expand(v_add_wrap(v_src02, v_128), v_mul4, v_resj0, v_resj1);
            v_res4 += v_resj0;
            v_res5 += v_resj1;
            v_mul_expand(v_add_wrap(v_src03, v_128), v_mul4, v_resj0, v_resj1);
            v_res6 += v_resj0;
            v_res7 += v_resj1;

            v_res0 += v_128_4;
            v_res1 += v_128_4;
            v_res2 += v_128_4;
            v_res3 += v_128_4;
            v_res4 += v_128_4;
            v_res5 += v_128_4;
            v_res6 += v_128_4;
            v_res7 += v_128_4;

            v_store(dst + i          , v_pack(v_reinterpret_as_u16(v_rshr_pack<16>(v_res0, v_res1)),
                                              v_reinterpret_as_u16(v_rshr_pack<16>(v_res2, v_res3))));
            v_store(dst + i + 2*VECSZ, v_pack(v_reinterpret_as_u16(v_rshr_pack<16>(v_res4, v_res5)),
                                              v_reinterpret_as_u16(v_rshr_pack<16>(v_res6, v_res7))));
        }
    }
#endif
    for (; i < len; i++)
        dst[i] = m[0] * src[0][i] + m[1] * src[1][i] + m[2] * src[2][i] + m[3] * src[3][i] + m[4] * src[4][i];
}
template <typename ET, typename FT>
void vlineSmooth5N14641(const FT* const * src, const FT*, int, ET* dst, int len)
{
    for (int i = 0; i < len; i++)
        dst[i] = (FT::WT(src[2][i])*(uint8_t)6 + ((FT::WT(src[1][i]) + FT::WT(src[3][i]))<<2) + FT::WT(src[0][i]) + FT::WT(src[4][i])) >> 4;
}
template <>
void vlineSmooth5N14641<uint8_t, ufixedpoint16>(const ufixedpoint16* const * src, const ufixedpoint16*, int, uint8_t* dst, int len)
{
    int i = 0;
#if CV_SIMD
    v_uint32 v_6 = vx_setall_u32(6);
    const int VECSZ = v_uint16::nlanes;
    for (; i <= len - 2*VECSZ; i += 2*VECSZ)
    {
        v_uint32 v_src00, v_src10, v_src20, v_src30, v_src40;
        v_uint32 v_src01, v_src11, v_src21, v_src31, v_src41;
        v_uint32 v_src02, v_src12, v_src22, v_src32, v_src42;
        v_uint32 v_src03, v_src13, v_src23, v_src33, v_src43;
        v_expand(vx_load((uint16_t*)(src[0]) + i), v_src00, v_src01);
        v_expand(vx_load((uint16_t*)(src[0]) + i + VECSZ), v_src02, v_src03);
        v_expand(vx_load((uint16_t*)(src[1]) + i), v_src10, v_src11);
        v_expand(vx_load((uint16_t*)(src[1]) + i + VECSZ), v_src12, v_src13);
        v_expand(vx_load((uint16_t*)(src[2]) + i), v_src20, v_src21);
        v_expand(vx_load((uint16_t*)(src[2]) + i + VECSZ), v_src22, v_src23);
        v_expand(vx_load((uint16_t*)(src[3]) + i), v_src30, v_src31);
        v_expand(vx_load((uint16_t*)(src[3]) + i + VECSZ), v_src32, v_src33);
        v_expand(vx_load((uint16_t*)(src[4]) + i), v_src40, v_src41);
        v_expand(vx_load((uint16_t*)(src[4]) + i + VECSZ), v_src42, v_src43);
        v_store(dst + i, v_pack(v_rshr_pack<12>(v_src20*v_6 + ((v_src10 + v_src30) << 2) + v_src00 + v_src40,
                                                v_src21*v_6 + ((v_src11 + v_src31) << 2) + v_src01 + v_src41),
                                v_rshr_pack<12>(v_src22*v_6 + ((v_src12 + v_src32) << 2) + v_src02 + v_src42,
                                                v_src23*v_6 + ((v_src13 + v_src33) << 2) + v_src03 + v_src43)));
    }
#endif
    for (; i < len; i++)
        dst[i] = ((uint32_t)(((uint16_t*)(src[2]))[i]) * 6 +
                  (((uint32_t)(((uint16_t*)(src[1]))[i]) + (uint32_t)(((uint16_t*)(src[3]))[i])) << 2) +
                  (uint32_t)(((uint16_t*)(src[0]))[i]) + (uint32_t)(((uint16_t*)(src[4]))[i]) + (1 << 11)) >> 12;
}
template <typename ET, typename FT>
void vlineSmooth(const FT* const * src, const FT* m, int n, ET* dst, int len)
{
    for (int i = 0; i < len; i++)
    {
        typename FT::WT val = m[0] * src[0][i];
        for (int j = 1; j < n; j++)
            val = val + m[j] * src[j][i];
        dst[i] = val;
    }
}
template <>
void vlineSmooth<uint8_t, ufixedpoint16>(const ufixedpoint16* const * src, const ufixedpoint16* m, int n, uint8_t* dst, int len)
{
    int i = 0;
#if CV_SIMD
    static const v_int16 v_128 = v_reinterpret_as_s16(vx_setall_u16((uint16_t)1 << 15));
    v_int32 v_128_4 = vx_setall_s32(128 << 16);
    const int VECSZ = v_uint16::nlanes;
    if (len >= VECSZ)
    {
        ufixedpoint16 msum = m[0] + m[1];
        for (int j = 2; j < n; j++)
            msum = msum + m[j];
        ufixedpoint32 val[] = { msum * ufixedpoint16((uint8_t)128) };
        v_128_4 = vx_setall_s32(*((int32_t*)val));
    }
    for (; i <= len - 4*VECSZ; i += 4*VECSZ)
    {
        v_int16 v_src00, v_src10, v_src01, v_src11, v_src02, v_src12, v_src03, v_src13;
        v_int16 v_tmp0, v_tmp1;

        v_int16 v_mul = v_reinterpret_as_s16(vx_setall_u32(*((uint32_t*)m)));

        const int16_t* src0 = (const int16_t*)src[0] + i;
        const int16_t* src1 = (const int16_t*)src[1] + i;
        v_src00 = vx_load(src0);
        v_src01 = vx_load(src0 + VECSZ);
        v_src02 = vx_load(src0 + 2*VECSZ);
        v_src03 = vx_load(src0 + 3*VECSZ);
        v_src10 = vx_load(src1);
        v_src11 = vx_load(src1 + VECSZ);
        v_src12 = vx_load(src1 + 2*VECSZ);
        v_src13 = vx_load(src1 + 3*VECSZ);
        v_zip(v_add_wrap(v_src00, v_128), v_add_wrap(v_src10, v_128), v_tmp0, v_tmp1);
        v_int32 v_res0 = v_dotprod(v_tmp0, v_mul);
        v_int32 v_res1 = v_dotprod(v_tmp1, v_mul);
        v_zip(v_add_wrap(v_src01, v_128), v_add_wrap(v_src11, v_128), v_tmp0, v_tmp1);
        v_int32 v_res2 = v_dotprod(v_tmp0, v_mul);
        v_int32 v_res3 = v_dotprod(v_tmp1, v_mul);
        v_zip(v_add_wrap(v_src02, v_128), v_add_wrap(v_src12, v_128), v_tmp0, v_tmp1);
        v_int32 v_res4 = v_dotprod(v_tmp0, v_mul);
        v_int32 v_res5 = v_dotprod(v_tmp1, v_mul);
        v_zip(v_add_wrap(v_src03, v_128), v_add_wrap(v_src13, v_128), v_tmp0, v_tmp1);
        v_int32 v_res6 = v_dotprod(v_tmp0, v_mul);
        v_int32 v_res7 = v_dotprod(v_tmp1, v_mul);

        int j = 2;
        for (; j < n - 1; j+=2)
        {
            v_mul = v_reinterpret_as_s16(vx_setall_u32(*((uint32_t*)(m+j))));

            const int16_t* srcj0 = (const int16_t*)src[j] + i;
            const int16_t* srcj1 = (const int16_t*)src[j + 1] + i;
            v_src00 = vx_load(srcj0);
            v_src01 = vx_load(srcj0 + VECSZ);
            v_src02 = vx_load(srcj0 + 2*VECSZ);
            v_src03 = vx_load(srcj0 + 3*VECSZ);
            v_src10 = vx_load(srcj1);
            v_src11 = vx_load(srcj1 + VECSZ);
            v_src12 = vx_load(srcj1 + 2*VECSZ);
            v_src13 = vx_load(srcj1 + 3*VECSZ);
            v_zip(v_add_wrap(v_src00, v_128), v_add_wrap(v_src10, v_128), v_tmp0, v_tmp1);
            v_res0 += v_dotprod(v_tmp0, v_mul);
            v_res1 += v_dotprod(v_tmp1, v_mul);
            v_zip(v_add_wrap(v_src01, v_128), v_add_wrap(v_src11, v_128), v_tmp0, v_tmp1);
            v_res2 += v_dotprod(v_tmp0, v_mul);
            v_res3 += v_dotprod(v_tmp1, v_mul);
            v_zip(v_add_wrap(v_src02, v_128), v_add_wrap(v_src12, v_128), v_tmp0, v_tmp1);
            v_res4 += v_dotprod(v_tmp0, v_mul);
            v_res5 += v_dotprod(v_tmp1, v_mul);
            v_zip(v_add_wrap(v_src03, v_128), v_add_wrap(v_src13, v_128), v_tmp0, v_tmp1);
            v_res6 += v_dotprod(v_tmp0, v_mul);
            v_res7 += v_dotprod(v_tmp1, v_mul);
        }
        if(j < n)
        {
            v_int32 v_resj0, v_resj1;
            v_mul = v_reinterpret_as_s16(vx_setall_u16(*((uint16_t*)(m + j))));
            const int16_t* srcj = (const int16_t*)src[j] + i;
            v_src00 = vx_load(srcj);
            v_src01 = vx_load(srcj + VECSZ);
            v_src02 = vx_load(srcj + 2*VECSZ);
            v_src03 = vx_load(srcj + 3*VECSZ);
            v_mul_expand(v_add_wrap(v_src00, v_128), v_mul, v_resj0, v_resj1);
            v_res0 += v_resj0;
            v_res1 += v_resj1;
            v_mul_expand(v_add_wrap(v_src01, v_128), v_mul, v_resj0, v_resj1);
            v_res2 += v_resj0;
            v_res3 += v_resj1;
            v_mul_expand(v_add_wrap(v_src02, v_128), v_mul, v_resj0, v_resj1);
            v_res4 += v_resj0;
            v_res5 += v_resj1;
            v_mul_expand(v_add_wrap(v_src03, v_128), v_mul, v_resj0, v_resj1);
            v_res6 += v_resj0;
            v_res7 += v_resj1;
        }
        v_res0 += v_128_4;
        v_res1 += v_128_4;
        v_res2 += v_128_4;
        v_res3 += v_128_4;
        v_res4 += v_128_4;
        v_res5 += v_128_4;
        v_res6 += v_128_4;
        v_res7 += v_128_4;

        v_store(dst + i          , v_pack(v_reinterpret_as_u16(v_rshr_pack<16>(v_res0, v_res1)),
                                          v_reinterpret_as_u16(v_rshr_pack<16>(v_res2, v_res3))));
        v_store(dst + i + 2*VECSZ, v_pack(v_reinterpret_as_u16(v_rshr_pack<16>(v_res4, v_res5)),
                                          v_reinterpret_as_u16(v_rshr_pack<16>(v_res6, v_res7))));
    }
#endif
    for (; i < len; i++)
    {
        ufixedpoint32 val = m[0] * src[0][i];
        for (int j = 1; j < n; j++)
        {
            val = val + m[j] * src[j][i];
        }
        dst[i] = val;
    }
}
template <typename ET, typename FT>
void vlineSmoothONa_yzy_a(const FT* const * src, const FT* m, int n, ET* dst, int len)
{
    int pre_shift = n / 2;
    for (int i = 0; i < len; i++)
    {
        typename FT::WT val = m[pre_shift] * src[pre_shift][i];
        for (int j = 0; j < pre_shift; j++)
            val = val + m[j] * src[j][i] + m[j] * src[(n - 1 - j)][i];
        dst[i] = val;
    }
}
template <>
void vlineSmoothONa_yzy_a<uint8_t, ufixedpoint16>(const ufixedpoint16* const * src, const ufixedpoint16* m, int n, uint8_t* dst, int len)
{
    int i = 0;
#if CV_SIMD
    int pre_shift = n / 2;
    static const v_int16 v_128 = v_reinterpret_as_s16(vx_setall_u16((uint16_t)1 << 15));
    v_int32 v_128_4 = vx_setall_s32(128 << 16);
    const int VECSZ = v_uint16::nlanes;
    if (len >= VECSZ)
    {
        ufixedpoint16 msum = m[0] + m[pre_shift] + m[n - 1];
        for (int j = 1; j < pre_shift; j++)
            msum = msum + m[j] + m[n - 1 - j];
        ufixedpoint32 val[] = { msum * ufixedpoint16((uint8_t)128) };
        v_128_4 = vx_setall_s32(*((int32_t*)val));
    }
    for (; i <= len - 4*VECSZ; i += 4*VECSZ)
    {
        v_int16 v_src00, v_src10, v_src20, v_src30, v_src01, v_src11, v_src21, v_src31;
        v_int32 v_res0, v_res1, v_res2, v_res3, v_res4, v_res5, v_res6, v_res7;
        v_int16 v_tmp0, v_tmp1, v_tmp2, v_tmp3, v_tmp4, v_tmp5, v_tmp6, v_tmp7;

        v_int16 v_mul = v_reinterpret_as_s16(vx_setall_u16(*((uint16_t*)(m + pre_shift))));
        const int16_t* srcp = (const int16_t*)src[pre_shift] + i;
        v_src00 = vx_load(srcp);
        v_src10 = vx_load(srcp + VECSZ);
        v_src20 = vx_load(srcp + 2*VECSZ);
        v_src30 = vx_load(srcp + 3*VECSZ);
        v_mul_expand(v_add_wrap(v_src00, v_128), v_mul, v_res0, v_res1);
        v_mul_expand(v_add_wrap(v_src10, v_128), v_mul, v_res2, v_res3);
        v_mul_expand(v_add_wrap(v_src20, v_128), v_mul, v_res4, v_res5);
        v_mul_expand(v_add_wrap(v_src30, v_128), v_mul, v_res6, v_res7);

        int j = 0;
        for (; j < pre_shift; j++)
        {
            v_mul = v_reinterpret_as_s16(vx_setall_u16(*((uint16_t*)(m + j))));

            const int16_t* srcj0 = (const int16_t*)src[j] + i;
            const int16_t* srcj1 = (const int16_t*)src[n - 1 - j] + i;
            v_src00 = vx_load(srcj0);
            v_src10 = vx_load(srcj0 + VECSZ);
            v_src20 = vx_load(srcj0 + 2*VECSZ);
            v_src30 = vx_load(srcj0 + 3*VECSZ);
            v_src01 = vx_load(srcj1);
            v_src11 = vx_load(srcj1 + VECSZ);
            v_src21 = vx_load(srcj1 + 2*VECSZ);
            v_src31 = vx_load(srcj1 + 3*VECSZ);
            v_zip(v_add_wrap(v_src00, v_128), v_add_wrap(v_src01, v_128), v_tmp0, v_tmp1);
            v_res0 += v_dotprod(v_tmp0, v_mul);
            v_res1 += v_dotprod(v_tmp1, v_mul);
            v_zip(v_add_wrap(v_src10, v_128), v_add_wrap(v_src11, v_128), v_tmp2, v_tmp3);
            v_res2 += v_dotprod(v_tmp2, v_mul);
            v_res3 += v_dotprod(v_tmp3, v_mul);
            v_zip(v_add_wrap(v_src20, v_128), v_add_wrap(v_src21, v_128), v_tmp4, v_tmp5);
            v_res4 += v_dotprod(v_tmp4, v_mul);
            v_res5 += v_dotprod(v_tmp5, v_mul);
            v_zip(v_add_wrap(v_src30, v_128), v_add_wrap(v_src31, v_128), v_tmp6, v_tmp7);
            v_res6 += v_dotprod(v_tmp6, v_mul);
            v_res7 += v_dotprod(v_tmp7, v_mul);
        }

        v_res0 += v_128_4;
        v_res1 += v_128_4;
        v_res2 += v_128_4;
        v_res3 += v_128_4;
        v_res4 += v_128_4;
        v_res5 += v_128_4;
        v_res6 += v_128_4;
        v_res7 += v_128_4;

        v_store(dst + i          , v_pack(v_reinterpret_as_u16(v_rshr_pack<16>(v_res0, v_res1)),
                                          v_reinterpret_as_u16(v_rshr_pack<16>(v_res2, v_res3))));
        v_store(dst + i + 2*VECSZ, v_pack(v_reinterpret_as_u16(v_rshr_pack<16>(v_res4, v_res5)),
                                          v_reinterpret_as_u16(v_rshr_pack<16>(v_res6, v_res7))));
    }
#endif
    for (; i < len; i++)
    {
        ufixedpoint32 val = m[0] * src[0][i];
        for (int j = 1; j < n; j++)
        {
            val = val + m[j] * src[j][i];
        }
        dst[i] = val;
    }
}
template <typename ET, typename FT>
class fixedSmoothInvoker : public ParallelLoopBody
{
public:
    fixedSmoothInvoker(const ET* _src, size_t _src_stride, ET* _dst, size_t _dst_stride,
                       int _width, int _height, int _cn, const FT* _kx, int _kxlen, const FT* _ky, int _kylen, int _borderType) : ParallelLoopBody(),
                       src(_src), dst(_dst), src_stride(_src_stride), dst_stride(_dst_stride),
                       width(_width), height(_height), cn(_cn), kx(_kx), ky(_ky), kxlen(_kxlen), kylen(_kylen), borderType(_borderType)
    {
        if (kxlen == 1)
        {
            if (kx[0] == FT::one())
                hlineSmoothFunc = hlineSmooth1N1;
            else
                hlineSmoothFunc = hlineSmooth1N;
        }
        else if (kxlen == 3)
        {
            if (kx[0] == (FT::one()>>2)&&kx[1] == (FT::one()>>1)&&kx[2] == (FT::one()>>2))
                hlineSmoothFunc = hlineSmooth3N121;
            else if ((kx[0] - kx[2]).isZero())
                    hlineSmoothFunc = hlineSmooth3Naba;
            else
                hlineSmoothFunc = hlineSmooth3N;
        }
        else if (kxlen == 5)
        {
            if (kx[2] == (FT::one()*(uint8_t)3>>3) &&
                kx[1] == (FT::one()>>2) && kx[3] == (FT::one()>>2) &&
                kx[0] == (FT::one()>>4) && kx[4] == (FT::one()>>4))
                hlineSmoothFunc = hlineSmooth5N14641;
            else if (kx[0] == kx[4] && kx[1] == kx[3])
                hlineSmoothFunc = hlineSmooth5Nabcba;
            else
                hlineSmoothFunc = hlineSmooth5N;
        }
        else if (kxlen % 2 == 1)
        {
            hlineSmoothFunc = hlineSmoothONa_yzy_a;
            for (int i = 0; i < kxlen / 2; i++)
                if (!(kx[i] == kx[kxlen - 1 - i]))
                {
                    hlineSmoothFunc = hlineSmooth;
                    break;
                }
        }
        else
            hlineSmoothFunc = hlineSmooth;
        if (kylen == 1)
        {
            if (ky[0] == FT::one())
                vlineSmoothFunc = vlineSmooth1N1;
            else
                vlineSmoothFunc = vlineSmooth1N;
        }
        else if (kylen == 3)
        {
            if (ky[0] == (FT::one() >> 2) && ky[1] == (FT::one() >> 1) && ky[2] == (FT::one() >> 2))
                vlineSmoothFunc = vlineSmooth3N121;
            else
                vlineSmoothFunc = vlineSmooth3N;
        }
        else if (kylen == 5)
        {
            if (ky[2] == (FT::one() * (uint8_t)3 >> 3) &&
                ky[1] == (FT::one() >> 2) && ky[3] == (FT::one() >> 2) &&
                ky[0] == (FT::one() >> 4) && ky[4] == (FT::one() >> 4))
                vlineSmoothFunc = vlineSmooth5N14641;
            else
                vlineSmoothFunc = vlineSmooth5N;
        }
        else if (kylen % 2 == 1)
        {
            vlineSmoothFunc = vlineSmoothONa_yzy_a;
            for (int i = 0; i < kylen / 2; i++)
                if (!(ky[i] == ky[kylen - 1 - i]))
                {
                    vlineSmoothFunc = vlineSmooth;
                    break;
                }
        }
        else
            vlineSmoothFunc = vlineSmooth;
    }
    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        AutoBuffer<FT> _buf(width*cn*kylen);
        FT* buf = _buf.data();
        AutoBuffer<FT*> _ptrs(kylen*2);
        FT** ptrs = _ptrs.data();

        if (kylen == 1)
        {
            ptrs[0] = buf;
            for (int i = range.start; i < range.end; i++)
            {
                hlineSmoothFunc(src + i * src_stride, cn, kx, kxlen, ptrs[0], width, borderType);
                vlineSmoothFunc(ptrs, ky, kylen, dst + i * dst_stride, width*cn);
            }
        }
        else if (borderType != BORDER_CONSTANT)// If BORDER_CONSTANT out of border values are equal to zero and could be skipped
        {
            int pre_shift = kylen / 2;
            int post_shift = kylen - pre_shift - 1;
            // First line evaluation
            int idst = range.start;
            int ifrom = max(0, idst - pre_shift);
            int ito = idst + post_shift + 1;
            int i = ifrom;
            int bufline = 0;
            for (; i < min(ito, height); i++, bufline++)
            {
                ptrs[bufline+kylen] = ptrs[bufline] = buf + bufline * width*cn;
                hlineSmoothFunc(src + i * src_stride, cn, kx, kxlen, ptrs[bufline], width, borderType);
            }
            for (; i < ito; i++, bufline++)
            {
                int src_idx = borderInterpolate(i, height, borderType);
                if (src_idx < ifrom)
                {
                    ptrs[bufline + kylen] = ptrs[bufline] = buf + bufline * width*cn;
                    hlineSmoothFunc(src + src_idx * src_stride, cn, kx, kxlen, ptrs[bufline], width, borderType);
                }
                else
                {
                    ptrs[bufline + kylen] = ptrs[bufline] = ptrs[src_idx - ifrom];
                }
            }
            for (int j = idst - pre_shift; j < 0; j++)
            {
                int src_idx = borderInterpolate(j, height, borderType);
                if (src_idx >= ito)
                {
                    ptrs[2*kylen + j] = ptrs[kylen + j] = buf + (kylen + j) * width*cn;
                    hlineSmoothFunc(src + src_idx * src_stride, cn, kx, kxlen, ptrs[kylen + j], width, borderType);
                }
                else
                {
                    ptrs[2*kylen + j] = ptrs[kylen + j] = ptrs[src_idx];
                }
            }
            vlineSmoothFunc(ptrs + bufline, ky, kylen, dst + idst*dst_stride, width*cn); idst++;

            // border mode dependent part evaluation
            // i points to last src row to evaluate in convolution
            bufline %= kylen; ito = min(height, range.end + post_shift);
            for (; i < min(kylen, ito); i++, idst++)
            {
                ptrs[bufline + kylen] = ptrs[bufline] = buf + bufline * width*cn;
                hlineSmoothFunc(src + i * src_stride, cn, kx, kxlen, ptrs[bufline], width, borderType);
                bufline = (bufline + 1) % kylen;
                vlineSmoothFunc(ptrs + bufline, ky, kylen, dst + idst*dst_stride, width*cn);
            }
            // Points inside the border
            for (; i < ito; i++, idst++)
            {
                hlineSmoothFunc(src + i * src_stride, cn, kx, kxlen, ptrs[bufline], width, borderType);
                bufline = (bufline + 1) % kylen;
                vlineSmoothFunc(ptrs + bufline, ky, kylen, dst + idst*dst_stride, width*cn);
            }
            // Points that could fall below border
            for (; i < range.end + post_shift; i++, idst++)
            {
                int src_idx = borderInterpolate(i, height, borderType);
                if ((i - src_idx) > kylen)
                    hlineSmoothFunc(src + src_idx * src_stride, cn, kx, kxlen, ptrs[bufline], width, borderType);
                else
                    ptrs[bufline + kylen] = ptrs[bufline] = ptrs[(bufline + kylen - (i - src_idx)) % kylen];
                bufline = (bufline + 1) % kylen;
                vlineSmoothFunc(ptrs + bufline, ky, kylen, dst + idst*dst_stride, width*cn);
            }
        }
        else
        {
            int pre_shift = kylen / 2;
            int post_shift = kylen - pre_shift - 1;
            // First line evaluation
            int idst = range.start;
            int ifrom = idst - pre_shift;
            int ito = min(idst + post_shift + 1, height);
            int i = max(0, ifrom);
            int bufline = 0;
            for (; i < ito; i++, bufline++)
            {
                ptrs[bufline + kylen] = ptrs[bufline] = buf + bufline * width*cn;
                hlineSmoothFunc(src + i * src_stride, cn, kx, kxlen, ptrs[bufline], width, borderType);
            }

            if (bufline == 1)
                vlineSmooth1N(ptrs, ky - min(ifrom, 0), bufline, dst + idst*dst_stride, width*cn);
            else if (bufline == 3)
                vlineSmooth3N(ptrs, ky - min(ifrom, 0), bufline, dst + idst*dst_stride, width*cn);
            else if (bufline == 5)
                vlineSmooth5N(ptrs, ky - min(ifrom, 0), bufline, dst + idst*dst_stride, width*cn);
            else
                vlineSmooth(ptrs, ky - min(ifrom, 0), bufline, dst + idst*dst_stride, width*cn);
            idst++;

            // border mode dependent part evaluation
            // i points to last src row to evaluate in convolution
            bufline %= kylen; ito = min(height, range.end + post_shift);
            for (; i < min(kylen, ito); i++, idst++)
            {
                ptrs[bufline + kylen] = ptrs[bufline] = buf + bufline * width*cn;
                hlineSmoothFunc(src + i * src_stride, cn, kx, kxlen, ptrs[bufline], width, borderType);
                bufline++;
                if (bufline == 3)
                    vlineSmooth3N(ptrs, ky + kylen - bufline, i + 1, dst + idst*dst_stride, width*cn);
                else if (bufline == 5)
                    vlineSmooth5N(ptrs, ky + kylen - bufline, i + 1, dst + idst*dst_stride, width*cn);
                else
                    vlineSmooth(ptrs, ky + kylen - bufline, i + 1, dst + idst*dst_stride, width*cn);
                bufline %= kylen;
            }
            // Points inside the border
            if (i - max(0, ifrom) >= kylen)
            {
                for (; i < ito; i++, idst++)
                {
                    hlineSmoothFunc(src + i * src_stride, cn, kx, kxlen, ptrs[bufline], width, borderType);
                    bufline = (bufline + 1) % kylen;
                    vlineSmoothFunc(ptrs + bufline, ky, kylen, dst + idst*dst_stride, width*cn);
                }

                // Points that could fall below border
                // i points to first src row to evaluate in convolution
                bufline = (bufline + 1) % kylen;
                for (i = idst - pre_shift; i < range.end - pre_shift; i++, idst++, bufline++)
                    if (height - i == 3)
                        vlineSmooth3N(ptrs + bufline, ky, height - i, dst + idst*dst_stride, width*cn);
                    else if (height - i == 5)
                        vlineSmooth5N(ptrs + bufline, ky, height - i, dst + idst*dst_stride, width*cn);
                    else
                        vlineSmooth(ptrs + bufline, ky, height - i, dst + idst*dst_stride, width*cn);
            }
            else
            {
                // i points to first src row to evaluate in convolution
                for (i = idst - pre_shift; i < min(range.end - pre_shift, 0); i++, idst++)
                    if (height == 3)
                        vlineSmooth3N(ptrs, ky - i, height, dst + idst*dst_stride, width*cn);
                    else if (height == 5)
                        vlineSmooth5N(ptrs, ky - i, height, dst + idst*dst_stride, width*cn);
                    else
                        vlineSmooth(ptrs, ky - i, height, dst + idst*dst_stride, width*cn);
                for (; i < range.end - pre_shift; i++, idst++)
                    if (height - i == 3)
                        vlineSmooth3N(ptrs + i - max(0, ifrom), ky, height - i, dst + idst*dst_stride, width*cn);
                    else if (height - i == 5)
                        vlineSmooth5N(ptrs + i - max(0, ifrom), ky, height - i, dst + idst*dst_stride, width*cn);
                    else
                        vlineSmooth(ptrs + i - max(0, ifrom), ky, height - i, dst + idst*dst_stride, width*cn);
            }
        }
    }
private:
    const ET* src;
    ET* dst;
    size_t src_stride, dst_stride;
    int width, height, cn;
    const FT *kx, *ky;
    int kxlen, kylen;
    int borderType;
    void(*hlineSmoothFunc)(const ET* src, int cn, const FT* m, int n, FT* dst, int len, int borderType);
    void(*vlineSmoothFunc)(const FT* const * src, const FT* m, int n, ET* dst, int len);

    fixedSmoothInvoker(const fixedSmoothInvoker&);
    fixedSmoothInvoker& operator=(const fixedSmoothInvoker&);
};

static void getGaussianKernel(int n, double sigma, int ktype, Mat& res) { res = getGaussianKernel(n, sigma, ktype); }
template <typename T> static void getGaussianKernel(int n, double sigma, int, std::vector<T>& res) { res = getFixedpointGaussianKernel<T>(n, sigma); }

template <typename T>
static void createGaussianKernels( T & kx, T & ky, int type, Size &ksize,
                                   double sigma1, double sigma2 )
{
    int depth = CV_MAT_DEPTH(type);
    if( sigma2 <= 0 )
        sigma2 = sigma1;

    // automatic detection of kernel size from sigma
    if( ksize.width <= 0 && sigma1 > 0 )
        ksize.width = cvRound(sigma1*(depth == CV_8U ? 3 : 4)*2 + 1)|1;
    if( ksize.height <= 0 && sigma2 > 0 )
        ksize.height = cvRound(sigma2*(depth == CV_8U ? 3 : 4)*2 + 1)|1;

    CV_Assert( ksize.width  > 0 && ksize.width  % 2 == 1 &&
               ksize.height > 0 && ksize.height % 2 == 1 );

    sigma1 = std::max( sigma1, 0. );
    sigma2 = std::max( sigma2, 0. );

    getGaussianKernel( ksize.width, sigma1, std::max(depth, CV_32F), kx );
    if( ksize.height == ksize.width && std::abs(sigma1 - sigma2) < DBL_EPSILON )
        ky = kx;
    else
        getGaussianKernel( ksize.height, sigma2, std::max(depth, CV_32F), ky );
}

}

cv::Ptr<cv::FilterEngine> cv::createGaussianFilter( int type, Size ksize,
                                        double sigma1, double sigma2,
                                        int borderType )
{
    Mat kx, ky;
    createGaussianKernels(kx, ky, type, ksize, sigma1, sigma2);

    return createSeparableLinearFilter( type, type, kx, ky, Point(-1,-1), 0, borderType );
}

namespace cv
{
#ifdef HAVE_OPENCL

static bool ocl_GaussianBlur_8UC1(InputArray _src, OutputArray _dst, Size ksize, int ddepth,
                                  InputArray _kernelX, InputArray _kernelY, int borderType)
{
    const ocl::Device & dev = ocl::Device::getDefault();
    int type = _src.type(), sdepth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);

    if ( !(dev.isIntel() && (type == CV_8UC1) &&
         (_src.offset() == 0) && (_src.step() % 4 == 0) &&
         ((ksize.width == 5 && (_src.cols() % 4 == 0)) ||
         (ksize.width == 3 && (_src.cols() % 16 == 0) && (_src.rows() % 2 == 0)))) )
        return false;

    Mat kernelX = _kernelX.getMat().reshape(1, 1);
    if (kernelX.cols % 2 != 1)
        return false;
    Mat kernelY = _kernelY.getMat().reshape(1, 1);
    if (kernelY.cols % 2 != 1)
        return false;

    if (ddepth < 0)
        ddepth = sdepth;

    Size size = _src.size();
    size_t globalsize[2] = { 0, 0 };
    size_t localsize[2] = { 0, 0 };

    if (ksize.width == 3)
    {
        globalsize[0] = size.width / 16;
        globalsize[1] = size.height / 2;
    }
    else if (ksize.width == 5)
    {
        globalsize[0] = size.width / 4;
        globalsize[1] = size.height / 1;
    }

    const char * const borderMap[] = { "BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_REFLECT", 0, "BORDER_REFLECT_101" };
    char build_opts[1024];
    sprintf(build_opts, "-D %s %s%s", borderMap[borderType & ~BORDER_ISOLATED],
            ocl::kernelToStr(kernelX, CV_32F, "KERNEL_MATRIX_X").c_str(),
            ocl::kernelToStr(kernelY, CV_32F, "KERNEL_MATRIX_Y").c_str());

    ocl::Kernel kernel;

    if (ksize.width == 3)
        kernel.create("gaussianBlur3x3_8UC1_cols16_rows2", cv::ocl::imgproc::gaussianBlur3x3_oclsrc, build_opts);
    else if (ksize.width == 5)
        kernel.create("gaussianBlur5x5_8UC1_cols4", cv::ocl::imgproc::gaussianBlur5x5_oclsrc, build_opts);

    if (kernel.empty())
        return false;

    UMat src = _src.getUMat();
    _dst.create(size, CV_MAKETYPE(ddepth, cn));
    if (!(_dst.offset() == 0 && _dst.step() % 4 == 0))
        return false;
    UMat dst = _dst.getUMat();

    int idxArg = kernel.set(0, ocl::KernelArg::PtrReadOnly(src));
    idxArg = kernel.set(idxArg, (int)src.step);
    idxArg = kernel.set(idxArg, ocl::KernelArg::PtrWriteOnly(dst));
    idxArg = kernel.set(idxArg, (int)dst.step);
    idxArg = kernel.set(idxArg, (int)dst.rows);
    idxArg = kernel.set(idxArg, (int)dst.cols);

    return kernel.run(2, globalsize, (localsize[0] == 0) ? NULL : localsize, false);
}

#endif

#ifdef HAVE_OPENVX

namespace ovx {
    template <> inline bool skipSmallImages<VX_KERNEL_GAUSSIAN_3x3>(int w, int h) { return w*h < 320 * 240; }
}
static bool openvx_gaussianBlur(InputArray _src, OutputArray _dst, Size ksize,
                                double sigma1, double sigma2, int borderType)
{
    if (sigma2 <= 0)
        sigma2 = sigma1;
    // automatic detection of kernel size from sigma
    if (ksize.width <= 0 && sigma1 > 0)
        ksize.width = cvRound(sigma1*6 + 1) | 1;
    if (ksize.height <= 0 && sigma2 > 0)
        ksize.height = cvRound(sigma2*6 + 1) | 1;

    if (_src.type() != CV_8UC1 ||
        _src.cols() < 3 || _src.rows() < 3 ||
        ksize.width != 3 || ksize.height != 3)
        return false;

    sigma1 = std::max(sigma1, 0.);
    sigma2 = std::max(sigma2, 0.);

    if (!(sigma1 == 0.0 || (sigma1 - 0.8) < DBL_EPSILON) || !(sigma2 == 0.0 || (sigma2 - 0.8) < DBL_EPSILON) ||
        ovx::skipSmallImages<VX_KERNEL_GAUSSIAN_3x3>(_src.cols(), _src.rows()))
        return false;

    Mat src = _src.getMat();
    Mat dst = _dst.getMat();

    if ((borderType & BORDER_ISOLATED) == 0 && src.isSubmatrix())
        return false; //Process isolated borders only
    vx_enum border;
    switch (borderType & ~BORDER_ISOLATED)
    {
    case BORDER_CONSTANT:
        border = VX_BORDER_CONSTANT;
        break;
    case BORDER_REPLICATE:
        border = VX_BORDER_REPLICATE;
        break;
    default:
        return false;
    }

    try
    {
        ivx::Context ctx = ovx::getOpenVXContext();

        Mat a;
        if (dst.data != src.data)
            a = src;
        else
            src.copyTo(a);

        ivx::Image
            ia = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U8,
                ivx::Image::createAddressing(a.cols, a.rows, 1, (vx_int32)(a.step)), a.data),
            ib = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U8,
                ivx::Image::createAddressing(dst.cols, dst.rows, 1, (vx_int32)(dst.step)), dst.data);

        //ATTENTION: VX_CONTEXT_IMMEDIATE_BORDER attribute change could lead to strange issues in multi-threaded environments
        //since OpenVX standard says nothing about thread-safety for now
        ivx::border_t prevBorder = ctx.immediateBorder();
        ctx.setImmediateBorder(border, (vx_uint8)(0));
        ivx::IVX_CHECK_STATUS(vxuGaussian3x3(ctx, ia, ib));
        ctx.setImmediateBorder(prevBorder);
    }
    catch (const ivx::RuntimeError & e)
    {
        VX_DbgThrow(e.what());
    }
    catch (const ivx::WrapperError & e)
    {
        VX_DbgThrow(e.what());
    }
    return true;
}

#endif

#if 0 //defined HAVE_IPP
// IW 2017u2 has bug which doesn't allow use of partial inMem with tiling
#if IPP_DISABLE_GAUSSIANBLUR_PARALLEL
#define IPP_GAUSSIANBLUR_PARALLEL 0
#else
#define IPP_GAUSSIANBLUR_PARALLEL 1
#endif

#ifdef HAVE_IPP_IW

class ipp_gaussianBlurParallel: public ParallelLoopBody
{
public:
    ipp_gaussianBlurParallel(::ipp::IwiImage &src, ::ipp::IwiImage &dst, int kernelSize, float sigma, ::ipp::IwiBorderType &border, bool *pOk):
        m_src(src), m_dst(dst), m_kernelSize(kernelSize), m_sigma(sigma), m_border(border), m_pOk(pOk) {
        *m_pOk = true;
    }
    ~ipp_gaussianBlurParallel()
    {
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION_IPP();

        if(!*m_pOk)
            return;

        try
        {
            ::ipp::IwiTile tile = ::ipp::IwiRoi(0, range.start, m_dst.m_size.width, range.end - range.start);
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilterGaussian, m_src, m_dst, m_kernelSize, m_sigma, ::ipp::IwDefault(), m_border, tile);
        }
        catch(const ::ipp::IwException &)
        {
            *m_pOk = false;
            return;
        }
    }
private:
    ::ipp::IwiImage &m_src;
    ::ipp::IwiImage &m_dst;

    int m_kernelSize;
    float m_sigma;
    ::ipp::IwiBorderType &m_border;

    volatile bool *m_pOk;
    const ipp_gaussianBlurParallel& operator= (const ipp_gaussianBlurParallel&);
};

#endif

static bool ipp_GaussianBlur(InputArray _src, OutputArray _dst, Size ksize,
                   double sigma1, double sigma2, int borderType )
{
#ifdef HAVE_IPP_IW
    CV_INSTRUMENT_REGION_IPP();

#if IPP_VERSION_X100 < 201800 && ((defined _MSC_VER && defined _M_IX86) || (defined __GNUC__ && defined __i386__))
    CV_UNUSED(_src); CV_UNUSED(_dst); CV_UNUSED(ksize); CV_UNUSED(sigma1); CV_UNUSED(sigma2); CV_UNUSED(borderType);
    return false; // bug on ia32
#else
    if(sigma1 != sigma2)
        return false;

    if(sigma1 < FLT_EPSILON)
        return false;

    if(ksize.width != ksize.height)
        return false;

    // Acquire data and begin processing
    try
    {
        Mat src = _src.getMat();
        Mat dst = _dst.getMat();
        ::ipp::IwiImage       iwSrc      = ippiGetImage(src);
        ::ipp::IwiImage       iwDst      = ippiGetImage(dst);
        ::ipp::IwiBorderSize  borderSize = ::ipp::iwiSizeToBorderSize(ippiGetSize(ksize));
        ::ipp::IwiBorderType  ippBorder(ippiGetBorder(iwSrc, borderType, borderSize));
        if(!ippBorder)
            return false;

        const int threads = ippiSuggestThreadsNum(iwDst, 2);
        if(IPP_GAUSSIANBLUR_PARALLEL && threads > 1) {
            bool ok;
            ipp_gaussianBlurParallel invoker(iwSrc, iwDst, ksize.width, (float) sigma1, ippBorder, &ok);

            if(!ok)
                return false;
            const Range range(0, (int) iwDst.m_size.height);
            parallel_for_(range, invoker, threads*4);

            if(!ok)
                return false;
        } else {
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilterGaussian, iwSrc, iwDst, ksize.width, sigma1, ::ipp::IwDefault(), ippBorder);
        }
    }
    catch (const ::ipp::IwException &)
    {
        return false;
    }

    return true;
#endif
#else
    CV_UNUSED(_src); CV_UNUSED(_dst); CV_UNUSED(ksize); CV_UNUSED(sigma1); CV_UNUSED(sigma2); CV_UNUSED(borderType);
    return false;
#endif
}
#endif
}

void cv::GaussianBlur( InputArray _src, OutputArray _dst, Size ksize,
                   double sigma1, double sigma2,
                   int borderType )
{
    CV_INSTRUMENT_REGION();

    int type = _src.type();
    Size size = _src.size();
    _dst.create( size, type );

    if( (borderType & ~BORDER_ISOLATED) != BORDER_CONSTANT &&
        ((borderType & BORDER_ISOLATED) != 0 || !_src.getMat().isSubmatrix()) )
    {
        if( size.height == 1 )
            ksize.height = 1;
        if( size.width == 1 )
            ksize.width = 1;
    }

    if( ksize.width == 1 && ksize.height == 1 )
    {
        _src.copyTo(_dst);
        return;
    }

    bool useOpenCL = (ocl::isOpenCLActivated() && _dst.isUMat() && _src.dims() <= 2 &&
               ((ksize.width == 3 && ksize.height == 3) ||
               (ksize.width == 5 && ksize.height == 5)) &&
               _src.rows() > ksize.height && _src.cols() > ksize.width);
    CV_UNUSED(useOpenCL);

    int sdepth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);

    Mat kx, ky;
    createGaussianKernels(kx, ky, type, ksize, sigma1, sigma2);

    CV_OCL_RUN(useOpenCL, ocl_GaussianBlur_8UC1(_src, _dst, ksize, CV_MAT_DEPTH(type), kx, ky, borderType));

    CV_OCL_RUN(_dst.isUMat() && _src.dims() <= 2 && (size_t)_src.rows() > kx.total() && (size_t)_src.cols() > kx.total(),
               ocl_sepFilter2D(_src, _dst, sdepth, kx, ky, Point(-1, -1), 0, borderType))

    Mat src = _src.getMat();
    Mat dst = _dst.getMat();

    Point ofs;
    Size wsz(src.cols, src.rows);
    if(!(borderType & BORDER_ISOLATED))
        src.locateROI( wsz, ofs );

    CALL_HAL(gaussianBlur, cv_hal_gaussianBlur, src.ptr(), src.step, dst.ptr(), dst.step, src.cols, src.rows, sdepth, cn,
             ofs.x, ofs.y, wsz.width - src.cols - ofs.x, wsz.height - src.rows - ofs.y, ksize.width, ksize.height,
             sigma1, sigma2, borderType&~BORDER_ISOLATED);

    CV_OVX_RUN(true,
               openvx_gaussianBlur(src, dst, ksize, sigma1, sigma2, borderType))

    //CV_IPP_RUN_FAST(ipp_GaussianBlur(src, dst, ksize, sigma1, sigma2, borderType));

    if(sdepth == CV_8U && ((borderType & BORDER_ISOLATED) || !_src.getMat().isSubmatrix()))
    {
        std::vector<ufixedpoint16> fkx, fky;
        createGaussianKernels(fkx, fky, type, ksize, sigma1, sigma2);
        if (src.data == dst.data)
            src = src.clone();
        fixedSmoothInvoker<uint8_t, ufixedpoint16> invoker(src.ptr<uint8_t>(), src.step1(), dst.ptr<uint8_t>(), dst.step1(), dst.cols, dst.rows, dst.channels(), &fkx[0], (int)fkx.size(), &fky[0], (int)fky.size(), borderType & ~BORDER_ISOLATED);
        parallel_for_(Range(0, dst.rows), invoker, std::max(1, std::min(getNumThreads(), getNumberOfCPUs())));
        return;
    }

    sepFilter2D(src, dst, sdepth, kx, ky, Point(-1, -1), 0, borderType);
}

/****************************************************************************************\
                                   Bilateral Filtering
\****************************************************************************************/

namespace cv
{

class BilateralFilter_8u_Invoker :
    public ParallelLoopBody
{
public:
    BilateralFilter_8u_Invoker(Mat& _dest, const Mat& _temp, int _radius, int _maxk,
        int* _space_ofs, float *_space_weight, float *_color_weight) :
        temp(&_temp), dest(&_dest), radius(_radius),
        maxk(_maxk), space_ofs(_space_ofs), space_weight(_space_weight), color_weight(_color_weight)
    {
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        int i, j, cn = dest->channels(), k;
        Size size = dest->size();
#if CV_SIMD128
        int CV_DECL_ALIGNED(16) buf[4];
        bool haveSIMD128 = hasSIMD128();
#endif

        for( i = range.start; i < range.end; i++ )
        {
            const uchar* sptr = temp->ptr(i+radius) + radius*cn;
            uchar* dptr = dest->ptr(i);

            if( cn == 1 )
            {
                for( j = 0; j < size.width; j++ )
                {
                    float sum = 0, wsum = 0;
                    int val0 = sptr[j];
                    k = 0;
#if CV_SIMD128
                    if( haveSIMD128 )
                    {
                        v_float32x4 _val0 = v_setall_f32(static_cast<float>(val0));
                        v_float32x4 vsumw = v_setzero_f32();
                        v_float32x4 vsumc = v_setzero_f32();

                        for( ; k <= maxk - 4; k += 4 )
                        {
                            v_float32x4 _valF = v_float32x4(sptr[j + space_ofs[k]],
                                sptr[j + space_ofs[k + 1]],
                                sptr[j + space_ofs[k + 2]],
                                sptr[j + space_ofs[k + 3]]);
                            v_float32x4 _val = v_abs(_valF - _val0);
                            v_store(buf, v_round(_val));

                            v_float32x4 _cw = v_float32x4(color_weight[buf[0]],
                                color_weight[buf[1]],
                                color_weight[buf[2]],
                                color_weight[buf[3]]);
                            v_float32x4 _sw = v_load(space_weight+k);
#if defined(_MSC_VER) && _MSC_VER == 1700/* MSVS 2012 */ && CV_AVX
                            // details: https://github.com/opencv/opencv/issues/11004
                            vsumw += _cw * _sw;
                            vsumc += _cw * _sw * _valF;
#else
                            v_float32x4 _w = _cw * _sw;
                            _cw = _w * _valF;

                            vsumw += _w;
                            vsumc += _cw;
#endif
                        }
                        float *bufFloat = (float*)buf;
                        v_float32x4 sum4 = v_reduce_sum4(vsumw, vsumc, vsumw, vsumc);
                        v_store(bufFloat, sum4);
                        sum += bufFloat[1];
                        wsum += bufFloat[0];
                    }
#endif
                    for( ; k < maxk; k++ )
                    {
                        int val = sptr[j + space_ofs[k]];
                        float w = space_weight[k]*color_weight[std::abs(val - val0)];
                        sum += val*w;
                        wsum += w;
                    }
                    // overflow is not possible here => there is no need to use cv::saturate_cast
                    CV_DbgAssert(fabs(wsum) > 0);
                    dptr[j] = (uchar)cvRound(sum/wsum);
                }
            }
            else
            {
                assert( cn == 3 );
                for( j = 0; j < size.width*3; j += 3 )
                {
                    float sum_b = 0, sum_g = 0, sum_r = 0, wsum = 0;
                    int b0 = sptr[j], g0 = sptr[j+1], r0 = sptr[j+2];
                    k = 0;
#if CV_SIMD128
                    if( haveSIMD128 )
                    {
                        v_float32x4 vsumw = v_setzero_f32();
                        v_float32x4 vsumb = v_setzero_f32();
                        v_float32x4 vsumg = v_setzero_f32();
                        v_float32x4 vsumr = v_setzero_f32();
                        const v_float32x4 _b0 = v_setall_f32(static_cast<float>(b0));
                        const v_float32x4 _g0 = v_setall_f32(static_cast<float>(g0));
                        const v_float32x4 _r0 = v_setall_f32(static_cast<float>(r0));

                        for( ; k <= maxk - 4; k += 4 )
                        {
                            const uchar* const sptr_k0  = sptr + j + space_ofs[k];
                            const uchar* const sptr_k1  = sptr + j + space_ofs[k+1];
                            const uchar* const sptr_k2  = sptr + j + space_ofs[k+2];
                            const uchar* const sptr_k3  = sptr + j + space_ofs[k+3];

                            v_float32x4 __b = v_cvt_f32(v_reinterpret_as_s32(v_load_expand_q(sptr_k0)));
                            v_float32x4 __g = v_cvt_f32(v_reinterpret_as_s32(v_load_expand_q(sptr_k1)));
                            v_float32x4 __r = v_cvt_f32(v_reinterpret_as_s32(v_load_expand_q(sptr_k2)));
                            v_float32x4 __z = v_cvt_f32(v_reinterpret_as_s32(v_load_expand_q(sptr_k3)));
                            v_float32x4 _b, _g, _r, _z;

                            v_transpose4x4(__b, __g, __r, __z, _b, _g, _r, _z);

                            v_float32x4 bt = v_abs(_b -_b0);
                            v_float32x4 gt = v_abs(_g -_g0);
                            v_float32x4 rt = v_abs(_r -_r0);

                            bt = rt + bt + gt;
                            v_store(buf, v_round(bt));

                            v_float32x4 _w  = v_float32x4(color_weight[buf[0]],color_weight[buf[1]],
                                                    color_weight[buf[2]],color_weight[buf[3]]);
                            v_float32x4 _sw = v_load(space_weight+k);

#if defined(_MSC_VER) && _MSC_VER == 1700/* MSVS 2012 */ && CV_AVX
                            // details: https://github.com/opencv/opencv/issues/11004
                            vsumw += _w * _sw;
                            vsumb += _w * _sw * _b;
                            vsumg += _w * _sw * _g;
                            vsumr += _w * _sw * _r;
#else
                            _w *= _sw;
                            _b *=  _w;
                            _g *=  _w;
                            _r *=  _w;

                            vsumw += _w;
                            vsumb += _b;
                            vsumg += _g;
                            vsumr += _r;
#endif
                        }
                        float *bufFloat = (float*)buf;
                        v_float32x4 sum4 = v_reduce_sum4(vsumw, vsumb, vsumg, vsumr);
                        v_store(bufFloat, sum4);
                        wsum += bufFloat[0];
                        sum_b += bufFloat[1];
                        sum_g += bufFloat[2];
                        sum_r += bufFloat[3];
                    }
#endif

                    for( ; k < maxk; k++ )
                    {
                        const uchar* sptr_k = sptr + j + space_ofs[k];
                        int b = sptr_k[0], g = sptr_k[1], r = sptr_k[2];
                        float w = space_weight[k]*color_weight[std::abs(b - b0) +
                                                               std::abs(g - g0) + std::abs(r - r0)];
                        sum_b += b*w; sum_g += g*w; sum_r += r*w;
                        wsum += w;
                    }
                    CV_DbgAssert(fabs(wsum) > 0);
                    wsum = 1.f/wsum;
                    b0 = cvRound(sum_b*wsum);
                    g0 = cvRound(sum_g*wsum);
                    r0 = cvRound(sum_r*wsum);
                    dptr[j] = (uchar)b0; dptr[j+1] = (uchar)g0; dptr[j+2] = (uchar)r0;
                }
            }
        }
    }

private:
    const Mat *temp;
    Mat *dest;
    int radius, maxk, *space_ofs;
    float *space_weight, *color_weight;
};

#ifdef HAVE_OPENCL

static bool ocl_bilateralFilter_8u(InputArray _src, OutputArray _dst, int d,
                                   double sigma_color, double sigma_space,
                                   int borderType)
{
#ifdef __ANDROID__
    if (ocl::Device::getDefault().isNVidia())
        return false;
#endif

    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    int i, j, maxk, radius;

    if (depth != CV_8U || cn > 4)
        return false;

    if (sigma_color <= 0)
        sigma_color = 1;
    if (sigma_space <= 0)
        sigma_space = 1;

    double gauss_color_coeff = -0.5 / (sigma_color * sigma_color);
    double gauss_space_coeff = -0.5 / (sigma_space * sigma_space);

    if ( d <= 0 )
        radius = cvRound(sigma_space * 1.5);
    else
        radius = d / 2;
    radius = MAX(radius, 1);
    d = radius * 2 + 1;

    UMat src = _src.getUMat(), dst = _dst.getUMat(), temp;
    if (src.u == dst.u)
        return false;

    copyMakeBorder(src, temp, radius, radius, radius, radius, borderType);
    std::vector<float> _space_weight(d * d);
    std::vector<int> _space_ofs(d * d);
    float * const space_weight = &_space_weight[0];
    int * const space_ofs = &_space_ofs[0];

    // initialize space-related bilateral filter coefficients
    for( i = -radius, maxk = 0; i <= radius; i++ )
        for( j = -radius; j <= radius; j++ )
        {
            double r = std::sqrt((double)i * i + (double)j * j);
            if ( r > radius )
                continue;
            space_weight[maxk] = (float)std::exp(r * r * gauss_space_coeff);
            space_ofs[maxk++] = (int)(i * temp.step + j * cn);
        }

    char cvt[3][40];
    String cnstr = cn > 1 ? format("%d", cn) : "";
    String kernelName("bilateral");
    size_t sizeDiv = 1;
    if ((ocl::Device::getDefault().isIntel()) &&
        (ocl::Device::getDefault().type() == ocl::Device::TYPE_GPU))
    {
            //Intel GPU
            if (dst.cols % 4 == 0 && cn == 1) // For single channel x4 sized images.
            {
                kernelName = "bilateral_float4";
                sizeDiv = 4;
            }
     }
     ocl::Kernel k(kernelName.c_str(), ocl::imgproc::bilateral_oclsrc,
            format("-D radius=%d -D maxk=%d -D cn=%d -D int_t=%s -D uint_t=uint%s -D convert_int_t=%s"
            " -D uchar_t=%s -D float_t=%s -D convert_float_t=%s -D convert_uchar_t=%s -D gauss_color_coeff=(float)%f",
            radius, maxk, cn, ocl::typeToStr(CV_32SC(cn)), cnstr.c_str(),
            ocl::convertTypeStr(CV_8U, CV_32S, cn, cvt[0]),
            ocl::typeToStr(type), ocl::typeToStr(CV_32FC(cn)),
            ocl::convertTypeStr(CV_32S, CV_32F, cn, cvt[1]),
            ocl::convertTypeStr(CV_32F, CV_8U, cn, cvt[2]), gauss_color_coeff));
    if (k.empty())
        return false;

    Mat mspace_weight(1, d * d, CV_32FC1, space_weight);
    Mat mspace_ofs(1, d * d, CV_32SC1, space_ofs);
    UMat ucolor_weight, uspace_weight, uspace_ofs;

    mspace_weight.copyTo(uspace_weight);
    mspace_ofs.copyTo(uspace_ofs);

    k.args(ocl::KernelArg::ReadOnlyNoSize(temp), ocl::KernelArg::WriteOnly(dst),
           ocl::KernelArg::PtrReadOnly(uspace_weight),
           ocl::KernelArg::PtrReadOnly(uspace_ofs));

    size_t globalsize[2] = { (size_t)dst.cols / sizeDiv, (size_t)dst.rows };
    return k.run(2, globalsize, NULL, false);
}

#endif
static void
bilateralFilter_8u( const Mat& src, Mat& dst, int d,
    double sigma_color, double sigma_space,
    int borderType )
{
    int cn = src.channels();
    int i, j, maxk, radius;
    Size size = src.size();

    CV_Assert( (src.type() == CV_8UC1 || src.type() == CV_8UC3) && src.data != dst.data );

    if( sigma_color <= 0 )
        sigma_color = 1;
    if( sigma_space <= 0 )
        sigma_space = 1;

    double gauss_color_coeff = -0.5/(sigma_color*sigma_color);
    double gauss_space_coeff = -0.5/(sigma_space*sigma_space);

    if( d <= 0 )
        radius = cvRound(sigma_space*1.5);
    else
        radius = d/2;
    radius = MAX(radius, 1);
    d = radius*2 + 1;

    Mat temp;
    copyMakeBorder( src, temp, radius, radius, radius, radius, borderType );

    std::vector<float> _color_weight(cn*256);
    std::vector<float> _space_weight(d*d);
    std::vector<int> _space_ofs(d*d);
    float* color_weight = &_color_weight[0];
    float* space_weight = &_space_weight[0];
    int* space_ofs = &_space_ofs[0];

    // initialize color-related bilateral filter coefficients

    for( i = 0; i < 256*cn; i++ )
        color_weight[i] = (float)std::exp(i*i*gauss_color_coeff);

    // initialize space-related bilateral filter coefficients
    for( i = -radius, maxk = 0; i <= radius; i++ )
    {
        j = -radius;

        for( ; j <= radius; j++ )
        {
            double r = std::sqrt((double)i*i + (double)j*j);
            if( r > radius )
                continue;
            space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
            space_ofs[maxk++] = (int)(i*temp.step + j*cn);
        }
    }

    BilateralFilter_8u_Invoker body(dst, temp, radius, maxk, space_ofs, space_weight, color_weight);
    parallel_for_(Range(0, size.height), body, dst.total()/(double)(1<<16));
}


class BilateralFilter_32f_Invoker :
    public ParallelLoopBody
{
public:

    BilateralFilter_32f_Invoker(int _cn, int _radius, int _maxk, int *_space_ofs,
        const Mat& _temp, Mat& _dest, float _scale_index, float *_space_weight, float *_expLUT) :
        cn(_cn), radius(_radius), maxk(_maxk), space_ofs(_space_ofs),
        temp(&_temp), dest(&_dest), scale_index(_scale_index), space_weight(_space_weight), expLUT(_expLUT)
    {
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        int i, j, k;
        Size size = dest->size();
#if CV_SIMD128
        int CV_DECL_ALIGNED(16) idxBuf[4];
        bool haveSIMD128 = hasSIMD128();
#endif

        for( i = range.start; i < range.end; i++ )
        {
            const float* sptr = temp->ptr<float>(i+radius) + radius*cn;
            float* dptr = dest->ptr<float>(i);

            if( cn == 1 )
            {
                for( j = 0; j < size.width; j++ )
                {
                    float sum = 0, wsum = 0;
                    float val0 = sptr[j];
                    k = 0;
#if CV_SIMD128
                    if( haveSIMD128 )
                    {
                        v_float32x4 vecwsum = v_setzero_f32();
                        v_float32x4 vecvsum = v_setzero_f32();
                        const v_float32x4 _val0 = v_setall_f32(sptr[j]);
                        const v_float32x4 _scale_index = v_setall_f32(scale_index);

                        for (; k <= maxk - 4; k += 4)
                        {
                            v_float32x4 _sw = v_load(space_weight + k);
                            v_float32x4 _val = v_float32x4(sptr[j + space_ofs[k]],
                                sptr[j + space_ofs[k + 1]],
                                sptr[j + space_ofs[k + 2]],
                                sptr[j + space_ofs[k + 3]]);
                            v_float32x4 _alpha = v_abs(_val - _val0) * _scale_index;

                            v_int32x4 _idx = v_round(_alpha);
                            v_store(idxBuf, _idx);
                            _alpha -= v_cvt_f32(_idx);

                            v_float32x4 _explut = v_float32x4(expLUT[idxBuf[0]],
                                expLUT[idxBuf[1]],
                                expLUT[idxBuf[2]],
                                expLUT[idxBuf[3]]);
                            v_float32x4 _explut1 = v_float32x4(expLUT[idxBuf[0] + 1],
                                expLUT[idxBuf[1] + 1],
                                expLUT[idxBuf[2] + 1],
                                expLUT[idxBuf[3] + 1]);

                            v_float32x4 _w = _sw * (_explut + (_alpha * (_explut1 - _explut)));
                            _val *= _w;

                            vecwsum += _w;
                            vecvsum += _val;
                        }
                        float *bufFloat = (float*)idxBuf;
                        v_float32x4 sum4 = v_reduce_sum4(vecwsum, vecvsum, vecwsum, vecvsum);
                        v_store(bufFloat, sum4);
                        sum += bufFloat[1];
                        wsum += bufFloat[0];
                    }
#endif

                    for( ; k < maxk; k++ )
                    {
                        float val = sptr[j + space_ofs[k]];
                        float alpha = (float)(std::abs(val - val0)*scale_index);
                        int idx = cvFloor(alpha);
                        alpha -= idx;
                        float w = space_weight[k]*(expLUT[idx] + alpha*(expLUT[idx+1] - expLUT[idx]));
                        sum += val*w;
                        wsum += w;
                    }
                    CV_DbgAssert(fabs(wsum) > 0);
                    dptr[j] = (float)(sum/wsum);
                }
            }
            else
            {
                CV_Assert( cn == 3 );
                for( j = 0; j < size.width*3; j += 3 )
                {
                    float sum_b = 0, sum_g = 0, sum_r = 0, wsum = 0;
                    float b0 = sptr[j], g0 = sptr[j+1], r0 = sptr[j+2];
                    k = 0;
#if CV_SIMD128
                    if( haveSIMD128 )
                    {
                        v_float32x4 sumw = v_setzero_f32();
                        v_float32x4 sumb = v_setzero_f32();
                        v_float32x4 sumg = v_setzero_f32();
                        v_float32x4 sumr = v_setzero_f32();
                        const v_float32x4 _b0 = v_setall_f32(b0);
                        const v_float32x4 _g0 = v_setall_f32(g0);
                        const v_float32x4 _r0 = v_setall_f32(r0);
                        const v_float32x4 _scale_index = v_setall_f32(scale_index);

                        for( ; k <= maxk-4; k += 4 )
                        {
                            v_float32x4 _sw = v_load(space_weight + k);

                            const float* const sptr_k0 = sptr + j + space_ofs[k];
                            const float* const sptr_k1 = sptr + j + space_ofs[k+1];
                            const float* const sptr_k2 = sptr + j + space_ofs[k+2];
                            const float* const sptr_k3 = sptr + j + space_ofs[k+3];

                            v_float32x4 _v0 = v_load(sptr_k0);
                            v_float32x4 _v1 = v_load(sptr_k1);
                            v_float32x4 _v2 = v_load(sptr_k2);
                            v_float32x4 _v3 = v_load(sptr_k3);
                            v_float32x4 _b, _g, _r, _dummy;

                            v_transpose4x4(_v0, _v1, _v2, _v3, _b, _g, _r, _dummy);

                            v_float32x4 _bt = v_abs(_b - _b0);
                            v_float32x4 _gt = v_abs(_g - _g0);
                            v_float32x4 _rt = v_abs(_r - _r0);
                            v_float32x4 _alpha = _scale_index * (_bt + _gt + _rt);

                            v_int32x4 _idx = v_round(_alpha);
                            v_store((int*)idxBuf, _idx);
                            _alpha -= v_cvt_f32(_idx);

                            v_float32x4 _explut = v_float32x4(expLUT[idxBuf[0]],
                                expLUT[idxBuf[1]],
                                expLUT[idxBuf[2]],
                                expLUT[idxBuf[3]]);
                            v_float32x4 _explut1 = v_float32x4(expLUT[idxBuf[0] + 1],
                                expLUT[idxBuf[1] + 1],
                                expLUT[idxBuf[2] + 1],
                                expLUT[idxBuf[3] + 1]);

                            v_float32x4 _w = _sw * (_explut + (_alpha * (_explut1 - _explut)));

                            _b *=  _w;
                            _g *=  _w;
                            _r *=  _w;
                            sumw += _w;
                            sumb += _b;
                            sumg += _g;
                            sumr += _r;
                        }
                        v_float32x4 sum4 = v_reduce_sum4(sumw, sumb, sumg, sumr);
                        float *bufFloat = (float*)idxBuf;
                        v_store(bufFloat, sum4);
                        wsum += bufFloat[0];
                        sum_b += bufFloat[1];
                        sum_g += bufFloat[2];
                        sum_r += bufFloat[3];
                    }
#endif

                    for(; k < maxk; k++ )
                    {
                        const float* sptr_k = sptr + j + space_ofs[k];
                        float b = sptr_k[0], g = sptr_k[1], r = sptr_k[2];
                        float alpha = (float)((std::abs(b - b0) +
                            std::abs(g - g0) + std::abs(r - r0))*scale_index);
                        int idx = cvFloor(alpha);
                        alpha -= idx;
                        float w = space_weight[k]*(expLUT[idx] + alpha*(expLUT[idx+1] - expLUT[idx]));
                        sum_b += b*w; sum_g += g*w; sum_r += r*w;
                        wsum += w;
                    }
                    CV_DbgAssert(fabs(wsum) > 0);
                    wsum = 1.f/wsum;
                    b0 = sum_b*wsum;
                    g0 = sum_g*wsum;
                    r0 = sum_r*wsum;
                    dptr[j] = b0; dptr[j+1] = g0; dptr[j+2] = r0;
                }
            }
        }
    }

private:
    int cn, radius, maxk, *space_ofs;
    const Mat* temp;
    Mat *dest;
    float scale_index, *space_weight, *expLUT;
};


static void
bilateralFilter_32f( const Mat& src, Mat& dst, int d,
                     double sigma_color, double sigma_space,
                     int borderType )
{
    int cn = src.channels();
    int i, j, maxk, radius;
    double minValSrc=-1, maxValSrc=1;
    const int kExpNumBinsPerChannel = 1 << 12;
    int kExpNumBins = 0;
    float lastExpVal = 1.f;
    float len, scale_index;
    Size size = src.size();

    CV_Assert( (src.type() == CV_32FC1 || src.type() == CV_32FC3) && src.data != dst.data );

    if( sigma_color <= 0 )
        sigma_color = 1;
    if( sigma_space <= 0 )
        sigma_space = 1;

    double gauss_color_coeff = -0.5/(sigma_color*sigma_color);
    double gauss_space_coeff = -0.5/(sigma_space*sigma_space);

    if( d <= 0 )
        radius = cvRound(sigma_space*1.5);
    else
        radius = d/2;
    radius = MAX(radius, 1);
    d = radius*2 + 1;
    // compute the min/max range for the input image (even if multichannel)

    minMaxLoc( src.reshape(1), &minValSrc, &maxValSrc );
    if(std::abs(minValSrc - maxValSrc) < FLT_EPSILON)
    {
        src.copyTo(dst);
        return;
    }

    // temporary copy of the image with borders for easy processing
    Mat temp;
    copyMakeBorder( src, temp, radius, radius, radius, radius, borderType );
    const double insteadNaNValue = -5. * sigma_color;
    patchNaNs( temp, insteadNaNValue ); // this replacement of NaNs makes the assumption that depth values are nonnegative
                                        // TODO: make insteadNaNValue avalible in the outside function interface to control the cases breaking the assumption
    // allocate lookup tables
    std::vector<float> _space_weight(d*d);
    std::vector<int> _space_ofs(d*d);
    float* space_weight = &_space_weight[0];
    int* space_ofs = &_space_ofs[0];

    // assign a length which is slightly more than needed
    len = (float)(maxValSrc - minValSrc) * cn;
    kExpNumBins = kExpNumBinsPerChannel * cn;
    std::vector<float> _expLUT(kExpNumBins+2);
    float* expLUT = &_expLUT[0];

    scale_index = kExpNumBins/len;

    // initialize the exp LUT
    for( i = 0; i < kExpNumBins+2; i++ )
    {
        if( lastExpVal > 0.f )
        {
            double val =  i / scale_index;
            expLUT[i] = (float)std::exp(val * val * gauss_color_coeff);
            lastExpVal = expLUT[i];
        }
        else
            expLUT[i] = 0.f;
    }

    // initialize space-related bilateral filter coefficients
    for( i = -radius, maxk = 0; i <= radius; i++ )
        for( j = -radius; j <= radius; j++ )
        {
            double r = std::sqrt((double)i*i + (double)j*j);
            if( r > radius )
                continue;
            space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
            space_ofs[maxk++] = (int)(i*(temp.step/sizeof(float)) + j*cn);
        }

    // parallel_for usage

    BilateralFilter_32f_Invoker body(cn, radius, maxk, space_ofs, temp, dst, scale_index, space_weight, expLUT);
    parallel_for_(Range(0, size.height), body, dst.total()/(double)(1<<16));
}

#ifdef HAVE_IPP
#define IPP_BILATERAL_PARALLEL 1

#ifdef HAVE_IPP_IW
class ipp_bilateralFilterParallel: public ParallelLoopBody
{
public:
    ipp_bilateralFilterParallel(::ipp::IwiImage &_src, ::ipp::IwiImage &_dst, int _radius, Ipp32f _valSquareSigma, Ipp32f _posSquareSigma, ::ipp::IwiBorderType _borderType, bool *_ok):
        src(_src), dst(_dst)
    {
        pOk = _ok;

        radius          = _radius;
        valSquareSigma  = _valSquareSigma;
        posSquareSigma  = _posSquareSigma;
        borderType      = _borderType;

        *pOk = true;
    }
    ~ipp_bilateralFilterParallel() {}

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        if(*pOk == false)
            return;

        try
        {
            ::ipp::IwiTile tile = ::ipp::IwiRoi(0, range.start, dst.m_size.width, range.end - range.start);
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilterBilateral, src, dst, radius, valSquareSigma, posSquareSigma, ::ipp::IwDefault(), borderType, tile);
        }
        catch(const ::ipp::IwException &)
        {
            *pOk = false;
            return;
        }
    }
private:
    ::ipp::IwiImage &src;
    ::ipp::IwiImage &dst;

    int                  radius;
    Ipp32f               valSquareSigma;
    Ipp32f               posSquareSigma;
    ::ipp::IwiBorderType borderType;

    bool  *pOk;
    const ipp_bilateralFilterParallel& operator= (const ipp_bilateralFilterParallel&);
};
#endif

static bool ipp_bilateralFilter(Mat &src, Mat &dst, int d, double sigmaColor, double sigmaSpace, int borderType)
{
#ifdef HAVE_IPP_IW
    CV_INSTRUMENT_REGION_IPP();

    int         radius         = IPP_MAX(((d <= 0)?cvRound(sigmaSpace*1.5):d/2), 1);
    Ipp32f      valSquareSigma = (Ipp32f)((sigmaColor <= 0)?1:sigmaColor*sigmaColor);
    Ipp32f      posSquareSigma = (Ipp32f)((sigmaSpace <= 0)?1:sigmaSpace*sigmaSpace);

    // Acquire data and begin processing
    try
    {
        ::ipp::IwiImage      iwSrc = ippiGetImage(src);
        ::ipp::IwiImage      iwDst = ippiGetImage(dst);
        ::ipp::IwiBorderSize borderSize(radius);
        ::ipp::IwiBorderType ippBorder(ippiGetBorder(iwSrc, borderType, borderSize));
        if(!ippBorder)
            return false;

        const int threads = ippiSuggestThreadsNum(iwDst, 2);
        if(IPP_BILATERAL_PARALLEL && threads > 1) {
            bool  ok      = true;
            Range range(0, (int)iwDst.m_size.height);
            ipp_bilateralFilterParallel invoker(iwSrc, iwDst, radius, valSquareSigma, posSquareSigma, ippBorder, &ok);
            if(!ok)
                return false;

            parallel_for_(range, invoker, threads*4);

            if(!ok)
                return false;
        } else {
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilterBilateral, iwSrc, iwDst, radius, valSquareSigma, posSquareSigma, ::ipp::IwDefault(), ippBorder);
        }
    }
    catch (const ::ipp::IwException &)
    {
        return false;
    }
    return true;
#else
    CV_UNUSED(src); CV_UNUSED(dst); CV_UNUSED(d); CV_UNUSED(sigmaColor); CV_UNUSED(sigmaSpace); CV_UNUSED(borderType);
    return false;
#endif
}
#endif

}

void cv::bilateralFilter( InputArray _src, OutputArray _dst, int d,
                      double sigmaColor, double sigmaSpace,
                      int borderType )
{
    CV_INSTRUMENT_REGION();

    _dst.create( _src.size(), _src.type() );

    CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat(),
               ocl_bilateralFilter_8u(_src, _dst, d, sigmaColor, sigmaSpace, borderType))

    Mat src = _src.getMat(), dst = _dst.getMat();

    CV_IPP_RUN_FAST(ipp_bilateralFilter(src, dst, d, sigmaColor, sigmaSpace, borderType));

    if( src.depth() == CV_8U )
        bilateralFilter_8u( src, dst, d, sigmaColor, sigmaSpace, borderType );
    else if( src.depth() == CV_32F )
        bilateralFilter_32f( src, dst, d, sigmaColor, sigmaSpace, borderType );
    else
        CV_Error( CV_StsUnsupportedFormat,
        "Bilateral filtering is only implemented for 8u and 32f images" );
}

//////////////////////////////////////////////////////////////////////////////////////////

CV_IMPL void
cvSmooth( const void* srcarr, void* dstarr, int smooth_type,
          int param1, int param2, double param3, double param4 )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst0 = cv::cvarrToMat(dstarr), dst = dst0;

    CV_Assert( dst.size() == src.size() &&
        (smooth_type == CV_BLUR_NO_SCALE || dst.type() == src.type()) );

    if( param2 <= 0 )
        param2 = param1;

    if( smooth_type == CV_BLUR || smooth_type == CV_BLUR_NO_SCALE )
        cv::boxFilter( src, dst, dst.depth(), cv::Size(param1, param2), cv::Point(-1,-1),
            smooth_type == CV_BLUR, cv::BORDER_REPLICATE );
    else if( smooth_type == CV_GAUSSIAN )
        cv::GaussianBlur( src, dst, cv::Size(param1, param2), param3, param4, cv::BORDER_REPLICATE );
    else if( smooth_type == CV_MEDIAN )
        cv::medianBlur( src, dst, param1 );
    else
        cv::bilateralFilter( src, dst, param1, param3, param4, cv::BORDER_REPLICATE );

    if( dst.data != dst0.data )
        CV_Error( CV_StsUnmatchedFormats, "The destination image does not have the proper type" );
}

/* End of file. */
