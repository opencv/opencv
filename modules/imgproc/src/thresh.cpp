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
#include "opencl_kernels_imgproc.hpp"

namespace cv
{

static void calc_thresh_table(const uchar* src, Size roi, const int srcStep, const int dstStep, const uchar* tab, int j_scalar, uchar* dst)
{
    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {
        int j = j_scalar;
#if CV_ENABLE_UNROLLED
        for (; j <= roi.width - 4; j += 4)
        {
            uchar t0 = tab[src[j]];
            uchar t1 = tab[src[j + 1]];

            dst[j] = t0;
            dst[j + 1] = t1;

            t0 = tab[src[j + 2]];
            t1 = tab[src[j + 3]];

            dst[j + 2] = t0;
            dst[j + 3] = t1;
        }
#endif
        for (; j < roi.width; j++)
            dst[j] = tab[src[j]];
    }
}

template<typename SrcType, int ThresholdType>
inline void calc_thresh_primitive(const SrcType* /*src*/, const Size /*roi*/, const int /*srcStep*/, const int /*dstStep*/, const SrcType /*thresh*/, const SrcType /*maxval*/, int /*j_scalar*/, SrcType* /*dst*/)
{
}

template<>
void calc_thresh_primitive<uchar, THRESH_BINARY>(const uchar* src, const Size roi, const int srcStep, const int dstStep, const uchar thresh, const uchar maxval, int j_scalar, uchar* dst)
{
    const int thresh_pivot = thresh + 1;
    uchar tab[256];
    memset(tab, 0, thresh_pivot);
    if (thresh_pivot < 256) {
        memset(tab + thresh_pivot, maxval, 256 - thresh_pivot);
    }

    calc_thresh_table(src, roi, srcStep, dstStep, tab, j_scalar, dst);
}

template<>
void calc_thresh_primitive<uchar, THRESH_BINARY_INV>(const uchar* src, const Size roi, const int srcStep, const int dstStep, const uchar thresh, const uchar maxval, int j_scalar, uchar* dst)
{
    const int thresh_pivot = thresh + 1;
    uchar tab[256];
    memset(tab, maxval, thresh_pivot);
    if (thresh_pivot < 256) {
        memset(tab + thresh_pivot, 0, 256 - thresh_pivot);
    }

    calc_thresh_table(src, roi, srcStep, dstStep, tab, j_scalar, dst);
}

template<>
void calc_thresh_primitive<uchar, THRESH_TRUNC>(const uchar* src, const Size roi, const int srcStep, const int dstStep, const uchar thresh, const uchar /*maxval*/, int j_scalar, uchar* dst)
{
    const int thresh_pivot = thresh + 1;
    uchar tab[256];
    for (int i = 0; i < thresh_pivot; i++)
        tab[i] = (uchar)i;
    if (thresh_pivot < 256) {
        memset(tab + thresh_pivot, thresh, 256 - thresh_pivot);
    }

    calc_thresh_table(src, roi, srcStep, dstStep, tab, j_scalar, dst);
}

template<>
void calc_thresh_primitive<uchar, THRESH_TOZERO>(const uchar* src, const Size roi, const int srcStep, const int dstStep, const uchar thresh, const uchar /*maxval*/, int j_scalar, uchar* dst)
{
    const int thresh_pivot = thresh + 1;
    uchar tab[256];
    memset(tab, 0, thresh_pivot);
    for (int i = thresh_pivot; i < 256; i++)
        tab[i] = (uchar)i;

    calc_thresh_table(src, roi, srcStep, dstStep, tab, j_scalar, dst);
}

template<>
void calc_thresh_primitive<uchar, THRESH_TOZERO_INV>(const uchar* src, const Size roi, const int srcStep, const int dstStep, const uchar thresh, const uchar /*maxval*/, int j_scalar, uchar* dst)
{
    const int thresh_pivot = thresh + 1;
    uchar tab[256];
    for (int i = 0; i < thresh_pivot; i++)
        tab[i] = (uchar)i;
    if (thresh_pivot < 256) {
        memset(tab + thresh_pivot, 0, 256 - thresh_pivot);
    }

    calc_thresh_table(src, roi, srcStep, dstStep, tab, j_scalar, dst);
}

template<typename SrcType>
void calc_thresh_primitive_thresh_binary(const SrcType* src, const Size roi, const int srcStep, const int dstStep, const SrcType thresh, const SrcType maxval, int j_scalar, SrcType* dst)
{
    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {
        for (int j = j_scalar; j < roi.width; j++)
            dst[j] = src[j] > thresh ? maxval : 0;
    }
}

template<typename SrcType>
void calc_thresh_primitive_thresh_binary_inv(const SrcType* src, const Size roi, const int srcStep, const int dstStep, const SrcType thresh, const SrcType maxval, int j_scalar, SrcType* dst)
{
    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {
        for (int j = j_scalar; j < roi.width; j++)
            dst[j] = src[j] <= thresh ? maxval : 0;
    }
}

template<typename SrcType>
void calc_thresh_primitive_thresh_trunc(const SrcType* src, const Size roi, const int srcStep, const int dstStep, const SrcType thresh, const SrcType /*maxval*/, int j_scalar, SrcType* dst)
{
    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {
        for (int j = j_scalar; j < roi.width; j++)
            dst[j] = std::min(src[j], thresh);
    }
}

template<typename SrcType>
void calc_thresh_primitive_thresh_tozero(const SrcType* src, const Size roi, const int srcStep, const int dstStep, const SrcType thresh, const SrcType /*maxval*/, int j_scalar, SrcType* dst)
{
    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {
        for (int j = j_scalar; j < roi.width; j++)
        {
            const SrcType v = src[j];
            dst[j] = v > thresh ? v : 0;
        }
    }
}

template<typename SrcType>
void calc_thresh_primitive_thresh_tozero_inv(const SrcType* src, const Size roi, const int srcStep, const int dstStep, const SrcType thresh, const SrcType /*maxval*/, int j_scalar, SrcType* dst)
{
    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {
        for (int j = j_scalar; j < roi.width; j++)
        {
            const SrcType v = src[j];
            dst[j] = v <= thresh ? v : 0;
        }
    }
}

template<>
inline void calc_thresh_primitive<short, THRESH_BINARY>(const short* src, const Size roi, const int srcStep, const int dstStep, const short thresh, const short maxval, int j_scalar, short* dst)
{
    calc_thresh_primitive_thresh_binary<short>(src, roi, srcStep, dstStep, thresh, maxval, j_scalar, dst);
}

template<>
inline void calc_thresh_primitive<short, THRESH_BINARY_INV>(const short* src, const Size roi, const int srcStep, const int dstStep, const short thresh, const short maxval, int j_scalar, short* dst)
{
    calc_thresh_primitive_thresh_binary_inv<short>(src, roi, srcStep, dstStep, thresh, maxval, j_scalar, dst);
}

template<>
inline void calc_thresh_primitive<short, THRESH_TRUNC>(const short* src, const Size roi, const int srcStep, const int dstStep, const short thresh, const short maxval, int j_scalar, short* dst)
{
    calc_thresh_primitive_thresh_trunc<short>(src, roi, srcStep, dstStep, thresh, maxval, j_scalar, dst);
}

template<>
inline void calc_thresh_primitive<short, THRESH_TOZERO>(const short* src, const Size roi, const int srcStep, const int dstStep, const short thresh, const short maxval, int j_scalar, short* dst)
{
    calc_thresh_primitive_thresh_tozero<short>(src, roi, srcStep, dstStep, thresh, maxval, j_scalar, dst);
}

template<>
inline void calc_thresh_primitive<short, THRESH_TOZERO_INV>(const short* src, const Size roi, const int srcStep, const int dstStep, const short thresh, const short maxval, int j_scalar, short* dst)
{
    calc_thresh_primitive_thresh_tozero_inv<short>(src, roi, srcStep, dstStep, thresh, maxval, j_scalar, dst);
}

template<>
inline void calc_thresh_primitive<float, THRESH_BINARY>(const float* src, const Size roi, const int srcStep, const int dstStep, const float thresh, const float maxval, int j_scalar, float* dst)
{
    calc_thresh_primitive_thresh_binary<float>(src, roi, srcStep, dstStep, thresh, maxval, j_scalar, dst);
}

template<>
inline void calc_thresh_primitive<float, THRESH_BINARY_INV>(const float* src, const Size roi, const int srcStep, const int dstStep, const float thresh, const float maxval, int j_scalar, float* dst)
{
    calc_thresh_primitive_thresh_binary_inv<float>(src, roi, srcStep, dstStep, thresh, maxval, j_scalar, dst);
}

template<>
inline void calc_thresh_primitive<float, THRESH_TRUNC>(const float* src, const Size roi, const int srcStep, const int dstStep, const float thresh, const float maxval, int j_scalar, float* dst)
{
    calc_thresh_primitive_thresh_trunc<float>(src, roi, srcStep, dstStep, thresh, maxval, j_scalar, dst);
}

template<>
inline void calc_thresh_primitive<float, THRESH_TOZERO>(const float* src, const Size roi, const int srcStep, const int dstStep, const float thresh, const float maxval, int j_scalar, float* dst)
{
    calc_thresh_primitive_thresh_tozero<float>(src, roi, srcStep, dstStep, thresh, maxval, j_scalar, dst);
}

template<>
inline void calc_thresh_primitive<float, THRESH_TOZERO_INV>(const float* src, const Size roi, const int srcStep, const int dstStep, const float thresh, const float maxval, int j_scalar, float* dst)
{
    calc_thresh_primitive_thresh_tozero_inv<float>(src, roi, srcStep, dstStep, thresh, maxval, j_scalar, dst);
}

template<>
inline void calc_thresh_primitive<double, THRESH_BINARY>(const double* src, const Size roi, const int srcStep, const int dstStep, const double thresh, const double maxval, int j_scalar, double* dst)
{
    calc_thresh_primitive_thresh_binary<double>(src, roi, srcStep, dstStep, thresh, maxval, j_scalar, dst);
}

template<>
inline void calc_thresh_primitive<double, THRESH_BINARY_INV>(const double* src, const Size roi, const int srcStep, const int dstStep, const double thresh, const double maxval, int j_scalar, double* dst)
{
    calc_thresh_primitive_thresh_binary_inv<double>(src, roi, srcStep, dstStep, thresh, maxval, j_scalar, dst);
}

template<>
inline void calc_thresh_primitive<double, THRESH_TRUNC>(const double* src, const Size roi, const int srcStep, const int dstStep, const double thresh, const double maxval, int j_scalar, double* dst)
{
    calc_thresh_primitive_thresh_trunc<double>(src, roi, srcStep, dstStep, thresh, maxval, j_scalar, dst);
}

template<>
inline void calc_thresh_primitive<double, THRESH_TOZERO>(const double* src, const Size roi, const int srcStep, const int dstStep, const double thresh, const double maxval, int j_scalar, double* dst)
{
    calc_thresh_primitive_thresh_tozero<double>(src, roi, srcStep, dstStep, thresh, maxval, j_scalar, dst);
}

template<>
inline void calc_thresh_primitive<double, THRESH_TOZERO_INV>(const double* src, const Size roi, const int srcStep, const int dstStep, const double thresh, const double maxval, int j_scalar, double* dst)
{
    calc_thresh_primitive_thresh_tozero_inv<double>(src, roi, srcStep, dstStep, thresh, maxval, j_scalar, dst);
}

#ifdef HAVE_TEGRA_OPTIMIZATION
template<typename SrcType>
inline bool calc_thresh_tegra(const Mat& _src, const int width, const int height, SrcType thresh, SrcType maxval, const int type, Mat& _dst)
{
    return false;
}

template<>
inline bool calc_thresh_tegra(const Mat& _src, const int width, const int height, uchar thresh, uchar maxval, const int type, Mat& _dst)
{
    return tegra::thresh_8u(_src, _dst, width, height, thresh, maxval, type);
}

template<>
inline bool calc_thresh_tegra(const Mat& _src, const int width, const int height, short thresh, short maxval, const int type, Mat& _dst)
{
    return tegra::thresh_16s(_src, _dst, width, height, thresh, maxval, type);
}

template<>
inline bool calc_thresh_tegra(const Mat& _src, const int width, const int height, float thresh, float maxval, const int type, Mat& _dst)
{
    return tegra::thresh_32f(_src, _dst, width, height, thresh, maxval, type);
}
#endif

#if defined(HAVE_IPP)
#ifndef HAVE_IPP_ICV_ONLY
template<typename SrcType, int type>
inline int calc_thresh_ipp_c1ir(const SrcType /*thresh*/, const IppiSize /*sz*/, const int /*dstStep*/, SrcType* /*dst*/)
{
    return -1;
}

template<>
inline int calc_thresh_ipp_c1ir<uchar, THRESH_TRUNC>(const uchar thresh, const IppiSize sz, const int dstStep, uchar* dst)
{
    return ippiThreshold_GT_8u_C1IR(dst, dstStep, sz, thresh);
}

template<>
inline int calc_thresh_ipp_c1ir<short, THRESH_TRUNC>(const short thresh, const IppiSize sz, const int dstStep, short* dst)
{
    return ippiThreshold_GT_16s_C1IR(dst, dstStep, sz, thresh);
}

template<>
inline int calc_thresh_ipp_c1ir<uchar, THRESH_TOZERO>(const uchar thresh, const IppiSize sz, const int dstStep, uchar* dst)
{
    return ippiThreshold_LTVal_8u_C1IR(dst, dstStep, sz, thresh + 1, 0);
}

template<>
inline int calc_thresh_ipp_c1ir<short, THRESH_TOZERO>(const short thresh, const IppiSize sz, const int dstStep, short* dst)
{
    return ippiThreshold_LTVal_16s_C1IR(dst, dstStep, sz, thresh + 1, 0);
}

template<>
inline int calc_thresh_ipp_c1ir<uchar, THRESH_TOZERO_INV>(const uchar thresh, const IppiSize sz, const int dstStep, uchar* dst)
{
    return ippiThreshold_GTVal_8u_C1IR(dst, dstStep, sz, thresh, 0);
}

template<>
inline int calc_thresh_ipp_c1ir<short, THRESH_TOZERO_INV>(const short thresh, const IppiSize sz, const int dstStep, short* dst)
{
    return ippiThreshold_GTVal_16s_C1IR(dst, dstStep, sz, thresh, 0);
}
#endif

template<typename SrcType, int type>
inline int calc_thresh_ipp_c1r(const SrcType* /*src*/, const int /*srcStep*/, const SrcType /*thresh*/, const IppiSize /*sz*/, const int /*dstStep*/, SrcType* /*dst*/)
{
    return -1;
}

template<>
inline int calc_thresh_ipp_c1r<uchar, THRESH_TRUNC>(const uchar* src, const int srcStep, const uchar thresh, const IppiSize sz, const int dstStep, uchar* dst)
{
    return ippiThreshold_GT_8u_C1R(src, srcStep, dst, dstStep, sz, thresh);
}

template<>
inline int calc_thresh_ipp_c1r<short, THRESH_TRUNC>(const short* src, const int srcStep, const short thresh, const IppiSize sz, const int dstStep, short* dst)
{
    return ippiThreshold_GT_16s_C1R(src, srcStep, dst, dstStep, sz, thresh);
}

template<>
inline int calc_thresh_ipp_c1r<float, THRESH_TRUNC>(const float* src, const int srcStep, const float thresh, const IppiSize sz, const int dstStep, float* dst)
{
    return ippiThreshold_GT_32f_C1R(src, srcStep, dst, dstStep, sz, thresh);
}

template<>
inline int calc_thresh_ipp_c1r<uchar, THRESH_TOZERO>(const uchar* src, const int srcStep, const uchar thresh, const IppiSize sz, const int dstStep, uchar* dst)
{
    return ippiThreshold_LTVal_8u_C1R(src, srcStep, dst, dstStep, sz, thresh + 1, 0);
}

template<>
inline int calc_thresh_ipp_c1r<short, THRESH_TOZERO>(const short* src, const int srcStep, const short thresh, const IppiSize sz, const int dstStep, short* dst)
{
    return ippiThreshold_LTVal_16s_C1R(src, srcStep, dst, dstStep, sz, thresh + 1, 0);
}

template<>
inline int calc_thresh_ipp_c1r<float, THRESH_TOZERO>(const float* src, const int srcStep, const float thresh, const IppiSize sz, const int dstStep, float* dst)
{
    return ippiThreshold_LTVal_32f_C1R(src, srcStep, dst, dstStep, sz, thresh + FLT_EPSILON, 0);
}

template<>
inline int calc_thresh_ipp_c1r<uchar, THRESH_TOZERO_INV>(const uchar* src, const int srcStep, const uchar thresh, const IppiSize sz, const int dstStep, uchar* dst)
{
    return ippiThreshold_GTVal_8u_C1R(src, srcStep, dst, dstStep, sz, thresh, 0);
}

template<>
inline int calc_thresh_ipp_c1r<short, THRESH_TOZERO_INV>(const short* src, const int srcStep, const short thresh, const IppiSize sz, const int dstStep, short* dst)
{
    return ippiThreshold_GTVal_16s_C1R(src, srcStep, dst, dstStep, sz, thresh, 0);
}

template<>
inline int calc_thresh_ipp_c1r<float, THRESH_TOZERO_INV>(const float* src, const int srcStep, const float thresh, const IppiSize sz, const int dstStep, float* dst)
{
    return ippiThreshold_GTVal_32f_C1R(src, srcStep, dst, dstStep, sz, thresh, 0);
}
#endif

template<typename SrcType, int ThresholdType, int SimdType>
inline int calc_thresh_simd(const SrcType* /*src*/, const int /*srcStep*/, const SrcType /*thresh*/, const SrcType /*maxval*/, const Size /*roi*/, const int /*dstStep*/, SrcType* /*dst*/)
{
    return 0;
}

#if CV_SSE
template<>
int calc_thresh_simd<float, THRESH_BINARY, CV_CPU_SSE>(const float* src, const int srcStep, const float thresh, const float maxval, const Size roi, const int dstStep, float* dst)
{
    int j_scalar = 0;
    __m128 thresh4 = _mm_set1_ps(thresh);
    __m128 maxval4 = _mm_set1_ps(maxval);

    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {
        for (j_scalar = 0; j_scalar <= roi.width - 8; j_scalar += 8)
        {
            __m128 v0 = _mm_loadu_ps(src + j_scalar);
            __m128 v1 = _mm_loadu_ps(src + j_scalar + 4);
            v0 = _mm_cmpgt_ps(v0, thresh4);
            v1 = _mm_cmpgt_ps(v1, thresh4);
            v0 = _mm_and_ps(v0, maxval4);
            v1 = _mm_and_ps(v1, maxval4);
            _mm_storeu_ps(dst + j_scalar, v0);
            _mm_storeu_ps(dst + j_scalar + 4, v1);
        }
    }

    return j_scalar;
}

template<>
int calc_thresh_simd<float, THRESH_BINARY_INV, CV_CPU_SSE>(const float* src, const int srcStep, const float thresh, const float maxval, const Size roi, const int dstStep, float* dst)
{
    int j_scalar = 0;
    __m128 thresh4 = _mm_set1_ps(thresh);
    __m128 maxval4 = _mm_set1_ps(maxval);

    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {
        for (j_scalar = 0; j_scalar <= roi.width - 8; j_scalar += 8)
        {
            __m128 v0 = _mm_loadu_ps(src + j_scalar);
            __m128 v1 = _mm_loadu_ps(src + j_scalar + 4);
            v0 = _mm_cmple_ps(v0, thresh4);
            v1 = _mm_cmple_ps(v1, thresh4);
            v0 = _mm_and_ps(v0, maxval4);
            v1 = _mm_and_ps(v1, maxval4);
            _mm_storeu_ps(dst + j_scalar, v0);
            _mm_storeu_ps(dst + j_scalar + 4, v1);
        }
    }

    return j_scalar;
}

template<>
int calc_thresh_simd<float, THRESH_TRUNC, CV_CPU_SSE>(const float* src, const int srcStep, const float thresh, const float /*maxval*/, const Size roi, const int dstStep, float* dst)
{
    int j_scalar = 0;
    __m128 thresh4 = _mm_set1_ps(thresh);

    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {
        for (j_scalar = 0; j_scalar <= roi.width - 8; j_scalar += 8)
        {
            __m128 v0 = _mm_loadu_ps(src + j_scalar);
            __m128 v1 = _mm_loadu_ps(src + j_scalar + 4);
            v0 = _mm_min_ps(v0, thresh4);
            v1 = _mm_min_ps(v1, thresh4);
            _mm_storeu_ps(dst + j_scalar, v0);
            _mm_storeu_ps(dst + j_scalar + 4, v1);
        }
    }

    return j_scalar;
}

template<>
int calc_thresh_simd<float, THRESH_TOZERO, CV_CPU_SSE>(const float* src, const int srcStep, const float thresh, const float /*maxval*/, const Size roi, const int dstStep, float* dst)
{
    int j_scalar = 0;
    __m128 thresh4 = _mm_set1_ps(thresh);

    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {
        for (j_scalar = 0; j_scalar <= roi.width - 8; j_scalar += 8)
        {
            __m128 v0 = _mm_loadu_ps(src + j_scalar);
            __m128 v1 = _mm_loadu_ps(src + j_scalar + 4);
            v0 = _mm_and_ps(v0, _mm_cmpgt_ps(v0, thresh4));
            v1 = _mm_and_ps(v1, _mm_cmpgt_ps(v1, thresh4));
            _mm_storeu_ps(dst + j_scalar, v0);
            _mm_storeu_ps(dst + j_scalar + 4, v1);
        }
    }

    return j_scalar;
}

template<>
int calc_thresh_simd<float, THRESH_TOZERO_INV, CV_CPU_SSE>(const float* src, const int srcStep, const float thresh, const float /*maxval*/, const Size roi, const int dstStep, float* dst)
{
    int j_scalar = 0;
    __m128 thresh4 = _mm_set1_ps(thresh);

    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {
        for (j_scalar = 0; j_scalar <= roi.width - 8; j_scalar += 8)
        {
            __m128 v0 = _mm_loadu_ps(src + j_scalar);
            __m128 v1 = _mm_loadu_ps(src + j_scalar + 4);
            v0 = _mm_and_ps(v0, _mm_cmple_ps(v0, thresh4));
            v1 = _mm_and_ps(v1, _mm_cmple_ps(v1, thresh4));
            _mm_storeu_ps(dst + j_scalar, v0);
            _mm_storeu_ps(dst + j_scalar + 4, v1);
        }
    }

    return j_scalar;
}
#endif

#if CV_SSE2
template<>
int calc_thresh_simd<uchar, THRESH_BINARY, CV_CPU_SSE2>(const uchar* src, const int srcStep, const uchar thresh, const uchar maxval, const Size roi, const int dstStep, uchar* dst)
{
    int j_scalar = 0;
    __m128i _x80 = _mm_set1_epi8('\x80');
    __m128i thresh_s = _mm_set1_epi8(thresh ^ 0x80);
    __m128i maxval_ = _mm_set1_epi8(maxval);

    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {
        for (j_scalar = 0; j_scalar <= roi.width - 32; j_scalar += 32)
        {
            __m128i v0 = _mm_loadu_si128((const __m128i*)(src + j_scalar));
            __m128i v1 = _mm_loadu_si128((const __m128i*)(src + j_scalar + 16));
            v0 = _mm_cmpgt_epi8(_mm_xor_si128(v0, _x80), thresh_s);
            v1 = _mm_cmpgt_epi8(_mm_xor_si128(v1, _x80), thresh_s);
            v0 = _mm_and_si128(v0, maxval_);
            v1 = _mm_and_si128(v1, maxval_);
            _mm_storeu_si128((__m128i*)(dst + j_scalar), v0);
            _mm_storeu_si128((__m128i*)(dst + j_scalar + 16), v1);
        }

        for (; j_scalar <= roi.width - 8; j_scalar += 8)
        {
            __m128i v0 = _mm_loadl_epi64((const __m128i*)(src + j_scalar));
            v0 = _mm_cmpgt_epi8(_mm_xor_si128(v0, _x80), thresh_s);
            v0 = _mm_and_si128(v0, maxval_);
            _mm_storel_epi64((__m128i*)(dst + j_scalar), v0);
        }
    }

    return j_scalar;
}

template<>
int calc_thresh_simd<uchar, THRESH_BINARY_INV, CV_CPU_SSE2>(const uchar* src, const int srcStep, const uchar thresh, const uchar maxval, const Size roi, const int dstStep, uchar* dst)
{
    int j_scalar = 0;
    __m128i _x80 = _mm_set1_epi8('\x80');
    __m128i thresh_s = _mm_set1_epi8(thresh ^ 0x80);
    __m128i maxval_ = _mm_set1_epi8(maxval);

    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {
        for (j_scalar = 0; j_scalar <= roi.width - 32; j_scalar += 32)
        {
            __m128i v0 = _mm_loadu_si128((const __m128i*)(src + j_scalar));
            __m128i v1 = _mm_loadu_si128((const __m128i*)(src + j_scalar + 16));
            v0 = _mm_cmpgt_epi8(_mm_xor_si128(v0, _x80), thresh_s);
            v1 = _mm_cmpgt_epi8(_mm_xor_si128(v1, _x80), thresh_s);
            v0 = _mm_andnot_si128(v0, maxval_);
            v1 = _mm_andnot_si128(v1, maxval_);
            _mm_storeu_si128((__m128i*)(dst + j_scalar), v0);
            _mm_storeu_si128((__m128i*)(dst + j_scalar + 16), v1);
        }

        for (; j_scalar <= roi.width - 8; j_scalar += 8)
        {
            __m128i v0 = _mm_loadl_epi64((const __m128i*)(src + j_scalar));
            v0 = _mm_cmpgt_epi8(_mm_xor_si128(v0, _x80), thresh_s);
            v0 = _mm_andnot_si128(v0, maxval_);
            _mm_storel_epi64((__m128i*)(dst + j_scalar), v0);
        }
    }

    return j_scalar;
}

template<>
int calc_thresh_simd<uchar, THRESH_TRUNC, CV_CPU_SSE2>(const uchar* src, const int srcStep, const uchar thresh, const uchar /*maxval*/, const Size roi, const int dstStep, uchar* dst)
{
    int j_scalar = 0;
    __m128i thresh_u = _mm_set1_epi8(thresh);

    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {
        for (j_scalar = 0; j_scalar <= roi.width - 32; j_scalar += 32)
        {
            __m128i v0 = _mm_loadu_si128((const __m128i*)(src + j_scalar));
            __m128i v1 = _mm_loadu_si128((const __m128i*)(src + j_scalar + 16));
            v0 = _mm_subs_epu8(v0, _mm_subs_epu8( v0, thresh_u));
            v1 = _mm_subs_epu8(v1, _mm_subs_epu8( v1, thresh_u));
            _mm_storeu_si128((__m128i*)(dst + j_scalar), v0);
            _mm_storeu_si128((__m128i*)(dst + j_scalar + 16), v1);
        }

        for(; j_scalar <= roi.width - 8; j_scalar += 8)
        {
            __m128i v0 = _mm_loadl_epi64((const __m128i*)(src + j_scalar));
            v0 = _mm_subs_epu8(v0, _mm_subs_epu8( v0, thresh_u));
            _mm_storel_epi64((__m128i*)(dst + j_scalar), v0);
        }
    }

    return j_scalar;
}

template<>
int calc_thresh_simd<uchar, THRESH_TOZERO, CV_CPU_SSE2>(const uchar* src, const int srcStep, const uchar thresh, const uchar /*maxval*/, const Size roi, const int dstStep, uchar* dst)
{
    int j_scalar = 0;
    __m128i _x80 = _mm_set1_epi8('\x80');
    __m128i thresh_s = _mm_set1_epi8(thresh ^ 0x80);

    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {
        for (j_scalar = 0; j_scalar <= roi.width - 32; j_scalar += 32)
        {
            __m128i v0 = _mm_loadu_si128((const __m128i*)(src + j_scalar));
            __m128i v1 = _mm_loadu_si128((const __m128i*)(src + j_scalar + 16));
            v0 = _mm_and_si128(v0, _mm_cmpgt_epi8(_mm_xor_si128(v0, _x80), thresh_s));
            v1 = _mm_and_si128(v1, _mm_cmpgt_epi8(_mm_xor_si128(v1, _x80), thresh_s));
            _mm_storeu_si128((__m128i*)(dst + j_scalar), v0);
            _mm_storeu_si128((__m128i*)(dst + j_scalar + 16), v1);
        }

        for (; j_scalar <= roi.width - 8; j_scalar += 8)
        {
            __m128i v0 = _mm_loadl_epi64((const __m128i*)(src + j_scalar));
            v0 = _mm_and_si128(v0, _mm_cmpgt_epi8(_mm_xor_si128(v0, _x80), thresh_s));
            _mm_storel_epi64((__m128i*)(dst + j_scalar), v0);
        }
    }

    return j_scalar;
}

template<>
int calc_thresh_simd<uchar, THRESH_TOZERO_INV, CV_CPU_SSE2>(const uchar* src, const int srcStep, const uchar thresh, const uchar /*maxval*/, const Size roi, const int dstStep, uchar* dst)
{
    int j_scalar = 0;
    __m128i _x80 = _mm_set1_epi8('\x80');
    __m128i thresh_s = _mm_set1_epi8(thresh ^ 0x80);
    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {
        for (j_scalar = 0; j_scalar <= roi.width - 32; j_scalar += 32)
        {
            __m128i v0 = _mm_loadu_si128((const __m128i*)(src + j_scalar));
            __m128i v1 = _mm_loadu_si128((const __m128i*)(src + j_scalar + 16));
            v0 = _mm_andnot_si128(_mm_cmpgt_epi8(_mm_xor_si128(v0, _x80), thresh_s), v0);
            v1 = _mm_andnot_si128(_mm_cmpgt_epi8(_mm_xor_si128(v1, _x80), thresh_s), v1);
            _mm_storeu_si128((__m128i*)(dst + j_scalar), v0);
            _mm_storeu_si128((__m128i*)(dst + j_scalar + 16), v1);
        }

        for (; j_scalar <= roi.width - 8; j_scalar += 8)
        {
            __m128i v0 = _mm_loadl_epi64((const __m128i*)(src + j_scalar));
            v0 = _mm_andnot_si128( _mm_cmpgt_epi8(_mm_xor_si128(v0, _x80), thresh_s), v0);
            _mm_storel_epi64((__m128i*)(dst + j_scalar), v0);
        }
    }

    return j_scalar;
}

template<>
int calc_thresh_simd<short, THRESH_BINARY, CV_CPU_SSE2>(const short* src, const int srcStep, const short thresh, const short maxval, const Size roi, const int dstStep, short* dst)
{
    int j_scalar = 0;
    __m128i thresh8 = _mm_set1_epi16(thresh);
    __m128i maxval8 = _mm_set1_epi16(maxval);

    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {
        for (j_scalar = 0; j_scalar <= roi.width - 16; j_scalar += 16)
        {
            __m128i v0 = _mm_loadu_si128((const __m128i*)(src + j_scalar));
            __m128i v1 = _mm_loadu_si128((const __m128i*)(src + j_scalar + 8));
            v0 = _mm_cmpgt_epi16(v0, thresh8);
            v1 = _mm_cmpgt_epi16(v1, thresh8);
            v0 = _mm_and_si128(v0, maxval8);
            v1 = _mm_and_si128(v1, maxval8);
            _mm_storeu_si128((__m128i*)(dst + j_scalar), v0);
            _mm_storeu_si128((__m128i*)(dst + j_scalar + 8), v1);
        }
    }

    return j_scalar;
}

template<>
int calc_thresh_simd<short, THRESH_BINARY_INV, CV_CPU_SSE2>(const short* src, const int srcStep, const short thresh, const short maxval, const Size roi, const int dstStep, short* dst)
{
    int j_scalar = 0;
    __m128i thresh8 = _mm_set1_epi16(thresh);
    __m128i maxval8 = _mm_set1_epi16(maxval);

    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {
        for (j_scalar = 0; j_scalar <= roi.width - 16; j_scalar += 16)
        {
            __m128i v0 = _mm_loadu_si128((const __m128i*)(src + j_scalar));
            __m128i v1 = _mm_loadu_si128((const __m128i*)(src + j_scalar + 8));
            v0 = _mm_cmpgt_epi16(v0, thresh8);
            v1 = _mm_cmpgt_epi16(v1, thresh8);
            v0 = _mm_andnot_si128(v0, maxval8);
            v1 = _mm_andnot_si128(v1, maxval8);
            _mm_storeu_si128((__m128i*)(dst + j_scalar), v0);
            _mm_storeu_si128((__m128i*)(dst + j_scalar + 8), v1);
        }
    }

    return j_scalar;
}

template<>
int calc_thresh_simd<short, THRESH_TRUNC, CV_CPU_SSE2>(const short* src, const int srcStep, const short thresh, const short /*maxval*/, const Size roi, const int dstStep, short* dst)
{
    int j_scalar = 0;
    __m128i thresh8 = _mm_set1_epi16(thresh);

    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {
        for (j_scalar = 0; j_scalar <= roi.width - 16; j_scalar += 16)
        {
            __m128i v0 = _mm_loadu_si128((const __m128i*)(src + j_scalar));
            __m128i v1 = _mm_loadu_si128((const __m128i*)(src + j_scalar + 8));
            v0 = _mm_min_epi16(v0, thresh8);
            v1 = _mm_min_epi16(v1, thresh8);
            _mm_storeu_si128((__m128i*)(dst + j_scalar), v0);
            _mm_storeu_si128((__m128i*)(dst + j_scalar + 8), v1);
        }
    }

    return j_scalar;
}

template<>
int calc_thresh_simd<short, THRESH_TOZERO, CV_CPU_SSE2>(const short* src, const int srcStep, const short thresh, const short /*maxval*/, const Size roi, const int dstStep, short* dst)
{
    int j_scalar = 0;
    __m128i thresh8 = _mm_set1_epi16(thresh);

    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {
        for (j_scalar = 0; j_scalar <= roi.width - 16; j_scalar += 16)
        {
            __m128i v0 = _mm_loadu_si128((const __m128i*)(src + j_scalar));
            __m128i v1 = _mm_loadu_si128((const __m128i*)(src + j_scalar + 8));
            v0 = _mm_and_si128(v0, _mm_cmpgt_epi16(v0, thresh8));
            v1 = _mm_and_si128(v1, _mm_cmpgt_epi16(v1, thresh8));
            _mm_storeu_si128((__m128i*)(dst + j_scalar), v0);
            _mm_storeu_si128((__m128i*)(dst + j_scalar + 8), v1);
        }
    }

    return j_scalar;
}

template<>
int calc_thresh_simd<short, THRESH_TOZERO_INV, CV_CPU_SSE2>(const short* src, const int srcStep, const short thresh, const short /*maxval*/, const Size roi, const int dstStep, short* dst)
{
    int j_scalar = 0;
    __m128i thresh8 = _mm_set1_epi16(thresh);

    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {
        for (j_scalar = 0; j_scalar <= roi.width - 16; j_scalar += 16)
        {
            __m128i v0 = _mm_loadu_si128((const __m128i*)(src + j_scalar));
            __m128i v1 = _mm_loadu_si128((const __m128i*)(src + j_scalar + 8));
            v0 = _mm_andnot_si128(_mm_cmpgt_epi16(v0, thresh8), v0);
            v1 = _mm_andnot_si128(_mm_cmpgt_epi16(v1, thresh8), v1);
            _mm_storeu_si128((__m128i*)(dst + j_scalar), v0);
            _mm_storeu_si128((__m128i*)(dst + j_scalar + 8), v1);
        }
    }

    return j_scalar;
}

template<>
inline int calc_thresh_simd<float, THRESH_BINARY, CV_CPU_SSE2>(const float* src, const int srcStep, const float thresh, const float maxval, const Size roi, const int dstStep, float* dst)
{
    return calc_thresh_simd<float, THRESH_BINARY, CV_CPU_SSE>(src, srcStep, thresh, maxval, roi, dstStep, dst);
}

template<>
inline int calc_thresh_simd<float, THRESH_BINARY_INV, CV_CPU_SSE2>(const float* src, const int srcStep, const float thresh, const float maxval, const Size roi, const int dstStep, float* dst)
{
    return calc_thresh_simd<float, THRESH_BINARY_INV, CV_CPU_SSE>(src, srcStep, thresh, maxval, roi, dstStep, dst);
}

template<>
inline int calc_thresh_simd<float, THRESH_TRUNC, CV_CPU_SSE2>(const float* src, const int srcStep, const float thresh, const float maxval, const Size roi, const int dstStep, float* dst)
{
    return calc_thresh_simd<float, THRESH_TRUNC, CV_CPU_SSE>(src, srcStep, thresh, maxval, roi, dstStep, dst);
}

template<>
inline int calc_thresh_simd<float, THRESH_TOZERO, CV_CPU_SSE2>(const float* src, const int srcStep, const float thresh, const float maxval, const Size roi, const int dstStep, float* dst)
{
    return calc_thresh_simd<float, THRESH_TOZERO, CV_CPU_SSE>(src, srcStep, thresh, maxval, roi, dstStep, dst);
}

template<>
inline int calc_thresh_simd<float, THRESH_TOZERO_INV, CV_CPU_SSE2>(const float* src, const int srcStep, const float thresh, const float maxval, const Size roi, const int dstStep, float* dst)
{
    return calc_thresh_simd<float, THRESH_TOZERO_INV, CV_CPU_SSE>(src, srcStep, thresh, maxval, roi, dstStep, dst);
}
#endif

#if CV_NEON
template<>
int calc_thresh_simd<uchar, THRESH_BINARY, CV_CPU_NEON>(const uchar* src, const int srcStep, const uchar thresh, const uchar maxval, const Size roi, const int dstStep, uchar* dst)
{
    int j_scalar = 0;
    uint8x16_t v_thresh = vdupq_n_u8(thresh);
    uint8x16_t v_maxval = vdupq_n_u8(maxval);

    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {
        for (j_scalar = 0; j_scalar <= roi.width - 16; j_scalar += 16)
            vst1q_u8(dst + j_scalar, vandq_u8(vcgtq_u8(vld1q_u8(src + j_scalar), v_thresh), v_maxval));
    }

    return j_scalar;
}

template<>
int calc_thresh_simd<uchar, THRESH_BINARY_INV, CV_CPU_NEON>(const uchar* src, const int srcStep, const uchar thresh, const uchar maxval, const Size roi, const int dstStep, uchar* dst)
{
    int j_scalar = 0;
    uint8x16_t v_thresh = vdupq_n_u8(thresh);
    uint8x16_t v_maxval = vdupq_n_u8(maxval);

    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {
        for (j_scalar = 0; j_scalar <= roi.width - 16; j_scalar += 16)
            vst1q_u8(dst + j_scalar, vandq_u8(vcleq_u8(vld1q_u8(src + j_scalar), v_thresh), v_maxval));
    }

    return j_scalar;
}

template<>
int calc_thresh_simd<uchar, THRESH_TRUNC, CV_CPU_NEON>(const uchar* src, const int srcStep, const uchar thresh, const uchar /*maxval*/, const Size roi, const int dstStep, uchar* dst)
{
    int j_scalar = 0;
    uint8x16_t v_thresh = vdupq_n_u8(thresh);

    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {
        for (j_scalar = 0; j_scalar <= roi.width - 16; j_scalar += 16)
            vst1q_u8(dst + j_scalar, vminq_u8(vld1q_u8(src + j_scalar), v_thresh));
    }

    return j_scalar;
}

template<>
int calc_thresh_simd<uchar, THRESH_TOZERO, CV_CPU_NEON>(const uchar* src, const int srcStep, const uchar thresh, const uchar /*maxval*/, const Size roi, const int dstStep, uchar* dst)
{
    int j_scalar = 0;
    uint8x16_t v_thresh = vdupq_n_u8(thresh);

    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {
        for (j_scalar = 0; j_scalar <= roi.width - 16; j_scalar += 16)
        {
            uint8x16_t v_src = vld1q_u8(src + j_scalar), v_mask = vcgtq_u8(v_src, v_thresh);
            vst1q_u8(dst + j_scalar, vandq_u8(v_mask, v_src));
        }
    }

    return j_scalar;
}

template<>
int calc_thresh_simd<uchar, THRESH_TOZERO_INV, CV_CPU_NEON>(const uchar* src, const int srcStep, const uchar thresh, const uchar /*maxval*/, const Size roi, const int dstStep, uchar* dst)
{
    int j_scalar = 0;
    uint8x16_t v_thresh = vdupq_n_u8(thresh);

    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {
        for (j_scalar = 0; j_scalar <= roi.width - 16; j_scalar += 16)
        {
            uint8x16_t v_src = vld1q_u8(src + j_scalar), v_mask = vcleq_u8(v_src, v_thresh);
            vst1q_u8(dst + j_scalar, vandq_u8(v_mask, v_src));
        }
    }

    return j_scalar;
}

template<>
int calc_thresh_simd<short, THRESH_BINARY, CV_CPU_NEON>(const short* src, const int srcStep, const short thresh, const short maxval, const Size roi, const int dstStep, short* dst)
{
    int j_scalar = 0;
    int16x8_t v_thresh = vdupq_n_s16(thresh);
    int16x8_t v_maxval = vdupq_n_s16(maxval);
    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {

        for (; j_scalar <= roi.width - 8; j_scalar += 8)
        {
            uint16x8_t v_mask = vcgtq_s16(vld1q_s16(src + j_scalar), v_thresh);
            vst1q_s16(dst + j_scalar, vandq_s16(vreinterpretq_s16_u16(v_mask), v_maxval));
        }
    }

    return j_scalar;
}

template<>
int calc_thresh_simd<short, THRESH_BINARY_INV, CV_CPU_NEON>(const short* src, const int srcStep, const short thresh, const short maxval, const Size roi, const int dstStep, short* dst)
{
    int j_scalar = 0;
    int16x8_t v_thresh = vdupq_n_s16(thresh);
    int16x8_t v_maxval = vdupq_n_s16(maxval);

    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {
        for (; j_scalar <= roi.width - 8; j_scalar += 8)
        {
            uint16x8_t v_mask = vcleq_s16(vld1q_s16(src + j_scalar), v_thresh);
            vst1q_s16(dst + j_scalar, vandq_s16(vreinterpretq_s16_u16(v_mask), v_maxval));
        }
    }

    return j_scalar;
}

template<>
int calc_thresh_simd<short, THRESH_TRUNC, CV_CPU_NEON>(const short* src, const int srcStep, const short thresh, const short /*maxval*/, const Size roi, const int dstStep, short* dst)
{
    int j_scalar = 0;
    int16x8_t v_thresh = vdupq_n_s16(thresh);

    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {
        for (; j_scalar <= roi.width - 8; j_scalar += 8)
            vst1q_s16(dst + j_scalar, vminq_s16(vld1q_s16(src + j_scalar), v_thresh));
    }

    return j_scalar;
}

template<>
int calc_thresh_simd<short, THRESH_TOZERO, CV_CPU_NEON>(const short* src, const int srcStep, const short thresh, const short /*maxval*/, const Size roi, const int dstStep, short* dst)
{
    int j_scalar = 0;
    int16x8_t v_thresh = vdupq_n_s16(thresh);

    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {
        for (; j_scalar <= roi.width - 8; j_scalar += 8)
        {
            int16x8_t v_src = vld1q_s16(src + j_scalar);
            uint16x8_t v_mask = vcgtq_s16(v_src, v_thresh);
            vst1q_s16(dst + j_scalar, vandq_s16(vreinterpretq_s16_u16(v_mask), v_src));
        }
    }

    return j_scalar;
}

template<>
int calc_thresh_simd<short, THRESH_TOZERO_INV, CV_CPU_NEON>(const short* src, const int srcStep, const short thresh, const short /*maxval*/, const Size roi, const int dstStep, short* dst)
{
    int j_scalar = 0;
    int16x8_t v_thresh = vdupq_n_s16(thresh);

    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {
        for (; j_scalar <= roi.width - 8; j_scalar += 8)
        {
            int16x8_t v_src = vld1q_s16(src + j_scalar);
            uint16x8_t v_mask = vcleq_s16(v_src, v_thresh);
            vst1q_s16(dst + j_scalar, vandq_s16(vreinterpretq_s16_u16(v_mask), v_src));
        }
    }

    return j_scalar;
}

template<>
int calc_thresh_simd<float, THRESH_BINARY, CV_CPU_NEON>(const float* src, const int srcStep, const float thresh, const float maxval, const Size roi, const int dstStep, float* dst)
{
    int j_scalar = 0;
    float32x4_t v_thresh = vdupq_n_f32(thresh);
    uint32x4_t v_maxval = vreinterpretq_u32_f32(vdupq_n_f32(maxval));

    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {
        for (j_scalar = 0; j_scalar <= roi.width - 4; j_scalar += 4)
        {
            float32x4_t v_src = vld1q_f32(src + j_scalar);
            uint32x4_t v_dst = vandq_u32(vcgtq_f32(v_src, v_thresh), v_maxval);
            vst1q_f32(dst + j_scalar, vreinterpretq_f32_u32(v_dst));
        }
    }

    return j_scalar;
}

template<>
int calc_thresh_simd<float, THRESH_BINARY_INV, CV_CPU_NEON>(const float* src, const int srcStep, const float thresh, const float maxval, const Size roi, const int dstStep, float* dst)
{
    int j_scalar = 0;
    float32x4_t v_thresh = vdupq_n_f32(thresh);
    uint32x4_t v_maxval = vreinterpretq_u32_f32(vdupq_n_f32(maxval));

    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {
        for (j_scalar = 0; j_scalar <= roi.width - 4; j_scalar += 4)
        {
            float32x4_t v_src = vld1q_f32(src + j_scalar);
            uint32x4_t v_dst = vandq_u32(vcleq_f32(v_src, v_thresh), v_maxval);
            vst1q_f32(dst + j_scalar, vreinterpretq_f32_u32(v_dst));
        }
    }

    return j_scalar;
}

template<>
int calc_thresh_simd<float, THRESH_TRUNC, CV_CPU_NEON>(const float* src, const int srcStep, const float thresh, const float /*maxval*/, const Size roi, const int dstStep, float* dst)
{
    int j_scalar = 0;
    float32x4_t v_thresh = vdupq_n_f32(thresh);

    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {
        for (j_scalar = 0; j_scalar <= roi.width - 4; j_scalar += 4)
            vst1q_f32(dst + j_scalar, vminq_f32(vld1q_f32(src + j_scalar), v_thresh));
    }

    return j_scalar;
}

template<>
int calc_thresh_simd<float, THRESH_TOZERO, CV_CPU_NEON>(const float* src, const int srcStep, const float thresh, const float /*maxval*/, const Size roi, const int dstStep, float* dst)
{
    int j_scalar = 0;
    float32x4_t v_thresh = vdupq_n_f32(thresh);

    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {
        for (j_scalar = 0; j_scalar <= roi.width - 4; j_scalar += 4)
        {
            float32x4_t v_src = vld1q_f32(src + j_scalar);
            uint32x4_t v_dst = vandq_u32(vcgtq_f32(v_src, v_thresh),
                                         vreinterpretq_u32_f32(v_src));
            vst1q_f32(dst + j_scalar, vreinterpretq_f32_u32(v_dst));
        }
    }

    return j_scalar;
}

template<>
int calc_thresh_simd<float, THRESH_TOZERO_INV, CV_CPU_NEON>(const float* src, const int srcStep, const float thresh, const float /*maxval*/, const Size roi, const int dstStep, float* dst)
{
    int j_scalar = 0;
    float32x4_t v_thresh = vdupq_n_f32(thresh);

    for (int i = 0; i < roi.height; i++, src += srcStep, dst += dstStep)
    {
        for (j_scalar = 0; j_scalar <= roi.width - 4; j_scalar += 4)
        {
            float32x4_t v_src = vld1q_f32(src + j_scalar);
            uint32x4_t v_dst = vandq_u32(vcleq_f32(v_src, v_thresh),
                                         vreinterpretq_u32_f32(v_src));
            vst1q_f32(dst + j_scalar, vreinterpretq_f32_u32(v_dst));
        }
    }

    return j_scalar;
}
#endif

template<typename SrcType, int ThresholdType>
#if defined(HAVE_IPP) && !defined(HAVE_IPP_ICV_ONLY)
void calc_thresh(const SrcType* src, const int srcStep, const SrcType thresh, const SrcType maxval, const Size roi, const bool c1irFlag, const int dstStep, SrcType* dst)
#else
void calc_thresh(const SrcType* src, const int srcStep, const SrcType thresh, const SrcType maxval, const Size roi, const bool /*c1irFlag*/, const int dstStep, SrcType* dst)
#endif
{
#if defined(HAVE_IPP)
    CV_IPP_CHECK()
    {
        CV_SUPPRESS_DEPRECATED_START
        IppiSize sz = {roi.width, roi.height};
#ifndef HAVE_IPP_ICV_ONLY
        if (c1irFlag && calc_thresh_ipp_c1ir<SrcType, ThresholdType>(thresh, sz, dstStep * sizeof(SrcType), dst) >= 0)
        {
            CV_IMPL_ADD(CV_IMPL_IPP);
            return;
        }
#endif
        if (calc_thresh_ipp_c1r<SrcType, ThresholdType>(src, srcStep * sizeof(SrcType), thresh, sz, dstStep * sizeof(SrcType), dst) >= 0)
        {
            CV_IMPL_ADD(CV_IMPL_IPP);
            return;
        }
        setIppErrorStatus();
    }
    CV_SUPPRESS_DEPRECATED_END
#endif

    int j_scalar = 0;
#if CV_SSE2
    if (checkHardwareSupport(CV_CPU_SSE2))
    {
        j_scalar = calc_thresh_simd<SrcType, ThresholdType, CV_CPU_SSE2>(src, srcStep, thresh, maxval, roi, dstStep, dst);
    }
#elif CV_SSE
    if (checkHardwareSupport(CV_CPU_SSE))
    {
        j_scalar = calc_thresh_simd<SrcType, ThresholdType, CV_CPU_SSE>(src, srcStep, thresh, maxval, roi, dstStep, dst);
    }
#elif CV_NEON
    if (checkHardwareSupport(CV_CPU_NEON))
    {
        j_scalar = calc_thresh_simd<SrcType, ThresholdType, CV_CPU_NEON>(src, srcStep, thresh, maxval, roi, dstStep, dst);
    }
#endif

    if (j_scalar < roi.width)
    {
        calc_thresh_primitive<SrcType, ThresholdType>(src, roi, srcStep, dstStep, thresh, maxval, j_scalar, dst);
    }
}

#ifdef HAVE_IPP
static bool ipp_getThreshVal_Otsu_8u( const unsigned char* _src, int step, Size size, unsigned char &thresh)
{
#if IPP_VERSION_X100 >= 810 && !HAVE_ICV
    int ippStatus = -1;
    IppiSize srcSize = { size.width, size.height };
    CV_SUPPRESS_DEPRECATED_START
    ippStatus = ippiComputeThreshold_Otsu_8u_C1R(_src, step, srcSize, &thresh);
    CV_SUPPRESS_DEPRECATED_END

    if(ippStatus >= 0)
        return true;
#else
    CV_UNUSED(_src); CV_UNUSED(step); CV_UNUSED(size); CV_UNUSED(thresh);
#endif
    return false;
}
#endif

static double
getThreshVal_Otsu_8u( const Mat& _src )
{
    Size size = _src.size();
    int step = (int) _src.step;
    if( _src.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
        step = size.width;
    }

#ifdef HAVE_IPP
    unsigned char thresh;
    CV_IPP_RUN(IPP_VERSION_X100 >= 810 && !HAVE_ICV, ipp_getThreshVal_Otsu_8u(_src.ptr(), step, size, thresh), thresh);
#endif

    const int N = 256;
    int i, j, h[N] = {0};
    for( i = 0; i < size.height; i++ )
    {
        const uchar* src = _src.ptr() + step*i;
        j = 0;
        #if CV_ENABLE_UNROLLED
        for( ; j <= size.width - 4; j += 4 )
        {
            int v0 = src[j], v1 = src[j+1];
            h[v0]++; h[v1]++;
            v0 = src[j+2]; v1 = src[j+3];
            h[v0]++; h[v1]++;
        }
        #endif
        for( ; j < size.width; j++ )
            h[src[j]]++;
    }

    double mu = 0, scale = 1./(size.width*size.height);
    for( i = 0; i < N; i++ )
        mu += i*(double)h[i];

    mu *= scale;
    double mu1 = 0, q1 = 0;
    double max_sigma = 0, max_val = 0;

    for( i = 0; i < N; i++ )
    {
        double p_i, q2, mu2, sigma;

        p_i = h[i]*scale;
        mu1 *= q1;
        q1 += p_i;
        q2 = 1. - q1;

        if( std::min(q1,q2) < FLT_EPSILON || std::max(q1,q2) > 1. - FLT_EPSILON )
            continue;

        mu1 = (mu1 + i*p_i)/q1;
        mu2 = (mu - q1*mu1)/q2;
        sigma = q1*q2*(mu1 - mu2)*(mu1 - mu2);
        if( sigma > max_sigma )
        {
            max_sigma = sigma;
            max_val = i;
        }
    }

    return max_val;
}

static double
getThreshVal_Triangle_8u( const Mat& _src )
{
    Size size = _src.size();
    int step = (int) _src.step;
    if( _src.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
        step = size.width;
    }

    const int N = 256;
    int i, j, h[N] = {0};
    for( i = 0; i < size.height; i++ )
    {
        const uchar* src = _src.ptr() + step*i;
        j = 0;
        #if CV_ENABLE_UNROLLED
        for( ; j <= size.width - 4; j += 4 )
        {
            int v0 = src[j], v1 = src[j+1];
            h[v0]++; h[v1]++;
            v0 = src[j+2]; v1 = src[j+3];
            h[v0]++; h[v1]++;
        }
        #endif
        for( ; j < size.width; j++ )
            h[src[j]]++;
    }

    int left_bound = 0, right_bound = 0, max_ind = 0, max = 0;
    int temp;
    bool isflipped = false;

    for( i = 0; i < N; i++ )
    {
        if( h[i] > 0 )
        {
            left_bound = i;
            break;
        }
    }
    if( left_bound > 0 )
        left_bound--;

    for( i = N-1; i > 0; i-- )
    {
        if( h[i] > 0 )
        {
            right_bound = i;
            break;
        }
    }
    if( right_bound < N-1 )
        right_bound++;

    for( i = 0; i < N; i++ )
    {
        if( h[i] > max)
        {
            max = h[i];
            max_ind = i;
        }
    }

    if( max_ind-left_bound < right_bound-max_ind)
    {
        isflipped = true;
        i = 0, j = N-1;
        while( i < j )
        {
            temp = h[i]; h[i] = h[j]; h[j] = temp;
            i++; j--;
        }
        left_bound = N-1-right_bound;
        max_ind = N-1-max_ind;
    }

    double thresh = left_bound;
    double a, b, dist = 0, tempdist;

    /*
     * We do not need to compute precise distance here. Distance is maximized, so some constants can
     * be omitted. This speeds up a computation a bit.
     */
    a = max; b = left_bound-max_ind;
    for( i = left_bound+1; i <= max_ind; i++ )
    {
        tempdist = a*i + b*h[i];
        if( tempdist > dist)
        {
            dist = tempdist;
            thresh = i;
        }
    }
    thresh--;

    if( isflipped )
        thresh = N-1-thresh;

    return thresh;
}

template <typename SrcType>
class ThresholdRunner : public ParallelLoopBody
{
public:
    ThresholdRunner(Mat _src, Mat _dst, SrcType _thresh, SrcType _maxval, int _thresholdType)
    {
        src = _src;
        dst = _dst;

        thresh = _thresh;
        maxval = _maxval;
        thresholdType = _thresholdType;
    }

    void operator () ( const Range& range ) const
    {
        int row0 = range.start;
        int row1 = range.end;

        Mat srcStripe = src.rowRange(row0, row1);
        Mat dstStripe = dst.rowRange(row0, row1);

        Size roi = srcStripe.size();
        roi.width *= srcStripe.channels();
        size_t src_step = srcStripe.step / sizeof(SrcType);
        size_t dst_step = dstStripe.step / sizeof(SrcType);
        if (srcStripe.isContinuous() && dstStripe.isContinuous())
        {
            roi.width *= roi.height;
            roi.height = 1;
            src_step = dst_step = roi.width;
        }

#ifdef HAVE_TEGRA_OPTIMIZATION
        if (tegra::useTegra() && cals_thresh_tegra(srcStripe, roi.width, roi.height, thresh, maxval, thresholdType, dstStripe);
        {
            return;
        }
#endif

        const SrcType* _src = srcStripe.ptr<SrcType>();
        SrcType* _dst = dstStripe.ptr<SrcType>();
        if (thresholdType == THRESH_BINARY)
        {
            calc_thresh<SrcType, THRESH_BINARY>(_src, (int)src_step, thresh, maxval, roi, srcStripe.data == dstStripe.data, (int)dst_step, _dst);
        }
        else if (thresholdType == THRESH_BINARY_INV)
        {
            calc_thresh<SrcType, THRESH_BINARY_INV>(_src, (int)src_step, thresh, maxval, roi, srcStripe.data == dstStripe.data, (int)dst_step, _dst);
        }
        else if (thresholdType == THRESH_TRUNC)
        {
            calc_thresh<SrcType, THRESH_TRUNC>(_src, (int)src_step, thresh, maxval, roi, srcStripe.data == dstStripe.data, (int)dst_step, _dst);
        }
        else if (thresholdType == THRESH_TOZERO)
        {
            calc_thresh<SrcType, THRESH_TOZERO>(_src, (int)src_step, thresh, maxval, roi, srcStripe.data == dstStripe.data, (int)dst_step, _dst);
        }
        else if (thresholdType == THRESH_TOZERO_INV)
        {
            calc_thresh<SrcType, THRESH_TOZERO_INV>(_src, (int)src_step, thresh, maxval, roi, srcStripe.data == dstStripe.data, (int)dst_step, _dst);
        }
    }

private:
    Mat src;
    Mat dst;

    SrcType thresh;
    SrcType maxval;
    int thresholdType;
};

#ifdef HAVE_OPENCL

static bool ocl_threshold( InputArray _src, OutputArray _dst, double & thresh, double maxval, int thresh_type )
{
    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type),
        kercn = ocl::predictOptimalVectorWidth(_src, _dst), ktype = CV_MAKE_TYPE(depth, kercn);
    bool doubleSupport = ocl::Device::getDefault().doubleFPConfig() > 0;

    if ( !(thresh_type == THRESH_BINARY || thresh_type == THRESH_BINARY_INV || thresh_type == THRESH_TRUNC ||
           thresh_type == THRESH_TOZERO || thresh_type == THRESH_TOZERO_INV) ||
         (!doubleSupport && depth == CV_64F))
        return false;

    const char * const thresholdMap[] = { "THRESH_BINARY", "THRESH_BINARY_INV", "THRESH_TRUNC",
                                          "THRESH_TOZERO", "THRESH_TOZERO_INV" };
    ocl::Device dev = ocl::Device::getDefault();
    int stride_size = dev.isIntel() && (dev.type() & ocl::Device::TYPE_GPU) ? 4 : 1;

    ocl::Kernel k("threshold", ocl::imgproc::threshold_oclsrc,
                  format("-D %s -D T=%s -D T1=%s -D STRIDE_SIZE=%d%s", thresholdMap[thresh_type],
                         ocl::typeToStr(ktype), ocl::typeToStr(depth), stride_size,
                         doubleSupport ? " -D DOUBLE_SUPPORT" : ""));
    if (k.empty())
        return false;

    UMat src = _src.getUMat();
    _dst.create(src.size(), type);
    UMat dst = _dst.getUMat();

    if (depth <= CV_32S)
        thresh = cvFloor(thresh);

    const double min_vals[] = { 0, CHAR_MIN, 0, SHRT_MIN, INT_MIN, -FLT_MAX, -DBL_MAX, 0 };
    double min_val = min_vals[depth];

    k.args(ocl::KernelArg::ReadOnlyNoSize(src), ocl::KernelArg::WriteOnly(dst, cn, kercn),
           ocl::KernelArg::Constant(Mat(1, 1, depth, Scalar::all(thresh))),
           ocl::KernelArg::Constant(Mat(1, 1, depth, Scalar::all(maxval))),
           ocl::KernelArg::Constant(Mat(1, 1, depth, Scalar::all(min_val))));

    size_t globalsize[2] = { (size_t)dst.cols * cn / kercn, (size_t)dst.rows };
    globalsize[1] = (globalsize[1] + stride_size - 1) / stride_size;
    return k.run(2, globalsize, NULL, false);
}

#endif

}

double cv::threshold( InputArray _src, OutputArray _dst, double thresh, double maxval, int type )
{
    CV_OCL_RUN_(_src.dims() <= 2 && _dst.isUMat(),
                ocl_threshold(_src, _dst, thresh, maxval, type), thresh)

    Mat src = _src.getMat();
    int automatic_thresh = (type & ~CV_THRESH_MASK);
    type &= THRESH_MASK;

    CV_Assert( automatic_thresh != (CV_THRESH_OTSU | CV_THRESH_TRIANGLE) );
    if( automatic_thresh == CV_THRESH_OTSU )
    {
        CV_Assert( src.type() == CV_8UC1 );
        thresh = getThreshVal_Otsu_8u( src );
    }
    else if( automatic_thresh == CV_THRESH_TRIANGLE )
    {
        CV_Assert( src.type() == CV_8UC1 );
        thresh = getThreshVal_Triangle_8u( src );
    }

    _dst.create( src.size(), src.type() );
    Mat dst = _dst.getMat();

    if( src.depth() == CV_8U )
    {
        int ithresh = cvFloor(thresh);
        thresh = ithresh;
        int imaxval = cvRound(maxval);
        if( type == THRESH_TRUNC )
            imaxval = ithresh;
        imaxval = saturate_cast<uchar>(imaxval);

        if( ithresh < 0 || ithresh >= 255 )
        {
            if( type == THRESH_BINARY || type == THRESH_BINARY_INV ||
                ((type == THRESH_TRUNC || type == THRESH_TOZERO_INV) && ithresh < 0) ||
                (type == THRESH_TOZERO && ithresh >= 255) )
            {
                int v = type == THRESH_BINARY ? (ithresh >= 255 ? 0 : imaxval) :
                        type == THRESH_BINARY_INV ? (ithresh >= 255 ? imaxval : 0) :
                        /*type == THRESH_TRUNC ? imaxval :*/ 0;
                dst.setTo(v);
            }
            else
                src.copyTo(dst);
            return thresh;
        }
        parallel_for_(Range(0, dst.rows),
                      ThresholdRunner<uchar>(src, dst, (uchar)ithresh, (uchar)imaxval, type),
                      dst.total()/(double)(1<<16));
    }
    else if( src.depth() == CV_16S )
    {
        int ithresh = cvFloor(thresh);
        thresh = ithresh;
        int imaxval = cvRound(maxval);
        if( type == THRESH_TRUNC )
            imaxval = ithresh;
        imaxval = saturate_cast<short>(imaxval);

        if( ithresh < SHRT_MIN || ithresh >= SHRT_MAX )
        {
            if( type == THRESH_BINARY || type == THRESH_BINARY_INV ||
               ((type == THRESH_TRUNC || type == THRESH_TOZERO_INV) && ithresh < SHRT_MIN) ||
               (type == THRESH_TOZERO && ithresh >= SHRT_MAX) )
            {
                int v = type == THRESH_BINARY ? (ithresh >= SHRT_MAX ? 0 : imaxval) :
                type == THRESH_BINARY_INV ? (ithresh >= SHRT_MAX ? imaxval : 0) :
                /*type == THRESH_TRUNC ? imaxval :*/ 0;
                dst.setTo(v);
            }
            else
                src.copyTo(dst);
            return thresh;
        }
        parallel_for_(Range(0, dst.rows),
                      ThresholdRunner<short>(src, dst, (short)ithresh, (short)imaxval, type),
                      dst.total()/(double)(1<<16));
    }
    else if( src.depth() == CV_32F ) {
        parallel_for_(Range(0, dst.rows),
                      ThresholdRunner<float>(src, dst, (float)thresh, (float)maxval, type),
                      dst.total()/(double)(1<<16));
    }
    else if( src.depth() == CV_64F ) {
        parallel_for_(Range(0, dst.rows),
                      ThresholdRunner<double>(src, dst, thresh, maxval, type),
                      dst.total()/(double)(1<<16));
    }
    else
        CV_Error( CV_StsUnsupportedFormat, "" );

    return thresh;
}


void cv::adaptiveThreshold( InputArray _src, OutputArray _dst, double maxValue,
                            int method, int type, int blockSize, double delta )
{
    Mat src = _src.getMat();
    CV_Assert( src.type() == CV_8UC1 );
    CV_Assert( blockSize % 2 == 1 && blockSize > 1 );
    Size size = src.size();

    _dst.create( size, src.type() );
    Mat dst = _dst.getMat();

    if( maxValue < 0 )
    {
        dst = Scalar(0);
        return;
    }

    Mat mean;

    if( src.data != dst.data )
        mean = dst;

    if (method == ADAPTIVE_THRESH_MEAN_C)
        boxFilter( src, mean, src.type(), Size(blockSize, blockSize),
                   Point(-1,-1), true, BORDER_REPLICATE );
    else if (method == ADAPTIVE_THRESH_GAUSSIAN_C)
    {
        Mat srcfloat,meanfloat;
        src.convertTo(srcfloat,CV_32F);
        meanfloat=srcfloat;
        GaussianBlur(srcfloat, meanfloat, Size(blockSize, blockSize), 0, 0, BORDER_REPLICATE);
        meanfloat.convertTo(mean, src.type());
    }
    else
        CV_Error( CV_StsBadFlag, "Unknown/unsupported adaptive threshold method" );

    int i, j;
    uchar imaxval = saturate_cast<uchar>(maxValue);
    int idelta = type == THRESH_BINARY ? cvCeil(delta) : cvFloor(delta);
    uchar tab[768];

    if( type == CV_THRESH_BINARY )
        for( i = 0; i < 768; i++ )
            tab[i] = (uchar)(i - 255 > -idelta ? imaxval : 0);
    else if( type == CV_THRESH_BINARY_INV )
        for( i = 0; i < 768; i++ )
            tab[i] = (uchar)(i - 255 <= -idelta ? imaxval : 0);
    else
        CV_Error( CV_StsBadFlag, "Unknown/unsupported threshold type" );

    if( src.isContinuous() && mean.isContinuous() && dst.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }

    for( i = 0; i < size.height; i++ )
    {
        const uchar* sdata = src.ptr(i);
        const uchar* mdata = mean.ptr(i);
        uchar* ddata = dst.ptr(i);

        for( j = 0; j < size.width; j++ )
            ddata[j] = tab[sdata[j] - mdata[j] + 255];
    }
}

CV_IMPL double
cvThreshold( const void* srcarr, void* dstarr, double thresh, double maxval, int type )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr), dst0 = dst;

    CV_Assert( src.size == dst.size && src.channels() == dst.channels() &&
        (src.depth() == dst.depth() || dst.depth() == CV_8U));

    thresh = cv::threshold( src, dst, thresh, maxval, type );
    if( dst0.data != dst.data )
        dst.convertTo( dst0, dst0.depth() );
    return thresh;
}


CV_IMPL void
cvAdaptiveThreshold( const void *srcIm, void *dstIm, double maxValue,
                     int method, int type, int blockSize, double delta )
{
    cv::Mat src = cv::cvarrToMat(srcIm), dst = cv::cvarrToMat(dstIm);
    CV_Assert( src.size == dst.size && src.type() == dst.type() );
    cv::adaptiveThreshold( src, dst, maxValue, method, type, blockSize, delta );
}

/* End of file. */
