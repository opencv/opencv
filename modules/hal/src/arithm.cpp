/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015, Itseez Inc., all rights reserved.
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
#include "arithm_simd.hpp"
#include "arithm_core.hpp"
#include "replacement.hpp"

namespace cv { namespace hal {

//=======================================

#undef CALL_HAL
#define CALL_HAL(fun) \
    int res = fun(src1, step1, src2, step2, dst, step, width, height); \
    if (res == Error::Ok) \
        return; \
    else if (res != Error::NotImplemented) \
        throw Failure(res);

#if (ARITHM_USE_IPP == 1)
static inline void fixSteps(width, height, size_t elemSize, size_t& step1, size_t& step2, size_t& step)
{
    if( height == 1 )
        step1 = step2 = step = width*elemSize;
}
#define CALL_IPP_BIN_12(fun) \
    CV_IPP_CHECK() \
    { \
        fixSteps(width, height, sizeof(dst[0]), step1, step2, step); \
        if (0 <= fun(src1, (int)step1, src2, (int)step2, dst, (int)step, ippiSize(width, height), 0)) \
        { \
            CV_IMPL_ADD(CV_IMPL_IPP); \
            return; \
        } \
        setIppErrorStatus(); \
    }
#else
#define CALL_IPP_BIN_12(fun)
#endif

//=======================================
// Add
//=======================================

void add8u( const uchar* src1, size_t step1,
                   const uchar* src2, size_t step2,
                   uchar* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_add8u)
    CALL_IPP_BIN_12(ippiAdd_8u_C1RSfs)
    (vBinOp<uchar, cv::OpAdd<uchar>, IF_SIMD(VAdd<uchar>)>(src1, step1, src2, step2, dst, step, width, height));
}

void add8s( const schar* src1, size_t step1,
                   const schar* src2, size_t step2,
                   schar* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_add8s)
    vBinOp<schar, cv::OpAdd<schar>, IF_SIMD(VAdd<schar>)>(src1, step1, src2, step2, dst, step, width, height);
}

void add16u( const ushort* src1, size_t step1,
                    const ushort* src2, size_t step2,
                    ushort* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_add16u)
    CALL_IPP_BIN_12(ippiAdd_16u_C1RSfs)
    (vBinOp<ushort, cv::OpAdd<ushort>, IF_SIMD(VAdd<ushort>)>(src1, step1, src2, step2, dst, step, width, height));
}

void add16s( const short* src1, size_t step1,
                    const short* src2, size_t step2,
                    short* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_add16s)
    CALL_IPP_BIN_12(ippiAdd_16s_C1RSfs)
    (vBinOp<short, cv::OpAdd<short>, IF_SIMD(VAdd<short>)>(src1, step1, src2, step2, dst, step, width, height));
}

void add32s( const int* src1, size_t step1,
                    const int* src2, size_t step2,
                    int* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_add32s)
    vBinOp32<int, cv::OpAdd<int>, IF_SIMD(VAdd<int>)>(src1, step1, src2, step2, dst, step, width, height);
}

void add32f( const float* src1, size_t step1,
                    const float* src2, size_t step2,
                    float* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_add32f)
    CALL_IPP_BIN_12(ippiAdd_32f_C1R)
    (vBinOp32<float, cv::OpAdd<float>, IF_SIMD(VAdd<float>)>(src1, step1, src2, step2, dst, step, width, height));
}

void add64f( const double* src1, size_t step1,
                    const double* src2, size_t step2,
                    double* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_add64f)
    vBinOp64<double, cv::OpAdd<double>, IF_SIMD(VAdd<double>)>(src1, step1, src2, step2, dst, step, width, height);
}

//=======================================

#if (ARITHM_USE_IPP == 1)
#define CALL_IPP_BIN_21(fun) \
    CV_IPP_CHECK() \
    { \
        fixSteps(width, height, sizeof(dst[0]), step1, step2, step); \
        if (0 <= fun(src2, (int)step2, src1, (int)step1, dst, (int)step, ippiSize(width, height), 0)) \
        { \
            CV_IMPL_ADD(CV_IMPL_IPP); \
            return; \
        } \
        setIppErrorStatus(); \
    }
#else
#define CALL_IPP_BIN_21(fun)
#endif

//=======================================
// Subtract
//=======================================

void sub8u( const uchar* src1, size_t step1,
                   const uchar* src2, size_t step2,
                   uchar* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_sub8u)
    CALL_IPP_BIN_21(ippiSub_8u_C1RSfs)
    (vBinOp<uchar, cv::OpSub<uchar>, IF_SIMD(VSub<uchar>)>(src1, step1, src2, step2, dst, step, width, height));
}

void sub8s( const schar* src1, size_t step1,
                   const schar* src2, size_t step2,
                   schar* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_sub8s)
    vBinOp<schar, cv::OpSub<schar>, IF_SIMD(VSub<schar>)>(src1, step1, src2, step2, dst, step, width, height);
}

void sub16u( const ushort* src1, size_t step1,
                    const ushort* src2, size_t step2,
                    ushort* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_sub16u)
    CALL_IPP_BIN_21(ippiSub_16u_C1RSfs)
    (vBinOp<ushort, cv::OpSub<ushort>, IF_SIMD(VSub<ushort>)>(src1, step1, src2, step2, dst, step, width, height));
}

void sub16s( const short* src1, size_t step1,
                    const short* src2, size_t step2,
                    short* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_sub16s)
    CALL_IPP_BIN_21(ippiSub_16s_C1RSfs)
    (vBinOp<short, cv::OpSub<short>, IF_SIMD(VSub<short>)>(src1, step1, src2, step2, dst, step, width, height));
}

void sub32s( const int* src1, size_t step1,
                    const int* src2, size_t step2,
                    int* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_sub32s)
    vBinOp32<int, cv::OpSub<int>, IF_SIMD(VSub<int>)>(src1, step1, src2, step2, dst, step, width, height);
}

void sub32f( const float* src1, size_t step1,
                   const float* src2, size_t step2,
                   float* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_sub32f)
    CALL_IPP_BIN_21(ippiSub_32f_C1R)
    (vBinOp32<float, cv::OpSub<float>, IF_SIMD(VSub<float>)>(src1, step1, src2, step2, dst, step, width, height));
}

void sub64f( const double* src1, size_t step1,
                    const double* src2, size_t step2,
                    double* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_sub64f)
    vBinOp64<double, cv::OpSub<double>, IF_SIMD(VSub<double>)>(src1, step1, src2, step2, dst, step, width, height);
}

//=======================================

#if (ARITHM_USE_IPP == 1)
#define CALL_IPP_MIN_MAX(fun, type) \
    CV_IPP_CHECK() \
    { \
        type* s1 = (type*)src1; \
        type* s2 = (type*)src2; \
        type* d  = dst; \
        fixSteps(width, height, sizeof(dst[0]), step1, step2, step); \
        int i = 0; \
        for(; i < height; i++) \
        { \
            if (0 > fun(s1, s2, d, width)) \
                break; \
            s1 = (type*)((uchar*)s1 + step1); \
            s2 = (type*)((uchar*)s2 + step2); \
            d  = (type*)((uchar*)d + step); \
        } \
        if (i == height) \
        { \
            CV_IMPL_ADD(CV_IMPL_IPP); \
            return; \
        } \
        setIppErrorStatus(); \
    }
#else
#define CALL_IPP_MIN_MAX(fun, type)
#endif

//=======================================
// Max
//=======================================

void max8u( const uchar* src1, size_t step1,
                   const uchar* src2, size_t step2,
                   uchar* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_max8u)
    CALL_IPP_MIN_MAX(ippsMaxEvery_8u, uchar)
    vBinOp<uchar, cv::OpMax<uchar>, IF_SIMD(VMax<uchar>)>(src1, step1, src2, step2, dst, step, width, height);
}

void max8s( const schar* src1, size_t step1,
                   const schar* src2, size_t step2,
                   schar* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_max8s)
    vBinOp<schar, cv::OpMax<schar>, IF_SIMD(VMax<schar>)>(src1, step1, src2, step2, dst, step, width, height);
}

void max16u( const ushort* src1, size_t step1,
                    const ushort* src2, size_t step2,
                    ushort* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_max16u)
    CALL_IPP_MIN_MAX(ippsMaxEvery_16u, ushort)
    vBinOp<ushort, cv::OpMax<ushort>, IF_SIMD(VMax<ushort>)>(src1, step1, src2, step2, dst, step, width, height);
}

void max16s( const short* src1, size_t step1,
                    const short* src2, size_t step2,
                    short* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_max16s)
    vBinOp<short, cv::OpMax<short>, IF_SIMD(VMax<short>)>(src1, step1, src2, step2, dst, step, width, height);
}

void max32s( const int* src1, size_t step1,
                    const int* src2, size_t step2,
                    int* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_max32s)
    vBinOp32<int, cv::OpMax<int>, IF_SIMD(VMax<int>)>(src1, step1, src2, step2, dst, step, width, height);
}

void max32f( const float* src1, size_t step1,
                    const float* src2, size_t step2,
                    float* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_max32f)
    CALL_IPP_MIN_MAX(ippsMaxEvery_32f, float)
    vBinOp32<float, cv::OpMax<float>, IF_SIMD(VMax<float>)>(src1, step1, src2, step2, dst, step, width, height);
}

void max64f( const double* src1, size_t step1,
                    const double* src2, size_t step2,
                    double* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_max64f)
    CALL_IPP_MIN_MAX(ippsMaxEvery_64f, double)
    vBinOp64<double, cv::OpMax<double>, IF_SIMD(VMax<double>)>(src1, step1, src2, step2, dst, step, width, height);
}

//=======================================
// Min
//=======================================

void min8u( const uchar* src1, size_t step1,
                   const uchar* src2, size_t step2,
                   uchar* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_min8u)
    CALL_IPP_MIN_MAX(ippsMinEvery_8u, uchar)
    vBinOp<uchar, cv::OpMin<uchar>, IF_SIMD(VMin<uchar>)>(src1, step1, src2, step2, dst, step, width, height);
}

void min8s( const schar* src1, size_t step1,
                   const schar* src2, size_t step2,
                   schar* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_min8s)
    vBinOp<schar, cv::OpMin<schar>, IF_SIMD(VMin<schar>)>(src1, step1, src2, step2, dst, step, width, height);
}

void min16u( const ushort* src1, size_t step1,
                    const ushort* src2, size_t step2,
                    ushort* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_min16u)
    CALL_IPP_MIN_MAX(ippsMinEvery_16u, ushort)
    vBinOp<ushort, cv::OpMin<ushort>, IF_SIMD(VMin<ushort>)>(src1, step1, src2, step2, dst, step, width, height);
}

void min16s( const short* src1, size_t step1,
                    const short* src2, size_t step2,
                    short* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_min16s)
    vBinOp<short, cv::OpMin<short>, IF_SIMD(VMin<short>)>(src1, step1, src2, step2, dst, step, width, height);
}

void min32s( const int* src1, size_t step1,
                    const int* src2, size_t step2,
                    int* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_min32s)
    vBinOp32<int, cv::OpMin<int>, IF_SIMD(VMin<int>)>(src1, step1, src2, step2, dst, step, width, height);
}

void min32f( const float* src1, size_t step1,
                    const float* src2, size_t step2,
                    float* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_min32f)
    CALL_IPP_MIN_MAX(ippsMinEvery_32f, float)
    vBinOp32<float, cv::OpMin<float>, IF_SIMD(VMin<float>)>(src1, step1, src2, step2, dst, step, width, height);
}

void min64f( const double* src1, size_t step1,
                    const double* src2, size_t step2,
                    double* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_min64f)
    CALL_IPP_MIN_MAX(ippsMinEvery_64f, double)
    vBinOp64<double, cv::OpMin<double>, IF_SIMD(VMin<double>)>(src1, step1, src2, step2, dst, step, width, height);
}

//=======================================
// AbsDiff
//=======================================

void absdiff8u( const uchar* src1, size_t step1,
                       const uchar* src2, size_t step2,
                       uchar* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_absdiff8u)
    CALL_IPP_BIN_12(ippiAbsDiff_8u_C1R)
    (vBinOp<uchar, cv::OpAbsDiff<uchar>, IF_SIMD(VAbsDiff<uchar>)>(src1, step1, src2, step2, dst, step, width, height));
}

void absdiff8s( const schar* src1, size_t step1,
                       const schar* src2, size_t step2,
                       schar* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_absdiff8s)
    vBinOp<schar, cv::OpAbsDiff<schar>, IF_SIMD(VAbsDiff<schar>)>(src1, step1, src2, step2, dst, step, width, height);
}

void absdiff16u( const ushort* src1, size_t step1,
                        const ushort* src2, size_t step2,
                        ushort* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_absdiff16u)
    CALL_IPP_BIN_12(ippiAbsDiff_16u_C1R)
    (vBinOp<ushort, cv::OpAbsDiff<ushort>, IF_SIMD(VAbsDiff<ushort>)>(src1, step1, src2, step2, dst, step, width, height));
}

void absdiff16s( const short* src1, size_t step1,
                        const short* src2, size_t step2,
                        short* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_absdiff16s)
    vBinOp<short, cv::OpAbsDiff<short>, IF_SIMD(VAbsDiff<short>)>(src1, step1, src2, step2, dst, step, width, height);
}

void absdiff32s( const int* src1, size_t step1,
                        const int* src2, size_t step2,
                        int* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_absdiff32s)
    vBinOp32<int, cv::OpAbsDiff<int>, IF_SIMD(VAbsDiff<int>)>(src1, step1, src2, step2, dst, step, width, height);
}

void absdiff32f( const float* src1, size_t step1,
                        const float* src2, size_t step2,
                        float* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_absdiff32f)
    CALL_IPP_BIN_12(ippiAbsDiff_32f_C1R)
    (vBinOp32<float, cv::OpAbsDiff<float>, IF_SIMD(VAbsDiff<float>)>(src1, step1, src2, step2, dst, step, width, height));
}

void absdiff64f( const double* src1, size_t step1,
                        const double* src2, size_t step2,
                        double* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_absdiff64f)
    vBinOp64<double, cv::OpAbsDiff<double>, IF_SIMD(VAbsDiff<double>)>(src1, step1, src2, step2, dst, step, width, height);
}

//=======================================
// Logical
//=======================================

void and8u( const uchar* src1, size_t step1,
                   const uchar* src2, size_t step2,
                   uchar* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_and8u)
    CALL_IPP_BIN_12(ippiAnd_8u_C1R)
    (vBinOp<uchar, cv::OpAnd<uchar>, IF_SIMD(VAnd<uchar>)>(src1, step1, src2, step2, dst, step, width, height));
}

void or8u( const uchar* src1, size_t step1,
                  const uchar* src2, size_t step2,
                  uchar* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_or8u)
    CALL_IPP_BIN_12(ippiOr_8u_C1R)
    (vBinOp<uchar, cv::OpOr<uchar>, IF_SIMD(VOr<uchar>)>(src1, step1, src2, step2, dst, step, width, height));
}

void xor8u( const uchar* src1, size_t step1,
                   const uchar* src2, size_t step2,
                   uchar* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_xor8u)
    CALL_IPP_BIN_12(ippiXor_8u_C1R)
    (vBinOp<uchar, cv::OpXor<uchar>, IF_SIMD(VXor<uchar>)>(src1, step1, src2, step2, dst, step, width, height));
}

void not8u( const uchar* src1, size_t step1,
                   const uchar* src2, size_t step2,
                   uchar* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(hal_not8u)
    CALL_IPP_BIN_12(ippiNot_8u_C1R)
    (vBinOp<uchar, cv::OpNot<uchar>, IF_SIMD(VNot<uchar>)>(src1, step1, src2, step2, dst, step, width, height));
}

//=======================================

#undef CALL_HAL
#define CALL_HAL(fun) \
    int res = fun(src1, step1, src2, step2, dst, step, width, height, *(int*)_cmpop); \
    if (res == Error::Ok) \
        return; \
    else if (res != Error::NotImplemented) \
        throw Failure(res);

#if ARITHM_USE_IPP
inline static IppCmpOp convert_cmp(int _cmpop)
{
    return _cmpop == CMP_EQ ? ippCmpEq :
        _cmpop == CMP_GT ? ippCmpGreater :
        _cmpop == CMP_GE ? ippCmpGreaterEq :
        _cmpop == CMP_LT ? ippCmpLess :
        _cmpop == CMP_LE ? ippCmpLessEq :
        (IppCmpOp)-1;
}
#define CALL_IPP_CMP(fun) \
    CV_IPP_CHECK() \
    { \
        IppCmpOp op = convert_cmp(*(int *)_cmpop); \
        if( op  >= 0 ) \
        { \
            fixSteps(width, height, sizeof(dst[0]), step1, step2, step); \
            if (0 <= fun(src1, (int)step1, src2, (int)step2, dst, (int)step, ippiSize(width, height), op)) \
            { \
                CV_IMPL_ADD(CV_IMPL_IPP); \
                return; \
            } \
            setIppErrorStatus(); \
        } \
    }
#else
#define CALL_IPP_CMP(fun)
#endif

//=======================================
// Compare
//=======================================

void cmp8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2,
                  uchar* dst, size_t step, int width, int height, void* _cmpop)
{
    CALL_HAL(hal_cmp8u)
    CALL_IPP_CMP(ippiCompare_8u_C1R)
  //vz optimized  cmp_(src1, step1, src2, step2, dst, step, width, height, *(int*)_cmpop);
    int code = *(int*)_cmpop;
    step1 /= sizeof(src1[0]);
    step2 /= sizeof(src2[0]);
    if( code == CMP_GE || code == CMP_LT )
    {
        std::swap(src1, src2);
        std::swap(step1, step2);
        code = code == CMP_GE ? CMP_LE : CMP_GT;
    }

    if( code == CMP_GT || code == CMP_LE )
    {
        int m = code == CMP_GT ? 0 : 255;
        for( ; height--; src1 += step1, src2 += step2, dst += step )
        {
            int x =0;
            #if CV_SSE2
            if( USE_SSE2 )
            {
                __m128i m128 = code == CMP_GT ? _mm_setzero_si128() : _mm_set1_epi8 (-1);
                __m128i c128 = _mm_set1_epi8 (-128);
                for( ; x <= width - 16; x += 16 )
                {
                    __m128i r00 = _mm_loadu_si128((const __m128i*)(src1 + x));
                    __m128i r10 = _mm_loadu_si128((const __m128i*)(src2 + x));
                    // no simd for 8u comparison, that's why we need the trick
                    r00 = _mm_sub_epi8(r00,c128);
                    r10 = _mm_sub_epi8(r10,c128);

                    r00 =_mm_xor_si128(_mm_cmpgt_epi8(r00, r10), m128);
                    _mm_storeu_si128((__m128i*)(dst + x),r00);

                }
            }
            #elif CV_NEON
            uint8x16_t mask = code == CMP_GT ? vdupq_n_u8(0) : vdupq_n_u8(255);

            for( ; x <= width - 16; x += 16 )
            {
                vst1q_u8(dst+x, veorq_u8(vcgtq_u8(vld1q_u8(src1+x), vld1q_u8(src2+x)), mask));
            }

           #endif

            for( ; x < width; x++ ){
                dst[x] = (uchar)(-(src1[x] > src2[x]) ^ m);
            }
        }
    }
    else if( code == CMP_EQ || code == CMP_NE )
    {
        int m = code == CMP_EQ ? 0 : 255;
        for( ; height--; src1 += step1, src2 += step2, dst += step )
        {
            int x = 0;
            #if CV_SSE2
            if( USE_SSE2 )
            {
                __m128i m128 =  code == CMP_EQ ? _mm_setzero_si128() : _mm_set1_epi8 (-1);
                for( ; x <= width - 16; x += 16 )
                {
                    __m128i r00 = _mm_loadu_si128((const __m128i*)(src1 + x));
                    __m128i r10 = _mm_loadu_si128((const __m128i*)(src2 + x));
                    r00 = _mm_xor_si128 ( _mm_cmpeq_epi8 (r00, r10), m128);
                    _mm_storeu_si128((__m128i*)(dst + x), r00);
                }
            }
            #elif CV_NEON
            uint8x16_t mask = code == CMP_EQ ? vdupq_n_u8(0) : vdupq_n_u8(255);

            for( ; x <= width - 16; x += 16 )
            {
                vst1q_u8(dst+x, veorq_u8(vceqq_u8(vld1q_u8(src1+x), vld1q_u8(src2+x)), mask));
            }
           #endif
           for( ; x < width; x++ )
                dst[x] = (uchar)(-(src1[x] == src2[x]) ^ m);
        }
    }
}

void cmp8s(const schar* src1, size_t step1, const schar* src2, size_t step2,
                  uchar* dst, size_t step, int width, int height, void* _cmpop)
{
    CALL_HAL(hal_cmp8s)
    cmp_(src1, step1, src2, step2, dst, step, width, height, *(int*)_cmpop);
}

void cmp16u(const ushort* src1, size_t step1, const ushort* src2, size_t step2,
                  uchar* dst, size_t step, int width, int height, void* _cmpop)
{
    CALL_HAL(hal_cmp16u)
    CALL_IPP_CMP(ippiCompare_16u_C1R)
    cmp_(src1, step1, src2, step2, dst, step, width, height, *(int*)_cmpop);
}

void cmp16s(const short* src1, size_t step1, const short* src2, size_t step2,
                  uchar* dst, size_t step, int width, int height, void* _cmpop)
{
    CALL_HAL(hal_cmp16s)
    CALL_IPP_CMP(ippiCompare_16s_C1R)
   //vz optimized cmp_(src1, step1, src2, step2, dst, step, width, height, *(int*)_cmpop);

    int code = *(int*)_cmpop;
    step1 /= sizeof(src1[0]);
    step2 /= sizeof(src2[0]);
    if( code == CMP_GE || code == CMP_LT )
    {
        std::swap(src1, src2);
        std::swap(step1, step2);
        code = code == CMP_GE ? CMP_LE : CMP_GT;
    }

    if( code == CMP_GT || code == CMP_LE )
    {
        int m = code == CMP_GT ? 0 : 255;
        for( ; height--; src1 += step1, src2 += step2, dst += step )
        {
            int x =0;
            #if CV_SSE2
            if( USE_SSE2)
            {
                __m128i m128 =  code == CMP_GT ? _mm_setzero_si128() : _mm_set1_epi16 (-1);
                for( ; x <= width - 16; x += 16 )
                {
                    __m128i r00 = _mm_loadu_si128((const __m128i*)(src1 + x));
                    __m128i r10 = _mm_loadu_si128((const __m128i*)(src2 + x));
                    r00 = _mm_xor_si128 ( _mm_cmpgt_epi16 (r00, r10), m128);
                    __m128i r01 = _mm_loadu_si128((const __m128i*)(src1 + x + 8));
                    __m128i r11 = _mm_loadu_si128((const __m128i*)(src2 + x + 8));
                    r01 = _mm_xor_si128 ( _mm_cmpgt_epi16 (r01, r11), m128);
                    r11 = _mm_packs_epi16(r00, r01);
                    _mm_storeu_si128((__m128i*)(dst + x), r11);
                }
                if( x <= width-8)
                {
                    __m128i r00 = _mm_loadu_si128((const __m128i*)(src1 + x));
                    __m128i r10 = _mm_loadu_si128((const __m128i*)(src2 + x));
                    r00 = _mm_xor_si128 ( _mm_cmpgt_epi16 (r00, r10), m128);
                    r10 = _mm_packs_epi16(r00, r00);
                    _mm_storel_epi64((__m128i*)(dst + x), r10);

                    x += 8;
                }
            }
            #elif CV_NEON
            uint8x16_t mask = code == CMP_GT ? vdupq_n_u8(0) : vdupq_n_u8(255);

            for( ; x <= width - 16; x += 16 )
            {
                int16x8_t in1 = vld1q_s16(src1 + x);
                int16x8_t in2 = vld1q_s16(src2 + x);
                uint8x8_t t1 = vmovn_u16(vcgtq_s16(in1, in2));

                in1 = vld1q_s16(src1 + x + 8);
                in2 = vld1q_s16(src2 + x + 8);
                uint8x8_t t2 = vmovn_u16(vcgtq_s16(in1, in2));

                vst1q_u8(dst+x, veorq_u8(vcombine_u8(t1, t2), mask));
            }
            #endif

            for( ; x < width; x++ ){
                 dst[x] = (uchar)(-(src1[x] > src2[x]) ^ m);
            }
        }
    }
    else if( code == CMP_EQ || code == CMP_NE )
    {
        int m = code == CMP_EQ ? 0 : 255;
        for( ; height--; src1 += step1, src2 += step2, dst += step )
        {
            int x = 0;
            #if CV_SSE2
            if( USE_SSE2 )
            {
                __m128i m128 =  code == CMP_EQ ? _mm_setzero_si128() : _mm_set1_epi16 (-1);
                for( ; x <= width - 16; x += 16 )
                {
                    __m128i r00 = _mm_loadu_si128((const __m128i*)(src1 + x));
                    __m128i r10 = _mm_loadu_si128((const __m128i*)(src2 + x));
                    r00 = _mm_xor_si128 ( _mm_cmpeq_epi16 (r00, r10), m128);
                    __m128i r01 = _mm_loadu_si128((const __m128i*)(src1 + x + 8));
                    __m128i r11 = _mm_loadu_si128((const __m128i*)(src2 + x + 8));
                    r01 = _mm_xor_si128 ( _mm_cmpeq_epi16 (r01, r11), m128);
                    r11 = _mm_packs_epi16(r00, r01);
                    _mm_storeu_si128((__m128i*)(dst + x), r11);
                }
                if( x <= width - 8)
                {
                    __m128i r00 = _mm_loadu_si128((const __m128i*)(src1 + x));
                    __m128i r10 = _mm_loadu_si128((const __m128i*)(src2 + x));
                    r00 = _mm_xor_si128 ( _mm_cmpeq_epi16 (r00, r10), m128);
                    r10 = _mm_packs_epi16(r00, r00);
                    _mm_storel_epi64((__m128i*)(dst + x), r10);

                    x += 8;
                }
            }
            #elif CV_NEON
            uint8x16_t mask = code == CMP_EQ ? vdupq_n_u8(0) : vdupq_n_u8(255);

            for( ; x <= width - 16; x += 16 )
            {
                int16x8_t in1 = vld1q_s16(src1 + x);
                int16x8_t in2 = vld1q_s16(src2 + x);
                uint8x8_t t1 = vmovn_u16(vceqq_s16(in1, in2));

                in1 = vld1q_s16(src1 + x + 8);
                in2 = vld1q_s16(src2 + x + 8);
                uint8x8_t t2 = vmovn_u16(vceqq_s16(in1, in2));

                vst1q_u8(dst+x, veorq_u8(vcombine_u8(t1, t2), mask));
            }
            #endif
            for( ; x < width; x++ )
                dst[x] = (uchar)(-(src1[x] == src2[x]) ^ m);
        }
    }
}

void cmp32s(const int* src1, size_t step1, const int* src2, size_t step2,
                   uchar* dst, size_t step, int width, int height, void* _cmpop)
{
    CALL_HAL(hal_cmp32s)
    cmp_(src1, step1, src2, step2, dst, step, width, height, *(int*)_cmpop);
}

void cmp32f(const float* src1, size_t step1, const float* src2, size_t step2,
                  uchar* dst, size_t step, int width, int height, void* _cmpop)
{
    CALL_HAL(hal_cmp32f)
    CALL_IPP_CMP(ippiCompare_32f_C1R)
    cmp_(src1, step1, src2, step2, dst, step, width, height, *(int*)_cmpop);
}

void cmp64f(const double* src1, size_t step1, const double* src2, size_t step2,
                  uchar* dst, size_t step, int width, int height, void* _cmpop)
{
    CALL_HAL(hal_cmp64f)
    cmp_(src1, step1, src2, step2, dst, step, width, height, *(int*)_cmpop);
}

//=======================================

#undef CALL_HAL
#define CALL_HAL(fun) \
    int res = fun(src1, step1, src2, step2, dst, step, width, height, *(const double*)scale); \
    if (res == Error::Ok) \
        return; \
    else if (res != Error::NotImplemented) \
        throw Failure(res);

#if defined HAVE_IPP
#define CALL_IPP_MUL(fun) \
    CV_IPP_CHECK() \
    { \
        if (std::fabs(fscale - 1) <= FLT_EPSILON) \
        { \
            if (fun(src1, (int)step1, src2, (int)step2, dst, (int)step, ippiSize(width, height), 0) >= 0) \
            { \
                CV_IMPL_ADD(CV_IMPL_IPP); \
                return; \
            } \
            setIppErrorStatus(); \
        } \
    }
#else
#define CALL_IPP_MUL(fun)
#endif

//=======================================
// Multilpy
//=======================================

void mul8u( const uchar* src1, size_t step1, const uchar* src2, size_t step2,
                   uchar* dst, size_t step, int width, int height, void* scale)
{
    CALL_HAL(hal_mul8u)
    float fscale = (float)*(const double*)scale;
    CALL_IPP_MUL(ippiMul_8u_C1RSfs)
    mul_(src1, step1, src2, step2, dst, step, width, height, fscale);
}

void mul8s( const schar* src1, size_t step1, const schar* src2, size_t step2,
                   schar* dst, size_t step, int width, int height, void* scale)
{
    CALL_HAL(hal_mul8s)
    mul_(src1, step1, src2, step2, dst, step, width, height, (float)*(const double*)scale);
}

void mul16u( const ushort* src1, size_t step1, const ushort* src2, size_t step2,
                    ushort* dst, size_t step, int width, int height, void* scale)
{
    CALL_HAL(hal_mul16u)
    float fscale = (float)*(const double*)scale;
    CALL_IPP_MUL(ippiMul_16u_C1RSfs)
    mul_(src1, step1, src2, step2, dst, step, width, height, fscale);
}

void mul16s( const short* src1, size_t step1, const short* src2, size_t step2,
                    short* dst, size_t step, int width, int height, void* scale)
{
    CALL_HAL(hal_mul16s)
    float fscale = (float)*(const double*)scale;
    CALL_IPP_MUL(ippiMul_16s_C1RSfs)
    mul_(src1, step1, src2, step2, dst, step, width, height, fscale);
}

void mul32s( const int* src1, size_t step1, const int* src2, size_t step2,
                    int* dst, size_t step, int width, int height, void* scale)
{
    CALL_HAL(hal_mul32s)
    mul_(src1, step1, src2, step2, dst, step, width, height, *(const double*)scale);
}

void mul32f( const float* src1, size_t step1, const float* src2, size_t step2,
                    float* dst, size_t step, int width, int height, void* scale)
{
    CALL_HAL(hal_mul32f)
    float fscale = (float)*(const double*)scale;
    CALL_IPP_MUL(ippiMul_32f_C1R)
    mul_(src1, step1, src2, step2, dst, step, width, height, fscale);
}

void mul64f( const double* src1, size_t step1, const double* src2, size_t step2,
                    double* dst, size_t step, int width, int height, void* scale)
{
    CALL_HAL(hal_mul64f)
    mul_(src1, step1, src2, step2, dst, step, width, height, *(const double*)scale);
}

//=======================================
// Divide
//=======================================

void div8u( const uchar* src1, size_t step1, const uchar* src2, size_t step2,
                   uchar* dst, size_t step, int width, int height, void* scale)
{
    CALL_HAL(hal_div8u)
    if( src1 )
        div_i(src1, step1, src2, step2, dst, step, width, height, *(const double*)scale);
    else
        recip_i(src1, step1, src2, step2, dst, step, width, height, *(const double*)scale);
}

void div8s( const schar* src1, size_t step1, const schar* src2, size_t step2,
                  schar* dst, size_t step, int width, int height, void* scale)
{
    CALL_HAL(hal_div8s)
    div_i(src1, step1, src2, step2, dst, step, width, height, *(const double*)scale);
}

void div16u( const ushort* src1, size_t step1, const ushort* src2, size_t step2,
                    ushort* dst, size_t step, int width, int height, void* scale)
{
    CALL_HAL(hal_div16u)
    div_i(src1, step1, src2, step2, dst, step, width, height, *(const double*)scale);
}

void div16s( const short* src1, size_t step1, const short* src2, size_t step2,
                    short* dst, size_t step, int width, int height, void* scale)
{
    CALL_HAL(hal_div16s)
    div_i(src1, step1, src2, step2, dst, step, width, height, *(const double*)scale);
}

void div32s( const int* src1, size_t step1, const int* src2, size_t step2,
                    int* dst, size_t step, int width, int height, void* scale)
{
    CALL_HAL(hal_div32s)
    div_i(src1, step1, src2, step2, dst, step, width, height, *(const double*)scale);
}

void div32f( const float* src1, size_t step1, const float* src2, size_t step2,
                    float* dst, size_t step, int width, int height, void* scale)
{
    CALL_HAL(hal_div32f)
    div_f(src1, step1, src2, step2, dst, step, width, height, *(const double*)scale);
}

void div64f( const double* src1, size_t step1, const double* src2, size_t step2,
                    double* dst, size_t step, int width, int height, void* scale)
{
    CALL_HAL(hal_div64f)
    div_f(src1, step1, src2, step2, dst, step, width, height, *(const double*)scale);
}

//=======================================
// Reciprocial
//=======================================

void recip8u( const uchar* src1, size_t step1, const uchar* src2, size_t step2,
                  uchar* dst, size_t step, int width, int height, void* scale)
{
    CALL_HAL(hal_recip8u)
    recip_i(src1, step1, src2, step2, dst, step, width, height, *(const double*)scale);
}

void recip8s( const schar* src1, size_t step1, const schar* src2, size_t step2,
                  schar* dst, size_t step, int width, int height, void* scale)
{
    CALL_HAL(hal_recip8s)
    recip_i(src1, step1, src2, step2, dst, step, width, height, *(const double*)scale);
}

void recip16u( const ushort* src1, size_t step1, const ushort* src2, size_t step2,
                   ushort* dst, size_t step, int width, int height, void* scale)
{
    CALL_HAL(hal_recip16u)
    recip_i(src1, step1, src2, step2, dst, step, width, height, *(const double*)scale);
}

void recip16s( const short* src1, size_t step1, const short* src2, size_t step2,
                   short* dst, size_t step, int width, int height, void* scale)
{
    CALL_HAL(hal_recip16s)
    recip_i(src1, step1, src2, step2, dst, step, width, height, *(const double*)scale);
}

void recip32s( const int* src1, size_t step1, const int* src2, size_t step2,
                   int* dst, size_t step, int width, int height, void* scale)
{
    CALL_HAL(hal_recip32s)
    recip_i(src1, step1, src2, step2, dst, step, width, height, *(const double*)scale);
}

void recip32f( const float* src1, size_t step1, const float* src2, size_t step2,
                   float* dst, size_t step, int width, int height, void* scale)
{
    CALL_HAL(hal_recip32f)
    recip_f(src1, step1, src2, step2, dst, step, width, height, *(const double*)scale);
}

void recip64f( const double* src1, size_t step1, const double* src2, size_t step2,
                   double* dst, size_t step, int width, int height, void* scale)
{
    CALL_HAL(hal_recip64f)
    recip_f(src1, step1, src2, step2, dst, step, width, height, *(const double*)scale);
}

//=======================================

#undef CALL_HAL
#define CALL_HAL(fun) \
    int res = fun(src1, step1, src2, step2, dst, step, width, height, scalars); \
    if (res == Error::Ok) \
        return; \
    else if (res != Error::NotImplemented) \
        throw Failure(res);

//=======================================
// Add weighted
//=======================================

void
addWeighted8u( const uchar* src1, size_t step1,
               const uchar* src2, size_t step2,
               uchar* dst, size_t step, int width, int height,
               void* scalars )
{
    CALL_HAL(hal_addWeighted8u)
    const double* scalars_ = (const double*)scalars;
    float alpha = (float)scalars_[0], beta = (float)scalars_[1], gamma = (float)scalars_[2];

    for( ; height--; src1 += step1, src2 += step2, dst += step )
    {
        int x = 0;

#if CV_SSE2
        if( USE_SSE2 )
        {
            __m128 a4 = _mm_set1_ps(alpha), b4 = _mm_set1_ps(beta), g4 = _mm_set1_ps(gamma);
            __m128i z = _mm_setzero_si128();

            for( ; x <= width - 8; x += 8 )
            {
                __m128i u = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(src1 + x)), z);
                __m128i v = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(src2 + x)), z);

                __m128 u0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(u, z));
                __m128 u1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(u, z));
                __m128 v0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v, z));
                __m128 v1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v, z));

                u0 = _mm_add_ps(_mm_mul_ps(u0, a4), _mm_mul_ps(v0, b4));
                u1 = _mm_add_ps(_mm_mul_ps(u1, a4), _mm_mul_ps(v1, b4));
                u0 = _mm_add_ps(u0, g4); u1 = _mm_add_ps(u1, g4);

                u = _mm_packs_epi32(_mm_cvtps_epi32(u0), _mm_cvtps_epi32(u1));
                u = _mm_packus_epi16(u, u);

                _mm_storel_epi64((__m128i*)(dst + x), u);
            }
        }
#elif CV_NEON
        float32x4_t g = vdupq_n_f32 (gamma);

        for( ; x <= width - 8; x += 8 )
        {
            uint8x8_t in1 = vld1_u8(src1+x);
            uint16x8_t in1_16 = vmovl_u8(in1);
            float32x4_t in1_f_l = vcvtq_f32_u32(vmovl_u16(vget_low_u16(in1_16)));
            float32x4_t in1_f_h = vcvtq_f32_u32(vmovl_u16(vget_high_u16(in1_16)));

            uint8x8_t in2 = vld1_u8(src2+x);
            uint16x8_t in2_16 = vmovl_u8(in2);
            float32x4_t in2_f_l = vcvtq_f32_u32(vmovl_u16(vget_low_u16(in2_16)));
            float32x4_t in2_f_h = vcvtq_f32_u32(vmovl_u16(vget_high_u16(in2_16)));

            float32x4_t out_f_l = vaddq_f32(vmulq_n_f32(in1_f_l, alpha), vmulq_n_f32(in2_f_l, beta));
            float32x4_t out_f_h = vaddq_f32(vmulq_n_f32(in1_f_h, alpha), vmulq_n_f32(in2_f_h, beta));
            out_f_l = vaddq_f32(out_f_l, g);
            out_f_h = vaddq_f32(out_f_h, g);

            uint16x4_t out_16_l = vqmovun_s32(cv_vrndq_s32_f32(out_f_l));
            uint16x4_t out_16_h = vqmovun_s32(cv_vrndq_s32_f32(out_f_h));

            uint16x8_t out_16 = vcombine_u16(out_16_l, out_16_h);
            uint8x8_t out = vqmovn_u16(out_16);

            vst1_u8(dst+x, out);
        }
#endif
        #if CV_ENABLE_UNROLLED
        for( ; x <= width - 4; x += 4 )
        {
            float t0, t1;
            t0 = CV_8TO32F(src1[x])*alpha + CV_8TO32F(src2[x])*beta + gamma;
            t1 = CV_8TO32F(src1[x+1])*alpha + CV_8TO32F(src2[x+1])*beta + gamma;

            dst[x] = saturate_cast<uchar>(t0);
            dst[x+1] = saturate_cast<uchar>(t1);

            t0 = CV_8TO32F(src1[x+2])*alpha + CV_8TO32F(src2[x+2])*beta + gamma;
            t1 = CV_8TO32F(src1[x+3])*alpha + CV_8TO32F(src2[x+3])*beta + gamma;

            dst[x+2] = saturate_cast<uchar>(t0);
            dst[x+3] = saturate_cast<uchar>(t1);
        }
        #endif

        for( ; x < width; x++ )
        {
            float t0 = CV_8TO32F(src1[x])*alpha + CV_8TO32F(src2[x])*beta + gamma;
            dst[x] = saturate_cast<uchar>(t0);
        }
    }
}

void addWeighted8s( const schar* src1, size_t step1, const schar* src2, size_t step2,
                           schar* dst, size_t step, int width, int height, void* scalars )
{
    CALL_HAL(hal_addWeighted8s)
    addWeighted_<schar, float>(src1, step1, src2, step2, dst, step, width, height, scalars);
}

void addWeighted16u( const ushort* src1, size_t step1, const ushort* src2, size_t step2,
                            ushort* dst, size_t step, int width, int height, void* scalars )
{
    CALL_HAL(hal_addWeighted16u)
    addWeighted_<ushort, float>(src1, step1, src2, step2, dst, step, width, height, scalars);
}

void addWeighted16s( const short* src1, size_t step1, const short* src2, size_t step2,
                            short* dst, size_t step, int width, int height, void* scalars )
{
    CALL_HAL(hal_addWeighted16s)
    addWeighted_<short, float>(src1, step1, src2, step2, dst, step, width, height, scalars);
}

void addWeighted32s( const int* src1, size_t step1, const int* src2, size_t step2,
                            int* dst, size_t step, int width, int height, void* scalars )
{
    CALL_HAL(hal_addWeighted32s)
    addWeighted_<int, double>(src1, step1, src2, step2, dst, step, width, height, scalars);
}

void addWeighted32f( const float* src1, size_t step1, const float* src2, size_t step2,
                            float* dst, size_t step, int width, int height, void* scalars )
{
    CALL_HAL(hal_addWeighted32f)
    addWeighted_<float, double>(src1, step1, src2, step2, dst, step, width, height, scalars);
}

void addWeighted64f( const double* src1, size_t step1, const double* src2, size_t step2,
                            double* dst, size_t step, int width, int height, void* scalars )
{
    CALL_HAL(hal_addWeighted64f)
    addWeighted_<double, double>(src1, step1, src2, step2, dst, step, width, height, scalars);
}

}} // cv::hal::
