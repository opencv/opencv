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

/* ////////////////////////////////////////////////////////////////////
//
//  Arithmetic and logical operations: +, -, *, /, &, |, ^, ~, abs ...
//
// */

#include "precomp.hpp"
#include "opencl_kernels.hpp"

namespace cv
{

struct NOP {};

#if CV_SSE2

#define FUNCTOR_TEMPLATE(name)          \
    template<typename T> struct name {}

FUNCTOR_TEMPLATE(VLoadStore128);
FUNCTOR_TEMPLATE(VLoadStore64);
FUNCTOR_TEMPLATE(VLoadStore128Aligned);

#endif

template<typename T, class Op, class VOp>
void vBinOp(const T* src1, size_t step1, const T* src2, size_t step2, T* dst, size_t step, Size sz)
{
#if CV_SSE2
    VOp vop;
#endif
    Op op;

    for( ; sz.height--; src1 += step1/sizeof(src1[0]),
                        src2 += step2/sizeof(src2[0]),
                        dst += step/sizeof(dst[0]) )
    {
        int x = 0;

#if CV_SSE2
        if( USE_SSE2 )
        {
            for( ; x <= sz.width - 32/(int)sizeof(T); x += 32/sizeof(T) )
            {
                typename VLoadStore128<T>::reg_type r0 = VLoadStore128<T>::load(src1 + x               );
                typename VLoadStore128<T>::reg_type r1 = VLoadStore128<T>::load(src1 + x + 16/sizeof(T));
                r0 = vop(r0, VLoadStore128<T>::load(src2 + x               ));
                r1 = vop(r1, VLoadStore128<T>::load(src2 + x + 16/sizeof(T)));
                VLoadStore128<T>::store(dst + x               , r0);
                VLoadStore128<T>::store(dst + x + 16/sizeof(T), r1);
            }
        }
#endif
#if CV_SSE2
        if( USE_SSE2 )
        {
            for( ; x <= sz.width - 8/(int)sizeof(T); x += 8/sizeof(T) )
            {
                typename VLoadStore64<T>::reg_type r = VLoadStore64<T>::load(src1 + x);
                r = vop(r, VLoadStore64<T>::load(src2 + x));
                VLoadStore64<T>::store(dst + x, r);
            }
        }
#endif
#if CV_ENABLE_UNROLLED
        for( ; x <= sz.width - 4; x += 4 )
        {
            T v0 = op(src1[x], src2[x]);
            T v1 = op(src1[x+1], src2[x+1]);
            dst[x] = v0; dst[x+1] = v1;
            v0 = op(src1[x+2], src2[x+2]);
            v1 = op(src1[x+3], src2[x+3]);
            dst[x+2] = v0; dst[x+3] = v1;
        }
#endif

        for( ; x < sz.width; x++ )
            dst[x] = op(src1[x], src2[x]);
    }
}

template<typename T, class Op, class Op32>
void vBinOp32(const T* src1, size_t step1, const T* src2, size_t step2,
              T* dst, size_t step, Size sz)
{
#if CV_SSE2
    Op32 op32;
#endif
    Op op;

    for( ; sz.height--; src1 += step1/sizeof(src1[0]),
        src2 += step2/sizeof(src2[0]),
        dst += step/sizeof(dst[0]) )
    {
        int x = 0;

#if CV_SSE2
        if( USE_SSE2 )
        {
            if( (((size_t)src1|(size_t)src2|(size_t)dst)&15) == 0 )
            {
                for( ; x <= sz.width - 8; x += 8 )
                {
                    typename VLoadStore128Aligned<T>::reg_type r0 = VLoadStore128Aligned<T>::load(src1 + x    );
                    typename VLoadStore128Aligned<T>::reg_type r1 = VLoadStore128Aligned<T>::load(src1 + x + 4);
                    r0 = op32(r0, VLoadStore128Aligned<T>::load(src2 + x    ));
                    r1 = op32(r1, VLoadStore128Aligned<T>::load(src2 + x + 4));
                    VLoadStore128Aligned<T>::store(dst + x    , r0);
                    VLoadStore128Aligned<T>::store(dst + x + 4, r1);
                }
            }
        }
#endif
#if CV_SSE2
        if( USE_SSE2 )
        {
            for( ; x <= sz.width - 8; x += 8 )
            {
                typename VLoadStore128<T>::reg_type r0 = VLoadStore128<T>::load(src1 + x    );
                typename VLoadStore128<T>::reg_type r1 = VLoadStore128<T>::load(src1 + x + 4);
                r0 = op32(r0, VLoadStore128<T>::load(src2 + x    ));
                r1 = op32(r1, VLoadStore128<T>::load(src2 + x + 4));
                VLoadStore128<T>::store(dst + x    , r0);
                VLoadStore128<T>::store(dst + x + 4, r1);
            }
        }
#endif
#if CV_ENABLE_UNROLLED
        for( ; x <= sz.width - 4; x += 4 )
        {
            T v0 = op(src1[x], src2[x]);
            T v1 = op(src1[x+1], src2[x+1]);
            dst[x] = v0; dst[x+1] = v1;
            v0 = op(src1[x+2], src2[x+2]);
            v1 = op(src1[x+3], src2[x+3]);
            dst[x+2] = v0; dst[x+3] = v1;
        }
#endif

        for( ; x < sz.width; x++ )
            dst[x] = op(src1[x], src2[x]);
    }
}


template<typename T, class Op, class Op64>
void vBinOp64(const T* src1, size_t step1, const T* src2, size_t step2,
               T* dst, size_t step, Size sz)
{
#if CV_SSE2
    Op64 op64;
#endif
    Op op;

    for( ; sz.height--; src1 += step1/sizeof(src1[0]),
        src2 += step2/sizeof(src2[0]),
        dst += step/sizeof(dst[0]) )
    {
        int x = 0;

#if CV_SSE2
        if( USE_SSE2 )
        {
            if( (((size_t)src1|(size_t)src2|(size_t)dst)&15) == 0 )
            {
                for( ; x <= sz.width - 4; x += 4 )
                {
                    typename VLoadStore128Aligned<T>::reg_type r0 = VLoadStore128Aligned<T>::load(src1 + x    );
                    typename VLoadStore128Aligned<T>::reg_type r1 = VLoadStore128Aligned<T>::load(src1 + x + 2);
                    r0 = op64(r0, VLoadStore128Aligned<T>::load(src2 + x    ));
                    r1 = op64(r1, VLoadStore128Aligned<T>::load(src2 + x + 2));
                    VLoadStore128Aligned<T>::store(dst + x    , r0);
                    VLoadStore128Aligned<T>::store(dst + x + 2, r1);
                }
            }
        }
#endif

        for( ; x <= sz.width - 4; x += 4 )
        {
            T v0 = op(src1[x], src2[x]);
            T v1 = op(src1[x+1], src2[x+1]);
            dst[x] = v0; dst[x+1] = v1;
            v0 = op(src1[x+2], src2[x+2]);
            v1 = op(src1[x+3], src2[x+3]);
            dst[x+2] = v0; dst[x+3] = v1;
        }

        for( ; x < sz.width; x++ )
            dst[x] = op(src1[x], src2[x]);
    }
}

#if CV_SSE2

#define FUNCTOR_LOADSTORE_CAST(name, template_arg, register_type, load_body, store_body)\
    template <>                                                                                  \
    struct name<template_arg>{                                                                   \
        typedef register_type reg_type;                                                          \
        static reg_type load(const template_arg * p) { return load_body ((const reg_type *)p); } \
        static void store(template_arg * p, reg_type v) { store_body ((reg_type *)p, v); }       \
    }

#define FUNCTOR_LOADSTORE(name, template_arg, register_type, load_body, store_body)\
    template <>                                                                \
    struct name<template_arg>{                                                 \
        typedef register_type reg_type;                                        \
        static reg_type load(const template_arg * p) { return load_body (p); } \
        static void store(template_arg * p, reg_type v) { store_body (p, v); } \
    }

#define FUNCTOR_CLOSURE_2arg(name, template_arg, body)\
    template<>                                                                 \
    struct name<template_arg>                                                  \
    {                                                                          \
        VLoadStore128<template_arg>::reg_type operator()(                      \
                        const VLoadStore128<template_arg>::reg_type & a,       \
                        const VLoadStore128<template_arg>::reg_type & b) const \
        {                                                                      \
            body;                                                              \
        }                                                                      \
    }

#define FUNCTOR_CLOSURE_1arg(name, template_arg, body)\
    template<>                                                                 \
    struct name<template_arg>                                                  \
    {                                                                          \
        VLoadStore128<template_arg>::reg_type operator()(                      \
                        const VLoadStore128<template_arg>::reg_type & a,       \
                        const VLoadStore128<template_arg>::reg_type &  ) const \
        {                                                                      \
            body;                                                              \
        }                                                                      \
    }

FUNCTOR_LOADSTORE_CAST(VLoadStore128,  uchar, __m128i, _mm_loadu_si128, _mm_storeu_si128);
FUNCTOR_LOADSTORE_CAST(VLoadStore128,  schar, __m128i, _mm_loadu_si128, _mm_storeu_si128);
FUNCTOR_LOADSTORE_CAST(VLoadStore128, ushort, __m128i, _mm_loadu_si128, _mm_storeu_si128);
FUNCTOR_LOADSTORE_CAST(VLoadStore128,  short, __m128i, _mm_loadu_si128, _mm_storeu_si128);
FUNCTOR_LOADSTORE_CAST(VLoadStore128,    int, __m128i, _mm_loadu_si128, _mm_storeu_si128);
FUNCTOR_LOADSTORE(     VLoadStore128,  float, __m128 , _mm_loadu_ps   , _mm_storeu_ps   );
FUNCTOR_LOADSTORE(     VLoadStore128, double, __m128d, _mm_loadu_pd   , _mm_storeu_pd   );

FUNCTOR_LOADSTORE_CAST(VLoadStore64,  uchar, __m128i, _mm_loadl_epi64, _mm_storel_epi64);
FUNCTOR_LOADSTORE_CAST(VLoadStore64,  schar, __m128i, _mm_loadl_epi64, _mm_storel_epi64);
FUNCTOR_LOADSTORE_CAST(VLoadStore64, ushort, __m128i, _mm_loadl_epi64, _mm_storel_epi64);
FUNCTOR_LOADSTORE_CAST(VLoadStore64,  short, __m128i, _mm_loadl_epi64, _mm_storel_epi64);

FUNCTOR_LOADSTORE_CAST(VLoadStore128Aligned,    int, __m128i, _mm_load_si128, _mm_store_si128);
FUNCTOR_LOADSTORE(     VLoadStore128Aligned,  float, __m128 , _mm_load_ps   , _mm_store_ps   );
FUNCTOR_LOADSTORE(     VLoadStore128Aligned, double, __m128d, _mm_load_pd   , _mm_store_pd   );

FUNCTOR_TEMPLATE(VAdd);
FUNCTOR_CLOSURE_2arg(VAdd,  uchar, return _mm_adds_epu8 (a, b));
FUNCTOR_CLOSURE_2arg(VAdd,  schar, return _mm_adds_epi8 (a, b));
FUNCTOR_CLOSURE_2arg(VAdd, ushort, return _mm_adds_epu16(a, b));
FUNCTOR_CLOSURE_2arg(VAdd,  short, return _mm_adds_epi16(a, b));
FUNCTOR_CLOSURE_2arg(VAdd,    int, return _mm_add_epi32 (a, b));
FUNCTOR_CLOSURE_2arg(VAdd,  float, return _mm_add_ps    (a, b));
FUNCTOR_CLOSURE_2arg(VAdd, double, return _mm_add_pd    (a, b));

FUNCTOR_TEMPLATE(VSub);
FUNCTOR_CLOSURE_2arg(VSub,  uchar, return _mm_subs_epu8 (a, b));
FUNCTOR_CLOSURE_2arg(VSub,  schar, return _mm_subs_epi8 (a, b));
FUNCTOR_CLOSURE_2arg(VSub, ushort, return _mm_subs_epu16(a, b));
FUNCTOR_CLOSURE_2arg(VSub,  short, return _mm_subs_epi16(a, b));
FUNCTOR_CLOSURE_2arg(VSub,    int, return _mm_sub_epi32 (a, b));
FUNCTOR_CLOSURE_2arg(VSub,  float, return _mm_sub_ps    (a, b));
FUNCTOR_CLOSURE_2arg(VSub, double, return _mm_sub_pd    (a, b));

FUNCTOR_TEMPLATE(VMin);
FUNCTOR_CLOSURE_2arg(VMin, uchar, return _mm_min_epu8(a, b));
FUNCTOR_CLOSURE_2arg(VMin, schar,
        __m128i m = _mm_cmpgt_epi8(a, b);
        return _mm_xor_si128(a, _mm_and_si128(_mm_xor_si128(a, b), m));
    );
FUNCTOR_CLOSURE_2arg(VMin, ushort, return _mm_subs_epu16(a, _mm_subs_epu16(a, b)));
FUNCTOR_CLOSURE_2arg(VMin,  short, return _mm_min_epi16(a, b));
FUNCTOR_CLOSURE_2arg(VMin,    int,
        __m128i m = _mm_cmpgt_epi32(a, b);
        return _mm_xor_si128(a, _mm_and_si128(_mm_xor_si128(a, b), m));
    );
FUNCTOR_CLOSURE_2arg(VMin,  float, return _mm_min_ps(a, b));
FUNCTOR_CLOSURE_2arg(VMin, double, return _mm_min_pd(a, b));

FUNCTOR_TEMPLATE(VMax);
FUNCTOR_CLOSURE_2arg(VMax, uchar, return _mm_max_epu8(a, b));
FUNCTOR_CLOSURE_2arg(VMax, schar,
        __m128i m = _mm_cmpgt_epi8(b, a);
        return _mm_xor_si128(a, _mm_and_si128(_mm_xor_si128(a, b), m));
    );
FUNCTOR_CLOSURE_2arg(VMax, ushort, return _mm_adds_epu16(_mm_subs_epu16(a, b), b));
FUNCTOR_CLOSURE_2arg(VMax,  short, return _mm_max_epi16(a, b));
FUNCTOR_CLOSURE_2arg(VMax,    int,
        __m128i m = _mm_cmpgt_epi32(b, a);
        return _mm_xor_si128(a, _mm_and_si128(_mm_xor_si128(a, b), m));
    );
FUNCTOR_CLOSURE_2arg(VMax,  float, return _mm_max_ps(a, b));
FUNCTOR_CLOSURE_2arg(VMax, double, return _mm_max_pd(a, b));


static unsigned int CV_DECL_ALIGNED(16) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
static unsigned int CV_DECL_ALIGNED(16) v64f_absmask[] = { 0xffffffff, 0x7fffffff, 0xffffffff, 0x7fffffff };

FUNCTOR_TEMPLATE(VAbsDiff);
FUNCTOR_CLOSURE_2arg(VAbsDiff,  uchar,
        return _mm_add_epi8(_mm_subs_epu8(a, b), _mm_subs_epu8(b, a));
    );
FUNCTOR_CLOSURE_2arg(VAbsDiff,  schar,
        __m128i d = _mm_subs_epi8(a, b);
        __m128i m = _mm_cmpgt_epi8(b, a);
        return _mm_subs_epi8(_mm_xor_si128(d, m), m);
    );
FUNCTOR_CLOSURE_2arg(VAbsDiff, ushort,
        return _mm_add_epi16(_mm_subs_epu16(a, b), _mm_subs_epu16(b, a));
    );
FUNCTOR_CLOSURE_2arg(VAbsDiff,  short,
        __m128i M = _mm_max_epi16(a, b);
        __m128i m = _mm_min_epi16(a, b);
        return _mm_subs_epi16(M, m);
    );
FUNCTOR_CLOSURE_2arg(VAbsDiff,    int,
        __m128i d = _mm_sub_epi32(a, b);
        __m128i m = _mm_cmpgt_epi32(b, a);
        return _mm_sub_epi32(_mm_xor_si128(d, m), m);
    );
FUNCTOR_CLOSURE_2arg(VAbsDiff,  float,
        return _mm_and_ps(_mm_sub_ps(a,b), *(const __m128*)v32f_absmask);
    );
FUNCTOR_CLOSURE_2arg(VAbsDiff, double,
        return _mm_and_pd(_mm_sub_pd(a,b), *(const __m128d*)v64f_absmask);
    );

FUNCTOR_TEMPLATE(VAnd);
FUNCTOR_CLOSURE_2arg(VAnd, uchar, return _mm_and_si128(a, b));
FUNCTOR_TEMPLATE(VOr);
FUNCTOR_CLOSURE_2arg(VOr , uchar, return _mm_or_si128 (a, b));
FUNCTOR_TEMPLATE(VXor);
FUNCTOR_CLOSURE_2arg(VXor, uchar, return _mm_xor_si128(a, b));
FUNCTOR_TEMPLATE(VNot);
FUNCTOR_CLOSURE_1arg(VNot, uchar, return _mm_xor_si128(_mm_set1_epi32(-1), a));
#endif

#if CV_SSE2
#define IF_SIMD(op) op
#else
#define IF_SIMD(op) NOP
#endif

template<> inline uchar OpAdd<uchar>::operator ()(uchar a, uchar b) const
{ return CV_FAST_CAST_8U(a + b); }
template<> inline uchar OpSub<uchar>::operator ()(uchar a, uchar b) const
{ return CV_FAST_CAST_8U(a - b); }

template<typename T> struct OpAbsDiff
{
    typedef T type1;
    typedef T type2;
    typedef T rtype;
    T operator()(T a, T b) const { return (T)std::abs(a - b); }
};

template<> inline short OpAbsDiff<short>::operator ()(short a, short b) const
{ return saturate_cast<short>(std::abs(a - b)); }

template<> inline schar OpAbsDiff<schar>::operator ()(schar a, schar b) const
{ return saturate_cast<schar>(std::abs(a - b)); }

template<typename T, typename WT=T> struct OpAbsDiffS
{
    typedef T type1;
    typedef WT type2;
    typedef T rtype;
    T operator()(T a, WT b) const { return saturate_cast<T>(std::abs(a - b)); }
};

template<typename T> struct OpAnd
{
    typedef T type1;
    typedef T type2;
    typedef T rtype;
    T operator()( T a, T b ) const { return a & b; }
};

template<typename T> struct OpOr
{
    typedef T type1;
    typedef T type2;
    typedef T rtype;
    T operator()( T a, T b ) const { return a | b; }
};

template<typename T> struct OpXor
{
    typedef T type1;
    typedef T type2;
    typedef T rtype;
    T operator()( T a, T b ) const { return a ^ b; }
};

template<typename T> struct OpNot
{
    typedef T type1;
    typedef T type2;
    typedef T rtype;
    T operator()( T a, T ) const { return ~a; }
};

#if (ARITHM_USE_IPP == 1)
static inline void fixSteps(Size sz, size_t elemSize, size_t& step1, size_t& step2, size_t& step)
{
    if( sz.height == 1 )
        step1 = step2 = step = sz.width*elemSize;
}
#endif

static void add8u( const uchar* src1, size_t step1,
                   const uchar* src2, size_t step2,
                   uchar* dst, size_t step, Size sz, void* )
{
#if (ARITHM_USE_IPP == 1)
    fixSteps(sz, sizeof(dst[0]), step1, step2, step);
    if (0 <= ippiAdd_8u_C1RSfs(src1, (int)step1, src2, (int)step2, dst, (int)step, ippiSize(sz), 0))
        return;
    setIppErrorStatus();
#endif
    (vBinOp<uchar, OpAdd<uchar>, IF_SIMD(VAdd<uchar>)>(src1, step1, src2, step2, dst, step, sz));
}

static void add8s( const schar* src1, size_t step1,
                   const schar* src2, size_t step2,
                   schar* dst, size_t step, Size sz, void* )
{
    vBinOp<schar, OpAdd<schar>, IF_SIMD(VAdd<schar>)>(src1, step1, src2, step2, dst, step, sz);
}

static void add16u( const ushort* src1, size_t step1,
                    const ushort* src2, size_t step2,
                    ushort* dst, size_t step, Size sz, void* )
{
#if (ARITHM_USE_IPP == 1)
    fixSteps(sz, sizeof(dst[0]), step1, step2, step);
    if (0 <= ippiAdd_16u_C1RSfs(src1, (int)step1, src2, (int)step2, dst, (int)step, ippiSize(sz), 0))
        return;
    setIppErrorStatus();
#endif
    (vBinOp<ushort, OpAdd<ushort>, IF_SIMD(VAdd<ushort>)>(src1, step1, src2, step2, dst, step, sz));
}

static void add16s( const short* src1, size_t step1,
                    const short* src2, size_t step2,
                    short* dst, size_t step, Size sz, void* )
{
#if (ARITHM_USE_IPP == 1)
    fixSteps(sz, sizeof(dst[0]), step1, step2, step);
    if (0 <= ippiAdd_16s_C1RSfs(src1, (int)step1, src2, (int)step2, dst, (int)step, ippiSize(sz), 0))
        return;
    setIppErrorStatus();
#endif
    (vBinOp<short, OpAdd<short>, IF_SIMD(VAdd<short>)>(src1, step1, src2, step2, dst, step, sz));
}

static void add32s( const int* src1, size_t step1,
                    const int* src2, size_t step2,
                    int* dst, size_t step, Size sz, void* )
{
    vBinOp32<int, OpAdd<int>, IF_SIMD(VAdd<int>)>(src1, step1, src2, step2, dst, step, sz);
}

static void add32f( const float* src1, size_t step1,
                    const float* src2, size_t step2,
                    float* dst, size_t step, Size sz, void* )
{
#if (ARITHM_USE_IPP == 1)
    fixSteps(sz, sizeof(dst[0]), step1, step2, step);
    if (0 <= ippiAdd_32f_C1R(src1, (int)step1, src2, (int)step2, dst, (int)step, ippiSize(sz)))
        return;
    setIppErrorStatus();
#endif
    (vBinOp32<float, OpAdd<float>, IF_SIMD(VAdd<float>)>(src1, step1, src2, step2, dst, step, sz));
}

static void add64f( const double* src1, size_t step1,
                    const double* src2, size_t step2,
                    double* dst, size_t step, Size sz, void* )
{
    vBinOp64<double, OpAdd<double>, IF_SIMD(VAdd<double>)>(src1, step1, src2, step2, dst, step, sz);
}

static void sub8u( const uchar* src1, size_t step1,
                   const uchar* src2, size_t step2,
                   uchar* dst, size_t step, Size sz, void* )
{
#if (ARITHM_USE_IPP == 1)
    fixSteps(sz, sizeof(dst[0]), step1, step2, step);
    if (0 <= ippiSub_8u_C1RSfs(src2, (int)step2, src1, (int)step1, dst, (int)step, ippiSize(sz), 0))
        return;
    setIppErrorStatus();
#endif
    (vBinOp<uchar, OpSub<uchar>, IF_SIMD(VSub<uchar>)>(src1, step1, src2, step2, dst, step, sz));
}

static void sub8s( const schar* src1, size_t step1,
                   const schar* src2, size_t step2,
                   schar* dst, size_t step, Size sz, void* )
{
    vBinOp<schar, OpSub<schar>, IF_SIMD(VSub<schar>)>(src1, step1, src2, step2, dst, step, sz);
}

static void sub16u( const ushort* src1, size_t step1,
                    const ushort* src2, size_t step2,
                    ushort* dst, size_t step, Size sz, void* )
{
#if (ARITHM_USE_IPP == 1)
    fixSteps(sz, sizeof(dst[0]), step1, step2, step);
    if (0 <= ippiSub_16u_C1RSfs(src2, (int)step2, src1, (int)step1, dst, (int)step, ippiSize(sz), 0))
        return;
    setIppErrorStatus();
#endif
    (vBinOp<ushort, OpSub<ushort>, IF_SIMD(VSub<ushort>)>(src1, step1, src2, step2, dst, step, sz));
}

static void sub16s( const short* src1, size_t step1,
                    const short* src2, size_t step2,
                    short* dst, size_t step, Size sz, void* )
{
#if (ARITHM_USE_IPP == 1)
    fixSteps(sz, sizeof(dst[0]), step1, step2, step);
    if (0 <= ippiSub_16s_C1RSfs(src2, (int)step2, src1, (int)step1, dst, (int)step, ippiSize(sz), 0))
        return;
    setIppErrorStatus();
#endif
    (vBinOp<short, OpSub<short>, IF_SIMD(VSub<short>)>(src1, step1, src2, step2, dst, step, sz));
}

static void sub32s( const int* src1, size_t step1,
                    const int* src2, size_t step2,
                    int* dst, size_t step, Size sz, void* )
{
    vBinOp32<int, OpSub<int>, IF_SIMD(VSub<int>)>(src1, step1, src2, step2, dst, step, sz);
}

static void sub32f( const float* src1, size_t step1,
                   const float* src2, size_t step2,
                   float* dst, size_t step, Size sz, void* )
{
#if (ARITHM_USE_IPP == 1)
    fixSteps(sz, sizeof(dst[0]), step1, step2, step);
    if (0 <= ippiSub_32f_C1R(src2, (int)step2, src1, (int)step1, dst, (int)step, ippiSize(sz)))
        return;
    setIppErrorStatus();
#endif
    (vBinOp32<float, OpSub<float>, IF_SIMD(VSub<float>)>(src1, step1, src2, step2, dst, step, sz));
}

static void sub64f( const double* src1, size_t step1,
                    const double* src2, size_t step2,
                    double* dst, size_t step, Size sz, void* )
{
    vBinOp64<double, OpSub<double>, IF_SIMD(VSub<double>)>(src1, step1, src2, step2, dst, step, sz);
}

template<> inline uchar OpMin<uchar>::operator ()(uchar a, uchar b) const { return CV_MIN_8U(a, b); }
template<> inline uchar OpMax<uchar>::operator ()(uchar a, uchar b) const { return CV_MAX_8U(a, b); }

static void max8u( const uchar* src1, size_t step1,
                   const uchar* src2, size_t step2,
                   uchar* dst, size_t step, Size sz, void* )
{
#if (ARITHM_USE_IPP == 1)
    uchar* s1 = (uchar*)src1;
    uchar* s2 = (uchar*)src2;
    uchar* d  = dst;
    fixSteps(sz, sizeof(dst[0]), step1, step2, step);
    int i = 0;
    for(; i < sz.height; i++)
    {
        if (0 > ippsMaxEvery_8u(s1, s2, d, sz.width))
            break;
        s1 += step1;
        s2 += step2;
        d  += step;
    }
    if (i == sz.height)
        return;
    setIppErrorStatus();
#endif
    vBinOp<uchar, OpMax<uchar>, IF_SIMD(VMax<uchar>)>(src1, step1, src2, step2, dst, step, sz);
}

static void max8s( const schar* src1, size_t step1,
                   const schar* src2, size_t step2,
                   schar* dst, size_t step, Size sz, void* )
{
    vBinOp<schar, OpMax<schar>, IF_SIMD(VMax<schar>)>(src1, step1, src2, step2, dst, step, sz);
}

static void max16u( const ushort* src1, size_t step1,
                    const ushort* src2, size_t step2,
                    ushort* dst, size_t step, Size sz, void* )
{
#if (ARITHM_USE_IPP == 1)
    ushort* s1 = (ushort*)src1;
    ushort* s2 = (ushort*)src2;
    ushort* d  = dst;
    fixSteps(sz, sizeof(dst[0]), step1, step2, step);
    int i = 0;
    for(; i < sz.height; i++)
    {
        if (0 > ippsMaxEvery_16u(s1, s2, d, sz.width))
            break;
        s1 = (ushort*)((uchar*)s1 + step1);
        s2 = (ushort*)((uchar*)s2 + step2);
        d  = (ushort*)((uchar*)d + step);
    }
    if (i == sz.height)
        return;
    setIppErrorStatus();
#endif
    vBinOp<ushort, OpMax<ushort>, IF_SIMD(VMax<ushort>)>(src1, step1, src2, step2, dst, step, sz);
}

static void max16s( const short* src1, size_t step1,
                    const short* src2, size_t step2,
                    short* dst, size_t step, Size sz, void* )
{
    vBinOp<short, OpMax<short>, IF_SIMD(VMax<short>)>(src1, step1, src2, step2, dst, step, sz);
}

static void max32s( const int* src1, size_t step1,
                    const int* src2, size_t step2,
                    int* dst, size_t step, Size sz, void* )
{
    vBinOp32<int, OpMax<int>, IF_SIMD(VMax<int>)>(src1, step1, src2, step2, dst, step, sz);
}

static void max32f( const float* src1, size_t step1,
                    const float* src2, size_t step2,
                    float* dst, size_t step, Size sz, void* )
{
#if (ARITHM_USE_IPP == 1)
    float* s1 = (float*)src1;
    float* s2 = (float*)src2;
    float* d  = dst;
    fixSteps(sz, sizeof(dst[0]), step1, step2, step);
    int i = 0;
    for(; i < sz.height; i++)
    {
        if (0 > ippsMaxEvery_32f(s1, s2, d, sz.width))
            break;
        s1 = (float*)((uchar*)s1 + step1);
        s2 = (float*)((uchar*)s2 + step2);
        d  = (float*)((uchar*)d + step);
    }
    if (i == sz.height)
        return;
    setIppErrorStatus();
#endif
    vBinOp32<float, OpMax<float>, IF_SIMD(VMax<float>)>(src1, step1, src2, step2, dst, step, sz);
}

static void max64f( const double* src1, size_t step1,
                    const double* src2, size_t step2,
                    double* dst, size_t step, Size sz, void* )
{
#if ARITHM_USE_IPP == 1
    double* s1 = (double*)src1;
    double* s2 = (double*)src2;
    double* d  = dst;
    fixSteps(sz, sizeof(dst[0]), step1, step2, step);
    int i = 0;
    for(; i < sz.height; i++)
    {
        if (0 > ippsMaxEvery_64f(s1, s2, d, sz.width))
            break;
        s1 = (double*)((uchar*)s1 + step1);
        s2 = (double*)((uchar*)s2 + step2);
        d  = (double*)((uchar*)d + step);
    }
    if (i == sz.height)
        return;
    setIppErrorStatus();
#endif
    vBinOp64<double, OpMax<double>, IF_SIMD(VMax<double>)>(src1, step1, src2, step2, dst, step, sz);
}

static void min8u( const uchar* src1, size_t step1,
                   const uchar* src2, size_t step2,
                   uchar* dst, size_t step, Size sz, void* )
{
#if (ARITHM_USE_IPP == 1)
    uchar* s1 = (uchar*)src1;
    uchar* s2 = (uchar*)src2;
    uchar* d  = dst;
    fixSteps(sz, sizeof(dst[0]), step1, step2, step);
    int i = 0;
    for(; i < sz.height; i++)
    {
        if (0 > ippsMinEvery_8u(s1, s2, d, sz.width))
            break;
        s1 += step1;
        s2 += step2;
        d  += step;
    }
    if (i == sz.height)
        return;
    setIppErrorStatus();
#endif
    vBinOp<uchar, OpMin<uchar>, IF_SIMD(VMin<uchar>)>(src1, step1, src2, step2, dst, step, sz);
}

static void min8s( const schar* src1, size_t step1,
                   const schar* src2, size_t step2,
                   schar* dst, size_t step, Size sz, void* )
{
    vBinOp<schar, OpMin<schar>, IF_SIMD(VMin<schar>)>(src1, step1, src2, step2, dst, step, sz);
}

static void min16u( const ushort* src1, size_t step1,
                    const ushort* src2, size_t step2,
                    ushort* dst, size_t step, Size sz, void* )
{
#if (ARITHM_USE_IPP == 1)
    ushort* s1 = (ushort*)src1;
    ushort* s2 = (ushort*)src2;
    ushort* d  = dst;
    fixSteps(sz, sizeof(dst[0]), step1, step2, step);
    int i = 0;
    for(; i < sz.height; i++)
    {
        if (0 > ippsMinEvery_16u(s1, s2, d, sz.width))
            break;
        s1 = (ushort*)((uchar*)s1 + step1);
        s2 = (ushort*)((uchar*)s2 + step2);
        d  = (ushort*)((uchar*)d + step);
    }
    if (i == sz.height)
        return;
    setIppErrorStatus();
#endif
    vBinOp<ushort, OpMin<ushort>, IF_SIMD(VMin<ushort>)>(src1, step1, src2, step2, dst, step, sz);
}

static void min16s( const short* src1, size_t step1,
                    const short* src2, size_t step2,
                    short* dst, size_t step, Size sz, void* )
{
    vBinOp<short, OpMin<short>, IF_SIMD(VMin<short>)>(src1, step1, src2, step2, dst, step, sz);
}

static void min32s( const int* src1, size_t step1,
                    const int* src2, size_t step2,
                    int* dst, size_t step, Size sz, void* )
{
    vBinOp32<int, OpMin<int>, IF_SIMD(VMin<int>)>(src1, step1, src2, step2, dst, step, sz);
}

static void min32f( const float* src1, size_t step1,
                    const float* src2, size_t step2,
                    float* dst, size_t step, Size sz, void* )
{
#if (ARITHM_USE_IPP == 1)
    float* s1 = (float*)src1;
    float* s2 = (float*)src2;
    float* d  = dst;
    fixSteps(sz, sizeof(dst[0]), step1, step2, step);
    int i = 0;
    for(; i < sz.height; i++)
    {
        if (0 > ippsMinEvery_32f(s1, s2, d, sz.width))
            break;
        s1 = (float*)((uchar*)s1 + step1);
        s2 = (float*)((uchar*)s2 + step2);
        d  = (float*)((uchar*)d + step);
    }
    if (i == sz.height)
        return;
    setIppErrorStatus();
#endif
    vBinOp32<float, OpMin<float>, IF_SIMD(VMin<float>)>(src1, step1, src2, step2, dst, step, sz);
}

static void min64f( const double* src1, size_t step1,
                    const double* src2, size_t step2,
                    double* dst, size_t step, Size sz, void* )
{
#if ARITHM_USE_IPP == 1
    double* s1 = (double*)src1;
    double* s2 = (double*)src2;
    double* d  = dst;
    fixSteps(sz, sizeof(dst[0]), step1, step2, step);
    int i = 0;
    for(; i < sz.height; i++)
    {
        if (0 > ippsMinEvery_64f(s1, s2, d, sz.width))
            break;
        s1 = (double*)((uchar*)s1 + step1);
        s2 = (double*)((uchar*)s2 + step2);
        d  = (double*)((uchar*)d + step);
    }
    if (i == sz.height)
        return;
    setIppErrorStatus();
#endif
    vBinOp64<double, OpMin<double>, IF_SIMD(VMin<double>)>(src1, step1, src2, step2, dst, step, sz);
}

static void absdiff8u( const uchar* src1, size_t step1,
                       const uchar* src2, size_t step2,
                       uchar* dst, size_t step, Size sz, void* )
{
#if (ARITHM_USE_IPP == 1)
    fixSteps(sz, sizeof(dst[0]), step1, step2, step);
    if (0 <= ippiAbsDiff_8u_C1R(src1, (int)step1, src2, (int)step2, dst, (int)step, ippiSize(sz)))
        return;
    setIppErrorStatus();
#endif
    (vBinOp<uchar, OpAbsDiff<uchar>, IF_SIMD(VAbsDiff<uchar>)>(src1, step1, src2, step2, dst, step, sz));
}

static void absdiff8s( const schar* src1, size_t step1,
                       const schar* src2, size_t step2,
                       schar* dst, size_t step, Size sz, void* )
{
    vBinOp<schar, OpAbsDiff<schar>, IF_SIMD(VAbsDiff<schar>)>(src1, step1, src2, step2, dst, step, sz);
}

static void absdiff16u( const ushort* src1, size_t step1,
                        const ushort* src2, size_t step2,
                        ushort* dst, size_t step, Size sz, void* )
{
#if (ARITHM_USE_IPP == 1)
    fixSteps(sz, sizeof(dst[0]), step1, step2, step);
    if (0 <= ippiAbsDiff_16u_C1R(src1, (int)step1, src2, (int)step2, dst, (int)step, ippiSize(sz)))
        return;
    setIppErrorStatus();
#endif
    (vBinOp<ushort, OpAbsDiff<ushort>, IF_SIMD(VAbsDiff<ushort>)>(src1, step1, src2, step2, dst, step, sz));
}

static void absdiff16s( const short* src1, size_t step1,
                        const short* src2, size_t step2,
                        short* dst, size_t step, Size sz, void* )
{
    vBinOp<short, OpAbsDiff<short>, IF_SIMD(VAbsDiff<short>)>(src1, step1, src2, step2, dst, step, sz);
}

static void absdiff32s( const int* src1, size_t step1,
                        const int* src2, size_t step2,
                        int* dst, size_t step, Size sz, void* )
{
    vBinOp32<int, OpAbsDiff<int>, IF_SIMD(VAbsDiff<int>)>(src1, step1, src2, step2, dst, step, sz);
}

static void absdiff32f( const float* src1, size_t step1,
                        const float* src2, size_t step2,
                        float* dst, size_t step, Size sz, void* )
{
#if (ARITHM_USE_IPP == 1)
    fixSteps(sz, sizeof(dst[0]), step1, step2, step);
    if (0 <= ippiAbsDiff_32f_C1R(src1, (int)step1, src2, (int)step2, dst, (int)step, ippiSize(sz)))
        return;
    setIppErrorStatus();
#endif
    (vBinOp32<float, OpAbsDiff<float>, IF_SIMD(VAbsDiff<float>)>(src1, step1, src2, step2, dst, step, sz));
}

static void absdiff64f( const double* src1, size_t step1,
                        const double* src2, size_t step2,
                        double* dst, size_t step, Size sz, void* )
{
    vBinOp64<double, OpAbsDiff<double>, IF_SIMD(VAbsDiff<double>)>(src1, step1, src2, step2, dst, step, sz);
}


static void and8u( const uchar* src1, size_t step1,
                   const uchar* src2, size_t step2,
                   uchar* dst, size_t step, Size sz, void* )
{
#if (ARITHM_USE_IPP == 1)
    fixSteps(sz, sizeof(dst[0]), step1, step2, step);
    if (0 <= ippiAnd_8u_C1R(src1, (int)step1, src2, (int)step2, dst, (int)step, ippiSize(sz)))
        return;
    setIppErrorStatus();
#endif
    (vBinOp<uchar, OpAnd<uchar>, IF_SIMD(VAnd<uchar>)>(src1, step1, src2, step2, dst, step, sz));
}

static void or8u( const uchar* src1, size_t step1,
                  const uchar* src2, size_t step2,
                  uchar* dst, size_t step, Size sz, void* )
{
#if (ARITHM_USE_IPP == 1)
    fixSteps(sz, sizeof(dst[0]), step1, step2, step);
    if (0 <= ippiOr_8u_C1R(src1, (int)step1, src2, (int)step2, dst, (int)step, ippiSize(sz)))
        return;
    setIppErrorStatus();
#endif
    (vBinOp<uchar, OpOr<uchar>, IF_SIMD(VOr<uchar>)>(src1, step1, src2, step2, dst, step, sz));
}

static void xor8u( const uchar* src1, size_t step1,
                   const uchar* src2, size_t step2,
                   uchar* dst, size_t step, Size sz, void* )
{
#if (ARITHM_USE_IPP == 1)
    fixSteps(sz, sizeof(dst[0]), step1, step2, step);
    if (0 <= ippiXor_8u_C1R(src1, (int)step1, src2, (int)step2, dst, (int)step, ippiSize(sz)))
        return;
    setIppErrorStatus();
#endif
    (vBinOp<uchar, OpXor<uchar>, IF_SIMD(VXor<uchar>)>(src1, step1, src2, step2, dst, step, sz));
}

static void not8u( const uchar* src1, size_t step1,
                   const uchar* src2, size_t step2,
                   uchar* dst, size_t step, Size sz, void* )
{
#if (ARITHM_USE_IPP == 1)
    fixSteps(sz, sizeof(dst[0]), step1, step2, step); (void)src2;
    if (0 <= ippiNot_8u_C1R(src1, (int)step1, dst, (int)step, ippiSize(sz)))
        return;
    setIppErrorStatus();
#endif
    (vBinOp<uchar, OpNot<uchar>, IF_SIMD(VNot<uchar>)>(src1, step1, src2, step2, dst, step, sz));
}

/****************************************************************************************\
*                                   logical operations                                   *
\****************************************************************************************/

void convertAndUnrollScalar( const Mat& sc, int buftype, uchar* scbuf, size_t blocksize )
{
    int scn = (int)sc.total(), cn = CV_MAT_CN(buftype);
    size_t esz = CV_ELEM_SIZE(buftype);
    getConvertFunc(sc.depth(), buftype)(sc.data, 1, 0, 1, scbuf, 1, Size(std::min(cn, scn), 1), 0);
    // unroll the scalar
    if( scn < cn )
    {
        CV_Assert( scn == 1 );
        size_t esz1 = CV_ELEM_SIZE1(buftype);
        for( size_t i = esz1; i < esz; i++ )
            scbuf[i] = scbuf[i - esz1];
    }
    for( size_t i = esz; i < blocksize*esz; i++ )
        scbuf[i] = scbuf[i - esz];
}


enum { OCL_OP_ADD=0, OCL_OP_SUB=1, OCL_OP_RSUB=2, OCL_OP_ABSDIFF=3, OCL_OP_MUL=4,
       OCL_OP_MUL_SCALE=5, OCL_OP_DIV_SCALE=6, OCL_OP_RECIP_SCALE=7, OCL_OP_ADDW=8,
       OCL_OP_AND=9, OCL_OP_OR=10, OCL_OP_XOR=11, OCL_OP_NOT=12, OCL_OP_MIN=13, OCL_OP_MAX=14,
       OCL_OP_RDIV_SCALE=15 };

#ifdef HAVE_OPENCL

static const char* oclop2str[] = { "OP_ADD", "OP_SUB", "OP_RSUB", "OP_ABSDIFF",
    "OP_MUL", "OP_MUL_SCALE", "OP_DIV_SCALE", "OP_RECIP_SCALE",
    "OP_ADDW", "OP_AND", "OP_OR", "OP_XOR", "OP_NOT", "OP_MIN", "OP_MAX", "OP_RDIV_SCALE", 0 };

static bool ocl_binary_op(InputArray _src1, InputArray _src2, OutputArray _dst,
                          InputArray _mask, bool bitwise, int oclop, bool haveScalar )
{
    bool haveMask = !_mask.empty();
    int srctype = _src1.type();
    int srcdepth = CV_MAT_DEPTH(srctype);
    int cn = CV_MAT_CN(srctype);

    const ocl::Device d = ocl::Device::getDefault();
    bool doubleSupport = d.doubleFPConfig() > 0;
    if( oclop < 0 || ((haveMask || haveScalar) && cn > 4) ||
            (!doubleSupport && srcdepth == CV_64F && !bitwise))
        return false;

    char opts[1024];
    int kercn = haveMask || haveScalar ? cn : ocl::predictOptimalVectorWidth(_src1, _src2, _dst);
    int scalarcn = kercn == 3 ? 4 : kercn;
    int rowsPerWI = d.isIntel() ? 4 : 1;

    sprintf(opts, "-D %s%s -D %s -D dstT=%s%s -D dstT_C1=%s -D workST=%s -D cn=%d -D rowsPerWI=%d",
            haveMask ? "MASK_" : "", haveScalar ? "UNARY_OP" : "BINARY_OP", oclop2str[oclop],
            bitwise ? ocl::memopTypeToStr(CV_MAKETYPE(srcdepth, kercn)) :
                ocl::typeToStr(CV_MAKETYPE(srcdepth, kercn)), doubleSupport ? " -D DOUBLE_SUPPORT" : "",
            bitwise ? ocl::memopTypeToStr(CV_MAKETYPE(srcdepth, 1)) :
                ocl::typeToStr(CV_MAKETYPE(srcdepth, 1)),
            bitwise ? ocl::memopTypeToStr(CV_MAKETYPE(srcdepth, scalarcn)) :
                ocl::typeToStr(CV_MAKETYPE(srcdepth, scalarcn)),
            kercn, rowsPerWI);

    ocl::Kernel k("KF", ocl::core::arithm_oclsrc, opts);
    if (k.empty())
        return false;

    UMat src1 = _src1.getUMat(), src2;
    UMat dst = _dst.getUMat(), mask = _mask.getUMat();

    ocl::KernelArg src1arg = ocl::KernelArg::ReadOnlyNoSize(src1, cn, kercn);
    ocl::KernelArg dstarg = haveMask ? ocl::KernelArg::ReadWrite(dst, cn, kercn) :
                                       ocl::KernelArg::WriteOnly(dst, cn, kercn);
    ocl::KernelArg maskarg = ocl::KernelArg::ReadOnlyNoSize(mask, 1);

    if( haveScalar )
    {
        size_t esz = CV_ELEM_SIZE1(srctype)*scalarcn;
        double buf[4] = {0,0,0,0};

        if( oclop != OCL_OP_NOT )
        {
            Mat src2sc = _src2.getMat();
            convertAndUnrollScalar(src2sc, srctype, (uchar*)buf, 1);
        }

        ocl::KernelArg scalararg = ocl::KernelArg(0, 0, 0, 0, buf, esz);

        if( !haveMask )
            k.args(src1arg, dstarg, scalararg);
        else
            k.args(src1arg, maskarg, dstarg, scalararg);
    }
    else
    {
        src2 = _src2.getUMat();
        ocl::KernelArg src2arg = ocl::KernelArg::ReadOnlyNoSize(src2, cn, kercn);

        if( !haveMask )
            k.args(src1arg, src2arg, dstarg);
        else
            k.args(src1arg, src2arg, maskarg, dstarg);
    }

    size_t globalsize[] = { src1.cols * cn / kercn, (src1.rows + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, 0, false);
}

#endif

static void binary_op( InputArray _src1, InputArray _src2, OutputArray _dst,
                       InputArray _mask, const BinaryFunc* tab,
                       bool bitwise, int oclop )
{
    const _InputArray *psrc1 = &_src1, *psrc2 = &_src2;
    int kind1 = psrc1->kind(), kind2 = psrc2->kind();
    int type1 = psrc1->type(), depth1 = CV_MAT_DEPTH(type1), cn = CV_MAT_CN(type1);
    int type2 = psrc2->type(), depth2 = CV_MAT_DEPTH(type2), cn2 = CV_MAT_CN(type2);
    int dims1 = psrc1->dims(), dims2 = psrc2->dims();
    Size sz1 = dims1 <= 2 ? psrc1->size() : Size();
    Size sz2 = dims2 <= 2 ? psrc2->size() : Size();
#ifdef HAVE_OPENCL
    bool use_opencl = (kind1 == _InputArray::UMAT || kind2 == _InputArray::UMAT) &&
            dims1 <= 2 && dims2 <= 2;
#endif
    bool haveMask = !_mask.empty(), haveScalar = false;
    BinaryFunc func;

    if( dims1 <= 2 && dims2 <= 2 && kind1 == kind2 && sz1 == sz2 && type1 == type2 && !haveMask )
    {
        _dst.create(sz1, type1);
        CV_OCL_RUN(use_opencl,
                   ocl_binary_op(*psrc1, *psrc2, _dst, _mask, bitwise, oclop, false))

        if( bitwise )
        {
            func = *tab;
            cn = (int)CV_ELEM_SIZE(type1);
        }
        else
            func = tab[depth1];

        Mat src1 = psrc1->getMat(), src2 = psrc2->getMat(), dst = _dst.getMat();
        Size sz = getContinuousSize(src1, src2, dst);
        size_t len = sz.width*(size_t)cn;
        if( len == (size_t)(int)len )
        {
            sz.width = (int)len;
            func(src1.data, src1.step, src2.data, src2.step, dst.data, dst.step, sz, 0);
            return;
        }
    }

    if( oclop == OCL_OP_NOT )
        haveScalar = true;
    else if( (kind1 == _InputArray::MATX) + (kind2 == _InputArray::MATX) == 1 ||
        !psrc1->sameSize(*psrc2) || type1 != type2 )
    {
        if( checkScalar(*psrc1, type2, kind1, kind2) )
        {
            // src1 is a scalar; swap it with src2
            swap(psrc1, psrc2);
            swap(type1, type2);
            swap(depth1, depth2);
            swap(cn, cn2);
            swap(sz1, sz2);
        }
        else if( !checkScalar(*psrc2, type1, kind2, kind1) )
            CV_Error( CV_StsUnmatchedSizes,
                      "The operation is neither 'array op array' (where arrays have the same size and type), "
                      "nor 'array op scalar', nor 'scalar op array'" );
        haveScalar = true;
    }
    else
    {
        CV_Assert( psrc1->sameSize(*psrc2) && type1 == type2 );
    }

    size_t esz = CV_ELEM_SIZE(type1);
    size_t blocksize0 = (BLOCK_SIZE + esz-1)/esz;
    BinaryFunc copymask = 0;
    bool reallocate = false;

    if( haveMask )
    {
        int mtype = _mask.type();
        CV_Assert( (mtype == CV_8U || mtype == CV_8S) && _mask.sameSize(*psrc1));
        copymask = getCopyMaskFunc(esz);
        reallocate = !_dst.sameSize(*psrc1) || _dst.type() != type1;
    }

    AutoBuffer<uchar> _buf;
    uchar *scbuf = 0, *maskbuf = 0;

    _dst.createSameSize(*psrc1, type1);
    // if this is mask operation and dst has been reallocated,
    // we have to clear the destination
    if( haveMask && reallocate )
        _dst.setTo(0.);

    CV_OCL_RUN(use_opencl,
               ocl_binary_op(*psrc1, *psrc2, _dst, _mask, bitwise, oclop, haveScalar))


    Mat src1 = psrc1->getMat(), src2 = psrc2->getMat();
    Mat dst = _dst.getMat(), mask = _mask.getMat();

    if( bitwise )
    {
        func = *tab;
        cn = (int)esz;
    }
    else
        func = tab[depth1];

    if( !haveScalar )
    {
        const Mat* arrays[] = { &src1, &src2, &dst, &mask, 0 };
        uchar* ptrs[4];

        NAryMatIterator it(arrays, ptrs);
        size_t total = it.size, blocksize = total;

        if( blocksize*cn > INT_MAX )
            blocksize = INT_MAX/cn;

        if( haveMask )
        {
            blocksize = std::min(blocksize, blocksize0);
            _buf.allocate(blocksize*esz);
            maskbuf = _buf;
        }

        for( size_t i = 0; i < it.nplanes; i++, ++it )
        {
            for( size_t j = 0; j < total; j += blocksize )
            {
                int bsz = (int)MIN(total - j, blocksize);

                func( ptrs[0], 0, ptrs[1], 0, haveMask ? maskbuf : ptrs[2], 0, Size(bsz*cn, 1), 0 );
                if( haveMask )
                {
                    copymask( maskbuf, 0, ptrs[3], 0, ptrs[2], 0, Size(bsz, 1), &esz );
                    ptrs[3] += bsz;
                }

                bsz *= (int)esz;
                ptrs[0] += bsz; ptrs[1] += bsz; ptrs[2] += bsz;
            }
        }
    }
    else
    {
        const Mat* arrays[] = { &src1, &dst, &mask, 0 };
        uchar* ptrs[3];

        NAryMatIterator it(arrays, ptrs);
        size_t total = it.size, blocksize = std::min(total, blocksize0);

        _buf.allocate(blocksize*(haveMask ? 2 : 1)*esz + 32);
        scbuf = _buf;
        maskbuf = alignPtr(scbuf + blocksize*esz, 16);

        convertAndUnrollScalar( src2, src1.type(), scbuf, blocksize);

        for( size_t i = 0; i < it.nplanes; i++, ++it )
        {
            for( size_t j = 0; j < total; j += blocksize )
            {
                int bsz = (int)MIN(total - j, blocksize);

                func( ptrs[0], 0, scbuf, 0, haveMask ? maskbuf : ptrs[1], 0, Size(bsz*cn, 1), 0 );
                if( haveMask )
                {
                    copymask( maskbuf, 0, ptrs[2], 0, ptrs[1], 0, Size(bsz, 1), &esz );
                    ptrs[2] += bsz;
                }

                bsz *= (int)esz;
                ptrs[0] += bsz; ptrs[1] += bsz;
            }
        }
    }
}

static BinaryFunc* getMaxTab()
{
    static BinaryFunc maxTab[] =
    {
        (BinaryFunc)GET_OPTIMIZED(max8u), (BinaryFunc)GET_OPTIMIZED(max8s),
        (BinaryFunc)GET_OPTIMIZED(max16u), (BinaryFunc)GET_OPTIMIZED(max16s),
        (BinaryFunc)GET_OPTIMIZED(max32s),
        (BinaryFunc)GET_OPTIMIZED(max32f), (BinaryFunc)max64f,
        0
    };

    return maxTab;
}

static BinaryFunc* getMinTab()
{
    static BinaryFunc minTab[] =
    {
        (BinaryFunc)GET_OPTIMIZED(min8u), (BinaryFunc)GET_OPTIMIZED(min8s),
        (BinaryFunc)GET_OPTIMIZED(min16u), (BinaryFunc)GET_OPTIMIZED(min16s),
        (BinaryFunc)GET_OPTIMIZED(min32s),
        (BinaryFunc)GET_OPTIMIZED(min32f), (BinaryFunc)min64f,
        0
    };

    return minTab;
}

}

void cv::bitwise_and(InputArray a, InputArray b, OutputArray c, InputArray mask)
{
    BinaryFunc f = (BinaryFunc)GET_OPTIMIZED(and8u);
    binary_op(a, b, c, mask, &f, true, OCL_OP_AND);
}

void cv::bitwise_or(InputArray a, InputArray b, OutputArray c, InputArray mask)
{
    BinaryFunc f = (BinaryFunc)GET_OPTIMIZED(or8u);
    binary_op(a, b, c, mask, &f, true, OCL_OP_OR);
}

void cv::bitwise_xor(InputArray a, InputArray b, OutputArray c, InputArray mask)
{
    BinaryFunc f = (BinaryFunc)GET_OPTIMIZED(xor8u);
    binary_op(a, b, c, mask, &f, true, OCL_OP_XOR);
}

void cv::bitwise_not(InputArray a, OutputArray c, InputArray mask)
{
    BinaryFunc f = (BinaryFunc)GET_OPTIMIZED(not8u);
    binary_op(a, a, c, mask, &f, true, OCL_OP_NOT);
}

void cv::max( InputArray src1, InputArray src2, OutputArray dst )
{
    binary_op(src1, src2, dst, noArray(), getMaxTab(), false, OCL_OP_MAX );
}

void cv::min( InputArray src1, InputArray src2, OutputArray dst )
{
    binary_op(src1, src2, dst, noArray(), getMinTab(), false, OCL_OP_MIN );
}

void cv::max(const Mat& src1, const Mat& src2, Mat& dst)
{
    OutputArray _dst(dst);
    binary_op(src1, src2, _dst, noArray(), getMaxTab(), false, OCL_OP_MAX );
}

void cv::min(const Mat& src1, const Mat& src2, Mat& dst)
{
    OutputArray _dst(dst);
    binary_op(src1, src2, _dst, noArray(), getMinTab(), false, OCL_OP_MIN );
}

void cv::max(const UMat& src1, const UMat& src2, UMat& dst)
{
    OutputArray _dst(dst);
    binary_op(src1, src2, _dst, noArray(), getMaxTab(), false, OCL_OP_MAX );
}

void cv::min(const UMat& src1, const UMat& src2, UMat& dst)
{
    OutputArray _dst(dst);
    binary_op(src1, src2, _dst, noArray(), getMinTab(), false, OCL_OP_MIN );
}


/****************************************************************************************\
*                                      add/subtract                                      *
\****************************************************************************************/

namespace cv
{

static int actualScalarDepth(const double* data, int len)
{
    int i = 0, minval = INT_MAX, maxval = INT_MIN;
    for(; i < len; ++i)
    {
        int ival = cvRound(data[i]);
        if( ival != data[i] )
            break;
        minval = MIN(minval, ival);
        maxval = MAX(maxval, ival);
    }
    return i < len ? CV_64F :
        minval >= 0 && maxval <= (int)UCHAR_MAX ? CV_8U :
        minval >= (int)SCHAR_MIN && maxval <= (int)SCHAR_MAX ? CV_8S :
        minval >= 0 && maxval <= (int)USHRT_MAX ? CV_16U :
        minval >= (int)SHRT_MIN && maxval <= (int)SHRT_MAX ? CV_16S :
        CV_32S;
}

#ifdef HAVE_OPENCL

static bool ocl_arithm_op(InputArray _src1, InputArray _src2, OutputArray _dst,
                          InputArray _mask, int wtype,
                          void* usrdata, int oclop,
                          bool haveScalar )
{
    const ocl::Device d = ocl::Device::getDefault();
    bool doubleSupport = d.doubleFPConfig() > 0;
    int type1 = _src1.type(), depth1 = CV_MAT_DEPTH(type1), cn = CV_MAT_CN(type1);
    bool haveMask = !_mask.empty();

    if ( (haveMask || haveScalar) && cn > 4 )
        return false;

    int dtype = _dst.type(), ddepth = CV_MAT_DEPTH(dtype), wdepth = std::max(CV_32S, CV_MAT_DEPTH(wtype));
    if (!doubleSupport)
        wdepth = std::min(wdepth, CV_32F);

    wtype = CV_MAKETYPE(wdepth, cn);
    int type2 = haveScalar ? wtype : _src2.type(), depth2 = CV_MAT_DEPTH(type2);
    if (!doubleSupport && (depth2 == CV_64F || depth1 == CV_64F))
        return false;

    int kercn = haveMask || haveScalar ? cn : ocl::predictOptimalVectorWidth(_src1, _src2, _dst);
    int scalarcn = kercn == 3 ? 4 : kercn, rowsPerWI = d.isIntel() ? 4 : 1;

    char cvtstr[4][32], opts[1024];
    sprintf(opts, "-D %s%s -D %s -D srcT1=%s -D srcT1_C1=%s -D srcT2=%s -D srcT2_C1=%s "
            "-D dstT=%s -D dstT_C1=%s -D workT=%s -D workST=%s -D scaleT=%s -D wdepth=%d -D convertToWT1=%s "
            "-D convertToWT2=%s -D convertToDT=%s%s -D cn=%d -D rowsPerWI=%d -D convertFromU=%s",
            (haveMask ? "MASK_" : ""), (haveScalar ? "UNARY_OP" : "BINARY_OP"),
            oclop2str[oclop], ocl::typeToStr(CV_MAKETYPE(depth1, kercn)),
            ocl::typeToStr(depth1), ocl::typeToStr(CV_MAKETYPE(depth2, kercn)),
            ocl::typeToStr(depth2), ocl::typeToStr(CV_MAKETYPE(ddepth, kercn)),
            ocl::typeToStr(ddepth), ocl::typeToStr(CV_MAKETYPE(wdepth, kercn)),
            ocl::typeToStr(CV_MAKETYPE(wdepth, scalarcn)),
            ocl::typeToStr(wdepth), wdepth,
            ocl::convertTypeStr(depth1, wdepth, kercn, cvtstr[0]),
            ocl::convertTypeStr(depth2, wdepth, kercn, cvtstr[1]),
            ocl::convertTypeStr(wdepth, ddepth, kercn, cvtstr[2]),
            doubleSupport ? " -D DOUBLE_SUPPORT" : "", kercn, rowsPerWI,
            oclop == OCL_OP_ABSDIFF && wdepth == CV_32S && ddepth == wdepth ?
            ocl::convertTypeStr(CV_8U, ddepth, kercn, cvtstr[3]) : "noconvert");

    size_t usrdata_esz = CV_ELEM_SIZE(wdepth);
    const uchar* usrdata_p = (const uchar*)usrdata;
    const double* usrdata_d = (const double*)usrdata;
    float usrdata_f[3];
    int i, n = oclop == OCL_OP_MUL_SCALE || oclop == OCL_OP_DIV_SCALE ||
        oclop == OCL_OP_RDIV_SCALE || oclop == OCL_OP_RECIP_SCALE ? 1 : oclop == OCL_OP_ADDW ? 3 : 0;
    if( n > 0 && wdepth == CV_32F )
    {
        for( i = 0; i < n; i++ )
            usrdata_f[i] = (float)usrdata_d[i];
        usrdata_p = (const uchar*)usrdata_f;
    }

    ocl::Kernel k("KF", ocl::core::arithm_oclsrc, opts);
    if (k.empty())
        return false;

    UMat src1 = _src1.getUMat(), src2;
    UMat dst = _dst.getUMat(), mask = _mask.getUMat();

    ocl::KernelArg src1arg = ocl::KernelArg::ReadOnlyNoSize(src1, cn, kercn);
    ocl::KernelArg dstarg = haveMask ? ocl::KernelArg::ReadWrite(dst, cn, kercn) :
                                       ocl::KernelArg::WriteOnly(dst, cn, kercn);
    ocl::KernelArg maskarg = ocl::KernelArg::ReadOnlyNoSize(mask, 1);

    if( haveScalar )
    {
        size_t esz = CV_ELEM_SIZE1(wtype)*scalarcn;
        double buf[4]={0,0,0,0};
        Mat src2sc = _src2.getMat();

        if( !src2sc.empty() )
            convertAndUnrollScalar(src2sc, wtype, (uchar*)buf, 1);
        ocl::KernelArg scalararg = ocl::KernelArg(0, 0, 0, 0, buf, esz);

        if( !haveMask )
        {
            if(n == 0)
                k.args(src1arg, dstarg, scalararg);
            else if(n == 1)
                k.args(src1arg, dstarg, scalararg,
                       ocl::KernelArg(0, 0, 0, 0, usrdata_p, usrdata_esz));
            else
                CV_Error(Error::StsNotImplemented, "unsupported number of extra parameters");
        }
        else
            k.args(src1arg, maskarg, dstarg, scalararg);
    }
    else
    {
        src2 = _src2.getUMat();
        ocl::KernelArg src2arg = ocl::KernelArg::ReadOnlyNoSize(src2, cn, kercn);

        if( !haveMask )
        {
            if (n == 0)
                k.args(src1arg, src2arg, dstarg);
            else if (n == 1)
                k.args(src1arg, src2arg, dstarg,
                       ocl::KernelArg(0, 0, 0, 0, usrdata_p, usrdata_esz));
            else if (n == 3)
                k.args(src1arg, src2arg, dstarg,
                       ocl::KernelArg(0, 0, 0, 0, usrdata_p, usrdata_esz),
                       ocl::KernelArg(0, 0, 0, 0, usrdata_p + usrdata_esz, usrdata_esz),
                       ocl::KernelArg(0, 0, 0, 0, usrdata_p + usrdata_esz*2, usrdata_esz));
            else
                CV_Error(Error::StsNotImplemented, "unsupported number of extra parameters");
        }
        else
            k.args(src1arg, src2arg, maskarg, dstarg);
    }

    size_t globalsize[] = { src1.cols * cn / kercn, (src1.rows + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, NULL, false);
}

#endif

static void arithm_op(InputArray _src1, InputArray _src2, OutputArray _dst,
                      InputArray _mask, int dtype, BinaryFunc* tab, bool muldiv=false,
                      void* usrdata=0, int oclop=-1 )
{
    const _InputArray *psrc1 = &_src1, *psrc2 = &_src2;
    int kind1 = psrc1->kind(), kind2 = psrc2->kind();
    bool haveMask = !_mask.empty();
    bool reallocate = false;
    int type1 = psrc1->type(), depth1 = CV_MAT_DEPTH(type1), cn = CV_MAT_CN(type1);
    int type2 = psrc2->type(), depth2 = CV_MAT_DEPTH(type2), cn2 = CV_MAT_CN(type2);
    int wtype, dims1 = psrc1->dims(), dims2 = psrc2->dims();
    Size sz1 = dims1 <= 2 ? psrc1->size() : Size();
    Size sz2 = dims2 <= 2 ? psrc2->size() : Size();
#ifdef HAVE_OPENCL
    bool use_opencl = _dst.isUMat() && dims1 <= 2 && dims2 <= 2;
#endif
    bool src1Scalar = checkScalar(*psrc1, type2, kind1, kind2);
    bool src2Scalar = checkScalar(*psrc2, type1, kind2, kind1);

    if( (kind1 == kind2 || cn == 1) && sz1 == sz2 && dims1 <= 2 && dims2 <= 2 && type1 == type2 &&
        !haveMask && ((!_dst.fixedType() && (dtype < 0 || CV_MAT_DEPTH(dtype) == depth1)) ||
                       (_dst.fixedType() && _dst.type() == type1)) &&
        ((src1Scalar && src2Scalar) || (!src1Scalar && !src2Scalar)) )
    {
        _dst.createSameSize(*psrc1, type1);
        CV_OCL_RUN(use_opencl,
            ocl_arithm_op(*psrc1, *psrc2, _dst, _mask,
                          (!usrdata ? type1 : std::max(depth1, CV_32F)),
                          usrdata, oclop, false))

        Mat src1 = psrc1->getMat(), src2 = psrc2->getMat(), dst = _dst.getMat();
        Size sz = getContinuousSize(src1, src2, dst, src1.channels());
        tab[depth1](src1.data, src1.step, src2.data, src2.step, dst.data, dst.step, sz, usrdata);
        return;
    }

    bool haveScalar = false, swapped12 = false;

    if( dims1 != dims2 || sz1 != sz2 || cn != cn2 ||
        (kind1 == _InputArray::MATX && (sz1 == Size(1,4) || sz1 == Size(1,1))) ||
        (kind2 == _InputArray::MATX && (sz2 == Size(1,4) || sz2 == Size(1,1))) )
    {
        if( checkScalar(*psrc1, type2, kind1, kind2) )
        {
            // src1 is a scalar; swap it with src2
            swap(psrc1, psrc2);
            swap(sz1, sz2);
            swap(type1, type2);
            swap(depth1, depth2);
            swap(cn, cn2);
            swap(dims1, dims2);
            swapped12 = true;
            if( oclop == OCL_OP_SUB )
                oclop = OCL_OP_RSUB;
            if ( oclop == OCL_OP_DIV_SCALE )
                oclop = OCL_OP_RDIV_SCALE;
        }
        else if( !checkScalar(*psrc2, type1, kind2, kind1) )
            CV_Error( CV_StsUnmatchedSizes,
                     "The operation is neither 'array op array' "
                     "(where arrays have the same size and the same number of channels), "
                     "nor 'array op scalar', nor 'scalar op array'" );
        haveScalar = true;
        CV_Assert(type2 == CV_64F && (sz2.height == 1 || sz2.height == 4));

        if (!muldiv)
        {
            Mat sc = psrc2->getMat();
            depth2 = actualScalarDepth(sc.ptr<double>(), cn);
            if( depth2 == CV_64F && (depth1 < CV_32S || depth1 == CV_32F) )
                depth2 = CV_32F;
        }
        else
            depth2 = CV_64F;
    }

    if( dtype < 0 )
    {
        if( _dst.fixedType() )
            dtype = _dst.type();
        else
        {
            if( !haveScalar && type1 != type2 )
                CV_Error(CV_StsBadArg,
                     "When the input arrays in add/subtract/multiply/divide functions have different types, "
                     "the output array type must be explicitly specified");
            dtype = type1;
        }
    }
    dtype = CV_MAT_DEPTH(dtype);

    if( depth1 == depth2 && dtype == depth1 )
        wtype = dtype;
    else if( !muldiv )
    {
        wtype = depth1 <= CV_8S && depth2 <= CV_8S ? CV_16S :
                depth1 <= CV_32S && depth2 <= CV_32S ? CV_32S : std::max(depth1, depth2);
        wtype = std::max(wtype, dtype);

        // when the result of addition should be converted to an integer type,
        // and just one of the input arrays is floating-point, it makes sense to convert that input to integer type before the operation,
        // instead of converting the other input to floating-point and then converting the operation result back to integers.
        if( dtype < CV_32F && (depth1 < CV_32F || depth2 < CV_32F) )
            wtype = CV_32S;
    }
    else
    {
        wtype = std::max(depth1, std::max(depth2, CV_32F));
        wtype = std::max(wtype, dtype);
    }

    dtype = CV_MAKETYPE(dtype, cn);
    wtype = CV_MAKETYPE(wtype, cn);

    if( haveMask )
    {
        int mtype = _mask.type();
        CV_Assert( (mtype == CV_8UC1 || mtype == CV_8SC1) && _mask.sameSize(*psrc1) );
        reallocate = !_dst.sameSize(*psrc1) || _dst.type() != dtype;
    }

    _dst.createSameSize(*psrc1, dtype);
    if( reallocate )
        _dst.setTo(0.);

    CV_OCL_RUN(use_opencl,
               ocl_arithm_op(*psrc1, *psrc2, _dst, _mask, wtype,
               usrdata, oclop, haveScalar))

    BinaryFunc cvtsrc1 = type1 == wtype ? 0 : getConvertFunc(type1, wtype);
    BinaryFunc cvtsrc2 = type2 == type1 ? cvtsrc1 : type2 == wtype ? 0 : getConvertFunc(type2, wtype);
    BinaryFunc cvtdst = dtype == wtype ? 0 : getConvertFunc(wtype, dtype);

    size_t esz1 = CV_ELEM_SIZE(type1), esz2 = CV_ELEM_SIZE(type2);
    size_t dsz = CV_ELEM_SIZE(dtype), wsz = CV_ELEM_SIZE(wtype);
    size_t blocksize0 = (size_t)(BLOCK_SIZE + wsz-1)/wsz;
    BinaryFunc copymask = getCopyMaskFunc(dsz);
    Mat src1 = psrc1->getMat(), src2 = psrc2->getMat(), dst = _dst.getMat(), mask = _mask.getMat();

    AutoBuffer<uchar> _buf;
    uchar *buf, *maskbuf = 0, *buf1 = 0, *buf2 = 0, *wbuf = 0;
    size_t bufesz = (cvtsrc1 ? wsz : 0) +
                    (cvtsrc2 || haveScalar ? wsz : 0) +
                    (cvtdst ? wsz : 0) +
                    (haveMask ? dsz : 0);
    BinaryFunc func = tab[CV_MAT_DEPTH(wtype)];

    if( !haveScalar )
    {
        const Mat* arrays[] = { &src1, &src2, &dst, &mask, 0 };
        uchar* ptrs[4];

        NAryMatIterator it(arrays, ptrs);
        size_t total = it.size, blocksize = total;

        if( haveMask || cvtsrc1 || cvtsrc2 || cvtdst )
            blocksize = std::min(blocksize, blocksize0);

        _buf.allocate(bufesz*blocksize + 64);
        buf = _buf;
        if( cvtsrc1 )
            buf1 = buf, buf = alignPtr(buf + blocksize*wsz, 16);
        if( cvtsrc2 )
            buf2 = buf, buf = alignPtr(buf + blocksize*wsz, 16);
        wbuf = maskbuf = buf;
        if( cvtdst )
            buf = alignPtr(buf + blocksize*wsz, 16);
        if( haveMask )
            maskbuf = buf;

        for( size_t i = 0; i < it.nplanes; i++, ++it )
        {
            for( size_t j = 0; j < total; j += blocksize )
            {
                int bsz = (int)MIN(total - j, blocksize);
                Size bszn(bsz*cn, 1);
                const uchar *sptr1 = ptrs[0], *sptr2 = ptrs[1];
                uchar* dptr = ptrs[2];
                if( cvtsrc1 )
                {
                    cvtsrc1( sptr1, 1, 0, 1, buf1, 1, bszn, 0 );
                    sptr1 = buf1;
                }
                if( ptrs[0] == ptrs[1] )
                    sptr2 = sptr1;
                else if( cvtsrc2 )
                {
                    cvtsrc2( sptr2, 1, 0, 1, buf2, 1, bszn, 0 );
                    sptr2 = buf2;
                }

                if( !haveMask && !cvtdst )
                    func( sptr1, 1, sptr2, 1, dptr, 1, bszn, usrdata );
                else
                {
                    func( sptr1, 1, sptr2, 1, wbuf, 0, bszn, usrdata );
                    if( !haveMask )
                        cvtdst( wbuf, 1, 0, 1, dptr, 1, bszn, 0 );
                    else if( !cvtdst )
                    {
                        copymask( wbuf, 1, ptrs[3], 1, dptr, 1, Size(bsz, 1), &dsz );
                        ptrs[3] += bsz;
                    }
                    else
                    {
                        cvtdst( wbuf, 1, 0, 1, maskbuf, 1, bszn, 0 );
                        copymask( maskbuf, 1, ptrs[3], 1, dptr, 1, Size(bsz, 1), &dsz );
                        ptrs[3] += bsz;
                    }
                }
                ptrs[0] += bsz*esz1; ptrs[1] += bsz*esz2; ptrs[2] += bsz*dsz;
            }
        }
    }
    else
    {
        const Mat* arrays[] = { &src1, &dst, &mask, 0 };
        uchar* ptrs[3];

        NAryMatIterator it(arrays, ptrs);
        size_t total = it.size, blocksize = std::min(total, blocksize0);

        _buf.allocate(bufesz*blocksize + 64);
        buf = _buf;
        if( cvtsrc1 )
            buf1 = buf, buf = alignPtr(buf + blocksize*wsz, 16);
        buf2 = buf; buf = alignPtr(buf + blocksize*wsz, 16);
        wbuf = maskbuf = buf;
        if( cvtdst )
            buf = alignPtr(buf + blocksize*wsz, 16);
        if( haveMask )
            maskbuf = buf;

        convertAndUnrollScalar( src2, wtype, buf2, blocksize);

        for( size_t i = 0; i < it.nplanes; i++, ++it )
        {
            for( size_t j = 0; j < total; j += blocksize )
            {
                int bsz = (int)MIN(total - j, blocksize);
                Size bszn(bsz*cn, 1);
                const uchar *sptr1 = ptrs[0];
                const uchar* sptr2 = buf2;
                uchar* dptr = ptrs[1];

                if( cvtsrc1 )
                {
                    cvtsrc1( sptr1, 1, 0, 1, buf1, 1, bszn, 0 );
                    sptr1 = buf1;
                }

                if( swapped12 )
                    std::swap(sptr1, sptr2);

                if( !haveMask && !cvtdst )
                    func( sptr1, 1, sptr2, 1, dptr, 1, bszn, usrdata );
                else
                {
                    func( sptr1, 1, sptr2, 1, wbuf, 1, bszn, usrdata );
                    if( !haveMask )
                        cvtdst( wbuf, 1, 0, 1, dptr, 1, bszn, 0 );
                    else if( !cvtdst )
                    {
                        copymask( wbuf, 1, ptrs[2], 1, dptr, 1, Size(bsz, 1), &dsz );
                        ptrs[2] += bsz;
                    }
                    else
                    {
                        cvtdst( wbuf, 1, 0, 1, maskbuf, 1, bszn, 0 );
                        copymask( maskbuf, 1, ptrs[2], 1, dptr, 1, Size(bsz, 1), &dsz );
                        ptrs[2] += bsz;
                    }
                }
                ptrs[0] += bsz*esz1; ptrs[1] += bsz*dsz;
            }
        }
    }
}

static BinaryFunc* getAddTab()
{
    static BinaryFunc addTab[] =
    {
        (BinaryFunc)GET_OPTIMIZED(add8u), (BinaryFunc)GET_OPTIMIZED(add8s),
        (BinaryFunc)GET_OPTIMIZED(add16u), (BinaryFunc)GET_OPTIMIZED(add16s),
        (BinaryFunc)GET_OPTIMIZED(add32s),
        (BinaryFunc)GET_OPTIMIZED(add32f), (BinaryFunc)add64f,
        0
    };

    return addTab;
}

static BinaryFunc* getSubTab()
{
    static BinaryFunc subTab[] =
    {
        (BinaryFunc)GET_OPTIMIZED(sub8u), (BinaryFunc)GET_OPTIMIZED(sub8s),
        (BinaryFunc)GET_OPTIMIZED(sub16u), (BinaryFunc)GET_OPTIMIZED(sub16s),
        (BinaryFunc)GET_OPTIMIZED(sub32s),
        (BinaryFunc)GET_OPTIMIZED(sub32f), (BinaryFunc)sub64f,
        0
    };

    return subTab;
}

static BinaryFunc* getAbsDiffTab()
{
    static BinaryFunc absDiffTab[] =
    {
        (BinaryFunc)GET_OPTIMIZED(absdiff8u), (BinaryFunc)GET_OPTIMIZED(absdiff8s),
        (BinaryFunc)GET_OPTIMIZED(absdiff16u), (BinaryFunc)GET_OPTIMIZED(absdiff16s),
        (BinaryFunc)GET_OPTIMIZED(absdiff32s),
        (BinaryFunc)GET_OPTIMIZED(absdiff32f), (BinaryFunc)absdiff64f,
        0
    };

    return absDiffTab;
}

}

void cv::add( InputArray src1, InputArray src2, OutputArray dst,
          InputArray mask, int dtype )
{
    arithm_op(src1, src2, dst, mask, dtype, getAddTab(), false, 0, OCL_OP_ADD );
}

void cv::subtract( InputArray _src1, InputArray _src2, OutputArray _dst,
               InputArray mask, int dtype )
{
#ifdef HAVE_TEGRA_OPTIMIZATION
    int kind1 = _src1.kind(), kind2 = _src2.kind();
    Mat src1 = _src1.getMat(), src2 = _src2.getMat();
    bool src1Scalar = checkScalar(src1, _src2.type(), kind1, kind2);
    bool src2Scalar = checkScalar(src2, _src1.type(), kind2, kind1);

    if (!src1Scalar && !src2Scalar &&
        src1.depth() == CV_8U && src2.type() == src1.type() &&
        src1.dims == 2 && src2.size() == src1.size() &&
        mask.empty())
    {
        if (dtype < 0)
        {
            if (_dst.fixedType())
            {
                dtype = _dst.depth();
            }
            else
            {
                dtype = src1.depth();
            }
        }

        dtype = CV_MAT_DEPTH(dtype);

        if (!_dst.fixedType() || dtype == _dst.depth())
        {
            _dst.create(src1.size(), CV_MAKE_TYPE(dtype, src1.channels()));

            if (dtype == CV_16S)
            {
                Mat dst = _dst.getMat();
                if(tegra::subtract_8u8u16s(src1, src2, dst))
                    return;
            }
            else if (dtype == CV_32F)
            {
                Mat dst = _dst.getMat();
                if(tegra::subtract_8u8u32f(src1, src2, dst))
                    return;
            }
            else if (dtype == CV_8S)
            {
                Mat dst = _dst.getMat();
                if(tegra::subtract_8u8u8s(src1, src2, dst))
                    return;
            }
        }
    }
#endif
    arithm_op(_src1, _src2, _dst, mask, dtype, getSubTab(), false, 0, OCL_OP_SUB );
}

void cv::absdiff( InputArray src1, InputArray src2, OutputArray dst )
{
    arithm_op(src1, src2, dst, noArray(), -1, getAbsDiffTab(), false, 0, OCL_OP_ABSDIFF);
}

/****************************************************************************************\
*                                    multiply/divide                                     *
\****************************************************************************************/

namespace cv
{

template<typename T, typename WT> static void
mul_( const T* src1, size_t step1, const T* src2, size_t step2,
      T* dst, size_t step, Size size, WT scale )
{
    step1 /= sizeof(src1[0]);
    step2 /= sizeof(src2[0]);
    step /= sizeof(dst[0]);

    if( scale == (WT)1. )
    {
        for( ; size.height--; src1 += step1, src2 += step2, dst += step )
        {
            int i=0;
            #if CV_ENABLE_UNROLLED
            for(; i <= size.width - 4; i += 4 )
            {
                T t0;
                T t1;
                t0 = saturate_cast<T>(src1[i  ] * src2[i  ]);
                t1 = saturate_cast<T>(src1[i+1] * src2[i+1]);
                dst[i  ] = t0;
                dst[i+1] = t1;

                t0 = saturate_cast<T>(src1[i+2] * src2[i+2]);
                t1 = saturate_cast<T>(src1[i+3] * src2[i+3]);
                dst[i+2] = t0;
                dst[i+3] = t1;
            }
            #endif
            for( ; i < size.width; i++ )
                dst[i] = saturate_cast<T>(src1[i] * src2[i]);
        }
    }
    else
    {
        for( ; size.height--; src1 += step1, src2 += step2, dst += step )
        {
            int i = 0;
            #if CV_ENABLE_UNROLLED
            for(; i <= size.width - 4; i += 4 )
            {
                T t0 = saturate_cast<T>(scale*(WT)src1[i]*src2[i]);
                T t1 = saturate_cast<T>(scale*(WT)src1[i+1]*src2[i+1]);
                dst[i] = t0; dst[i+1] = t1;

                t0 = saturate_cast<T>(scale*(WT)src1[i+2]*src2[i+2]);
                t1 = saturate_cast<T>(scale*(WT)src1[i+3]*src2[i+3]);
                dst[i+2] = t0; dst[i+3] = t1;
            }
            #endif
            for( ; i < size.width; i++ )
                dst[i] = saturate_cast<T>(scale*(WT)src1[i]*src2[i]);
        }
    }
}

template<typename T> static void
div_( const T* src1, size_t step1, const T* src2, size_t step2,
      T* dst, size_t step, Size size, double scale )
{
    step1 /= sizeof(src1[0]);
    step2 /= sizeof(src2[0]);
    step /= sizeof(dst[0]);

    for( ; size.height--; src1 += step1, src2 += step2, dst += step )
    {
        int i = 0;
        #if CV_ENABLE_UNROLLED
        for( ; i <= size.width - 4; i += 4 )
        {
            if( src2[i] != 0 && src2[i+1] != 0 && src2[i+2] != 0 && src2[i+3] != 0 )
            {
                double a = (double)src2[i] * src2[i+1];
                double b = (double)src2[i+2] * src2[i+3];
                double d = scale/(a * b);
                b *= d;
                a *= d;

                T z0 = saturate_cast<T>(src2[i+1] * ((double)src1[i] * b));
                T z1 = saturate_cast<T>(src2[i] * ((double)src1[i+1] * b));
                T z2 = saturate_cast<T>(src2[i+3] * ((double)src1[i+2] * a));
                T z3 = saturate_cast<T>(src2[i+2] * ((double)src1[i+3] * a));

                dst[i] = z0; dst[i+1] = z1;
                dst[i+2] = z2; dst[i+3] = z3;
            }
            else
            {
                T z0 = src2[i] != 0 ? saturate_cast<T>(src1[i]*scale/src2[i]) : 0;
                T z1 = src2[i+1] != 0 ? saturate_cast<T>(src1[i+1]*scale/src2[i+1]) : 0;
                T z2 = src2[i+2] != 0 ? saturate_cast<T>(src1[i+2]*scale/src2[i+2]) : 0;
                T z3 = src2[i+3] != 0 ? saturate_cast<T>(src1[i+3]*scale/src2[i+3]) : 0;

                dst[i] = z0; dst[i+1] = z1;
                dst[i+2] = z2; dst[i+3] = z3;
            }
        }
        #endif
        for( ; i < size.width; i++ )
            dst[i] = src2[i] != 0 ? saturate_cast<T>(src1[i]*scale/src2[i]) : 0;
    }
}

template<typename T> static void
recip_( const T*, size_t, const T* src2, size_t step2,
        T* dst, size_t step, Size size, double scale )
{
    step2 /= sizeof(src2[0]);
    step /= sizeof(dst[0]);

    for( ; size.height--; src2 += step2, dst += step )
    {
        int i = 0;
        #if CV_ENABLE_UNROLLED
        for( ; i <= size.width - 4; i += 4 )
        {
            if( src2[i] != 0 && src2[i+1] != 0 && src2[i+2] != 0 && src2[i+3] != 0 )
            {
                double a = (double)src2[i] * src2[i+1];
                double b = (double)src2[i+2] * src2[i+3];
                double d = scale/(a * b);
                b *= d;
                a *= d;

                T z0 = saturate_cast<T>(src2[i+1] * b);
                T z1 = saturate_cast<T>(src2[i] * b);
                T z2 = saturate_cast<T>(src2[i+3] * a);
                T z3 = saturate_cast<T>(src2[i+2] * a);

                dst[i] = z0; dst[i+1] = z1;
                dst[i+2] = z2; dst[i+3] = z3;
            }
            else
            {
                T z0 = src2[i] != 0 ? saturate_cast<T>(scale/src2[i]) : 0;
                T z1 = src2[i+1] != 0 ? saturate_cast<T>(scale/src2[i+1]) : 0;
                T z2 = src2[i+2] != 0 ? saturate_cast<T>(scale/src2[i+2]) : 0;
                T z3 = src2[i+3] != 0 ? saturate_cast<T>(scale/src2[i+3]) : 0;

                dst[i] = z0; dst[i+1] = z1;
                dst[i+2] = z2; dst[i+3] = z3;
            }
        }
        #endif
        for( ; i < size.width; i++ )
            dst[i] = src2[i] != 0 ? saturate_cast<T>(scale/src2[i]) : 0;
    }
}


static void mul8u( const uchar* src1, size_t step1, const uchar* src2, size_t step2,
                   uchar* dst, size_t step, Size sz, void* scale)
{
    float fscale = (float)*(const double*)scale;
#if defined HAVE_IPP
    if (std::fabs(fscale - 1) <= FLT_EPSILON)
    {
        if (ippiMul_8u_C1RSfs(src1, (int)step1, src2, (int)step2, dst, (int)step, ippiSize(sz), 0) >= 0)
            return;
        setIppErrorStatus();
    }
#endif
    mul_(src1, step1, src2, step2, dst, step, sz, fscale);
}

static void mul8s( const schar* src1, size_t step1, const schar* src2, size_t step2,
                   schar* dst, size_t step, Size sz, void* scale)
{
    mul_(src1, step1, src2, step2, dst, step, sz, (float)*(const double*)scale);
}

static void mul16u( const ushort* src1, size_t step1, const ushort* src2, size_t step2,
                    ushort* dst, size_t step, Size sz, void* scale)
{
    float fscale = (float)*(const double*)scale;
#if defined HAVE_IPP
    if (std::fabs(fscale - 1) <= FLT_EPSILON)
    {
        if (ippiMul_16u_C1RSfs(src1, (int)step1, src2, (int)step2, dst, (int)step, ippiSize(sz), 0) >= 0)
            return;
        setIppErrorStatus();
    }
#endif
    mul_(src1, step1, src2, step2, dst, step, sz, fscale);
}

static void mul16s( const short* src1, size_t step1, const short* src2, size_t step2,
                    short* dst, size_t step, Size sz, void* scale)
{
    float fscale = (float)*(const double*)scale;
#if defined HAVE_IPP
    if (std::fabs(fscale - 1) <= FLT_EPSILON)
    {
        if (ippiMul_16s_C1RSfs(src1, (int)step1, src2, (int)step2, dst, (int)step, ippiSize(sz), 0) >= 0)
            return;
        setIppErrorStatus();
    }
#endif
    mul_(src1, step1, src2, step2, dst, step, sz, fscale);
}

static void mul32s( const int* src1, size_t step1, const int* src2, size_t step2,
                    int* dst, size_t step, Size sz, void* scale)
{
    mul_(src1, step1, src2, step2, dst, step, sz, *(const double*)scale);
}

static void mul32f( const float* src1, size_t step1, const float* src2, size_t step2,
                    float* dst, size_t step, Size sz, void* scale)
{
    float fscale = (float)*(const double*)scale;
#if defined HAVE_IPP
    if (std::fabs(fscale - 1) <= FLT_EPSILON)
    {
        if (ippiMul_32f_C1R(src1, (int)step1, src2, (int)step2, dst, (int)step, ippiSize(sz)) >= 0)
            return;
        setIppErrorStatus();
    }
#endif
    mul_(src1, step1, src2, step2, dst, step, sz, fscale);
}

static void mul64f( const double* src1, size_t step1, const double* src2, size_t step2,
                    double* dst, size_t step, Size sz, void* scale)
{
    mul_(src1, step1, src2, step2, dst, step, sz, *(const double*)scale);
}

static void div8u( const uchar* src1, size_t step1, const uchar* src2, size_t step2,
                   uchar* dst, size_t step, Size sz, void* scale)
{
    if( src1 )
        div_(src1, step1, src2, step2, dst, step, sz, *(const double*)scale);
    else
        recip_(src1, step1, src2, step2, dst, step, sz, *(const double*)scale);
}

static void div8s( const schar* src1, size_t step1, const schar* src2, size_t step2,
                  schar* dst, size_t step, Size sz, void* scale)
{
    div_(src1, step1, src2, step2, dst, step, sz, *(const double*)scale);
}

static void div16u( const ushort* src1, size_t step1, const ushort* src2, size_t step2,
                    ushort* dst, size_t step, Size sz, void* scale)
{
    div_(src1, step1, src2, step2, dst, step, sz, *(const double*)scale);
}

static void div16s( const short* src1, size_t step1, const short* src2, size_t step2,
                    short* dst, size_t step, Size sz, void* scale)
{
    div_(src1, step1, src2, step2, dst, step, sz, *(const double*)scale);
}

static void div32s( const int* src1, size_t step1, const int* src2, size_t step2,
                    int* dst, size_t step, Size sz, void* scale)
{
    div_(src1, step1, src2, step2, dst, step, sz, *(const double*)scale);
}

static void div32f( const float* src1, size_t step1, const float* src2, size_t step2,
                    float* dst, size_t step, Size sz, void* scale)
{
    div_(src1, step1, src2, step2, dst, step, sz, *(const double*)scale);
}

static void div64f( const double* src1, size_t step1, const double* src2, size_t step2,
                    double* dst, size_t step, Size sz, void* scale)
{
    div_(src1, step1, src2, step2, dst, step, sz, *(const double*)scale);
}

static void recip8u( const uchar* src1, size_t step1, const uchar* src2, size_t step2,
                  uchar* dst, size_t step, Size sz, void* scale)
{
    recip_(src1, step1, src2, step2, dst, step, sz, *(const double*)scale);
}

static void recip8s( const schar* src1, size_t step1, const schar* src2, size_t step2,
                  schar* dst, size_t step, Size sz, void* scale)
{
    recip_(src1, step1, src2, step2, dst, step, sz, *(const double*)scale);
}

static void recip16u( const ushort* src1, size_t step1, const ushort* src2, size_t step2,
                   ushort* dst, size_t step, Size sz, void* scale)
{
    recip_(src1, step1, src2, step2, dst, step, sz, *(const double*)scale);
}

static void recip16s( const short* src1, size_t step1, const short* src2, size_t step2,
                   short* dst, size_t step, Size sz, void* scale)
{
    recip_(src1, step1, src2, step2, dst, step, sz, *(const double*)scale);
}

static void recip32s( const int* src1, size_t step1, const int* src2, size_t step2,
                   int* dst, size_t step, Size sz, void* scale)
{
    recip_(src1, step1, src2, step2, dst, step, sz, *(const double*)scale);
}

static void recip32f( const float* src1, size_t step1, const float* src2, size_t step2,
                   float* dst, size_t step, Size sz, void* scale)
{
    recip_(src1, step1, src2, step2, dst, step, sz, *(const double*)scale);
}

static void recip64f( const double* src1, size_t step1, const double* src2, size_t step2,
                   double* dst, size_t step, Size sz, void* scale)
{
    recip_(src1, step1, src2, step2, dst, step, sz, *(const double*)scale);
}


static BinaryFunc* getMulTab()
{
    static BinaryFunc mulTab[] =
    {
        (BinaryFunc)mul8u, (BinaryFunc)mul8s, (BinaryFunc)mul16u,
        (BinaryFunc)mul16s, (BinaryFunc)mul32s, (BinaryFunc)mul32f,
        (BinaryFunc)mul64f, 0
    };

    return mulTab;
}

static BinaryFunc* getDivTab()
{
    static BinaryFunc divTab[] =
    {
        (BinaryFunc)div8u, (BinaryFunc)div8s, (BinaryFunc)div16u,
        (BinaryFunc)div16s, (BinaryFunc)div32s, (BinaryFunc)div32f,
        (BinaryFunc)div64f, 0
    };

    return divTab;
}

static BinaryFunc* getRecipTab()
{
    static BinaryFunc recipTab[] =
    {
        (BinaryFunc)recip8u, (BinaryFunc)recip8s, (BinaryFunc)recip16u,
        (BinaryFunc)recip16s, (BinaryFunc)recip32s, (BinaryFunc)recip32f,
        (BinaryFunc)recip64f, 0
    };

    return recipTab;
}

}

void cv::multiply(InputArray src1, InputArray src2,
                  OutputArray dst, double scale, int dtype)
{
    arithm_op(src1, src2, dst, noArray(), dtype, getMulTab(),
              true, &scale, std::abs(scale - 1.0) < DBL_EPSILON ? OCL_OP_MUL : OCL_OP_MUL_SCALE);
}

void cv::divide(InputArray src1, InputArray src2,
                OutputArray dst, double scale, int dtype)
{
    arithm_op(src1, src2, dst, noArray(), dtype, getDivTab(), true, &scale, OCL_OP_DIV_SCALE);
}

void cv::divide(double scale, InputArray src2,
                OutputArray dst, int dtype)
{
    arithm_op(src2, src2, dst, noArray(), dtype, getRecipTab(), true, &scale, OCL_OP_RECIP_SCALE);
}

/****************************************************************************************\
*                                      addWeighted                                       *
\****************************************************************************************/

namespace cv
{

template<typename T, typename WT> static void
addWeighted_( const T* src1, size_t step1, const T* src2, size_t step2,
              T* dst, size_t step, Size size, void* _scalars )
{
    const double* scalars = (const double*)_scalars;
    WT alpha = (WT)scalars[0], beta = (WT)scalars[1], gamma = (WT)scalars[2];
    step1 /= sizeof(src1[0]);
    step2 /= sizeof(src2[0]);
    step /= sizeof(dst[0]);

    for( ; size.height--; src1 += step1, src2 += step2, dst += step )
    {
        int x = 0;
        #if CV_ENABLE_UNROLLED
        for( ; x <= size.width - 4; x += 4 )
        {
            T t0 = saturate_cast<T>(src1[x]*alpha + src2[x]*beta + gamma);
            T t1 = saturate_cast<T>(src1[x+1]*alpha + src2[x+1]*beta + gamma);
            dst[x] = t0; dst[x+1] = t1;

            t0 = saturate_cast<T>(src1[x+2]*alpha + src2[x+2]*beta + gamma);
            t1 = saturate_cast<T>(src1[x+3]*alpha + src2[x+3]*beta + gamma);
            dst[x+2] = t0; dst[x+3] = t1;
        }
        #endif
        for( ; x < size.width; x++ )
            dst[x] = saturate_cast<T>(src1[x]*alpha + src2[x]*beta + gamma);
    }
}


static void
addWeighted8u( const uchar* src1, size_t step1,
               const uchar* src2, size_t step2,
               uchar* dst, size_t step, Size size,
               void* _scalars )
{
    const double* scalars = (const double*)_scalars;
    float alpha = (float)scalars[0], beta = (float)scalars[1], gamma = (float)scalars[2];

    for( ; size.height--; src1 += step1, src2 += step2, dst += step )
    {
        int x = 0;

#if CV_SSE2
        if( USE_SSE2 )
        {
            __m128 a4 = _mm_set1_ps(alpha), b4 = _mm_set1_ps(beta), g4 = _mm_set1_ps(gamma);
            __m128i z = _mm_setzero_si128();

            for( ; x <= size.width - 8; x += 8 )
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
#endif
        #if CV_ENABLE_UNROLLED
        for( ; x <= size.width - 4; x += 4 )
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

        for( ; x < size.width; x++ )
        {
            float t0 = CV_8TO32F(src1[x])*alpha + CV_8TO32F(src2[x])*beta + gamma;
            dst[x] = saturate_cast<uchar>(t0);
        }
    }
}

static void addWeighted8s( const schar* src1, size_t step1, const schar* src2, size_t step2,
                           schar* dst, size_t step, Size sz, void* scalars )
{
    addWeighted_<schar, float>(src1, step1, src2, step2, dst, step, sz, scalars);
}

static void addWeighted16u( const ushort* src1, size_t step1, const ushort* src2, size_t step2,
                            ushort* dst, size_t step, Size sz, void* scalars )
{
    addWeighted_<ushort, float>(src1, step1, src2, step2, dst, step, sz, scalars);
}

static void addWeighted16s( const short* src1, size_t step1, const short* src2, size_t step2,
                            short* dst, size_t step, Size sz, void* scalars )
{
    addWeighted_<short, float>(src1, step1, src2, step2, dst, step, sz, scalars);
}

static void addWeighted32s( const int* src1, size_t step1, const int* src2, size_t step2,
                            int* dst, size_t step, Size sz, void* scalars )
{
    addWeighted_<int, double>(src1, step1, src2, step2, dst, step, sz, scalars);
}

static void addWeighted32f( const float* src1, size_t step1, const float* src2, size_t step2,
                            float* dst, size_t step, Size sz, void* scalars )
{
    addWeighted_<float, double>(src1, step1, src2, step2, dst, step, sz, scalars);
}

static void addWeighted64f( const double* src1, size_t step1, const double* src2, size_t step2,
                            double* dst, size_t step, Size sz, void* scalars )
{
    addWeighted_<double, double>(src1, step1, src2, step2, dst, step, sz, scalars);
}

static BinaryFunc* getAddWeightedTab()
{
    static BinaryFunc addWeightedTab[] =
    {
        (BinaryFunc)GET_OPTIMIZED(addWeighted8u), (BinaryFunc)GET_OPTIMIZED(addWeighted8s), (BinaryFunc)GET_OPTIMIZED(addWeighted16u),
        (BinaryFunc)GET_OPTIMIZED(addWeighted16s), (BinaryFunc)GET_OPTIMIZED(addWeighted32s), (BinaryFunc)addWeighted32f,
        (BinaryFunc)addWeighted64f, 0
    };

    return addWeightedTab;
}

}

void cv::addWeighted( InputArray src1, double alpha, InputArray src2,
                      double beta, double gamma, OutputArray dst, int dtype )
{
    double scalars[] = {alpha, beta, gamma};
    arithm_op(src1, src2, dst, noArray(), dtype, getAddWeightedTab(), true, scalars, OCL_OP_ADDW);
}


/****************************************************************************************\
*                                          compare                                       *
\****************************************************************************************/

namespace cv
{

template<typename T> static void
cmp_(const T* src1, size_t step1, const T* src2, size_t step2,
     uchar* dst, size_t step, Size size, int code)
{
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
        for( ; size.height--; src1 += step1, src2 += step2, dst += step )
        {
            int x = 0;
            #if CV_ENABLE_UNROLLED
            for( ; x <= size.width - 4; x += 4 )
            {
                int t0, t1;
                t0 = -(src1[x] > src2[x]) ^ m;
                t1 = -(src1[x+1] > src2[x+1]) ^ m;
                dst[x] = (uchar)t0; dst[x+1] = (uchar)t1;
                t0 = -(src1[x+2] > src2[x+2]) ^ m;
                t1 = -(src1[x+3] > src2[x+3]) ^ m;
                dst[x+2] = (uchar)t0; dst[x+3] = (uchar)t1;
            }
            #endif
            for( ; x < size.width; x++ )
                dst[x] = (uchar)(-(src1[x] > src2[x]) ^ m);
               }
    }
    else if( code == CMP_EQ || code == CMP_NE )
    {
        int m = code == CMP_EQ ? 0 : 255;
        for( ; size.height--; src1 += step1, src2 += step2, dst += step )
        {
            int x = 0;
            #if CV_ENABLE_UNROLLED
            for( ; x <= size.width - 4; x += 4 )
            {
                int t0, t1;
                t0 = -(src1[x] == src2[x]) ^ m;
                t1 = -(src1[x+1] == src2[x+1]) ^ m;
                dst[x] = (uchar)t0; dst[x+1] = (uchar)t1;
                t0 = -(src1[x+2] == src2[x+2]) ^ m;
                t1 = -(src1[x+3] == src2[x+3]) ^ m;
                dst[x+2] = (uchar)t0; dst[x+3] = (uchar)t1;
            }
            #endif
            for( ; x < size.width; x++ )
                dst[x] = (uchar)(-(src1[x] == src2[x]) ^ m);
        }
    }
}

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
#endif

static void cmp8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2,
                  uchar* dst, size_t step, Size size, void* _cmpop)
{
#if ARITHM_USE_IPP
    IppCmpOp op = convert_cmp(*(int *)_cmpop);
    if( op  >= 0 )
    {
        fixSteps(size, sizeof(dst[0]), step1, step2, step);
        if (0 <= ippiCompare_8u_C1R(src1, (int)step1, src2, (int)step2, dst, (int)step, ippiSize(size), op))
            return;
        setIppErrorStatus();
    }
#endif
  //vz optimized  cmp_(src1, step1, src2, step2, dst, step, size, *(int*)_cmpop);
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
        for( ; size.height--; src1 += step1, src2 += step2, dst += step )
        {
            int x =0;
            #if CV_SSE2
            if( USE_SSE2 ){
                __m128i m128 = code == CMP_GT ? _mm_setzero_si128() : _mm_set1_epi8 (-1);
                __m128i c128 = _mm_set1_epi8 (-128);
                for( ; x <= size.width - 16; x += 16 )
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
           #endif

            for( ; x < size.width; x++ ){
                dst[x] = (uchar)(-(src1[x] > src2[x]) ^ m);
            }
        }
    }
    else if( code == CMP_EQ || code == CMP_NE )
    {
        int m = code == CMP_EQ ? 0 : 255;
        for( ; size.height--; src1 += step1, src2 += step2, dst += step )
        {
            int x = 0;
            #if CV_SSE2
            if( USE_SSE2 ){
                __m128i m128 =  code == CMP_EQ ? _mm_setzero_si128() : _mm_set1_epi8 (-1);
                for( ; x <= size.width - 16; x += 16 )
                {
                    __m128i r00 = _mm_loadu_si128((const __m128i*)(src1 + x));
                    __m128i r10 = _mm_loadu_si128((const __m128i*)(src2 + x));
                    r00 = _mm_xor_si128 ( _mm_cmpeq_epi8 (r00, r10), m128);
                    _mm_storeu_si128((__m128i*)(dst + x), r00);
                }
            }
           #endif
           for( ; x < size.width; x++ )
                dst[x] = (uchar)(-(src1[x] == src2[x]) ^ m);
        }
    }
}

static void cmp8s(const schar* src1, size_t step1, const schar* src2, size_t step2,
                  uchar* dst, size_t step, Size size, void* _cmpop)
{
    cmp_(src1, step1, src2, step2, dst, step, size, *(int*)_cmpop);
}

static void cmp16u(const ushort* src1, size_t step1, const ushort* src2, size_t step2,
                  uchar* dst, size_t step, Size size, void* _cmpop)
{
#if ARITHM_USE_IPP
    IppCmpOp op = convert_cmp(*(int *)_cmpop);
    if( op  >= 0 )
    {
        fixSteps(size, sizeof(dst[0]), step1, step2, step);
        if (0 <= ippiCompare_16u_C1R(src1, (int)step1, src2, (int)step2, dst, (int)step, ippiSize(size), op))
            return;
        setIppErrorStatus();
    }
#endif
    cmp_(src1, step1, src2, step2, dst, step, size, *(int*)_cmpop);
}

static void cmp16s(const short* src1, size_t step1, const short* src2, size_t step2,
                  uchar* dst, size_t step, Size size, void* _cmpop)
{
#if ARITHM_USE_IPP
    IppCmpOp op = convert_cmp(*(int *)_cmpop);
    if( op  > 0 )
    {
        fixSteps(size, sizeof(dst[0]), step1, step2, step);
        if (0 <= ippiCompare_16s_C1R(src1, (int)step1, src2, (int)step2, dst, (int)step, ippiSize(size), op))
            return;
        setIppErrorStatus();
    }
#endif
   //vz optimized cmp_(src1, step1, src2, step2, dst, step, size, *(int*)_cmpop);

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
        for( ; size.height--; src1 += step1, src2 += step2, dst += step )
        {
            int x =0;
            #if CV_SSE2
            if( USE_SSE2){//
                __m128i m128 =  code == CMP_GT ? _mm_setzero_si128() : _mm_set1_epi16 (-1);
                for( ; x <= size.width - 16; x += 16 )
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
                if( x <= size.width-8)
                {
                    __m128i r00 = _mm_loadu_si128((const __m128i*)(src1 + x));
                    __m128i r10 = _mm_loadu_si128((const __m128i*)(src2 + x));
                    r00 = _mm_xor_si128 ( _mm_cmpgt_epi16 (r00, r10), m128);
                    r10 = _mm_packs_epi16(r00, r00);
                    _mm_storel_epi64((__m128i*)(dst + x), r10);

                    x += 8;
                }
            }
           #endif

            for( ; x < size.width; x++ ){
                 dst[x] = (uchar)(-(src1[x] > src2[x]) ^ m);
            }
        }
    }
    else if( code == CMP_EQ || code == CMP_NE )
    {
        int m = code == CMP_EQ ? 0 : 255;
        for( ; size.height--; src1 += step1, src2 += step2, dst += step )
        {
            int x = 0;
            #if CV_SSE2
            if( USE_SSE2 ){
                __m128i m128 =  code == CMP_EQ ? _mm_setzero_si128() : _mm_set1_epi16 (-1);
                for( ; x <= size.width - 16; x += 16 )
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
                if( x <= size.width - 8)
                {
                    __m128i r00 = _mm_loadu_si128((const __m128i*)(src1 + x));
                    __m128i r10 = _mm_loadu_si128((const __m128i*)(src2 + x));
                    r00 = _mm_xor_si128 ( _mm_cmpeq_epi16 (r00, r10), m128);
                    r10 = _mm_packs_epi16(r00, r00);
                    _mm_storel_epi64((__m128i*)(dst + x), r10);

                    x += 8;
                }
            }
           #endif
           for( ; x < size.width; x++ )
                dst[x] = (uchar)(-(src1[x] == src2[x]) ^ m);
        }
    }
}

static void cmp32s(const int* src1, size_t step1, const int* src2, size_t step2,
                   uchar* dst, size_t step, Size size, void* _cmpop)
{
    cmp_(src1, step1, src2, step2, dst, step, size, *(int*)_cmpop);
}

static void cmp32f(const float* src1, size_t step1, const float* src2, size_t step2,
                  uchar* dst, size_t step, Size size, void* _cmpop)
{
#if ARITHM_USE_IPP
    IppCmpOp op = convert_cmp(*(int *)_cmpop);
    if( op  >= 0 )
    {
        fixSteps(size, sizeof(dst[0]), step1, step2, step);
        if (0 <= ippiCompare_32f_C1R(src1, (int)step1, src2, (int)step2, dst, (int)step, ippiSize(size), op))
            return;
        setIppErrorStatus();
    }
#endif
    cmp_(src1, step1, src2, step2, dst, step, size, *(int*)_cmpop);
}

static void cmp64f(const double* src1, size_t step1, const double* src2, size_t step2,
                  uchar* dst, size_t step, Size size, void* _cmpop)
{
    cmp_(src1, step1, src2, step2, dst, step, size, *(int*)_cmpop);
}

static BinaryFunc getCmpFunc(int depth)
{
    static BinaryFunc cmpTab[] =
    {
        (BinaryFunc)GET_OPTIMIZED(cmp8u), (BinaryFunc)GET_OPTIMIZED(cmp8s),
        (BinaryFunc)GET_OPTIMIZED(cmp16u), (BinaryFunc)GET_OPTIMIZED(cmp16s),
        (BinaryFunc)GET_OPTIMIZED(cmp32s),
        (BinaryFunc)GET_OPTIMIZED(cmp32f), (BinaryFunc)cmp64f,
        0
    };

    return cmpTab[depth];
}

static double getMinVal(int depth)
{
    static const double tab[] = {0, -128, 0, -32768, INT_MIN, -FLT_MAX, -DBL_MAX, 0};
    return tab[depth];
}

static double getMaxVal(int depth)
{
    static const double tab[] = {255, 127, 65535, 32767, INT_MAX, FLT_MAX, DBL_MAX, 0};
    return tab[depth];
}

#ifdef HAVE_OPENCL

static bool ocl_compare(InputArray _src1, InputArray _src2, OutputArray _dst, int op, bool haveScalar)
{
    const ocl::Device& dev = ocl::Device::getDefault();
    bool doubleSupport = dev.doubleFPConfig() > 0;
    int type1 = _src1.type(), depth1 = CV_MAT_DEPTH(type1), cn = CV_MAT_CN(type1),
            type2 = _src2.type(), depth2 = CV_MAT_DEPTH(type2);

    if (!doubleSupport && depth1 == CV_64F)
        return false;

    if (!haveScalar && (!_src1.sameSize(_src2) || type1 != type2))
            return false;

    int kercn = haveScalar ? cn : ocl::predictOptimalVectorWidth(_src1, _src2, _dst), rowsPerWI = dev.isIntel() ? 4 : 1;
    // Workaround for bug with "?:" operator in AMD OpenCL compiler
    if (depth1 >= CV_16U)
        kercn = 1;

    int scalarcn = kercn == 3 ? 4 : kercn;
    const char * const operationMap[] = { "==", ">", ">=", "<", "<=", "!=" };
    char cvt[40];

    String opts = format("-D %s -D srcT1=%s -D dstT=%s -D workT=srcT1 -D cn=%d"
                         " -D convertToDT=%s -D OP_CMP -D CMP_OPERATOR=%s -D srcT1_C1=%s"
                         " -D srcT2_C1=%s -D dstT_C1=%s -D workST=%s -D rowsPerWI=%d%s",
                         haveScalar ? "UNARY_OP" : "BINARY_OP",
                         ocl::typeToStr(CV_MAKE_TYPE(depth1, kercn)),
                         ocl::typeToStr(CV_8UC(kercn)), kercn,
                         ocl::convertTypeStr(depth1, CV_8U, kercn, cvt),
                         operationMap[op], ocl::typeToStr(depth1),
                         ocl::typeToStr(depth1), ocl::typeToStr(CV_8U),
                         ocl::typeToStr(CV_MAKE_TYPE(depth1, scalarcn)), rowsPerWI,
                         doubleSupport ? " -D DOUBLE_SUPPORT" : "");

    ocl::Kernel k("KF", ocl::core::arithm_oclsrc, opts);
    if (k.empty())
        return false;

    UMat src1 = _src1.getUMat();
    Size size = src1.size();
    _dst.create(size, CV_8UC(cn));
    UMat dst = _dst.getUMat();

    if (haveScalar)
    {
        size_t esz = CV_ELEM_SIZE1(type1) * scalarcn;
        double buf[4] = { 0, 0, 0, 0 };
        Mat src2 = _src2.getMat();

        if( depth1 > CV_32S )
            convertAndUnrollScalar( src2, depth1, (uchar *)buf, kercn );
        else
        {
            double fval = 0;
            getConvertFunc(depth2, CV_64F)(src2.data, 1, 0, 1, (uchar *)&fval, 1, Size(1, 1), 0);
            if( fval < getMinVal(depth1) )
                return dst.setTo(Scalar::all(op == CMP_GT || op == CMP_GE || op == CMP_NE ? 255 : 0)), true;

            if( fval > getMaxVal(depth1) )
                return dst.setTo(Scalar::all(op == CMP_LT || op == CMP_LE || op == CMP_NE ? 255 : 0)), true;

            int ival = cvRound(fval);
            if( fval != ival )
            {
                if( op == CMP_LT || op == CMP_GE )
                    ival = cvCeil(fval);
                else if( op == CMP_LE || op == CMP_GT )
                    ival = cvFloor(fval);
                else
                    return dst.setTo(Scalar::all(op == CMP_NE ? 255 : 0)), true;
            }
            convertAndUnrollScalar(Mat(1, 1, CV_32S, &ival), depth1, (uchar *)buf, kercn);
        }

        ocl::KernelArg scalararg = ocl::KernelArg(0, 0, 0, 0, buf, esz);

        k.args(ocl::KernelArg::ReadOnlyNoSize(src1, cn, kercn),
               ocl::KernelArg::WriteOnly(dst, cn, kercn), scalararg);
    }
    else
    {
        UMat src2 = _src2.getUMat();

        k.args(ocl::KernelArg::ReadOnlyNoSize(src1),
               ocl::KernelArg::ReadOnlyNoSize(src2),
               ocl::KernelArg::WriteOnly(dst, cn, kercn));
    }

    size_t globalsize[2] = { dst.cols * cn / kercn, (dst.rows + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, NULL, false);
}

#endif

}

void cv::compare(InputArray _src1, InputArray _src2, OutputArray _dst, int op)
{
    CV_Assert( op == CMP_LT || op == CMP_LE || op == CMP_EQ ||
               op == CMP_NE || op == CMP_GE || op == CMP_GT );

    bool haveScalar = false;

    if ((_src1.isMatx() + _src2.isMatx()) == 1
            || !_src1.sameSize(_src2)
            || _src1.type() != _src2.type())
    {
        if (checkScalar(_src1, _src2.type(), _src1.kind(), _src2.kind()))
        {
            op = op == CMP_LT ? CMP_GT : op == CMP_LE ? CMP_GE :
                op == CMP_GE ? CMP_LE : op == CMP_GT ? CMP_LT : op;
            // src1 is a scalar; swap it with src2
            compare(_src2, _src1, _dst, op);
            return;
        }
        else if( !checkScalar(_src2, _src1.type(), _src2.kind(), _src1.kind()) )
            CV_Error( CV_StsUnmatchedSizes,
                     "The operation is neither 'array op array' (where arrays have the same size and the same type), "
                     "nor 'array op scalar', nor 'scalar op array'" );
        haveScalar = true;
    }

    CV_OCL_RUN(_src1.dims() <= 2 && _src2.dims() <= 2 && _dst.isUMat(),
               ocl_compare(_src1, _src2, _dst, op, haveScalar))

    int kind1 = _src1.kind(), kind2 = _src2.kind();
    Mat src1 = _src1.getMat(), src2 = _src2.getMat();

    if( kind1 == kind2 && src1.dims <= 2 && src2.dims <= 2 && src1.size() == src2.size() && src1.type() == src2.type() )
    {
        int cn = src1.channels();
        _dst.create(src1.size(), CV_8UC(cn));
        Mat dst = _dst.getMat();
        Size sz = getContinuousSize(src1, src2, dst, src1.channels());
        getCmpFunc(src1.depth())(src1.data, src1.step, src2.data, src2.step, dst.data, dst.step, sz, &op);
        return;
    }

    int cn = src1.channels(), depth1 = src1.depth(), depth2 = src2.depth();

    _dst.create(src1.dims, src1.size, CV_8UC(cn));
    src1 = src1.reshape(1); src2 = src2.reshape(1);
    Mat dst = _dst.getMat().reshape(1);

    size_t esz = src1.elemSize();
    size_t blocksize0 = (size_t)(BLOCK_SIZE + esz-1)/esz;
    BinaryFunc func = getCmpFunc(depth1);

    if( !haveScalar )
    {
        const Mat* arrays[] = { &src1, &src2, &dst, 0 };
        uchar* ptrs[3];

        NAryMatIterator it(arrays, ptrs);
        size_t total = it.size;

        for( size_t i = 0; i < it.nplanes; i++, ++it )
            func( ptrs[0], 0, ptrs[1], 0, ptrs[2], 0, Size((int)total, 1), &op );
    }
    else
    {
        const Mat* arrays[] = { &src1, &dst, 0 };
        uchar* ptrs[2];

        NAryMatIterator it(arrays, ptrs);
        size_t total = it.size, blocksize = std::min(total, blocksize0);

        AutoBuffer<uchar> _buf(blocksize*esz);
        uchar *buf = _buf;

        if( depth1 > CV_32S )
            convertAndUnrollScalar( src2, depth1, buf, blocksize );
        else
        {
            double fval=0;
            getConvertFunc(depth2, CV_64F)(src2.data, 1, 0, 1, (uchar*)&fval, 1, Size(1,1), 0);
            if( fval < getMinVal(depth1) )
            {
                dst = Scalar::all(op == CMP_GT || op == CMP_GE || op == CMP_NE ? 255 : 0);
                return;
            }

            if( fval > getMaxVal(depth1) )
            {
                dst = Scalar::all(op == CMP_LT || op == CMP_LE || op == CMP_NE ? 255 : 0);
                return;
            }

            int ival = cvRound(fval);
            if( fval != ival )
            {
                if( op == CMP_LT || op == CMP_GE )
                    ival = cvCeil(fval);
                else if( op == CMP_LE || op == CMP_GT )
                    ival = cvFloor(fval);
                else
                {
                    dst = Scalar::all(op == CMP_NE ? 255 : 0);
                    return;
                }
            }
            convertAndUnrollScalar(Mat(1, 1, CV_32S, &ival), depth1, buf, blocksize);
        }

        for( size_t i = 0; i < it.nplanes; i++, ++it )
        {
            for( size_t j = 0; j < total; j += blocksize )
            {
                int bsz = (int)MIN(total - j, blocksize);
                func( ptrs[0], 0, buf, 0, ptrs[1], 0, Size(bsz, 1), &op);
                ptrs[0] += bsz*esz;
                ptrs[1] += bsz;
            }
        }
    }
}

/****************************************************************************************\
*                                        inRange                                         *
\****************************************************************************************/

namespace cv
{

template<typename T> static void
inRange_(const T* src1, size_t step1, const T* src2, size_t step2,
         const T* src3, size_t step3, uchar* dst, size_t step,
         Size size)
{
    step1 /= sizeof(src1[0]);
    step2 /= sizeof(src2[0]);
    step3 /= sizeof(src3[0]);

    for( ; size.height--; src1 += step1, src2 += step2, src3 += step3, dst += step )
    {
        int x = 0;
        #if CV_ENABLE_UNROLLED
        for( ; x <= size.width - 4; x += 4 )
        {
            int t0, t1;
            t0 = src2[x] <= src1[x] && src1[x] <= src3[x];
            t1 = src2[x+1] <= src1[x+1] && src1[x+1] <= src3[x+1];
            dst[x] = (uchar)-t0; dst[x+1] = (uchar)-t1;
            t0 = src2[x+2] <= src1[x+2] && src1[x+2] <= src3[x+2];
            t1 = src2[x+3] <= src1[x+3] && src1[x+3] <= src3[x+3];
            dst[x+2] = (uchar)-t0; dst[x+3] = (uchar)-t1;
        }
        #endif
        for( ; x < size.width; x++ )
            dst[x] = (uchar)-(src2[x] <= src1[x] && src1[x] <= src3[x]);
    }
}


static void inRange8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2,
                      const uchar* src3, size_t step3, uchar* dst, size_t step, Size size)
{
    inRange_(src1, step1, src2, step2, src3, step3, dst, step, size);
}

static void inRange8s(const schar* src1, size_t step1, const schar* src2, size_t step2,
                      const schar* src3, size_t step3, uchar* dst, size_t step, Size size)
{
    inRange_(src1, step1, src2, step2, src3, step3, dst, step, size);
}

static void inRange16u(const ushort* src1, size_t step1, const ushort* src2, size_t step2,
                       const ushort* src3, size_t step3, uchar* dst, size_t step, Size size)
{
    inRange_(src1, step1, src2, step2, src3, step3, dst, step, size);
}

static void inRange16s(const short* src1, size_t step1, const short* src2, size_t step2,
                       const short* src3, size_t step3, uchar* dst, size_t step, Size size)
{
    inRange_(src1, step1, src2, step2, src3, step3, dst, step, size);
}

static void inRange32s(const int* src1, size_t step1, const int* src2, size_t step2,
                       const int* src3, size_t step3, uchar* dst, size_t step, Size size)
{
    inRange_(src1, step1, src2, step2, src3, step3, dst, step, size);
}

static void inRange32f(const float* src1, size_t step1, const float* src2, size_t step2,
                       const float* src3, size_t step3, uchar* dst, size_t step, Size size)
{
    inRange_(src1, step1, src2, step2, src3, step3, dst, step, size);
}

static void inRange64f(const double* src1, size_t step1, const double* src2, size_t step2,
                       const double* src3, size_t step3, uchar* dst, size_t step, Size size)
{
    inRange_(src1, step1, src2, step2, src3, step3, dst, step, size);
}

static void inRangeReduce(const uchar* src, uchar* dst, size_t len, int cn)
{
    int k = cn % 4 ? cn % 4 : 4;
    size_t i, j;
    if( k == 1 )
        for( i = j = 0; i < len; i++, j += cn )
            dst[i] = src[j];
    else if( k == 2 )
        for( i = j = 0; i < len; i++, j += cn )
            dst[i] = src[j] & src[j+1];
    else if( k == 3 )
        for( i = j = 0; i < len; i++, j += cn )
            dst[i] = src[j] & src[j+1] & src[j+2];
    else
        for( i = j = 0; i < len; i++, j += cn )
            dst[i] = src[j] & src[j+1] & src[j+2] & src[j+3];

    for( ; k < cn; k += 4 )
    {
        for( i = 0, j = k; i < len; i++, j += cn )
            dst[i] &= src[j] & src[j+1] & src[j+2] & src[j+3];
    }
}

typedef void (*InRangeFunc)( const uchar* src1, size_t step1, const uchar* src2, size_t step2,
                             const uchar* src3, size_t step3, uchar* dst, size_t step, Size sz );

static InRangeFunc getInRangeFunc(int depth)
{
    static InRangeFunc inRangeTab[] =
    {
        (InRangeFunc)GET_OPTIMIZED(inRange8u), (InRangeFunc)GET_OPTIMIZED(inRange8s), (InRangeFunc)GET_OPTIMIZED(inRange16u),
        (InRangeFunc)GET_OPTIMIZED(inRange16s), (InRangeFunc)GET_OPTIMIZED(inRange32s), (InRangeFunc)GET_OPTIMIZED(inRange32f),
        (InRangeFunc)inRange64f, 0
    };

    return inRangeTab[depth];
}

#ifdef HAVE_OPENCL

static bool ocl_inRange( InputArray _src, InputArray _lowerb,
                         InputArray _upperb, OutputArray _dst )
{
    const ocl::Device & d = ocl::Device::getDefault();
    int skind = _src.kind(), lkind = _lowerb.kind(), ukind = _upperb.kind();
    Size ssize = _src.size(), lsize = _lowerb.size(), usize = _upperb.size();
    int stype = _src.type(), ltype = _lowerb.type(), utype = _upperb.type();
    int sdepth = CV_MAT_DEPTH(stype), ldepth = CV_MAT_DEPTH(ltype), udepth = CV_MAT_DEPTH(utype);
    int cn = CV_MAT_CN(stype), rowsPerWI = d.isIntel() ? 4 : 1;
    bool lbScalar = false, ubScalar = false;

    if( (lkind == _InputArray::MATX && skind != _InputArray::MATX) ||
        ssize != lsize || stype != ltype )
    {
        if( !checkScalar(_lowerb, stype, lkind, skind) )
            CV_Error( CV_StsUnmatchedSizes,
                     "The lower bounary is neither an array of the same size and same type as src, nor a scalar");
        lbScalar = true;
    }

    if( (ukind == _InputArray::MATX && skind != _InputArray::MATX) ||
        ssize != usize || stype != utype )
    {
        if( !checkScalar(_upperb, stype, ukind, skind) )
            CV_Error( CV_StsUnmatchedSizes,
                     "The upper bounary is neither an array of the same size and same type as src, nor a scalar");
        ubScalar = true;
    }

    if (lbScalar != ubScalar)
        return false;

    bool doubleSupport = d.doubleFPConfig() > 0,
            haveScalar = lbScalar && ubScalar;

    if ( (!doubleSupport && sdepth == CV_64F) ||
         (!haveScalar && (sdepth != ldepth || sdepth != udepth)) )
        return false;

    ocl::Kernel ker("inrange", ocl::core::inrange_oclsrc,
                    format("%s-D cn=%d -D T=%s%s", haveScalar ? "-D HAVE_SCALAR " : "",
                           cn, ocl::typeToStr(sdepth), doubleSupport ? " -D DOUBLE_SUPPORT" : ""));
    if (ker.empty())
        return false;

    _dst.create(ssize, CV_8UC1);
    UMat src = _src.getUMat(), dst = _dst.getUMat(), lscalaru, uscalaru;
    Mat lscalar, uscalar;

    if (lbScalar && ubScalar)
    {
        lscalar = _lowerb.getMat();
        uscalar = _upperb.getMat();

        size_t esz = src.elemSize();
        size_t blocksize = 36;

        AutoBuffer<uchar> _buf(blocksize*(((int)lbScalar + (int)ubScalar)*esz + cn) + 2*cn*sizeof(int) + 128);
        uchar *buf = alignPtr(_buf + blocksize*cn, 16);

        if( ldepth != sdepth && sdepth < CV_32S )
        {
            int* ilbuf = (int*)alignPtr(buf + blocksize*esz, 16);
            int* iubuf = ilbuf + cn;

            BinaryFunc sccvtfunc = getConvertFunc(ldepth, CV_32S);
            sccvtfunc(lscalar.data, 1, 0, 1, (uchar*)ilbuf, 1, Size(cn, 1), 0);
            sccvtfunc(uscalar.data, 1, 0, 1, (uchar*)iubuf, 1, Size(cn, 1), 0);
            int minval = cvRound(getMinVal(sdepth)), maxval = cvRound(getMaxVal(sdepth));

            for( int k = 0; k < cn; k++ )
            {
                if( ilbuf[k] > iubuf[k] || ilbuf[k] > maxval || iubuf[k] < minval )
                    ilbuf[k] = minval+1, iubuf[k] = minval;
            }
            lscalar = Mat(cn, 1, CV_32S, ilbuf);
            uscalar = Mat(cn, 1, CV_32S, iubuf);
        }

        lscalar.convertTo(lscalar, stype);
        uscalar.convertTo(uscalar, stype);
    }
    else
    {
        lscalaru = _lowerb.getUMat();
        uscalaru = _upperb.getUMat();
    }

    ocl::KernelArg srcarg = ocl::KernelArg::ReadOnlyNoSize(src),
            dstarg = ocl::KernelArg::WriteOnly(dst);

    if (haveScalar)
    {
        lscalar.copyTo(lscalaru);
        uscalar.copyTo(uscalaru);

        ker.args(srcarg, dstarg, ocl::KernelArg::PtrReadOnly(lscalaru),
               ocl::KernelArg::PtrReadOnly(uscalaru), rowsPerWI);
    }
    else
        ker.args(srcarg, dstarg, ocl::KernelArg::ReadOnlyNoSize(lscalaru),
               ocl::KernelArg::ReadOnlyNoSize(uscalaru), rowsPerWI);

    size_t globalsize[2] = { ssize.width, (ssize.height + rowsPerWI - 1) / rowsPerWI };
    return ker.run(2, globalsize, NULL, false);
}

#endif

}

void cv::inRange(InputArray _src, InputArray _lowerb,
                 InputArray _upperb, OutputArray _dst)
{
    CV_OCL_RUN(_src.dims() <= 2 && _lowerb.dims() <= 2 &&
               _upperb.dims() <= 2 && _dst.isUMat(),
               ocl_inRange(_src, _lowerb, _upperb, _dst))

    int skind = _src.kind(), lkind = _lowerb.kind(), ukind = _upperb.kind();
    Mat src = _src.getMat(), lb = _lowerb.getMat(), ub = _upperb.getMat();

    bool lbScalar = false, ubScalar = false;

    if( (lkind == _InputArray::MATX && skind != _InputArray::MATX) ||
        src.size != lb.size || src.type() != lb.type() )
    {
        if( !checkScalar(lb, src.type(), lkind, skind) )
            CV_Error( CV_StsUnmatchedSizes,
                     "The lower bounary is neither an array of the same size and same type as src, nor a scalar");
        lbScalar = true;
    }

    if( (ukind == _InputArray::MATX && skind != _InputArray::MATX) ||
        src.size != ub.size || src.type() != ub.type() )
    {
        if( !checkScalar(ub, src.type(), ukind, skind) )
            CV_Error( CV_StsUnmatchedSizes,
                     "The upper bounary is neither an array of the same size and same type as src, nor a scalar");
        ubScalar = true;
    }

    CV_Assert(lbScalar == ubScalar);

    int cn = src.channels(), depth = src.depth();

    size_t esz = src.elemSize();
    size_t blocksize0 = (size_t)(BLOCK_SIZE + esz-1)/esz;

    _dst.create(src.dims, src.size, CV_8UC1);
    Mat dst = _dst.getMat();
    InRangeFunc func = getInRangeFunc(depth);

    const Mat* arrays_sc[] = { &src, &dst, 0 };
    const Mat* arrays_nosc[] = { &src, &dst, &lb, &ub, 0 };
    uchar* ptrs[4];

    NAryMatIterator it(lbScalar && ubScalar ? arrays_sc : arrays_nosc, ptrs);
    size_t total = it.size, blocksize = std::min(total, blocksize0);

    AutoBuffer<uchar> _buf(blocksize*(((int)lbScalar + (int)ubScalar)*esz + cn) + 2*cn*sizeof(int) + 128);
    uchar *buf = _buf, *mbuf = buf, *lbuf = 0, *ubuf = 0;
    buf = alignPtr(buf + blocksize*cn, 16);

    if( lbScalar && ubScalar )
    {
        lbuf = buf;
        ubuf = buf = alignPtr(buf + blocksize*esz, 16);

        CV_Assert( lb.type() == ub.type() );
        int scdepth = lb.depth();

        if( scdepth != depth && depth < CV_32S )
        {
            int* ilbuf = (int*)alignPtr(buf + blocksize*esz, 16);
            int* iubuf = ilbuf + cn;

            BinaryFunc sccvtfunc = getConvertFunc(scdepth, CV_32S);
            sccvtfunc(lb.data, 1, 0, 1, (uchar*)ilbuf, 1, Size(cn, 1), 0);
            sccvtfunc(ub.data, 1, 0, 1, (uchar*)iubuf, 1, Size(cn, 1), 0);
            int minval = cvRound(getMinVal(depth)), maxval = cvRound(getMaxVal(depth));

            for( int k = 0; k < cn; k++ )
            {
                if( ilbuf[k] > iubuf[k] || ilbuf[k] > maxval || iubuf[k] < minval )
                    ilbuf[k] = minval+1, iubuf[k] = minval;
            }
            lb = Mat(cn, 1, CV_32S, ilbuf);
            ub = Mat(cn, 1, CV_32S, iubuf);
        }

        convertAndUnrollScalar( lb, src.type(), lbuf, blocksize );
        convertAndUnrollScalar( ub, src.type(), ubuf, blocksize );
    }

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        for( size_t j = 0; j < total; j += blocksize )
        {
            int bsz = (int)MIN(total - j, blocksize);
            size_t delta = bsz*esz;
            uchar *lptr = lbuf, *uptr = ubuf;
            if( !lbScalar )
            {
                lptr = ptrs[2];
                ptrs[2] += delta;
            }
            if( !ubScalar )
            {
                int idx = !lbScalar ? 3 : 2;
                uptr = ptrs[idx];
                ptrs[idx] += delta;
            }
            func( ptrs[0], 0, lptr, 0, uptr, 0, cn == 1 ? ptrs[1] : mbuf, 0, Size(bsz*cn, 1));
            if( cn > 1 )
                inRangeReduce(mbuf, ptrs[1], bsz, cn);
            ptrs[0] += delta;
            ptrs[1] += bsz;
        }
    }
}

/****************************************************************************************\
*                                Earlier API: cvAdd etc.                                 *
\****************************************************************************************/

CV_IMPL void
cvNot( const CvArr* srcarr, CvArr* dstarr )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src.size == dst.size && src.type() == dst.type() );
    cv::bitwise_not( src, dst );
}


CV_IMPL void
cvAnd( const CvArr* srcarr1, const CvArr* srcarr2, CvArr* dstarr, const CvArr* maskarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), src2 = cv::cvarrToMat(srcarr2),
        dst = cv::cvarrToMat(dstarr), mask;
    CV_Assert( src1.size == dst.size && src1.type() == dst.type() );
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::bitwise_and( src1, src2, dst, mask );
}


CV_IMPL void
cvOr( const CvArr* srcarr1, const CvArr* srcarr2, CvArr* dstarr, const CvArr* maskarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), src2 = cv::cvarrToMat(srcarr2),
        dst = cv::cvarrToMat(dstarr), mask;
    CV_Assert( src1.size == dst.size && src1.type() == dst.type() );
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::bitwise_or( src1, src2, dst, mask );
}


CV_IMPL void
cvXor( const CvArr* srcarr1, const CvArr* srcarr2, CvArr* dstarr, const CvArr* maskarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), src2 = cv::cvarrToMat(srcarr2),
        dst = cv::cvarrToMat(dstarr), mask;
    CV_Assert( src1.size == dst.size && src1.type() == dst.type() );
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::bitwise_xor( src1, src2, dst, mask );
}


CV_IMPL void
cvAndS( const CvArr* srcarr, CvScalar s, CvArr* dstarr, const CvArr* maskarr )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr), mask;
    CV_Assert( src.size == dst.size && src.type() == dst.type() );
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::bitwise_and( src, (const cv::Scalar&)s, dst, mask );
}


CV_IMPL void
cvOrS( const CvArr* srcarr, CvScalar s, CvArr* dstarr, const CvArr* maskarr )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr), mask;
    CV_Assert( src.size == dst.size && src.type() == dst.type() );
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::bitwise_or( src, (const cv::Scalar&)s, dst, mask );
}


CV_IMPL void
cvXorS( const CvArr* srcarr, CvScalar s, CvArr* dstarr, const CvArr* maskarr )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr), mask;
    CV_Assert( src.size == dst.size && src.type() == dst.type() );
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::bitwise_xor( src, (const cv::Scalar&)s, dst, mask );
}


CV_IMPL void cvAdd( const CvArr* srcarr1, const CvArr* srcarr2, CvArr* dstarr, const CvArr* maskarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), src2 = cv::cvarrToMat(srcarr2),
        dst = cv::cvarrToMat(dstarr), mask;
    CV_Assert( src1.size == dst.size && src1.channels() == dst.channels() );
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::add( src1, src2, dst, mask, dst.type() );
}


CV_IMPL void cvSub( const CvArr* srcarr1, const CvArr* srcarr2, CvArr* dstarr, const CvArr* maskarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), src2 = cv::cvarrToMat(srcarr2),
        dst = cv::cvarrToMat(dstarr), mask;
    CV_Assert( src1.size == dst.size && src1.channels() == dst.channels() );
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::subtract( src1, src2, dst, mask, dst.type() );
}


CV_IMPL void cvAddS( const CvArr* srcarr1, CvScalar value, CvArr* dstarr, const CvArr* maskarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1),
        dst = cv::cvarrToMat(dstarr), mask;
    CV_Assert( src1.size == dst.size && src1.channels() == dst.channels() );
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::add( src1, (const cv::Scalar&)value, dst, mask, dst.type() );
}


CV_IMPL void cvSubRS( const CvArr* srcarr1, CvScalar value, CvArr* dstarr, const CvArr* maskarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1),
        dst = cv::cvarrToMat(dstarr), mask;
    CV_Assert( src1.size == dst.size && src1.channels() == dst.channels() );
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::subtract( (const cv::Scalar&)value, src1, dst, mask, dst.type() );
}


CV_IMPL void cvMul( const CvArr* srcarr1, const CvArr* srcarr2,
                    CvArr* dstarr, double scale )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), src2 = cv::cvarrToMat(srcarr2),
        dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && src1.channels() == dst.channels() );
    cv::multiply( src1, src2, dst, scale, dst.type() );
}


CV_IMPL void cvDiv( const CvArr* srcarr1, const CvArr* srcarr2,
                    CvArr* dstarr, double scale )
{
    cv::Mat src2 = cv::cvarrToMat(srcarr2),
        dst = cv::cvarrToMat(dstarr), mask;
    CV_Assert( src2.size == dst.size && src2.channels() == dst.channels() );

    if( srcarr1 )
        cv::divide( cv::cvarrToMat(srcarr1), src2, dst, scale, dst.type() );
    else
        cv::divide( scale, src2, dst, dst.type() );
}


CV_IMPL void
cvAddWeighted( const CvArr* srcarr1, double alpha,
               const CvArr* srcarr2, double beta,
               double gamma, CvArr* dstarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), src2 = cv::cvarrToMat(srcarr2),
        dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && src1.channels() == dst.channels() );
    cv::addWeighted( src1, alpha, src2, beta, gamma, dst, dst.type() );
}


CV_IMPL  void
cvAbsDiff( const CvArr* srcarr1, const CvArr* srcarr2, CvArr* dstarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && src1.type() == dst.type() );

    cv::absdiff( src1, cv::cvarrToMat(srcarr2), dst );
}


CV_IMPL void
cvAbsDiffS( const CvArr* srcarr1, CvArr* dstarr, CvScalar scalar )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && src1.type() == dst.type() );

    cv::absdiff( src1, (const cv::Scalar&)scalar, dst );
}


CV_IMPL void
cvInRange( const void* srcarr1, const void* srcarr2,
           const void* srcarr3, void* dstarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && dst.type() == CV_8U );

    cv::inRange( src1, cv::cvarrToMat(srcarr2), cv::cvarrToMat(srcarr3), dst );
}


CV_IMPL void
cvInRangeS( const void* srcarr1, CvScalar lowerb, CvScalar upperb, void* dstarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && dst.type() == CV_8U );

    cv::inRange( src1, (const cv::Scalar&)lowerb, (const cv::Scalar&)upperb, dst );
}


CV_IMPL void
cvCmp( const void* srcarr1, const void* srcarr2, void* dstarr, int cmp_op )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && dst.type() == CV_8U );

    cv::compare( src1, cv::cvarrToMat(srcarr2), dst, cmp_op );
}


CV_IMPL void
cvCmpS( const void* srcarr1, double value, void* dstarr, int cmp_op )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && dst.type() == CV_8U );

    cv::compare( src1, value, dst, cmp_op );
}


CV_IMPL void
cvMin( const void* srcarr1, const void* srcarr2, void* dstarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && src1.type() == dst.type() );

    cv::min( src1, cv::cvarrToMat(srcarr2), dst );
}


CV_IMPL void
cvMax( const void* srcarr1, const void* srcarr2, void* dstarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && src1.type() == dst.type() );

    cv::max( src1, cv::cvarrToMat(srcarr2), dst );
}


CV_IMPL void
cvMinS( const void* srcarr1, double value, void* dstarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && src1.type() == dst.type() );

    cv::min( src1, value, dst );
}


CV_IMPL void
cvMaxS( const void* srcarr1, double value, void* dstarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && src1.type() == dst.type() );

    cv::max( src1, value, dst );
}

/* End of file. */
