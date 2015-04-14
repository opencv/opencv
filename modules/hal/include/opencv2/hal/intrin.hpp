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

#ifndef __OPENCV_HAL_INTRIN_HPP__
#define __OPENCV_HAL_INTRIN_HPP__

#include <algorithm>
#include <cmath>
#include <stdlib.h>

#define OPENCV_HAL_ADD(a, b) ((a) + (b))
#define OPENCV_HAL_AND(a, b) ((a) & (b))
#define OPENCV_HAL_NOP(a) (a)
#define OPENCV_HAL_1ST(a, b) (a)

namespace cv { namespace hal {

template<typename _Tp> struct TypeTraits
{
    typedef _Tp int_type;
    typedef _Tp uint_type;
    typedef _Tp abs_type;
    typedef _Tp sum_type;

    enum { delta = 0, shift = 0 };

    static int_type reinterpret_int(_Tp x) { return x; }
    static uint_type reinterpet_uint(_Tp x) { return x; }
    static _Tp reinterpret_from_int(int_type x) { return (_Tp)x; }
};

template<> struct TypeTraits<uchar>
{
    typedef uchar value_type;
    typedef schar int_type;
    typedef uchar uint_type;
    typedef uchar abs_type;
    typedef int sum_type;

    typedef ushort w_type;

    enum { delta = 128, shift = 8 };

    static int_type reinterpret_int(value_type x) { return (int_type)x; }
    static uint_type reinterpret_uint(value_type x) { return (uint_type)x; }
    static value_type reinterpret_from_int(int_type x) { return (value_type)x; }
};

template<> struct TypeTraits<schar>
{
    typedef schar value_type;
    typedef schar int_type;
    typedef uchar uint_type;
    typedef uchar abs_type;
    typedef int sum_type;

    typedef short w_type;

    enum { delta = 128, shift = 8 };

    static int_type reinterpret_int(value_type x) { return (int_type)x; }
    static uint_type reinterpret_uint(value_type x) { return (uint_type)x; }
    static value_type reinterpret_from_int(int_type x) { return (value_type)x; }
};

template<> struct TypeTraits<ushort>
{
    typedef ushort value_type;
    typedef short int_type;
    typedef ushort uint_type;
    typedef ushort abs_type;
    typedef int sum_type;

    typedef unsigned w_type;
    typedef uchar nu_type;

    enum { delta = 32768, shift = 16 };

    static int_type reinterpret_int(value_type x) { return (int_type)x; }
    static uint_type reinterpret_uint(value_type x) { return (uint_type)x; }
    static value_type reinterpret_from_int(int_type x) { return (value_type)x; }
};

template<> struct TypeTraits<short>
{
    typedef short value_type;
    typedef short int_type;
    typedef ushort uint_type;
    typedef ushort abs_type;
    typedef int sum_type;

    typedef int w_type;
    typedef uchar nu_type;
    typedef schar n_type;

    enum { delta = 128, shift = 8 };

    static int_type reinterpret_int(value_type x) { return (int_type)x; }
    static uint_type reinterpret_uint(value_type x) { return (uint_type)x; }
    static value_type reinterpret_from_int(int_type x) { return (value_type)x; }
};

template<> struct TypeTraits<unsigned>
{
    typedef unsigned value_type;
    typedef int int_type;
    typedef unsigned uint_type;
    typedef unsigned abs_type;
    typedef unsigned sum_type;

    typedef ushort nu_type;

    static int_type reinterpret_int(value_type x) { return (int_type)x; }
    static uint_type reinterpret_uint(value_type x) { return (uint_type)x; }
    static value_type reinterpret_from_int(int_type x) { return (value_type)x; }
};

template<> struct TypeTraits<int>
{
    typedef int value_type;
    typedef int int_type;
    typedef unsigned uint_type;
    typedef unsigned abs_type;
    typedef int sum_type;

    typedef short n_type;
    typedef ushort nu_type;

    static int_type reinterpret_int(value_type x) { return (int_type)x; }
    static uint_type reinterpret_uint(value_type x) { return (uint_type)x; }
    static value_type reinterpret_from_int(int_type x) { return (value_type)x; }
};

template<> struct TypeTraits<float>
{
    typedef float value_type;
    typedef int int_type;
    typedef unsigned uint_type;
    typedef float abs_type;
    typedef float sum_type;

    typedef double w_type;

    static int_type reinterpret_int(value_type x)
    {
        Cv32suf u;
        u.f = x;
        return u.i;
    }
    static uint_type reinterpet_uint(value_type x)
    {
        Cv32suf u;
        u.f = x;
        return u.u;
    }
    static value_type reinterpret_from_int(int_type x)
    {
        Cv32suf u;
        u.i = x;
        return u.f;
    }
};

template<> struct TypeTraits<double>
{
    typedef double value_type;
    typedef int64 int_type;
    typedef uint64 uint_type;
    typedef double abs_type;
    typedef double sum_type;
    static int_type reinterpret_int(value_type x)
    {
        Cv64suf u;
        u.f = x;
        return u.i;
    }
    static uint_type reinterpet_uint(value_type x)
    {
        Cv64suf u;
        u.f = x;
        return u.u;
    }
    static value_type reinterpret_from_int(int_type x)
    {
        Cv64suf u;
        u.i = x;
        return u.f;
    }
};

template<typename _Tp, int n> struct v_reg
{
    typedef _Tp scalar_type;
    typedef v_reg<typename TypeTraits<_Tp>::int_type, n> int_vec;
    typedef v_reg<typename TypeTraits<_Tp>::abs_type, n> abs_vec;
    enum { channels = n };

    explicit v_reg(const _Tp* ptr) { for( int i = 0; i < n; i++ ) s[i] = ptr[i]; }
    v_reg(_Tp s0, _Tp s1) { s[0] = s0; s[1] = s1; }
    v_reg(_Tp s0, _Tp s1, _Tp s2, _Tp s3) { s[0] = s0; s[1] = s1; s[2] = s2; s[3] = s3; }
    v_reg(_Tp s0, _Tp s1, _Tp s2, _Tp s3,
           _Tp s4, _Tp s5, _Tp s6, _Tp s7)
    {
        s[0] = s0; s[1] = s1; s[2] = s2; s[3] = s3;
        s[4] = s4; s[5] = s5; s[6] = s6; s[7] = s7;
    }
    v_reg(_Tp s0, _Tp s1, _Tp s2, _Tp s3,
           _Tp s4, _Tp s5, _Tp s6, _Tp s7,
           _Tp s8, _Tp s9, _Tp s10, _Tp s11,
           _Tp s12, _Tp s13, _Tp s14, _Tp s15)
    {
        s[0] = s0; s[1] = s1; s[2] = s2; s[3] = s3;
        s[4] = s4; s[5] = s5; s[6] = s6; s[7] = s7;
        s[8] = s8; s[9] = s9; s[10] = s10; s[11] = s11;
        s[12] = s12; s[13] = s13; s[14] = s14; s[15] = s15;
    }

    v_reg() {}
    v_reg(const v_reg<_Tp, n> & r)
    {
        for( int i = 0; i < n; i++ )
            s[i] = r.s[i];
    }

    _Tp get(const int i) const { return s[i]; }
    _Tp get0() const { return s[0]; }
    v_reg<_Tp, n> high() const
    {
        v_reg<_Tp, n> c;
        int i;
        for( i = 0; i < n/2; i++ )
        {
            c.s[i] = s[i+(n/2)];
            c.s[i+(n/2)] = 0;
        }
        return c;
    }

    static v_reg<_Tp, n> zero()
    {
        v_reg<_Tp, n> c;
        for( int i = 0; i < n; i++ )
            c.s[i] = (_Tp)0;
        return c;
    }

    static v_reg<_Tp, n> all(_Tp s)
    {
        v_reg<_Tp, n> c;
        for( int i = 0; i < n; i++ )
            c.s[i] = s;
        return c;
    }

    template<typename _Tp2, int n2> static v_reg<_Tp2, n2> reinterpret_as(const v_reg<_Tp, n>& a)
    {
        size_t bytes = std::min(sizeof(_Tp2)*n2, sizeof(_Tp)*n);
        v_reg<_Tp2, n2> c;
        memcpy(&c.s[0], &a.s[0], bytes);
        return c;
    }

    _Tp s[n];
};

#define OPENCV_HAL_IMPL_BIN_OP(bin_op) \
template<typename _Tp, int n> inline v_reg<_Tp, n> operator bin_op (const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b) \
{ \
    v_reg<_Tp, n> c; \
    for( int i = 0; i < n; i++ ) \
        c.s[i] = saturate_cast<_Tp>(a.s[i] bin_op b.s[i]); \
    return c; \
} \
template<typename _Tp, int n> inline v_reg<_Tp, n>& operator bin_op##= (v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b) \
{ \
    for( int i = 0; i < n; i++ ) \
        a.s[i] = saturate_cast<_Tp>(a.s[i] bin_op b.s[i]); \
    return a; \
}

OPENCV_HAL_IMPL_BIN_OP(+)
OPENCV_HAL_IMPL_BIN_OP(-)
OPENCV_HAL_IMPL_BIN_OP(*)
OPENCV_HAL_IMPL_BIN_OP(/)

#define OPENCV_HAL_IMPL_BIT_OP(bit_op) \
template<typename _Tp, int n> inline v_reg<_Tp, n> operator bit_op (const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b) \
{ \
    v_reg<_Tp, n> c; \
    typedef typename TypeTraits<_Tp>::int_type itype; \
    for( int i = 0; i < n; i++ ) \
        c.s[i] = TypeTraits<_Tp>::reinterpret_from_int((itype)(TypeTraits<_Tp>::reinterpret_int(a.s[i]) bit_op \
                                                        TypeTraits<_Tp>::reinterpret_int(b.s[i]))); \
    return c; \
} \
template<typename _Tp, int n> inline v_reg<_Tp, n>& operator bit_op##= (v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b) \
{ \
    typedef typename TypeTraits<_Tp>::int_type itype; \
    for( int i = 0; i < n; i++ ) \
        a.s[i] = TypeTraits<_Tp>::reinterpret_from_int((itype)(TypeTraits<_Tp>::reinterpret_int(a.s[i]) bit_op \
                                                        TypeTraits<_Tp>::reinterpret_int(b.s[i]))); \
    return a; \
}

OPENCV_HAL_IMPL_BIT_OP(&)
OPENCV_HAL_IMPL_BIT_OP(|)
OPENCV_HAL_IMPL_BIT_OP(^)

template<typename _Tp, int n> inline v_reg<_Tp, n> operator ~ (const v_reg<_Tp, n>& a)
{
    v_reg<_Tp, n> c;
    typedef typename TypeTraits<_Tp>::int_type itype;
    for( int i = 0; i < n; i++ )
        c.s[i] = TypeTraits<_Tp>::reinterpret_from_int(~TypeTraits<_Tp>::reinterpret_int(a.s[i]));
        return c;
}

#define OPENCV_HAL_IMPL_MATH_FUNC(func, cfunc, _Tp2) \
template<typename _Tp, int n> inline v_reg<_Tp2, n> func(const v_reg<_Tp, n>& a) \
{ \
    v_reg<_Tp2, n> c; \
    for( int i = 0; i < n; i++ ) \
        c.s[i] = cfunc(a.s[i]); \
    return c; \
}

OPENCV_HAL_IMPL_MATH_FUNC(v_sqrt, std::sqrt, _Tp)
OPENCV_HAL_IMPL_MATH_FUNC(v_sin, std::sin, _Tp)
OPENCV_HAL_IMPL_MATH_FUNC(v_cos, std::cos, _Tp)
OPENCV_HAL_IMPL_MATH_FUNC(v_exp, std::exp, _Tp)
OPENCV_HAL_IMPL_MATH_FUNC(v_log, std::log, _Tp)
OPENCV_HAL_IMPL_MATH_FUNC(v_abs, (typename TypeTraits<_Tp>::abs_type)std::abs, typename TypeTraits<_Tp>::abs_type)
OPENCV_HAL_IMPL_MATH_FUNC(v_round, cvRound, int)
OPENCV_HAL_IMPL_MATH_FUNC(v_floor, cvFloor, int)
OPENCV_HAL_IMPL_MATH_FUNC(v_ceil, cvCeil, int)
OPENCV_HAL_IMPL_MATH_FUNC(v_trunc, int, int)

#define OPENCV_HAL_IMPL_MINMAX_FUNC(func, hfunc, cfunc) \
template<typename _Tp, int n> inline v_reg<_Tp, n> func(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b) \
{ \
    v_reg<_Tp, n> c; \
    for( int i = 0; i < n; i++ ) \
        c.s[i] = cfunc(a.s[i], b.s[i]); \
    return c; \
} \
template<typename _Tp, int n> inline _Tp hfunc(const v_reg<_Tp, n>& a) \
{ \
    _Tp c = a.s[0]; \
    for( int i = 1; i < n; i++ ) \
        c = cfunc(c, a.s[i]); \
    return c; \
}

OPENCV_HAL_IMPL_MINMAX_FUNC(v_min, v_reduce_min, std::min)
OPENCV_HAL_IMPL_MINMAX_FUNC(v_max, v_reduce_max, std::max)

template<typename _Tp, int n> inline void v_minmax(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b,
                                                   v_reg<_Tp, n>& minval, v_reg<_Tp, n>& maxval)
{
    for( int i = 0; i < n; i++ )
    {
        minval.s[i] = std::min(a.s[i], b.s[i]);
        maxval.s[i] = std::max(a.s[i], b.s[i]);
    }
}


#define OPENCV_HAL_IMPL_CMP_OP(cmp_op) \
template<typename _Tp, int n> inline v_reg<_Tp, n> \
    operator cmp_op(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b) \
{ \
    typedef typename TypeTraits<_Tp>::int_type itype; \
    v_reg<_Tp, n> c; \
    for( int i = 0; i < n; i++ ) \
        c.s[i] = TypeTraits<_Tp>::reinterpret_from_int((itype)-(int)(a.s[i] cmp_op b.s[i])); \
    return c; \
}

OPENCV_HAL_IMPL_CMP_OP(<)
OPENCV_HAL_IMPL_CMP_OP(>)
OPENCV_HAL_IMPL_CMP_OP(<=)
OPENCV_HAL_IMPL_CMP_OP(>=)
OPENCV_HAL_IMPL_CMP_OP(==)
OPENCV_HAL_IMPL_CMP_OP(!=)

#define OPENCV_HAL_IMPL_ADDSUB_OP(func, bin_op, cast_op, _Tp2) \
template<typename _Tp, int n> inline v_reg<_Tp2, n> func(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b) \
{ \
    typedef _Tp2 rtype; \
    v_reg<rtype, n> c; \
    for( int i = 0; i < n; i++ ) \
        c.s[i] = cast_op(a.s[i] bin_op b.s[i]); \
    return c; \
}

OPENCV_HAL_IMPL_ADDSUB_OP(v_add_wrap, +, (_Tp), _Tp)
OPENCV_HAL_IMPL_ADDSUB_OP(v_sub_wrap, -, (_Tp), _Tp)
OPENCV_HAL_IMPL_ADDSUB_OP(v_absdiff, -, (rtype)std::abs, typename TypeTraits<_Tp>::abs_type)

template<typename _Tp, int n> inline v_reg<_Tp, n> v_magnitude(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{
    v_reg<_Tp, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = std::sqrt(a.s[i]*a.s[i] + b.s[i]*b.s[i]);
    return c;
}

template<typename _Tp, int n> inline v_reg<_Tp, n> v_sqr_magnitude(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{
    v_reg<_Tp, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = a.s[i]*a.s[i] + b.s[i]*b.s[i];
    return c;
}

template<typename _Tp, int n> inline v_reg<_Tp, n> v_muladd(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b,
                                                            const v_reg<_Tp, n>& c)
{
    v_reg<_Tp, n> d;
    for( int i = 0; i < n; i++ )
        d.s[i] = a.s[i]*b.s[i] + c.s[i];
    return d;
}

template<typename _Tp, int n> inline v_reg<_Tp, n> v_mullo(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{
    v_reg<_Tp, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = (_Tp)(a.s[i]*b.s[i]);
    return c;
}

template<typename _Tp, int n> inline v_reg<_Tp, n> v_mulhi2(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{
    v_reg<_Tp, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = (_Tp)((a.s[i]*b.s[i]*2 + TypeTraits<_Tp>::delta) >> TypeTraits<_Tp>::shift);
    return c;
}

#define OPENCV_HAL_IMPL_SHIFT_OP(shift_op) \
template<typename _Tp, int n> inline v_reg<_Tp, n> operator shift_op(const v_reg<_Tp, n>& a, int imm) \
{ \
    v_reg<_Tp, n> c; \
    for( int i = 0; i < n; i++ ) \
        c.s[i] = (_Tp)(a.s[i] shift_op imm); \
    return c; \
}

OPENCV_HAL_IMPL_SHIFT_OP(<<)
OPENCV_HAL_IMPL_SHIFT_OP(>>)

template<typename _Tp, int n> inline typename TypeTraits<_Tp>::sum_type v_reduce_sum(const v_reg<_Tp, n>& a)
{
    typename TypeTraits<_Tp>::sum_type c = a.s[0];
    for( int i = 1; i < n; i++ )
        c += a.s[i];
    return c;
}

template<typename _Tp, int n> inline int v_signmask(const v_reg<_Tp, n>& a)
{
    int mask = 0;
    for( int i = 0; i < n; i++ )
        mask |= (TypeTraits<_Tp>::reinterpret_int(a.s[i]) < 0) << i;
    return mask;
}

template<typename _Tp, int n> inline bool v_check_all(const v_reg<_Tp, n>& a)
{
    for( int i = 0; i < n; i++ )
        if( TypeTraits<_Tp>::reinterpret_int(a.s[i]) >= 0 )
            return false;
    return true;
}

template<typename _Tp, int n> inline bool v_check_any(const v_reg<_Tp, n>& a)
{
    for( int i = 0; i < n; i++ )
        if( TypeTraits<_Tp>::reinterpret_int(a.s[i]) < 0 )
            return true;
    return false;
}

template<typename _Tp, int n> inline v_reg<_Tp, n> v_select(const v_reg<_Tp, n>& mask,
                                                           const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{
    v_reg<_Tp, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = TypeTraits<_Tp>::reinterpret_int(mask.s[i]) < 0 ? b.s[i] : a.s[i];
    return c;
}

template<typename _Tp, int n> inline void v_expand(const v_reg<_Tp, n>& a,
                            v_reg<typename TypeTraits<_Tp>::w_type, n/2>& b0,
                            v_reg<typename TypeTraits<_Tp>::w_type, n/2>& b1)
{
    for( int i = 0; i < (n/2); i++ )
    {
        b0.s[i] = a.s[i];
        b1.s[i] = a.s[i+(n/2)];
    }
}

template<typename _Tp, int n> inline v_reg<typename TypeTraits<_Tp>::int_type, n>
    v_reinterpret_int(const v_reg<_Tp, n>& a)
{
    v_reg<typename TypeTraits<_Tp>::int_type, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = TypeTraits<_Tp>::reinterpret_int(a.s[i]);
    return c;
}

template<typename _Tp, int n> inline v_reg<typename TypeTraits<_Tp>::uint_type, n>
    v_reinterpret_uint(const v_reg<_Tp, n>& a)
{
    v_reg<typename TypeTraits<_Tp>::uint_type, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = TypeTraits<_Tp>::reinterpret_uint(a.s[i]);
    return c;
}

template<typename _Tp, int n> inline void v_zip( const v_reg<_Tp, n>& a0, const v_reg<_Tp, n>& a1,
                                               v_reg<_Tp, n>& b0, v_reg<_Tp, n>& b1 )
{
    int i;
    for( i = 0; i < n/2; i++ )
    {
        b0.s[i*2] = a0.s[i];
        b0.s[i*2+1] = a1.s[i];
    }
    for( ; i < n; i++ )
    {
        b1.s[i*2-n] = a0.s[i];
        b1.s[i*2-n+1] = a1.s[i];
    }
}

template<typename _Tp, int n> inline v_reg<_Tp, n> v_load(const _Tp* ptr)
{
    return v_reg<_Tp, n>(ptr);
}

template<typename _Tp, int n> inline v_reg<_Tp, n> v_load_aligned(const _Tp* ptr)
{
    return v_reg<_Tp, n>(ptr);
}

template<typename _Tp, int n> inline void v_load_halves(const _Tp* loptr, const _Tp* hiptr)
{
    v_reg<_Tp, n> c;
    for( int i = 0; i < n/2; i++ )
    {
        c.s[i] = loptr[i];
        c.s[i+n/2] = hiptr[i];
    }
    return c;
}

template<typename _Tp, int n> inline v_reg<typename TypeTraits<_Tp>::w_type, n> v_load_expand(const _Tp* ptr)
{
    typedef typename TypeTraits<_Tp>::w_type w_type;
    v_reg<w_type, n> c;
    for( int i = 0; i < n; i++ )
    {
        c.s[i] = ptr[i];
    }
    return c;
}

template<typename _Tp, int n> inline v_reg<typename
    TypeTraits<typename TypeTraits<_Tp>::w_type>::w_type, n> v_load_expand_q(const _Tp* ptr)
{
    typedef typename TypeTraits<typename TypeTraits<_Tp>::w_type>::w_type w_type;
    v_reg<w_type, n> c;
    for( int i = 0; i < n; i++ )
    {
        c.s[i] = ptr[i];
    }
    return c;
}

template<typename _Tp, int n> inline void v_load_deinterleave(const _Tp* ptr, v_reg<_Tp, n>& a,
                                                            v_reg<_Tp, n>& b, v_reg<_Tp, n>& c)
{
    int i, i3;
    for( i = i3 = 0; i < n; i++, i3 += 3 )
    {
        a.s[i] = ptr[i3];
        b.s[i] = ptr[i3+1];
        c.s[i] = ptr[i3+2];
    }
}

template<typename _Tp, int n> inline void v_load_deinterleave(const _Tp* ptr, v_reg<_Tp, n>& a,
                                                            v_reg<_Tp, n>& b, v_reg<_Tp, n>& c,
                                                            v_reg<_Tp, n>& d)
{
    int i, i4;
    for( i = i4 = 0; i < n; i++, i4 += 4 )
    {
        a.s[i] = ptr[i4];
        b.s[i] = ptr[i4+1];
        c.s[i] = ptr[i4+2];
        d.s[i] = ptr[i4+3];
    }
}

template<typename _Tp, int n> inline void v_store_interleave( _Tp* ptr, const v_reg<_Tp, n>& a,
                                                            const v_reg<_Tp, n>& b, const v_reg<_Tp, n>& c)
{
    int i, i3;
    for( i = i3 = 0; i < n; i++, i3 += 3 )
    {
        ptr[i3] = a.s[i];
        ptr[i3+1] = b.s[i];
        ptr[i3+2] = c.s[i];
    }
}

template<typename _Tp, int n> inline void v_store_interleave( _Tp* ptr, const v_reg<_Tp, n>& a,
                                                            const v_reg<_Tp, n>& b, const v_reg<_Tp, n>& c,
                                                            const v_reg<_Tp, n>& d)
{
    int i, i4;
    for( i = i4 = 0; i < n; i++, i4 += 4 )
    {
        ptr[i4] = a.s[i];
        ptr[i4+1] = b.s[i];
        ptr[i4+2] = c.s[i];
        ptr[i4+3] = d.s[i];
    }
}

template<typename _Tp, int n> inline void v_store(_Tp* ptr, const v_reg<_Tp, n>& a)
{
    for( int i = 0; i < n; i++ )
        ptr[i] = a.s[i];
}

template<typename _Tp, int n> inline void v_store_low(_Tp* ptr, const v_reg<_Tp, n>& a)
{
    for( int i = 0; i < (n/2); i++ )
        ptr[i] = a.s[i];
}

template<typename _Tp, int n> inline void v_store_high(_Tp* ptr, const v_reg<_Tp, n>& a)
{
    for( int i = 0; i < (n/2); i++ )
        ptr[i] = a.s[i+(n/2)];
}

template<typename _Tp, int n> inline void v_store_aligned(_Tp* ptr, const v_reg<_Tp, n>& a)
{
    for( int i = 0; i < n; i++ )
        ptr[i] = a.s[i];
}

template<typename _Tp, int n> inline v_reg<_Tp, n> v_combine_low(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{
    v_reg<_Tp, n> c;
    for( int i = 0; i < (n/2); i++ )
    {
        c.s[i] = a.s[i];
        c.s[i+(n/2)] = b.s[i];
    }
}

template<typename _Tp, int n> inline v_reg<_Tp, n> v_combine_high(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{
    v_reg<_Tp, n> c;
    for( int i = 0; i < (n/2); i++ )
    {
        c.s[i] = a.s[i+(n/2)];
        c.s[i+(n/2)] = b.s[i+(n/2)];
    }
}

template<typename _Tp, int n> inline void v_recombine(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b,
                                                      v_reg<_Tp, n>& low, v_reg<_Tp, n>& high)
{
    for( int i = 0; i < (n/2); i++ )
    {
        low.s[i] = a.s[i];
        low.s[i+(n/2)] = b.s[i];
        high.s[i] = a.s[i+(n/2)];
        high.s[i+(n/2)] = b.s[i+(n/2)];
    }
}

template<int n> inline v_reg<int, n> v_round(const v_reg<float, n>& a)
{
    v_reg<int, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = cvRound(a.s[i]);
    return c;
}

template<int n> inline v_reg<int, n> v_floor(const v_reg<float, n>& a)
{
    v_reg<int, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = cvFloor(a.s[i]);
    return c;
}

template<int n> inline v_reg<int, n> v_ceil(const v_reg<float, n>& a)
{
    v_reg<int, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = cvCeil(a.s[i]);
    return c;
}

template<int n> inline v_reg<int, n> v_trunc(const v_reg<float, n>& a)
{
    v_reg<int, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = (int)(a.s[i]);
    return c;
}

template<int n> inline v_reg<int, n*2> v_round(const v_reg<double, n>& a)
{
    v_reg<int, n*2> c;
    for( int i = 0; i < n; i++ )
    {
        c.s[i] = cvRound(a.s[i]);
        c.s[i+n] = 0;
    }
    return c;
}

template<int n> inline v_reg<int, n*2> v_floor(const v_reg<double, n>& a)
{
    v_reg<int, n> c;
    for( int i = 0; i < n; i++ )
    {
        c.s[i] = cvFloor(a.s[i]);
        c.s[i+n] = 0;
    }
    return c;
}

template<int n> inline v_reg<int, n*2> v_ceil(const v_reg<double, n>& a)
{
    v_reg<int, n> c;
    for( int i = 0; i < n; i++ )
    {
        c.s[i] = cvCeil(a.s[i]);
        c.s[i+n] = 0;
    }
    return c;
}

template<int n> inline v_reg<int, n*2> v_trunc(const v_reg<double, n>& a)
{
    v_reg<int, n> c;
    for( int i = 0; i < n; i++ )
    {
        c.s[i] = cvCeil(a.s[i]);
        c.s[i+n] = 0;
    }
    return c;
}

template<int n> inline v_reg<float, n> v_cvt_f32(const v_reg<int, n>& a)
{
    v_reg<float, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = (float)a.s[i];
    return c;
}

template<int n> inline v_reg<double, n> v_cvt_f64(const v_reg<int, n*2>& a)
{
    v_reg<double, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = (double)a.s[i];
    return c;
}

template<int n> inline v_reg<double, n> v_cvt_f64(const v_reg<float, n*2>& a)
{
    v_reg<double, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = (double)a.s[i];
    return c;
}

template<typename _Tp, int n, typename _Tp2> inline v_reg<_Tp2, n*2> v_cvtsat(const v_reg<_Tp, n>& a,
                                                                              const v_reg<_Tp, n>& b)
{
    v_reg<_Tp2, n*2> c;
    for( int i = 0; i < n; i++ )
    {
        c.s[i] = saturate_cast<_Tp2>(a.s[i]);
        c.s[i+n] = saturate_cast<_Tp2>(b.s[i]);
    }
    return c;
}

template<typename _Tp, int n, typename _Tp2> inline v_reg<_Tp2, n*2> v_cvtsat(const v_reg<_Tp, n>& a,
                                                                              const v_reg<_Tp, n>& b,
                                                                              int rshift)
{
    v_reg<_Tp2, n*2> c;
    for( int i = 0; i < n; i++ )
    {
        c.s[i] = saturate_cast<_Tp2>((a.s[i] + (1<<(rshift-1))) >> rshift);
        c.s[i+n] = saturate_cast<_Tp2>((b.s[i] + (1<<(rshift-1))) >> rshift);
    }
    return c;
}

template<typename _Tp, int n, typename _Tp2> inline void v_storesat(_Tp2* ptr, const v_reg<_Tp, n>& a)
{
    for( int i = 0; i < n; i++ )
    {
        ptr[i] = saturate_cast<_Tp2>(a.s[i]);
    }
}

template<typename _Tp, int n, typename _Tp2> inline void v_storesat(_Tp2* ptr, const v_reg<_Tp, n>& a, int rshift)
{
    for( int i = 0; i < n; i++ )
    {
        ptr[i] = saturate_cast<_Tp2>((a.s[i] + (1<<(rshift-1))) >> rshift);
    }
}

template<typename _Tp> inline void v_transpose4x4(const v_reg<_Tp, 4>& a0, const v_reg<_Tp, 4>& a1,
                                                  const v_reg<_Tp, 4>& a2, const v_reg<_Tp, 4>& a3,
                                                  v_reg<_Tp, 4>& b0, v_reg<_Tp, 4>& b1,
                                                  v_reg<_Tp, 4>& b2, v_reg<_Tp, 4>& b3)
{
    b0 = v_reg<_Tp, 4>(a0.s[0], a1.s[0], a2.s[0], a3.s[0]);
    b1 = v_reg<_Tp, 4>(a0.s[1], a1.s[1], a2.s[1], a3.s[1]);
    b2 = v_reg<_Tp, 4>(a0.s[2], a1.s[2], a2.s[2], a3.s[2]);
    b3 = v_reg<_Tp, 4>(a0.s[3], a1.s[3], a2.s[3], a3.s[3]);
}

#if CV_SSE2

#define CV_SIMD128 1
#define CV_SIMD128_64F 1

struct v_uint8x16
{
    explicit v_uint8x16(__m128i v) : val(v) {}
    v_uint8x16(uchar v0, uchar v1, uchar v2, uchar v3, uchar v4, uchar v5, uchar v6, uchar v7,
               uchar v8, uchar v9, uchar v10, uchar v11, uchar v12, uchar v13, uchar v14, uchar v15)
    {
        val = _mm_setr_epi8((char)v0, (char)v1, (char)v2, (char)v3,
                            (char)v4, (char)v5, (char)v6, (char)v7,
                            (char)v8, (char)v9, (char)v10, (char)v11,
                            (char)v12, (char)v13, (char)v14, (char)v15);
    }
    uchar get(const int i) const
    {
        return (uchar)(_mm_extract_epi16(val, i/2) >> ((i&1)*8));
    }
    uchar get0() const
    {
        return (uchar)_mm_cvtsi128_si32(val);
    }

    __m128i val;
};

struct v_int8x16
{
    explicit v_int8x16(__m128i v) : val(v) {}
    v_int8x16(schar v0, schar v1, schar v2, schar v3, schar v4, schar v5, schar v6, schar v7,
              schar v8, schar v9, schar v10, schar v11, schar v12, schar v13, schar v14, schar v15)
    {
        val = _mm_setr_epi8((char)v0, (char)v1, (char)v2, (char)v3,
                            (char)v4, (char)v5, (char)v6, (char)v7,
                            (char)v8, (char)v9, (char)v10, (char)v11,
                            (char)v12, (char)v13, (char)v14, (char)v15);
    }
    schar get(const int i) const
    {
        return (schar)(_mm_extract_epi16(val, i/2) >> ((i&1)*8));
    }
    schar get0() const
    {
        return (schar)_mm_cvtsi128_si32(val);
    }

    __m128i val;
};

struct v_uint16x8
{
    explicit v_uint16x8(__m128i v) : val(v) {}
    v_uint16x8(ushort v0, ushort v1, ushort v2, ushort v3, ushort v4, ushort v5, ushort v6, ushort v7)
    {
        val = _mm_setr_epi16((short)v0, (short)v1, (short)v2, (short)v3,
                             (short)v4, (short)v5, (short)v6, (short)v7);
    }
    ushort get(const int i) const
    {
        return (ushort)_mm_extract_epi16(val, i);
    }
    uchar get0() const
    {
        return (ushort)_mm_cvtsi128_si32(val);
    }

    __m128i val;
};

struct v_int16x8
{
    explicit v_int16x8(__m128i v) : val(v) {}
    v_int16x8(short v0, short v1, short v2, short v3, short v4, short v5, short v6, short v7)
    {
        val = _mm_setr_epi16((short)v0, (short)v1, (short)v2, (short)v3,
                             (short)v4, (short)v5, (short)v6, (short)v7);
    }
    short get(const int i) const
    {
        return (short)_mm_extract_epi16(val, i);
    }
    short get0() const
    {
        return (short)_mm_cvtsi128_si32(val);
    }
    __m128i val;
};

struct v_uint32x4
{
    explicit v_uint32x4(__m128i v) : val(v) {}
    v_uint32x4(unsigned v0, unsigned v1, unsigned v2, unsigned v3)
    {
        val = _mm_setr_epi32((int)v0, (int)v1, (int)v2, (int)v3);
    }
    unsigned get(const int i) const
    {
        unsigned CV_DECL_ALIGNED(16) buf[4];
        _mm_store_si128((__m128i*)buf, val);
        return buf[i];
    }
    unsigned get0() const
    {
        return (unsigned)_mm_cvtsi128_si32(val);
    }
    __m128i val;
};

struct v_int32x4
{
    explicit v_int32x4(__m128i v) : val(v) {}
    v_int32x4(int v0, int v1, int v2, int v3)
    {
        val = _mm_setr_epi32(v0, v1, v2, v3);
    }
    int get(int i) const
    {
        int CV_DECL_ALIGNED(16) buf[4];
        _mm_store_si128((__m128i*)buf, val);
        return buf[i];
    }
    int get0() const
    {
        return _mm_cvtsi128_si32(val);
    }
    __m128i val;
};

struct v_float32x4
{
    explicit v_float32x4(__m128 v) : val(v) {}
    v_float32x4(float v0, float v1, float v2, float v3)
    {
        val = _mm_setr_ps(v0, v1, v2, v3);
    }
    float get(int i) const
    {
        float CV_DECL_ALIGNED(16) buf[4];
        _mm_store_ps(buf, val);
        return buf[i];
    }
    float get0() const
    {
        return _mm_cvtss_f32(val);
    }
    __m128 val;
};

struct v_float64x2
{
    explicit v_float64x2(__m128d v) : val(v) {}
    v_float64x2(double v0, double v1)
    {
        val = _mm_setr_pd(v0, v1);
    }
    double get(int i) const
    {
        double CV_DECL_ALIGNED(16) buf[2];
        _mm_store_pd(buf, val);
        return buf[i];
    }
    double get0() const
    {
        return _mm_cvtsd_f64(val);
    }
    __m128d val;
};

inline v_uint8x16 v_setzero_u8() { return v_uint8x16(_mm_setzero_si128()); }
inline v_int8x16 v_setzero_s8() { return v_int8x16(_mm_setzero_si128()); }
inline v_uint16x8 v_setzero_u16() { return v_uint16x8(_mm_setzero_si128()); }
inline v_int16x8 v_setzero_s16() { return v_int16x8(_mm_setzero_si128()); }
inline v_uint32x4 v_setzero_u32() { return v_uint32x4(_mm_setzero_si128()); }
inline v_int32x4 v_setzero_s32() { return v_int32x4(_mm_setzero_si128()); }
inline v_float32x4 v_setzero_f32() { return v_float32x4(_mm_setzero_ps()); }
inline v_float64x2 v_setzero_f64() { return v_float64x2(_mm_setzero_pd()); }

inline v_uint8x16 v_setall_u8(uchar v) { return v_uint8x16(_mm_set1_epi8((char)v)); }
inline v_int8x16 v_setall_s8(schar v) { return v_int8x16(_mm_set1_epi8((char)v)); }
inline v_uint16x8 v_setall_u16(ushort v) { return v_uint16x8(_mm_set1_epi16((short)v)); }
inline v_int16x8 v_setall_s16(short v) { return v_int16x8(_mm_set1_epi16((short)v)); }
inline v_uint32x4 v_setall_u32(unsigned v) { return v_uint32x4(_mm_set1_epi32((int)v)); }
inline v_int32x4 v_setall_s32(int v) { return v_int32x4(_mm_set1_epi32(v)); }
inline v_float32x4 v_setall_f32(float v) { return v_float32x4(_mm_set1_ps(v)); }
inline v_float64x2 v_setall_f64(double v) { return v_float64x2(_mm_set1_pd(v)); }

template<typename _Tpvec> inline v_uint8x16 v_reinterpret_u8(const _Tpvec& a)
{ return v_uint8x16(a.val); }

inline v_uint8x16 v_reinterpret_u8(const v_float32x4& a)
{ return v_uint8x16(_mm_castps_si128(a.val)); }

inline v_uint8x16 v_reinterpret_u8(const v_float64x2& a)
{ return v_uint8x16(_mm_castpd_si128(a.val)); }

template<typename _Tpvec> inline v_int8x16 v_reinterpret_s8(const _Tpvec& a)
{ return v_int8x16(a.val); }

inline v_int8x16 v_reinterpret_s8(const v_float32x4& a)
{ return v_int8x16(_mm_castps_si128(a.val)); }

inline v_int8x16 v_reinterpret_s8(const v_float64x2& a)
{ return v_int8x16(_mm_castpd_si128(a.val)); }

template<typename _Tpvec> inline v_uint16x8 v_reinterpret_u16(const _Tpvec& a)
{ return v_uint16x8(a.val); }

inline v_uint16x8 v_reinterpret_u16(const v_float32x4& a)
{ return v_uint16x8(_mm_castps_si128(a.val)); }

inline v_uint16x8 v_reinterpret_u16(const v_float64x2& a)
{ return v_uint16x8(_mm_castpd_si128(a.val)); }

template<typename _Tpvec> inline v_int16x8 v_reinterpret_s16(const _Tpvec& a)
{ return v_int16x8(a.val); }

inline v_int16x8 v_reinterpret_s16(const v_float32x4& a)
{ return v_int16x8(_mm_castps_si128(a.val)); }

inline v_int16x8 v_reinterpret_s16(const v_float64x2& a)
{ return v_int16x8(_mm_castpd_si128(a.val)); }

template<typename _Tpvec> inline v_uint32x4 v_reinterpret_u32(const _Tpvec& a)
{ return v_uint32x4(a.val); }

inline v_uint32x4 v_reinterpret_u32(const v_float32x4& a)
{ return v_uint32x4(_mm_castps_si128(a.val)); }

inline v_uint32x4 v_reinterpret_u32(const v_float64x2& a)
{ return v_uint32x4(_mm_castpd_si128(a.val)); }

template<typename _Tpvec> inline v_int32x4 v_reinterpret_s32(const _Tpvec& a)
{ return v_int32x4(a.val); }

inline v_int32x4 v_reinterpret_s32(const v_float32x4& a)
{ return v_int32x4(_mm_castps_si128(a.val)); }

inline v_int32x4 v_reinterpret_s32(const v_float64x2& a)
{ return v_int32x4(_mm_castpd_si128(a.val)); }

template<typename _Tpvec> inline v_float32x4 v_reinterpret_f32(const _Tpvec& a)
{ return v_float32x4(_mm_castsi128_ps(a.val)); }

inline v_float32x4 v_reinterpret_f32(const v_float64x2& a)
{ return v_float32x4(_mm_castpd_ps(a.val)); }

template<typename _Tpvec> inline v_float64x2 v_reinterpret_f64(const _Tpvec& a)
{ return v_float64x2(_mm_castsi128_pd(a.val)); }

inline v_float64x2 v_reinterpret_f64(const v_float64x2& a)
{ return v_float64x2(_mm_castps_pd(a.val)); }

inline v_uint8x16 v_sat_u8(const v_uint16x8& a, const v_uint16x8& b)
{
    __m128i delta = _mm_set1_epi16(255);
    return v_uint8x16(_mm_packus_epi16(_mm_adds_epu16(_mm_subs_epu16(a.val, delta), delta),
                                       _mm_adds_epu16(_mm_subs_epu16(b.val, delta), delta)));
}
inline v_uint8x16 v_sat_u8(const v_uint16x8& a, const v_uint16x8& b, int n)
{
    // we assume that n > 0, and so the shifted 16-bit values can be treated as signed numbers.
    __m128i delta = _mm_set1_epi16((short)(1 << (n-1)));
    return v_uint8x16(_mm_packus_epi16(_mm_srli_epi16(_mm_add_epi16(a.val, delta), n),
                                       _mm_srli_epi16(_mm_add_epi16(b.val, delta), n)));
}

inline v_uint8x16 v_sat_u8(const v_int16x8& a, const v_int16x8& b)
{ return v_uint8x16(_mm_packus_epi16(a.val, b.val)); }
inline v_uint8x16 v_sat_u8(const v_int16x8& a, const v_int16x8& b, int n)
{
    __m128i delta = _mm_set1_epi16((short)(1 << (n-1)));
    return v_uint8x16(_mm_packus_epi16(_mm_srai_epi16(_mm_add_epi16(a.val, delta), n),
                                       _mm_srai_epi16(_mm_add_epi16(b.val, delta), n)));
}

inline void v_storesat_u8(uchar* ptr, const v_uint16x8& a)
{
    __m128i delta = _mm_set1_epi16(255);
    _mm_storel_epi64((__m128i*)ptr,
                     _mm_packus_epi16(_mm_adds_epu16(_mm_subs_epu16(a.val, delta), delta), delta));
}

inline void v_storesat_u8(uchar* ptr, const v_uint16x8& a, int n)
{
    __m128i delta = _mm_set1_epi16((short)(1 << (n-1)));
    _mm_storel_epi64((__m128i*)ptr,
                     _mm_packus_epi16(_mm_srli_epi16(_mm_add_epi16(a.val, delta), n), delta));
}

inline void v_storesat_u8(uchar* ptr, const v_int16x8& a)
{ _mm_storel_epi64((__m128i*)ptr, _mm_packus_epi16(a.val, a.val)); }

inline void v_storesat_u8(uchar* ptr, const v_int16x8& a, int n)
{
    __m128i delta = _mm_set1_epi16((short)(1 << (n-1)));
    _mm_storel_epi64((__m128i*)ptr,
        _mm_packus_epi16(_mm_srai_epi16(_mm_add_epi16(a.val, delta), n), delta));
}

inline v_int8x16 v_sat_s8(const v_uint16x8& a, const v_uint16x8& b)
{
    __m128i delta = _mm_set1_epi16(127);
    return v_int8x16(_mm_packs_epi16(_mm_adds_epu16(_mm_subs_epu16(a.val, delta), delta),
                                       _mm_adds_epu16(_mm_subs_epu16(b.val, delta), delta)));
}

inline v_int8x16 v_sat_s8(const v_uint16x8& a, const v_uint16x8& b, int n)
{
    // we assume that n > 0, and so the shifted 16-bit values can be treated as signed numbers.
    __m128i delta = _mm_set1_epi16((short)(1 << (n-1)));
    return v_int8x16(_mm_packs_epi16(_mm_srli_epi16(_mm_add_epi16(a.val, delta), n),
                                       _mm_srli_epi16(_mm_add_epi16(b.val, delta), n)));
}

inline v_int8x16 v_sat_s8(const v_int16x8& a, const v_int16x8& b)
{ return v_int8x16(_mm_packs_epi16(a.val, b.val)); }

inline v_int8x16 v_sat_s8(const v_int16x8& a, const v_int16x8& b, int n)
{
    __m128i delta = _mm_set1_epi16((short)(1 << (n-1)));
    return v_int8x16(_mm_packs_epi16(_mm_srai_epi16(_mm_add_epi16(a.val, delta), n),
                                       _mm_srai_epi16(_mm_add_epi16(b.val, delta), n)));
}

inline void v_storesat_s8(schar* ptr, const v_uint16x8& a)
{
    __m128i delta = _mm_set1_epi16(127);
    _mm_storel_epi64((__m128i*)ptr,
                     _mm_packs_epi16(_mm_adds_epu16(_mm_subs_epu16(a.val, delta), delta), delta));
}

inline void v_storesat_s8(schar* ptr, const v_uint16x8& a, int n)
{
    __m128i delta = _mm_set1_epi16((short)(1 << (n-1)));
    _mm_storel_epi64((__m128i*)ptr,
                     _mm_packs_epi16(_mm_srli_epi16(_mm_add_epi16(a.val, delta), n), delta));
}
inline void v_storesat_s8(schar* ptr, const v_int16x8& a)
{ _mm_storel_epi64((__m128i*)ptr, _mm_packs_epi16(a.val, a.val)); }

inline void v_storesat_s8(schar* ptr, const v_int16x8& a, int n)
{
    __m128i delta = _mm_set1_epi16((short)(1 << (n-1)));
    _mm_storel_epi64((__m128i*)ptr,
                     _mm_packs_epi16(_mm_srai_epi16(_mm_add_epi16(a.val, delta), n), delta));
}

// bit-wise "mask ? a : b"
inline __m128i v_select_si128(__m128i mask, __m128i a, __m128i b)
{
    return _mm_xor_si128(b, _mm_and_si128(_mm_xor_si128(a, b), mask));
}

inline v_uint16x8 v_sat_u16(const v_uint32x4& a, const v_uint32x4& b)
{
    __m128i z = _mm_setzero_si128(), maxval32 = _mm_set1_epi32(65535), delta32 = _mm_set1_epi32(32768);
    __m128i a1 = _mm_sub_epi32(v_select_si128(_mm_cmpgt_epi32(z, a.val), maxval32, a.val), delta32);
    __m128i b1 = _mm_sub_epi32(v_select_si128(_mm_cmpgt_epi32(z, b.val), maxval32, b.val), delta32);
    __m128i r = _mm_packs_epi32(a1, b1);
    return v_uint16x8(_mm_sub_epi16(r, _mm_set1_epi16(-32768)));
}
inline v_uint16x8 v_sat_u16(const v_uint32x4& a, const v_uint32x4& b, int n)
{
    __m128i delta = _mm_set1_epi32(1 << (n-1)), delta32 = _mm_set1_epi32(32768);
    __m128i a1 = _mm_sub_epi32(_mm_srli_epi32(_mm_add_epi32(a.val, delta), n), delta32);
    __m128i b1 = _mm_sub_epi32(_mm_srli_epi32(_mm_add_epi32(b.val, delta), n), delta32);
    return v_uint16x8(_mm_sub_epi16(_mm_packs_epi32(a1, b1), _mm_set1_epi16(-32768)));
}
inline v_uint16x8 v_sat_u16(const v_int32x4& a, const v_int32x4& b)
{
    __m128i delta32 = _mm_set1_epi32(32768);
    __m128i r = _mm_packs_epi32(_mm_sub_epi32(a.val, delta32), _mm_sub_epi32(b.val, delta32));
    return v_uint16x8(_mm_sub_epi16(r, _mm_set1_epi16(-32768)));
}
inline v_uint16x8 v_sat_u16(const v_int32x4& a, const v_int32x4& b, int n)
{
    __m128i delta = _mm_set1_epi32(1 << (n-1)), delta32 = _mm_set1_epi32(32768);
    __m128i a1 = _mm_sub_epi32(_mm_srai_epi32(_mm_add_epi32(a.val, delta), n), delta32);
    __m128i b1 = _mm_sub_epi32(_mm_srai_epi32(_mm_add_epi32(b.val, delta), n), delta32);
    return v_uint16x8(_mm_sub_epi16(_mm_packs_epi32(a1, b1), _mm_set1_epi16(-32768)));
}

inline void v_storesat_u16(ushort* ptr, const v_uint32x4& a)
{
    __m128i z = _mm_setzero_si128(), maxval32 = _mm_set1_epi32(65535), delta32 = _mm_set1_epi32(32768);
    __m128i a1 = _mm_sub_epi32(v_select_si128(_mm_cmpgt_epi32(z, a.val), maxval32, a.val), delta32);
    __m128i r = _mm_sub_epi16(_mm_packs_epi32(a1, delta32), _mm_set1_epi16(-32768));
    _mm_storel_epi64((__m128i*)ptr, r);
}
inline void v_storesat_u16(ushort* ptr, const v_uint32x4& a, int n)
{
    __m128i delta = _mm_set1_epi32(1 << (n-1)), delta32 = _mm_set1_epi32(32768);
    __m128i a1 = _mm_sub_epi32(_mm_srli_epi32(_mm_add_epi32(a.val, delta), n), delta32);
    __m128i r = _mm_sub_epi16(_mm_packs_epi32(a1, delta32), _mm_set1_epi16(-32768));
    _mm_storel_epi64((__m128i*)ptr, r);
}
inline void v_storesat_u16(ushort* ptr, const v_int32x4& a)
{
    __m128i delta32 = _mm_set1_epi32(32768);
    __m128i a1 = _mm_sub_epi32(a.val, delta32);
    __m128i r = _mm_sub_epi16(_mm_packs_epi32(a1, delta32), _mm_set1_epi16(-32768));
    _mm_storel_epi64((__m128i*)ptr, r);
}
inline void v_storesat_u16(ushort* ptr, const v_int32x4& a, int n)
{
    __m128i delta = _mm_set1_epi32(1 << (n-1)), delta32 = _mm_set1_epi32(32768);
    __m128i a1 = _mm_sub_epi32(_mm_srai_epi32(_mm_add_epi32(a.val, delta), n), delta32);
    __m128i r = _mm_sub_epi16(_mm_packs_epi32(a1, delta32), _mm_set1_epi16(-32768));
    _mm_storel_epi64((__m128i*)ptr, r);
}

inline v_int16x8 v_sat_s16(const v_uint32x4& a, const v_uint32x4& b)
{
    __m128i z = _mm_setzero_si128(), maxval32 = _mm_set1_epi32(32767);
    __m128i a1 = v_select_si128(_mm_cmpgt_epi32(z, a.val), maxval32, a.val);
    __m128i b1 = v_select_si128(_mm_cmpgt_epi32(z, b.val), maxval32, b.val);
    return v_int16x8(_mm_packs_epi32(a1, b1));
}
inline v_int16x8 v_sat_s16(const v_uint32x4& a, const v_uint32x4& b, int n)
{
    __m128i delta = _mm_set1_epi32(1 << (n-1));
    return v_int16x8(_mm_packs_epi32(_mm_srli_epi32(_mm_add_epi32(a.val, delta), n),
                                     _mm_srli_epi32(_mm_add_epi32(b.val, delta), n)));
}
inline v_int16x8 v_sat_s16(const v_int32x4& a, const v_int32x4& b)
{ return v_int16x8(_mm_packs_epi32(a.val, b.val)); }
inline v_int16x8 v_sat_s16(const v_int32x4& a, const v_int32x4& b, int n)
{
    __m128i delta = _mm_set1_epi32(1 << (n-1));
    return v_int16x8(_mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(a.val, delta), n),
                                     _mm_srai_epi32(_mm_add_epi32(b.val, delta), n)));
}

inline void v_storesat_s16(short* ptr, const v_uint32x4& a)
{
    __m128i z = _mm_setzero_si128(), maxval32 = _mm_set1_epi32(32767);
    __m128i a1 = v_select_si128(_mm_cmpgt_epi32(z, a.val), maxval32, a.val);
    _mm_storel_epi64((__m128i*)ptr, _mm_packs_epi32(a1, a1));
}
inline void v_storesat_s16(short* ptr, const v_uint32x4& a, int n)
{
    __m128i delta = _mm_set1_epi32(1 << (n-1));
    __m128i a1 = _mm_srli_epi32(_mm_add_epi32(a.val, delta), n);
    _mm_storel_epi64((__m128i*)ptr, _mm_packs_epi32(a1, a1));
}
inline void v_storesat_s16(short* ptr, const v_int32x4& a)
{
    _mm_storel_epi64((__m128i*)ptr, _mm_packs_epi32(a.val, a.val));
}
inline void v_storesat_s16(short* ptr, const v_int32x4& a, int n)
{
    __m128i delta = _mm_set1_epi32(1 << (n-1));
    __m128i a1 = _mm_srai_epi32(_mm_add_epi32(a.val, delta), n);
    _mm_storel_epi64((__m128i*)ptr, _mm_packs_epi32(a1, a1));
}

inline v_float32x4 v_matmul(const v_float32x4& v, const v_float32x4& m0,
                            const v_float32x4& m1, const v_float32x4& m2,
                            const v_float32x4& m3)
{
    __m128 v0 = _mm_mul_ps(_mm_shuffle_ps(v.val, v.val, _MM_SHUFFLE(0, 0, 0, 0)), m0.val);
    __m128 v1 = _mm_mul_ps(_mm_shuffle_ps(v.val, v.val, _MM_SHUFFLE(1, 1, 1, 1)), m1.val);
    __m128 v2 = _mm_mul_ps(_mm_shuffle_ps(v.val, v.val, _MM_SHUFFLE(2, 2, 2, 2)), m2.val);
    __m128 v3 = _mm_mul_ps(_mm_shuffle_ps(v.val, v.val, _MM_SHUFFLE(3, 3, 3, 3)), m3.val);

    return v_float32x4(_mm_add_ps(_mm_add_ps(v0, v1), _mm_add_ps(v2, v3)));
}


#define OPENCV_HAL_IMPL_SSE_BIN_OP(bin_op, _Tpvec, intrin) \
    inline _Tpvec operator bin_op (const _Tpvec& a, const _Tpvec& b) \
    { \
        return _Tpvec(intrin(a.val, b.val)); \
    } \
    inline _Tpvec& operator bin_op##= (_Tpvec& a, const _Tpvec& b) \
    { \
        a.val = intrin(a.val, b.val); \
        return a; \
    }

OPENCV_HAL_IMPL_SSE_BIN_OP(+, v_uint8x16, _mm_adds_epu8)
OPENCV_HAL_IMPL_SSE_BIN_OP(-, v_uint8x16, _mm_subs_epu8)
OPENCV_HAL_IMPL_SSE_BIN_OP(+, v_int8x16, _mm_adds_epi8)
OPENCV_HAL_IMPL_SSE_BIN_OP(-, v_int8x16, _mm_subs_epi8)
OPENCV_HAL_IMPL_SSE_BIN_OP(+, v_uint16x8, _mm_adds_epu16)
OPENCV_HAL_IMPL_SSE_BIN_OP(-, v_uint16x8, _mm_subs_epu16)
OPENCV_HAL_IMPL_SSE_BIN_OP(*, v_uint16x8, _mm_mullo_epi16)
OPENCV_HAL_IMPL_SSE_BIN_OP(+, v_int16x8, _mm_adds_epi16)
OPENCV_HAL_IMPL_SSE_BIN_OP(-, v_int16x8, _mm_subs_epi16)
OPENCV_HAL_IMPL_SSE_BIN_OP(*, v_int16x8, _mm_mullo_epi16)
OPENCV_HAL_IMPL_SSE_BIN_OP(+, v_int32x4, _mm_add_epi32)
OPENCV_HAL_IMPL_SSE_BIN_OP(-, v_int32x4, _mm_sub_epi32)
OPENCV_HAL_IMPL_SSE_BIN_OP(+, v_float32x4, _mm_add_ps)
OPENCV_HAL_IMPL_SSE_BIN_OP(-, v_float32x4, _mm_sub_ps)
OPENCV_HAL_IMPL_SSE_BIN_OP(*, v_float32x4, _mm_mul_ps)
OPENCV_HAL_IMPL_SSE_BIN_OP(/, v_float32x4, _mm_div_ps)
OPENCV_HAL_IMPL_SSE_BIN_OP(+, v_float64x2, _mm_add_pd)
OPENCV_HAL_IMPL_SSE_BIN_OP(-, v_float64x2, _mm_sub_pd)
OPENCV_HAL_IMPL_SSE_BIN_OP(*, v_float64x2, _mm_mul_pd)
OPENCV_HAL_IMPL_SSE_BIN_OP(/, v_float64x2, _mm_div_pd)

inline v_uint32x4 operator * (const v_uint32x4& a, const v_uint32x4& b)
{
    __m128i c0 = _mm_mul_epu32(a.val, b.val);
    __m128i c1 = _mm_mul_epu32(_mm_srli_epi64(a.val, 32), _mm_srli_epi64(b.val, 32));
    __m128i d0 = _mm_unpacklo_epi32(c0, c1);
    __m128i d1 = _mm_unpackhi_epi32(c0, c1);
    return v_uint32x4(_mm_unpacklo_epi64(d0, d1));
}
inline v_int32x4 operator * (const v_int32x4& a, const v_int32x4& b)
{
    __m128i c0 = _mm_mul_epu32(a.val, b.val);
    __m128i c1 = _mm_mul_epu32(_mm_srli_epi64(a.val, 32), _mm_srli_epi64(b.val, 32));
    __m128i d0 = _mm_unpacklo_epi32(c0, c1);
    __m128i d1 = _mm_unpackhi_epi32(c0, c1);
    return v_int32x4(_mm_unpacklo_epi64(d0, d1));
}

#define OPENCV_HAL_IMPL_SSE_LOGIC_OP(_Tpvec, suffix, not_const) \
    OPENCV_HAL_IMPL_SSE_BIN_OP(&, _Tpvec, _mm_and_##suffix) \
    OPENCV_HAL_IMPL_SSE_BIN_OP(|, _Tpvec, _mm_or_##suffix) \
    OPENCV_HAL_IMPL_SSE_BIN_OP(^, _Tpvec, _mm_xor_##suffix) \
    inline _Tpvec operator ~ (const _Tpvec& a) \
    { \
        return _Tpvec(_mm_xor_##suffix(a.val, not_const)); \
    }

OPENCV_HAL_IMPL_SSE_LOGIC_OP(v_uint8x16, si128, _mm_set1_epi32(-1))
OPENCV_HAL_IMPL_SSE_LOGIC_OP(v_int8x16, si128, _mm_set1_epi32(-1))
OPENCV_HAL_IMPL_SSE_LOGIC_OP(v_uint16x8, si128, _mm_set1_epi32(-1))
OPENCV_HAL_IMPL_SSE_LOGIC_OP(v_int16x8, si128, _mm_set1_epi32(-1))
OPENCV_HAL_IMPL_SSE_LOGIC_OP(v_uint32x4, si128, _mm_set1_epi32(-1))
OPENCV_HAL_IMPL_SSE_LOGIC_OP(v_int32x4, si128, _mm_set1_epi32(-1))
OPENCV_HAL_IMPL_SSE_LOGIC_OP(v_float32x4, ps, _mm_castsi128_ps(_mm_set1_epi32(-1)))
OPENCV_HAL_IMPL_SSE_LOGIC_OP(v_float64x2, pd, _mm_castsi128_pd(_mm_set1_epi32(-1)))

inline v_float32x4 v_sqrt(v_float32x4 x)
{ return v_float32x4(_mm_sqrt_ps(x.val)); }
inline v_float64x2 v_sqrt(v_float64x2 x)
{ return v_float64x2(_mm_sqrt_pd(x.val)); }

inline v_float32x4 v_abs(v_float32x4 x)
{ return v_float32x4(_mm_and_ps(x.val, _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff)))); }
inline v_float64x2 v_abs(v_float64x2 x)
{
    return v_float64x2(_mm_and_pd(x.val,
        _mm_castsi128_pd(_mm_srli_epi64(_mm_set1_epi32(-1), 1))));
}

// TODO: exp, log, sin, cos

#define OPENCV_HAL_IMPL_SSE_BIN_FUNC(_Tpvec, func, intrin) \
inline _Tpvec func(const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(intrin(a.val, b.val)); \
}

OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_uint8x16, v_min, _mm_min_epu8)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_uint8x16, v_max, _mm_max_epu8)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_int16x8, v_min, _mm_min_epi16)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_int16x8, v_max, _mm_max_epi16)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_float32x4, v_min, _mm_min_ps)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_float32x4, v_max, _mm_max_ps)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_float64x2, v_min, _mm_min_pd)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_float64x2, v_max, _mm_max_pd)

inline v_int8x16 v_min(const v_int8x16& a, const v_int8x16& b)
{
    __m128i delta = _mm_set1_epi8((char)0x80);
    return v_int8x16(_mm_xor_si128(delta, _mm_min_epu8(_mm_xor_si128(a.val, delta),
                                                       _mm_xor_si128(b.val, delta))));
}
inline v_int8x16 v_max(const v_int8x16& a, const v_int8x16& b)
{
    __m128i delta = _mm_set1_epi8((char)0x80);
    return v_int8x16(_mm_xor_si128(delta, _mm_max_epu8(_mm_xor_si128(a.val, delta),
                                                       _mm_xor_si128(b.val, delta))));
}
inline v_uint16x8 v_min(const v_uint16x8& a, const v_uint16x8& b)
{
    return v_uint16x8(_mm_subs_epu16(a.val, _mm_subs_epu16(a.val, b.val)));
}
inline v_uint16x8 v_max(const v_uint16x8& a, const v_uint16x8& b)
{
    return v_uint16x8(_mm_adds_epu16(_mm_subs_epu16(a.val, b.val), b.val));
}
inline v_uint32x4 v_min(const v_uint32x4& a, const v_uint32x4& b)
{
    __m128i delta = _mm_set1_epi32((int)0x80000000);
    __m128i mask = _mm_cmpgt_epi32(_mm_xor_si128(a.val, delta), _mm_xor_si128(b.val, delta));
    return v_uint32x4(v_select_si128(mask, b.val, a.val));
}
inline v_uint32x4 v_max(const v_uint32x4& a, const v_uint32x4& b)
{
    __m128i delta = _mm_set1_epi32((int)0x80000000);
    __m128i mask = _mm_cmpgt_epi32(_mm_xor_si128(a.val, delta), _mm_xor_si128(b.val, delta));
    return v_uint32x4(v_select_si128(mask, a.val, b.val));
}
inline v_int32x4 v_min(const v_int32x4& a, const v_int32x4& b)
{
    return v_int32x4(v_select_si128(_mm_cmpgt_epi32(a.val, b.val), b.val, a.val));
}
inline v_int32x4 v_max(const v_int32x4& a, const v_int32x4& b)
{
    return v_int32x4(v_select_si128(_mm_cmpgt_epi32(a.val, b.val), a.val, b.val));
}

#define OPENCV_HAL_IMPL_SSE_INT_CMP_OP(_Tpuvec, _Tpsvec, suffix, sbit) \
inline _Tpuvec operator == (const _Tpuvec& a, const _Tpuvec& b) \
{ return _Tpuvec(_mm_cmpeq_##suffix(a.val, b.val)); } \
inline _Tpuvec operator != (const _Tpuvec& a, const _Tpuvec& b) \
{ \
    __m128i not_mask = _mm_set1_epi32(-1); \
    return _Tpuvec(_mm_xor_si128(_mm_cmpeq_##suffix(a.val, b.val), not_mask)); \
} \
inline _Tpsvec operator == (const _Tpsvec& a, const _Tpsvec& b) \
{ return _Tpsvec(_mm_cmpeq_##suffix(a.val, b.val)); } \
inline _Tpsvec operator != (const _Tpsvec& a, const _Tpsvec& b) \
{ \
    __m128i not_mask = _mm_set1_epi32(-1); \
    return _Tpsvec(_mm_xor_si128(_mm_cmpeq_##suffix(a.val, b.val), not_mask)); \
} \
inline _Tpuvec operator < (const _Tpuvec& a, const _Tpuvec& b) \
{ \
    __m128i smask = _mm_set1_##suffix(sbit); \
    return _Tpuvec(_mm_cmpgt_##suffix(_mm_xor_si128(b.val, smask), _mm_xor_si128(a.val, smask))); \
} \
inline _Tpuvec operator > (const _Tpuvec& a, const _Tpuvec& b) \
{ \
    __m128i smask = _mm_set1_##suffix(sbit); \
    return _Tpuvec(_mm_cmpgt_##suffix(_mm_xor_si128(a.val, smask), _mm_xor_si128(b.val, smask))); \
} \
inline _Tpuvec operator <= (const _Tpuvec& a, const _Tpuvec& b) \
{ \
    __m128i smask = _mm_set1_##suffix(sbit); \
    __m128i not_mask = _mm_set1_epi32(-1); \
    __m128i res = _mm_cmpgt_##suffix(_mm_xor_si128(a.val, smask), _mm_xor_si128(b.val, smask)); \
    return _Tpuvec(_mm_xor_si128(res, not_mask)); \
} \
inline _Tpuvec operator >= (const _Tpuvec& a, const _Tpuvec& b) \
{ \
    __m128i smask = _mm_set1_##suffix(sbit); \
    __m128i not_mask = _mm_set1_epi32(-1); \
    __m128i res = _mm_cmpgt_##suffix(_mm_xor_si128(b.val, smask), _mm_xor_si128(a.val, smask)); \
    return _Tpuvec(_mm_xor_si128(res, not_mask)); \
} \
inline _Tpsvec operator < (const _Tpsvec& a, const _Tpsvec& b) \
{ \
    return _Tpsvec(_mm_cmpgt_##suffix(b.val, a.val)); \
} \
inline _Tpsvec operator > (const _Tpsvec& a, const _Tpsvec& b) \
{ \
    return _Tpsvec(_mm_cmpgt_##suffix(a.val, b.val)); \
} \
inline _Tpsvec operator <= (const _Tpsvec& a, const _Tpsvec& b) \
{ \
    __m128i not_mask = _mm_set1_epi32(-1); \
    return _Tpsvec(_mm_xor_si128(_mm_cmpgt_##suffix(a.val, b.val), not_mask)); \
} \
inline _Tpsvec operator >= (const _Tpsvec& a, const _Tpsvec& b) \
{ \
    __m128i not_mask = _mm_set1_epi32(-1); \
    return _Tpsvec(_mm_xor_si128(_mm_cmpgt_##suffix(b.val, a.val), not_mask)); \
}

OPENCV_HAL_IMPL_SSE_INT_CMP_OP(v_uint8x16, v_int8x16, epi8, (char)0x80)
OPENCV_HAL_IMPL_SSE_INT_CMP_OP(v_uint16x8, v_int16x8, epi16, (short)0x8000)
OPENCV_HAL_IMPL_SSE_INT_CMP_OP(v_uint32x4, v_int32x4, epi32, (int)0x80000000)

#define OPENCV_HAL_IMPL_SSE_FLT_CMP_OP(_Tpvec, suffix) \
inline _Tpvec operator == (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(_mm_cmpeq_##suffix(a.val, b.val)); } \
inline _Tpvec operator != (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(_mm_cmpneq_##suffix(a.val, b.val)); } \
inline _Tpvec operator < (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(_mm_cmplt_##suffix(a.val, b.val)); } \
inline _Tpvec operator > (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(_mm_cmpgt_##suffix(a.val, b.val)); } \
inline _Tpvec operator <= (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(_mm_cmple_##suffix(a.val, b.val)); } \
inline _Tpvec operator >= (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(_mm_cmpge_##suffix(a.val, b.val)); }

OPENCV_HAL_IMPL_SSE_FLT_CMP_OP(v_float32x4, ps)
OPENCV_HAL_IMPL_SSE_FLT_CMP_OP(v_float64x2, pd)

OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_uint8x16, v_add_wrap, _mm_add_epi8)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_int8x16, v_add_wrap, _mm_add_epi8)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_uint16x8, v_add_wrap, _mm_add_epi16)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_int16x8, v_add_wrap, _mm_add_epi16)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_uint8x16, v_sub_wrap, _mm_sub_epi8)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_int8x16, v_sub_wrap, _mm_sub_epi8)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_uint16x8, v_sub_wrap, _mm_sub_epi16)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_int16x8, v_sub_wrap, _mm_sub_epi16)

#define OPENCV_HAL_IMPL_SSE_ABSDIFF_8_16(_Tpuvec, _Tpsvec, bits, sbit) \
inline _Tpuvec v_absdiff(const _Tpuvec& a, const _Tpuvec& b) \
{ \
    return _Tpuvec(_mm_add_epi##bits(_mm_subs_epu##bits(a.val, b.val), _mm_subs_epu##bits(b.val, a.val))); \
} \
inline _Tpuvec v_absdiff(const _Tpsvec& a, const _Tpsvec& b) \
{ \
    __m128i smask = _mm_set1_epi8(sbit); \
    __m128i a1 = _mm_xor_si128(a.val, smask); \
    __m128i b1 = _mm_xor_si128(b.val, smask); \
    return _Tpuvec(_mm_add_epi##bits(_mm_subs_epu##bits(a1, b1), _mm_subs_epu##bits(b1, a1))); \
}

OPENCV_HAL_IMPL_SSE_ABSDIFF_8_16(v_uint8x16, v_int8x16, 8, (char)0x80)
OPENCV_HAL_IMPL_SSE_ABSDIFF_8_16(v_uint16x8, v_int16x8, 16, (char)0x8000)

#define OPENCV_HAL_IMPL_SSE_MISC_FLT_OP(_Tpvec, _Tp, _Tpreg, suffix, absmask_vec) \
inline _Tpvec v_absdiff(const _Tpvec& a, const _Tpvec& b) \
{ \
    _Tpreg absmask = _mm_castsi128_##suffix(absmask_vec); \
    return _Tpvec(_mm_and_##suffix(_mm_sub_##suffix(a.val, b.val), absmask)); \
} \
inline _Tpvec v_magnitude(const _Tpvec& a, const _Tpvec& b) \
{ \
    _Tpreg res = _mm_add_##suffix(_mm_mul_##suffix(a.val, a.val), _mm_mul_##suffix(b.val, b.val)); \
    return _Tpvec(_mm_sqrt_##suffix(res)); \
} \
inline _Tpvec v_sqr_magnitude(const _Tpvec& a, const _Tpvec& b) \
{ \
    _Tpreg res = _mm_add_##suffix(_mm_mul_##suffix(a.val, a.val), _mm_mul_##suffix(b.val, b.val)); \
    return _Tpvec(res); \
} \
inline _Tpvec v_muladd(const _Tpvec& a, const _Tpvec& b, const _Tpvec& c) \
{ \
    return _Tpvec(_mm_add_##suffix(_mm_mul_##suffix(a.val, b.val), c.val)); \
}

OPENCV_HAL_IMPL_SSE_MISC_FLT_OP(v_float32x4, float, __m128, ps, _mm_set1_epi32((int)0x7fffffff))
OPENCV_HAL_IMPL_SSE_MISC_FLT_OP(v_float64x2, double, __m128d, pd, _mm_srli_epi64(_mm_set1_epi32(-1), 1))

#define OPENCV_HAL_IMPL_SSE_SHIFT_OP(_Tpuvec, _Tpsvec, suffix) \
inline _Tpuvec operator << (const _Tpuvec& a, int imm) \
{ \
    return _Tpuvec(_mm_slli_##suffix(a.val, imm)); \
} \
inline _Tpsvec operator << (const _Tpsvec& a, int imm) \
{ \
    return _Tpsvec(_mm_slli_##suffix(a.val, imm)); \
} \
inline _Tpuvec operator >> (const _Tpuvec& a, int imm) \
{ \
    return _Tpuvec(_mm_srli_##suffix(a.val, imm)); \
} \
inline _Tpsvec operator >> (const _Tpsvec& a, int imm) \
{ \
    return _Tpsvec(_mm_srai_##suffix(a.val, imm)); \
}

OPENCV_HAL_IMPL_SSE_SHIFT_OP(v_uint16x8, v_int16x8, epi16)
OPENCV_HAL_IMPL_SSE_SHIFT_OP(v_uint32x4, v_int32x4, epi32)

inline v_int16x8 v_mullo(const v_int16x8& a, const v_int16x8& b)
{
    return v_int16x8(_mm_mullo_epi16(a.val, b.val));
}
inline v_uint16x8 v_mullo(const v_uint16x8& a, const v_uint16x8& b)
{
    return v_uint16x8(_mm_mullo_epi16(a.val, b.val));
}
inline v_int16x8 v_mulhi2(const v_int16x8& a, const v_int16x8& b)
{
    return v_int16x8(_mm_slli_epi16(_mm_mulhi_epi16(a.val, b.val), 1));
}
inline v_uint16x8 v_mulhi2(const v_uint16x8& a, const v_uint16x8& b)
{
    return v_uint16x8(_mm_slli_epi16(_mm_mulhi_epu16(a.val, b.val), 1));
}

#define OPENCV_HAL_IMPL_SSE_LOADSTORE_INT_OP(_Tpvec, _Tp) \
inline _Tpvec v_load(const _Tp* ptr) \
{ return _Tpvec(_mm_loadu_si128((const __m128i*)ptr)); } \
inline _Tpvec v_load_aligned(const _Tp* ptr) \
{ return _Tpvec(_mm_load_si128((const __m128i*)ptr)); } \
inline _Tpvec v_load_halves(const _Tp* ptr0, const _Tp* ptr1) \
{ \
    return _Tpvec(_mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)ptr0), \
                                     _mm_loadl_epi64((const __m128i*)ptr1))); \
} \
inline void v_store(_Tp* ptr, const _Tpvec& a) \
{ _mm_storeu_si128((__m128i*)ptr, a.val); } \
inline void v_store_aligned(_Tp* ptr, const _Tpvec& a) \
{ _mm_store_si128((__m128i*)ptr, a.val); } \
inline void v_store_low(_Tp* ptr, const _Tpvec& a) \
{ _mm_storel_epi64((__m128i*)ptr, a.val); } \
inline void v_store_high(_Tp* ptr, const _Tpvec& a) \
{ _mm_storel_epi64((__m128i*)ptr, _mm_unpackhi_epi64(a.val, a.val)); }

OPENCV_HAL_IMPL_SSE_LOADSTORE_INT_OP(v_uint8x16, uchar)
OPENCV_HAL_IMPL_SSE_LOADSTORE_INT_OP(v_int8x16, schar)
OPENCV_HAL_IMPL_SSE_LOADSTORE_INT_OP(v_uint16x8, ushort)
OPENCV_HAL_IMPL_SSE_LOADSTORE_INT_OP(v_int16x8, short)
OPENCV_HAL_IMPL_SSE_LOADSTORE_INT_OP(v_uint32x4, unsigned)
OPENCV_HAL_IMPL_SSE_LOADSTORE_INT_OP(v_int32x4, int)

#define OPENCV_HAL_IMPL_SSE_LOADSTORE_FLT_OP(_Tpvec, _Tp, suffix) \
inline _Tpvec v_load(const _Tp* ptr) \
{ return _Tpvec(_mm_loadu_##suffix(ptr)); } \
inline _Tpvec v_load_aligned(const _Tp* ptr) \
{ return _Tpvec(_mm_load_##suffix(ptr)); } \
inline _Tpvec v_load_halves(const _Tp* ptr0, const _Tp* ptr1) \
{ \
    return _Tpvec(_mm_castsi128_##suffix( \
        _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)ptr0), \
                           _mm_loadl_epi64((const __m128i*)ptr1)))); \
} \
inline void v_store(_Tp* ptr, const _Tpvec& a) \
{ _mm_storeu_##suffix(ptr, a.val); } \
inline void v_store_aligned(_Tp* ptr, const _Tpvec& a) \
{ _mm_store_##suffix(ptr, a.val); } \
inline void v_store_low(_Tp* ptr, const _Tpvec& a) \
{ _mm_storel_epi64((__m128i*)ptr, _mm_cast##suffix##_si128(a.val)); } \
inline void v_store_high(_Tp* ptr, const _Tpvec& a) \
{ \
    __m128i a1 = _mm_cast##suffix##_si128(a.val); \
    _mm_storel_epi64((__m128i*)ptr, _mm_unpackhi_epi64(a1, a1)); \
}

OPENCV_HAL_IMPL_SSE_LOADSTORE_FLT_OP(v_float32x4, float, ps)
OPENCV_HAL_IMPL_SSE_LOADSTORE_FLT_OP(v_float64x2, double, pd)

#define OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(_Tpvec, scalartype, func, scalar_func) \
inline scalartype v_reduce_##func(const _Tpvec& a) \
{ \
    scalartype CV_DECL_ALIGNED(16) buf[4]; \
    v_store_aligned(buf, a); \
    scalartype s0 = scalar_func(buf[0], buf[1]); \
    scalartype s1 = scalar_func(buf[2], buf[3]); \
    return scalar_func(s0, s1); \
}

OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_uint32x4, unsigned, sum, OPENCV_HAL_ADD)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_uint32x4, unsigned, max, std::max)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_uint32x4, unsigned, min, std::min)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_int32x4, int, sum, OPENCV_HAL_ADD)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_int32x4, int, max, std::max)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_int32x4, int, min, std::min)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_float32x4, float, sum, OPENCV_HAL_ADD)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_float32x4, float, max, std::max)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_float32x4, float, min, std::min)

#define OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(_Tpvec, suffix, pack_op, and_op, signmask, allmask) \
inline bool v_signmask(const _Tpvec& a) \
{ \
    return and_op(_mm_movemask_##suffix(pack_op(a.val)), signmask); \
} \
inline bool v_check_all(const _Tpvec& a) \
{ return and_op(_mm_movemask_##suffix(a.val), allmask) == allmask; } \
inline bool v_check_any(const _Tpvec& a) \
{ return and_op(_mm_movemask_##suffix(a.val), allmask) != 0; }

#define OPENCV_HAL_PACKS(a) _mm_packs_epi16(a, a)
inline __m128i v_packq_epi32(__m128i a)
{
    __m128i b = _mm_packs_epi32(a, a);
    return _mm_packs_epi16(b, b);
}

OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_uint8x16, epi8, OPENCV_HAL_NOP, OPENCV_HAL_1ST, 65535, 65535)
OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_int8x16, epi8, OPENCV_HAL_NOP, OPENCV_HAL_1ST, 65535, 65535)
OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_uint16x8, epi8, OPENCV_HAL_PACKS, OPENCV_HAL_AND, 255, (int)0xaaaa)
OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_int16x8, epi8, OPENCV_HAL_PACKS, OPENCV_HAL_AND, 255, (int)0xaaaa)
OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_uint32x4, epi8, v_packq_epi32, OPENCV_HAL_AND, 15, (int)0x8888)
OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_int32x4, epi8, v_packq_epi32, OPENCV_HAL_AND, 15, (int)0x8888)
OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_float32x4, ps, OPENCV_HAL_NOP, OPENCV_HAL_1ST, 15, 15)
OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_float64x2, pd, OPENCV_HAL_NOP, OPENCV_HAL_1ST, 3, 3)

#define OPENCV_HAL_IMPL_SSE_SELECT(_Tpvec, suffix) \
inline _Tpvec v_select(const _Tpvec& mask, const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(_mm_xor_##suffix(b.val, _mm_and_##suffix(_mm_xor_##suffix(b.val, a.val), mask.val))); \
}

OPENCV_HAL_IMPL_SSE_SELECT(v_uint8x16, si128)
OPENCV_HAL_IMPL_SSE_SELECT(v_int8x16, si128)
OPENCV_HAL_IMPL_SSE_SELECT(v_uint16x8, si128)
OPENCV_HAL_IMPL_SSE_SELECT(v_int16x8, si128)
OPENCV_HAL_IMPL_SSE_SELECT(v_uint32x4, si128)
OPENCV_HAL_IMPL_SSE_SELECT(v_int32x4, si128)
OPENCV_HAL_IMPL_SSE_SELECT(v_float32x4, ps)
OPENCV_HAL_IMPL_SSE_SELECT(v_float64x2, pd)

#define OPENCV_HAL_IMPL_SSE_EXPAND(_Tpuvec, _Tpwuvec, _Tpu, _Tpsvec, _Tpwsvec, _Tps, suffix, wsuffix, shift) \
inline void v_expand(const _Tpuvec& a, _Tpwuvec& b0, _Tpwuvec& b1) \
{ \
    __m128i z = _mm_setzero_si128(); \
    b0.val = _mm_unpacklo_##suffix(a.val, z); \
    b1.val = _mm_unpackhi_##suffix(a.val, z); \
} \
inline _Tpwuvec v_load_expand(const _Tpu* ptr) \
{ \
    __m128i z = _mm_setzero_si128(); \
    return _Tpwuvec(_mm_unpacklo_##suffix(_mm_loadl_epi64((const __m128i*)ptr), z)); \
} \
inline void v_expand(const _Tpsvec& a, _Tpwsvec& b0, _Tpwsvec& b1) \
{ \
    b0.val = _mm_srai_##wsuffix(_mm_unpacklo_##suffix(a.val, a.val), shift); \
    b1.val = _mm_srai_##wsuffix(_mm_unpackhi_##suffix(a.val, a.val), shift); \
} \
inline _Tpwsvec v_load_expand(const _Tps* ptr) \
{ \
    __m128i a = _mm_loadl_epi64((const __m128i*)ptr); \
    return _Tpwsvec(_mm_srai_##wsuffix(_mm_unpacklo_##suffix(a, a), shift)); \
}

OPENCV_HAL_IMPL_SSE_EXPAND(v_uint8x16, v_uint16x8, uchar, v_int8x16, v_int16x8, schar, epi8, epi16, 8)
OPENCV_HAL_IMPL_SSE_EXPAND(v_uint16x8, v_uint32x4, ushort, v_int16x8, v_int32x4, short, epi16, epi32, 16)

inline v_uint32x4 v_load_expand_q(const uchar* ptr)
{
    __m128i z = _mm_setzero_si128();
    __m128i a = _mm_cvtsi32_si128(*(const int*)ptr);
    return v_uint32x4(_mm_unpacklo_epi16(_mm_unpacklo_epi8(a, z), z));
}

inline v_int32x4 v_load_expand_q(const schar* ptr)
{
    __m128i a = _mm_cvtsi32_si128(*(const int*)ptr);
    a = _mm_unpacklo_epi8(a, a);
    a = _mm_unpacklo_epi8(a, a);
    return v_int32x4(_mm_srai_epi32(a, 24));
}

#define OPENCV_HAL_IMPL_SSE_UNPACKS(_Tpvec, suffix, cast_from, cast_to) \
inline void v_zip(const _Tpvec& a0, const _Tpvec& a1, _Tpvec& b0, _Tpvec& b1) \
{ \
    b0.val = _mm_unpacklo_##suffix(a0.val, a1.val); \
    b1.val = _mm_unpackhi_##suffix(a0.val, a1.val); \
} \
inline _Tpvec v_combine_low(const _Tpvec& a, const _Tpvec& b) \
{ \
    __m128i a1 = cast_from(a.val), b1 = cast_from(b.val); \
    return _Tpvec(cast_to(_mm_unpacklo_epi64(a1, b1))); \
} \
inline _Tpvec v_combine_high(const _Tpvec& a, const _Tpvec& b) \
{ \
    __m128i a1 = cast_from(a.val), b1 = cast_from(b.val); \
    return _Tpvec(cast_to(_mm_unpackhi_epi64(a1, b1))); \
} \
inline void v_recombine(const _Tpvec& a, const _Tpvec& b, _Tpvec& c, _Tpvec& d) \
{ \
    __m128i a1 = cast_from(a.val), b1 = cast_from(b.val); \
    c.val = cast_to(_mm_unpacklo_epi64(a1, b1)); \
    d.val = cast_to(_mm_unpackhi_epi64(a1, b1)); \
}

OPENCV_HAL_IMPL_SSE_UNPACKS(v_uint8x16, epi8, OPENCV_HAL_NOP, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_UNPACKS(v_int8x16, epi8, OPENCV_HAL_NOP, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_UNPACKS(v_uint16x8, epi16, OPENCV_HAL_NOP, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_UNPACKS(v_int16x8, epi16, OPENCV_HAL_NOP, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_UNPACKS(v_uint32x4, epi32, OPENCV_HAL_NOP, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_UNPACKS(v_int32x4, epi32, OPENCV_HAL_NOP, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_UNPACKS(v_float32x4, ps, _mm_castps_si128, _mm_castsi128_ps)
OPENCV_HAL_IMPL_SSE_UNPACKS(v_float64x2, pd, _mm_castpd_si128, _mm_castsi128_pd)

inline v_int32x4 v_round(const v_float32x4& a)
{ return v_int32x4(_mm_cvtps_epi32(a.val)); }

inline v_int32x4 v_floor(const v_float32x4& a)
{
    __m128i a1 = _mm_cvtps_epi32(a.val);
    __m128i mask = _mm_castps_si128(_mm_cmpgt_ps(_mm_cvtepi32_ps(a1), a.val));
    return v_int32x4(_mm_add_epi32(a1, mask));
}

inline v_int32x4 v_ceil(const v_float32x4& a)
{
    __m128i a1 = _mm_cvtps_epi32(a.val);
    __m128i mask = _mm_castps_si128(_mm_cmpgt_ps(a.val, _mm_cvtepi32_ps(a1)));
    return v_int32x4(_mm_sub_epi32(a1, mask));
}

inline v_int32x4 v_trunc(const v_float32x4& a)
{ return v_int32x4(_mm_cvttps_epi32(a.val)); }

inline v_int32x4 v_round(const v_float64x2& a)
{ return v_int32x4(_mm_cvtpd_epi32(a.val)); }

inline v_int32x4 v_floor(const v_float64x2& a)
{
    __m128i a1 = _mm_cvtpd_epi32(a.val);
    __m128i mask = _mm_castpd_si128(_mm_cmpgt_pd(_mm_cvtepi32_pd(a1), a.val));
    mask = _mm_srli_si128(_mm_slli_si128(mask, 4), 8); // m0 m0 m1 m1 => m0 m1 0 0
    return v_int32x4(_mm_add_epi32(a1, mask));
}

inline v_int32x4 v_ceil(const v_float64x2& a)
{
    __m128i a1 = _mm_cvtpd_epi32(a.val);
    __m128i mask = _mm_castpd_si128(_mm_cmpgt_pd(a.val, _mm_cvtepi32_pd(a1)));
    mask = _mm_srli_si128(_mm_slli_si128(mask, 4), 8); // m0 m0 m1 m1 => m0 m1 0 0
    return v_int32x4(_mm_sub_epi32(a1, mask));
}

inline v_int32x4 v_trunc(const v_float64x2& a)
{ return v_int32x4(_mm_cvttpd_epi32(a.val)); }

#define OPENCV_HAL_IMPL_SSE_TRANSPOSE4x4(_Tpvec, suffix, cast_from, cast_to) \
inline void v_transpose4x4(const _Tpvec& a0, const _Tpvec& a1, \
                           const _Tpvec& a2, const _Tpvec& a3, \
                           _Tpvec& b0, _Tpvec& b1, \
                           _Tpvec& b2, _Tpvec& b3) \
{ \
    __m128i t0 = cast_from(_mm_unpacklo_##suffix(a0.val, a1.val)); \
    __m128i t1 = cast_from(_mm_unpacklo_##suffix(a2.val, a3.val)); \
    __m128i t2 = cast_from(_mm_unpackhi_##suffix(a0.val, a1.val)); \
    __m128i t3 = cast_from(_mm_unpacklo_##suffix(a2.val, a3.val)); \
\
    b0.val = cast_to(_mm_unpacklo_epi64(t0, t1)); \
    b1.val = cast_to(_mm_unpackhi_epi64(t0, t1)); \
    b2.val = cast_to(_mm_unpacklo_epi64(t2, t3)); \
    b3.val = cast_to(_mm_unpackhi_epi64(t2, t3)); \
}

OPENCV_HAL_IMPL_SSE_TRANSPOSE4x4(v_uint32x4, epi32, OPENCV_HAL_NOP, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_TRANSPOSE4x4(v_int32x4, epi32, OPENCV_HAL_NOP, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_TRANSPOSE4x4(v_float32x4, ps, _mm_castps_si128, _mm_castsi128_ps)

#if 0
inline void v_load_deinterleave(const uchar*, v_uint8x16&, v_uint8x16&, v_uint8x16&)
{
    // !!! TODO !!!
}
#endif

inline void v_load_deinterleave(const uchar* ptr, v_uint8x16& a, v_uint8x16& b, v_uint8x16& c, v_uint8x16& d)
{
    __m128i u0 = _mm_loadu_si128((const __m128i*)ptr); // a0 b0 c0 d0 a1 b1 c1 d1 ...
    __m128i u1 = _mm_loadu_si128((const __m128i*)(ptr + 16)); // a4 b4 c4 d4 ...
    __m128i u2 = _mm_loadu_si128((const __m128i*)(ptr + 32)); // a8 b8 c8 d8 ...
    __m128i u3 = _mm_loadu_si128((const __m128i*)(ptr + 48)); // a12 b12 c12 d12 ...

    __m128 v0 = _mm_unpacklo_epi8(u0, u2); // a0 a8 b0 b8 ...
    __m128 v1 = _mm_unpackhi_epi8(u0, u2); // a2 a10 b2 b10 ...
    __m128 v2 = _mm_unpacklo_epi8(u1, u3); // a4 a12 b4 b12 ...
    __m128 v3 = _mm_unpackhi_epi8(u1, u3); // a6 a14 b4 b14 ...

    u0 = _mm_unpacklo_epi8(v0, v2); // a0 a4 a8 a12 ...
    u1 = _mm_unpacklo_epi8(v1, v3); // a2 a6 a10 a14 ...
    u2 = _mm_unpackhi_epi8(v0, v2); // a1 a5 a9 a13 ...
    u3 = _mm_unpackhi_epi8(v1, v3); // a3 a7 a11 a15 ...

    v0 = _mm_unpacklo_epi8(u0, u1); // a0 a2 a4 a6 ...
    v1 = _mm_unpacklo_epi8(u2, u3); // a1 a3 a5 a7 ...
    v2 = _mm_unpackhi_epi8(u0, u1); // b0 b2 b4 b6 ...
    v3 = _mm_unpackhi_epi8(u2, u3); // b1 b3 b5 b7 ...

    a.val = _mm_unpacklo_epi8(v0, v1);
    b.val = _mm_unpacklo_epi8(v2, v3);
    c.val = _mm_unpackhi_epi8(v0, v1);
    d.val = _mm_unpacklo_epi8(v2, v3);
}

#if 0
inline void v_load_deinterleave(const ushort*, v_uint16x8&, v_uint16x8&, v_uint16x8&)
{
    // !!! TODO !!!
}
#endif

inline void v_load_deinterleave(const ushort* ptr, v_uint16x8& a, v_uint16x8& b, v_uint16x8& c, v_uint16x8& d)
{
    __m128i u0 = _mm_loadu_si128((const __m128i*)ptr); // a0 b0 c0 d0 a1 b1 c1 d1
    __m128i u1 = _mm_loadu_si128((const __m128i*)(ptr + 8)); // a2 b2 c2 d2 ...
    __m128i u2 = _mm_loadu_si128((const __m128i*)(ptr + 16)); // a4 b4 c4 d4 ...
    __m128i u3 = _mm_loadu_si128((const __m128i*)(ptr + 24)); // a6 b6 c6 d6 ...

    __m128 v0 = _mm_unpacklo_epi16(u0, u2); // a0 a4 b0 b4 ...
    __m128 v1 = _mm_unpackhi_epi16(u0, u2); // a1 a5 b1 b5 ...
    __m128 v2 = _mm_unpacklo_epi16(u1, u3); // a2 a6 b2 b6 ...
    __m128 v3 = _mm_unpackhi_epi16(u1, u3); // a3 a7 b3 b7 ...

    u0 = _mm_unpacklo_epi16(v0, v2); // a0 a2 a4 a6 ...
    u1 = _mm_unpacklo_epi16(v1, v3); // a1 a3 a5 a7 ...
    u2 = _mm_unpackhi_epi16(v0, v2); // c0 c2 c4 c6 ...
    u3 = _mm_unpackhi_epi16(v1, v3); // c1 c3 c5 c7 ...

    a.val = _mm_unpacklo_epi16(u0, u1);
    b.val = _mm_unpackhi_epi16(u0, u1);
    c.val = _mm_unpacklo_epi16(u2, u3);
    d.val = _mm_unpackhi_epi16(u2, u3);
}

#if 0
inline void v_load_deinterleave(const unsigned*, v_uint32x4&, v_uint32x4&, v_uint32x4&)
{
    // !!! TODO !!!
}
#endif

inline void v_load_deinterleave(const unsigned* ptr, v_uint32x4& a, v_uint32x4& b, v_uint32x4& c, v_uint32x4& d)
{
    v_uint32x4 u0(_mm_loadu_si128((const __m128i*)ptr));        // a0 b0 c0 d0
    v_uint32x4 u1(_mm_loadu_si128((const __m128i*)(ptr + 4))); // a1 b1 c1 d1
    v_uint32x4 u2(_mm_loadu_si128((const __m128i*)(ptr + 8))); // a2 b2 c2 d2
    v_uint32x4 u3(_mm_loadu_si128((const __m128i*)(ptr + 12))); // a3 b3 c3 d3

    v_transpose4x4(u0, u1, u2, u3, a, b, c, d);
}

inline void v_load_deinterleave(const float*, v_float32x4&, v_float32x4&, v_float32x4&)
{
    // !!! TODO !!!
}

inline void v_load_deinterleave(const float* ptr, v_float32x4& a, v_float32x4& b, v_float32x4& c, v_float32x4& d)
{
    v_float32x4 u0(_mm_loadu_ps(ptr));
    v_float32x4 u1(_mm_loadu_ps(ptr + 4));
    v_float32x4 u2(_mm_loadu_ps(ptr + 8));
    v_float32x4 u3(_mm_loadu_ps(ptr + 12));

    v_transpose4x4(u0, u1, u2, u3, a, b, c, d);
}

inline void v_store_interleave( uchar* ptr, const v_uint8x16& a, const v_uint8x16& b,
                                const v_uint8x16& c, const v_uint8x16& d)
{
    // a0 a1 a2 a3 ....
    // b0 b1 b2 b3 ....
    // c0 c1 c2 c3 ....
    // d0 d1 d2 d3 ....
    __m128i u0 = _mm_unpacklo_epi8(a.val, c.val); // a0 c0 a1 c1 ...
    __m128i u1 = _mm_unpackhi_epi8(a.val, c.val); // a8 c8 a9 c9 ...
    __m128i u2 = _mm_unpacklo_epi8(b.val, d.val); // b0 d0 b1 d1 ...
    __m128i u3 = _mm_unpackhi_epi8(b.val, d.val); // b8 d8 b9 d9 ...

    __m128i v0 = _mm_unpacklo_epi8(u0, u2); // a0 b0 c0 d0 ...
    __m128i v1 = _mm_unpacklo_epi8(u1, u3); // a8 b8 c8 d8 ...
    __m128i v2 = _mm_unpackhi_epi8(u0, u2); // a4 b4 c4 d4 ...
    __m128i v3 = _mm_unpackhi_epi8(u1, u3); // a12 b12 c12 d12 ...

    _mm_storeu_si128((__m128i*)ptr, v0);
    _mm_storeu_si128((__m128i*)(ptr + 16), v2);
    _mm_storeu_si128((__m128i*)(ptr + 32), v1);
    _mm_storeu_si128((__m128i*)(ptr + 48), v3);
}

inline void v_store_interleave( ushort* ptr, const v_uint16x8& a, const v_uint16x8& b,
                                const v_uint16x8& c, const v_uint16x8& d)
{
    // a0 a1 a2 a3 ....
    // b0 b1 b2 b3 ....
    // c0 c1 c2 c3 ....
    // d0 d1 d2 d3 ....
    __m128i u0 = _mm_unpacklo_epi16(a.val, c.val); // a0 c0 a1 c1 ...
    __m128i u1 = _mm_unpackhi_epi16(a.val, c.val); // a4 c4 a5 c5 ...
    __m128i u2 = _mm_unpacklo_epi16(b.val, d.val); // b0 d0 b1 d1 ...
    __m128i u3 = _mm_unpackhi_epi16(b.val, d.val); // b4 d4 b5 d5 ...

    __m128i v0 = _mm_unpacklo_epi16(u0, u2); // a0 b0 c0 d0 ...
    __m128i v1 = _mm_unpacklo_epi16(u1, u3); // a4 b4 c4 d4 ...
    __m128i v2 = _mm_unpackhi_epi16(u0, u2); // a2 b2 c2 d2 ...
    __m128i v3 = _mm_unpackhi_epi16(u1, u3); // a6 b6 c6 d6 ...

    _mm_storeu_si128((__m128i*)ptr, v0);
    _mm_storeu_si128((__m128i*)(ptr + 8), v2);
    _mm_storeu_si128((__m128i*)(ptr + 16), v1);
    _mm_storeu_si128((__m128i*)(ptr + 24), v3);
}

inline v_float32x4 v_cvt_f32(const v_int32x4& a)
{
    return v_float32x4(_mm_cvtepi32_ps(a.val));
}

inline v_float32x4 v_cvt_f32(const v_float64x2& a)
{
    return v_float32x4(_mm_cvtpd_ps(a.val));
}

inline v_float64x2 v_cvt_f64(const v_int32x4& a)
{
    return v_float64x2(_mm_cvtepi32_pd(a.val));
}

inline v_float64x2 v_cvt_f64(const v_float32x4& a)
{
    return v_float64x2(_mm_cvtps_pd(a.val));
}

#elif CV_NEON

#define CV_SIMD128 1

struct v_uint8x16
{
    uint8x16_t val;
};

struct v_int8x16
{
    int8x16_t val;
};

struct v_uint16x8
{
    uint16x8_t val;
};

struct v_int16x8
{
    int16x8_t val;
};

struct v_uint32x4
{
    uint32x4_t val;
};

struct v_int32x4
{
    int32x4_t val;
};

struct v_float32x4
{
    float32x4_t val;
};

typedef v_reg<double, 2> v_float64x2;
typedef v_reg<double, 4> v_float64x4;

#else

typedef v_reg<uchar, 16> v_uint8x16;
typedef v_reg<schar, 16> v_int8x16;
typedef v_reg<ushort, 8> v_uint16x8;
typedef v_reg<short, 8> v_int16x8;
typedef v_reg<unsigned, 4> v_uint32x4;
typedef v_reg<int, 4> v_int32x4;
typedef v_reg<float, 4> v_float32x4;
typedef v_reg<float, 8> v_float32x8;
typedef v_reg<double, 2> v_float64x2;
typedef v_reg<double, 4> v_float64x4;

inline v_uint8x16 v_setzero_u8() { return v_uint8x16::zero(); }
inline v_int8x16 v_setzero_s8() { return v_int8x16::zero(); }
inline v_uint16x8 v_setzero_u16() { return v_uint16x8::zero(); }
inline v_int16x8 v_setzero_s16() { return v_int16x8::zero(); }
inline v_uint32x4 v_setzero_u32() { return v_uint32x4::zero(); }
inline v_int32x4 v_setzero_s32() { return v_int32x4::zero(); }
inline v_float32x4 v_setzero_f32() { return v_float32x4::zero(); }
inline v_float64x2 v_setzero_f64() { return v_float64x2::zero(); }

inline v_uint8x16 v_setall_u8(uchar v) { return v_uint8x16::all(v); }
inline v_int8x16 v_setall_s8(schar v) { return v_int8x16::all(v); }
inline v_uint16x8 v_setall_u16(ushort v) { return v_uint16x8::all(v); }
inline v_int16x8 v_setall_s16(short v) { return v_int16x8::all(v); }
inline v_uint32x4 v_setall_u32(unsigned v) { return v_uint32x4::all(v); }
inline v_int32x4 v_setall_s32(int v) { return v_int32x4::all(v); }
inline v_float32x4 v_setall_f32(float v) { return v_float32x4::all(v); }
inline v_float64x2 v_setall_f64(double v) { return v_float64x2::all(v); }

template<typename _Tp, int n> inline v_uint8x16 v_reinterpret_u8(const v_reg<_Tp, n>& a)
{ return v_reg<_Tp, n>::template reinterpret_as<uchar, 16>(a); }

template<typename _Tp, int n> inline v_int8x16 v_reinterpret_s8(const v_reg<_Tp, n>& a)
{ return v_reg<_Tp, n>::template reinterpret_as<schar, 16>(a); }

template<typename _Tp, int n> inline v_uint16x8 v_reinterpret_u16(const v_reg<_Tp, n>& a)
{ return v_reg<_Tp, n>::template reinterpret_as<ushort, 8>(a); }

template<typename _Tp, int n> inline v_int16x8 v_reinterpret_s16(const v_reg<_Tp, n>& a)
{ return v_reg<_Tp, n>::template reinterpret_as<short, 8>(a); }

template<typename _Tp, int n> inline v_uint32x4 v_reinterpret_u32(const v_reg<_Tp, n>& a)
{ return v_reg<_Tp, n>::template reinterpret_as<uint, 4>(a); }

template<typename _Tp, int n> inline v_int32x4 v_reinterpret_s32(const v_reg<_Tp, n>& a)
{ return v_reg<_Tp, n>::template reinterpret_as<int, 4>(a); }

template<typename _Tp, int n> inline v_float32x4 v_reinterpret_f32(const v_reg<_Tp, n>& a)
{ return v_reg<_Tp, n>::template reinterpret_as<float, 4>(a); }

template<typename _Tp, int n> inline v_float64x2 v_reinterpret_f64(const v_reg<_Tp, n>& a)
{ return v_reg<_Tp, n>::template reinterpret_as<double, 2>(a); }

inline v_uint8x16 v_sat_u8(const v_uint16x8& a, const v_uint16x8& b)
{ return v_cvtsat<ushort, 8, uchar>(a, b); }
inline v_uint8x16 v_sat_u8(const v_uint16x8& a, const v_uint16x8& b, int n)
{ return v_cvtsat<ushort, 8, uchar>(a, b, n); }
inline v_uint8x16 v_sat_u8(const v_int16x8& a, const v_int16x8& b)
{ return v_cvtsat<short, 8, uchar>(a, b); }
inline v_uint8x16 v_sat_u8(const v_int16x8& a, const v_int16x8& b, int n)
{ return v_cvtsat<short, 8, uchar>(a, b, n); }

inline void v_storesat_u8(uchar* ptr, const v_uint16x8& b)
{ return v_storesat(ptr, b); }
inline void v_storesat_u8(uchar* ptr, const v_uint16x8& b, int n)
{ return v_storesat(ptr, b, n); }
inline void v_storesat_u8(uchar* ptr, const v_int16x8& b)
{ return v_storesat(ptr, b); }
inline void v_storesat_u8(uchar* ptr, const v_int16x8& b, int n)
{ return v_storesat(ptr, b, n); }

inline v_int8x16 v_sat_s8(const v_uint16x8& a, const v_uint16x8& b)
{ return v_cvtsat<ushort, 8, schar>(a, b); }
inline v_int8x16 v_sat_s8(const v_uint16x8& a, const v_uint16x8& b, int n)
{ return v_cvtsat<ushort, 8, schar>(a, b, n); }
inline v_int8x16 v_sat_s8(const v_int16x8& a, const v_int16x8& b)
{ return v_cvtsat<short, 8, schar>(a, b); }
inline v_int8x16 v_sat_s8(const v_int16x8& a, const v_int16x8& b, int n)
{ return v_cvtsat<short, 8, schar>(a, b, n); }

inline void v_storesat_s8(schar* ptr, const v_uint16x8& b)
{ return v_storesat(ptr, b); }
inline void v_storesat_s8(schar* ptr, const v_uint16x8& b, int n)
{ return v_storesat(ptr, b, n); }
inline void v_storesat_s8(schar* ptr, const v_int16x8& b)
{ return v_storesat(ptr, b); }
inline void v_storesat_s8(schar* ptr, const v_int16x8& b, int n)
{ return v_storesat(ptr, b, n); }

inline v_uint16x8 v_sat_u16(const v_uint32x4& a, const v_uint32x4& b)
{ return v_cvtsat<uint, 4, ushort>(a, b); }
inline v_uint16x8 v_sat_u16(const v_uint32x4& a, const v_uint32x4& b, int n)
{ return v_cvtsat<uint, 4, ushort>(a, b, n); }
inline v_uint16x8 v_sat_u16(const v_int32x4& a, const v_int32x4& b)
{ return v_cvtsat<int, 4, ushort>(a, b); }
inline v_uint16x8 v_sat_u16(const v_int32x4& a, const v_int32x4& b, int n)
{ return v_cvtsat<int, 4, ushort>(a, b, n); }

inline void v_storesat_u16(ushort* ptr, const v_uint32x4& b)
{ return v_storesat(ptr, b); }
inline void v_storesat_u16(ushort* ptr, const v_uint32x4& b, int n)
{ return v_storesat(ptr, b, n); }
inline void v_storesat_u16(ushort* ptr, const v_int32x4& b)
{ return v_storesat(ptr, b); }
inline void v_storesat_u16(ushort* ptr, const v_int32x4& b, int n)
{ return v_storesat(ptr, b, n); }

inline v_int16x8 v_sat_s16(const v_uint32x4& a, const v_uint32x4& b)
{ return v_cvtsat<uint, 4, short>(a, b); }
inline v_int16x8 v_sat_s16(const v_uint32x4& a, const v_uint32x4& b, int n)
{ return v_cvtsat<uint, 4, short>(a, b, n); }
inline v_int16x8 v_sat_s16(const v_int32x4& a, const v_int32x4& b)
{ return v_cvtsat<int, 4, short>(a, b); }
inline v_int16x8 v_sat_s16(const v_int32x4& a, const v_int32x4& b, int n)
{ return v_cvtsat<int, 4, short>(a, b, n); }

inline void v_storesat_s16(short* ptr, const v_uint32x4& b)
{ return v_storesat(ptr, b); }
inline void v_storesat_s16(short* ptr, const v_uint32x4& b, int n)
{ return v_storesat(ptr, b, n); }
inline void v_storesat_s16(short* ptr, const v_int32x4& b)
{ return v_storesat(ptr, b); }
inline void v_storesat_s16(short* ptr, const v_int32x4& b, int n)
{ return v_storesat(ptr, b, n); }

inline v_float32x4 v_matmul(const v_float32x4& v, const v_float32x4& m0,
                            const v_float32x4& m1, const v_float32x4& m2,
                            const v_float32x4& m3)
{
    return v_float32x4(v.s[0]*m0.s[0] + v.s[1]*m1.s[0] + v.s[2]*m2.s[0] + v.s[3]*m3.s[0],
                       v.s[0]*m0.s[1] + v.s[1]*m1.s[1] + v.s[2]*m2.s[1] + v.s[3]*m3.s[1],
                       v.s[0]*m0.s[2] + v.s[1]*m1.s[2] + v.s[2]*m2.s[2] + v.s[3]*m3.s[2],
                       v.s[0]*m0.s[3] + v.s[1]*m1.s[3] + v.s[2]*m2.s[3] + v.s[3]*m3.s[3]);
}

#endif

}}

#endif
