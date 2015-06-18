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

#ifndef __OPENCV_HAL_INTRIN_CPP_HPP__
#define __OPENCV_HAL_INTRIN_CPP_HPP__

namespace cv
{

template<typename _Tp, int n> struct v_reg
{
    typedef _Tp lane_type;
    typedef v_reg<typename V_TypeTraits<_Tp>::int_type, n> int_vec;
    typedef v_reg<typename V_TypeTraits<_Tp>::abs_type, n> abs_vec;
    enum { nlanes = n };

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

    template<typename _Tp2, int n2> v_reg<_Tp2, n2> reinterpret_as() const
    {
        size_t bytes = std::min(sizeof(_Tp2)*n2, sizeof(_Tp)*n);
        v_reg<_Tp2, n2> c;
        memcpy(&c.s[0], &s[0], bytes);
        return c;
    }

    _Tp s[n];
};

#define OPENCV_HAL_IMPL_BIN_OP(bin_op) \
template<typename _Tp, int n> inline v_reg<_Tp, n> \
    operator bin_op (const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b) \
{ \
    v_reg<_Tp, n> c; \
    for( int i = 0; i < n; i++ ) \
        c.s[i] = saturate_cast<_Tp>(a.s[i] bin_op b.s[i]); \
    return c; \
} \
template<typename _Tp, int n> inline v_reg<_Tp, n>& \
    operator bin_op##= (v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b) \
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
template<typename _Tp, int n> inline v_reg<_Tp, n> operator bit_op \
    (const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b) \
{ \
    v_reg<_Tp, n> c; \
    typedef typename V_TypeTraits<_Tp>::int_type itype; \
    for( int i = 0; i < n; i++ ) \
        c.s[i] = V_TypeTraits<_Tp>::reinterpret_from_int((itype)(V_TypeTraits<_Tp>::reinterpret_int(a.s[i]) bit_op \
                                                        V_TypeTraits<_Tp>::reinterpret_int(b.s[i]))); \
    return c; \
} \
template<typename _Tp, int n> inline v_reg<_Tp, n>& operator \
    bit_op##= (v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b) \
{ \
    typedef typename V_TypeTraits<_Tp>::int_type itype; \
    for( int i = 0; i < n; i++ ) \
        a.s[i] = V_TypeTraits<_Tp>::reinterpret_from_int((itype)(V_TypeTraits<_Tp>::reinterpret_int(a.s[i]) bit_op \
                                                        V_TypeTraits<_Tp>::reinterpret_int(b.s[i]))); \
    return a; \
}

OPENCV_HAL_IMPL_BIT_OP(&)
OPENCV_HAL_IMPL_BIT_OP(|)
OPENCV_HAL_IMPL_BIT_OP(^)

template<typename _Tp, int n> inline v_reg<_Tp, n> operator ~ (const v_reg<_Tp, n>& a)
{
    v_reg<_Tp, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = V_TypeTraits<_Tp>::reinterpret_from_int(~V_TypeTraits<_Tp>::reinterpret_int(a.s[i]));
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
OPENCV_HAL_IMPL_MATH_FUNC(v_abs, (typename V_TypeTraits<_Tp>::abs_type)std::abs,
                          typename V_TypeTraits<_Tp>::abs_type)
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

template<typename _Tp, int n>
inline void v_minmax( const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b,
                      v_reg<_Tp, n>& minval, v_reg<_Tp, n>& maxval )
{
    for( int i = 0; i < n; i++ )
    {
        minval.s[i] = std::min(a.s[i], b.s[i]);
        maxval.s[i] = std::max(a.s[i], b.s[i]);
    }
}


#define OPENCV_HAL_IMPL_CMP_OP(cmp_op) \
template<typename _Tp, int n> \
inline v_reg<_Tp, n> operator cmp_op(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b) \
{ \
    typedef typename V_TypeTraits<_Tp>::int_type itype; \
    v_reg<_Tp, n> c; \
    for( int i = 0; i < n; i++ ) \
        c.s[i] = V_TypeTraits<_Tp>::reinterpret_from_int((itype)-(int)(a.s[i] cmp_op b.s[i])); \
    return c; \
}

OPENCV_HAL_IMPL_CMP_OP(<)
OPENCV_HAL_IMPL_CMP_OP(>)
OPENCV_HAL_IMPL_CMP_OP(<=)
OPENCV_HAL_IMPL_CMP_OP(>=)
OPENCV_HAL_IMPL_CMP_OP(==)
OPENCV_HAL_IMPL_CMP_OP(!=)

#define OPENCV_HAL_IMPL_ADD_SUB_OP(func, bin_op, cast_op, _Tp2) \
template<typename _Tp, int n> \
inline v_reg<_Tp2, n> func(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b) \
{ \
    typedef _Tp2 rtype; \
    v_reg<rtype, n> c; \
    for( int i = 0; i < n; i++ ) \
        c.s[i] = cast_op(a.s[i] bin_op b.s[i]); \
    return c; \
}

OPENCV_HAL_IMPL_ADD_SUB_OP(v_add_wrap, +, (_Tp), _Tp)
OPENCV_HAL_IMPL_ADD_SUB_OP(v_sub_wrap, -, (_Tp), _Tp)
OPENCV_HAL_IMPL_ADD_SUB_OP(v_absdiff, -, (rtype)std::abs, typename V_TypeTraits<_Tp>::abs_type)

template<typename _Tp, int n>
inline v_reg<_Tp, n> v_invsqrt(const v_reg<_Tp, n>& a)
{
    v_reg<_Tp, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = 1.f/std::sqrt(a.s[i]);
    return c;
}

template<typename _Tp, int n>
inline v_reg<_Tp, n> v_magnitude(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{
    v_reg<_Tp, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = std::sqrt(a.s[i]*a.s[i] + b.s[i]*b.s[i]);
    return c;
}


template<typename _Tp, int n>
inline v_reg<_Tp, n> v_sqr_magnitude(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{
    v_reg<_Tp, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = a.s[i]*a.s[i] + b.s[i]*b.s[i];
    return c;
}

template<typename _Tp, int n>
inline v_reg<_Tp, n> v_muladd(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b,
                              const v_reg<_Tp, n>& c)
{
    v_reg<_Tp, n> d;
    for( int i = 0; i < n; i++ )
        d.s[i] = a.s[i]*b.s[i] + c.s[i];
    return d;
}

template<typename _Tp, int n> inline v_reg<typename V_TypeTraits<_Tp>::w_type, n/2>
    v_dotprod(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{
    typedef typename V_TypeTraits<_Tp>::w_type w_type;
    v_reg<w_type, n/2> c;
    for( int i = 0; i < (n/2); i++ )
        c.s[i] = (w_type)a.s[i*2]*b.s[i*2] + (w_type)a.s[i*2+1]*b.s[i*2+1];
    return c;
}

template<typename _Tp, int n> inline void v_mul_expand(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b,
                                                       v_reg<typename V_TypeTraits<_Tp>::w_type, n/2>& c,
                                                       v_reg<typename V_TypeTraits<_Tp>::w_type, n/2>& d)
{
    typedef typename V_TypeTraits<_Tp>::w_type w_type;
    for( int i = 0; i < (n/2); i++ )
    {
        c.s[i] = (w_type)a.s[i]*b.s[i]*2;
        d.s[i] = (w_type)a.s[i+(n/2)]*b.s[i+(n/2)];
    }
}

template<typename _Tp, int n> inline void v_hsum(const v_reg<_Tp, n>& a,
                                                 v_reg<typename V_TypeTraits<_Tp>::w_type, n/2>& c)
{
    typedef typename V_TypeTraits<_Tp>::w_type w_type;
    for( int i = 0; i < (n/2); i++ )
    {
        c.s[i] = (w_type)a.s[i*2] + a.s[i*2+1];
    }
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

template<typename _Tp, int n> inline typename V_TypeTraits<_Tp>::sum_type v_reduce_sum(const v_reg<_Tp, n>& a)
{
    typename V_TypeTraits<_Tp>::sum_type c = a.s[0];
    for( int i = 1; i < n; i++ )
        c += a.s[i];
    return c;
}

template<typename _Tp, int n> inline int v_signmask(const v_reg<_Tp, n>& a)
{
    int mask = 0;
    for( int i = 0; i < n; i++ )
        mask |= (V_TypeTraits<_Tp>::reinterpret_int(a.s[i]) < 0) << i;
    return mask;
}

template<typename _Tp, int n> inline bool v_check_all(const v_reg<_Tp, n>& a)
{
    for( int i = 0; i < n; i++ )
        if( V_TypeTraits<_Tp>::reinterpret_int(a.s[i]) >= 0 )
            return false;
    return true;
}

template<typename _Tp, int n> inline bool v_check_any(const v_reg<_Tp, n>& a)
{
    for( int i = 0; i < n; i++ )
        if( V_TypeTraits<_Tp>::reinterpret_int(a.s[i]) < 0 )
            return true;
    return false;
}

template<typename _Tp, int n> inline v_reg<_Tp, n> v_select(const v_reg<_Tp, n>& mask,
                                                           const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{
    v_reg<_Tp, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = V_TypeTraits<_Tp>::reinterpret_int(mask.s[i]) < 0 ? b.s[i] : a.s[i];
    return c;
}

template<typename _Tp, int n> inline void v_expand(const v_reg<_Tp, n>& a,
                            v_reg<typename V_TypeTraits<_Tp>::w_type, n/2>& b0,
                            v_reg<typename V_TypeTraits<_Tp>::w_type, n/2>& b1)
{
    for( int i = 0; i < (n/2); i++ )
    {
        b0.s[i] = a.s[i];
        b1.s[i] = a.s[i+(n/2)];
    }
}

template<typename _Tp, int n> inline v_reg<typename V_TypeTraits<_Tp>::int_type, n>
    v_reinterpret_as_int(const v_reg<_Tp, n>& a)
{
    v_reg<typename V_TypeTraits<_Tp>::int_type, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = V_TypeTraits<_Tp>::reinterpret_int(a.s[i]);
    return c;
}

template<typename _Tp, int n> inline v_reg<typename V_TypeTraits<_Tp>::uint_type, n>
    v_reinterpret_as_uint(const v_reg<_Tp, n>& a)
{
    v_reg<typename V_TypeTraits<_Tp>::uint_type, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = V_TypeTraits<_Tp>::reinterpret_uint(a.s[i]);
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

template<typename _Tp, int n> inline v_reg<typename V_TypeTraits<_Tp>::w_type, n> v_load_expand(const _Tp* ptr)
{
    typedef typename V_TypeTraits<_Tp>::w_type w_type;
    v_reg<w_type, n> c;
    for( int i = 0; i < n; i++ )
    {
        c.s[i] = ptr[i];
    }
    return c;
}

template<typename _Tp, int n> inline v_reg<typename
    V_TypeTraits<typename V_TypeTraits<_Tp>::w_type>::w_type, n> v_load_expand_q(const _Tp* ptr)
{
    typedef typename V_TypeTraits<typename V_TypeTraits<_Tp>::w_type>::w_type w_type;
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

template<typename _Tp, int n>
inline void v_load_deinterleave(const _Tp* ptr, v_reg<_Tp, n>& a,
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

template<typename _Tp, int n>
inline void v_store_interleave( _Tp* ptr, const v_reg<_Tp, n>& a,
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

template<typename _Tp, int n>
inline void v_store(_Tp* ptr, const v_reg<_Tp, n>& a)
{
    for( int i = 0; i < n; i++ )
        ptr[i] = a.s[i];
}

template<typename _Tp, int n>
inline void v_store_low(_Tp* ptr, const v_reg<_Tp, n>& a)
{
    for( int i = 0; i < (n/2); i++ )
        ptr[i] = a.s[i];
}

template<typename _Tp, int n>
inline void v_store_high(_Tp* ptr, const v_reg<_Tp, n>& a)
{
    for( int i = 0; i < (n/2); i++ )
        ptr[i] = a.s[i+(n/2)];
}

template<typename _Tp, int n>
inline void v_store_aligned(_Tp* ptr, const v_reg<_Tp, n>& a)
{
    for( int i = 0; i < n; i++ )
        ptr[i] = a.s[i];
}

template<typename _Tp, int n>
inline v_reg<_Tp, n> v_combine_low(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{
    v_reg<_Tp, n> c;
    for( int i = 0; i < (n/2); i++ )
    {
        c.s[i] = a.s[i];
        c.s[i+(n/2)] = b.s[i];
    }
}

template<typename _Tp, int n>
inline v_reg<_Tp, n> v_combine_high(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{
    v_reg<_Tp, n> c;
    for( int i = 0; i < (n/2); i++ )
    {
        c.s[i] = a.s[i+(n/2)];
        c.s[i+(n/2)] = b.s[i+(n/2)];
    }
}

template<typename _Tp, int n>
inline void v_recombine(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b,
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

template<typename _Tp>
inline void v_transpose4x4( v_reg<_Tp, 4>& a0, const v_reg<_Tp, 4>& a1,
                            const v_reg<_Tp, 4>& a2, const v_reg<_Tp, 4>& a3,
                            v_reg<_Tp, 4>& b0, v_reg<_Tp, 4>& b1,
                            v_reg<_Tp, 4>& b2, v_reg<_Tp, 4>& b3 )
{
    b0 = v_reg<_Tp, 4>(a0.s[0], a1.s[0], a2.s[0], a3.s[0]);
    b1 = v_reg<_Tp, 4>(a0.s[1], a1.s[1], a2.s[1], a3.s[1]);
    b2 = v_reg<_Tp, 4>(a0.s[2], a1.s[2], a2.s[2], a3.s[2]);
    b3 = v_reg<_Tp, 4>(a0.s[3], a1.s[3], a2.s[3], a3.s[3]);
}

typedef v_reg<uchar, 16> v_uint8x16;
typedef v_reg<schar, 16> v_int8x16;
typedef v_reg<ushort, 8> v_uint16x8;
typedef v_reg<short, 8> v_int16x8;
typedef v_reg<unsigned, 4> v_uint32x4;
typedef v_reg<int, 4> v_int32x4;
typedef v_reg<float, 4> v_float32x4;
typedef v_reg<float, 8> v_float32x8;
typedef v_reg<double, 2> v_float64x2;
typedef v_reg<uint64, 2> v_uint64x2;
typedef v_reg<int64, 2> v_int64x2;

#define OPENCV_HAL_IMPL_C_INIT(_Tpvec, _Tp, suffix) \
inline _Tpvec v_setzero_##suffix() { return _Tpvec::zero(); } \
inline _Tpvec v_setall_##suffix(_Tp val) { return _Tpvec::all(val); } \
template<typename _Tp0, int n0> inline _Tpvec \
    v_reinterpret_as_##suffix(const v_reg<_Tp0, n0>& a) \
{ return a.template reinterpret_as<_Tp, _Tpvec::nlanes>(a); }

OPENCV_HAL_IMPL_C_INIT(v_uint8x16, uchar, u8)
OPENCV_HAL_IMPL_C_INIT(v_int8x16, schar, s8)
OPENCV_HAL_IMPL_C_INIT(v_uint16x8, ushort, u16)
OPENCV_HAL_IMPL_C_INIT(v_int16x8, short, s16)
OPENCV_HAL_IMPL_C_INIT(v_uint32x4, unsigned, u32)
OPENCV_HAL_IMPL_C_INIT(v_int32x4, int, s32)
OPENCV_HAL_IMPL_C_INIT(v_float32x4, float, f32)
OPENCV_HAL_IMPL_C_INIT(v_float64x2, double, f64)
OPENCV_HAL_IMPL_C_INIT(v_uint64x2, uint64, u64)
OPENCV_HAL_IMPL_C_INIT(v_uint64x2, int64, s64)

#define OPENCV_HAL_IMPL_C_SHIFT(_Tpvec, _Tp) \
template<int n> inline _Tpvec v_shl(const _Tpvec& a) \
{ return a << n; } \
template<int n> inline _Tpvec v_shr(const _Tpvec& a) \
{ return a >> n; } \
template<int n> inline _Tpvec v_rshr(const _Tpvec& a) \
{ \
    _Tpvec c; \
    for( int i = 0; i < _Tpvec::nlanes; i++ ) \
        c.s[i] = (_Tp)((a.s[i] + ((_Tp)1 << (n - 1))) >> n); \
    return c; \
}

OPENCV_HAL_IMPL_C_SHIFT(v_uint16x8, ushort)
OPENCV_HAL_IMPL_C_SHIFT(v_int16x8, short)
OPENCV_HAL_IMPL_C_SHIFT(v_uint32x4, unsigned)
OPENCV_HAL_IMPL_C_SHIFT(v_int32x4, int)
OPENCV_HAL_IMPL_C_SHIFT(v_uint64x2, uint64)
OPENCV_HAL_IMPL_C_SHIFT(v_int64x2, int64)


#define OPENCV_HAL_IMPL_C_PACK(_Tpvec, _Tp, _Tpnvec, _Tpn, pack_suffix) \
inline _Tpnvec v_##pack_suffix(const _Tpvec& a, const _Tpvec& b) \
{ \
    _Tpnvec c; \
    for( int i = 0; i < _Tpvec::nlanes; i++ ) \
    { \
        c.s[i] = saturate_cast<_Tpn>(a.s[i]); \
        c.s[i+_Tpvec::nlanes] = saturate_cast<_Tpn>(b.s[i]); \
    } \
    return c; \
} \
template<int n> inline _Tpnvec v_rshr_##pack_suffix(const _Tpvec& a, const _Tpvec& b) \
{ \
    _Tpnvec c; \
    for( int i = 0; i < _Tpvec::nlanes; i++ ) \
    { \
        c.s[i] = saturate_cast<_Tpn>((a.s[i] + ((_Tp)1 << (n - 1))) >> n); \
        c.s[i+_Tpvec::nlanes] = saturate_cast<_Tpn>((b.s[i] + ((_Tp)1 << (n - 1))) >> n); \
    } \
    return c; \
} \
inline void v_##pack_suffix##_store(_Tpn* ptr, const _Tpvec& a) \
{ \
    for( int i = 0; i < _Tpvec::nlanes; i++ ) \
        ptr[i] = saturate_cast<_Tpn>(a.s[i]); \
} \
template<int n> inline void v_rshr_##pack_suffix##_store(_Tpn* ptr, const _Tpvec& a) \
{ \
    for( int i = 0; i < _Tpvec::nlanes; i++ ) \
        ptr[i] = saturate_cast<_Tpn>((a.s[i] + ((_Tp)1 << (n - 1))) >> n); \
}

OPENCV_HAL_IMPL_C_PACK(v_uint16x8, ushort, v_uint8x16, uchar, pack)
OPENCV_HAL_IMPL_C_PACK(v_int16x8, short, v_int8x16, schar, pack)
OPENCV_HAL_IMPL_C_PACK(v_int16x8, short, v_uint8x16, uchar, pack_u)
OPENCV_HAL_IMPL_C_PACK(v_uint32x4, unsigned, v_uint16x8, ushort, pack)
OPENCV_HAL_IMPL_C_PACK(v_int32x4, int, v_int16x8, short, pack)
OPENCV_HAL_IMPL_C_PACK(v_int32x4, int, v_uint16x8, ushort, pack_u)
OPENCV_HAL_IMPL_C_PACK(v_uint64x2, uint64, v_uint32x4, unsigned, pack)
OPENCV_HAL_IMPL_C_PACK(v_int64x2, int64, v_int32x4, int, pack)

inline v_float32x4 v_matmul(const v_float32x4& v, const v_float32x4& m0,
                            const v_float32x4& m1, const v_float32x4& m2,
                            const v_float32x4& m3)
{
    return v_float32x4(v.s[0]*m0.s[0] + v.s[1]*m1.s[0] + v.s[2]*m2.s[0] + v.s[3]*m3.s[0],
                       v.s[0]*m0.s[1] + v.s[1]*m1.s[1] + v.s[2]*m2.s[1] + v.s[3]*m3.s[1],
                       v.s[0]*m0.s[2] + v.s[1]*m1.s[2] + v.s[2]*m2.s[2] + v.s[3]*m3.s[2],
                       v.s[0]*m0.s[3] + v.s[1]*m1.s[3] + v.s[2]*m2.s[3] + v.s[3]*m3.s[3]);
}

}

#endif
