// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_HAL_INTRIN_RVV_HPP
#define OPENCV_HAL_INTRIN_RVV_HPP

#include <limits>
#include <cstring>
#include <algorithm>
#include "opencv2/core/saturate.hpp"

#define CV_SIMD128_CPP 1
#if defined(CV_FORCE_SIMD128_CPP) || defined(CV_DOXYGEN)
#define CV_SIMD128 1
#define CV_SIMD128_64F 1
#endif

namespace cv
{

#ifndef CV_DOXYGEN
CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN
#endif


template<typename _Tp, int n> struct v_reg
{
    typedef _Tp lane_type;
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
    _Tp get0() const { return s[0]; }

    _Tp get(const int i) const { return s[i]; }
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
        std::memcpy(&c.s[0], &s[0], bytes);
        return c;
    }

    v_reg& operator=(const v_reg<_Tp, n> & r)
    {
        for( int i = 0; i < n; i++ )
            s[i] = r.s[i];
        return *this;
    }

    _Tp s[n];
};

typedef v_reg<uchar, 16> v_uint8x16;
typedef v_reg<schar, 16> v_int8x16;
typedef v_reg<ushort, 8> v_uint16x8;
typedef v_reg<short, 8> v_int16x8;
typedef v_reg<unsigned, 4> v_uint32x4;
typedef v_reg<int, 4> v_int32x4;
typedef v_reg<float, 4> v_float32x4;
typedef v_reg<double, 2> v_float64x2;
typedef v_reg<uint64, 2> v_uint64x2;
typedef v_reg<int64, 2> v_int64x2;

template<typename _Tp, int n> CV_INLINE v_reg<_Tp, n> operator+(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b);
template<typename _Tp, int n> CV_INLINE v_reg<_Tp, n>& operator+=(v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b);

template<typename _Tp, int n> CV_INLINE v_reg<_Tp, n> operator-(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b);
template<typename _Tp, int n> CV_INLINE v_reg<_Tp, n>& operator-=(v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b);

template<typename _Tp, int n> CV_INLINE v_reg<_Tp, n> operator*(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b);
template<typename _Tp, int n> CV_INLINE v_reg<_Tp, n>& operator*=(v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b);

template<typename _Tp, int n> CV_INLINE v_reg<_Tp, n> operator/(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b);
template<typename _Tp, int n> CV_INLINE v_reg<_Tp, n>& operator/=(v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b);


template<typename _Tp, int n> CV_INLINE v_reg<_Tp, n> operator&(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b);
template<typename _Tp, int n> CV_INLINE v_reg<_Tp, n>& operator&=(v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b);

template<typename _Tp, int n> CV_INLINE v_reg<_Tp, n> operator|(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b);
template<typename _Tp, int n> CV_INLINE v_reg<_Tp, n>& operator|=(v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b);

template<typename _Tp, int n> CV_INLINE v_reg<_Tp, n> operator^(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b);
template<typename _Tp, int n> CV_INLINE v_reg<_Tp, n>& operator^=(v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b);

template<typename _Tp, int n> CV_INLINE v_reg<_Tp, n> operator~(const v_reg<_Tp, n>& a);


#ifndef CV_DOXYGEN

#define CV__HAL_INTRIN_EXPAND_WITH_INTEGER_TYPES(macro_name, ...) \
__CV_EXPAND(macro_name(uchar, __VA_ARGS__)) \
__CV_EXPAND(macro_name(schar, __VA_ARGS__)) \
__CV_EXPAND(macro_name(ushort, __VA_ARGS__)) \
__CV_EXPAND(macro_name(short, __VA_ARGS__)) \
__CV_EXPAND(macro_name(unsigned, __VA_ARGS__)) \
__CV_EXPAND(macro_name(int, __VA_ARGS__)) \
__CV_EXPAND(macro_name(uint64, __VA_ARGS__)) \
__CV_EXPAND(macro_name(int64, __VA_ARGS__)) \

#define CV__HAL_INTRIN_EXPAND_WITH_FP_TYPES(macro_name, ...) \
__CV_EXPAND(macro_name(float, __VA_ARGS__)) \
__CV_EXPAND(macro_name(double, __VA_ARGS__)) \

#define CV__HAL_INTRIN_EXPAND_WITH_ALL_TYPES(macro_name, ...) \
CV__HAL_INTRIN_EXPAND_WITH_INTEGER_TYPES(macro_name, __VA_ARGS__) \
CV__HAL_INTRIN_EXPAND_WITH_FP_TYPES(macro_name, __VA_ARGS__) \

#define CV__HAL_INTRIN_IMPL_BIN_OP_(_Tp, bin_op) \
template<int n> inline \
v_reg<_Tp, n> operator bin_op (const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b) \
{ \
    v_reg<_Tp, n> c; \
    for( int i = 0; i < n; i++ ) \
        c.s[i] = saturate_cast<_Tp>(a.s[i] bin_op b.s[i]); \
    return c; \
} \
template<int n> inline \
v_reg<_Tp, n>& operator bin_op##= (v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b) \
{ \
    for( int i = 0; i < n; i++ ) \
        a.s[i] = saturate_cast<_Tp>(a.s[i] bin_op b.s[i]); \
    return a; \
}

#define CV__HAL_INTRIN_IMPL_BIN_OP(bin_op) CV__HAL_INTRIN_EXPAND_WITH_ALL_TYPES(CV__HAL_INTRIN_IMPL_BIN_OP_, bin_op)

CV__HAL_INTRIN_IMPL_BIN_OP(+)
CV__HAL_INTRIN_IMPL_BIN_OP(-)
CV__HAL_INTRIN_IMPL_BIN_OP(*)
CV__HAL_INTRIN_EXPAND_WITH_FP_TYPES(CV__HAL_INTRIN_IMPL_BIN_OP_, /)

#define CV__HAL_INTRIN_IMPL_BIT_OP_(_Tp, bit_op) \
template<int n> CV_INLINE \
v_reg<_Tp, n> operator bit_op (const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b) \
{ \
    v_reg<_Tp, n> c; \
    typedef typename V_TypeTraits<_Tp>::int_type itype; \
    for( int i = 0; i < n; i++ ) \
        c.s[i] = V_TypeTraits<_Tp>::reinterpret_from_int((itype)(V_TypeTraits<_Tp>::reinterpret_int(a.s[i]) bit_op \
                                                        V_TypeTraits<_Tp>::reinterpret_int(b.s[i]))); \
    return c; \
} \
template<int n> CV_INLINE \
v_reg<_Tp, n>& operator bit_op##= (v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b) \
{ \
    typedef typename V_TypeTraits<_Tp>::int_type itype; \
    for( int i = 0; i < n; i++ ) \
        a.s[i] = V_TypeTraits<_Tp>::reinterpret_from_int((itype)(V_TypeTraits<_Tp>::reinterpret_int(a.s[i]) bit_op \
                                                        V_TypeTraits<_Tp>::reinterpret_int(b.s[i]))); \
    return a; \
}

#define CV__HAL_INTRIN_IMPL_BIT_OP(bit_op) \
CV__HAL_INTRIN_EXPAND_WITH_INTEGER_TYPES(CV__HAL_INTRIN_IMPL_BIT_OP_, bit_op) \
CV__HAL_INTRIN_EXPAND_WITH_FP_TYPES(CV__HAL_INTRIN_IMPL_BIT_OP_, bit_op) /* TODO: FIXIT remove this after masks refactoring */


CV__HAL_INTRIN_IMPL_BIT_OP(&)
CV__HAL_INTRIN_IMPL_BIT_OP(|)
CV__HAL_INTRIN_IMPL_BIT_OP(^)

#define CV__HAL_INTRIN_IMPL_BITWISE_NOT_(_Tp, dummy) \
template<int n> CV_INLINE \
v_reg<_Tp, n> operator ~ (const v_reg<_Tp, n>& a) \
{ \
    v_reg<_Tp, n> c; \
    for( int i = 0; i < n; i++ ) \
        c.s[i] = V_TypeTraits<_Tp>::reinterpret_from_int(~V_TypeTraits<_Tp>::reinterpret_int(a.s[i])); \
    return c; \
} \

CV__HAL_INTRIN_EXPAND_WITH_INTEGER_TYPES(CV__HAL_INTRIN_IMPL_BITWISE_NOT_, ~)

#endif


#define OPENCV_HAL_IMPL_MATH_FUNC(func, cfunc, _Tp2) \
template<typename _Tp, int n> inline v_reg<_Tp2, n> func(const v_reg<_Tp, n>& a) \
{ \
    v_reg<_Tp2, n> c; \
    for( int i = 0; i < n; i++ ) \
        c.s[i] = cfunc(a.s[i]); \
    return c; \
}

#define OPENCV_HAL_IMPL_MATH_FUNC_FLOAT(func, cfunc) \
inline v_reg<int, 4> func(const v_reg<float, 4>& a) \
{ \
    v_reg<int, 4> c; \
    for( int i = 0; i < 4; i++ ) \
        c.s[i] = cfunc(a.s[i]); \
    return c; \
} \
inline v_reg<int, 4> func(const v_reg<double, 2>& a) \
{ \
    v_reg<int, 4> c; \
    for( int i = 0; i < 2; i++ ) \
    { \
        c.s[i] = cfunc(a.s[i]); \
        c.s[i + 2] = 0; \
    } \
    return c; \
}

OPENCV_HAL_IMPL_MATH_FUNC(v_sqrt, std::sqrt, _Tp)

OPENCV_HAL_IMPL_MATH_FUNC(v_sin, std::sin, _Tp)
OPENCV_HAL_IMPL_MATH_FUNC(v_cos, std::cos, _Tp)
OPENCV_HAL_IMPL_MATH_FUNC(v_exp, std::exp, _Tp)
OPENCV_HAL_IMPL_MATH_FUNC(v_log, std::log, _Tp)

OPENCV_HAL_IMPL_MATH_FUNC(v_abs, (typename V_TypeTraits<_Tp>::abs_type)std::abs,
                          typename V_TypeTraits<_Tp>::abs_type)

OPENCV_HAL_IMPL_MATH_FUNC_FLOAT(v_round, cvRound)

OPENCV_HAL_IMPL_MATH_FUNC_FLOAT(v_floor, cvFloor)

OPENCV_HAL_IMPL_MATH_FUNC_FLOAT(v_ceil, cvCeil)

OPENCV_HAL_IMPL_MATH_FUNC_FLOAT(v_trunc, int)

#define OPENCV_HAL_IMPL_MINMAX_FUNC(func, cfunc) \
template<typename _Tp, int n> inline v_reg<_Tp, n> func(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b) \
{ \
    v_reg<_Tp, n> c; \
    for( int i = 0; i < n; i++ ) \
        c.s[i] = cfunc(a.s[i], b.s[i]); \
    return c; \
}

#define OPENCV_HAL_IMPL_REDUCE_MINMAX_FUNC(func, cfunc) \
template<typename _Tp, int n> inline _Tp func(const v_reg<_Tp, n>& a) \
{ \
    _Tp c = a.s[0]; \
    for( int i = 1; i < n; i++ ) \
        c = cfunc(c, a.s[i]); \
    return c; \
}

OPENCV_HAL_IMPL_MINMAX_FUNC(v_min, std::min)

OPENCV_HAL_IMPL_MINMAX_FUNC(v_max, std::max)

OPENCV_HAL_IMPL_REDUCE_MINMAX_FUNC(v_reduce_min, std::min)

OPENCV_HAL_IMPL_REDUCE_MINMAX_FUNC(v_reduce_max, std::max)

static const unsigned char popCountTable[] =
{
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8,
};
template<typename _Tp, int n>
inline v_reg<typename V_TypeTraits<_Tp>::abs_type, n> v_popcount(const v_reg<_Tp, n>& a)
{
    v_reg<typename V_TypeTraits<_Tp>::abs_type, n> b = v_reg<typename V_TypeTraits<_Tp>::abs_type, n>::zero();
    for (int i = 0; i < n*(int)sizeof(_Tp); i++)
        b.s[i/sizeof(_Tp)] += popCountTable[v_reinterpret_as_u8(a).s[i]];
    return b;
}


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

template<int n>
inline v_reg<float, n> v_not_nan(const v_reg<float, n>& a)
{
    typedef typename V_TypeTraits<float>::int_type itype;
    v_reg<float, n> c;
    for (int i = 0; i < n; i++)
        c.s[i] = V_TypeTraits<float>::reinterpret_from_int((itype)-(int)(a.s[i] == a.s[i]));
    return c;
}
template<int n>
inline v_reg<double, n> v_not_nan(const v_reg<double, n>& a)
{
    typedef typename V_TypeTraits<double>::int_type itype;
    v_reg<double, n> c;
    for (int i = 0; i < n; i++)
        c.s[i] = V_TypeTraits<double>::reinterpret_from_int((itype)-(int)(a.s[i] == a.s[i]));
    return c;
}

#define OPENCV_HAL_IMPL_ARITHM_OP(func, bin_op, cast_op, _Tp2) \
template<typename _Tp, int n> \
inline v_reg<_Tp2, n> func(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b) \
{ \
    typedef _Tp2 rtype; \
    v_reg<rtype, n> c; \
    for( int i = 0; i < n; i++ ) \
        c.s[i] = cast_op(a.s[i] bin_op b.s[i]); \
    return c; \
}

OPENCV_HAL_IMPL_ARITHM_OP(v_add_wrap, +, (_Tp), _Tp)

OPENCV_HAL_IMPL_ARITHM_OP(v_sub_wrap, -, (_Tp), _Tp)

OPENCV_HAL_IMPL_ARITHM_OP(v_mul_wrap, *, (_Tp), _Tp)

template<typename T> inline T _absdiff(T a, T b)
{
    return a > b ? a - b : b - a;
}

template<typename _Tp, int n>
inline v_reg<typename V_TypeTraits<_Tp>::abs_type, n> v_absdiff(const v_reg<_Tp, n>& a, const v_reg<_Tp, n> & b)
{
    typedef typename V_TypeTraits<_Tp>::abs_type rtype;
    v_reg<rtype, n> c;
    const rtype mask = (rtype)(std::numeric_limits<_Tp>::is_signed ? (1 << (sizeof(rtype)*8 - 1)) : 0);
    for( int i = 0; i < n; i++ )
    {
        rtype ua = a.s[i] ^ mask;
        rtype ub = b.s[i] ^ mask;
        c.s[i] = _absdiff(ua, ub);
    }
    return c;
}

inline v_float32x4 v_absdiff(const v_float32x4& a, const v_float32x4& b)
{
    v_float32x4 c;
    for( int i = 0; i < c.nlanes; i++ )
        c.s[i] = _absdiff(a.s[i], b.s[i]);
    return c;
}

inline v_float64x2 v_absdiff(const v_float64x2& a, const v_float64x2& b)
{
    v_float64x2 c;
    for( int i = 0; i < c.nlanes; i++ )
        c.s[i] = _absdiff(a.s[i], b.s[i]);
    return c;
}

template<typename _Tp, int n>
inline v_reg<_Tp, n> v_absdiffs(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{
    v_reg<_Tp, n> c;
    for( int i = 0; i < n; i++)
        c.s[i] = saturate_cast<_Tp>(std::abs(a.s[i] - b.s[i]));
    return c;
}

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
inline v_reg<_Tp, n> v_fma(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b,
                           const v_reg<_Tp, n>& c)
{
    v_reg<_Tp, n> d;
    for( int i = 0; i < n; i++ )
        d.s[i] = a.s[i]*b.s[i] + c.s[i];
    return d;
}

template<typename _Tp, int n>
inline v_reg<_Tp, n> v_muladd(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b,
                              const v_reg<_Tp, n>& c)
{
    return v_fma(a, b, c);
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

template<typename _Tp, int n> inline v_reg<typename V_TypeTraits<_Tp>::w_type, n/2>
v_dotprod(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b,
          const v_reg<typename V_TypeTraits<_Tp>::w_type, n / 2>& c)
{
    typedef typename V_TypeTraits<_Tp>::w_type w_type;
    v_reg<w_type, n/2> s;
    for( int i = 0; i < (n/2); i++ )
        s.s[i] = (w_type)a.s[i*2]*b.s[i*2] + (w_type)a.s[i*2+1]*b.s[i*2+1] + c.s[i];
    return s;
}

template<typename _Tp, int n> inline v_reg<typename V_TypeTraits<_Tp>::w_type, n/2>
v_dotprod_fast(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{ return v_dotprod(a, b); }

template<typename _Tp, int n> inline v_reg<typename V_TypeTraits<_Tp>::w_type, n/2>
v_dotprod_fast(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b,
               const v_reg<typename V_TypeTraits<_Tp>::w_type, n / 2>& c)
{ return v_dotprod(a, b, c); }

template<typename _Tp, int n> inline v_reg<typename V_TypeTraits<_Tp>::q_type, n/4>
v_dotprod_expand(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{
    typedef typename V_TypeTraits<_Tp>::q_type q_type;
    v_reg<q_type, n/4> s;
    for( int i = 0; i < (n/4); i++ )
        s.s[i] = (q_type)a.s[i*4    ]*b.s[i*4    ] + (q_type)a.s[i*4 + 1]*b.s[i*4 + 1] +
                 (q_type)a.s[i*4 + 2]*b.s[i*4 + 2] + (q_type)a.s[i*4 + 3]*b.s[i*4 + 3];
    return s;
}

template<typename _Tp, int n> inline v_reg<typename V_TypeTraits<_Tp>::q_type, n/4>
v_dotprod_expand(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b,
                 const v_reg<typename V_TypeTraits<_Tp>::q_type, n / 4>& c)
{
    typedef typename V_TypeTraits<_Tp>::q_type q_type;
    v_reg<q_type, n/4> s;
    for( int i = 0; i < (n/4); i++ )
        s.s[i] = (q_type)a.s[i*4    ]*b.s[i*4    ] + (q_type)a.s[i*4 + 1]*b.s[i*4 + 1] +
                 (q_type)a.s[i*4 + 2]*b.s[i*4 + 2] + (q_type)a.s[i*4 + 3]*b.s[i*4 + 3] + c.s[i];
    return s;
}

template<typename _Tp, int n> inline v_reg<typename V_TypeTraits<_Tp>::q_type, n/4>
v_dotprod_expand_fast(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{ return v_dotprod_expand(a, b); }

template<typename _Tp, int n> inline v_reg<typename V_TypeTraits<_Tp>::q_type, n/4>
v_dotprod_expand_fast(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b,
                      const v_reg<typename V_TypeTraits<_Tp>::q_type, n / 4>& c)
{ return v_dotprod_expand(a, b, c); }

template<typename _Tp, int n> inline void v_mul_expand(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b,
                                                       v_reg<typename V_TypeTraits<_Tp>::w_type, n/2>& c,
                                                       v_reg<typename V_TypeTraits<_Tp>::w_type, n/2>& d)
{
    typedef typename V_TypeTraits<_Tp>::w_type w_type;
    for( int i = 0; i < (n/2); i++ )
    {
        c.s[i] = (w_type)a.s[i]*b.s[i];
        d.s[i] = (w_type)a.s[i+(n/2)]*b.s[i+(n/2)];
    }
}

template<typename _Tp, int n> inline v_reg<_Tp, n> v_mul_hi(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{
    typedef typename V_TypeTraits<_Tp>::w_type w_type;
    v_reg<_Tp, n> c;
    for (int i = 0; i < n; i++)
        c.s[i] = (_Tp)(((w_type)a.s[i] * b.s[i]) >> sizeof(_Tp)*8);
    return c;
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

OPENCV_HAL_IMPL_SHIFT_OP(<< )

OPENCV_HAL_IMPL_SHIFT_OP(>> )

#define OPENCV_HAL_IMPL_ROTATE_SHIFT_OP(suffix,opA,opB) \
template<int imm, typename _Tp, int n> inline v_reg<_Tp, n> v_rotate_##suffix(const v_reg<_Tp, n>& a) \
{ \
    v_reg<_Tp, n> b; \
    for (int i = 0; i < n; i++) \
    { \
        int sIndex = i opA imm; \
        if (0 <= sIndex && sIndex < n) \
        { \
            b.s[i] = a.s[sIndex]; \
        } \
        else \
        { \
            b.s[i] = 0; \
        } \
    } \
    return b; \
} \
template<int imm, typename _Tp, int n> inline v_reg<_Tp, n> v_rotate_##suffix(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b) \
{ \
    v_reg<_Tp, n> c; \
    for (int i = 0; i < n; i++) \
    { \
        int aIndex = i opA imm; \
        int bIndex = i opA imm opB n; \
        if (0 <= bIndex && bIndex < n) \
        { \
            c.s[i] = b.s[bIndex]; \
        } \
        else if (0 <= aIndex && aIndex < n) \
        { \
            c.s[i] = a.s[aIndex]; \
        } \
        else \
        { \
            c.s[i] = 0; \
        } \
    } \
    return c; \
}

OPENCV_HAL_IMPL_ROTATE_SHIFT_OP(left,  -, +)

OPENCV_HAL_IMPL_ROTATE_SHIFT_OP(right, +, -)

template<typename _Tp, int n> inline typename V_TypeTraits<_Tp>::sum_type v_reduce_sum(const v_reg<_Tp, n>& a)
{
    typename V_TypeTraits<_Tp>::sum_type c = a.s[0];
    for( int i = 1; i < n; i++ )
        c += a.s[i];
    return c;
}

inline v_float32x4 v_reduce_sum4(const v_float32x4& a, const v_float32x4& b,
                                 const v_float32x4& c, const v_float32x4& d)
{
    v_float32x4 r;
    r.s[0] = a.s[0] + a.s[1] + a.s[2] + a.s[3];
    r.s[1] = b.s[0] + b.s[1] + b.s[2] + b.s[3];
    r.s[2] = c.s[0] + c.s[1] + c.s[2] + c.s[3];
    r.s[3] = d.s[0] + d.s[1] + d.s[2] + d.s[3];
    return r;
}

template<typename _Tp, int n> inline typename V_TypeTraits< typename V_TypeTraits<_Tp>::abs_type >::sum_type v_reduce_sad(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{
    typename V_TypeTraits< typename V_TypeTraits<_Tp>::abs_type >::sum_type c = _absdiff(a.s[0], b.s[0]);
    for (int i = 1; i < n; i++)
        c += _absdiff(a.s[i], b.s[i]);
    return c;
}

template<typename _Tp, int n> inline int v_signmask(const v_reg<_Tp, n>& a)
{
    int mask = 0;
    for( int i = 0; i < n; i++ )
        mask |= (V_TypeTraits<_Tp>::reinterpret_int(a.s[i]) < 0) << i;
    return mask;
}

template <typename _Tp, int n> inline int v_scan_forward(const v_reg<_Tp, n>& a)
{
    for (int i = 0; i < n; i++)
        if(V_TypeTraits<_Tp>::reinterpret_int(a.s[i]) < 0)
            return i;
    return 0;
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
    typedef V_TypeTraits<_Tp> Traits;
    typedef typename Traits::int_type int_type;
    v_reg<_Tp, n> c;
    for( int i = 0; i < n; i++ )
    {
        int_type m = Traits::reinterpret_int(mask.s[i]);
        CV_DbgAssert(m == 0 || m == (~(int_type)0));  // restrict mask values: 0 or 0xff/0xffff/etc
        c.s[i] = m ? a.s[i] : b.s[i];
    }
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

template<typename _Tp, int n>
inline v_reg<typename V_TypeTraits<_Tp>::w_type, n/2>
v_expand_low(const v_reg<_Tp, n>& a)
{
    v_reg<typename V_TypeTraits<_Tp>::w_type, n/2> b;
    for( int i = 0; i < (n/2); i++ )
        b.s[i] = a.s[i];
    return b;
}

template<typename _Tp, int n>
inline v_reg<typename V_TypeTraits<_Tp>::w_type, n/2>
v_expand_high(const v_reg<_Tp, n>& a)
{
    v_reg<typename V_TypeTraits<_Tp>::w_type, n/2> b;
    for( int i = 0; i < (n/2); i++ )
        b.s[i] = a.s[i+(n/2)];
    return b;
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

template<typename _Tp>
inline v_reg<_Tp, V_TypeTraits<_Tp>::nlanes128> v_load(const _Tp* ptr)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
    return v_reg<_Tp, V_TypeTraits<_Tp>::nlanes128>(ptr);
}

template<typename _Tp>
inline v_reg<_Tp, V_TypeTraits<_Tp>::nlanes128> v_load_aligned(const _Tp* ptr)
{
    CV_Assert(isAligned<sizeof(v_reg<_Tp, V_TypeTraits<_Tp>::nlanes128>)>(ptr));
    return v_reg<_Tp, V_TypeTraits<_Tp>::nlanes128>(ptr);
}

template<typename _Tp>
inline v_reg<_Tp, V_TypeTraits<_Tp>::nlanes128> v_load_low(const _Tp* ptr)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
    v_reg<_Tp, V_TypeTraits<_Tp>::nlanes128> c;
    for( int i = 0; i < c.nlanes/2; i++ )
    {
        c.s[i] = ptr[i];
    }
    return c;
}

template<typename _Tp>
inline v_reg<_Tp, V_TypeTraits<_Tp>::nlanes128> v_load_halves(const _Tp* loptr, const _Tp* hiptr)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(loptr));
    CV_Assert(isAligned<sizeof(_Tp)>(hiptr));
#endif
    v_reg<_Tp, V_TypeTraits<_Tp>::nlanes128> c;
    for( int i = 0; i < c.nlanes/2; i++ )
    {
        c.s[i] = loptr[i];
        c.s[i+c.nlanes/2] = hiptr[i];
    }
    return c;
}

template<typename _Tp>
inline v_reg<typename V_TypeTraits<_Tp>::w_type, V_TypeTraits<_Tp>::nlanes128 / 2>
v_load_expand(const _Tp* ptr)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
    typedef typename V_TypeTraits<_Tp>::w_type w_type;
    v_reg<w_type, V_TypeTraits<w_type>::nlanes128> c;
    for( int i = 0; i < c.nlanes; i++ )
    {
        c.s[i] = ptr[i];
    }
    return c;
}

template<typename _Tp>
inline v_reg<typename V_TypeTraits<_Tp>::q_type, V_TypeTraits<_Tp>::nlanes128 / 4>
v_load_expand_q(const _Tp* ptr)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
    typedef typename V_TypeTraits<_Tp>::q_type q_type;
    v_reg<q_type, V_TypeTraits<q_type>::nlanes128> c;
    for( int i = 0; i < c.nlanes; i++ )
    {
        c.s[i] = ptr[i];
    }
    return c;
}

template<typename _Tp, int n> inline void v_load_deinterleave(const _Tp* ptr, v_reg<_Tp, n>& a,
                                                            v_reg<_Tp, n>& b)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
    int i, i2;
    for( i = i2 = 0; i < n; i++, i2 += 2 )
    {
        a.s[i] = ptr[i2];
        b.s[i] = ptr[i2+1];
    }
}

template<typename _Tp, int n> inline void v_load_deinterleave(const _Tp* ptr, v_reg<_Tp, n>& a,
                                                            v_reg<_Tp, n>& b, v_reg<_Tp, n>& c)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
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
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
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
                               const v_reg<_Tp, n>& b,
                               hal::StoreMode /*mode*/=hal::STORE_UNALIGNED)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
    int i, i2;
    for( i = i2 = 0; i < n; i++, i2 += 2 )
    {
        ptr[i2] = a.s[i];
        ptr[i2+1] = b.s[i];
    }
}

template<typename _Tp, int n>
inline void v_store_interleave( _Tp* ptr, const v_reg<_Tp, n>& a,
                                const v_reg<_Tp, n>& b, const v_reg<_Tp, n>& c,
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
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
                                                            const v_reg<_Tp, n>& d,
                                                            hal::StoreMode /*mode*/=hal::STORE_UNALIGNED)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
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
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
    for( int i = 0; i < n; i++ )
        ptr[i] = a.s[i];
}

template<typename _Tp, int n>
inline void v_store(_Tp* ptr, const v_reg<_Tp, n>& a, hal::StoreMode /*mode*/)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
    v_store(ptr, a);
}

template<typename _Tp, int n>
inline void v_store_low(_Tp* ptr, const v_reg<_Tp, n>& a)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
    for( int i = 0; i < (n/2); i++ )
        ptr[i] = a.s[i];
}

template<typename _Tp, int n>
inline void v_store_high(_Tp* ptr, const v_reg<_Tp, n>& a)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
    for( int i = 0; i < (n/2); i++ )
        ptr[i] = a.s[i+(n/2)];
}

template<typename _Tp, int n>
inline void v_store_aligned(_Tp* ptr, const v_reg<_Tp, n>& a)
{
    CV_Assert(isAligned<sizeof(v_reg<_Tp, n>)>(ptr));
    v_store(ptr, a);
}

template<typename _Tp, int n>
inline void v_store_aligned_nocache(_Tp* ptr, const v_reg<_Tp, n>& a)
{
    CV_Assert(isAligned<sizeof(v_reg<_Tp, n>)>(ptr));
    v_store(ptr, a);
}

template<typename _Tp, int n>
inline void v_store_aligned(_Tp* ptr, const v_reg<_Tp, n>& a, hal::StoreMode /*mode*/)
{
    CV_Assert(isAligned<sizeof(v_reg<_Tp, n>)>(ptr));
    v_store(ptr, a);
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
    return c;
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
    return c;
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

template<typename _Tp, int n>
inline v_reg<_Tp, n> v_reverse(const v_reg<_Tp, n>& a)
{
    v_reg<_Tp, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = a.s[n-i-1];
    return c;
}

template<int s, typename _Tp, int n>
inline v_reg<_Tp, n> v_extract(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{
    v_reg<_Tp, n> r;
    const int shift = n - s;
    int i = 0;
    for (; i < shift; ++i)
        r.s[i] = a.s[i+s];
    for (; i < n; ++i)
        r.s[i] = b.s[i-shift];
    return r;
}

template<int s, typename _Tp, int n>
inline _Tp v_extract_n(const v_reg<_Tp, n>& v)
{
    CV_DbgAssert(s >= 0 && s < n);
    return v.s[s];
}

template<int i, typename _Tp, int n>
inline v_reg<_Tp, n> v_broadcast_element(const v_reg<_Tp, n>& a)
{
    CV_DbgAssert(i >= 0 && i < n);
    return v_reg<_Tp, n>::all(a.s[i]);
}

template<int n> inline v_reg<int, n> v_round(const v_reg<float, n>& a)
{
    v_reg<int, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = cvRound(a.s[i]);
    return c;
}

template<int n> inline v_reg<int, n*2> v_round(const v_reg<double, n>& a, const v_reg<double, n>& b)
{
    v_reg<int, n*2> c;
    for( int i = 0; i < n; i++ )
    {
        c.s[i] = cvRound(a.s[i]);
        c.s[i+n] = cvRound(b.s[i]);
    }
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

template<int n> inline v_reg<float, n*2> v_cvt_f32(const v_reg<double, n>& a)
{
    v_reg<float, n*2> c;
    for( int i = 0; i < n; i++ )
    {
        c.s[i] = (float)a.s[i];
        c.s[i+n] = 0;
    }
    return c;
}

template<int n> inline v_reg<float, n*2> v_cvt_f32(const v_reg<double, n>& a, const v_reg<double, n>& b)
{
    v_reg<float, n*2> c;
    for( int i = 0; i < n; i++ )
    {
        c.s[i] = (float)a.s[i];
        c.s[i+n] = (float)b.s[i];
    }
    return c;
}

CV_INLINE v_reg<double, 2> v_cvt_f64(const v_reg<int, 4>& a)
{
    enum { n = 2 };
    v_reg<double, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = (double)a.s[i];
    return c;
}

CV_INLINE v_reg<double, 2> v_cvt_f64_high(const v_reg<int, 4>& a)
{
    enum { n = 2 };
    v_reg<double, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = (double)a.s[i + 2];
    return c;
}

CV_INLINE v_reg<double, 2> v_cvt_f64(const v_reg<float, 4>& a)
{
    enum { n = 2 };
    v_reg<double, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = (double)a.s[i];
    return c;
}

CV_INLINE v_reg<double, 2> v_cvt_f64_high(const v_reg<float, 4>& a)
{
    enum { n = 2 };
    v_reg<double, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = (double)a.s[i + 2];
    return c;
}

CV_INLINE v_reg<double, 2> v_cvt_f64(const v_reg<int64, 2>& a)
{
    enum { n = 2 };
    v_reg<double, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = (double)a.s[i];
    return c;
}

CV_INLINE v_reg<double, 2> v_cvt_f64_high(const v_reg<int64, 2>& a)
{
    enum { n = 2 };
    v_reg<double, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = (double)a.s[i];
    return c;
}


template<typename _Tp> inline v_reg<_Tp, V_TypeTraits<_Tp>::nlanes128> v_lut(const _Tp* tab, const int* idx)
{
    v_reg<_Tp, V_TypeTraits<_Tp>::nlanes128> c;
    for (int i = 0; i < V_TypeTraits<_Tp>::nlanes128; i++)
        c.s[i] = tab[idx[i]];
    return c;
}
template<typename _Tp> inline v_reg<_Tp, V_TypeTraits<_Tp>::nlanes128> v_lut_pairs(const _Tp* tab, const int* idx)
{
    v_reg<_Tp, V_TypeTraits<_Tp>::nlanes128> c;
    for (int i = 0; i < V_TypeTraits<_Tp>::nlanes128; i++)
        c.s[i] = tab[idx[i / 2] + i % 2];
    return c;
}
template<typename _Tp> inline v_reg<_Tp, V_TypeTraits<_Tp>::nlanes128> v_lut_quads(const _Tp* tab, const int* idx)
{
    v_reg<_Tp, V_TypeTraits<_Tp>::nlanes128> c;
    for (int i = 0; i < V_TypeTraits<_Tp>::nlanes128; i++)
        c.s[i] = tab[idx[i / 4] + i % 4];
    return c;
}

template<int n> inline v_reg<int, n> v_lut(const int* tab, const v_reg<int, n>& idx)
{
    v_reg<int, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = tab[idx.s[i]];
    return c;
}

template<int n> inline v_reg<unsigned, n> v_lut(const unsigned* tab, const v_reg<int, n>& idx)
{
    v_reg<int, n> c;
    for (int i = 0; i < n; i++)
        c.s[i] = tab[idx.s[i]];
    return c;
}

template<int n> inline v_reg<float, n> v_lut(const float* tab, const v_reg<int, n>& idx)
{
    v_reg<float, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = tab[idx.s[i]];
    return c;
}

template<int n> inline v_reg<double, n> v_lut(const double* tab, const v_reg<int, n*2>& idx)
{
    v_reg<double, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = tab[idx.s[i]];
    return c;
}


inline v_int32x4 v_lut(const int* tab, const v_int32x4& idxvec)
{
    return v_lut(tab, idxvec.s);
}

inline v_uint32x4 v_lut(const unsigned* tab, const v_int32x4& idxvec)
{
    return v_lut(tab, idxvec.s);
}

inline v_float32x4 v_lut(const float* tab, const v_int32x4& idxvec)
{
    return v_lut(tab, idxvec.s);
}

inline v_float64x2 v_lut(const double* tab, const v_int32x4& idxvec)
{
    return v_lut(tab, idxvec.s);
}


template<int n> inline void v_lut_deinterleave(const float* tab, const v_reg<int, n>& idx,
                                               v_reg<float, n>& x, v_reg<float, n>& y)
{
    for( int i = 0; i < n; i++ )
    {
        int j = idx.s[i];
        x.s[i] = tab[j];
        y.s[i] = tab[j+1];
    }
}

template<int n> inline void v_lut_deinterleave(const double* tab, const v_reg<int, n*2>& idx,
                                               v_reg<double, n>& x, v_reg<double, n>& y)
{
    for( int i = 0; i < n; i++ )
    {
        int j = idx.s[i];
        x.s[i] = tab[j];
        y.s[i] = tab[j+1];
    }
}

template<typename _Tp, int n> inline v_reg<_Tp, n> v_interleave_pairs(const v_reg<_Tp, n>& vec)
{
    v_reg<_Tp, n> c;
    for (int i = 0; i < n/4; i++)
    {
        c.s[4*i  ] = vec.s[4*i  ];
        c.s[4*i+1] = vec.s[4*i+2];
        c.s[4*i+2] = vec.s[4*i+1];
        c.s[4*i+3] = vec.s[4*i+3];
    }
    return c;
}

template<typename _Tp, int n> inline v_reg<_Tp, n> v_interleave_quads(const v_reg<_Tp, n>& vec)
{
    v_reg<_Tp, n> c;
    for (int i = 0; i < n/8; i++)
    {
        c.s[8*i  ] = vec.s[8*i  ];
        c.s[8*i+1] = vec.s[8*i+4];
        c.s[8*i+2] = vec.s[8*i+1];
        c.s[8*i+3] = vec.s[8*i+5];
        c.s[8*i+4] = vec.s[8*i+2];
        c.s[8*i+5] = vec.s[8*i+6];
        c.s[8*i+6] = vec.s[8*i+3];
        c.s[8*i+7] = vec.s[8*i+7];
    }
    return c;
}

template<typename _Tp, int n> inline v_reg<_Tp, n> v_pack_triplets(const v_reg<_Tp, n>& vec)
{
    v_reg<_Tp, n> c;
    for (int i = 0; i < n/4; i++)
    {
        c.s[3*i  ] = vec.s[4*i  ];
        c.s[3*i+1] = vec.s[4*i+1];
        c.s[3*i+2] = vec.s[4*i+2];
    }
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

#define OPENCV_HAL_IMPL_C_INIT_ZERO(_Tpvec, _Tp, suffix) \
inline _Tpvec v_setzero_##suffix() { return _Tpvec::zero(); }

OPENCV_HAL_IMPL_C_INIT_ZERO(v_uint8x16, uchar, u8)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_int8x16, schar, s8)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_uint16x8, ushort, u16)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_int16x8, short, s16)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_uint32x4, unsigned, u32)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_int32x4, int, s32)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_float32x4, float, f32)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_float64x2, double, f64)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_uint64x2, uint64, u64)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_int64x2, int64, s64)

#define OPENCV_HAL_IMPL_C_INIT_VAL(_Tpvec, _Tp, suffix) \
inline _Tpvec v_setall_##suffix(_Tp val) { return _Tpvec::all(val); }

OPENCV_HAL_IMPL_C_INIT_VAL(v_uint8x16, uchar, u8)
OPENCV_HAL_IMPL_C_INIT_VAL(v_int8x16, schar, s8)
OPENCV_HAL_IMPL_C_INIT_VAL(v_uint16x8, ushort, u16)
OPENCV_HAL_IMPL_C_INIT_VAL(v_int16x8, short, s16)
OPENCV_HAL_IMPL_C_INIT_VAL(v_uint32x4, unsigned, u32)
OPENCV_HAL_IMPL_C_INIT_VAL(v_int32x4, int, s32)
OPENCV_HAL_IMPL_C_INIT_VAL(v_float32x4, float, f32)
OPENCV_HAL_IMPL_C_INIT_VAL(v_float64x2, double, f64)
OPENCV_HAL_IMPL_C_INIT_VAL(v_uint64x2, uint64, u64)
OPENCV_HAL_IMPL_C_INIT_VAL(v_int64x2, int64, s64)

#define OPENCV_HAL_IMPL_C_REINTERPRET(_Tpvec, _Tp, suffix) \
template<typename _Tp0, int n0> inline _Tpvec \
    v_reinterpret_as_##suffix(const v_reg<_Tp0, n0>& a) \
{ return a.template reinterpret_as<_Tp, _Tpvec::nlanes>(); }

OPENCV_HAL_IMPL_C_REINTERPRET(v_uint8x16, uchar, u8)
OPENCV_HAL_IMPL_C_REINTERPRET(v_int8x16, schar, s8)
OPENCV_HAL_IMPL_C_REINTERPRET(v_uint16x8, ushort, u16)
OPENCV_HAL_IMPL_C_REINTERPRET(v_int16x8, short, s16)
OPENCV_HAL_IMPL_C_REINTERPRET(v_uint32x4, unsigned, u32)
OPENCV_HAL_IMPL_C_REINTERPRET(v_int32x4, int, s32)
OPENCV_HAL_IMPL_C_REINTERPRET(v_float32x4, float, f32)
OPENCV_HAL_IMPL_C_REINTERPRET(v_float64x2, double, f64)
OPENCV_HAL_IMPL_C_REINTERPRET(v_uint64x2, uint64, u64)
OPENCV_HAL_IMPL_C_REINTERPRET(v_int64x2, int64, s64)

#define OPENCV_HAL_IMPL_C_SHIFTL(_Tpvec, _Tp) \
template<int n> inline _Tpvec v_shl(const _Tpvec& a) \
{ return a << n; }

OPENCV_HAL_IMPL_C_SHIFTL(v_uint16x8, ushort)
OPENCV_HAL_IMPL_C_SHIFTL(v_int16x8, short)
OPENCV_HAL_IMPL_C_SHIFTL(v_uint32x4, unsigned)
OPENCV_HAL_IMPL_C_SHIFTL(v_int32x4, int)
OPENCV_HAL_IMPL_C_SHIFTL(v_uint64x2, uint64)
OPENCV_HAL_IMPL_C_SHIFTL(v_int64x2, int64)

#define OPENCV_HAL_IMPL_C_SHIFTR(_Tpvec, _Tp) \
template<int n> inline _Tpvec v_shr(const _Tpvec& a) \
{ return a >> n; }

OPENCV_HAL_IMPL_C_SHIFTR(v_uint16x8, ushort)
OPENCV_HAL_IMPL_C_SHIFTR(v_int16x8, short)
OPENCV_HAL_IMPL_C_SHIFTR(v_uint32x4, unsigned)
OPENCV_HAL_IMPL_C_SHIFTR(v_int32x4, int)
OPENCV_HAL_IMPL_C_SHIFTR(v_uint64x2, uint64)
OPENCV_HAL_IMPL_C_SHIFTR(v_int64x2, int64)

#define OPENCV_HAL_IMPL_C_RSHIFTR(_Tpvec, _Tp) \
template<int n> inline _Tpvec v_rshr(const _Tpvec& a) \
{ \
    _Tpvec c; \
    for( int i = 0; i < _Tpvec::nlanes; i++ ) \
        c.s[i] = (_Tp)((a.s[i] + ((_Tp)1 << (n - 1))) >> n); \
    return c; \
}

OPENCV_HAL_IMPL_C_RSHIFTR(v_uint16x8, ushort)
OPENCV_HAL_IMPL_C_RSHIFTR(v_int16x8, short)
OPENCV_HAL_IMPL_C_RSHIFTR(v_uint32x4, unsigned)
OPENCV_HAL_IMPL_C_RSHIFTR(v_int32x4, int)
OPENCV_HAL_IMPL_C_RSHIFTR(v_uint64x2, uint64)
OPENCV_HAL_IMPL_C_RSHIFTR(v_int64x2, int64)

#define OPENCV_HAL_IMPL_C_PACK(_Tpvec, _Tpnvec, _Tpn, pack_suffix, cast) \
inline _Tpnvec v_##pack_suffix(const _Tpvec& a, const _Tpvec& b) \
{ \
    _Tpnvec c; \
    for( int i = 0; i < _Tpvec::nlanes; i++ ) \
    { \
        c.s[i] = cast<_Tpn>(a.s[i]); \
        c.s[i+_Tpvec::nlanes] = cast<_Tpn>(b.s[i]); \
    } \
    return c; \
}

OPENCV_HAL_IMPL_C_PACK(v_uint16x8, v_uint8x16, uchar, pack, saturate_cast)
OPENCV_HAL_IMPL_C_PACK(v_int16x8, v_int8x16, schar, pack, saturate_cast)
OPENCV_HAL_IMPL_C_PACK(v_uint32x4, v_uint16x8, ushort, pack, saturate_cast)
OPENCV_HAL_IMPL_C_PACK(v_int32x4, v_int16x8, short, pack, saturate_cast)
OPENCV_HAL_IMPL_C_PACK(v_uint64x2, v_uint32x4, unsigned, pack, static_cast)
OPENCV_HAL_IMPL_C_PACK(v_int64x2, v_int32x4, int, pack, static_cast)
OPENCV_HAL_IMPL_C_PACK(v_int16x8, v_uint8x16, uchar, pack_u, saturate_cast)
OPENCV_HAL_IMPL_C_PACK(v_int32x4, v_uint16x8, ushort, pack_u, saturate_cast)

#define OPENCV_HAL_IMPL_C_RSHR_PACK(_Tpvec, _Tp, _Tpnvec, _Tpn, pack_suffix, cast) \
template<int n> inline _Tpnvec v_rshr_##pack_suffix(const _Tpvec& a, const _Tpvec& b) \
{ \
    _Tpnvec c; \
    for( int i = 0; i < _Tpvec::nlanes; i++ ) \
    { \
        c.s[i] = cast<_Tpn>((a.s[i] + ((_Tp)1 << (n - 1))) >> n); \
        c.s[i+_Tpvec::nlanes] = cast<_Tpn>((b.s[i] + ((_Tp)1 << (n - 1))) >> n); \
    } \
    return c; \
}

OPENCV_HAL_IMPL_C_RSHR_PACK(v_uint16x8, ushort, v_uint8x16, uchar, pack, saturate_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK(v_int16x8, short, v_int8x16, schar, pack, saturate_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK(v_uint32x4, unsigned, v_uint16x8, ushort, pack, saturate_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK(v_int32x4, int, v_int16x8, short, pack, saturate_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK(v_uint64x2, uint64, v_uint32x4, unsigned, pack, static_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK(v_int64x2, int64, v_int32x4, int, pack, static_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK(v_int16x8, short, v_uint8x16, uchar, pack_u, saturate_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK(v_int32x4, int, v_uint16x8, ushort, pack_u, saturate_cast)

#define OPENCV_HAL_IMPL_C_PACK_STORE(_Tpvec, _Tp, _Tpnvec, _Tpn, pack_suffix, cast) \
inline void v_##pack_suffix##_store(_Tpn* ptr, const _Tpvec& a) \
{ \
    for( int i = 0; i < _Tpvec::nlanes; i++ ) \
        ptr[i] = cast<_Tpn>(a.s[i]); \
}

OPENCV_HAL_IMPL_C_PACK_STORE(v_uint16x8, ushort, v_uint8x16, uchar, pack, saturate_cast)
OPENCV_HAL_IMPL_C_PACK_STORE(v_int16x8, short, v_int8x16, schar, pack, saturate_cast)
OPENCV_HAL_IMPL_C_PACK_STORE(v_uint32x4, unsigned, v_uint16x8, ushort, pack, saturate_cast)
OPENCV_HAL_IMPL_C_PACK_STORE(v_int32x4, int, v_int16x8, short, pack, saturate_cast)
OPENCV_HAL_IMPL_C_PACK_STORE(v_uint64x2, uint64, v_uint32x4, unsigned, pack, static_cast)
OPENCV_HAL_IMPL_C_PACK_STORE(v_int64x2, int64, v_int32x4, int, pack, static_cast)
OPENCV_HAL_IMPL_C_PACK_STORE(v_int16x8, short, v_uint8x16, uchar, pack_u, saturate_cast)
OPENCV_HAL_IMPL_C_PACK_STORE(v_int32x4, int, v_uint16x8, ushort, pack_u, saturate_cast)

#define OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(_Tpvec, _Tp, _Tpnvec, _Tpn, pack_suffix, cast) \
template<int n> inline void v_rshr_##pack_suffix##_store(_Tpn* ptr, const _Tpvec& a) \
{ \
    for( int i = 0; i < _Tpvec::nlanes; i++ ) \
        ptr[i] = cast<_Tpn>((a.s[i] + ((_Tp)1 << (n - 1))) >> n); \
}

OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(v_uint16x8, ushort, v_uint8x16, uchar, pack, saturate_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(v_int16x8, short, v_int8x16, schar, pack, saturate_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(v_uint32x4, unsigned, v_uint16x8, ushort, pack, saturate_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(v_int32x4, int, v_int16x8, short, pack, saturate_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(v_uint64x2, uint64, v_uint32x4, unsigned, pack, static_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(v_int64x2, int64, v_int32x4, int, pack, static_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(v_int16x8, short, v_uint8x16, uchar, pack_u, saturate_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(v_int32x4, int, v_uint16x8, ushort, pack_u, saturate_cast)

template<typename _Tpm, typename _Tp, int n>
inline void _pack_b(_Tpm* mptr, const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{
    for (int i = 0; i < n; ++i)
    {
        mptr[i] = (_Tpm)a.s[i];
        mptr[i + n] = (_Tpm)b.s[i];
    }
}



inline v_uint8x16 v_pack_b(const v_uint16x8& a, const v_uint16x8& b)
{
    v_uint8x16 mask;
    _pack_b(mask.s, a, b);
    return mask;
}


inline v_uint8x16 v_pack_b(const v_uint32x4& a, const v_uint32x4& b,
                           const v_uint32x4& c, const v_uint32x4& d)
{
    v_uint8x16 mask;
    _pack_b(mask.s, a, b);
    _pack_b(mask.s + 8, c, d);
    return mask;
}

inline v_uint8x16 v_pack_b(const v_uint64x2& a, const v_uint64x2& b, const v_uint64x2& c,
                           const v_uint64x2& d, const v_uint64x2& e, const v_uint64x2& f,
                           const v_uint64x2& g, const v_uint64x2& h)
{
    v_uint8x16 mask;
    _pack_b(mask.s, a, b);
    _pack_b(mask.s + 4, c, d);
    _pack_b(mask.s + 8, e, f);
    _pack_b(mask.s + 12, g, h);
    return mask;
}

inline v_float32x4 v_matmul(const v_float32x4& v, const v_float32x4& m0,
                            const v_float32x4& m1, const v_float32x4& m2,
                            const v_float32x4& m3)
{
    return v_float32x4(v.s[0]*m0.s[0] + v.s[1]*m1.s[0] + v.s[2]*m2.s[0] + v.s[3]*m3.s[0],
                       v.s[0]*m0.s[1] + v.s[1]*m1.s[1] + v.s[2]*m2.s[1] + v.s[3]*m3.s[1],
                       v.s[0]*m0.s[2] + v.s[1]*m1.s[2] + v.s[2]*m2.s[2] + v.s[3]*m3.s[2],
                       v.s[0]*m0.s[3] + v.s[1]*m1.s[3] + v.s[2]*m2.s[3] + v.s[3]*m3.s[3]);
}

inline v_float32x4 v_matmuladd(const v_float32x4& v, const v_float32x4& m0,
                               const v_float32x4& m1, const v_float32x4& m2,
                               const v_float32x4& m3)
{
    return v_float32x4(v.s[0]*m0.s[0] + v.s[1]*m1.s[0] + v.s[2]*m2.s[0] + m3.s[0],
                       v.s[0]*m0.s[1] + v.s[1]*m1.s[1] + v.s[2]*m2.s[1] + m3.s[1],
                       v.s[0]*m0.s[2] + v.s[1]*m1.s[2] + v.s[2]*m2.s[2] + m3.s[2],
                       v.s[0]*m0.s[3] + v.s[1]*m1.s[3] + v.s[2]*m2.s[3] + m3.s[3]);
}


inline v_float64x2 v_dotprod_expand(const v_int32x4& a, const v_int32x4& b)
{ return v_fma(v_cvt_f64(a), v_cvt_f64(b), v_cvt_f64_high(a) * v_cvt_f64_high(b)); }
inline v_float64x2 v_dotprod_expand(const v_int32x4& a, const v_int32x4& b, const v_float64x2& c)
{ return v_fma(v_cvt_f64(a), v_cvt_f64(b), v_fma(v_cvt_f64_high(a), v_cvt_f64_high(b), c)); }

inline v_float64x2 v_dotprod_expand_fast(const v_int32x4& a, const v_int32x4& b)
{ return v_dotprod_expand(a, b); }
inline v_float64x2 v_dotprod_expand_fast(const v_int32x4& a, const v_int32x4& b, const v_float64x2& c)
{ return v_dotprod_expand(a, b, c); }

////// FP16 support ///////

inline v_reg<float, V_TypeTraits<float>::nlanes128>
v_load_expand(const float16_t* ptr)
{
    v_reg<float, V_TypeTraits<float>::nlanes128> v;
    for( int i = 0; i < v.nlanes; i++ )
    {
        v.s[i] = ptr[i];
    }
    return v;
}

inline void
v_pack_store(float16_t* ptr, const v_reg<float, V_TypeTraits<float>::nlanes128>& v)
{
    for( int i = 0; i < v.nlanes; i++ )
    {
        ptr[i] = float16_t(v.s[i]);
    }
}

inline void v_cleanup() {}


#ifndef CV_DOXYGEN
CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END
#endif
}

#endif
