// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_HAL_INTRIN_WASM_HPP
#define OPENCV_HAL_INTRIN_WASM_HPP

#include <limits>
#include <cstring>
#include <algorithm>
#include "opencv2/core/saturate.hpp"

#define CV_SIMD128 1
#define CV_SIMD128_64F 0 // Now all implementation of f64 use fallback, so disable it.
#define CV_SIMD128_FP16 0

namespace cv
{

//! @cond IGNORED

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN

#if (__EMSCRIPTEN_major__ * 1000000 + __EMSCRIPTEN_minor__ * 1000 + __EMSCRIPTEN_tiny__) < (1038046)
// handle renames: https://github.com/emscripten-core/emscripten/pull/9440 (https://github.com/emscripten-core/emscripten/commit/755d5b46cb84d0aa120c10981b11d05646c29673)
#define wasm_i32x4_trunc_saturate_f32x4 wasm_trunc_saturate_i32x4_f32x4
#define wasm_u32x4_trunc_saturate_f32x4 wasm_trunc_saturate_u32x4_f32x4
#define wasm_i64x2_trunc_saturate_f64x2 wasm_trunc_saturate_i64x2_f64x2
#define wasm_u64x2_trunc_saturate_f64x2 wasm_trunc_saturate_u64x2_f64x2
#define wasm_f32x4_convert_i32x4 wasm_convert_f32x4_i32x4
#define wasm_f32x4_convert_u32x4 wasm_convert_f32x4_u32x4
#define wasm_f64x2_convert_i64x2 wasm_convert_f64x2_i64x2
#define wasm_f64x2_convert_u64x2 wasm_convert_f64x2_u64x2
#endif // COMPATIBILITY: <1.38.46

///////// Types ///////////

struct v_uint8x16
{
    typedef uchar lane_type;
    typedef v128_t vector_type;
    enum { nlanes = 16 };

    v_uint8x16() : val(wasm_i8x16_splat(0)) {}
    explicit v_uint8x16(v128_t v) : val(v) {}
    v_uint8x16(uchar v0, uchar v1, uchar v2, uchar v3, uchar v4, uchar v5, uchar v6, uchar v7,
            uchar v8, uchar v9, uchar v10, uchar v11, uchar v12, uchar v13, uchar v14, uchar v15)
    {
        uchar v[] = {v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15};
        val = wasm_v128_load(v);
    }
    uchar get0() const
    {
        return (uchar)wasm_i8x16_extract_lane(val, 0);
    }

    v128_t val;
};

struct v_int8x16
{
    typedef schar lane_type;
    typedef v128_t vector_type;
    enum { nlanes = 16 };

    v_int8x16() : val(wasm_i8x16_splat(0)) {}
    explicit v_int8x16(v128_t v) : val(v) {}
    v_int8x16(schar v0, schar v1, schar v2, schar v3, schar v4, schar v5, schar v6, schar v7,
            schar v8, schar v9, schar v10, schar v11, schar v12, schar v13, schar v14, schar v15)
    {
        schar v[] = {v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15};
        val = wasm_v128_load(v);
    }
    schar get0() const
    {
        return wasm_i8x16_extract_lane(val, 0);
    }

    v128_t val;
};

struct v_uint16x8
{
    typedef ushort lane_type;
    typedef v128_t vector_type;
    enum { nlanes = 8 };

    v_uint16x8() : val(wasm_i16x8_splat(0)) {}
    explicit v_uint16x8(v128_t v) : val(v) {}
    v_uint16x8(ushort v0, ushort v1, ushort v2, ushort v3, ushort v4, ushort v5, ushort v6, ushort v7)
    {
        ushort v[] = {v0, v1, v2, v3, v4, v5, v6, v7};
        val = wasm_v128_load(v);
    }
    ushort get0() const
    {
        return (ushort)wasm_i16x8_extract_lane(val, 0);    // wasm_u16x8_extract_lane() unimplemented yet
    }

    v128_t val;
};

struct v_int16x8
{
    typedef short lane_type;
    typedef v128_t vector_type;
    enum { nlanes = 8 };

    v_int16x8() : val(wasm_i16x8_splat(0)) {}
    explicit v_int16x8(v128_t v) : val(v) {}
    v_int16x8(short v0, short v1, short v2, short v3, short v4, short v5, short v6, short v7)
    {
        short v[] = {v0, v1, v2, v3, v4, v5, v6, v7};
        val = wasm_v128_load(v);
    }
    short get0() const
    {
        return wasm_i16x8_extract_lane(val, 0);
    }

    v128_t val;
};

struct v_uint32x4
{
    typedef unsigned lane_type;
    typedef v128_t vector_type;
    enum { nlanes = 4 };

    v_uint32x4() : val(wasm_i32x4_splat(0)) {}
    explicit v_uint32x4(v128_t v) : val(v) {}
    v_uint32x4(unsigned v0, unsigned v1, unsigned v2, unsigned v3)
    {
        unsigned v[] = {v0, v1, v2, v3};
        val = wasm_v128_load(v);
    }
    unsigned get0() const
    {
        return (unsigned)wasm_i32x4_extract_lane(val, 0);
    }

    v128_t val;
};

struct v_int32x4
{
    typedef int lane_type;
    typedef v128_t vector_type;
    enum { nlanes = 4 };

    v_int32x4() : val(wasm_i32x4_splat(0)) {}
    explicit v_int32x4(v128_t v) : val(v) {}
    v_int32x4(int v0, int v1, int v2, int v3)
    {
        int v[] = {v0, v1, v2, v3};
        val = wasm_v128_load(v);
    }
    int get0() const
    {
        return wasm_i32x4_extract_lane(val, 0);
    }

    v128_t val;
};

struct v_float32x4
{
    typedef float lane_type;
    typedef v128_t vector_type;
    enum { nlanes = 4 };

    v_float32x4() : val(wasm_f32x4_splat(0)) {}
    explicit v_float32x4(v128_t v) : val(v) {}
    v_float32x4(float v0, float v1, float v2, float v3)
    {
        float v[] = {v0, v1, v2, v3};
        val = wasm_v128_load(v);
    }
    float get0() const
    {
        return wasm_f32x4_extract_lane(val, 0);
    }

    v128_t val;
};

struct v_uint64x2
{
    typedef uint64 lane_type;
    typedef v128_t vector_type;
    enum { nlanes = 2 };

#ifdef __wasm_unimplemented_simd128__
    v_uint64x2() : val(wasm_i64x2_splat(0)) {}
#else
    v_uint64x2() : val(wasm_i32x4_splat(0)) {}
#endif
    explicit v_uint64x2(v128_t v) : val(v) {}
    v_uint64x2(uint64 v0, uint64 v1)
    {
        uint64 v[] = {v0, v1};
        val = wasm_v128_load(v);
    }
    uint64 get0() const
    {
#ifdef __wasm_unimplemented_simd128__
        return (uint64)wasm_i64x2_extract_lane(val, 0);
#else
        uint64 des[2];
        wasm_v128_store(des, val);
        return des[0];
#endif
    }

    v128_t val;
};

struct v_int64x2
{
    typedef int64 lane_type;
    typedef v128_t vector_type;
    enum { nlanes = 2 };

#ifdef __wasm_unimplemented_simd128__
    v_int64x2() : val(wasm_i64x2_splat(0)) {}
#else
    v_int64x2() : val(wasm_i32x4_splat(0)) {}
#endif
    explicit v_int64x2(v128_t v) : val(v) {}
    v_int64x2(int64 v0, int64 v1)
    {
        int64 v[] = {v0, v1};
        val = wasm_v128_load(v);
    }
    int64 get0() const
    {
#ifdef __wasm_unimplemented_simd128__
        return wasm_i64x2_extract_lane(val, 0);
#else
        int64 des[2];
        wasm_v128_store(des, val);
        return des[0];
#endif
    }

    v128_t val;
};

struct v_float64x2
{
    typedef double lane_type;
    typedef v128_t vector_type;
    enum { nlanes = 2 };

#ifdef __wasm_unimplemented_simd128__
    v_float64x2() : val(wasm_f64x2_splat(0)) {}
#else
    v_float64x2() : val(wasm_f32x4_splat(0)) {}
#endif
    explicit v_float64x2(v128_t v) : val(v) {}
    v_float64x2(double v0, double v1)
    {
        double v[] = {v0, v1};
        val = wasm_v128_load(v);
    }
    double get0() const
    {
#ifdef __wasm_unimplemented_simd128__
        return wasm_f64x2_extract_lane(val, 0);
#else
        double des[2];
        wasm_v128_store(des, val);
        return des[0];
#endif
    }

    v128_t val;
};

namespace fallback
{

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

    v_reg(const cv::v_uint8x16& v) { wasm_v128_store(&s, v.val); }
    v_reg(const cv::v_int8x16& v) { wasm_v128_store(&s, v.val); }
    v_reg(const cv::v_uint16x8& v) { wasm_v128_store(&s, v.val); }
    v_reg(const cv::v_int16x8& v) { wasm_v128_store(&s, v.val); }
    v_reg(const cv::v_uint32x4& v) { wasm_v128_store(&s, v.val); }
    v_reg(const cv::v_int32x4& v) { wasm_v128_store(&s, v.val); }
    v_reg(const cv::v_float32x4& v) { wasm_v128_store(&s, v.val); }
    v_reg(const cv::v_float64x2& v) { wasm_v128_store(&s, v.val); }
    v_reg(const cv::v_uint64x2& v) { wasm_v128_store(&s, v.val); }
    v_reg(const cv::v_int64x2& v) { wasm_v128_store(&s, v.val); }

    operator cv::v_uint8x16() const { return cv::v_uint8x16(wasm_v128_load(&s)); }
    operator cv::v_int8x16() const { return cv::v_int8x16(wasm_v128_load(&s)); }
    operator cv::v_uint16x8() const { return cv::v_uint16x8(wasm_v128_load(&s)); }
    operator cv::v_int16x8() const { return cv::v_int16x8(wasm_v128_load(&s)); }
    operator cv::v_uint32x4() const { return cv::v_uint32x4(wasm_v128_load(&s)); }
    operator cv::v_int32x4() const { return cv::v_int32x4(wasm_v128_load(&s)); }
    operator cv::v_float32x4() const { return cv::v_float32x4(wasm_v128_load(&s)); }
    operator cv::v_float64x2() const { return cv::v_float64x2(wasm_v128_load(&s)); }
    operator cv::v_uint64x2() const { return cv::v_uint64x2(wasm_v128_load(&s)); }
    operator cv::v_int64x2() const { return cv::v_int64x2(wasm_v128_load(&s)); }

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
    {
        c.s[i] = V_TypeTraits<_Tp>::reinterpret_from_int(~V_TypeTraits<_Tp>::reinterpret_int(a.s[i]));
    }
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
    for (int i = 0; i < (int)(n*sizeof(_Tp)); i++)
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
    v_dotprod(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b, const v_reg<typename V_TypeTraits<_Tp>::w_type, n / 2>& c)
{
    typedef typename V_TypeTraits<_Tp>::w_type w_type;
    v_reg<w_type, n/2> s;
    for( int i = 0; i < (n/2); i++ )
        s.s[i] = (w_type)a.s[i*2]*b.s[i*2] + (w_type)a.s[i*2+1]*b.s[i*2+1] + c.s[i];
    return s;
}

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
    return v_reg<_Tp, V_TypeTraits<_Tp>::nlanes128>(ptr);
}

template<typename _Tp>
inline v_reg<_Tp, V_TypeTraits<_Tp>::nlanes128> v_load_aligned(const _Tp* ptr)
{
    return v_reg<_Tp, V_TypeTraits<_Tp>::nlanes128>(ptr);
}

template<typename _Tp>
inline v_reg<_Tp, V_TypeTraits<_Tp>::nlanes128> v_load_low(const _Tp* ptr)
{
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
                               const v_reg<_Tp, n>& b,
                               hal::StoreMode /*mode*/=hal::STORE_UNALIGNED)
{
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
inline void v_store(_Tp* ptr, const v_reg<_Tp, n>& a, hal::StoreMode /*mode*/ = hal::STORE_UNALIGNED)
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
inline void v_store_aligned_nocache(_Tp* ptr, const v_reg<_Tp, n>& a)
{
    for( int i = 0; i < n; i++ )
        ptr[i] = a.s[i];
}

template<typename _Tp, int n>
inline void v_store_aligned(_Tp* ptr, const v_reg<_Tp, n>& a, hal::StoreMode /*mode*/)
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
    v_reg<int, n*2> c;
    for( int i = 0; i < n; i++ )
    {
        c.s[i] = cvFloor(a.s[i]);
        c.s[i+n] = 0;
    }
    return c;
}

template<int n> inline v_reg<int, n*2> v_ceil(const v_reg<double, n>& a)
{
    v_reg<int, n*2> c;
    for( int i = 0; i < n; i++ )
    {
        c.s[i] = cvCeil(a.s[i]);
        c.s[i+n] = 0;
    }
    return c;
}

template<int n> inline v_reg<int, n*2> v_trunc(const v_reg<double, n>& a)
{
    v_reg<int, n*2> c;
    for( int i = 0; i < n; i++ )
    {
        c.s[i] = (int)(a.s[i]);
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

inline v_float64x2 v_cvt_f64(const v_int32x4& a)
{
    v_float64x2 c;
    for( int i = 0; i < 2; i++ )
        c.s[i] = (double)a.s[i];
    return c;
}

inline v_float64x2 v_cvt_f64_high(const v_int32x4& a)
{
    v_float64x2 c;
    for( int i = 0; i < 2; i++ )
        c.s[i] = (double)a.s[i+2];
    return c;
}

inline v_float64x2 v_cvt_f64(const v_float32x4& a)
{
    v_float64x2 c;
    for( int i = 0; i < 2; i++ )
        c.s[i] = (double)a.s[i];
    return c;
}

inline v_float64x2 v_cvt_f64_high(const v_float32x4& a)
{
    v_float64x2 c;
    for( int i = 0; i < 2; i++ )
        c.s[i] = (double)a.s[i+2];
    return c;
}

inline v_float64x2 v_cvt_f64(const v_int64x2& a)
{
    v_float64x2 c;
    for( int i = 0; i < 2; i++ )
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
}  // namespace fallback

static v128_t wasm_unpacklo_i8x16(v128_t a, v128_t b) {
    return wasm_v8x16_shuffle(a, b, 0,16,1,17,2,18,3,19,4,20,5,21,6,22,7,23);
}

static v128_t wasm_unpacklo_i16x8(v128_t a, v128_t b) {
    return wasm_v8x16_shuffle(a, b, 0,1,16,17,2,3,18,19,4,5,20,21,6,7,22,23);
}

static v128_t wasm_unpacklo_i32x4(v128_t a, v128_t b) {
    return wasm_v8x16_shuffle(a, b, 0,1,2,3,16,17,18,19,4,5,6,7,20,21,22,23);
}

static v128_t wasm_unpacklo_i64x2(v128_t a, v128_t b) {
    return wasm_v8x16_shuffle(a, b, 0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23);
}

static v128_t wasm_unpackhi_i8x16(v128_t a, v128_t b) {
    return wasm_v8x16_shuffle(a, b, 8,24,9,25,10,26,11,27,12,28,13,29,14,30,15,31);
}

static v128_t wasm_unpackhi_i16x8(v128_t a, v128_t b) {
    return wasm_v8x16_shuffle(a, b, 8,9,24,25,10,11,26,27,12,13,28,29,14,15,30,31);
}

static v128_t wasm_unpackhi_i32x4(v128_t a, v128_t b) {
    return wasm_v8x16_shuffle(a, b, 8,9,10,11,24,25,26,27,12,13,14,15,28,29,30,31);
}

static v128_t wasm_unpackhi_i64x2(v128_t a, v128_t b) {
    return wasm_v8x16_shuffle(a, b, 8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31);
}

/** Convert **/
// 8 >> 16
inline v128_t v128_cvtu8x16_i16x8(const v128_t& a)
{
    const v128_t z = wasm_i8x16_splat(0);
    return wasm_unpacklo_i8x16(a, z);
}
inline v128_t v128_cvti8x16_i16x8(const v128_t& a)
{ return wasm_i16x8_shr(wasm_unpacklo_i8x16(a, a), 8); }
// 8 >> 32
inline v128_t v128_cvtu8x16_i32x4(const v128_t& a)
{
    const v128_t z = wasm_i8x16_splat(0);
    return wasm_unpacklo_i16x8(wasm_unpacklo_i8x16(a, z), z);
}
inline v128_t v128_cvti8x16_i32x4(const v128_t& a)
{
    v128_t r = wasm_unpacklo_i8x16(a, a);
    r = wasm_unpacklo_i8x16(r, r);
    return wasm_i32x4_shr(r, 24);
}
// 16 >> 32
inline v128_t v128_cvtu16x8_i32x4(const v128_t& a)
{
    const v128_t z = wasm_i8x16_splat(0);
    return wasm_unpacklo_i16x8(a, z);
}
inline v128_t v128_cvti16x8_i32x4(const v128_t& a)
{ return wasm_i32x4_shr(wasm_unpacklo_i16x8(a, a), 16); }
// 32 >> 64
inline v128_t v128_cvtu32x4_i64x2(const v128_t& a)
{
    const v128_t z = wasm_i8x16_splat(0);
    return wasm_unpacklo_i32x4(a, z);
}
inline v128_t v128_cvti32x4_i64x2(const v128_t& a)
{ return wasm_unpacklo_i32x4(a, wasm_i32x4_shr(a, 31)); }

// 16 << 8
inline v128_t v128_cvtu8x16_i16x8_high(const v128_t& a)
{
    const v128_t z = wasm_i8x16_splat(0);
    return wasm_unpackhi_i8x16(a, z);
}
inline v128_t v128_cvti8x16_i16x8_high(const v128_t& a)
{ return wasm_i16x8_shr(wasm_unpackhi_i8x16(a, a), 8); }
// 32 << 16
inline v128_t v128_cvtu16x8_i32x4_high(const v128_t& a)
{
    const v128_t z = wasm_i8x16_splat(0);
    return wasm_unpackhi_i16x8(a, z);
}
inline v128_t v128_cvti16x8_i32x4_high(const v128_t& a)
{ return wasm_i32x4_shr(wasm_unpackhi_i16x8(a, a), 16); }
// 64 << 32
inline v128_t v128_cvtu32x4_i64x2_high(const v128_t& a)
{
    const v128_t z = wasm_i8x16_splat(0);
    return wasm_unpackhi_i32x4(a, z);
}
inline v128_t v128_cvti32x4_i64x2_high(const v128_t& a)
{ return wasm_unpackhi_i32x4(a, wasm_i32x4_shr(a, 31)); }

#define OPENCV_HAL_IMPL_WASM_INITVEC(_Tpvec, _Tp, suffix, zsuffix, _Tps) \
inline _Tpvec v_setzero_##suffix() { return _Tpvec(wasm_##zsuffix##_splat((_Tps)0)); } \
inline _Tpvec v_setall_##suffix(_Tp v) { return _Tpvec(wasm_##zsuffix##_splat((_Tps)v)); } \
template<typename _Tpvec0> inline _Tpvec v_reinterpret_as_##suffix(const _Tpvec0& a) \
{ return _Tpvec(a.val); }

OPENCV_HAL_IMPL_WASM_INITVEC(v_uint8x16, uchar, u8, i8x16, schar)
OPENCV_HAL_IMPL_WASM_INITVEC(v_int8x16, schar, s8, i8x16, schar)
OPENCV_HAL_IMPL_WASM_INITVEC(v_uint16x8, ushort, u16, i16x8, short)
OPENCV_HAL_IMPL_WASM_INITVEC(v_int16x8, short, s16, i16x8, short)
OPENCV_HAL_IMPL_WASM_INITVEC(v_uint32x4, unsigned, u32, i32x4, int)
OPENCV_HAL_IMPL_WASM_INITVEC(v_int32x4, int, s32, i32x4, int)
OPENCV_HAL_IMPL_WASM_INITVEC(v_float32x4, float, f32, f32x4, float)

#ifdef __wasm_unimplemented_simd128__
OPENCV_HAL_IMPL_WASM_INITVEC(v_uint64x2, uint64, u64, i64x2, int64)
OPENCV_HAL_IMPL_WASM_INITVEC(v_int64x2, int64, s64, i64x2, int64)
OPENCV_HAL_IMPL_WASM_INITVEC(v_float64x2, double, f64, f64x2, double)
#else
#define OPENCV_HAL_IMPL_FALLBACK_INITVEC(_Tpvec, _Tp, suffix, _Tps) \
inline _Tpvec v_setzero_##suffix() { return _Tpvec((_Tps)0, (_Tps)0); } \
inline _Tpvec v_setall_##suffix(_Tp v) { return _Tpvec((_Tps)v, (_Tps)v); } \
template<typename _Tpvec0> inline _Tpvec v_reinterpret_as_##suffix(const _Tpvec0& a) \
{ return _Tpvec(a.val); }

OPENCV_HAL_IMPL_FALLBACK_INITVEC(v_uint64x2, uint64, u64, int64)
OPENCV_HAL_IMPL_FALLBACK_INITVEC(v_int64x2, int64, s64, int64)
OPENCV_HAL_IMPL_FALLBACK_INITVEC(v_float64x2, double, f64, double)
#endif

//////////////// PACK ///////////////
inline v_uint8x16 v_pack(const v_uint16x8& a, const v_uint16x8& b)
{
    v128_t maxval = wasm_i16x8_splat(255);
    v128_t a1 = wasm_v128_bitselect(maxval, a.val, wasm_u16x8_gt(a.val, maxval));
    v128_t b1 = wasm_v128_bitselect(maxval, b.val, wasm_u16x8_gt(b.val, maxval));
    return v_uint8x16(wasm_v8x16_shuffle(a1, b1, 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30));
}
inline v_int8x16 v_pack(const v_int16x8& a, const v_int16x8& b)
{
    v128_t maxval = wasm_i16x8_splat(127);
    v128_t minval = wasm_i16x8_splat(-128);
    v128_t a1 = wasm_v128_bitselect(maxval, a.val, wasm_i16x8_gt(a.val, maxval));
    v128_t b1 = wasm_v128_bitselect(maxval, b.val, wasm_i16x8_gt(b.val, maxval));
    v128_t a2 = wasm_v128_bitselect(minval, a1, wasm_i16x8_lt(a1, minval));
    v128_t b2 = wasm_v128_bitselect(minval, b1, wasm_i16x8_lt(b1, minval));
    return v_int8x16(wasm_v8x16_shuffle(a2, b2, 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30));
}
inline v_uint16x8 v_pack(const v_uint32x4& a, const v_uint32x4& b)
{
    v128_t maxval = wasm_i32x4_splat(65535);
    v128_t a1 = wasm_v128_bitselect(maxval, a.val, wasm_u32x4_gt(a.val, maxval));
    v128_t b1 = wasm_v128_bitselect(maxval, b.val, wasm_u32x4_gt(b.val, maxval));
    return v_uint16x8(wasm_v8x16_shuffle(a1, b1, 0,1,4,5,8,9,12,13,16,17,20,21,24,25,28,29));
}
inline v_int16x8 v_pack(const v_int32x4& a, const v_int32x4& b)
{
    v128_t maxval = wasm_i32x4_splat(32767);
    v128_t minval = wasm_i32x4_splat(-32768);
    v128_t a1 = wasm_v128_bitselect(maxval, a.val, wasm_i32x4_gt(a.val, maxval));
    v128_t b1 = wasm_v128_bitselect(maxval, b.val, wasm_i32x4_gt(b.val, maxval));
    v128_t a2 = wasm_v128_bitselect(minval, a1, wasm_i32x4_lt(a1, minval));
    v128_t b2 = wasm_v128_bitselect(minval, b1, wasm_i32x4_lt(b1, minval));
    return v_int16x8(wasm_v8x16_shuffle(a2, b2, 0,1,4,5,8,9,12,13,16,17,20,21,24,25,28,29));
}
inline v_uint32x4 v_pack(const v_uint64x2& a, const v_uint64x2& b)
{
    return v_uint32x4(wasm_v8x16_shuffle(a.val, b.val, 0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27));
}
inline v_int32x4 v_pack(const v_int64x2& a, const v_int64x2& b)
{
    return v_int32x4(wasm_v8x16_shuffle(a.val, b.val, 0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27));
}
inline v_uint8x16 v_pack_u(const v_int16x8& a, const v_int16x8& b)
{
    v128_t maxval = wasm_i16x8_splat(255);
    v128_t minval = wasm_i16x8_splat(0);
    v128_t a1 = wasm_v128_bitselect(maxval, a.val, wasm_i16x8_gt(a.val, maxval));
    v128_t b1 = wasm_v128_bitselect(maxval, b.val, wasm_i16x8_gt(b.val, maxval));
    v128_t a2 = wasm_v128_bitselect(minval, a1, wasm_i16x8_lt(a1, minval));
    v128_t b2 = wasm_v128_bitselect(minval, b1, wasm_i16x8_lt(b1, minval));
    return v_uint8x16(wasm_v8x16_shuffle(a2, b2, 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30));
}
inline v_uint16x8 v_pack_u(const v_int32x4& a, const v_int32x4& b)
{
    v128_t maxval = wasm_i32x4_splat(65535);
    v128_t minval = wasm_i32x4_splat(0);
    v128_t a1 = wasm_v128_bitselect(maxval, a.val, wasm_i32x4_gt(a.val, maxval));
    v128_t b1 = wasm_v128_bitselect(maxval, b.val, wasm_i32x4_gt(b.val, maxval));
    v128_t a2 = wasm_v128_bitselect(minval, a1, wasm_i32x4_lt(a1, minval));
    v128_t b2 = wasm_v128_bitselect(minval, b1, wasm_i32x4_lt(b1, minval));
    return v_uint16x8(wasm_v8x16_shuffle(a2, b2, 0,1,4,5,8,9,12,13,16,17,20,21,24,25,28,29));
}

template<int n>
inline v_uint8x16 v_rshr_pack(const v_uint16x8& a, const v_uint16x8& b)
{
    v128_t delta = wasm_i16x8_splat(((short)1 << (n-1)));
    v128_t a1 = wasm_u16x8_shr(wasm_i16x8_add(a.val, delta), n);
    v128_t b1 = wasm_u16x8_shr(wasm_i16x8_add(b.val, delta), n);
    v128_t maxval = wasm_i16x8_splat(255);
    v128_t a2 = wasm_v128_bitselect(maxval, a1, wasm_u16x8_gt(a1, maxval));
    v128_t b2 = wasm_v128_bitselect(maxval, b1, wasm_u16x8_gt(b1, maxval));
    return v_uint8x16(wasm_v8x16_shuffle(a2, b2, 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30));
}
template<int n>
inline v_int8x16 v_rshr_pack(const v_int16x8& a, const v_int16x8& b)
{
    v128_t delta = wasm_i16x8_splat(((short)1 << (n-1)));
    v128_t a1 = wasm_i16x8_shr(wasm_i16x8_add(a.val, delta), n);
    v128_t b1 = wasm_i16x8_shr(wasm_i16x8_add(b.val, delta), n);
    v128_t maxval = wasm_i16x8_splat(127);
    v128_t minval = wasm_i16x8_splat(-128);
    v128_t a2 = wasm_v128_bitselect(maxval, a1, wasm_i16x8_gt(a1, maxval));
    v128_t b2 = wasm_v128_bitselect(maxval, b1, wasm_i16x8_gt(b1, maxval));
    v128_t a3 = wasm_v128_bitselect(minval, a2, wasm_i16x8_lt(a1, minval));
    v128_t b3 = wasm_v128_bitselect(minval, b2, wasm_i16x8_lt(b1, minval));
    return v_int8x16(wasm_v8x16_shuffle(a3, b3, 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30));
}
template<int n>
inline v_uint16x8 v_rshr_pack(const v_uint32x4& a, const v_uint32x4& b)
{
    v128_t delta = wasm_i32x4_splat(((int)1 << (n-1)));
    v128_t a1 = wasm_u32x4_shr(wasm_i32x4_add(a.val, delta), n);
    v128_t b1 = wasm_u32x4_shr(wasm_i32x4_add(b.val, delta), n);
    v128_t maxval = wasm_i32x4_splat(65535);
    v128_t a2 = wasm_v128_bitselect(maxval, a1, wasm_u32x4_gt(a1, maxval));
    v128_t b2 = wasm_v128_bitselect(maxval, b1, wasm_u32x4_gt(b1, maxval));
    return v_uint16x8(wasm_v8x16_shuffle(a2, b2, 0,1,4,5,8,9,12,13,16,17,20,21,24,25,28,29));
}
template<int n>
inline v_int16x8 v_rshr_pack(const v_int32x4& a, const v_int32x4& b)
{
    v128_t delta = wasm_i32x4_splat(((int)1 << (n-1)));
    v128_t a1 = wasm_i32x4_shr(wasm_i32x4_add(a.val, delta), n);
    v128_t b1 = wasm_i32x4_shr(wasm_i32x4_add(b.val, delta), n);
    v128_t maxval = wasm_i32x4_splat(32767);
    v128_t minval = wasm_i16x8_splat(-32768);
    v128_t a2 = wasm_v128_bitselect(maxval, a1, wasm_i32x4_gt(a1, maxval));
    v128_t b2 = wasm_v128_bitselect(maxval, b1, wasm_i32x4_gt(b1, maxval));
    v128_t a3 = wasm_v128_bitselect(minval, a2, wasm_i32x4_lt(a1, minval));
    v128_t b3 = wasm_v128_bitselect(minval, b2, wasm_i32x4_lt(b1, minval));
    return v_int16x8(wasm_v8x16_shuffle(a3, b3, 0,1,4,5,8,9,12,13,16,17,20,21,24,25,28,29));
}
template<int n>
inline v_uint32x4 v_rshr_pack(const v_uint64x2& a, const v_uint64x2& b)
{
#ifdef __wasm_unimplemented_simd128__
    v128_t delta = wasm_i64x2_splat(((int64)1 << (n-1)));
    v128_t a1 = wasm_u64x2_shr(wasm_i64x2_add(a.val, delta), n);
    v128_t b1 = wasm_u64x2_shr(wasm_i64x2_add(b.val, delta), n);
    return v_uint32x4(wasm_v8x16_shuffle(a1, b1, 0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27));
#else
    fallback::v_uint64x2 a_(a), b_(b);
    return fallback::v_rshr_pack<n>(a_, b_);
#endif
}
template<int n>
inline v_int32x4 v_rshr_pack(const v_int64x2& a, const v_int64x2& b)
{
#ifdef __wasm_unimplemented_simd128__
    v128_t delta = wasm_i64x2_splat(((int64)1 << (n-1)));
    v128_t a1 = wasm_i64x2_shr(wasm_i64x2_add(a.val, delta), n);
    v128_t b1 = wasm_i64x2_shr(wasm_i64x2_add(b.val, delta), n);
    return v_int32x4(wasm_v8x16_shuffle(a1, b1, 0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27));
#else
    fallback::v_int64x2 a_(a), b_(b);
    return fallback::v_rshr_pack<n>(a_, b_);
#endif
}
template<int n>
inline v_uint8x16 v_rshr_pack_u(const v_int16x8& a, const v_int16x8& b)
{
    v128_t delta = wasm_i16x8_splat(((short)1 << (n-1)));
    v128_t a1 = wasm_i16x8_shr(wasm_i16x8_add(a.val, delta), n);
    v128_t b1 = wasm_i16x8_shr(wasm_i16x8_add(b.val, delta), n);
    v128_t maxval = wasm_i16x8_splat(255);
    v128_t minval = wasm_i16x8_splat(0);
    v128_t a2 = wasm_v128_bitselect(maxval, a1, wasm_i16x8_gt(a1, maxval));
    v128_t b2 = wasm_v128_bitselect(maxval, b1, wasm_i16x8_gt(b1, maxval));
    v128_t a3 = wasm_v128_bitselect(minval, a2, wasm_i16x8_lt(a1, minval));
    v128_t b3 = wasm_v128_bitselect(minval, b2, wasm_i16x8_lt(b1, minval));
    return v_uint8x16(wasm_v8x16_shuffle(a3, b3, 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30));
}
template<int n>
inline v_uint16x8 v_rshr_pack_u(const v_int32x4& a, const v_int32x4& b)
{
    v128_t delta = wasm_i32x4_splat(((int)1 << (n-1)));
    v128_t a1 = wasm_i32x4_shr(wasm_i32x4_add(a.val, delta), n);
    v128_t b1 = wasm_i32x4_shr(wasm_i32x4_add(b.val, delta), n);
    v128_t maxval = wasm_i32x4_splat(65535);
    v128_t minval = wasm_i16x8_splat(0);
    v128_t a2 = wasm_v128_bitselect(maxval, a1, wasm_i32x4_gt(a1, maxval));
    v128_t b2 = wasm_v128_bitselect(maxval, b1, wasm_i32x4_gt(b1, maxval));
    v128_t a3 = wasm_v128_bitselect(minval, a2, wasm_i32x4_lt(a1, minval));
    v128_t b3 = wasm_v128_bitselect(minval, b2, wasm_i32x4_lt(b1, minval));
    return v_uint16x8(wasm_v8x16_shuffle(a3, b3, 0,1,4,5,8,9,12,13,16,17,20,21,24,25,28,29));
}

inline void v_pack_store(uchar* ptr, const v_uint16x8& a)
{
    v128_t maxval = wasm_i16x8_splat(255);
    v128_t a1 = wasm_v128_bitselect(maxval, a.val, wasm_u16x8_gt(a.val, maxval));
    v128_t r = wasm_v8x16_shuffle(a1, a1, 0,2,4,6,8,10,12,14,0,2,4,6,8,10,12,14);
    uchar t_ptr[16];
    wasm_v128_store(t_ptr, r);
    for (int i=0; i<8; ++i) {
        ptr[i] = t_ptr[i];
    }
}
inline void v_pack_store(schar* ptr, const v_int16x8& a)
{
    v128_t maxval = wasm_i16x8_splat(127);
    v128_t minval = wasm_i16x8_splat(-128);
    v128_t a1 = wasm_v128_bitselect(maxval, a.val, wasm_i16x8_gt(a.val, maxval));
    v128_t a2 = wasm_v128_bitselect(minval, a1, wasm_i16x8_lt(a1, minval));
    v128_t r = wasm_v8x16_shuffle(a2, a2, 0,2,4,6,8,10,12,14,0,2,4,6,8,10,12,14);
    schar t_ptr[16];
    wasm_v128_store(t_ptr, r);
    for (int i=0; i<8; ++i) {
        ptr[i] = t_ptr[i];
    }
}
inline void v_pack_store(ushort* ptr, const v_uint32x4& a)
{
    v128_t maxval = wasm_i32x4_splat(65535);
    v128_t a1 = wasm_v128_bitselect(maxval, a.val, wasm_u32x4_gt(a.val, maxval));
    v128_t r = wasm_v8x16_shuffle(a1, a1, 0,1,4,5,8,9,12,13,0,1,4,5,8,9,12,13);
    ushort t_ptr[8];
    wasm_v128_store(t_ptr, r);
    for (int i=0; i<4; ++i) {
        ptr[i] = t_ptr[i];
    }
}
inline void v_pack_store(short* ptr, const v_int32x4& a)
{
    v128_t maxval = wasm_i32x4_splat(32767);
    v128_t minval = wasm_i32x4_splat(-32768);
    v128_t a1 = wasm_v128_bitselect(maxval, a.val, wasm_i32x4_gt(a.val, maxval));
    v128_t a2 = wasm_v128_bitselect(minval, a1, wasm_i32x4_lt(a1, minval));
    v128_t r = wasm_v8x16_shuffle(a2, a2, 0,1,4,5,8,9,12,13,0,1,4,5,8,9,12,13);
    short t_ptr[8];
    wasm_v128_store(t_ptr, r);
    for (int i=0; i<4; ++i) {
        ptr[i] = t_ptr[i];
    }
}
inline void v_pack_store(unsigned* ptr, const v_uint64x2& a)
{
    v128_t r = wasm_v8x16_shuffle(a.val, a.val, 0,1,2,3,8,9,10,11,0,1,2,3,8,9,10,11);
    unsigned t_ptr[4];
    wasm_v128_store(t_ptr, r);
    for (int i=0; i<2; ++i) {
        ptr[i] = t_ptr[i];
    }
}
inline void v_pack_store(int* ptr, const v_int64x2& a)
{
    v128_t r = wasm_v8x16_shuffle(a.val, a.val, 0,1,2,3,8,9,10,11,0,1,2,3,8,9,10,11);
    int t_ptr[4];
    wasm_v128_store(t_ptr, r);
    for (int i=0; i<2; ++i) {
        ptr[i] = t_ptr[i];
    }
}
inline void v_pack_u_store(uchar* ptr, const v_int16x8& a)
{
    v128_t maxval = wasm_i16x8_splat(255);
    v128_t minval = wasm_i16x8_splat(0);
    v128_t a1 = wasm_v128_bitselect(maxval, a.val, wasm_i16x8_gt(a.val, maxval));
    v128_t a2 = wasm_v128_bitselect(minval, a1, wasm_i16x8_lt(a1, minval));
    v128_t r = wasm_v8x16_shuffle(a2, a2, 0,2,4,6,8,10,12,14,0,2,4,6,8,10,12,14);
    uchar t_ptr[16];
    wasm_v128_store(t_ptr, r);
    for (int i=0; i<8; ++i) {
        ptr[i] = t_ptr[i];
    }
}
inline void v_pack_u_store(ushort* ptr, const v_int32x4& a)
{
    v128_t maxval = wasm_i32x4_splat(65535);
    v128_t minval = wasm_i32x4_splat(0);
    v128_t a1 = wasm_v128_bitselect(maxval, a.val, wasm_i32x4_gt(a.val, maxval));
    v128_t a2 = wasm_v128_bitselect(minval, a1, wasm_i32x4_lt(a1, minval));
    v128_t r = wasm_v8x16_shuffle(a2, a2, 0,1,4,5,8,9,12,13,0,1,4,5,8,9,12,13);
    ushort t_ptr[8];
    wasm_v128_store(t_ptr, r);
    for (int i=0; i<4; ++i) {
        ptr[i] = t_ptr[i];
    }
}

template<int n>
inline void v_rshr_pack_store(uchar* ptr, const v_uint16x8& a)
{
    v128_t delta = wasm_i16x8_splat((short)(1 << (n-1)));
    v128_t a1 = wasm_u16x8_shr(wasm_i16x8_add(a.val, delta), n);
    v128_t maxval = wasm_i16x8_splat(255);
    v128_t a2 = wasm_v128_bitselect(maxval, a1, wasm_u16x8_gt(a1, maxval));
    v128_t r = wasm_v8x16_shuffle(a2, a2, 0,2,4,6,8,10,12,14,0,2,4,6,8,10,12,14);
    uchar t_ptr[16];
    wasm_v128_store(t_ptr, r);
    for (int i=0; i<8; ++i) {
        ptr[i] = t_ptr[i];
    }
}
template<int n>
inline void v_rshr_pack_store(schar* ptr, const v_int16x8& a)
{
    v128_t delta = wasm_i16x8_splat(((short)1 << (n-1)));
    v128_t a1 = wasm_i16x8_shr(wasm_i16x8_add(a.val, delta), n);
    v128_t maxval = wasm_i16x8_splat(127);
    v128_t minval = wasm_i16x8_splat(-128);
    v128_t a2 = wasm_v128_bitselect(maxval, a1, wasm_i16x8_gt(a1, maxval));
    v128_t a3 = wasm_v128_bitselect(minval, a2, wasm_i16x8_lt(a1, minval));
    v128_t r = wasm_v8x16_shuffle(a3, a3, 0,2,4,6,8,10,12,14,0,2,4,6,8,10,12,14);
    schar t_ptr[16];
    wasm_v128_store(t_ptr, r);
    for (int i=0; i<8; ++i) {
        ptr[i] = t_ptr[i];
    }
}
template<int n>
inline void v_rshr_pack_store(ushort* ptr, const v_uint32x4& a)
{
    v128_t delta = wasm_i32x4_splat(((int)1 << (n-1)));
    v128_t a1 = wasm_u32x4_shr(wasm_i32x4_add(a.val, delta), n);
    v128_t maxval = wasm_i32x4_splat(65535);
    v128_t a2 = wasm_v128_bitselect(maxval, a1, wasm_u32x4_gt(a1, maxval));
    v128_t r = wasm_v8x16_shuffle(a2, a2, 0,1,4,5,8,9,12,13,0,1,4,5,8,9,12,13);
    ushort t_ptr[8];
    wasm_v128_store(t_ptr, r);
    for (int i=0; i<4; ++i) {
        ptr[i] = t_ptr[i];
    }
}
template<int n>
inline void v_rshr_pack_store(short* ptr, const v_int32x4& a)
{
    v128_t delta = wasm_i32x4_splat(((int)1 << (n-1)));
    v128_t a1 = wasm_i32x4_shr(wasm_i32x4_add(a.val, delta), n);
    v128_t maxval = wasm_i32x4_splat(32767);
    v128_t minval = wasm_i32x4_splat(-32768);
    v128_t a2 = wasm_v128_bitselect(maxval, a1, wasm_i32x4_gt(a1, maxval));
    v128_t a3 = wasm_v128_bitselect(minval, a2, wasm_i32x4_lt(a1, minval));
    v128_t r = wasm_v8x16_shuffle(a3, a3, 0,1,4,5,8,9,12,13,0,1,4,5,8,9,12,13);
    short t_ptr[8];
    wasm_v128_store(t_ptr, r);
    for (int i=0; i<4; ++i) {
        ptr[i] = t_ptr[i];
    }
}
template<int n>
inline void v_rshr_pack_store(unsigned* ptr, const v_uint64x2& a)
{
#ifdef __wasm_unimplemented_simd128__
    v128_t delta = wasm_i64x2_splat(((int64)1 << (n-1)));
    v128_t a1 = wasm_u64x2_shr(wasm_i64x2_add(a.val, delta), n);
    v128_t r = wasm_v8x16_shuffle(a1, a1, 0,1,2,3,8,9,10,11,0,1,2,3,8,9,10,11);
    unsigned t_ptr[4];
    wasm_v128_store(t_ptr, r);
    for (int i=0; i<2; ++i) {
        ptr[i] = t_ptr[i];
    }
#else
    fallback::v_uint64x2 _a(a);
    fallback::v_rshr_pack_store<n>(ptr, _a);
#endif
}
template<int n>
inline void v_rshr_pack_store(int* ptr, const v_int64x2& a)
{
#ifdef __wasm_unimplemented_simd128__
    v128_t delta = wasm_i64x2_splat(((int64)1 << (n-1)));
    v128_t a1 = wasm_i64x2_shr(wasm_i64x2_add(a.val, delta), n);
    v128_t r = wasm_v8x16_shuffle(a1, a1, 0,1,2,3,8,9,10,11,0,1,2,3,8,9,10,11);
    int t_ptr[4];
    wasm_v128_store(t_ptr, r);
    for (int i=0; i<2; ++i) {
        ptr[i] = t_ptr[i];
    }
#else
    fallback::v_int64x2 _a(a);
    fallback::v_rshr_pack_store<n>(ptr, _a);
#endif
}
template<int n>
inline void v_rshr_pack_u_store(uchar* ptr, const v_int16x8& a)
{
    v128_t delta = wasm_i16x8_splat(((short)1 << (n-1)));
    v128_t a1 = wasm_i16x8_shr(wasm_i16x8_add(a.val, delta), n);
    v128_t maxval = wasm_i16x8_splat(255);
    v128_t minval = wasm_i16x8_splat(0);
    v128_t a2 = wasm_v128_bitselect(maxval, a1, wasm_i16x8_gt(a1, maxval));
    v128_t a3 = wasm_v128_bitselect(minval, a2, wasm_i16x8_lt(a1, minval));
    v128_t r = wasm_v8x16_shuffle(a3, a3, 0,2,4,6,8,10,12,14,0,2,4,6,8,10,12,14);
    uchar t_ptr[16];
    wasm_v128_store(t_ptr, r);
    for (int i=0; i<8; ++i) {
        ptr[i] = t_ptr[i];
    }
}
template<int n>
inline void v_rshr_pack_u_store(ushort* ptr, const v_int32x4& a)
{
    v128_t delta = wasm_i32x4_splat(((int)1 << (n-1)));
    v128_t a1 = wasm_i32x4_shr(wasm_i32x4_add(a.val, delta), n);
    v128_t maxval = wasm_i32x4_splat(65535);
    v128_t minval = wasm_i32x4_splat(0);
    v128_t a2 = wasm_v128_bitselect(maxval, a1, wasm_i32x4_gt(a1, maxval));
    v128_t a3 = wasm_v128_bitselect(minval, a2, wasm_i32x4_lt(a1, minval));
    v128_t r = wasm_v8x16_shuffle(a3, a3, 0,1,4,5,8,9,12,13,0,1,4,5,8,9,12,13);
    ushort t_ptr[8];
    wasm_v128_store(t_ptr, r);
    for (int i=0; i<4; ++i) {
        ptr[i] = t_ptr[i];
    }
}

inline v_uint8x16 v_pack_b(const v_uint16x8& a, const v_uint16x8& b)
{
    v128_t maxval = wasm_i16x8_splat(255);
    v128_t a1 = wasm_v128_bitselect(maxval, a.val, wasm_u16x8_gt(a.val, maxval));
    v128_t b1 = wasm_v128_bitselect(maxval, b.val, wasm_u16x8_gt(b.val, maxval));
    return v_uint8x16(wasm_v8x16_shuffle(a1, b1, 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30));
}

inline v_uint8x16 v_pack_b(const v_uint32x4& a, const v_uint32x4& b,
                           const v_uint32x4& c, const v_uint32x4& d)
{
    v128_t maxval = wasm_i32x4_splat(255);
    v128_t a1 = wasm_v128_bitselect(maxval, a.val, wasm_u32x4_gt(a.val, maxval));
    v128_t b1 = wasm_v128_bitselect(maxval, b.val, wasm_u32x4_gt(b.val, maxval));
    v128_t c1 = wasm_v128_bitselect(maxval, c.val, wasm_u32x4_gt(c.val, maxval));
    v128_t d1 = wasm_v128_bitselect(maxval, d.val, wasm_u32x4_gt(d.val, maxval));
    v128_t ab = wasm_v8x16_shuffle(a1, b1, 0,4,8,12,16,20,24,28,0,4,8,12,16,20,24,28);
    v128_t cd = wasm_v8x16_shuffle(c1, d1, 0,4,8,12,16,20,24,28,0,4,8,12,16,20,24,28);
    return v_uint8x16(wasm_v8x16_shuffle(ab, cd, 0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23));
}

inline v_uint8x16 v_pack_b(const v_uint64x2& a, const v_uint64x2& b, const v_uint64x2& c,
                           const v_uint64x2& d, const v_uint64x2& e, const v_uint64x2& f,
                           const v_uint64x2& g, const v_uint64x2& h)
{
#ifdef __wasm_unimplemented_simd128__
    v128_t maxval = wasm_i32x4_splat(255);
    v128_t a1 = wasm_v128_bitselect(maxval, a.val, ((__u64x2)(a.val) > (__u64x2)maxval));
    v128_t b1 = wasm_v128_bitselect(maxval, b.val, ((__u64x2)(b.val) > (__u64x2)maxval));
    v128_t c1 = wasm_v128_bitselect(maxval, c.val, ((__u64x2)(c.val) > (__u64x2)maxval));
    v128_t d1 = wasm_v128_bitselect(maxval, d.val, ((__u64x2)(d.val) > (__u64x2)maxval));
    v128_t e1 = wasm_v128_bitselect(maxval, e.val, ((__u64x2)(e.val) > (__u64x2)maxval));
    v128_t f1 = wasm_v128_bitselect(maxval, f.val, ((__u64x2)(f.val) > (__u64x2)maxval));
    v128_t g1 = wasm_v128_bitselect(maxval, g.val, ((__u64x2)(g.val) > (__u64x2)maxval));
    v128_t h1 = wasm_v128_bitselect(maxval, h.val, ((__u64x2)(h.val) > (__u64x2)maxval));
    v128_t ab = wasm_v8x16_shuffle(a1, b1, 0,8,16,24,0,8,16,24,0,8,16,24,0,8,16,24);
    v128_t cd = wasm_v8x16_shuffle(c1, d1, 0,8,16,24,0,8,16,24,0,8,16,24,0,8,16,24);
    v128_t ef = wasm_v8x16_shuffle(e1, f1, 0,8,16,24,0,8,16,24,0,8,16,24,0,8,16,24);
    v128_t gh = wasm_v8x16_shuffle(g1, h1, 0,8,16,24,0,8,16,24,0,8,16,24,0,8,16,24);
    v128_t abcd = wasm_v8x16_shuffle(ab, cd, 0,1,2,3,16,17,18,19,0,1,2,3,16,17,18,19);
    v128_t efgh = wasm_v8x16_shuffle(ef, gh, 0,1,2,3,16,17,18,19,0,1,2,3,16,17,18,19);
    return v_uint8x16(wasm_v8x16_shuffle(abcd, efgh, 0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23));
#else
    fallback::v_uint64x2 a_(a), b_(b), c_(c), d_(d), e_(e), f_(f), g_(g), h_(h);
    return fallback::v_pack_b(a_, b_, c_, d_, e_, f_, g_, h_);
#endif
}

inline v_float32x4 v_matmul(const v_float32x4& v, const v_float32x4& m0,
                            const v_float32x4& m1, const v_float32x4& m2,
                            const v_float32x4& m3)
{
    v128_t v0 = wasm_f32x4_splat(wasm_f32x4_extract_lane(v.val, 0));
    v128_t v1 = wasm_f32x4_splat(wasm_f32x4_extract_lane(v.val, 1));
    v128_t v2 = wasm_f32x4_splat(wasm_f32x4_extract_lane(v.val, 2));
    v128_t v3 = wasm_f32x4_splat(wasm_f32x4_extract_lane(v.val, 3));
    v0 = wasm_f32x4_mul(v0, m0.val);
    v1 = wasm_f32x4_mul(v1, m1.val);
    v2 = wasm_f32x4_mul(v2, m2.val);
    v3 = wasm_f32x4_mul(v3, m3.val);

    return v_float32x4(wasm_f32x4_add(wasm_f32x4_add(v0, v1), wasm_f32x4_add(v2, v3)));
}

inline v_float32x4 v_matmuladd(const v_float32x4& v, const v_float32x4& m0,
                               const v_float32x4& m1, const v_float32x4& m2,
                               const v_float32x4& a)
{
    v128_t v0 = wasm_f32x4_splat(wasm_f32x4_extract_lane(v.val, 0));
    v128_t v1 = wasm_f32x4_splat(wasm_f32x4_extract_lane(v.val, 1));
    v128_t v2 = wasm_f32x4_splat(wasm_f32x4_extract_lane(v.val, 2));
    v0 = wasm_f32x4_mul(v0, m0.val);
    v1 = wasm_f32x4_mul(v1, m1.val);
    v2 = wasm_f32x4_mul(v2, m2.val);

    return v_float32x4(wasm_f32x4_add(wasm_f32x4_add(v0, v1), wasm_f32x4_add(v2, a.val)));
}

#define OPENCV_HAL_IMPL_WASM_BIN_OP(bin_op, _Tpvec, intrin) \
inline _Tpvec operator bin_op (const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(intrin(a.val, b.val)); \
} \
inline _Tpvec& operator bin_op##= (_Tpvec& a, const _Tpvec& b) \
{ \
    a.val = intrin(a.val, b.val); \
    return a; \
}

OPENCV_HAL_IMPL_WASM_BIN_OP(+, v_uint8x16, wasm_u8x16_add_saturate)
OPENCV_HAL_IMPL_WASM_BIN_OP(-, v_uint8x16, wasm_u8x16_sub_saturate)
OPENCV_HAL_IMPL_WASM_BIN_OP(+, v_int8x16, wasm_i8x16_add_saturate)
OPENCV_HAL_IMPL_WASM_BIN_OP(-, v_int8x16, wasm_i8x16_sub_saturate)
OPENCV_HAL_IMPL_WASM_BIN_OP(+, v_uint16x8, wasm_u16x8_add_saturate)
OPENCV_HAL_IMPL_WASM_BIN_OP(-, v_uint16x8, wasm_u16x8_sub_saturate)
OPENCV_HAL_IMPL_WASM_BIN_OP(+, v_int16x8, wasm_i16x8_add_saturate)
OPENCV_HAL_IMPL_WASM_BIN_OP(-, v_int16x8, wasm_i16x8_sub_saturate)
OPENCV_HAL_IMPL_WASM_BIN_OP(+, v_uint32x4, wasm_i32x4_add)
OPENCV_HAL_IMPL_WASM_BIN_OP(-, v_uint32x4, wasm_i32x4_sub)
OPENCV_HAL_IMPL_WASM_BIN_OP(*, v_uint32x4, wasm_i32x4_mul)
OPENCV_HAL_IMPL_WASM_BIN_OP(+, v_int32x4, wasm_i32x4_add)
OPENCV_HAL_IMPL_WASM_BIN_OP(-, v_int32x4, wasm_i32x4_sub)
OPENCV_HAL_IMPL_WASM_BIN_OP(*, v_int32x4, wasm_i32x4_mul)
OPENCV_HAL_IMPL_WASM_BIN_OP(+, v_float32x4, wasm_f32x4_add)
OPENCV_HAL_IMPL_WASM_BIN_OP(-, v_float32x4, wasm_f32x4_sub)
OPENCV_HAL_IMPL_WASM_BIN_OP(*, v_float32x4, wasm_f32x4_mul)
OPENCV_HAL_IMPL_WASM_BIN_OP(/, v_float32x4, wasm_f32x4_div)

#ifdef __wasm_unimplemented_simd128__
OPENCV_HAL_IMPL_WASM_BIN_OP(+, v_uint64x2, wasm_i64x2_add)
OPENCV_HAL_IMPL_WASM_BIN_OP(-, v_uint64x2, wasm_i64x2_sub)
OPENCV_HAL_IMPL_WASM_BIN_OP(+, v_int64x2, wasm_i64x2_add)
OPENCV_HAL_IMPL_WASM_BIN_OP(-, v_int64x2, wasm_i64x2_sub)
OPENCV_HAL_IMPL_WASM_BIN_OP(+, v_float64x2, wasm_f64x2_add)
OPENCV_HAL_IMPL_WASM_BIN_OP(-, v_float64x2, wasm_f64x2_sub)
OPENCV_HAL_IMPL_WASM_BIN_OP(*, v_float64x2, wasm_f64x2_mul)
OPENCV_HAL_IMPL_WASM_BIN_OP(/, v_float64x2, wasm_f64x2_div)
#else
#define OPENCV_HAL_IMPL_FALLBACK_BIN_OP(bin_op, _Tpvec) \
inline _Tpvec operator bin_op (const _Tpvec& a, const _Tpvec& b) \
{ \
    fallback::_Tpvec a_(a), b_(b); \
    return _Tpvec((a_) bin_op (b_)); \
} \
inline _Tpvec& operator bin_op##= (_Tpvec& a, const _Tpvec& b) \
{ \
    fallback::_Tpvec a_(a), b_(b); \
    a_ bin_op##= b_; \
    a = _Tpvec(a_); \
    return a; \
}

OPENCV_HAL_IMPL_FALLBACK_BIN_OP(+, v_uint64x2)
OPENCV_HAL_IMPL_FALLBACK_BIN_OP(-, v_uint64x2)
OPENCV_HAL_IMPL_FALLBACK_BIN_OP(+, v_int64x2)
OPENCV_HAL_IMPL_FALLBACK_BIN_OP(-, v_int64x2)
OPENCV_HAL_IMPL_FALLBACK_BIN_OP(+, v_float64x2)
OPENCV_HAL_IMPL_FALLBACK_BIN_OP(-, v_float64x2)
OPENCV_HAL_IMPL_FALLBACK_BIN_OP(*, v_float64x2)
OPENCV_HAL_IMPL_FALLBACK_BIN_OP(/, v_float64x2)
#endif

// saturating multiply 8-bit, 16-bit
#define OPENCV_HAL_IMPL_WASM_MUL_SAT(_Tpvec, _Tpwvec)        \
inline _Tpvec operator * (const _Tpvec& a, const _Tpvec& b)  \
{                                                            \
    _Tpwvec c, d;                                            \
    v_mul_expand(a, b, c, d);                                \
    return v_pack(c, d);                                     \
}                                                            \
inline _Tpvec& operator *= (_Tpvec& a, const _Tpvec& b)      \
{ a = a * b; return a; }

OPENCV_HAL_IMPL_WASM_MUL_SAT(v_uint8x16, v_uint16x8)
OPENCV_HAL_IMPL_WASM_MUL_SAT(v_int8x16,  v_int16x8)
OPENCV_HAL_IMPL_WASM_MUL_SAT(v_uint16x8, v_uint32x4)
OPENCV_HAL_IMPL_WASM_MUL_SAT(v_int16x8,  v_int32x4)

//  Multiply and expand
inline void v_mul_expand(const v_uint8x16& a, const v_uint8x16& b,
                         v_uint16x8& c, v_uint16x8& d)
{
    v_uint16x8 a0, a1, b0, b1;
    v_expand(a, a0, a1);
    v_expand(b, b0, b1);
    c = v_mul_wrap(a0, b0);
    d = v_mul_wrap(a1, b1);
}

inline void v_mul_expand(const v_int8x16& a, const v_int8x16& b,
                         v_int16x8& c, v_int16x8& d)
{
    v_int16x8 a0, a1, b0, b1;
    v_expand(a, a0, a1);
    v_expand(b, b0, b1);
    c = v_mul_wrap(a0, b0);
    d = v_mul_wrap(a1, b1);
}

inline void v_mul_expand(const v_int16x8& a, const v_int16x8& b,
                         v_int32x4& c, v_int32x4& d)
{
    v_int32x4 a0, a1, b0, b1;
    v_expand(a, a0, a1);
    v_expand(b, b0, b1);
    c.val = wasm_i32x4_mul(a0.val, b0.val);
    d.val = wasm_i32x4_mul(a1.val, b1.val);
}

inline void v_mul_expand(const v_uint16x8& a, const v_uint16x8& b,
                         v_uint32x4& c, v_uint32x4& d)
{
    v_uint32x4 a0, a1, b0, b1;
    v_expand(a, a0, a1);
    v_expand(b, b0, b1);
    c.val = wasm_i32x4_mul(a0.val, b0.val);
    d.val = wasm_i32x4_mul(a1.val, b1.val);
}

inline void v_mul_expand(const v_uint32x4& a, const v_uint32x4& b,
                         v_uint64x2& c, v_uint64x2& d)
{
#ifdef __wasm_unimplemented_simd128__
    v_uint64x2 a0, a1, b0, b1;
    v_expand(a, a0, a1);
    v_expand(b, b0, b1);
    c.val = ((__u64x2)(a0.val) * (__u64x2)(b0.val));
    d.val = ((__u64x2)(a1.val) * (__u64x2)(b1.val));
#else
    fallback::v_uint32x4 a_(a), b_(b);
    fallback::v_uint64x2 c_, d_;
    fallback::v_mul_expand(a_, b_, c_, d_);
    c = v_uint64x2(c_);
    d = v_uint64x2(d_);
#endif
}

inline v_int16x8 v_mul_hi(const v_int16x8& a, const v_int16x8& b)
{
    v_int32x4 a0, a1, b0, b1;
    v_expand(a, a0, a1);
    v_expand(b, b0, b1);
    v128_t c = wasm_i32x4_mul(a0.val, b0.val);
    v128_t d = wasm_i32x4_mul(a1.val, b1.val);
    return v_int16x8(wasm_v8x16_shuffle(c, d, 2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31));
}
inline v_uint16x8 v_mul_hi(const v_uint16x8& a, const v_uint16x8& b)
{
    v_uint32x4 a0, a1, b0, b1;
    v_expand(a, a0, a1);
    v_expand(b, b0, b1);
    v128_t c = wasm_i32x4_mul(a0.val, b0.val);
    v128_t d = wasm_i32x4_mul(a1.val, b1.val);
    return v_uint16x8(wasm_v8x16_shuffle(c, d, 2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31));
}

//////// Dot Product ////////

inline v_int32x4 v_dotprod(const v_int16x8& a, const v_int16x8& b)
{
    v128_t a0 = wasm_i32x4_shr(wasm_i32x4_shl(a.val, 16), 16);
    v128_t a1 = wasm_i32x4_shr(a.val, 16);
    v128_t b0 = wasm_i32x4_shr(wasm_i32x4_shl(b.val, 16), 16);
    v128_t b1 = wasm_i32x4_shr(b.val, 16);
    v128_t c = wasm_i32x4_mul(a0, b0);
    v128_t d = wasm_i32x4_mul(a1, b1);
    return v_int32x4(wasm_i32x4_add(c, d));
}

inline v_int32x4 v_dotprod(const v_int16x8& a, const v_int16x8& b, const v_int32x4& c)
{ return v_dotprod(a, b) + c; }

inline v_int64x2 v_dotprod(const v_int32x4& a, const v_int32x4& b)
{
#ifdef __wasm_unimplemented_simd128__
    v128_t a0 = wasm_i64x2_shr(wasm_i64x2_shl(a.val, 32), 32);
    v128_t a1 = wasm_i64x2_shr(a.val, 32);
    v128_t b0 = wasm_i64x2_shr(wasm_i64x2_shl(b.val, 32), 32);
    v128_t b1 = wasm_i64x2_shr(b.val, 32);
    v128_t c = (v128_t)((__i64x2)a0 * (__i64x2)b0);
    v128_t d = (v128_t)((__i64x2)a1 * (__i64x2)b1);
    return v_int64x2(wasm_i64x2_add(c, d));
#else
    fallback::v_int32x4 a_(a);
    fallback::v_int32x4 b_(b);
    return fallback::v_dotprod(a_, b_);
#endif
}
inline v_int64x2 v_dotprod(const v_int32x4& a, const v_int32x4& b, const v_int64x2& c)
{
#ifdef __wasm_unimplemented_simd128__
    return v_dotprod(a, b) + c;
#else
    fallback::v_int32x4 a_(a);
    fallback::v_int32x4 b_(b);
    fallback::v_int64x2 c_(c);
    return fallback::v_dotprod(a_, b_, c_);
#endif
}

// 8 >> 32
inline v_uint32x4 v_dotprod_expand(const v_uint8x16& a, const v_uint8x16& b)
{
    v128_t a0 = wasm_u16x8_shr(wasm_i16x8_shl(a.val, 8), 8);
    v128_t a1 = wasm_u16x8_shr(a.val, 8);
    v128_t b0 = wasm_u16x8_shr(wasm_i16x8_shl(b.val, 8), 8);
    v128_t b1 = wasm_u16x8_shr(b.val, 8);
    return v_uint32x4((
        v_dotprod(v_int16x8(a0), v_int16x8(b0)) +
        v_dotprod(v_int16x8(a1), v_int16x8(b1))).val
    );
}
inline v_uint32x4 v_dotprod_expand(const v_uint8x16& a, const v_uint8x16& b, const v_uint32x4& c)
{ return v_dotprod_expand(a, b) + c; }

inline v_int32x4 v_dotprod_expand(const v_int8x16& a, const v_int8x16& b)
{
    v128_t a0 = wasm_i16x8_shr(wasm_i16x8_shl(a.val, 8), 8);
    v128_t a1 = wasm_i16x8_shr(a.val, 8);
    v128_t b0 = wasm_i16x8_shr(wasm_i16x8_shl(b.val, 8), 8);
    v128_t b1 = wasm_i16x8_shr(b.val, 8);
    return v_int32x4(
        v_dotprod(v_int16x8(a0), v_int16x8(b0)) +
        v_dotprod(v_int16x8(a1), v_int16x8(b1))
    );
}
inline v_int32x4 v_dotprod_expand(const v_int8x16& a, const v_int8x16& b, const v_int32x4& c)
{ return v_dotprod_expand(a, b) + c; }

// 16 >> 64
inline v_uint64x2 v_dotprod_expand(const v_uint16x8& a, const v_uint16x8& b)
{
    fallback::v_uint16x8 a_(a);
    fallback::v_uint16x8 b_(b);
    return fallback::v_dotprod_expand(a_, b_);
}
inline v_uint64x2 v_dotprod_expand(const v_uint16x8& a, const v_uint16x8& b, const v_uint64x2& c)
{
    fallback::v_uint16x8 a_(a);
    fallback::v_uint16x8 b_(b);
    fallback::v_uint64x2 c_(c);
    return fallback::v_dotprod_expand(a_, b_, c_);
}

inline v_int64x2 v_dotprod_expand(const v_int16x8& a, const v_int16x8& b)
{
    fallback::v_int16x8 a_(a);
    fallback::v_int16x8 b_(b);
    return fallback::v_dotprod_expand(a_, b_);
}

inline v_int64x2 v_dotprod_expand(const v_int16x8& a, const v_int16x8& b, const v_int64x2& c)
{
    fallback::v_int16x8 a_(a);
    fallback::v_int16x8 b_(b);
    fallback::v_int64x2 c_(c);
    return fallback::v_dotprod_expand(a_, b_, c_);
}

// 32 >> 64f
inline v_float64x2 v_dotprod_expand(const v_int32x4& a, const v_int32x4& b)
{ return v_cvt_f64(v_dotprod(a, b)); }
inline v_float64x2 v_dotprod_expand(const v_int32x4& a, const v_int32x4& b, const v_float64x2& c)
{ return v_dotprod_expand(a, b) + c; }

//////// Fast Dot Product ////////

// 16 >> 32
inline v_int32x4 v_dotprod_fast(const v_int16x8& a, const v_int16x8& b)
{ return v_dotprod(a, b); }
inline v_int32x4 v_dotprod_fast(const v_int16x8& a, const v_int16x8& b, const v_int32x4& c)
{ return v_dotprod(a, b, c); }

// 32 >> 64
inline v_int64x2 v_dotprod_fast(const v_int32x4& a, const v_int32x4& b)
{ return v_dotprod(a, b); }
inline v_int64x2 v_dotprod_fast(const v_int32x4& a, const v_int32x4& b, const v_int64x2& c)
{ return v_dotprod(a, b, c); }

// 8 >> 32
inline v_uint32x4 v_dotprod_expand_fast(const v_uint8x16& a, const v_uint8x16& b)
{ return v_dotprod_expand(a, b); }
inline v_uint32x4 v_dotprod_expand_fast(const v_uint8x16& a, const v_uint8x16& b, const v_uint32x4& c)
{ return v_dotprod_expand(a, b, c); }
inline v_int32x4 v_dotprod_expand_fast(const v_int8x16& a, const v_int8x16& b)
{ return v_dotprod_expand(a, b); }
inline v_int32x4 v_dotprod_expand_fast(const v_int8x16& a, const v_int8x16& b, const v_int32x4& c)
{ return v_dotprod_expand(a, b, c); }

// 16 >> 64
inline v_uint64x2 v_dotprod_expand_fast(const v_uint16x8& a, const v_uint16x8& b)
{ return v_dotprod_expand(a, b); }
inline v_uint64x2 v_dotprod_expand_fast(const v_uint16x8& a, const v_uint16x8& b, const v_uint64x2& c)
{ return v_dotprod_expand(a, b, c); }
inline v_int64x2 v_dotprod_expand_fast(const v_int16x8& a, const v_int16x8& b)
{ return v_dotprod_expand(a, b); }
inline v_int64x2 v_dotprod_expand_fast(const v_int16x8& a, const v_int16x8& b, const v_int64x2& c)
{ return v_dotprod_expand(a, b, c); }

// 32 >> 64f
inline v_float64x2 v_dotprod_expand_fast(const v_int32x4& a, const v_int32x4& b)
{ return v_dotprod_expand(a, b); }
inline v_float64x2 v_dotprod_expand_fast(const v_int32x4& a, const v_int32x4& b, const v_float64x2& c)
{ return v_dotprod_expand(a, b, c); }

#define OPENCV_HAL_IMPL_WASM_LOGIC_OP(_Tpvec) \
OPENCV_HAL_IMPL_WASM_BIN_OP(&, _Tpvec, wasm_v128_and) \
OPENCV_HAL_IMPL_WASM_BIN_OP(|, _Tpvec, wasm_v128_or) \
OPENCV_HAL_IMPL_WASM_BIN_OP(^, _Tpvec, wasm_v128_xor) \
inline _Tpvec operator ~ (const _Tpvec& a) \
{ \
    return _Tpvec(wasm_v128_not(a.val)); \
}

OPENCV_HAL_IMPL_WASM_LOGIC_OP(v_uint8x16)
OPENCV_HAL_IMPL_WASM_LOGIC_OP(v_int8x16)
OPENCV_HAL_IMPL_WASM_LOGIC_OP(v_uint16x8)
OPENCV_HAL_IMPL_WASM_LOGIC_OP(v_int16x8)
OPENCV_HAL_IMPL_WASM_LOGIC_OP(v_uint32x4)
OPENCV_HAL_IMPL_WASM_LOGIC_OP(v_int32x4)
OPENCV_HAL_IMPL_WASM_LOGIC_OP(v_uint64x2)
OPENCV_HAL_IMPL_WASM_LOGIC_OP(v_int64x2)
OPENCV_HAL_IMPL_WASM_LOGIC_OP(v_float32x4)
OPENCV_HAL_IMPL_WASM_LOGIC_OP(v_float64x2)

inline v_float32x4 v_sqrt(const v_float32x4& x)
{
#ifdef __wasm_unimplemented_simd128__
    return v_float32x4(wasm_f32x4_sqrt(x.val));
#else
    fallback::v_float32x4 x_(x);
    return fallback::v_sqrt(x_);
#endif
}

inline v_float32x4 v_invsqrt(const v_float32x4& x)
{
#ifdef __wasm_unimplemented_simd128__
    const v128_t _1_0 = wasm_f32x4_splat(1.0);
    return v_float32x4(wasm_f32x4_div(_1_0, wasm_f32x4_sqrt(x.val)));
#else
    fallback::v_float32x4 x_(x);
    return fallback::v_invsqrt(x_);
#endif
}

inline v_float64x2 v_sqrt(const v_float64x2& x)
{
#ifdef __wasm_unimplemented_simd128__
    return v_float64x2(wasm_f64x2_sqrt(x.val));
#else
    fallback::v_float64x2 x_(x);
    return fallback::v_sqrt(x_);
#endif
}

inline v_float64x2 v_invsqrt(const v_float64x2& x)
{
#ifdef __wasm_unimplemented_simd128__
    const v128_t _1_0 = wasm_f64x2_splat(1.0);
    return v_float64x2(wasm_f64x2_div(_1_0, wasm_f64x2_sqrt(x.val)));
#else
    fallback::v_float64x2 x_(x);
    return fallback::v_invsqrt(x_);
#endif
}

#define OPENCV_HAL_IMPL_WASM_ABS_INT_FUNC(_Tpuvec, _Tpsvec, suffix, zsuffix, shiftWidth) \
inline _Tpuvec v_abs(const _Tpsvec& x) \
{ \
    v128_t s = wasm_##suffix##_shr(x.val, shiftWidth); \
    v128_t f = wasm_##zsuffix##_shr(x.val, shiftWidth); \
    return _Tpuvec(wasm_##zsuffix##_add(wasm_v128_xor(x.val, f), s)); \
}

OPENCV_HAL_IMPL_WASM_ABS_INT_FUNC(v_uint8x16, v_int8x16, u8x16, i8x16, 7)
OPENCV_HAL_IMPL_WASM_ABS_INT_FUNC(v_uint16x8, v_int16x8, u16x8, i16x8, 15)
OPENCV_HAL_IMPL_WASM_ABS_INT_FUNC(v_uint32x4, v_int32x4, u32x4, i32x4, 31)

inline v_float32x4 v_abs(const v_float32x4& x)
{ return v_float32x4(wasm_f32x4_abs(x.val)); }
inline v_float64x2 v_abs(const v_float64x2& x)
{
#ifdef __wasm_unimplemented_simd128__
    return v_float64x2(wasm_f64x2_abs(x.val));
#else
    fallback::v_float64x2 x_(x);
    return fallback::v_abs(x_);
#endif
}

// TODO: exp, log, sin, cos

#define OPENCV_HAL_IMPL_WASM_BIN_FUNC(_Tpvec, func, intrin) \
inline _Tpvec func(const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(intrin(a.val, b.val)); \
}

OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_float32x4, v_min, wasm_f32x4_min)
OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_float32x4, v_max, wasm_f32x4_max)

#ifdef __wasm_unimplemented_simd128__
OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_float64x2, v_min, wasm_f64x2_min)
OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_float64x2, v_max, wasm_f64x2_max)
#else
#define OPENCV_HAL_IMPL_WASM_MINMAX_64f_FUNC(func) \
inline v_float64x2 func(const v_float64x2& a, const v_float64x2& b) \
{ \
    fallback::v_float64x2 a_(a), b_(b); \
    return fallback::func(a_, b_); \
}

OPENCV_HAL_IMPL_WASM_MINMAX_64f_FUNC(v_min)
OPENCV_HAL_IMPL_WASM_MINMAX_64f_FUNC(v_max)
#endif

#define OPENCV_HAL_IMPL_WASM_MINMAX_S_INIT_FUNC(_Tpvec, suffix) \
inline _Tpvec v_min(const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(wasm_v128_bitselect(b.val, a.val, wasm_##suffix##_gt(a.val, b.val))); \
} \
inline _Tpvec v_max(const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(wasm_v128_bitselect(a.val, b.val, wasm_##suffix##_gt(a.val, b.val))); \
}

OPENCV_HAL_IMPL_WASM_MINMAX_S_INIT_FUNC(v_int8x16, i8x16)
OPENCV_HAL_IMPL_WASM_MINMAX_S_INIT_FUNC(v_int16x8, i16x8)
OPENCV_HAL_IMPL_WASM_MINMAX_S_INIT_FUNC(v_int32x4, i32x4)

#define OPENCV_HAL_IMPL_WASM_MINMAX_U_INIT_FUNC(_Tpvec, suffix, deltaNum) \
inline _Tpvec v_min(const _Tpvec& a, const _Tpvec& b) \
{ \
    v128_t delta = wasm_##suffix##_splat(deltaNum); \
    v128_t mask = wasm_##suffix##_gt(wasm_v128_xor(a.val, delta), wasm_v128_xor(b.val, delta)); \
    return _Tpvec(wasm_v128_bitselect(b.val, a.val, mask)); \
} \
inline _Tpvec v_max(const _Tpvec& a, const _Tpvec& b) \
{ \
    v128_t delta = wasm_##suffix##_splat(deltaNum); \
    v128_t mask = wasm_##suffix##_gt(wasm_v128_xor(a.val, delta), wasm_v128_xor(b.val, delta)); \
    return _Tpvec(wasm_v128_bitselect(a.val, b.val, mask)); \
}

OPENCV_HAL_IMPL_WASM_MINMAX_U_INIT_FUNC(v_uint8x16, i8x16, (schar)0x80)
OPENCV_HAL_IMPL_WASM_MINMAX_U_INIT_FUNC(v_uint16x8, i16x8, (short)0x8000)
OPENCV_HAL_IMPL_WASM_MINMAX_U_INIT_FUNC(v_uint32x4, i32x4, (int)0x80000000)

#define OPENCV_HAL_IMPL_WASM_INIT_CMP_OP(_Tpvec, suffix, esuffix) \
inline _Tpvec operator == (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(wasm_##esuffix##_eq(a.val, b.val)); } \
inline _Tpvec operator != (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(wasm_##esuffix##_ne(a.val, b.val)); } \
inline _Tpvec operator < (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(wasm_##suffix##_lt(a.val, b.val)); } \
inline _Tpvec operator > (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(wasm_##suffix##_gt(a.val, b.val)); } \
inline _Tpvec operator <= (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(wasm_##suffix##_le(a.val, b.val)); } \
inline _Tpvec operator >= (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(wasm_##suffix##_ge(a.val, b.val)); }

OPENCV_HAL_IMPL_WASM_INIT_CMP_OP(v_uint8x16, u8x16, i8x16)
OPENCV_HAL_IMPL_WASM_INIT_CMP_OP(v_int8x16, i8x16, i8x16)
OPENCV_HAL_IMPL_WASM_INIT_CMP_OP(v_uint16x8, u16x8, i16x8)
OPENCV_HAL_IMPL_WASM_INIT_CMP_OP(v_int16x8, i16x8, i16x8)
OPENCV_HAL_IMPL_WASM_INIT_CMP_OP(v_uint32x4, u32x4, i32x4)
OPENCV_HAL_IMPL_WASM_INIT_CMP_OP(v_int32x4, i32x4, i32x4)
OPENCV_HAL_IMPL_WASM_INIT_CMP_OP(v_float32x4, f32x4, f32x4)

#ifdef __wasm_unimplemented_simd128__
OPENCV_HAL_IMPL_WASM_INIT_CMP_OP(v_float64x2, f64x2, f64x2)
#else
#define OPENCV_HAL_IMPL_INIT_FALLBACK_CMP_OP(_Tpvec, bin_op) \
inline _Tpvec operator bin_op (const _Tpvec& a, const _Tpvec& b) \
{ \
    fallback::_Tpvec a_(a), b_(b); \
    return _Tpvec((a_) bin_op (b_));\
} \

OPENCV_HAL_IMPL_INIT_FALLBACK_CMP_OP(v_float64x2, ==)
OPENCV_HAL_IMPL_INIT_FALLBACK_CMP_OP(v_float64x2, !=)
OPENCV_HAL_IMPL_INIT_FALLBACK_CMP_OP(v_float64x2, <)
OPENCV_HAL_IMPL_INIT_FALLBACK_CMP_OP(v_float64x2, >)
OPENCV_HAL_IMPL_INIT_FALLBACK_CMP_OP(v_float64x2, <=)
OPENCV_HAL_IMPL_INIT_FALLBACK_CMP_OP(v_float64x2, >=)
#endif

#define OPENCV_HAL_IMPL_WASM_64BIT_CMP_OP(_Tpvec, cast) \
inline _Tpvec operator == (const _Tpvec& a, const _Tpvec& b) \
{ return cast(v_reinterpret_as_f64(a) == v_reinterpret_as_f64(b)); } \
inline _Tpvec operator != (const _Tpvec& a, const _Tpvec& b) \
{ return cast(v_reinterpret_as_f64(a) != v_reinterpret_as_f64(b)); }

OPENCV_HAL_IMPL_WASM_64BIT_CMP_OP(v_uint64x2, v_reinterpret_as_u64)
OPENCV_HAL_IMPL_WASM_64BIT_CMP_OP(v_int64x2, v_reinterpret_as_s64)

inline v_float32x4 v_not_nan(const v_float32x4& a)
{
    v128_t z = wasm_i32x4_splat(0x7fffffff);
    v128_t t = wasm_i32x4_splat(0x7f800000);
    return v_float32x4(wasm_u32x4_lt(wasm_v128_and(a.val, z), t));
}
inline v_float64x2 v_not_nan(const v_float64x2& a)
{
#ifdef __wasm_unimplemented_simd128__
    v128_t z = wasm_i64x2_splat(0x7fffffffffffffff);
    v128_t t = wasm_i64x2_splat(0x7ff0000000000000);
    return v_float64x2((__u64x2)(wasm_v128_and(a.val, z)) < (__u64x2)t);
#else
    fallback::v_float64x2 a_(a);
    return fallback::v_not_nan(a_);
#endif
}

OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_uint8x16, v_add_wrap, wasm_i8x16_add)
OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_int8x16, v_add_wrap, wasm_i8x16_add)
OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_uint16x8, v_add_wrap, wasm_i16x8_add)
OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_int16x8, v_add_wrap, wasm_i16x8_add)
OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_uint8x16, v_sub_wrap, wasm_i8x16_sub)
OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_int8x16, v_sub_wrap, wasm_i8x16_sub)
OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_uint16x8, v_sub_wrap, wasm_i16x8_sub)
OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_int16x8, v_sub_wrap, wasm_i16x8_sub)
OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_uint8x16, v_mul_wrap, wasm_i8x16_mul)
OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_int8x16, v_mul_wrap, wasm_i8x16_mul)
OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_uint16x8, v_mul_wrap, wasm_i16x8_mul)
OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_int16x8, v_mul_wrap, wasm_i16x8_mul)


/** Absolute difference **/

inline v_uint8x16 v_absdiff(const v_uint8x16& a, const v_uint8x16& b)
{ return v_add_wrap(a - b,  b - a); }
inline v_uint16x8 v_absdiff(const v_uint16x8& a, const v_uint16x8& b)
{ return v_add_wrap(a - b,  b - a); }
inline v_uint32x4 v_absdiff(const v_uint32x4& a, const v_uint32x4& b)
{ return v_max(a, b) - v_min(a, b); }

inline v_uint8x16 v_absdiff(const v_int8x16& a, const v_int8x16& b)
{
    v_int8x16 d = v_sub_wrap(a, b);
    v_int8x16 m = a < b;
    return v_reinterpret_as_u8(v_sub_wrap(d ^ m, m));
}
inline v_uint16x8 v_absdiff(const v_int16x8& a, const v_int16x8& b)
{
    return v_reinterpret_as_u16(v_sub_wrap(v_max(a, b), v_min(a, b)));
}
inline v_uint32x4 v_absdiff(const v_int32x4& a, const v_int32x4& b)
{
    v_int32x4 d = a - b;
    v_int32x4 m = a < b;
    return v_reinterpret_as_u32((d ^ m) - m);
}

/** Saturating absolute difference **/
inline v_int8x16 v_absdiffs(const v_int8x16& a, const v_int8x16& b)
{
    v_int8x16 d = a - b;
    v_int8x16 m = a < b;
    return (d ^ m) - m;
 }
inline v_int16x8 v_absdiffs(const v_int16x8& a, const v_int16x8& b)
{ return v_max(a, b) - v_min(a, b); }


inline v_int32x4 v_fma(const v_int32x4& a, const v_int32x4& b, const v_int32x4& c)
{
    return a * b + c;
}

inline v_int32x4 v_muladd(const v_int32x4& a, const v_int32x4& b, const v_int32x4& c)
{
    return v_fma(a, b, c);
}

inline v_float32x4 v_fma(const v_float32x4& a, const v_float32x4& b, const v_float32x4& c)
{
    return a * b + c;
}

inline v_float64x2 v_fma(const v_float64x2& a, const v_float64x2& b, const v_float64x2& c)
{
    return a * b + c;
}

inline v_float32x4 v_absdiff(const v_float32x4& a, const v_float32x4& b)
{
    v128_t absmask_vec = wasm_i32x4_splat(0x7fffffff);
    return v_float32x4(wasm_v128_and(wasm_f32x4_sub(a.val, b.val), absmask_vec));
}
inline v_float64x2 v_absdiff(const v_float64x2& a, const v_float64x2& b)
{
#ifdef __wasm_unimplemented_simd128__
    v128_t absmask_vec = wasm_u64x2_shr(wasm_i32x4_splat(-1), 1);
    return v_float64x2(wasm_v128_and(wasm_f64x2_sub(a.val, b.val), absmask_vec));
#else
    fallback::v_float64x2 a_(a), b_(b);
    return fallback::v_absdiff(a_, b_);
#endif
}

#define OPENCV_HAL_IMPL_WASM_MISC_FLT_OP(_Tpvec) \
inline _Tpvec v_magnitude(const _Tpvec& a, const _Tpvec& b) \
{ \
    fallback::_Tpvec a_(a), b_(b); \
    return fallback::v_magnitude(a_, b_); \
} \
inline _Tpvec v_sqr_magnitude(const _Tpvec& a, const _Tpvec& b) \
{ \
    return v_fma(a, a, b*b); \
} \
inline _Tpvec v_muladd(const _Tpvec& a, const _Tpvec& b, const _Tpvec& c) \
{ \
    return v_fma(a, b, c); \
}

OPENCV_HAL_IMPL_WASM_MISC_FLT_OP(v_float32x4)
OPENCV_HAL_IMPL_WASM_MISC_FLT_OP(v_float64x2)

#define OPENCV_HAL_IMPL_WASM_SHIFT_OP(_Tpuvec, _Tpsvec, suffix, ssuffix) \
inline _Tpuvec operator << (const _Tpuvec& a, int imm) \
{ \
    return _Tpuvec(wasm_##suffix##_shl(a.val, imm)); \
} \
inline _Tpsvec operator << (const _Tpsvec& a, int imm) \
{ \
    return _Tpsvec(wasm_##suffix##_shl(a.val, imm)); \
} \
inline _Tpuvec operator >> (const _Tpuvec& a, int imm) \
{ \
    return _Tpuvec(wasm_##ssuffix##_shr(a.val, imm)); \
} \
inline _Tpsvec operator >> (const _Tpsvec& a, int imm) \
{ \
    return _Tpsvec(wasm_##suffix##_shr(a.val, imm)); \
} \
template<int imm> \
inline _Tpuvec v_shl(const _Tpuvec& a) \
{ \
    return _Tpuvec(wasm_##suffix##_shl(a.val, imm)); \
} \
template<int imm> \
inline _Tpsvec v_shl(const _Tpsvec& a) \
{ \
    return _Tpsvec(wasm_##suffix##_shl(a.val, imm)); \
} \
template<int imm> \
inline _Tpuvec v_shr(const _Tpuvec& a) \
{ \
    return _Tpuvec(wasm_##ssuffix##_shr(a.val, imm)); \
} \
template<int imm> \
inline _Tpsvec v_shr(const _Tpsvec& a) \
{ \
    return _Tpsvec(wasm_##suffix##_shr(a.val, imm)); \
}

OPENCV_HAL_IMPL_WASM_SHIFT_OP(v_uint8x16, v_int8x16, i8x16, u8x16)
OPENCV_HAL_IMPL_WASM_SHIFT_OP(v_uint16x8, v_int16x8, i16x8, u16x8)
OPENCV_HAL_IMPL_WASM_SHIFT_OP(v_uint32x4, v_int32x4, i32x4, u32x4)

#ifdef __wasm_unimplemented_simd128__
OPENCV_HAL_IMPL_WASM_SHIFT_OP(v_uint64x2, v_int64x2, i64x2, u64x2)
#else
#define OPENCV_HAL_IMPL_FALLBACK_SHIFT_OP(_Tpvec) \
inline _Tpvec operator << (const _Tpvec& a, int imm) \
{ \
    fallback::_Tpvec a_(a); \
    return a_ << imm; \
} \
inline _Tpvec operator >> (const _Tpvec& a, int imm) \
{ \
    fallback::_Tpvec a_(a); \
    return a_ >> imm; \
} \
template<int imm> \
inline _Tpvec v_shl(const _Tpvec& a) \
{ \
    fallback::_Tpvec a_(a); \
    return fallback::v_shl<imm>(a_); \
} \
template<int imm> \
inline _Tpvec v_shr(const _Tpvec& a) \
{ \
    fallback::_Tpvec a_(a); \
    return fallback::v_shr<imm>(a_); \
} \

OPENCV_HAL_IMPL_FALLBACK_SHIFT_OP(v_uint64x2)
OPENCV_HAL_IMPL_FALLBACK_SHIFT_OP(v_int64x2)
#endif

namespace hal_wasm_internal
{
    template <int imm,
        bool is_invalid = ((imm < 0) || (imm > 16)),
        bool is_first = (imm == 0),
        bool is_second = (imm == 16),
        bool is_other = (((imm > 0) && (imm < 16)))>
    class v_wasm_palignr_u8_class;

    template <int imm>
    class v_wasm_palignr_u8_class<imm, true, false, false, false>;

    template <int imm>
    class v_wasm_palignr_u8_class<imm, false, true, false, false>
    {
    public:
        inline v128_t operator()(const v128_t& a, const v128_t&) const
        {
            return a;
        }
    };

    template <int imm>
    class v_wasm_palignr_u8_class<imm, false, false, true, false>
    {
    public:
        inline v128_t operator()(const v128_t&, const v128_t& b) const
        {
            return b;
        }
    };

    template <int imm>
    class v_wasm_palignr_u8_class<imm, false, false, false, true>
    {
    public:
        inline v128_t operator()(const v128_t& a, const v128_t& b) const
        {
            enum { imm2 = (sizeof(v128_t) - imm) };
            return wasm_v8x16_shuffle(a, b,
                                      imm, imm+1, imm+2, imm+3,
                                      imm+4, imm+5, imm+6, imm+7,
                                      imm+8, imm+9, imm+10, imm+11,
                                      imm+12, imm+13, imm+14, imm+15);
        }
    };

    template <int imm>
    inline v128_t v_wasm_palignr_u8(const v128_t& a, const v128_t& b)
    {
        CV_StaticAssert((imm >= 0) && (imm <= 16), "Invalid imm for v_wasm_palignr_u8.");
        return v_wasm_palignr_u8_class<imm>()(a, b);
    }
}

template<int imm, typename _Tpvec>
inline _Tpvec v_rotate_right(const _Tpvec &a)
{
    using namespace hal_wasm_internal;
    enum { imm2 = (imm * sizeof(typename _Tpvec::lane_type)) };
    v128_t z = wasm_i8x16_splat(0);
    return _Tpvec(v_wasm_palignr_u8<imm2>(a.val, z));
}

template<int imm, typename _Tpvec>
inline _Tpvec v_rotate_left(const _Tpvec &a)
{
    using namespace hal_wasm_internal;
    enum { imm2 = ((_Tpvec::nlanes - imm) * sizeof(typename _Tpvec::lane_type)) };
    v128_t z = wasm_i8x16_splat(0);
    return _Tpvec(v_wasm_palignr_u8<imm2>(z, a.val));
}

template<int imm, typename _Tpvec>
inline _Tpvec v_rotate_right(const _Tpvec &a, const _Tpvec &b)
{
    using namespace hal_wasm_internal;
    enum { imm2 = (imm * sizeof(typename _Tpvec::lane_type)) };
    return _Tpvec(v_wasm_palignr_u8<imm2>(a.val, b.val));
}

template<int imm, typename _Tpvec>
inline _Tpvec v_rotate_left(const _Tpvec &a, const _Tpvec &b)
{
    using namespace hal_wasm_internal;
    enum { imm2 = ((_Tpvec::nlanes - imm) * sizeof(typename _Tpvec::lane_type)) };
    return _Tpvec(v_wasm_palignr_u8<imm2>(b.val, a.val));
}

#define OPENCV_HAL_IMPL_WASM_LOADSTORE_INT_OP(_Tpvec, _Tp) \
inline _Tpvec v_load(const _Tp* ptr) \
{ return _Tpvec(wasm_v128_load(ptr)); } \
inline _Tpvec v_load_aligned(const _Tp* ptr) \
{ return _Tpvec(wasm_v128_load(ptr)); } \
inline _Tpvec v_load_low(const _Tp* ptr) \
{ \
    _Tp tmp[_Tpvec::nlanes] = {0}; \
    for (int i=0; i<_Tpvec::nlanes/2; ++i) { \
        tmp[i] = ptr[i]; \
    } \
    return _Tpvec(wasm_v128_load(tmp)); \
} \
inline _Tpvec v_load_halves(const _Tp* ptr0, const _Tp* ptr1) \
{ \
    _Tp tmp[_Tpvec::nlanes]; \
    for (int i=0; i<_Tpvec::nlanes/2; ++i) { \
        tmp[i] = ptr0[i]; \
        tmp[i+_Tpvec::nlanes/2] = ptr1[i]; \
    } \
    return _Tpvec(wasm_v128_load(tmp)); \
} \
inline void v_store(_Tp* ptr, const _Tpvec& a) \
{ wasm_v128_store(ptr, a.val); } \
inline void v_store_aligned(_Tp* ptr, const _Tpvec& a) \
{ wasm_v128_store(ptr, a.val); } \
inline void v_store_aligned_nocache(_Tp* ptr, const _Tpvec& a) \
{ wasm_v128_store(ptr, a.val); } \
inline void v_store(_Tp* ptr, const _Tpvec& a, hal::StoreMode /*mode*/) \
{ \
    wasm_v128_store(ptr, a.val); \
} \
inline void v_store_low(_Tp* ptr, const _Tpvec& a) \
{ \
    fallback::_Tpvec a_(a); \
    fallback::v_store_low(ptr, a_); \
} \
inline void v_store_high(_Tp* ptr, const _Tpvec& a) \
{ \
    fallback::_Tpvec a_(a); \
    fallback::v_store_high(ptr, a_); \
}

OPENCV_HAL_IMPL_WASM_LOADSTORE_INT_OP(v_uint8x16, uchar)
OPENCV_HAL_IMPL_WASM_LOADSTORE_INT_OP(v_int8x16, schar)
OPENCV_HAL_IMPL_WASM_LOADSTORE_INT_OP(v_uint16x8, ushort)
OPENCV_HAL_IMPL_WASM_LOADSTORE_INT_OP(v_int16x8, short)
OPENCV_HAL_IMPL_WASM_LOADSTORE_INT_OP(v_uint32x4, unsigned)
OPENCV_HAL_IMPL_WASM_LOADSTORE_INT_OP(v_int32x4, int)
OPENCV_HAL_IMPL_WASM_LOADSTORE_INT_OP(v_uint64x2, uint64)
OPENCV_HAL_IMPL_WASM_LOADSTORE_INT_OP(v_int64x2, int64)
OPENCV_HAL_IMPL_WASM_LOADSTORE_INT_OP(v_float32x4, float)
OPENCV_HAL_IMPL_WASM_LOADSTORE_INT_OP(v_float64x2, double)


/** Reverse **/
inline v_uint8x16 v_reverse(const v_uint8x16 &a)
{ return v_uint8x16(wasm_v8x16_shuffle(a.val, a.val, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)); }

inline v_int8x16 v_reverse(const v_int8x16 &a)
{ return v_reinterpret_as_s8(v_reverse(v_reinterpret_as_u8(a))); }

inline v_uint16x8 v_reverse(const v_uint16x8 &a)
{ return v_uint16x8(wasm_v8x16_shuffle(a.val, a.val, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1)); }

inline v_int16x8 v_reverse(const v_int16x8 &a)
{ return v_reinterpret_as_s16(v_reverse(v_reinterpret_as_u16(a))); }

inline v_uint32x4 v_reverse(const v_uint32x4 &a)
{ return v_uint32x4(wasm_v8x16_shuffle(a.val, a.val, 12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3)); }

inline v_int32x4 v_reverse(const v_int32x4 &a)
{ return v_reinterpret_as_s32(v_reverse(v_reinterpret_as_u32(a))); }

inline v_float32x4 v_reverse(const v_float32x4 &a)
{ return v_reinterpret_as_f32(v_reverse(v_reinterpret_as_u32(a))); }

inline v_uint64x2 v_reverse(const v_uint64x2 &a)
{ return v_uint64x2(wasm_v8x16_shuffle(a.val, a.val, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7)); }

inline v_int64x2 v_reverse(const v_int64x2 &a)
{ return v_reinterpret_as_s64(v_reverse(v_reinterpret_as_u64(a))); }

inline v_float64x2 v_reverse(const v_float64x2 &a)
{ return v_reinterpret_as_f64(v_reverse(v_reinterpret_as_u64(a))); }


#define OPENCV_HAL_IMPL_WASM_REDUCE_OP_4_SUM(_Tpvec, scalartype, regtype, suffix, esuffix) \
inline scalartype v_reduce_sum(const _Tpvec& a) \
{ \
    regtype val = a.val; \
    val = wasm_##suffix##_add(val, wasm_v8x16_shuffle(val, val, 8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7)); \
    val = wasm_##suffix##_add(val, wasm_v8x16_shuffle(val, val, 4,5,6,7,8,9,10,11,12,13,14,15,0,1,2,3)); \
    return (scalartype)wasm_##esuffix##_extract_lane(val, 0); \
}

OPENCV_HAL_IMPL_WASM_REDUCE_OP_4_SUM(v_uint32x4, unsigned, v128_t, i32x4, i32x4)
OPENCV_HAL_IMPL_WASM_REDUCE_OP_4_SUM(v_int32x4, int, v128_t, i32x4, i32x4)
OPENCV_HAL_IMPL_WASM_REDUCE_OP_4_SUM(v_float32x4, float, v128_t, f32x4, f32x4)

// To do: Optimize v_reduce_sum with wasm intrin.
//        Now use fallback implementation as there is no widening op in wasm intrin.

#define OPENCV_HAL_IMPL_FALLBACK_REDUCE_OP_SUM(_Tpvec, scalartype) \
inline scalartype v_reduce_sum(const _Tpvec& a) \
{ \
    fallback::_Tpvec a_(a); \
    return fallback::v_reduce_sum(a_); \
}

OPENCV_HAL_IMPL_FALLBACK_REDUCE_OP_SUM(v_uint8x16, unsigned)
OPENCV_HAL_IMPL_FALLBACK_REDUCE_OP_SUM(v_int8x16, int)
OPENCV_HAL_IMPL_FALLBACK_REDUCE_OP_SUM(v_uint16x8, unsigned)
OPENCV_HAL_IMPL_FALLBACK_REDUCE_OP_SUM(v_int16x8, int)
OPENCV_HAL_IMPL_FALLBACK_REDUCE_OP_SUM(v_uint64x2, uint64)
OPENCV_HAL_IMPL_FALLBACK_REDUCE_OP_SUM(v_int64x2, int64)
OPENCV_HAL_IMPL_FALLBACK_REDUCE_OP_SUM(v_float64x2, double)

inline v_float32x4 v_reduce_sum4(const v_float32x4& a, const v_float32x4& b,
                                 const v_float32x4& c, const v_float32x4& d)
{
    v128_t ac = wasm_f32x4_add(wasm_unpacklo_i32x4(a.val, c.val), wasm_unpackhi_i32x4(a.val, c.val));
    v128_t bd = wasm_f32x4_add(wasm_unpacklo_i32x4(b.val, d.val), wasm_unpackhi_i32x4(b.val, d.val));
    return v_float32x4(wasm_f32x4_add(wasm_unpacklo_i32x4(ac, bd), wasm_unpackhi_i32x4(ac, bd)));
}

#define OPENCV_HAL_IMPL_WASM_REDUCE_OP(_Tpvec, scalartype, func, scalar_func) \
inline scalartype v_reduce_##func(const _Tpvec& a) \
{ \
    scalartype buf[_Tpvec::nlanes]; \
    v_store(buf, a); \
    scalartype tmp = buf[0]; \
    for (int i=1; i<_Tpvec::nlanes; ++i) { \
        tmp = scalar_func(tmp, buf[i]); \
    } \
    return tmp; \
}

OPENCV_HAL_IMPL_WASM_REDUCE_OP(v_uint8x16, uchar, max, std::max)
OPENCV_HAL_IMPL_WASM_REDUCE_OP(v_uint8x16, uchar, min, std::min)
OPENCV_HAL_IMPL_WASM_REDUCE_OP(v_int8x16, schar, max, std::max)
OPENCV_HAL_IMPL_WASM_REDUCE_OP(v_int8x16, schar, min, std::min)
OPENCV_HAL_IMPL_WASM_REDUCE_OP(v_uint16x8, ushort, max, std::max)
OPENCV_HAL_IMPL_WASM_REDUCE_OP(v_uint16x8, ushort, min, std::min)
OPENCV_HAL_IMPL_WASM_REDUCE_OP(v_int16x8, short, max, std::max)
OPENCV_HAL_IMPL_WASM_REDUCE_OP(v_int16x8, short, min, std::min)
OPENCV_HAL_IMPL_WASM_REDUCE_OP(v_uint32x4, unsigned, max, std::max)
OPENCV_HAL_IMPL_WASM_REDUCE_OP(v_uint32x4, unsigned, min, std::min)
OPENCV_HAL_IMPL_WASM_REDUCE_OP(v_int32x4, int, max, std::max)
OPENCV_HAL_IMPL_WASM_REDUCE_OP(v_int32x4, int, min, std::min)
OPENCV_HAL_IMPL_WASM_REDUCE_OP(v_float32x4, float, max, std::max)
OPENCV_HAL_IMPL_WASM_REDUCE_OP(v_float32x4, float, min, std::min)

inline unsigned v_reduce_sad(const v_uint8x16& a, const v_uint8x16& b)
{
    v_uint16x8 l16, h16;
    v_uint32x4 l16_l32, l16_h32, h16_l32, h16_h32;
    v_expand(v_absdiff(a, b), l16, h16);
    v_expand(l16, l16_l32, l16_h32);
    v_expand(h16, h16_l32, h16_h32);
    return v_reduce_sum(l16_l32+l16_h32+h16_l32+h16_h32);
}
inline unsigned v_reduce_sad(const v_int8x16& a, const v_int8x16& b)
{
    v_uint16x8 l16, h16;
    v_uint32x4 l16_l32, l16_h32, h16_l32, h16_h32;
    v_expand(v_absdiff(a, b), l16, h16);
    v_expand(l16, l16_l32, l16_h32);
    v_expand(h16, h16_l32, h16_h32);
    return v_reduce_sum(l16_l32+l16_h32+h16_l32+h16_h32);
}
inline unsigned v_reduce_sad(const v_uint16x8& a, const v_uint16x8& b)
{
    v_uint32x4 l, h;
    v_expand(v_absdiff(a, b), l, h);
    return v_reduce_sum(l + h);
}
inline unsigned v_reduce_sad(const v_int16x8& a, const v_int16x8& b)
{
    v_uint32x4 l, h;
    v_expand(v_absdiff(a, b), l, h);
    return v_reduce_sum(l + h);
}
inline unsigned v_reduce_sad(const v_uint32x4& a, const v_uint32x4& b)
{
    return v_reduce_sum(v_absdiff(a, b));
}
inline unsigned v_reduce_sad(const v_int32x4& a, const v_int32x4& b)
{
    return v_reduce_sum(v_absdiff(a, b));
}
inline float v_reduce_sad(const v_float32x4& a, const v_float32x4& b)
{
    return v_reduce_sum(v_absdiff(a, b));
}

inline v_uint8x16 v_popcount(const v_uint8x16& a)
{
    v128_t m1 = wasm_i32x4_splat(0x55555555);
    v128_t m2 = wasm_i32x4_splat(0x33333333);
    v128_t m4 = wasm_i32x4_splat(0x0f0f0f0f);
    v128_t p = a.val;
    p = wasm_i32x4_add(wasm_v128_and(wasm_u32x4_shr(p, 1), m1), wasm_v128_and(p, m1));
    p = wasm_i32x4_add(wasm_v128_and(wasm_u32x4_shr(p, 2), m2), wasm_v128_and(p, m2));
    p = wasm_i32x4_add(wasm_v128_and(wasm_u32x4_shr(p, 4), m4), wasm_v128_and(p, m4));
    return v_uint8x16(p);
}
inline v_uint16x8 v_popcount(const v_uint16x8& a)
{
    v_uint8x16 p = v_popcount(v_reinterpret_as_u8(a));
    p += v_rotate_right<1>(p);
    return v_reinterpret_as_u16(p) & v_setall_u16(0x00ff);
}
inline v_uint32x4 v_popcount(const v_uint32x4& a)
{
    v_uint8x16 p = v_popcount(v_reinterpret_as_u8(a));
    p += v_rotate_right<1>(p);
    p += v_rotate_right<2>(p);
    return v_reinterpret_as_u32(p) & v_setall_u32(0x000000ff);
}
inline v_uint64x2 v_popcount(const v_uint64x2& a)
{
    fallback::v_uint64x2 a_(a);
    return fallback::v_popcount(a_);
}
inline v_uint8x16 v_popcount(const v_int8x16& a)
{ return v_popcount(v_reinterpret_as_u8(a)); }
inline v_uint16x8 v_popcount(const v_int16x8& a)
{ return v_popcount(v_reinterpret_as_u16(a)); }
inline v_uint32x4 v_popcount(const v_int32x4& a)
{ return v_popcount(v_reinterpret_as_u32(a)); }
inline v_uint64x2 v_popcount(const v_int64x2& a)
{ return v_popcount(v_reinterpret_as_u64(a)); }

#define OPENCV_HAL_IMPL_WASM_CHECK_SIGNS(_Tpvec, suffix, scalarType) \
inline int v_signmask(const _Tpvec& a) \
{ \
    fallback::_Tpvec a_(a); \
    return fallback::v_signmask(a_); \
} \
inline bool v_check_all(const _Tpvec& a) \
{ return wasm_i8x16_all_true(wasm_##suffix##_lt(a.val, wasm_##suffix##_splat(0))); } \
inline bool v_check_any(const _Tpvec& a) \
{ return wasm_i8x16_any_true(wasm_##suffix##_lt(a.val, wasm_##suffix##_splat(0)));; }

OPENCV_HAL_IMPL_WASM_CHECK_SIGNS(v_uint8x16, i8x16, schar)
OPENCV_HAL_IMPL_WASM_CHECK_SIGNS(v_int8x16, i8x16, schar)
OPENCV_HAL_IMPL_WASM_CHECK_SIGNS(v_uint16x8, i16x8, short)
OPENCV_HAL_IMPL_WASM_CHECK_SIGNS(v_int16x8, i16x8, short)
OPENCV_HAL_IMPL_WASM_CHECK_SIGNS(v_uint32x4, i32x4, int)
OPENCV_HAL_IMPL_WASM_CHECK_SIGNS(v_int32x4, i32x4, int)
OPENCV_HAL_IMPL_WASM_CHECK_SIGNS(v_float32x4, i32x4, float)

inline int v_signmask(const v_float64x2& a)
{
    fallback::v_float64x2 a_(a);
    return fallback::v_signmask(a_);
}
inline bool v_check_all(const v_float64x2& a)
{
#ifdef __wasm_unimplemented_simd128__
    return wasm_i8x16_all_true((__i64x2)(a.val) < (__i64x2)(wasm_i64x2_splat(0)));
#else
    fallback::v_float64x2 a_(a);
    return fallback::v_check_all(a_);
#endif
}
inline bool v_check_any(const v_float64x2& a)
{
#ifdef __wasm_unimplemented_simd128__
    return wasm_i8x16_any_true((__i64x2)(a.val) < (__i64x2)(wasm_i64x2_splat(0)));;
#else
    fallback::v_float64x2 a_(a);
    return fallback::v_check_any(a_);
#endif
}

inline int v_scan_forward(const v_int8x16& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))); }
inline int v_scan_forward(const v_uint8x16& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))); }
inline int v_scan_forward(const v_int16x8& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))) / 2; }
inline int v_scan_forward(const v_uint16x8& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))) / 2; }
inline int v_scan_forward(const v_int32x4& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))) / 4; }
inline int v_scan_forward(const v_uint32x4& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))) / 4; }
inline int v_scan_forward(const v_float32x4& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))) / 4; }
inline int v_scan_forward(const v_int64x2& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))) / 8; }
inline int v_scan_forward(const v_uint64x2& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))) / 8; }
inline int v_scan_forward(const v_float64x2& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))) / 8; }

#define OPENCV_HAL_IMPL_WASM_SELECT(_Tpvec) \
inline _Tpvec v_select(const _Tpvec& mask, const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(wasm_v128_bitselect(a.val, b.val, mask.val)); \
}

OPENCV_HAL_IMPL_WASM_SELECT(v_uint8x16)
OPENCV_HAL_IMPL_WASM_SELECT(v_int8x16)
OPENCV_HAL_IMPL_WASM_SELECT(v_uint16x8)
OPENCV_HAL_IMPL_WASM_SELECT(v_int16x8)
OPENCV_HAL_IMPL_WASM_SELECT(v_uint32x4)
OPENCV_HAL_IMPL_WASM_SELECT(v_int32x4)
// OPENCV_HAL_IMPL_WASM_SELECT(v_uint64x2)
// OPENCV_HAL_IMPL_WASM_SELECT(v_int64x2)
OPENCV_HAL_IMPL_WASM_SELECT(v_float32x4)
OPENCV_HAL_IMPL_WASM_SELECT(v_float64x2)

#define OPENCV_HAL_IMPL_WASM_EXPAND(_Tpvec, _Tpwvec, _Tp, intrin)    \
inline void v_expand(const _Tpvec& a, _Tpwvec& b0, _Tpwvec& b1)      \
{                                                                    \
    b0.val = intrin(a.val);                                          \
    b1.val = __CV_CAT(intrin, _high)(a.val);                         \
}                                                                    \
inline _Tpwvec v_expand_low(const _Tpvec& a)                         \
{ return _Tpwvec(intrin(a.val)); }                                   \
inline _Tpwvec v_expand_high(const _Tpvec& a)                        \
{ return _Tpwvec(__CV_CAT(intrin, _high)(a.val)); }                  \
inline _Tpwvec v_load_expand(const _Tp* ptr)                         \
{                                                                    \
    v128_t a = wasm_v128_load(ptr);                                  \
    return _Tpwvec(intrin(a));                                       \
}

OPENCV_HAL_IMPL_WASM_EXPAND(v_uint8x16, v_uint16x8, uchar, v128_cvtu8x16_i16x8)
OPENCV_HAL_IMPL_WASM_EXPAND(v_int8x16,  v_int16x8,  schar, v128_cvti8x16_i16x8)
OPENCV_HAL_IMPL_WASM_EXPAND(v_uint16x8, v_uint32x4, ushort, v128_cvtu16x8_i32x4)
OPENCV_HAL_IMPL_WASM_EXPAND(v_int16x8,  v_int32x4,  short, v128_cvti16x8_i32x4)
OPENCV_HAL_IMPL_WASM_EXPAND(v_uint32x4, v_uint64x2, unsigned, v128_cvtu32x4_i64x2)
OPENCV_HAL_IMPL_WASM_EXPAND(v_int32x4,  v_int64x2,  int, v128_cvti32x4_i64x2)

#define OPENCV_HAL_IMPL_WASM_EXPAND_Q(_Tpvec, _Tp, intrin)  \
inline _Tpvec v_load_expand_q(const _Tp* ptr)               \
{                                                           \
    v128_t a = wasm_v128_load(ptr);                         \
    return _Tpvec(intrin(a));                               \
}

OPENCV_HAL_IMPL_WASM_EXPAND_Q(v_uint32x4, uchar, v128_cvtu8x16_i32x4)
OPENCV_HAL_IMPL_WASM_EXPAND_Q(v_int32x4, schar, v128_cvti8x16_i32x4)

#define OPENCV_HAL_IMPL_WASM_UNPACKS(_Tpvec, suffix) \
inline void v_zip(const _Tpvec& a0, const _Tpvec& a1, _Tpvec& b0, _Tpvec& b1) \
{ \
    b0.val = wasm_unpacklo_##suffix(a0.val, a1.val); \
    b1.val = wasm_unpackhi_##suffix(a0.val, a1.val); \
} \
inline _Tpvec v_combine_low(const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(wasm_unpacklo_i64x2(a.val, b.val)); \
} \
inline _Tpvec v_combine_high(const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(wasm_unpackhi_i64x2(a.val, b.val)); \
} \
inline void v_recombine(const _Tpvec& a, const _Tpvec& b, _Tpvec& c, _Tpvec& d) \
{ \
    c.val = wasm_unpacklo_i64x2(a.val, b.val); \
    d.val = wasm_unpackhi_i64x2(a.val, b.val); \
}

OPENCV_HAL_IMPL_WASM_UNPACKS(v_uint8x16, i8x16)
OPENCV_HAL_IMPL_WASM_UNPACKS(v_int8x16, i8x16)
OPENCV_HAL_IMPL_WASM_UNPACKS(v_uint16x8, i16x8)
OPENCV_HAL_IMPL_WASM_UNPACKS(v_int16x8, i16x8)
OPENCV_HAL_IMPL_WASM_UNPACKS(v_uint32x4, i32x4)
OPENCV_HAL_IMPL_WASM_UNPACKS(v_int32x4, i32x4)
OPENCV_HAL_IMPL_WASM_UNPACKS(v_float32x4, i32x4)
OPENCV_HAL_IMPL_WASM_UNPACKS(v_float64x2, i64x2)

template<int s, typename _Tpvec>
inline _Tpvec v_extract(const _Tpvec& a, const _Tpvec& b)
{
    return v_rotate_right<s>(a, b);
}

inline v_int32x4 v_round(const v_float32x4& a)
{
    v128_t h = wasm_f32x4_splat(0.5);
    return v_int32x4(wasm_i32x4_trunc_saturate_f32x4(wasm_f32x4_add(a.val, h)));
}

inline v_int32x4 v_floor(const v_float32x4& a)
{
    v128_t a1 = wasm_i32x4_trunc_saturate_f32x4(a.val);
    v128_t mask = wasm_f32x4_lt(a.val, wasm_f32x4_convert_i32x4(a1));
    return v_int32x4(wasm_i32x4_add(a1, mask));
}

inline v_int32x4 v_ceil(const v_float32x4& a)
{
    v128_t a1 = wasm_i32x4_trunc_saturate_f32x4(a.val);
    v128_t mask = wasm_f32x4_gt(a.val, wasm_f32x4_convert_i32x4(a1));
    return v_int32x4(wasm_i32x4_sub(a1, mask));
}

inline v_int32x4 v_trunc(const v_float32x4& a)
{ return v_int32x4(wasm_i32x4_trunc_saturate_f32x4(a.val)); }

#define OPENCV_HAL_IMPL_WASM_MATH_FUNC(func, cfunc, _Tpvec, _Tpnvec, _Tp, _Tpn) \
inline _Tpnvec func(const _Tpvec& a) \
{ \
    fallback::_Tpvec a_(a); \
    return fallback::func(a_); \
}

OPENCV_HAL_IMPL_WASM_MATH_FUNC(v_round, cvRound, v_float64x2, v_int32x4, double, int)
OPENCV_HAL_IMPL_WASM_MATH_FUNC(v_floor, cvFloor, v_float64x2, v_int32x4, double, int)
OPENCV_HAL_IMPL_WASM_MATH_FUNC(v_ceil, cvCeil, v_float64x2, v_int32x4, double, int)
OPENCV_HAL_IMPL_WASM_MATH_FUNC(v_trunc, int, v_float64x2, v_int32x4, double, int)

inline v_int32x4 v_round(const v_float64x2& a, const v_float64x2& b)
{
    fallback::v_float64x2 a_(a), b_(b);
    return fallback::v_round(a_, b_);
}

#define OPENCV_HAL_IMPL_WASM_TRANSPOSE4x4(_Tpvec, suffix) \
inline void v_transpose4x4(const _Tpvec& a0, const _Tpvec& a1, \
                           const _Tpvec& a2, const _Tpvec& a3, \
                           _Tpvec& b0, _Tpvec& b1, \
                           _Tpvec& b2, _Tpvec& b3) \
{ \
    v128_t t0 = wasm_unpacklo_##suffix(a0.val, a1.val); \
    v128_t t1 = wasm_unpacklo_##suffix(a2.val, a3.val); \
    v128_t t2 = wasm_unpackhi_##suffix(a0.val, a1.val); \
    v128_t t3 = wasm_unpackhi_##suffix(a2.val, a3.val); \
\
    b0.val = wasm_unpacklo_i64x2(t0, t1); \
    b1.val = wasm_unpackhi_i64x2(t0, t1); \
    b2.val = wasm_unpacklo_i64x2(t2, t3); \
    b3.val = wasm_unpackhi_i64x2(t2, t3); \
}

OPENCV_HAL_IMPL_WASM_TRANSPOSE4x4(v_uint32x4, i32x4)
OPENCV_HAL_IMPL_WASM_TRANSPOSE4x4(v_int32x4, i32x4)
OPENCV_HAL_IMPL_WASM_TRANSPOSE4x4(v_float32x4, i32x4)

// load deinterleave
inline void v_load_deinterleave(const uchar* ptr, v_uint8x16& a, v_uint8x16& b)
{
    v128_t t00 = wasm_v128_load(ptr);
    v128_t t01 = wasm_v128_load(ptr + 16);

    a.val = wasm_v8x16_shuffle(t00, t01, 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30);
    b.val = wasm_v8x16_shuffle(t00, t01, 1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31);
}

inline void v_load_deinterleave(const uchar* ptr, v_uint8x16& a, v_uint8x16& b, v_uint8x16& c)
{
    v128_t t00 = wasm_v128_load(ptr);
    v128_t t01 = wasm_v128_load(ptr + 16);
    v128_t t02 = wasm_v128_load(ptr + 32);

    v128_t t10 = wasm_v8x16_shuffle(t00, t01, 0,3,6,9,12,15,18,21,24,27,30,1,2,4,5,7);
    v128_t t11 = wasm_v8x16_shuffle(t00, t01, 1,4,7,10,13,16,19,22,25,28,31,0,2,3,5,6);
    v128_t t12 = wasm_v8x16_shuffle(t00, t01, 2,5,8,11,14,17,20,23,26,29,0,1,3,4,6,7);

    a.val = wasm_v8x16_shuffle(t10, t02, 0,1,2,3,4,5,6,7,8,9,10,17,20,23,26,29);
    b.val = wasm_v8x16_shuffle(t11, t02, 0,1,2,3,4,5,6,7,8,9,10,18,21,24,27,30);
    c.val = wasm_v8x16_shuffle(t12, t02, 0,1,2,3,4,5,6,7,8,9,16,19,22,25,28,31);
}

inline void v_load_deinterleave(const uchar* ptr, v_uint8x16& a, v_uint8x16& b, v_uint8x16& c, v_uint8x16& d)
{
    v128_t u0 = wasm_v128_load(ptr); // a0 b0 c0 d0 a1 b1 c1 d1 ...
    v128_t u1 = wasm_v128_load(ptr + 16); // a4 b4 c4 d4 ...
    v128_t u2 = wasm_v128_load(ptr + 32); // a8 b8 c8 d8 ...
    v128_t u3 = wasm_v128_load(ptr + 48); // a12 b12 c12 d12 ...

    v128_t v0 = wasm_v8x16_shuffle(u0, u1, 0,4,8,12,16,20,24,28,1,5,9,13,17,21,25,29);
    v128_t v1 = wasm_v8x16_shuffle(u2, u3, 0,4,8,12,16,20,24,28,1,5,9,13,17,21,25,29);
    v128_t v2 = wasm_v8x16_shuffle(u0, u1, 2,6,10,14,18,22,26,30,3,7,11,15,19,23,27,31);
    v128_t v3 = wasm_v8x16_shuffle(u2, u3, 2,6,10,14,18,22,26,30,3,7,11,15,19,23,27,31);

    a.val = wasm_v8x16_shuffle(v0, v1, 0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23);
    b.val = wasm_v8x16_shuffle(v0, v1, 8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31);
    c.val = wasm_v8x16_shuffle(v2, v3, 0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23);
    d.val = wasm_v8x16_shuffle(v2, v3, 8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31);
}

inline void v_load_deinterleave(const ushort* ptr, v_uint16x8& a, v_uint16x8& b)
{
    v128_t v0 = wasm_v128_load(ptr);     // a0 b0 a1 b1 a2 b2 a3 b3
    v128_t v1 = wasm_v128_load(ptr + 8); // a4 b4 a5 b5 a6 b6 a7 b7

    a.val = wasm_v8x16_shuffle(v0, v1, 0,1,4,5,8,9,12,13,16,17,20,21,24,25,28,29); // a0 a1 a2 a3 a4 a5 a6 a7
    b.val = wasm_v8x16_shuffle(v0, v1, 2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31); // b0 b1 ab b3 b4 b5 b6 b7
}

inline void v_load_deinterleave(const ushort* ptr, v_uint16x8& a, v_uint16x8& b, v_uint16x8& c)
{
    v128_t t00 = wasm_v128_load(ptr);        // a0 b0 c0 a1 b1 c1 a2 b2
    v128_t t01 = wasm_v128_load(ptr + 8);    // c2 a3 b3 c3 a4 b4 c4 a5
    v128_t t02 = wasm_v128_load(ptr + 16);  // b5 c5 a6 b6 c6 a7 b7 c7

    v128_t t10 = wasm_v8x16_shuffle(t00, t01, 0,1,6,7,12,13,18,19,24,25,30,31,2,3,4,5);
    v128_t t11 = wasm_v8x16_shuffle(t00, t01, 2,3,8,9,14,15,20,21,26,27,0,1,4,5,6,7);
    v128_t t12 = wasm_v8x16_shuffle(t00, t01, 4,5,10,11,16,17,22,23,28,29,0,1,2,3,6,7);

    a.val = wasm_v8x16_shuffle(t10, t02, 0,1,2,3,4,5,6,7,8,9,10,11,20,21,26,27);
    b.val = wasm_v8x16_shuffle(t11, t02, 0,1,2,3,4,5,6,7,8,9,16,17,22,23,28,29);
    c.val = wasm_v8x16_shuffle(t12, t02, 0,1,2,3,4,5,6,7,8,9,18,19,24,25,30,31);
}

inline void v_load_deinterleave(const ushort* ptr, v_uint16x8& a, v_uint16x8& b, v_uint16x8& c, v_uint16x8& d)
{
    v128_t u0 = wasm_v128_load(ptr); // a0 b0 c0 d0 a1 b1 c1 d1
    v128_t u1 = wasm_v128_load(ptr + 8); // a2 b2 c2 d2 ...
    v128_t u2 = wasm_v128_load(ptr + 16); // a4 b4 c4 d4 ...
    v128_t u3 = wasm_v128_load(ptr + 24); // a6 b6 c6 d6 ...

    v128_t v0 = wasm_v8x16_shuffle(u0, u1, 0,1,8,9,16,17,24,25,2,3,10,11,18,19,26,27); // a0 a1 a2 a3 b0 b1 b2 b3
    v128_t v1 = wasm_v8x16_shuffle(u2, u3, 0,1,8,9,16,17,24,25,2,3,10,11,18,19,26,27); // a4 a5 a6 a7 b4 b5 b6 b7
    v128_t v2 = wasm_v8x16_shuffle(u0, u1, 4,5,12,13,20,21,28,29,6,7,14,15,22,23,30,31); // c0 c1 c2 c3 d0 d1 d2 d3
    v128_t v3 = wasm_v8x16_shuffle(u2, u3, 4,5,12,13,20,21,28,29,6,7,14,15,22,23,30,31); // c4 c5 c6 c7 d4 d5 d6 d7

    a.val = wasm_v8x16_shuffle(v0, v1, 0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23);
    b.val = wasm_v8x16_shuffle(v0, v1, 8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31);
    c.val = wasm_v8x16_shuffle(v2, v3, 0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23);
    d.val = wasm_v8x16_shuffle(v2, v3, 8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31);
}

inline void v_load_deinterleave(const unsigned* ptr, v_uint32x4& a, v_uint32x4& b)
{
    v128_t v0 = wasm_v128_load(ptr);     // a0 b0 a1 b1
    v128_t v1 = wasm_v128_load(ptr + 4); // a2 b2 a3 b3

    a.val = wasm_v8x16_shuffle(v0, v1, 0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27); // a0 a1 a2 a3
    b.val = wasm_v8x16_shuffle(v0, v1, 4,5,6,7,12,13,14,15,20,21,22,23,28,29,30,31); // b0 b1 b2 b3
}

inline void v_load_deinterleave(const unsigned* ptr, v_uint32x4& a, v_uint32x4& b, v_uint32x4& c)
{
    v128_t t00 = wasm_v128_load(ptr);        // a0 b0 c0 a1
    v128_t t01 = wasm_v128_load(ptr + 4);     // b2 c2 a3 b3
    v128_t t02 = wasm_v128_load(ptr + 8);    // c3 a4 b4 c4

    v128_t t10 = wasm_v8x16_shuffle(t00, t01, 0,1,2,3,12,13,14,15,24,25,26,27,4,5,6,7);
    v128_t t11 = wasm_v8x16_shuffle(t00, t01, 4,5,6,7,16,17,18,19,28,29,30,31,0,1,2,3);
    v128_t t12 = wasm_v8x16_shuffle(t00, t01, 8,9,10,11,20,21,22,23,0,1,2,3,4,5,6,7);

    a.val = wasm_v8x16_shuffle(t10, t02, 0,1,2,3,4,5,6,7,8,9,10,11,20,21,22,23);
    b.val = wasm_v8x16_shuffle(t11, t02, 0,1,2,3,4,5,6,7,8,9,10,11,24,25,26,27);
    c.val = wasm_v8x16_shuffle(t12, t02, 0,1,2,3,4,5,6,7,16,17,18,19,28,29,30,31);
}

inline void v_load_deinterleave(const unsigned* ptr, v_uint32x4& a, v_uint32x4& b, v_uint32x4& c, v_uint32x4& d)
{
    v_uint32x4 s0(wasm_v128_load(ptr));      // a0 b0 c0 d0
    v_uint32x4 s1(wasm_v128_load(ptr + 4));  // a1 b1 c1 d1
    v_uint32x4 s2(wasm_v128_load(ptr + 8));  // a2 b2 c2 d2
    v_uint32x4 s3(wasm_v128_load(ptr + 12)); // a3 b3 c3 d3

    v_transpose4x4(s0, s1, s2, s3, a, b, c, d);
}

inline void v_load_deinterleave(const float* ptr, v_float32x4& a, v_float32x4& b)
{
    v128_t v0 = wasm_v128_load(ptr);       // a0 b0 a1 b1
    v128_t v1 = wasm_v128_load((ptr + 4)); // a2 b2 a3 b3

    a.val = wasm_v8x16_shuffle(v0, v1, 0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27); // a0 a1 a2 a3
    b.val = wasm_v8x16_shuffle(v0, v1, 4,5,6,7,12,13,14,15,20,21,22,23,28,29,30,31); // b0 b1 b2 b3
}

inline void v_load_deinterleave(const float* ptr, v_float32x4& a, v_float32x4& b, v_float32x4& c)
{
    v128_t t00 = wasm_v128_load(ptr);        // a0 b0 c0 a1
    v128_t t01 = wasm_v128_load(ptr + 4);     // b2 c2 a3 b3
    v128_t t02 = wasm_v128_load(ptr + 8);    // c3 a4 b4 c4

    v128_t t10 = wasm_v8x16_shuffle(t00, t01, 0,1,2,3,12,13,14,15,24,25,26,27,4,5,6,7);
    v128_t t11 = wasm_v8x16_shuffle(t00, t01, 4,5,6,7,16,17,18,19,28,29,30,31,0,1,2,3);
    v128_t t12 = wasm_v8x16_shuffle(t00, t01, 8,9,10,11,20,21,22,23,0,1,2,3,4,5,6,7);

    a.val = wasm_v8x16_shuffle(t10, t02, 0,1,2,3,4,5,6,7,8,9,10,11,20,21,22,23);
    b.val = wasm_v8x16_shuffle(t11, t02, 0,1,2,3,4,5,6,7,8,9,10,11,24,25,26,27);
    c.val = wasm_v8x16_shuffle(t12, t02, 0,1,2,3,4,5,6,7,16,17,18,19,28,29,30,31);
}

inline void v_load_deinterleave(const float* ptr, v_float32x4& a, v_float32x4& b, v_float32x4& c, v_float32x4& d)
{
    v_float32x4 s0(wasm_v128_load(ptr));      // a0 b0 c0 d0
    v_float32x4 s1(wasm_v128_load(ptr + 4));  // a1 b1 c1 d1
    v_float32x4 s2(wasm_v128_load(ptr + 8));  // a2 b2 c2 d2
    v_float32x4 s3(wasm_v128_load(ptr + 12)); // a3 b3 c3 d3

    v_transpose4x4(s0, s1, s2, s3, a, b, c, d);
}

inline void v_load_deinterleave(const uint64 *ptr, v_uint64x2& a, v_uint64x2& b)
{
    v128_t t0 = wasm_v128_load(ptr);      // a0 b0
    v128_t t1 = wasm_v128_load(ptr + 2);  // a1 b1

    a.val = wasm_unpacklo_i64x2(t0, t1);
    b.val = wasm_unpackhi_i64x2(t0, t1);
}

inline void v_load_deinterleave(const uint64 *ptr, v_uint64x2& a, v_uint64x2& b, v_uint64x2& c)
{
    v128_t t0 = wasm_v128_load(ptr);     // a0, b0
    v128_t t1 = wasm_v128_load(ptr + 2); // c0, a1
    v128_t t2 = wasm_v128_load(ptr + 4); // b1, c1

    a.val = wasm_v8x16_shuffle(t0, t1, 0,1,2,3,4,5,6,7,24,25,26,27,28,29,30,31);
    b.val = wasm_v8x16_shuffle(t0, t2, 8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23);
    c.val = wasm_v8x16_shuffle(t1, t2, 0,1,2,3,4,5,6,7,24,25,26,27,28,29,30,31);
}

inline void v_load_deinterleave(const uint64 *ptr, v_uint64x2& a,
                                v_uint64x2& b, v_uint64x2& c, v_uint64x2& d)
{
    v128_t t0 = wasm_v128_load(ptr);     // a0 b0
    v128_t t1 = wasm_v128_load(ptr + 2); // c0 d0
    v128_t t2 = wasm_v128_load(ptr + 4); // a1 b1
    v128_t t3 = wasm_v128_load(ptr + 6); // c1 d1

    a.val = wasm_unpacklo_i64x2(t0, t2);
    b.val = wasm_unpackhi_i64x2(t0, t2);
    c.val = wasm_unpacklo_i64x2(t1, t3);
    d.val = wasm_unpackhi_i64x2(t1, t3);
}

// store interleave

inline void v_store_interleave( uchar* ptr, const v_uint8x16& a, const v_uint8x16& b,
                                hal::StoreMode /*mode*/ = hal::STORE_UNALIGNED)
{
    v128_t v0 = wasm_unpacklo_i8x16(a.val, b.val);
    v128_t v1 = wasm_unpackhi_i8x16(a.val, b.val);

    wasm_v128_store(ptr, v0);
    wasm_v128_store(ptr + 16, v1);
}

inline void v_store_interleave( uchar* ptr, const v_uint8x16& a, const v_uint8x16& b,
                                const v_uint8x16& c, hal::StoreMode /*mode*/ = hal::STORE_UNALIGNED)
{
    v128_t t00 = wasm_v8x16_shuffle(a.val, b.val, 0,16,0,1,17,0,2,18,0,3,19,0,4,20,0,5);
    v128_t t01 = wasm_v8x16_shuffle(a.val, b.val, 21,0,6,22,0,7,23,0,8,24,0,9,25,0,10,26);
    v128_t t02 = wasm_v8x16_shuffle(a.val, b.val, 0,11,27,0,12,28,0,13,29,0,14,30,0,15,31,0);

    v128_t t10 = wasm_v8x16_shuffle(t00, c.val, 0,1,16,3,4,17,6,7,18,9,10,19,12,13,20,15);
    v128_t t11 = wasm_v8x16_shuffle(t01, c.val, 0,21,2,3,22,5,6,23,8,9,24,11,12,25,14,15);
    v128_t t12 = wasm_v8x16_shuffle(t02, c.val, 26,1,2,27,4,5,28,7,8,29,10,11,30,13,14,31);

    wasm_v128_store(ptr, t10);
    wasm_v128_store(ptr + 16, t11);
    wasm_v128_store(ptr + 32, t12);
}

inline void v_store_interleave( uchar* ptr, const v_uint8x16& a, const v_uint8x16& b,
                                const v_uint8x16& c, const v_uint8x16& d,
                                hal::StoreMode /*mode*/ = hal::STORE_UNALIGNED)
{
    // a0 a1 a2 a3 ....
    // b0 b1 b2 b3 ....
    // c0 c1 c2 c3 ....
    // d0 d1 d2 d3 ....
    v128_t u0 = wasm_unpacklo_i8x16(a.val, c.val); // a0 c0 a1 c1 ...
    v128_t u1 = wasm_unpackhi_i8x16(a.val, c.val); // a8 c8 a9 c9 ...
    v128_t u2 = wasm_unpacklo_i8x16(b.val, d.val); // b0 d0 b1 d1 ...
    v128_t u3 = wasm_unpackhi_i8x16(b.val, d.val); // b8 d8 b9 d9 ...

    v128_t v0 = wasm_unpacklo_i8x16(u0, u2); // a0 b0 c0 d0 ...
    v128_t v1 = wasm_unpackhi_i8x16(u0, u2); // a4 b4 c4 d4 ...
    v128_t v2 = wasm_unpacklo_i8x16(u1, u3); // a8 b8 c8 d8 ...
    v128_t v3 = wasm_unpackhi_i8x16(u1, u3); // a12 b12 c12 d12 ...

    wasm_v128_store(ptr, v0);
    wasm_v128_store(ptr + 16, v1);
    wasm_v128_store(ptr + 32, v2);
    wasm_v128_store(ptr + 48, v3);
}

inline void v_store_interleave( ushort* ptr, const v_uint16x8& a, const v_uint16x8& b,
                                hal::StoreMode /*mode*/ = hal::STORE_UNALIGNED)
{
    v128_t v0 = wasm_unpacklo_i16x8(a.val, b.val);
    v128_t v1 = wasm_unpackhi_i16x8(a.val, b.val);

    wasm_v128_store(ptr, v0);
    wasm_v128_store(ptr + 8, v1);
}

inline void v_store_interleave( ushort* ptr, const v_uint16x8& a,
                                const v_uint16x8& b, const v_uint16x8& c,
                                hal::StoreMode /*mode*/ = hal::STORE_UNALIGNED)
{
    v128_t t00 = wasm_v8x16_shuffle(a.val, b.val, 0,1,16,17,0,0,2,3,18,19,0,0,4,5,20,21);
    v128_t t01 = wasm_v8x16_shuffle(a.val, b.val, 0,0,6,7,22,23,0,0,8,9,24,25,0,0,10,11);
    v128_t t02 = wasm_v8x16_shuffle(a.val, b.val, 26,27,0,0,12,13,28,29,0,0,14,15,30,31,0,0);

    v128_t t10 = wasm_v8x16_shuffle(t00, c.val, 0,1,2,3,16,17,6,7,8,9,18,19,12,13,14,15);
    v128_t t11 = wasm_v8x16_shuffle(t01, c.val, 20,21,2,3,4,5,22,23,8,9,10,11,24,25,14,15);
    v128_t t12 = wasm_v8x16_shuffle(t02, c.val, 0,1,26,27,4,5,6,7,28,29,10,11,12,13,30,31);

    wasm_v128_store(ptr, t10);
    wasm_v128_store(ptr + 8, t11);
    wasm_v128_store(ptr + 16, t12);
}

inline void v_store_interleave( ushort* ptr, const v_uint16x8& a, const v_uint16x8& b,
                                const v_uint16x8& c, const v_uint16x8& d,
                                hal::StoreMode /*mode*/ = hal::STORE_UNALIGNED)
{
    // a0 a1 a2 a3 ....
    // b0 b1 b2 b3 ....
    // c0 c1 c2 c3 ....
    // d0 d1 d2 d3 ....
    v128_t u0 = wasm_unpacklo_i16x8(a.val, c.val); // a0 c0 a1 c1 ...
    v128_t u1 = wasm_unpackhi_i16x8(a.val, c.val); // a4 c4 a5 c5 ...
    v128_t u2 = wasm_unpacklo_i16x8(b.val, d.val); // b0 d0 b1 d1 ...
    v128_t u3 = wasm_unpackhi_i16x8(b.val, d.val); // b4 d4 b5 d5 ...

    v128_t v0 = wasm_unpacklo_i16x8(u0, u2); // a0 b0 c0 d0 ...
    v128_t v1 = wasm_unpackhi_i16x8(u0, u2); // a2 b2 c2 d2 ...
    v128_t v2 = wasm_unpacklo_i16x8(u1, u3); // a4 b4 c4 d4 ...
    v128_t v3 = wasm_unpackhi_i16x8(u1, u3); // a6 b6 c6 d6 ...

    wasm_v128_store(ptr, v0);
    wasm_v128_store(ptr + 8, v1);
    wasm_v128_store(ptr + 16, v2);
    wasm_v128_store(ptr + 24, v3);
}

inline void v_store_interleave( unsigned* ptr, const v_uint32x4& a, const v_uint32x4& b,
                                hal::StoreMode /*mode*/ = hal::STORE_UNALIGNED)
{
    v128_t v0 = wasm_unpacklo_i32x4(a.val, b.val);
    v128_t v1 = wasm_unpackhi_i32x4(a.val, b.val);

    wasm_v128_store(ptr, v0);
    wasm_v128_store(ptr + 4, v1);
}

inline void v_store_interleave( unsigned* ptr, const v_uint32x4& a, const v_uint32x4& b,
                                const v_uint32x4& c, hal::StoreMode /*mode*/ = hal::STORE_UNALIGNED)
{
    v128_t t00 = wasm_v8x16_shuffle(a.val, b.val, 0,1,2,3,16,17,18,19,0,0,0,0,4,5,6,7);
    v128_t t01 = wasm_v8x16_shuffle(a.val, b.val, 20,21,22,23,0,0,0,0,8,9,10,11,24,25,26,27);
    v128_t t02 = wasm_v8x16_shuffle(a.val, b.val, 0,0,0,0,12,13,14,15,28,29,30,31,0,0,0,0);

    v128_t t10 = wasm_v8x16_shuffle(t00, c.val, 0,1,2,3,4,5,6,7,16,17,18,19,12,13,14,15);
    v128_t t11 = wasm_v8x16_shuffle(t01, c.val, 0,1,2,3,20,21,22,23,8,9,10,11,12,13,14,15);
    v128_t t12 = wasm_v8x16_shuffle(t02, c.val, 24,25,26,27,4,5,6,7,8,9,10,11,28,29,30,31);

    wasm_v128_store(ptr, t10);
    wasm_v128_store(ptr + 4, t11);
    wasm_v128_store(ptr + 8, t12);
}

inline void v_store_interleave(unsigned* ptr, const v_uint32x4& a, const v_uint32x4& b,
                               const v_uint32x4& c, const v_uint32x4& d,
                               hal::StoreMode /*mode*/ = hal::STORE_UNALIGNED)
{
    v_uint32x4 v0, v1, v2, v3;
    v_transpose4x4(a, b, c, d, v0, v1, v2, v3);

    wasm_v128_store(ptr, v0.val);
    wasm_v128_store(ptr + 4, v1.val);
    wasm_v128_store(ptr + 8, v2.val);
    wasm_v128_store(ptr + 12, v3.val);
}

// 2-channel, float only
inline void v_store_interleave(float* ptr, const v_float32x4& a, const v_float32x4& b,
                               hal::StoreMode /*mode*/ = hal::STORE_UNALIGNED)
{
    v128_t v0 = wasm_unpacklo_i32x4(a.val, b.val);
    v128_t v1 = wasm_unpackhi_i32x4(a.val, b.val);

    wasm_v128_store(ptr, v0);
    wasm_v128_store(ptr + 4, v1);
}

inline void v_store_interleave(float* ptr, const v_float32x4& a, const v_float32x4& b,
                               const v_float32x4& c, hal::StoreMode /*mode*/ = hal::STORE_UNALIGNED)
{
    v128_t t00 = wasm_v8x16_shuffle(a.val, b.val, 0,1,2,3,16,17,18,19,0,0,0,0,4,5,6,7);
    v128_t t01 = wasm_v8x16_shuffle(a.val, b.val, 20,21,22,23,0,0,0,0,8,9,10,11,24,25,26,27);
    v128_t t02 = wasm_v8x16_shuffle(a.val, b.val, 0,0,0,0,12,13,14,15,28,29,30,31,0,0,0,0);

    v128_t t10 = wasm_v8x16_shuffle(t00, c.val, 0,1,2,3,4,5,6,7,16,17,18,19,12,13,14,15);
    v128_t t11 = wasm_v8x16_shuffle(t01, c.val, 0,1,2,3,20,21,22,23,8,9,10,11,12,13,14,15);
    v128_t t12 = wasm_v8x16_shuffle(t02, c.val, 24,25,26,27,4,5,6,7,8,9,10,11,28,29,30,31);

    wasm_v128_store(ptr, t10);
    wasm_v128_store(ptr + 4, t11);
    wasm_v128_store(ptr + 8, t12);
}

inline void v_store_interleave(float* ptr, const v_float32x4& a, const v_float32x4& b,
                               const v_float32x4& c, const v_float32x4& d,
                               hal::StoreMode /*mode*/ = hal::STORE_UNALIGNED)
{
    v_float32x4 v0, v1, v2, v3;
    v_transpose4x4(a, b, c, d, v0, v1, v2, v3);

    wasm_v128_store(ptr, v0.val);
    wasm_v128_store(ptr + 4, v1.val);
    wasm_v128_store(ptr + 8, v2.val);
    wasm_v128_store(ptr + 12, v3.val);
}

inline void v_store_interleave(uint64 *ptr, const v_uint64x2& a, const v_uint64x2& b,
                               hal::StoreMode /*mode*/ = hal::STORE_UNALIGNED)
{
    v128_t v0 = wasm_unpacklo_i64x2(a.val, b.val);
    v128_t v1 = wasm_unpackhi_i64x2(a.val, b.val);

    wasm_v128_store(ptr, v0);
    wasm_v128_store(ptr + 2, v1);
}

inline void v_store_interleave(uint64 *ptr, const v_uint64x2& a, const v_uint64x2& b,
                               const v_uint64x2& c, hal::StoreMode /*mode*/ = hal::STORE_UNALIGNED)
{
    v128_t v0 = wasm_v8x16_shuffle(a.val, b.val, 0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23);
    v128_t v1 = wasm_v8x16_shuffle(a.val, c.val, 16,17,18,19,20,21,22,23,8,9,10,11,12,13,14,15);
    v128_t v2 = wasm_v8x16_shuffle(b.val, c.val, 8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31);

    wasm_v128_store(ptr, v0);
    wasm_v128_store(ptr + 2, v1);
    wasm_v128_store(ptr + 4, v2);
}

inline void v_store_interleave(uint64 *ptr, const v_uint64x2& a, const v_uint64x2& b,
                               const v_uint64x2& c, const v_uint64x2& d,
                               hal::StoreMode /*mode*/ = hal::STORE_UNALIGNED)
{
    v128_t v0 = wasm_unpacklo_i64x2(a.val, b.val);
    v128_t v1 = wasm_unpacklo_i64x2(c.val, d.val);
    v128_t v2 = wasm_unpackhi_i64x2(a.val, b.val);
    v128_t v3 = wasm_unpackhi_i64x2(c.val, d.val);

    wasm_v128_store(ptr, v0);
    wasm_v128_store(ptr + 2, v1);
    wasm_v128_store(ptr + 4, v2);
    wasm_v128_store(ptr + 6, v3);
}

#define OPENCV_HAL_IMPL_WASM_LOADSTORE_INTERLEAVE(_Tpvec0, _Tp0, suffix0, _Tpvec1, _Tp1, suffix1) \
inline void v_load_deinterleave( const _Tp0* ptr, _Tpvec0& a0, _Tpvec0& b0 ) \
{ \
    _Tpvec1 a1, b1; \
    v_load_deinterleave((const _Tp1*)ptr, a1, b1); \
    a0 = v_reinterpret_as_##suffix0(a1); \
    b0 = v_reinterpret_as_##suffix0(b1); \
} \
inline void v_load_deinterleave( const _Tp0* ptr, _Tpvec0& a0, _Tpvec0& b0, _Tpvec0& c0 ) \
{ \
    _Tpvec1 a1, b1, c1; \
    v_load_deinterleave((const _Tp1*)ptr, a1, b1, c1); \
    a0 = v_reinterpret_as_##suffix0(a1); \
    b0 = v_reinterpret_as_##suffix0(b1); \
    c0 = v_reinterpret_as_##suffix0(c1); \
} \
inline void v_load_deinterleave( const _Tp0* ptr, _Tpvec0& a0, _Tpvec0& b0, _Tpvec0& c0, _Tpvec0& d0 ) \
{ \
    _Tpvec1 a1, b1, c1, d1; \
    v_load_deinterleave((const _Tp1*)ptr, a1, b1, c1, d1); \
    a0 = v_reinterpret_as_##suffix0(a1); \
    b0 = v_reinterpret_as_##suffix0(b1); \
    c0 = v_reinterpret_as_##suffix0(c1); \
    d0 = v_reinterpret_as_##suffix0(d1); \
} \
inline void v_store_interleave( _Tp0* ptr, const _Tpvec0& a0, const _Tpvec0& b0, \
                                hal::StoreMode mode = hal::STORE_UNALIGNED ) \
{ \
    _Tpvec1 a1 = v_reinterpret_as_##suffix1(a0); \
    _Tpvec1 b1 = v_reinterpret_as_##suffix1(b0); \
    v_store_interleave((_Tp1*)ptr, a1, b1, mode);      \
} \
inline void v_store_interleave( _Tp0* ptr, const _Tpvec0& a0, const _Tpvec0& b0, \
                                const _Tpvec0& c0, hal::StoreMode mode = hal::STORE_UNALIGNED ) \
{ \
    _Tpvec1 a1 = v_reinterpret_as_##suffix1(a0); \
    _Tpvec1 b1 = v_reinterpret_as_##suffix1(b0); \
    _Tpvec1 c1 = v_reinterpret_as_##suffix1(c0); \
    v_store_interleave((_Tp1*)ptr, a1, b1, c1, mode);  \
} \
inline void v_store_interleave( _Tp0* ptr, const _Tpvec0& a0, const _Tpvec0& b0, \
                                const _Tpvec0& c0, const _Tpvec0& d0, \
                                hal::StoreMode mode = hal::STORE_UNALIGNED ) \
{ \
    _Tpvec1 a1 = v_reinterpret_as_##suffix1(a0); \
    _Tpvec1 b1 = v_reinterpret_as_##suffix1(b0); \
    _Tpvec1 c1 = v_reinterpret_as_##suffix1(c0); \
    _Tpvec1 d1 = v_reinterpret_as_##suffix1(d0); \
    v_store_interleave((_Tp1*)ptr, a1, b1, c1, d1, mode); \
}

OPENCV_HAL_IMPL_WASM_LOADSTORE_INTERLEAVE(v_int8x16, schar, s8, v_uint8x16, uchar, u8)
OPENCV_HAL_IMPL_WASM_LOADSTORE_INTERLEAVE(v_int16x8, short, s16, v_uint16x8, ushort, u16)
OPENCV_HAL_IMPL_WASM_LOADSTORE_INTERLEAVE(v_int32x4, int, s32, v_uint32x4, unsigned, u32)
OPENCV_HAL_IMPL_WASM_LOADSTORE_INTERLEAVE(v_int64x2, int64, s64, v_uint64x2, uint64, u64)
OPENCV_HAL_IMPL_WASM_LOADSTORE_INTERLEAVE(v_float64x2, double, f64, v_uint64x2, uint64, u64)

inline v_float32x4 v_cvt_f32(const v_int32x4& a)
{
    return v_float32x4(wasm_f32x4_convert_i32x4(a.val));
}

inline v_float32x4 v_cvt_f32(const v_float64x2& a)
{
    fallback::v_float64x2 a_(a);
    return fallback::v_cvt_f32(a_);
}

inline v_float32x4 v_cvt_f32(const v_float64x2& a, const v_float64x2& b)
{
    fallback::v_float64x2 a_(a), b_(b);
    return fallback::v_cvt_f32(a_, b_);
}

inline v_float64x2 v_cvt_f64(const v_int32x4& a)
{
#ifdef __wasm_unimplemented_simd128__
    v128_t p = v128_cvti32x4_i64x2(a.val);
    return v_float64x2(wasm_f64x2_convert_i64x2(p));
#else
    fallback::v_int32x4 a_(a);
    return fallback::v_cvt_f64(a_);
#endif
}

inline v_float64x2 v_cvt_f64_high(const v_int32x4& a)
{
#ifdef __wasm_unimplemented_simd128__
    v128_t p = v128_cvti32x4_i64x2_high(a.val);
    return v_float64x2(wasm_f64x2_convert_i64x2(p));
#else
    fallback::v_int32x4 a_(a);
    return fallback::v_cvt_f64_high(a_);
#endif
}

inline v_float64x2 v_cvt_f64(const v_float32x4& a)
{
    fallback::v_float32x4 a_(a);
    return fallback::v_cvt_f64(a_);
}

inline v_float64x2 v_cvt_f64_high(const v_float32x4& a)
{
    fallback::v_float32x4 a_(a);
    return fallback::v_cvt_f64_high(a_);
}

inline v_float64x2 v_cvt_f64(const v_int64x2& a)
{
#ifdef __wasm_unimplemented_simd128__
    return v_float64x2(wasm_f64x2_convert_i64x2(a.val));
#else
    fallback::v_int64x2 a_(a);
    return fallback::v_cvt_f64(a_);
#endif
}

////////////// Lookup table access ////////////////////

inline v_int8x16 v_lut(const schar* tab, const int* idx)
{
    return v_int8x16(tab[idx[0]], tab[idx[1]], tab[idx[ 2]], tab[idx[ 3]], tab[idx[ 4]], tab[idx[ 5]], tab[idx[ 6]], tab[idx[ 7]],
                     tab[idx[8]], tab[idx[9]], tab[idx[10]], tab[idx[11]], tab[idx[12]], tab[idx[13]], tab[idx[14]], tab[idx[15]]);
}
inline v_int8x16 v_lut_pairs(const schar* tab, const int* idx)
{
    return v_int8x16(tab[idx[0]], tab[idx[0]+1], tab[idx[1]], tab[idx[1]+1], tab[idx[2]], tab[idx[2]+1], tab[idx[3]], tab[idx[3]+1],
                     tab[idx[4]], tab[idx[4]+1], tab[idx[5]], tab[idx[5]+1], tab[idx[6]], tab[idx[6]+1], tab[idx[7]], tab[idx[7]+1]);
}
inline v_int8x16 v_lut_quads(const schar* tab, const int* idx)
{
    return v_int8x16(tab[idx[0]], tab[idx[0]+1], tab[idx[0]+2], tab[idx[0]+3], tab[idx[1]], tab[idx[1]+1], tab[idx[1]+2], tab[idx[1]+3],
                     tab[idx[2]], tab[idx[2]+1], tab[idx[2]+2], tab[idx[2]+3], tab[idx[3]], tab[idx[3]+1], tab[idx[3]+2], tab[idx[3]+3]);
}
inline v_uint8x16 v_lut(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v_lut((const schar *)tab, idx)); }
inline v_uint8x16 v_lut_pairs(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v_lut_pairs((const schar *)tab, idx)); }
inline v_uint8x16 v_lut_quads(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v_lut_quads((const schar *)tab, idx)); }

inline v_int16x8 v_lut(const short* tab, const int* idx)
{
    return v_int16x8(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]],
                     tab[idx[4]], tab[idx[5]], tab[idx[6]], tab[idx[7]]);
}
inline v_int16x8 v_lut_pairs(const short* tab, const int* idx)
{
    return v_int16x8(tab[idx[0]], tab[idx[0]+1], tab[idx[1]], tab[idx[1]+1],
                     tab[idx[2]], tab[idx[2]+1], tab[idx[3]], tab[idx[3]+1]);
}
inline v_int16x8 v_lut_quads(const short* tab, const int* idx)
{
    return v_int16x8(tab[idx[0]], tab[idx[0]+1], tab[idx[0]+2], tab[idx[0]+3],
                     tab[idx[1]], tab[idx[1]+1], tab[idx[1]+2], tab[idx[1]+3]);
}
inline v_uint16x8 v_lut(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v_lut((const short *)tab, idx)); }
inline v_uint16x8 v_lut_pairs(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v_lut_pairs((const short *)tab, idx)); }
inline v_uint16x8 v_lut_quads(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v_lut_quads((const short *)tab, idx)); }

inline v_int32x4 v_lut(const int* tab, const int* idx)
{
    return v_int32x4(tab[idx[0]], tab[idx[1]],
                     tab[idx[2]], tab[idx[3]]);
}
inline v_int32x4 v_lut_pairs(const int* tab, const int* idx)
{
    return v_int32x4(tab[idx[0]], tab[idx[0]+1],
                     tab[idx[1]], tab[idx[1]+1]);
}
inline v_int32x4 v_lut_quads(const int* tab, const int* idx)
{
    return v_int32x4(wasm_v128_load(tab + idx[0]));
}
inline v_uint32x4 v_lut(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut((const int *)tab, idx)); }
inline v_uint32x4 v_lut_pairs(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut_pairs((const int *)tab, idx)); }
inline v_uint32x4 v_lut_quads(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut_quads((const int *)tab, idx)); }

inline v_int64x2 v_lut(const int64_t* tab, const int* idx)
{
    return v_int64x2(tab[idx[0]], tab[idx[1]]);
}
inline v_int64x2 v_lut_pairs(const int64_t* tab, const int* idx)
{
    return v_int64x2(wasm_v128_load(tab + idx[0]));
}
inline v_uint64x2 v_lut(const uint64_t* tab, const int* idx) { return v_reinterpret_as_u64(v_lut((const int64_t *)tab, idx)); }
inline v_uint64x2 v_lut_pairs(const uint64_t* tab, const int* idx) { return v_reinterpret_as_u64(v_lut_pairs((const int64_t *)tab, idx)); }

inline v_float32x4 v_lut(const float* tab, const int* idx)
{
    return v_float32x4(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]]);
}
inline v_float32x4 v_lut_pairs(const float* tab, const int* idx) { return v_reinterpret_as_f32(v_lut_pairs((const int *)tab, idx)); }
inline v_float32x4 v_lut_quads(const float* tab, const int* idx) { return v_reinterpret_as_f32(v_lut_quads((const int *)tab, idx)); }

inline v_float64x2 v_lut(const double* tab, const int* idx)
{
    return v_float64x2(tab[idx[0]], tab[idx[1]]);
}
inline v_float64x2 v_lut_pairs(const double* tab, const int* idx)
{
    return v_float64x2(wasm_v128_load(tab + idx[0]));
}

inline v_int32x4 v_lut(const int* tab, const v_int32x4& idxvec)
{
    return v_int32x4(tab[wasm_i32x4_extract_lane(idxvec.val, 0)],
                     tab[wasm_i32x4_extract_lane(idxvec.val, 1)],
                     tab[wasm_i32x4_extract_lane(idxvec.val, 2)],
                     tab[wasm_i32x4_extract_lane(idxvec.val, 3)]);
}

inline v_uint32x4 v_lut(const unsigned* tab, const v_int32x4& idxvec)
{
    return v_reinterpret_as_u32(v_lut((const int *)tab, idxvec));
}

inline v_float32x4 v_lut(const float* tab, const v_int32x4& idxvec)
{
    return v_float32x4(tab[wasm_i32x4_extract_lane(idxvec.val, 0)],
                       tab[wasm_i32x4_extract_lane(idxvec.val, 1)],
                       tab[wasm_i32x4_extract_lane(idxvec.val, 2)],
                       tab[wasm_i32x4_extract_lane(idxvec.val, 3)]);
}

inline v_float64x2 v_lut(const double* tab, const v_int32x4& idxvec)
{
    return v_float64x2(tab[wasm_i32x4_extract_lane(idxvec.val, 0)],
                       tab[wasm_i32x4_extract_lane(idxvec.val, 1)]);
}

// loads pairs from the table and deinterleaves them, e.g. returns:
//   x = (tab[idxvec[0], tab[idxvec[1]], tab[idxvec[2]], tab[idxvec[3]]),
//   y = (tab[idxvec[0]+1], tab[idxvec[1]+1], tab[idxvec[2]+1], tab[idxvec[3]+1])
// note that the indices are float's indices, not the float-pair indices.
// in theory, this function can be used to implement bilinear interpolation,
// when idxvec are the offsets within the image.
inline void v_lut_deinterleave(const float* tab, const v_int32x4& idxvec, v_float32x4& x, v_float32x4& y)
{
    x = v_float32x4(tab[wasm_i32x4_extract_lane(idxvec.val, 0)],
                    tab[wasm_i32x4_extract_lane(idxvec.val, 1)],
                    tab[wasm_i32x4_extract_lane(idxvec.val, 2)],
                    tab[wasm_i32x4_extract_lane(idxvec.val, 3)]);
    y = v_float32x4(tab[wasm_i32x4_extract_lane(idxvec.val, 0)+1],
                    tab[wasm_i32x4_extract_lane(idxvec.val, 1)+1],
                    tab[wasm_i32x4_extract_lane(idxvec.val, 2)+1],
                    tab[wasm_i32x4_extract_lane(idxvec.val, 3)+1]);
}

inline void v_lut_deinterleave(const double* tab, const v_int32x4& idxvec, v_float64x2& x, v_float64x2& y)
{
    v128_t xy0 = wasm_v128_load(tab + wasm_i32x4_extract_lane(idxvec.val, 0));
    v128_t xy1 = wasm_v128_load(tab + wasm_i32x4_extract_lane(idxvec.val, 1));
    x.val = wasm_unpacklo_i64x2(xy0, xy1);
    y.val = wasm_unpacklo_i64x2(xy0, xy1);
}

inline v_int8x16 v_interleave_pairs(const v_int8x16& vec)
{
    return v_int8x16(wasm_v8x16_shuffle(vec.val, vec.val, 0,2,1,3,4,6,5,7,8,10,9,11,12,14,13,15));
}
inline v_uint8x16 v_interleave_pairs(const v_uint8x16& vec) { return v_reinterpret_as_u8(v_interleave_pairs(v_reinterpret_as_s8(vec))); }
inline v_int8x16 v_interleave_quads(const v_int8x16& vec)
{
    return v_int8x16(wasm_v8x16_shuffle(vec.val, vec.val, 0,4,1,5,2,6,3,7,8,12,9,13,10,14,11,15));
}
inline v_uint8x16 v_interleave_quads(const v_uint8x16& vec) { return v_reinterpret_as_u8(v_interleave_quads(v_reinterpret_as_s8(vec))); }

inline v_int16x8 v_interleave_pairs(const v_int16x8& vec)
{
    return v_int16x8(wasm_v8x16_shuffle(vec.val, vec.val, 0,1,4,5,2,3,6,7,8,9,12,13,10,11,14,15));
}
inline v_uint16x8 v_interleave_pairs(const v_uint16x8& vec) { return v_reinterpret_as_u16(v_interleave_pairs(v_reinterpret_as_s16(vec))); }
inline v_int16x8 v_interleave_quads(const v_int16x8& vec)
{
    return v_int16x8(wasm_v8x16_shuffle(vec.val, vec.val, 0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15));
}
inline v_uint16x8 v_interleave_quads(const v_uint16x8& vec) { return v_reinterpret_as_u16(v_interleave_quads(v_reinterpret_as_s16(vec))); }

inline v_int32x4 v_interleave_pairs(const v_int32x4& vec)
{
    return v_int32x4(wasm_v8x16_shuffle(vec.val, vec.val, 0,1,2,3,8,9,10,11,4,5,6,7,12,13,14,15));
}
inline v_uint32x4 v_interleave_pairs(const v_uint32x4& vec) { return v_reinterpret_as_u32(v_interleave_pairs(v_reinterpret_as_s32(vec))); }
inline v_float32x4 v_interleave_pairs(const v_float32x4& vec)
{
    return v_float32x4(wasm_v8x16_shuffle(vec.val, vec.val, 0,1,2,3,8,9,10,11,4,5,6,7,12,13,14,15));
}

inline v_int8x16 v_pack_triplets(const v_int8x16& vec)
{
    return v_int8x16(wasm_v8x16_shuffle(vec.val, vec.val, 0,1,2,4,5,6,8,9,10,12,13,14,16,16,16,16));
}
inline v_uint8x16 v_pack_triplets(const v_uint8x16& vec) { return v_reinterpret_as_u8(v_pack_triplets(v_reinterpret_as_s8(vec))); }

inline v_int16x8 v_pack_triplets(const v_int16x8& vec)
{
    return v_int16x8(wasm_v8x16_shuffle(vec.val, vec.val, 0,1,2,3,4,5,8,9,10,11,12,13,14,15,6,7));
}
inline v_uint16x8 v_pack_triplets(const v_uint16x8& vec) { return v_reinterpret_as_u16(v_pack_triplets(v_reinterpret_as_s16(vec))); }

inline v_int32x4 v_pack_triplets(const v_int32x4& vec) { return vec; }
inline v_uint32x4 v_pack_triplets(const v_uint32x4& vec) { return vec; }
inline v_float32x4 v_pack_triplets(const v_float32x4& vec) { return vec; }

template<int i, typename _Tp>
inline typename _Tp::lane_type v_extract_n(const _Tp& a)
{
    return v_rotate_right<i>(a).get0();
}

template<int i>
inline v_uint32x4 v_broadcast_element(const v_uint32x4& a)
{
    return v_setall_u32(v_extract_n<i>(a));
}
template<int i>
inline v_int32x4 v_broadcast_element(const v_int32x4& a)
{
    return v_setall_s32(v_extract_n<i>(a));
}
template<int i>
inline v_float32x4 v_broadcast_element(const v_float32x4& a)
{
    return v_setall_f32(v_extract_n<i>(a));
}


////////////// FP16 support ///////////////////////////

inline v_float32x4 v_load_expand(const float16_t* ptr)
{
    return fallback::v_load_expand(ptr);
}

inline void v_pack_store(float16_t* ptr, const v_float32x4& v)
{
    fallback::v_float32x4 v_(v);
    fallback::v_pack_store(ptr, v_);
}

inline void v_cleanup() {}

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END

//! @endcond

}

#endif
