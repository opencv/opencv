// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

// NB: allow including this *.hpp several times!
// #pragma once -- don't: this file is NOT once!

#if !defined(GAPI_STANDALONE)

#include "opencv2/gapi/own/saturate.hpp"

#include "opencv2/core.hpp"
#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/core/hal/hal.hpp>

#include <cstdint>
#include <cstring>

#include <algorithm>
#include <limits>
#include <vector>

#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wstrict-overflow"
#endif

using cv::gapi::own::saturate;

namespace cv {
namespace gapi {
namespace fluid {

CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

#define DIV_SIMD(SRC, DST)                                     \
int div_simd(const SRC in1[], const SRC in2[], DST out[],      \
             const int length, double _scale);

DIV_SIMD(uchar, uchar)
DIV_SIMD(ushort, uchar)
DIV_SIMD(short, uchar)
DIV_SIMD(float, uchar)
DIV_SIMD(short, short)
DIV_SIMD(ushort, short)
DIV_SIMD(uchar, short)
DIV_SIMD(float, short)
DIV_SIMD(ushort, ushort)
DIV_SIMD(uchar, ushort)
DIV_SIMD(short, ushort)
DIV_SIMD(float, ushort)
DIV_SIMD(uchar, float)
DIV_SIMD(ushort, float)
DIV_SIMD(short, float)
DIV_SIMD(float, float)

#undef DIV_SIMD

#define MUL_SIMD(SRC, DST)                                     \
int mul_simd(const SRC in1[], const SRC in2[], DST out[],      \
             const int length, double _scale);

MUL_SIMD(uchar, uchar)
MUL_SIMD(ushort, uchar)
MUL_SIMD(short, uchar)
MUL_SIMD(float, uchar)
MUL_SIMD(short, short)
MUL_SIMD(ushort, short)
MUL_SIMD(uchar, short)
MUL_SIMD(float, short)
MUL_SIMD(ushort, ushort)
MUL_SIMD(uchar, ushort)
MUL_SIMD(short, ushort)
MUL_SIMD(float, ushort)
MUL_SIMD(uchar, float)
MUL_SIMD(ushort, float)
MUL_SIMD(short, float)
MUL_SIMD(float, float)

#undef MUL_SIMD

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

struct scale_tag {};
struct not_scale_tag {};

template<typename scalar_t>
struct vector_type_of;

template<typename scalar_t>
using vector_type_of_t = typename vector_type_of<scalar_t>::type;

template<> struct vector_type_of<uchar> { using type = v_uint8; };
template<> struct vector_type_of<ushort> { using type = v_uint16; };
template<> struct vector_type_of<short> { using type = v_int16; };

CV_ALWAYS_INLINE v_float32 vg_load_f32(const float* in)
{
    return vx_load(in);
}

CV_ALWAYS_INLINE v_float32 vg_load_f32(const ushort* in)
{
    return v_cvt_f32(v_reinterpret_as_s32(vx_load_expand(in)));
}

CV_ALWAYS_INLINE v_float32 vg_load_f32(const short* in)
{
    return v_cvt_f32(v_reinterpret_as_s32(vx_load_expand(in)));
}

CV_ALWAYS_INLINE v_float32 vg_load_f32(const uchar* in)
{
    return v_cvt_f32(v_reinterpret_as_s32(vx_load_expand_q(in)));
}

CV_ALWAYS_INLINE v_float32 mul_op(scale_tag, const v_float32& a, const v_float32& b, const v_float32& scale)
{
    return (scale*a * b);
}

CV_ALWAYS_INLINE v_float32 mul_op(not_scale_tag, const v_float32& a, const v_float32& b, const v_float32&)
{
    return a * b;
}

CV_ALWAYS_INLINE v_float32 div_op(scale_tag, const v_float32& a, const v_float32& div, const v_float32& scale)
{
    return (a*scale/div);
}

CV_ALWAYS_INLINE v_float32 div_op(not_scale_tag, const v_float32& a, const v_float32& div, const v_float32&)
{
    return a / div;
}

CV_ALWAYS_INLINE void v_store_i16(short* dst, v_int32& res1, v_int32& res2)
{
    vx_store(dst, v_pack(res1, res2));
}

CV_ALWAYS_INLINE void v_store_i16(ushort* dst, v_int32& res1, v_int32& res2)
{
    vx_store(dst, v_pack_u(res1, res2));
}

CV_ALWAYS_INLINE void v_store_select(short* dst, const v_int16& div, const v_int16& v_zero,
                                     const v_int32& res1, const v_int32& res2)
{
    vx_store(dst, v_select(div == v_zero, v_zero, v_pack(res1, res2)));
}

CV_ALWAYS_INLINE void v_store_select(ushort* dst, const v_int16& div, const v_int16& v_zero,
                                     const v_int32& res1, const v_int32& res2)
{
    v_uint16 sel = v_reinterpret_as_u16(v_select(div == v_zero, v_zero, v_pack(res1, res2)));
    vx_store(dst, sel);
}

//=================================================================================================

template<typename scale_tag_t, typename SRC, typename DST>
CV_ALWAYS_INLINE
typename std::enable_if<(std::is_same<SRC, short>::value && std::is_same<DST, ushort>::value) ||
                        (std::is_same<SRC, ushort>::value && std::is_same<DST, ushort>::value) ||
                        (std::is_same<SRC, ushort>::value && std::is_same<DST, short>::value), int>::type
div_hal(scale_tag_t t, const SRC in1[], const SRC in2[], DST out[], const int length, double _scale)
{
    constexpr int nlanes = vector_type_of_t<DST>::nlanes;

    if (length < nlanes)
        return 0;

    v_int16 v_zero = vx_setall_s16(0);
    v_float32 scale = vx_setall_f32(static_cast<float>(_scale));

    int x = 0;
    for (;;)
    {
        for (; x <= length - nlanes; x += nlanes)
        {
            v_float32 a1 = vg_load_f32(&in1[x]);
            v_float32 a2 = vg_load_f32(&in1[x + nlanes / 2]);

            v_int16 div = v_reinterpret_as_s16(vx_load(&in2[x]));

            v_float32 fdiv1 = v_cvt_f32(v_expand_low(div));
            v_float32 fdiv2 = v_cvt_f32(v_expand_high(div));

            v_int32 r1 = v_round(div_op(t, a1, fdiv1, scale));
            v_int32 r2 = v_round(div_op(t, a2, fdiv2, scale));

            v_store_select(&out[x], div, v_zero, r1, r2);
        }

        if (x < length)
        {
            x = length - nlanes;
            continue;  // process one more time (unaligned tail)
        }
        break;
    }
    return x;
}

//-------------------------------------------------------------------------------------------------

template<typename scale_tag_t, typename SRC>
CV_ALWAYS_INLINE
typename std::enable_if<std::is_same<SRC, short>::value ||
                        std::is_same<SRC, ushort>::value, int>::type
div_hal(scale_tag_t t, const SRC in1[], const SRC in2[], uchar out[], const int length, double _scale)
{
    constexpr int nlanes = v_uint8::nlanes;

    if (length < nlanes)
        return 0;

    v_float32 scale = vx_setall_f32(static_cast<float>(_scale));
    v_int16 v_zero = vx_setall_s16(0);

    int x = 0;
    for (;;)
    {
        for (; x <= length - nlanes; x += nlanes)
        {
            v_float32 a1 = vg_load_f32(&in1[x]);
            v_float32 a2 = vg_load_f32(&in1[x + nlanes / 4]);
            v_float32 a3 = vg_load_f32(&in1[x + nlanes / 2]);
            v_float32 a4 = vg_load_f32(&in1[x + 3 * nlanes / 4]);

            v_int16 div1 = v_reinterpret_as_s16(vx_load(&in2[x]));
            v_int16 div2 = v_reinterpret_as_s16(vx_load(&in2[x + nlanes/2]));

            v_float32 fdiv1 = v_cvt_f32(v_expand_low(div1));
            v_float32 fdiv2 = v_cvt_f32(v_expand_high(div1));
            v_float32 fdiv3 = v_cvt_f32(v_expand_low(div2));
            v_float32 fdiv4 = v_cvt_f32(v_expand_high(div2));

            v_int32 sum1 = v_round(div_op(t, a1, fdiv1, scale)),
                    sum2 = v_round(div_op(t, a2, fdiv2, scale)),
                    sum3 = v_round(div_op(t, a3, fdiv3, scale)),
                    sum4 = v_round(div_op(t, a4, fdiv4, scale));

            v_int16 res1 = v_select((div1 == v_zero), v_zero, v_pack(sum1, sum2));
            v_int16 res2 = v_select((div2 == v_zero), v_zero, v_pack(sum3, sum4));

            vx_store(&out[x], v_pack_u(res1, res2));
        }

        if (x < length)
        {
            x = length - nlanes;
            continue;  // process one more time (unaligned tail)
        }
        break;
    }
    return x;
}

//-------------------------------------------------------------------------------------------------

template<typename scale_tag_t>
CV_ALWAYS_INLINE int div_hal(scale_tag_t t, const float in1[], const float in2[], uchar out[],
                             const int length, double _scale)
{
    constexpr int nlanes = v_uint8::nlanes;

    if (length < nlanes)
        return 0;

    v_float32 scale = vx_setall_f32(static_cast<float>(_scale));
    v_float32 v_zero = vx_setall_f32(0);
    int x = 0;
    for (;;)
    {
        for (; x <= length - nlanes; x += nlanes)
        {
            v_float32 a1 = vg_load_f32(&in1[x]);
            v_float32 a2 = vg_load_f32(&in1[x + nlanes / 4]);
            v_float32 a3 = vg_load_f32(&in1[x + nlanes / 2]);
            v_float32 a4 = vg_load_f32(&in1[x + 3 * nlanes / 4]);

            v_float32 div1 = vg_load_f32(&in2[x]);
            v_float32 div2 = vg_load_f32(&in2[x + nlanes / 4]);
            v_float32 div3 = vg_load_f32(&in2[x + nlanes / 2]);
            v_float32 div4 = vg_load_f32(&in2[x + 3 * nlanes / 4]);

            v_float32 r1 = div_op(t, a1, div1, scale);
            v_float32 r2 = div_op(t, a2, div2, scale);
            v_float32 r3 = div_op(t, a3, div3, scale);
            v_float32 r4 = div_op(t, a4, div4, scale);

            v_float32 sel1 = v_select((div1 == v_zero), v_zero, r1);
            v_float32 sel2 = v_select((div2 == v_zero), v_zero, r2);
            v_float32 sel3 = v_select((div3 == v_zero), v_zero, r3);
            v_float32 sel4 = v_select((div4 == v_zero), v_zero, r4);

            v_int32 res1 = v_round(sel1);
            v_int32 res2 = v_round(sel2);
            v_int32 res3 = v_round(sel3);
            v_int32 res4 = v_round(sel4);

            vx_store(&out[x], v_pack_u(v_pack(res1, res2), v_pack(res3, res4)));
        }

        if (x < length)
        {
            x = length - nlanes;
            continue;  // process one more time (unaligned tail)
        }
        break;
    }
    return x;
}

//-------------------------------------------------------------------------------------------------

template<typename scale_tag_t, typename DST>
CV_ALWAYS_INLINE
typename std::enable_if<std::is_same<DST, short>::value ||
                        std::is_same<DST, ushort>::value, int>::type
div_hal(scale_tag_t t, const uchar in1[], const uchar in2[], DST out[], const int length, double _scale)
{
    constexpr int nlanes = vector_type_of_t<DST>::nlanes;

    if (length < nlanes)
        return 0;

    v_float32 scale = vx_setall_f32(static_cast<float>(_scale));
    v_int16 v_zero = vx_setall_s16(0);

    int x = 0;
    for (;;)
    {
        for (; x <= length - nlanes; x += nlanes)
        {
            v_float32 a1 = vg_load_f32(&in1[x]);
            v_float32 a2 = vg_load_f32(&in1[x + nlanes / 2]);

            v_int16 div = v_reinterpret_as_s16(vx_load_expand(&in2[x]));

            v_float32 fdiv1 = v_cvt_f32(v_expand_low(div));
            v_float32 fdiv2 = v_cvt_f32(v_expand_high(div));

            v_int32 r1 = v_round(div_op(t, a1, fdiv1, scale));
            v_int32 r2 = v_round(div_op(t, a2, fdiv2, scale));

            v_store_select(&out[x], div, v_zero, r1, r2);
        }

        if (x < length)
        {
            x = length - nlanes;
            continue;  // process one more time (unaligned tail)
        }
        break;
    }
    return x;
}

//-------------------------------------------------------------------------------------------------

template<typename scale_tag_t, typename DST>
CV_ALWAYS_INLINE
typename std::enable_if<std::is_same<DST, short>::value ||
                        std::is_same<DST, ushort>::value, int>::type
div_hal(scale_tag_t t, const float in1[], const float in2[], DST out[], const int length, double _scale)
{
    constexpr int nlanes = vector_type_of_t<DST>::nlanes;

    if (length < nlanes)
        return 0;

    v_float32 scale = vx_setall_f32(static_cast<float>(_scale));
    v_float32 v_zero = vx_setall_f32(0);
    int x = 0;
    for (;;)
    {
        for (; x <= length - nlanes; x += nlanes)
        {
            v_float32 a1 = vg_load_f32(&in1[x]);
            v_float32 a2 = vg_load_f32(&in1[x + nlanes / 2]);

            v_float32 fdiv1 = vg_load_f32(&in2[x]);
            v_float32 fdiv2 = vg_load_f32(&in2[x + nlanes / 2]);

            v_float32 r1 = div_op(t, a1, fdiv1, scale);
            v_float32 r2 = div_op(t, a2, fdiv2, scale);

            v_int32 res1 = v_round(v_select((fdiv1 == v_zero), v_zero, r1));
            v_int32 res2 = v_round(v_select((fdiv2 == v_zero), v_zero, r2));

            v_store_i16(&out[x], res1, res2);
        }

        if (x < length)
        {
            x = length - nlanes;
            continue;  // process one more time (unaligned tail)
        }
        break;
    }
    return x;
}

//-------------------------------------------------------------------------------------------------

template<typename scale_tag_t, typename SRC>
CV_ALWAYS_INLINE int div_hal(scale_tag_t t, const SRC in1[], const SRC in2[], float out[],
                             const int length, double _scale)
{
    constexpr int nlanes = v_float32::nlanes;

    if (length < nlanes)
        return 0;

    v_float32 scale = vx_setall_f32(static_cast<float>(_scale));

    int x = 0;
    for (;;)
    {
        for (; x <= length - nlanes; x += nlanes)
        {
            v_float32 a1 = vg_load_f32(&in1[x]);
            v_float32 b1 = vg_load_f32(&in2[x]);

            vx_store(&out[x], div_op(t, a1, b1, scale));
        }

        if (x < length)
        {
            x = length - nlanes;
            continue;  // process one more time (unaligned tail)
        }
        break;
    }
    return x;
}

//-------------------------------------------------------------------------------------------------

template<typename scale_tag_t>
CV_ALWAYS_INLINE int div_hal(scale_tag_t, const uchar in1[], const uchar in2[], uchar out[],
                             const int length, double scale)
{
    hal::div8u(in1, static_cast<size_t>(length), in2, static_cast<size_t>(length),
               out, static_cast<size_t>(length), length, 1, &scale);
    return length;
}

template<typename scale_tag_t>
CV_ALWAYS_INLINE int div_hal(scale_tag_t, const short in1[], const short in2[], short out[],
                             const int length, double scale)
{
    hal::div16s(in1, static_cast<size_t>(length), in2, static_cast<size_t>(length),
                out, static_cast<size_t>(length), length, 1, &scale);
    return length;
}

//-------------------------------------------------------------------------------------------------

#define DIV_SIMD(SRC, DST)                                                      \
int div_simd(const SRC in1[], const SRC in2[], DST out[],                       \
                              const int length, double _scale)                  \
{                                                                               \
    int x = 0;                                                                  \
    float fscale = static_cast<float>(_scale);                                  \
    if (std::fabs(fscale - 1.0f) <= FLT_EPSILON)                                \
    {                                                                           \
        not_scale_tag t;                                                        \
        x = div_hal(t, in1, in2, out, length, _scale);                          \
    }                                                                           \
    else                                                                        \
    {                                                                           \
        scale_tag t;                                                            \
        x = div_hal(t, in1, in2, out, length, _scale);                          \
    }                                                                           \
    return x;                                                                   \
}

DIV_SIMD(uchar, uchar)
DIV_SIMD(ushort, uchar)
DIV_SIMD(short, uchar)
DIV_SIMD(float, uchar)
DIV_SIMD(short, short)
DIV_SIMD(ushort, short)
DIV_SIMD(uchar, short)
DIV_SIMD(float, short)
DIV_SIMD(ushort, ushort)
DIV_SIMD(uchar, ushort)
DIV_SIMD(short, ushort)
DIV_SIMD(float, ushort)
DIV_SIMD(uchar, float)
DIV_SIMD(ushort, float)
DIV_SIMD(short, float)
DIV_SIMD(float, float)

#undef DIV_SIMD

//-------------------------
//
// Fluid kernels: Multiply
//
//-------------------------

template<typename scale_tag_t, typename SRC, typename DST>
CV_ALWAYS_INLINE
typename std::enable_if<(std::is_same<SRC, short>::value && std::is_same<DST, ushort>::value) ||
                        (std::is_same<SRC, ushort>::value && std::is_same<DST, ushort>::value) ||
                        (std::is_same<SRC, short>::value && std::is_same<DST, short>::value) ||
                        (std::is_same<SRC, ushort>::value && std::is_same<DST, short>::value), int>::type
mul_hal(scale_tag_t t, const SRC in1[], const SRC in2[], DST out[], const int length, double _scale)
{
    constexpr int nlanes = vector_type_of_t<DST>::nlanes;

    if (length < nlanes)
        return 0;

    v_float32 scale = vx_setall_f32(static_cast<float>(_scale));

    int x = 0;
    for (;;)
    {
        for (; x <= length - nlanes; x += nlanes)
        {
            v_int16 a = v_reinterpret_as_s16(vx_load(&in1[x]));
            v_int16 b = v_reinterpret_as_s16(vx_load(&in2[x]));

            v_float32 a1 = v_cvt_f32(v_expand_low(a));
            v_float32 a2 = v_cvt_f32(v_expand_high(a));

            v_float32 b1 = v_cvt_f32(v_expand_low(b));
            v_float32 b2 = v_cvt_f32(v_expand_high(b));

            v_int32 r1 = v_round(mul_op(t, a1, b1, scale));
            v_int32 r2 = v_round(mul_op(t, a2, b2, scale));

            v_store_i16(&out[x], r1, r2);
        }

        if (x < length)
        {
            x = length - nlanes;
            continue;  // process one more time (unaligned tail)
        }
        break;
    }
    return x;
}

//-------------------------------------------------------------------------------------------------

template<typename scale_tag_t, typename SRC>
CV_ALWAYS_INLINE
typename std::enable_if<std::is_same<SRC, short>::value ||
                        std::is_same<SRC, ushort>::value, int>::type
mul_hal(scale_tag_t t, const SRC in1[], const SRC in2[], uchar out[], const int length, double _scale)
{
    constexpr int nlanes = v_uint8::nlanes;

    if (length < nlanes)
        return 0;

    v_float32 scale = vx_setall_f32(static_cast<float>(_scale));

    int x = 0;
    for (;;)
    {
        for (; x <= length - nlanes; x += nlanes)
        {
            v_int16 a1 = v_reinterpret_as_s16(vx_load(&in1[x]));
            v_int16 a2 = v_reinterpret_as_s16(vx_load(&in1[x + nlanes / 2]));

            v_float32 fa1 = v_cvt_f32(v_expand_low(a1));
            v_float32 fa2 = v_cvt_f32(v_expand_high(a1));
            v_float32 fa3 = v_cvt_f32(v_expand_low(a2));
            v_float32 fa4 = v_cvt_f32(v_expand_high(a2));

            v_int16 b1 = v_reinterpret_as_s16(vx_load(&in2[x]));
            v_int16 b2 = v_reinterpret_as_s16(vx_load(&in2[x + nlanes/2]));

            v_float32 fb1 = v_cvt_f32(v_expand_low(b1));
            v_float32 fb2 = v_cvt_f32(v_expand_high(b1));
            v_float32 fb3 = v_cvt_f32(v_expand_low(b2));
            v_float32 fb4 = v_cvt_f32(v_expand_high(b2));

            v_int32 sum1 = v_round(mul_op(t, fa1, fb1, scale)),
                    sum2 = v_round(mul_op(t, fa2, fb2, scale)),
                    sum3 = v_round(mul_op(t, fa3, fb3, scale)),
                    sum4 = v_round(mul_op(t, fa4, fb4, scale));

            v_int16 res1 = v_pack(sum1, sum2);
            v_int16 res2 = v_pack(sum3, sum4);

            vx_store(&out[x], v_pack_u(res1, res2));
        }

        if (x < length)
        {
            x = length - nlanes;
            continue;  // process one more time (unaligned tail)
        }
        break;
    }
    return x;
}

//-------------------------------------------------------------------------------------------------

template<typename scale_tag_t>
CV_ALWAYS_INLINE int mul_hal(scale_tag_t t, const float in1[], const float in2[], uchar out[],
                             const int length, double _scale)
{
    constexpr int nlanes = v_uint8::nlanes;

    if (length < nlanes)
        return 0;

    v_float32 scale = vx_setall_f32(static_cast<float>(_scale));
    int x = 0;
    for (;;)
    {
        for (; x <= length - nlanes; x += nlanes)
        {
            v_float32 a1 = vg_load_f32(&in1[x]);
            v_float32 a2 = vg_load_f32(&in1[x + nlanes / 4]);
            v_float32 a3 = vg_load_f32(&in1[x + nlanes / 2]);
            v_float32 a4 = vg_load_f32(&in1[x + 3 * nlanes / 4]);

            v_float32 b1 = vg_load_f32(&in2[x]);
            v_float32 b2 = vg_load_f32(&in2[x + nlanes / 4]);
            v_float32 b3 = vg_load_f32(&in2[x + nlanes / 2]);
            v_float32 b4 = vg_load_f32(&in2[x + 3 * nlanes / 4]);

            v_int32 res1 = v_round(mul_op(t, a1, b1, scale));
            v_int32 res2 = v_round(mul_op(t, a2, b2, scale));
            v_int32 res3 = v_round(mul_op(t, a3, b3, scale));
            v_int32 res4 = v_round(mul_op(t, a4, b4, scale));

            vx_store(&out[x], v_pack_u(v_pack(res1, res2), v_pack(res3, res4)));
        }

        if (x < length)
        {
            x = length - nlanes;
            continue;  // process one more time (unaligned tail)
        }
        break;
    }
    return x;
}

template<typename scale_tag_t, typename DST>
CV_ALWAYS_INLINE
typename std::enable_if<std::is_same<DST, short>::value ||
                        std::is_same<DST, ushort>::value, int>::type
mul_hal(scale_tag_t t, const uchar in1[], const uchar in2[], DST out[], const int length, double _scale)
{
    constexpr int nlanes = vector_type_of_t<DST>::nlanes;

    if (length < nlanes)
        return 0;

    v_float32 scale = vx_setall_f32(static_cast<float>(_scale));

    int x = 0;
    for (;;)
    {
        for (; x <= length - nlanes; x += nlanes)
        {
            v_int16 a = v_reinterpret_as_s16(vx_load_expand(&in1[x]));
            v_int16 b = v_reinterpret_as_s16(vx_load_expand(&in2[x]));

            v_float32 a1 = v_cvt_f32(v_expand_low(a));
            v_float32 a2 = v_cvt_f32(v_expand_high(a));

            v_float32 b1 = v_cvt_f32(v_expand_low(b));
            v_float32 b2 = v_cvt_f32(v_expand_high(b));

            v_int32 r1 = v_round(mul_op(t, a1, b1, scale));
            v_int32 r2 = v_round(mul_op(t, a2, b2, scale));

            v_store_i16(&out[x], r1, r2);
        }

        if (x < length)
        {
            x = length - nlanes;
            continue;  // process one more time (unaligned tail)
        }
        break;
    }
    return x;
}

//-------------------------------------------------------------------------------------------------

template<typename scale_tag_t, typename DST>
CV_ALWAYS_INLINE
typename std::enable_if<std::is_same<DST, short>::value ||
                        std::is_same<DST, ushort>::value, int>::type
mul_hal(scale_tag_t t, const float in1[], const float in2[], DST out[], const int length, double _scale)
{
    constexpr int nlanes = vector_type_of_t<DST>::nlanes;

    if (length < nlanes)
        return 0;

    v_float32 scale = vx_setall_f32(static_cast<float>(_scale));

    int x = 0;
    for (;;)
    {
        for (; x <= length - nlanes; x += nlanes)
        {
            v_float32 a1 = vg_load_f32(&in1[x]);
            v_float32 a2 = vg_load_f32(&in1[x + nlanes / 2]);

            v_float32 b1 = vg_load_f32(&in2[x]);
            v_float32 b2 = vg_load_f32(&in2[x + nlanes / 2]);

            v_int32 res1 = v_round(mul_op(t, a1, b1, scale));
            v_int32 res2 = v_round(mul_op(t, a2, b2, scale));

            v_store_i16(&out[x], res1, res2);
        }

        if (x < length)
        {
            x = length - nlanes;
            continue;  // process one more time (unaligned tail)
        }
        break;
    }
    return x;
}

//-------------------------------------------------------------------------------------------------

template<typename scale_tag_t, typename SRC>
CV_ALWAYS_INLINE int mul_hal(scale_tag_t t, const SRC in1[], const SRC in2[], float out[],
                             const int length, double _scale)
{
    constexpr int nlanes = v_float32::nlanes;

    if (length < nlanes)
        return 0;

    v_float32 scale = vx_setall_f32(static_cast<float>(_scale));

    int x = 0;
    for (;;)
    {
        for (; x <= length - nlanes; x += nlanes)
        {
            v_float32 a1 = vg_load_f32(&in1[x]);
            v_float32 b1 = vg_load_f32(&in2[x]);

            vx_store(&out[x], mul_op(t, a1, b1, scale));
        }

        if (x < length)
        {
            x = length - nlanes;
            continue;  // process one more time (unaligned tail)
        }
        break;
    }
    return x;
}

//-------------------------------------------------------------------------------------------------

template<typename scale_tag_t>
CV_ALWAYS_INLINE int mul_hal(scale_tag_t, const uchar in1[], const uchar in2[], uchar out[],
                             const int length, double scale)
{
    hal::mul8u(in1, static_cast<size_t>(length), in2, static_cast<size_t>(length),
               out, static_cast<size_t>(length), length, 1, &scale);
    return length;
}

#define MUL_SIMD(SRC, DST)                                                      \
int mul_simd(const SRC in1[], const SRC in2[], DST out[],                       \
             const int length, double _scale)                                   \
{                                                                               \
    int x = 0;                                                                  \
    float fscale = static_cast<float>(_scale);                                  \
    if (std::fabs(fscale - 1.0f) <= FLT_EPSILON)                                \
    {                                                                           \
        not_scale_tag t;                                                        \
        x = mul_hal(t, in1, in2, out, length, _scale);                          \
    }                                                                           \
    else                                                                        \
    {                                                                           \
        scale_tag t;                                                            \
        x = mul_hal(t, in1, in2, out, length, _scale);                          \
    }                                                                           \
    return x;                                                                   \
}

MUL_SIMD(uchar, uchar)
MUL_SIMD(ushort, uchar)
MUL_SIMD(short, uchar)
MUL_SIMD(float, uchar)
MUL_SIMD(short, short)
MUL_SIMD(ushort, short)
MUL_SIMD(uchar, short)
MUL_SIMD(float, short)
MUL_SIMD(ushort, ushort)
MUL_SIMD(uchar, ushort)
MUL_SIMD(short, ushort)
MUL_SIMD(float, ushort)
MUL_SIMD(uchar, float)
MUL_SIMD(ushort, float)
MUL_SIMD(short, float)
MUL_SIMD(float, float)

#undef MUL_SIMD

#endif  // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END

}  // namespace fluid
}  // namespace gapi
}  // namespace cv

#endif // !defined(GAPI_STANDALONE)
