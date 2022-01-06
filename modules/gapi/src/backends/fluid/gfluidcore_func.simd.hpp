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

#define ADDC_SIMD(SRC, DST)                                                              \
int addc_simd(const SRC in[], const float scalar[], DST out[],                           \
              const int length, const int chan);

ADDC_SIMD(uchar, uchar)
ADDC_SIMD(ushort, uchar)
ADDC_SIMD(short, uchar)
ADDC_SIMD(float, uchar)
ADDC_SIMD(short, short)
ADDC_SIMD(ushort, short)
ADDC_SIMD(uchar, short)
ADDC_SIMD(float, short)
ADDC_SIMD(ushort, ushort)
ADDC_SIMD(uchar, ushort)
ADDC_SIMD(short, ushort)
ADDC_SIMD(float, ushort)
ADDC_SIMD(uchar, float)
ADDC_SIMD(ushort, float)
ADDC_SIMD(short, float)
ADDC_SIMD(float, float)

#undef ADDC_SIMD

#define SUBC_SIMD(SRC, DST)                                                              \
int subc_simd(const SRC in[], const float scalar[], DST out[],                           \
              const int length, const int chan);

SUBC_SIMD(uchar, uchar)
SUBC_SIMD(ushort, uchar)
SUBC_SIMD(short, uchar)
SUBC_SIMD(float, uchar)
SUBC_SIMD(short, short)
SUBC_SIMD(ushort, short)
SUBC_SIMD(uchar, short)
SUBC_SIMD(float, short)
SUBC_SIMD(ushort, ushort)
SUBC_SIMD(uchar, ushort)
SUBC_SIMD(short, ushort)
SUBC_SIMD(float, ushort)
SUBC_SIMD(uchar, float)
SUBC_SIMD(ushort, float)
SUBC_SIMD(short, float)
SUBC_SIMD(float, float)

#undef SUBC_SIMD

#define SUBRC_SIMD(SRC, DST)                                                              \
int subrc_simd(const float scalar[], const SRC in[], DST out[],                           \
               const int length, const int chan);

SUBRC_SIMD(uchar, uchar)
SUBRC_SIMD(ushort, uchar)
SUBRC_SIMD(short, uchar)
SUBRC_SIMD(float, uchar)
SUBRC_SIMD(short, short)
SUBRC_SIMD(ushort, short)
SUBRC_SIMD(uchar, short)
SUBRC_SIMD(float, short)
SUBRC_SIMD(ushort, ushort)
SUBRC_SIMD(uchar, ushort)
SUBRC_SIMD(short, ushort)
SUBRC_SIMD(float, ushort)
SUBRC_SIMD(uchar, float)
SUBRC_SIMD(ushort, float)
SUBRC_SIMD(short, float)
SUBRC_SIMD(float, float)

#undef SUBRC_SIMD

#define MULC_SIMD(SRC, DST)                                                              \
int mulc_simd(const SRC in[], const float scalar[], DST out[],                           \
              const int length, const int chan, const float scale);

MULC_SIMD(uchar, uchar)
MULC_SIMD(ushort, uchar)
MULC_SIMD(short, uchar)
MULC_SIMD(float, uchar)
MULC_SIMD(short, short)
MULC_SIMD(ushort, short)
MULC_SIMD(uchar, short)
MULC_SIMD(float, short)
MULC_SIMD(ushort, ushort)
MULC_SIMD(uchar, ushort)
MULC_SIMD(short, ushort)
MULC_SIMD(float, ushort)
MULC_SIMD(uchar, float)
MULC_SIMD(ushort, float)
MULC_SIMD(short, float)
MULC_SIMD(float, float)

#undef MULC_SIMD

#define ABSDIFFC_SIMD(T)                                            \
int absdiffc_simd(const T in[], const float scalar[], T out[],      \
                  const int length, const int chan);

ABSDIFFC_SIMD(uchar)
ABSDIFFC_SIMD(short)
ABSDIFFC_SIMD(ushort)
ABSDIFFC_SIMD(float)

#undef ABSDIFFC_SIMD

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
template<> struct vector_type_of<float> { using type = v_float32; };

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

CV_ALWAYS_INLINE void v_store_i16(short* dst, const v_int32& res1, const v_int32& res2)
{
    vx_store(dst, v_pack(res1, res2));
}

CV_ALWAYS_INLINE void v_store_i16(ushort* dst, const v_int32& res1, const v_int32& res2)
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

//-------------------------
//
// Fluid kernels: AddC, SubC, SubRC
//
//-------------------------

struct add_tag {};
struct sub_tag {};
struct subr_tag {};
struct mul_tag {};
struct absdiff_tag {};

CV_ALWAYS_INLINE void arithmOpScalar_pack_store_c3(short* outx,       const v_int32& c1,
                                                   const v_int32& c2, const v_int32& c3,
                                                   const v_int32& c4, const v_int32& c5,
                                                   const v_int32& c6)
{
    constexpr int nlanes = v_int16::nlanes;
    vx_store(outx,           v_pack(c1, c2));
    vx_store(&outx[nlanes],   v_pack(c3, c4));
    vx_store(&outx[2*nlanes], v_pack(c5, c6));
}

CV_ALWAYS_INLINE void arithmOpScalar_pack_store_c3(ushort* outx,      const v_int32& c1,
                                                   const v_int32& c2, const v_int32& c3,
                                                   const v_int32& c4, const v_int32& c5,
                                                   const v_int32& c6)
{
    constexpr int nlanes = v_uint16::nlanes;
    vx_store(outx,            v_pack_u(c1, c2));
    vx_store(&outx[nlanes],   v_pack_u(c3, c4));
    vx_store(&outx[2*nlanes], v_pack_u(c5, c6));
}

CV_ALWAYS_INLINE v_float32 oper(add_tag, const v_float32& a, const v_float32& sc)
{
    return a + sc;
}

CV_ALWAYS_INLINE v_float32 oper(sub_tag, const v_float32& a, const v_float32& sc)
{
    return a - sc;
}

CV_ALWAYS_INLINE v_float32 oper(subr_tag, const v_float32& a, const v_float32& sc)
{
    return sc - a;
}

CV_ALWAYS_INLINE v_float32 oper(mul_tag, const v_float32& a, const v_float32& sc)
{
    return a * sc;
}

CV_ALWAYS_INLINE v_float32 oper(absdiff_tag, const v_float32& a, const v_float32& sc)
{
    return v_absdiff(a, sc);
}

//-------------------------------------------------------------------------------------------------

template<typename oper_tag, typename SRC, typename DST>
CV_ALWAYS_INLINE
typename std::enable_if<(std::is_same<DST, ushort>::value ||
                         std::is_same<DST, short>::value), void>::type
arithmOpScalar_simd_common_impl(oper_tag t, const SRC* inx, DST* outx,
                                const v_float32& sc, const int nlanes)
{
    v_float32 a1 = vg_load_f32(inx);
    v_float32 a2 = vg_load_f32(&inx[nlanes/2]);

    v_store_i16(outx, v_round(oper(t, a1, sc)), v_round(oper(t, a2, sc)));
}

//-------------------------------------------------------------------------------------------------

template<typename oper_tag, typename SRC>
CV_ALWAYS_INLINE void arithmOpScalar_simd_common_impl(oper_tag t, const SRC* inx,
                                                      uchar* outx, const v_float32& sc,
                                                      const int nlanes)
{
    v_float32 a1 = vg_load_f32(inx);
    v_float32 a2 = vg_load_f32(&inx[nlanes/4]);
    v_float32 a3 = vg_load_f32(&inx[nlanes/2]);
    v_float32 a4 = vg_load_f32(&inx[3 * nlanes/4]);

    vx_store(outx, v_pack_u(v_pack(v_round(oper(t, a1, sc)),
                                   v_round(oper(t, a2, sc))),
                            v_pack(v_round(oper(t, a3, sc)),
                                   v_round(oper(t, a4, sc)))));
}

//-------------------------------------------------------------------------------------------------

template<typename oper_tag, typename SRC>
CV_ALWAYS_INLINE void arithmOpScalar_simd_common_impl(oper_tag t, const SRC* inx,
                                                      float* outx, const v_float32& sc, const int)
{
    v_float32 a1 = vg_load_f32(inx);
    vx_store(outx, oper(t, a1, sc));
}

//-------------------------------------------------------------------------------------------------

template<typename oper_tag, typename SRC, typename DST>
CV_ALWAYS_INLINE
typename std::enable_if<std::is_same<DST, short>::value ||
                        std::is_same<DST, ushort>::value, void>::type
arithmOpScalar_simd_c3_impl(oper_tag t, const SRC* inx, DST* outx, const v_float32& s1, const v_float32& s2,
                            const v_float32& s3, const int nlanes)
{
    v_float32 a1 = vg_load_f32(inx);
    v_float32 a2 = vg_load_f32(&inx[nlanes / 2]);
    v_float32 a3 = vg_load_f32(&inx[nlanes]);
    v_float32 a4 = vg_load_f32(&inx[3 * nlanes / 2]);
    v_float32 a5 = vg_load_f32(&inx[2 * nlanes]);
    v_float32 a6 = vg_load_f32(&inx[5 * nlanes / 2]);

    arithmOpScalar_pack_store_c3(outx, v_round(oper(t, a1, s1)),
                                       v_round(oper(t, a2, s2)),
                                       v_round(oper(t, a3, s3)),
                                       v_round(oper(t, a4, s1)),
                                       v_round(oper(t, a5, s2)),
                                       v_round(oper(t, a6, s3)));
}

//-------------------------------------------------------------------------------------------------

template<typename oper_tag, typename SRC>
CV_ALWAYS_INLINE void arithmOpScalar_simd_c3_impl(oper_tag t, const SRC* inx, uchar* outx,
                                                  const v_float32& s1, const v_float32& s2,
                                                  const v_float32& s3, const int nlanes)
{
    vx_store(outx,
               v_pack_u(v_pack(v_round(oper(t, vg_load_f32(inx), s1)),
                               v_round(oper(t, vg_load_f32(&inx[nlanes/4]), s2))),
                        v_pack(v_round(oper(t, vg_load_f32(&inx[nlanes/2]), s3)),
                               v_round(oper(t, vg_load_f32(&inx[3*nlanes/4]), s1)))));

    vx_store(&outx[nlanes],
                v_pack_u(v_pack(v_round(oper(t, vg_load_f32(&inx[nlanes]), s2)),
                                v_round(oper(t, vg_load_f32(&inx[5*nlanes/4]), s3))),
                         v_pack(v_round(oper(t, vg_load_f32(&inx[3*nlanes/2]), s1)),
                                v_round(oper(t, vg_load_f32(&inx[7*nlanes/4]), s2)))));

    vx_store(&outx[2 * nlanes],
                v_pack_u(v_pack(v_round(oper(t, vg_load_f32(&inx[2*nlanes]), s3)),
                                v_round(oper(t, vg_load_f32(&inx[9*nlanes/4]), s1))),
                         v_pack(v_round(oper(t, vg_load_f32(&inx[5*nlanes/2]), s2)),
                                v_round(oper(t, vg_load_f32(&inx[11*nlanes/4]), s3)))));
}

//-------------------------------------------------------------------------------------------------

template<typename oper_tag, typename SRC>
CV_ALWAYS_INLINE void arithmOpScalar_simd_c3_impl(oper_tag t, const SRC* in, float* out,
                                                  const v_float32& s1, const v_float32& s2,
                                                  const v_float32& s3, const int nlanes)
{
    v_float32 a1 = vg_load_f32(in);
    v_float32 a2 = vg_load_f32(&in[nlanes]);
    v_float32 a3 = vg_load_f32(&in[2*nlanes]);

    vx_store(out, oper(t, a1, s1));
    vx_store(&out[nlanes], oper(t, a2, s2));
    vx_store(&out[2*nlanes], oper(t, a3, s3));
}

//-------------------------------------------------------------------------------------------------

template<typename oper_tag, typename SRC, typename DST>
CV_ALWAYS_INLINE int arithmOpScalar_simd_c3(oper_tag t, const SRC in[],
                                            const float scalar[], DST out[],
                                            const int length)
{
    constexpr int chan = 3;
    constexpr int nlanes = vector_type_of_t<DST>::nlanes;
    constexpr int lanes = chan * nlanes;

    if (length < lanes)
        return 0;

    v_float32 s1 = vx_load(scalar);
#if CV_SIMD_WIDTH == 32
    v_float32 s2 = vx_load(&scalar[2]);
    v_float32 s3 = vx_load(&scalar[1]);
#else
    v_float32 s2 = vx_load(&scalar[1]);
    v_float32 s3 = vx_load(&scalar[2]);
#endif

    int x = 0;
    for (;;)
    {
        for (; x <= length - lanes; x += lanes)
        {
            arithmOpScalar_simd_c3_impl(t, &in[x], &out[x], s1, s2, s3, nlanes);
        }

        if (x < length)
        {
            x = length - lanes;
            continue;  // process unaligned tail
        }
        break;
    }
    return x;
}

//-------------------------------------------------------------------------------------------------

template<typename oper_tag, typename SRC, typename DST>
CV_ALWAYS_INLINE int arithmOpScalar_simd_common(oper_tag t, const SRC in[],
                                                const float scalar[], DST out[],
                                                const int length)
{
    constexpr int nlanes = vector_type_of_t<DST>::nlanes;

    if (length < nlanes)
        return 0;

    v_float32 sc = vx_load(scalar);

    int x = 0;
    for (;;)
    {
        for (; x <= length - nlanes; x += nlanes)
        {
            arithmOpScalar_simd_common_impl(t, &in[x], &out[x], sc, nlanes);
        }

        if (x < length)
        {
            x = length - nlanes;
            continue;  // process unaligned tail
        }
        break;
    }
    return x;
}

//-------------------------------------------------------------------------------------------------

#define ADDC_SIMD(SRC, DST)                                                         \
int addc_simd(const SRC in[], const float scalar[], DST out[],                      \
              const int length, const int chan)                                     \
{                                                                                   \
    switch (chan)                                                                   \
    {                                                                               \
    case 1:                                                                         \
    case 2:                                                                         \
    case 4:                                                                         \
        return arithmOpScalar_simd_common(add_tag{}, in, scalar, out, length);      \
    case 3:                                                                         \
        return arithmOpScalar_simd_c3(add_tag{}, in, scalar, out, length);          \
    default:                                                                        \
        GAPI_Assert(chan <= 4);                                                     \
        break;                                                                      \
    }                                                                               \
    return 0;                                                                       \
}

ADDC_SIMD(uchar, uchar)
ADDC_SIMD(ushort, uchar)
ADDC_SIMD(short, uchar)
ADDC_SIMD(float, uchar)
ADDC_SIMD(short, short)
ADDC_SIMD(ushort, short)
ADDC_SIMD(uchar, short)
ADDC_SIMD(float, short)
ADDC_SIMD(ushort, ushort)
ADDC_SIMD(uchar, ushort)
ADDC_SIMD(short, ushort)
ADDC_SIMD(float, ushort)
ADDC_SIMD(uchar, float)
ADDC_SIMD(ushort, float)
ADDC_SIMD(short, float)
ADDC_SIMD(float, float)

#undef ADDC_SIMD

//-------------------------------------------------------------------------------------------------

#define SUBC_SIMD(SRC, DST)                                                         \
int subc_simd(const SRC in[], const float scalar[], DST out[],                      \
              const int length, const int chan)                                     \
{                                                                                   \
    switch (chan)                                                                   \
    {                                                                               \
    case 1:                                                                         \
    case 2:                                                                         \
    case 4:                                                                         \
        return arithmOpScalar_simd_common(sub_tag{}, in, scalar, out, length);      \
    case 3:                                                                         \
        return arithmOpScalar_simd_c3(sub_tag{}, in, scalar, out, length);          \
    default:                                                                        \
        GAPI_Assert(chan <= 4);                                                     \
        break;                                                                      \
    }                                                                               \
    return 0;                                                                       \
}

SUBC_SIMD(uchar, uchar)
SUBC_SIMD(ushort, uchar)
SUBC_SIMD(short, uchar)
SUBC_SIMD(float, uchar)
SUBC_SIMD(short, short)
SUBC_SIMD(ushort, short)
SUBC_SIMD(uchar, short)
SUBC_SIMD(float, short)
SUBC_SIMD(ushort, ushort)
SUBC_SIMD(uchar, ushort)
SUBC_SIMD(short, ushort)
SUBC_SIMD(float, ushort)
SUBC_SIMD(uchar, float)
SUBC_SIMD(ushort, float)
SUBC_SIMD(short, float)
SUBC_SIMD(float, float)

#undef SUBC_SIMD

//-------------------------------------------------------------------------------------------------

#define SUBRC_SIMD(SRC, DST)                                                        \
int subrc_simd(const float scalar[], const SRC in[], DST out[],                     \
               const int length, const int chan)                                    \
{                                                                                   \
    switch (chan)                                                                   \
    {                                                                               \
    case 1:                                                                         \
    case 2:                                                                         \
    case 4:                                                                         \
        return arithmOpScalar_simd_common(subr_tag{}, in, scalar, out, length);     \
    case 3:                                                                         \
        return arithmOpScalar_simd_c3(subr_tag{}, in, scalar, out, length);         \
    default:                                                                        \
        GAPI_Assert(chan <= 4);                                                     \
        break;                                                                      \
    }                                                                               \
    return 0;                                                                       \
}

SUBRC_SIMD(uchar, uchar)
SUBRC_SIMD(ushort, uchar)
SUBRC_SIMD(short, uchar)
SUBRC_SIMD(float, uchar)
SUBRC_SIMD(short, short)
SUBRC_SIMD(ushort, short)
SUBRC_SIMD(uchar, short)
SUBRC_SIMD(float, short)
SUBRC_SIMD(ushort, ushort)
SUBRC_SIMD(uchar, ushort)
SUBRC_SIMD(short, ushort)
SUBRC_SIMD(float, ushort)
SUBRC_SIMD(uchar, float)
SUBRC_SIMD(ushort, float)
SUBRC_SIMD(short, float)
SUBRC_SIMD(float, float)

#undef SUBRC_SIMD

//-------------------------
//
// Fluid kernels: MulC
//
//-------------------------

template<typename SRC, typename DST>
CV_ALWAYS_INLINE
typename std::enable_if<std::is_same<DST, short>::value ||
                        std::is_same<DST, ushort>::value, void>::type
mulc_scale_simd_c3_impl(const SRC* inx, DST* outx, const v_float32& s1, const v_float32& s2,
                        const v_float32& s3, const v_float32& scale, const int nlanes)
{
    v_float32 a1 = vg_load_f32(inx);
    v_float32 a2 = vg_load_f32(&inx[nlanes / 2]);
    v_float32 a3 = vg_load_f32(&inx[nlanes]);
    v_float32 a4 = vg_load_f32(&inx[3 * nlanes / 2]);
    v_float32 a5 = vg_load_f32(&inx[2 * nlanes]);
    v_float32 a6 = vg_load_f32(&inx[5 * nlanes / 2]);

    arithmOpScalar_pack_store_c3(outx, v_round(scale*a1*s1),
                                       v_round(scale*a2*s2),
                                       v_round(scale*a3*s3),
                                       v_round(scale*a4*s1),
                                       v_round(scale*a5*s2),
                                       v_round(scale*a6*s3));
}

//-------------------------------------------------------------------------------------------------

template<typename SRC>
CV_ALWAYS_INLINE void mulc_scale_simd_c3_impl(const SRC* inx, uchar* outx,
                                              const v_float32& s1, const v_float32& s2,
                                              const v_float32& s3, const v_float32& scale, const int nlanes)
{
    vx_store(outx,
               v_pack_u(v_pack(v_round(scale * vg_load_f32(inx)* s1),
                               v_round(scale * vg_load_f32(&inx[nlanes/4])* s2)),
                        v_pack(v_round(scale * vg_load_f32(&inx[nlanes/2])* s3),
                               v_round(scale * vg_load_f32(&inx[3*nlanes/4])* s1))));

    vx_store(&outx[nlanes],
                v_pack_u(v_pack(v_round(scale * vg_load_f32(&inx[nlanes])* s2),
                                v_round(scale * vg_load_f32(&inx[5*nlanes/4])* s3)),
                         v_pack(v_round(scale * vg_load_f32(&inx[3*nlanes/2])* s1),
                                v_round(scale * vg_load_f32(&inx[7*nlanes/4])* s2))));

    vx_store(&outx[2 * nlanes],
                v_pack_u(v_pack(v_round(scale * vg_load_f32(&inx[2*nlanes])* s3),
                                v_round(scale * vg_load_f32(&inx[9*nlanes/4])* s1)),
                         v_pack(v_round(scale * vg_load_f32(&inx[5*nlanes/2])* s2),
                                v_round(scale * vg_load_f32(&inx[11*nlanes/4])* s3))));
}

//-------------------------------------------------------------------------------------------------

template<typename SRC>
CV_ALWAYS_INLINE void mulc_scale_simd_c3_impl(const SRC* in, float* out,
                                        const v_float32& s1, const v_float32& s2,
                                        const v_float32& s3, const v_float32& scale, const int nlanes)
{
    v_float32 a1 = vg_load_f32(in);
    v_float32 a2 = vg_load_f32(&in[nlanes]);
    v_float32 a3 = vg_load_f32(&in[2*nlanes]);

    vx_store(out, scale * a1* s1);
    vx_store(&out[nlanes], scale * a2* s2);
    vx_store(&out[2*nlanes], scale * a3* s3);
}

//-------------------------------------------------------------------------------------------------

template<typename SRC, typename DST>
CV_ALWAYS_INLINE int mulc_scale_simd_c3(const SRC in[],
                                        const float scalar[], DST out[],
                                        const int length, const float _scale)
{
    constexpr int chan = 3;
    constexpr int nlanes = vector_type_of_t<DST>::nlanes;
    constexpr int lanes = chan * nlanes;

    if (length < lanes)
        return 0;

    v_float32 scale = vx_setall_f32(_scale);

    v_float32 s1 = vx_load(scalar);
#if CV_SIMD_WIDTH == 32
    v_float32 s2 = vx_load(&scalar[2]);
    v_float32 s3 = vx_load(&scalar[1]);
#else
    v_float32 s2 = vx_load(&scalar[1]);
    v_float32 s3 = vx_load(&scalar[2]);
#endif

    int x = 0;
    for (;;)
    {
        for (; x <= length - lanes; x += lanes)
        {
            mulc_scale_simd_c3_impl(&in[x], &out[x], s1, s2, s3, scale, nlanes);
        }

        if (x < length)
        {
            x = length - lanes;
            continue;  // process unaligned tail
        }
        break;
    }
    return x;
}

//-------------------------------------------------------------------------------------------------

template<typename SRC, typename DST>
CV_ALWAYS_INLINE
typename std::enable_if<(std::is_same<DST, ushort>::value ||
                         std::is_same<DST, short>::value), void>::type
mulc_scale_simd_common_impl(const SRC* inx, DST* outx,
                            const v_float32& sc, const v_float32& scale,
                            const int nlanes)
{
    v_float32 a1 = vg_load_f32(inx);
    v_float32 a2 = vg_load_f32(&inx[nlanes/2]);

    v_store_i16(outx, v_round(scale * a1* sc), v_round(scale * a2* sc));
}

//-------------------------------------------------------------------------------------------------

template<typename SRC>
CV_ALWAYS_INLINE void mulc_scale_simd_common_impl(const SRC* inx,
                                                  uchar* outx, const v_float32& sc,
                                                  const v_float32& scale, const int nlanes)
{
    v_float32 a1 = vg_load_f32(inx);
    v_float32 a2 = vg_load_f32(&inx[nlanes/4]);
    v_float32 a3 = vg_load_f32(&inx[nlanes/2]);
    v_float32 a4 = vg_load_f32(&inx[3 * nlanes/4]);

    vx_store(outx, v_pack_u(v_pack(v_round(scale * a1* sc),
                                   v_round(scale * a2* sc)),
                            v_pack(v_round(scale * a3* sc),
                                   v_round(scale * a4* sc))));
}

//-------------------------------------------------------------------------------------------------

template<typename SRC>
CV_ALWAYS_INLINE void mulc_scale_simd_common_impl(const SRC* inx,
                                                  float* outx, const v_float32& sc,
                                                  const v_float32& scale, const int)
{
    v_float32 a1 = vg_load_f32(inx);
    vx_store(outx, scale * a1* sc);
}

//-------------------------------------------------------------------------------------------------

template<typename SRC, typename DST>
CV_ALWAYS_INLINE int mulc_scale_simd_common(const SRC in[],
                                            const float scalar[], DST out[],
                                            const int length, const float _scale)
{
    constexpr int nlanes = vector_type_of_t<DST>::nlanes;

    if (length < nlanes)
        return 0;

    v_float32 _scalar = vx_load(scalar);
    v_float32 scale = vx_setall_f32(_scale);

    int x = 0;
    for (;;)
    {
        for (; x <= length - nlanes; x += nlanes)
        {
            mulc_scale_simd_common_impl(&in[x], &out[x], _scalar, scale, nlanes);
        }

        if (x < length)
        {
            x = length - nlanes;
            continue;  // process unaligned tail
        }
        break;
    }
    return x;
}

#define MULC_SIMD(SRC, DST)                                                    \
int mulc_simd(const SRC in[], const float scalar[], DST out[],                 \
              const int length, const int chan, const float scale)             \
{                                                                              \
    mul_tag op_t;                                                              \
    switch (chan)                                                              \
    {                                                                          \
    case 1:                                                                    \
    case 2:                                                                    \
    case 4:                                                                    \
    {                                                                          \
        if (std::fabs(scale - 1.0f) <= FLT_EPSILON)                            \
        {                                                                      \
            return arithmOpScalar_simd_common(op_t, in, scalar,                \
                                              out, length);                    \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            return mulc_scale_simd_common(in, scalar, out, length, scale);     \
        }                                                                      \
    }                                                                          \
    case 3:                                                                    \
    {                                                                          \
        if (std::fabs(scale - 1.0f) <= FLT_EPSILON)                            \
        {                                                                      \
            return arithmOpScalar_simd_c3(op_t, in, scalar,                    \
                                          out, length);                        \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            return mulc_scale_simd_c3(in, scalar, out, length, scale);         \
        }                                                                      \
    }                                                                          \
    default:                                                                   \
        GAPI_Assert(chan <= 4);                                                \
        break;                                                                 \
    }                                                                          \
    return 0;                                                                  \
}

MULC_SIMD(uchar, uchar)
MULC_SIMD(ushort, uchar)
MULC_SIMD(short, uchar)
MULC_SIMD(float, uchar)
MULC_SIMD(short, short)
MULC_SIMD(ushort, short)
MULC_SIMD(uchar, short)
MULC_SIMD(float, short)
MULC_SIMD(ushort, ushort)
MULC_SIMD(uchar, ushort)
MULC_SIMD(short, ushort)
MULC_SIMD(float, ushort)
MULC_SIMD(uchar, float)
MULC_SIMD(ushort, float)
MULC_SIMD(short, float)
MULC_SIMD(float, float)

#undef MULC_SIMD

//-------------------------
//
// Fluid kernels: AbsDiffC
//
//-------------------------

#define ABSDIFFC_SIMD(SRC)                                                          \
int absdiffc_simd(const SRC in[], const float scalar[], SRC out[],                  \
              const int length, const int chan)                                     \
{                                                                                   \
    switch (chan)                                                                   \
    {                                                                               \
    case 1:                                                                         \
    case 2:                                                                         \
    case 4:                                                                         \
        return arithmOpScalar_simd_common(absdiff_tag{}, in, scalar, out, length);  \
    case 3:                                                                         \
        return arithmOpScalar_simd_c3(absdiff_tag{}, in, scalar, out, length);      \
    default:                                                                        \
        GAPI_Assert(chan <= 4);                                                     \
        break;                                                                      \
    }                                                                               \
    return 0;                                                                       \
}

ABSDIFFC_SIMD(uchar)
ABSDIFFC_SIMD(short)
ABSDIFFC_SIMD(ushort)
ABSDIFFC_SIMD(float)

#undef ABSDIFFC_SIMD

#endif  // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END

}  // namespace fluid
}  // namespace gapi
}  // namespace cv

#endif // !defined(GAPI_STANDALONE)
