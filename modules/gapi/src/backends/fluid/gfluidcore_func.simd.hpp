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

#define MULC_SIMD(SRC, DST)                                               \
int mulc_simd(const SRC in[], const float scalar[], DST out[],            \
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

#define DIVC_SIMD(SRC, DST)                                              \
int divc_simd(const SRC in[], const float scalar[], DST out[],           \
              const int length, const int chan, const float scale,       \
              const int set_mask_flag);

DIVC_SIMD(uchar, uchar)
DIVC_SIMD(ushort, uchar)
DIVC_SIMD(short, uchar)
DIVC_SIMD(float, uchar)
DIVC_SIMD(short, short)
DIVC_SIMD(ushort, short)
DIVC_SIMD(uchar, short)
DIVC_SIMD(float, short)
DIVC_SIMD(ushort, ushort)
DIVC_SIMD(uchar, ushort)
DIVC_SIMD(short, ushort)
DIVC_SIMD(float, ushort)
DIVC_SIMD(uchar, float)
DIVC_SIMD(ushort, float)
DIVC_SIMD(short, float)
DIVC_SIMD(float, float)

#undef DIVC_SIMD

#define ABSDIFFC_SIMD(T)                                            \
int absdiffc_simd(const T in[], const float scalar[], T out[],      \
                  const int length, const int chan);

ABSDIFFC_SIMD(uchar)
ABSDIFFC_SIMD(short)
ABSDIFFC_SIMD(ushort)
ABSDIFFC_SIMD(float)

#undef ABSDIFFC_SIMD

#define DIVRC_SIMD(SRC, DST)                                           \
int divrc_simd(const float scalar[], const SRC in[], DST out[],        \
               const int length, const int chan, const float scale);

DIVRC_SIMD(uchar, uchar)
DIVRC_SIMD(ushort, uchar)
DIVRC_SIMD(short, uchar)
DIVRC_SIMD(float, uchar)
DIVRC_SIMD(short, short)
DIVRC_SIMD(ushort, short)
DIVRC_SIMD(uchar, short)
DIVRC_SIMD(float, short)
DIVRC_SIMD(ushort, ushort)
DIVRC_SIMD(uchar, ushort)
DIVRC_SIMD(short, ushort)
DIVRC_SIMD(float, ushort)
DIVRC_SIMD(uchar, float)
DIVRC_SIMD(ushort, float)
DIVRC_SIMD(short, float)
DIVRC_SIMD(float, float)

#undef DIVRC_SIMD

#define ADD_SIMD(SRC, DST)                                                      \
int add_simd(const SRC in1[], const SRC in2[], DST out[], const int length);

ADD_SIMD(uchar, uchar)
ADD_SIMD(ushort, uchar)
ADD_SIMD(short, uchar)
ADD_SIMD(float, uchar)
ADD_SIMD(short, short)
ADD_SIMD(ushort, short)
ADD_SIMD(uchar, short)
ADD_SIMD(float, short)
ADD_SIMD(ushort, ushort)
ADD_SIMD(uchar, ushort)
ADD_SIMD(short, ushort)
ADD_SIMD(float, ushort)
ADD_SIMD(uchar, float)
ADD_SIMD(ushort, float)
ADD_SIMD(short, float)
ADD_SIMD(float, float)

#undef ADD_SIMD

#define SUB_SIMD(SRC, DST)                                                      \
int sub_simd(const SRC in1[], const SRC in2[], DST out[], const int length);

SUB_SIMD(uchar, uchar)
SUB_SIMD(ushort, uchar)
SUB_SIMD(short, uchar)
SUB_SIMD(float, uchar)
SUB_SIMD(short, short)
SUB_SIMD(ushort, short)
SUB_SIMD(uchar, short)
SUB_SIMD(float, short)
SUB_SIMD(ushort, ushort)
SUB_SIMD(uchar, ushort)
SUB_SIMD(short, ushort)
SUB_SIMD(float, ushort)
SUB_SIMD(uchar, float)
SUB_SIMD(ushort, float)
SUB_SIMD(short, float)
SUB_SIMD(float, float)

#undef SUB_SIMD

#define CONVERTTO_NOCOEF_SIMD(SRC, DST)                                             \
int convertto_simd(const SRC in[], DST out[], const int length);

CONVERTTO_NOCOEF_SIMD(ushort, uchar)
CONVERTTO_NOCOEF_SIMD(short, uchar)
CONVERTTO_NOCOEF_SIMD(float, uchar)
CONVERTTO_NOCOEF_SIMD(ushort, short)
CONVERTTO_NOCOEF_SIMD(uchar, short)
CONVERTTO_NOCOEF_SIMD(float, short)
CONVERTTO_NOCOEF_SIMD(uchar, ushort)
CONVERTTO_NOCOEF_SIMD(short, ushort)
CONVERTTO_NOCOEF_SIMD(float, ushort)
CONVERTTO_NOCOEF_SIMD(uchar, float)
CONVERTTO_NOCOEF_SIMD(ushort, float)
CONVERTTO_NOCOEF_SIMD(short, float)

#undef CONVERTTO_NOCOEF_SIMD

#define CONVERTTO_SCALED_SIMD(SRC, DST)                                     \
int convertto_scaled_simd(const SRC in[], DST out[], const float alpha,     \
                          const float beta, const int length);

CONVERTTO_SCALED_SIMD(uchar, uchar)
CONVERTTO_SCALED_SIMD(ushort, uchar)
CONVERTTO_SCALED_SIMD(short, uchar)
CONVERTTO_SCALED_SIMD(float, uchar)
CONVERTTO_SCALED_SIMD(short, short)
CONVERTTO_SCALED_SIMD(ushort, short)
CONVERTTO_SCALED_SIMD(uchar, short)
CONVERTTO_SCALED_SIMD(float, short)
CONVERTTO_SCALED_SIMD(ushort, ushort)
CONVERTTO_SCALED_SIMD(uchar, ushort)
CONVERTTO_SCALED_SIMD(short, ushort)
CONVERTTO_SCALED_SIMD(float, ushort)
CONVERTTO_SCALED_SIMD(uchar, float)
CONVERTTO_SCALED_SIMD(ushort, float)
CONVERTTO_SCALED_SIMD(short, float)
CONVERTTO_SCALED_SIMD(float, float)

#undef CONVERTTO_SCALED_SIMD

int split3_simd(const uchar in[], uchar out1[], uchar out2[],
                uchar out3[], const int width);

int split4_simd(const uchar in[], uchar out1[], uchar out2[],
                uchar out3[], uchar out4[], const int width);

#define MERGE3_SIMD(T)                                          \
int merge3_simd(const T in1[], const T in2[], const T in3[],    \
                T out[], const int width);

MERGE3_SIMD(uchar)
MERGE3_SIMD(short)
MERGE3_SIMD(ushort)
MERGE3_SIMD(float)

#undef MERGE3_SIMD

int merge4_simd(const uchar in1[], const uchar in2[], const uchar in3[],
                const uchar in4[], uchar out[], const int width);


#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

#define SRC_SHORT_OR_USHORT std::is_same<SRC, short>::value || std::is_same<SRC, ushort>::value
#define DST_SHORT_OR_USHORT std::is_same<DST, short>::value || std::is_same<DST, ushort>::value
#define SRC_DST_SHORT_AND_USHORT (std::is_same<SRC, short>::value && std::is_same<DST, ushort>::value) || (std::is_same<SRC, ushort>::value && std::is_same<DST, short>::value)
#define SRC_DST_SHORT_OR_USHORT (std::is_same<SRC, short>::value && std::is_same<DST, short>::value) || (std::is_same<SRC, ushort>::value && std::is_same<DST, ushort>::value)

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

template<typename scalar_t>
struct zero_vec_type_of;

template<typename scalar_t>
using zero_vec_type_of_t = typename zero_vec_type_of<scalar_t>::type;

template<> struct zero_vec_type_of<uchar> { using type = v_int16; };
template<> struct zero_vec_type_of<ushort> { using type = v_int16; };
template<> struct zero_vec_type_of<short> { using type = v_int16; };
template<> struct zero_vec_type_of<float> { using type = v_float32; };

template<typename scalar_t>
struct univ_zero_vec_type_of;

template<typename scalar_t>
using univ_zero_vec_type_of_t = typename univ_zero_vec_type_of<scalar_t>::type;

template<> struct univ_zero_vec_type_of<uchar> { using type = v_uint8; };
template<> struct univ_zero_vec_type_of<ushort> { using type = v_int16; };
template<> struct univ_zero_vec_type_of<short> { using type = v_int16; };
template<> struct univ_zero_vec_type_of<float> { using type = v_float32; };

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
    vx_store(dst, v_select(v_reinterpret_as_u16(div == v_zero),
                           v_reinterpret_as_u16(v_zero), v_pack_u(res1, res2)));
}

//=============================================================================

template<typename scale_tag_t>
CV_ALWAYS_INLINE
void div_simd_impl(scale_tag_t s_tag, const v_float32& a1, const v_float32& a2,
                   const v_float32& a3, const v_float32& a4, const uchar* in2x,
                   uchar* outx, const v_float32& v_scale, const v_int16& v_zero)
{
    constexpr int nlanes = v_uint8::nlanes;

    v_int16 div1 = v_reinterpret_as_s16(vx_load_expand(in2x));
    v_int16 div2 = v_reinterpret_as_s16(vx_load_expand(&in2x[nlanes/2]));

    v_float32 fdiv1 = v_cvt_f32(v_expand_low(div1));
    v_float32 fdiv2 = v_cvt_f32(v_expand_high(div1));
    v_float32 fdiv3 = v_cvt_f32(v_expand_low(div2));
    v_float32 fdiv4 = v_cvt_f32(v_expand_high(div2));

    v_int32 sum1 = v_round(div_op(s_tag, a1, fdiv1, v_scale)),
            sum2 = v_round(div_op(s_tag, a2, fdiv2, v_scale)),
            sum3 = v_round(div_op(s_tag, a3, fdiv3, v_scale)),
            sum4 = v_round(div_op(s_tag, a4, fdiv4, v_scale));

    v_int16 res1 = v_select((div1 == v_zero), v_zero, v_pack(sum1, sum2));
    v_int16 res2 = v_select((div2 == v_zero), v_zero, v_pack(sum3, sum4));

    vx_store(outx, v_pack_u(res1, res2));
}

template<typename scale_tag_t, typename SRC>
CV_ALWAYS_INLINE
typename std::enable_if<std::is_same<SRC, short>::value ||
                        std::is_same<SRC, ushort>::value, void>::type
div_simd_impl(scale_tag_t s_tag, const v_float32& a1, const v_float32& a2,
              const v_float32& a3, const v_float32& a4, const SRC* in2x,
              uchar* outx, const v_float32& v_scale, const v_int16& v_zero)
{
    constexpr int nlanes = v_uint8::nlanes;

    v_int16 div1 = v_reinterpret_as_s16(vx_load(in2x));
    v_int16 div2 = v_reinterpret_as_s16(vx_load(&in2x[nlanes/2]));

    v_float32 fdiv1 = v_cvt_f32(v_expand_low(div1));
    v_float32 fdiv2 = v_cvt_f32(v_expand_high(div1));
    v_float32 fdiv3 = v_cvt_f32(v_expand_low(div2));
    v_float32 fdiv4 = v_cvt_f32(v_expand_high(div2));

    v_int32 sum1 = v_round(div_op(s_tag, a1, fdiv1, v_scale)),
            sum2 = v_round(div_op(s_tag, a2, fdiv2, v_scale)),
            sum3 = v_round(div_op(s_tag, a3, fdiv3, v_scale)),
            sum4 = v_round(div_op(s_tag, a4, fdiv4, v_scale));

    v_int16 res1 = v_select((div1 == v_zero), v_zero, v_pack(sum1, sum2));
    v_int16 res2 = v_select((div2 == v_zero), v_zero, v_pack(sum3, sum4));

    vx_store(outx, v_pack_u(res1, res2));
}

template<typename scale_tag_t>
CV_ALWAYS_INLINE void div_simd_impl(scale_tag_t s_tag, const v_float32& a1,
                                    const v_float32& a2, const v_float32& a3,
                                    const v_float32& a4, const float* in2x, uchar* outx,
                                    const v_float32& v_scale, const v_float32& v_zero)
{
    constexpr int nlanes = v_uint8::nlanes;

    v_float32 div1 = vg_load_f32(in2x);
    v_float32 div2 = vg_load_f32(&in2x[nlanes / 4]);
    v_float32 div3 = vg_load_f32(&in2x[nlanes / 2]);
    v_float32 div4 = vg_load_f32(&in2x[3 * nlanes / 4]);

    v_float32 r1 = div_op(s_tag, a1, div1, v_scale);
    v_float32 r2 = div_op(s_tag, a2, div2, v_scale);
    v_float32 r3 = div_op(s_tag, a3, div3, v_scale);
    v_float32 r4 = div_op(s_tag, a4, div4, v_scale);

    v_float32 sel1 = v_select((div1 == v_zero), v_zero, r1);
    v_float32 sel2 = v_select((div2 == v_zero), v_zero, r2);
    v_float32 sel3 = v_select((div3 == v_zero), v_zero, r3);
    v_float32 sel4 = v_select((div4 == v_zero), v_zero, r4);

    v_int32 res1 = v_round(sel1);
    v_int32 res2 = v_round(sel2);
    v_int32 res3 = v_round(sel3);
    v_int32 res4 = v_round(sel4);

    vx_store(outx, v_pack_u(v_pack(res1, res2), v_pack(res3, res4)));
}

template<typename scale_tag_t, typename SRC, typename Vtype>
CV_ALWAYS_INLINE void div_hal(scale_tag_t s_tag, const SRC* in1x, const SRC* in2x, uchar* outx,
                              const v_float32& v_scale, const Vtype& v_zero)
{
    constexpr int nlanes = v_uint8::nlanes;

    v_float32 a1 = vg_load_f32(in1x);
    v_float32 a2 = vg_load_f32(&in1x[nlanes / 4]);
    v_float32 a3 = vg_load_f32(&in1x[nlanes / 2]);
    v_float32 a4 = vg_load_f32(&in1x[3 * nlanes / 4]);

    div_simd_impl(s_tag, a1, a2, a3, a4, in2x, outx, v_scale, v_zero);
}

//-------------------------------------------------------------------------------------------------

template<typename scale_tag_t, typename DST>
CV_ALWAYS_INLINE
typename std::enable_if<std::is_same<DST, short>::value ||
                        std::is_same<DST, ushort>::value, void>::type
div_simd_impl(scale_tag_t s_tag, const v_float32& a1, const v_float32& a2,
              const uchar* in2x, DST* outx, const v_float32& v_scale,
              const v_int16& v_zero)
{
    v_int16 div = v_reinterpret_as_s16(vx_load_expand(in2x));

    v_float32 fdiv1 = v_cvt_f32(v_expand_low(div));
    v_float32 fdiv2 = v_cvt_f32(v_expand_high(div));

    v_int32 r1 = v_round(div_op(s_tag, a1, fdiv1, v_scale));
    v_int32 r2 = v_round(div_op(s_tag, a2, fdiv2, v_scale));

    v_store_select(outx, div, v_zero, r1, r2);
}

template<typename scale_tag_t, typename SRC, typename DST>
CV_ALWAYS_INLINE
typename std::enable_if<(std::is_same<SRC, short>::value  && std::is_same<DST, ushort>::value) ||
                        (std::is_same<SRC, ushort>::value && std::is_same<DST, ushort>::value) ||
                        (std::is_same<SRC, short>::value  && std::is_same<DST, short>::value)  ||
                        (std::is_same<SRC, ushort>::value && std::is_same<DST, short>::value), void>::type
div_simd_impl(scale_tag_t s_tag, const v_float32& a1, const v_float32& a2,
              const SRC* in2x, DST* outx, const v_float32& v_scale, const v_int16& v_zero)
{
    v_int16 div = v_reinterpret_as_s16(vx_load(in2x));

    v_float32 fdiv1 = v_cvt_f32(v_expand_low(div));
    v_float32 fdiv2 = v_cvt_f32(v_expand_high(div));

    v_int32 r1 = v_round(div_op(s_tag, a1, fdiv1, v_scale));
    v_int32 r2 = v_round(div_op(s_tag, a2, fdiv2, v_scale));

    v_store_select(outx, div, v_zero, r1, r2);
}

template<typename scale_tag_t, typename DST>
CV_ALWAYS_INLINE
typename std::enable_if<std::is_same<DST, short>::value ||
                        std::is_same<DST, ushort>::value, void>::type
div_simd_impl(scale_tag_t s_tag, const v_float32& a1, const v_float32& a2,
              const float* in2x, DST* outx, const v_float32& v_scale,
              const v_float32& v_zero)
{
    constexpr int nlanes = vector_type_of_t<DST>::nlanes;

    v_float32 fdiv1 = vg_load_f32(in2x);
    v_float32 fdiv2 = vg_load_f32(&in2x[nlanes / 2]);

    v_float32 r1 = div_op(s_tag, a1, fdiv1, v_scale);
    v_float32 r2 = div_op(s_tag, a2, fdiv2, v_scale);

    v_int32 res1 = v_round(v_select((fdiv1 == v_zero), v_zero, r1));
    v_int32 res2 = v_round(v_select((fdiv2 == v_zero), v_zero, r2));

    v_store_i16(outx, res1, res2);
}

template<typename scale_tag_t, typename SRC, typename DST, typename Vtype>
CV_ALWAYS_INLINE
typename std::enable_if<std::is_same<DST, short>::value ||
                        std::is_same<DST, ushort>::value, void>::type
div_hal(scale_tag_t s_tag, const SRC* in1x, const SRC* in2x, DST* outx,
        const v_float32& v_scale, const Vtype& v_zero)
{
    constexpr int nlanes = vector_type_of_t<DST>::nlanes;

    v_float32 a1 = vg_load_f32(in1x);
    v_float32 a2 = vg_load_f32(&in1x[nlanes / 2]);

    div_simd_impl(s_tag, a1, a2, in2x, outx, v_scale, v_zero);
}

//-------------------------------------------------------------------------------------------------

template<typename scale_tag_t, typename SRC>
CV_ALWAYS_INLINE void div_simd_impl(scale_tag_t s_tag, const v_float32& a1, const SRC* in2x,
                                           float* outx, const v_float32& v_scale)
{
    v_float32 b1 = vg_load_f32(in2x);
    vx_store(outx, div_op(s_tag, a1, b1, v_scale));
}

template<typename scale_tag_t, typename SRC, typename Tvec>
CV_ALWAYS_INLINE void div_hal(scale_tag_t s_tag, const SRC* in1x, const SRC* in2x, float* outx,
                              const v_float32& v_scale, const Tvec&)
{
    v_float32 a1 = vg_load_f32(in1x);
    div_simd_impl(s_tag, a1, in2x, outx, v_scale);
}

//-------------------------------------------------------------------------------------------------

template<typename scale_tag_t, typename SRC, typename DST>
CV_ALWAYS_INLINE int div_simd_common(scale_tag_t s_tag, const SRC in1[], const SRC in2[],
                                     DST out[], const int length, float scale)
{
    constexpr int nlanes = vector_type_of_t<DST>::nlanes;

    if (length < nlanes)
        return 0;

    const zero_vec_type_of_t<SRC> v_zero = vx_setall<typename zero_vec_type_of_t<SRC>::lane_type>(0);
    v_float32 v_scale = vx_setall_f32(scale);

    int x = 0;
    for (;;)
    {
        for (; x <= length - nlanes; x += nlanes)
        {
            div_hal(s_tag, &in1[x], &in2[x], &out[x], v_scale, v_zero);
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

#define DIV_SIMD(SRC, DST)                                                      \
int div_simd(const SRC in1[], const SRC in2[], DST out[],                       \
                              const int length, double _scale)                  \
{                                                                               \
    int x = 0;                                                                  \
    float fscale = static_cast<float>(_scale);                                  \
    if (std::fabs(fscale - 1.0f) <= FLT_EPSILON)                                \
    {                                                                           \
        x = div_simd_common(not_scale_tag{}, in1, in2, out, length, fscale);    \
    }                                                                           \
    else                                                                        \
    {                                                                           \
        x = div_simd_common(scale_tag{}, in1, in2, out, length, fscale);        \
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
struct div_tag {};
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

CV_ALWAYS_INLINE v_float32 oper_scaled(mul_tag, const v_float32& a, const v_float32& v_scalar, const v_float32& v_scale)
{
    return v_scale * a * v_scalar;
}

CV_ALWAYS_INLINE v_float32 oper(div_tag, const v_float32& a, const v_float32& sc)
{
    return a / sc;
}

CV_ALWAYS_INLINE v_float32 oper_scaled(div_tag, const v_float32& a, const v_float32& v_scalar, const v_float32& v_scale)
{
    return a*v_scale / v_scalar;
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
// Fluid kernels: MulC, DivC
//
//-------------------------

template<typename oper_tag, typename SRC, typename DST>
CV_ALWAYS_INLINE
typename std::enable_if<std::is_same<DST, short>::value ||
                        std::is_same<DST, ushort>::value, void>::type
arithmOpScalarScaled_simd_c3_impl(oper_tag op, SRC* inx, DST* outx, const v_float32& s1,
                                  const v_float32& s2, const v_float32& s3,
                                  const v_float32& v_scale, const int nlanes)
{
    v_float32 a1 = vg_load_f32(inx);
    v_float32 a2 = vg_load_f32(&inx[nlanes / 2]);
    v_float32 a3 = vg_load_f32(&inx[nlanes]);
    v_float32 a4 = vg_load_f32(&inx[3 * nlanes / 2]);
    v_float32 a5 = vg_load_f32(&inx[2 * nlanes]);
    v_float32 a6 = vg_load_f32(&inx[5 * nlanes / 2]);

    arithmOpScalar_pack_store_c3(outx, v_round(oper_scaled(op, a1, s1, v_scale)),
                                       v_round(oper_scaled(op, a2, s2, v_scale)),
                                       v_round(oper_scaled(op, a3, s3, v_scale)),
                                       v_round(oper_scaled(op, a4, s1, v_scale)),
                                       v_round(oper_scaled(op, a5, s2, v_scale)),
                                       v_round(oper_scaled(op, a6, s3, v_scale)));
}

//-------------------------------------------------------------------------------------------------

template<typename oper_tag, typename SRC>
CV_ALWAYS_INLINE void arithmOpScalarScaled_simd_c3_impl(oper_tag op, const SRC* inx, uchar* outx,
                                                        const v_float32& s1, const v_float32& s2,
                                                        const v_float32& s3, const v_float32& v_scale,
                                                        const int nlanes)
{
    vx_store(outx,
               v_pack_u(v_pack(v_round(oper_scaled(op, vg_load_f32(inx), s1, v_scale)),
                               v_round(oper_scaled(op, vg_load_f32(&inx[nlanes/4]), s2, v_scale))),
                        v_pack(v_round(oper_scaled(op, vg_load_f32(&inx[nlanes/2]), s3, v_scale)),
                               v_round(oper_scaled(op, vg_load_f32(&inx[3*nlanes/4]), s1, v_scale)))));

    vx_store(&outx[nlanes],
                v_pack_u(v_pack(v_round(oper_scaled(op, vg_load_f32(&inx[nlanes]), s2, v_scale)),
                                v_round(oper_scaled(op, vg_load_f32(&inx[5*nlanes/4]), s3, v_scale))),
                         v_pack(v_round(oper_scaled(op, vg_load_f32(&inx[3*nlanes/2]), s1, v_scale)),
                                v_round(oper_scaled(op, vg_load_f32(&inx[7*nlanes/4]), s2, v_scale)))));

    vx_store(&outx[2 * nlanes],
                v_pack_u(v_pack(v_round(oper_scaled(op, vg_load_f32(&inx[2*nlanes]), s3, v_scale)),
                                v_round(oper_scaled(op, vg_load_f32(&inx[9*nlanes/4]), s1, v_scale))),
                         v_pack(v_round(oper_scaled(op, vg_load_f32(&inx[5*nlanes/2]), s2, v_scale)),
                                v_round(oper_scaled(op, vg_load_f32(&inx[11*nlanes/4]), s3, v_scale)))));
}

//-------------------------------------------------------------------------------------------------

template<typename oper_tag, typename SRC>
CV_ALWAYS_INLINE void arithmOpScalarScaled_simd_c3_impl(oper_tag op, const SRC* in, float* out,
                                                        const v_float32& s1, const v_float32& s2,
                                                        const v_float32& s3, const v_float32& v_scale,
                                                        const int nlanes)
{
    v_float32 a1 = vg_load_f32(in);
    v_float32 a2 = vg_load_f32(&in[nlanes]);
    v_float32 a3 = vg_load_f32(&in[2*nlanes]);

    vx_store(out, oper_scaled(op, a1, s1, v_scale));
    vx_store(&out[nlanes], oper_scaled(op, a2, s2, v_scale));
    vx_store(&out[2*nlanes], oper_scaled(op, a3, s3, v_scale));
}

//-------------------------------------------------------------------------------------------------

template<typename oper_tag, typename SRC, typename DST>
CV_ALWAYS_INLINE int arithmOpScalarScaled_simd_c3(oper_tag op, const SRC in[],
                                                  const float scalar[], DST out[],
                                                  const int length, const float scale)
{
    constexpr int chan = 3;
    constexpr int nlanes = vector_type_of_t<DST>::nlanes;
    constexpr int lanes = chan * nlanes;

    if (length < lanes)
        return 0;

    v_float32 v_scale = vx_setall_f32(scale);

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
            arithmOpScalarScaled_simd_c3_impl(op, &in[x], &out[x], s1, s2, s3, v_scale, nlanes);
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
CV_ALWAYS_INLINE
typename std::enable_if<(std::is_same<DST, ushort>::value ||
                         std::is_same<DST, short>::value), void>::type
arithmOpScalarScaled_simd_common_impl(oper_tag op, const SRC* inx, DST* outx,
                                      const v_float32& v_scalar, const v_float32& v_scale,
                                      const int nlanes)
{
    v_float32 a1 = vg_load_f32(inx);
    v_float32 a2 = vg_load_f32(&inx[nlanes/2]);

    v_store_i16(outx, v_round(oper_scaled(op, a1, v_scalar, v_scale)), v_round(oper_scaled(op, a2, v_scalar, v_scale)));
}

//-------------------------------------------------------------------------------------------------

template<typename oper_tag, typename SRC>
CV_ALWAYS_INLINE void arithmOpScalarScaled_simd_common_impl(oper_tag op, const SRC* inx,
                                                            uchar* outx, const v_float32& v_scalar,
                                                            const v_float32& v_scale, const int nlanes)
{
    v_float32 a1 = vg_load_f32(inx);
    v_float32 a2 = vg_load_f32(&inx[nlanes/4]);
    v_float32 a3 = vg_load_f32(&inx[nlanes/2]);
    v_float32 a4 = vg_load_f32(&inx[3 * nlanes/4]);

    vx_store(outx, v_pack_u(v_pack(v_round(oper_scaled(op, a1, v_scalar, v_scale)),
                                   v_round(oper_scaled(op, a2, v_scalar, v_scale))),
                            v_pack(v_round(oper_scaled(op, a3, v_scalar, v_scale)),
                                   v_round(oper_scaled(op, a4, v_scalar, v_scale)))));
}

//-------------------------------------------------------------------------------------------------

template<typename oper_tag, typename SRC>
CV_ALWAYS_INLINE void arithmOpScalarScaled_simd_common_impl(oper_tag op, const SRC* inx,
                                                            float* outx, const v_float32& v_scalar,
                                                            const v_float32& v_scale, const int)
{
    v_float32 a = vg_load_f32(inx);
    vx_store(outx, oper_scaled(op, a, v_scalar, v_scale));
}

//-------------------------------------------------------------------------------------------------

template<typename oper_tag, typename SRC, typename DST>
CV_ALWAYS_INLINE int arithmOpScalarScaled_simd_common(oper_tag op, const SRC in[],
                                                      const float scalar[], DST out[],
                                                      const int length, const float scale)
{
    constexpr int nlanes = vector_type_of_t<DST>::nlanes;

    if (length < nlanes)
        return 0;

    v_float32 v_scalar = vx_load(scalar);
    v_float32 v_scale = vx_setall_f32(scale);

    int x = 0;
    for (;;)
    {
        for (; x <= length - nlanes; x += nlanes)
        {
            arithmOpScalarScaled_simd_common_impl(op, &in[x], &out[x], v_scalar, v_scale, nlanes);
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
            return arithmOpScalarScaled_simd_common(op_t, in, scalar, out,     \
                                                    length, scale);            \
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
            return arithmOpScalarScaled_simd_c3(op_t, in, scalar, out,         \
                                                length, scale);                \
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

//-------------------------------------------------------------------------------------------------

template<typename scale_tag_t, typename SRC, typename DST>
CV_ALWAYS_INLINE
typename std::enable_if<(std::is_same<DST, ushort>::value ||
                         std::is_same<DST, short>::value), int>::type
divc_simd_common_impl(scale_tag_t s_tag, const SRC in[], DST out[],
                      const v_float32& v_scalar, const v_float32& v_scale,
                      const int length)
{
    constexpr int nlanes = vector_type_of_t<DST>::nlanes;

    v_float32 v_zero = vx_setzero_f32();
    v_float32 v_mask = (v_scalar == v_zero);

    int x = 0;
    for (;;)
    {
        for (; x <= length - nlanes; x += nlanes)
        {
            v_float32 a1 = vg_load_f32(&in[x]);
            v_float32 a2 = vg_load_f32(&in[x + nlanes/2]);

            v_store_i16(&out[x], v_round(v_select(v_mask, v_zero, div_op(s_tag, a1, v_scalar, v_scale))),
                                 v_round(v_select(v_mask, v_zero, div_op(s_tag, a2, v_scalar, v_scale))));
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

template<typename scale_tag_t, typename SRC>
CV_ALWAYS_INLINE int divc_simd_common_impl(scale_tag_t s_tag, const SRC in[],
                                           uchar out[], const v_float32& v_scalar,
                                           const v_float32& v_scale, const int length)
{
    constexpr int nlanes = v_uint8::nlanes;

    v_float32 v_zero = vx_setzero_f32();
    v_float32 v_mask = (v_scalar == v_zero);

    int x = 0;
    for (;;)
    {
        for (; x <= length - nlanes; x += nlanes)
        {
            v_float32 a1 = vg_load_f32(&in[x]);
            v_float32 a2 = vg_load_f32(&in[x + nlanes/4]);
            v_float32 a3 = vg_load_f32(&in[x + nlanes/2]);
            v_float32 a4 = vg_load_f32(&in[x + 3 * nlanes/4]);

            vx_store(&out[x], v_pack_u(v_pack(v_round(v_select(v_mask, v_zero, div_op(s_tag, a1, v_scalar, v_scale))),
                                              v_round(v_select(v_mask, v_zero, div_op(s_tag, a2, v_scalar, v_scale)))),
                                       v_pack(v_round(v_select(v_mask, v_zero, div_op(s_tag, a3, v_scalar, v_scale))),
                                              v_round(v_select(v_mask, v_zero, div_op(s_tag, a4, v_scalar, v_scale))))));
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

template<typename scale_tag_t, typename SRC>
CV_ALWAYS_INLINE int divc_simd_common_impl(scale_tag_t s_tag, const SRC in[],
                                           float out[], const v_float32& v_scalar,
                                           const v_float32& v_scale, const int length)
{
    constexpr int nlanes = v_float32::nlanes;
    int x = 0;
    for (;;)
    {
        for (; x <= length - nlanes; x += nlanes)
        {
            v_float32 a1 = vg_load_f32(&in[x]);
            vx_store(&out[x], div_op(s_tag, a1, v_scalar, v_scale));
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

template<typename scale_tag_t, typename SRC, typename DST>
CV_ALWAYS_INLINE int divc_mask_simd_common(scale_tag_t tag, const SRC in[],
                                           const float scalar[], DST out[],
                                           const int length, const float scale)
{
    constexpr int nlanes = vector_type_of_t<DST>::nlanes;

    if (length < nlanes)
        return 0;

    v_float32 v_scalar = vx_load(scalar);
    v_float32 v_scale = vx_setall_f32(scale);
    return divc_simd_common_impl(tag, in, out, v_scalar, v_scale, length);
}

//-------------------------------------------------------------------------------------------------

template<typename scale_tag_t, typename SRC, typename DST>
CV_ALWAYS_INLINE
typename std::enable_if<std::is_same<DST, short>::value ||
                        std::is_same<DST, ushort>::value, int>::type
divc_simd_c3_impl(scale_tag_t s_tag, SRC in[], DST out[], const v_float32& s1,
                  const v_float32& s2, const v_float32& s3,
                  const v_float32& v_scale, const int length,
                  const int nlanes, const int lanes)
{
    v_float32 v_zero = vx_setzero_f32();
    v_float32 v_mask1 = (s1 == v_zero);
    v_float32 v_mask2 = (s2 == v_zero);
    v_float32 v_mask3 = (s3 == v_zero);

    int x = 0;
    for (;;)
    {
        for (; x <= length - lanes; x += lanes)
        {
            v_float32 a1 = vg_load_f32(&in[x]);
            v_float32 a2 = vg_load_f32(&in[x + nlanes / 2]);
            v_float32 a3 = vg_load_f32(&in[x + nlanes]);
            v_float32 a4 = vg_load_f32(&in[x + 3 * nlanes / 2]);
            v_float32 a5 = vg_load_f32(&in[x + 2 * nlanes]);
            v_float32 a6 = vg_load_f32(&in[x + 5 * nlanes / 2]);

            arithmOpScalar_pack_store_c3(&out[x], v_round(v_select(v_mask1, v_zero, div_op(s_tag, a1, s1, v_scale))),
                                                  v_round(v_select(v_mask2, v_zero, div_op(s_tag, a2, s2, v_scale))),
                                                  v_round(v_select(v_mask3, v_zero, div_op(s_tag, a3, s3, v_scale))),
                                                  v_round(v_select(v_mask1, v_zero, div_op(s_tag, a4, s1, v_scale))),
                                                  v_round(v_select(v_mask2, v_zero, div_op(s_tag, a5, s2, v_scale))),
                                                  v_round(v_select(v_mask3, v_zero, div_op(s_tag, a6, s3, v_scale))));
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

template<typename scale_tag_t, typename SRC>
CV_ALWAYS_INLINE int divc_simd_c3_impl(scale_tag_t s_tag, const SRC* in, uchar* out,
                                       const v_float32& s1, const v_float32& s2,
                                       const v_float32& s3, const v_float32& v_scale,
                                       const int length, const int nlanes, const int lanes)
{
    v_float32 v_zero = vx_setzero_f32();
    v_float32 v_mask1 = (s1 == v_zero);
    v_float32 v_mask2 = (s2 == v_zero);
    v_float32 v_mask3 = (s3 == v_zero);

    int x = 0;
    for (;;)
    {
        for (; x <= length - lanes; x += lanes)
        {
            vx_store(&out[x],
                       v_pack_u(v_pack(v_round(v_select(v_mask1, v_zero, div_op(s_tag, vg_load_f32(&in[x]), s1, v_scale))),
                                       v_round(v_select(v_mask2, v_zero, div_op(s_tag, vg_load_f32(&in[x + nlanes/4]), s2, v_scale)))),
                                v_pack(v_round(v_select(v_mask3, v_zero, div_op(s_tag, vg_load_f32(&in[x + nlanes/2]), s3, v_scale))),
                                       v_round(v_select(v_mask1, v_zero, div_op(s_tag, vg_load_f32(&in[x + 3*nlanes/4]), s1, v_scale))))));

            vx_store(&out[x + nlanes],
                        v_pack_u(v_pack(v_round(v_select(v_mask2, v_zero, div_op(s_tag, vg_load_f32(&in[x + nlanes]), s2, v_scale))),
                                        v_round(v_select(v_mask3, v_zero, div_op(s_tag, vg_load_f32(&in[x + 5*nlanes/4]), s3, v_scale)))),
                                 v_pack(v_round(v_select(v_mask1, v_zero, div_op(s_tag, vg_load_f32(&in[x + 3*nlanes/2]), s1, v_scale))),
                                        v_round(v_select(v_mask2, v_zero, div_op(s_tag, vg_load_f32(&in[x + 7*nlanes/4]), s2, v_scale))))));

            vx_store(&out[x + 2 * nlanes],
                        v_pack_u(v_pack(v_round(v_select(v_mask3, v_zero, div_op(s_tag, vg_load_f32(&in[x + 2*nlanes]), s3, v_scale))),
                                        v_round(v_select(v_mask1, v_zero, div_op(s_tag, vg_load_f32(&in[x + 9*nlanes/4]), s1, v_scale)))),
                                 v_pack(v_round(v_select(v_mask2, v_zero, div_op(s_tag, vg_load_f32(&in[x + 5*nlanes/2]), s2, v_scale))),
                                        v_round(v_select(v_mask3, v_zero, div_op(s_tag, vg_load_f32(&in[x + 11*nlanes/4]), s3, v_scale))))));
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

template<typename scale_tag_t, typename SRC>
CV_ALWAYS_INLINE int divc_simd_c3_impl(scale_tag_t s_tag, const SRC* in, float* out,
                                       const v_float32& s1, const v_float32& s2,
                                       const v_float32& s3, const v_float32& v_scale, const int length,
                                       const int nlanes, const int lanes)
{
    int x = 0;
    for (;;)
    {
        for (; x <= length - lanes; x += lanes)
        {
            v_float32 a1 = vg_load_f32(&in[x]);
            v_float32 a2 = vg_load_f32(&in[x + nlanes]);
            v_float32 a3 = vg_load_f32(&in[x + 2*nlanes]);

            vx_store(&out[x], div_op(s_tag, a1, s1, v_scale));
            vx_store(&out[x + nlanes], div_op(s_tag, a2, s2, v_scale));
            vx_store(&out[x + 2*nlanes], div_op(s_tag, a3, s3, v_scale));
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

template<typename scale_tag_t, typename SRC, typename DST>
CV_ALWAYS_INLINE int divc_mask_simd_c3(scale_tag_t s_tag, const SRC in[],
                                       const float scalar[], DST out[],
                                       const int length, const float scale)
{
    constexpr int chan = 3;
    constexpr int nlanes = vector_type_of_t<DST>::nlanes;
    constexpr int lanes = chan * nlanes;

    if (length < lanes)
        return 0;

    v_float32 v_scale = vx_setall_f32(scale);

    v_float32 s1 = vx_load(scalar);
#if CV_SIMD_WIDTH == 32
    v_float32 s2 = vx_load(&scalar[2]);
    v_float32 s3 = vx_load(&scalar[1]);
#else
    v_float32 s2 = vx_load(&scalar[1]);
    v_float32 s3 = vx_load(&scalar[2]);
#endif
     return divc_simd_c3_impl(s_tag, in, out, s1, s2, s3, v_scale, length, nlanes, lanes);
}

//-------------------------------------------------------------------------------------------------

#define DIVC_SIMD(SRC, DST)                                                    \
int divc_simd(const SRC in[], const float scalar[], DST out[],                 \
              const int length, const int chan, const float scale,             \
              const int set_mask_flag)                                         \
{                                                                              \
    switch (chan)                                                              \
    {                                                                          \
    case 1:                                                                    \
    case 2:                                                                    \
    case 4:                                                                    \
    {                                                                          \
        if (std::fabs(scale - 1.0f) <= FLT_EPSILON)                            \
        {                                                                      \
            if (set_mask_flag == 1)                                            \
                return divc_mask_simd_common(not_scale_tag{}, in, scalar,      \
                                             out, length, scale);              \
            else                                                               \
                return arithmOpScalar_simd_common(div_tag{}, in, scalar,       \
                                                  out, length);                \
        }                                                                      \
        else                                                                   \
        {   if (set_mask_flag == 1)                                            \
                return divc_mask_simd_common(scale_tag{}, in, scalar,          \
                                             out, length, scale);              \
            else                                                               \
                return arithmOpScalarScaled_simd_common(div_tag{}, in, scalar, \
                                                        out, length, scale);   \
        }                                                                      \
    }                                                                          \
    case 3:                                                                    \
    {                                                                          \
        if (std::fabs(scale - 1.0f) <= FLT_EPSILON)                            \
        {                                                                      \
            if (set_mask_flag == 1)                                            \
                return divc_mask_simd_c3(not_scale_tag{}, in, scalar,          \
                                             out, length, scale);              \
            else                                                               \
                return arithmOpScalar_simd_c3(div_tag{}, in, scalar,           \
                                              out, length);                    \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            if (set_mask_flag == 1)                                            \
                return divc_mask_simd_c3(scale_tag{}, in, scalar,              \
                                         out, length, scale);                  \
            else                                                               \
                return arithmOpScalarScaled_simd_c3(div_tag{}, in, scalar, out,\
                                                    length, scale);            \
        }                                                                      \
    }                                                                          \
    default:                                                                   \
        GAPI_Assert(chan <= 4);                                                \
        break;                                                                 \
    }                                                                          \
    return 0;                                                                  \
}

DIVC_SIMD(uchar, uchar)
DIVC_SIMD(ushort, uchar)
DIVC_SIMD(short, uchar)
DIVC_SIMD(float, uchar)
DIVC_SIMD(short, short)
DIVC_SIMD(ushort, short)
DIVC_SIMD(uchar, short)
DIVC_SIMD(float, short)
DIVC_SIMD(ushort, ushort)
DIVC_SIMD(uchar, ushort)
DIVC_SIMD(short, ushort)
DIVC_SIMD(float, ushort)
DIVC_SIMD(uchar, float)
DIVC_SIMD(ushort, float)
DIVC_SIMD(short, float)
DIVC_SIMD(float, float)

#undef DIVC_SIMD

//-------------------------
//
// Fluid kernels: AbsDiffC
//
//-------------------------

#define ABSDIFFC_SIMD(SRC)                                                          \
int absdiffc_simd(const SRC in[], const float scalar[], SRC out[],                  \
                  const int length, const int chan)                                 \
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

//-------------------------------------------------------------------------------------------------

template<typename scale_tag_t, typename SRC, typename DST, typename Tvec>
CV_ALWAYS_INLINE
typename std::enable_if<std::is_same<DST, short>::value ||
                        std::is_same<DST, ushort>::value, void>::type
divrc_simd_common_impl(scale_tag_t s_tag, const SRC* inx,
                       const v_float32& v_scalar, DST* outx,
                       const v_float32& v_scale, const Tvec& v_zero)
{
    div_simd_impl(s_tag, v_scalar, v_scalar, inx, outx, v_scale, v_zero);
}

template<typename scale_tag_t, typename SRC, typename DST, typename Tvec>
CV_ALWAYS_INLINE
typename std::enable_if<std::is_same<DST, uchar>::value, void>::type
divrc_simd_common_impl(scale_tag_t s_tag, const SRC* inx,
                       const v_float32& v_scalar, DST* outx,
                       const v_float32& v_scale, const Tvec& v_zero)
{
    div_simd_impl(s_tag, v_scalar, v_scalar, v_scalar, v_scalar, inx, outx, v_scale, v_zero);
}

template<typename scale_tag_t, typename SRC, typename DST, typename Tvec>
CV_ALWAYS_INLINE
typename std::enable_if<std::is_same<DST, float>::value, void>::type
divrc_simd_common_impl(scale_tag_t s_tag, const SRC* inx,
                       const v_float32& v_scalar, DST* outx,
                       const v_float32& v_scale, const Tvec&)
{
    div_simd_impl(s_tag, v_scalar, inx, outx, v_scale);
}

template<typename scale_tag_t, typename SRC, typename DST>
CV_ALWAYS_INLINE int divrc_simd_common(scale_tag_t s_tag, const SRC in[],
                                       const float scalar[], DST out[],
                                       const int length, const float scale)
{
    constexpr int nlanes = vector_type_of_t<DST>::nlanes;

    if (length < nlanes)
        return 0;

    v_float32 v_scalar = vx_load(scalar);
    v_float32 v_scale = vx_setall_f32(scale);
    zero_vec_type_of_t<SRC> v_zero =
                         vx_setall<typename zero_vec_type_of_t<SRC>::lane_type>(0);

    int x = 0;
    for (;;)
    {
        for (; x <= length - nlanes; x += nlanes)
        {
            divrc_simd_common_impl(s_tag, &in[x], v_scalar, &out[x], v_scale, v_zero);
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

template<typename scale_tag_t>
CV_ALWAYS_INLINE void divrc_simd_c3_calc(scale_tag_t s_tag, const uchar* inx, uchar* outx,
                                         const v_float32& s1, const v_float32& s2,
                                         const v_float32& s3, const v_float32& v_scale,
                                         const v_uint8& v_zero)
{
    v_uint8 div = vx_load(inx);
    v_uint8 v_mask = (div == v_zero);

    v_uint16 div1 = v_expand_low(div);
    v_uint16 div2 = v_expand_high(div);

    v_float32 fdiv1 = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(div1)));
    v_float32 fdiv2 = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(div1)));
    v_float32 fdiv3 = v_cvt_f32(v_reinterpret_as_s32(v_expand_low(div2)));
    v_float32 fdiv4 = v_cvt_f32(v_reinterpret_as_s32(v_expand_high(div2)));

    vx_store(outx,
        v_select(v_mask, v_zero, v_pack_u(v_pack(v_round(div_op(s_tag, s1, fdiv1, v_scale)),
                                                 v_round(div_op(s_tag, s2, fdiv2, v_scale))),
                                          v_pack(v_round(div_op(s_tag, s3, fdiv3, v_scale)),
                                                 v_round(div_op(s_tag, s1, fdiv4, v_scale))))));
}

template<typename scale_tag_t, typename SRC>
CV_ALWAYS_INLINE
typename std::enable_if<std::is_same<SRC, short>::value ||
                        std::is_same<SRC, ushort>::value, void>::type
divrc_simd_c3_calc(scale_tag_t s_tag, const SRC* inx, uchar* outx,
                   const v_float32& s1, const v_float32& s2,
                   const v_float32& s3, const v_float32& v_scale,
                   const v_int16& v_zero)
{
    constexpr int nlanes = v_uint8::nlanes;

    v_int16 div1 = v_reinterpret_as_s16(vx_load(inx));
    v_int16 div2 = v_reinterpret_as_s16(vx_load(&inx[nlanes / 2]));

    v_int16 v_mask1 = (div1 == v_zero);
    v_int16 v_mask2 = (div2 == v_zero);

    v_float32 fdiv1 = v_cvt_f32(v_expand_low(div1));
    v_float32 fdiv2 = v_cvt_f32(v_expand_high(div1));
    v_float32 fdiv3 = v_cvt_f32(v_expand_low(div2));
    v_float32 fdiv4 = v_cvt_f32(v_expand_high(div2));

    vx_store(outx,
             v_pack_u(v_select(v_mask1, v_zero,
                               v_pack(v_round(div_op(s_tag, s1, fdiv1, v_scale)),
                                      v_round(div_op(s_tag, s2, fdiv2, v_scale)))),
                      v_select(v_mask2, v_zero,
                               v_pack(v_round(div_op(s_tag, s3, fdiv3, v_scale)),
                                      v_round(div_op(s_tag, s1, fdiv4, v_scale))))));
}

template<typename scale_tag_t>
CV_ALWAYS_INLINE void divrc_simd_c3_calc(scale_tag_t s_tag, const float* inx, uchar* outx,
                                         const v_float32& s1, const v_float32& s2,
                                         const v_float32& s3, const v_float32& v_scale,
                                         const v_float32& v_zero)
{
    constexpr int nlanes = v_uint8::nlanes;

    v_float32 fdiv1 = vg_load_f32(inx);
    v_float32 fdiv2 = vg_load_f32(&inx[nlanes / 4]);
    v_float32 fdiv3 = vg_load_f32(&inx[nlanes / 2]);
    v_float32 fdiv4 = vg_load_f32(&inx[3 * nlanes / 4]);

    v_float32 v_mask1 = (fdiv1 == v_zero);
    v_float32 v_mask2 = (fdiv2 == v_zero);
    v_float32 v_mask3 = (fdiv3 == v_zero);
    v_float32 v_mask4 = (fdiv4 == v_zero);

    vx_store(outx,
             v_pack_u(v_pack(v_round(v_select(v_mask1, v_zero, div_op(s_tag, s1, fdiv1, v_scale))),
                             v_round(v_select(v_mask2, v_zero, div_op(s_tag, s2, fdiv2, v_scale)))),
                      v_pack(v_round(v_select(v_mask3, v_zero, div_op(s_tag, s3, fdiv3, v_scale))),
                             v_round(v_select(v_mask4, v_zero, div_op(s_tag, s1, fdiv4, v_scale))))));

}

template<typename scale_tag_t, typename SRC>
CV_ALWAYS_INLINE int divrc_simd_c3_impl(scale_tag_t s_tag, const SRC in[], uchar out[],
                                        const v_float32& s1, const v_float32& s2,
                                        const v_float32& s3, const v_float32& v_scale,
                                        const int length, const int nlanes, const int lanes)
{
    univ_zero_vec_type_of_t<SRC> v_zero =
        vx_setall<typename univ_zero_vec_type_of_t<SRC>::lane_type>(0);

    int x = 0;
    for (;;)
    {
        for (; x <= length - lanes; x += lanes)
        {
            divrc_simd_c3_calc(s_tag, &in[x], &out[x], s1, s2, s3, v_scale, v_zero);
            divrc_simd_c3_calc(s_tag, &in[x + nlanes], &out[x + nlanes], s2, s3, s1, v_scale, v_zero);
            divrc_simd_c3_calc(s_tag, &in[x + 2 * nlanes], &out[x + 2 * nlanes], s3, s1, s2, v_scale, v_zero);
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

//---------------------------------------------------------------------------------------

template<typename scale_tag_t, typename DST>
CV_ALWAYS_INLINE
typename std::enable_if<std::is_same<DST, short>::value ||
                        std::is_same<DST, ushort>::value, void>::type
divrc_simd_c3_calc(scale_tag_t s_tag, const uchar* inx, DST* outx,
                   const v_float32& s1, const v_float32& s2,
                   const v_float32& s3, const v_float32& v_scale,
                   const v_int16& v_zero)
{
    constexpr int nlanes = vector_type_of_t<DST>::nlanes;
    v_uint8 div = vx_load(inx);

    v_int16 div1 = v_reinterpret_as_s16(v_expand_low(div));
    v_int16 div2 = v_reinterpret_as_s16(v_expand_high(div));
    v_int16 div3 = v_reinterpret_as_s16(vx_load_expand(&inx[2 * nlanes]));

    v_float32 fdiv1 = v_cvt_f32(v_expand_low(div1));
    v_float32 fdiv2 = v_cvt_f32(v_expand_high(div1));
    v_float32 fdiv3 = v_cvt_f32(v_expand_low(div2));
    v_float32 fdiv4 = v_cvt_f32(v_expand_high(div2));
    v_float32 fdiv5 = v_cvt_f32(v_expand_low(div3));
    v_float32 fdiv6 = v_cvt_f32(v_expand_high(div3));

    v_store_select(outx, div1, v_zero, v_round(div_op(s_tag, s1, fdiv1, v_scale)),
                                       v_round(div_op(s_tag, s2, fdiv2, v_scale)));
    v_store_select(&outx[nlanes], div2, v_zero, v_round(div_op(s_tag, s3, fdiv3, v_scale)),
                                                v_round(div_op(s_tag, s1, fdiv4, v_scale)));
    v_store_select(&outx[2*nlanes], div3, v_zero, v_round(div_op(s_tag, s2, fdiv5, v_scale)),
                                                  v_round(div_op(s_tag, s3, fdiv6, v_scale)));
}

template<typename scale_tag_t, typename SRC, typename DST>
CV_ALWAYS_INLINE
typename std::enable_if<(std::is_same<SRC, short>::value  && std::is_same<DST, ushort>::value) ||
                        (std::is_same<SRC, ushort>::value && std::is_same<DST, ushort>::value) ||
                        (std::is_same<SRC, short>::value  && std::is_same<DST, short>::value)  ||
                        (std::is_same<SRC, ushort>::value && std::is_same<DST, short>::value), void>::type
divrc_simd_c3_calc(scale_tag_t s_tag, const SRC* inx, DST* outx,
                   const v_float32& s1, const v_float32& s2,
                   const v_float32& s3, const v_float32& v_scale,
                   const v_int16& v_zero)
{
    constexpr int nlanes = vector_type_of_t<DST>::nlanes;

    v_int16 div1 = v_reinterpret_as_s16(vx_load(inx));
    v_int16 div2 = v_reinterpret_as_s16(vx_load(&inx[nlanes]));
    v_int16 div3 = v_reinterpret_as_s16(vx_load(&inx[2*nlanes]));

    v_float32 fdiv1 = v_cvt_f32(v_expand_low(div1));
    v_float32 fdiv2 = v_cvt_f32(v_expand_high(div1));
    v_float32 fdiv3 = v_cvt_f32(v_expand_low(div2));
    v_float32 fdiv4 = v_cvt_f32(v_expand_high(div2));
    v_float32 fdiv5 = v_cvt_f32(v_expand_low(div3));
    v_float32 fdiv6 = v_cvt_f32(v_expand_high(div3));

    v_store_select(outx, div1, v_zero, v_round(div_op(s_tag, s1, fdiv1, v_scale)),
                                       v_round(div_op(s_tag, s2, fdiv2, v_scale)));
    v_store_select(&outx[nlanes], div2, v_zero, v_round(div_op(s_tag, s3, fdiv3, v_scale)),
                                                v_round(div_op(s_tag, s1, fdiv4, v_scale)));
    v_store_select(&outx[2*nlanes], div3, v_zero, v_round(div_op(s_tag, s2, fdiv5, v_scale)),
                                                  v_round(div_op(s_tag, s3, fdiv6, v_scale)));
}

template<typename scale_tag_t, typename DST>
CV_ALWAYS_INLINE
typename std::enable_if<std::is_same<DST, short>::value ||
                        std::is_same<DST, ushort>::value, void>::type
divrc_simd_c3_calc(scale_tag_t s_tag, const float* inx, DST* outx,
                   const v_float32& s1, const v_float32& s2,
                   const v_float32& s3, const v_float32& v_scale,
                   const v_float32& v_zero)
{
    constexpr int nlanes = vector_type_of_t<DST>::nlanes;

    v_float32 fdiv1 = vg_load_f32(inx);
    v_float32 fdiv2 = vg_load_f32(&inx[nlanes/2]);
    v_float32 fdiv3 = vg_load_f32(&inx[nlanes]);
    v_float32 fdiv4 = vg_load_f32(&inx[3*nlanes/2]);
    v_float32 fdiv5 = vg_load_f32(&inx[2*nlanes]);
    v_float32 fdiv6 = vg_load_f32(&inx[5*nlanes/2]);

    v_store_i16(outx, v_round(v_select(fdiv1 == v_zero, v_zero, div_op(s_tag, s1, fdiv1, v_scale))),
                      v_round(v_select(fdiv2 == v_zero, v_zero, div_op(s_tag, s2, fdiv2, v_scale))));
    v_store_i16(&outx[nlanes], v_round(v_select(fdiv3 == v_zero, v_zero, div_op(s_tag, s3, fdiv3, v_scale))),
                               v_round(v_select(fdiv4 == v_zero, v_zero, div_op(s_tag, s1, fdiv4, v_scale))));
    v_store_i16(&outx[2*nlanes], v_round(v_select(fdiv5 == v_zero, v_zero, div_op(s_tag, s2, fdiv5, v_scale))),
                                 v_round(v_select(fdiv6 == v_zero, v_zero, div_op(s_tag, s3, fdiv6, v_scale))));
}

template<typename scale_tag_t, typename SRC, typename DST>
CV_ALWAYS_INLINE
typename std::enable_if<std::is_same<DST, short>::value ||
                        std::is_same<DST, ushort>::value, int>::type
divrc_simd_c3_impl(scale_tag_t s_tag, const SRC in[], DST out[], const v_float32& s1,
                   const v_float32& s2, const v_float32& s3,
                   const v_float32& v_scale, const int length,
                   const int, const int lanes)
{
    zero_vec_type_of_t<SRC> v_zero =
        vx_setall<typename zero_vec_type_of_t<SRC>::lane_type>(0);

    int x = 0;
    for (;;)
    {
        for (; x <= length - lanes; x += lanes)
        {
            divrc_simd_c3_calc(s_tag, &in[x], &out[x], s1, s2, s3, v_scale, v_zero);
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

//---------------------------------------------------------------------------------------

template<typename scale_tag_t, typename SRC>
CV_ALWAYS_INLINE int divrc_simd_c3_impl(scale_tag_t s_tag, const SRC* in, float* out,
                                        const v_float32& s1, const v_float32& s2,
                                        const v_float32& s3, const v_float32& v_scale,
                                        const int length, const int nlanes, const int lanes)
{
    int x = 0;
    for (;;)
    {
        for (; x <= length - lanes; x += lanes)
        {
            v_float32 div1 = vg_load_f32(&in[x]);
            v_float32 div2 = vg_load_f32(&in[x + nlanes]);
            v_float32 div3 = vg_load_f32(&in[x + 2*nlanes]);

            vx_store(&out[x], div_op(s_tag, s1, div1, v_scale));
            vx_store(&out[x + nlanes], div_op(s_tag, s2, div2, v_scale));
            vx_store(&out[x + 2*nlanes], div_op(s_tag, s3, div3, v_scale));
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

template<typename scale_tag_t, typename SRC, typename DST>
CV_ALWAYS_INLINE int divrc_simd_c3(scale_tag_t s_tag, const SRC in[],
                                   const float scalar[], DST out[],
                                   const int length, const float scale)
{
    constexpr int chan = 3;
    constexpr int nlanes = vector_type_of_t<DST>::nlanes;
    constexpr int lanes = chan * nlanes;

    if (length < lanes)
        return 0;

    v_float32 v_scale = vx_setall_f32(scale);

    v_float32 s1 = vx_load(scalar);
#if CV_SIMD_WIDTH == 32
    v_float32 s2 = vx_load(&scalar[2]);
    v_float32 s3 = vx_load(&scalar[1]);
#else
    v_float32 s2 = vx_load(&scalar[1]);
    v_float32 s3 = vx_load(&scalar[2]);
#endif
     return divrc_simd_c3_impl(s_tag, in, out, s1, s2, s3, v_scale, length, nlanes, lanes);
}

#define DIVRC_SIMD(SRC, DST)                                                   \
int divrc_simd(const float scalar[], const SRC in[], DST out[],                \
               const int length, const int chan, const float scale)            \
{                                                                              \
    switch (chan)                                                              \
    {                                                                          \
    case 1:                                                                    \
    case 2:                                                                    \
    case 4:                                                                    \
    {                                                                          \
        if (std::fabs(scale - 1.0f) <= FLT_EPSILON)                            \
        {                                                                      \
            return divrc_simd_common(not_scale_tag{}, in, scalar,              \
                                     out, length, scale);                      \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            return divrc_simd_common(scale_tag{}, in, scalar, out,             \
                                     length, scale);                           \
        }                                                                      \
    }                                                                          \
    case 3:                                                                    \
    {                                                                          \
       if (std::fabs(scale - 1.0f) <= FLT_EPSILON)                             \
        {                                                                      \
            return divrc_simd_c3(not_scale_tag{}, in, scalar,                  \
                                 out, length, scale);                          \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            return divrc_simd_c3(scale_tag{}, in, scalar, out,                 \
                                 length, scale);                               \
        }                                                                      \
    }                                                                          \
    default:                                                                   \
        GAPI_Assert(chan <= 4);                                                \
        break;                                                                 \
    }                                                                          \
    return 0;                                                                  \
}

DIVRC_SIMD(uchar, uchar)
DIVRC_SIMD(ushort, uchar)
DIVRC_SIMD(short, uchar)
DIVRC_SIMD(float, uchar)
DIVRC_SIMD(short, short)
DIVRC_SIMD(ushort, short)
DIVRC_SIMD(uchar, short)
DIVRC_SIMD(float, short)
DIVRC_SIMD(ushort, ushort)
DIVRC_SIMD(uchar, ushort)
DIVRC_SIMD(short, ushort)
DIVRC_SIMD(float, ushort)
DIVRC_SIMD(uchar, float)
DIVRC_SIMD(ushort, float)
DIVRC_SIMD(short, float)
DIVRC_SIMD(float, float)

#undef DIVRC_SIMD

//-------------------------
//
// Fluid kernels: Split3
//
//-------------------------

int split3_simd(const uchar in[], uchar out1[], uchar out2[], uchar out3[],
                const int width)
{
    constexpr int nlanes = v_uint8::nlanes;
    if (width < nlanes)
        return 0;

    int x = 0;
    for (;;)
    {
        for (; x <= width - nlanes; x += nlanes)
        {
            v_uint8 a, b, c;
            v_load_deinterleave(&in[3 * x], a, b, c);
            vx_store(&out1[x], a);
            vx_store(&out2[x], b);
            vx_store(&out3[x], c);
        }
        if (x < width)
        {
            x = width - nlanes;
            continue;
        }
        break;
    }
    return x;
}

//-------------------------
//
// Fluid kernels: Split4
//
//-------------------------

int split4_simd(const uchar in[], uchar out1[], uchar out2[],
                uchar out3[], uchar out4[], const int width)
{
    constexpr int nlanes = v_uint8::nlanes;
    if (width < nlanes)
        return 0;

    int x = 0;
    for (;;)
    {
        for (; x <= width - nlanes; x += nlanes)
        {
            v_uint8 a, b, c, d;
            v_load_deinterleave(&in[4 * x], a, b, c, d);
            vx_store(&out1[x], a);
            vx_store(&out2[x], b);
            vx_store(&out3[x], c);
            vx_store(&out4[x], d);
        }
        if (x < width)
        {
            x = width - nlanes;
            continue;
        }
        break;
    }
    return x;
}

//-------------------------
//
// Fluid kernels: Merge3
//
//-------------------------

#define MERGE3_SIMD(T)                                              \
int merge3_simd(const T in1[], const T in2[], const T in3[],        \
                T out[], const int width)                           \
{                                                                   \
    constexpr int nlanes = vector_type_of_t<T>::nlanes;             \
    if (width < nlanes)                                             \
        return 0;                                                   \
                                                                    \
    int x = 0;                                                      \
    for (;;)                                                        \
    {                                                               \
        for (; x <= width - nlanes; x += nlanes)                    \
        {                                                           \
            vector_type_of_t<T> a, b, c;                            \
            a = vx_load(&in1[x]);                                   \
            b = vx_load(&in2[x]);                                   \
            c = vx_load(&in3[x]);                                   \
            v_store_interleave(&out[3 * x], a, b, c);               \
        }                                                           \
        if (x < width)                                              \
        {                                                           \
            x = width - nlanes;                                     \
            continue;                                               \
        }                                                           \
        break;                                                      \
    }                                                               \
    return x;                                                       \
}

MERGE3_SIMD(uchar)
MERGE3_SIMD(short)
MERGE3_SIMD(ushort)
MERGE3_SIMD(float)

#undef MERGE3_SIMD

//-------------------------
//
// Fluid kernels: Merge4
//
//-------------------------

int merge4_simd(const uchar in1[], const uchar in2[], const uchar in3[],
                const uchar in4[], uchar out[], const int width)
{
    constexpr int nlanes = v_uint8::nlanes;
    if (width < nlanes)
        return 0;

    int x = 0;
    for (;;)
    {
        for (; x <= width - nlanes; x += nlanes)
        {
            v_uint8 a, b, c, d;
            a = vx_load(&in1[x]);
            b = vx_load(&in2[x]);
            c = vx_load(&in3[x]);
            d = vx_load(&in4[x]);
            v_store_interleave(&out[4 * x], a, b, c, d);
        }
        if (x < width)
        {
            x = width - nlanes;
            continue;
        }
        break;
    }
    return x;
}

//-------------------------
//
// Fluid kernels: Add
//
//-------------------------
template<typename VT>
CV_ALWAYS_INLINE VT oper(add_tag, const VT& a, const VT& b)
{
    return a + b;
}

template<typename VT>
CV_ALWAYS_INLINE VT oper(sub_tag, const VT& a, const VT& b)
{
    return a - b;
}

CV_ALWAYS_INLINE void pack_store_uchar(uchar* outx, const v_uint16& c1, const v_uint16& c2)
{
    vx_store(outx, v_pack(c1, c2));
}

CV_ALWAYS_INLINE void pack_store_uchar(uchar* outx, const v_int16& c1, const v_int16& c2)
{
    vx_store(outx, v_pack_u(c1, c2));
}

template<typename oper_tag, typename SRC, typename DST>
CV_ALWAYS_INLINE
typename std::enable_if<std::is_same<SRC, DST>::value, void>::type
arithmOp_simd_impl(oper_tag op, const SRC* in1x, const SRC* in2x, DST* outx)
{
    vector_type_of_t<SRC> a = vx_load(in1x);
    vector_type_of_t<SRC> b = vx_load(in2x);
    vx_store(outx, oper(op, a, b));
}

template<typename oper_tag, typename SRC>
CV_ALWAYS_INLINE
typename std::enable_if<std::is_same<SRC, short>::value ||
                        std::is_same<SRC, ushort>::value, void>::type
arithmOp_simd_impl(oper_tag op, const SRC* in1x, const SRC* in2x, uchar* outx)
{
    constexpr int nlanes = v_uint8::nlanes;

    vector_type_of_t<SRC> a1 = vx_load(in1x);
    vector_type_of_t<SRC> a2 = vx_load(&in1x[nlanes / 2]);
    vector_type_of_t<SRC> b1 = vx_load(in2x);
    vector_type_of_t<SRC> b2 = vx_load(&in2x[nlanes / 2]);

    pack_store_uchar(outx, oper(op, a1, b1), oper(op, a2, b2));
}

template<typename oper_tag>
CV_ALWAYS_INLINE void arithmOp_simd_impl(oper_tag op, const float* in1x,
                                         const float* in2x, uchar* outx)
{
    constexpr int nlanes = v_uint8::nlanes;

    v_float32 a1 = vx_load(in1x);
    v_float32 a2 = vx_load(&in1x[nlanes / 4]);
    v_float32 a3 = vx_load(&in1x[2 * nlanes / 4]);
    v_float32 a4 = vx_load(&in1x[3 * nlanes / 4]);

    v_float32 b1 = vx_load(in2x);
    v_float32 b2 = vx_load(&in2x[nlanes / 4]);
    v_float32 b3 = vx_load(&in2x[2 * nlanes / 4]);
    v_float32 b4 = vx_load(&in2x[3 * nlanes / 4]);

    vx_store(outx, v_pack_u(v_pack(v_round(oper(op, a1, b1)), v_round(oper(op, a2, b2))),
                            v_pack(v_round(oper(op, a3, b3)), v_round(oper(op, a4, b4)))));
}

template<typename oper_tag>
CV_ALWAYS_INLINE void arithmOp_simd_impl(oper_tag op, const uchar* in1x,
                                         const uchar* in2x, short* outx)
{
    v_int16 a = v_reinterpret_as_s16(vx_load_expand(in1x));
    v_int16 b = v_reinterpret_as_s16(vx_load_expand(in2x));

    vx_store(outx, oper(op, a, b));
}

template<typename oper_tag>
CV_ALWAYS_INLINE void arithmOp_simd_impl(oper_tag op, const uchar* in1x,
                                         const uchar* in2x, ushort* outx)
{
    v_uint16 a = vx_load_expand(in1x);
    v_uint16 b = vx_load_expand(in2x);

    vx_store(outx, oper(op, a, b));
}

template<typename oper_tag, typename DST>
CV_ALWAYS_INLINE
typename std::enable_if<std::is_same<DST, short>::value ||
                        std::is_same<DST, ushort>::value, void>::type
arithmOp_simd_impl(oper_tag op, const float* in1x, const float* in2x, DST* outx)
{
    constexpr int nlanes = vector_type_of_t<DST>::nlanes;
    v_float32 a1 = vx_load(in1x);
    v_float32 a2 = vx_load(&in1x[nlanes/2]);
    v_float32 b1 = vx_load(in2x);
    v_float32 b2 = vx_load(&in2x[nlanes/2]);

    v_store_i16(outx, v_round(oper(op, a1, b1)), v_round(oper(op, a2, b2)));
}

template<typename oper_tag>
CV_ALWAYS_INLINE void arithmOp_simd_impl(oper_tag op, const short* in1x,
                                         const short* in2x, ushort* outx)
{
    v_int16 a = vx_load(in1x);
    v_int32 a1 = v_expand_low(a);
    v_int32 a2 = v_expand_high(a);

    v_int16 b = vx_load(in2x);
    v_int32 b1 = v_expand_low(b);
    v_int32 b2 = v_expand_high(b);

    vx_store(outx, v_pack_u(oper(op, a1, b1), oper(op, a2, b2)));
}

template<typename oper_tag>
CV_ALWAYS_INLINE void arithmOp_simd_impl(oper_tag op, const ushort* in1x,
                                         const ushort* in2x, short* outx)
{
    v_int16 a = v_reinterpret_as_s16(vx_load(in1x));
    v_int32 a1 = v_expand_low(a);
    v_int32 a2 = v_expand_high(a);

    v_int16 b = v_reinterpret_as_s16(vx_load(in2x));
    v_int32 b1 = v_expand_low(b);
    v_int32 b2 = v_expand_high(b);

    vx_store(outx, v_pack(oper(op, a1, b1), oper(op, a2, b2)));
}

template<typename oper_tag, typename SRC>
CV_ALWAYS_INLINE void arithmOp_simd_impl(oper_tag op, const SRC* in1x, const SRC* in2x, float* outx)
{
    v_float32 a = vg_load_f32(in1x);
    v_float32 b = vg_load_f32(in2x);

    vx_store(outx, oper(op, a, b));
}

template<typename oper_tag, typename SRC, typename DST>
CV_ALWAYS_INLINE int arithmOp_simd(oper_tag op, const SRC in1[], const SRC in2[],
                                   DST out[], const int length)
{
    constexpr int nlanes = vector_type_of_t<DST>::nlanes;

    if (length < nlanes)
        return 0;

    int x = 0;
    for (;;)
    {
        for (; x <= length - nlanes; x += nlanes)
        {
            arithmOp_simd_impl(op, &in1[x], &in2[x], &out[x]);
        }

        if (x < length)
        {
            x = length - nlanes;
            continue;
        }
        break;
    }

    return x;
}

#define ADD_SIMD(SRC, DST)                                                      \
int add_simd(const SRC in1[], const SRC in2[], DST out[], const int length)     \
{                                                                               \
    return arithmOp_simd(add_tag{}, in1, in2, out, length);                     \
}                                                                               \

ADD_SIMD(uchar, uchar)
ADD_SIMD(ushort, uchar)
ADD_SIMD(short, uchar)
ADD_SIMD(float, uchar)
ADD_SIMD(short, short)
ADD_SIMD(ushort, short)
ADD_SIMD(uchar, short)
ADD_SIMD(float, short)
ADD_SIMD(ushort, ushort)
ADD_SIMD(uchar, ushort)
ADD_SIMD(short, ushort)
ADD_SIMD(float, ushort)
ADD_SIMD(uchar, float)
ADD_SIMD(ushort, float)
ADD_SIMD(short, float)
ADD_SIMD(float, float)

#undef ADD_SIMD

//-------------------------
//
// Fluid kernels: Sub
//
//-------------------------

#define SUB_SIMD(SRC, DST)                                                      \
int sub_simd(const SRC in1[], const SRC in2[], DST out[], const int length)     \
{                                                                               \
    return arithmOp_simd(sub_tag{}, in1, in2, out, length);                     \
}                                                                               \

SUB_SIMD(uchar, uchar)
SUB_SIMD(ushort, uchar)
SUB_SIMD(short, uchar)
SUB_SIMD(float, uchar)
SUB_SIMD(short, short)
SUB_SIMD(ushort, short)
SUB_SIMD(uchar, short)
SUB_SIMD(float, short)
SUB_SIMD(ushort, ushort)
SUB_SIMD(uchar, ushort)
SUB_SIMD(short, ushort)
SUB_SIMD(float, ushort)
SUB_SIMD(uchar, float)
SUB_SIMD(ushort, float)
SUB_SIMD(short, float)
SUB_SIMD(float, float)

#undef SUB_SIMD

//-------------------------
//
// Fluid kernels: ConvertTo
//
//-------------------------

CV_ALWAYS_INLINE void store_i16(ushort* outx, const v_uint16& res)
{
    vx_store(outx, res);
}

CV_ALWAYS_INLINE void store_i16(short* outx, const v_uint16& res)
{
    vx_store(outx, v_reinterpret_as_s16(res));
}

CV_ALWAYS_INLINE void store_i16(ushort* outx, const v_int16& res)
{
    vx_store(outx, v_reinterpret_as_u16(res));
}

CV_ALWAYS_INLINE void store_i16(short* outx, const v_int16& res)
{
    vx_store(outx, res);
}

CV_ALWAYS_INLINE void convertto_simd_nocoeff_impl(const float* inx, uchar* outx)
{
    constexpr int nlanes = v_uint8::nlanes;

    v_int32 a1 = v_round(vx_load(inx));
    v_int32 a2 = v_round(vx_load(&inx[nlanes/4]));
    v_int32 a3 = v_round(vx_load(&inx[nlanes/2]));
    v_int32 a4 = v_round(vx_load(&inx[3*nlanes/4]));

    v_int16 r1 = v_pack(a1, a2);
    v_int16 r2 = v_pack(a3, a4);

    vx_store(outx, v_pack_u(r1, r2));
}

template<typename SRC>
CV_ALWAYS_INLINE
typename std::enable_if<SRC_SHORT_OR_USHORT, void>::type
convertto_simd_nocoeff_impl(const SRC* inx, uchar* outx)
{
    constexpr int nlanes = v_uint8::nlanes;

    vector_type_of_t<SRC> a1 = vx_load(inx);
    vector_type_of_t<SRC> a2 = vx_load(&inx[nlanes/2]);

    pack_store_uchar(outx, a1, a2);
}

//---------------------------------------------------------------------------------------

template<typename DST>
CV_ALWAYS_INLINE
typename std::enable_if<DST_SHORT_OR_USHORT, void>::type
convertto_simd_nocoeff_impl(const float* inx, DST* outx)
{
    constexpr int nlanes = vector_type_of_t<DST>::nlanes;

    v_int32 a1 = v_round(vx_load(inx));
    v_int32 a2 = v_round(vx_load(&inx[nlanes/2]));

    v_store_i16(outx, a1, a2);
}

template<typename DST>
CV_ALWAYS_INLINE
typename std::enable_if<DST_SHORT_OR_USHORT, void>::type
convertto_simd_nocoeff_impl(const uchar* inx, DST* outx)
{
    v_uint8 a = vx_load(inx);
    v_uint16 res = v_expand_low(a);

    store_i16(outx, res);
}

template<typename SRC, typename DST>
CV_ALWAYS_INLINE
typename std::enable_if<SRC_DST_SHORT_AND_USHORT, void>::type
convertto_simd_nocoeff_impl(const SRC* inx, DST* outx)
{
    vector_type_of_t<SRC> a = vx_load(inx);
    store_i16(outx, a);
}

//---------------------------------------------------------------------------------------

template<typename SRC>
CV_ALWAYS_INLINE void convertto_simd_nocoeff_impl(const SRC* inx, float* outx)
{
    v_float32 a = vg_load_f32(inx);
    vx_store(outx, a);
}

#define CONVERTTO_NOCOEF_SIMD(SRC, DST)                            \
int convertto_simd(const SRC in[], DST out[], const int length)    \
{                                                                  \
    constexpr int nlanes = vector_type_of_t<DST>::nlanes;          \
    if (length < nlanes)                                           \
        return 0;                                                  \
                                                                   \
    int x = 0;                                                     \
    for (;;)                                                       \
    {                                                              \
        for (; x <= length - nlanes; x += nlanes)                  \
        {                                                          \
            convertto_simd_nocoeff_impl(&in[x], &out[x]);          \
        }                                                          \
        if (x < length)                                            \
        {                                                          \
            x = length - nlanes;                                   \
            continue;                                              \
        }                                                          \
        break;                                                     \
    }                                                              \
    return x;                                                      \
}

CONVERTTO_NOCOEF_SIMD(ushort, uchar)
CONVERTTO_NOCOEF_SIMD(short, uchar)
CONVERTTO_NOCOEF_SIMD(float, uchar)
CONVERTTO_NOCOEF_SIMD(ushort, short)
CONVERTTO_NOCOEF_SIMD(uchar, short)
CONVERTTO_NOCOEF_SIMD(float, short)
CONVERTTO_NOCOEF_SIMD(uchar, ushort)
CONVERTTO_NOCOEF_SIMD(short, ushort)
CONVERTTO_NOCOEF_SIMD(float, ushort)
CONVERTTO_NOCOEF_SIMD(uchar, float)
CONVERTTO_NOCOEF_SIMD(ushort, float)
CONVERTTO_NOCOEF_SIMD(short, float)

#undef CONVERTTO_NOCOEF_SIMD

CV_ALWAYS_INLINE void convertto_scaled_simd_impl(const float* inx, uchar* outx,
                                                 const v_float32& v_alpha,
                                                 const v_float32& v_beta)
{
    constexpr int nlanes = v_uint8::nlanes;

    v_float32 a1 = vx_load(inx);
    v_float32 a2 = vx_load(&inx[nlanes / 4]);
    v_float32 a3 = vx_load(&inx[nlanes / 2]);
    v_float32 a4 = vx_load(&inx[3 * nlanes / 4]);

    v_int32 r1 = v_round(v_fma(a1, v_alpha, v_beta));
    v_int32 r2 = v_round(v_fma(a2, v_alpha, v_beta));
    v_int32 r3 = v_round(v_fma(a3, v_alpha, v_beta));
    v_int32 r4 = v_round(v_fma(a4, v_alpha, v_beta));

    vx_store(outx, v_pack_u(v_pack(r1, r2), v_pack(r3, r4)));
}

template<typename SRC>
CV_ALWAYS_INLINE
typename std::enable_if<SRC_SHORT_OR_USHORT, void>::type
convertto_scaled_simd_impl(const SRC* inx, uchar* outx, const v_float32& v_alpha,
                           const v_float32& v_beta)
{
    constexpr int nlanes = v_uint8::nlanes;

    v_int16 a = v_reinterpret_as_s16(vx_load(inx));
    v_int16 b = v_reinterpret_as_s16(vx_load(&inx[nlanes / 2]));

    v_float32 a1 = v_cvt_f32(v_expand_low(a));
    v_float32 a2 = v_cvt_f32(v_expand_high(a));
    v_float32 b1 = v_cvt_f32(v_expand_low(b));
    v_float32 b2 = v_cvt_f32(v_expand_high(b));

    v_int32 r1 = v_round(v_fma(a1, v_alpha, v_beta));
    v_int32 r2 = v_round(v_fma(a2, v_alpha, v_beta));
    v_int32 r3 = v_round(v_fma(b1, v_alpha, v_beta));
    v_int32 r4 = v_round(v_fma(b2, v_alpha, v_beta));

    vx_store(outx, v_pack_u(v_pack(r1, r2), v_pack(r3, r4)));
}

CV_ALWAYS_INLINE void convertto_scaled_simd_impl(const uchar* inx, uchar* outx,
                                                 const v_float32& v_alpha,
                                                 const v_float32& v_beta)
{
    v_uint8 a = vx_load(inx);
    v_int16 a1 = v_reinterpret_as_s16(v_expand_low(a));
    v_int16 a2 = v_reinterpret_as_s16(v_expand_high(a));

    v_float32 f1 = v_cvt_f32(v_expand_low(a1));
    v_float32 f2 = v_cvt_f32(v_expand_high(a1));

    v_float32 f3 = v_cvt_f32(v_expand_low(a2));
    v_float32 f4 = v_cvt_f32(v_expand_high(a2));

    v_int32 r1 = v_round(v_fma(f1, v_alpha, v_beta));
    v_int32 r2 = v_round(v_fma(f2, v_alpha, v_beta));
    v_int32 r3 = v_round(v_fma(f3, v_alpha, v_beta));
    v_int32 r4 = v_round(v_fma(f4, v_alpha, v_beta));

    vx_store(outx, v_pack_u(v_pack(r1, r2), v_pack(r3, r4)));
}

template<typename DST>
CV_ALWAYS_INLINE
typename std::enable_if<DST_SHORT_OR_USHORT, void>::type
convertto_scaled_simd_impl(const float* inx, DST* outx,
                           const v_float32& v_alpha,
                           const v_float32& v_beta)
{
    constexpr int nlanes = vector_type_of_t<DST>::nlanes;

    v_float32 a1 = vx_load(inx);
    v_float32 a2 = vx_load(&inx[nlanes / 2]);

    v_int32 r1 = v_round(v_fma(a1, v_alpha, v_beta));
    v_int32 r2 = v_round(v_fma(a2, v_alpha, v_beta));

    v_store_i16(outx, r1, r2);
}

template<typename DST>
CV_ALWAYS_INLINE
typename std::enable_if<DST_SHORT_OR_USHORT, void>::type
convertto_scaled_simd_impl(const uchar* inx, DST* outx,
                           const v_float32& v_alpha,
                           const v_float32& v_beta)
{
    v_int16 a = v_reinterpret_as_s16(vx_load_expand(inx));

    v_float32 a1 = v_cvt_f32(v_expand_low(a));
    v_float32 a2 = v_cvt_f32(v_expand_high(a));

    v_int32 r1 = v_round(v_fma(a1, v_alpha, v_beta));
    v_int32 r2 = v_round(v_fma(a2, v_alpha, v_beta));

    v_store_i16(outx, r1, r2);
}

template<typename SRC, typename DST>
CV_ALWAYS_INLINE
typename std::enable_if<SRC_DST_SHORT_AND_USHORT ||
                        SRC_DST_SHORT_OR_USHORT, void>::type
convertto_scaled_simd_impl(const SRC* inx, DST* outx,
                           const v_float32& v_alpha,
                           const v_float32& v_beta)
{
    v_int16 a = v_reinterpret_as_s16(vx_load(inx));

    v_float32 a1 = v_cvt_f32(v_expand_low(a));
    v_float32 a2 = v_cvt_f32(v_expand_high(a));

    v_int32 r1 = v_round(v_fma(a1, v_alpha, v_beta));
    v_int32 r2 = v_round(v_fma(a2, v_alpha, v_beta));

    v_store_i16(outx, r1, r2);
}

template<typename SRC>
CV_ALWAYS_INLINE void convertto_scaled_simd_impl(const SRC* inx, float* outx,
                                                 const v_float32& v_alpha,
                                                 const v_float32& v_beta)
{
    v_float32 a = vg_load_f32(inx);
    vx_store(outx, v_fma(a, v_alpha, v_beta));
}

#define CONVERTTO_SCALED_SIMD(SRC, DST)                                     \
int convertto_scaled_simd(const SRC in[], DST out[], const float alpha,     \
                          const float beta, const int length)               \
{                                                                           \
    constexpr int nlanes = vector_type_of_t<DST>::nlanes;                   \
    if (length < nlanes)                                                    \
        return 0;                                                           \
                                                                            \
    v_float32 v_alpha = vx_setall_f32(alpha);                               \
    v_float32 v_beta = vx_setall_f32(beta);                                 \
                                                                            \
    int x = 0;                                                              \
    for (;;)                                                                \
    {                                                                       \
        for (; x <= length - nlanes; x += nlanes)                           \
        {                                                                   \
            convertto_scaled_simd_impl(&in[x], &out[x], v_alpha, v_beta);   \
        }                                                                   \
        if (x < length)                                                     \
        {                                                                   \
            x = length - nlanes;                                            \
            continue;                                                       \
        }                                                                   \
        break;                                                              \
    }                                                                       \
    return x;                                                               \
}

CONVERTTO_SCALED_SIMD(uchar, uchar)
CONVERTTO_SCALED_SIMD(ushort, uchar)
CONVERTTO_SCALED_SIMD(short, uchar)
CONVERTTO_SCALED_SIMD(float, uchar)
CONVERTTO_SCALED_SIMD(short, short)
CONVERTTO_SCALED_SIMD(ushort, short)
CONVERTTO_SCALED_SIMD(uchar, short)
CONVERTTO_SCALED_SIMD(float, short)
CONVERTTO_SCALED_SIMD(ushort, ushort)
CONVERTTO_SCALED_SIMD(uchar, ushort)
CONVERTTO_SCALED_SIMD(short, ushort)
CONVERTTO_SCALED_SIMD(float, ushort)
CONVERTTO_SCALED_SIMD(uchar, float)
CONVERTTO_SCALED_SIMD(ushort, float)
CONVERTTO_SCALED_SIMD(short, float)
CONVERTTO_SCALED_SIMD(float, float)

#undef CONVERTTO_SCALED_SIMD

#endif  // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END

}  // namespace fluid
}  // namespace gapi
}  // namespace cv

#endif // !defined(GAPI_STANDALONE)
