// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation

// NB: allow including this *.hpp several times!
// #pragma once -- don't: this file is NOT once!

#if !defined(GAPI_STANDALONE)

#include "gfluidimgproc_func.hpp"

#include "opencv2/gapi/own/saturate.hpp"

#include "opencv2/core.hpp"
#include "opencv2/core/hal/intrin.hpp"

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

//----------------------------------
//
// Fluid kernels: RGB2Gray, BGR2Gray
//
//----------------------------------

void run_rgb2gray_impl(uchar out[], const uchar in[], int width,
                       float coef_r, float coef_g, float coef_b);

//--------------------------------------
//
// Fluid kernels: RGB-to-HSV
//
//--------------------------------------

void run_rgb2hsv_impl(uchar out[], const uchar in[], const int sdiv_table[],
                      const int hdiv_table[], int width);

//--------------------------------------
//
// Fluid kernels: RGB-to-BayerGR
//
//--------------------------------------

void run_bayergr2rgb_bg_impl(uchar out[], const uchar **in, int width);

void run_bayergr2rgb_gr_impl(uchar out[], const uchar **in, int width);

//--------------------------------------
//
// Fluid kernels: RGB-to-YUV, RGB-to-YUV422, YUV-to-RGB
//
//--------------------------------------

void run_rgb2yuv_impl(uchar out[], const uchar in[], int width, const float coef[5]);

void run_yuv2rgb_impl(uchar out[], const uchar in[], int width, const float coef[4]);

void run_rgb2yuv422_impl(uchar out[], const uchar in[], int width);

//-------------------------
//
// Fluid kernels: sepFilter
//
//-------------------------

#define RUN_SEPFILTER3X3_IMPL(DST, SRC)                                     \
void run_sepfilter3x3_impl(DST out[], const SRC *in[], int width, int chan, \
                           const float kx[], const float ky[], int border,  \
                           float scale, float delta,                        \
                           float *buf[], int y, int y0);

RUN_SEPFILTER3X3_IMPL(uchar , uchar )
RUN_SEPFILTER3X3_IMPL( short, uchar )
RUN_SEPFILTER3X3_IMPL( float, uchar )
RUN_SEPFILTER3X3_IMPL(ushort, ushort)
RUN_SEPFILTER3X3_IMPL( short, ushort)
RUN_SEPFILTER3X3_IMPL( float, ushort)
RUN_SEPFILTER3X3_IMPL( short,  short)
RUN_SEPFILTER3X3_IMPL( float,  short)
RUN_SEPFILTER3X3_IMPL( float,  float)

#undef RUN_SEPFILTER3X3_IMPL

#define RUN_SEPFILTER5x5_IMPL(DST, SRC)                                     \
void run_sepfilter5x5_impl(DST out[], const SRC *in[], int width, int chan, \
                           const float kx[], const float ky[], int border,  \
                           float scale, float delta,                        \
                           float *buf[], int y, int y0);

RUN_SEPFILTER5x5_IMPL(uchar, uchar)
RUN_SEPFILTER5x5_IMPL(short, uchar)
RUN_SEPFILTER5x5_IMPL(float, uchar)
RUN_SEPFILTER5x5_IMPL(ushort, ushort)
RUN_SEPFILTER5x5_IMPL(short, ushort)
RUN_SEPFILTER5x5_IMPL(float, ushort)
RUN_SEPFILTER5x5_IMPL(short, short)
RUN_SEPFILTER5x5_IMPL(float, short)
RUN_SEPFILTER5x5_IMPL(float, float)

#undef RUN_SEPFILTER5x5_IMPL
//-------------------------
//
// Fluid kernels: Filter 2D
//
//-------------------------

#define RUN_FILTER2D_3X3_IMPL(DST, SRC)                                     \
void run_filter2d_3x3_impl(DST out[], const SRC *in[], int width, int chan, \
                           const float kernel[], float scale, float delta);

RUN_FILTER2D_3X3_IMPL(uchar , uchar )
RUN_FILTER2D_3X3_IMPL(ushort, ushort)
RUN_FILTER2D_3X3_IMPL( short,  short)
RUN_FILTER2D_3X3_IMPL( float, uchar )
RUN_FILTER2D_3X3_IMPL( float, ushort)
RUN_FILTER2D_3X3_IMPL( float,  short)
RUN_FILTER2D_3X3_IMPL( float,  float)

#undef RUN_FILTER2D_3X3_IMPL

//-----------------------------
//
// Fluid kernels: Erode, Dilate
//
//-----------------------------

#define RUN_MORPHOLOGY3X3_IMPL(T)                                        \
void run_morphology3x3_impl(T out[], const T *in[], int width, int chan, \
                            const uchar k[], MorphShape k_type,          \
                            Morphology morphology);

RUN_MORPHOLOGY3X3_IMPL(uchar )
RUN_MORPHOLOGY3X3_IMPL(ushort)
RUN_MORPHOLOGY3X3_IMPL( short)
RUN_MORPHOLOGY3X3_IMPL( float)

#undef RUN_MORPHOLOGY3X3_IMPL

//---------------------------
//
// Fluid kernels: Median blur
//
//---------------------------

#define RUN_MEDBLUR3X3_IMPL(T) \
void run_medblur3x3_impl(T out[], const T *in[], int width, int chan);

RUN_MEDBLUR3X3_IMPL(uchar )
RUN_MEDBLUR3X3_IMPL(ushort)
RUN_MEDBLUR3X3_IMPL( short)
RUN_MEDBLUR3X3_IMPL( float)

#undef RUN_MEDBLUR3X3_IMPL

//----------------------------------------------------------------------

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

#if CV_SIMD
template<typename SRC>
static inline v_float32 vx_load_f32(const SRC* ptr)
{
    if (std::is_same<SRC,uchar>::value)
    {
        v_uint32 tmp = vx_load_expand_q(reinterpret_cast<const uchar*>(ptr));
        return v_cvt_f32(v_reinterpret_as_s32(tmp));
    }

    if (std::is_same<SRC,ushort>::value)
    {
        v_uint32 tmp = vx_load_expand(reinterpret_cast<const ushort*>(ptr));
        return v_cvt_f32(v_reinterpret_as_s32(tmp));
    }

    if (std::is_same<SRC,short>::value)
    {
        v_int32 tmp = vx_load_expand(reinterpret_cast<const short*>(ptr));
        return v_cvt_f32(tmp);
    }

    if (std::is_same<SRC,float>::value)
    {
        v_float32 tmp = vx_load(reinterpret_cast<const float*>(ptr));
        return tmp;
    }

    CV_Error(cv::Error::StsBadArg, "unsupported type");
}
#endif  // CV_SIMD

//----------------------------------
//
// Fluid kernels: RGB2Gray, BGR2Gray
//
//----------------------------------

void run_rgb2gray_impl(uchar out[], const uchar in[], int width,
                       float coef_r, float coef_g, float coef_b)
{
    // assume:
    // - coefficients are less than 1
    // - and their sum equals 1

    constexpr int unity = 1 << 16;  // Q0.0.16 inside ushort:
    ushort rc = static_cast<ushort>(coef_r * unity + 0.5f);
    ushort gc = static_cast<ushort>(coef_g * unity + 0.5f);
    ushort bc = static_cast<ushort>(coef_b * unity + 0.5f);

    GAPI_Assert(rc + gc + bc <= unity);
    GAPI_Assert(rc + gc + bc >= USHRT_MAX);

#if CV_SIMD
    constexpr int nlanes = v_uint8::nlanes;
    if (width >= nlanes)
    {
        for (int w=0; w < width; )
        {
            // process main part of pixels row
            for ( ; w <= width - nlanes; w += nlanes)
            {
                v_uint8 r, g, b;
                v_load_deinterleave(&in[3*w], r, g, b);

                v_uint16 r0, r1, g0, g1, b0, b1;
                v_expand(r, r0, r1);
                v_expand(g, g0, g1);
                v_expand(b, b0, b1);

                v_uint16 y0, y1;
                static const ushort half = 1 << 7; // Q0.8.8
                y0 = (v_mul_hi(r0 << 8, vx_setall_u16(rc)) +
                      v_mul_hi(g0 << 8, vx_setall_u16(gc)) +
                      v_mul_hi(b0 << 8, vx_setall_u16(bc)) +
                                        vx_setall_u16(half)) >> 8;
                y1 = (v_mul_hi(r1 << 8, vx_setall_u16(rc)) +
                      v_mul_hi(g1 << 8, vx_setall_u16(gc)) +
                      v_mul_hi(b1 << 8, vx_setall_u16(bc)) +
                                        vx_setall_u16(half)) >> 8;

                v_uint8 y;
                y = v_pack(y0, y1);
                v_store(&out[w], y);
            }

            // process tail (if any)
            if (w < width)
            {
                GAPI_DbgAssert(width - nlanes >= 0);
                w = width - nlanes;
            }
        }

        return;
    }
#endif

    for (int w=0; w < width; w++)
    {
        uchar r = in[3*w    ];
        uchar g = in[3*w + 1];
        uchar b = in[3*w + 2];

        static const int half = 1 << 15;  // Q0.0.16
        ushort y = (r*rc + b*bc + g*gc + half) >> 16;
        out[w] = static_cast<uchar>(y);
    }
}

//--------------------------------------
//
// Fluid kernels: RGB-to-HSV
//
//--------------------------------------
//
void run_rgb2hsv_impl(uchar out[], const uchar in[], const int sdiv_table[],
                      const int hdiv_table[], int width)
{
    const int hsv_shift = 12;
    const int hr = 180;

    int j = 0;

    #if CV_SIMD128
        const int vectorStep = 16;

        uint8_t ff = 0xff;
        v_uint8x16 mask1(ff, 0, 0, 0, ff, 0, 0, 0, ff, 0, 0, 0, ff, 0, 0, 0);
        v_uint8x16 mask2(0, ff, 0, 0, 0, ff, 0, 0, 0, ff, 0, 0, 0, ff, 0, 0);
        v_uint8x16 mask3(0, 0, ff, 0, 0, 0, ff, 0, 0, 0, ff, 0, 0, 0, ff, 0);
        v_uint8x16 mask4(0, 0, 0, ff, 0, 0, 0, ff, 0, 0, 0, ff, 0, 0, 0, ff);

        for (int w = 0; w <= 3 * (width - vectorStep); w += 3 * vectorStep)
        {
            v_uint8x16 r, g, b;
            v_load_deinterleave(in + w, r, g, b);

            v_uint8x16 v_min_rgb = v_min(v_min(r, g), b);
            v_uint8x16 v_max_rgb = v_max(v_max(r, g), b);

            v_uint8x16 v_diff = v_max_rgb - v_min_rgb;

            v_uint8x16 v_r_eq_max = (r == v_max_rgb);
            v_uint8x16 v_g_eq_max = (g == v_max_rgb);

            v_uint8x16 v;
            // get V-ch
            v = v_max_rgb;

            // divide v into 4x4 vectors because later int32 required
            v_uint32x4 v_idx[4];
            v_idx[0] = v_reinterpret_as_u32(v & mask1);
            v_idx[1] = v_reinterpret_as_u32(v & mask2) >> 8;
            v_idx[2] = v_reinterpret_as_u32(v & mask3) >> 16;
            v_idx[3] = v_reinterpret_as_u32(v & mask4) >> 24;

            v_uint32x4 sv_elems_32[4];
            sv_elems_32[0] = v_reinterpret_as_u32(v_lut(sdiv_table, v_reinterpret_as_s32(v_idx[0])));
            sv_elems_32[1] = v_reinterpret_as_u32(v_lut(sdiv_table, v_reinterpret_as_s32(v_idx[1])));
            sv_elems_32[2] = v_reinterpret_as_u32(v_lut(sdiv_table, v_reinterpret_as_s32(v_idx[2])));
            sv_elems_32[3] = v_reinterpret_as_u32(v_lut(sdiv_table, v_reinterpret_as_s32(v_idx[3])));

            // divide and calculate s according to above feature
            v_uint32x4 ss[4];

            v_uint32x4 v_add = v_setall_u32(1) << (hsv_shift - 1);

            v_uint32x4 v_diff_exp[4];
            v_diff_exp[0] = v_reinterpret_as_u32(v_reinterpret_as_u8(v_diff) & mask1);
            v_diff_exp[1] = v_reinterpret_as_u32(v_reinterpret_as_u8(v_diff) & mask2) >> 8;
            v_diff_exp[2] = v_reinterpret_as_u32(v_reinterpret_as_u8(v_diff) & mask3) >> 16;
            v_diff_exp[3] = v_reinterpret_as_u32(v_reinterpret_as_u8(v_diff) & mask4) >> 24;

            // s = (diff * sdiv_table[v] + (1 << (hsv_shift-1))) >> hsv_shift;
            ss[0] = (v_diff_exp[0] * sv_elems_32[0] + v_add) >> hsv_shift;
            ss[1] = (v_diff_exp[1] * sv_elems_32[1] + v_add) >> hsv_shift;
            ss[2] = (v_diff_exp[2] * sv_elems_32[2] + v_add) >> hsv_shift;
            ss[3] = (v_diff_exp[3] * sv_elems_32[3] + v_add) >> hsv_shift;

            // reconstruct order of S-ch
            v_uint32x4 zip[8];
            v_zip(ss[0], ss[2], zip[0], zip[1]);
            v_zip(ss[1], ss[3], zip[2], zip[3]);

            v_zip(zip[0], zip[2], zip[4], zip[5]);
            v_zip(zip[1], zip[3], zip[6], zip[7]);

            v_uint8x16 s = v_pack(v_pack(zip[4], zip[5]), v_pack(zip[6], zip[7]));

            // the same divination for H-ch
            // FIXME: REALLY UGLY and slow
            v_uint32x4 gg[4];
            v_uint16x8 tmp_exp[2];
            v_expand(g, tmp_exp[0], tmp_exp[1]);
            v_expand(tmp_exp[0], gg[0], gg[1]);
            v_expand(tmp_exp[1], gg[2], gg[3]);

            v_uint32x4 rr[4];
            v_expand(r, tmp_exp[0], tmp_exp[1]);
            v_expand(tmp_exp[0], rr[0], rr[1]);
            v_expand(tmp_exp[1], rr[2], rr[3]);

            v_uint32x4 bb[4];
            v_expand(b, tmp_exp[0], tmp_exp[1]);
            v_expand(tmp_exp[0], bb[0], bb[1]);
            v_expand(tmp_exp[1], bb[2], bb[3]);

            v_int32x4 e[4];
            v_int16x8 sig_exp[2];
            v_expand(v_reinterpret_as_s8(v_r_eq_max), sig_exp[0], sig_exp[1]);
            v_expand(sig_exp[0], e[0], e[1]);
            v_expand(sig_exp[1], e[2], e[3]);

            v_int32x4 p[4];
            v_expand(v_reinterpret_as_s8(v_g_eq_max), sig_exp[0], sig_exp[1]);
            v_expand(sig_exp[0], p[0], p[1]);
            v_expand(sig_exp[1], p[2], p[3]);

            // reconstruct order of v_diff
            v_zip(v_diff_exp[0], v_diff_exp[2], zip[0], zip[1]);
            v_zip(v_diff_exp[1], v_diff_exp[3], zip[2], zip[3]);

            v_zip(zip[0], zip[2], zip[4], zip[5]);
            v_zip(zip[1], zip[3], zip[6], zip[7]);

            v_uint8x16 vd = v_pack(v_pack(zip[4], zip[5]), v_pack(zip[6], zip[7]));

            v_uint32x4 vdd[4];
            v_uint16x8 vvdd[2];
            v_expand(vd, vvdd[0], vvdd[1]);
            v_expand(vvdd[0], vdd[0], vdd[1]);
            v_expand(vvdd[1], vdd[2], vdd[3]);

            // start computing H-ch
            //h = (_vr & (g - b)) + (~_vr & ((_vg & (b - r + 2 * diff)) + ((~_vg) & (r - g + 4 * diff))));
            v_int32x4 hh[4];
            hh[0] = v_reinterpret_as_s32(v_select(e[0], v_reinterpret_as_s32(gg[0] - bb[0]),
                                         v_select(p[0], v_reinterpret_as_s32(bb[0] - rr[0] + v_setall_u32(2) * vdd[0]),
                                                        v_reinterpret_as_s32(rr[0] - gg[0] + v_setall_u32(4) * vdd[0]))));
            hh[1] = v_reinterpret_as_s32(v_select(e[1], v_reinterpret_as_s32(gg[1] - bb[1]),
                                         v_select(p[1], v_reinterpret_as_s32(bb[1] - rr[1] + v_setall_u32(2) * vdd[1]),
                                                        v_reinterpret_as_s32(rr[1] - gg[1] + v_setall_u32(4) * vdd[1]))));
            hh[2] = v_reinterpret_as_s32(v_select(e[2], v_reinterpret_as_s32(gg[2] - bb[2]),
                                         v_select(p[2], v_reinterpret_as_s32(bb[2] - rr[2] + v_setall_u32(2) * vdd[2]),
                                                        v_reinterpret_as_s32(rr[2] - gg[2] + v_setall_u32(4) * vdd[2]))));
            hh[3] = v_reinterpret_as_s32(v_select(e[3], v_reinterpret_as_s32(gg[3] - bb[3]),
                                         v_select(p[3], v_reinterpret_as_s32(bb[3] - rr[3] + v_setall_u32(2) * vdd[3]),
                                                        v_reinterpret_as_s32(rr[3] - gg[3] + v_setall_u32(4) * vdd[3]))));

            //h = (h * hdiv_table[diff] + (1 << (hsv_shift-1))) >> hsv_shift;
            v_uint32x4 h_elems_32[4];
            h_elems_32[0] = v_reinterpret_as_u32(v_lut(hdiv_table, v_reinterpret_as_s32(vdd[0])));
            h_elems_32[1] = v_reinterpret_as_u32(v_lut(hdiv_table, v_reinterpret_as_s32(vdd[1])));
            h_elems_32[2] = v_reinterpret_as_u32(v_lut(hdiv_table, v_reinterpret_as_s32(vdd[2])));
            h_elems_32[3] = v_reinterpret_as_u32(v_lut(hdiv_table, v_reinterpret_as_s32(vdd[3])));

            hh[0] = (hh[0] * v_reinterpret_as_s32(h_elems_32[0]) + v_reinterpret_as_s32(v_add)) >> hsv_shift;
            hh[1] = (hh[1] * v_reinterpret_as_s32(h_elems_32[1]) + v_reinterpret_as_s32(v_add)) >> hsv_shift;
            hh[2] = (hh[2] * v_reinterpret_as_s32(h_elems_32[2]) + v_reinterpret_as_s32(v_add)) >> hsv_shift;
            hh[3] = (hh[3] * v_reinterpret_as_s32(h_elems_32[3]) + v_reinterpret_as_s32(v_add)) >> hsv_shift;

            // check for negative H
            v_int32x4 v_h_less_0[4];
            v_h_less_0[0] = (hh[0] < v_setall_s32(0));
            v_h_less_0[1] = (hh[1] < v_setall_s32(0));
            v_h_less_0[2] = (hh[2] < v_setall_s32(0));
            v_h_less_0[3] = (hh[3] < v_setall_s32(0));

            v_int32x4 v_h_180[4];
            v_h_180[0] = hh[0] + v_setall_s32(180);
            v_h_180[1] = hh[1] + v_setall_s32(180);
            v_h_180[2] = hh[2] + v_setall_s32(180);
            v_h_180[3] = hh[3] + v_setall_s32(180);

            hh[0] = v_select(v_h_less_0[0], v_h_180[0], hh[0]);
            hh[1] = v_select(v_h_less_0[1], v_h_180[1], hh[1]);
            hh[2] = v_select(v_h_less_0[2], v_h_180[2], hh[2]);
            hh[3] = v_select(v_h_less_0[3], v_h_180[3], hh[3]);

            // pack H-ch
            v_uint16x8 hh_16_1 = v_pack(v_reinterpret_as_u32(hh[0]), v_reinterpret_as_u32(hh[1]));
            v_uint16x8 hh_16_2 = v_pack(v_reinterpret_as_u32(hh[2]), v_reinterpret_as_u32(hh[3]));

            v_uint8x16 h = v_pack(hh_16_1, hh_16_2);

            v_store_interleave(out + w, h, s, v);

            // output offset
            j += vectorStep;
        }
    v_cleanup();
    #endif

    for (; j < width; ++j)
    {
        int r = in[j * 3    ],
            g = in[j * 3 + 1],
            b = in[j * 3 + 2];

        int h, s, v = b;
        int vmin = std::min({r, g, b});
        v = std::max({r, g, b});
        int _vr, _vg;

        uchar diff = cv::saturate_cast<uchar>(v - vmin);
        _vr = v == r ? -1 : 0;
        _vg = v == g ? -1 : 0;

        s = (diff * sdiv_table[v] + (1 << (hsv_shift-1))) >> hsv_shift;

        h = (_vr & (g - b)) +
            (~_vr & ((_vg & (b - r + 2 * diff)) + ((~_vg) & (r - g + 4 * diff))));

        h = (h * hdiv_table[diff] + (1 << (hsv_shift-1))) >> hsv_shift;
        h += h < 0 ? hr : 0;

        out[j * 3    ] = cv::saturate_cast<uchar>(h);
        out[j * 3 + 1] = (uchar)(s);
        out[j * 3 + 2] = (uchar)(v);
    }
}

//--------------------------------------
//
// Fluid kernels: RGB-to-BayerGR
//
//--------------------------------------

void run_bayergr2rgb_bg_impl(uchar out[], const uchar **in, int width)
{

    int j = 0;

    #if CV_SIMD128
        const int vectorStep = 16;

        v_uint16x8 l_1, r_1, l_2, r_2;
        v_uint16x8 l_3, r_3, l_4, r_4;

        for (int w = 0; w <= width - 2 * vectorStep - 2; w += 2 * vectorStep) // -2 for offset vectors
        {
            v_uint8x16 g1, r1, g1_offset, r1_offset; // 1 line
            v_uint8x16 b2, g2, b2_offset, g2_offset; // 2 line
            v_uint8x16 g3, r3, g3_offset, r3_offset; // 3 line

            v_load_deinterleave(in[0] + w + 1, r1, g1);
            v_load_deinterleave(in[0] + w + 2 + 1, r1_offset, g1_offset);

            v_load_deinterleave(in[1] + w, b2, g2);
            v_load_deinterleave(in[1] + w + 2, b2_offset, g2_offset);

            v_load_deinterleave(in[2] + w + 1, r3, g3);
            v_load_deinterleave(in[2] + w + 2 + 1, r3_offset, g3_offset);


            // calculate b-channel
            v_expand(b2, l_1, r_1);
            v_expand(b2_offset, l_2, r_2);
            v_uint8x16 b2_sum = v_rshr_pack<1>(l_1 + l_2, r_1 + r_2);

            v_uint8x16 b_low, b_high;
            v_zip(b2_sum, b2_offset, b_low, b_high);


            // calculate r-channel
            v_expand(r1, l_1, r_1);
            v_expand(r1_offset, l_2, r_2);
            v_expand(r3, l_3, r_3);
            v_expand(r3_offset, l_4, r_4);

            v_uint8x16 r13offset_sum, r13_sum;
            r13offset_sum = v_rshr_pack<2>(l_1 + l_2 + l_3 + l_4,
                                           r_1 + r_2 + r_3 + r_4);
            r13_sum = v_rshr_pack<1>(l_1 + l_3, r_1 + r_3);

            v_uint8x16 r_low, r_high;
            v_zip(r13_sum, r13offset_sum, r_low, r_high);


            // calculate g-channel
            v_expand(g1, l_1, r_1);
            v_expand(g3, l_2, r_2);
            v_expand(g2, l_3, r_3);
            v_expand(g2_offset, l_4, r_4);

            v_uint8x16 g_out_sum = v_rshr_pack<2>(l_1 + l_2 + l_3 + l_4,
                                                  r_1 + r_2 + r_3 + r_4);

            v_uint8x16 g_low, g_high;
            v_zip(g2, g_out_sum, g_low, g_high);


            v_store_interleave(out + w * 3 + 3, b_low, g_low, r_low);
            v_store_interleave(out + w * 3 + vectorStep * 3 + 3, b_high, g_high, r_high);

            // output offset for scalar code
            j += vectorStep * 2;
        }
    #endif

    bool curr_red = true;
    int t0, t1, t2;

    int i = 1;

    for (; j < width - 1; ++j, curr_red = !curr_red)
    {
        if (!curr_red)
        {
            t0 = (in[i][j - 1] + in[i][j + 1] + 1) >> 1;
            t1 =  in[i][j];
            t2 = (in[i - 1][j] + in[i + 1][j] + 1) >> 1;


            out[j * 3 + 0] = (uchar)t0;
            out[j * 3 + 1] = (uchar)t1;
            out[j * 3 + 2] = (uchar)t2;
        }
        else
        {
            t2 = (in[i - 1][j - 1] + in[i - 1][j + 1] +
                  in[i + 1][j - 1] + in[i + 1][j + 1] + 2) >> 2;
            t1 = (in[i][j - 1] + in[i][j + 1] +
                  in[i - 1][j] + in[i + 1][j] + 2) >> 2;
            t0 = in[i][j];

            out[j * 3 + 0] = (uchar)t0;
            out[j * 3 + 1] = (uchar)t1;
            out[j * 3 + 2] = (uchar)t2;
        }
    }

    out[0] = out[3];
    out[1] = out[4];
    out[2] = out[5];

    out[3 * (width - 1)    ] = out[3 * (width - 2)    ];
    out[3 * (width - 1) + 1] = out[3 * (width - 2) + 1];
    out[3 * (width - 1) + 2] = out[3 * (width - 2) + 2];
}

void run_bayergr2rgb_gr_impl(uchar out[], const uchar **in, int width)
{

    int j = 0;

    #if CV_SIMD128
        const int vectorStep = 16;

        v_uint16x8 l_1, r_1, l_2, r_2;
        v_uint16x8 l_3, r_3, l_4, r_4;

        for (int w = 0; w <= width - 2 * vectorStep - 2; w += 2 * vectorStep) // -2 for offset vectors
        {
            v_uint8x16 b1, g1, b1_offset, g1_offset; // 1 line
            v_uint8x16 g2, r2, g2_offset, r2_offset; // 2 line
            v_uint8x16 b3, g3, b3_offset, g3_offset; // 3 line

            v_load_deinterleave(in[0] + w, b1, g1);
            v_load_deinterleave(in[0] + w + 2, b1_offset, g1_offset);

            v_load_deinterleave(in[1] + w, g2, r2);
            v_load_deinterleave(in[1] + w + 2, g2_offset, r2_offset);

            v_load_deinterleave(in[2] + w, b3, g3);
            v_load_deinterleave(in[2] + w + 2, b3_offset, g3_offset);

            // calculate r-channel
            v_expand(r2, l_1, r_1);
            v_expand(r2_offset, l_2, r_2);
            v_uint8x16 r2_sum = v_rshr_pack<1>(l_1 + l_2, r_1 + r_2);

            v_uint8x16 r_low, r_high;
            v_zip(r2, r2_sum, r_low, r_high);


            // calculate b-channel
            v_expand(b1, l_1, r_1);
            v_expand(b1_offset, l_2, r_2);
            v_expand(b3, l_3, r_3);
            v_expand(b3_offset, l_4, r_4);

            v_uint8x16 b13offset_sum, b13_sum;
            b13offset_sum = v_rshr_pack<2>(l_1 + l_2 + l_3 + l_4,
                                           r_1 + r_2 + r_3 + r_4);
            b13_sum = v_rshr_pack<1>(l_2 + l_4, r_2 + r_4);

            v_uint8x16 b_low, b_high;
            v_zip(b13offset_sum, b13_sum, b_low, b_high);


            // calculate g-channel
            v_expand(g1, l_1, r_1);
            v_expand(g3, l_2, r_2);
            v_expand(g2, l_3, r_3);
            v_expand(g2_offset, l_4, r_4);

            v_uint8x16 g_out_sum = v_rshr_pack<2>(l_1 + l_2 + l_3 + l_4,
                                                  r_1 + r_2 + r_3 + r_4);

            v_uint8x16 g_low, g_high;
            v_zip(g_out_sum, g2_offset, g_low, g_high);


            v_store_interleave(out + w * 3 + 3, b_low, g_low, r_low);
            v_store_interleave(out + w * 3 + vectorStep * 3 + 3, b_high, g_high, r_high);

            // output offset for scalar code
            j += vectorStep * 2;
        }
    #endif

    bool curr_blue = false;
    int t0, t1, t2;

    int i = 1;

    for (; j < width - 1; ++j, curr_blue = !curr_blue)
    {
        if (!curr_blue)
        {
            // pixel at green at bgbg line
            t2 = (in[i][j - 1] + in[i][j + 1] + 1) >> 1;
            t1 =  in[i][j];
            t0 = (in[i - 1][j] + in[i + 1][j] + 1) >> 1;

            out[j * 3 + 0] = (uchar)t0;
            out[j * 3 + 1] = (uchar)t1;
            out[j * 3 + 2] = (uchar)t2;
        }
        else
        {
            // pixel at red at grgr line
            t2 = in[i][j];

            t1 = (in[i][j - 1] + in[i][j + 1] +
                  in[i - 1][j] + in[i + 1][j] + 2) >> 2;

            t0 = (in[i - 1][j - 1] + in[i - 1][j + 1] +
                  in[i + 1][j - 1] + in[i + 1][j + 1] + 2) >> 2;

            out[j * 3 + 0] = (uchar)t0;
            out[j * 3 + 1] = (uchar)t1;
            out[j * 3 + 2] = (uchar)t2;

        }
    }

    out[0] = out[3];
    out[1] = out[4];
    out[2] = out[5];

    out[3 * (width - 1)    ] = out[3 * (width - 2)    ];
    out[3 * (width - 1) + 1] = out[3 * (width - 2) + 1];
    out[3 * (width - 1) + 2] = out[3 * (width - 2) + 2];
}

//--------------------------------------
//
// Fluid kernels: RGB-to-YUV, YUV-to-RGB
//
//--------------------------------------

void run_rgb2yuv_impl(uchar out[], const uchar in[], int width, const float coef[5])
{
    ushort c0 = static_cast<ushort>(coef[0]*(1 << 16) + 0.5f);  // Q0.0.16 un-signed
    ushort c1 = static_cast<ushort>(coef[1]*(1 << 16) + 0.5f);
    ushort c2 = static_cast<ushort>(coef[2]*(1 << 16) + 0.5f);
    short c3 = static_cast<short>(coef[3]*(1 << 12) + 0.5f);    // Q1.0.12 signed
    short c4 = static_cast<short>(coef[4]*(1 << 12) + 0.5f);

    int w = 0;

#if CV_SIMD
    static const int nlanes = v_uint8::nlanes;
    for ( ; w <= width - nlanes; w += nlanes)
    {
        v_uint8 r, g, b;
        v_load_deinterleave(&in[3*w], r, g, b);

        v_uint16 _r0, _r1, _g0, _g1, _b0, _b1;
        v_expand(r, _r0, _r1);
        v_expand(g, _g0, _g1);
        v_expand(b, _b0, _b1);

        _r0 = _r0 << 7;                         // Q0.9.7 un-signed
        _r1 = _r1 << 7;
        _g0 = _g0 << 7;
        _g1 = _g1 << 7;
        _b0 = _b0 << 7;
        _b1 = _b1 << 7;

        v_uint16 _y0, _y1;
        _y0 = v_mul_hi(vx_setall_u16(c0), _r0)  // Q0.9.7
            + v_mul_hi(vx_setall_u16(c1), _g0)
            + v_mul_hi(vx_setall_u16(c2), _b0);
        _y1 = v_mul_hi(vx_setall_u16(c0), _r1)
            + v_mul_hi(vx_setall_u16(c1), _g1)
            + v_mul_hi(vx_setall_u16(c2), _b1);

        v_int16 r0, r1, b0, b1, y0, y1;
        r0 = v_reinterpret_as_s16(_r0);         // Q1.8.7 signed
        r1 = v_reinterpret_as_s16(_r1);
        b0 = v_reinterpret_as_s16(_b0);
        b1 = v_reinterpret_as_s16(_b1);
        y0 = v_reinterpret_as_s16(_y0);
        y1 = v_reinterpret_as_s16(_y1);

        v_int16 u0, u1, v0, v1;
        u0 = v_mul_hi(vx_setall_s16(c3), b0 - y0);  // Q1.12.3
        u1 = v_mul_hi(vx_setall_s16(c3), b1 - y1);
        v0 = v_mul_hi(vx_setall_s16(c4), r0 - y0);
        v1 = v_mul_hi(vx_setall_s16(c4), r1 - y1);

        v_uint8 y, u, v;
        y = v_pack((_y0 + vx_setall_u16(1 << 6)) >> 7,
                   (_y1 + vx_setall_u16(1 << 6)) >> 7);
        u = v_pack_u((u0 + vx_setall_s16(257 << 2)) >> 3,  // 257 << 2 = 128.5 * (1 << 3)
                     (u1 + vx_setall_s16(257 << 2)) >> 3);
        v = v_pack_u((v0 + vx_setall_s16(257 << 2)) >> 3,
                     (v1 + vx_setall_s16(257 << 2)) >> 3);

        v_store_interleave(&out[3*w], y, u, v);
    }
#endif

    for ( ; w < width; w++)
    {
        short r = in[3*w    ] << 7;                            // Q1.8.7 signed
        short g = in[3*w + 1] << 7;
        short b = in[3*w + 2] << 7;
        short y = (c0*r + c1*g + c2*b) >> 16;                  // Q1.8.7
        short u =  c3*(b - y) >> 16;                           // Q1.12.3
        short v =  c4*(r - y) >> 16;
        out[3*w    ] = static_cast<uchar>((y              + (1 << 6)) >> 7);
        out[3*w + 1] =    saturate<uchar>((u + (128 << 3) + (1 << 2)) >> 3);
        out[3*w + 2] =    saturate<uchar>((v + (128 << 3) + (1 << 2)) >> 3);
    }
}

void run_yuv2rgb_impl(uchar out[], const uchar in[], int width, const float coef[4])
{
    short c0 = static_cast<short>(coef[0] * (1 << 12) + 0.5f);  // Q1.3.12
    short c1 = static_cast<short>(coef[1] * (1 << 12) + 0.5f);
    short c2 = static_cast<short>(coef[2] * (1 << 12) + 0.5f);
    short c3 = static_cast<short>(coef[3] * (1 << 12) + 0.5f);

    int w = 0;

#if CV_SIMD
    static const int nlanes = v_uint8::nlanes;
    for ( ; w <= width - nlanes; w += nlanes)
    {
        v_uint8 y, u, v;
        v_load_deinterleave(&in[3*w], y, u, v);

        v_uint16 _y0, _y1, _u0, _u1, _v0, _v1;
        v_expand(y, _y0, _y1);
        v_expand(u, _u0, _u1);
        v_expand(v, _v0, _v1);

        v_int16 y0, y1, u0, u1, v0, v1;
        y0 = v_reinterpret_as_s16(_y0);
        y1 = v_reinterpret_as_s16(_y1);
        u0 = v_reinterpret_as_s16(_u0);
        u1 = v_reinterpret_as_s16(_u1);
        v0 = v_reinterpret_as_s16(_v0);
        v1 = v_reinterpret_as_s16(_v1);

        y0 =  y0 << 3;                              // Q1.12.3
        y1 =  y1 << 3;
        u0 = (u0 - vx_setall_s16(128)) << 7;        // Q1.8.7
        u1 = (u1 - vx_setall_s16(128)) << 7;
        v0 = (v0 - vx_setall_s16(128)) << 7;
        v1 = (v1 - vx_setall_s16(128)) << 7;

        v_int16 r0, r1, g0, g1, b0, b1;
        r0 = y0 + v_mul_hi(vx_setall_s16(c0), v0);  // Q1.12.3
        r1 = y1 + v_mul_hi(vx_setall_s16(c0), v1);
        g0 = y0 + v_mul_hi(vx_setall_s16(c1), u0)
                + v_mul_hi(vx_setall_s16(c2), v0);
        g1 = y1 + v_mul_hi(vx_setall_s16(c1), u1)
                + v_mul_hi(vx_setall_s16(c2), v1);
        b0 = y0 + v_mul_hi(vx_setall_s16(c3), u0);
        b1 = y1 + v_mul_hi(vx_setall_s16(c3), u1);

        v_uint8 r, g, b;
        r = v_pack_u((r0 + vx_setall_s16(1 << 2)) >> 3,
                     (r1 + vx_setall_s16(1 << 2)) >> 3);
        g = v_pack_u((g0 + vx_setall_s16(1 << 2)) >> 3,
                     (g1 + vx_setall_s16(1 << 2)) >> 3);
        b = v_pack_u((b0 + vx_setall_s16(1 << 2)) >> 3,
                     (b1 + vx_setall_s16(1 << 2)) >> 3);

        v_store_interleave(&out[3*w], r, g, b);
    }
#endif

    for ( ; w < width; w++)
    {
        short y =  in[3*w    ]        << 3;  // Q1.12.3
        short u = (in[3*w + 1] - 128) << 7;  // Q1.8.7
        short v = (in[3*w + 2] - 128) << 7;
        short r = y + (        c0*v  >> 16); // Q1.12.3
        short g = y + ((c1*u + c2*v) >> 16);
        short b = y + ((c3*u       ) >> 16);
        out[3*w    ] = saturate<uchar>((r + (1 << 2)) >> 3);
        out[3*w + 1] = saturate<uchar>((g + (1 << 2)) >> 3);
        out[3*w + 2] = saturate<uchar>((b + (1 << 2)) >> 3);
    }
}

// Y' = 0.299*R' + 0.587*G' + 0.114*B'
// U' = (B' - Y')*0.492
// V' = (R' - Y')*0.877
static const float coef[5] = {0.299f, 0.587f, 0.114f, 0.492f, 0.877f};

// don't use expressions (avoid any dynamic initialization): https://github.com/opencv/opencv/issues/15690
static const ushort c0 = 19595;  // static_cast<ushort>(coef[0]*(1 << 16) + 0.5f);
static const ushort c1 = 38470;  // static_cast<ushort>(coef[1]*(1 << 16) + 0.5f);
static const ushort c2 = 7471;   // static_cast<ushort>(coef[2]*(1 << 16) + 0.5f);
static const short c3 = 2015;    // static_cast<short>(coef[3]*(1 << 12) + 0.5f);
static const short c4 = 3592;    // static_cast<short>(coef[4]*(1 << 12) + 0.5f);

void run_rgb2yuv422_impl(uchar out[], const uchar in[], int width)
{
    int w = 0, j = 0;

    #if CV_SIMD128
        const int vectorStep = 16;

        for (; w <= 3 * (width - vectorStep); w += 3 * vectorStep)
        {
            v_uint8x16 r, g, b;
            v_load_deinterleave(in + w, r, g, b);

            // TODO: compute u and v  x2 less times
            v_uint8x16 y, u, v;

            v_uint16x8 rr1, gg1, bb1, rr2, gg2, bb2;
            v_expand(r, rr1, rr2);
            v_expand(g, gg1, gg2);
            v_expand(b, bb1, bb2);

            rr1 = rr1 << 7;
            rr2 = rr2 << 7;
            gg1 = gg1 << 7;
            gg2 = gg2 << 7;
            bb1 = bb1 << 7;
            bb2 = bb2 << 7;

            v_uint16x8 yy1, yy2;

            yy1 = v_mul_hi(v_setall_u16(c0), rr1) +
                  v_mul_hi(v_setall_u16(c1), gg1) +
                  v_mul_hi(v_setall_u16(c2), bb1);

            yy2 = v_mul_hi(v_setall_u16(c0), rr2) +
                  v_mul_hi(v_setall_u16(c1), gg2) +
                  v_mul_hi(v_setall_u16(c2), bb2);

            v_int16x8 u1, u2, v1, v2;

            u1 = v_mul_hi(v_setall_s16(c3), v_reinterpret_as_s16(bb1) - v_reinterpret_as_s16(yy1));
            u2 = v_mul_hi(v_setall_s16(c3), v_reinterpret_as_s16(bb2) - v_reinterpret_as_s16(yy2));
            v1 = v_mul_hi(v_setall_s16(c4), v_reinterpret_as_s16(rr1) - v_reinterpret_as_s16(yy1));
            v2 = v_mul_hi(v_setall_s16(c4), v_reinterpret_as_s16(rr2) - v_reinterpret_as_s16(yy2));

            y = v_pack((yy1 + v_setall_u16(1 << 6)) >> 7,
                       (yy2 + v_setall_u16(1 << 6)) >> 7);
            u = v_pack_u((u1 + v_setall_s16(257 << 2)) >> 3,
                         (u2 + v_setall_s16(257 << 2)) >> 3);
            v = v_pack_u((v1 + v_setall_s16(257 << 2)) >> 3,
                         (v2 + v_setall_s16(257 << 2)) >> 3);

            uint8_t ff = 0xff;
            v_uint8x16 mask(ff, 0, ff, 0, ff, 0, ff, 0, ff, 0, ff, 0, ff, 0, ff, 0);
            v_uint8x16 uu = u & mask;
            v_uint8x16 vv = v & mask;
            // extract even u and v
            v_uint8x16 u_low = v_pack(v_reinterpret_as_u16(uu), v_reinterpret_as_u16(uu));
            v_uint8x16 v_low = v_pack(v_reinterpret_as_u16(vv), v_reinterpret_as_u16(vv));

            v_uint8x16 out1, out2;
            v_zip(u_low, v_low, out1, out2);

            v_store_interleave(out + j, out1, y);

            // offset for output buffer
            j += vectorStep * 2;
        }
    v_cleanup();
    #endif

    for (; w < width * 3; w += 6)
    {
        short r = in[w] << 7;
        short g = in[w + 1] << 7;
        short b = in[w + 2] << 7;
        short y1 = (c0 * r + c1 * g + c2 * b) >> 16;
        short u =  c3*(b - y1) >> 16;
        short v =  c4*(r - y1) >> 16;

        out[j]     = cv::saturate_cast<uchar>((u + (128 << 3) + (1 << 2)) >> 3); // u
        out[j + 1] = cv::saturate_cast<uchar>((y1 + (1 << 6)) >> 7); // y1
        out[j + 2] = cv::saturate_cast<uchar>((v + (128 << 3) + (1 << 2)) >> 3); // v

        r = in[w + 3] << 7;
        g = in[w + 4] << 7;
        b = in[w + 5] << 7;
        short y2 = (c0 * r + c1 * g + c2 * b) >> 16;

        out[j + 3] = cv::saturate_cast<uchar>((y2 + (1 << 6)) >> 7); // y2

        // offset for output buffer
        j += 4;
    }
}

//-----------------------------
//
// Fluid kernels: sepFilter 3x3
//
//-----------------------------

#if CV_SIMD
// this variant not using buf[] appears 15% faster than reference any-2-float code below
template<bool noscale, typename SRC>
static void run_sepfilter3x3_any2float(float out[], const SRC *in[], int width, int chan,
                                       const float kx[], const float ky[], int border,
                                       float scale, float delta)
{
    const int length = width * chan;
    const int shift = border * chan;

    const float kx0 = kx[0], kx1 = kx[1], kx2 = kx[2];
    const float ky0 = ky[0], ky1 = ky[1], ky2 = ky[2];

    for (int l=0; l < length; )
    {
        static const int nlanes = v_float32::nlanes;

        // main part
        for ( ; l <= length - nlanes; l += nlanes)
        {
            auto xsum = [l, shift, kx0, kx1, kx2](const SRC i[])
            {
                v_float32 t0 = vx_load_f32(&i[l - shift]);
                v_float32 t1 = vx_load_f32(&i[l        ]);
                v_float32 t2 = vx_load_f32(&i[l + shift]);
                v_float32 t = t0 * vx_setall_f32(kx0);
                    t = v_fma(t1,  vx_setall_f32(kx1), t);
                    t = v_fma(t2,  vx_setall_f32(kx2), t);
                return t;
            };

            v_float32 s0 = xsum(in[0]);
            v_float32 s1 = xsum(in[1]);
            v_float32 s2 = xsum(in[2]);
            v_float32 s = s0 * vx_setall_f32(ky0);
                s = v_fma(s1,  vx_setall_f32(ky1), s);
                s = v_fma(s2,  vx_setall_f32(ky2), s);

            if (!noscale)
            {
                s = v_fma(s, vx_setall_f32(scale), vx_setall_f32(delta));
            }

            v_store(&out[l], s);
        }

        // tail (if any)
        if (l < length)
        {
            GAPI_DbgAssert(length >= nlanes);
            l = length - nlanes;
        }
    }
}

// this variant with manually vectored rounding to short/ushort appears 10-40x faster
// than reference code below
template<bool noscale, typename DST, typename SRC>
static void run_sepfilter3x3_any2short(DST out[], const SRC *in[], int width, int chan,
                                       const float kx[], const float ky[], int border,
                                       float scale, float delta,
                                       float *buf[], int y, int y0)
{
    int r[3];
    r[0] = (y - y0    ) % 3;  // buf[r[0]]: previous
    r[1] = (y - y0 + 1) % 3;  //            this
    r[2] = (y - y0 + 2) % 3;  //            next row

    const int length = width * chan;
    const int shift = border * chan;

    const float kx0 = kx[0], kx1 = kx[1], kx2 = kx[2];
    const float ky0 = ky[0], ky1 = ky[1], ky2 = ky[2];

    // horizontal pass

    int k0 = (y == y0)? 0: 2;

    for (int k = k0; k < 3; k++)
    {
        //                      previous , this , next pixel
        const SRC *s[3] = {in[k] - shift , in[k], in[k] + shift};

        // rely on compiler vectoring
        for (int l=0; l < length; l++)
        {
            buf[r[k]][l] = s[0][l]*kx0 + s[1][l]*kx1 + s[2][l]*kx2;
        }
    }

    // vertical pass

    const int r0=r[0], r1=r[1], r2=r[2];

    for (int l=0; l < length;)
    {
        constexpr int nlanes = v_int16::nlanes;

        // main part of row
        for (; l <= length - nlanes; l += nlanes)
        {
            v_float32 sum0 = vx_load(&buf[r0][l])            * vx_setall_f32(ky0);
                sum0 = v_fma(vx_load(&buf[r1][l]),             vx_setall_f32(ky1), sum0);
                sum0 = v_fma(vx_load(&buf[r2][l]),             vx_setall_f32(ky2), sum0);

            v_float32 sum1 = vx_load(&buf[r0][l + nlanes/2]) * vx_setall_f32(ky0);
                sum1 = v_fma(vx_load(&buf[r1][l + nlanes/2]),  vx_setall_f32(ky1), sum1);
                sum1 = v_fma(vx_load(&buf[r2][l + nlanes/2]),  vx_setall_f32(ky2), sum1);

            if (!noscale)
            {
                sum0 = v_fma(sum0, vx_setall_f32(scale), vx_setall_f32(delta));
                sum1 = v_fma(sum1, vx_setall_f32(scale), vx_setall_f32(delta));
            }

            v_int32 isum0 = v_round(sum0),
                    isum1 = v_round(sum1);

            if (std::is_same<DST, short>::value)
            {
                // signed short
                v_int16 res = v_pack(isum0, isum1);
                v_store(reinterpret_cast<short*>(&out[l]), res);
            } else
            {
                // unsigned short
                v_uint16 res = v_pack_u(isum0, isum1);
                v_store(reinterpret_cast<ushort*>(&out[l]), res);
            }
        }

        // tail (if any)
        if (l < length)
        {
            GAPI_DbgAssert(length >= nlanes);
            l = length - nlanes;
        }
    }
}

// this code with manually vectored rounding to uchar is 10-40x faster than reference
template<bool noscale, typename SRC>
static void run_sepfilter3x3_any2char(uchar out[], const SRC *in[], int width, int chan,
                                      const float kx[], const float ky[], int border,
                                      float scale, float delta,
                                      float *buf[], int y, int y0)
{
    int r[3];
    r[0] = (y - y0    ) % 3;  // buf[r[0]]: previous
    r[1] = (y - y0 + 1) % 3;  //            this
    r[2] = (y - y0 + 2) % 3;  //            next row

    const int length = width * chan;
    const int shift = border * chan;

    const float kx0 = kx[0], kx1 = kx[1], kx2 = kx[2];
    const float ky0 = ky[0], ky1 = ky[1], ky2 = ky[2];

    // horizontal pass

    int k0 = (y == y0)? 0: 2;

    for (int k = k0; k < 3; k++)
    {
        //                      previous , this , next pixel
        const SRC *s[3] = {in[k] - shift , in[k], in[k] + shift};

        // rely on compiler vectoring
        for (int l=0; l < length; l++)
        {
            buf[r[k]][l] = s[0][l]*kx0 + s[1][l]*kx1 + s[2][l]*kx2;
        }
    }

    // vertical pass

    const int r0=r[0], r1=r[1], r2=r[2];

    for (int l=0; l < length;)
    {
        constexpr int nlanes = v_uint8::nlanes;

        // main part of row
        for (; l <= length - nlanes; l += nlanes)
        {
            v_float32 sum0 = vx_load(&buf[r0][l])              * vx_setall_f32(ky0);
                sum0 = v_fma(vx_load(&buf[r1][l]),               vx_setall_f32(ky1), sum0);
                sum0 = v_fma(vx_load(&buf[r2][l]),               vx_setall_f32(ky2), sum0);

            v_float32 sum1 = vx_load(&buf[r0][l +   nlanes/4]) * vx_setall_f32(ky0);
                sum1 = v_fma(vx_load(&buf[r1][l +   nlanes/4]),  vx_setall_f32(ky1), sum1);
                sum1 = v_fma(vx_load(&buf[r2][l +   nlanes/4]),  vx_setall_f32(ky2), sum1);

            v_float32 sum2 = vx_load(&buf[r0][l + 2*nlanes/4]) * vx_setall_f32(ky0);
                sum2 = v_fma(vx_load(&buf[r1][l + 2*nlanes/4]),  vx_setall_f32(ky1), sum2);
                sum2 = v_fma(vx_load(&buf[r2][l + 2*nlanes/4]),  vx_setall_f32(ky2), sum2);

            v_float32 sum3 = vx_load(&buf[r0][l + 3*nlanes/4]) * vx_setall_f32(ky0);
                sum3 = v_fma(vx_load(&buf[r1][l + 3*nlanes/4]),  vx_setall_f32(ky1), sum3);
                sum3 = v_fma(vx_load(&buf[r2][l + 3*nlanes/4]),  vx_setall_f32(ky2), sum3);

            if (!noscale)
            {
                sum0 = v_fma(sum0, vx_setall_f32(scale), vx_setall_f32(delta));
                sum1 = v_fma(sum1, vx_setall_f32(scale), vx_setall_f32(delta));
                sum2 = v_fma(sum2, vx_setall_f32(scale), vx_setall_f32(delta));
                sum3 = v_fma(sum3, vx_setall_f32(scale), vx_setall_f32(delta));
            }

            v_int32 isum0 = v_round(sum0),
                    isum1 = v_round(sum1),
                    isum2 = v_round(sum2),
                    isum3 = v_round(sum3);

            v_int16 ires0 = v_pack(isum0, isum1),
                    ires1 = v_pack(isum2, isum3);

            v_uint8 res = v_pack_u(ires0, ires1);
            v_store(reinterpret_cast<uchar*>(&out[l]), res);
        }

        // tail (if any)
        if (l < length)
        {
            GAPI_DbgAssert(length >= nlanes);
            l = length - nlanes;
        }
    }
}

// this code manually vectored for int16 not much faster than generic any-to-short code above
#define USE_SEPFILTER3X3_CHAR2SHORT 1

#if USE_SEPFILTER3X3_CHAR2SHORT
template<bool noscale>
static void run_sepfilter3x3_char2short(short out[], const uchar *in[], int width, int chan,
                                        const float kx[], const float ky[], int border,
                                        float scale, float delta,
                                        float *buf[], int y, int y0)
{
    const schar ikx0 = saturate<schar>(kx[0], rintf);
    const schar ikx1 = saturate<schar>(kx[1], rintf);
    const schar ikx2 = saturate<schar>(kx[2], rintf);

    const schar iky0 = saturate<schar>(ky[0], rintf);
    const schar iky1 = saturate<schar>(ky[1], rintf);
    const schar iky2 = saturate<schar>(ky[2], rintf);

    const short iscale = saturate<short>(scale * (1 << 15), rintf);
    const short idelta = saturate<short>(delta            , rintf);

    // check if this code is applicable
    if (ikx0 != kx[0] || ikx1 != kx[1] || ikx2 != kx[2] ||
        iky0 != ky[0] || iky1 != ky[1] || iky2 != ky[2] ||
        idelta != delta ||
        std::abs(scale) > 1 || std::abs(scale) < 0.01)
    {
        run_sepfilter3x3_any2short<noscale>(out, in, width, chan, kx, ky, border, scale, delta,
                                            buf, y, y0);
        return;
    }

    short *ibuf[3];
    ibuf[0] = reinterpret_cast<short*>(buf[0]);
    ibuf[1] = reinterpret_cast<short*>(buf[1]);
    ibuf[2] = reinterpret_cast<short*>(buf[2]);

    int r[3];
    r[0] = (y - y0    ) % 3;  // buf[r[0]]: previous
    r[1] = (y - y0 + 1) % 3;  //            this
    r[2] = (y - y0 + 2) % 3;  //            next row

    const int length = width * chan;
    const int shift = border * chan;

    // horizontal pass

    int k0 = (y == y0)? 0: 2;

    for (int k = k0; k < 3; k++)
    {
        for (int l=0; l < length;)
        {
            constexpr int nlanes = v_int16::nlanes;

            // main part of output row
            for (; l <= length - nlanes; l += nlanes)
            {
                v_uint16 t0 = vx_load_expand(&in[k][l - shift]);  // previous
                v_uint16 t1 = vx_load_expand(&in[k][l        ]);  // current
                v_uint16 t2 = vx_load_expand(&in[k][l + shift]);  // next pixel
                v_int16 t = v_reinterpret_as_s16(t0) * vx_setall_s16(ikx0) +
                            v_reinterpret_as_s16(t1) * vx_setall_s16(ikx1) +
                            v_reinterpret_as_s16(t2) * vx_setall_s16(ikx2);
                v_store(&ibuf[r[k]][l], t);
            }

            // tail (if any)
            if (l < length)
            {
                GAPI_DbgAssert(length >= nlanes);
                l = length - nlanes;
            }
        }
    }

    // vertical pass

    for (int l=0; l < length;)
    {
        constexpr int nlanes = v_int16::nlanes;

        // main part of output row
        for (; l <= length - nlanes; l += nlanes)
        {
            v_int16 s0 = vx_load(&ibuf[r[0]][l]);  // previous
            v_int16 s1 = vx_load(&ibuf[r[1]][l]);  // current
            v_int16 s2 = vx_load(&ibuf[r[2]][l]);  // next row
            v_int16 s = s0 * vx_setall_s16(iky0) +
                        s1 * vx_setall_s16(iky1) +
                        s2 * vx_setall_s16(iky2);

            if (!noscale)
            {
                s = v_mul_hi(s << 1, vx_setall_s16(iscale)) + vx_setall_s16(idelta);
            }

            v_store(&out[l], s);
        }

        // tail (if any)
        if (l < length)
        {
            GAPI_DbgAssert(length >= nlanes);
            l = length - nlanes;
        }
    }
}
#endif //USE_SEPFILTER3X3_CHAR2SHORT

#endif  // CV_SIMD

template<bool noscale, typename DST, typename SRC>
static void run_sepfilter3x3_reference(DST out[], const SRC *in[], int width, int chan,
                                       const float kx[], const float ky[], int border,
                                       float scale, float delta,
                                       float *buf[], int y, int y0)
{
    int r[3];
    r[0] = (y - y0)     % 3;  // buf[r[0]]: previous
    r[1] = (y - y0 + 1) % 3;  //            this
    r[2] = (y - y0 + 2) % 3;  //            next row

    int length = width * chan;
    int shift = border * chan;

    // horizontal pass

    // full horizontal pass is needed only if very 1st row in ROI;
    // for 2nd and further rows, it is enough to convolve only the
    // "next" row - as we can reuse buffers from previous calls to
    // this kernel (Fluid does rows consequently: y=y0, y0+1, ...)

    int k0 = (y == y0)? 0: 2;

    for (int k = k0; k < 3; k++)
    {
        //                      previous , this , next pixel
        const SRC *s[3] = {in[k] - shift , in[k], in[k] + shift};

        // rely on compiler vectoring
        for (int l=0; l < length; l++)
        {
            buf[r[k]][l] = s[0][l]*kx[0] + s[1][l]*kx[1] + s[2][l]*kx[2];
        }
    }

    // vertical pass

    for (int l=0; l < length; l++)
    {
        float sum = buf[r[0]][l]*ky[0] + buf[r[1]][l]*ky[1] + buf[r[2]][l]*ky[2];

        if (!noscale)
        {
            sum = sum*scale + delta;
        }

        out[l] = saturate<DST>(sum, rintf);
    }
}

template<bool noscale, typename DST, typename SRC>
static void run_sepfilter3x3_code(DST out[], const SRC *in[], int width, int chan,
                                  const float kx[], const float ky[], int border,
                                  float scale, float delta,
                                  float *buf[], int y, int y0)
{
#if CV_SIMD
    int length = width * chan;

    // length variable may be unused if types do not match at 'if' statements below
    (void) length;

#if USE_SEPFILTER3X3_CHAR2SHORT
    if (std::is_same<DST, short>::value && std::is_same<SRC, uchar>::value &&
        length >= v_int16::nlanes)
    {
        // only slightly faster than more generic any-to-short (see below)
        run_sepfilter3x3_char2short<noscale>(reinterpret_cast<short*>(out),
                                             reinterpret_cast<const uchar**>(in),
                                             width, chan, kx, ky, border, scale, delta,
                                             buf, y, y0);
        return;
    }
#endif

    if (std::is_same<DST, float>::value && std::is_same<SRC, float>::value &&
        length >= v_float32::nlanes)
    {
        // appears 15% faster than reference any-to-float code (called below)
        run_sepfilter3x3_any2float<noscale>(reinterpret_cast<float*>(out), in,
                                            width, chan, kx, ky, border, scale, delta);
        return;
    }

    if (std::is_same<DST, short>::value && length >= v_int16::nlanes)
    {
        // appears 10-40x faster than reference due to much faster rounding
        run_sepfilter3x3_any2short<noscale>(reinterpret_cast<short*>(out), in,
                                            width, chan, kx, ky, border, scale, delta,
                                            buf, y, y0);
        return;
    }

    if (std::is_same<DST, ushort>::value && length >= v_uint16::nlanes)
    {
        // appears 10-40x faster than reference due to much faster rounding
        run_sepfilter3x3_any2short<noscale>(reinterpret_cast<ushort*>(out), in,
                                            width, chan, kx, ky, border, scale, delta,
                                            buf, y, y0);
        return;
    }

    if (std::is_same<DST, uchar>::value && length >= v_uint8::nlanes)
    {
        // appears 10-40x faster than reference due to much faster rounding
        run_sepfilter3x3_any2char<noscale>(reinterpret_cast<uchar*>(out), in,
                                           width, chan, kx, ky, border, scale, delta,
                                           buf, y, y0);
        return;
    }
#endif  // CV_SIMD

    // reference code is quite fast for any-to-float case,
    // but not for any-to-integral due to very slow rounding
    run_sepfilter3x3_reference<noscale>(out, in, width, chan, kx, ky, border,
                                        scale, delta, buf, y, y0);
}

#define RUN_SEPFILTER3X3_IMPL(DST, SRC)                                      \
void run_sepfilter3x3_impl(DST out[], const SRC *in[], int width, int chan,  \
                           const float kx[], const float ky[], int border,   \
                           float scale, float delta,                         \
                           float *buf[], int y, int y0)                      \
{                                                                            \
    if (scale == 1 && delta == 0)                                            \
    {                                                                        \
        constexpr bool noscale = true;                                       \
        run_sepfilter3x3_code<noscale>(out, in, width, chan, kx, ky, border, \
                                       scale, delta, buf, y, y0);            \
    }                                                                        \
    else                                                                     \
    {                                                                        \
        constexpr bool noscale = false;                                      \
        run_sepfilter3x3_code<noscale>(out, in, width, chan, kx, ky, border, \
                                       scale, delta, buf, y, y0);            \
    }                                                                        \
}

RUN_SEPFILTER3X3_IMPL(uchar, uchar)
RUN_SEPFILTER3X3_IMPL(short, uchar)
RUN_SEPFILTER3X3_IMPL(float, uchar)
RUN_SEPFILTER3X3_IMPL(ushort, ushort)
RUN_SEPFILTER3X3_IMPL(short, ushort)
RUN_SEPFILTER3X3_IMPL(float, ushort)
RUN_SEPFILTER3X3_IMPL(short, short)
RUN_SEPFILTER3X3_IMPL(float, short)
RUN_SEPFILTER3X3_IMPL(float, float)

#undef RUN_SEPFILTER3X3_IMPL

//-----------------------------
//
// Fluid kernels: sepFilter 5x5
//
//-----------------------------

#if CV_SIMD

// this code with manually vectored rounding to uchar
template<bool noscale, typename SRC>
static void run_sepfilter5x5_any2char(uchar out[], const SRC *in[], int width, int chan,
                                      const float kx[], const float ky[], int border,
                                      float scale, float delta,
                                      float *buf[], int y, int y0)
{
    constexpr int kxLen = 5;
    constexpr int kyLen = kxLen;
    constexpr int buffSize = 5;

    int r[buffSize];
    for (int n = 0; n < buffSize; ++n)
    {
        r[n] = (y - y0 + n) % 5;  // previous, this, next rows
    }

    const int length = width * chan;
    const int shift = chan;

    // horizontal pass

    int k0 = (y == y0) ? 0 : 4;

    for (int k = k0; k < kxLen; ++k)
    {
        const SRC *s[kxLen] = { nullptr };

        for (int i = 0; i < kxLen; ++i)
        {
            //  previous , this , next pixels
            s[i] = in[k] + (i - border)*shift;
        }

        // rely on compiler vectoring
        for (int l = 0; l < length; ++l)
        {
            float sum = 0;
            for (int j = 0; j < kxLen; ++j)
            {
                sum += s[j][l] * kx[j];
            }
            buf[r[k]][l] = sum;
        }
    }

    // vertical pass

    constexpr int nlanes = v_uint8::nlanes;

    for (int l = 0; l < length;)
    {
        // main part of row
        for (; l <= length - nlanes; l += nlanes)
        {
            v_float32 sum0 = vx_load(&buf[r[0]][l]) * vx_setall_f32(ky[0]);
            v_float32 sum1 = vx_load(&buf[r[0]][l + nlanes / 4]) * vx_setall_f32(ky[0]);
            v_float32 sum2 = vx_load(&buf[r[0]][l + 2 * nlanes / 4]) * vx_setall_f32(ky[0]);
            v_float32 sum3 = vx_load(&buf[r[0]][l + 3 * nlanes / 4]) * vx_setall_f32(ky[0]);

            for (int n = 1; n < kyLen; ++n)
            {
                sum0 = v_fma(vx_load(&buf[r[n]][l]), vx_setall_f32(ky[n]), sum0);
                sum1 = v_fma(vx_load(&buf[r[n]][l + nlanes / 4]), vx_setall_f32(ky[n]), sum1);
                sum2 = v_fma(vx_load(&buf[r[n]][l + 2 * nlanes / 4]), vx_setall_f32(ky[n]), sum2);
                sum3 = v_fma(vx_load(&buf[r[n]][l + 3 * nlanes / 4]), vx_setall_f32(ky[n]), sum3);
            }

            if (!noscale)
            {
                sum0 = v_fma(sum0, vx_setall_f32(scale), vx_setall_f32(delta));
                sum1 = v_fma(sum1, vx_setall_f32(scale), vx_setall_f32(delta));
                sum2 = v_fma(sum2, vx_setall_f32(scale), vx_setall_f32(delta));
                sum3 = v_fma(sum3, vx_setall_f32(scale), vx_setall_f32(delta));
            }

            v_int32 isum0 = v_round(sum0),
                    isum1 = v_round(sum1),
                    isum2 = v_round(sum2),
                    isum3 = v_round(sum3);

            v_int16 ires0 = v_pack(isum0, isum1),
                    ires1 = v_pack(isum2, isum3);

            v_uint8 res = v_pack_u(ires0, ires1);
            v_store(reinterpret_cast<uchar*>(&out[l]), res);
        }

        // tail (if any)
        if (l < length)
        {
            GAPI_DbgAssert(length >= nlanes);
            l = length - nlanes;
        }
    }
    return;
}

// this variant with manually vectored rounding to short/ushort
template<bool noscale, typename DST, typename SRC>
static void run_sepfilter5x5_any2short(DST out[], const SRC *in[], int width, int chan,
                                       const float kx[], const float ky[], int border,
                                       float scale, float delta,
                                       float *buf[], int y, int y0)
{
    constexpr int kxLen = 5;
    constexpr int kyLen = kxLen;
    constexpr int buffSize = 5;

    int r[buffSize];
    for (int n = 0; n < buffSize; ++n)
    {
        r[n] = (y - y0 + n) % 5;  // previous, this, next rows
    }

    const int length = width * chan;
    const int shift = chan;

    // horizontal pass

    int k0 = (y == y0) ? 0 : 4;

    for (int k = k0; k < kyLen; ++k)
    {
        const SRC *s[kxLen] = { nullptr };

        for (int i = 0; i < kxLen; ++i)
        {
            //  previous , this , next pixels
            s[i] = in[k] + (i - border)*shift;
        }

        // rely on compiler vectoring
        for (int l = 0; l < length; ++l)
        {
            float sum = 0;
            for (int j = 0; j < kxLen; ++j)
            {
                sum += s[j][l] * kx[j];
            }
            buf[r[k]][l] = sum;
        }
    }

    // vertical pass

    constexpr int nlanes = v_int16::nlanes;
    for (int l = 0; l < length;)
    {
        //GAPI_Assert(length >= nlanes);
        // main part of row
        for (; l <= length - nlanes; l += nlanes)
        {
            v_float32 sum0 = vx_load(&buf[r[0]][l]) * vx_setall_f32(ky[0]);
            v_float32 sum1 = vx_load(&buf[r[0]][l + nlanes / 2]) * vx_setall_f32(ky[0]);

            for (int j = 1; j < kyLen; ++j)
            {
                sum0 = v_fma(vx_load(&buf[r[j]][l]), vx_setall_f32(ky[j]), sum0);
                sum1 = v_fma(vx_load(&buf[r[j]][l + nlanes / 2]), vx_setall_f32(ky[j]), sum1);
            }

            if (!noscale)
            {
                sum0 = v_fma(sum0, vx_setall_f32(scale), vx_setall_f32(delta));
                sum1 = v_fma(sum1, vx_setall_f32(scale), vx_setall_f32(delta));
            }

            v_int32 isum0 = v_round(sum0),
                    isum1 = v_round(sum1);

            if (std::is_same<DST, short>::value)
            {
                // signed short
                v_int16 res = v_pack(isum0, isum1);
                v_store(reinterpret_cast<short*>(&out[l]), res);
            }
            else
            {
                // unsigned short
                v_uint16 res = v_pack_u(isum0, isum1);
                v_store(reinterpret_cast<ushort*>(&out[l]), res);
            }
        }

        // tail (if any)
        if (l < length)
        {
            GAPI_DbgAssert(length >= nlanes);
            l = length - nlanes;
        }
    }
    return;
}

// this variant not using buf[]
template<bool noscale, typename SRC>
static void run_sepfilter5x5_any2float(float out[], const SRC *in[], int width, int chan,
                                       const float kx[], const float ky[], int border,
                                       float scale, float delta)
{
    constexpr int kxLen = 5;
    constexpr int kyLen = kxLen;
    constexpr int buffSize = 5;

    const int length = width * chan;
    const int shift = chan;

    static const int nlanes = v_float32::nlanes;
    for (int l = 0; l < length; )
    {
        //GAPI_Assert(length >= nlanes);
        // main part
        for (; l <= length - nlanes; l += nlanes)
        {
            auto xsum = [l, border, shift, kx](const SRC inp[])
            {
                v_float32 t[5];
                for (int i = 0; i < 5; ++i)
                {
                    t[i] = vx_load_f32(&inp[l + (i - border)*shift]);
                }

                v_float32 sum = t[0] * vx_setall_f32(kx[0]);
                for (int j = 1; j < 5; ++j)
                {
                    sum = v_fma(t[j], vx_setall_f32(kx[j]), sum);
                }

                return sum;
            };

            v_float32 s[buffSize];
            for (int m = 0; m < buffSize; ++m)
            {
                s[m] = xsum(in[m]);
            }

            v_float32 sum = s[0] * vx_setall_f32(ky[0]);
            for (int n = 1; n < kyLen; ++n)
            {
                sum = v_fma(s[n], vx_setall_f32(ky[n]), sum);
            }

            if (!noscale)
            {
                sum = v_fma(sum, vx_setall_f32(scale), vx_setall_f32(delta));
            }

            v_store(&out[l], sum);
        }

        // tail (if any)
        if (l < length)
        {
            GAPI_DbgAssert(length >= nlanes);
            l = length - nlanes;
        }
    }
    return;
}

#define USE_SEPFILTER5X5_CHAR2SHORT 1

#if USE_SEPFILTER5X5_CHAR2SHORT
template<bool noscale>
static void run_sepfilter5x5_char2short(short out[], const uchar *in[], int width, int chan,
                                        const float kx[], const float ky[], int border,
                                        float scale, float delta,
                                        float *buf[], int y, int y0)
{
    constexpr int kxLen = 5;
    constexpr int kyLen = kxLen;

    constexpr int buffSize = 5;

    schar ikx[kxLen];
    schar iky[kyLen];

    for (int i = 0; i < kxLen; ++i)
    {
        ikx[i] = saturate<schar>(kx[i], rintf);
        iky[i] = saturate<schar>(ky[i], rintf);
    }

    const short iscale = saturate<short>(scale * (1 << 15), rintf);
    const short idelta = saturate<short>(delta, rintf);

    // check if this code is applicable
    if (ikx[0] != kx[0] || ikx[1] != kx[1] || ikx[2] != kx[2] || ikx[3] != kx[3] || ikx[4] != kx[4] ||
        iky[0] != ky[0] || iky[1] != ky[1] || iky[2] != ky[2] || iky[3] != ky[3] || iky[4] != ky[4] ||
        idelta != delta ||
        std::abs(scale) > 1 || std::abs(scale) < 0.01)
    {
        run_sepfilter5x5_any2short<noscale>(out, in, width, chan, kx, ky, border, scale, delta,
                                            buf, y, y0);
        return;
    }

    short *ibuf[buffSize];
    int r[buffSize];

    for (int n = 0; n < buffSize; ++n)
    {
        ibuf[n] = reinterpret_cast<short*>(buf[n]);
        r[n] = (y - y0 + n) % 5;  // previous, this, next rows
    }

    const int length = width * chan;
    const int shift = chan;

    // horizontal pass
    // full horizontal pass is needed only if the very 1st row in ROI is handled;
    // for 2nd and further rows, it's enough to convolve only the
    // "next" row - as we can reuse buffers from previous calls to
    // this kernel (Fluid does rows consequently: y=y0, y0+1, ...)
    int k0 = (y == y0) ? 0 : 4;

    constexpr int nlanes = v_int16::nlanes;

    for (int k = k0; k < kyLen; ++k)
    {
        for (int l = 0; l < length;)
        {
            GAPI_Assert(length >= nlanes);

            // main part of output row
            for (; l <= length - nlanes; l += nlanes)
            {
                v_uint16 t[kxLen];
                v_int16 sum = vx_setzero_s16();

                for (int i = 0; i < kxLen; ++i)
                {
                    // previous, current, next pixels
                    t[i] = vx_load_expand(&in[k][l + (i - border)*shift]);

                    sum += v_reinterpret_as_s16(t[i]) * vx_setall_s16(ikx[i]);
                }

                v_store(&ibuf[r[k]][l], sum);
            }

            // tail (if any)
            if (l < length)
            {
                GAPI_DbgAssert(length >= nlanes);
                l = length - nlanes;
            }
        }
    }

    // vertical pass

    for (int l = 0; l < length;)
    {
        //GAPI_Assert(length >= nlanes);
        // main part of output row
        for (; l <= length - nlanes; l += nlanes)
        {
            v_int16 s[buffSize];
            v_int16 sum = vx_setzero_s16();

            for (int i = 0; i < kyLen; ++i)
            {
                // previous, current, next rows
                s[i] = vx_load(&ibuf[r[i]][l]);

                sum += s[i] * vx_setall_s16(iky[i]);
            }

            if (!noscale)
            {
                sum = v_mul_hi(sum << 1, vx_setall_s16(iscale)) + vx_setall_s16(idelta);
            }

            v_store(&out[l], sum);
        }

        // tail (if any)
        if (l < length)
        {
            GAPI_DbgAssert(length >= nlanes);
            l = length - nlanes;
        }
    }
    return;
}
#endif //USE_SEPFILTER5X5_CHAR2SHORT

#endif //CV_SIMD

template<bool noscale, typename DST, typename SRC>
static void run_sepfilter5x5_reference(DST out[], const SRC *in[], int width, int chan,
                                       const float kx[], const float ky[], int border,
                                       float scale, float delta, float *buf[], int y, int y0)
{
    constexpr int kxLen = 5; // kernel size
    constexpr int kyLen = kxLen;
    int r[kyLen];
    for (int n = 0; n < kyLen; ++n)
    {
        r[n] = (y - y0 + n) % 5; // previous, this, next rows
    }

    int length = width * chan;
    int shift = chan;

    // horizontal pass

    // full horizontal pass is needed only if very 1st row in ROI;
    // for 2nd and further rows, it is enough to convolve only the
    // "next" row - as we can reuse buffers from previous calls to
    // this kernel (Fluid does rows consequently: y=y0, y0+1, ...)

    int k0 = (y == y0) ? 0 : 4;

    for (int k = k0; k < kyLen; ++k)
    {
        const SRC *s[kxLen] = { nullptr };

        for (int i = 0; i < kxLen; ++i)
        {
            //  previous , this , next pixels
            s[i] = in[k] + (i - border)*shift;
        }

        // rely on compiler vectoring
        for (int l = 0; l < length; ++l)
        {
            float sum = 0;
            for (int i = 0; i < kxLen; ++i)
            {
                sum += s[i][l] * kx[i];
            }
            buf[r[k]][l] = sum;
        }
    }

    // vertical pass

    for (int l = 0; l < length; ++l)
    {
        float sum = 0;
        for (int j = 0; j < kyLen; ++j)
        {
            sum += buf[r[j]][l] * ky[j];
        }

        if (!noscale)
        {
            sum = sum * scale + delta;
        }

        out[l] = saturate<DST>(sum, rintf);
    }
    return;
}

template<bool noscale, typename DST, typename SRC>
static void run_sepfilter5x5_code(DST out[], const SRC *in[], int width, int chan,
                                  const float kx[], const float ky[], int border,
                                  float scale, float delta, float *buf[], int y, int y0)
{
#if CV_SIMD
    int length = width * chan;

    // length variable may be unused if types do not match at 'if' statements below
    (void)length;

    if (std::is_same<DST, short>::value && std::is_same<SRC, uchar>::value &&
        length >= v_int16::nlanes)
    {
        run_sepfilter5x5_char2short<noscale>(reinterpret_cast<short*>(out),
                                             reinterpret_cast<const uchar**>(in),
                                             width, chan, kx, ky, border, scale, delta,
                                             buf, y, y0);
        return;
    }

    if (std::is_same<DST, float>::value && std::is_same<SRC, float>::value &&
        length >= v_float32::nlanes)
    {
        run_sepfilter5x5_any2float<noscale>(reinterpret_cast<float*>(out), in, width,
                                            chan, kx, ky, border, scale, delta);
        return;
    }

    if (std::is_same<DST, short>::value && length >= v_int16::nlanes)
    {
        run_sepfilter5x5_any2short<noscale>(reinterpret_cast<short*>(out), in, width,
                                            chan, kx, ky, border, scale, delta,
                                            buf, y, y0);
        return;
    }

    if (std::is_same<DST, ushort>::value && length >= v_uint16::nlanes)
    {
        run_sepfilter5x5_any2short<noscale>(reinterpret_cast<ushort*>(out), in, width,
                                            chan, kx, ky, border, scale, delta,
                                            buf, y, y0);
        return;
    }

    if (std::is_same<DST, uchar>::value && length >= v_uint8::nlanes)
    {
        run_sepfilter5x5_any2char<noscale>(reinterpret_cast<uchar*>(out), in, width,
                                           chan, kx, ky, border, scale, delta,
                                           buf, y, y0);
        return;
    }
#endif  // CV_SIMD

    // reference code is quite fast for any-to-float case,
    // but not for any-to-integral due to very slow rounding
    run_sepfilter5x5_reference<noscale>(out, in, width, chan, kx, ky, border,
        scale, delta, buf, y, y0);
}
#define RUN_SEPFILTER5x5_IMPL(DST, SRC)                                                                        \
void run_sepfilter5x5_impl(DST out[], const SRC *in[], int width, int chan, const float kx[],                  \
                           const float ky[], int border, float scale, float delta,                             \
                           float *buf[], int y, int y0)                                                        \
{                                                                                                              \
    if (scale == 1 && delta == 0)                                                                              \
    {                                                                                                          \
        constexpr bool noscale = true;                                                                         \
        run_sepfilter5x5_code<noscale>(out, in, width, chan, kx, ky, border,                                   \
                                       scale, delta, buf, y, y0);                                              \
    }                                                                                                          \
    else                                                                                                       \
    {                                                                                                          \
        constexpr bool noscale = false;                                                                        \
        run_sepfilter5x5_code<noscale>(out, in, width, chan, kx, ky, border,                                   \
                                       scale, delta, buf, y, y0);                                              \
    }                                                                                                          \
    return;                                                                                                    \
}

RUN_SEPFILTER5x5_IMPL(uchar, uchar)
RUN_SEPFILTER5x5_IMPL(short, uchar)
RUN_SEPFILTER5x5_IMPL(float, uchar)
RUN_SEPFILTER5x5_IMPL(ushort, ushort)
RUN_SEPFILTER5x5_IMPL(short, ushort)
RUN_SEPFILTER5x5_IMPL(float, ushort)
RUN_SEPFILTER5x5_IMPL(short, short)
RUN_SEPFILTER5x5_IMPL(float, short)
RUN_SEPFILTER5x5_IMPL(float, float)

#undef RUN_SEPFILTER5x5_IMPL

//-------------------------
//
// Fluid kernels: Filter 2D
//
//-------------------------

template<bool noscale, typename DST, typename SRC>
static void run_filter2d_3x3_reference(DST out[], const SRC *in[], int width, int chan,
                                       const float kernel[], float scale, float delta)
{
    static constexpr int ksize = 3;
    static constexpr int border = (ksize - 1) / 2;

    const int length = width * chan;
    const int shift = border * chan;

    const float k[3][3] = {{ kernel[0], kernel[1], kernel[2] },
                           { kernel[3], kernel[4], kernel[5] },
                           { kernel[6], kernel[7], kernel[8] }};

    for (int l=0; l < length; l++)
    {
        float sum = in[0][l - shift] * k[0][0] + in[0][l] * k[0][1] + in[0][l + shift] * k[0][2]
                  + in[1][l - shift] * k[1][0] + in[1][l] * k[1][1] + in[1][l + shift] * k[1][2]
                  + in[2][l - shift] * k[2][0] + in[2][l] * k[2][1] + in[2][l + shift] * k[2][2];

        if (!noscale)
        {
            sum = sum*scale + delta;
        }

        out[l] = saturate<DST>(sum, rintf);
    }
}

#if CV_SIMD
// assume DST is short or ushort
template<bool noscale, typename DST, typename SRC>
static void run_filter2d_3x3_any2short(DST out[], const SRC *in[], int width, int chan,
                                       const float kernel[], float scale, float delta)
{
    static constexpr int ksize = 3;
    static constexpr int border = (ksize - 1) / 2;

    const int length = width * chan;
    const int shift = border * chan;

    const float k[3][3] = {
        { kernel[0], kernel[1], kernel[2] },
        { kernel[3], kernel[4], kernel[5] },
        { kernel[6], kernel[7], kernel[8] }
    };

    for (int l=0; l < length;)
    {
        static constexpr int nlanes = v_int16::nlanes;

        // main part of output row
        for (; l <= length - nlanes; l += nlanes)
        {
            auto sumx = [in, shift, &k](int i, int j)
            {
                v_float32 s = vx_load_f32(&in[i][j - shift]) * vx_setall_f32(k[i][0]);
                    s = v_fma(vx_load_f32(&in[i][j        ]),  vx_setall_f32(k[i][1]), s);
                    s = v_fma(vx_load_f32(&in[i][j + shift]),  vx_setall_f32(k[i][2]), s);
                return s;
            };

            int l0 = l;
            int l1 = l + nlanes/2;
            v_float32 sum0 = sumx(0, l0) + sumx(1, l0) + sumx(2, l0);
            v_float32 sum1 = sumx(0, l1) + sumx(1, l1) + sumx(2, l1);

            if (!noscale)
            {
                sum0 = v_fma(sum0, vx_setall_f32(scale), vx_setall_f32(delta));
                sum1 = v_fma(sum1, vx_setall_f32(scale), vx_setall_f32(delta));
            }

            v_int32 res0 = v_round(sum0);
            v_int32 res1 = v_round(sum1);

            if (std::is_same<DST, ushort>::value)
            {
                v_uint16 res = v_pack_u(res0, res1);
                v_store(reinterpret_cast<ushort*>(&out[l]), res);
            }
            else // if DST == short
            {
                v_int16 res = v_pack(res0, res1);
                v_store(reinterpret_cast<short*>(&out[l]), res);
            }
        }

        // tail (if any)
        if (l < length)
        {
            GAPI_DbgAssert(length >= nlanes);
            l = length - nlanes;
        }
    }
}

template<bool noscale, typename SRC>
static void run_filter2d_3x3_any2char(uchar out[], const SRC *in[], int width, int chan,
                                      const float kernel[], float scale, float delta)
{
    static constexpr int ksize = 3;
    static constexpr int border = (ksize - 1) / 2;

    const int length = width * chan;
    const int shift = border * chan;

    const float k[3][3] = {
        { kernel[0], kernel[1], kernel[2] },
        { kernel[3], kernel[4], kernel[5] },
        { kernel[6], kernel[7], kernel[8] }
    };

    for (int l=0; l < length;)
    {
        static constexpr int nlanes = v_uint8::nlanes;

        // main part of output row
        for (; l <= length - nlanes; l += nlanes)
        {
            auto sumx = [in, shift, &k](int i, int j)
            {
                v_float32 s = vx_load_f32(&in[i][j - shift]) * vx_setall_f32(k[i][0]);
                    s = v_fma(vx_load_f32(&in[i][j        ]),  vx_setall_f32(k[i][1]), s);
                    s = v_fma(vx_load_f32(&in[i][j + shift]),  vx_setall_f32(k[i][2]), s);
                return s;
            };

            int l0 = l;
            int l1 = l +   nlanes/4;
            int l2 = l + 2*nlanes/4;
            int l3 = l + 3*nlanes/4;
            v_float32 sum0 = sumx(0, l0) + sumx(1, l0) + sumx(2, l0);
            v_float32 sum1 = sumx(0, l1) + sumx(1, l1) + sumx(2, l1);
            v_float32 sum2 = sumx(0, l2) + sumx(1, l2) + sumx(2, l2);
            v_float32 sum3 = sumx(0, l3) + sumx(1, l3) + sumx(2, l3);

            if (!noscale)
            {
                sum0 = v_fma(sum0, vx_setall_f32(scale), vx_setall_f32(delta));
                sum1 = v_fma(sum1, vx_setall_f32(scale), vx_setall_f32(delta));
                sum2 = v_fma(sum2, vx_setall_f32(scale), vx_setall_f32(delta));
                sum3 = v_fma(sum3, vx_setall_f32(scale), vx_setall_f32(delta));
            }

            v_int32 res0 = v_round(sum0);
            v_int32 res1 = v_round(sum1);
            v_int32 res2 = v_round(sum2);
            v_int32 res3 = v_round(sum3);

            v_int16 resl = v_pack(res0, res1);
            v_int16 resh = v_pack(res2, res3);
            v_uint8 res = v_pack_u(resl, resh);

            v_store(&out[l], res);
        }

        // tail (if any)
        if (l < length)
        {
            GAPI_DbgAssert(length >= nlanes);
            l = length - nlanes;
        }
    }
}
#endif

template<bool noscale, typename DST, typename SRC>
static void run_filter2d_3x3_code(DST out[], const SRC *in[], int width, int chan,
                                  const float kernel[], float scale, float delta)
{
#if CV_SIMD
    int length = width * chan;

    // length variable may be unused if types do not match at 'if' statements below
    (void) length;

    if (std::is_same<DST, short>::value && length >= v_int16::nlanes)
    {
        run_filter2d_3x3_any2short<noscale>(reinterpret_cast<short*>(out), in,
                                            width, chan, kernel, scale, delta);
        return;
    }

    if (std::is_same<DST, ushort>::value && length >= v_uint16::nlanes)
    {
        run_filter2d_3x3_any2short<noscale>(reinterpret_cast<ushort*>(out), in,
                                            width, chan, kernel, scale, delta);
        return;
    }


    if (std::is_same<DST, uchar>::value && length >= v_uint8::nlanes)
    {
        run_filter2d_3x3_any2char<noscale>(reinterpret_cast<uchar*>(out), in,
                                           width, chan, kernel, scale, delta);
        return;
    }
#endif  // CV_SIMD

    run_filter2d_3x3_reference<noscale>(out, in, width, chan, kernel, scale, delta);
}

#define RUN_FILTER2D_3X3_IMPL(DST, SRC)                                             \
void run_filter2d_3x3_impl(DST out[], const SRC *in[], int width, int chan,         \
                           const float kernel[], float scale, float delta)          \
{                                                                                   \
    if (scale == 1 && delta == 0)                                                   \
    {                                                                               \
        constexpr bool noscale = true;                                              \
        run_filter2d_3x3_code<noscale>(out, in, width, chan, kernel, scale, delta); \
    }                                                                               \
    else                                                                            \
    {                                                                               \
        constexpr bool noscale = false;                                             \
        run_filter2d_3x3_code<noscale>(out, in, width, chan, kernel, scale, delta); \
    }                                                                               \
}

RUN_FILTER2D_3X3_IMPL(uchar , uchar )
RUN_FILTER2D_3X3_IMPL(ushort, ushort)
RUN_FILTER2D_3X3_IMPL( short,  short)
RUN_FILTER2D_3X3_IMPL( float, uchar )
RUN_FILTER2D_3X3_IMPL( float, ushort)
RUN_FILTER2D_3X3_IMPL( float,  short)
RUN_FILTER2D_3X3_IMPL( float,  float)

#undef RUN_FILTER2D_3X3_IMPL

//-----------------------------
//
// Fluid kernels: Erode, Dilate
//
//-----------------------------

template<typename T>
static void run_morphology3x3_reference(T out[], const T *in[], int width, int chan,
                                        const uchar k[], MorphShape k_type,
                                        Morphology morphology)
{
    constexpr int k_size = 3;
    constexpr int border = (k_size - 1) / 2;

    const uchar kernel[3][3] = {{k[0], k[1], k[2]}, {k[3], k[4], k[5]}, {k[6], k[7], k[8]}};

    const int length = width * chan;
    const int shift = border * chan;

    if (M_ERODE == morphology)
    {
        if (M_FULL == k_type)
        {
            for (int l=0; l < length; l++)
            {
                T result = std::numeric_limits<T>::max();

                result = (std::min)(result, in[0][l - shift]);
                result = (std::min)(result, in[0][l        ]);
                result = (std::min)(result, in[0][l + shift]);

                result = (std::min)(result, in[1][l - shift]);
                result = (std::min)(result, in[1][l        ]);
                result = (std::min)(result, in[1][l + shift]);

                result = (std::min)(result, in[2][l - shift]);
                result = (std::min)(result, in[2][l        ]);
                result = (std::min)(result, in[2][l + shift]);

                out[l] = result;
            }
            return;
        }

        if (M_CROSS == k_type)
        {
            for (int l=0; l < length; l++)
            {
                T result = std::numeric_limits<T>::max();

            //  result = (std::min)(result, in[0][l - shift]);
                result = (std::min)(result, in[0][l        ]);
            //  result = (std::min)(result, in[0][l + shift]);

                result = (std::min)(result, in[1][l - shift]);
                result = (std::min)(result, in[1][l        ]);
                result = (std::min)(result, in[1][l + shift]);

            //  result = (std::min)(result, in[2][l - shift]);
                result = (std::min)(result, in[2][l        ]);
            //  result = (std::min)(result, in[2][l + shift]);

                out[l] = result;
            }
            return;
        }

        for (int l=0; l < length; l++)
        {
            T result = std::numeric_limits<T>::max();

            result = kernel[0][0]? (std::min)(result, in[0][l - shift]): result;
            result = kernel[0][1]? (std::min)(result, in[0][l        ]): result;
            result = kernel[0][2]? (std::min)(result, in[0][l + shift]): result;

            result = kernel[1][0]? (std::min)(result, in[1][l - shift]): result;
            result = kernel[1][1]? (std::min)(result, in[1][l        ]): result;
            result = kernel[1][2]? (std::min)(result, in[1][l + shift]): result;

            result = kernel[2][0]? (std::min)(result, in[2][l - shift]): result;
            result = kernel[2][1]? (std::min)(result, in[2][l        ]): result;
            result = kernel[2][2]? (std::min)(result, in[2][l + shift]): result;

            out[l] = result;
        }
        return;
    }

    if (M_DILATE == morphology)
    {
        if (M_FULL == k_type)
        {
            for (int l=0; l < length; l++)
            {
                T result = std::numeric_limits<T>::min();

                result = (std::max)(result, in[0][l - shift]);
                result = (std::max)(result, in[0][l        ]);
                result = (std::max)(result, in[0][l + shift]);

                result = (std::max)(result, in[1][l - shift]);
                result = (std::max)(result, in[1][l        ]);
                result = (std::max)(result, in[1][l + shift]);

                result = (std::max)(result, in[2][l - shift]);
                result = (std::max)(result, in[2][l        ]);
                result = (std::max)(result, in[2][l + shift]);

                out[l] = result;
            }
            return;
        }

        if (M_CROSS == k_type)
        {
            for (int l=0; l < length; l++)
            {
                T result = std::numeric_limits<T>::min();

            //  result = (std::max)(result, in[0][l - shift]);
                result = (std::max)(result, in[0][l        ]);
            //  result = (std::max)(result, in[0][l + shift]);

                result = (std::max)(result, in[1][l - shift]);
                result = (std::max)(result, in[1][l        ]);
                result = (std::max)(result, in[1][l + shift]);

            //  result = (std::max)(result, in[2][l - shift]);
                result = (std::max)(result, in[2][l        ]);
            //  result = (std::max)(result, in[2][l + shift]);

                out[l] = result;
            }
            return;
        }

        for (int l=0; l < length; l++)
        {
            T result = std::numeric_limits<T>::min();

            result = kernel[0][0]? (std::max)(result, in[0][l - shift]): result;
            result = kernel[0][1]? (std::max)(result, in[0][l        ]): result;
            result = kernel[0][2]? (std::max)(result, in[0][l + shift]): result;

            result = kernel[1][0]? (std::max)(result, in[1][l - shift]): result;
            result = kernel[1][1]? (std::max)(result, in[1][l        ]): result;
            result = kernel[1][2]? (std::max)(result, in[1][l + shift]): result;

            result = kernel[2][0]? (std::max)(result, in[2][l - shift]): result;
            result = kernel[2][1]? (std::max)(result, in[2][l        ]): result;
            result = kernel[2][2]? (std::max)(result, in[2][l + shift]): result;

            out[l] = result;
        }
        return;
    }

    CV_Error(cv::Error::StsBadArg, "unsupported morphology");
}

#if CV_SIMD
template<typename T, typename VT, typename S>
static void run_morphology3x3_simd(T out[], const T *in[], int width, int chan,
                                   const uchar k[], MorphShape k_type,
                                   Morphology morphology,
                                   S setall)
{
    constexpr int k_size = 3;
    constexpr int border = (k_size - 1) / 2;

    const uchar kernel[3][3] = {{k[0], k[1], k[2]}, {k[3], k[4], k[5]}, {k[6], k[7], k[8]}};

    const int length = width * chan;
    const int shift = border * chan;

    if (M_ERODE == morphology)
    {
        if (M_FULL == k_type)
        {
            for (int l=0; l < length;)
            {
                constexpr int nlanes = VT::nlanes;

                // main part of output row
                for (; l <= length - nlanes; l += nlanes)
                {
                    VT r = setall(std::numeric_limits<T>::max());

                    r = v_min(r, vx_load(&in[0][l - shift]));
                    r = v_min(r, vx_load(&in[0][l        ]));
                    r = v_min(r, vx_load(&in[0][l + shift]));

                    r = v_min(r, vx_load(&in[1][l - shift]));
                    r = v_min(r, vx_load(&in[1][l        ]));
                    r = v_min(r, vx_load(&in[1][l + shift]));

                    r = v_min(r, vx_load(&in[2][l - shift]));
                    r = v_min(r, vx_load(&in[2][l        ]));
                    r = v_min(r, vx_load(&in[2][l + shift]));

                    v_store(&out[l], r);
                }

                // tail (if any)
                if (l < length)
                {
                    GAPI_DbgAssert(length >= nlanes);
                    l = length - nlanes;
                }
            }
            return;
        }

        if (M_CROSS == k_type)
        {
            for (int l=0; l < length;)
            {
                constexpr int nlanes = VT::nlanes;

                // main part of output row
                for (; l <= length - nlanes; l += nlanes)
                {
                    VT r = setall(std::numeric_limits<T>::max());

                //  r = v_min(r, vx_load(&in[0][l - shift]));
                    r = v_min(r, vx_load(&in[0][l        ]));
                //  r = v_min(r, vx_load(&in[0][l + shift]));

                    r = v_min(r, vx_load(&in[1][l - shift]));
                    r = v_min(r, vx_load(&in[1][l        ]));
                    r = v_min(r, vx_load(&in[1][l + shift]));

                //  r = v_min(r, vx_load(&in[2][l - shift]));
                    r = v_min(r, vx_load(&in[2][l        ]));
                //  r = v_min(r, vx_load(&in[2][l + shift]));

                    v_store(&out[l], r);
                }

                // tail (if any)
                if (l < length)
                {
                    GAPI_DbgAssert(length >= nlanes);
                    l = length - nlanes;
                }
            }
            return;
        }

        for (int l=0; l < length;)
        {
            constexpr int nlanes = VT::nlanes;

            // main part of output row
            for (; l <= length - nlanes; l += nlanes)
            {
                VT r = setall(std::numeric_limits<T>::max());

                if (kernel[0][0]) r = v_min(r, vx_load(&in[0][l - shift]));
                if (kernel[0][1]) r = v_min(r, vx_load(&in[0][l        ]));
                if (kernel[0][2]) r = v_min(r, vx_load(&in[0][l + shift]));

                if (kernel[1][0]) r = v_min(r, vx_load(&in[1][l - shift]));
                if (kernel[1][1]) r = v_min(r, vx_load(&in[1][l        ]));
                if (kernel[1][2]) r = v_min(r, vx_load(&in[1][l + shift]));

                if (kernel[2][0]) r = v_min(r, vx_load(&in[2][l - shift]));
                if (kernel[2][1]) r = v_min(r, vx_load(&in[2][l        ]));
                if (kernel[2][2]) r = v_min(r, vx_load(&in[2][l + shift]));

                v_store(&out[l], r);
            }

            // tail (if any)
            if (l < length)
            {
                GAPI_DbgAssert(length >= nlanes);
                l = length - nlanes;
            }
        }
        return;
    }

    if (M_DILATE == morphology)
    {
        if (M_FULL == k_type)
        {
            for (int l=0; l < length;)
            {
                constexpr int nlanes = VT::nlanes;

                // main part of output row
                for (; l <= length - nlanes; l += nlanes)
                {
                    VT r = setall(std::numeric_limits<T>::min());

                    r = v_max(r, vx_load(&in[0][l - shift]));
                    r = v_max(r, vx_load(&in[0][l        ]));
                    r = v_max(r, vx_load(&in[0][l + shift]));

                    r = v_max(r, vx_load(&in[1][l - shift]));
                    r = v_max(r, vx_load(&in[1][l        ]));
                    r = v_max(r, vx_load(&in[1][l + shift]));

                    r = v_max(r, vx_load(&in[2][l - shift]));
                    r = v_max(r, vx_load(&in[2][l        ]));
                    r = v_max(r, vx_load(&in[2][l + shift]));

                    v_store(&out[l], r);
                }

                // tail (if any)
                if (l < length)
                {
                    GAPI_DbgAssert(length >= nlanes);
                    l = length - nlanes;
                }
            }
            return;
        }

        if (M_CROSS == k_type)
        {
            for (int l=0; l < length;)
            {
                constexpr int nlanes = VT::nlanes;

                // main part of output row
                for (; l <= length - nlanes; l += nlanes)
                {
                    VT r = setall(std::numeric_limits<T>::min());

                //  r = v_max(r, vx_load(&in[0][l - shift]));
                    r = v_max(r, vx_load(&in[0][l        ]));
                //  r = v_max(r, vx_load(&in[0][l + shift]));

                    r = v_max(r, vx_load(&in[1][l - shift]));
                    r = v_max(r, vx_load(&in[1][l        ]));
                    r = v_max(r, vx_load(&in[1][l + shift]));

                //  r = v_max(r, vx_load(&in[2][l - shift]));
                    r = v_max(r, vx_load(&in[2][l        ]));
                //  r = v_max(r, vx_load(&in[2][l + shift]));

                    v_store(&out[l], r);
                }

                // tail (if any)
                if (l < length)
                {
                    GAPI_DbgAssert(length >= nlanes);
                    l = length - nlanes;
                }
            }
            return;
        }

        for (int l=0; l < length;)
        {
            constexpr int nlanes = VT::nlanes;

            // main part of output row
            for (; l <= length - nlanes; l += nlanes)
            {
                VT r = setall(std::numeric_limits<T>::min());

                if (kernel[0][0]) r = v_max(r, vx_load(&in[0][l - shift]));
                if (kernel[0][1]) r = v_max(r, vx_load(&in[0][l        ]));
                if (kernel[0][2]) r = v_max(r, vx_load(&in[0][l + shift]));

                if (kernel[1][0]) r = v_max(r, vx_load(&in[1][l - shift]));
                if (kernel[1][1]) r = v_max(r, vx_load(&in[1][l        ]));
                if (kernel[1][2]) r = v_max(r, vx_load(&in[1][l + shift]));

                if (kernel[2][0]) r = v_max(r, vx_load(&in[2][l - shift]));
                if (kernel[2][1]) r = v_max(r, vx_load(&in[2][l        ]));
                if (kernel[2][2]) r = v_max(r, vx_load(&in[2][l + shift]));

                v_store(&out[l], r);
            }

            // tail (if any)
            if (l < length)
            {
                GAPI_DbgAssert(length >= nlanes);
                l = length - nlanes;
            }
        }
        return;
    }

    CV_Error(cv::Error::StsBadArg, "unsupported morphology");
}
#endif

template<typename T>
static void run_morphology3x3_code(T out[], const T *in[], int width, int chan,
                                   const uchar k[], MorphShape k_type,
                                   Morphology morphology)
{
#if CV_SIMD
    int length = width * chan;

    // length variable may be unused if types do not match at 'if' statements below
    (void) length;

    if (std::is_same<T, float>::value && length >= v_float32::nlanes)
    {
        run_morphology3x3_simd<float, v_float32>(reinterpret_cast<float*>(out),
                                                 reinterpret_cast<const float**>(in),
                                                 width, chan, k, k_type, morphology,
                                                 vx_setall_f32);
        return;
    }

    if (std::is_same<T, short>::value && length >= v_int16::nlanes)
    {
        run_morphology3x3_simd<short, v_int16>(reinterpret_cast<short*>(out),
                                               reinterpret_cast<const short**>(in),
                                               width, chan, k, k_type, morphology,
                                               vx_setall_s16);
        return;
    }

    if (std::is_same<T, ushort>::value && length >= v_uint16::nlanes)
    {
        run_morphology3x3_simd<ushort, v_uint16>(reinterpret_cast<ushort*>(out),
                                                 reinterpret_cast<const ushort**>(in),
                                                 width, chan, k, k_type, morphology,
                                                 vx_setall_u16);
        return;
    }

    if (std::is_same<T, uchar>::value && length >= v_uint8::nlanes)
    {
        run_morphology3x3_simd<uchar, v_uint8>(reinterpret_cast<uchar*>(out),
                                               reinterpret_cast<const uchar**>(in),
                                               width, chan, k, k_type, morphology,
                                               vx_setall_u8);
        return;
    }
#endif  // CV_SIMD

    run_morphology3x3_reference(out, in, width, chan, k, k_type, morphology);
}

#define RUN_MORPHOLOGY3X3_IMPL(T)                                        \
void run_morphology3x3_impl(T out[], const T *in[], int width, int chan, \
                            const uchar k[], MorphShape k_type,          \
                            Morphology morphology)                       \
{                                                                        \
    run_morphology3x3_code(out, in, width, chan, k, k_type, morphology); \
}

RUN_MORPHOLOGY3X3_IMPL(uchar )
RUN_MORPHOLOGY3X3_IMPL(ushort)
RUN_MORPHOLOGY3X3_IMPL( short)
RUN_MORPHOLOGY3X3_IMPL( float)

#undef RUN_MORPHOLOGY3X3_IMPL

//---------------------------
//
// Fluid kernels: Median blur
//
//---------------------------

template<typename T>
static void run_medblur3x3_reference(T out[], const T *in[], int width, int chan)
{
    constexpr int ksize = 3;
    constexpr int border = (ksize - 1) / 2;

    const int length = width * chan;
    const int shift = border * chan;

    for (int l=0; l < length; l++)
    {
        T t[3][3];

        // neighbourhood 3x3
        t[0][0] = in[0][l - shift];    t[0][1] = in[0][l];    t[0][2] = in[0][l + shift];
        t[1][0] = in[1][l - shift];    t[1][1] = in[1][l];    t[1][2] = in[1][l + shift];
        t[2][0] = in[2][l - shift];    t[2][1] = in[2][l];    t[2][2] = in[2][l + shift];

        // sort 2 values
        auto sort = [](T& a, T& b)
        {
            T u=a, v=b;
            a = (std::min)(u, v);
            b = (std::max)(u, v);
        };

        // horizontal: 3-elements bubble-sort per each row
        sort(t[0][0], t[0][1]);    sort(t[0][1], t[0][2]);    sort(t[0][0], t[0][1]);
        sort(t[1][0], t[1][1]);    sort(t[1][1], t[1][2]);    sort(t[1][0], t[1][1]);
        sort(t[2][0], t[2][1]);    sort(t[2][1], t[2][2]);    sort(t[2][0], t[2][1]);

        // vertical: columns bubble-sort (although partial)
        sort(t[0][0], t[1][0]);    sort(t[0][1], t[1][1]);  /*sort(t[0][2], t[1][2]);*/
        sort(t[1][0], t[2][0]);    sort(t[1][1], t[2][1]);    sort(t[1][2], t[2][2]);
      /*sort(t[0][0], t[1][0]);*/  sort(t[0][1], t[1][1]);    sort(t[0][2], t[1][2]);

        // diagonal: bubble-sort (in opposite order!)
        sort(t[1][1], t[0][2]);    sort(t[2][0], t[1][1]);    sort(t[1][1], t[0][2]);

        out[l] = t[1][1];
    }
}

#if CV_SIMD
template<typename VT, typename T>
static void run_medblur3x3_simd(T out[], const T *in[], int width, int chan)
{
    constexpr int ksize = 3;
    constexpr int border = (ksize - 1) / 2;

    const int length = width * chan;
    const int shift = border * chan;

    for (int l=0; l < length;)
    {
        constexpr int nlanes = VT::nlanes;

        // main part of output row
        for (; l <= length - nlanes; l += nlanes)
        {
            VT t00, t01, t02, t10, t11, t12, t20, t21, t22;

            // neighbourhood 3x3

            t00 = vx_load(&in[0][l - shift]);
            t01 = vx_load(&in[0][l        ]);
            t02 = vx_load(&in[0][l + shift]);

            t10 = vx_load(&in[1][l - shift]);
            t11 = vx_load(&in[1][l        ]);
            t12 = vx_load(&in[1][l + shift]);

            t20 = vx_load(&in[2][l - shift]);
            t21 = vx_load(&in[2][l        ]);
            t22 = vx_load(&in[2][l + shift]);

            // sort 2 values
            auto sort = [](VT& a, VT& b)
            {
                VT u=a, v=b;
                a = v_min(u, v);
                b = v_max(u, v);
            };

            // horizontal: 3-elements bubble-sort per each row
            sort(t00, t01);    sort(t01, t02);    sort(t00, t01);
            sort(t10, t11);    sort(t11, t12);    sort(t10, t11);
            sort(t20, t21);    sort(t21, t22);    sort(t20, t21);

            // vertical: columns bubble-sort (although partial)
            sort(t00, t10);    sort(t01, t11);  /*sort(t02, t12);*/
            sort(t10, t20);    sort(t11, t21);    sort(t12, t22);
          /*sort(t00, t10);*/  sort(t01, t11);    sort(t02, t12);

            // diagonal: bubble-sort (in opposite order!)
            sort(t11, t02);    sort(t20, t11);    sort(t11, t02);

            v_store(&out[l], t11);
        }

        // tail (if any)
        if (l < length)
        {
            GAPI_DbgAssert(length >= nlanes);
            l = length - nlanes;
        }
    }
}
#endif

template<typename T>
static void run_medblur3x3_code(T out[], const T *in[], int width, int chan)
{
#if CV_SIMD
    int length = width * chan;

    // length variable may be unused if types do not match at 'if' statements below
    (void) length;

    if (std::is_same<T, float>::value && length >= v_float32::nlanes)
    {
        run_medblur3x3_simd<v_float32>(reinterpret_cast<float*>(out),
                                       reinterpret_cast<const float**>(in),
                                       width, chan);
        return;
    }

    if (std::is_same<T, short>::value && length >= v_int16::nlanes)
    {
        run_medblur3x3_simd<v_int16>(reinterpret_cast<short*>(out),
                                     reinterpret_cast<const short**>(in),
                                     width, chan);
        return;
    }

    if (std::is_same<T, ushort>::value && length >= v_uint16::nlanes)
    {
        run_medblur3x3_simd<v_uint16>(reinterpret_cast<ushort*>(out),
                                      reinterpret_cast<const ushort**>(in),
                                      width, chan);
        return;
    }

    if (std::is_same<T, uchar>::value && length >= v_uint8::nlanes)
    {
        run_medblur3x3_simd<v_uint8>(reinterpret_cast<uchar*>(out),
                                     reinterpret_cast<const uchar**>(in),
                                     width, chan);
        return;
    }
#endif

    run_medblur3x3_reference(out, in, width, chan);
}

#define RUN_MEDBLUR3X3_IMPL(T)                                        \
void run_medblur3x3_impl(T out[], const T *in[], int width, int chan) \
{                                                                     \
    run_medblur3x3_code(out, in, width, chan);                        \
}

RUN_MEDBLUR3X3_IMPL(uchar )
RUN_MEDBLUR3X3_IMPL(ushort)
RUN_MEDBLUR3X3_IMPL( short)
RUN_MEDBLUR3X3_IMPL( float)

#undef RUN_MEDBLUR3X3_IMPL

//------------------------------------------------------------------------------

#endif  // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END

}  // namespace fluid
}  // namespace gapi
}  // namespace cv

#endif // !defined(GAPI_STANDALONE)
