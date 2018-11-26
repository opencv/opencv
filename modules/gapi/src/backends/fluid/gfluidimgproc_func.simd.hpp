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
// Fluid kernels: RGB-to-YUV, YUV-to-RGB
//
//--------------------------------------

void run_rgb2yuv_impl(uchar out[], const uchar in[], int width, const float coef[5]);

void run_yuv2rgb_impl(uchar out[], const uchar in[], int width, const float coef[4]);

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

//-------------------------
//
// Fluid kernels: sepFilter
//
//-------------------------

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
#endif

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

//------------------------------------------------------------------------------

#endif  // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END

}  // namespace fluid
}  // namespace gapi
}  // namespace cv

#endif // !defined(GAPI_STANDALONE)
