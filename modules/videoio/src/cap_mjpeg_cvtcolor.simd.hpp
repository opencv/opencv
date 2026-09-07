// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, Advanced Micro Devices, Inc., all rights reserved.

#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include "opencv2/videoio/container_avi.private.hpp"

namespace cv {

CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void mjpeg_convertToYUV(int colorspace, int channels, int input_channels,
                        short* UV_data, short* Y_data, const uchar* pix_data,
                        int y_limit, int x_limit, int step, int u_plane_ofs, int v_plane_ofs);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

namespace {

#define DCT_DESCALE12(x) (((x) + (1 << 11)) >> 12)

static const int y_r = 1225;   // fix(0.299, 12)
static const int y_g = 2404;   // fix(0.587, 12)
static const int y_b = 467;    // fix(0.114, 12)
static const int cb_r = -691;  // -fix(0.1687, 12)
static const int cb_g = -1357; // -fix(0.3313, 12)
static const int cb_b = 2048;  //  fix(0.5, 12)
static const int cr_r = 2048;
static const int cr_g = -1715; // -fix(0.4187, 12)
static const int cr_b = -333;  // -fix(0.0813, 12)

#if CV_SIMD
static inline v_int32 descale12(const v_int32& x)
{
    return v_shr<12>(v_add(x, vx_setall_s32(1 << 11)));
}

static inline void yuv_from_rgb(const v_int32& r, const v_int32& g, const v_int32& b,
                                v_int32& Y, v_int32& U, v_int32& V)
{
    Y = v_sub(descale12(v_add(v_add(v_mul(r, vx_setall_s32(y_r)),
                                    v_mul(g, vx_setall_s32(y_g))),
                              v_mul(b, vx_setall_s32(y_b)))),
              vx_setall_s32(128));
    U = descale12(v_add(v_add(v_mul(r, vx_setall_s32(cb_r)),
                              v_mul(g, vx_setall_s32(cb_g))),
                        v_mul(b, vx_setall_s32(cb_b))));
    V = descale12(v_add(v_add(v_mul(r, vx_setall_s32(cr_r)),
                              v_mul(g, vx_setall_s32(cr_g))),
                        v_mul(b, vx_setall_s32(cr_b))));
}

// 16 BGR pixels (one MCU row): 128-bit deinterleave, then v_int32 lanes (4/8/16).
static inline void cvt_bgr16_two_rows(const uchar* row0, const uchar* row1,
                                      short* Y0, short* Y1, short* UV)
{
    v_uint8x16 b0, g0, r0, b1, g1, r1;
    v_load_deinterleave(row0, b0, g0, r0);
    v_load_deinterleave(row1, b1, g1, r1);

    alignas(64) uchar B0[16], G0[16], R0[16], B1[16], G1[16], R1[16];
    v_store(B0, b0); v_store(G0, g0); v_store(R0, r0);
    v_store(B1, b1); v_store(G1, g1); v_store(R1, r1);

    const int n32 = VTraits<v_int32>::vlanes();
    alignas(64) int uacc[16], vacc[16];
    for (int j = 0; j < 16; j += n32)
    {
        v_int32 b = v_reinterpret_as_s32(vx_load_expand_q(B0 + j));
        v_int32 g = v_reinterpret_as_s32(vx_load_expand_q(G0 + j));
        v_int32 r = v_reinterpret_as_s32(vx_load_expand_q(R0 + j));
        v_int32 Y, U, V;
        yuv_from_rgb(r, g, b, Y, U, V);
        v_pack_store(Y0 + j, Y);
        v_store(uacc + j, U);
        v_store(vacc + j, V);

        b = v_reinterpret_as_s32(vx_load_expand_q(B1 + j));
        g = v_reinterpret_as_s32(vx_load_expand_q(G1 + j));
        r = v_reinterpret_as_s32(vx_load_expand_q(R1 + j));
        yuv_from_rgb(r, g, b, Y, U, V);
        v_pack_store(Y1 + j, Y);
        v_store(uacc + j, v_add(vx_load(uacc + j), U));
        v_store(vacc + j, v_add(vx_load(vacc + j), V));
    }

    for (int k = 0; k < 16; k += 2)
    {
        UV[k >> 1] = (short)(uacc[k] + uacc[k + 1]);
        UV[(k >> 1) + 8] = (short)(vacc[k] + vacc[k + 1]);
    }
}

static inline void cvt_yuv444p_16x2(const uchar* pix, short* Y, short* UV,
                                    int step, int Y_step, int u_plane_ofs, int v_plane_ofs)
{
    const v_uint16x8 masklo = v_setall_u16(255);
    const v_int16x8 uv_bias = v_setall_s16((short)(128 * 4));
    const v_int16x8 y_bias = v_setall_s16(128);

    v_uint16x8 lane = v_load((const ushort*)(pix + v_plane_ofs));
    v_uint16x8 t1 = v_add(v_shr<8>(lane), v_and(lane, masklo));
    lane = v_load((const ushort*)(pix + v_plane_ofs + step));
    t1 = v_add(t1, v_add(v_shr<8>(lane), v_and(lane, masklo)));
    v_store(UV, v_sub(v_reinterpret_as_s16(t1), uv_bias));

    lane = v_load((const ushort*)(pix + u_plane_ofs));
    t1 = v_add(v_shr<8>(lane), v_and(lane, masklo));
    lane = v_load((const ushort*)(pix + u_plane_ofs + step));
    t1 = v_add(t1, v_add(v_shr<8>(lane), v_and(lane, masklo)));
    v_store(UV + 8, v_sub(v_reinterpret_as_s16(t1), uv_bias));

    v_store(Y, v_sub(v_reinterpret_as_s16(v_load_expand(pix)), y_bias));
    v_store(Y + 8, v_sub(v_reinterpret_as_s16(v_load_expand(pix + 8)), y_bias));
    v_store(Y + Y_step, v_sub(v_reinterpret_as_s16(v_load_expand(pix + step)), y_bias));
    v_store(Y + Y_step + 8, v_sub(v_reinterpret_as_s16(v_load_expand(pix + step + 8)), y_bias));
}
#endif

static inline void yuv_from_rgb_scalar(int r, int g, int b, int& Y, int& U, int& V)
{
    Y = DCT_DESCALE12(r * y_r + g * y_g + b * y_b) - 128;
    U = DCT_DESCALE12(r * cb_r + g * cb_g + b * cb_b);
    V = DCT_DESCALE12(r * cr_r + g * cr_g + b * cr_b);
}

static inline void cvt_yuv444p_2x2_scalar(const uchar* pix, short* Y, short* UV,
                                          int x_limit, int step, int Y_step,
                                          int input_channels, int u_plane_ofs, int v_plane_ofs)
{
    int j = 0;
    for (; j + 2 <= x_limit; j += 2, pix += 2)
    {
        Y[j] = (short)(pix[0] - 128);
        Y[j + 1] = (short)(pix[1] - 128);
        Y[j + Y_step] = (short)(pix[step] - 128);
        Y[j + Y_step + 1] = (short)(pix[step + 1] - 128);
        UV[j >> 1] = (short)(pix[v_plane_ofs] + pix[v_plane_ofs + 1] +
                             pix[v_plane_ofs + step] + pix[v_plane_ofs + step + 1] - 128 * 4);
        UV[(j >> 1) + 8] = (short)(pix[u_plane_ofs] + pix[u_plane_ofs + 1] +
                                   pix[u_plane_ofs + step] + pix[u_plane_ofs + step + 1] - 128 * 4);
    }
    for (; j < x_limit; j++, pix += input_channels)
    {
        Y[j] = (short)(pix[0] - 128);
        Y[j + Y_step] = (short)(pix[step] - 128);
        UV[j >> 1] = (short)(UV[j >> 1] + pix[v_plane_ofs] + pix[v_plane_ofs + step] - 128 * 2);
        UV[(j >> 1) + 8] = (short)(UV[(j >> 1) + 8] + pix[u_plane_ofs] + pix[u_plane_ofs + step] - 128 * 2);
    }
}

static inline void cvt_color_row_scalar(int colorspace, int input_channels, int x_scale,
                                        short* UV, short* Y, const uchar* pix,
                                        int x_limit, int u_plane_ofs, int v_plane_ofs)
{
    for (int j = 0; j < x_limit; j++, pix += input_channels)
    {
        int y, u, v;
        if (colorspace == COLORSPACE_BGR)
            yuv_from_rgb_scalar(pix[2], pix[1], pix[0], y, u, v);
        else if (colorspace == COLORSPACE_RGBA)
            yuv_from_rgb_scalar(pix[0], pix[1], pix[2], y, u, v);
        else
        {
            y = pix[0] - 128;
            u = pix[v_plane_ofs] - 128;
            v = pix[u_plane_ofs] - 128;
        }
        const int j2 = j >> (x_scale - 1);
        Y[j] = (short)y;
        UV[j2] = (short)(UV[j2] + u);
        UV[j2 + 8] = (short)(UV[j2 + 8] + v);
    }
}

} // namespace

void mjpeg_convertToYUV(int colorspace, int channels, int input_channels,
                        short* UV_data, short* Y_data, const uchar* pix_data,
                        int y_limit, int x_limit, int step, int u_plane_ofs, int v_plane_ofs)
{
    const int UV_step = 16;
    const int x_scale = channels > 1 ? 2 : 1;
    const int y_scale = x_scale;
    const int Y_step = x_scale * 8;

    if (channels > 1)
    {
        int i = 0;
        if (colorspace == COLORSPACE_YUV444P)
        {
#if CV_SIMD
            if (x_limit == 16)
            {
                for (; i + 2 <= y_limit; i += 2, pix_data += step * 2, Y_data += Y_step * 2, UV_data += UV_step)
                    cvt_yuv444p_16x2(pix_data, Y_data, UV_data, step, Y_step, u_plane_ofs, v_plane_ofs);
            }
#endif
            for (; i + 2 <= y_limit; i += 2, pix_data += step * 2, Y_data += Y_step * 2, UV_data += UV_step)
                cvt_yuv444p_2x2_scalar(pix_data, Y_data, UV_data, x_limit, step, Y_step,
                                       input_channels, u_plane_ofs, v_plane_ofs);
        }
        else
        {
#if CV_SIMD
            if (colorspace == COLORSPACE_BGR && input_channels == 3 && x_limit == 16 && x_scale == 2)
            {
                for (; i + 2 <= y_limit; i += 2, pix_data += step * 2, Y_data += Y_step * 2, UV_data += UV_step)
                    cvt_bgr16_two_rows(pix_data, pix_data + step, Y_data, Y_data + Y_step, UV_data);
            }
#endif
            for (; i < y_limit; i++, pix_data += step, Y_data += Y_step)
            {
                cvt_color_row_scalar(colorspace, input_channels, x_scale, UV_data, Y_data, pix_data,
                                     x_limit, u_plane_ofs, v_plane_ofs);
                if (((i + 1) & (y_scale - 1)) == 0)
                    UV_data += UV_step;
            }
        }
        if (i < y_limit)
        {
            for (; i < y_limit; i++, pix_data += step, Y_data += Y_step)
            {
                cvt_color_row_scalar(colorspace, input_channels, x_scale, UV_data, Y_data, pix_data,
                                     x_limit, u_plane_ofs, v_plane_ofs);
                if (((i + 1) & (y_scale - 1)) == 0)
                    UV_data += UV_step;
            }
        }
#if CV_SIMD
        vx_cleanup();
#endif
        return;
    }

    for (int i = 0; i < y_limit; i++, pix_data += step, Y_data += Y_step)
    {
        int j = 0;
#if CV_SIMD
        const int n32 = VTraits<v_int32>::vlanes();
        for (; j + n32 <= x_limit; j += n32)
        {
            v_int32 y = v_sub(v_shl<2>(v_reinterpret_as_s32(vx_load_expand_q(pix_data + j))),
                              vx_setall_s32(512));
            v_pack_store(Y_data + j, y);
        }
#endif
        for (; j < x_limit; j++)
            Y_data[j] = (short)(pix_data[j] * 4 - 128 * 4);
    }
#if CV_SIMD
    vx_cleanup();
#endif
}

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace cv
