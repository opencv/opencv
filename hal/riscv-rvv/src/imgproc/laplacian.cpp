// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2025.

#include "rvv_hal.hpp"
#include "common.hpp"

#include <vector>
#include <limits>
#include <algorithm>

namespace cv { namespace rvv_hal { namespace imgproc {

#if CV_HAL_RVV_1P0_ENABLED

namespace {

static inline int borderMap(int x, int len, int border_type)
{
    return common::borderInterpolate(x, len, border_type);
}

static inline uint8_t sample_u8(const uint8_t* src, size_t step, int width, int height, int y, int x, int border_type, uint8_t border_value)
{
    int yy = borderMap(y, height, border_type);
    int xx = borderMap(x, width , border_type);
    if (yy < 0 || xx < 0)
        return border_value;
    return src[yy * step + xx];
}

static inline bool isLaplacian3x3Supported(int width, int height, int border_type)
{
    return width >= 8 && height >= 1 &&
           (border_type == BORDER_CONSTANT ||
            border_type == BORDER_REPLICATE);
}

static inline bool isLaplacianOpenCVSupported(int width, int height, int border_type)
{
    return width >= 8 && height >= 1 &&
           (border_type == BORDER_CONSTANT   ||
            border_type == BORDER_REPLICATE  ||
            border_type == BORDER_REFLECT    ||
            border_type == BORDER_REFLECT_101);
}

static inline vint16m2_t vw_u8_to_i16(vuint8m1_t v, size_t vl)
{
    vuint16m2_t u16 = __riscv_vwaddu_vx_u16m2(v, 0, vl);
    return __riscv_vreinterpret_v_u16m2_i16m2(u16);
}

static inline vuint8m1_t clamp_to_u8(vint16m2_t v, size_t vl)
{
    vint16m2_t zero = __riscv_vmv_v_x_i16m2(0, vl);
    vint16m2_t maxv = __riscv_vmv_v_x_i16m2(255, vl);
    v = __riscv_vmax_vv_i16m2(v, zero, vl);
    v = __riscv_vmin_vv_i16m2(v, maxv, vl);
    vuint16m2_t u16 = __riscv_vreinterpret_v_i16m2_u16m2(v);
    return __riscv_vnclipu(u16, 0, __RISCV_VXRM_RNU, vl);
}

static int laplacian3x3_u8(int start, int end,
                           const uint8_t* src_data, size_t src_step,
                           uint8_t* dst_data, size_t dst_step,
                           int width, int height,
                           int border_type, uint8_t border_value)
{
    if (!isLaplacian3x3Supported(width, height, border_type))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    for (int y = start; y < end; ++y)
    {
        int y0 = borderMap(y - 1, height, border_type);
        int y1 = borderMap(y    , height, border_type);
        int y2 = borderMap(y + 1, height, border_type);

        const uint8_t* r0 = (y0 < 0) ? nullptr : src_data + y0 * src_step;
        const uint8_t* r1 = (y1 < 0) ? nullptr : src_data + y1 * src_step;
        const uint8_t* r2 = (y2 < 0) ? nullptr : src_data + y2 * src_step;

        uint8_t* drow = dst_data + y * dst_step;

        int left = 1;
        int right = width - 1;
        if (right < left)
        {
            for (int x = 0; x < width; ++x)
            {
                int v0l = sample_u8(src_data, src_step, width, height, y - 1, x - 1, border_type, border_value);
                int v0r = sample_u8(src_data, src_step, width, height, y - 1, x + 1, border_type, border_value);
                int v2l = sample_u8(src_data, src_step, width, height, y + 1, x - 1, border_type, border_value);
                int v2r = sample_u8(src_data, src_step, width, height, y + 1, x + 1, border_type, border_value);
                int v1 = sample_u8(src_data, src_step, width, height, y, x, border_type, border_value);
                int val = (v0l + v0r + v2l + v2r - 4 * v1) * 2;
                val = std::max(0, std::min(255, val));
                drow[x] = (uint8_t)val;
            }
            continue;
        }

        for (int x = 0; x < left; ++x)
        {
            int v0l = sample_u8(src_data, src_step, width, height, y - 1, x - 1, border_type, border_value);
            int v0r = sample_u8(src_data, src_step, width, height, y - 1, x + 1, border_type, border_value);
            int v2l = sample_u8(src_data, src_step, width, height, y + 1, x - 1, border_type, border_value);
            int v2r = sample_u8(src_data, src_step, width, height, y + 1, x + 1, border_type, border_value);
            int v1 = sample_u8(src_data, src_step, width, height, y, x, border_type, border_value);
            int val = (v0l + v0r + v2l + v2r - 4 * v1) * 2;
            val = std::max(0, std::min(255, val));
            drow[x] = (uint8_t)val;
        }

        int x = left;
        for (; x < right; )
        {
            size_t vl = __riscv_vsetvl_e8m1(right - x);

            __builtin_prefetch(r0 ? (r0 + x - 1) : nullptr, 0, 3);
            __builtin_prefetch(r1 + x, 0, 3);
            __builtin_prefetch(r2 ? (r2 + x - 1) : nullptr, 0, 3);

            vuint8m1_t v0l = r0 ? __riscv_vle8_v_u8m1(r0 + x - 1, vl) : __riscv_vmv_v_x_u8m1(border_value, vl);
            vuint8m1_t v0r = r0 ? __riscv_vle8_v_u8m1(r0 + x + 1, vl) : __riscv_vmv_v_x_u8m1(border_value, vl);
            vuint8m1_t v2l = r2 ? __riscv_vle8_v_u8m1(r2 + x - 1, vl) : __riscv_vmv_v_x_u8m1(border_value, vl);
            vuint8m1_t v2r = r2 ? __riscv_vle8_v_u8m1(r2 + x + 1, vl) : __riscv_vmv_v_x_u8m1(border_value, vl);
            vuint8m1_t v1  = r1 ? __riscv_vle8_v_u8m1(r1 + x, vl) : __riscv_vmv_v_x_u8m1(border_value, vl);

            vint16m2_t s0l = vw_u8_to_i16(v0l, vl);
            vint16m2_t s0r = vw_u8_to_i16(v0r, vl);
            vint16m2_t s2l = vw_u8_to_i16(v2l, vl);
            vint16m2_t s2r = vw_u8_to_i16(v2r, vl);
            vint16m2_t s1  = vw_u8_to_i16(v1, vl);

            vint16m2_t sum = __riscv_vadd_vv_i16m2(s0l, s0r, vl);
            sum = __riscv_vadd_vv_i16m2(sum, s2l, vl);
            sum = __riscv_vadd_vv_i16m2(sum, s2r, vl);
            sum = __riscv_vsub_vv_i16m2(sum, __riscv_vsll_vx_i16m2(s1, 2, vl), vl);
            sum = __riscv_vsll_vx_i16m2(sum, 1, vl);

            vuint8m1_t out = clamp_to_u8(sum, vl);
            __riscv_vse8_v_u8m1(drow + x, out, vl);
            x += vl;
        }

        for (; x < width; ++x)
        {
            int v0l = sample_u8(src_data, src_step, width, height, y - 1, x - 1, border_type, border_value);
            int v0r = sample_u8(src_data, src_step, width, height, y - 1, x + 1, border_type, border_value);
            int v2l = sample_u8(src_data, src_step, width, height, y + 1, x - 1, border_type, border_value);
            int v2r = sample_u8(src_data, src_step, width, height, y + 1, x + 1, border_type, border_value);
            int v1 = sample_u8(src_data, src_step, width, height, y, x, border_type, border_value);
            int val = (v0l + v0r + v2l + v2r - 4 * v1) * 2;
            val = std::max(0, std::min(255, val));
            drow[x] = (uint8_t)val;
        }
    }

    return CV_HAL_ERROR_OK;
}

static int laplacian1(int start, int end,
                      const uint8_t* src_data, size_t src_step,
                      int16_t* dst_data, size_t dst_step,
                      int width, int height,
                      int border_type, uint8_t border_value)
{
    if (!isLaplacianOpenCVSupported(width, height, border_type))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    for (int y = start; y < end; ++y)
    {
        int y0 = borderMap(y - 1, height, border_type);
        int y1 = borderMap(y    , height, border_type);
        int y2 = borderMap(y + 1, height, border_type);

        const uint8_t* r0 = (y0 < 0) ? nullptr : src_data + y0 * src_step;
        const uint8_t* r1 = (y1 < 0) ? nullptr : src_data + y1 * src_step;
        const uint8_t* r2 = (y2 < 0) ? nullptr : src_data + y2 * src_step;

        int16_t* drow = reinterpret_cast<int16_t*>(reinterpret_cast<uint8_t*>(dst_data) + y * dst_step);

        int left = 1;
        int right = width - 1;

        for (int x = 0; x < left; ++x)
        {
            int v0 = sample_u8(src_data, src_step, width, height, y - 1, x, border_type, border_value);
            int v2 = sample_u8(src_data, src_step, width, height, y + 1, x, border_type, border_value);
            int v1l = sample_u8(src_data, src_step, width, height, y, x - 1, border_type, border_value);
            int v1r = sample_u8(src_data, src_step, width, height, y, x + 1, border_type, border_value);
            int v1 = sample_u8(src_data, src_step, width, height, y, x, border_type, border_value);
            drow[x] = (int16_t)(v0 + v2 + v1l + v1r - 4 * v1);
        }

        int x = left;
        for (; x < right; )
        {
            size_t vl = __riscv_vsetvl_e8m1(right - x);
            vuint8m1_t v0 = r0 ? __riscv_vle8_v_u8m1(r0 + x, vl) : __riscv_vmv_v_x_u8m1(border_value, vl);
            vuint8m1_t v2 = r2 ? __riscv_vle8_v_u8m1(r2 + x, vl) : __riscv_vmv_v_x_u8m1(border_value, vl);
            vuint8m1_t v1 = r1 ? __riscv_vle8_v_u8m1(r1 + x, vl) : __riscv_vmv_v_x_u8m1(border_value, vl);
            vuint8m1_t v1l = r1 ? __riscv_vle8_v_u8m1(r1 + x - 1, vl) : __riscv_vmv_v_x_u8m1(border_value, vl);
            vuint8m1_t v1r = r1 ? __riscv_vle8_v_u8m1(r1 + x + 1, vl) : __riscv_vmv_v_x_u8m1(border_value, vl);

            vint16m2_t s0 = vw_u8_to_i16(v0, vl);
            vint16m2_t s2 = vw_u8_to_i16(v2, vl);
            vint16m2_t s1 = vw_u8_to_i16(v1, vl);
            vint16m2_t s1l = vw_u8_to_i16(v1l, vl);
            vint16m2_t s1r = vw_u8_to_i16(v1r, vl);

            vint16m2_t sum = __riscv_vadd_vv_i16m2(s0, s2, vl);
            sum = __riscv_vadd_vv_i16m2(sum, s1l, vl);
            sum = __riscv_vadd_vv_i16m2(sum, s1r, vl);
            sum = __riscv_vsub_vv_i16m2(sum, __riscv_vsll_vx_i16m2(s1, 2, vl), vl);

            __riscv_vse16_v_i16m2(drow + x, sum, vl);
            x += vl;
        }

        for (; x < width; ++x)
        {
            int v0 = sample_u8(src_data, src_step, width, height, y - 1, x, border_type, border_value);
            int v2 = sample_u8(src_data, src_step, width, height, y + 1, x, border_type, border_value);
            int v1l = sample_u8(src_data, src_step, width, height, y, x - 1, border_type, border_value);
            int v1r = sample_u8(src_data, src_step, width, height, y, x + 1, border_type, border_value);
            int v1 = sample_u8(src_data, src_step, width, height, y, x, border_type, border_value);
            drow[x] = (int16_t)(v0 + v2 + v1l + v1r - 4 * v1);
        }
    }

    return CV_HAL_ERROR_OK;
}

static int laplacian3(int start, int end,
                      const uint8_t* src_data, size_t src_step,
                      int16_t* dst_data, size_t dst_step,
                      int width, int height,
                      int border_type, uint8_t border_value)
{
    if (!isLaplacianOpenCVSupported(width, height, border_type))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    for (int y = start; y < end; ++y)
    {
        int y0 = borderMap(y - 1, height, border_type);
        int y1 = borderMap(y    , height, border_type);
        int y2 = borderMap(y + 1, height, border_type);

        const uint8_t* r0 = (y0 < 0) ? nullptr : src_data + y0 * src_step;
        const uint8_t* r1 = (y1 < 0) ? nullptr : src_data + y1 * src_step;
        const uint8_t* r2 = (y2 < 0) ? nullptr : src_data + y2 * src_step;

        int16_t* drow = reinterpret_cast<int16_t*>(reinterpret_cast<uint8_t*>(dst_data) + y * dst_step);

        int left = 1;
        int right = width - 1;

        for (int x = 0; x < left; ++x)
        {
            int v0l = sample_u8(src_data, src_step, width, height, y - 1, x - 1, border_type, border_value);
            int v0r = sample_u8(src_data, src_step, width, height, y - 1, x + 1, border_type, border_value);
            int v2l = sample_u8(src_data, src_step, width, height, y + 1, x - 1, border_type, border_value);
            int v2r = sample_u8(src_data, src_step, width, height, y + 1, x + 1, border_type, border_value);
            int v1  = sample_u8(src_data, src_step, width, height, y, x, border_type, border_value);

            int res = (v0l + v0r + v2l + v2r - 4 * v1) * 2;
            drow[x] = (int16_t)res;
        }

        int x = left;
        for (; x < right; )
        {
            size_t vl = __riscv_vsetvl_e8m1(right - x);

            vuint8m1_t v0l = r0 ? __riscv_vle8_v_u8m1(r0 + x - 1, vl) : __riscv_vmv_v_x_u8m1(border_value, vl);
            vuint8m1_t v0r = r0 ? __riscv_vle8_v_u8m1(r0 + x + 1, vl) : __riscv_vmv_v_x_u8m1(border_value, vl);
            vuint8m1_t v2l = r2 ? __riscv_vle8_v_u8m1(r2 + x - 1, vl) : __riscv_vmv_v_x_u8m1(border_value, vl);
            vuint8m1_t v2r = r2 ? __riscv_vle8_v_u8m1(r2 + x + 1, vl) : __riscv_vmv_v_x_u8m1(border_value, vl);
            vuint8m1_t v1  = r1 ? __riscv_vle8_v_u8m1(r1 + x, vl) : __riscv_vmv_v_x_u8m1(border_value, vl);

            vint16m2_t s0l = vw_u8_to_i16(v0l, vl);
            vint16m2_t s0r = vw_u8_to_i16(v0r, vl);
            vint16m2_t s2l = vw_u8_to_i16(v2l, vl);
            vint16m2_t s2r = vw_u8_to_i16(v2r, vl);
            vint16m2_t s1  = vw_u8_to_i16(v1, vl);

            vint16m2_t sum = __riscv_vadd_vv_i16m2(s0l, s0r, vl);
            sum = __riscv_vadd_vv_i16m2(sum, s2l, vl);
            sum = __riscv_vadd_vv_i16m2(sum, s2r, vl);

            sum = __riscv_vsub_vv_i16m2(sum, __riscv_vsll_vx_i16m2(s1, 2, vl), vl);
            sum = __riscv_vsll_vx_i16m2(sum, 1, vl);

            __riscv_vse16_v_i16m2(drow + x, sum, vl);
            x += vl;
        }

        for (; x < width; ++x)
        {
            int v0l = sample_u8(src_data, src_step, width, height, y - 1, x - 1, border_type, border_value);
            int v0r = sample_u8(src_data, src_step, width, height, y - 1, x + 1, border_type, border_value);
            int v2l = sample_u8(src_data, src_step, width, height, y + 1, x - 1, border_type, border_value);
            int v2r = sample_u8(src_data, src_step, width, height, y + 1, x + 1, border_type, border_value);
            int v1  = sample_u8(src_data, src_step, width, height, y, x, border_type, border_value);

            int res = (v0l + v0r + v2l + v2r - 4 * v1) * 2;
            drow[x] = (int16_t)res;
        }
    }

    return CV_HAL_ERROR_OK;
}

static int laplacian5(int start, int end,
                      const uint8_t* src_data, size_t src_step,
                      int16_t* dst_data, size_t dst_step,
                      int width, int height,
                      int border_type, uint8_t border_value)
{
    if (!isLaplacianOpenCVSupported(width, height, border_type))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    for (int y = start; y < end; ++y)
    {
        int y0 = borderMap(y - 2, height, border_type);
        int y1 = borderMap(y - 1, height, border_type);
        int y2 = borderMap(y    , height, border_type);
        int y3 = borderMap(y + 1, height, border_type);
        int y4 = borderMap(y + 2, height, border_type);

        const uint8_t* r0 = (y0 < 0) ? nullptr : src_data + y0 * src_step;
        const uint8_t* r1 = (y1 < 0) ? nullptr : src_data + y1 * src_step;
        const uint8_t* r2 = (y2 < 0) ? nullptr : src_data + y2 * src_step;
        const uint8_t* r3 = (y3 < 0) ? nullptr : src_data + y3 * src_step;
        const uint8_t* r4 = (y4 < 0) ? nullptr : src_data + y4 * src_step;

        int16_t* drow = reinterpret_cast<int16_t*>(reinterpret_cast<uint8_t*>(dst_data) + y * dst_step);

        int left = 2;
        int right = width - 2;

        for (int x = 0; x < left; ++x)
        {
            auto v = [&](int yy, int xx){ return sample_u8(src_data, src_step, width, height, yy, xx, border_type, border_value); };

            int pprevx = v(y-2,x-2) + 2*v(y-1,x-2) + 2*v(y,x-2) + 2*v(y+1,x-2) + v(y+2,x-2);
            int prevx  = 2*v(y-2,x-1) - 4*v(y,x-1) + 2*v(y+2,x-1);
            int currx  = 2*v(y-2,x) - 4*v(y-1,x) - 12*v(y,x) - 4*v(y+1,x) + 2*v(y+2,x);
            int nextx  = 2*v(y-2,x+1) - 4*v(y,x+1) + 2*v(y+2,x+1);
            int nnextx = v(y-2,x+2) + 2*v(y-1,x+2) + 2*v(y,x+2) + 2*v(y+1,x+2) + v(y+2,x+2);

            int res = (pprevx + prevx + currx + nextx + nnextx) * 2;
            drow[x] = (int16_t)res;
        }

        int x = left;
        for (; x < right; )
        {
            size_t vl = __riscv_vsetvl_e8m1(right - x);

            vuint8m1_t r0m2 = r0 ? __riscv_vle8_v_u8m1(r0 + x - 2, vl) : __riscv_vmv_v_x_u8m1(border_value, vl);
            vuint8m1_t r0m1 = r0 ? __riscv_vle8_v_u8m1(r0 + x - 1, vl) : __riscv_vmv_v_x_u8m1(border_value, vl);
            vuint8m1_t r0p0 = r0 ? __riscv_vle8_v_u8m1(r0 + x,     vl) : __riscv_vmv_v_x_u8m1(border_value, vl);
            vuint8m1_t r0p1 = r0 ? __riscv_vle8_v_u8m1(r0 + x + 1, vl) : __riscv_vmv_v_x_u8m1(border_value, vl);
            vuint8m1_t r0p2 = r0 ? __riscv_vle8_v_u8m1(r0 + x + 2, vl) : __riscv_vmv_v_x_u8m1(border_value, vl);

            vuint8m1_t r1m2 = r1 ? __riscv_vle8_v_u8m1(r1 + x - 2, vl) : __riscv_vmv_v_x_u8m1(border_value, vl);
            vuint8m1_t r1p0 = r1 ? __riscv_vle8_v_u8m1(r1 + x,     vl) : __riscv_vmv_v_x_u8m1(border_value, vl);
            vuint8m1_t r1p2 = r1 ? __riscv_vle8_v_u8m1(r1 + x + 2, vl) : __riscv_vmv_v_x_u8m1(border_value, vl);

            vuint8m1_t r2m2 = r2 ? __riscv_vle8_v_u8m1(r2 + x - 2, vl) : __riscv_vmv_v_x_u8m1(border_value, vl);
            vuint8m1_t r2m1 = r2 ? __riscv_vle8_v_u8m1(r2 + x - 1, vl) : __riscv_vmv_v_x_u8m1(border_value, vl);
            vuint8m1_t r2p0 = r2 ? __riscv_vle8_v_u8m1(r2 + x,     vl) : __riscv_vmv_v_x_u8m1(border_value, vl);
            vuint8m1_t r2p1 = r2 ? __riscv_vle8_v_u8m1(r2 + x + 1, vl) : __riscv_vmv_v_x_u8m1(border_value, vl);
            vuint8m1_t r2p2 = r2 ? __riscv_vle8_v_u8m1(r2 + x + 2, vl) : __riscv_vmv_v_x_u8m1(border_value, vl);

            vuint8m1_t r3m2 = r3 ? __riscv_vle8_v_u8m1(r3 + x - 2, vl) : __riscv_vmv_v_x_u8m1(border_value, vl);
            vuint8m1_t r3p0 = r3 ? __riscv_vle8_v_u8m1(r3 + x,     vl) : __riscv_vmv_v_x_u8m1(border_value, vl);
            vuint8m1_t r3p2 = r3 ? __riscv_vle8_v_u8m1(r3 + x + 2, vl) : __riscv_vmv_v_x_u8m1(border_value, vl);

            vuint8m1_t r4m2 = r4 ? __riscv_vle8_v_u8m1(r4 + x - 2, vl) : __riscv_vmv_v_x_u8m1(border_value, vl);
            vuint8m1_t r4m1 = r4 ? __riscv_vle8_v_u8m1(r4 + x - 1, vl) : __riscv_vmv_v_x_u8m1(border_value, vl);
            vuint8m1_t r4p0 = r4 ? __riscv_vle8_v_u8m1(r4 + x,     vl) : __riscv_vmv_v_x_u8m1(border_value, vl);
            vuint8m1_t r4p1 = r4 ? __riscv_vle8_v_u8m1(r4 + x + 1, vl) : __riscv_vmv_v_x_u8m1(border_value, vl);
            vuint8m1_t r4p2 = r4 ? __riscv_vle8_v_u8m1(r4 + x + 2, vl) : __riscv_vmv_v_x_u8m1(border_value, vl);

            auto I = [&](vuint8m1_t v){ return vw_u8_to_i16(v, vl); };

            vint16m2_t pprevx = I(r0m2);
            pprevx = __riscv_vadd_vv_i16m2(pprevx, __riscv_vsll_vx_i16m2(I(r1m2),1,vl), vl);
            pprevx = __riscv_vadd_vv_i16m2(pprevx, __riscv_vsll_vx_i16m2(I(r2m2),1,vl), vl);
            pprevx = __riscv_vadd_vv_i16m2(pprevx, __riscv_vsll_vx_i16m2(I(r3m2),1,vl), vl);
            pprevx = __riscv_vadd_vv_i16m2(pprevx, I(r4m2), vl);

            vint16m2_t prevx = __riscv_vsll_vx_i16m2(I(r0m1),1,vl);
            prevx = __riscv_vsub_vv_i16m2(prevx, __riscv_vsll_vx_i16m2(I(r2m1),2,vl), vl);
            prevx = __riscv_vadd_vv_i16m2(prevx, __riscv_vsll_vx_i16m2(I(r4m1),1,vl), vl);

            vint16m2_t currx = __riscv_vsll_vx_i16m2(I(r0p0),1,vl);
            currx = __riscv_vsub_vv_i16m2(currx, __riscv_vsll_vx_i16m2(I(r1p0),2,vl), vl);
            currx = __riscv_vsub_vv_i16m2(currx, __riscv_vsll_vx_i16m2(I(r2p0),3,vl), vl);
            currx = __riscv_vsub_vv_i16m2(currx, __riscv_vsll_vx_i16m2(I(r3p0),2,vl), vl);
            currx = __riscv_vadd_vv_i16m2(currx, __riscv_vsll_vx_i16m2(I(r4p0),1,vl), vl);

            vint16m2_t nextx = __riscv_vsll_vx_i16m2(I(r0p1),1,vl);
            nextx = __riscv_vsub_vv_i16m2(nextx, __riscv_vsll_vx_i16m2(I(r2p1),2,vl), vl);
            nextx = __riscv_vadd_vv_i16m2(nextx, __riscv_vsll_vx_i16m2(I(r4p1),1,vl), vl);

            vint16m2_t nnextx = I(r0p2);
            nnextx = __riscv_vadd_vv_i16m2(nnextx, __riscv_vsll_vx_i16m2(I(r1p2),1,vl), vl);
            nnextx = __riscv_vadd_vv_i16m2(nnextx, __riscv_vsll_vx_i16m2(I(r2p2),1,vl), vl);
            nnextx = __riscv_vadd_vv_i16m2(nnextx, __riscv_vsll_vx_i16m2(I(r3p2),1,vl), vl);
            nnextx = __riscv_vadd_vv_i16m2(nnextx, I(r4p2), vl);

            vint16m2_t sum = __riscv_vadd_vv_i16m2(pprevx, prevx, vl);
            sum = __riscv_vadd_vv_i16m2(sum, currx, vl);
            sum = __riscv_vadd_vv_i16m2(sum, nextx, vl);
            sum = __riscv_vadd_vv_i16m2(sum, nnextx, vl);
            sum = __riscv_vsll_vx_i16m2(sum, 1, vl);

            __riscv_vse16_v_i16m2(drow + x, sum, vl);
            x += vl;
        }

        for (; x < width; ++x)
        {
            auto v = [&](int yy, int xx){ return sample_u8(src_data, src_step, width, height, yy, xx, border_type, border_value); };

            int pprevx = v(y-2,x-2) + 2*v(y-1,x-2) + 2*v(y,x-2) + 2*v(y+1,x-2) + v(y+2,x-2);
            int prevx  = 2*v(y-2,x-1) - 4*v(y,x-1) + 2*v(y+2,x-1);
            int currx  = 2*v(y-2,x) - 4*v(y-1,x) - 12*v(y,x) - 4*v(y+1,x) + 2*v(y+2,x);
            int nextx  = 2*v(y-2,x+1) - 4*v(y,x+1) + 2*v(y+2,x+1);
            int nnextx = v(y-2,x+2) + 2*v(y-1,x+2) + 2*v(y,x+2) + 2*v(y+1,x+2) + v(y+2,x+2);

            int res = (pprevx + prevx + currx + nextx + nnextx) * 2;
            drow[x] = (int16_t)res;
        }
    }

    return CV_HAL_ERROR_OK;
}

} // anonymous

int laplacian(const uint8_t* src_data, size_t src_step,
              uint8_t* dst_data, size_t dst_step,
              int width, int height,
              int src_depth, int dst_depth, int cn,
              int ksize, int border_type, uint8_t border_value)
{
    if (src_data == dst_data)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if (src_depth != CV_8U || cn != 1)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if (ksize != 1 && ksize != 3 && ksize != 5)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if (dst_depth != CV_8U && dst_depth != CV_16S)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if (dst_depth == CV_8U)
    {
        if (ksize != 3) // only 3x3 path for u8
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        return common::invoke(height, {laplacian3x3_u8},
                              src_data, src_step, dst_data, dst_step,
                              width, height, border_type, border_value);
    }

    if (ksize == 1)
    {
        return common::invoke(height, {laplacian1},
                              src_data, src_step,
                              reinterpret_cast<int16_t*>(dst_data), dst_step,
                              width, height, border_type, border_value);
    }
    if (ksize == 3)
    {
        return common::invoke(height, {laplacian3},
                              src_data, src_step,
                              reinterpret_cast<int16_t*>(dst_data), dst_step,
                              width, height, border_type, border_value);
    }
    if (ksize == 5)
    {
        return common::invoke(height, {laplacian5},
                              src_data, src_step,
                              reinterpret_cast<int16_t*>(dst_data), dst_step,
                              width, height, border_type, border_value);
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

#endif // CV_HAL_RVV_1P0_ENABLED

}}} // cv::rvv_hal::imgproc