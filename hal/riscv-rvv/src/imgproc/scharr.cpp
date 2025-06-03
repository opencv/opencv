// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, SpaceMIT Inc., all rights reserved.

#include "rvv_hal.hpp"
#include "common.hpp"

namespace cv
{
    namespace rvv_hal
    {
        namespace imgproc
        {

#if CV_HAL_RVV_1P0_ENABLED

#define borderValue 0

            static int scharr_border(const uint8_t *src_data, size_t src_step,
                                     int width, int height,
                                     int border_type,
                                     int margin_left, int margin_top,
                                     int margin_right, int margin_bottom)
            {
                const int paddedWidth = width + margin_left + margin_right;
                const int paddedHeight = height + margin_top + margin_bottom;
                const int startX = margin_left;
                const int startY = margin_top;

                uint8_t *tempBuffer = new uint8_t[paddedWidth * paddedHeight]();

                for (int y = 0; y < height; y++)
                {
                    const uint8_t *src_row = src_data + y * src_step;
                    uint8_t *dst_row = tempBuffer + (y + startY) * paddedWidth + startX;
                    memcpy(dst_row, src_row, width * sizeof(uint8_t));
                }

                if (margin_left > 0)
                {
                    for (int y = startY; y < startY + height; y++)
                    {
                        for (int x = 0; x < margin_left; x++)
                        {
                            uint8_t *ptr = tempBuffer + y * paddedWidth + x;
                            switch (border_type)
                            {
                            case BORDER_REPLICATE:
                                *ptr = tempBuffer[y * paddedWidth + startX];
                                break;
                            case BORDER_REFLECT:
                            {
                                int offset = (startX - x) % width;
                                *ptr = tempBuffer[y * paddedWidth + (startX + offset)];
                                break;
                            }
                            case BORDER_REFLECT_101:
                            {
                                int relativeX = x - startX;
                                int period = 2 * (width - 1);
                                if (period <= 0)
                                {
                                    relativeX = 0;
                                }
                                else
                                {
                                    relativeX = relativeX % period;
                                    if (relativeX < 0)
                                        relativeX += period;
                                    if (relativeX >= width)
                                        relativeX = 2 * (width - 1) - relativeX;
                                }
                                int originalX = startX + relativeX;
                                *ptr = tempBuffer[y * paddedWidth + originalX];
                                break;
                            }
                            case BORDER_CONSTANT:
                                *ptr = borderValue;
                                break;
                            default:
                                delete[] tempBuffer;
                                return CV_HAL_ERROR_NOT_IMPLEMENTED;
                            }
                        }
                    }
                }

                if (margin_right > 0)
                {
                    const int right_start = startX + width;
                    for (int y = startY; y < startY + height; y++)
                    {
                        for (int x = right_start; x < paddedWidth; x++)
                        {
                            uint8_t *ptr = tempBuffer + y * paddedWidth + x;
                            switch (border_type)
                            {
                            case BORDER_REPLICATE:
                                *ptr = tempBuffer[y * paddedWidth + (right_start - 1)];
                                break;
                            case BORDER_REFLECT:
                            {
                                int offset = (x - (right_start - 1)) % width;
                                *ptr = tempBuffer[y * paddedWidth + (right_start - 1 - offset)];
                                break;
                            }
                            case BORDER_REFLECT_101:
                            {
                                int relativeX = x - startX;
                                int period = 2 * (width - 1);
                                if (period <= 0)
                                {
                                    relativeX = 0;
                                }
                                else
                                {
                                    relativeX = relativeX % period;
                                    if (relativeX < 0)
                                        relativeX += period;
                                    if (relativeX >= width)
                                        relativeX = 2 * (width - 1) - relativeX;
                                }
                                int originalX = startX + relativeX;
                                *ptr = tempBuffer[y * paddedWidth + originalX];
                                break;
                            }
                            case BORDER_CONSTANT:
                                *ptr = borderValue;
                                break;
                            default:
                                delete[] tempBuffer;
                                return CV_HAL_ERROR_NOT_IMPLEMENTED;
                            }
                        }
                    }
                }

                if (margin_top > 0)
                {
                    for (int y = 0; y < margin_top; y++)
                    {
                        for (int x = 0; x < paddedWidth; x++)
                        {
                            uint8_t *ptr = tempBuffer + y * paddedWidth + x;
                            switch (border_type)
                            {
                            case BORDER_REPLICATE:
                                *ptr = tempBuffer[margin_top * paddedWidth + x];
                                break;
                            case BORDER_REFLECT:
                            {
                                int offset = (margin_top - y) % height;
                                *ptr = tempBuffer[(margin_top + offset) * paddedWidth + x];
                                break;
                            }
                            case BORDER_REFLECT_101:
                            {
                                int relativeY = y - startY;
                                int period = 2 * (height - 1);
                                if (period <= 0)
                                {
                                    relativeY = 0;
                                }
                                else
                                {
                                    relativeY = relativeY % period;
                                    if (relativeY < 0)
                                        relativeY += period;
                                    if (relativeY >= height)
                                        relativeY = 2 * (height - 1) - relativeY;
                                }
                                int originalY = startY + relativeY;
                                *ptr = tempBuffer[originalY * paddedWidth + x];
                                break;
                            }
                            case BORDER_CONSTANT:
                                *ptr = borderValue;
                                break;
                            default:
                                delete[] tempBuffer;
                                return CV_HAL_ERROR_NOT_IMPLEMENTED;
                            }
                        }
                    }
                }

                if (margin_bottom > 0)
                {
                    const int bottom_start = startY + height;
                    for (int y = bottom_start; y < paddedHeight; y++)
                    {
                        for (int x = 0; x < paddedWidth; x++)
                        {
                            uint8_t *ptr = tempBuffer + y * paddedWidth + x;
                            switch (border_type)
                            {
                            case BORDER_REPLICATE:
                                *ptr = tempBuffer[(bottom_start - 1) * paddedWidth + x];
                                break;
                            case BORDER_REFLECT:
                            {
                                int offset = (y - (bottom_start - 1)) % height;
                                *ptr = tempBuffer[(bottom_start - 1 - offset) * paddedWidth + x];
                                break;
                            }
                            case BORDER_REFLECT_101:
                            {
                                int relativeY = y - startY;
                                int period = 2 * (height - 1);
                                if (period <= 0)
                                {
                                    relativeY = 0;
                                }
                                else
                                {
                                    relativeY = relativeY % period;
                                    if (relativeY < 0)
                                        relativeY += period;
                                    if (relativeY >= height)
                                        relativeY = 2 * (height - 1) - relativeY;
                                }
                                int originalY = startY + relativeY;
                                *ptr = tempBuffer[originalY * paddedWidth + x];
                                break;
                            }
                            case BORDER_CONSTANT:
                                *ptr = borderValue;
                                break;
                            default:
                                delete[] tempBuffer;
                                return CV_HAL_ERROR_NOT_IMPLEMENTED;
                            }
                        }
                    }
                }

                return CV_HAL_ERROR_OK;
            }

            int scharr(const uint8_t *src_data, size_t src_step, uint8_t *dst_data, size_t dst_step, int width, int height, int src_depth, int dst_depth, int cn, int margin_left, int margin_top, int margin_right, int margin_bottom, int dx, int dy, double scale, double delta, int border_type)
            {
                if (src_depth != CV_8U || (dst_depth != CV_8U && dst_depth != CV_16S && dst_depth != CV_32F) || cn != 1 || width < 3)
                {
                    return CV_HAL_ERROR_NOT_IMPLEMENTED;
                }

                if (scale != 1 || delta != 0)
                {
                    return CV_HAL_ERROR_NOT_IMPLEMENTED;
                }

                if (border_type != BORDER_REPLICATE &&
                    border_type != BORDER_CONSTANT)
                {
                    return CV_HAL_ERROR_NOT_IMPLEMENTED;
                }

                if (margin_left != 0 || margin_right != 0 || margin_top != 0 || margin_bottom != 0)
                {
                    int ret = scharr_border(src_data, src_step,
                                            width, height,
                                            border_type,
                                            margin_left, margin_top,
                                            margin_right, margin_bottom);
                    if (ret != CV_HAL_ERROR_OK)
                        return ret;
                }

                if (dst_depth == CV_8U)
                {
                    int align_size = (width + 2 + 7) & -8;
                    constexpr size_t alignment = 64;
                    size_t bufferSize = (align_size << 1) + alignment;
                    uint8_t *_tempBuf = static_cast<uint8_t *>(
                        aligned_alloc(alignment, bufferSize));

                    uint8_t *trow0 = reinterpret_cast<uint8_t *>(
                        (reinterpret_cast<uintptr_t>(_tempBuf) + alignment - 1) & ~(alignment - 1));
                    uint8_t *trow1 = trow0 + align_size;

                    for (int y = 0; y < height; y++)
                    {
                        const uint8_t *srow1 = src_data + y * src_step;
                        const uint8_t *srow0, *srow2;
                        if (border_type == BORDER_REPLICATE)
                        {
                            srow0 = src_data + (y > 0 ? y - 1 : 0) * src_step;
                            srow2 = src_data + (y < height - 1 ? y + 1 : height - 1) * src_step;
                        }
                        else if (border_type == BORDER_CONSTANT)
                        {
                            if (y > 0)
                                srow0 = src_data + (y - 1) * src_step;
                            else
                                srow0 = nullptr;

                            if (y < height - 1)
                                srow2 = src_data + (y + 1) * src_step;
                            else
                                srow2 = nullptr;
                        }

                        uint8_t *drow0 = dst_data + (y > 0 ? y - 1 : 0) * dst_step;
                        uint8_t *drow1 = dst_data + y * dst_step;

                        uint8_t *trow = (y % 2) ? trow1 : trow0;

                        size_t vl = __riscv_vsetvl_e8m4(width - 1);
                        int16_t prevx = 0, rowx = 0, nextx = 0, res = 0;
                        if (border_type == BORDER_REPLICATE)
                        {
                            if (dy == 0)
                            {
                                prevx = 3 * srow2[0] + 10 * srow1[0] + 3 * srow0[0];
                                nextx = 3 * srow2[1] + 10 * srow1[1] + 3 * srow0[1];
                            }
                            else if (dy == 2)
                            {
                                prevx = 3 * srow2[0] - 10 * srow1[0] + 3 * srow0[0];
                                nextx = 3 * srow2[1] - 10 * srow1[1] + 3 * srow0[1];
                            }
                            else
                            {
                                prevx = 3 * srow2[0] - 3 * srow0[0];
                                nextx = 3 * srow2[1] - 3 * srow0[1];
                            }
                            rowx = prevx;
                        }

                        else if (border_type == BORDER_CONSTANT)
                        {
                            if (dy == 0)
                            {
                                prevx = 3 * (srow2 ? srow2[0] : borderValue) +
                                        10 * srow1[0] +
                                        3 * (srow0 ? srow0[0] : borderValue);
                                nextx = 3 * (srow2 ? srow2[1] : borderValue) +
                                        10 * srow1[1] +
                                        3 * (srow0 ? srow0[1] : borderValue);
                            }
                            else if (dy == 2)
                            {
                                prevx = 3 * (srow2 ? srow2[0] : borderValue) -
                                        10 * srow1[0] +
                                        3 * (srow0 ? srow0[0] : borderValue);
                                nextx = 3 * (srow2 ? srow2[1] : borderValue) -
                                        10 * srow1[1] +
                                        3 * (srow0 ? srow0[1] : borderValue);
                            }
                            else
                            {
                                prevx = 3 * (srow2 ? srow2[0] : borderValue) -
                                        3 * (srow0 ? srow0[0] : borderValue);
                                nextx = 3 * (srow2 ? srow2[1] : borderValue) -
                                        3 * (srow0 ? srow0[1] : borderValue);
                            }
                        }

                        if (dx == 1)
                        {
                            res = 3 * nextx - 3 * prevx;
                        }
                        else if (dx == 0)
                        {
                            res = 3 * prevx + 10 * rowx + 3 * nextx;
                        }
                        else
                        {
                            res = 3 * prevx - 10 * rowx + 3 * nextx;
                        }

                        if (res < 0)
                            res = 0;
                        if (res > 255)
                            res = 255;

                        vint16m8_t vrowx, vprevx, vnextx;
                        vrowx = __riscv_vmv_v_x_i16m8(rowx, vl);
                        vprevx = vrowx;

                        for (int x = 1; x < width - 1; x += vl)
                        {
                            vl = __riscv_vsetvl_e8m4(width - x - 1);
                            vint16m8_t vsrow0, vsrow1, vsrow2;
                            if (border_type == BORDER_REPLICATE)
                            {
                                vsrow0 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow0 + x + 1, vl), vl));
                                vsrow1 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow1 + x + 1, vl), vl));
                                vsrow2 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow2 + x + 1, vl), vl));
                            }

                            else if (border_type == BORDER_CONSTANT)
                            {
                                vsrow0 = srow0 ? __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow0 + x, vl), vl)) : __riscv_vmv_v_x_i16m8(borderValue, vl);
                                vsrow1 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow1 + x, vl), vl));
                                vsrow2 = srow2 ? __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow2 + x, vl), vl)) : __riscv_vmv_v_x_i16m8(borderValue, vl);
                            }

                            if (dy == 0)
                            {
                                vint16m8_t vsrow2_vsrow0 = __riscv_vmul_vx_i16m8(__riscv_vadd_vv_i16m8(vsrow2, vsrow0, vl), 3, vl);
                                vint16m8_t vsrow1_2x = __riscv_vmul_vx_i16m8(vsrow1, 10, vl);
                                vnextx = __riscv_vadd_vv_i16m8(vsrow2_vsrow0, vsrow1_2x, vl);
                            }
                            else if (dy == 2)
                            {
                                vint16m8_t vsrow2_vsrow0 = __riscv_vmul_vx_i16m8(__riscv_vadd_vv_i16m8(vsrow2, vsrow0, vl), 3, vl);
                                vint16m8_t vsrow1_2x = __riscv_vmul_vx_i16m8(vsrow1, 10, vl);
                                vnextx = __riscv_vsub_vv_i16m8(vsrow2_vsrow0, vsrow1_2x, vl);
                            }
                            else
                            {
                                vnextx = __riscv_vmul_vx_i16m8(__riscv_vsub_vv_i16m8(vsrow2, vsrow0, vl), 3, vl);
                            }

                            vint16m8_t grad;
                            if (dx == 1)
                            {
                                grad = __riscv_vmul_vx_i16m8(__riscv_vsub_vv_i16m8(vnextx, vprevx, vl), 3, vl);
                            }
                            else if (dx == 0)
                            {
                                grad = __riscv_vadd_vv_i16m8(__riscv_vmul_vx_i16m8(__riscv_vadd_vv_i16m8(vprevx, vnextx, vl), 3, vl), __riscv_vmul_vx_i16m8(vrowx, 3, vl), vl);
                            }
                            else
                            {
                                grad = __riscv_vsub_vv_i16m8(__riscv_vmul_vx_i16m8(__riscv_vadd_vv_i16m8(vprevx, vnextx, vl), 3, vl), __riscv_vmul_vx_i16m8(vrowx, 3, vl), vl);
                            }

                            vrowx = __riscv_vslidedown_vx_i16m8(vrowx, vl - 1, vl);
                            vprevx = __riscv_vslidedown_vx_i16m8(vprevx, vl - 1, vl);

                            vprevx = vrowx;
                            vrowx = vnextx;

                            vrowx = __riscv_vslidedown_vx_i16m8(vrowx, vl - 1, vl);
                            vprevx = __riscv_vslidedown_vx_i16m8(vprevx, vl - 1, vl);

                            vuint8m4_t vres = __riscv_vncvt_x_x_w_u8m4(__riscv_vreinterpret_v_i16m8_u16m8(__riscv_vmax_vx_i16m8(__riscv_vmin_vx_i16m8(grad, 255, vl), 0, vl)), vl);
                            __riscv_vse8_v_u8m4(trow + x, vres, vl);
                        }

                        if (border_type == BORDER_REPLICATE)
                        {
                            if (dy == 0)
                            {
                                prevx = 3 * srow2[width - 2] + 10 * srow1[width - 2] + 3 * srow0[width - 2];
                                rowx = 3 * srow2[width - 1] + 10 * srow1[width - 1] + 3 * srow0[width - 1];
                            }
                            else if (dy == 2)
                            {
                                prevx = 3 * srow2[width - 2] - 10 * srow1[width - 2] + 3 * srow0[width - 2];
                                rowx = 3 * srow2[width - 1] - 10 * srow1[width - 1] + 3 * srow0[width - 1];
                            }
                            else
                            {
                                prevx = 3 * srow2[width - 2] - 3 * srow0[width - 2];
                                rowx = 3 * srow2[width - 1] - 3 * srow0[width - 1];
                            }
                            nextx = rowx;
                        }
                        else if (border_type == BORDER_CONSTANT)
                        {

                            if (dy == 0)
                            {
                                uint8_t s0 = srow0 ? srow0[width - 1] : borderValue;
                                uint8_t s1 = srow1[width - 1];
                                uint8_t s2 = srow2 ? srow2[width - 1] : borderValue;
                                rowx = 3 * s2 + 10 * s1 + 3 * s0;
                            }
                            else if (dy == 2)
                            {
                                uint8_t s0 = srow0 ? srow0[width - 1] : borderValue;
                                uint8_t s1 = srow1[width - 1];
                                uint8_t s2 = srow2 ? srow2[width - 1] : borderValue;
                                rowx = 3 * s2 - 10 * s1 + 3 * s0;
                            }
                            else
                            {
                                uint8_t s0 = srow0 ? srow0[width - 1] : borderValue;
                                uint8_t s2 = srow2 ? srow2[width - 1] : borderValue;
                                rowx = 3 * s2 - 3 * s0;
                            }
                            nextx = rowx;
                        }

                        if (dx == 1)
                        {
                            res = 3 * nextx - 3 * prevx;
                        }
                        else if (dx == 0)
                        {
                            res = 3 * prevx + 10 * rowx + 3 * nextx;
                        }
                        else
                        {
                            res = 3 * prevx - 10 * rowx + 3 * nextx;
                        }

                        if (res < 0)
                            res = 0;
                        if (res > 255)
                            res = 255;
                        trow[width - 1] = (uint8_t)res;

                        if (y > 0)
                        {
                            uint8_t *trow_res = (y % 2) ? trow0 : trow1;
                            for (int x = 0; x < width; x += vl)
                            {
                                vl = __riscv_vsetvl_e8m8(width - x);
                                vuint8m8_t vdata = __riscv_vle8_v_u8m8(trow_res + x, vl);
                                __riscv_vse8_v_u8m8(drow0 + x, vdata, vl);
                            }
                        }

                        if (y == height - 1)
                        {
                            uint8_t *trow_res = (!(y % 2)) ? trow0 : trow1;
                            for (int x = 0; x < width; x += vl)
                            {
                                vl = __riscv_vsetvl_e8m8(width - x);
                                vuint8m8_t vdata = __riscv_vle8_v_u8m8(trow_res + x, vl);
                                __riscv_vse8_v_u8m8(drow1 + x, vdata, vl);
                            }
                        }
                    }

                    free(_tempBuf);
                }

                else if (dst_depth == CV_16S)
                {
                    int align_size = (width + 2 + 7) & -8;
                    constexpr size_t alignment = 64;
                    size_t bufferSize = (align_size << 1) + alignment;
                    int16_t *_tempBuf = static_cast<int16_t *>(
                        aligned_alloc(alignment, bufferSize * sizeof(int16_t)));

                    int16_t *trow0 = reinterpret_cast<int16_t *>(
                        (reinterpret_cast<uintptr_t>(_tempBuf) + alignment - 1) & ~(alignment - 1));
                    int16_t *trow1 = trow0 + align_size;

                    for (int y = 0; y < height; y++)
                    {
                        const uint8_t *srow1 = src_data + y * src_step;
                        const uint8_t *srow0, *srow2;
                        if (border_type == BORDER_REPLICATE)
                        {
                            srow0 = src_data + (y > 0 ? y - 1 : 0) * src_step;
                            srow2 = src_data + (y < height - 1 ? y + 1 : height - 1) * src_step;
                        }
                        else if (border_type == BORDER_CONSTANT)
                        {
                            if (y > 0)
                                srow0 = src_data + (y - 1) * src_step;
                            else
                                srow0 = nullptr;

                            if (y < height - 1)
                                srow2 = src_data + (y + 1) * src_step;
                            else
                                srow2 = nullptr;
                        }

                        int16_t *drow0 = (int16_t *)(dst_data + (y > 0 ? y - 1 : 0) * dst_step);
                        int16_t *drow1 = (int16_t *)(dst_data + y * dst_step);

                        int16_t *trow = (y % 2) ? trow1 : trow0;

                        size_t vl = __riscv_vsetvl_e8m4(width - 1);

                        int x = 0;
                        int16_t prevx = 0, rowx = 0, nextx = 0, res = 0;
                        if (border_type == BORDER_REPLICATE)
                        {
                            if (dy == 0)
                            {
                                prevx = 3 * srow2[0] + 10 * srow1[0] + 3 * srow0[0];
                                nextx = 3 * srow2[1] + 10 * srow1[1] + 3 * srow0[1];
                            }
                            else if (dy == 2)
                            {
                                prevx = 3 * srow2[0] - 10 * srow1[0] + 3 * srow0[0];
                                nextx = 3 * srow2[1] - 10 * srow1[1] + 3 * srow0[1];
                            }
                            else
                            {
                                prevx = 3 * srow2[0] - 3 * srow0[0];
                                nextx = 3 * srow2[1] - 3 * srow0[1];
                            }
                            rowx = prevx;
                        }

                        else if (border_type == BORDER_CONSTANT)
                        {
                            if (dy == 0)
                            {
                                prevx = 3 * (srow2 ? srow2[0] : borderValue) +
                                        10 * srow1[0] +
                                        3 * (srow0 ? srow0[0] : borderValue);
                                nextx = 3 * (srow2 ? srow2[1] : borderValue) +
                                        10 * srow1[1] +
                                        3 * (srow0 ? srow0[1] : borderValue);
                            }
                            else if (dy == 2)
                            {
                                prevx = 3 * (srow2 ? srow2[0] : borderValue) -
                                        10 * srow1[0] +
                                        3 * (srow0 ? srow0[0] : borderValue);
                                nextx = 3 * (srow2 ? srow2[1] : borderValue) -
                                        10 * srow1[1] +
                                        3 * (srow0 ? srow0[1] : borderValue);
                            }
                            else
                            {
                                prevx = 3 * (srow2 ? srow2[0] : borderValue) -
                                        3 * (srow0 ? srow0[0] : borderValue);
                                nextx = 3 * (srow2 ? srow2[1] : borderValue) -
                                        3 * (srow0 ? srow0[1] : borderValue);
                            }
                        }

                        if (dx == 1)
                        {
                            res = 3 * nextx - 3 * prevx;
                        }
                        else if (dx == 0)
                        {
                            res = 3 * prevx + 10 * rowx + 3 * nextx;
                        }
                        else
                        {
                            res = 3 * prevx - 10 * rowx + 3 * nextx;
                        }

                        vint16m8_t vrowx, vprevx, vnextx;
                        vrowx = __riscv_vmv_v_x_i16m8(rowx, vl);
                        vprevx = vrowx;

                        for (x = 1; x < width - 1; x += vl)
                        {
                            vl = __riscv_vsetvl_e8m4(width - x - 1);
                            vint16m8_t vsrow0, vsrow1, vsrow2;
                            if (border_type == BORDER_REPLICATE)
                            {
                                vsrow0 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow0 + x + 1, vl), vl));
                                vsrow1 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow1 + x + 1, vl), vl));
                                vsrow2 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow2 + x + 1, vl), vl));
                            }

                            else if (border_type == BORDER_CONSTANT)
                            {
                                vsrow0 = srow0 ? __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow0 + x, vl), vl)) : __riscv_vmv_v_x_i16m8(borderValue, vl);
                                vsrow1 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow1 + x, vl), vl));
                                vsrow2 = srow2 ? __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow2 + x, vl), vl)) : __riscv_vmv_v_x_i16m8(borderValue, vl);
                            }

                            if (dy == 0)
                            {
                                vint16m8_t vsrow2_vsrow0 = __riscv_vmul_vx_i16m8(__riscv_vadd_vv_i16m8(vsrow2, vsrow0, vl), 3, vl);
                                vint16m8_t vsrow1_2x = __riscv_vmul_vx_i16m8(vsrow1, 10, vl);
                                vnextx = __riscv_vadd_vv_i16m8(vsrow2_vsrow0, vsrow1_2x, vl);
                            }
                            else if (dy == 2)
                            {
                                vint16m8_t vsrow2_vsrow0 = __riscv_vmul_vx_i16m8(__riscv_vadd_vv_i16m8(vsrow2, vsrow0, vl), 3, vl);
                                vint16m8_t vsrow1_2x = __riscv_vmul_vx_i16m8(vsrow1, 10, vl);
                                vnextx = __riscv_vsub_vv_i16m8(vsrow2_vsrow0, vsrow1_2x, vl);
                            }
                            else
                            {
                                vnextx = __riscv_vmul_vx_i16m8(__riscv_vsub_vv_i16m8(vsrow2, vsrow0, vl), 3, vl);
                            }

                            vrowx = __riscv_vslideup_vx_i16m8(vrowx, vnextx, 1, vl);
                            vprevx = __riscv_vslideup_vx_i16m8(vprevx, vrowx, 1, vl);

                            vint16m8_t grad;
                            if (dx == 1)
                            {
                                grad = __riscv_vsub_vv_i16m8(vnextx, vprevx, vl);
                            }
                            else if (dx == 0)
                            {
                                grad = __riscv_vadd_vv_i16m8(__riscv_vmul_vx_i16m8(__riscv_vadd_vv_i16m8(vprevx, vnextx, vl), 3, vl), __riscv_vmul_vx_i16m8(vrowx, 10, vl), vl);
                            }
                            else
                            {
                                grad = __riscv_vsub_vv_i16m8(__riscv_vmul_vx_i16m8(__riscv_vadd_vv_i16m8(vprevx, vnextx, vl), 3, vl), __riscv_vmul_vx_i16m8(vrowx, 10, vl), vl);
                            }

                            vprevx = vrowx;
                            vrowx = vnextx;

                            vrowx = __riscv_vslidedown_vx_i16m8(vrowx, vl - 1, vl);
                            vprevx = __riscv_vslidedown_vx_i16m8(vprevx, vl - 1, vl);

                            __riscv_vse16_v_i16m8(trow + x, grad, vl);
                        }

                        if (border_type == BORDER_REPLICATE)
                        {
                            if (dy == 0)
                            {
                                prevx = 3 * srow2[width - 2] + 10 * srow1[width - 2] + 3 * srow0[width - 2];
                                rowx = 3 * srow2[width - 1] + 10 * srow1[width - 1] + 3 * srow0[width - 1];
                            }
                            else if (dy == 2)
                            {
                                prevx = 3 * srow2[width - 2] - 10 * srow1[width - 2] + 3 * srow0[width - 2];
                                rowx = 3 * srow2[width - 1] - 10 * srow1[width - 1] + 3 * srow0[width - 1];
                            }
                            else
                            {
                                prevx = 3 * srow2[width - 2] - 3 * srow0[width - 2];
                                rowx = 3 * srow2[width - 1] - 3 * srow0[width - 1];
                            }
                            nextx = rowx;
                        }

                        if (dx == 1)
                        {
                            res = 3 * nextx - 3 * prevx;
                        }
                        else if (dx == 0)
                        {
                            res = 3 * prevx + 10 * rowx + 3 * nextx;
                        }
                        else
                        {
                            res = 3 * prevx - 10 * rowx + 3 * nextx;
                        }

                        trow[width - 1] = res;

                        if (y > 0)
                        {
                            int16_t *trow_res = (y % 2) ? trow0 : trow1;
                            for (x = 0; x < width; x += vl)
                            {
                                vl = __riscv_vsetvl_e16m8(width - x);
                                vint16m8_t vdata = __riscv_vle16_v_i16m8(trow_res + x, vl);
                                __riscv_vse16_v_i16m8(drow0 + x, vdata, vl);
                            }
                        }

                        if (y == height - 1)
                        {
                            int16_t *trow_res = (!(y % 2)) ? trow0 : trow1;
                            for (x = 0; x < width; x += vl)
                            {
                                vl = __riscv_vsetvl_e16m8(width - x);
                                vint16m8_t vdata = __riscv_vle16_v_i16m8(trow_res + x, vl);
                                __riscv_vse16_v_i16m8(drow1 + x, vdata, vl);
                            }
                        }
                    }

                    free(_tempBuf);
                }

                else if (dst_depth == CV_32F)
                {
                    int align_size = (width + 2 + 7) & -8;
                    constexpr size_t alignment = 64;
                    size_t bufferSize = (align_size << 1) + alignment;
                    float *_tempBuf = static_cast<float *>(
                        aligned_alloc(alignment, bufferSize * sizeof(float)));

                    float *trow0 = reinterpret_cast<float *>(
                        (reinterpret_cast<uintptr_t>(_tempBuf) + alignment - 1) & ~(alignment - 1));
                    float *trow1 = trow0 + align_size;

                    for (int y = 0; y < height; y++)
                    {
                        const uint8_t *srow0, *srow1, *srow2;
                        if (border_type == BORDER_REPLICATE)
                        {
                            srow1 = src_data + y * src_step;
                            srow0 = src_data + (y > 0 ? y - 1 : 0) * src_step;
                            srow2 = src_data + (y < height - 1 ? y + 1 : height - 1) * src_step;
                        }
                        else if (border_type == BORDER_CONSTANT)
                        {
                            srow1 = src_data + y * src_step;
                            if (y > 0)
                                srow0 = src_data + (y - 1) * src_step;
                            else
                                srow0 = nullptr;

                            if (y < height - 1)
                                srow2 = src_data + (y + 1) * src_step;
                            else
                                srow2 = nullptr;
                        }

                        float *drow0 = (float *)(dst_data + (y > 0 ? y - 1 : 0) * dst_step);
                        float *drow1 = (float *)(dst_data + y * dst_step);

                        float *trow = (y % 2) ? trow1 : trow0;

                        size_t vl = __riscv_vsetvl_e8m4(width - 1);

                        float prevx = 0, rowx = 0, nextx = 0, res = 0;
                        if (border_type == BORDER_REPLICATE)
                        {
                            if (dy == 0)
                            {
                                prevx = 3 * srow2[0] + 10 * srow1[0] + 3 * srow0[0];
                                nextx = 3 * srow2[1] + 10 * srow1[1] + 3 * srow0[1];
                            }
                            else if (dy == 2)
                            {
                                prevx = 3 * srow2[0] - 10 * srow1[0] + 3 * srow0[0];
                                nextx = 3 * srow2[1] - 10 * srow1[1] + 3 * srow0[1];
                            }
                            else
                            {
                                prevx = 3 * srow2[0] - 3 * srow0[0];
                                nextx = 3 * srow2[1] - 3 * srow0[1];
                            }
                        }
                        else if (border_type == BORDER_CONSTANT)
                        {
                            if (dy == 0)
                            {
                                prevx = 3 * (srow2 ? srow2[0] : borderValue) +
                                        10 * srow1[0] +
                                        3 * (srow0 ? srow0[0] : borderValue);
                                nextx = 3 * (srow2 ? srow2[1] : borderValue) +
                                        10 * srow1[1] +
                                        3 * (srow0 ? srow0[1] : borderValue);
                            }
                            else if (dy == 2)
                            {
                                prevx = 3 * (srow2 ? srow2[0] : borderValue) -
                                        10 * srow1[0] +
                                        3 * (srow0 ? srow0[0] : borderValue);
                                nextx = 3 * (srow2 ? srow2[1] : borderValue) -
                                        10 * srow1[1] +
                                        3 * (srow0 ? srow0[1] : borderValue);
                            }
                            else
                            {
                                prevx = 3 * (srow2 ? srow2[0] : borderValue) -
                                        3 * (srow0 ? srow0[0] : borderValue);
                                nextx = 3 * (srow2 ? srow2[1] : borderValue) -
                                        3 * (srow0 ? srow0[1] : borderValue);
                            }
                        }
                        rowx = prevx;

                        if (dx == 1)
                        {
                            res = 3 * nextx - 3 * prevx;
                        }
                        else if (dx == 0)
                        {
                            res = 3 * prevx + 10 * rowx + 3 * nextx;
                        }
                        else
                        {
                            res = 3 * prevx - 10 * rowx + 3 * nextx;
                        }

                        trow[0] = res;

                        // Vector processing
                        vfloat32m8_t vrowx = __riscv_vfmv_v_f_f32m8(rowx, vl);
                        vfloat32m8_t vprevx = vrowx;
                        vfloat32m8_t vnextx;

                        for (int x = 1; x < width - 1; x += vl)
                        {

                            vl = __riscv_vsetvl_e8m2(width - x - 1);
                            vuint8m2_t vsrow0_u8, vsrow1_u8, vsrow2_u8;

                            // Convert uint8 to float32
                            if (border_type == BORDER_REPLICATE)
                            {
                                vsrow0_u8 = __riscv_vle8_v_u8m2(srow0 + x + 1, vl);
                                vsrow1_u8 = __riscv_vle8_v_u8m2(srow1 + x + 1, vl);
                                vsrow2_u8 = __riscv_vle8_v_u8m2(srow2 + x + 1, vl);
                            }
                            else if (border_type == BORDER_CONSTANT)
                            {
                                vsrow0_u8 = srow0 ? __riscv_vle8_v_u8m2(srow0 + x, vl) : __riscv_vmv_v_x_u8m2(borderValue, vl);
                                vsrow1_u8 = __riscv_vle8_v_u8m2(srow1 + x, vl);
                                vsrow2_u8 = srow2 ? __riscv_vle8_v_u8m2(srow2 + x, vl) : __riscv_vmv_v_x_u8m2(borderValue, vl);
                            }
                            vfloat32m8_t vsrow0 = __riscv_vfwcvt_f_xu_v_f32m8(__riscv_vwcvtu_x_x_v_u16m4(vsrow0_u8, vl), vl);
                            vfloat32m8_t vsrow1 = __riscv_vfwcvt_f_xu_v_f32m8(__riscv_vwcvtu_x_x_v_u16m4(vsrow1_u8, vl), vl);
                            vfloat32m8_t vsrow2 = __riscv_vfwcvt_f_xu_v_f32m8(__riscv_vwcvtu_x_x_v_u16m4(vsrow2_u8, vl), vl);

                            if (dy == 0)
                            {
                                vnextx = __riscv_vfmul_vf_f32m8(vsrow1, 10.0f, vl);
                                vnextx = __riscv_vfmacc_vf_f32m8(vnextx, 3.0f, vsrow0, vl);
                                vnextx = __riscv_vfmacc_vf_f32m8(vnextx, 3.0f, vsrow2, vl);
                            }
                            else if (dy == 2)
                            {
                                vnextx = __riscv_vfmul_vf_f32m8(vsrow1, -10.0f, vl);
                                vnextx = __riscv_vfmacc_vf_f32m8(vnextx, 3.0f, vsrow0, vl);
                                vnextx = __riscv_vfmacc_vf_f32m8(vnextx, 3.0f, vsrow2, vl);
                            }
                            else
                            {
                                vnextx = __riscv_vfmul_vf_f32m8(
                                    __riscv_vfsub_vv_f32m8(vsrow2, vsrow0, vl), 3.0f, vl);
                            }

                            vrowx = __riscv_vslideup_vx_f32m8(vrowx, vnextx, 1, vl);
                            vprevx = __riscv_vslideup_vx_f32m8(vprevx, vrowx, 1, vl);

                            vfloat32m8_t grad;
                            if (dx == 1)
                            {
                                grad = __riscv_vfsub_vv_f32m8(vnextx, vprevx, vl);
                            }
                            else if (dx == 0)
                            {
                                grad = __riscv_vfmul_vf_f32m8(vrowx, 10.0f, vl);
                                grad = __riscv_vfmacc_vf_f32m8(grad, 3.0f, vprevx, vl);
                                grad = __riscv_vfmacc_vf_f32m8(grad, 3.0f, vnextx, vl);
                            }
                            else
                            {
                                grad = __riscv_vfmul_vf_f32m8(vrowx, -10.0f, vl);
                                grad = __riscv_vfmacc_vf_f32m8(grad, 3.0f, vprevx, vl);
                                grad = __riscv_vfmacc_vf_f32m8(grad, 3.0f, vnextx, vl);
                            }

                            __riscv_vse32_v_f32m8(trow + x, grad, vl);

                            vprevx = vrowx;
                            vrowx = vnextx;

                            vrowx = __riscv_vslidedown_vx_f32m8(vrowx, vl - 1, vl);
                            vprevx = __riscv_vslidedown_vx_f32m8(vprevx, vl - 1, vl);
                        }

                        // Last pixel
                        if (border_type == BORDER_REPLICATE)
                        {
                            if (dy == 0)
                            {
                                prevx = 3 * srow2[width - 2] + 10 * srow1[width - 2] + 3 * srow0[width - 2];
                                rowx = 3 * srow2[width - 1] + 10 * srow1[width - 1] + 3 * srow0[width - 1];
                            }
                            else if (dy == 2)
                            {
                                prevx = 3 * srow2[width - 2] - 10 * srow1[width - 2] + 3 * srow0[width - 2];
                                rowx = 3 * srow2[width - 1] - 10 * srow1[width - 1] + 3 * srow0[width - 1];
                            }
                            else
                            {
                                prevx = 3 * srow2[width - 2] - 3 * srow0[width - 2];
                                rowx = 3 * srow2[width - 1] - 3 * srow0[width - 1];
                            }
                        }
                        else if (border_type == BORDER_CONSTANT)
                        {
                            if (dy == 0)
                            {
                                float s0 = srow0 ? srow0[width - 1] : borderValue;
                                float s1 = srow1[width - 1];
                                float s2 = srow2 ? srow2[width - 1] : borderValue;
                                rowx = 3 * s2 + 10 * s1 + 3 * s0;
                            }
                            else if (dy == 2)
                            {
                                float s0 = srow0 ? srow0[width - 1] : borderValue;
                                float s1 = srow1[width - 1];
                                float s2 = srow2 ? srow2[width - 1] : borderValue;
                                rowx = 3 * s2 - 10 * s1 + 3 * s0;
                            }
                            else
                            {
                                float s0 = srow0 ? srow0[width - 1] : borderValue;
                                float s2 = srow2 ? srow2[width - 1] : borderValue;
                                rowx = 3 * s2 - 3 * s0;
                            }
                        }
                        nextx = rowx;

                        if (dx == 1)
                        {
                            res = 3 * nextx - 3 * prevx;
                        }
                        else if (dx == 0)
                        {
                            res = 3 * prevx + 10 * rowx + 3 * nextx;
                        }
                        else
                        {
                            res = 3 * prevx - 10 * rowx + 3 * nextx;
                        }

                        trow[width - 1] = res;

                        if (y > 0)
                        {
                            float *trow_res = (y % 2) ? trow0 : trow1;
                            for (int x = 0; x < width; x += vl)
                            {
                                vl = __riscv_vsetvl_e32m8(width - x);
                                vfloat32m8_t vdata = __riscv_vle32_v_f32m8(trow_res + x, vl);
                                __riscv_vse32_v_f32m8(drow0 + x, vdata, vl);
                            }
                        }

                        if (y == height - 1)
                        {
                            float *trow_res = (!(y % 2)) ? trow0 : trow1;
                            for (int x = 0; x < width; x += vl)
                            {
                                vl = __riscv_vsetvl_e32m8(width - x);
                                vfloat32m8_t vdata = __riscv_vle32_v_f32m8(trow_res + x, vl);
                                __riscv_vse32_v_f32m8(drow1 + x, vdata, vl);
                            }
                        }
                    }

                    free(_tempBuf);
                }

                return CV_HAL_ERROR_OK;
            }

#endif // CV_HAL_RVV_1P0_ENABLED

        }
    }
} // cv::rvv_hal::imgproc
