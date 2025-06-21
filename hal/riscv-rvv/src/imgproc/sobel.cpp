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

            static int sobel_border(const uint8_t *src_data, size_t src_step,
                                    int width, int height,
                                    int ksize,
                                    int border_type,
                                    int margin_left, int margin_top,
                                    int margin_right, int margin_bottom)
            {
                if (ksize % 2 == 0)
                    return CV_HAL_ERROR_NOT_IMPLEMENTED;

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

            int sobel(const uint8_t *src_data, size_t src_step, uint8_t *dst_data, size_t dst_step, int width, int height, int src_depth, int dst_depth, int cn, int margin_left, int margin_top, int margin_right, int margin_bottom, int dx, int dy, int ksize, double scale, double delta, int border_type)
            {
                if (src_depth != CV_8U || (dst_depth != CV_8U && dst_depth != CV_16S && dst_depth != CV_32F) || cn != 1)
                {
                    return CV_HAL_ERROR_NOT_IMPLEMENTED;
                }

                if (width <= ksize)
                {
                    return CV_HAL_ERROR_NOT_IMPLEMENTED;
                }

                // TODO: Add support for different matters
                if (scale != 1 || delta != 0)
                {
                    return CV_HAL_ERROR_NOT_IMPLEMENTED;
                }

                if (margin_left != 0 || margin_right != 0 || margin_top != 0 || margin_bottom != 0)
                {
                    return CV_HAL_ERROR_NOT_IMPLEMENTED;
                    int ret = sobel_border(src_data, src_step, width, height, ksize, border_type, margin_left, margin_top, margin_right, margin_bottom);
                    if (ret != CV_HAL_ERROR_OK)
                    {
                        return ret;
                    }
                }

                if (dst_depth == CV_8U)
                {
                    if (ksize == 3)
                    {
                        if (border_type == BORDER_REPLICATE)
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
                                const uint8_t *srow0 = src_data + (y > 0 ? y - 1 : 0) * src_step;
                                const uint8_t *srow2 = src_data + (y < height - 1 ? y + 1 : height - 1) * src_step;

                                uint8_t *drow0 = dst_data + (y > 0 ? y - 1 : 0) * dst_step;
                                uint8_t *drow1 = dst_data + y * dst_step;

                                uint8_t *trow = (y % 2) ? trow1 : trow0;

                                size_t vl = __riscv_vsetvl_e8m4(width - 1);

                                int x = 0;
                                int16_t prevx = 0, rowx = 0, nextx = 0, res = 0;
                                if (dy == 0)
                                {
                                    prevx = srow2[0] + 2 * srow1[0] + srow0[0];
                                    nextx = srow2[1] + 2 * srow1[1] + srow0[1];
                                }
                                else if (dy == 2)
                                {
                                    prevx = srow2[0] - 2 * srow1[0] + srow0[0];
                                    nextx = srow2[1] - 2 * srow1[1] + srow0[1];
                                }
                                else
                                {
                                    prevx = srow2[0] - srow0[0];
                                    nextx = srow2[1] - srow0[1];
                                }
                                rowx = prevx;
                                if (dx == 1)
                                {
                                    res = nextx - prevx;
                                }
                                else if (dx == 0)
                                {
                                    res = prevx + 2 * rowx + nextx;
                                }
                                else
                                {
                                    res = prevx - 2 * rowx + nextx;
                                }

                                if (res < 0)
                                    res = 0;
                                if (res > 255)
                                    res = 255;

                                vint16m8_t vrowx, vprevx, vnextx;
                                vrowx = __riscv_vmv_v_x_i16m8(rowx, vl);
                                vprevx = vrowx;

                                for (x = 0; x < width - 1; x += vl)
                                {
                                    vl = __riscv_vsetvl_e8m4(width - x - 1);

                                    vint16m8_t vsrow0 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow0 + x + 1, vl), vl));
                                    vint16m8_t vsrow1 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow1 + x + 1, vl), vl));
                                    vint16m8_t vsrow2 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow2 + x + 1, vl), vl));

                                    if (dy == 0)
                                    {
                                        vint16m8_t vsrow2_vsrow0 = __riscv_vadd_vv_i16m8(vsrow2, vsrow0, vl);
                                        vint16m8_t vsrow1_2x = __riscv_vadd_vv_i16m8(vsrow1, vsrow1, vl);
                                        vnextx = __riscv_vadd_vv_i16m8(vsrow2_vsrow0, vsrow1_2x, vl);

                                        vrowx = __riscv_vslideup_vx_i16m8(vrowx, vnextx, 1, vl);
                                        vprevx = __riscv_vslideup_vx_i16m8(vprevx, vrowx, 1, vl);
                                    }
                                    else if (dy == 2)
                                    {
                                        vint16m8_t vsrow2_vsrow0 = __riscv_vadd_vv_i16m8(vsrow2, vsrow0, vl);
                                        vint16m8_t vsrow1_2x = __riscv_vadd_vv_i16m8(vsrow1, vsrow1, vl);
                                        vnextx = __riscv_vsub_vv_i16m8(vsrow2_vsrow0, vsrow1_2x, vl);

                                        vrowx = __riscv_vslideup_vx_i16m8(vrowx, vnextx, 1, vl);
                                        vprevx = __riscv_vslideup_vx_i16m8(vprevx, vrowx, 1, vl);
                                    }
                                    else
                                    {
                                        vnextx = __riscv_vsub_vv_i16m8(vsrow2, vsrow0, vl);

                                        vrowx = __riscv_vslideup_vx_i16m8(vrowx, vnextx, 1, vl);
                                        vprevx = __riscv_vslideup_vx_i16m8(vprevx, vrowx, 1, vl);
                                    }

                                    vint16m8_t grad;
                                    if (dx == 1)
                                    {
                                        grad = __riscv_vsub_vv_i16m8(vnextx, vprevx, vl);
                                    }
                                    else if (dx == 0)
                                    {
                                        grad = __riscv_vadd_vv_i16m8(__riscv_vadd_vv_i16m8(vprevx, vnextx, vl), __riscv_vadd_vv_i16m8(vrowx, vrowx, vl), vl);
                                    }
                                    else
                                    {
                                        grad = __riscv_vsub_vv_i16m8(__riscv_vadd_vv_i16m8(vprevx, vnextx, vl), __riscv_vadd_vv_i16m8(vrowx, vrowx, vl), vl);
                                    }

                                    vprevx = vrowx;
                                    vrowx = vnextx;

                                    vrowx = __riscv_vslidedown_vx_i16m8(vrowx, vl - 1, vl);
                                    vprevx = __riscv_vslidedown_vx_i16m8(vprevx, vl - 1, vl);

                                    vuint8m4_t vres = __riscv_vncvt_x_x_w_u8m4(__riscv_vreinterpret_v_i16m8_u16m8(__riscv_vmax_vx_i16m8(__riscv_vmin_vx_i16m8(grad, 255, vl), 0, vl)), vl);
                                    __riscv_vse8_v_u8m4(trow + x, vres, vl);
                                }

                                if (dy == 0)
                                {
                                    prevx = srow2[width - 2] + 2 * srow1[width - 2] + srow0[width - 2];
                                    rowx = srow2[width - 1] + 2 * srow1[width - 1] + srow0[width - 1];
                                }
                                else if (dy == 2)
                                {
                                    prevx = srow2[width - 2] - 2 * srow1[width - 2] + srow0[width - 2];
                                    rowx = srow2[width - 1] - 2 * srow1[width - 1] + srow0[width - 1];
                                }
                                else
                                {
                                    prevx = srow2[width - 2] - srow0[width - 2];
                                    rowx = srow2[width - 1] - srow0[width - 1];
                                }
                                nextx = rowx;
                                if (dx == 1)
                                {
                                    res = nextx - prevx;
                                }
                                else if (dx == 0)
                                {
                                    res = prevx + 2 * rowx + nextx;
                                }
                                else
                                {
                                    res = prevx - 2 * rowx + nextx;
                                }

                                if (res < 0)
                                    res = 0;
                                if (res > 255)
                                    res = 255;
                                trow[width - 1] = (uint8_t)res;

                                if (y > 0)
                                {
                                    uint8_t *trow_res = (y % 2) ? trow0 : trow1;
                                    for (x = 0; x < width; x += vl)
                                    {
                                        vl = __riscv_vsetvl_e8m8(width - x);
                                        vuint8m8_t vdata = __riscv_vle8_v_u8m8(trow_res + x, vl);
                                        __riscv_vse8_v_u8m8(drow0 + x, vdata, vl);
                                    }
                                }

                                if (y == height - 1)
                                {
                                    uint8_t *trow_res = (!(y % 2)) ? trow0 : trow1;
                                    for (x = 0; x < width; x += vl)
                                    {
                                        vl = __riscv_vsetvl_e8m8(width - x);
                                        vuint8m8_t vdata = __riscv_vle8_v_u8m8(trow_res + x, vl);
                                        __riscv_vse8_v_u8m8(drow1 + x, vdata, vl);
                                    }
                                }
                            }

                            free(_tempBuf);
                        }

                        else if (border_type == BORDER_CONSTANT)
                        {
                            int align_size = (width + 2 + 7) & -8;
                            constexpr size_t alignment = 64;
                            size_t bufferSize = (align_size << 1) + alignment;
                            uint8_t *_tempBuf = static_cast<uint8_t *>(
                                aligned_alloc(alignment, bufferSize));

                            uint8_t *trow0 = reinterpret_cast<uint8_t *>(
                                (reinterpret_cast<uintptr_t>(_tempBuf) + alignment - 1) & ~(alignment - 1));
                            uint8_t *trow1 = trow0 + align_size;

                            vint16m8_t vborder = __riscv_vmv_v_x_i16m8(borderValue, __riscv_vsetvlmax_e16m8());

                            for (int y = 0; y < height; y++)
                            {
                                const uint8_t *srow1 = src_data + y * src_step;
                                const uint8_t *srow0 = nullptr;
                                const uint8_t *srow2 = nullptr;

                                if (y > 0)
                                    srow0 = src_data + (y - 1) * src_step;
                                if (y < height - 1)
                                    srow2 = src_data + (y + 1) * src_step;

                                uint8_t *drow0 = dst_data + (y > 0 ? y - 1 : 0) * dst_step;
                                uint8_t *drow1 = dst_data + y * dst_step;
                                uint8_t *trow = (y % 2) ? trow1 : trow0;

                                size_t vl = __riscv_vsetvl_e8m4(width - 1);
                                int x = 0;
                                int16_t prevx = 0, rowx = 0, nextx = 0, res = 0;

                                if (dy == 0)
                                {
                                    int16_t s0 = srow0 ? srow0[0] : borderValue;
                                    int16_t s1 = srow1[0];
                                    int16_t s2 = srow2 ? srow2[0] : borderValue;
                                    prevx = s2 + 2 * s1 + s0;

                                    s0 = srow0 ? srow0[1] : borderValue;
                                    s1 = srow1[1];
                                    s2 = srow2 ? srow2[1] : borderValue;
                                    nextx = s2 + 2 * s1 + s0;
                                }
                                else if (dy == 2)
                                {
                                    int16_t s0 = srow0 ? srow0[0] : borderValue;
                                    int16_t s1 = srow1[0];
                                    int16_t s2 = srow2 ? srow2[0] : borderValue;
                                    prevx = s2 - 2 * s1 + s0;

                                    s0 = srow0 ? srow0[1] : borderValue;
                                    s1 = srow1[1];
                                    s2 = srow2 ? srow2[1] : borderValue;
                                    nextx = s2 - 2 * s1 + s0;
                                }
                                else
                                {
                                    int16_t s0 = srow0 ? srow0[0] : borderValue;
                                    int16_t s2 = srow2 ? srow2[0] : borderValue;
                                    prevx = s2 - s0;

                                    s0 = srow0 ? srow0[1] : borderValue;
                                    s2 = srow2 ? srow2[1] : borderValue;
                                    nextx = s2 - s0;
                                }

                                rowx = prevx;
                                if (dx == 1)
                                    res = nextx - prevx;
                                else if (dx == 0)
                                    res = prevx + 2 * rowx + nextx;
                                else
                                    res = prevx - 2 * rowx + nextx;

                                if (res < 0)
                                    res = 0;
                                if (res > 255)
                                    res = 255;

                                trow[0] = (uint8_t)res;

                                vint16m8_t vrowx = __riscv_vmv_v_x_i16m8(rowx, vl);
                                vint16m8_t vprevx = vrowx, vnextx;

                                for (x = 0; x < width - 1; x += vl)
                                {
                                    vl = __riscv_vsetvl_e8m4(width - x - 1);

                                    vint16m8_t vsrow1 = __riscv_vreinterpret_v_u16m8_i16m8(
                                        __riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow1 + x + 1, vl), vl));

                                    vint16m8_t vsrow0, vsrow2;
                                    if (srow0)
                                        vsrow0 = __riscv_vreinterpret_v_u16m8_i16m8(
                                            __riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow0 + x + 1, vl), vl));
                                    else
                                        vsrow0 = vborder;

                                    if (srow2)
                                        vsrow2 = __riscv_vreinterpret_v_u16m8_i16m8(
                                            __riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow2 + x + 1, vl), vl));
                                    else
                                        vsrow2 = vborder;

                                    if (dy == 0)
                                    {
                                        vint16m8_t vsrow2_vsrow0 = __riscv_vadd_vv_i16m8(vsrow2, vsrow0, vl);
                                        vint16m8_t vsrow1_2x = __riscv_vadd_vv_i16m8(vsrow1, vsrow1, vl);
                                        vnextx = __riscv_vadd_vv_i16m8(vsrow2_vsrow0, vsrow1_2x, vl);

                                        vrowx = __riscv_vslideup_vx_i16m8(vrowx, vnextx, 1, vl);
                                        vprevx = __riscv_vslideup_vx_i16m8(vprevx, vrowx, 1, vl);
                                    }
                                    else if (dy == 2)
                                    {
                                        vint16m8_t vsrow2_vsrow0 = __riscv_vadd_vv_i16m8(vsrow2, vsrow0, vl);
                                        vint16m8_t vsrow1_2x = __riscv_vadd_vv_i16m8(vsrow1, vsrow1, vl);
                                        vnextx = __riscv_vsub_vv_i16m8(vsrow2_vsrow0, vsrow1_2x, vl);

                                        vrowx = __riscv_vslideup_vx_i16m8(vrowx, vnextx, 1, vl);
                                        vprevx = __riscv_vslideup_vx_i16m8(vprevx, vrowx, 1, vl);
                                    }
                                    else
                                    {
                                        vnextx = __riscv_vsub_vv_i16m8(vsrow2, vsrow0, vl);

                                        vrowx = __riscv_vslideup_vx_i16m8(vrowx, vnextx, 1, vl);
                                        vprevx = __riscv_vslideup_vx_i16m8(vprevx, vrowx, 1, vl);
                                    }

                                    vint16m8_t grad;
                                    if (dx == 1)
                                        grad = __riscv_vsub_vv_i16m8(vnextx, vprevx, vl);
                                    else if (dx == 0)
                                        grad = __riscv_vadd_vv_i16m8(__riscv_vadd_vv_i16m8(vprevx, vnextx, vl),
                                                                     __riscv_vadd_vv_i16m8(vrowx, vrowx, vl), vl);
                                    else
                                        grad = __riscv_vsub_vv_i16m8(__riscv_vadd_vv_i16m8(vprevx, vnextx, vl),
                                                                     __riscv_vadd_vv_i16m8(vrowx, vrowx, vl), vl);

                                    vprevx = vrowx;
                                    vrowx = vnextx;

                                    vrowx = __riscv_vslidedown_vx_i16m8(vrowx, vl - 1, vl);
                                    vprevx = __riscv_vslidedown_vx_i16m8(vprevx, vl - 1, vl);

                                    vuint8m4_t vres = __riscv_vncvt_x_x_w_u8m4(
                                        __riscv_vreinterpret_v_i16m8_u16m8(
                                            __riscv_vmax_vx_i16m8(
                                                __riscv_vmin_vx_i16m8(grad, 255, vl), 0, vl)),
                                        vl);
                                    __riscv_vse8_v_u8m4(trow + x, vres, vl);
                                }

                                if (dy == 0)
                                {
                                    int16_t s0 = srow0 ? srow0[width - 1] : borderValue;
                                    int16_t s1 = srow1[width - 1];
                                    int16_t s2 = srow2 ? srow2[width - 1] : borderValue;
                                    rowx = s2 + 2 * s1 + s0;
                                }
                                else if (dy == 2)
                                {
                                    int16_t s0 = srow0 ? srow0[width - 1] : borderValue;
                                    int16_t s1 = srow1[width - 1];
                                    int16_t s2 = srow2 ? srow2[width - 1] : borderValue;
                                    rowx = s2 - 2 * s1 + s0;
                                }
                                else
                                {
                                    int16_t s0 = srow0 ? srow0[width - 1] : borderValue;
                                    int16_t s2 = srow2 ? srow2[width - 1] : borderValue;
                                    rowx = s2 - s0;
                                }
                                nextx = rowx;

                                if (dx == 1)
                                    res = nextx - prevx;
                                else if (dx == 0)
                                    res = prevx + 2 * rowx + nextx;
                                else
                                    res = prevx - 2 * rowx + nextx;

                                if (res < 0)
                                    res = 0;
                                if (res > 255)
                                    res = 255;
                                trow[width - 1] = (uint8_t)res;

                                if (y > 0)
                                {
                                    uint8_t *trow_res = (y % 2) ? trow0 : trow1;
                                    for (x = 0; x < width; x += vl)
                                    {
                                        vl = __riscv_vsetvl_e8m8(width - x);
                                        vuint8m8_t vdata = __riscv_vle8_v_u8m8(trow_res + x, vl);
                                        __riscv_vse8_v_u8m8(drow0 + x, vdata, vl);
                                    }
                                }

                                if (y == height - 1)
                                {
                                    uint8_t *trow_res = (!(y % 2)) ? trow0 : trow1;
                                    for (x = 0; x < width; x += vl)
                                    {
                                        vl = __riscv_vsetvl_e8m8(width - x);
                                        vuint8m8_t vdata = __riscv_vle8_v_u8m8(trow_res + x, vl);
                                        __riscv_vse8_v_u8m8(drow1 + x, vdata, vl);
                                    }
                                }
                            }

                            free(_tempBuf);
                        }

                        else
                        {
                            return CV_HAL_ERROR_NOT_IMPLEMENTED;
                        }
                    }

                    else if (ksize == 5)
                    {
                        if (border_type != BORDER_REPLICATE &&
                            border_type != BORDER_CONSTANT &&
                            border_type != BORDER_REFLECT &&
                            border_type != BORDER_REFLECT_101)
                        {
                            return CV_HAL_ERROR_NOT_IMPLEMENTED;
                        }
                        int align_size = (width + 4 + 7) & -8;
                        constexpr size_t alignment = 64;

                        size_t bufferBytes = align_size * 3 * sizeof(int16_t);
                        size_t bufferSize = (bufferBytes + alignment - 1) & ~(alignment - 1);
                        int16_t *_tempBuf = static_cast<int16_t *>(aligned_alloc(alignment, bufferSize));

                        int16_t *trow0 = _tempBuf;
                        int16_t *trow1 = trow0 + align_size;
                        int16_t *trow2 = trow1 + align_size;

                        const uint8_t *srowm2, *srowm1, *srow0, *srowp1, *srowp2;
                        uint8_t *border = (uint8_t *)malloc((4 * height) * sizeof(uint8_t));

                        for (int y = 0; y < height; y++)
                        {
                            if (border_type == BORDER_REPLICATE)
                            {
                                srowm2 = src_data + (y > 2 ? y - 2 : 0) * src_step;
                                srowm1 = src_data + (y > 1 ? y - 1 : 0) * src_step;
                                srow0 = src_data + y * src_step;
                                srowp1 = src_data + (y + 1 < height - 1 ? y + 1 : height - 1) * src_step;
                                srowp2 = src_data + (y + 2 < height - 1 ? y + 2 : height - 1) * src_step;
                            }
                            else if (border_type == BORDER_CONSTANT)
                            {
                                uint8_t *zero_row = (uint8_t *)malloc(width * sizeof(uint8_t));
                                memset(zero_row, 0, width * sizeof(uint8_t));
                                srowm2 = y > 2 ? (src_data + (y - 2) * src_step) : zero_row;
                                srowm1 = y > 1 ? (src_data + (y - 1) * src_step) : zero_row;
                                srow0 = src_data + y * src_step;
                                srowp1 = y + 1 < height - 1 ? (src_data + (y + 1) * src_step) : zero_row;
                                srowp2 = y + 2 < height - 1 ? (src_data + (y + 2) * src_step) : zero_row;
                            }
                            else if (border_type == BORDER_REFLECT)
                            {
                                srowm2 = y >= 2 ? (src_data + (y - 2) * src_step) : (y == 1 ? src_data : (src_data + src_step));
                                srowm1 = y >= 1 ? (src_data + (y - 1) * src_step) : src_data;
                                srow0 = src_data + y * src_step;
                                srowp1 = y + 1 <= height - 1 ? (src_data + (y + 1) * src_step) : src_data + y * src_step;
                                srowp2 = y + 2 <= height - 1 ? (src_data + (y + 2) * src_step) : (y == height - 2 ? (src_data + (y - 1) * src_step) : (src_data + y * src_step));
                            }
                            else if (border_type == BORDER_REFLECT_101)
                            {
                                srowm2 = y > 2 ? (src_data + (y - 2) * src_step) : (src_data + 2 * src_step);
                                srowm1 = y >= 1 ? (src_data + (y - 1) * src_step) : (src_data + src_step);
                                srow0 = src_data + y * src_step;
                                srowp1 = y + 1 <= height - 1 ? (src_data + (y + 1) * src_step) : (src_data + (y - 1) * src_step);
                                srowp2 = y + 2 <= height - 1 ? (src_data + (y + 2) * src_step) : (y == height - 2 ? (src_data + (y - 2) * src_step) : (src_data + y * src_step));
                            }

                            int16_t *trow;
                            if (y % 3 == 0)
                            {
                                trow = trow0;
                            }
                            else if (y % 3 == 1)
                            {
                                trow = trow1;
                            }
                            else
                            {
                                trow = trow2;
                            }

                            int16_t prevxm2 = 0, prevxm1 = 0, rowx = 0, nextxp1 = 0, nextxp2 = 0;

                            int x = 0;
                            if (dy == 0)
                            {
                                rowx = srowm2[0] + 4 * srowm1[0] + 6 * srow0[0] + 4 * srowp1[0] + srowp2[0];
                                nextxp1 = srowm2[1] + 4 * srowm1[1] + 6 * srow0[1] + 4 * srowp1[1] + srowp2[1];
                                nextxp2 = srowm2[2] + 4 * srowm1[2] + 6 * srow0[2] + 4 * srowp1[2] + srowp2[2];
                            }
                            else if (dy == 1)
                            {
                                rowx = -1 * srowm2[0] - 2 * srowm1[0] + 0 * srow0[0] + 2 * srowp1[0] + srowp2[0];
                                nextxp1 = -1 * srowm2[1] - 2 * srowm1[1] + 0 * srow0[1] + 2 * srowp1[1] + srowp2[1];
                                nextxp2 = -1 * srowm2[2] - 2 * srowm1[2] + 0 * srow0[2] + 2 * srowp1[2] + srowp2[2];
                            }
                            else
                            {
                                rowx = srowm2[0] + 0 * srowm1[0] - 2 * srow0[0] + 0 * srowp1[0] + srowp2[0];
                                nextxp1 = srowm2[1] + 0 * srowm1[1] - 2 * srow0[1] + 0 * srowp1[1] + srowp2[1];
                                nextxp2 = srowm2[2] + 0 * srowm1[2] - 2 * srow0[2] + 0 * srowp1[2] + srowp2[2];
                            }
                            if (border_type == BORDER_REPLICATE)
                            {
                                prevxm1 = rowx;
                                prevxm2 = prevxm1;
                            }
                            // border_type == BORDER_CONSTANT : prevxm1 = borderValue, prevxm2 = borderValue;
                            else if (border_type == BORDER_REFLECT)
                            {
                                prevxm1 = rowx;
                                prevxm2 = nextxp1;
                            }
                            else if (border_type == BORDER_REFLECT_101)
                            {
                                prevxm1 = nextxp1;
                                prevxm2 = nextxp2;
                            }

                            int16_t res = 0;
                            if (dx == 0)
                            {
                                res = prevxm2 + 4 * prevxm1 + 6 * rowx + 4 * nextxp1 + nextxp2;
                            }
                            else if (dx == 1)
                            {
                                res = -1 * prevxm2 - 2 * prevxm1 + 0 * rowx + 2 * nextxp1 + nextxp2;
                            }
                            else
                            {
                                res = prevxm2 + 0 * prevxm1 - 2 * rowx + 0 * nextxp1 + nextxp2;
                            }

                            if (res < 0)
                                res = 0;
                            if (res > 255)
                                res = 255;

                            border[y * 4] = (uint8_t)res;

                            trow[x] = rowx;

                            x = 1;
                            if (border_type == BORDER_REPLICATE)
                            {
                                rowx = nextxp1;
                                nextxp1 = nextxp2;
                            }
                            else if (border_type == BORDER_CONSTANT)
                            {
                                prevxm1 = rowx;
                                rowx = nextxp1;
                                nextxp1 = nextxp2;
                            }
                            else if (border_type == BORDER_REFLECT)
                            {
                                prevxm1 = rowx;
                                prevxm2 = prevxm1;
                                rowx = nextxp1;
                                nextxp1 = nextxp2;
                            }
                            else if (border_type == BORDER_REFLECT_101)
                            {
                                prevxm1 = rowx;
                                rowx = nextxp1;
                                nextxp1 = nextxp2;
                                prevxm2 = rowx;
                            }

                            if (dy == 0)
                            {
                                nextxp2 = srowm2[3] + 4 * srowm1[3] + 6 * srow0[3] + 4 * srowp1[3] + srowp2[3];
                            }
                            else if (dy == 1)
                            {
                                nextxp2 = -1 * srowm2[3] - 2 * srowm1[3] + 0 * srow0[3] + 2 * srowp1[3] + srowp2[3];
                            }
                            else
                            {
                                nextxp2 = srowm2[3] + 0 * srowm1[3] - 2 * srow0[3] + 0 * srowp1[3] + srowp2[3];
                            }

                            if (dx == 0)
                            {
                                res = prevxm2 + 4 * prevxm1 + 6 * rowx + 4 * nextxp1 + nextxp2;
                            }
                            else if (dx == 1)
                            {
                                res = -1 * prevxm2 - 2 * prevxm1 + 0 * rowx + 2 * nextxp1 + nextxp2;
                            }
                            else
                            {
                                res = prevxm2 + 0 * prevxm1 - 2 * rowx + 0 * nextxp1 + nextxp2;
                            }

                            if (res < 0)
                                res = 0;
                            if (res > 255)
                                res = 255;

                            border[y * 4 + 1] = (uint8_t)res;
                            trow[x] = rowx;

                            x = 2;
                            size_t vl = 0;
                            vint16m8_t vsrowm2, vsrowm1, vsrow0, vsrowp1, vsrowp2;
                            vint16m8_t vrowx;

                            for (; x < width - 2; x += vl)
                            {
                                vl = __riscv_vsetvl_e8m4(width - 2 - x);
                                vsrowm2 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srowm2 + x, vl), vl));
                                vsrowm1 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srowm1 + x, vl), vl));
                                vsrow0 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow0 + x, vl), vl));
                                vsrowp1 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srowp1 + x, vl), vl));
                                vsrowp2 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srowp2 + x, vl), vl));
                                if (dy == 0)
                                {
                                    vrowx = __riscv_vadd_vv_i16m8(vsrowm2, vsrowp2, vl);
                                    vrowx = __riscv_vmacc_vx_i16m8(vrowx, 4, vsrowm1, vl);
                                    vrowx = __riscv_vmacc_vx_i16m8(vrowx, 6, vsrow0, vl);
                                    vrowx = __riscv_vmacc_vx_i16m8(vrowx, 4, vsrowp1, vl);
                                }
                                else if (dy == 1)
                                {
                                    vrowx = __riscv_vsub_vv_i16m8(vsrowp2, vsrowm2, vl);
                                    vrowx = __riscv_vmacc_vx_i16m8(vrowx, -2, vsrowm1, vl);
                                    vrowx = __riscv_vmacc_vx_i16m8(vrowx, 2, vsrowp1, vl);
                                }
                                else
                                {
                                    vrowx = __riscv_vadd_vv_i16m8(vsrowm2, vsrowp2, vl);
                                    vrowx = __riscv_vmacc_vx_i16m8(vrowx, -2, vsrow0, vl);
                                }

                                __riscv_vse16_v_i16m8(trow + x, vrowx, vl);
                            }

                            x = width - 2;
                            if (dy == 0)
                            {
                                prevxm2 = srowm2[x - 2] + 4 * srowm1[x - 2] + 6 * srow0[x - 2] + 4 * srowp1[x - 2] + srowp2[x - 2];
                                prevxm1 = srowm2[x - 1] + 4 * srowm1[x - 1] + 6 * srow0[x - 1] + 4 * srowp1[x - 1] + srowp2[x - 1];
                                rowx = srowm2[x] + 4 * srowm1[x] + 6 * srow0[x] + 4 * srowp1[x] + srowp2[x];
                                nextxp1 = srowm2[x + 1] + 4 * srowm1[x + 1] + 6 * srow0[x + 1] + 4 * srowp1[x + 1] + srowp2[x + 1];
                            }
                            else if (dy == 1)
                            {
                                prevxm2 = -1 * srowm2[x - 2] - 2 * srowm1[x - 2] + 0 * srow0[x - 2] + 2 * srowp1[x - 2] + srowp2[x - 2];
                                prevxm1 = -1 * srowm2[x - 1] - 2 * srowm1[x - 1] + 0 * srow0[x - 1] + 2 * srowp1[x - 1] + srowp2[x - 1];
                                rowx = -1 * srowm2[x] - 2 * srowm1[x] + 0 * srow0[x] + 2 * srowp1[x] + srowp2[x];
                                nextxp1 = -1 * srowm2[x + 1] - 2 * srowm1[x + 1] + 0 * srow0[x + 1] + 2 * srowp1[x + 1] + srowp2[x + 1];
                            }
                            else
                            {
                                prevxm2 = srowm2[x - 2] + 0 * srowm1[x - 2] - 2 * srow0[x - 2] + 0 * srowp1[x - 2] + srowp2[x - 2];
                                prevxm1 = srowm2[x - 1] + 0 * srowm1[x - 1] - 2 * srow0[x - 1] + 0 * srowp1[x - 1] + srowp2[x - 1];
                                rowx = srowm2[x] + 0 * srowm1[x] - 2 * srow0[x] + 0 * srowp1[x] + srowp2[x];
                                nextxp1 = srowm2[x + 1] + 0 * srowm1[x + 1] - 2 * srow0[x + 1] + 0 * srowp1[x + 1] + srowp2[x + 1];
                            }
                            if (border_type == BORDER_REPLICATE)
                            {
                                nextxp2 = nextxp1;
                            }
                            else if (border_type == BORDER_CONSTANT)
                            {
                                nextxp2 = borderValue;
                            }
                            else if (border_type == BORDER_REFLECT)
                            {
                                nextxp2 = nextxp1;
                            }
                            else if (border_type == BORDER_REFLECT_101)
                            {
                                nextxp2 = rowx;
                            }

                            if (dx == 0)
                            {
                                res = prevxm2 + 4 * prevxm1 + 6 * rowx + 4 * nextxp1 + nextxp2;
                            }
                            else if (dx == 1)
                            {
                                res = -1 * prevxm2 - 2 * prevxm1 + 0 * rowx + 2 * nextxp1 + nextxp2;
                            }
                            else
                            {
                                res = prevxm2 + 0 * prevxm1 - 2 * rowx + 0 * nextxp1 + nextxp2;
                            }

                            if (res < 0)
                                res = 0;
                            if (res > 255)
                                res = 255;

                            border[y * 4 + 2] = (uint8_t)res;
                            trow[x] = rowx;

                            x = width - 1;
                            if (border_type == BORDER_REPLICATE)
                            {
                                prevxm2 = prevxm1;
                                prevxm1 = rowx;
                                rowx = nextxp1;
                                nextxp1 = nextxp2;
                            }
                            else if (border_type == BORDER_CONSTANT)
                            {
                                prevxm2 = prevxm1;
                                prevxm1 = rowx;
                                rowx = nextxp1;
                                nextxp1 = borderValue;
                            }
                            else if (border_type == BORDER_REFLECT)
                            {
                                prevxm2 = prevxm1;
                                prevxm1 = rowx;
                                rowx = nextxp1;
                                nextxp1 = rowx;
                                nextxp2 = prevxm1;
                            }
                            else if (border_type == BORDER_REFLECT_101)
                            {
                                prevxm2 = prevxm1;
                                prevxm1 = rowx;
                                rowx = nextxp1;
                                nextxp1 = prevxm1;
                                nextxp2 = prevxm2;
                            }

                            if (dx == 0)
                            {
                                res = prevxm2 + 4 * prevxm1 + 6 * rowx + 4 * nextxp1 + nextxp2;
                            }
                            else if (dx == 1)
                            {
                                res = -1 * prevxm2 - 2 * prevxm1 + 0 * rowx + 2 * nextxp1 + nextxp2;
                            }
                            else
                            {
                                res = prevxm2 + 0 * prevxm1 - 2 * rowx + 0 * nextxp1 + nextxp2;
                            }

                            if (res < 0)
                                res = 0;
                            if (res > 255)
                                res = 255;

                            border[y * 4 + 3] = (uint8_t)res;
                            trow[x] = rowx;
                            if (y > 1)
                            {
                                int target_y = y - 2;
                                int16_t *target_trow;
                                if (target_y % 3 == 0)
                                    target_trow = trow0;
                                else if (target_y % 3 == 1)
                                    target_trow = trow1;
                                else
                                    target_trow = trow2;

                                uint8_t *drow_target = dst_data + target_y * dst_step;

                                drow_target[0] = border[target_y * 4];
                                drow_target[1] = border[target_y * 4 + 1];
                                drow_target[width - 2] = border[target_y * 4 + 2];
                                drow_target[width - 1] = border[target_y * 4 + 3];

                                for (x = 2; x < width - 2; x += vl)
                                {
                                    vl = __riscv_vsetvl_e16m8(width - 2 - x);
                                    vint16m8_t vprevxm2 = __riscv_vle16_v_i16m8(target_trow + x - 2, vl);
                                    vint16m8_t vprevxm1 = __riscv_vle16_v_i16m8(target_trow + x - 1, vl);
                                    vint16m8_t vrowx = __riscv_vle16_v_i16m8(target_trow + x, vl);
                                    vint16m8_t vnextxp1 = __riscv_vle16_v_i16m8(target_trow + x + 1, vl);
                                    vint16m8_t vnextxp2 = __riscv_vle16_v_i16m8(target_trow + x + 2, vl);

                                    vint16m8_t grad;
                                    if (dx == 0)
                                    {
                                        grad = __riscv_vadd_vv_i16m8(vprevxm2, vnextxp2, vl);
                                        grad = __riscv_vmacc_vx_i16m8(grad, 4, vprevxm1, vl);
                                        grad = __riscv_vmacc_vx_i16m8(grad, 6, vrowx, vl);
                                        grad = __riscv_vmacc_vx_i16m8(grad, 4, vnextxp1, vl);
                                    }
                                    else if (dx == 1)
                                    {
                                        grad = __riscv_vsub_vv_i16m8(vnextxp2, vprevxm2, vl);
                                        grad = __riscv_vmacc_vx_i16m8(grad, -2, vprevxm1, vl);
                                        grad = __riscv_vmacc_vx_i16m8(grad, 2, vnextxp1, vl);
                                    }
                                    else
                                    {
                                        grad = __riscv_vadd_vv_i16m8(vprevxm2, vnextxp2, vl);
                                        grad = __riscv_vmacc_vx_i16m8(grad, -2, vrowx, vl);
                                    }

                                    vuint8m4_t vres = __riscv_vncvt_x_x_w_u8m4(__riscv_vreinterpret_v_i16m8_u16m8(__riscv_vmax_vx_i16m8(__riscv_vmin_vx_i16m8(grad, 255, vl), 0, vl)), vl);

                                    __riscv_vse8_v_u8m4(drow_target + x, vres, vl);
                                }
                            }
                        }

                        for (int remaining_y = (height >= 2 ? height - 2 : 0); remaining_y < height; remaining_y++)
                        {
                            int16_t *current_trow;
                            if (remaining_y % 3 == 0)
                                current_trow = trow0;
                            else if (remaining_y % 3 == 1)
                                current_trow = trow1;
                            else
                                current_trow = trow2;

                            uint8_t *drow_current = (uint8_t *)(dst_data + remaining_y * dst_step);

                            drow_current[0] = border[remaining_y * 4];
                            drow_current[1] = border[remaining_y * 4 + 1];
                            drow_current[width - 2] = border[remaining_y * 4 + 2];
                            drow_current[width - 1] = border[remaining_y * 4 + 3];

                            size_t vl = 0;
                            int x = 0;
                            for (x = 2; x < width - 2; x += vl)
                            {
                                vl = __riscv_vsetvl_e16m8(width - 2 - x);
                                vint16m8_t vprevxm2 = __riscv_vle16_v_i16m8(current_trow + x - 2, vl);
                                vint16m8_t vprevxm1 = __riscv_vle16_v_i16m8(current_trow + x - 1, vl);
                                vint16m8_t vrowx = __riscv_vle16_v_i16m8(current_trow + x, vl);
                                vint16m8_t vnextxp1 = __riscv_vle16_v_i16m8(current_trow + x + 1, vl);
                                vint16m8_t vnextxp2 = __riscv_vle16_v_i16m8(current_trow + x + 2, vl);

                                vint16m8_t grad;
                                if (dx == 0)
                                {
                                    grad = __riscv_vadd_vv_i16m8(vprevxm2, vnextxp2, vl);
                                    grad = __riscv_vmacc_vx_i16m8(grad, 4, vprevxm1, vl);
                                    grad = __riscv_vmacc_vx_i16m8(grad, 6, vrowx, vl);
                                    grad = __riscv_vmacc_vx_i16m8(grad, 4, vnextxp1, vl);
                                }
                                else if (dx == 1)
                                {
                                    grad = __riscv_vsub_vv_i16m8(vnextxp2, vprevxm2, vl);
                                    grad = __riscv_vmacc_vx_i16m8(grad, -2, vprevxm1, vl);
                                    grad = __riscv_vmacc_vx_i16m8(grad, 2, vnextxp1, vl);
                                }
                                else
                                {
                                    grad = __riscv_vadd_vv_i16m8(vprevxm2, vnextxp2, vl);
                                    grad = __riscv_vmacc_vx_i16m8(grad, -2, vrowx, vl);
                                }

                                vuint8m4_t vres = __riscv_vncvt_x_x_w_u8m4(__riscv_vreinterpret_v_i16m8_u16m8(__riscv_vmax_vx_i16m8(__riscv_vmin_vx_i16m8(grad, 255, vl), 0, vl)), vl);

                                __riscv_vse8_v_u8m4(drow_current + x, vres, vl);
                            }
                        }
                    }

                    else
                    {
                        return CV_HAL_ERROR_NOT_IMPLEMENTED;
                    }
                }

                else if (dst_depth == CV_16S)
                {
                    if (ksize == 3)
                    {
                        if (border_type == BORDER_REPLICATE)
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
                                const uint8_t *srow0 = src_data + (y > 0 ? y - 1 : 0) * src_step;
                                const uint8_t *srow2 = src_data + (y < height - 1 ? y + 1 : height - 1) * src_step;

                                int16_t *drow0 = (int16_t *)(dst_data + (y > 0 ? y - 1 : 0) * dst_step);
                                int16_t *drow1 = (int16_t *)(dst_data + y * dst_step);

                                int16_t *trow = (y % 2) ? trow1 : trow0;

                                size_t vl = __riscv_vsetvl_e8m4(width - 1);

                                int x = 0;
                                int16_t prevx = 0, rowx = 0, nextx = 0, res = 0;
                                if (dy == 0)
                                {
                                    prevx = srow2[0] + 2 * srow1[0] + srow0[0];
                                    nextx = srow2[1] + 2 * srow1[1] + srow0[1];
                                }
                                else if (dy == 2)
                                {
                                    prevx = srow2[0] - 2 * srow1[0] + srow0[0];
                                    nextx = srow2[1] - 2 * srow1[1] + srow0[1];
                                }
                                else
                                {
                                    prevx = srow2[0] - srow0[0];
                                    nextx = srow2[1] - srow0[1];
                                }
                                rowx = prevx;
                                if (dx == 1)
                                {
                                    res = nextx - prevx;
                                }
                                else if (dx == 0)
                                {
                                    res = prevx + 2 * rowx + nextx;
                                }
                                else
                                {
                                    res = prevx - 2 * rowx + nextx;
                                }

                                vint16m8_t vrowx, vprevx, vnextx;
                                vrowx = __riscv_vmv_v_x_i16m8(rowx, vl);
                                vprevx = vrowx;

                                for (x = 0; x < width - 1; x += vl)
                                {
                                    vl = __riscv_vsetvl_e8m4(width - x - 1);

                                    vint16m8_t vsrow0 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow0 + x + 1, vl), vl));
                                    vint16m8_t vsrow1 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow1 + x + 1, vl), vl));
                                    vint16m8_t vsrow2 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow2 + x + 1, vl), vl));

                                    if (dy == 0)
                                    {
                                        vint16m8_t vsrow2_vsrow0 = __riscv_vadd_vv_i16m8(vsrow2, vsrow0, vl);
                                        vint16m8_t vsrow1_2x = __riscv_vadd_vv_i16m8(vsrow1, vsrow1, vl);
                                        vnextx = __riscv_vadd_vv_i16m8(vsrow2_vsrow0, vsrow1_2x, vl);

                                        vrowx = __riscv_vslideup_vx_i16m8(vrowx, vnextx, 1, vl);
                                        vprevx = __riscv_vslideup_vx_i16m8(vprevx, vrowx, 1, vl);
                                    }
                                    else if (dy == 2)
                                    {
                                        vint16m8_t vsrow2_vsrow0 = __riscv_vadd_vv_i16m8(vsrow2, vsrow0, vl);
                                        vint16m8_t vsrow1_2x = __riscv_vadd_vv_i16m8(vsrow1, vsrow1, vl);
                                        vnextx = __riscv_vsub_vv_i16m8(vsrow2_vsrow0, vsrow1_2x, vl);

                                        vrowx = __riscv_vslideup_vx_i16m8(vrowx, vnextx, 1, vl);
                                        vprevx = __riscv_vslideup_vx_i16m8(vprevx, vrowx, 1, vl);
                                    }
                                    else
                                    {
                                        vnextx = __riscv_vsub_vv_i16m8(vsrow2, vsrow0, vl);

                                        vrowx = __riscv_vslideup_vx_i16m8(vrowx, vnextx, 1, vl);
                                        vprevx = __riscv_vslideup_vx_i16m8(vprevx, vrowx, 1, vl);
                                    }

                                    vint16m8_t grad;
                                    if (dx == 1)
                                    {
                                        grad = __riscv_vsub_vv_i16m8(vnextx, vprevx, vl);
                                    }
                                    else if (dx == 0)
                                    {
                                        grad = __riscv_vadd_vv_i16m8(__riscv_vadd_vv_i16m8(vprevx, vnextx, vl), __riscv_vadd_vv_i16m8(vrowx, vrowx, vl), vl);
                                    }
                                    else
                                    {
                                        grad = __riscv_vsub_vv_i16m8(__riscv_vadd_vv_i16m8(vprevx, vnextx, vl), __riscv_vadd_vv_i16m8(vrowx, vrowx, vl), vl);
                                    }

                                    vprevx = vrowx;
                                    vrowx = vnextx;

                                    vrowx = __riscv_vslidedown_vx_i16m8(vrowx, vl - 1, vl);
                                    vprevx = __riscv_vslidedown_vx_i16m8(vprevx, vl - 1, vl);

                                    __riscv_vse16_v_i16m8(trow + x, grad, vl);
                                }

                                if (dy == 0)
                                {
                                    prevx = srow2[width - 2] + 2 * srow1[width - 2] + srow0[width - 2];
                                    rowx = srow2[width - 1] + 2 * srow1[width - 1] + srow0[width - 1];
                                }
                                else if (dy == 2)
                                {
                                    prevx = srow2[width - 2] - 2 * srow1[width - 2] + srow0[width - 2];
                                    rowx = srow2[width - 1] - 2 * srow1[width - 1] + srow0[width - 1];
                                }
                                else
                                {
                                    prevx = srow2[width - 2] - srow0[width - 2];
                                    rowx = srow2[width - 1] - srow0[width - 1];
                                }
                                nextx = rowx;
                                if (dx == 1)
                                {
                                    res = nextx - prevx;
                                }
                                else if (dx == 0)
                                {
                                    res = prevx + 2 * rowx + nextx;
                                }
                                else
                                {
                                    res = prevx - 2 * rowx + nextx;
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

                        else if (border_type == BORDER_CONSTANT)
                        {
                            int align_size = (width + 2 + 7) & -8;
                            constexpr size_t alignment = 64;
                            size_t bufferSize = (align_size << 1) + alignment;
                            int16_t *_tempBuf = static_cast<int16_t *>(
                                aligned_alloc(alignment, bufferSize * sizeof(int16_t)));

                            int16_t *trow0 = reinterpret_cast<int16_t *>(
                                (reinterpret_cast<uintptr_t>(_tempBuf) + alignment - 1) & ~(alignment - 1));
                            int16_t *trow1 = trow0 + align_size;

                            vint16m8_t vborder = __riscv_vmv_v_x_i16m8(borderValue, __riscv_vsetvlmax_e16m8());

                            for (int y = 0; y < height; y++)
                            {
                                const uint8_t *srow1 = src_data + y * src_step;
                                const uint8_t *srow0 = nullptr;
                                const uint8_t *srow2 = nullptr;

                                if (y > 0)
                                    srow0 = src_data + (y - 1) * src_step;
                                if (y < height - 1)
                                    srow2 = src_data + (y + 1) * src_step;

                                int16_t *drow0 = (int16_t *)(dst_data + (y > 0 ? y - 1 : 0) * dst_step);
                                int16_t *drow1 = (int16_t *)(dst_data + y * dst_step);
                                int16_t *trow = (y % 2) ? trow1 : trow0;

                                size_t vl = __riscv_vsetvl_e8m4(width - 1);
                                int x = 0;
                                int16_t prevx = 0, rowx = 0, nextx = 0, res = 0;

                                if (dy == 0)
                                {
                                    int16_t s0 = srow0 ? srow0[0] : borderValue;
                                    int16_t s1 = srow1[0];
                                    int16_t s2 = srow2 ? srow2[0] : borderValue;
                                    prevx = s2 + 2 * s1 + s0;

                                    s0 = srow0 ? srow0[1] : borderValue;
                                    s1 = srow1[1];
                                    s2 = srow2 ? srow2[1] : borderValue;
                                    nextx = s2 + 2 * s1 + s0;
                                }
                                else if (dy == 2)
                                {
                                    int16_t s0 = srow0 ? srow0[0] : borderValue;
                                    int16_t s1 = srow1[0];
                                    int16_t s2 = srow2 ? srow2[0] : borderValue;
                                    prevx = s2 - 2 * s1 + s0;

                                    s0 = srow0 ? srow0[1] : borderValue;
                                    s1 = srow1[1];
                                    s2 = srow2 ? srow2[1] : borderValue;
                                    nextx = s2 - 2 * s1 + s0;
                                }
                                else
                                {
                                    int16_t s0 = srow0 ? srow0[0] : borderValue;
                                    int16_t s2 = srow2 ? srow2[0] : borderValue;
                                    prevx = s2 - s0;

                                    s0 = srow0 ? srow0[1] : borderValue;
                                    s2 = srow2 ? srow2[1] : borderValue;
                                    nextx = s2 - s0;
                                }

                                rowx = prevx;
                                if (dx == 1)
                                    res = nextx - prevx;
                                else if (dx == 0)
                                    res = prevx + 2 * rowx + nextx;
                                else
                                    res = prevx - 2 * rowx + nextx;

                                trow[0] = res;

                                vint16m8_t vrowx = __riscv_vmv_v_x_i16m8(rowx, vl);
                                vint16m8_t vprevx = vrowx, vnextx;

                                for (x = 0; x < width - 1; x += vl)
                                {
                                    vl = __riscv_vsetvl_e8m4(width - x - 1);

                                    vint16m8_t vsrow1 = __riscv_vreinterpret_v_u16m8_i16m8(
                                        __riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow1 + x + 1, vl), vl));

                                    vint16m8_t vsrow0, vsrow2;
                                    if (srow0)
                                        vsrow0 = __riscv_vreinterpret_v_u16m8_i16m8(
                                            __riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow0 + x + 1, vl), vl));
                                    else
                                        vsrow0 = vborder;

                                    if (srow2)
                                        vsrow2 = __riscv_vreinterpret_v_u16m8_i16m8(
                                            __riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow2 + x + 1, vl), vl));
                                    else
                                        vsrow2 = vborder;

                                    if (dy == 0)
                                    {
                                        vint16m8_t vsrow2_vsrow0 = __riscv_vadd_vv_i16m8(vsrow2, vsrow0, vl);
                                        vint16m8_t vsrow1_2x = __riscv_vadd_vv_i16m8(vsrow1, vsrow1, vl);
                                        vnextx = __riscv_vadd_vv_i16m8(vsrow2_vsrow0, vsrow1_2x, vl);
                                    }
                                    else if (dy == 2)
                                    {
                                        vint16m8_t vsrow2_vsrow0 = __riscv_vadd_vv_i16m8(vsrow2, vsrow0, vl);
                                        vint16m8_t vsrow1_2x = __riscv_vadd_vv_i16m8(vsrow1, vsrow1, vl);
                                        vnextx = __riscv_vsub_vv_i16m8(vsrow2_vsrow0, vsrow1_2x, vl);
                                    }
                                    else
                                    {
                                        vnextx = __riscv_vsub_vv_i16m8(vsrow2, vsrow0, vl);
                                    }

                                    vrowx = __riscv_vslideup_vx_i16m8(vrowx, vnextx, 1, vl);
                                    vprevx = __riscv_vslideup_vx_i16m8(vprevx, vrowx, 1, vl);

                                    vint16m8_t grad;
                                    if (dx == 1)
                                        grad = __riscv_vsub_vv_i16m8(vnextx, vprevx, vl);
                                    else if (dx == 0)
                                        grad = __riscv_vadd_vv_i16m8(__riscv_vadd_vv_i16m8(vprevx, vnextx, vl),
                                                                     __riscv_vadd_vv_i16m8(vrowx, vrowx, vl), vl);
                                    else
                                        grad = __riscv_vsub_vv_i16m8(__riscv_vadd_vv_i16m8(vprevx, vnextx, vl),
                                                                     __riscv_vadd_vv_i16m8(vrowx, vrowx, vl), vl);

                                    vprevx = vrowx;
                                    vrowx = vnextx;

                                    vrowx = __riscv_vslidedown_vx_i16m8(vrowx, vl - 1, vl);
                                    vprevx = __riscv_vslidedown_vx_i16m8(vprevx, vl - 1, vl);

                                    __riscv_vse16_v_i16m8(trow + x, grad, vl);
                                }

                                if (dy == 0)
                                {
                                    int16_t s0 = srow0 ? srow0[width - 1] : borderValue;
                                    int16_t s1 = srow1[width - 1];
                                    int16_t s2 = srow2 ? srow2[width - 1] : borderValue;
                                    rowx = s2 + 2 * s1 + s0;
                                }
                                else if (dy == 2)
                                {
                                    int16_t s0 = srow0 ? srow0[width - 1] : borderValue;
                                    int16_t s1 = srow1[width - 1];
                                    int16_t s2 = srow2 ? srow2[width - 1] : borderValue;
                                    rowx = s2 - 2 * s1 + s0;
                                }
                                else
                                {
                                    int16_t s0 = srow0 ? srow0[width - 1] : borderValue;
                                    int16_t s2 = srow2 ? srow2[width - 1] : borderValue;
                                    rowx = s2 - s0;
                                }
                                nextx = rowx;

                                if (dx == 1)
                                    res = nextx - prevx;
                                else if (dx == 0)
                                    res = prevx + 2 * rowx + nextx;
                                else
                                    res = prevx - 2 * rowx + nextx;

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

                        else
                        {
                            return CV_HAL_ERROR_NOT_IMPLEMENTED;
                        }
                    }

                    else if (ksize == 5)
                    {
                        if (border_type != BORDER_REPLICATE &&
                            border_type != BORDER_CONSTANT &&
                            border_type != BORDER_REFLECT &&
                            border_type != BORDER_REFLECT_101)
                        {
                            return CV_HAL_ERROR_NOT_IMPLEMENTED;
                        }
                        int align_size = (width + 4 + 7) & -8;
                        constexpr size_t alignment = 64;
                        size_t bufferSize = (align_size * 3) + alignment;
                        int16_t *_tempBuf = static_cast<int16_t *>(
                            aligned_alloc(alignment, bufferSize * sizeof(int16_t)));

                        int16_t *trow0 = reinterpret_cast<int16_t *>(
                            (reinterpret_cast<uintptr_t>(_tempBuf) + alignment - 1) & ~(alignment - 1));
                        int16_t *trow1 = trow0 + align_size;
                        int16_t *trow2 = trow1 + align_size;

                        const uint8_t *srowm2, *srowm1, *srow0, *srowp1, *srowp2;
                        int16_t *border = (int16_t *)malloc((4 * height) * sizeof(int16_t));
                        for (int y = 0; y < height; y++)
                        {
                            if (border_type == BORDER_REPLICATE)
                            {
                                srowm2 = src_data + (y > 2 ? y - 2 : 0) * src_step;
                                srowm1 = src_data + (y > 1 ? y - 1 : 0) * src_step;
                                srow0 = src_data + y * src_step;
                                srowp1 = src_data + (y + 1 < height - 1 ? y + 1 : height - 1) * src_step;
                                srowp2 = src_data + (y + 2 < height - 1 ? y + 2 : height - 1) * src_step;
                            }
                            else if (border_type == BORDER_CONSTANT)
                            {
                                uint8_t *zero_row = (uint8_t *)malloc(width * sizeof(uint8_t));
                                memset(zero_row, 0, width * sizeof(uint8_t));
                                srowm2 = y > 2 ? (src_data + (y - 2) * src_step) : zero_row;
                                srowm1 = y > 1 ? (src_data + (y - 1) * src_step) : zero_row;
                                srow0 = src_data + y * src_step;
                                srowp1 = y + 1 < height - 1 ? (src_data + (y + 1) * src_step) : zero_row;
                                srowp2 = y + 2 < height - 1 ? (src_data + (y + 2) * src_step) : zero_row;
                            }
                            else if (border_type == BORDER_REFLECT)
                            {
                                srowm2 = y >= 2 ? (src_data + (y - 2) * src_step) : (y == 1 ? src_data : (src_data + src_step));
                                srowm1 = y >= 1 ? (src_data + (y - 1) * src_step) : src_data;
                                srow0 = src_data + y * src_step;
                                srowp1 = y + 1 <= height - 1 ? (src_data + (y + 1) * src_step) : src_data + y * src_step;
                                srowp2 = y + 2 <= height - 1 ? (src_data + (y + 2) * src_step) : (y == height - 2 ? (src_data + (y - 1) * src_step) : (src_data + y * src_step));
                            }
                            else if (border_type == BORDER_REFLECT_101)
                            {
                                srowm2 = y > 2 ? (src_data + (y - 2) * src_step) : (src_data + 2 * src_step);
                                srowm1 = y >= 1 ? (src_data + (y - 1) * src_step) : (src_data + src_step);
                                srow0 = src_data + y * src_step;
                                srowp1 = y + 1 <= height - 1 ? (src_data + (y + 1) * src_step) : (src_data + (y - 1) * src_step);
                                srowp2 = y + 2 <= height - 1 ? (src_data + (y + 2) * src_step) : (y == height - 2 ? (src_data + (y - 2) * src_step) : (src_data + y * src_step));
                            }

                            int16_t *trow;
                            if (y % 3 == 0)
                            {
                                trow = trow0;
                            }
                            else if (y % 3 == 1)
                            {
                                trow = trow1;
                            }
                            else
                            {
                                trow = trow2;
                            }

                            int16_t prevxm2 = 0, prevxm1 = 0, rowx = 0, nextxp1 = 0, nextxp2 = 0;

                            int x = 0;
                            if (dy == 0)
                            {
                                rowx = srowm2[0] + 4 * srowm1[0] + 6 * srow0[0] + 4 * srowp1[0] + srowp2[0];
                                nextxp1 = srowm2[1] + 4 * srowm1[1] + 6 * srow0[1] + 4 * srowp1[1] + srowp2[1];
                                nextxp2 = srowm2[2] + 4 * srowm1[2] + 6 * srow0[2] + 4 * srowp1[2] + srowp2[2];
                            }
                            else if (dy == 1)
                            {
                                rowx = -1 * srowm2[0] - 2 * srowm1[0] + 0 * srow0[0] + 2 * srowp1[0] + srowp2[0];
                                nextxp1 = -1 * srowm2[1] - 2 * srowm1[1] + 0 * srow0[1] + 2 * srowp1[1] + srowp2[1];
                                nextxp2 = -1 * srowm2[2] - 2 * srowm1[2] + 0 * srow0[2] + 2 * srowp1[2] + srowp2[2];
                            }
                            else
                            {
                                rowx = srowm2[0] + 0 * srowm1[0] - 2 * srow0[0] + 0 * srowp1[0] + srowp2[0];
                                nextxp1 = srowm2[1] + 0 * srowm1[1] - 2 * srow0[1] + 0 * srowp1[1] + srowp2[1];
                                nextxp2 = srowm2[2] + 0 * srowm1[2] - 2 * srow0[2] + 0 * srowp1[2] + srowp2[2];
                            }
                            if (border_type == BORDER_REPLICATE)
                            {
                                prevxm1 = rowx;
                                prevxm2 = prevxm1;
                            }
                            // border_type == BORDER_CONSTANT : prevxm1 = borderValue, prevxm2 = borderValue;
                            else if (border_type == BORDER_REFLECT)
                            {
                                prevxm1 = rowx;
                                prevxm2 = nextxp1;
                            }
                            else if (border_type == BORDER_REFLECT_101)
                            {
                                prevxm1 = nextxp1;
                                prevxm2 = nextxp2;
                            }

                            int16_t res = 0;
                            if (dx == 0)
                            {
                                res = prevxm2 + 4 * prevxm1 + 6 * rowx + 4 * nextxp1 + nextxp2;
                            }
                            else if (dx == 1)
                            {
                                res = -1 * prevxm2 - 2 * prevxm1 + 0 * rowx + 2 * nextxp1 + nextxp2;
                            }
                            else
                            {
                                res = prevxm2 + 0 * prevxm1 - 2 * rowx + 0 * nextxp1 + nextxp2;
                            }
                            border[y * 4] = res;
                            trow[x] = rowx;

                            x = 1;
                            if (border_type == BORDER_REPLICATE)
                            {
                                rowx = nextxp1;
                                nextxp1 = nextxp2;
                            }
                            else if (border_type == BORDER_CONSTANT)
                            {
                                prevxm1 = rowx;
                                rowx = nextxp1;
                                nextxp1 = nextxp2;
                            }
                            else if (border_type == BORDER_REFLECT)
                            {
                                prevxm1 = rowx;
                                prevxm2 = prevxm1;
                                rowx = nextxp1;
                                nextxp1 = nextxp2;
                            }
                            else if (border_type == BORDER_REFLECT_101)
                            {
                                prevxm1 = rowx;
                                rowx = nextxp1;
                                nextxp1 = nextxp2;
                                prevxm2 = rowx;
                            }

                            if (dy == 0)
                            {
                                nextxp2 = srowm2[3] + 4 * srowm1[3] + 6 * srow0[3] + 4 * srowp1[3] + srowp2[3];
                            }
                            else if (dy == 1)
                            {
                                nextxp2 = -1 * srowm2[3] - 2 * srowm1[3] + 0 * srow0[3] + 2 * srowp1[3] + srowp2[3];
                            }
                            else
                            {
                                nextxp2 = srowm2[3] + 0 * srowm1[3] - 2 * srow0[3] + 0 * srowp1[3] + srowp2[3];
                            }

                            if (dx == 0)
                            {
                                res = prevxm2 + 4 * prevxm1 + 6 * rowx + 4 * nextxp1 + nextxp2;
                            }
                            else if (dx == 1)
                            {
                                res = -1 * prevxm2 - 2 * prevxm1 + 0 * rowx + 2 * nextxp1 + nextxp2;
                            }
                            else
                            {
                                res = prevxm2 + 0 * prevxm1 - 2 * rowx + 0 * nextxp1 + nextxp2;
                            }

                            border[y * 4 + 1] = res;
                            trow[x] = rowx;

                            x = 2;
                            size_t vl = 0;
                            vint16m8_t vsrowm2, vsrowm1, vsrow0, vsrowp1, vsrowp2;
                            vint16m8_t vrowx;

                            for (; x < width - 2; x += vl)
                            {
                                vl = __riscv_vsetvl_e8m4(width - 2 - x);
                                vsrowm2 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srowm2 + x, vl), vl));
                                vsrowm1 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srowm1 + x, vl), vl));
                                vsrow0 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow0 + x, vl), vl));
                                vsrowp1 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srowp1 + x, vl), vl));
                                vsrowp2 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srowp2 + x, vl), vl));
                                if (dy == 0)
                                {
                                    vrowx = __riscv_vadd_vv_i16m8(vsrowm2, vsrowp2, vl);
                                    vrowx = __riscv_vmacc_vx_i16m8(vrowx, 4, vsrowm1, vl);
                                    vrowx = __riscv_vmacc_vx_i16m8(vrowx, 6, vsrow0, vl);
                                    vrowx = __riscv_vmacc_vx_i16m8(vrowx, 4, vsrowp1, vl);
                                }
                                else if (dy == 1)
                                {
                                    vrowx = __riscv_vsub_vv_i16m8(vsrowp2, vsrowm2, vl);
                                    vrowx = __riscv_vmacc_vx_i16m8(vrowx, -2, vsrowm1, vl);
                                    vrowx = __riscv_vmacc_vx_i16m8(vrowx, 2, vsrowp1, vl);
                                }
                                else
                                {
                                    vrowx = __riscv_vadd_vv_i16m8(vsrowm2, vsrowp2, vl);
                                    vrowx = __riscv_vmacc_vx_i16m8(vrowx, -2, vsrow0, vl);
                                }

                                __riscv_vse16_v_i16m8(trow + x, vrowx, vl);
                            }

                            x = width - 2;
                            if (dy == 0)
                            {
                                prevxm2 = srowm2[x - 2] + 4 * srowm1[x - 2] + 6 * srow0[x - 2] + 4 * srowp1[x - 2] + srowp2[x - 2];
                                prevxm1 = srowm2[x - 1] + 4 * srowm1[x - 1] + 6 * srow0[x - 1] + 4 * srowp1[x - 1] + srowp2[x - 1];
                                rowx = srowm2[x] + 4 * srowm1[x] + 6 * srow0[x] + 4 * srowp1[x] + srowp2[x];
                                nextxp1 = srowm2[x + 1] + 4 * srowm1[x + 1] + 6 * srow0[x + 1] + 4 * srowp1[x + 1] + srowp2[x + 1];
                            }
                            else if (dy == 1)
                            {
                                prevxm2 = -1 * srowm2[x - 2] - 2 * srowm1[x - 2] + 0 * srow0[x - 2] + 2 * srowp1[x - 2] + srowp2[x - 2];
                                prevxm1 = -1 * srowm2[x - 1] - 2 * srowm1[x - 1] + 0 * srow0[x - 1] + 2 * srowp1[x - 1] + srowp2[x - 1];
                                rowx = -1 * srowm2[x] - 2 * srowm1[x] + 0 * srow0[x] + 2 * srowp1[x] + srowp2[x];
                                nextxp1 = -1 * srowm2[x + 1] - 2 * srowm1[x + 1] + 0 * srow0[x + 1] + 2 * srowp1[x + 1] + srowp2[x + 1];
                            }
                            else
                            {
                                prevxm2 = srowm2[x - 2] + 0 * srowm1[x - 2] - 2 * srow0[x - 2] + 0 * srowp1[x - 2] + srowp2[x - 2];
                                prevxm1 = srowm2[x - 1] + 0 * srowm1[x - 1] - 2 * srow0[x - 1] + 0 * srowp1[x - 1] + srowp2[x - 1];
                                rowx = srowm2[x] + 0 * srowm1[x] - 2 * srow0[x] + 0 * srowp1[x] + srowp2[x];
                                nextxp1 = srowm2[x + 1] + 0 * srowm1[x + 1] - 2 * srow0[x + 1] + 0 * srowp1[x + 1] + srowp2[x + 1];
                            }
                            if (border_type == BORDER_REPLICATE)
                            {
                                nextxp2 = nextxp1;
                            }
                            else if (border_type == BORDER_CONSTANT)
                            {
                                nextxp2 = borderValue;
                            }
                            else if (border_type == BORDER_REFLECT)
                            {
                                nextxp2 = nextxp1;
                            }
                            else if (border_type == BORDER_REFLECT_101)
                            {
                                nextxp2 = rowx;
                            }

                            if (dx == 0)
                            {
                                res = prevxm2 + 4 * prevxm1 + 6 * rowx + 4 * nextxp1 + nextxp2;
                            }
                            else if (dx == 1)
                            {
                                res = -1 * prevxm2 - 2 * prevxm1 + 0 * rowx + 2 * nextxp1 + nextxp2;
                            }
                            else
                            {
                                res = prevxm2 + 0 * prevxm1 - 2 * rowx + 0 * nextxp1 + nextxp2;
                            }

                            border[y * 4 + 2] = res;
                            trow[x] = rowx;

                            x = width - 1;
                            if (border_type == BORDER_REPLICATE)
                            {
                                prevxm2 = prevxm1;
                                prevxm1 = rowx;
                                rowx = nextxp1;
                                nextxp1 = nextxp2;
                            }
                            else if (border_type == BORDER_CONSTANT)
                            {
                                prevxm2 = prevxm1;
                                prevxm1 = rowx;
                                rowx = nextxp1;
                                nextxp1 = borderValue;
                            }
                            else if (border_type == BORDER_REFLECT)
                            {
                                prevxm2 = prevxm1;
                                prevxm1 = rowx;
                                rowx = nextxp1;
                                nextxp1 = rowx;
                                nextxp2 = prevxm1;
                            }
                            else if (border_type == BORDER_REFLECT_101)
                            {
                                prevxm2 = prevxm1;
                                prevxm1 = rowx;
                                rowx = nextxp1;
                                nextxp1 = prevxm1;
                                nextxp2 = prevxm2;
                            }

                            if (dx == 0)
                            {
                                res = prevxm2 + 4 * prevxm1 + 6 * rowx + 4 * nextxp1 + nextxp2;
                            }
                            else if (dx == 1)
                            {
                                res = -1 * prevxm2 - 2 * prevxm1 + 0 * rowx + 2 * nextxp1 + nextxp2;
                            }
                            else
                            {
                                res = prevxm2 + 0 * prevxm1 - 2 * rowx + 0 * nextxp1 + nextxp2;
                            }

                            border[y * 4 + 3] = res;
                            trow[x] = rowx;

                            if (y > 1)
                            {
                                int target_y = y - 2;
                                int16_t *target_trow;
                                if (target_y % 3 == 0)
                                    target_trow = trow0;
                                else if (target_y % 3 == 1)
                                    target_trow = trow1;
                                else
                                    target_trow = trow2;

                                int16_t *drow_target = (int16_t *)(dst_data + target_y * dst_step);

                                drow_target[0] = border[target_y * 4];
                                drow_target[1] = border[target_y * 4 + 1];
                                drow_target[width - 2] = border[target_y * 4 + 2];
                                drow_target[width - 1] = border[target_y * 4 + 3];

                                for (x = 2; x < width - 2; x += vl)
                                {
                                    vl = __riscv_vsetvl_e16m8(width - 2 - x);
                                    vint16m8_t vprevxm2 = __riscv_vle16_v_i16m8(target_trow + x - 2, vl);
                                    vint16m8_t vprevxm1 = __riscv_vle16_v_i16m8(target_trow + x - 1, vl);
                                    vint16m8_t vrowx = __riscv_vle16_v_i16m8(target_trow + x, vl);
                                    vint16m8_t vnextxp1 = __riscv_vle16_v_i16m8(target_trow + x + 1, vl);
                                    vint16m8_t vnextxp2 = __riscv_vle16_v_i16m8(target_trow + x + 2, vl);

                                    vint16m8_t grad;
                                    if (dx == 0)
                                    {
                                        grad = __riscv_vadd_vv_i16m8(vprevxm2, vnextxp2, vl);
                                        grad = __riscv_vmacc_vx_i16m8(grad, 4, vprevxm1, vl);
                                        grad = __riscv_vmacc_vx_i16m8(grad, 6, vrowx, vl);
                                        grad = __riscv_vmacc_vx_i16m8(grad, 4, vnextxp1, vl);
                                    }
                                    else if (dx == 1)
                                    {
                                        grad = __riscv_vsub_vv_i16m8(vnextxp2, vprevxm2, vl);
                                        grad = __riscv_vmacc_vx_i16m8(grad, -2, vprevxm1, vl);
                                        grad = __riscv_vmacc_vx_i16m8(grad, 2, vnextxp1, vl);
                                    }
                                    else
                                    {
                                        grad = __riscv_vadd_vv_i16m8(vprevxm2, vnextxp2, vl);
                                        grad = __riscv_vmacc_vx_i16m8(grad, -2, vrowx, vl);
                                    }

                                    __riscv_vse16_v_i16m8(drow_target + x, grad, vl);
                                }
                            }
                        }

                        for (int remaining_y = (height >= 2 ? height - 2 : 0); remaining_y < height; remaining_y++)
                        {
                            int16_t *current_trow;
                            if (remaining_y % 3 == 0)
                                current_trow = trow0;
                            else if (remaining_y % 3 == 1)
                                current_trow = trow1;
                            else
                                current_trow = trow2;

                            int16_t *drow_current = (int16_t *)(dst_data + remaining_y * dst_step);

                            drow_current[0] = border[remaining_y * 4];
                            drow_current[1] = border[remaining_y * 4 + 1];
                            drow_current[width - 2] = border[remaining_y * 4 + 2];
                            drow_current[width - 1] = border[remaining_y * 4 + 3];

                            size_t vl = 0;
                            int x = 0;
                            for (x = 2; x < width - 2; x += vl)
                            {
                                vl = __riscv_vsetvl_e16m8(width - 2 - x);
                                vint16m8_t vprevxm2 = __riscv_vle16_v_i16m8(current_trow + x - 2, vl);
                                vint16m8_t vprevxm1 = __riscv_vle16_v_i16m8(current_trow + x - 1, vl);
                                vint16m8_t vrowx = __riscv_vle16_v_i16m8(current_trow + x, vl);
                                vint16m8_t vnextxp1 = __riscv_vle16_v_i16m8(current_trow + x + 1, vl);
                                vint16m8_t vnextxp2 = __riscv_vle16_v_i16m8(current_trow + x + 2, vl);

                                vint16m8_t grad;
                                if (dx == 0)
                                {
                                    grad = __riscv_vadd_vv_i16m8(vprevxm2, vnextxp2, vl);
                                    grad = __riscv_vmacc_vx_i16m8(grad, 4, vprevxm1, vl);
                                    grad = __riscv_vmacc_vx_i16m8(grad, 6, vrowx, vl);
                                    grad = __riscv_vmacc_vx_i16m8(grad, 4, vnextxp1, vl);
                                }
                                else if (dx == 1)
                                {
                                    grad = __riscv_vsub_vv_i16m8(vnextxp2, vprevxm2, vl);
                                    grad = __riscv_vmacc_vx_i16m8(grad, -2, vprevxm1, vl);
                                    grad = __riscv_vmacc_vx_i16m8(grad, 2, vnextxp1, vl);
                                }
                                else
                                {
                                    grad = __riscv_vadd_vv_i16m8(vprevxm2, vnextxp2, vl);
                                    grad = __riscv_vmacc_vx_i16m8(grad, -2, vrowx, vl);
                                }

                                __riscv_vse16_v_i16m8(drow_current + x, grad, vl);
                            }
                        }
                    }

                    else
                    {
                        return CV_HAL_ERROR_NOT_IMPLEMENTED;
                    }
                }

                else if (dst_depth == CV_32F)
                {
                    if (ksize == 3)
                    {
                        if (border_type == BORDER_REPLICATE)
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
                                const uint8_t *srow1 = src_data + y * src_step;
                                const uint8_t *srow0 = src_data + (y > 0 ? y - 1 : 0) * src_step;
                                const uint8_t *srow2 = src_data + (y < height - 1 ? y + 1 : height - 1) * src_step;

                                float *drow0 = (float *)(dst_data + (y > 0 ? y - 1 : 0) * dst_step);
                                float *drow1 = (float *)(dst_data + y * dst_step);
                                float *trow = (y % 2) ? trow1 : trow0;

                                int x = 0;

                                size_t vl = __riscv_vsetvl_e8m2(width - 1);
                                float prevx = 0, rowx = 0, nextx = 0;
                                if (dy == 0)
                                {
                                    prevx = srow2[0] + 2 * srow1[0] + srow0[0];
                                    nextx = srow2[1] + 2 * srow1[1] + srow0[1];
                                }
                                else if (dy == 2)
                                {
                                    prevx = srow2[0] - 2 * srow1[0] + srow0[0];
                                    nextx = srow2[1] - 2 * srow1[1] + srow0[1];
                                }
                                else
                                {
                                    prevx = srow2[0] - srow0[0];
                                    nextx = srow2[1] - srow0[1];
                                }

                                rowx = prevx;
                                float res;
                                if (dx == 1)
                                    res = nextx - prevx;
                                else if (dx == 0)
                                    res = prevx + 2 * rowx + nextx;
                                else
                                    res = prevx - 2 * rowx + nextx;

                                trow[0] = res;

                                vfloat32m8_t vrowx = __riscv_vfmv_v_f_f32m8(rowx, vl);
                                vfloat32m8_t vprevx = vrowx;
                                vfloat32m8_t vnextx;

                                for (x = 0; x < width - 1; x += vl)
                                {
                                    vl = __riscv_vsetvl_e8m2(width - x - 1);

                                    vuint16m4_t vsrow0_u16 = __riscv_vwcvtu_x_x_v_u16m4(__riscv_vle8_v_u8m2(srow0 + x + 1, vl), vl);
                                    vuint16m4_t vsrow1_u16 = __riscv_vwcvtu_x_x_v_u16m4(__riscv_vle8_v_u8m2(srow1 + x + 1, vl), vl);
                                    vuint16m4_t vsrow2_u16 = __riscv_vwcvtu_x_x_v_u16m4(__riscv_vle8_v_u8m2(srow2 + x + 1, vl), vl);

                                    vfloat32m8_t vsrow0 = __riscv_vfwcvt_f_xu_v_f32m8(vsrow0_u16, vl);
                                    vfloat32m8_t vsrow1 = __riscv_vfwcvt_f_xu_v_f32m8(vsrow1_u16, vl);
                                    vfloat32m8_t vsrow2 = __riscv_vfwcvt_f_xu_v_f32m8(vsrow2_u16, vl);

                                    if (dy == 0)
                                    {
                                        vfloat32m8_t vsrow1_2x = __riscv_vfadd_vv_f32m8(vsrow1, vsrow1, vl);
                                        vnextx = __riscv_vfadd_vv_f32m8(__riscv_vfadd_vv_f32m8(vsrow2, vsrow0, vl), vsrow1_2x, vl);
                                    }
                                    else if (dy == 2)
                                    {
                                        vfloat32m8_t vsrow1_2x = __riscv_vfadd_vv_f32m8(vsrow1, vsrow1, vl);
                                        vnextx = __riscv_vfsub_vv_f32m8(__riscv_vfadd_vv_f32m8(vsrow2, vsrow0, vl), vsrow1_2x, vl);
                                    }
                                    else
                                    {
                                        vnextx = __riscv_vfsub_vv_f32m8(vsrow2, vsrow0, vl);
                                    }

                                    vrowx = __riscv_vslideup_vx_f32m8(vrowx, vnextx, 1, vl);
                                    vprevx = __riscv_vslideup_vx_f32m8(vprevx, vrowx, 1, vl);

                                    vfloat32m8_t grad;
                                    if (dx == 1)
                                        grad = __riscv_vfsub_vv_f32m8(vnextx, vprevx, vl);
                                    else if (dx == 0)
                                        grad = __riscv_vfadd_vv_f32m8(__riscv_vfadd_vv_f32m8(vprevx, vnextx, vl),
                                                                      __riscv_vfadd_vv_f32m8(vrowx, vrowx, vl), vl);
                                    else
                                        grad = __riscv_vfsub_vv_f32m8(__riscv_vfadd_vv_f32m8(vprevx, vnextx, vl),
                                                                      __riscv_vfadd_vv_f32m8(vrowx, vrowx, vl), vl);

                                    vprevx = vrowx;
                                    vrowx = vnextx;

                                    vrowx = __riscv_vslidedown_vx_f32m8(vrowx, vl - 1, vl);
                                    vprevx = __riscv_vslidedown_vx_f32m8(vprevx, vl - 1, vl);

                                    __riscv_vse32_v_f32m8(trow + x, grad, vl);
                                }

                                if (dy == 0)
                                {
                                    prevx = srow2[width - 2] + 2 * srow1[width - 2] + srow0[width - 2];
                                    rowx = srow2[width - 1] + 2 * srow1[width - 1] + srow0[width - 1];
                                }
                                else if (dy == 2)
                                {
                                    prevx = srow2[width - 2] - 2 * srow1[width - 2] + srow0[width - 2];
                                    rowx = srow2[width - 1] - 2 * srow1[width - 1] + srow0[width - 1];
                                }
                                else
                                {
                                    prevx = srow2[width - 2] - srow0[width - 2];
                                    rowx = srow2[width - 1] - srow0[width - 1];
                                }
                                nextx = rowx;

                                if (dx == 1)
                                    res = nextx - prevx;
                                else if (dx == 0)
                                    res = prevx + 2 * rowx + nextx;
                                else
                                    res = prevx - 2 * rowx + nextx;

                                trow[width - 1] = res;

                                if (y > 0)
                                {
                                    float *trow_res = (y % 2) ? trow0 : trow1;
                                    for (x = 0; x < width; x += vl)
                                    {
                                        vl = __riscv_vsetvl_e32m8(width - x);
                                        vfloat32m8_t vdata = __riscv_vle32_v_f32m8(trow_res + x, vl);
                                        __riscv_vse32_v_f32m8(drow0 + x, vdata, vl);
                                    }
                                }

                                if (y == height - 1)
                                {
                                    float *trow_res = (!(y % 2)) ? trow0 : trow1;
                                    for (x = 0; x < width; x += vl)
                                    {
                                        vl = __riscv_vsetvl_e32m8(width - x);
                                        vfloat32m8_t vdata = __riscv_vle32_v_f32m8(trow_res + x, vl);
                                        __riscv_vse32_v_f32m8(drow1 + x, vdata, vl);
                                    }
                                }
                            }

                            free(_tempBuf);
                        }

                        else if (border_type == BORDER_CONSTANT)
                        {
                            int align_size = (width + 2 + 7) & -8;
                            constexpr size_t alignment = 64;
                            size_t bufferSize = (align_size << 1) + alignment;
                            float *_tempBuf = static_cast<float *>(
                                aligned_alloc(alignment, bufferSize * sizeof(float)));

                            float *trow0 = reinterpret_cast<float *>(
                                (reinterpret_cast<uintptr_t>(_tempBuf) + alignment - 1) & ~(alignment - 1));
                            float *trow1 = trow0 + align_size;

                            vfloat32m8_t vborder = __riscv_vfmv_v_f_f32m8(borderValue, __riscv_vsetvlmax_e32m8());

                            for (int y = 0; y < height; y++)
                            {
                                const uint8_t *srow1 = src_data + y * src_step;
                                const uint8_t *srow0 = nullptr;
                                const uint8_t *srow2 = nullptr;

                                if (y > 0)
                                    srow0 = src_data + (y - 1) * src_step;
                                if (y < height - 1)
                                    srow2 = src_data + (y + 1) * src_step;

                                float *drow0 = (float *)(dst_data + (y > 0 ? y - 1 : 0) * dst_step);
                                float *drow1 = (float *)(dst_data + y * dst_step);
                                float *trow = (y % 2) ? trow1 : trow0;

                                size_t vl = __riscv_vsetvl_e8m2(width - 1);
                                float prevx = 0, rowx = 0, nextx = 0;

                                int x = 0;

                                if (dy == 0)
                                {
                                    float s0 = srow0 ? srow0[0] : borderValue;
                                    float s1 = srow1[0];
                                    float s2 = srow2 ? srow2[0] : borderValue;
                                    prevx = s2 + 2 * s1 + s0;

                                    s0 = srow0 ? srow0[1] : borderValue;
                                    s1 = srow1[1];
                                    s2 = srow2 ? srow2[1] : borderValue;
                                    nextx = s2 + 2 * s1 + s0;
                                }
                                else if (dy == 2)
                                {
                                    float s0 = srow0 ? srow0[0] : borderValue;
                                    float s1 = srow1[0];
                                    float s2 = srow2 ? srow2[0] : borderValue;
                                    prevx = s2 - 2 * s1 + s0;

                                    s0 = srow0 ? srow0[1] : borderValue;
                                    s1 = srow1[1];
                                    s2 = srow2 ? srow2[1] : borderValue;
                                    nextx = s2 - 2 * s1 + s0;
                                }
                                else
                                {
                                    float s0 = srow0 ? srow0[0] : borderValue;
                                    float s2 = srow2 ? srow2[0] : borderValue;
                                    prevx = s2 - s0;

                                    s0 = srow0 ? srow0[1] : borderValue;
                                    s2 = srow2 ? srow2[1] : borderValue;
                                    nextx = s2 - s0;
                                }

                                rowx = prevx;
                                float res;
                                if (dx == 1)
                                    res = nextx - prevx;
                                else if (dx == 0)
                                    res = prevx + 2 * rowx + nextx;
                                else
                                    res = prevx - 2 * rowx + nextx;

                                trow[0] = res;

                                vfloat32m8_t vrowx = __riscv_vfmv_v_f_f32m8(rowx, vl);
                                vfloat32m8_t vprevx = vrowx;
                                vfloat32m8_t vnextx;

                                for (x = 0; x < width - 1; x += vl)
                                {
                                    vl = __riscv_vsetvl_e8m2(width - x - 1);

                                    vuint16m4_t vsrow1_u16 = __riscv_vwcvtu_x_x_v_u16m4(
                                        __riscv_vle8_v_u8m2(srow1 + x + 1, vl), vl);
                                    vfloat32m8_t vsrow1 = __riscv_vfwcvt_f_xu_v_f32m8(vsrow1_u16, vl);

                                    vfloat32m8_t vsrow0, vsrow2;
                                    if (srow0)
                                    {
                                        vuint16m4_t vsrow0_u16 = __riscv_vwcvtu_x_x_v_u16m4(
                                            __riscv_vle8_v_u8m2(srow0 + x + 1, vl), vl);
                                        vsrow0 = __riscv_vfwcvt_f_xu_v_f32m8(vsrow0_u16, vl);
                                    }
                                    else
                                        vsrow0 = vborder;

                                    if (srow2)
                                    {
                                        vuint16m4_t vsrow2_u16 = __riscv_vwcvtu_x_x_v_u16m4(
                                            __riscv_vle8_v_u8m2(srow2 + x + 1, vl), vl);
                                        vsrow2 = __riscv_vfwcvt_f_xu_v_f32m8(vsrow2_u16, vl);
                                    }
                                    else
                                        vsrow2 = vborder;

                                    if (dy == 0)
                                    {
                                        vfloat32m8_t vsrow1_2x = __riscv_vfadd_vv_f32m8(vsrow1, vsrow1, vl);
                                        vnextx = __riscv_vfadd_vv_f32m8(
                                            __riscv_vfadd_vv_f32m8(vsrow2, vsrow0, vl), vsrow1_2x, vl);
                                    }
                                    else if (dy == 2)
                                    {
                                        vfloat32m8_t vsrow1_2x = __riscv_vfadd_vv_f32m8(vsrow1, vsrow1, vl);
                                        vnextx = __riscv_vfsub_vv_f32m8(
                                            __riscv_vfadd_vv_f32m8(vsrow2, vsrow0, vl), vsrow1_2x, vl);
                                    }
                                    else
                                    {
                                        vnextx = __riscv_vfsub_vv_f32m8(vsrow2, vsrow0, vl);
                                    }

                                    vrowx = __riscv_vslideup_vx_f32m8(vrowx, vnextx, 1, vl);
                                    vprevx = __riscv_vslideup_vx_f32m8(vprevx, vrowx, 1, vl);

                                    vfloat32m8_t grad;
                                    if (dx == 1)
                                        grad = __riscv_vfsub_vv_f32m8(vnextx, vprevx, vl);
                                    else if (dx == 0)
                                        grad = __riscv_vfadd_vv_f32m8(
                                            __riscv_vfadd_vv_f32m8(vprevx, vnextx, vl),
                                            __riscv_vfadd_vv_f32m8(vrowx, vrowx, vl), vl);
                                    else
                                        grad = __riscv_vfsub_vv_f32m8(
                                            __riscv_vfadd_vv_f32m8(vprevx, vnextx, vl),
                                            __riscv_vfadd_vv_f32m8(vrowx, vrowx, vl), vl);

                                    vprevx = vrowx;
                                    vrowx = vnextx;

                                    vrowx = __riscv_vslidedown_vx_f32m8(vrowx, vl - 1, vl);
                                    vprevx = __riscv_vslidedown_vx_f32m8(vprevx, vl - 1, vl);

                                    __riscv_vse32_v_f32m8(trow + x, grad, vl);
                                }

                                if (dy == 0)
                                {
                                    float s0 = srow0 ? srow0[width - 1] : borderValue;
                                    float s1 = srow1[width - 1];
                                    float s2 = srow2 ? srow2[width - 1] : borderValue;
                                    rowx = s2 + 2 * s1 + s0;
                                }
                                else if (dy == 2)
                                {
                                    float s0 = srow0 ? srow0[width - 1] : borderValue;
                                    float s1 = srow1[width - 1];
                                    float s2 = srow2 ? srow2[width - 1] : borderValue;
                                    rowx = s2 - 2 * s1 + s0;
                                }
                                else
                                {
                                    float s0 = srow0 ? srow0[width - 1] : borderValue;
                                    float s2 = srow2 ? srow2[width - 1] : borderValue;
                                    rowx = s2 - s0;
                                }
                                nextx = rowx;

                                if (dx == 1)
                                    res = nextx - prevx;
                                else if (dx == 0)
                                    res = prevx + 2 * rowx + nextx;
                                else
                                    res = prevx - 2 * rowx + nextx;

                                trow[width - 1] = res;

                                if (y > 0)
                                {
                                    float *trow_res = (y % 2) ? trow0 : trow1;
                                    for (x = 0; x < width; x += vl)
                                    {
                                        vl = __riscv_vsetvl_e32m8(width - x);
                                        vfloat32m8_t vdata = __riscv_vle32_v_f32m8(trow_res + x, vl);
                                        __riscv_vse32_v_f32m8(drow0 + x, vdata, vl);
                                    }
                                }

                                if (y == height - 1)
                                {
                                    float *trow_res = (!(y % 2)) ? trow0 : trow1;
                                    for (x = 0; x < width; x += vl)
                                    {
                                        vl = __riscv_vsetvl_e32m8(width - x);
                                        vfloat32m8_t vdata = __riscv_vle32_v_f32m8(trow_res + x, vl);
                                        __riscv_vse32_v_f32m8(drow1 + x, vdata, vl);
                                    }
                                }
                            }

                            free(_tempBuf);
                        }

                        else
                        {
                            return CV_HAL_ERROR_NOT_IMPLEMENTED;
                        }
                    }

                    else if (ksize == 5)
                    {
                        if (border_type != BORDER_REPLICATE &&
                            border_type != BORDER_CONSTANT &&
                            border_type != BORDER_REFLECT &&
                            border_type != BORDER_REFLECT_101)
                        {
                            return CV_HAL_ERROR_NOT_IMPLEMENTED;
                        }
                        int align_size = (width + 4 + 7) & -8;
                        constexpr size_t alignment = 64;
                        size_t bufferSize = (align_size * 3) + alignment;
                        float *_tempBuf = static_cast<float *>(
                            aligned_alloc(alignment, bufferSize * sizeof(float)));

                        float *trow0 = reinterpret_cast<float *>(
                            (reinterpret_cast<uintptr_t>(_tempBuf) + alignment - 1) & ~(alignment - 1));
                        float *trow1 = trow0 + align_size;
                        float *trow2 = trow1 + align_size;

                        const uint8_t *srowm2, *srowm1, *srow0, *srowp1, *srowp2;
                        float *border = (float *)malloc((4 * height) * sizeof(float));
                        for (int y = 0; y < height; y++)
                        {
                            if (border_type == BORDER_REPLICATE)
                            {
                                srowm2 = src_data + (y > 2 ? y - 2 : 0) * src_step;
                                srowm1 = src_data + (y > 1 ? y - 1 : 0) * src_step;
                                srow0 = src_data + y * src_step;
                                srowp1 = src_data + (y + 1 < height - 1 ? y + 1 : height - 1) * src_step;
                                srowp2 = src_data + (y + 2 < height - 1 ? y + 2 : height - 1) * src_step;
                            }
                            else if (border_type == BORDER_CONSTANT)
                            {
                                uint8_t *zero_row = (uint8_t *)malloc(width * sizeof(uint8_t));
                                memset(zero_row, 0, width * sizeof(uint8_t));
                                srowm2 = y > 2 ? (src_data + (y - 2) * src_step) : zero_row;
                                srowm1 = y > 1 ? (src_data + (y - 1) * src_step) : zero_row;
                                srow0 = src_data + y * src_step;
                                srowp1 = y + 1 < height - 1 ? (src_data + (y + 1) * src_step) : zero_row;
                                srowp2 = y + 2 < height - 1 ? (src_data + (y + 2) * src_step) : zero_row;
                            }
                            else if (border_type == BORDER_REFLECT)
                            {
                                srowm2 = y >= 2 ? (src_data + (y - 2) * src_step) : (y == 1 ? src_data : (src_data + src_step));
                                srowm1 = y >= 1 ? (src_data + (y - 1) * src_step) : src_data;
                                srow0 = src_data + y * src_step;
                                srowp1 = y + 1 <= height - 1 ? (src_data + (y + 1) * src_step) : src_data + y * src_step;
                                srowp2 = y + 2 <= height - 1 ? (src_data + (y + 2) * src_step) : (y == height - 2 ? (src_data + (y - 1) * src_step) : (src_data + y * src_step));
                            }
                            else if (border_type == BORDER_REFLECT_101)
                            {
                                srowm2 = y > 2 ? (src_data + (y - 2) * src_step) : (src_data + 2 * src_step);
                                srowm1 = y >= 1 ? (src_data + (y - 1) * src_step) : (src_data + src_step);
                                srow0 = src_data + y * src_step;
                                srowp1 = y + 1 <= height - 1 ? (src_data + (y + 1) * src_step) : (src_data + (y - 1) * src_step);
                                srowp2 = y + 2 <= height - 1 ? (src_data + (y + 2) * src_step) : (y == height - 2 ? (src_data + (y - 2) * src_step) : (src_data + y * src_step));
                            }

                            float *trow;
                            if (y % 3 == 0)
                            {
                                trow = trow0;
                            }
                            else if (y % 3 == 1)
                            {
                                trow = trow1;
                            }
                            else
                            {
                                trow = trow2;
                            }

                            float prevxm2 = 0, prevxm1 = 0, rowx = 0, nextxp1 = 0, nextxp2 = 0;

                            int x = 0;
                            if (dy == 0)
                            {
                                rowx = srowm2[0] + 4 * srowm1[0] + 6 * srow0[0] + 4 * srowp1[0] + srowp2[0];
                                nextxp1 = srowm2[1] + 4 * srowm1[1] + 6 * srow0[1] + 4 * srowp1[1] + srowp2[1];
                                nextxp2 = srowm2[2] + 4 * srowm1[2] + 6 * srow0[2] + 4 * srowp1[2] + srowp2[2];
                            }
                            else if (dy == 1)
                            {
                                rowx = -1 * srowm2[0] - 2 * srowm1[0] + 0 * srow0[0] + 2 * srowp1[0] + srowp2[0];
                                nextxp1 = -1 * srowm2[1] - 2 * srowm1[1] + 0 * srow0[1] + 2 * srowp1[1] + srowp2[1];
                                nextxp2 = -1 * srowm2[2] - 2 * srowm1[2] + 0 * srow0[2] + 2 * srowp1[2] + srowp2[2];
                            }
                            else
                            {
                                rowx = srowm2[0] + 0 * srowm1[0] - 2 * srow0[0] + 0 * srowp1[0] + srowp2[0];
                                nextxp1 = srowm2[1] + 0 * srowm1[1] - 2 * srow0[1] + 0 * srowp1[1] + srowp2[1];
                                nextxp2 = srowm2[2] + 0 * srowm1[2] - 2 * srow0[2] + 0 * srowp1[2] + srowp2[2];
                            }
                            if (border_type == BORDER_REPLICATE)
                            {
                                prevxm1 = rowx;
                                prevxm2 = prevxm1;
                            }
                            // border_type == BORDER_CONSTANT : prevxm1 = borderValue, prevxm2 = borderValue;
                            else if (border_type == BORDER_REFLECT)
                            {
                                prevxm1 = rowx;
                                prevxm2 = nextxp1;
                            }
                            else if (border_type == BORDER_REFLECT_101)
                            {
                                prevxm1 = nextxp1;
                                prevxm2 = nextxp2;
                            }

                            float res = 0;
                            if (dx == 0)
                            {
                                res = prevxm2 + 4 * prevxm1 + 6 * rowx + 4 * nextxp1 + nextxp2;
                            }
                            else if (dx == 1)
                            {
                                res = -1 * prevxm2 - 2 * prevxm1 + 0 * rowx + 2 * nextxp1 + nextxp2;
                            }
                            else
                            {
                                res = prevxm2 + 0 * prevxm1 - 2 * rowx + 0 * nextxp1 + nextxp2;
                            }
                            border[y * 4] = res;
                            trow[x] = rowx;

                            x = 1;
                            if (border_type == BORDER_REPLICATE)
                            {
                                rowx = nextxp1;
                                nextxp1 = nextxp2;
                            }
                            else if (border_type == BORDER_CONSTANT)
                            {
                                prevxm1 = rowx;
                                rowx = nextxp1;
                                nextxp1 = nextxp2;
                            }
                            else if (border_type == BORDER_REFLECT)
                            {
                                prevxm1 = rowx;
                                prevxm2 = prevxm1;
                                rowx = nextxp1;
                                nextxp1 = nextxp2;
                            }
                            else if (border_type == BORDER_REFLECT_101)
                            {
                                prevxm1 = rowx;
                                rowx = nextxp1;
                                nextxp1 = nextxp2;
                                prevxm2 = rowx;
                            }

                            if (dy == 0)
                            {
                                nextxp2 = srowm2[3] + 4 * srowm1[3] + 6 * srow0[3] + 4 * srowp1[3] + srowp2[3];
                            }
                            else if (dy == 1)
                            {
                                nextxp2 = -1 * srowm2[3] - 2 * srowm1[3] + 0 * srow0[3] + 2 * srowp1[3] + srowp2[3];
                            }
                            else
                            {
                                nextxp2 = srowm2[3] + 0 * srowm1[3] - 2 * srow0[3] + 0 * srowp1[3] + srowp2[3];
                            }

                            if (dx == 0)
                            {
                                res = prevxm2 + 4 * prevxm1 + 6 * rowx + 4 * nextxp1 + nextxp2;
                            }
                            else if (dx == 1)
                            {
                                res = -1 * prevxm2 - 2 * prevxm1 + 0 * rowx + 2 * nextxp1 + nextxp2;
                            }
                            else
                            {
                                res = prevxm2 + 0 * prevxm1 - 2 * rowx + 0 * nextxp1 + nextxp2;
                            }

                            border[y * 4 + 1] = res;
                            trow[x] = rowx;

                            x = 2;
                            size_t vl = 0;
                            vfloat32m8_t vsrowm2, vsrowm1, vsrow0, vsrowp1, vsrowp2;
                            vfloat32m8_t vrowx;

                            for (; x < width - 2; x += vl)
                            {
                                vl = __riscv_vsetvl_e8m2(width - 2 - x);

                                vsrowm2 = __riscv_vfwcvt_f_xu_v_f32m8(__riscv_vwcvtu_x_x_v_u16m4(__riscv_vle8_v_u8m2(srowm2 + x, vl), vl), vl);
                                vsrowm1 = __riscv_vfwcvt_f_xu_v_f32m8(__riscv_vwcvtu_x_x_v_u16m4(__riscv_vle8_v_u8m2(srowm1 + x, vl), vl), vl);
                                vsrow0 = __riscv_vfwcvt_f_xu_v_f32m8(__riscv_vwcvtu_x_x_v_u16m4(__riscv_vle8_v_u8m2(srow0 + x, vl), vl), vl);
                                vsrowp1 = __riscv_vfwcvt_f_xu_v_f32m8(__riscv_vwcvtu_x_x_v_u16m4(__riscv_vle8_v_u8m2(srowp1 + x, vl), vl), vl);
                                vsrowp2 = __riscv_vfwcvt_f_xu_v_f32m8(__riscv_vwcvtu_x_x_v_u16m4(__riscv_vle8_v_u8m2(srowp2 + x, vl), vl), vl);

                                if (dy == 0)
                                {
                                    vrowx = __riscv_vfadd_vv_f32m8(vsrowm2, vsrowp2, vl);
                                    vrowx = __riscv_vfmacc_vf_f32m8(vrowx, 4, vsrowm1, vl);
                                    vrowx = __riscv_vfmacc_vf_f32m8(vrowx, 6, vsrow0, vl);
                                    vrowx = __riscv_vfmacc_vf_f32m8(vrowx, 4, vsrowp1, vl);
                                }
                                else if (dy == 1)
                                {
                                    vrowx = __riscv_vfsub_vv_f32m8(vsrowp2, vsrowm2, vl);
                                    vrowx = __riscv_vfmacc_vf_f32m8(vrowx, -2, vsrowm1, vl);
                                    vrowx = __riscv_vfmacc_vf_f32m8(vrowx, 2, vsrowp1, vl);
                                }
                                else
                                {
                                    vrowx = __riscv_vfadd_vv_f32m8(vsrowm2, vsrowp2, vl);
                                    vrowx = __riscv_vfmacc_vf_f32m8(vrowx, -2, vsrow0, vl);
                                }

                                __riscv_vse32_v_f32m8(trow + x, vrowx, vl);
                            }

                            x = width - 2;
                            if (dy == 0)
                            {
                                prevxm2 = srowm2[x - 2] + 4 * srowm1[x - 2] + 6 * srow0[x - 2] + 4 * srowp1[x - 2] + srowp2[x - 2];
                                prevxm1 = srowm2[x - 1] + 4 * srowm1[x - 1] + 6 * srow0[x - 1] + 4 * srowp1[x - 1] + srowp2[x - 1];
                                rowx = srowm2[x] + 4 * srowm1[x] + 6 * srow0[x] + 4 * srowp1[x] + srowp2[x];
                                nextxp1 = srowm2[x + 1] + 4 * srowm1[x + 1] + 6 * srow0[x + 1] + 4 * srowp1[x + 1] + srowp2[x + 1];
                            }
                            else if (dy == 1)
                            {
                                prevxm2 = -1 * srowm2[x - 2] - 2 * srowm1[x - 2] + 0 * srow0[x - 2] + 2 * srowp1[x - 2] + srowp2[x - 2];
                                prevxm1 = -1 * srowm2[x - 1] - 2 * srowm1[x - 1] + 0 * srow0[x - 1] + 2 * srowp1[x - 1] + srowp2[x - 1];
                                rowx = -1 * srowm2[x] - 2 * srowm1[x] + 0 * srow0[x] + 2 * srowp1[x] + srowp2[x];
                                nextxp1 = -1 * srowm2[x + 1] - 2 * srowm1[x + 1] + 0 * srow0[x + 1] + 2 * srowp1[x + 1] + srowp2[x + 1];
                            }
                            else
                            {
                                prevxm2 = srowm2[x - 2] + 0 * srowm1[x - 2] - 2 * srow0[x - 2] + 0 * srowp1[x - 2] + srowp2[x - 2];
                                prevxm1 = srowm2[x - 1] + 0 * srowm1[x - 1] - 2 * srow0[x - 1] + 0 * srowp1[x - 1] + srowp2[x - 1];
                                rowx = srowm2[x] + 0 * srowm1[x] - 2 * srow0[x] + 0 * srowp1[x] + srowp2[x];
                                nextxp1 = srowm2[x + 1] + 0 * srowm1[x + 1] - 2 * srow0[x + 1] + 0 * srowp1[x + 1] + srowp2[x + 1];
                            }
                            if (border_type == BORDER_REPLICATE)
                            {
                                nextxp2 = nextxp1;
                            }
                            else if (border_type == BORDER_CONSTANT)
                            {
                                nextxp2 = borderValue;
                            }
                            else if (border_type == BORDER_REFLECT)
                            {
                                nextxp2 = nextxp1;
                            }
                            else if (border_type == BORDER_REFLECT_101)
                            {
                                nextxp2 = rowx;
                            }

                            if (dx == 0)
                            {
                                res = prevxm2 + 4 * prevxm1 + 6 * rowx + 4 * nextxp1 + nextxp2;
                            }
                            else if (dx == 1)
                            {
                                res = -1 * prevxm2 - 2 * prevxm1 + 0 * rowx + 2 * nextxp1 + nextxp2;
                            }
                            else
                            {
                                res = prevxm2 + 0 * prevxm1 - 2 * rowx + 0 * nextxp1 + nextxp2;
                            }

                            border[y * 4 + 2] = res;
                            trow[x] = rowx;

                            x = width - 1;
                            if (border_type == BORDER_REPLICATE)
                            {
                                prevxm2 = prevxm1;
                                prevxm1 = rowx;
                                rowx = nextxp1;
                                nextxp1 = nextxp2;
                            }
                            else if (border_type == BORDER_CONSTANT)
                            {
                                prevxm2 = prevxm1;
                                prevxm1 = rowx;
                                rowx = nextxp1;
                                nextxp1 = borderValue;
                            }
                            else if (border_type == BORDER_REFLECT)
                            {
                                prevxm2 = prevxm1;
                                prevxm1 = rowx;
                                rowx = nextxp1;
                                nextxp1 = rowx;
                                nextxp2 = prevxm1;
                            }
                            else if (border_type == BORDER_REFLECT_101)
                            {
                                prevxm2 = prevxm1;
                                prevxm1 = rowx;
                                rowx = nextxp1;
                                nextxp1 = prevxm1;
                                nextxp2 = prevxm2;
                            }

                            if (dx == 0)
                            {
                                res = prevxm2 + 4 * prevxm1 + 6 * rowx + 4 * nextxp1 + nextxp2;
                            }
                            else if (dx == 1)
                            {
                                res = -1 * prevxm2 - 2 * prevxm1 + 0 * rowx + 2 * nextxp1 + nextxp2;
                            }
                            else
                            {
                                res = prevxm2 + 0 * prevxm1 - 2 * rowx + 0 * nextxp1 + nextxp2;
                            }

                            border[y * 4 + 3] = res;
                            trow[x] = rowx;

                            if (y > 1)
                            {
                                int target_y = y - 2;
                                float *target_trow;
                                if (target_y % 3 == 0)
                                    target_trow = trow0;
                                else if (target_y % 3 == 1)
                                    target_trow = trow1;
                                else
                                    target_trow = trow2;

                                float *drow_target = (float *)(dst_data + target_y * dst_step);

                                drow_target[0] = border[target_y * 4];
                                drow_target[1] = border[target_y * 4 + 1];
                                drow_target[width - 2] = border[target_y * 4 + 2];
                                drow_target[width - 1] = border[target_y * 4 + 3];

                                for (x = 2; x < width - 2; x += vl)
                                {
                                    vl = __riscv_vsetvl_e32m8(width - 2 - x);
                                    vfloat32m8_t vprevxm2 = __riscv_vle32_v_f32m8(target_trow + x - 2, vl);
                                    vfloat32m8_t vprevxm1 = __riscv_vle32_v_f32m8(target_trow + x - 1, vl);
                                    vfloat32m8_t vrowx = __riscv_vle32_v_f32m8(target_trow + x, vl);
                                    vfloat32m8_t vnextxp1 = __riscv_vle32_v_f32m8(target_trow + x + 1, vl);
                                    vfloat32m8_t vnextxp2 = __riscv_vle32_v_f32m8(target_trow + x + 2, vl);

                                    vfloat32m8_t grad;
                                    if (dx == 0)
                                    {
                                        grad = __riscv_vfadd_vv_f32m8(vprevxm2, vnextxp2, vl);
                                        grad = __riscv_vfmacc_vf_f32m8(grad, 4, vprevxm1, vl);
                                        grad = __riscv_vfmacc_vf_f32m8(grad, 6, vrowx, vl);
                                        grad = __riscv_vfmacc_vf_f32m8(grad, 4, vnextxp1, vl);
                                    }
                                    else if (dx == 1)
                                    {
                                        grad = __riscv_vfsub_vv_f32m8(vnextxp2, vprevxm2, vl);
                                        grad = __riscv_vfmacc_vf_f32m8(grad, -2, vprevxm1, vl);
                                        grad = __riscv_vfmacc_vf_f32m8(grad, 2, vnextxp1, vl);
                                    }
                                    else
                                    {
                                        grad = __riscv_vfadd_vv_f32m8(vprevxm2, vnextxp2, vl);
                                        grad = __riscv_vfmacc_vf_f32m8(grad, -2, vrowx, vl);
                                    }

                                    __riscv_vse32_v_f32m8(drow_target + x, grad, vl);
                                }
                            }
                        }

                        for (int remaining_y = (height >= 2 ? height - 2 : 0); remaining_y < height; remaining_y++)
                        {
                            float *current_trow;
                            if (remaining_y % 3 == 0)
                                current_trow = trow0;
                            else if (remaining_y % 3 == 1)
                                current_trow = trow1;
                            else
                                current_trow = trow2;

                            float *drow_current = (float *)(dst_data + remaining_y * dst_step);

                            drow_current[0] = border[remaining_y * 4];
                            drow_current[1] = border[remaining_y * 4 + 1];
                            drow_current[width - 2] = border[remaining_y * 4 + 2];
                            drow_current[width - 1] = border[remaining_y * 4 + 3];

                            size_t vl = 0;
                            int x = 0;
                            for (x = 2; x < width - 2; x += vl)
                            {
                                vl = __riscv_vsetvl_e32m8(width - 2 - x);
                                vfloat32m8_t vprevxm2 = __riscv_vle32_v_f32m8(current_trow + x - 2, vl);
                                vfloat32m8_t vprevxm1 = __riscv_vle32_v_f32m8(current_trow + x - 1, vl);
                                vfloat32m8_t vrowx = __riscv_vle32_v_f32m8(current_trow + x, vl);
                                vfloat32m8_t vnextxp1 = __riscv_vle32_v_f32m8(current_trow + x + 1, vl);
                                vfloat32m8_t vnextxp2 = __riscv_vle32_v_f32m8(current_trow + x + 2, vl);

                                vfloat32m8_t grad;
                                if (dx == 0)
                                {
                                    grad = __riscv_vfadd_vv_f32m8(vprevxm2, vnextxp2, vl);
                                    grad = __riscv_vfmacc_vf_f32m8(grad, 4, vprevxm1, vl);
                                    grad = __riscv_vfmacc_vf_f32m8(grad, 6, vrowx, vl);
                                    grad = __riscv_vfmacc_vf_f32m8(grad, 4, vnextxp1, vl);
                                }
                                else if (dx == 1)
                                {
                                    grad = __riscv_vfsub_vv_f32m8(vnextxp2, vprevxm2, vl);
                                    grad = __riscv_vfmacc_vf_f32m8(grad, -2, vprevxm1, vl);
                                    grad = __riscv_vfmacc_vf_f32m8(grad, 2, vnextxp1, vl);
                                }
                                else
                                {
                                    grad = __riscv_vfadd_vv_f32m8(vprevxm2, vnextxp2, vl);
                                    grad = __riscv_vfmacc_vf_f32m8(grad, -2, vrowx, vl);
                                }

                                __riscv_vse32_v_f32m8(drow_current + x, grad, vl);
                            }
                        }
                    }

                    else
                    {
                        return CV_HAL_ERROR_NOT_IMPLEMENTED;
                    }
                }

                return CV_HAL_ERROR_OK;
            }

#endif // CV_HAL_RVV_1P0_ENABLED

        }
    }
} // cv::rvv_hal::imgproc
