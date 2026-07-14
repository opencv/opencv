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
            int sobel_3X3(int start, int end, const uint8_t *src_data, size_t src_step, uint8_t *dst_data, size_t dst_step, int width, int height, int src_depth __attribute__((unused)), int dst_depth, int cn __attribute__((unused)), int margin_left __attribute__((unused)), int margin_top __attribute__((unused)), int margin_right __attribute__((unused)), int margin_bottom __attribute__((unused)), int dx, int dy, double scale __attribute__((unused)), double delta __attribute__((unused)), int border_type);

            int sobel_3X3(int start, int end, const uint8_t *src_data, size_t src_step, uint8_t *dst_data, size_t dst_step, int width, int height, int src_depth __attribute__((unused)), int dst_depth, int cn __attribute__((unused)), int margin_left __attribute__((unused)), int margin_top __attribute__((unused)), int margin_right __attribute__((unused)), int margin_bottom __attribute__((unused)), int dx, int dy, double scale __attribute__((unused)), double delta __attribute__((unused)), int border_type)
            {
                if (border_type != BORDER_REPLICATE && border_type != BORDER_CONSTANT)
                    return CV_HAL_ERROR_NOT_IMPLEMENTED;

                int dx0, dx1, dx2, dy1;

                if (dx == 0)
                {
                    dx0 = 1;
                    dx1 = 2;
                    dx2 = 1;
                }
                else if (dx == 1)
                {
                    dx0 = -1;
                    dx1 = 0;
                    dx2 = 1;
                }
                else if (dx == 2)
                {
                    dx0 = 1;
                    dx1 = -2;
                    dx2 = 1;
                }
                else
                {
                    dx0 = 1;
                    dx1 = 1;
                    dx2 = 1;

                }

                if (dy == 0)
                {
                    dy1 = 2;
                }
                else if (dy == 1)
                {
                    dy1 = 0;
                }
                else if (dy == 2)
                {
                    dy1 = -2;
                }
                else
                {
                    dy1 = 0;
                }

                // int16_t trow[width];

                std::vector<int16_t> trow_buf(width);
                int16_t *trow = trow_buf.data();

                static thread_local std::vector<uint8_t> zero_row;
                if (zero_row.size() < static_cast<size_t>(width))
                {
                    zero_row.clear();
                    zero_row.resize(width, 0);
                }

                for (int y = start; y < end; ++y)
                {
                    const uint8_t *srow1 = src_data + y * src_step;
                    const uint8_t *srow0;
                    const uint8_t *srow2;

                    if (y != 0)
                    {
                        srow0 = src_data + (y - 1) * src_step;
                    }
                    else
                    {
                        if (border_type == BORDER_REPLICATE)
                        {
                            srow0 = srow1;
                        }
                        else
                        {
                            srow0 = zero_row.data();
                        }
                    }

                    if (y != height - 1)
                    {
                        srow2 = src_data + (y + 1) * src_step;
                    }
                    else
                    {
                        if (border_type == BORDER_REPLICATE)
                        {
                            srow2 = srow1;
                        }
                        else
                        {
                            srow2 = zero_row.data();
                        }
                    }

                    size_t vl = 0;

                    // Vector processing

                    if (dy == 1)
                    {
                        for (int x = 0; x < width; x += vl)
                        {
                            vl = __riscv_vsetvl_e8m4(width - x);

                            vint16m8_t vsrow0 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow0 + x, vl), vl));
                            vint16m8_t vsrow2 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow2 + x, vl), vl));

                            vint16m8_t vrowx = __riscv_vsub_vv_i16m8(vsrow2, vsrow0, vl);

                            __riscv_vse16_v_i16m8(trow + x, vrowx, vl);
                        }
                    }
                    else
                    {
                        for (int x = 0; x < width; x += vl)
                        {
                            vl = __riscv_vsetvl_e8m4(width - x);

                            vint16m8_t vsrow0 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow0 + x, vl), vl));
                            vint16m8_t vsrow1 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow1 + x, vl), vl));
                            vint16m8_t vsrow2 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow2 + x, vl), vl));

                            vint16m8_t vrowx = __riscv_vadd_vv_i16m8(vsrow0, vsrow2, vl);
                            vrowx = __riscv_vmacc_vx_i16m8(vrowx, dy1, vsrow1, vl);

                            __riscv_vse16_v_i16m8(trow + x, vrowx, vl);
                        }
                    }

                    if (dst_depth == CV_8U)
                    {
                        int x = 0;

                        uint8_t *drow = (uint8_t *)(dst_data + y * dst_step);

                        int16_t temp = trow[0] * dx1 + trow[1] * dx2;

                        if (border_type == BORDER_REPLICATE)
                        {
                            temp += trow[0] * dx0;
                        }

                        drow[x] = (uint8_t)(temp > 0 ? (temp < 255 ? temp : 255) : 0);

                        x = 1;

                        if (dx == 1)
                        {
                            for (; x < width - 1; x += vl)
                            {
                                vl = __riscv_vsetvl_e16m8(width - x - 1);
                                vint16m8_t vleft = __riscv_vle16_v_i16m8(trow + x - 1, vl);
                                vint16m8_t vright = __riscv_vle16_v_i16m8(trow + x + 1, vl);

                                vint16m8_t vres = __riscv_vsub_vv_i16m8(vright, vleft, vl);

                                vres = __riscv_vmax_vx_i16m8(vres, 0, vl);
                                vres = __riscv_vmin_vx_i16m8(vres, 255, vl);

                                __riscv_vse8_v_u8m4(drow + x, __riscv_vncvt_x_x_w_u8m4(__riscv_vreinterpret_v_i16m8_u16m8(vres), vl), vl);
                            }
                        }
                        else
                        {
                            for (; x < width - 1; x += vl)
                            {
                                vl = __riscv_vsetvl_e16m8(width - x - 1);
                                vint16m8_t vleft = __riscv_vle16_v_i16m8(trow + x - 1, vl);
                                vint16m8_t vcenter = __riscv_vle16_v_i16m8(trow + x, vl);
                                vint16m8_t vright = __riscv_vle16_v_i16m8(trow + x + 1, vl);

                                vint16m8_t vres = __riscv_vadd_vv_i16m8(vleft, vright, vl);
                                vres = __riscv_vmacc_vx_i16m8(vres, dx1, vcenter, vl);

                                vres = __riscv_vmax_vx_i16m8(vres, 0, vl);
                                vres = __riscv_vmin_vx_i16m8(vres, 255, vl);

                                __riscv_vse8_v_u8m4(drow + x, __riscv_vncvt_x_x_w_u8m4(__riscv_vreinterpret_v_i16m8_u16m8(vres), vl), vl);
                            }
                        }

                        x = width - 1;

                        temp = trow[width - 2] * dx0 + trow[width - 1] * dx1;

                        if (border_type == BORDER_REPLICATE)
                        {
                            temp += trow[width - 1] * dx2;
                        }

                        drow[x] = (uint8_t)(temp > 0 ? (temp < 255 ? temp : 255) : 0);
                    }

                    else if (dst_depth == CV_16S)
                    {
                        int x = 0;

                        int16_t *drow = (int16_t *)(dst_data + y * dst_step);

                        drow[0] = trow[0] * dx1 + trow[1] * dx2;

                        if (border_type == BORDER_REPLICATE)
                        {
                            drow[x] += trow[0] * dx0;
                        }

                        x = 1;

                        if (dx == 1)
                        {
                            for (; x < width - 1; x += vl)
                            {
                                vl = __riscv_vsetvl_e16m8(width - x - 1);
                                vint16m8_t vleft = __riscv_vle16_v_i16m8(trow + x - 1, vl);
                                vint16m8_t vright = __riscv_vle16_v_i16m8(trow + x + 1, vl);

                                vint16m8_t vres = __riscv_vsub_vv_i16m8(vright, vleft, vl);

                                __riscv_vse16_v_i16m8(drow + x, vres, vl);
                            }
                        }
                        else
                        {
                            for (; x < width - 1; x += vl)
                            {
                                vl = __riscv_vsetvl_e16m8(width - x - 1);
                                vint16m8_t vleft = __riscv_vle16_v_i16m8(trow + x - 1, vl);
                                vint16m8_t vcenter = __riscv_vle16_v_i16m8(trow + x, vl);
                                vint16m8_t vright = __riscv_vle16_v_i16m8(trow + x + 1, vl);

                                vint16m8_t vres = __riscv_vadd_vv_i16m8(vleft, vright, vl);
                                vres = __riscv_vmacc_vx_i16m8(vres, dx1, vcenter, vl);

                                __riscv_vse16_v_i16m8(drow + x, vres, vl);
                            }
                        }

                        x = width - 1;

                        drow[x] = trow[width - 2] * dx0 + trow[width - 1] * dx1;

                        if (border_type == BORDER_REPLICATE)
                        {
                            drow[x] += trow[width - 1] * dx2;
                        }
                    }

                    else if (dst_depth == CV_32F)
                    {
                        int x = 0;

                        float *drow = (float *)(dst_data + y * dst_step);

                        int temp = trow[0] * dx1 + trow[1] * dx2;

                        if (border_type == BORDER_REPLICATE)
                        {
                            temp += trow[0] * dx0;
                        }

                        drow[0] = (float)temp;

                        x = 1;

                        if (dx == 1)
                        {
                            for (; x < width - 1; x += vl)
                            {
                                vl = __riscv_vsetvl_e16m4(width - x - 1);
                                vint16m4_t vleft = __riscv_vle16_v_i16m4(trow + x - 1, vl);
                                vint16m4_t vright = __riscv_vle16_v_i16m4(trow + x + 1, vl);

                                vint16m4_t vres = __riscv_vsub_vv_i16m4(vright, vleft, vl);

                                __riscv_vse32_v_f32m8(drow + x, __riscv_vfwcvt_f_x_v_f32m8(vres, vl), vl);
                            }
                        }
                        else
                        {
                            for (; x < width - 1; x += vl)
                            {
                                vl = __riscv_vsetvl_e16m4(width - x - 1);
                                vint16m4_t vleft = __riscv_vle16_v_i16m4(trow + x - 1, vl);
                                vint16m4_t vcenter = __riscv_vle16_v_i16m4(trow + x, vl);
                                vint16m4_t vright = __riscv_vle16_v_i16m4(trow + x + 1, vl);

                                vint16m4_t vres = __riscv_vadd_vv_i16m4(vleft, vright, vl);
                                vres = __riscv_vmacc_vx_i16m4(vres, dx1, vcenter, vl);

                                __riscv_vse32_v_f32m8(drow + x, __riscv_vfwcvt_f_x_v_f32m8(vres, vl), vl);
                            }
                        }

                        x = width - 1;

                        drow[x] = trow[width - 2] * dx0 + trow[width - 1] * dx1;

                        if (border_type == BORDER_REPLICATE)
                        {
                            drow[x] += trow[width - 1] * dx2;
                        }
                    }
                }

                return CV_HAL_ERROR_OK;
            }
            int sobel_5X5(int start, int end, const uint8_t *src_data, size_t src_step, uint8_t *dst_data, size_t dst_step, int width, int height, int src_depth, int dst_depth, int cn, int margin_left, int margin_top, int margin_right, int margin_bottom, int dx, int dy, double scale, double delta, int border_type);
            int sobel_5X5(int start, int end, const uint8_t *src_data, size_t src_step, uint8_t *dst_data, size_t dst_step, int width, int height, int src_depth __attribute__((unused)), int dst_depth, int cn __attribute__((unused)), int margin_left __attribute__((unused)), int margin_top __attribute__((unused)), int margin_right __attribute__((unused)), int margin_bottom __attribute__((unused)), int dx, int dy, double scale __attribute__((unused)), double delta __attribute__((unused)), int border_type)
            {
                if (border_type != BORDER_REPLICATE && border_type != BORDER_CONSTANT && border_type != BORDER_REFLECT && border_type != BORDER_REFLECT_101)
                    return CV_HAL_ERROR_NOT_IMPLEMENTED;

                int dx0, dx1, dx2, dx3, dx4,  dy1, dy2, dy3;

                if (dx == 0)
                {
                    dx0 = 1;
                    dx1 = 4;
                    dx2 = 6;
                    dx3 = 4;
                    dx4 = 1;
                }
                else if (dx == 1)
                {
                    dx0 = -1;
                    dx1 = -2;
                    dx2 = 0;
                    dx3 = 2;
                    dx4 = 1;
                }
                else if (dx == 2)
                {
                    dx0 = 1;
                    dx1 = 0;
                    dx2 = -2;
                    dx3 = 0;
                    dx4 = 1;
                }
                else
                {
                    dx0 = 1;
                    dx1 = 1;
                    dx2 = 1;
                    dx3 = 1;
                    dx4 = 1;
                }

                if (dy == 0)
                {
                    dy1 = 4;
                    dy2 = 6;
                    dy3 = 4;
                }
                else if (dy == 1)
                {
                    dy1 = -2;
                    dy2 = 0;
                    dy3 = 2;
                }
                else if (dy == 2)
                {
                    dy1 = 0;
                    dy2 = -2;
                    dy3 = 0;
                }
                else
                {
                    dy1 = 0;
                    dy2 = 0;
                    dy3 = 0;
                }

                // int16_t trow[width];

                std::vector<int16_t> trow_buf(width);
                int16_t *trow = trow_buf.data();

                static thread_local std::vector<uint8_t> zero_row;
                if (zero_row.size() < static_cast<size_t>(width))
                {
                    zero_row.clear();
                    zero_row.resize(width, 0);
                }

                for (int y = start; y < end; ++y)
                {
                    const uint8_t *srowm2, *srowm1, *srow0, *srowp1, *srowp2;

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
                        srowm2 = y >= 2 ? (src_data + (y - 2) * src_step) : zero_row.data();
                        srowm1 = y >= 1 ? (src_data + (y - 1) * src_step) : zero_row.data();
                        srow0 = src_data + y * src_step;
                        srowp1 = y + 1 <= height - 1 ? (src_data + (y + 1) * src_step) : zero_row.data();
                        srowp2 = y + 2 <= height - 1 ? (src_data + (y + 2) * src_step) : zero_row.data();
                    }
                    else if (border_type == BORDER_REFLECT)
                    {
                        if (y >= 2)
                        {
                            srowm2 = src_data + (y - 2) * src_step;
                        }
                        else if (y == 1)
                        {
                            srowm2 = src_data + (y - 1) * src_step;
                        }
                        else // y == 0
                        {
                            srowm2 = src_data + (y + 1) * src_step;
                        }

                        if (y >= 1)
                        {
                            srowm1 = src_data + (y - 1) * src_step;
                        }
                        else // y == 0
                        {
                            srowm1 = src_data + y * src_step;
                        }

                        srow0 = src_data + y * src_step;

                        if (y + 1 <= height - 1) // height - 2
                        {
                            srowp1 = src_data + (y + 1) * src_step;
                        }
                        else if (y + 1 == height) // height - 1
                        {
                            srowp1 = src_data + y * src_step;
                        }
                        else
                        {
                            srowp1 = src_data;
                        }

                        if (y + 2 <= height - 1) // height - 3
                        {
                            srowp2 = src_data + (y + 2) * src_step;
                        }
                        else if (y + 2 == height) // height - 2
                        {
                            srowp2 = src_data + (y + 1) * src_step;
                        }
                        else // y == height - 1
                        {
                            srowp2 = src_data + (y - 1) * src_step;
                        }
                    }
                    else if (border_type == BORDER_REFLECT_101)
                    {
                        if (y >= 2)
                        {
                            srowm2 = src_data + (y - 2) * src_step;
                        }
                        else if (y == 1)
                        {
                            srowm2 = src_data + y * src_step;
                        }
                        else // y == 0
                        {
                            srowm2 = src_data + (y + 2) * src_step;
                        }

                        if (y >= 1)
                        {
                            srowm1 = src_data + (y - 1) * src_step;
                        }
                        else // y == 0
                        {
                            srowm1 = src_data + (y + 1) * src_step;
                        }

                        srow0 = src_data + y * src_step;

                        if (y + 1 <= height - 1) // height - 2
                        {
                            srowp1 = src_data + (y + 1) * src_step;
                        }
                        else if (y + 1 == height) // height - 1
                        {
                            srowp1 = src_data + (y - 1) * src_step;
                        }
                        else
                        {
                            srowp1 = src_data;
                        }

                        if (y + 2 <= height - 1) // height - 3
                        {
                            srowp2 = src_data + (y + 2) * src_step;
                        }
                        else if (y + 2 == height) // height - 2
                        {
                            srowp2 = src_data + y * src_step;
                        }
                        else // y == height - 1
                        {
                            srowp2 = src_data + (y - 2) * src_step;
                        }
                    }
                    else
                    {
                        srowm1 = src_data;
                    }

                    size_t vl = 0;

                    // Vector processing

                    if (dy == 0)
                    {
                        for (int x = 0; x < width; x += vl)
                        {
                            vl = __riscv_vsetvl_e8m4(width - x);

                            vuint8m4_t vsrow0 = __riscv_vle8_v_u8m4(srowm2 + x, vl);
                            vuint8m4_t vsrow1 = __riscv_vle8_v_u8m4(srowm1 + x, vl);
                            vuint8m4_t vsrow2 = __riscv_vle8_v_u8m4(srow0 + x, vl);
                            vuint8m4_t vsrow3 = __riscv_vle8_v_u8m4(srowp1 + x, vl);
                            vuint8m4_t vsrow4 = __riscv_vle8_v_u8m4(srowp2 + x, vl);

                            vuint16m8_t vrowx = __riscv_vwaddu_vv_u16m8(vsrow4, vsrow0, vl);
                            vrowx = __riscv_vwmaccu_vx_u16m8(vrowx, dy1, vsrow1, vl);
                            vrowx = __riscv_vwmaccu_vx_u16m8(vrowx, dy2, vsrow2, vl);
                            vrowx = __riscv_vwmaccu_vx_u16m8(vrowx, dy3, vsrow3, vl);

                            __riscv_vse16_v_i16m8(trow + x, __riscv_vreinterpret_v_u16m8_i16m8(vrowx), vl);
                        }
                    }
                    else if (dy == 1)
                    {
                        for (int x = 0; x < width; x += vl)
                        {
                            vl = __riscv_vsetvl_e8m4(width - x);

                            vint16m8_t vsrow0 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srowm2 + x, vl), vl));
                            vint16m8_t vsrow1 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srowm1 + x, vl), vl));
                            vint16m8_t vsrow3 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srowp1 + x, vl), vl));
                            vint16m8_t vsrow4 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srowp2 + x, vl), vl));

                            vint16m8_t vrowx = __riscv_vsub_vv_i16m8(vsrow4, vsrow0, vl);
                            vrowx = __riscv_vmacc_vx_i16m8(vrowx, dy1, vsrow1, vl);
                            vrowx = __riscv_vmacc_vx_i16m8(vrowx, dy3, vsrow3, vl);

                            __riscv_vse16_v_i16m8(trow + x, vrowx, vl);
                        }
                    }
                    else
                    {
                        for (int x = 0; x < width; x += vl)
                        {
                            vl = __riscv_vsetvl_e8m4(width - x);

                            vint16m8_t vsrow0 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srowm2 + x, vl), vl));
                            vint16m8_t vsrow2 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow0 + x, vl), vl));
                            vint16m8_t vsrow4 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srowp2 + x, vl), vl));

                            vint16m8_t vrowx = __riscv_vadd_vv_i16m8(vsrow4, vsrow0, vl);
                            vrowx = __riscv_vmacc_vx_i16m8(vrowx, dy2, vsrow2, vl);

                            __riscv_vse16_v_i16m8(trow + x, vrowx, vl);
                        }
                    }

                    if (dst_depth == CV_8U)
                    {
                        return CV_HAL_ERROR_NOT_IMPLEMENTED;
                        uint8_t *drow = (uint8_t *)(dst_data + y * dst_step);

                        int16_t temp = trow[0] * dx2 + trow[1] * dx3 + trow[2] * dx4;

                        if (border_type == BORDER_REPLICATE)
                        {
                            temp += trow[0] * dx0 + trow[0] * dx1;
                        }
                        else if (border_type == BORDER_REFLECT)
                        {
                            temp += trow[1] * dx0 + trow[0] * dx1;
                        }
                        else if (border_type == BORDER_REFLECT_101)
                        {
                            temp += trow[2] * dx0 + trow[1] * dx1;
                        }

                        drow[0] = (uint8_t)(temp > 0 ? (temp < 255 ? temp : 255) : 0);

                        temp = trow[0] * dx1 + trow[1] * dx2 + trow[2] * dx3 + trow[3] * dx4;

                        if (border_type == BORDER_REPLICATE)
                        {
                            temp += trow[0] * dx0;
                        }
                        else if (border_type == BORDER_REFLECT)
                        {
                            temp += trow[0] * dx0;
                        }
                        else if (border_type == BORDER_REFLECT_101)
                        {
                            temp += trow[1] * dx0;
                        }

                        drow[1] = (uint8_t)(temp > 0 ? (temp < 255 ? temp : 255) : 0);

                        if (dx == 0)
                        {
                            for (int x = 2; x < width - 2; x += vl)
                            {
                                vl = __riscv_vsetvl_e16m8(width - x - 2);
                                vint16m8_t vleft2 = __riscv_vle16_v_i16m8(trow + x - 2, vl);
                                vint16m8_t vleft1 = __riscv_vle16_v_i16m8(trow + x - 1, vl);
                                vint16m8_t vcenter = __riscv_vle16_v_i16m8(trow + x, vl);
                                vint16m8_t vright1 = __riscv_vle16_v_i16m8(trow + x + 1, vl);
                                vint16m8_t vright2 = __riscv_vle16_v_i16m8(trow + x + 2, vl);

                                vint16m8_t vres = __riscv_vadd_vv_i16m8(vright2, vleft2, vl);
                                vres = __riscv_vmacc_vx_i16m8(vres, dx1, vleft1, vl);
                                vres = __riscv_vmacc_vx_i16m8(vres, dx2, vcenter, vl);
                                vres = __riscv_vmacc_vx_i16m8(vres, dx3, vright1, vl);

                                vres = __riscv_vmax_vx_i16m8(vres, 0, vl);
                                vres = __riscv_vmin_vx_i16m8(vres, 255, vl);

                                __riscv_vse8_v_u8m4(drow + x, __riscv_vncvt_x_x_w_u8m4(__riscv_vreinterpret_v_i16m8_u16m8(vres), vl), vl);
                            }
                        }
                        else if (dx == 1)
                        {
                            for (int x = 2; x < width - 2; x += vl)
                            {
                                vl = __riscv_vsetvl_e16m8(width - x - 2);
                                vint16m8_t vleft2 = __riscv_vle16_v_i16m8(trow + x - 2, vl);
                                vint16m8_t vleft1 = __riscv_vle16_v_i16m8(trow + x - 1, vl);
                                vint16m8_t vright1 = __riscv_vle16_v_i16m8(trow + x + 1, vl);
                                vint16m8_t vright2 = __riscv_vle16_v_i16m8(trow + x + 2, vl);

                                vint16m8_t vres = __riscv_vsub_vv_i16m8(vright2, vleft2, vl);
                                vres = __riscv_vmacc_vx_i16m8(vres, dx1, vleft1, vl);
                                vres = __riscv_vmacc_vx_i16m8(vres, dx3, vright1, vl);

                                vres = __riscv_vmax_vx_i16m8(vres, 0, vl);
                                vres = __riscv_vmin_vx_i16m8(vres, 255, vl);

                                __riscv_vse8_v_u8m4(drow + x, __riscv_vncvt_x_x_w_u8m4(__riscv_vreinterpret_v_i16m8_u16m8(vres), vl), vl);
                            }
                        }
                        else
                        {
                            for (int x = 2; x < width - 2; x += vl)
                            {
                                vl = __riscv_vsetvl_e16m8(width - x - 2);
                                vint16m8_t vleft2 = __riscv_vle16_v_i16m8(trow + x - 2, vl);
                                vint16m8_t vcenter = __riscv_vle16_v_i16m8(trow + x, vl);
                                vint16m8_t vright2 = __riscv_vle16_v_i16m8(trow + x + 2, vl);

                                vint16m8_t vres = __riscv_vadd_vv_i16m8(vright2, vleft2, vl);
                                vres = __riscv_vmacc_vx_i16m8(vres, dx2, vcenter, vl);

                                vres = __riscv_vmax_vx_i16m8(vres, 0, vl);
                                vres = __riscv_vmin_vx_i16m8(vres, 255, vl);

                                __riscv_vse8_v_u8m4(drow + x, __riscv_vncvt_x_x_w_u8m4(__riscv_vreinterpret_v_i16m8_u16m8(vres), vl), vl);
                            }
                        }

                        size_t x = width - 2;

                        temp = trow[x - 2] * dx0 + trow[x - 1] * dx1 + trow[x] * dx2 + trow[x + 1] * dx3;

                        if (border_type == BORDER_REPLICATE)
                        {
                            temp += trow[x + 1] * dx4;
                        }
                        else if (border_type == BORDER_REFLECT)
                        {
                            temp += trow[width - 1] * dx4;
                        }
                        else if (border_type == BORDER_REFLECT_101)
                        {
                            temp += trow[width - 2] * dx4;
                        }

                        drow[x] = (uint8_t)(temp > 0 ? (temp < 255 ? temp : 255) : 0);

                        x = width - 1;

                        temp = trow[width - 3] * dx0 + trow[width - 2] * dx1 + trow[width - 1] * dx2;

                        if (border_type == BORDER_REPLICATE)
                        {
                            temp += trow[width - 1] * dx3 + trow[width - 1] * dx4;
                        }
                        else if (border_type == BORDER_REFLECT)
                        {
                            temp += trow[width - 1] * dx3 + trow[width - 2] * dx4;
                        }
                        else if (border_type == BORDER_REFLECT_101)
                        {
                            temp += trow[width - 2] * dx3 + trow[width - 3] * dx4;
                        }

                        drow[x] = (uint8_t)(temp > 0 ? (temp < 255 ? temp : 255) : 0);
                    }

                    else if (dst_depth == CV_16S)
                    {
                        size_t x = 0;

                        int16_t *drow = (int16_t *)(dst_data + y * dst_step);

                        int16_t temp = trow[0] * dx2 + trow[1] * dx3 + trow[2] * dx4;

                        if (border_type == BORDER_REPLICATE)
                        {
                            temp += trow[0] * dx0 + trow[0] * dx1;
                        }
                        else if (border_type == BORDER_REFLECT)
                        {
                            temp += trow[1] * dx0 + trow[0] * dx1;
                        }
                        else if (border_type == BORDER_REFLECT_101)
                        {
                            temp += trow[2] * dx0 + trow[1] * dx1;
                        }

                        drow[0] = temp;

                        temp = trow[0] * dx1 + trow[1] * dx2 + trow[2] * dx3 + trow[3] * dx4;

                        if (border_type == BORDER_REPLICATE)
                        {
                            temp += trow[0] * dx0;
                        }
                        else if (border_type == BORDER_REFLECT)
                        {
                            temp += trow[0] * dx0;
                        }
                        else if (border_type == BORDER_REFLECT_101)
                        {
                            temp += trow[1] * dx0;
                        }

                        drow[1] = temp;

                        x = 1;

                        if (dx == 0)
                        {
                            for (int xx = 2; xx < width - 2; xx += vl)
                            {
                                vl = __riscv_vsetvl_e16m8(width - xx - 2);
                                vint16m8_t vleft2 = __riscv_vle16_v_i16m8(trow + xx - 2, vl);
                                vint16m8_t vleft1 = __riscv_vle16_v_i16m8(trow + xx - 1, vl);
                                vint16m8_t vcenter = __riscv_vle16_v_i16m8(trow + xx, vl);
                                vint16m8_t vright1 = __riscv_vle16_v_i16m8(trow + xx + 1, vl);
                                vint16m8_t vright2 = __riscv_vle16_v_i16m8(trow + xx + 2, vl);

                                vint16m8_t vres = __riscv_vadd_vv_i16m8(vright2, vleft2, vl);
                                vres = __riscv_vmacc_vx_i16m8(vres, dx1, vleft1, vl);
                                vres = __riscv_vmacc_vx_i16m8(vres, dx2, vcenter, vl);
                                vres = __riscv_vmacc_vx_i16m8(vres, dx3, vright1, vl);

                                __riscv_vse16_v_i16m8(drow + xx, vres, vl);
                            }
                        }
                        else if (dx == 1)
                        {
                            for (int xx = 2; xx < width - 2; xx += vl)
                            {
                                vl = __riscv_vsetvl_e16m8(width - xx - 2);
                                vint16m8_t vleft2 = __riscv_vle16_v_i16m8(trow + xx - 2, vl);
                                vint16m8_t vleft1 = __riscv_vle16_v_i16m8(trow + xx - 1, vl);
                                vint16m8_t vright1 = __riscv_vle16_v_i16m8(trow + xx + 1, vl);
                                vint16m8_t vright2 = __riscv_vle16_v_i16m8(trow + xx + 2, vl);

                                vint16m8_t vres = __riscv_vsub_vv_i16m8(vright2, vleft2, vl);
                                vres = __riscv_vmacc_vx_i16m8(vres, dx1, vleft1, vl);
                                vres = __riscv_vmacc_vx_i16m8(vres, dx3, vright1, vl);

                                __riscv_vse16_v_i16m8(drow + xx, vres, vl);
                            }
                        }
                        else
                        {
                            for (int xx = 2; xx < width - 2; xx += vl)
                            {
                                vl = __riscv_vsetvl_e16m8(width - xx - 2);
                                vint16m8_t vleft2 = __riscv_vle16_v_i16m8(trow + xx - 2, vl);
                                vint16m8_t vcenter = __riscv_vle16_v_i16m8(trow + xx, vl);
                                vint16m8_t vright2 = __riscv_vle16_v_i16m8(trow + xx + 2, vl);

                                vint16m8_t vres = __riscv_vadd_vv_i16m8(vright2, vleft2, vl);
                                vres = __riscv_vmacc_vx_i16m8(vres, dx2, vcenter, vl);

                                __riscv_vse16_v_i16m8(drow + xx, vres, vl);
                            }
                        }
                        x = width - 2;

                        temp = trow[x - 2] * dx0 + trow[x - 1] * dx1 + trow[x] * dx2 + trow[x + 1] * dx3;

                        if (border_type == BORDER_REPLICATE)
                        {
                            temp += trow[width - 1] * dx4;
                        }
                        else if (border_type == BORDER_REFLECT)
                        {
                            temp += trow[width - 1] * dx4;
                        }
                        else if (border_type == BORDER_REFLECT_101)
                        {
                            temp += trow[width - 2] * dx4;
                        }

                        drow[x] = temp;

                        x = width - 1;

                        temp = trow[width - 3] * dx0 + trow[width - 2] * dx1 + trow[width - 1] * dx2;

                        if (border_type == BORDER_REPLICATE)
                        {
                            temp += trow[width - 1] * dx3 + trow[width - 1] * dx4;
                        }
                        else if (border_type == BORDER_REFLECT)
                        {
                            temp += trow[width - 1] * dx3 + trow[width - 2] * dx4;
                        }
                        else if (border_type == BORDER_REFLECT_101)
                        {
                            temp += trow[width - 2] * dx3 + trow[width - 3] * dx4;
                        }

                        drow[x] = temp;
                    }

                    else if (dst_depth == CV_32F)
                    {
                        size_t x = 0;

                        float *drow = (float *)(dst_data + y * dst_step);

                        int16_t temp = trow[0] * dx2 + trow[1] * dx3 + trow[2] * dx4;

                        if (border_type == BORDER_REPLICATE)
                        {
                            temp += trow[0] * dx0 + trow[0] * dx1;
                        }
                        else if (border_type == BORDER_REFLECT)
                        {
                            temp += trow[1] * dx0 + trow[0] * dx1;
                        }
                        else if (border_type == BORDER_REFLECT_101)
                        {
                            temp += trow[2] * dx0 + trow[1] * dx1;
                        }

                        drow[0] = (float)(temp);

                        temp = trow[0] * dx1 + trow[1] * dx2 + trow[2] * dx3 + trow[3] * dx4;

                        if (border_type == BORDER_REPLICATE)
                        {
                            temp += trow[0] * dx0;
                        }
                        else if (border_type == BORDER_REFLECT)
                        {
                            temp += trow[0] * dx0;
                        }
                        else if (border_type == BORDER_REFLECT_101)
                        {
                            temp += trow[1] * dx0;
                        }

                        drow[1] = (float)temp;

                        x = 1;

                        if (dx == 0)
                        {
                            for (int xx = 2; xx < width - 2; xx += vl)
                            {
                                vl = __riscv_vsetvl_e16m4(width - xx - 2);
                                vint16m4_t vleft2 = __riscv_vle16_v_i16m4(trow + xx - 2, vl);
                                vint16m4_t vleft1 = __riscv_vle16_v_i16m4(trow + xx - 1, vl);
                                vint16m4_t vcenter = __riscv_vle16_v_i16m4(trow + xx, vl);
                                vint16m4_t vright1 = __riscv_vle16_v_i16m4(trow + xx + 1, vl);
                                vint16m4_t vright2 = __riscv_vle16_v_i16m4(trow + xx + 2, vl);

                                vint16m4_t vres = __riscv_vadd_vv_i16m4(vright2, vleft2, vl);
                                vres = __riscv_vmacc_vx_i16m4(vres, dx1, vleft1, vl);
                                vres = __riscv_vmacc_vx_i16m4(vres, dx2, vcenter, vl);
                                vres = __riscv_vmacc_vx_i16m4(vres, dx3, vright1, vl);

                                __riscv_vse32_v_f32m8(drow + xx, __riscv_vfwcvt_f_x_v_f32m8(vres, vl), vl);
                            }
                        }
                        else if (dx == 1)
                        {
                            for (int xx = 2; xx < width - 2; xx += vl)
                            {
                                vl = __riscv_vsetvl_e16m4(width - xx - 2);
                                vint16m4_t vleft2 = __riscv_vle16_v_i16m4(trow + xx - 2, vl);
                                vint16m4_t vleft1 = __riscv_vle16_v_i16m4(trow + xx - 1, vl);
                                vint16m4_t vright1 = __riscv_vle16_v_i16m4(trow + xx + 1, vl);
                                vint16m4_t vright2 = __riscv_vle16_v_i16m4(trow + xx + 2, vl);

                                vint16m4_t vres = __riscv_vsub_vv_i16m4(vright2, vleft2, vl);
                                vres = __riscv_vmacc_vx_i16m4(vres, dx1, vleft1, vl);
                                vres = __riscv_vmacc_vx_i16m4(vres, dx3, vright1, vl);

                                __riscv_vse32_v_f32m8(drow + xx, __riscv_vfwcvt_f_x_v_f32m8(vres, vl), vl);
                            }
                        }
                        else
                        {
                            for (int xx = 2; xx < width - 2; xx += vl)
                            {
                                vl = __riscv_vsetvl_e16m4(width - xx - 2);
                                vint16m4_t vleft2 = __riscv_vle16_v_i16m4(trow + xx - 2, vl);
                                vint16m4_t vcenter = __riscv_vle16_v_i16m4(trow + xx, vl);
                                vint16m4_t vright2 = __riscv_vle16_v_i16m4(trow + xx + 2, vl);

                                vint16m4_t vres = __riscv_vadd_vv_i16m4(vright2, vleft2, vl);
                                vres = __riscv_vmacc_vx_i16m4(vres, dx2, vcenter, vl);

                                __riscv_vse32_v_f32m8(drow + xx, __riscv_vfwcvt_f_x_v_f32m8(vres, vl), vl);
                            }
                        }

                        x = width - 2;

                        temp = trow[width - 4] * dx0 + trow[width - 3] * dx1 + trow[width - 2] * dx2 + trow[width - 1] * dx3;

                        if (border_type == BORDER_REPLICATE)
                        {
                            temp += trow[width - 1] * dx4;
                        }
                        else if (border_type == BORDER_REFLECT)
                        {
                            temp += trow[width - 1] * dx4;
                        }
                        else if (border_type == BORDER_REFLECT_101)
                        {
                            temp += trow[width - 2] * dx4;
                        }

                        drow[x] = (float)temp;

                        x = width - 1;

                        temp = trow[width - 3] * dx0 + trow[width - 2] * dx1 + trow[width - 1] * dx2;

                        if (border_type == BORDER_REPLICATE)
                        {
                            temp += trow[width - 1] * dx3 + trow[width - 1] * dx4;
                        }
                        else if (border_type == BORDER_REFLECT)
                        {
                            temp += trow[width - 1] * dx3 + trow[width - 2] * dx4;
                        }
                        else if (border_type == BORDER_REFLECT_101)
                        {
                            temp += trow[width - 2] * dx3 + trow[width - 3] * dx4;
                        }

                        drow[x] = (float)temp;
                    }
                }

                return CV_HAL_ERROR_OK;
            }

            class sobelInvoker : public ParallelLoopBody
            {
            public:
                explicit sobelInvoker(std::function<int(int, int)> _func)
                    : func(std::move(_func)) {}

                void operator()(const Range &range) const override
                {
                    func(range.start, range.end);
                }

            private:
                std::function<int(int, int)> func;
            };

            template <typename... Args>
            inline int invoke(int height,
                              int (*f)(int, int, Args...),
                              Args... args)
            {
                using namespace std::placeholders;
                auto bound = std::bind(f, _1, _2, std::forward<Args>(args)...);
                cv::parallel_for_(Range(1, height), sobelInvoker(bound), cv::getNumThreads());
                return f(0, 1, std::forward<Args>(args)...);
            }

            int sobel(const uint8_t *src_data, size_t src_step,
                      uint8_t *dst_data, size_t dst_step,
                      int width, int height,
                      int src_depth, int dst_depth, int cn,
                      int margin_left, int margin_top,
                      int margin_right, int margin_bottom,
                      int dx, int dy, int ksize,
                      double scale, double delta,
                      int border_type)
            {
                if (src_depth != CV_8U ||
                    (dst_depth != CV_8U && dst_depth != CV_16S && dst_depth != CV_32F) ||
                    cn != 1 || width < 3)
                    return CV_HAL_ERROR_NOT_IMPLEMENTED;

                if (scale != 1 || delta != 0)
                    return CV_HAL_ERROR_NOT_IMPLEMENTED;

                if (margin_left != 0 || margin_top != 0 || margin_right != 0 || margin_bottom != 0)
                    return CV_HAL_ERROR_NOT_IMPLEMENTED;

                if (height < ksize || width < ksize)
                    return CV_HAL_ERROR_NOT_IMPLEMENTED;

                int size = 1;
                if (dst_depth == CV_16S)
                {
                    size = 2;
                }
                else if (dst_depth == CV_32F)
                {
                    size = 4;
                }

                std::vector<uint8_t> temp_buffer;
                uint8_t *actual_dst_data = dst_data;
                if (src_data == dst_data)
                {
                    temp_buffer.resize(height * dst_step);
                    actual_dst_data = temp_buffer.data();
                }

                int res = 0;

                if (ksize == 3)
                {
                    res = invoke(height,
                                 &sobel_3X3,
                                 src_data, src_step,
                                 actual_dst_data, dst_step,
                                 width, height,
                                 src_depth, dst_depth, cn,
                                 margin_left, margin_top,
                                 margin_right, margin_bottom,
                                 dx, dy,
                                 scale, delta,
                                 border_type);
                }
                else if (ksize == 5)
                {
                    res = invoke(height,
                                 &sobel_5X5,
                                 src_data, src_step,
                                 actual_dst_data, dst_step,
                                 width, height,
                                 src_depth, dst_depth, cn,
                                 margin_left, margin_top,
                                 margin_right, margin_bottom,
                                 dx, dy,
                                 scale, delta,
                                 border_type);
                }
                else
                {
                    return CV_HAL_ERROR_NOT_IMPLEMENTED;
                }

                if (src_data == dst_data && res == CV_HAL_ERROR_OK)
                {
                    for (int y = 0; y < height; ++y)
                    {
                        memcpy(dst_data + y * dst_step,
                               actual_dst_data + y * dst_step,
                               width * size);
                    }
                }

                return res;
            }

#endif // CV_HAL_RVV_1P0_ENABLED

        }
    }
} // cv::rvv_hal::imgproc
