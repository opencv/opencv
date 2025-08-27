// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html  .

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

            static inline int scharr_dx0(int start, int end, const uint8_t *src_data, size_t src_step,
                                         uint8_t *dst_data, size_t dst_step,
                                         int width, int height,
                                         int src_depth  __attribute__((unused)), int dst_depth, int cn __attribute__((unused)),
                                         int margin_left __attribute__((unused)), int margin_top __attribute__((unused)),
                                         int margin_right __attribute__((unused)), int margin_bottom __attribute__((unused)),
                                         double scale __attribute__((unused)), double delta __attribute__((unused)),
                                         int border_type)
            {
                int dx0, dx1, dx2, dy0, dy1, dy2;

                dy0 = -1;
                dx0 = 3;

                dy1 = 0;
                dx1 = 10;

                dy2 = 1;
                dx2 = 3;

                int16_t trow[width];

                static thread_local std::vector<uint8_t> zero_row;
                if (zero_row.size() < static_cast<size_t>(width))
                {
                    zero_row.clear();
                    zero_row.resize(width, 0);
                }

                if (dst_depth == CV_8U)
                {
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

                        trow[0] = (dy0 * srow0[0] + dy1 * srow1[0] + dy2 * srow2[0]);

                        // Last pixel

                        trow[width - 1] = dy0 * srow0[width - 1] + dy1 * srow1[width - 1] + dy2 * srow2[width - 1];

                        size_t vl = 0;

                        // Vector processing

                        for (int x = 1; x < width - 1; x += vl)
                        {
                            vl = __riscv_vsetvl_e8m4(width - x - 1);

                            vint16m8_t vsrow0 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow0 + x, vl), vl));
                            vint16m8_t vsrow2 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow2 + x, vl), vl));

                            vint16m8_t vrowx = __riscv_vsub_vv_i16m8(vsrow2, vsrow0, vl);

                            __riscv_vse16_v_i16m8(trow + x, vrowx, vl);
                        }

                        int x = 0;

                        uint8_t *drow = (uint8_t *)(dst_data + y * dst_step);

                        int16_t temp = trow[0] * dx1 + trow[1] * dx2;

                        if (border_type == BORDER_REPLICATE)
                        {
                            temp += trow[0] * dx0;
                        }

                        drow[x] = (uint8_t)(temp > 0 ? (temp < 255 ? temp : 255) : 0);

                        x = 1;
                        for (; x < width - 1; x += vl)
                        {
                            vl = __riscv_vsetvl_e16m8(width - x - 1);
                            vint16m8_t vleft = __riscv_vle16_v_i16m8(trow + x - 1, vl);
                            vint16m8_t vcenter = __riscv_vle16_v_i16m8(trow + x, vl);
                            vint16m8_t vright = __riscv_vle16_v_i16m8(trow + x + 1, vl);

                            vint16m8_t vres = __riscv_vmul_vx_i16m8(vcenter, dx1, vl);
                            vcenter = __riscv_vadd_vv_i16m8(vleft, vright, vl);
                            vres = __riscv_vmacc_vx_i16m8(vres, dx0, vcenter, vl);

                            vres = __riscv_vmax_vx_i16m8(vres, 0, vl);
                            vres = __riscv_vmin_vx_i16m8(vres, 255, vl);

                            __riscv_vse8_v_u8m4(drow + x, __riscv_vncvt_x_x_w_u8m4(__riscv_vreinterpret_v_i16m8_u16m8(vres), vl), vl);
                        }

                        x = width - 1;

                        temp = trow[width - 2] * dx0 + trow[width - 1] * dx1;

                        if (border_type == BORDER_REPLICATE)
                        {
                            temp += trow[width - 1] * dx2;
                        }

                        drow[x] = (uint8_t)(temp > 0 ? (temp < 255 ? temp : 255) : 0);
                    }
                }

                else if (dst_depth == CV_16S)
                {

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

                        if (y == 0)
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
                        else if (y == height - 1)
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

                        for (int x = 0; x < width; x += vl)
                        {
                            vl = __riscv_vsetvl_e8m4(width - x);

                            vint16m8_t vsrow0 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow0 + x, vl), vl));
                            vint16m8_t vsrow2 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow2 + x, vl), vl));

                            vint16m8_t vrowx = __riscv_vsub_vv_i16m8(vsrow2, vsrow0, vl);

                            __riscv_vse16_v_i16m8(trow + x, vrowx, vl);
                        }

                        int x = 0;

                        int16_t *drow = (int16_t *)(dst_data + y * dst_step);

                        drow[0] = trow[0] * dx1 + trow[1] * dx2;

                        if (border_type == BORDER_REPLICATE)
                        {
                            drow[x] += trow[0] * dx0;
                        }

                        x = 1;
                        for (; x < width - 1; x += vl)
                        {
                            vl = __riscv_vsetvl_e16m8(width - x - 1);
                            vint16m8_t vleft = __riscv_vle16_v_i16m8(trow + x - 1, vl);
                            vint16m8_t vcenter = __riscv_vle16_v_i16m8(trow + x, vl);
                            vint16m8_t vright = __riscv_vle16_v_i16m8(trow + x + 1, vl);

                            vint16m8_t vres = __riscv_vmul_vx_i16m8(vcenter, dx1, vl);
                            vcenter = __riscv_vadd_vv_i16m8(vleft, vright, vl);
                            vres = __riscv_vmacc_vx_i16m8(vres, dx0, vcenter, vl);

                            __riscv_vse16_v_i16m8(drow + x, vres, vl);
                        }

                        x = width - 1;

                        drow[x] = trow[width - 2] * dx0 + trow[width - 1] * dx1;

                        if (border_type == BORDER_REPLICATE)
                        {
                            drow[x] += trow[width - 1] * dx2;
                        }
                    }
                }

                else if (dst_depth == CV_32F)
                {
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

                        if (y == 0)
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
                        else if (y == height - 1)
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
                        for (int x = 0; x < width; x += vl)
                        {
                            vl = __riscv_vsetvl_e8m4(width - x);

                            vint16m8_t vsrow0 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow0 + x, vl), vl));
                            vint16m8_t vsrow2 = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vwcvtu_x_x_v_u16m8(__riscv_vle8_v_u8m4(srow2 + x, vl), vl));

                            vint16m8_t vrowx = __riscv_vsub_vv_i16m8(vsrow2, vsrow0, vl);

                            __riscv_vse16_v_i16m8(trow + x, vrowx, vl);
                        }

                        int x = 0;

                        float *drow = (float *)(dst_data + y * dst_step);

                        drow[x] = trow[0] * dx1 + trow[1] * dx2;

                        if (border_type == BORDER_REPLICATE)
                        {
                            drow[x] += trow[0] * dx0;
                        }

                        x = 1;

                        for (; x < width - 1; x += vl)
                        {
                            vl = __riscv_vsetvl_e16m4(width - x - 1);
                            vint16m4_t vleft = __riscv_vle16_v_i16m4(trow + x - 1, vl);
                            vint16m4_t vcenter = __riscv_vle16_v_i16m4(trow + x, vl);
                            vint16m4_t vright = __riscv_vle16_v_i16m4(trow + x + 1, vl);

                            vint16m4_t vres = __riscv_vmul_vx_i16m4(vcenter, dx1, vl);
                            vcenter = __riscv_vadd_vv_i16m4(vleft, vright, vl);
                            vres = __riscv_vmacc_vx_i16m4(vres, dx0, vcenter, vl);

                            __riscv_vse32_v_f32m8(drow + x, __riscv_vfwcvt_f_x_v_f32m8(vres, vl), vl);
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

            static inline int scharr_dx1(int start, int end, const uint8_t *src_data, size_t src_step,
                                         uint8_t *dst_data, size_t dst_step,
                                         int width, int height,
                                         int src_depth __attribute__((unused)), int dst_depth, int cn __attribute__((unused)),
                                         int margin_left __attribute__((unused)), int margin_top __attribute__((unused)),
                                         int margin_right __attribute__((unused)), int margin_bottom __attribute__((unused)),
                                         double scale __attribute__((unused)), double delta __attribute__((unused)),
                                         int border_type)
            {
                int dx0, dx1, dx2, dy0, dy1, dy2;
                dx0 = -1;
                dy0 = 3;

                dx1 = 0;
                dy1 = 10;

                dx2 = 1;
                dy2 = 3;

                int16_t trow[width];

                static thread_local std::vector<uint8_t> zero_row;
                if (zero_row.size() < static_cast<size_t>(width))
                {
                    zero_row.clear();
                    zero_row.resize(width, 0);
                }

                if (dst_depth == CV_8U)
                {
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

                        if (y == 0)
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
                        else if (y == height - 1)
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

                        trow[0] = (dy0 * srow0[0] + dy1 * srow1[0] + dy2 * srow2[0]);

                        // Last pixel

                        trow[width - 1] = dy0 * srow0[width - 1] + dy1 * srow1[width - 1] + dy2 * srow2[width - 1];

                        size_t vl = 0;

                        // Vector processing

                        for (int x = 1; x < width - 1; x += vl)
                        {
                            vl = __riscv_vsetvl_e8m4(width - x - 1);

                            vuint8m4_t vsrow0 = __riscv_vle8_v_u8m4(srow0 + x, vl);
                            vuint8m4_t vsrow1 = __riscv_vle8_v_u8m4(srow1 + x, vl);
                            vuint8m4_t vsrow2 = __riscv_vle8_v_u8m4(srow2 + x, vl);

                            vuint16m8_t vrowx = __riscv_vwmulu_vx_u16m8(vsrow1, dy1, vl);
                            vuint16m8_t vtemp = __riscv_vwaddu_vv_u16m8(vsrow0, vsrow2, vl);
                            vrowx = __riscv_vmacc_vx_u16m8(vrowx, dy0, vtemp, vl);

                            __riscv_vse16_v_i16m8(trow + x, __riscv_vreinterpret_v_u16m8_i16m8(vrowx), vl);
                        }

                        int x = 0;

                        uint8_t *drow = (uint8_t *)(dst_data + y * dst_step);

                        int16_t temp = trow[0] * dx1 + trow[1] * dx2;

                        if (border_type == BORDER_REPLICATE)
                        {
                            temp += trow[0] * dx0;
                        }

                        drow[x] = (uint8_t)(temp > 0 ? (temp < 255 ? temp : 255) : 0);

                        x = 1;

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

                        x = width - 1;

                        temp = trow[width - 2] * dx0 + trow[width - 1] * dx1;

                        if (border_type == BORDER_REPLICATE)
                        {
                            temp += trow[width - 1] * dx2;
                        }

                        drow[x] = (uint8_t)(temp > 0 ? (temp < 255 ? temp : 255) : 0);
                    }
                }

                else if (dst_depth == CV_16S)
                {
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

                        if (y == 0)
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
                        else if (y == height - 1)
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

                        for (int x = 0; x < width; x += vl)
                        {
                            vl = __riscv_vsetvl_e8m4(width - x);

                            vuint8m4_t vsrow0 = __riscv_vle8_v_u8m4(srow0 + x, vl);
                            vuint8m4_t vsrow1 = __riscv_vle8_v_u8m4(srow1 + x, vl);
                            vuint8m4_t vsrow2 = __riscv_vle8_v_u8m4(srow2 + x, vl);

                            vuint16m8_t vrowx = __riscv_vwmulu_vx_u16m8(vsrow1, dy1, vl);
                            vuint16m8_t vtemp = __riscv_vwaddu_vv_u16m8(vsrow0, vsrow2, vl);
                            vrowx = __riscv_vmacc_vx_u16m8(vrowx, dy0, vtemp, vl);

                            __riscv_vse16_v_i16m8(trow + x, __riscv_vreinterpret_v_u16m8_i16m8(vrowx), vl);
                        }

                        int x = 0;

                        int16_t *drow = (int16_t *)(dst_data + y * dst_step);

                        drow[x] = trow[0] * dx1 + trow[1] * dx2;

                        if (border_type == BORDER_REPLICATE)
                        {
                            drow[x] += trow[0] * dx0;
                        }

                        x = 1;
                        for (; x < width - 1; x += vl)
                        {
                            vl = __riscv_vsetvl_e16m8(width - x - 1);
                            vint16m8_t vleft = __riscv_vle16_v_i16m8(trow + x - 1, vl);

                            vint16m8_t vright = __riscv_vle16_v_i16m8(trow + x + 1, vl);
                            vint16m8_t vres = __riscv_vsub_vv_i16m8(vright, vleft, vl);

                            __riscv_vse16_v_i16m8(drow + x, vres, vl);
                        }

                        x = width - 1;

                        drow[x] = trow[width - 2] * dx0 + trow[width - 1] * dx1;

                        if (border_type == BORDER_REPLICATE)
                        {
                            drow[x] += trow[width - 1] * dx2;
                        }
                    }
                }

                else if (dst_depth == CV_32F)
                {
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

                        if (y == 0)
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
                        else if (y == height - 1)
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
                        for (int x = 0; x < width; x += vl)
                        {
                            vl = __riscv_vsetvl_e8m4(width - x);

                            vuint8m4_t vsrow0 = __riscv_vle8_v_u8m4(srow0 + x, vl);
                            vuint8m4_t vsrow1 = __riscv_vle8_v_u8m4(srow1 + x, vl);
                            vuint8m4_t vsrow2 = __riscv_vle8_v_u8m4(srow2 + x, vl);

                            vuint16m8_t vrowx = __riscv_vwmulu_vx_u16m8(vsrow1, dy1, vl);
                            vuint16m8_t vtemp = __riscv_vwaddu_vv_u16m8(vsrow0, vsrow2, vl);
                            vrowx = __riscv_vmacc_vx_u16m8(vrowx, dy0, vtemp, vl);

                            __riscv_vse16_v_i16m8(trow + x, __riscv_vreinterpret_v_u16m8_i16m8(vrowx), vl);
                        }

                        int x = 0;

                        float *drow = (float *)(dst_data + y * dst_step);

                        drow[x] = trow[0] * dx1 + trow[1] * dx2;

                        if (border_type == BORDER_REPLICATE)
                        {
                            drow[x] += trow[0] * dx0;
                        }

                        x = 1;

                        for (; x < width - 1; x += vl)
                        {
                            vl = __riscv_vsetvl_e16m4(width - x - 1);
                            vint16m4_t vleft = __riscv_vle16_v_i16m4(trow + x - 1, vl);

                            vint16m4_t vright = __riscv_vle16_v_i16m4(trow + x + 1, vl);
                            vint16m4_t vres = __riscv_vsub_vv_i16m4(vright, vleft, vl);

                            __riscv_vse32_v_f32m8(drow + x, __riscv_vfwcvt_f_x_v_f32m8(vres, vl), vl);
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

            class ScharrInvoker : public ParallelLoopBody
            {
            public:
                explicit ScharrInvoker(std::function<int(int, int)> _func)
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
                cv::parallel_for_(Range(1, height), ScharrInvoker(bound), cv::getNumThreads());
                return f(0, 1, std::forward<Args>(args)...);
            }

            int scharr(const uint8_t *src_data, size_t src_step,
                       uint8_t *dst_data, size_t dst_step,
                       int width, int height,
                       int src_depth, int dst_depth, int cn,
                       int margin_left, int margin_top,
                       int margin_right, int margin_bottom,
                       int dx, int dy __attribute__((unused)),
                       double scale, double delta,
                       int border_type)
            {
                if (src_depth != CV_8U ||
                    (dst_depth != CV_8U && dst_depth != CV_16S && dst_depth != CV_32F) ||
                    cn != 1 || width < 3)
                    return CV_HAL_ERROR_NOT_IMPLEMENTED;

                if (scale != 1 || delta != 0)
                    return CV_HAL_ERROR_NOT_IMPLEMENTED;

                if (border_type != BORDER_REPLICATE && border_type != BORDER_CONSTANT)
                    return CV_HAL_ERROR_NOT_IMPLEMENTED;

                if (margin_left != 0 || margin_top != 0 || margin_right != 0 || margin_bottom != 0)
                    return CV_HAL_ERROR_NOT_IMPLEMENTED;

                // This is copied from opencv/modules/imgproc/src/deriv.cpp
                // if (!(dx >= 0 && dy >= 0 && dx + dy == 1))
                // {
                //     return CV_HAL_ERROR_NOT_IMPLEMENTED;
                // }

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

                int res = invoke(height,
                                 dx == 1 ? &scharr_dx1 : &scharr_dx0,
                                 src_data, src_step,
                                 actual_dst_data, dst_step,
                                 width, height,
                                 src_depth, dst_depth, cn,
                                 margin_left, margin_top,
                                 margin_right, margin_bottom,
                                 scale, delta,
                                 border_type);
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