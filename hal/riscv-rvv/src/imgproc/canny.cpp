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

            typedef struct
            {
                int *buffer;
                int *prev;
                int *curr;
                int *next;
                size_t size;
            } LineBuffer;

            LineBuffer *createLineBuffer(size_t width, size_t align)
            {
                LineBuffer *buf = (LineBuffer *)malloc(sizeof(LineBuffer));
                if (!buf)
                    return NULL;

                size_t aligned_width = ((width + align - 1) & ~(align - 1));
                buf->buffer = (int *)aligned_alloc(align, 3 * aligned_width * sizeof(int));
                if (!buf->buffer)
                {
                    free(buf);
                    return NULL;
                }

                buf->prev = buf->buffer;
                buf->curr = buf->prev + aligned_width;
                buf->next = buf->curr + aligned_width;
                buf->size = aligned_width;

                return buf;
            }

            void rotateLineBuffer(LineBuffer *buf)
            {
                int *temp = buf->prev;
                buf->prev = buf->curr;
                buf->curr = buf->next;
                buf->next = temp;
            }

            void freeLineBuffer(LineBuffer *buf)
            {
                if (buf)
                {
                    free(buf->buffer);
                    free(buf);
                }
            }

            typedef struct
            {
                uint8_t **data;
                int capacity;
                int size;
            } Stack;

            static Stack *createStack(int capacity)
            {
                Stack *stack = (Stack *)malloc(sizeof(Stack));
                stack->data = (uint8_t **)malloc(capacity * sizeof(uint8_t *));
                stack->capacity = capacity;
                stack->size = 0;
                return stack;
            }

            static void push(Stack *stack, uint8_t *item)
            {
                if (stack->size < stack->capacity)
                {
                    stack->data[stack->size++] = item;
                }
            }

            static uint8_t *pop(Stack *stack)
            {
                if (stack->size > 0)
                {
                    return stack->data[--stack->size];
                }
                return NULL;
            }

            static void freeStack(Stack *stack)
            {
                free(stack->data);
                free(stack);
            }

            int sobel(const uint8_t *src_data, size_t src_step, uint8_t *dst_data, size_t dst_step, int width, int height, int src_depth, int dst_depth, int cn, int margin_left, int margin_top, int margin_right, int margin_bottom, int dx, int dy, int ksize, double scale, double delta, int border_type);

            int canny(const uint8_t *src_data, size_t src_step,
                      uint8_t *dst_data, size_t dst_step,
                      int width, int height, int cn,
                      double low_thresh, double high_thresh,
                      int ksize, bool L2gradient)
            {
                if (cn != 1 || (ksize != 3 && ksize != 5))
                {
                    return CV_HAL_ERROR_NOT_IMPLEMENTED;
                }

                if (L2gradient)
                {
                    low_thresh = 32767.0 < low_thresh ? 32767.0 : low_thresh;
                    high_thresh = 32767.0 < high_thresh ? 32767.0 : high_thresh;

                    if (low_thresh > 0)
                        low_thresh *= low_thresh;
                    if (high_thresh > 0)
                        high_thresh *= high_thresh;
                }

                int i = (int)low_thresh;
                low_thresh = i - (i > low_thresh);
                i = (int)high_thresh;
                high_thresh = i - (i > high_thresh);

                int16_t *dx_data = (int16_t *)malloc(height * width * sizeof(int16_t));
                int16_t *dy_data = (int16_t *)malloc(height * width * sizeof(int16_t));
                if (!dx_data || !dy_data)
                    return CV_HAL_ERROR_NOT_IMPLEMENTED;

                const size_t grad_step = width * sizeof(int16_t);

                int ret = sobel(src_data, src_step, (uint8_t *)dx_data, grad_step,
                                width, height, CV_8U, CV_16S, 1,
                                0, 0, 0, 0, 1, 0, ksize, 0.0, 0.0, BORDER_REPLICATE);
                if (ret != CV_HAL_ERROR_OK)
                {
                    free(dx_data);
                    free(dy_data);
                    return ret;
                }

                ret = sobel(src_data, src_step, (uint8_t *)dy_data, grad_step,
                            width, height, CV_8U, CV_16S, 1,
                            0, 0, 0, 0, 0, 1, ksize, 0.0, 0.0, BORDER_REPLICATE);
                if (ret != CV_HAL_ERROR_OK)
                {
                    free(dx_data);
                    free(dy_data);
                    return ret;
                }

                const int mapstep = width + 2;
                const size_t mapsize = (height + 2) * mapstep;
                uint8_t *edge_map = (uint8_t *)calloc(mapsize, sizeof(uint8_t));
                if (!edge_map)
                {
                    free(dx_data);
                    free(dy_data);
                    return CV_HAL_ERROR_NOT_IMPLEMENTED;
                }

                Stack *stack = createStack(width * height);
                Stack *borderPeaksLocal = createStack(width * height);
                if (!stack || !borderPeaksLocal)
                {
                    free(dx_data);
                    free(dy_data);
                    free(edge_map);
                    return CV_HAL_ERROR_NOT_IMPLEMENTED;
                }

                memset(edge_map, 1, mapstep);
                memset(edge_map + (height + 1) * mapstep, 1, mapstep);
                uint8_t *map = edge_map + mapstep + 1;

                const int simd_align = 16;
                LineBuffer *lineBuf = createLineBuffer(width, simd_align);
                if (!lineBuf)
                {
                    free(dx_data);
                    free(dy_data);
                    free(edge_map);
                    freeStack(stack);
                    freeStack(borderPeaksLocal);
                    return CV_HAL_ERROR_NOT_IMPLEMENTED;
                }

                const int TG22 = 13573;
                for (int i = 0; i <= height; i++)
                {
                    if (i < height)
                    {
                        int16_t *dx = (int16_t *)(dx_data + i * width);
                        int16_t *dy = (int16_t *)(dy_data + i * width);

                        if (L2gradient)
                        {
                            size_t vl = 0;
                            for (int j = 0; j < width; j += vl)
                            {
                                vl = __riscv_vsetvl_e32m8(width - j);
                                vint32m8_t vdx = __riscv_vwcvt_x_x_v_i32m8(__riscv_vle16_v_i16m4(dx + j, vl), vl);
                                vint32m8_t vdy = __riscv_vwcvt_x_x_v_i32m8(__riscv_vle16_v_i16m4(dy + j, vl), vl);
                                vint32m8_t vres = __riscv_vmul_vv_i32m8(vdx, vdx, vl);
                                vres = __riscv_vmacc_vv_i32m8(vres, vdy, vdy, vl);
                                __riscv_vse32_v_i32m8(lineBuf->next + j, vres, vl);
                            }
                        }
                        else
                        {
                            size_t vl = 0;
                            for (int j = 0; j < width; j += vl)
                            {
                                vl = __riscv_vsetvl_e32m8(width - j);
                                vint32m8_t vdx = __riscv_vwcvt_x_x_v_i32m8(__riscv_vle16_v_i16m4(dx + j, vl), vl);
                                vint32m8_t vdy = __riscv_vwcvt_x_x_v_i32m8(__riscv_vle16_v_i16m4(dy + j, vl), vl);
                                vbool4_t mask = __riscv_vmslt_vx_i32m8_b4(vdx, 0, vl);
                                vdx = __riscv_vmul_vx_i32m8_m(mask, vdx, -1, vl);
                                mask = __riscv_vmslt_vx_i32m8_b4(vdy, 0, vl);
                                vdy = __riscv_vmul_vx_i32m8_m(mask, vdy, -1, vl);
                                vint32m8_t vres = __riscv_vadd_vv_i32m8(vdx, vdy, vl);
                                __riscv_vse32_v_i32m8(lineBuf->next + j, vres, vl);
                            }
                        }
                    }
                    else
                    {
                        memset(lineBuf->next, 0, width * sizeof(int));
                    }

                    uint8_t *pmap = map + i * mapstep;

                    int16_t *dx = (int16_t *)(dx_data + i * width);
                    int16_t *dy = (int16_t *)(dy_data + i * width);

                    size_t vl = 0;
                    for (int j = 0; j < width; j += vl)
                    {
                        vl = __riscv_vsetvl_e32m8(width - j);

                        vint32m8_t vm = __riscv_vle32_v_i32m8(lineBuf->curr + j, vl);

                        vint32m8_t vdx = __riscv_vwcvt_x_x_v_i32m8(__riscv_vle16_v_i16m4(dx + j, vl), vl);
                        vint32m8_t vdy = __riscv_vwcvt_x_x_v_i32m8(__riscv_vle16_v_i16m4(dy + j, vl), vl);

                        vbool4_t mask_dx = __riscv_vmslt_vx_i32m8_b4(vdx, 0, vl);
                        vbool4_t mask_dy = __riscv_vmslt_vx_i32m8_b4(vdy, 0, vl);
                        vdx = __riscv_vmul_vx_i32m8_m(mask_dx, vdx, -1, vl);
                        vdy = __riscv_vmul_vx_i32m8_m(mask_dy, vdy, -1, vl);

                        vint32m8_t vy = __riscv_vsll_vx_i32m8(vdy, 15, vl);
                        vint32m8_t vtg22x = __riscv_vmul_vx_i32m8(vdx, TG22, vl);

                        vbool4_t vcmp_tg22 = __riscv_vmslt_vv_i32m8_b4(vy, vtg22x, vl);

                        vint32m8_t vprev = __riscv_vle32_v_i32m8(lineBuf->curr + j - 1, vl);
                        vint32m8_t vnext = __riscv_vle32_v_i32m8(lineBuf->curr + j + 1, vl);

                        vbool4_t vmax_h = __riscv_vmsgt_vv_i32m8_b4(vm, vprev, vl);
                        vbool4_t vmax_ge = __riscv_vmsge_vv_i32m8_b4(vm, vnext, vl);
                        vbool4_t vmax = __riscv_vmand_mm_b4(vmax_h, vmax_ge, vl);

                        vbool4_t vhigh = __riscv_vmsgt_vx_i32m8_b4(vm, high_thresh, vl);
                        vbool4_t vlow = __riscv_vmsgt_vx_i32m8_b4(vm, low_thresh, vl);

                        vuint8m2_t vres = __riscv_vmv_v_x_u8m2(1, vl);
                        vres = __riscv_vmerge_vxm_u8m2(vres, 0, vlow, vl);
                        vres = __riscv_vmerge_vxm_u8m2(vres, 2, vhigh, vl);

                        __riscv_vse8_v_u8m2(pmap + j, vres, vl);

                        vbool4_t vstrong = __riscv_vmand_mm_b4(vhigh, vmax, vl);
                        int32_t vidx = __riscv_vfirst_m_b4(vstrong, vl);
                        while (vidx >= 0)
                        {
                            push(stack, pmap + j + vidx);
                            vstrong = __riscv_vmand_mm_b4(vstrong, __riscv_vmclr_m_b4(vl), vl);
                            vidx = __riscv_vfirst_m_b4(vstrong, vl);
                        }
                    }

                    rotateLineBuffer(lineBuf);
                }

                uint8_t *pmapLower = edge_map;
                uint32_t pmapDiff = (uint32_t)((edge_map + mapsize) - pmapLower);

                while (stack->size > 0)
                {
                    uint8_t *m = pop(stack);
                    if ((uint32_t)(m - pmapLower) < pmapDiff)
                    {
                        if (!m[-mapstep - 1])
                        {
                            *(m - mapstep - 1) = 2;
                            push(stack, m - mapstep - 1);
                        }
                        if (!m[-mapstep])
                        {
                            *(m - mapstep) = 2;
                            push(stack, m - mapstep);
                        }
                        if (!m[-mapstep + 1])
                        {
                            *(m - mapstep + 1) = 2;
                            push(stack, m - mapstep + 1);
                        }
                        if (!m[-1])
                        {
                            *(m - 1) = 2;
                            push(stack, m - 1);
                        }
                        if (!m[1])
                        {
                            *(m + 1) = 2;
                            push(stack, m + 1);
                        }
                        if (!m[mapstep - 1])
                        {
                            *(m + mapstep - 1) = 2;
                            push(stack, m + mapstep - 1);
                        }
                        if (!m[mapstep])
                        {
                            *(m + mapstep) = 2;
                            push(stack, m + mapstep);
                        }
                        if (!m[mapstep + 1])
                        {
                            *(m + mapstep + 1) = 2;
                            push(stack, m + mapstep + 1);
                        }
                    }
                    else
                    {
                        push(borderPeaksLocal, m);
                        ptrdiff_t mapstep2 = m < pmapLower ? mapstep : -mapstep;
                        if (!m[-1])
                        {
                            *(m - 1) = 2;
                            push(stack, m - 1);
                        }
                        if (!m[1])
                        {
                            *(m + 1) = 2;
                            push(stack, m + 1);
                        }
                        if (!m[mapstep2 - 1])
                        {
                            *(m + mapstep2 - 1) = 2;
                            push(stack, m + mapstep2 - 1);
                        }
                        if (!m[mapstep2])
                        {
                            *(m + mapstep2) = 2;
                            push(stack, m + mapstep2);
                        }
                        if (!m[mapstep2 + 1])
                        {
                            *(m + mapstep2 + 1) = 2;
                            push(stack, m + mapstep2 + 1);
                        }
                    }
                }

                if (borderPeaksLocal->size > 0)
                {
                    for (int i = 0; i < borderPeaksLocal->size; i++)
                    {
                        push(stack, borderPeaksLocal->data[i]);
                    }
                }

                for (int i = 0; i < height; i++)
                {
                    uint8_t *pdst = dst_data + i * dst_step;
                    uint8_t *pmap = map + i * mapstep + 1;
                    size_t vl = 0;
                    for (int j = 0; j < width; j += vl)
                    {
                        vl = __riscv_vsetvl_e8m8(width - j);
                        vuint8m8_t vres = __riscv_vle8_v_u8m8(pmap + j, vl);
                        vres = __riscv_vsrl_vx_u8m8(vres, 1, vl);
                        vres = __riscv_vneg_v_u8m8(vres, vl);
                        __riscv_vse8_v_u8m8(pdst + j, vres, vl);
                    }
                }

                freeLineBuffer(lineBuf);
                freeStack(borderPeaksLocal);
                freeStack(stack);
                free(dx_data);
                free(dy_data);
                free(edge_map);

                return CV_HAL_ERROR_OK;
            }
        }
    }
#endif // CV_HAL_RVV_1P0_ENABLED
} // namespace cv::rvv_hal::imgproc
