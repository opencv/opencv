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

            LineBuffer *createLineBuffer(size_t width, size_t align);

            void rotateLineBuffer(LineBuffer *buf);

            void freeLineBuffer(LineBuffer *buf);

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

            static Stack *canny_createStack(int capacity);
            static void canny_push(Stack *stack, uint8_t *item);
            static uint8_t *canny_pop(Stack *stack);
            static void freeStack(Stack *stack);

            static Stack *canny_createStack(int capacity)
            {
                Stack *stack = (Stack *)malloc(sizeof(Stack));
                stack->data = (uint8_t **)malloc(capacity * sizeof(uint8_t *));
                stack->capacity = capacity;
                stack->size = 0;
                return stack;
            }

            static void canny_push(Stack *stack, uint8_t *item)
            {
                if (stack->size < stack->capacity)
                {
                    stack->data[stack->size++] = item;
                }
            }

            static uint8_t *canny_pop(Stack *stack)
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
                return CV_HAL_ERROR_NOT_IMPLEMENTED;
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

                int low = (int)low_thresh;
                low_thresh = low - (low > low_thresh);
                int high = (int)high_thresh;
                high_thresh = high - (high > high_thresh);

                int16_t *dx_data = (int16_t *)malloc(height * width * sizeof(int16_t));
                int16_t *dy_data = (int16_t *)malloc(height * width * sizeof(int16_t));
                if (!dx_data || !dy_data)
                    return CV_HAL_ERROR_NOT_IMPLEMENTED;

                const size_t grad_step = width * sizeof(int16_t);

                int ret = sobel(src_data, src_step, (uint8_t *)dx_data, grad_step,
                                width, height, CV_8U, CV_16S, 1,
                                0, 0, 0, 0, 1, 0, ksize, 1.0, 0.0, BORDER_REPLICATE);
                if (ret != CV_HAL_ERROR_OK)
                {
                    free(dx_data);
                    free(dy_data);
                    return CV_HAL_ERROR_NOT_IMPLEMENTED;
                }

                ret = sobel(src_data, src_step, (uint8_t *)dy_data, grad_step,
                            width, height, CV_8U, CV_16S, 1,
                            0, 0, 0, 0, 0, 1, ksize, 1.0, 0.0, BORDER_REPLICATE);
                if (ret != CV_HAL_ERROR_OK)
                {
                    free(dx_data);
                    free(dy_data);
                    return CV_HAL_ERROR_NOT_IMPLEMENTED;
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

                Stack *stack = canny_createStack(width * height);
                Stack *borderPeaksLocal = canny_createStack(width * height);
                if (!stack || !borderPeaksLocal)
                {
                    free(dx_data);
                    free(dy_data);
                    free(edge_map);
                    return CV_HAL_ERROR_NOT_IMPLEMENTED;
                }

                memset(edge_map, 1, mapstep);
                memset(edge_map + (height + 1) * mapstep, 1, mapstep);

                for (int i = 1; i <= height; i++)
                {
                    edge_map[i * mapstep] = 1;
                    edge_map[i * mapstep + width + 1] = 1;
                }

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
                        vint32m8_t vdx_abs = __riscv_vmul_vx_i32m8_m(mask_dx, vdx, -1, vl);
                        vint32m8_t vdy_abs = __riscv_vmul_vx_i32m8_m(mask_dy, vdy, -1, vl);

                        vdy_abs = __riscv_vsll_vx_i32m8(vdy_abs, 15, vl);

                        // Calculate threshold values
                        vint32m8_t vtg22x = __riscv_vmul_vx_i32m8(vdx_abs, TG22, vl);
                        vint32m8_t vtg67x = __riscv_vadd_vv_i32m8(vtg22x, __riscv_vsll_vx_i32m8(vdx_abs, 16, vl), vl);

                        // Create masks for different angle ranges
                        vbool4_t mask1 = __riscv_vmslt_vv_i32m8_b4(vdy_abs, vtg22x, vl);
                        vbool4_t mask2 = __riscv_vmsgt_vv_i32m8_b4(vdy_abs, vtg67x, vl);
                        vbool4_t mask3 = __riscv_vmnot_m_b4(__riscv_vmor_mm_b4(mask1, mask2, vl), vl);

                        // Load neighbor pixels for all conditions
                        vint32m8_t prev_curr = __riscv_vle32_v_i32m8(lineBuf->curr + (j > 0 ? j - 1 : j), vl);
                        vint32m8_t next_curr = __riscv_vle32_v_i32m8(lineBuf->curr + j + 1, vl);
                        vint32m8_t prev_line = __riscv_vle32_v_i32m8(lineBuf->prev + j, vl);
                        vint32m8_t next_line = __riscv_vle32_v_i32m8(lineBuf->next + j, vl);

                        // Condition 1: Horizontal/Vertical edges (compare left/right)
                        vbool4_t cond1_max = __riscv_vmand_mm_b4(
                            __riscv_vmsgt_vv_i32m8_b4(vm, prev_curr, vl),
                            __riscv_vmsge_vv_i32m8_b4(vm, next_curr, vl),
                            vl);

                        // Condition 2: Diagonal edges (compare top/bottom)
                        vbool4_t cond2_max = __riscv_vmand_mm_b4(
                            __riscv_vmsgt_vv_i32m8_b4(vm, prev_line, vl),
                            __riscv_vmsge_vv_i32m8_b4(vm, next_line, vl),
                            vl);

                        // Condition 3: Other diagonals (calculate s)
                        vint32m8_t vxor = __riscv_vxor_vv_i32m8(vdx, vdy, vl);
                        vbool4_t s_mask = __riscv_vmslt_vx_i32m8_b4(vxor, 0, vl);

                        vint32m8_t prev_s1 = __riscv_vle32_v_i32m8(lineBuf->prev + (j > 0 ? j - 1 : j), vl);
                        vint32m8_t prev_s2 = __riscv_vle32_v_i32m8(lineBuf->prev + j + 1, vl);
                        vint32m8_t next_s1 = __riscv_vle32_v_i32m8(lineBuf->next + j + 1, vl);
                        vint32m8_t next_s2 = __riscv_vle32_v_i32m8(lineBuf->next + (j > 0 ? j - 1 : j), vl);

                        vint32m8_t prev_sel = __riscv_vmerge_vvm_i32m8(prev_s1, prev_s2, s_mask, vl);
                        vint32m8_t next_sel = __riscv_vmerge_vvm_i32m8(next_s1, next_s2, s_mask, vl);

                        vbool4_t cond3_max = __riscv_vmand_mm_b4(
                            __riscv_vmsgt_vv_i32m8_b4(vm, prev_sel, vl),
                            __riscv_vmsgt_vv_i32m8_b4(vm, next_sel, vl),
                            vl);

                        // Combine results from all conditions
                        vbool4_t vmax = __riscv_vmor_mm_b4(
                            __riscv_vmand_mm_b4(mask1, cond1_max, vl),
                            __riscv_vmand_mm_b4(mask2, cond2_max, vl),
                            vl);
                        vmax = __riscv_vmor_mm_b4(
                            vmax,
                            __riscv_vmand_mm_b4(mask3, cond3_max, vl),
                            vl);

                        // Threshold checks
                        vbool4_t vlow = __riscv_vmsgt_vx_i32m8_b4(vm, low_thresh, vl);
                        vbool4_t vhigh = __riscv_vmsgt_vx_i32m8_b4(vm, high_thresh, vl);
                        vbool4_t valid_edges = __riscv_vmand_mm_b4(vmax, vlow, vl);
                        vbool4_t strong_edges = __riscv_vmand_mm_b4(valid_edges, vhigh, vl);

                        // Generate result map
                        vuint8m2_t vres = __riscv_vmv_v_x_u8m2(1, vl);
                        vres = __riscv_vmerge_vxm_u8m2(vres, 0, valid_edges, vl);
                        vres = __riscv_vmerge_vxm_u8m2(vres, 2, strong_edges, vl);
                        __riscv_vse8_v_u8m2(pmap + j, vres, vl);

                        // canny_Push strong edges to stack
                        int32_t vidx = __riscv_vfirst_m_b4(strong_edges, vl);
                        while (vidx >= 0)
                        {
                            canny_push(stack, pmap + j + vidx);
                            strong_edges = __riscv_vmand_mm_b4(strong_edges,
                                                               __riscv_vmclr_m_b4(vl), vl);
                            vidx = __riscv_vfirst_m_b4(strong_edges, vl);
                        }
                    }

                    rotateLineBuffer(lineBuf);
                }

                uint8_t *pmapLower = edge_map;
                uint32_t pmapDiff = (uint32_t)((edge_map + mapsize) - pmapLower);

                while (stack->size > 0)
                {
                    uint8_t *m = canny_pop(stack);

                    if ((uint64_t)m < (uint64_t)pmapLower || (uint64_t)m >= (uint64_t)(pmapLower + pmapDiff))
                        continue;

                    const int offsets[8] = {
                        -mapstep - 1, -mapstep, -mapstep + 1,
                        -1, +1,
                        +mapstep - 1, +mapstep, +mapstep + 1};

                    for (int k = 0; k < 8; k++)
                    {
                        uint8_t *neighbor = m + offsets[k];
                        if ((uint64_t)neighbor < (uint64_t)edge_map ||
                            (uint64_t)neighbor >= (uint64_t)(edge_map + mapsize))
                            continue;

                        if (*neighbor == 0)
                        {
                            *neighbor = 2;
                            canny_push(stack, neighbor);
                        }
                    }
                }

                if (borderPeaksLocal->size > 0)
                {
                    for (int i = 0; i < borderPeaksLocal->size; i++)
                    {
                        canny_push(stack, borderPeaksLocal->data[i]);
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
                        vres = __riscv_vsub_vv_u8m8(__riscv_vmv_v_x_u8m8(0, vl), vres, vl);
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
