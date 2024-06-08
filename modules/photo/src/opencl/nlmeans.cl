// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#ifdef cl_amd_printf
#pragma OPENCL_EXTENSION cl_amd_printf:enable
#endif

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif


#ifdef OP_CALC_WEIGHTS

__kernel void calcAlmostDist2Weight(__global wlut_t * almostDist2Weight, int almostMaxDist,
                                    FT almostDist2ActualDistMultiplier, int fixedPointMult,
                                    w_t den, FT WEIGHT_THRESHOLD)
{
    int almostDist = get_global_id(0);

    if (almostDist < almostMaxDist)
    {
        FT dist = almostDist * almostDist2ActualDistMultiplier;
#ifdef ABS
        w_t w = exp((w_t)(-dist*dist) * den);
#else
        w_t w = exp((w_t)(-dist) * den);
#endif
        wlut_t weight = convert_wlut_t(fixedPointMult * (isnan(w) ? (w_t)1.0 : w));
        almostDist2Weight[almostDist] =
            weight < (wlut_t)(WEIGHT_THRESHOLD * fixedPointMult) ? (wlut_t)0 : weight;
    }
}

#elif defined OP_CALC_FASTNLMEANS

#define noconvert

#define SEARCH_SIZE_SQ (SEARCH_SIZE * SEARCH_SIZE)

inline int calcDist(pixel_t a, pixel_t b)
{
#ifdef ABS
    int_t retval = convert_int_t(abs_diff(a, b));
#else
    int_t diff = convert_int_t(a) - convert_int_t(b);
    int_t retval = diff * diff;
#endif

#if cn == 1
    return retval;
#elif cn == 2
    return retval.x + retval.y;
#elif cn == 3
    return retval.x + retval.y + retval.z;
#elif cn == 4
    return retval.x + retval.y + retval.z + retval.w;
#else
#error "cn should be either 1, 2, 3 or 4"
#endif
}

#ifdef ABS
inline int calcDistUpDown(pixel_t down_value, pixel_t down_value_t, pixel_t up_value, pixel_t up_value_t)
{
    return calcDist(down_value, down_value_t) - calcDist(up_value, up_value_t);
}
#else
inline int calcDistUpDown(pixel_t down_value, pixel_t down_value_t, pixel_t up_value, pixel_t up_value_t)
{
    int_t A = convert_int_t(down_value) - convert_int_t(down_value_t);
    int_t B = convert_int_t(up_value) - convert_int_t(up_value_t);
    int_t retval = (A - B) * (A + B);

#if cn == 1
    return retval;
#elif cn == 2
    return retval.x + retval.y;
#elif cn == 3
    return retval.x + retval.y + retval.z;
#elif cn == 4
    return retval.x + retval.y + retval.z + retval.w;
#else
#error "cn should be either 1, 2, 3 or 4"
#endif
}
#endif

#define COND if (x == 0 && y == 0)

inline void calcFirstElementInRow(__global const uchar * src, int src_step, int src_offset,
                                  __local int * dists, int y, int x, int id,
                                  __global int * col_dists, __global int * up_col_dists)
{
    y -= TEMPLATE_SIZE2;
    int sx = x - SEARCH_SIZE2, sy = y - SEARCH_SIZE2;
    int col_dists_current_private[TEMPLATE_SIZE];

    for (int i = id; i < SEARCH_SIZE_SQ; i += CTA_SIZE)
    {
        int dist = 0, value;

        __global const pixel_t * src_template = (__global const pixel_t *)(src +
            mad24(sy + i / SEARCH_SIZE, src_step, mad24(psz, sx + i % SEARCH_SIZE, src_offset)));
        __global const pixel_t * src_current = (__global const pixel_t *)(src + mad24(y, src_step, mad24(psz, x, src_offset)));
        __global int * col_dists_current = col_dists + i * TEMPLATE_SIZE;

        #pragma unroll
        for (int j = 0; j < TEMPLATE_SIZE; ++j)
            col_dists_current_private[j] = 0;

        for (int ty = 0; ty < TEMPLATE_SIZE; ++ty)
        {
            #pragma unroll
            for (int tx = -TEMPLATE_SIZE2; tx <= TEMPLATE_SIZE2; ++tx)
            {
                value = calcDist(src_template[tx], src_current[tx]);

                col_dists_current_private[tx + TEMPLATE_SIZE2] += value;
                dist += value;
            }

            src_current = (__global const pixel_t *)((__global const uchar *)src_current + src_step);
            src_template = (__global const pixel_t *)((__global const uchar *)src_template + src_step);
        }

        #pragma unroll
        for (int j = 0; j < TEMPLATE_SIZE; ++j)
            col_dists_current[j] = col_dists_current_private[j];

        dists[i] = dist;
        up_col_dists[0 + i] = col_dists[TEMPLATE_SIZE - 1];
    }
}

inline void calcElementInFirstRow(__global const uchar * src, int src_step, int src_offset,
                                  __local int * dists, int y, int x0, int x, int id, int first,
                                  __global int * col_dists, __global int * up_col_dists)
{
    x += TEMPLATE_SIZE2;
    y -= TEMPLATE_SIZE2;
    int sx = x - SEARCH_SIZE2, sy = y - SEARCH_SIZE2;

    for (int i = id; i < SEARCH_SIZE_SQ; i += CTA_SIZE)
    {
        __global const pixel_t * src_current = (__global const pixel_t *)(src + mad24(y, src_step, mad24(psz, x, src_offset)));
        __global const pixel_t * src_template = (__global const pixel_t *)(src +
            mad24(sy + i / SEARCH_SIZE, src_step, mad24(psz, sx + i % SEARCH_SIZE, src_offset)));
        __global int * col_dists_current = col_dists + TEMPLATE_SIZE * i;

        int col_dist = 0;

        #pragma unroll
        for (int ty = 0; ty < TEMPLATE_SIZE; ++ty)
        {
            col_dist += calcDist(src_current[0], src_template[0]);

            src_current = (__global const pixel_t *)((__global const uchar *)src_current + src_step);
            src_template = (__global const pixel_t *)((__global const uchar *)src_template + src_step);
        }

        dists[i] += col_dist - col_dists_current[first];
        col_dists_current[first] = col_dist;
        up_col_dists[mad24(x0, SEARCH_SIZE_SQ, i)] = col_dist;
    }
}

inline void calcElement(__global const uchar * src, int src_step, int src_offset,
                        __local int * dists, int y, int x0, int x, int id, int first,
                        __global int * col_dists, __global int * up_col_dists)
{
    int sx = x + TEMPLATE_SIZE2;
    int sy_up = y - TEMPLATE_SIZE2 - 1;
    int sy_down = y + TEMPLATE_SIZE2;

    pixel_t up_value = *(__global const pixel_t *)(src + mad24(sy_up, src_step, mad24(psz, sx, src_offset)));
    pixel_t down_value = *(__global const pixel_t *)(src + mad24(sy_down, src_step, mad24(psz, sx, src_offset)));

    sx -= SEARCH_SIZE2;
    sy_up -= SEARCH_SIZE2;
    sy_down -= SEARCH_SIZE2;

    for (int i = id; i < SEARCH_SIZE_SQ; i += CTA_SIZE)
    {
        int wx = i % SEARCH_SIZE, wy = i / SEARCH_SIZE;

        pixel_t up_value_t = *(__global const pixel_t *)(src + mad24(sy_up + wy, src_step, mad24(psz, sx + wx, src_offset)));
        pixel_t down_value_t = *(__global const pixel_t *)(src + mad24(sy_down + wy, src_step, mad24(psz, sx + wx, src_offset)));

        __global int * col_dists_current = col_dists + mad24(i, TEMPLATE_SIZE, first);
        __global int * up_col_dists_current = up_col_dists + mad24(x0, SEARCH_SIZE_SQ, i);

        int col_dist = up_col_dists_current[0] + calcDistUpDown(down_value, down_value_t, up_value, up_value_t);

        dists[i] += col_dist - col_dists_current[0];
        col_dists_current[0] = col_dist;
        up_col_dists_current[0] = col_dist;
    }
}

inline void convolveWindow(__global const uchar * src, int src_step, int src_offset,
                           __local int * dists, __global const wlut_t * almostDist2Weight,
                           __global uchar * dst, int dst_step, int dst_offset,
                           int y, int x, int id, __local weight_t * weights_local,
                           __local sum_t * weighted_sum_local, int almostTemplateWindowSizeSqBinShift)
{
    int sx = x - SEARCH_SIZE2, sy = y - SEARCH_SIZE2;
    weight_t weights = (weight_t)0;
    sum_t weighted_sum = (sum_t)0;

    for (int i = id; i < SEARCH_SIZE_SQ; i += CTA_SIZE)
    {
        int src_index = mad24(sy + i / SEARCH_SIZE, src_step, mad24(i % SEARCH_SIZE + sx, psz, src_offset));
        sum_t src_value = convert_sum_t(*(__global const pixel_t *)(src + src_index));

        int almostAvgDist = dists[i] >> almostTemplateWindowSizeSqBinShift;
        weight_t weight = convert_weight_t(almostDist2Weight[almostAvgDist]);

        weights += weight;
        weighted_sum += (sum_t)weight * src_value;
    }

    weights_local[id] = weights;
    weighted_sum_local[id] = weighted_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int lsize = CTA_SIZE >> 1; lsize > 2; lsize >>= 1)
    {
        if (id < lsize)
        {
           int id2 = lsize + id;
           weights_local[id] += weights_local[id2];
           weighted_sum_local[id] += weighted_sum_local[id2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (id == 0)
    {
        int dst_index = mad24(y, dst_step, mad24(psz, x, dst_offset));
        sum_t weighted_sum_local_0 = weighted_sum_local[0] + weighted_sum_local[1] +
            weighted_sum_local[2] + weighted_sum_local[3];
        weight_t weights_local_0 = weights_local[0] + weights_local[1] + weights_local[2] + weights_local[3];

        *(__global pixel_t *)(dst + dst_index) = convert_pixel_t(weighted_sum_local_0 / (sum_t)weights_local_0);
    }
}

__kernel void fastNlMeansDenoising(__global const uchar * src, int src_step, int src_offset,
                                   __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                                   __global const wlut_t * almostDist2Weight, __global uchar * buffer,
                                   int almostTemplateWindowSizeSqBinShift)
{
    int block_x = get_group_id(0), nblocks_x = get_num_groups(0);
    int block_y = get_group_id(1);
    int id = get_local_id(0), first;

    __local int dists[SEARCH_SIZE_SQ];
    __local weight_t weights[CTA_SIZE];
    __local sum_t weighted_sum[CTA_SIZE];

    int x0 = block_x * BLOCK_COLS, x1 = min(x0 + BLOCK_COLS, dst_cols);
    int y0 = block_y * BLOCK_ROWS, y1 = min(y0 + BLOCK_ROWS, dst_rows);

    // for each group we need SEARCH_SIZE_SQ * TEMPLATE_SIZE integer buffer for storing part column sum for current element
    // and SEARCH_SIZE_SQ * BLOCK_COLS integer buffer for storing last column sum for each element of search window of up row
    int block_data_start = SEARCH_SIZE_SQ * (mad24(block_y, dst_cols, x0) + mad24(block_y, nblocks_x, block_x) * TEMPLATE_SIZE);
    __global int * col_dists = (__global int *)(buffer + block_data_start * sizeof(int));
    __global int * up_col_dists = col_dists + SEARCH_SIZE_SQ * TEMPLATE_SIZE;

    for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x)
        {
            if (x == x0)
            {
                calcFirstElementInRow(src, src_step, src_offset, dists, y, x, id, col_dists, up_col_dists);
                first = 0;
            }
            else
            {
                if (y == y0)
                    calcElementInFirstRow(src, src_step, src_offset, dists, y, x - x0, x, id, first, col_dists, up_col_dists);
                else
                    calcElement(src, src_step, src_offset, dists, y, x - x0, x, id, first, col_dists, up_col_dists);

                first = (first + 1) % TEMPLATE_SIZE;
            }

            convolveWindow(src, src_step, src_offset, dists, almostDist2Weight, dst, dst_step, dst_offset,
                y, x, id, weights, weighted_sum, almostTemplateWindowSizeSqBinShift);
        }
}

#endif
