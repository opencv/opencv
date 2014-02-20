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

__kernel void calcAlmostDist2Weight(__global int * almostDist2Weight, int almostMaxDist,
                                    FT almostDist2ActualDistMultiplier, int fixedPointMult,
                                    FT den, FT WEIGHT_THRESHOLD)
{
    int almostDist = get_global_id(0);

    if (almostDist < almostMaxDist)
    {
        FT dist = almostDist * almostDist2ActualDistMultiplier;
        int weight = convert_int_sat_rte(fixedPointMult * exp(-dist * den));

        if (weight < WEIGHT_THRESHOLD * fixedPointMult)
            weight = 0;

        almostDist2Weight[almostDist] = weight;
    }
}

#elif defined OP_CALC_FASTNLMEANS

#define noconvert

#define SEARCH_SIZE_SQ (SEARCH_SIZE * SEARCH_SIZE)

inline int calcDist(uchar_t a, uchar_t b)
{
    int_t diff = convert_int_t(a) - convert_int_t(b);
    int_t retval = diff * diff;

#if cn == 1
    return retval;
#elif cn == 2
    return retval.x + retval.y;
#else
#error "cn should be either 1 or 2"
#endif
}

inline int calcDistUpDown(uchar_t down_value, uchar_t down_value_t, uchar_t up_value, uchar_t up_value_t)
{
    int_t A = convert_int_t(down_value) - convert_int_t(down_value_t);
    int_t B = convert_int_t(up_value) - convert_int_t(up_value_t);
    int_t retval = (A - B) * (A + B);

#if cn == 1
    return retval;
#elif cn == 2
    return retval.x + retval.y;
#else
#error "cn should be either 1 or 2"
#endif
}

#define COND if (x == 0 && y == 0)

inline void calcFirstElementInRow(__global const uchar * src, int src_step, int src_offset,
                                  __local int * dists, int y, int x, int id,
                                  __global int * col_dists, __global int * up_col_dists)
{
    y -= TEMPLATE_SIZE2;
    int sx = x - SEARCH_SIZE2, sy = y - SEARCH_SIZE2;
    int col_dists_current_private[TEMPLATE_SIZE];

    for (int i = id, size = SEARCH_SIZE_SQ; i < size; i += CTA_SIZE)
    {
        int dist = 0, value;

        __global const uchar_t * src_template = (__global const uchar_t *)(src +
            mad24(sy + i / SEARCH_SIZE, src_step, mad24(cn, sx + i % SEARCH_SIZE, src_offset)));
        __global const uchar_t * src_current = (__global const uchar_t *)(src + mad24(y, src_step, mad24(cn, x, src_offset)));
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

            src_current = (__global const uchar_t *)((__global const uchar *)src_current + src_step);
            src_template = (__global const uchar_t *)((__global const uchar *)src_template + src_step);
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

    for (int i = id, size = SEARCH_SIZE_SQ; i < size; i += CTA_SIZE)
    {
        __global const uchar_t * src_current = (__global const uchar_t *)(src + mad24(y, src_step, mad24(cn, x, src_offset)));
        __global const uchar_t * src_template = (__global const uchar_t *)(src +
            mad24(sy + i / SEARCH_SIZE, src_step, mad24(cn, sx + i % SEARCH_SIZE, src_offset)));
        __global int * col_dists_current = col_dists + TEMPLATE_SIZE * i;

        int col_dist = 0;

        #pragma unroll
        for (int ty = 0; ty < TEMPLATE_SIZE; ++ty)
        {
            col_dist += calcDist(src_current[0], src_template[0]);

            src_current = (__global const uchar_t *)((__global const uchar *)src_current + src_step);
            src_template = (__global const uchar_t *)((__global const uchar *)src_template + src_step);
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

    uchar_t up_value = *(__global const uchar_t *)(src + mad24(sy_up, src_step, mad24(cn, sx, src_offset)));
    uchar_t down_value = *(__global const uchar_t *)(src + mad24(sy_down, src_step, mad24(cn, sx, src_offset)));

    sx -= SEARCH_SIZE2;
    sy_up -= SEARCH_SIZE2;
    sy_down -= SEARCH_SIZE2;

    for (int i = id, size = SEARCH_SIZE_SQ; i < size; i += CTA_SIZE)
    {
        int wx = i % SEARCH_SIZE, wy = i / SEARCH_SIZE;

        uchar_t up_value_t = *(__global const uchar_t *)(src + mad24(sy_up + wy, src_step, mad24(cn, sx + wx, src_offset)));
        uchar_t down_value_t = *(__global const uchar_t *)(src + mad24(sy_down + wy, src_step, mad24(cn, sx + wx, src_offset)));

        __global int * col_dists_current = col_dists + mad24(i, TEMPLATE_SIZE, first);
        __global int * up_col_dists_current = up_col_dists + mad24(x0, SEARCH_SIZE_SQ, i);

        int col_dist = up_col_dists_current[0] + calcDistUpDown(down_value, down_value_t, up_value, up_value_t);

        dists[i] += col_dist - col_dists_current[0];
        col_dists_current[0] = col_dist;
        up_col_dists_current[0] = col_dist;
    }
}

inline void convolveWindow(__global const uchar * src, int src_step, int src_offset,
                           __local int * dists, __global const int * almostDist2Weight,
                           __global uchar * dst, int dst_step, int dst_offset,
                           int y, int x, int id, __local int * weights_local,
                           __local int_t * weighted_sum_local, int almostTemplateWindowSizeSqBinShift)
{
    int sx = x - SEARCH_SIZE2, sy = y - SEARCH_SIZE2, weights = 0;
    int_t weighted_sum = (int_t)(0);

    for (int i = id, size = SEARCH_SIZE_SQ; i < size; i += CTA_SIZE)
    {
        int src_index = mad24(sy + i / SEARCH_SIZE, src_step, mad24(i % SEARCH_SIZE + sx, cn, src_offset));
        int_t src_value = convert_int_t(*(__global const uchar_t *)(src + src_index));

        int almostAvgDist = dists[i] >> almostTemplateWindowSizeSqBinShift;
        int weight = almostDist2Weight[almostAvgDist];

        weights += weight;
        weighted_sum += (int_t)(weight) * src_value;
    }

    if (id >= CTA_SIZE2)
    {
        int id2 = id - CTA_SIZE2;
        weights_local[id2] = weights;
        weighted_sum_local[id2] = weighted_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (id < CTA_SIZE2)
    {
        weights_local[id] += weights;
        weighted_sum_local[id] += weighted_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int lsize = CTA_SIZE2 >> 1; lsize > 2; lsize >>= 1)
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
        int dst_index = mad24(y, dst_step, mad24(cn, x, dst_offset));
        int_t weighted_sum_local_0 = weighted_sum_local[0] + weighted_sum_local[1] +
            weighted_sum_local[2] + weighted_sum_local[3];
        int weights_local_0 = weights_local[0] + weights_local[1] + weights_local[2] + weights_local[3];

        *(__global uchar_t *)(dst + dst_index) = convert_uchar_t(weighted_sum_local_0 / (int_t)(weights_local_0));
    }
}

__kernel void fastNlMeansDenoising(__global const uchar * src, int src_step, int src_offset,
                                   __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                                   __global const int * almostDist2Weight, __global uchar * buffer,
                                   int almostTemplateWindowSizeSqBinShift)
{
    int block_x = get_group_id(0), nblocks_x = get_num_groups(0);
    int block_y = get_group_id(1);
    int id = get_local_id(0), first;

    __local int dists[SEARCH_SIZE_SQ], weights[CTA_SIZE2];
    __local int_t weighted_sum[CTA_SIZE2];

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
            barrier(CLK_LOCAL_MEM_FENCE);

            convolveWindow(src, src_step, src_offset, dists, almostDist2Weight, dst, dst_step, dst_offset,
                y, x, id, weights, weighted_sum, almostTemplateWindowSizeSqBinShift);
        }
}

#endif
