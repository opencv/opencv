// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#ifdef OP_CALC_WEIGHTS

__kernel void calcAlmostDist2Weight(__global int * almostDist2Weight, int almostMaxDist,
                                    float almostDist2ActualDistMultiplier, int fixedPointMult,
                                    float den, float WEIGHT_THRESHOLD)
{
    int almostDist = get_global_id(0);

    if (almostDist < almostMaxDist)
    {
        float dist = almostDist * almostDist2ActualDistMultiplier;
        int weight = convert_int_sat_rte(fixedPointMult * exp(-dist * den));

        if (weight < WEIGHT_THRESHOLD * fixedPointMult)
            weight = 0;

        almostDist2Weight[almostDist] = weight;
    }
}

#elif defined OP_CALC_FASTNLMEANS

#define SEARCH_SIZE_SQ (SEARCH_SIZE * SEARCH_SIZE)

inline int_t calcDist(uchar_t a, uchar_t b)
{
    int_t diff = convert_int_t(a) -convert_int_t(b);
    return diff * diff;
}

inline void calcFirstElementInRow(__global const uchar * src, int src_step, int src_offset,
                                  __local int_t * dists, int y, int x, int id,
                                  __global int_t * col_dists, __global int_t * up_col_dists)
{
    int sx = x - SEARCH_SIZE2, sy = y - SEARCH_SIZE2;

    for (int i = 0, size = SEARCH_SIZE_SQ; i < size; i += CTA_SIZE)
    {
        int_t dist = (int_t)(0), value;

        sx += i % SEARCH_SIZE;
        sy += i / SEARCH_SIZE;

        __global const uchar_t * src_template = (__global const uchar_t *)(src + mad24(sy, src_step, mad24(cn, x, src_offset)));
        __global const uchar_t * src_current = (__global const uchar_t *)(src + mad24(y, src_step, mad24(cn, x, src_offset)));
        __global int_t * col_dists_current = col_dists + i * TEMPLATE_SIZE;

        #pragma unroll
        for (int j = 0; j < TEMPLATE_SIZE; ++j)
            col_dists_current[j] = (int_t)(0);

        #pragma unroll
        for (int ty = -TEMPLATE_SIZE2; ty <= TEMPLATE_SIZE2; ++ty)
        {
            #pragma unroll
            for (int tx = -TEMPLATE_SIZE2; tx <= TEMPLATE_SIZE2; ++tx)
            {
                value = calcDist(src_template[tx], src_current[tx]);

                col_dists_current[tx + TEMPLATE_SIZE2] += value;
                dist += value;
            }

            src_current += src_step;
            src_template += src_step;
        }

        dists[i] = dist;
        up_col_dists[i] = col_dists[TEMPLATE_SIZE - 1];
    }
}

inline void calcElementInFirstRow(__global const uchar * src, int src_step, int src_offset,
                                  __local int_t * dists, int y, int x, int id, int first,
                                  __global int_t * col_dists, __global int_t * up_col_dists)
{
    x += TEMPLATE_SIZE2;
    int sx = x - SEARCH_SIZE2, sy = y - SEARCH_SIZE2;

    for (int i = 0, size = SEARCH_SIZE_SQ; i < size; i += CTA_SIZE)
    {
        sx += i % SEARCH_SIZE;
        sy += i / SEARCH_SIZE;

        __global const uchar_t * src_current = (__global const uchar_t *)(src + mad24(y, src_step, mad24(cn, x, src_offset)));
        __global const uchar_t * src_template = (__global const uchar_t *)(src + mad24(sy, src_step, mad24(cn, x, src_offset)));
        __global int_t * col_dists_current = col_dists + TEMPLATE_SIZE * i;

        int_t value;
        dists[id] -= col_dists_current[first];
        col_dists_current[first] = (int_t)(0);

        #pragma unroll
        for (int ty = -TEMPLATE_SIZE2; ty <= TEMPLATE_SIZE2; ++ty)
        {
            value = calcDist(src_current[0], src_template[0]);
            col_dists_current[first] += value;

            src_current += src_step;
            src_template += src_step;
        }

        dists[id] += col_dists_current[first];
        up_col_dists[id] = col_dists_current[first];
    }
}

inline void calcElement(__global const uchar * src, int src_step, int src_offset,
                        __local int_t * dists, int y, int x, int id, int first,
                        __global int_t * col_dists, __global int_t * up_col_dists)
{
    int sx_up = x + TEMPLATE_SIZE2, sy_up = y - TEMPLATE_SIZE2 - 1;
    int sx_down = x + TEMPLATE_SIZE2, sy_down = y + TEMPLATE_SIZE2;

    uchar_t up_value = *(__global const uchar_t *)(src + mad24(sy_up, src_step, mad24(cn, sx_up, src_offset)));
    uchar_t down_value = *(__global const uchar_t *)(src + mad24(sy_down, src_step, mad24(cn, sx_down, src_offset)));

    for (int i = 0, size = SEARCH_SIZE_SQ; i < size; i += CTA_SIZE)
    {
        int wx = i % SEARCH_SIZE;
        int wy = i / SEARCH_SIZE;

        sx_up += wx, sx_down += wx;
        sy_up += wy, sy_down += wy;

        uchar_t up_value_t = *(__global const uchar_t *)(src + mad24(sy_up, src_step, mad24(cn, sx_up, src_offset)));
        uchar_t down_value_t = *(__global const uchar_t *)(src + mad24(sy_down, src_step, mad24(cn, sx_down, src_offset)));

        __global int_t * col_dists_current = col_dists + i * TEMPLATE_SIZE;
        __global int_t * up_col_dists_current = up_col_dists + i;

        dists[i] -= col_dists_current[first];
        col_dists_current[first] = up_col_dists_current[id] + calcDist(down_value, down_value_t) - calcDist(up_value, up_value_t);
        dists[i] += col_dists_current[first];
        up_col_dists_current[id] = col_dists_current[first];
    }
}

inline void convolveWindow(__global const uchar * src, int src_step, int src_offset,
                           __local int * dists, __global const int * almostDist2Weight,
                           __global uchar * dst, int dst_step, int dst_offset,
                           int y, int x, int id, __local int * weights_local,
                           __local int * weighted_sum_local, int almostTemplateWindowSizeSqBinShift)
{
    int sx = x - SEARCH_SIZE2, sy = y - SEARCH_SIZE2, weights = 0;
    int_t weighted_sum = (int_t)(0);

    for (int i = 0, size = SEARCH_SIZE_SQ; i < size; i += id)
    {
        int src_index = mad24(sy + i / SEARCH_SIZE, src_step, (i % SEARCH_SIZE + sx) * cn + src_offset);
        __global const uchar_t * src_search = (__global const uchar_t *)(src + src_index);

        int almostAvgDist = dists[i] >> almostTemplateWindowSizeSqBinShift;
        int weight = almostDist2Weight[almostAvgDist];

        weights += weight;
        weighted_sum += (int_t)(weight) * convert_int_t(src_search[0]);
    }

    if (id >= CTA_SIZE2)
    {
        weights_local[id - CTA_SIZE2] = weights;
        weighted_sum_local[id - CTA_SIZE2] = weighted_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (id < CTA_SIZE2)
    {
        weights_local[id] += weights;
        weighted_sum_local[id] += weighted_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int lsize = CTA_SIZE2 >> 1; lsize >= 4; lsize >>= 1)
    {
        if (id < lsize)
        {
           int id2 = lsize + id;
           weights_local[id] = weights + weights_local[id2];
           weighted_sum_local[id] = weighted_sum + weighted_sum_local[id2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (id == 0)
    {
        int dst_index = mad24(y, dst_step, dst_offset + x * cn);

        int_t weights_local_0 = (int_t)(weights_local[0] + weights_local[1] + weights_local[2] + weights_local[3]);
        int_t weighted_sum_local_0 = weighted_sum_local[0] + weighted_sum_local[1] + weighted_sum_local[2] + weighted_sum_local[3];

        *(__global uchar_t *)(dst + dst_index) = convert_uchar_t((weighted_sum_local_0 + weights_local_0 >> 1) / weights_local_0);
    }
}

__kernel void fastNlMeansDenoising(__global const uchar * src, int src_step, int src_offset,
                                   __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                                   __global const int * almostDist2Weight, int nblocksy, int nblocksx,
                                   __global uchar * buffer, int almostTemplateWindowSizeSqBinShift)
{
    int block_x = get_global_id(0);
    int block_y = get_global_id(1);
    int id = get_local_id(0), first;

    __local int_t dists[SEARCH_SIZE_SQ], weighted_sum[CTA_SIZE2];
    __local int weights[CTA_SIZE2];

    int block_data_start = mad24(block_y, nblocksx, block_x) * SEARCH_SIZE_SQ * (TEMPLATE_SIZE + BLOCK_COLS);
    __global int_t * col_dists = (__global int_t *)(buffer + block_data_start * sizeof(int_t));
    __global int_t * up_col_dists = (__global int_t *)(buffer + sizeof(int_t) * (block_data_start + SEARCH_SIZE_SQ * TEMPLATE_SIZE));

    if (block_x < nblocksx && block_y < nblocksy)
    {
        int x0 = block_x * BLOCK_COLS, x1 = min(x0 + BLOCK_COLS, dst_cols);
        int y0 = block_y * BLOCK_ROWS, y1 = min(y0 + BLOCK_ROWS, dst_rows);

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
                        calcElementInFirstRow(src, src_step, src_offset, dists, y, x, id, first, col_dists, up_col_dists);
                    else
                    {
                        calcElement(src, src_step, src_offset, dists, y, x, id, first, col_dists, up_col_dists);
                        first = (first + 1) % TEMPLATE_SIZE;
                    }

                    convolveWindow(src, src_step, src_offset, dists, almostDist2Weight, dst, dst_step, dst_offset,
                        y, x, id, weights, weighted_sum, almostTemplateWindowSizeSqBinShift);
                }
            }
    }
}

#endif
