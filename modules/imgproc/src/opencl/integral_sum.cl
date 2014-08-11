/*M///////////////////////////////////////////////////////////////////////////////////////
// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2014, Itseez, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//M*/

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

#ifndef LOCAL_SUM_SIZE
#define LOCAL_SUM_SIZE      16
#endif

#define LOCAL_SUM_STRIDE    (LOCAL_SUM_SIZE + 1)


kernel void integral_sum_cols(__global const uchar *src_ptr, int src_step, int src_offset, int rows, int cols,
                              __global uchar *buf_ptr, int buf_step, int buf_offset
#ifdef SUM_SQUARE
                              ,__global uchar *buf_sq_ptr, int buf_sq_step, int buf_sq_offset
#endif
                              )
{
    __local sumT lm_sum[LOCAL_SUM_STRIDE * LOCAL_SUM_SIZE];
#ifdef SUM_SQUARE
    __local sumSQT lm_sum_sq[LOCAL_SUM_STRIDE * LOCAL_SUM_SIZE];
#endif
    int lid = get_local_id(0);
    int gid = get_group_id(0);

    int x = get_global_id(0);
    int src_index = x + src_offset;

    sumT accum = 0;
#ifdef SUM_SQUARE
    sumSQT accum_sq = 0;
#endif
    for (int y = 0; y < rows; y += LOCAL_SUM_SIZE)
    {
        int lsum_index = lid;
        #pragma unroll
        for (int yin = 0; yin < LOCAL_SUM_SIZE; yin++, src_index+=src_step, lsum_index += LOCAL_SUM_STRIDE)
        {
            if ((x < cols) && (y + yin < rows))
            {
                __global const uchar *src = src_ptr + src_index;
                accum += src[0];
#ifdef SUM_SQUARE
                sumSQT temp = src[0] * src[0];
                accum_sq += temp;
#endif
            }
            lm_sum[lsum_index] = accum;
#ifdef SUM_SQUARE
            lm_sum_sq[lsum_index] = accum_sq;
#endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        //int buf_index = buf_offset + buf_step * LOCAL_SUM_COLS * gid + sizeof(sumT) * y + sizeof(sumT) * lid;
        int buf_index = mad24(buf_step, LOCAL_SUM_SIZE * gid, mad24((int)sizeof(sumT), y + lid, buf_offset));
#ifdef SUM_SQUARE
        int buf_sq_index = mad24(buf_sq_step, LOCAL_SUM_SIZE * gid, mad24((int)sizeof(sumSQT), y + lid, buf_sq_offset));
#endif

        lsum_index = LOCAL_SUM_STRIDE * lid;
        #pragma unroll
        for (int yin = 0; yin < LOCAL_SUM_SIZE; yin++, lsum_index ++)
        {
            __global sumT *buf = (__global sumT *)(buf_ptr + buf_index);
            buf[0] = lm_sum[lsum_index];
            buf_index += buf_step;
#ifdef SUM_SQUARE
            __global sumSQT *bufsq = (__global sumSQT *)(buf_sq_ptr + buf_sq_index);
            bufsq[0] = lm_sum_sq[lsum_index];
            buf_sq_index += buf_sq_step;
#endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

kernel void integral_sum_rows(__global const uchar *buf_ptr, int buf_step, int buf_offset,
#ifdef SUM_SQUARE
                              __global uchar *buf_sq_ptr, int buf_sq_step, int buf_sq_offset,
#endif
                              __global uchar *dst_ptr, int dst_step, int dst_offset, int rows, int cols
#ifdef SUM_SQUARE
                              ,__global uchar *dst_sq_ptr, int dst_sq_step, int dst_sq_offset
#endif
                              )
{
    __local sumT lm_sum[LOCAL_SUM_STRIDE * LOCAL_SUM_SIZE];
#ifdef SUM_SQUARE
    __local sumSQT lm_sum_sq[LOCAL_SUM_STRIDE * LOCAL_SUM_SIZE];
#endif
    int lid = get_local_id(0);
    int gid = get_group_id(0);

    int gs = get_global_size(0);

    int x = get_global_id(0);

    __global sumT *dst = (__global sumT *)(dst_ptr + dst_offset);
    for (int xin = x; xin < cols; xin += gs)
    {
        dst[xin] = 0;
    }
    dst_offset += dst_step;

    if (x < rows - 1)
    {
        dst = (__global sumT *)(dst_ptr + mad24(x, dst_step, dst_offset));
        dst[0] = 0;
    }

    int buf_index = mad24((int)sizeof(sumT), x, buf_offset);
    sumT accum = 0;

#ifdef SUM_SQUARE
    __global sumSQT *dst_sq = (__global sumT *)(dst_sq_ptr + dst_sq_offset);
    for (int xin = x; xin < cols; xin += gs)
    {
        dst_sq[xin] = 0;
    }
    dst_sq_offset += dst_sq_step;

    if (x < rows - 1)
    {
        dst_sq = (__global sumSQT *)(dst_sq_ptr + mad24(x, dst_sq_step, dst_sq_offset));
        dst_sq[0] = 0;
    }

    int buf_sq_index = mad24((int)sizeof(sumSQT), x, buf_sq_offset);
    sumSQT accum_sq = 0;
#endif

    for (int y = 1; y < cols; y += LOCAL_SUM_SIZE)
    {
        int lsum_index = lid;
        #pragma unroll
        for (int yin = 0; yin < LOCAL_SUM_SIZE; yin++, lsum_index += LOCAL_SUM_STRIDE)
        {
            __global const sumT *buf = (__global const sumT *)(buf_ptr + buf_index);
            accum += buf[0];
            lm_sum[lsum_index] = accum;
            buf_index += buf_step;
#ifdef SUM_SQUARE
            __global const sumSQT *buf_sq = (__global const sumSQT *)(buf_sq_ptr + buf_sq_index);
            accum_sq += buf_sq[0];
            lm_sum_sq[lsum_index] = accum_sq;
            buf_sq_index += buf_sq_step;
#endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (y + lid < cols)
        {
            //int dst_index = dst_offset + dst_step *  LOCAL_SUM_COLS * gid + sizeof(sumT) * y + sizeof(sumT) * lid;
            int dst_index = mad24(dst_step, LOCAL_SUM_SIZE * gid, mad24((int)sizeof(sumT), y + lid, dst_offset));
#ifdef SUM_SQUARE
            int dst_sq_index = mad24(dst_sq_step, LOCAL_SUM_SIZE * gid, mad24((int)sizeof(sumSQT), y + lid, dst_sq_offset));
#endif
            lsum_index = LOCAL_SUM_STRIDE * lid;
            int yin_max = min(rows - 1 -  LOCAL_SUM_SIZE * gid, LOCAL_SUM_SIZE);
            #pragma unroll
            for (int yin = 0; yin < yin_max; yin++, lsum_index++)
            {
                dst = (__global sumT *)(dst_ptr + dst_index);
                dst[0] = lm_sum[lsum_index];
                dst_index += dst_step;
#ifdef SUM_SQUARE
                dst_sq = (__global sumSQT *)(dst_sq_ptr + dst_sq_index);
                dst_sq[0] = lm_sum_sq[lsum_index];
                dst_sq_index += dst_sq_step;
#endif
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
