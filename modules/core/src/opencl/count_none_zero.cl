// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Itseez, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

__kernel void countNoneZero(__global const uchar * srcptr, int src_step, int src_offset, int rows, int cols,
                            __global uchar * bufptr, int buf_step, int buf_offset)
{
    int x = get_global_id(0);
#ifdef HALF_ROWS
    if (x < cols)
    {
        int src_index = mad24(x, (int)sizeof(srcT), src_offset);
        __global int * buf = (__global int *)(bufptr + buf_offset);
        int accum = 0;
        for (int y = 0; y < rows / 2; ++y, src_index += src_step)
        {
            __global const srcT *src = (__global const srcT *)(srcptr + src_index);
            if (0 != (*src))
                accum++;
        }
        buf[x] = accum;
    }
    else if (x < 2 * cols)
    {
        int y = rows / 2;
        int xtemp = x % cols;
        int src_index = mad24(y, src_step, mad24(xtemp, (int)sizeof(srcT), src_offset));
        __global int * buf = (__global int *)(bufptr + buf_offset);
        int accum = 0;
        for (; y < rows; ++y, src_index += src_step)
        {
            __global const srcT *src = (__global const srcT *)(srcptr + src_index);
            if (0 != (*src))
                accum++;
        }
        buf[x] = accum;
    }
#else
    if (x < cols)
    {
        int src_index = mad24(x, (int)sizeof(srcT), src_offset);
        __global int * buf = (__global int *)(bufptr + buf_offset);
        int accum = 0;
        for (int y = 0; y < rows; ++y, src_index += src_step)
        {
            __global const srcT *src = (__global const srcT *)(srcptr + src_index);
            if (0 != (*src))
                accum++;
        }
        buf[x] = accum;
    }
#endif
}

#ifndef BUF_COLS
#define BUF_COLS  32
#endif

__kernel void sumLine(__global uchar * bufptr, int buf_step, int buf_offset, int rows, int cols)
{
    int x = get_global_id(0);
    if (x < BUF_COLS)
    {
        int src_index = mad24(x, 4, buf_offset);
        int src_last = mad24(cols, 4, buf_offset);
         __global int * src = (__global int *)(bufptr + src_index);
         __global int * dst = (__global int *)(bufptr + src_index);
         __global int * srcend = (__global int *)(bufptr + src_last);
        int temp = 0;
        for (; src < srcend; src += BUF_COLS)
        {
            temp = add_sat(temp, src[0]);
        }
        dst[0] = temp;
    }
}
