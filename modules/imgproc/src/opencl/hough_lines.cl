// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Itseez, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#ifdef MAKE_POINT_LIST

__kernel void make_point_list(__global const uchar * src_ptr, int src_step, int src_offset, int src_rows, int src_cols,
                              __global uchar * list_ptr, int list_step, int list_offset, __global int* global_offset)
{
    int x = get_local_id(0);
    int y = get_group_id(1);
    
    __local int l_index;
    __local int l_points[LOCAL_SIZE];
    __global const uchar * src = src_ptr + mad24(y, src_step, src_offset);
    __global int * list = (__global int*)(list_ptr + list_offset);

    if (x == 0)
        l_index = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (y < src_rows)
    {
        for (int i=x; i < src_cols; i+=GROUP_SIZE)
        {
            if (src[i])
            {
                int val = (y << 16) | i;
                int index = atomic_inc(&l_index);
                l_points[index] = val;
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    
    int offset;
    if (x == 0)
        offset = atomic_add(global_offset, l_index);

    barrier(CLK_LOCAL_MEM_FENCE);
    
    list += offset;
    for (int i=x; i < l_index; i+=GROUP_SIZE)
    {
        list[i] = l_points[i];
    }
}

#elif defined FILL_ACCUM

__kernel void fill_accum(__global const uchar * list_ptr, int list_step, int list_offset,
                         __global uchar * accum_ptr, int accum_step, int accum_offset, int accum_rows, int accum_cols,
                         int count, float irho, float theta, int numrho)
{
    int theta_idx = get_global_id(0);
    int count_idx = get_global_id(1);
    float cosVal;
    float sinVal = sincos(theta * theta_idx, &cosVal);
    sinVal *= irho;
    cosVal *= irho;

    __global const int * list = (__global const int*)(list_ptr + list_offset);
    __global int* accum = (__global int*)(accum_ptr + mad24(theta_idx, accum_step, accum_offset));
    const int shift = (numrho - 1) / 2;

    for (int i = count_idx; i < count; i += GROUP_SIZE)
    {
        const int val = list[i];
        const int x = (val & 0xFFFF);
        const int y = (val >> 16) & 0xFFFF;

        int r = round(x * cosVal + y * sinVal) + shift;
        atomic_inc(accum + r + 1);
    }
}

#endif

