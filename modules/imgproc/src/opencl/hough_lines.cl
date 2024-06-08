// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Itseez, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#define ACCUM(ptr) *((__global int*)(ptr))

#ifdef MAKE_POINTS_LIST

__kernel void make_point_list(__global const uchar * src_ptr, int src_step, int src_offset, int src_rows, int src_cols,
                              __global uchar * list_ptr, int list_step, int list_offset, __global int* global_offset)
{
    int x = get_local_id(0);
    int y = get_group_id(1);

    __local int l_index, l_offset;
    __local int l_points[LOCAL_SIZE];
    __global const uchar * src = src_ptr + mad24(y, src_step, src_offset);
    __global int * list = (__global int*)(list_ptr + list_offset);

    if (x == 0)
        l_index = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (y < src_rows)
    {
        y <<= 16;

        for (int i=x; i < src_cols; i+=GROUP_SIZE)
        {
            if (src[i])
            {
                int val = y | i;
                int index = atomic_inc(&l_index);
                l_points[index] = val;
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x == 0)
        l_offset = atomic_add(global_offset, l_index);

    barrier(CLK_LOCAL_MEM_FENCE);

    list += l_offset;
    for (int i=x; i < l_index; i+=GROUP_SIZE)
    {
        list[i] = l_points[i];
    }
}

#elif defined FILL_ACCUM_GLOBAL

__kernel void fill_accum_global(__global const uchar * list_ptr, int list_step, int list_offset,
                                __global uchar * accum_ptr, int accum_step, int accum_offset,
                                int total_points, float irho, float theta, int numrho, int numangle)
{
    int theta_idx = get_global_id(1);
    int count_idx = get_global_id(0);
    int glob_size = get_global_size(0);
    float cosVal;
    float sinVal = sincos(theta * ((float)theta_idx), &cosVal);
    sinVal *= irho;
    cosVal *= irho;

    __global const int * list = (__global const int*)(list_ptr + list_offset);
    __global int* accum = (__global int*)(accum_ptr + mad24(theta_idx + 1, accum_step, accum_offset));
    const int shift = (numrho - 1) / 2;

    if (theta_idx < numangle)
    {
        for (int i = count_idx; i < total_points; i += glob_size)
        {
            const int val = list[i];
            const int x = (val & 0xFFFF);
            const int y = (val >> 16) & 0xFFFF;

            int r = convert_int_rte(mad((float)x, cosVal, y * sinVal)) + shift;
            atomic_inc(accum + r + 1);
        }
    }
}

#elif defined FILL_ACCUM_LOCAL

__kernel void fill_accum_local(__global const uchar * list_ptr, int list_step, int list_offset,
                               __global uchar * accum_ptr, int accum_step, int accum_offset,
                               int total_points, float irho, float theta, int numrho, int numangle)
{
    int theta_idx = get_group_id(1);
    int count_idx = get_local_id(0);
    __local int l_accum[BUFFER_SIZE];

    if (theta_idx > 0 && theta_idx < numangle + 1)
    {
        float cosVal;
        float sinVal = sincos(theta * (float) (theta_idx-1), &cosVal);
        sinVal *= irho;
        cosVal *= irho;

        for (int i=count_idx; i<BUFFER_SIZE; i+=LOCAL_SIZE)
            l_accum[i] = 0;

        barrier(CLK_LOCAL_MEM_FENCE);

        __global const int * list = (__global const int*)(list_ptr + list_offset);
        const int shift = (numrho - 1) / 2;

        for (int i = count_idx; i < total_points; i += LOCAL_SIZE)
        {
            const int point = list[i];
            const int x = (point & 0xFFFF);
            const int y = point >> 16;

            int r = convert_int_rte(mad((float)x, cosVal, y * sinVal)) + shift;
            atomic_inc(l_accum + r + 1);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        __global int* accum = (__global int*)(accum_ptr + mad24(theta_idx, accum_step, accum_offset));
        for (int i=count_idx; i<BUFFER_SIZE; i+=LOCAL_SIZE)
            accum[i] = l_accum[i];
    }
    else if (theta_idx < numangle + 2)
    {
        __global int* accum = (__global int*)(accum_ptr + mad24(theta_idx, accum_step, accum_offset));
        for (int i=count_idx; i<BUFFER_SIZE; i+=LOCAL_SIZE)
            accum[i] = 0;
    }
}

#elif defined GET_LINES

__kernel void get_lines(__global uchar * accum_ptr, int accum_step, int accum_offset, int accum_rows, int accum_cols,
                         __global uchar * lines_ptr, int lines_step, int lines_offset, __global int* lines_index_ptr,
                         int linesMax, int threshold, float rho, float theta)
{
    int x0 = get_global_id(0);
    int y = get_global_id(1);
    int glob_size = get_global_size(0);

    if (y < accum_rows-2)
    {
        __global uchar* accum = accum_ptr + mad24(y+1, accum_step, mad24(x0+1, (int) sizeof(int), accum_offset));
        __global float2* lines = (__global float2*)(lines_ptr + lines_offset);
        __global int* lines_index = lines_index_ptr + 1;

        for (int x=x0; x<accum_cols-2; x+=glob_size)
        {
            int curVote = ACCUM(accum);

            if (curVote > threshold && curVote > ACCUM(accum - sizeof(int)) && curVote >= ACCUM(accum + sizeof(int)) &&
                curVote > ACCUM(accum - accum_step) && curVote >= ACCUM(accum + accum_step))
            {
                int index = atomic_inc(lines_index);

                if (index < linesMax)
                {
                    float radius = (x - (accum_cols - 3) * 0.5f) * rho;
                    float angle = y * theta;

                    lines[index] = (float2)(radius, angle);
                }
            }

            accum += glob_size * (int) sizeof(int);
        }
    }
}

#elif GET_LINES_PROBABOLISTIC

__kernel void get_lines(__global const uchar * accum_ptr, int accum_step, int accum_offset, int accum_rows, int accum_cols,
                        __global const uchar * src_ptr, int src_step, int src_offset, int src_rows, int src_cols,
                        __global uchar * lines_ptr, int lines_step, int lines_offset, __global int* lines_index_ptr,
                        int linesMax, int threshold, int lineLength, int lineGap, float rho, float theta)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < accum_rows-2)
    {
        __global const uchar* accum = accum_ptr + mad24(y+1, accum_step, mad24(x+1, (int) sizeof(int), accum_offset));
        __global int4* lines = (__global int4*)(lines_ptr + lines_offset);
        __global int* lines_index = lines_index_ptr + 1;

        int curVote = ACCUM(accum);

        if (curVote >= threshold &&
            curVote > ACCUM(accum - accum_step - sizeof(int)) &&
            curVote > ACCUM(accum - accum_step) &&
            curVote > ACCUM(accum - accum_step + sizeof(int)) &&
            curVote > ACCUM(accum - sizeof(int)) &&
            curVote > ACCUM(accum + sizeof(int)) &&
            curVote > ACCUM(accum + accum_step - sizeof(int)) &&
            curVote > ACCUM(accum + accum_step) &&
            curVote > ACCUM(accum + accum_step + sizeof(int)))
        {
            const float radius = (x - (accum_cols - 2 - 1) * 0.5f) * rho;
            const float angle = y * theta;

            float cosa;
            float sina = sincos(angle, &cosa);

            float2 p0 = (float2)(cosa * radius, sina * radius);
            float2 dir = (float2)(-sina, cosa);

            float2 pb[4] = { (float2)(-1, -1), (float2)(-1, -1), (float2)(-1, -1), (float2)(-1, -1) };
            float a;

            if (dir.x != 0)
            {
                a = -p0.x / dir.x;
                pb[0].x = 0;
                pb[0].y = p0.y + a * dir.y;

                a = (src_cols - 1 - p0.x) / dir.x;
                pb[1].x = src_cols - 1;
                pb[1].y = p0.y + a * dir.y;
            }

            if (dir.y != 0)
            {
                a = -p0.y / dir.y;
                pb[2].x = p0.x + a * dir.x;
                pb[2].y = 0;

                a = (src_rows - 1 - p0.y) / dir.y;
                pb[3].x = p0.x + a * dir.x;
                pb[3].y = src_rows - 1;
            }

            if (pb[0].x == 0 && (pb[0].y >= 0 && pb[0].y < src_rows))
            {
                p0 = pb[0];
                if (dir.x < 0)
                    dir = -dir;
            }
            else if (pb[1].x == src_cols - 1 && (pb[1].y >= 0 && pb[1].y < src_rows))
            {
                p0 = pb[1];
                if (dir.x > 0)
                    dir = -dir;
            }
            else if (pb[2].y == 0 && (pb[2].x >= 0 && pb[2].x < src_cols))
            {
                p0 = pb[2];
                if (dir.y < 0)
                    dir = -dir;
            }
            else if (pb[3].y == src_rows - 1 && (pb[3].x >= 0 && pb[3].x < src_cols))
            {
                p0 = pb[3];
                if (dir.y > 0)
                    dir = -dir;
            }

            dir /= max(fabs(dir.x), fabs(dir.y));

            float2 line_end[2];
            int gap;
            bool inLine = false;

            if (p0.x < 0 || p0.x >= src_cols || p0.y < 0 || p0.y >= src_rows)
                return;

            for (;;)
            {
                if (*(src_ptr + mad24(p0.y, src_step, p0.x + src_offset)))
                {
                    gap = 0;

                    if (!inLine)
                    {
                        line_end[0] = p0;
                        line_end[1] = p0;
                        inLine = true;
                    }
                    else
                    {
                        line_end[1] = p0;
                    }
                }
                else if (inLine)
                {
                    if (++gap > lineGap)
                    {
                        bool good_line = fabs(line_end[1].x - line_end[0].x) >= lineLength ||
                                         fabs(line_end[1].y - line_end[0].y) >= lineLength;

                        if (good_line)
                        {
                            int index = atomic_inc(lines_index);
                            if (index < linesMax)
                                lines[index] = (int4)(line_end[0].x, line_end[0].y, line_end[1].x, line_end[1].y);
                        }

                        gap = 0;
                        inLine = false;
                    }
                }

                p0 = p0 + dir;
                if (p0.x < 0 || p0.x >= src_cols || p0.y < 0 || p0.y >= src_rows)
                {
                    if (inLine)
                    {
                        bool good_line = fabs(line_end[1].x - line_end[0].x) >= lineLength ||
                                         fabs(line_end[1].y - line_end[0].y) >= lineLength;

                        if (good_line)
                        {
                            int index = atomic_inc(lines_index);
                            if (index < linesMax)
                                lines[index] = (int4)(line_end[0].x, line_end[0].y, line_end[1].x, line_end[1].y);
                        }

                    }
                    break;
                }
            }

        }
    }
}

#endif
