/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Peng Xiao, pengxiao@multicorewareinc.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

// Smoothing perpendicular to the derivative direction with a triangle filter
// only support 3x3 Sobel kernel
// h (-1) =  1, h (0) =  2, h (1) =  1
// h'(-1) = -1, h'(0) =  0, h'(1) =  1
// thus sobel 2D operator can be calculated as:
// h'(x, y) = h'(x)h(y) for x direction
//
// src		input 8bit single channel image data
// dx_buf	output dx buffer
// dy_buf	output dy buffer

__kernel void __attribute__((reqd_work_group_size(16, 16, 1)))
calcSobelRowPass
    (__global const uchar * src, int src_step, int src_offset, int rows, int cols,
     __global uchar * dx_buf, int dx_buf_step, int dx_buf_offset,
     __global uchar * dy_buf, int dy_buf_step, int dy_buf_offset)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    int lidx = get_local_id(0);
    int lidy = get_local_id(1);

    __local int smem[16][18];

    smem[lidy][lidx + 1] = src[mad24(src_step, min(gidy, rows - 1), gidx + src_offset)];
    if (lidx == 0)
    {
        smem[lidy][0]  = src[mad24(src_step, min(gidy, rows - 1), max(gidx - 1,  0)        + src_offset)];
        smem[lidy][17] = src[mad24(src_step, min(gidy, rows - 1), min(gidx + 16, cols - 1) + src_offset)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gidy < rows && gidx < cols)
    {
        *(__global short *)(dx_buf + mad24(gidy, dx_buf_step, gidx * (int)sizeof(short) + dx_buf_offset)) =
            smem[lidy][lidx + 2] - smem[lidy][lidx];
        *(__global short *)(dy_buf + mad24(gidy, dy_buf_step, gidx * (int)sizeof(short) + dy_buf_offset)) =
            smem[lidy][lidx] + 2 * smem[lidy][lidx + 1] + smem[lidy][lidx + 2];
    }
}

inline int calc(short x, short y)
{
#ifdef L2GRAD
    return x * x + y * y;
#else
    return (x >= 0 ? x : -x) + (y >= 0 ? y : -y);
#endif
}

// calculate the magnitude of the filter pass combining both x and y directions
// This is the non-buffered version(non-3x3 sobel)
//
// dx_buf		dx buffer, calculated from calcSobelRowPass
// dy_buf		dy buffer, calculated from calcSobelRowPass
// dx			direvitive in x direction output
// dy			direvitive in y direction output
// mag			magnitude direvitive of xy output

__kernel void calcMagnitude(__global const uchar * dxptr, int dx_step, int dx_offset,
                            __global const uchar * dyptr, int dy_step, int dy_offset,
                            __global uchar * magptr, int mag_step, int mag_offset, int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < rows && x < cols)
    {
        int dx_index = mad24(dx_step, y, x * (int)sizeof(short) + dx_offset);
        int dy_index = mad24(dy_step, y, x * (int)sizeof(short) + dy_offset);
        int mag_index = mad24(mag_step, y + 1, (x + 1) * (int)sizeof(int) + mag_offset);

        __global const short * dx = (__global const short *)(dxptr + dx_index);
        __global const short * dy = (__global const short *)(dyptr + dy_index);
        __global int * mag = (__global int *)(magptr + mag_index);

        mag[0] = calc(dx[0], dy[0]);
    }
}

// calculate the magnitude of the filter pass combining both x and y directions
// This is the buffered version(3x3 sobel)
//
// dx_buf		dx buffer, calculated from calcSobelRowPass
// dy_buf		dy buffer, calculated from calcSobelRowPass
// dx			direvitive in x direction output
// dy			direvitive in y direction output
// mag			magnitude direvitive of xy output
__kernel void __attribute__((reqd_work_group_size(16, 16, 1)))
calcMagnitude_buf
    (__global const short * dx_buf, int dx_buf_step, int dx_buf_offset,
     __global const short * dy_buf, int dy_buf_step, int dy_buf_offset,
     __global short * dx, int dx_step, int dx_offset,
     __global short * dy, int dy_step, int dy_offset,
     __global int * mag, int mag_step, int mag_offset,
     int rows, int cols)
{
    dx_buf_step    /= sizeof(*dx_buf);
    dx_buf_offset  /= sizeof(*dx_buf);
    dy_buf_step    /= sizeof(*dy_buf);
    dy_buf_offset  /= sizeof(*dy_buf);
    dx_step    /= sizeof(*dx);
    dx_offset  /= sizeof(*dx);
    dy_step    /= sizeof(*dy);
    dy_offset  /= sizeof(*dy);
    mag_step   /= sizeof(*mag);
    mag_offset /= sizeof(*mag);

    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    int lidx = get_local_id(0);
    int lidy = get_local_id(1);

    __local short sdx[18][16];
    __local short sdy[18][16];

    sdx[lidy + 1][lidx] = dx_buf[gidx + min(gidy, rows - 1) * dx_buf_step + dx_buf_offset];
    sdy[lidy + 1][lidx] = dy_buf[gidx + min(gidy, rows - 1) * dy_buf_step + dy_buf_offset];
    if (lidy == 0)
    {
        sdx[0][lidx]  = dx_buf[gidx + min(max(gidy - 1, 0), rows - 1) * dx_buf_step + dx_buf_offset];
        sdx[17][lidx] = dx_buf[gidx + min(gidy + 16, rows - 1)        * dx_buf_step + dx_buf_offset];

        sdy[0][lidx]  = dy_buf[gidx + min(max(gidy - 1, 0), rows - 1) * dy_buf_step + dy_buf_offset];
        sdy[17][lidx] = dy_buf[gidx + min(gidy + 16, rows - 1)        * dy_buf_step + dy_buf_offset];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gidx < cols && gidy < rows)
    {
        short x =  sdx[lidy][lidx] + 2 * sdx[lidy + 1][lidx] + sdx[lidy + 2][lidx];
        short y = -sdy[lidy][lidx] + sdy[lidy + 2][lidx];

        dx[gidx + gidy * dx_step + dx_offset] = x;
        dy[gidx + gidy * dy_step + dy_offset] = y;

        mag[(gidx + 1) + (gidy + 1) * mag_step + mag_offset] = calc(x, y);
    }
}


//////////////////////////////////////////////////////////////////////////////////////////
// 0.4142135623730950488016887242097 is tan(22.5)

#define CANNY_SHIFT 15
#define TG22        (int)(0.4142135623730950488016887242097f*(1<<CANNY_SHIFT) + 0.5f)

// First pass of edge detection and non-maximum suppression
// edgetype is set to for each pixel:
// 0 - below low thres, not an edge
// 1 - maybe an edge
// 2 - is an edge, either magnitude is greater than high thres, or
//     Given estimates of the image gradients, a search is then carried out
//     to determine if the gradient magnitude assumes a local maximum in the gradient direction.
//     if the rounded gradient angle is zero degrees (i.e. the edge is in the north-south direction) the point will be considered to be on the edge if its gradient magnitude is greater than the magnitudes in the west and east directions,
//     if the rounded gradient angle is 90 degrees (i.e. the edge is in the east-west direction) the point will be considered to be on the edge if its gradient magnitude is greater than the magnitudes in the north and south directions,
//     if the rounded gradient angle is 135 degrees (i.e. the edge is in the north east-south west direction) the point will be considered to be on the edge if its gradient magnitude is greater than the magnitudes in the north west and south east directions,
//     if the rounded gradient angle is 45 degrees (i.e. the edge is in the north west-south east direction)the point will be considered to be on the edge if its gradient magnitude is greater than the magnitudes in the north east and south west directions.
//
// dx, dy		direvitives of x and y direction
// mag			magnitudes calculated from calcMagnitude function
// map			output containing raw edge types

__kernel void __attribute__((reqd_work_group_size(16,16,1)))
calcMap(
    __global const uchar * dx, int dx_step, int dx_offset,
    __global const uchar * dy, int dy_step, int dy_offset,
    __global const uchar * mag, int mag_step, int mag_offset,
    __global uchar * map, int map_step, int map_offset,
    int rows, int cols, int low_thresh, int high_thresh)
{
    __local int smem[18][18];

    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    int lidx = get_local_id(0);
    int lidy = get_local_id(1);

    int grp_idx = get_global_id(0) & 0xFFFFF0;
    int grp_idy = get_global_id(1) & 0xFFFFF0;

    int tid = lidx + lidy * 16;
    int lx = tid % 18;
    int ly = tid / 18;

    mag += mag_offset;
    if (ly < 14)
        smem[ly][lx] = *(__global const int *)(mag +
            mad24(mag_step, min(grp_idy + ly, rows - 1), (int)sizeof(int) * (grp_idx + lx)));
    if (ly < 4 && grp_idy + ly + 14 <= rows && grp_idx + lx <= cols)
        smem[ly + 14][lx] = *(__global const int *)(mag +
            mad24(mag_step, min(grp_idy + ly + 14, rows - 1), (int)sizeof(int) * (grp_idx + lx)));
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gidy < rows && gidx < cols)
    {
        // 0 - the pixel can not belong to an edge
        // 1 - the pixel might belong to an edge
        // 2 - the pixel does belong to an edge
        int edge_type = 0;
        int m = smem[lidy + 1][lidx + 1];

        if (m > low_thresh)
        {
            short xs = *(__global const short *)(dx + mad24(gidy, dx_step, dx_offset + (int)sizeof(short) * gidx));
            short ys = *(__global const short *)(dy + mad24(gidy, dy_step, dy_offset + (int)sizeof(short) * gidx));
            int x = abs(xs), y = abs(ys);

            int tg22x = x * TG22;
            y <<= CANNY_SHIFT;

            if (y < tg22x)
            {
                if (m > smem[lidy + 1][lidx] && m >= smem[lidy + 1][lidx + 2])
                    edge_type = 1 + (int)(m > high_thresh);
            }
            else
            {
                int tg67x = tg22x + (x << (1 + CANNY_SHIFT));
                if (y > tg67x)
                {
                    if (m > smem[lidy][lidx + 1]&& m >= smem[lidy + 2][lidx + 1])
                        edge_type = 1 + (int)(m > high_thresh);
                }
                else
                {
                    int s = (xs ^ ys) < 0 ? -1 : 1;
                    if (m > smem[lidy][lidx + 1 - s]&& m > smem[lidy + 2][lidx + 1 + s])
                        edge_type = 1 + (int)(m > high_thresh);
                }
            }
        }
        *(__global int *)(map + mad24(map_step, gidy + 1, (gidx + 1) * (int)sizeof(int) + map_offset)) = edge_type;
    }
}

#undef CANNY_SHIFT
#undef TG22

struct PtrStepSz
{
    __global uchar * ptr;
    int step, rows, cols;
};

inline int get(struct PtrStepSz data, int y, int x)
{
    return *(__global int *)(data.ptr + mad24(data.step, y + 1, (int)sizeof(int) * (x + 1)));
}

inline void set(struct PtrStepSz data, int y, int x, int value)
{
    *(__global int *)(data.ptr + mad24(data.step, y + 1, (int)sizeof(int) * (x + 1))) = value;
}

// perform Hysteresis for pixel whose edge type is 1
//
// If candidate pixel (edge type is 1) has a neighbour pixel (in 3x3 area) with type 2, it is believed to be part of an edge and
// marked as edge. Each thread will iterate for 16 times to connect local edges.
// Candidate pixel being identified as edge will then be tested if there is nearby potiential edge points. If there is, counter will
// be incremented by 1 and the point location is stored. These potiential candidates will be processed further in next kernel.
//
// map		raw edge type results calculated from calcMap.
// stack	the potiential edge points found in this kernel call
// counter	the number of potiential edge points

__kernel void __attribute__((reqd_work_group_size(16,16,1)))
edgesHysteresisLocal
    (__global uchar * map_ptr, int map_step, int map_offset,
     __global ushort2 * st, __global unsigned int * counter,
    int rows, int cols)
{
    struct PtrStepSz map = { map_ptr + map_offset, map_step, rows + 1, cols + 1 };

    __local int smem[18][18];

    int2 blockIdx = (int2)(get_group_id(0), get_group_id(1));
    int2 blockDim = (int2)(get_local_size(0), get_local_size(1));
    int2 threadIdx = (int2)(get_local_id(0), get_local_id(1));

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    smem[threadIdx.y + 1][threadIdx.x + 1] = x < map.cols && y < map.rows ? get(map, y, x) : 0;
    if (threadIdx.y == 0)
        smem[0][threadIdx.x + 1] = x < map.cols ? get(map, y - 1, x) : 0;
    if (threadIdx.y == blockDim.y - 1)
        smem[blockDim.y + 1][threadIdx.x + 1] = y + 1 < map.rows ? get(map, y + 1, x) : 0;
    if (threadIdx.x == 0)
        smem[threadIdx.y + 1][0] = y < map.rows ? get(map, y, x - 1) : 0;
    if (threadIdx.x == blockDim.x - 1)
        smem[threadIdx.y + 1][blockDim.x + 1] = x + 1 < map.cols && y < map.rows ? get(map, y, x + 1) : 0;
    if (threadIdx.x == 0 && threadIdx.y == 0)
        smem[0][0] = y > 0 && x > 0 ? get(map, y - 1, x - 1) : 0;
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0)
        smem[0][blockDim.x + 1] = y > 0 && x + 1 < map.cols ? get(map, y - 1, x + 1) : 0;
    if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1)
        smem[blockDim.y + 1][0] = y + 1 < map.rows && x > 0 ? get(map, y + 1, x - 1) : 0;
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1)
        smem[blockDim.y + 1][blockDim.x + 1] = y + 1 < map.rows && x + 1 < map.cols ? get(map, y + 1, x + 1) : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x >= cols || y >= rows)
        return;

    int n;

    #pragma unroll
    for (int k = 0; k < 16; ++k)
    {
        n = 0;

        if (smem[threadIdx.y + 1][threadIdx.x + 1] == 1)
        {
            n += smem[threadIdx.y    ][threadIdx.x    ] == 2;
            n += smem[threadIdx.y    ][threadIdx.x + 1] == 2;
            n += smem[threadIdx.y    ][threadIdx.x + 2] == 2;

            n += smem[threadIdx.y + 1][threadIdx.x    ] == 2;
            n += smem[threadIdx.y + 1][threadIdx.x + 2] == 2;

            n += smem[threadIdx.y + 2][threadIdx.x    ] == 2;
            n += smem[threadIdx.y + 2][threadIdx.x + 1] == 2;
            n += smem[threadIdx.y + 2][threadIdx.x + 2] == 2;
        }

        if (n > 0)
            smem[threadIdx.y + 1][threadIdx.x + 1] = 2;
    }

    const int e = smem[threadIdx.y + 1][threadIdx.x + 1];
    set(map, y, x, e);
    n = 0;

    if (e == 2)
    {
        n += smem[threadIdx.y    ][threadIdx.x    ] == 1;
        n += smem[threadIdx.y    ][threadIdx.x + 1] == 1;
        n += smem[threadIdx.y    ][threadIdx.x + 2] == 1;

        n += smem[threadIdx.y + 1][threadIdx.x    ] == 1;
        n += smem[threadIdx.y + 1][threadIdx.x + 2] == 1;

        n += smem[threadIdx.y + 2][threadIdx.x    ] == 1;
        n += smem[threadIdx.y + 2][threadIdx.x + 1] == 1;
        n += smem[threadIdx.y + 2][threadIdx.x + 2] == 1;
    }

    if (n > 0)
    {
        const int ind = atomic_inc(counter);
        st[ind] = (ushort2)(x + 1, y + 1);
    }
}

__constant int c_dx[8] = {-1,  0,  1, -1, 1, -1, 0, 1};
__constant int c_dy[8] = {-1, -1, -1,  0, 0,  1, 1, 1};


#define stack_size 512
#define map_index mad24(map_step, pos.y, pos.x * (int)sizeof(int))

__kernel void __attribute__((reqd_work_group_size(128, 1, 1)))
edgesHysteresisGlobal(__global uchar * map, int map_step, int map_offset,
    __global ushort2 * st1, __global ushort2 * st2, __global int * counter,
    int rows, int cols, int count)
{
    map += map_offset;

    int lidx = get_local_id(0);

    int grp_idx = get_group_id(0);
    int grp_idy = get_group_id(1);

    __local unsigned int s_counter, s_ind;
    __local ushort2 s_st[stack_size];

    if (lidx == 0)
        s_counter = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    int ind = mad24(grp_idy, (int)get_local_size(0), grp_idx);

    if (ind < count)
    {
        ushort2 pos = st1[ind];
        if (lidx < 8)
        {
            pos.x += c_dx[lidx];
            pos.y += c_dy[lidx];
            if (pos.x > 0 && pos.x <= cols && pos.y > 0 && pos.y <= rows && *(__global int *)(map + map_index) == 1)
            {
                *(__global int *)(map + map_index) = 2;
                ind = atomic_inc(&s_counter);
                s_st[ind] = pos;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        while (s_counter > 0 && s_counter <= stack_size - get_local_size(0))
        {
            const int subTaskIdx = lidx >> 3;
            const int portion = min(s_counter, (uint)(get_local_size(0)>> 3));

            if (subTaskIdx < portion)
                pos = s_st[s_counter - 1 - subTaskIdx];
            barrier(CLK_LOCAL_MEM_FENCE);

            if (lidx == 0)
                s_counter -= portion;
            barrier(CLK_LOCAL_MEM_FENCE);

            if (subTaskIdx < portion)
            {
                pos.x += c_dx[lidx & 7];
                pos.y += c_dy[lidx & 7];
                if (pos.x > 0 && pos.x <= cols && pos.y > 0 && pos.y <= rows && *(__global int *)(map + map_index) == 1)
                {
                    *(__global int *)(map + map_index) = 2;
                    ind = atomic_inc(&s_counter);
                    s_st[ind] = pos;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (s_counter > 0)
        {
            if (lidx == 0)
            {
                ind = atomic_add(counter, s_counter);
                s_ind = ind - s_counter;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            ind = s_ind;
            for (int i = lidx; i < (int)s_counter; i += get_local_size(0))
                st2[ind + i] = s_st[i];
        }
    }
}

#undef map_index
#undef stack_size

// Get the edge result. egde type of value 2 will be marked as an edge point and set to 255. Otherwise 0.
// map		edge type mappings
// dst		edge output

__kernel void getEdges(__global const uchar * mapptr, int map_step, int map_offset,
                       __global uchar * dst, int dst_step, int dst_offset, int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < rows && x < cols)
    {
        int map_index = mad24(map_step, y + 1, (x + 1) * (int)sizeof(int) + map_offset);
        int dst_index = mad24(dst_step, y, x + dst_offset);

        __global const int * map = (__global const int *)(mapptr + map_index);

        dst[dst_index] = (uchar)(-(map[0] >> 1));
    }
}
