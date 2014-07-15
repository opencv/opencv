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

#ifdef WITH_SOBEL

#define loadpix(addr) vload3(0, (__global uchar *)(addr))
#define storepix(value, addr) *(__global uchar *)(addr) = (uchar)value 

#define CANNY_SHIFT 15
#define TG22        (int)(0.4142135623730950488016887242097f * (1 << CANNY_SHIFT) + 0.5f)
/*
    stage1:
        Sobel operator,
        Non maxima suppression
        Double thresholding
*/


__kernel void stage1_with_sobel(__global const uchar *src, int src_step, int src_offset, int rows, int cols
                                __global uchar *map, int map_step, int map_offset,
                                int low_thr, int high_thr)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    int lidx = get_local_id(0);
    int lidy = get_local_id(1);

#define ptr(x, y) \
    (src + mad24(src_step, clamp(gidy + y, 0, rows - 1), clamp(gidx + x, 0, cols - 1)) * cn + src_offset)) 
    //// SOBEL
    // 

    uchar3 dx = loadpix(ptr(1, -1)) - loadpix(ptr(-1, -1)) 
                + (uchar)2 * (loadpix(ptr(1, 0)) - loadpix(ptr(-1, 0)))
                + loadpix(ptr(1, 1)) - loadpix(ptr(-1, 1));

    uchar3 dy = loadpix(ptr(-1, 1)) - loadpix(ptr(-1, -1))
                + (uchar)2 * (loadpix(ptr(0, 1)) - loadpix(ptr(0, -1)))
                + loadpix(ptr(1, 1)) - loadpix(ptr(1, -1));

    __local int mag[16][16]; // PROBLEM with alignment (bank conflict) what to do?

#ifdef L2_GRAD
    int3 mag3 = convert_int3(dx * dx + dy * dy);
#else
    int3 mag3 = convert_int3(dx + dy);
#endif
    mag[lidy][lidx] = max(max(mag3.x, mag3.y), mag3.z);

    barrier(CLK_LOCAL_MEM_FENCE);

    lidy = clamp(lidy, 1, 14);
    lidx = clamp(lidx, 1, 14);
    int mag0 = mag[lidy][lidx];
    /*
        0 - pixel doesn't belong to an edge
        1 - might belong to an edge
        2 - belong to an edge
    */
    uchar value = 0;
    if (mag0 > low_thr)
    {
        value = 1;
        int tg22x = dx * TG22;
        dy <<= CANNY_SHIFT;
        int tg67x = tg22x + (dx << (1 + CANNY_SHIFT));
        
        if (dy < tg22x)
        {
            if (mag0 > mag[lidy, lidx - 1] && mag0 > mag[lidy, lidx + 1])
                value = 2;
        }
        else if(dy < tg67x)
        {
            int delta = (dx ^ dy < 0) ? -1 : 1;
            if (mag0 > mag[lidy + delta][lidx - 1] && mag0 > mag[lidy - delta][lidx + 1])
                value = 2;
        }
        else
        {
            if (mag0 > mag[lidy - 1, lidx] && mag0 > mag[lidy + 1, lidx])
                value = 2;
        }
    }          
    storepix(value, map + mad24(gidy, map_step, gidx) + map_offset); // sizeof(map[i]) ???
}

#elif defined WITHOUT_SOBEL

#define loadpix(addr) (__global uchar *)(addr)
#define storepix(val, addr) *(__global uchar *)(addr) = uchar(val)

#define CANNY_SHIFT 15
#define TG22        (int)(0.4142135623730950488016887242097f * (1 << CANNY_SHIFT) + 0.5f)


inline int dist(short x, short y)
{
#ifdef L2_GRAD
    return (x * x + y * y);
#else
    return (abs(x) + abs(y));
#endif    
}

__kernel void stage1_without_sobel(__global const uchar *dxptr, int dx_step, int dx_offset, /* int rows, int cols  <- where it can be used */
                                   __global const uchar *dyptr, int dy_step, int dy_offset,
                                   __global uchar *map, int map_step, int map_offset,
                                    low_thr, int high_thr)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    int lidx = get_local_id(0);
    int lidy = get_local_id(1);

    __local int mag[16][16];

    int dx_index = mad24(gidy, dx_step, mad24(gidx, (int)sizeof(short) * cn, dx_offset));
    int dy_index = mad24(gidy, dy_step, mad24(gidx, (int)sizeof(short) * cn, dy_offset));

    __global short *dx = (__global short *)loadpix(dxptr + dx_index);
    __global short *dy = (__global short *)loadpix(dyptr + dy_index);

    int mag0 = dist(dx[0], dy[0]);
#if cn > 1
    #pragma unroll
    for (int i = 1; i < cn; i++)
    {
        mag1 = dist(dx[i], dy[i]); 
        if (mag1 > mag0)
        {
            mag0 = mag1;
            cdx = dx[i];
            cdy = dy[i]; 
        }
    }
    dx[0] = cdx;
    dy[0] = cdy;
#endif 
    mag[lidy][lidx] = mag0;

    barrier(CLK_LOCAL_MEM_FENCE);

    lidy = clamp(lidy, 1, 14);
    lidx = clamp(lidx, 1, 14);
    int mag0 = mag[lidy][lidx];
    /*
        0 - pixel doesn't belong to an edge
        1 - might belong to an edge
        2 - belong to an edge
    */
    uchar value = 0;
    if (mag0 > low_thr)
    {
        value = 1;
        int tg22x = dx[0] * TG22;
        y = dy[0] << CANNY_SHIFT;
        int tg67x = tg22x + (dx << (1 + CANNY_SHIFT));
        
        if (y < tg22x)
        {
            if (mag0 > mag[lidy, lidx - 1] && mag0 > mag[lidy, lidx + 1])
                value = 2;
        }
        else if(y < tg67x)
        {
            int delta = (dx[0] ^ dy[0] < 0) ? -1 : 1;
            if (mag0 > mag[lidy + delta][lidx - 1] && mag0 > mag[lidy - delta][lidx + 1])
                value = 2;
        }
        else
        {
            if (mag0 > mag[lidy - 1, lidx] && mag0 > mag[lidy + 1, lidx])
                value = 2;
        }
    }
    storepix(value, map + mad24(gidy, map_step, gidx) + map_offset); // sizeof(map[i]) ???
}

#elif defined STAGE2

#define STACK_SIZE 512
/*
    stage2:
        hysteresis (add edges labeled 1 if they are connected with an edge labeled 2)
*/
int move_dir[2][8] = {
    { -1, -1 }, { -1, 0 }, { -1, 1 }, { 0, -1 }, { 0, 1 }, { 1, -1 }, { 1, 0 }, { 1, 1 }
};
__kernel void stage2(__global uchar *map, int map_step, int map_offset
                     __global const ushort2 *common_stack, int c_stack_offset) // what about size of common_stack?
{
    __local ushort2 stack[STACK_SIZE];
    int counter = 0;

    while ()
}

#elif defined GET_EDGES

// Get the edge result. egde type of value 2 will be marked as an edge point and set to 255. Otherwise 0.
// map      edge type mappings
// dst      edge output

__kernel void getEdges(__global const uchar * mapptr, int map_step, int map_offset,
                       __global uchar * dst, int dst_step, int dst_offset, int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < rows && x < cols)
    {
        int map_index = mad24(map_step, y + 1, mad24(x + 1, (int)sizeof(int), map_offset)); // sizeof(map[i]) ???
        int dst_index = mad24(dst_step, y, x + dst_offset);

        __global const int * map = (__global const int *)(mapptr + map_index);

        dst[dst_index] = (uchar)(-(map[0] >> 1));
    }
}

#endif