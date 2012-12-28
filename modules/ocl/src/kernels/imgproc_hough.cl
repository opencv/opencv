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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
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
// This software is provided by the copyright holders and contributors "as is" and
// any express or bpied warranties, including, but not limited to, the bpied
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

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

////////////////////////////////////////////////////////////////////////
// buildPointList

#define PIXELS_PER_THREAD 16

// TODO: add offset to support ROI
__kernel void buildPointList(__global const uchar* src,
                             int cols,
                             int rows,
                             int step,
                             __global unsigned int* list,
                             __global int* counter)
{
    __local unsigned int s_queues[4][32 * PIXELS_PER_THREAD];
    __local int s_qsize[4];
    __local int s_globStart[4];

    const int x = get_group_id(0) * get_local_size(0) * PIXELS_PER_THREAD + get_local_id(0);
    const int y = get_global_id(1);

    if (get_local_id(0) == 0)
        s_qsize[get_local_id(1)] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
        
    if (y < rows)
    {
        // fill the queue
        __global const uchar* srcRow = &src[y * step];
        for (int i = 0, xx = x; i < PIXELS_PER_THREAD && xx < cols; ++i, xx += get_local_size(0))
        {
            if (srcRow[xx])
            {
                const unsigned int val = (y << 16) | xx;
                const int qidx = atomic_add(&s_qsize[get_local_id(1)], 1);
                s_queues[get_local_id(1)][qidx] = val;
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // let one work-item reserve the space required in the global list
    if (get_local_id(0) == 0 && get_local_id(1) == 0)
    {
        // find how many items are stored in each list
        int totalSize = 0;
        for (int i = 0; i < get_local_size(1); ++i)
        {
            s_globStart[i] = totalSize;
            totalSize += s_qsize[i];
        }

        // calculate the offset in the global list
        const int globalOffset = atomic_add(counter, totalSize);
        for (int i = 0; i < get_local_size(1); ++i)
            s_globStart[i] += globalOffset;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
    
    // copy local queues to global queue
    const int qsize = s_qsize[get_local_id(1)];
    int gidx = s_globStart[get_local_id(1)] + get_local_id(0);
    for(int i = get_local_id(0); i < qsize; i += get_local_size(0), gidx += get_local_size(0))
        list[gidx] = s_queues[get_local_id(1)][i];
}

////////////////////////////////////////////////////////////////////////
// circlesAccumCenters

// TODO: add offset to support ROI
__kernel void circlesAccumCenters(__global const unsigned int* list,
                                  const int count,
                                  __global const int* dx,
                                  const int dxStep,
                                  __global const int* dy,
                                  const int dyStep,
                                  __global int* accum,
                                  const int accumStep,
                                  const int width,
                                  const int height,
                                  const int minRadius,
                                  const int maxRadius,
                                  const float idp)
{
    const int dxStepInPixel    = dxStep    / sizeof(int);
    const int dyStepInPixel    = dyStep    / sizeof(int);
    const int accumStepInPixel = accumStep / sizeof(int);
    
    const int SHIFT = 10;
    const int ONE = 1 << SHIFT;

    // const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int wid = get_global_id(0);

    if (wid >= count)
        return;

    const unsigned int val = list[wid];

    const int x = (val & 0xFFFF);
    const int y = (val >> 16) & 0xFFFF;

    const int vx = dx[mad24(y, dxStepInPixel, x)];
    const int vy = dy[mad24(y, dyStepInPixel, x)];

    if (vx == 0 && vy == 0)
        return;

    const float mag = sqrt(convert_float(vx * vx + vy * vy));

    const int x0 = convert_int_rte((x * idp) * ONE);
    const int y0 = convert_int_rte((y * idp) * ONE);

    int sx = convert_int_rte((vx * idp) * ONE / mag);
    int sy = convert_int_rte((vy * idp) * ONE / mag);

    // Step from minRadius to maxRadius in both directions of the gradient
    for (int k1 = 0; k1 < 2; ++k1)
    {
        int x1 = x0 + minRadius * sx;
        int y1 = y0 + minRadius * sy;

        for (int r = minRadius; r <= maxRadius; x1 += sx, y1 += sy, ++r)
        {
            const int x2 = x1 >> SHIFT;
            const int y2 = y1 >> SHIFT;

            if (x2 < 0 || x2 >= width || y2 < 0 || y2 >= height)
                break;

            atomic_add(&accum[mad24(y2+1, accumStepInPixel, x2+1)], 1);
        }

        sx = -sx;
        sy = -sy;
    }
}

// ////////////////////////////////////////////////////////////////////////
// // buildCentersList

// TODO: add offset to support ROI
__kernel void buildCentersList(__global const int* accum,
                               const int accumCols,
                               const int accumRows,
                               const int accumStep,
                               __global unsigned int* centers,
                               const int threshold,
                               __global int* counter)
{
    const int accumStepInPixel = accumStep/sizeof(int);
    
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x < accumCols - 2 && y < accumRows - 2)
    {
        const int top    = accum[mad24(y,     accumStepInPixel, x + 1)];

        const int left   = accum[mad24(y + 1, accumStepInPixel, x)];
        const int cur    = accum[mad24(y + 1, accumStepInPixel, x + 1)];
        const int right  = accum[mad24(y + 1, accumStepInPixel, x + 2)];
        
        const int bottom = accum[mad24(y + 2, accumStepInPixel, x + 1)];;

        if (cur > threshold && cur > top && cur >= bottom && cur >  left && cur >= right)
        {
            const unsigned int val = (y << 16) | x;
            const int idx = atomic_add(counter, 1);
            centers[idx] = val;
        }
    }
}


// ////////////////////////////////////////////////////////////////////////
// // circlesAccumRadius

// TODO: add offset to support ROI
__kernel void circlesAccumRadius(__global const unsigned int* centers,
                                 __global const unsigned int* list, const int count,
                                 __global float4* circles, const int maxCircles,
                                 const float dp,
                                 const int minRadius, const int maxRadius,
                                 const int histSize,
                                 const int threshold,
                                 __local int* smem,
                                 __global int* counter)
{
    for (int i = get_local_id(0); i < histSize + 2; i += get_local_size(0))
        smem[i] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int val = centers[get_group_id(0)];

    float cx = convert_float(val & 0xFFFF);
    float cy = convert_float((val >> 16) & 0xFFFF);

    cx = (cx + 0.5f) * dp;
    cy = (cy + 0.5f) * dp;

    for (int i = get_local_id(0); i < count; i += get_local_size(0))
    {
        val = list[i];

        const int x = (val & 0xFFFF);
        const int y = (val >> 16) & 0xFFFF;

        const float rad = sqrt((cx - x) * (cx - x) + (cy - y) * (cy - y));
        if (rad >= minRadius && rad <= maxRadius)
        {
            const int r = convert_int_rte(rad - minRadius);

            atomic_add(&smem[r + 1], 1);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = get_local_id(0); i < histSize; i += get_local_size(0))
    {
        const int curVotes = smem[i + 1];

        if (curVotes >= threshold && curVotes > smem[i] && curVotes >= smem[i + 2])
            
        {
            const int ind = atomic_add(counter, 1);
            if (ind < maxCircles)
            {
                circles[ind] = (float4)(cx, cy, convert_float(i + minRadius), 0.0f);
            }
        }
    }
}
