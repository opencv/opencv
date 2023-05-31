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
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
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

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

#if cn != 3
#define loadpix(addr) *(__global const ST *)(addr)
#define storepix(val, addr)  *(__global DT *)(addr) = val
#define SRCSIZE (int)sizeof(ST)
#define DSTSIZE (int)sizeof(DT)
#else
#define loadpix(addr) vload3(0, (__global const ST1 *)(addr))
#define storepix(val, addr) vstore3(val, 0, (__global DT1 *)(addr))
#define SRCSIZE (int)sizeof(ST1)*cn
#define DSTSIZE (int)sizeof(DT1)*cn
#endif

#ifdef BORDER_CONSTANT
#elif defined BORDER_REPLICATE
#define EXTRAPOLATE(x, y, minX, minY, maxX, maxY) \
    { \
        x = max(min(x, maxX - 1), minX); \
        y = max(min(y, maxY - 1), minY); \
    }
#elif defined BORDER_WRAP
#define EXTRAPOLATE(x, y, minX, minY, maxX, maxY) \
    { \
        if (x < minX) \
            x -= ((x - maxX + 1) / maxX) * maxX; \
        if (x >= maxX) \
            x %= maxX; \
        if (y < minY) \
            y -= ((y - maxY + 1) / maxY) * maxY; \
        if (y >= maxY) \
            y %= maxY; \
    }
#elif defined(BORDER_REFLECT) || defined(BORDER_REFLECT_101)
#define EXTRAPOLATE_(x, y, minX, minY, maxX, maxY, delta) \
    { \
        if (maxX - minX == 1) \
            x = minX; \
        else \
            do \
            { \
                if (x < minX) \
                    x = minX - (x - minX) - 1 + delta; \
                else \
                    x = maxX - 1 - (x - maxX) - delta; \
            } \
            while (x >= maxX || x < minX); \
        \
        if (maxY - minY == 1) \
            y = minY; \
        else \
            do \
            { \
                if (y < minY) \
                    y = minY - (y - minY) - 1 + delta; \
                else \
                    y = maxY - 1 - (y - maxY) - delta; \
            } \
            while (y >= maxY || y < minY); \
    }
#ifdef BORDER_REFLECT
#define EXTRAPOLATE(x, y, minX, minY, maxX, maxY) EXTRAPOLATE_(x, y, minX, minY, maxX, maxY, 0)
#elif defined(BORDER_REFLECT_101)
#define EXTRAPOLATE(x, y, minX, minY, maxX, maxY) EXTRAPOLATE_(x, y, minX, minY, maxX, maxY, 1)
#endif
#else
#error No extrapolation method
#endif

#define noconvert

#ifdef SQR
#define PROCESS_ELEM(value) (value * value)
#else
#define PROCESS_ELEM(value) value
#endif

struct RectCoords
{
    int x1, y1, x2, y2;
};

inline WT readSrcPixel(int2 pos, __global const uchar * srcptr, int src_step, const struct RectCoords srcCoords)
{
#ifdef BORDER_ISOLATED
    if (pos.x >= srcCoords.x1 && pos.y >= srcCoords.y1 && pos.x < srcCoords.x2 && pos.y < srcCoords.y2)
#else
    if (pos.x >= 0 && pos.y >= 0 && pos.x < srcCoords.x2 && pos.y < srcCoords.y2)
#endif
    {
        int src_index = mad24(pos.y, src_step, pos.x * SRCSIZE);
        WT value = convertToWT(loadpix(srcptr + src_index));

        return PROCESS_ELEM(value);
    }
    else
    {
#ifdef BORDER_CONSTANT
        return (WT)(0);
#else
        int selected_col = pos.x, selected_row = pos.y;

        EXTRAPOLATE(selected_col, selected_row,
#ifdef BORDER_ISOLATED
            srcCoords.x1, srcCoords.y1,
#else
            0, 0,
#endif
            srcCoords.x2, srcCoords.y2);

        int src_index = mad24(selected_row, src_step, selected_col * SRCSIZE);
        WT value = convertToWT(loadpix(srcptr + src_index));

        return PROCESS_ELEM(value);
#endif
    }
}

__kernel void boxFilter(__global const uchar * srcptr, int src_step, int srcOffsetX, int srcOffsetY, int srcEndX, int srcEndY,
                        __global uchar * dstptr, int dst_step, int dst_offset, int rows, int cols
#ifdef NORMALIZE
                        , float alpha
#endif
                       )
{
    const struct RectCoords srcCoords = { srcOffsetX, srcOffsetY, srcEndX, srcEndY }; // for non-isolated border: offsetX, offsetY, wholeX, wholeY

    int x = get_local_id(0) + (LOCAL_SIZE_X - (KERNEL_SIZE_X - 1)) * get_group_id(0) - ANCHOR_X;
    int y = get_global_id(1) * BLOCK_SIZE_Y;
    int local_id = get_local_id(0);

    WT data[KERNEL_SIZE_Y];
    __local WT sumOfCols[LOCAL_SIZE_X];
    int2 srcPos = (int2)(srcCoords.x1 + x, srcCoords.y1 + y - ANCHOR_Y);

    #pragma unroll
    for (int sy = 0; sy < KERNEL_SIZE_Y; sy++, srcPos.y++)
        data[sy] = readSrcPixel(srcPos, srcptr, src_step, srcCoords);

    WT tmp_sum = (WT)(0);
    #pragma unroll
    for (int sy = 0; sy < KERNEL_SIZE_Y; sy++)
        tmp_sum += data[sy];

    sumOfCols[local_id] = tmp_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    int dst_index = mad24(y, dst_step, mad24(x, DSTSIZE, dst_offset));
    __global DT * dst = (__global DT *)(dstptr + dst_index);

    int sy_index = 0; // current index in data[] array
    for (int i = 0, stepY = min(rows - y, BLOCK_SIZE_Y); i < stepY; ++i)
    {
        if (local_id >= ANCHOR_X && local_id < LOCAL_SIZE_X - (KERNEL_SIZE_X - 1 - ANCHOR_X) &&
            x >= 0 && x < cols)
        {
            WT total_sum = (WT)(0);

            #pragma unroll
            for (int sx = 0; sx < KERNEL_SIZE_X; sx++)
                total_sum += sumOfCols[local_id + sx - ANCHOR_X];

#ifdef NORMALIZE
            DT dstval = convertToDT((WT)(alpha) * total_sum);
#else
            DT dstval = convertToDT(total_sum);
#endif
            storepix(dstval, dst);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        tmp_sum = sumOfCols[local_id];
        tmp_sum -= data[sy_index];

        data[sy_index] = readSrcPixel(srcPos, srcptr, src_step, srcCoords);
        srcPos.y++;

        tmp_sum += data[sy_index];
        sumOfCols[local_id] = tmp_sum;

        sy_index = sy_index + 1 < KERNEL_SIZE_Y ? sy_index + 1 : 0;
        barrier(CLK_LOCAL_MEM_FENCE);

        dst = (__global DT *)((__global uchar *)dst + dst_step);
    }
}
