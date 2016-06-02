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

#ifdef EXTRA_EXTRAPOLATION // border > src image size
#ifdef BORDER_CONSTANT
// CCCCCC|abcdefgh|CCCCCCC
#define EXTRAPOLATE(x, minV, maxV)
#elif defined BORDER_REPLICATE
// aaaaaa|abcdefgh|hhhhhhh
#define EXTRAPOLATE(x, minV, maxV) \
    { \
        (x) = clamp((x), (minV), (maxV)-1); \
    }
#elif defined BORDER_WRAP
// cdefgh|abcdefgh|abcdefg
#define EXTRAPOLATE(x, minV, maxV) \
    { \
        if ((x) < (minV)) \
            (x) += ((maxV) - (minV)); \
        if ((x) >= (maxV)) \
            (x) -= ((maxV) - (minV)); \
    }
#elif defined BORDER_REFLECT
// fedcba|abcdefgh|hgfedcb
#define EXTRAPOLATE(x, minV, maxV) \
    { \
        if ((maxV) - (minV) == 1) \
            (x) = (minV); \
        else \
            while ((x) >= (maxV) || (x) < (minV)) \
            { \
                if ((x) < (minV)) \
                    (x) = (minV) - ((x) - (minV)) - 1; \
                else \
                    (x) = (maxV) - 1 - ((x) - (maxV)); \
            } \
    }
#elif defined BORDER_REFLECT_101 || defined BORDER_REFLECT101
// gfedcb|abcdefgh|gfedcba
#define EXTRAPOLATE(x, minV, maxV) \
    { \
        if ((maxV) - (minV) == 1) \
            (x) = (minV); \
        else \
            while ((x) >= (maxV) || (x) < (minV)) \
            { \
                if ((x) < (minV)) \
                    (x) = (minV) - ((x) - (minV)); \
                else \
                    (x) = (maxV) - 1 - ((x) - (maxV)) - 1; \
            } \
    }
#else
#error No extrapolation method
#endif
#else
#ifdef BORDER_CONSTANT
// CCCCCC|abcdefgh|CCCCCCC
#define EXTRAPOLATE(x, minV, maxV)
#elif defined BORDER_REPLICATE
// aaaaaa|abcdefgh|hhhhhhh
#define EXTRAPOLATE(x, minV, maxV) \
    { \
        (x) = clamp((x), (minV), (maxV)-1); \
    }
#elif defined BORDER_WRAP
// cdefgh|abcdefgh|abcdefg
#define EXTRAPOLATE(x, minV, maxV) \
    { \
        if ((x) < (minV)) \
            (x) += (((minV) - (x)) / ((maxV) - (minV)) + 1) * ((maxV) - (minV)); \
        if ((x) >= (maxV)) \
            (x) = ((x) - (minV)) % ((maxV) - (minV)) + (minV); \
    }
#elif defined BORDER_REFLECT
// fedcba|abcdefgh|hgfedcb
#define EXTRAPOLATE(x, minV, maxV) \
    { \
        (x) = clamp((x), 2 * (minV) - (x) - 1, 2 * (maxV) - (x) - 1); \
    }
#elif defined BORDER_REFLECT_101 || defined BORDER_REFLECT101
// gfedcb|abcdefgh|gfedcba
#define EXTRAPOLATE(x, minV, maxV) \
    { \
        (x) = clamp((x), 2 * (minV) - (x), 2 * (maxV) - (x) - 2); \
    }
#else
#error No extrapolation method
#endif
#endif //EXTRA_EXTRAPOLATION


#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

#if cn != 3
#define loadpix(addr) *(__global const srcT *)(addr)
#define storepix(val, addr)  *(__global dstT *)(addr) = val
#define SRCSIZE (int)sizeof(srcT)
#define DSTSIZE (int)sizeof(dstT)
#else
#define loadpix(addr) vload3(0, (__global const srcT1 *)(addr))
#define storepix(val, addr) vstore3(val, 0, (__global dstT1 *)(addr))
#define SRCSIZE (int)sizeof(srcT1) * cn
#define DSTSIZE (int)sizeof(dstT1) * cn
#endif

#define UPDATE_COLUMN_SUM(col) \
    __constant WT1 * k = &kernelData[KERNEL_SIZE_Y2_ALIGNED * col]; \
    WT tmp_sum = 0;                                                 \
    for (int sy = 0; sy < KERNEL_SIZE_Y; sy++)                      \
        tmp_sum += data[sy] * k[sy];                                \
    sumOfCols[local_id] = tmp_sum;                                  \
    barrier(CLK_LOCAL_MEM_FENCE);

#define UPDATE_TOTAL_SUM(col) \
    int id = local_id + col - ANCHOR_X; \
    if (id >= 0 && id < LOCAL_SIZE)     \
        total_sum += sumOfCols[id];     \
    barrier(CLK_LOCAL_MEM_FENCE);

#define noconvert

#define DIG(a) a,
__constant WT1 kernelData[] = { COEFF };

__kernel void filter2D(__global const uchar * srcptr, int src_step, int srcOffsetX, int srcOffsetY, int srcEndX, int srcEndY,
                       __global uchar * dstptr, int dst_step, int dst_offset, int rows, int cols, float delta)
{
    int local_id = get_local_id(0);
    int x = local_id + (LOCAL_SIZE - (KERNEL_SIZE_X - 1)) * get_group_id(0) - ANCHOR_X;
    int y = get_global_id(1);

    WT data[KERNEL_SIZE_Y];
    __local WT sumOfCols[LOCAL_SIZE];

#ifdef BORDER_ISOLATED
    int srcBeginX = srcOffsetX;
    int srcBeginY = srcOffsetY;
#else
    int srcBeginX = 0;
    int srcBeginY = 0;
#endif

    int srcX = srcOffsetX + x;
    int srcY = srcOffsetY + y - ANCHOR_Y;

    __global dstT *dst = (__global dstT *)(dstptr + mad24(y, dst_step, mad24(x, DSTSIZE, dst_offset))); // Pointer can be out of bounds!

#ifdef BORDER_CONSTANT
    if (srcX >= srcBeginX && srcX < srcEndX)
    {
        for (int sy = 0, sy_index = 0; sy < KERNEL_SIZE_Y; sy++, srcY++)
        {
            if (srcY >= srcBeginY && srcY < srcEndY)
                data[sy + sy_index] = convertToWT(loadpix(srcptr + mad24(srcY, src_step, srcX * SRCSIZE)));
            else
                data[sy + sy_index] = (WT)(0);
        }
    }
    else
    {
        for (int sy = 0, sy_index = 0; sy < KERNEL_SIZE_Y; sy++, srcY++)
        {
             data[sy + sy_index] = (WT)(0);
        }
    }
#else
    EXTRAPOLATE(srcX, srcBeginX, srcEndX);
    for (int sy = 0, sy_index = 0; sy < KERNEL_SIZE_Y; sy++, srcY++)
    {
        int tempY = srcY;
        EXTRAPOLATE(tempY, srcBeginY, srcEndY);
        data[sy + sy_index] = convertToWT(loadpix(srcptr + mad24(tempY, src_step, srcX * SRCSIZE)));
    }
#endif

    WT total_sum = 0;
    for (int sx = 0; sx < ANCHOR_X; sx++)
    {
        UPDATE_COLUMN_SUM(sx);
        UPDATE_TOTAL_SUM(sx);
    }

    __constant WT1 * k = &kernelData[KERNEL_SIZE_Y2_ALIGNED * ANCHOR_X];
    for (int sy = 0; sy < KERNEL_SIZE_Y; sy++)
        total_sum += data[sy] * k[sy];

    for (int sx = ANCHOR_X + 1; sx < KERNEL_SIZE_X; sx++)
    {
        UPDATE_COLUMN_SUM(sx);
        UPDATE_TOTAL_SUM(sx);
    }

    if (local_id >= ANCHOR_X && local_id < LOCAL_SIZE - (KERNEL_SIZE_X - 1 - ANCHOR_X) && x >= 0 && x < cols)
        storepix(convertToDstT(total_sum + (WT)(delta)), dst);
}
