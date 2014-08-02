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

#ifdef BORDER_REPLICATE
//BORDER_REPLICATE:     aaaaaa|abcdefgh|hhhhhhh
#define ADDR_L(i, l_edge, r_edge)  ((i) <  (l_edge) ? (l_edge)   : (i))
#define ADDR_R(i, r_edge, addr)    ((i) >= (r_edge) ? (r_edge)-1 : (addr))
#define ADDR_H(i, t_edge, b_edge)  ((i) <  (t_edge) ? (t_edge)   :(i))
#define ADDR_B(i, b_edge, addr)    ((i) >= (b_edge) ? (b_edge)-1 :(addr))
#endif

#ifdef BORDER_REFLECT
//BORDER_REFLECT:       fedcba|abcdefgh|hgfedcb
#define ADDR_L(i, l_edge, r_edge)  ((i) <  (l_edge) ? -(i)-1               : (i))
#define ADDR_R(i, r_edge, addr)    ((i) >= (r_edge) ? -(i)-1+((r_edge)<<1) : (addr))
#define ADDR_H(i, t_edge, b_edge)  ((i) <  (t_edge) ? -(i)-1 : (i))
#define ADDR_B(i, b_edge, addr)    ((i) >= (b_edge) ? -(i)-1+((b_edge)<<1) : (addr))
#endif

#ifdef BORDER_REFLECT_101
//BORDER_REFLECT_101:   gfedcb|abcdefgh|gfedcba
#define ADDR_L(i, l_edge, r_edge)  ((i) <  (l_edge) ? -(i)                 : (i))
#define ADDR_R(i, r_edge, addr)    ((i) >= (r_edge) ? -(i)-2+((r_edge)<<1) : (addr))
#define ADDR_H(i, t_edge, b_edge)  ((i) <  (t_edge) ? -(i)                 : (i))
#define ADDR_B(i, b_edge, addr)    ((i) >= (b_edge) ? -(i)-2+((b_edge)<<1) : (addr))
#endif

//blur function does not support BORDER_WRAP
#ifdef BORDER_WRAP
//BORDER_WRAP:          cdefgh|abcdefgh|abcdefg
#define ADDR_L(i, l_edge, r_edge)  ((i) <  (l_edge) ? (i)+(r_edge) : (i))
#define ADDR_R(i, r_edge, addr)    ((i) >= (r_edge) ? (i)-(r_edge) : (addr))
#define ADDR_H(i, t_edge, b_edge)  ((i) <  (t_edge) ? (i)+(b_edge) : (i))
#define ADDR_B(i, b_edge, addr)    ((i) >= (b_edge) ? (i)-(b_edge) : (addr))
#endif

#ifdef EXTRA_EXTRAPOLATION // border > src image size
#ifdef BORDER_CONSTANT
// None
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
#elif defined(BORDER_REFLECT_101) || defined(BORDER_REFLECT101)
#define EXTRAPOLATE(x, y, minX, minY, maxX, maxY) EXTRAPOLATE_(x, y, minX, minY, maxX, maxY, 1)
#endif
#else
#error No extrapolation method
#endif
#else
#define EXTRAPOLATE(x, y, minX, minY, maxX, maxY) \
    { \
        int _row = y - minY, _col = x - minX; \
        _row = ADDR_H(_row, 0, maxY - minY); \
        _row = ADDR_B(_row, maxY - minY, _row); \
        y = _row + minY; \
        \
        _col = ADDR_L(_col, 0, maxX - minX); \
        _col = ADDR_R(_col, maxX - minX, _col); \
        x = _col + minX; \
    }
#endif

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

#define noconvert

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
        return convertToWT(loadpix(srcptr + mad24(pos.y, src_step, pos.x * SRCSIZE)));
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
                srcCoords.x2, srcCoords.y2
         );

        return convertToWT(loadpix(srcptr + mad24(selected_row, src_step, selected_col * SRCSIZE)));
#endif
    }
}

#define DIG(a) a,
__constant WT1 kernelData[] = { COEFF };

__kernel void filter2D(__global const uchar * srcptr, int src_step, int srcOffsetX, int srcOffsetY, int srcEndX, int srcEndY,
                       __global uchar * dstptr, int dst_step, int dst_offset, int rows, int cols, float delta)
{
    const struct RectCoords srcCoords = { srcOffsetX, srcOffsetY, srcEndX, srcEndY }; // for non-isolated border: offsetX, offsetY, wholeX, wholeY

    int local_id = get_local_id(0);
    int x = local_id + (LOCAL_SIZE - (KERNEL_SIZE_X - 1)) * get_group_id(0) - ANCHOR_X;
    int y = get_global_id(1) * BLOCK_SIZE_Y;

    WT data[KERNEL_SIZE_Y];
    __local WT sumOfCols[LOCAL_SIZE];

    int2 srcPos = (int2)(srcCoords.x1 + x, srcCoords.y1 + y - ANCHOR_Y);

    int2 pos = (int2)(x, y);
    __global dstT * dst = (__global dstT *)(dstptr + mad24(pos.y, dst_step, mad24(pos.x, DSTSIZE, dst_offset))); // Pointer can be out of bounds!
    bool writeResult = local_id >= ANCHOR_X && local_id < LOCAL_SIZE - (KERNEL_SIZE_X - 1 - ANCHOR_X) &&
                        pos.x >= 0 && pos.x < cols;

#if BLOCK_SIZE_Y > 1
    bool readAllpixels = true;
    int sy_index = 0; // current index in data[] array

    dstRowsMax = min(rows, pos.y + BLOCK_SIZE_Y);
    for ( ;
          pos.y < dstRowsMax;
          pos.y++, dst = (__global dstT *)((__global uchar *)dst + dst_step))
#endif
    {
        for (
#if BLOCK_SIZE_Y > 1
            int sy = readAllpixels ? 0 : -1; sy < (readAllpixels ? KERNEL_SIZE_Y : 0);
#else
            int sy = 0, sy_index = 0; sy < KERNEL_SIZE_Y;
#endif
            sy++, srcPos.y++)
        {
            data[sy + sy_index] = readSrcPixel(srcPos, srcptr, src_step, srcCoords);
        }

        WT total_sum = 0;
        for (int sx = 0; sx < KERNEL_SIZE_X; sx++)
        {
            {
                __constant WT1 * k = &kernelData[KERNEL_SIZE_Y2_ALIGNED * sx
#if BLOCK_SIZE_Y > 1
                                                   + KERNEL_SIZE_Y - sy_index
#endif
                                                   ];
                WT tmp_sum = 0;
                for (int sy = 0; sy < KERNEL_SIZE_Y; sy++)
                    tmp_sum += data[sy] * k[sy];

                sumOfCols[local_id] = tmp_sum;
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            int id = local_id + sx - ANCHOR_X;
            if (id >= 0 && id < LOCAL_SIZE)
               total_sum += sumOfCols[id];

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (writeResult)
            storepix(convertToDstT(total_sum + (WT)(delta)), dst);

#if BLOCK_SIZE_Y > 1
        readAllpixels = false;
#if BLOCK_SIZE_Y > KERNEL_SIZE_Y
        sy_index = sy_index + 1 <= KERNEL_SIZE_Y ? sy_index + 1 : 1;
#else
        sy_index++;
#endif
#endif // BLOCK_SIZE_Y == 1
    }
}
