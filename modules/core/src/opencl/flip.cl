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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the OpenCV Foundation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#if kercn != 3
#define loadpix(addr) *(__global const T *)(addr)
#define storepix(val, addr)  *(__global T *)(addr) = val
#define storepix_2(val0, val1, addr0, addr1) \
    *(__global T *)(addr0) = val0; *(__global T *)(addr1) = val1
#define TSIZE (int)sizeof(T)
#else
#define loadpix(addr) vload3(0, (__global const T1 *)(addr))
#define storepix(val, addr) vstore3(val, 0, (__global T1 *)(addr))
#if DEPTH == 2 || DEPTH == 3
#define storepix_2(val0, val1, addr0, addr1) \
    ((__global T1 *)(addr0))[0] = val0.x; \
    ((__global T1 *)(addr1))[0] = val1.x; \
    ((__global T1 *)(addr0))[1] = val0.y; \
    ((__global T1 *)(addr1))[1] = val1.y; \
    ((__global T1 *)(addr0))[2] = val0.z; \
    ((__global T1 *)(addr1))[2] = val1.z
#else
#define storepix_2(val0, val1, addr0, addr1) \
    storepix(val0, addr0); \
    storepix(val1, addr1)
#endif
#define TSIZE ((int)sizeof(T1)*3)
#endif
#define LDS_STEP (TILE_SIZE + 1)

#ifndef INPLACE

__kernel void arithm_flip_rows(__global const uchar * srcptr, int src_step, int src_offset,
                               __global uchar * dstptr, int dst_step, int dst_offset,
                               int rows, int cols, int thread_rows, int thread_cols)
{
    int x = get_global_id(0);
    int y0 = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        int src_index0 = mad24(y0, src_step, mad24(x, TSIZE, src_offset));
        int src_index1 = mad24(rows - y0 - 1, src_step, mad24(x, TSIZE, src_offset));
        int dst_index0 = mad24(y0, dst_step, mad24(x, TSIZE, dst_offset));
        int dst_index1 = mad24(rows - y0 - 1, dst_step, mad24(x, TSIZE, dst_offset));

        #pragma unroll
        for (int y = y0, y1 = min(thread_rows, y0 + PIX_PER_WI_Y); y < y1; ++y)
        {
            T src0 = loadpix(srcptr + src_index0);
            T src1 = loadpix(srcptr + src_index1);

            storepix_2(src1, src0, dstptr + dst_index0, dstptr + dst_index1);

            src_index0 += src_step;
            src_index1 -= src_step;
            dst_index0 += dst_step;
            dst_index1 -= dst_step;
        }
    }
}

__kernel void arithm_flip_rows_cols(__global const uchar * srcptr, int src_step, int src_offset,
                                    __global uchar * dstptr, int dst_step, int dst_offset,
                                    int rows, int cols, int thread_rows, int thread_cols)
{
    int x = get_global_id(0);
    int y0 = get_global_id(1)*PIX_PER_WI_Y;

    if (x < cols)
    {
        int src_index0 = mad24(y0, src_step, mad24(x, TSIZE, src_offset));
        int src_index1 = mad24(rows - y0 - 1, src_step, mad24(cols - x - 1, TSIZE, src_offset));
        int dst_index0 = mad24(y0, dst_step, mad24(x, TSIZE, dst_offset));
        int dst_index1 = mad24(rows - y0 - 1, dst_step, mad24(cols - x - 1, TSIZE, dst_offset));

        #pragma unroll
        for (int y = y0, y1 = min(thread_rows, y0 + PIX_PER_WI_Y); y < y1; ++y)
        {
            T src0 = loadpix(srcptr + src_index0);
            T src1 = loadpix(srcptr + src_index1);

#if kercn == 2
#if cn == 1
            src0 = src0.s10;
            src1 = src1.s10;
#endif
#elif kercn == 4
#if cn == 1
            src0 = src0.s3210;
            src1 = src1.s3210;
#elif cn == 2
            src0 = src0.s2301;
            src1 = src1.s2301;
#endif
#endif

            storepix_2(src1, src0, dstptr + dst_index0, dstptr + dst_index1);

            src_index0 += src_step;
            src_index1 -= src_step;
            dst_index0 += dst_step;
            dst_index1 -= dst_step;
        }
    }
}

__kernel void arithm_flip_cols(__global const uchar * srcptr, int src_step, int src_offset,
                               __global uchar * dstptr, int dst_step, int dst_offset,
                               int rows, int cols, int thread_rows, int thread_cols)
{
    int x = get_global_id(0);
    int y0 = get_global_id(1)*PIX_PER_WI_Y;

    if (x < thread_cols)
    {
        int src_index0 = mad24(y0, src_step, mad24(x, TSIZE, src_offset));
        int src_index1 = mad24(y0, src_step, mad24(cols - x - 1, TSIZE, src_offset));
        int dst_index0 = mad24(y0, dst_step, mad24(x, TSIZE, dst_offset));
        int dst_index1 = mad24(y0, dst_step, mad24(cols - x - 1, TSIZE, dst_offset));

        #pragma unroll
        for (int y = y0, y1 = min(rows, y0 + PIX_PER_WI_Y); y < y1; ++y)
        {
            T src0 = loadpix(srcptr + src_index0);
            T src1 = loadpix(srcptr + src_index1);

#if kercn == 2
#if cn == 1
            src0 = src0.s10;
            src1 = src1.s10;
#endif
#elif kercn == 4
#if cn == 1
            src0 = src0.s3210;
            src1 = src1.s3210;
#elif cn == 2
            src0 = src0.s2301;
            src1 = src1.s2301;
#endif
#endif

            storepix_2(src1, src0, dstptr + dst_index0, dstptr + dst_index1);

            src_index0 += src_step;
            src_index1 += src_step;
            dst_index0 += dst_step;
            dst_index1 += dst_step;
        }
    }
}

#else

__kernel void arithm_flip_rows_inplace(__global uchar * srcptr, int src_step, int src_offset,
                                       int rows, int cols)
{
    int gp_x = get_group_id(0);
    int gp_y = get_group_id(1);
    int lx = get_local_id(0);
    int ly = get_local_id(1);

    __local T tile_top[TILE_SIZE * LDS_STEP];
    __local T tile_bottom[TILE_SIZE * LDS_STEP];

    int half_rows = (rows + 1) / 2;

    int x = gp_x * TILE_SIZE + lx;
    int y_top = gp_y * TILE_SIZE + ly;
    int y_bottom = rows - 1 - y_top;

    #pragma unroll
    for (int i = 0; i < TILE_SIZE; i += BLOCK_ROWS)
    {
        if (x < cols && y_top + i < half_rows)
        {
            int curr_y_top = y_top + i;
            int curr_y_bottom = rows - 1 - curr_y_top;

            T val_top = loadpix(srcptr + mad24(curr_y_top, src_step, mad24(x, TSIZE, src_offset)));
            T val_bottom = loadpix(srcptr + mad24(curr_y_bottom, src_step, mad24(x, TSIZE, src_offset)));

            tile_top[mad24(ly + i, LDS_STEP, lx)] = val_top;
            tile_bottom[mad24(ly + i, LDS_STEP, lx)] = val_bottom;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    #pragma unroll
    for (int i = 0; i < TILE_SIZE; i += BLOCK_ROWS)
    {
        if (x < cols && y_top + i < half_rows)
        {
            int curr_y_top = y_top + i;
            int curr_y_bottom = rows - 1 - curr_y_top;

            storepix(tile_bottom[mad24(ly + i, LDS_STEP, lx)],
                    srcptr + mad24(curr_y_top, src_step, mad24(x, TSIZE, src_offset)));

            if (curr_y_top != curr_y_bottom)
            {
                storepix(tile_top[mad24(ly + i, LDS_STEP, lx)],
                        srcptr + mad24(curr_y_bottom, src_step, mad24(x, TSIZE, src_offset)));
            }
        }
    }
}

__kernel void arithm_flip_rows_cols_inplace(__global uchar * srcptr, int src_step, int src_offset,
                                            int rows, int cols)
{
    int gp_x = get_group_id(0);
    int gp_y = get_group_id(1);
    int lx = get_local_id(0);
    int ly = get_local_id(1);

    __local T tile_first[TILE_SIZE * LDS_STEP];
    __local T tile_second[TILE_SIZE * LDS_STEP];

    int total_pixels = rows * cols;
    int half_pixels = (total_pixels + 1) / 2;

    int x_first = gp_x * TILE_SIZE + lx;
    int y_first = gp_y * TILE_SIZE + ly;

    int x_second = cols - 1 - x_first;
    int y_second = rows - 1 - y_first;

    #pragma unroll
    for (int i = 0; i < TILE_SIZE; i += BLOCK_ROWS)
    {
        int curr_y_first = y_first + i;
        int curr_y_second = rows - 1 - curr_y_first;
        int linear_idx = curr_y_first * cols + x_first;

        if (x_first < cols && curr_y_first < rows && linear_idx < half_pixels)
        {
            T val_first = loadpix(srcptr + mad24(curr_y_first, src_step, mad24(x_first, TSIZE, src_offset)));
            T val_second = loadpix(srcptr + mad24(curr_y_second, src_step, mad24(x_second, TSIZE, src_offset)));

#if kercn == 2
#if cn == 1
            val_first = val_first.s10;
            val_second = val_second.s10;
#endif
#elif kercn == 4
#if cn == 1
            val_first = val_first.s3210;
            val_second = val_second.s3210;
#elif cn == 2
            val_first = val_first.s2301;
            val_second = val_second.s2301;
#endif
#endif
            tile_first[mad24(ly + i, LDS_STEP, lx)] = val_first;
            tile_second[mad24(ly + i, LDS_STEP, lx)] = val_second;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    #pragma unroll
    for (int i = 0; i < TILE_SIZE; i += BLOCK_ROWS)
    {
        int curr_y_first = y_first + i;
        int curr_y_second = rows - 1 - curr_y_first;
        int linear_idx = curr_y_first * cols + x_first;

        if (x_first < cols && curr_y_first < rows && linear_idx < half_pixels)
        {
            storepix(tile_second[mad24(ly + i, LDS_STEP, lx)],
                    srcptr + mad24(curr_y_first, src_step, mad24(x_first, TSIZE, src_offset)));

            if (linear_idx != total_pixels - 1 - linear_idx)
            {
                storepix(tile_first[mad24(ly + i, LDS_STEP, lx)],
                        srcptr + mad24(curr_y_second, src_step, mad24(x_second, TSIZE, src_offset)));
            }
        }
    }
}

__kernel void arithm_flip_cols_inplace(__global uchar * srcptr, int src_step, int src_offset,
                                       int rows, int cols)
{
    int gp_x = get_group_id(0);
    int gp_y = get_group_id(1);
    int lx = get_local_id(0);
    int ly = get_local_id(1);

    __local T tile_left[TILE_SIZE * LDS_STEP];
    __local T tile_right[TILE_SIZE * LDS_STEP];

    int half_cols = (cols + 1) / 2;

    int x_left = gp_x * TILE_SIZE + lx;
    int y = gp_y * TILE_SIZE + ly;
    int x_right = cols - 1 - x_left;

    #pragma unroll
    for (int i = 0; i < TILE_SIZE; i += BLOCK_ROWS)
    {
        if (y + i < rows && x_left < half_cols)
        {
            T val_left = loadpix(srcptr + mad24(y + i, src_step, mad24(x_left, TSIZE, src_offset)));
            T val_right = loadpix(srcptr + mad24(y + i, src_step, mad24(x_right, TSIZE, src_offset)));

#if kercn == 2
#if cn == 1
            val_left = val_left.s10;
            val_right = val_right.s10;
#endif
#elif kercn == 4
#if cn == 1
            val_left = val_left.s3210;
            val_right = val_right.s3210;
#elif cn == 2
            val_left = val_left.s2301;
            val_right = val_right.s2301;
#endif
#endif
            tile_left[mad24(ly + i, LDS_STEP, lx)] = val_left;
            tile_right[mad24(ly + i, LDS_STEP, lx)] = val_right;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    #pragma unroll
    for (int i = 0; i < TILE_SIZE; i += BLOCK_ROWS)
    {
        if (y + i < rows && x_left < half_cols)
        {
            storepix(tile_right[mad24(ly + i, LDS_STEP, lx)],
                    srcptr + mad24(y + i, src_step, mad24(x_left, TSIZE, src_offset)));

            if (x_left != x_right)
            {
                storepix(tile_left[mad24(ly + i, LDS_STEP, lx)],
                        srcptr + mad24(y + i, src_step, mad24(x_right, TSIZE, src_offset)));
            }
        }
    }
}

#endif // INPLACE