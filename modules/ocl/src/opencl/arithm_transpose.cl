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
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Jia Haipeng, jiahaipeng95@gmail.com
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

#define TILE_DIM      32
#define BLOCK_ROWS    8
#define LDS_STEP      TILE_DIM

__kernel void transpose(__global const T* src, __global T* dst,
    int src_cols, int src_rows,
    int src_step, int dst_step,
    int src_offset, int dst_offset)
{
    int gp_x = get_group_id(0),   gp_y = get_group_id(1);
    int gs_x = get_num_groups(0), gs_y = get_num_groups(1);

    int groupId_x, groupId_y;

    if(src_rows == src_cols)
    {
        groupId_y = gp_x;
        groupId_x = (gp_x + gp_y) % gs_x;
    }
    else
    {
        int bid = gp_x + gs_x * gp_y;
        groupId_y =  bid % gs_y;
        groupId_x = ((bid / gs_y) + groupId_y) % gs_x;
    }

    int lx = get_local_id(0);
    int ly = get_local_id(1);

    int x = groupId_x * TILE_DIM + lx;
    int y = groupId_y * TILE_DIM + ly;

    int x_index = groupId_y * TILE_DIM + lx;
    int y_index = groupId_x * TILE_DIM + ly;

    __local T title[TILE_DIM * LDS_STEP];

    if (x < src_cols && y < src_rows)
    {
        int index_src = mad24(y, src_step, x);

        for(int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
        {
            if (y + i < src_rows)
            {
                title[(ly + i) * LDS_STEP + lx] = src[src_offset + index_src];
                index_src = mad24(BLOCK_ROWS, src_step, index_src);
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x_index < src_rows && y_index < src_cols)
    {
        int index_dst = mad24(y_index, dst_step, x_index);

        for(int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
        {
            if ((y_index + i) < src_cols)
            {
                dst[dst_offset + index_dst] = title[lx * LDS_STEP + ly + i];
                index_dst +=  dst_step * BLOCK_ROWS;
            }
        }
    }
}

__kernel void transpose_inplace(__global T* src, __global T* dst,
    int src_cols, int src_rows,
    int src_step, int dst_step,
    int src_offset, int dst_offset)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < src_rows && x < y)
    {
        int srcIdx = mad24(y, src_step, src_offset + x);
        int dstIdx = mad24(x, dst_step, dst_offset + y);

        T tmp = dst[dstIdx];
        dst[dstIdx] = src[srcIdx];
        src[srcIdx] = tmp;
    }
}
