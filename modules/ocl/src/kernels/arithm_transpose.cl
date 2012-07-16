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
//     and/or other oclMaterials provided with the distribution.
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

#define TILE_DIM      32 
#define BLOCK_ROWS    8 
#define LDS_STEP     (TILE_DIM + 1)


//8UC1 is not unoptimized, as the size of write per thread is 8 
//which will use completepath
__kernel void transpose_C1_D0(__global uchar* src, int src_step, int src_offset, 
                              __global uchar* dst, int dst_step, int dst_offset,
                              int src_rows, int src_cols)
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

    __local uchar title[TILE_DIM * LDS_STEP];

    if(x < src_cols && y < src_rows)
    {
        int index_src = mad24(y, src_step, x);

        #pragma unroll 
        for(int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
        {
            if(y + i < src_rows)
            {
                title[(ly + i) * LDS_STEP + lx] =*(src + src_offset + index_src);
                index_src = mad24(BLOCK_ROWS, src_step, index_src);
            }
        }
     }

     barrier(CLK_LOCAL_MEM_FENCE);

    if(x_index < src_rows && y_index < src_cols)
    {
        int index_dst = mad24(y_index, dst_step, x_index);

        #pragma unroll
        for(int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
        {
            if((y_index + i) < src_cols)
            {
                *(dst + dst_offset + index_dst ) = title[lx * LDS_STEP + ly + i];  
                index_dst +=  dst_step * BLOCK_ROWS ;
            }
        }
    }
}

__kernel void transpose_C1_D4(__global int* src, int src_step, int src_offset, 
                              __global int* dst, int dst_step, int dst_offset,
                              int src_rows, int src_cols)
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

    __local int title[TILE_DIM * LDS_STEP];

    if(x < src_cols && y < src_rows)
    {
        int index_src = mad24(y, src_step, (x << 2));

        #pragma unroll 
        for(int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
        {
            if(y + i < src_rows)
            {
                title[(ly + i) * LDS_STEP + lx] = *((__global int *)((__global char*)src + src_offset + index_src));
                index_src = mad24(BLOCK_ROWS, src_step, index_src);
            }
        }
     }

     barrier(CLK_LOCAL_MEM_FENCE);

    if(x_index < src_rows && y_index < src_cols)
    {
        int index_dst = mad24(y_index, dst_step, (x_index << 2));

        #pragma unroll
        for(int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
        {
            if((y_index + i) < src_cols)
            {
                *((__global int*)((__global char*)dst + dst_offset + index_dst )) = title[lx * LDS_STEP + ly + i];  
                index_dst +=  dst_step * BLOCK_ROWS ;
            }
        }
    }
}
__kernel void transpose_C1_D5(__global float* src, int src_step, int src_offset, 
                              __global float* dst, int dst_step, int dst_offset,
                              int src_rows, int src_cols)
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

    __local float title[TILE_DIM * LDS_STEP];

    if(x < src_cols && y < src_rows)
    {
        int index_src = mad24(y, src_step, (x << 2));

        #pragma unroll 
        for(int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
        {
            if(y + i < src_rows)
            {
                title[(ly + i) * LDS_STEP + lx] = *((__global float *)((__global char*)src + src_offset + index_src));
                index_src = mad24(BLOCK_ROWS, src_step, index_src);
            }
        }
     }

     barrier(CLK_LOCAL_MEM_FENCE);

    if(x_index < src_rows && y_index < src_cols)
    {
        int index_dst = mad24(y_index, dst_step, (x_index << 2));

        #pragma unroll
        for(int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
        {
            if((y_index + i) < src_cols)
            {
                *((__global float*)((__global char*)dst + dst_offset + index_dst )) = title[lx * LDS_STEP + ly + i];  
                index_dst +=  dst_step * BLOCK_ROWS ;
            }
        }
    }
}

__kernel void transpose_C2_D2(__global ushort* src, int src_step, int src_offset, 
                              __global ushort* dst, int dst_step, int dst_offset,
                              int src_rows, int src_cols)
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

    __local ushort2 title[TILE_DIM * LDS_STEP];

    if(x < src_cols && y < src_rows)
    {
        int index_src = mad24(y, src_step, (x << 2));

        #pragma unroll 
        for(int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
        {
            if(y + i < src_rows)
            {
                title[(ly + i) * LDS_STEP + lx] = *((__global ushort2 *)((__global char*)src + src_offset + index_src));
                index_src = mad24(BLOCK_ROWS, src_step, index_src);
            }
        }
     }

     barrier(CLK_LOCAL_MEM_FENCE);

    if(x_index < src_rows && y_index < src_cols)
    {
        int index_dst = mad24(y_index, dst_step, (x_index << 2));

        #pragma unroll
        for(int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
        {
            if((y_index + i) < src_cols)
            {
                *((__global ushort2*)((__global char*)dst + dst_offset + index_dst )) = title[lx * LDS_STEP + ly + i];  
                index_dst +=  dst_step * BLOCK_ROWS ;
            }
        }
    }
}
__kernel void transpose_C2_D3(__global short* src, int src_step, int src_offset, 
                              __global short* dst, int dst_step, int dst_offset,
                              int src_rows, int src_cols)
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

    __local short2 title[TILE_DIM * LDS_STEP];

    if(x < src_cols && y < src_rows)
    {
        int index_src = mad24(y, src_step, (x << 2));

        #pragma unroll 
        for(int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
        {
            if(y + i < src_rows)
            {
                title[(ly + i) * LDS_STEP + lx] = *((__global short2 *)((__global char*)src + src_offset + index_src));
                index_src = mad24(BLOCK_ROWS, src_step, index_src);
            }
        }
     }

     barrier(CLK_LOCAL_MEM_FENCE);

    if(x_index < src_rows && y_index < src_cols)
    {
        int index_dst = mad24(y_index, dst_step, (x_index << 2));

        #pragma unroll
        for(int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
        {
            if((y_index + i) < src_cols)
            {
                *((__global short2*)((__global char*)dst + dst_offset + index_dst )) = title[lx * LDS_STEP + ly + i];  
                index_dst +=  dst_step * BLOCK_ROWS ;
            }
        }
    }
}
__kernel void transpose_C4_D0(__global uchar* src, int src_step, int src_offset, 
                              __global uchar* dst, int dst_step, int dst_offset,
                              int src_rows, int src_cols)
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

    __local uchar4 title[TILE_DIM * LDS_STEP];

    if(x < src_cols && y < src_rows)
    {
        int index_src = mad24(y, src_step, (x << 2));

        #pragma unroll 
        for(int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
        {
            if(y + i < src_rows)
            {
                title[(ly + i) * LDS_STEP + lx] = *((__global uchar4 *)(src + src_offset + index_src));
                index_src = mad24(BLOCK_ROWS, src_step, index_src);
            }
        }
     }

     barrier(CLK_LOCAL_MEM_FENCE);

    if(x_index < src_rows && y_index < src_cols)
    {
        int index_dst = mad24(y_index, dst_step, (x_index << 2));

        #pragma unroll
        for(int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
        {
            if((y_index + i) < src_cols)
            {
                *((__global uchar4*)(dst + dst_offset + index_dst )) = title[lx * LDS_STEP + ly + i];  
                index_dst +=  dst_step * BLOCK_ROWS ;
            }
        }
    }
}

__kernel void transpose_C4_D1(__global char* src, int src_step, int src_offset, 
                              __global char* dst, int dst_step, int dst_offset,
                              int src_rows, int src_cols)
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

    __local char4 title[TILE_DIM * LDS_STEP];

    if(x < src_cols && y < src_rows)
    {
        int index_src = mad24(y, src_step, (x << 2));

        #pragma unroll 
        for(int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
        {
            if(y + i < src_rows)
            {
                title[(ly + i) * LDS_STEP + lx] = *((__global char4 *)(src + src_offset + index_src));
                index_src = mad24(BLOCK_ROWS, src_step, index_src);
            }
        }
     }

     barrier(CLK_LOCAL_MEM_FENCE);

    if(x_index < src_rows && y_index < src_cols)
    {
        int index_dst = mad24(y_index, dst_step, (x_index << 2));

        #pragma unroll
        for(int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
        {
            if((y_index + i) < src_cols)
            {
                *((__global char4*)(dst + dst_offset + index_dst )) = title[lx * LDS_STEP + ly + i];  
                index_dst +=  dst_step * BLOCK_ROWS ;
            }
        }
    }
}
