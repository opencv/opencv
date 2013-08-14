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
//    Pang Erping, erping@multicorewareinc.com
//    Jia Haipeng, jiahaipeng95@gmail.com
//    Peng Xiao, pengxiao@outlook.com
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

///////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////Macro for border type////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef BORDER_REPLICATE

//BORDER_REPLICATE:     aaaaaa|abcdefgh|hhhhhhh
#define ADDR_L(i, l_edge, r_edge)  ((i) <  (l_edge) ? (l_edge)   : (i))
#define ADDR_R(i, r_edge, addr)    ((i) >= (r_edge) ? (r_edge)-1 : (addr))
#define ADDR_H(i, t_edge, b_edge)  ((i) <  (t_edge) ? (t_edge)   : (i))
#define ADDR_B(i, b_edge, addr)    ((i) >= (b_edge) ? (b_edge)-1 :(addr))
#endif

#ifdef BORDER_REFLECT
#define ADDR_L(i, l_edge, r_edge)  ((i) <  (l_edge) ? ((l_edge)<<1)-(i)-1                 : (i))
#define ADDR_R(i, r_edge, addr)    ((i) >= (r_edge) ? -(i)-1+((r_edge)<<1) : (addr))
#define ADDR_H(i, t_edge, b_edge)  ((i) <  (t_edge) ? ((t_edge)<<1)-(i)-1                 : (i))
#define ADDR_B(i, b_edge, addr)    ((i) >= (b_edge) ? -(i)-1+((b_edge)<<1) : (addr))
#endif

#ifdef BORDER_REFLECT_101
//BORDER_REFLECT_101:   gfedcb|abcdefgh|gfedcba
#define ADDR_L(i, l_edge, r_edge)  ((i) <  (l_edge) ? ((l_edge)<<1)-(i)                 : (i))
#define ADDR_R(i, r_edge, addr)    ((i) >= (r_edge) ? -(i)-2+((r_edge)<<1) : (addr))
#define ADDR_H(i, t_edge, b_edge)  ((i) <  (t_edge) ? ((t_edge)<<1)-(i)                 : (i))
#define ADDR_B(i, b_edge, addr)    ((i) >= (b_edge) ? -(i)-2+((b_edge)<<1) : (addr))
#endif

#ifdef IMG_C_1_0
#define T_IMG   uchar
#define T_IMGx4 uchar4
#define T_IMG_C1 uchar
#define CONVERT_TYPE   convert_uchar_sat
#define CONVERT_TYPEx4 convert_uchar4_sat
#endif
#ifdef IMG_C_4_0
#define T_IMG   uchar4
#define T_IMGx4 uchar16
#define T_IMG_C1 uchar
#define CONVERT_TYPE   convert_uchar4_sat
#define CONVERT_TYPEx4 convert_uchar16_sat
#endif
#ifdef IMG_C_1_5
#define T_IMG   float
#define T_IMGx4 float4
#define T_IMG_C1 float
#define CONVERT_TYPE   convert_float
#define CONVERT_TYPEx4 convert_float4
#endif
#ifdef IMG_C_4_5
#define T_IMG   float4
#define T_IMGx4 float16
#define T_IMG_C1 float
#define CONVERT_TYPE   convert_float4
#define CONVERT_TYPEx4 convert_float16
#endif

#ifndef CN
#define CN 1
#endif

#if CN == 1
#define T_SUM   float
#define T_SUMx4 float4
#define CONVERT_TYPE_SUM   convert_float
#define CONVERT_TYPE_SUMx4 convert_float4
#define SUM_ZERO   (0.0f)
#define SUM_ZEROx4 (0.0f, 0.0f, 0.0f, 0.0f)
#define VLOAD4 vload4
#define SX x
#define SY y
#define SZ z
#define SW w
#elif CN == 4
#define T_SUM float4
#define T_SUMx4 float16
#define CONVERT_TYPE_SUM   convert_float4
#define CONVERT_TYPE_SUMx4 convert_float16
#define SUM_ZERO   (0.0f, 0.0f, 0.0f, 0.0f)
#define SUM_ZEROx4 (0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f)
#define VLOAD4 vload16
#define SX s0123
#define SY s4567
#define SZ s89ab
#define SW scdef
#endif

#ifndef FILTER_SIZE
#define FILTER_SIZE 3
#endif

#define LOCAL_GROUP_SIZE 16

#define LOCAL_WIDTH  ((FILTER_SIZE/2)*2 + LOCAL_GROUP_SIZE)
#define LOCAL_HEIGHT ((FILTER_SIZE/2)*2 + LOCAL_GROUP_SIZE)

#define FILTER_RADIUS (FILTER_SIZE >> 1)

__kernel void filter2D(
    __global T_IMG *src,
    __global T_IMG *dst,
    int src_step,
    int dst_step,
    __constant float *mat_kernel,
    __local T_IMG *local_data,
    int wholerows,
    int wholecols,
    int src_offset_x,
    int src_offset_y,
    int dst_offset_x,
    int dst_offset_y,
    int cols,
    int rows,
    int operate_cols
)
{
    int groupStartCol = get_group_id(0) * get_local_size(0);
    int groupStartRow = get_group_id(1) * get_local_size(1);

    int localCol = get_local_id(0);
    int localRow = get_local_id(1);
    int globalCol = groupStartCol + localCol;
    int globalRow = groupStartRow + localRow;
    const int src_offset = mad24(src_offset_y, src_step, src_offset_x);
    const int dst_offset = mad24(dst_offset_y, dst_step, dst_offset_x);
#ifdef BORDER_CONSTANT
    for(int i = localRow; i < LOCAL_HEIGHT; i += get_local_size(1))
    {
        int curRow = groupStartRow + i;
        for(int j = localCol; j < LOCAL_WIDTH; j += get_local_size(0))
        {
            int curCol = groupStartCol + j;
            if(curRow < FILTER_RADIUS - src_offset_y || (curRow - FILTER_RADIUS) >= wholerows - src_offset_y||
                curCol < FILTER_RADIUS - src_offset_x || (curCol - FILTER_RADIUS) >= wholecols - src_offset_x)
            {
                local_data[(i) * LOCAL_WIDTH + j] = 0;
            }
            else
            {
                local_data[(i) * LOCAL_WIDTH + j] = src[(curRow - FILTER_RADIUS) * src_step + curCol - FILTER_RADIUS + src_offset];
            }
        }
    }
#else
    for(int i = localRow; i < LOCAL_HEIGHT; i += get_local_size(1))
    {
        int curRow = groupStartRow + i;

        curRow = ADDR_H(curRow, FILTER_RADIUS - src_offset_y, wholerows - src_offset_y);

        curRow = ADDR_B(curRow - FILTER_RADIUS, wholerows - src_offset_y, curRow - FILTER_RADIUS);

        for(int j = localCol; j < LOCAL_WIDTH; j += get_local_size(0))
        {
            int curCol = groupStartCol + j;
            curCol = ADDR_L(curCol, FILTER_RADIUS - src_offset_x, wholecols - src_offset_x);
            curCol = ADDR_R(curCol - FILTER_RADIUS, wholecols - src_offset_x, curCol - FILTER_RADIUS);
            if(curRow < wholerows  && curCol < wholecols)
            {
                local_data[(i) * LOCAL_WIDTH + j] = src[(curRow) * src_step + curCol + src_offset];
            }
        }
    }
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
    if(globalRow < rows && globalCol < cols)
    {
        T_SUM sum = (T_SUM)SUM_ZERO;
        int filterIdx = 0;
        for(int i = 0; i < FILTER_SIZE; i++)
        {
            int offset = (i + localRow) * LOCAL_WIDTH;

            for(int j = 0; j < FILTER_SIZE; j++)
            {
                sum += CONVERT_TYPE_SUM(local_data[offset + j + localCol]) * mat_kernel[filterIdx++];
            }
        }
        dst[(globalRow)*dst_step + (globalCol) + dst_offset] = CONVERT_TYPE(sum);
    }
}

/// following is specific for 3x3 kernels

//////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////Macro for define elements number per thread/////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
#define ANX                     1
#define ANY                     1

#define ROWS_PER_GROUP          4
#define ROWS_PER_GROUP_BITS     2
#define ROWS_FETCH              (ROWS_PER_GROUP + ANY + ANY)   //(ROWS_PER_GROUP + anY * 2)

#define THREADS_PER_ROW         64
#define THREADS_PER_ROW_BIT     6

#define ELEMENTS_PER_THREAD     4
#define ELEMENTS_PER_THREAD_BIT 2

#define LOCAL_MEM_STEP          260 //divup((get_local_size(0) + anX * 2), 4) * 4

///////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////8uC1////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void filter2D_3x3(
    __global T_IMG *src,
    __global T_IMG *dst,
    int src_step,
    int dst_step,
    __constant float *mat_kernel,
    __local T_IMG *local_data,
    int wholerows,
    int wholecols,
    int src_offset_x,
    int src_offset_y,
    int dst_offset_x,
    int dst_offset_y,
    int cols,
    int rows,
    int operate_cols
)
{
    int gX = get_global_id(0);
    int gY = get_global_id(1);

    int lX = get_local_id(0);

    int groupX_size = get_local_size(0);
    int groupX_id   = get_group_id(0);

#define dst_align (dst_offset_x & 3)
    int cols_start_index_group = src_offset_x - dst_align + groupX_size * groupX_id - ANX;
    int rows_start_index       = src_offset_y + (gY << ROWS_PER_GROUP_BITS) - ANY;

    if((gY << 2) < rows)
    {
        for(int i = 0; i < ROWS_FETCH; ++i)
        {
            if((rows_start_index - src_offset_y) + i < rows + ANY)
            {
#ifdef BORDER_CONSTANT
                int selected_row  = rows_start_index + i;
                int selected_cols = cols_start_index_group + lX;

                T_IMG data = src[mad24(selected_row, src_step, selected_cols)];
                int con = selected_row >= 0 && selected_row < wholerows && selected_cols >= 0 && selected_cols < wholecols;
                data = con ? data : 0;
                local_data[mad24(i, LOCAL_MEM_STEP, lX)] = data;

                if(lX < (ANX << 1))
                {
                    selected_cols = cols_start_index_group + lX + groupX_size;

                    data  = src[mad24(selected_row, src_step, selected_cols)];
                    con = selected_row >= 0 && selected_row < wholerows && selected_cols >= 0 && selected_cols < wholecols;
                    data = con ? data : 0;
                    local_data[mad24(i, LOCAL_MEM_STEP, lX) + groupX_size] = data;
                }
#else
                int selected_row = ADDR_H(rows_start_index + i,  0, wholerows);
                selected_row     = ADDR_B(rows_start_index + i, wholerows, selected_row);

                int selected_cols = ADDR_L(cols_start_index_group + lX, 0, wholecols);
                selected_cols     = ADDR_R(cols_start_index_group + lX, wholecols, selected_cols);

                T_IMG data = src[mad24(selected_row, src_step, selected_cols)];

                local_data[mad24(i, LOCAL_MEM_STEP, lX)] = data;

                if(lX < (ANX << 1))
                {
                    selected_cols = cols_start_index_group + lX + groupX_size;
                    selected_cols = ADDR_R(selected_cols, wholecols, selected_cols);

                    data = src[mad24(selected_row, src_step, selected_cols)];
                    local_data[mad24(i, LOCAL_MEM_STEP, lX) + groupX_size] = data;
                }
#endif
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int process_col = groupX_size * groupX_id + ((lX % THREADS_PER_ROW) << 2);
    if(((gY << 2) < rows) && (process_col < operate_cols))
    {
        int dst_cols_start = dst_offset_x;
        int dst_cols_end   = dst_offset_x + cols;
        int dst_cols_index = (dst_offset_x + process_col) & 0xfffffffc;

        int dst_rows_end   = dst_offset_y + rows;
        int dst_rows_index = dst_offset_y + (gY << ROWS_PER_GROUP_BITS) + (lX >> THREADS_PER_ROW_BIT);
        dst = dst + mad24(dst_rows_index, dst_step, dst_cols_index);

        T_IMGx4 dst_data = *(__global T_IMGx4 *)dst;

        T_SUMx4 sum = (T_SUMx4)SUM_ZEROx4;
        T_IMGx4 data;

        for(int i = 0; i < FILTER_SIZE; i++)
        {
#pragma unroll
            for(int j = 0; j < FILTER_SIZE; j++)
            {
                if(dst_rows_index < dst_rows_end)
                {
                    int local_row = (lX >> THREADS_PER_ROW_BIT) + i;
                    int local_cols = ((lX % THREADS_PER_ROW) << ELEMENTS_PER_THREAD_BIT) + j;

                    data = VLOAD4(0, (__local T_IMG_C1 *)(local_data + local_row * LOCAL_MEM_STEP + local_cols));
                    sum = sum + (mat_kernel[i * FILTER_SIZE + j] * CONVERT_TYPE_SUMx4(data));
                }
            }
        }
        if(dst_rows_index < dst_rows_end)
        {
            T_IMGx4 tmp_dst = CONVERT_TYPEx4(sum);
            tmp_dst.SX = ((dst_cols_index + 0 >= dst_cols_start) && (dst_cols_index + 0 < dst_cols_end)) ?
                         tmp_dst.SX : dst_data.SX;
            tmp_dst.SY = ((dst_cols_index + 1 >= dst_cols_start) && (dst_cols_index + 1 < dst_cols_end)) ?
                         tmp_dst.SY : dst_data.SY;
            tmp_dst.SZ = ((dst_cols_index + 2 >= dst_cols_start) && (dst_cols_index + 2 < dst_cols_end)) ?
                         tmp_dst.SZ : dst_data.SZ;
            tmp_dst.SW = ((dst_cols_index + 3 >= dst_cols_start) && (dst_cols_index + 3 < dst_cols_end)) ?
                         tmp_dst.SW : dst_data.SW;
            *(__global T_IMGx4 *)dst = tmp_dst;
        }
    }
}
