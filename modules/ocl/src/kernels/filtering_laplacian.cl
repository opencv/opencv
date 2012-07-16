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
#define BORDER_REFLECT_101

///////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////Macro for border type////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
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

#ifdef BORDER_WRAP
//BORDER_WRAP:          cdefgh|abcdefgh|abcdefg
#define ADDR_L(i, l_edge, r_edge)  ((i) <  (l_edge) ? (i)+(r_edge) : (i))
#define ADDR_R(i, r_edge, addr)    ((i) >= (r_edge) ? (i)-(r_edge) : (addr))
#define ADDR_H(i, t_edge, b_edge)  ((i) <  (t_edge) ? (i)+(b_edge) : (i))
#define ADDR_B(i, b_edge, addr)    ((i) >= (b_edge) ? (i)-(b_edge) : (addr))
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////Macro for define elements number per thread/////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
#define ANCHOR                  3
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
__kernel void filter2D_C1_D0(__global uchar *src, int src_step, int src_offset_x, int src_offset_y, 
                             __global uchar *dst, int dst_step, int dst_offset_x, int dst_offset_y, 
                             __constant int *mat_kernel __attribute__((max_constant_size (16384))),
                             int cols,int rows, int operate_cols, int wholecols, int wholerows) 
{
    int gX = get_global_id(0);
    int gY = get_global_id(1);

    int lX = get_local_id(0);

    int groupX_size = get_local_size(0);
    int groupX_id   = get_group_id(0);

    #define dst_align (dst_offset_x & 3)     
    int cols_start_index_group = src_offset_x - dst_align + groupX_size * groupX_id - ANX; 
    int rows_start_index       = src_offset_y + (gY << ROWS_PER_GROUP_BITS) - ANY; 
        
    __local uchar local_data[LOCAL_MEM_STEP * ROWS_FETCH];
    if((gY << 2) < rows)
    {
        for(int i = 0; i < ROWS_FETCH; ++i)
        {
            if((rows_start_index - src_offset_y) + i < rows + ANY)  
            {
                #ifdef BORDER_CONSTANT
                int selected_row  = rows_start_index + i;
                int selected_cols = cols_start_index_group + lX;

                uchar data = *(src + selected_row * src_step + selected_cols);
                int con = selected_row >=0 && selected_row < wholerows && selected_cols >=0 && selected_cols < wholecols;
                data = con ? data : 0;
                local_data[i * LOCAL_MEM_STEP + lX ] =data; 

                if(lX < (ANX << 1))
                {
                    selected_cols = cols_start_index_group + lX + groupX_size;

                    data = *(src + selected_row * src_step + selected_cols);
                    con = selected_row >=0 && selected_row < wholerows && selected_cols >=0 && selected_cols < wholecols;
                    data = con ? data : 0;
                    local_data[i * LOCAL_MEM_STEP + lX + groupX_size] =data; 
                }
                #else
                int selected_row = ADDR_H(rows_start_index + i,  0, wholerows);
                selected_row     = ADDR_B(rows_start_index + i, wholerows, selected_row);

                int selected_cols = ADDR_L(cols_start_index_group + lX, 0, wholecols);
                selected_cols     = ADDR_R(cols_start_index_group + lX, wholecols, selected_cols);

                uchar data = *(src + selected_row * src_step + selected_cols);

                local_data[i * LOCAL_MEM_STEP + lX ] =data; 

                if(lX < (ANX << 1))
                {
                    selected_cols = cols_start_index_group + lX + groupX_size;
                    selected_cols = ADDR_R(selected_cols, wholecols, selected_cols);

                    data = *(src + selected_row * src_step + selected_cols);
                    local_data[i * LOCAL_MEM_STEP + lX + groupX_size] =data; 
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

        uchar4 dst_data = *((__global uchar4 *)(dst + dst_rows_index * dst_step + dst_cols_index));

        int4 sum = (int4)(0);
        uchar4 data;

        for(int i = 0; i < ANCHOR; i++)
        {
           #pragma unroll 3
           for(int j = 0; j < ANCHOR; j++)
           {
                if(dst_rows_index < dst_rows_end)
                {
                     int local_row = (lX >> THREADS_PER_ROW_BIT) + i;
                     int local_cols = ((lX % THREADS_PER_ROW) << ELEMENTS_PER_THREAD_BIT) + j; 

                     data = vload4(0, local_data+local_row * LOCAL_MEM_STEP + local_cols); 
                     sum = sum + (mat_kernel[i * ANCHOR + j] * convert_int4_sat(data));
                 }
            }
        }

        if(dst_rows_index < dst_rows_end)
        {
            sum.x = ((dst_cols_index + 0 >= dst_cols_start) && (dst_cols_index + 0 < dst_cols_end)) ? sum.x : dst_data.x;
            sum.y = ((dst_cols_index + 1 >= dst_cols_start) && (dst_cols_index + 1 < dst_cols_end)) ? sum.y : dst_data.y;
            sum.z = ((dst_cols_index + 2 >= dst_cols_start) && (dst_cols_index + 2 < dst_cols_end)) ? sum.z : dst_data.z;
            sum.w = ((dst_cols_index + 3 >= dst_cols_start) && (dst_cols_index + 3 < dst_cols_end)) ? sum.w : dst_data.w;
            *((__global uchar4 *)(dst + dst_rows_index * dst_step + dst_cols_index)) = convert_uchar4_sat(sum); 
        }
   }
}
///////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////32FC1////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void filter2D_C1_D5(__global float *src, int src_step, int src_offset_x, int src_offset_y, 
                             __global float *dst, int dst_step, int dst_offset_x, int dst_offset_y, 
                             __constant int *mat_kernel __attribute__((max_constant_size (16384))),
                             int cols,int rows, int operate_cols, int wholecols, int wholerows) 
{
    int gX = get_global_id(0);
    int gY = get_global_id(1);

    int lX = get_local_id(0);

    int groupX_size = get_local_size(0);
    int groupX_id   = get_group_id(0);

    #define dst_align (dst_offset_x & 3)     
    int cols_start_index_group = src_offset_x - dst_align + groupX_size * groupX_id - ANX; 
    int rows_start_index       = src_offset_y + (gY << ROWS_PER_GROUP_BITS) - ANY; 
        
    __local float local_data[LOCAL_MEM_STEP * ROWS_FETCH];
    if(((gY << 2) < rows))
    {
        for(int i = 0; i < ROWS_FETCH; ++i)
        {
            if((rows_start_index - src_offset_y) + i < rows + ANY)  
            {
                #ifdef BORDER_CONSTANT
                int selected_row  = rows_start_index + i;
                int selected_cols = cols_start_index_group + lX;

                float data = *((__global float *)((__global char *)src + selected_row * src_step + (selected_cols << 2)));
                int con = selected_row >=0 && selected_row < wholerows && selected_cols >=0 && selected_cols < wholecols;
                data = con ? data : 0;
                local_data[i * LOCAL_MEM_STEP + lX ] =data; 

                if(lX < (ANX << 1))
                {
                    selected_cols = cols_start_index_group + lX + groupX_size;

                    data = *((__global float *)((__global char *)src + selected_row * src_step + (selected_cols << 2)));
                    con = selected_row >=0 && selected_row < wholerows && selected_cols >=0 && selected_cols < wholecols;
                    data = con ? data : 0;
                    local_data[i * LOCAL_MEM_STEP + lX + groupX_size] =data; 
                }
                #else
                int selected_row = ADDR_H(rows_start_index + i,  0, wholerows);
                selected_row     = ADDR_B(rows_start_index + i, wholerows, selected_row);

                int selected_cols = ADDR_L(cols_start_index_group + lX, 0, wholecols);
                selected_cols     = ADDR_R(cols_start_index_group + lX, wholecols, selected_cols);

                float data = *((__global float *)((__global char *)src + selected_row * src_step + (selected_cols << 2)));
                local_data[i * LOCAL_MEM_STEP + lX] =data; 

                if(lX < (ANX << 1))
                {
                    selected_cols = cols_start_index_group + lX + groupX_size;
                    selected_cols = ADDR_R(selected_cols, wholecols, selected_cols);

                    data = *((__global float *)((__global char *)src + selected_row * src_step + (selected_cols << 2)));
                    local_data[i * LOCAL_MEM_STEP + lX + groupX_size] =data; 
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

        float4 dst_data = *((__global float4*)((__global char *)dst + dst_rows_index * dst_step + (dst_cols_index << 2)));

        float4 sum = (float4)(0);
        float4 data;

        for(int i = 0; i < ANCHOR; i++)
        {
           #pragma unroll 3
           for(int j = 0; j < ANCHOR; j++)
           {
                if(dst_rows_index < dst_rows_end)
                {
                     int local_row = (lX >> THREADS_PER_ROW_BIT) + i;
                     int local_cols = ((lX % THREADS_PER_ROW) << ELEMENTS_PER_THREAD_BIT) + j; 

                     data = vload4(0, local_data+local_row * LOCAL_MEM_STEP + local_cols); 
                     sum = sum + (mat_kernel[i * ANCHOR + j] * data);
                 }
            }
        }

        if(dst_rows_index < dst_rows_end)
        {
            sum.x = ((dst_cols_index + 0 >= dst_cols_start) && (dst_cols_index + 0 < dst_cols_end)) ? sum.x : dst_data.x;
            sum.y = ((dst_cols_index + 1 >= dst_cols_start) && (dst_cols_index + 1 < dst_cols_end)) ? sum.y : dst_data.y;
            sum.z = ((dst_cols_index + 2 >= dst_cols_start) && (dst_cols_index + 2 < dst_cols_end)) ? sum.z : dst_data.z;
            sum.w = ((dst_cols_index + 3 >= dst_cols_start) && (dst_cols_index + 3 < dst_cols_end)) ? sum.w : dst_data.w;

            *((__global float4 *)((__global char *)dst + dst_rows_index * dst_step + (dst_cols_index << 2))) = sum; 
        }
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////8uC4////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void filter2D_C4_D0(__global uchar4 *src, int src_step, int src_offset_x, int src_offset_y, 
                             __global uchar4 *dst, int dst_step, int dst_offset_x, int dst_offset_y, 
                             __constant int *mat_kernel __attribute__((max_constant_size (16384))),
                             int cols,int rows, int operate_cols, int wholecols, int wholerows) 
{
    int gX = get_global_id(0);
    int gY = get_global_id(1);

    int lX = get_local_id(0);

    int groupX_size = get_local_size(0);
    int groupX_id   = get_group_id(0);

    #define dst_align (dst_offset_x & 3)     
    int cols_start_index_group = src_offset_x - dst_align + groupX_size * groupX_id - ANX; 
    int rows_start_index       = src_offset_y + (gY << ROWS_PER_GROUP_BITS) - ANY; 
        
    __local uchar4 local_data[LOCAL_MEM_STEP * ROWS_FETCH];
        
    if(((gY << 2) < rows))
    {
        for(int i = 0; i < ROWS_FETCH; ++i)
        {
            if((rows_start_index - src_offset_y) + i < rows + ANY)  
            {
                #ifdef BORDER_CONSTANT
                int selected_row  = rows_start_index + i;
                int selected_cols = cols_start_index_group + lX;

                uchar4 data = *((__global uchar4*)((__global char*)src + selected_row * src_step + (selected_cols << 2)));
                int con = selected_row >=0 && selected_row < wholerows && selected_cols >=0 && selected_cols < wholecols;
                data = con ? data : 0;
                local_data[i * LOCAL_MEM_STEP + lX ] =data; 

                if(lX < (ANX << 1))
                {
                    selected_cols = cols_start_index_group + lX + groupX_size;

                    data = *((__global uchar4*)((__global char*)src + selected_row * src_step + (selected_cols << 2)));
                    con = selected_row >=0 && selected_row < wholerows && selected_cols >=0 && selected_cols < wholecols;
                    data = con ? data : 0;
                    local_data[i * LOCAL_MEM_STEP + lX + groupX_size] =data; 
                }
                #else
                int selected_row = ADDR_H(rows_start_index + i,  0, wholerows);
                selected_row     = ADDR_B(rows_start_index + i, wholerows, selected_row);

                int selected_cols = ADDR_L(cols_start_index_group + lX, 0, wholecols);
                selected_cols     = ADDR_R(cols_start_index_group + lX, wholecols, selected_cols);

                uchar4 data = *((__global uchar4*)((__global char*)src + selected_row * src_step + (selected_cols << 2)));

                local_data[i * LOCAL_MEM_STEP + lX] =data; 

                if(lX < (ANX << 1))
                {
                    selected_cols = cols_start_index_group + lX + groupX_size;
                    selected_cols = ADDR_R(selected_cols, wholecols, selected_cols);

                    data = *((__global uchar4*)((__global char*)src + selected_row * src_step + (selected_cols << 2)));
                    local_data[i * LOCAL_MEM_STEP + lX + groupX_size] =data; 
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

        uchar16 dst_data;
        dst_data = *((__global uchar16*)((__global char *)dst + dst_rows_index * dst_step + (dst_cols_index << 2)));

        int16 sum = (int16)(0);
        uchar16 data;

        for(int i = 0; i < ANCHOR; i++)
        {
           #pragma unroll 3
           for(int j = 0; j < ANCHOR; j++)
           {
                if(dst_rows_index < dst_rows_end)
                {
                     int local_row = (lX >> THREADS_PER_ROW_BIT) + i;
                     int local_cols = ((lX % THREADS_PER_ROW) << ELEMENTS_PER_THREAD_BIT) + j; 

                     data = vload16(0, (__local uchar *)(local_data+local_row * LOCAL_MEM_STEP + local_cols)); 
                     sum = sum + (mat_kernel[i * ANCHOR + j] * convert_int16_sat(data));
                 }
            }
        }

        if(dst_rows_index < dst_rows_end)
        {
            uchar16 sum1 = convert_uchar16_sat(sum);
            sum1.s0123 = ((dst_cols_index + 0 >= dst_cols_start) && (dst_cols_index + 0 < dst_cols_end))?  
                         sum1.s0123 : dst_data.s0123;
            sum1.s4567 = ((dst_cols_index + 1 >= dst_cols_start) && (dst_cols_index + 1 < dst_cols_end))? 
                         sum1.s4567 : dst_data.s4567;
            sum1.s89ab = ((dst_cols_index + 2 >= dst_cols_start) && (dst_cols_index + 2 < dst_cols_end))? 
                         sum1.s89ab : dst_data.s89ab;
            sum1.scdef = ((dst_cols_index + 3 >= dst_cols_start) && (dst_cols_index + 3 < dst_cols_end))? 
                         sum1.scdef : dst_data.scdef;

            *((__global uchar16*)((__global char *)dst + dst_rows_index * dst_step + (dst_cols_index << 2))) = sum1; 
        }
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////32FC4////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
#define ROWS_FETCH_C4              (1 + ANY + ANY)   //(ROWS_PER_GROUP + anY * 2)
#define LOCAL_MEM_STEP_C4           260 //divup((get_local_size(0) + anX * 2), 4) * 4)
__kernel void filter2D_C4_D5(__global float4 *src, int src_step, int src_offset_x, int src_offset_y, 
                             __global float4 *dst, int dst_step, int dst_offset_x, int dst_offset_y, 
                             __constant int *mat_kernel __attribute__((max_constant_size (16384))),
                             int cols,int rows, int operate_cols, int wholecols, int wholerows) 
{
    int gX = get_global_id(0);
    int gY = get_global_id(1);

    int lX = get_local_id(0);

    int groupX_size = get_local_size(0);
    int groupX_id   = get_group_id(0);

    int cols_start_index_group = src_offset_x + groupX_size * groupX_id - ANX; 
    int rows_start_index       = src_offset_y + gY - ANY; 
        
    __local float4 local_data[LOCAL_MEM_STEP_C4 * ROWS_FETCH_C4];
    if((gY < rows) && (gX < (operate_cols + ANX + ANX)))
    {
        for(int i = 0; i < ROWS_FETCH_C4; ++i)
        {
            if((rows_start_index - src_offset_y) + i < rows + ANY)  
            {
                #ifdef BORDER_CONSTANT
                int selected_row  = rows_start_index + i;
                int selected_cols = cols_start_index_group + lX;

                float4 data = *((__global float4*)((__global char*)src + selected_row * src_step + (selected_cols << 4)));
                int con = selected_row >=0 && selected_row < wholerows && selected_cols >=0 && selected_cols < wholecols;
                data = con ? data : 0;
                local_data[i * LOCAL_MEM_STEP + lX ] =data; 

                if(lX < (ANX << 1))
                {
                    selected_cols = cols_start_index_group + lX + groupX_size;

                    data = *((__global float4*)((__global char*)src + selected_row * src_step + (selected_cols << 4)));
                    con = selected_row >=0 && selected_row < wholerows && selected_cols >=0 && selected_cols < wholecols;
                    data = con ? data : 0;
                    local_data[i * LOCAL_MEM_STEP + lX + groupX_size] =data; 
                }
                #else
                int selected_row = ADDR_H(rows_start_index + i,  0, wholerows);
                selected_row     = ADDR_B(rows_start_index + i, wholerows, selected_row);

                int selected_cols = ADDR_L(cols_start_index_group + lX, 0, wholecols);
                selected_cols     = ADDR_R(cols_start_index_group + lX, wholecols, selected_cols);

                float4 data = *((__global float4*)((__global char*)src + selected_row * src_step + (selected_cols << 4)));
                local_data[i * LOCAL_MEM_STEP_C4 + lX] =data; 

                if(lX < (ANX << 1))
                {
                    selected_cols = cols_start_index_group + lX + groupX_size;
                    selected_cols = ADDR_R(selected_cols, wholecols, selected_cols);

                    data = *((__global float4*)((__global char*)src + selected_row * src_step + (selected_cols << 4)));
                    local_data[i * LOCAL_MEM_STEP_C4 + lX + groupX_size] =data; 
                }
                #endif
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if((gY < rows) && (gX < operate_cols))
    {
        int dst_cols_index = dst_offset_x + gX;  
        int dst_rows_index = dst_offset_y + gY;

        float4 sum = (float4)(0);

        for(int i = 0; i < ANCHOR; i++)
        {
           for(int j = 0; j < ANCHOR; j++)
           {
               int local_cols = lX + j; 
               sum = sum + mat_kernel[i * ANCHOR + j] * local_data[i * LOCAL_MEM_STEP_C4 + local_cols];
            }
        }

        *((__global float4*)((__global char *)dst + dst_rows_index * dst_step + (dst_cols_index << 4))) = sum; 
    }
}
