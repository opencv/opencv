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
//    Shengen Yan,yanshengen@gmail.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other GpuMaterials provided with the distribution.
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

#if defined (DOUBLE_SUPPORT)
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#elif defined (cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#endif
#endif
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

#ifdef BORDER_REFLECT101
//BORDER_REFLECT101:   gfedcb|abcdefgh|gfedcba
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

#define THREADS 256
#define ELEM(i, l_edge, r_edge, elem1, elem2) (i) >= (l_edge) && (i) < (r_edge) ? (elem1) : (elem2)
///////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////calcHarris////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void calcHarris(__global const float *Dx,__global const float *Dy, __global float *dst,
                              int dx_offset, int dx_whole_rows, int dx_whole_cols, int dx_step,
                              int dy_offset, int dy_whole_rows, int dy_whole_cols, int dy_step,
                              int dst_offset, int dst_rows, int dst_cols, int dst_step,
                              float k)
{
    int col = get_local_id(0);
    const int gX = get_group_id(0);
    const int gY = get_group_id(1);
    const int glx = get_global_id(0);
    const int gly = get_global_id(1);

    int dx_x_off = (dx_offset % dx_step) >> 2;
    int dx_y_off = dx_offset / dx_step;
    int dy_x_off = (dy_offset % dy_step) >> 2;
    int dy_y_off = dy_offset / dy_step;
    int dst_x_off = (dst_offset % dst_step) >> 2;
    int dst_y_off = dst_offset / dst_step;

    int dx_startX = gX * (THREADS-ksX+1) - anX + dx_x_off;
    int dx_startY = (gY << 1) - anY + dx_y_off;
    int dy_startX = gX * (THREADS-ksX+1) - anX + dy_x_off;
    int dy_startY = (gY << 1) - anY + dy_y_off;
    int dst_startX = gX * (THREADS-ksX+1) + dst_x_off;
    int dst_startY = (gY << 1) + dst_y_off;

    float dx_data[ksY+1],dy_data[ksY+1],data[3][ksY+1];
    __local float temp[6][THREADS];
#ifdef BORDER_CONSTANT
    bool dx_con,dy_con;
    float dx_s,dy_s;
    for(int i=0; i < ksY+1; i++)
    {
        dx_con = dx_startX+col >= 0 && dx_startX+col < dx_whole_cols && dx_startY+i >= 0 && dx_startY+i < dx_whole_rows;
        dx_s = Dx[(dx_startY+i)*(dx_step>>2)+(dx_startX+col)];
        dx_data[i] = dx_con ? dx_s : 0.0;
        dy_con = dy_startX+col >= 0 && dy_startX+col < dy_whole_cols && dy_startY+i >= 0 && dy_startY+i < dy_whole_rows;
        dy_s = Dy[(dy_startY+i)*(dy_step>>2)+(dy_startX+col)];
        dy_data[i] = dy_con ? dy_s : 0.0;
        data[0][i] = dx_data[i] * dx_data[i];
        data[1][i] = dx_data[i] * dy_data[i];
        data[2][i] = dy_data[i] * dy_data[i];
    }
#else
   for(int i=0; i < ksY+1; i++)
   {
        int dx_selected_row;
        int dx_selected_col;
        dx_selected_row = ADDR_H(dx_startY+i, 0, dx_whole_rows);
        dx_selected_row = ADDR_B(dx_startY+i, dx_whole_rows, dx_selected_row);
        dx_selected_col = ADDR_L(dx_startX+col, 0, dx_whole_cols);
        dx_selected_col = ADDR_R(dx_startX+col, dx_whole_cols, dx_selected_col);
        dx_data[i] = Dx[dx_selected_row * (dx_step>>2) + dx_selected_col];

        int dy_selected_row;
        int dy_selected_col;
        dy_selected_row = ADDR_H(dy_startY+i, 0, dy_whole_rows);
        dy_selected_row = ADDR_B(dy_startY+i, dy_whole_rows, dy_selected_row);
        dy_selected_col = ADDR_L(dy_startX+col, 0, dy_whole_cols);
        dy_selected_col = ADDR_R(dy_startX+col, dy_whole_cols, dy_selected_col);
        dy_data[i] = Dy[dy_selected_row * (dy_step>>2) + dy_selected_col];

        data[0][i] = dx_data[i] * dx_data[i];
        data[1][i] = dx_data[i] * dy_data[i];
        data[2][i] = dy_data[i] * dy_data[i];
   }
#endif
    float sum0 = 0.0, sum1 = 0.0, sum2 = 0.0;
    for(int i=1; i < ksY; i++)
    {
        sum0 += (data[0][i]);
        sum1 += (data[1][i]);
        sum2 += (data[2][i]);
    }
    float sum01,sum02,sum11,sum12,sum21,sum22;
    sum01 = sum0 + (data[0][0]);
    sum02 = sum0 + (data[0][ksY]);
    temp[0][col] = sum01;
    temp[1][col] = sum02;
    sum11 = sum1 + (data[1][0]);
    sum12 = sum1 + (data[1][ksY]);
    temp[2][col] = sum11;
    temp[3][col] = sum12;
    sum21 = sum2 + (data[2][0]);
    sum22 = sum2 + (data[2][ksY]);
    temp[4][col] = sum21;
    temp[5][col] = sum22;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(col < (THREADS-(ksX-1)))
    {
        col += anX;
        int posX = dst_startX - dst_x_off + col - anX;
        int posY = (gly << 1);
        int till = (ksX + 1)%2;
        float tmp_sum[6]={ 0.0, 0.0 , 0.0, 0.0, 0.0, 0.0 };
        for(int k=0; k<6; k++)
            for(int i=-anX; i<=anX - till; i++)
            {
                tmp_sum[k] += temp[k][col+i];
            }

        if(posX < dst_cols && (posY) < dst_rows)
        {
            dst[(dst_startY+0) * (dst_step>>2)+ dst_startX + col - anX] =
                    tmp_sum[0] * tmp_sum[4] - tmp_sum[2] * tmp_sum[2] - k * (tmp_sum[0] + tmp_sum[4]) * (tmp_sum[0] + tmp_sum[4]);
        }
        if(posX < dst_cols && (posY + 1) < dst_rows)
        {
            dst[(dst_startY+1) * (dst_step>>2)+ dst_startX + col - anX] =
                    tmp_sum[1] * tmp_sum[5] - tmp_sum[3] * tmp_sum[3] - k * (tmp_sum[1] + tmp_sum[5]) * (tmp_sum[1] + tmp_sum[5]);
        }
    }
}
