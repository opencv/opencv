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

///////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////Macro for border type////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef BORDER_CONSTANT
#elif defined BORDER_REPLICATE
#define EXTRAPOLATE(x, maxV) \
    { \
        x = max(min(x, maxV - 1), 0); \
    }
#elif defined BORDER_WRAP
#define EXTRAPOLATE(x, maxV) \
    { \
        if (x < 0) \
            x -= ((x - maxV + 1) / maxV) * maxV; \
        if (x >= maxV) \
            x %= maxV; \
    }
#elif defined(BORDER_REFLECT) || defined(BORDER_REFLECT101)
#define EXTRAPOLATE_(x, maxV, delta) \
    { \
        if (maxV == 1) \
            x = 0; \
        else \
            do \
            { \
                if ( x < 0 ) \
                    x = -x - 1 + delta; \
                else \
                    x = maxV - 1 - (x - maxV) - delta; \
            } \
            while (x >= maxV || x < 0); \
    }
#ifdef BORDER_REFLECT
#define EXTRAPOLATE(x, maxV) EXTRAPOLATE_(x, maxV, 0)
#else
#define EXTRAPOLATE(x, maxV) EXTRAPOLATE_(x, maxV, 1)
#endif
#else
#error No extrapolation method
#endif

#define THREADS 256

///////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////calcHarris////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void calcMinEigenVal(__global const float *Dx,__global const float *Dy, __global float *dst,
                              int dx_offset, int dx_whole_rows, int dx_whole_cols, int dx_step,
                              int dy_offset, int dy_whole_rows, int dy_whole_cols, int dy_step,
                              int dst_offset, int dst_rows, int dst_cols, int dst_step, float k)
{
    int col = get_local_id(0);
    int gX = get_group_id(0);
    int gY = get_group_id(1);
    int gly = get_global_id(1);

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

    float dx_data[ksY+1], dy_data[ksY+1], data[3][ksY+1];
    __local float temp[6][THREADS];

#ifdef BORDER_CONSTANT
    for (int i=0; i < ksY+1; i++)
    {
        bool dx_con = dx_startX+col >= 0 && dx_startX+col < dx_whole_cols && dx_startY+i >= 0 && dx_startY+i < dx_whole_rows;
        int indexDx = (dx_startY+i)*(dx_step>>2)+(dx_startX+col);
        float dx_s = dx_con ? Dx[indexDx] : 0.0f;
        dx_data[i] = dx_s;

        bool dy_con = dy_startX+col >= 0 && dy_startX+col < dy_whole_cols && dy_startY+i >= 0 && dy_startY+i < dy_whole_rows;
        int indexDy = (dy_startY+i)*(dy_step>>2)+(dy_startX+col);
        float dy_s = dy_con ? Dy[indexDy] : 0.0f;
        dy_data[i] = dy_s;

        data[0][i] = dx_data[i] * dx_data[i];
        data[1][i] = dx_data[i] * dy_data[i];
        data[2][i] = dy_data[i] * dy_data[i];
    }
#else
    int clamped_col = min(dst_cols, col);
    for (int i=0; i < ksY+1; i++)
    {
        int dx_selected_row = dx_startY+i, dx_selected_col = dx_startX+clamped_col;
        EXTRAPOLATE(dx_selected_row, dx_whole_rows)
        EXTRAPOLATE(dx_selected_col, dx_whole_cols)
        dx_data[i] = Dx[dx_selected_row * (dx_step>>2) + dx_selected_col];

        int dy_selected_row = dy_startY+i, dy_selected_col = dy_startX+clamped_col;
        EXTRAPOLATE(dy_selected_row, dy_whole_rows)
        EXTRAPOLATE(dy_selected_col, dy_whole_cols)
        dy_data[i] = Dy[dy_selected_row * (dy_step>>2) + dy_selected_col];

        data[0][i] = dx_data[i] * dx_data[i];
        data[1][i] = dx_data[i] * dy_data[i];
        data[2][i] = dy_data[i] * dy_data[i];
    }
#endif
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f;
    for (int i=1; i < ksY; i++)
    {
        sum0 += (data[0][i]);
        sum1 += (data[1][i]);
        sum2 += (data[2][i]);
    }

    float sum01 = sum0 + (data[0][0]);
    float sum02 = sum0 + (data[0][ksY]);
    temp[0][col] = sum01;
    temp[1][col] = sum02;
    float sum11 = sum1 + (data[1][0]);
    float sum12 = sum1 + (data[1][ksY]);
    temp[2][col] = sum11;
    temp[3][col] = sum12;
    float sum21 = sum2 + (data[2][0]);
    float sum22 = sum2 + (data[2][ksY]);
    temp[4][col] = sum21;
    temp[5][col] = sum22;
    barrier(CLK_LOCAL_MEM_FENCE);

    if(col < (THREADS-(ksX-1)))
    {
        col += anX;
        int posX = dst_startX - dst_x_off + col - anX;
        int posY = (gly << 1);
        int till = (ksX + 1)%2;
        float tmp_sum[6] = { 0.0f, 0.0f , 0.0f, 0.0f, 0.0f, 0.0f };
        for (int k=0; k<6; k++)
            for (int i=-anX; i<=anX - till; i++)
                tmp_sum[k] += temp[k][col+i];

        if(posX < dst_cols && (posY) < dst_rows)
        {
            float a = tmp_sum[0] * 0.5f;
            float b = tmp_sum[2];
            float c = tmp_sum[4] * 0.5f;
            dst[(dst_startY+0) * (dst_step>>2)+ dst_startX + col - anX] = (float)((a+c) - sqrt((a-c)*(a-c) + b*b));
        }
        if (posX < dst_cols && (posY + 1) < dst_rows)
        {
            float a = tmp_sum[1] * 0.5f;
            float b = tmp_sum[3];
            float c = tmp_sum[5] * 0.5f;
            dst[(dst_startY+1) * (dst_step>>2)+ dst_startX + col - anX] = (float)((a+c) - sqrt((a-c)*(a-c) + b*b));
        }
    }
}
