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
//    Sen Liu, swjtuls1987@126.com
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

#define ROWSperTHREAD 21     // the number of rows a thread will process
#define BLOCK_W       128    // the thread block width (464)
#define N_DISPARITIES 8

#define STEREO_MIND 0                    // The minimum d range to check
#define STEREO_DISP_STEP N_DISPARITIES   // the d step, must be <= 1 to avoid aliasing

#ifndef radius
#define radius 64
#endif

inline unsigned int CalcSSD(__local unsigned int *col_ssd)
{
    unsigned int cache = col_ssd[0];

#pragma unroll
    for(int i = 1; i <= (radius << 1); i++)
        cache += col_ssd[i];

    return cache;
}

inline uint2 MinSSD(__local unsigned int *col_ssd)
{
    unsigned int ssd[N_DISPARITIES];
    const int win_size = (radius << 1);

    //See above:  #define COL_SSD_SIZE (BLOCK_W + WIN_SIZE)
    ssd[0] = CalcSSD(col_ssd + 0 * (BLOCK_W + win_size));
    ssd[1] = CalcSSD(col_ssd + 1 * (BLOCK_W + win_size));
    ssd[2] = CalcSSD(col_ssd + 2 * (BLOCK_W + win_size));
    ssd[3] = CalcSSD(col_ssd + 3 * (BLOCK_W + win_size));
    ssd[4] = CalcSSD(col_ssd + 4 * (BLOCK_W + win_size));
    ssd[5] = CalcSSD(col_ssd + 5 * (BLOCK_W + win_size));
    ssd[6] = CalcSSD(col_ssd + 6 * (BLOCK_W + win_size));
    ssd[7] = CalcSSD(col_ssd + 7 * (BLOCK_W + win_size));

    unsigned int mssd = min(min(min(ssd[0], ssd[1]), min(ssd[4], ssd[5])), min(min(ssd[2], ssd[3]), min(ssd[6], ssd[7])));

    int bestIdx = 0;

    for (int i = 0; i < N_DISPARITIES; i++)
    {
        if (mssd == ssd[i])
            bestIdx = i;
    }

    return (uint2)(mssd, bestIdx);
}

inline void StepDown(int idx1, int idx2, __global unsigned char* imageL,
              __global unsigned char* imageR, int d,   __local unsigned int *col_ssd)
{
    uint8 imgR1 = convert_uint8(vload8(0, imageR + (idx1 - d - 7)));
    uint8 imgR2 = convert_uint8(vload8(0, imageR + (idx2 - d - 7)));
    uint8 diff1 = (uint8)(imageL[idx1]) - imgR1;
    uint8 diff2 = (uint8)(imageL[idx2]) - imgR2;
    uint8 res = diff2 * diff2 - diff1 * diff1;
    const int win_size = (radius << 1);
    col_ssd[0 * (BLOCK_W + win_size)] += res.s7;
    col_ssd[1 * (BLOCK_W + win_size)] += res.s6;
    col_ssd[2 * (BLOCK_W + win_size)] += res.s5;
    col_ssd[3 * (BLOCK_W + win_size)] += res.s4;
    col_ssd[4 * (BLOCK_W + win_size)] += res.s3;
    col_ssd[5 * (BLOCK_W + win_size)] += res.s2;
    col_ssd[6 * (BLOCK_W + win_size)] += res.s1;
    col_ssd[7 * (BLOCK_W + win_size)] += res.s0;
}

inline void InitColSSD(int x_tex, int y_tex, int im_pitch, __global unsigned char* imageL,
                __global unsigned char* imageR, int d,
                 __local unsigned int *col_ssd)
{
    uint8 leftPixel1;
    uint8 diffa = 0;
    int idx = y_tex * im_pitch + x_tex;
    const int win_size = (radius << 1);
    for(int i = 0; i < (win_size + 1); i++)
    {
        leftPixel1 = (uint8)(imageL[idx]);
        uint8 imgR = convert_uint8(vload8(0, imageR + (idx - d - 7)));
        uint8 res = leftPixel1 - imgR;
        diffa += res * res;

        idx += im_pitch;
    }
    //See above:  #define COL_SSD_SIZE (BLOCK_W + WIN_SIZE)
    col_ssd[0 * (BLOCK_W + win_size)] = diffa.s7;
    col_ssd[1 * (BLOCK_W + win_size)] = diffa.s6;
    col_ssd[2 * (BLOCK_W + win_size)] = diffa.s5;
    col_ssd[3 * (BLOCK_W + win_size)] = diffa.s4;
    col_ssd[4 * (BLOCK_W + win_size)] = diffa.s3;
    col_ssd[5 * (BLOCK_W + win_size)] = diffa.s2;
    col_ssd[6 * (BLOCK_W + win_size)] = diffa.s1;
    col_ssd[7 * (BLOCK_W + win_size)] = diffa.s0;
}

__kernel void stereoKernel(__global unsigned char *left, __global unsigned char *right,
                           __global unsigned int *cminSSDImage, int cminSSD_step,
                           __global unsigned char *disp, int disp_step,int cwidth, int cheight,
                           int img_step, int maxdisp,
                           __local unsigned int *col_ssd_cache)
{
    __local unsigned int *col_ssd = col_ssd_cache + get_local_id(0);
    __local unsigned int *col_ssd_extra = get_local_id(0) < (radius << 1) ? col_ssd + BLOCK_W : 0;

    int X = get_group_id(0) * BLOCK_W + get_local_id(0) + maxdisp + radius;

#define Y (int)(get_group_id(1) * ROWSperTHREAD + radius)

    __global unsigned int* minSSDImage = cminSSDImage + X + Y * cminSSD_step;
    __global unsigned char* disparImage = disp + X + Y * disp_step;

    int end_row = ROWSperTHREAD < (cheight - Y) ? ROWSperTHREAD:(cheight - Y);
    int y_tex;
    int x_tex = X - radius;

    //if (x_tex >= cwidth)
    //    return;

    for(int d = STEREO_MIND; d < maxdisp; d += STEREO_DISP_STEP)
    {
        y_tex = Y - radius;

        InitColSSD(x_tex, y_tex, img_step, left, right, d, col_ssd);
        if (col_ssd_extra > 0)
            if (x_tex + BLOCK_W < cwidth)
                InitColSSD(x_tex + BLOCK_W, y_tex, img_step, left, right, d, col_ssd_extra);

        barrier(CLK_LOCAL_MEM_FENCE); //before MinSSD function

        uint2 minSSD = MinSSD(col_ssd);
        if (X < cwidth - radius && Y < cheight - radius)
        {
            if (minSSD.x < minSSDImage[0])
            {
                disparImage[0] = (unsigned char)(d + minSSD.y);
                minSSDImage[0] = minSSD.x;
            }
        }

        for(int row = 1; row < end_row; row++)
        {
            int idx1 = y_tex * img_step + x_tex;
            int idx2 = min(y_tex + ((radius << 1) + 1), cheight - 1) * img_step + x_tex;

            barrier(CLK_LOCAL_MEM_FENCE);

            StepDown(idx1, idx2, left, right, d, col_ssd);
            if (col_ssd_extra > 0)
                if (x_tex + BLOCK_W < cwidth)
                    StepDown(idx1, idx2, left + BLOCK_W, right + BLOCK_W, d, col_ssd_extra);

            barrier(CLK_LOCAL_MEM_FENCE);

            uint2 minSSD = MinSSD(col_ssd);
            if (X < cwidth - radius && row < cheight - radius - Y)
            {
                int idx = row * cminSSD_step;
                if (minSSD.x < minSSDImage[idx])
                {
                    disparImage[disp_step * row] = (unsigned char)(d + minSSD.y);
                    minSSDImage[idx] = minSSD.x;
                }
            }

            y_tex++;
        } // for row loop
    } // for d loop
}
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////// Sobel Prefiler (signal channel)//////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void prefilter_xsobel(__global unsigned char *input, __global unsigned char *output,
                               int rows, int cols, int prefilterCap)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < cols && y < rows)
    {
        int cov = input[(y-1) * cols + (x-1)] * (-1) + input[(y-1) * cols + (x+1)] * (1) +
                  input[(y)   * cols + (x-1)] * (-2) + input[(y)   * cols + (x+1)] * (2) +
                  input[(y+1) * cols + (x-1)] * (-1) + input[(y+1) * cols + (x+1)] * (1);

        cov = min(min(max(-prefilterCap, cov), prefilterCap) + prefilterCap, 255);
        output[y * cols + x] = cov & 0xFF;
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// Textureness filtering ////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

inline float sobel(__global unsigned char *input, int x, int y, int rows, int cols)
{
    float conv = 0;
    int y1 = y==0? 0 : y-1;
    int x1 = x==0? 0 : x-1;
    if(x < cols && y < rows && x > 0 && y > 0)
    {
        conv = (float)input[(y1)  * cols + (x1)] * (-1) + (float)input[(y1)  * cols + (x+1)] * (1) +
               (float)input[(y)   * cols + (x1)] * (-2) + (float)input[(y)   * cols + (x+1)] * (2) +
               (float)input[(y+1) * cols + (x1)] * (-1) + (float)input[(y+1) * cols + (x+1)] * (1);

    }
    return fabs(conv);
}

inline float CalcSums(__local float *cols, __local float *cols_cache, int winsz)
{
    unsigned int cache = cols[0];

    for(int i = 1; i <= winsz; i++)
        cache += cols[i];

    return cache;
}

#define RpT (2 * ROWSperTHREAD)  // got experimentally
__kernel void textureness_kernel(__global unsigned char *disp, int disp_rows, int disp_cols,
                                 int disp_step, __global unsigned char *input, int input_rows,
                                 int input_cols,int winsz, float threshold,
                                 __local float *cols_cache)
{
    int winsz2 = winsz/2;
    int n_dirty_pixels = (winsz2) * 2;

    int local_id_x = get_local_id(0);
    int group_size_x = get_local_size(0);
    int group_id_y = get_group_id(1);

    __local float *cols = cols_cache + group_size_x + local_id_x;
    __local float *cols_extra = local_id_x < n_dirty_pixels ? cols + group_size_x : 0;

    int x = get_global_id(0);
    int beg_row = group_id_y * RpT;
    int end_row = min(beg_row + RpT, disp_rows);


    int y = beg_row;

    float sum = 0;
    float sum_extra = 0;

    for(int i = y - winsz2; i <= y + winsz2; ++i)
    {
        sum += sobel(input, x - winsz2, i, input_rows, input_cols);
        if (cols_extra)
            sum_extra += sobel(input, x + group_size_x - winsz2, i, input_rows, input_cols);
    }
    *cols = sum;
    if (cols_extra)
        *cols_extra = sum_extra;

    barrier(CLK_LOCAL_MEM_FENCE);

    float sum_win = CalcSums(cols, cols_cache + local_id_x, winsz) * 255;
    if (sum_win < threshold)
        disp[y * disp_step + x] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for(int y = beg_row + 1; y < end_row; ++y)
    {
        sum = sum - sobel(input, x - winsz2, y - winsz2 - 1, input_rows, input_cols) +
              sobel(input, x - winsz2, y + winsz2, input_rows, input_cols);
        *cols = sum;

        if (cols_extra)
        {
            sum_extra = sum_extra - sobel(input, x + group_size_x - winsz2, y - winsz2 - 1,input_rows, input_cols)
                        + sobel(input, x + group_size_x - winsz2, y + winsz2, input_rows, input_cols);
            *cols_extra = sum_extra;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (x < disp_cols)
        {
            float sum_win = CalcSums(cols, cols_cache + local_id_x, winsz) * 255;
            if (sum_win < threshold)
                disp[y * disp_step + x] = 0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

}
