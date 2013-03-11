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

#define ROWSperTHREAD 21     // the number of rows a thread will process
#define BLOCK_W       128    // the thread block width (464)
#define N_DISPARITIES 8

#define STEREO_MIND 0                    // The minimum d range to check
#define STEREO_DISP_STEP N_DISPARITIES   // the d step, must be <= 1 to avoid aliasing

int SQ(int a)
{
    return a * a;
}

unsigned int CalcSSD(volatile __local unsigned int *col_ssd_cache, 
                     volatile __local unsigned int *col_ssd, int radius)
{	
    unsigned int cache = 0;
    unsigned int cache2 = 0;

    for(int i = 1; i <= radius; i++)
        cache += col_ssd[i];

    col_ssd_cache[0] = cache;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_local_id(0) < BLOCK_W - radius)
        cache2 = col_ssd_cache[radius];
    else
        for(int i = radius + 1; i < (2 * radius + 1); i++)
            cache2 += col_ssd[i];

    return col_ssd[0] + cache + cache2;
}

uint2 MinSSD(volatile __local unsigned int *col_ssd_cache, 
             volatile __local unsigned int *col_ssd, int radius)
{
    unsigned int ssd[N_DISPARITIES];

    //See above:  #define COL_SSD_SIZE (BLOCK_W + 2 * radius)
    ssd[0] = CalcSSD(col_ssd_cache, col_ssd + 0 * (BLOCK_W + 2 * radius), radius);
    barrier(CLK_LOCAL_MEM_FENCE);
    ssd[1] = CalcSSD(col_ssd_cache, col_ssd + 1 * (BLOCK_W + 2 * radius), radius);
    barrier(CLK_LOCAL_MEM_FENCE);
    ssd[2] = CalcSSD(col_ssd_cache, col_ssd + 2 * (BLOCK_W + 2 * radius), radius);
    barrier(CLK_LOCAL_MEM_FENCE);
    ssd[3] = CalcSSD(col_ssd_cache, col_ssd + 3 * (BLOCK_W + 2 * radius), radius);
    barrier(CLK_LOCAL_MEM_FENCE);
    ssd[4] = CalcSSD(col_ssd_cache, col_ssd + 4 * (BLOCK_W + 2 * radius), radius);
    barrier(CLK_LOCAL_MEM_FENCE);
    ssd[5] = CalcSSD(col_ssd_cache, col_ssd + 5 * (BLOCK_W + 2 * radius), radius);
    barrier(CLK_LOCAL_MEM_FENCE);
    ssd[6] = CalcSSD(col_ssd_cache, col_ssd + 6 * (BLOCK_W + 2 * radius), radius);
    barrier(CLK_LOCAL_MEM_FENCE);
    ssd[7] = CalcSSD(col_ssd_cache, col_ssd + 7 * (BLOCK_W + 2 * radius), radius);
    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int mssd = min(min(min(ssd[0], ssd[1]), min(ssd[4], ssd[5])), min(min(ssd[2], ssd[3]), min(ssd[6], ssd[7])));

    int bestIdx = 0;
    for (int i = 0; i < N_DISPARITIES; i++)
    {
        if (mssd == ssd[i])
            bestIdx = i;
    }

    return (uint2)(mssd, bestIdx);
}

void StepDown(int idx1, int idx2, __global unsigned char* imageL, 
              __global unsigned char* imageR, int d, volatile  __local unsigned int *col_ssd, int radius)
{
    unsigned char leftPixel1;
    unsigned char leftPixel2;
    unsigned char rightPixel1[8];
    unsigned char rightPixel2[8];
    unsigned int diff1, diff2;

    leftPixel1 = imageL[idx1];
    leftPixel2 = imageL[idx2];

    idx1 = idx1 - d;
    idx2 = idx2 - d;

    rightPixel1[7] = imageR[idx1 - 7];
    rightPixel1[0] = imageR[idx1 - 0];
    rightPixel1[1] = imageR[idx1 - 1];
    rightPixel1[2] = imageR[idx1 - 2];
    rightPixel1[3] = imageR[idx1 - 3];
    rightPixel1[4] = imageR[idx1 - 4];
    rightPixel1[5] = imageR[idx1 - 5];
    rightPixel1[6] = imageR[idx1 - 6];

    rightPixel2[7] = imageR[idx2 - 7];
    rightPixel2[0] = imageR[idx2 - 0];
    rightPixel2[1] = imageR[idx2 - 1];
    rightPixel2[2] = imageR[idx2 - 2];
    rightPixel2[3] = imageR[idx2 - 3];
    rightPixel2[4] = imageR[idx2 - 4];
    rightPixel2[5] = imageR[idx2 - 5];
    rightPixel2[6] = imageR[idx2 - 6];

    //See above:  #define COL_SSD_SIZE (BLOCK_W + 2 * radius)
    diff1 = leftPixel1 - rightPixel1[0];
    diff2 = leftPixel2 - rightPixel2[0];
    col_ssd[0 * (BLOCK_W + 2 * radius)] += SQ(diff2) - SQ(diff1);

    diff1 = leftPixel1 - rightPixel1[1];
    diff2 = leftPixel2 - rightPixel2[1];
    col_ssd[1 * (BLOCK_W + 2 * radius)] += SQ(diff2) - SQ(diff1);

    diff1 = leftPixel1 - rightPixel1[2];
    diff2 = leftPixel2 - rightPixel2[2];
    col_ssd[2 * (BLOCK_W + 2 * radius)] += SQ(diff2) - SQ(diff1);

    diff1 = leftPixel1 - rightPixel1[3];
    diff2 = leftPixel2 - rightPixel2[3];
    col_ssd[3 * (BLOCK_W + 2 * radius)] += SQ(diff2) - SQ(diff1);

    diff1 = leftPixel1 - rightPixel1[4];
    diff2 = leftPixel2 - rightPixel2[4];
    col_ssd[4 * (BLOCK_W + 2 * radius)] += SQ(diff2) - SQ(diff1);

    diff1 = leftPixel1 - rightPixel1[5];
    diff2 = leftPixel2 - rightPixel2[5];
    col_ssd[5 * (BLOCK_W + 2 * radius)] += SQ(diff2) - SQ(diff1);

    diff1 = leftPixel1 - rightPixel1[6];
    diff2 = leftPixel2 - rightPixel2[6];
    col_ssd[6 * (BLOCK_W + 2 * radius)] += SQ(diff2) - SQ(diff1);

    diff1 = leftPixel1 - rightPixel1[7];
    diff2 = leftPixel2 - rightPixel2[7];
    col_ssd[7 * (BLOCK_W + 2 * radius)] += SQ(diff2) - SQ(diff1);
}

void InitColSSD(int x_tex, int y_tex, int im_pitch, __global unsigned char* imageL, 
                __global unsigned char* imageR, int d, 
                volatile __local unsigned int *col_ssd, int radius)
{
    unsigned char leftPixel1;
    int idx;
    unsigned int diffa[] = {0, 0, 0, 0, 0, 0, 0, 0};

    for(int i = 0; i < (2 * radius + 1); i++)
    {
        idx = y_tex * im_pitch + x_tex;
        leftPixel1 = imageL[idx];
        idx = idx - d;

        diffa[0] += SQ(leftPixel1 - imageR[idx - 0]);
        diffa[1] += SQ(leftPixel1 - imageR[idx - 1]);
        diffa[2] += SQ(leftPixel1 - imageR[idx - 2]);
        diffa[3] += SQ(leftPixel1 - imageR[idx - 3]);
        diffa[4] += SQ(leftPixel1 - imageR[idx - 4]);
        diffa[5] += SQ(leftPixel1 - imageR[idx - 5]);
        diffa[6] += SQ(leftPixel1 - imageR[idx - 6]);
        diffa[7] += SQ(leftPixel1 - imageR[idx - 7]);

        y_tex += 1;
    }
    //See above:  #define COL_SSD_SIZE (BLOCK_W + 2 * radius)
    col_ssd[0 * (BLOCK_W + 2 * radius)] = diffa[0];
    col_ssd[1 * (BLOCK_W + 2 * radius)] = diffa[1];
    col_ssd[2 * (BLOCK_W + 2 * radius)] = diffa[2];
    col_ssd[3 * (BLOCK_W + 2 * radius)] = diffa[3];
    col_ssd[4 * (BLOCK_W + 2 * radius)] = diffa[4];
    col_ssd[5 * (BLOCK_W + 2 * radius)] = diffa[5];
    col_ssd[6 * (BLOCK_W + 2 * radius)] = diffa[6];
    col_ssd[7 * (BLOCK_W + 2 * radius)] = diffa[7];
}

__kernel void stereoKernel(__global unsigned char *left, __global unsigned char *right,  
                           __global unsigned int *cminSSDImage, int cminSSD_step,
                           __global unsigned char *disp, int disp_step,int cwidth, int cheight,
                           int img_step, int maxdisp, int radius,  
                           __local unsigned int *col_ssd_cache)
{

    volatile __local unsigned int *col_ssd = col_ssd_cache + BLOCK_W + get_local_id(0);
    volatile __local unsigned int *col_ssd_extra = get_local_id(0) < (2 * radius) ? col_ssd + BLOCK_W : 0;  

    int X = get_group_id(0) * BLOCK_W + get_local_id(0) + maxdisp + radius;
   // int Y = get_group_id(1) * ROWSperTHREAD + radius;

    #define Y (get_group_id(1) * ROWSperTHREAD + radius)

    volatile __global unsigned int* minSSDImage = cminSSDImage + X + Y * cminSSD_step;
    __global unsigned char* disparImage = disp + X + Y * disp_step;

    int end_row = ROWSperTHREAD < (cheight - Y) ? ROWSperTHREAD:(cheight - Y);
    int y_tex;
    int x_tex = X - radius;

    if (x_tex >= cwidth)
        return;

    for(int d = STEREO_MIND; d < maxdisp; d += STEREO_DISP_STEP)
    {
        y_tex = Y - radius;

        InitColSSD(x_tex, y_tex, img_step, left, right, d, col_ssd, radius);
        if (col_ssd_extra > 0)
            if (x_tex + BLOCK_W < cwidth)
                InitColSSD(x_tex + BLOCK_W, y_tex, img_step, left, right, d, col_ssd_extra, radius);

        barrier(CLK_LOCAL_MEM_FENCE); //before MinSSD function

        if (X < cwidth - radius && Y < cheight - radius)
        {
            uint2 minSSD = MinSSD(col_ssd_cache + get_local_id(0), col_ssd, radius);
            if (minSSD.x < minSSDImage[0])
            {
                disparImage[0] = (unsigned char)(d + minSSD.y);
                minSSDImage[0] = minSSD.x;
            }
        }

        for(int row = 1; row < end_row; row++)
        {
            int idx1 = y_tex * img_step + x_tex;
            int idx2 = (y_tex + (2 * radius + 1)) * img_step + x_tex;

            barrier(CLK_GLOBAL_MEM_FENCE); 
            barrier(CLK_LOCAL_MEM_FENCE); 

            StepDown(idx1, idx2, left, right, d, col_ssd, radius);
            if (col_ssd_extra > 0)
                if (x_tex + BLOCK_W < cwidth)
                    StepDown(idx1, idx2, left + BLOCK_W, right + BLOCK_W, d, col_ssd_extra, radius);

            y_tex += 1;

            barrier(CLK_LOCAL_MEM_FENCE); 

            if (X < cwidth - radius && row < cheight - radius - Y)
            {
                int idx = row * cminSSD_step;
                uint2 minSSD = MinSSD(col_ssd_cache + get_local_id(0), col_ssd, radius);
                if (minSSD.x < minSSDImage[idx])
                {
                    disparImage[disp_step * row] = (unsigned char)(d + minSSD.y);
                    minSSDImage[idx] = minSSD.x;
                }
            }
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

float sobel(__global unsigned char *input, int x, int y, int rows, int cols)
{
    float conv = 0;
    int y1 = y==0? 0 : y-1;
    int x1 = x==0? 0 : x-1;
    if(x < cols && y < rows)
    {
        conv = (float)input[(y1)  * cols + (x1)] * (-1) + (float)input[(y1)  * cols + (x+1)] * (1) + 
               (float)input[(y)   * cols + (x1)] * (-2) + (float)input[(y)   * cols + (x+1)] * (2) +
               (float)input[(y+1) * cols + (x1)] * (-1) + (float)input[(y+1) * cols + (x+1)] * (1);
    
    }
    return fabs(conv);
}

float CalcSums(__local float *cols, __local float *cols_cache, int winsz)
{
    float cache = 0;
    float cache2 = 0;
    int winsz2 = winsz/2;

    int x = get_local_id(0);
    int group_size_x = get_local_size(0);

    for(int i = 1; i <= winsz2; i++)
        cache += cols[i];

    cols_cache[0] = cache;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x < group_size_x - winsz2)
        cache2 = cols_cache[winsz2];
    else
        for(int i = winsz2 + 1; i < winsz; i++)
            cache2 += cols[i];

    return cols[0] + cache + cache2;
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

 //   if (x < disp_cols)
 //   {
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
            float sum_win = CalcSums(cols, cols_cache + local_id_x, winsz) * 255;
            if (sum_win < threshold)
                disp[y * disp_step + x] = 0;

            barrier(CLK_LOCAL_MEM_FENCE);
        }
  //  }
}
