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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "cuda_shared.hpp"

#define ROWSperTHREAD 21     // the number of rows a thread will process
#define BLOCK_W 128          // the thread block width (464)
#define N_DISPARITIES 8

#define STEREO_MIND 0                    // The minimum d range to check 
#define STEREO_DISP_STEP N_DISPARITIES   // the d step, must be <= 1 to avoid aliasing
#define RADIUS 9                         // Kernel Radius 5V & 5H = 11x11 kernel

#define WINSZ (2 * RADIUS + 1)
#define N_DIRTY_PIXELS (2 * RADIUS)
#define COL_SSD_SIZE (BLOCK_W + N_DIRTY_PIXELS)
#define SHARED_MEM_SIZE (COL_SSD_SIZE) // amount of shared memory used

__constant__ unsigned int* cminSSDImage;
__constant__ size_t cminSSD_step;
__constant__ int cwidth;
__constant__ int cheight;

namespace device_code 
{

__device__ int SQ(int a)
{
    return a * a;    
}

__device__ unsigned int CalcSSD(unsigned int *col_ssd_cache, unsigned int *col_ssd)
{
    unsigned int cache = 0;
    unsigned int cache2 = 0;

    for(int i = 1; i <= RADIUS; i++)
        cache += col_ssd[i];
    
    col_ssd_cache[0] = cache;

    __syncthreads();

    if (threadIdx.x < BLOCK_W - RADIUS)
        cache2 = col_ssd_cache[RADIUS];
    else
        for(int i = RADIUS + 1; i < WINSZ; i++)
            cache2 += col_ssd[i];

    return col_ssd[0] + cache + cache2;
}

__device__ uint2 MinSSD(unsigned int *col_ssd_cache, unsigned int *col_ssd)
{
    unsigned int ssd[N_DISPARITIES];

    ssd[0] = CalcSSD(col_ssd_cache, col_ssd + 0 * SHARED_MEM_SIZE);
    ssd[1] = CalcSSD(col_ssd_cache, col_ssd + 1 * SHARED_MEM_SIZE);
    ssd[2] = CalcSSD(col_ssd_cache, col_ssd + 2 * SHARED_MEM_SIZE);
    ssd[3] = CalcSSD(col_ssd_cache, col_ssd + 3 * SHARED_MEM_SIZE);
    ssd[4] = CalcSSD(col_ssd_cache, col_ssd + 4 * SHARED_MEM_SIZE);
    ssd[5] = CalcSSD(col_ssd_cache, col_ssd + 5 * SHARED_MEM_SIZE);
    ssd[6] = CalcSSD(col_ssd_cache, col_ssd + 6 * SHARED_MEM_SIZE);
    ssd[7] = CalcSSD(col_ssd_cache, col_ssd + 7 * SHARED_MEM_SIZE);

    int mssd = min(min(min(ssd[0], ssd[1]), min(ssd[4], ssd[5])), min(min(ssd[2], ssd[3]), min(ssd[6], ssd[7])));

    int bestIdx = 0;
    for (int i = 0; i < N_DISPARITIES; i++)
    {
        if (mssd == ssd[i])
            bestIdx = i;
    }

    return make_uint2(mssd, bestIdx);
}

__device__ void StepDown(int idx1, int idx2, unsigned char* imageL, unsigned char* imageR, int d, unsigned int *col_ssd)
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
    

    diff1 = leftPixel1 - rightPixel1[0];                
    diff2 = leftPixel2 - rightPixel2[0];    
    col_ssd[0 * SHARED_MEM_SIZE] += SQ(diff2) - SQ(diff1);

    diff1 = leftPixel1 - rightPixel1[1];
    diff2 = leftPixel2 - rightPixel2[1];
    col_ssd[1 * SHARED_MEM_SIZE] += SQ(diff2) - SQ(diff1);
     
    diff1 = leftPixel1 - rightPixel1[2];
    diff2 = leftPixel2 - rightPixel2[2];
    col_ssd[2 * SHARED_MEM_SIZE] += SQ(diff2) - SQ(diff1);

    diff1 = leftPixel1 - rightPixel1[3];
    diff2 = leftPixel2 - rightPixel2[3];
    col_ssd[3 * SHARED_MEM_SIZE] += SQ(diff2) - SQ(diff1);
    
    diff1 = leftPixel1 - rightPixel1[4]; 
    diff2 = leftPixel2 - rightPixel2[4];               
    col_ssd[4 * SHARED_MEM_SIZE] += SQ(diff2) - SQ(diff1);
    
    diff1 = leftPixel1 - rightPixel1[5];
    diff2 = leftPixel2 - rightPixel2[5];
    col_ssd[5 * SHARED_MEM_SIZE] += SQ(diff2) - SQ(diff1);
    
    diff1 = leftPixel1 - rightPixel1[6];
    diff2 = leftPixel2 - rightPixel2[6];
    col_ssd[6 * SHARED_MEM_SIZE] += SQ(diff2) - SQ(diff1);
        
    diff1 = leftPixel1 - rightPixel1[7];
    diff2 = leftPixel2 - rightPixel2[7];
    col_ssd[7 * SHARED_MEM_SIZE] += SQ(diff2) - SQ(diff1);
}

__device__ void InitColSSD(int x_tex, int y_tex, int im_pitch, unsigned char* imageL, unsigned char* imageR, int d, unsigned int *col_ssd)
{
    unsigned char leftPixel1;
    int idx;
    unsigned int diffa[] = {0, 0, 0, 0, 0, 0, 0, 0};

    for(int i = 0; i < WINSZ; i++)
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

    col_ssd[0 * SHARED_MEM_SIZE] = diffa[0];
    col_ssd[1 * SHARED_MEM_SIZE] = diffa[1];
    col_ssd[2 * SHARED_MEM_SIZE] = diffa[2];
    col_ssd[3 * SHARED_MEM_SIZE] = diffa[3];
    col_ssd[4 * SHARED_MEM_SIZE] = diffa[4];
    col_ssd[5 * SHARED_MEM_SIZE] = diffa[5];
    col_ssd[6 * SHARED_MEM_SIZE] = diffa[6];
    col_ssd[7 * SHARED_MEM_SIZE] = diffa[7];
}

extern "C" __global__ void stereoKernel(unsigned char *left, unsigned char *right, size_t img_step, unsigned char* disp, size_t disp_pitch, int maxdisp)
{
    extern __shared__ unsigned int col_ssd_cache[];
    unsigned int *col_ssd = col_ssd_cache + BLOCK_W + threadIdx.x;    
    unsigned int *col_ssd_extra = threadIdx.x < N_DIRTY_PIXELS ? col_ssd + BLOCK_W : 0;

    //#define X (blockIdx.x * BLOCK_W + threadIdx.x + STEREO_MAXD)
    int X = (blockIdx.x * BLOCK_W + threadIdx.x + maxdisp);
    //#define Y (__mul24(blockIdx.y, ROWSperTHREAD) + RADIUS)
    #define Y (blockIdx.y * ROWSperTHREAD + RADIUS)
    //int Y = blockIdx.y * ROWSperTHREAD + RADIUS;

    unsigned int* minSSDImage = cminSSDImage + X + Y * cminSSD_step;
    unsigned char* disparImage = disp + X + Y * disp_pitch;
 /*   if (X < cwidth)
    {        
        unsigned int *minSSDImage_end = minSSDImage + min(ROWSperTHREAD, cheight - Y) * minssd_step;
        for(uint *ptr = minSSDImage; ptr != minSSDImage_end; ptr += minssd_step )
            *ptr = 0xFFFFFFFF;        
    }*/
    int end_row = min(ROWSperTHREAD, cheight - Y);
    int y_tex;    
    int x_tex = X - RADIUS;
    for(int d = STEREO_MIND; d < maxdisp; d += STEREO_DISP_STEP)
    {
        y_tex = Y - RADIUS;

        InitColSSD(x_tex, y_tex, img_step, left, right, d, col_ssd); 

        if (col_ssd_extra > 0)
            InitColSSD(x_tex + BLOCK_W, y_tex, img_step, left, right, d, col_ssd_extra);

        __syncthreads(); //before MinSSD function

        if (X < cwidth - RADIUS && Y < cheight - RADIUS)
        {
            uint2 minSSD = MinSSD(col_ssd_cache + threadIdx.x, col_ssd);
            if (minSSD.x < minSSDImage[0])
            {
                disparImage[0] = (unsigned char)(d + minSSD.y);
                minSSDImage[0] = minSSD.x;
            }
        }

        for(int row = 1; row < end_row; row++)
        {
            int idx1 = y_tex * img_step + x_tex;
            int idx2 = (y_tex + WINSZ) * img_step + x_tex;

            __syncthreads();

            StepDown(idx1, idx2, left, right, d, col_ssd);

            if (col_ssd_extra)
                StepDown(idx1, idx2, left + BLOCK_W, right + BLOCK_W, d, col_ssd_extra);

            y_tex += 1;
 
            __syncthreads(); //before MinSSD function

            if (X < cwidth - RADIUS && row < cheight - RADIUS - Y)
            {       
                int idx = row * cminSSD_step;         
                uint2 minSSD = MinSSD(col_ssd_cache + threadIdx.x, col_ssd);  
                if (minSSD.x < minSSDImage[idx])
                {
                    disparImage[disp_pitch * row] = (unsigned char)(d + minSSD.y);
                    minSSDImage[idx] = minSSD.x;
                }
            }
        } // for row loop
    } // for d loop
}

}

extern "C" void cv::gpu::impl::stereoBM_GPU(const DevMem2D& left, const DevMem2D& right, DevMem2D& disp, int maxdisp, DevMem2D_<unsigned int>& minSSD_buf)
{   
    //cudaSafeCall( cudaFuncSetCacheConfig(&stereoKernel, cudaFuncCachePreferL1) );
    //cudaSafeCall( cudaFuncSetCacheConfig(&stereoKernel, cudaFuncCachePreferShared) );
    
    size_t smem_size = (BLOCK_W + N_DISPARITIES * SHARED_MEM_SIZE) * sizeof(unsigned int);      

    cudaSafeCall( cudaMemset2D(disp.ptr, disp.step, 0, disp.cols, disp. rows) );
    cudaSafeCall( cudaMemset2D(minSSD_buf.ptr, minSSD_buf.step, 0xFF, minSSD_buf.cols * minSSD_buf.elemSize(), disp. rows) );        

    dim3 grid(1,1,1);
    dim3 threads(BLOCK_W, 1, 1);    
    
    grid.x = divUp(left.cols - maxdisp - 2 * RADIUS, BLOCK_W);
    grid.y = divUp(left.rows - 2 * RADIUS, ROWSperTHREAD);
    
    cudaSafeCall( cudaMemcpyToSymbol(  cwidth, &left.cols, sizeof (left.cols) ) );
    cudaSafeCall( cudaMemcpyToSymbol( cheight, &left.rows, sizeof (left.rows) ) );
    cudaSafeCall( cudaMemcpyToSymbol( cminSSDImage,  &minSSD_buf.ptr, sizeof (minSSD_buf.ptr) ) );

    size_t minssd_step = minSSD_buf.step/minSSD_buf.elemSize();
    cudaSafeCall( cudaMemcpyToSymbol( cminSSD_step,  &minssd_step, sizeof (minssd_step) ) );
         
    device_code::stereoKernel<<<grid, threads, smem_size>>>(left.ptr, right.ptr, left.step, disp.ptr, disp.step, maxdisp);
    cudaSafeCall( cudaThreadSynchronize() );
}