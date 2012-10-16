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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Dachuan Zhao, dachuan@multicorewareinc.com
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

//#pragma OPENCL EXTENSION cl_amd_printf : enable

__kernel void arithm_muls_D5 (__global float *src1, int src1_step, int src1_offset,
                             __global float *dst,  int dst_step,  int dst_offset,
                             int rows, int cols, int dst_step1, float scalar)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 2) + src1_offset);
        int dst_index  = mad24(y, dst_step,  (x << 2) + dst_offset);

        float data1 = *((__global float *)((__global char *)src1 + src1_index));
        float tmp = data1 * scalar;

        *((__global float *)((__global char *)dst + dst_index)) = tmp;
    }
}


__kernel void calcSharrDeriv_vertical_C1_D0(__global const uchar* src, int srcStep, int rows, int cols, int cn, __global short* dx_buf, int dx_bufStep, __global short* dy_buf, int dy_bufStep)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (y < rows && x < cols * cn)
    {
        const uchar src_val0 = (src + (y > 0 ? y-1 : rows > 1 ? 1 : 0) * srcStep)[x];
        const uchar src_val1 = (src + y * srcStep)[x];
        const uchar src_val2 = (src + (y < rows-1 ? y+1 : rows > 1 ? rows-2 : 0) * srcStep)[x];
        
        ((__global short*)((__global char*)dx_buf + y * dx_bufStep / 2))[x] = (src_val0 + src_val2) * 3 + src_val1 * 10;
        ((__global short*)((__global char*)dy_buf + y * dy_bufStep / 2))[x] = src_val2 - src_val0;
    }
}

__kernel void calcSharrDeriv_vertical_C4_D0(__global const uchar* src, int srcStep, int rows, int cols, int cn, __global short* dx_buf, int dx_bufStep, __global short* dy_buf, int dy_bufStep)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (y < rows && x < cols * cn)
    {
        const uchar src_val0 = (src + (y > 0 ? y - 1 : 1) * srcStep)[x];
        const uchar src_val1 = (src + y * srcStep)[x];
        const uchar src_val2 = (src + (y < rows - 1 ? y + 1 : rows - 2) * srcStep)[x];
        
        ((__global short*)((__global char*)dx_buf + y * dx_bufStep / 2))[x] = (src_val0 + src_val2) * 3 + src_val1 * 10;
        ((__global short*)((__global char*)dy_buf + y * dy_bufStep / 2))[x] = src_val2 - src_val0;
    }
}

__kernel void calcSharrDeriv_horizontal_C1_D0(int rows, int cols, int cn, __global const short* dx_buf, int dx_bufStep, __global const short* dy_buf, int dy_bufStep, __global short* dIdx, int dIdxStep, __global short* dIdy, int dIdyStep)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    const int colsn = cols * cn;

    if (y < rows && x < colsn)
    {
        __global const short* dx_buf_row = dx_buf + y * dx_bufStep;
        __global const short* dy_buf_row = dy_buf + y * dy_bufStep;

        const int xr = x + cn < colsn ? x + cn : (cols - 2) * cn + x + cn - colsn;
        const int xl = x - cn >= 0 ? x - cn : cn + x;

        ((__global short*)((__global char*)dIdx + y * dIdxStep / 2))[x] = dx_buf_row[xr] - dx_buf_row[xl];
        ((__global short*)((__global char*)dIdy + y * dIdyStep / 2))[x] = (dy_buf_row[xr] + dy_buf_row[xl]) * 3 + dy_buf_row[x] * 10;
    }
}

__kernel void calcSharrDeriv_horizontal_C4_D0(int rows, int cols, int cn, __global const short* dx_buf, int dx_bufStep, __global const short* dy_buf, int dy_bufStep, __global short* dIdx, int dIdxStep, __global short* dIdy, int dIdyStep)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    const int colsn = cols * cn;

    if (y < rows && x < colsn)
    {
        __global const short* dx_buf_row = dx_buf + y * dx_bufStep;
        __global const short* dy_buf_row = dy_buf + y * dy_bufStep;

        const int xr = x + cn < colsn ? x + cn : (cols - 2) * cn + x + cn - colsn;
        const int xl = x - cn >= 0 ? x - cn : cn + x;

        ((__global short*)((__global char*)dIdx + y * dIdxStep / 2))[x] = dx_buf_row[xr] - dx_buf_row[xl];
        ((__global short*)((__global char*)dIdy + y * dIdyStep / 2))[x] = (dy_buf_row[xr] + dy_buf_row[xl]) * 3 + dy_buf_row[x] * 10;
    }
}

#define W_BITS 14
#define W_BITS1 14

#define  CV_DESCALE(x, n)     (((x) + (1 << ((n)-1))) >> (n))

int linearFilter_uchar(__global const uchar* src, int srcStep, int cn, float2 pt, int x, int y)
{
    int2 ipt;
    ipt.x = convert_int_sat_rtn(pt.x);
    ipt.y = convert_int_sat_rtn(pt.y);

    float a = pt.x - ipt.x;
    float b = pt.y - ipt.y;

    int iw00 = convert_int_sat_rte((1.0f - a) * (1.0f - b) * (1 << W_BITS));
    int iw01 = convert_int_sat_rte(a * (1.0f - b) * (1 << W_BITS));
    int iw10 = convert_int_sat_rte((1.0f - a) * b * (1 << W_BITS));
    int iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

    __global const uchar* src_row = src + (ipt.y + y) * srcStep + ipt.x * cn;
    __global const uchar* src_row1 = src + (ipt.y + y + 1) * srcStep + ipt.x * cn;

    return CV_DESCALE(src_row[x] * iw00 + src_row[x + cn] * iw01 + src_row1[x] * iw10 + src_row1[x + cn] * iw11, W_BITS1 - 5);
}

int linearFilter_short(__global const short* src, int srcStep, int cn, float2 pt, int x, int y)
{
    int2 ipt;
    ipt.x = convert_int_sat_rtn(pt.x);
    ipt.y = convert_int_sat_rtn(pt.y);

    float a = pt.x - ipt.x;
    float b = pt.y - ipt.y;

    int iw00 = convert_int_sat_rte((1.0f - a) * (1.0f - b) * (1 << W_BITS));
    int iw01 = convert_int_sat_rte(a * (1.0f - b) * (1 << W_BITS));
    int iw10 = convert_int_sat_rte((1.0f - a) * b * (1 << W_BITS));
    int iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

    __global const short* src_row = src + (ipt.y + y) * srcStep + ipt.x * cn;
    __global const short* src_row1 = src + (ipt.y + y + 1) * srcStep + ipt.x * cn;

    return CV_DESCALE(src_row[x] * iw00 + src_row[x + cn] * iw01 + src_row1[x] * iw10 + src_row1[x + cn] * iw11, W_BITS1);
}

float linearFilter_float(__global const float* src, int srcStep, int cn, float2 pt, float x, float y)
{
    int2 ipt;
    ipt.x = convert_int_sat_rtn(pt.x);
    ipt.y = convert_int_sat_rtn(pt.y);

    float a = pt.x - ipt.x;
    float b = pt.y - ipt.y;

    float iw00 = ((1.0f - a) * (1.0f - b) * (1 << W_BITS));
    float iw01 = (a * (1.0f - b) * (1 << W_BITS));
    float iw10 = ((1.0f - a) * b * (1 << W_BITS));
    float iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

    __global const float* src_row = src + (int)(ipt.y + y) * srcStep / 4 + ipt.x * cn;
    __global const float* src_row1 = src + (int)(ipt.y + y + 1) * srcStep / 4 + ipt.x * cn;

    return src_row[(int)x] * iw00 + src_row[(int)x + cn] * iw01 + src_row1[(int)x] * iw10 + src_row1[(int)x + cn] * iw11, W_BITS1 - 5;
}

void reduce3(float val1, float val2, float val3, __local float* smem1, __local float* smem2, __local float* smem3, int tid)
{
    smem1[tid] = val1;
    smem2[tid] = val2;
    smem3[tid] = val3;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 128) 
    { 
        smem1[tid] = val1 += smem1[tid + 128]; 
        smem2[tid] = val2 += smem2[tid + 128]; 
        smem3[tid] = val3 += smem3[tid + 128]; 
    } 
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 64) 
    { 
        smem1[tid] = val1 += smem1[tid + 64]; 
        smem2[tid] = val2 += smem2[tid + 64]; 
        smem3[tid] = val3 += smem3[tid + 64];
    } 
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 32)
    {
        volatile __local float* vmem1 = smem1;
        volatile __local float* vmem2 = smem2;
        volatile __local float* vmem3 = smem3;

        vmem1[tid] = val1 += vmem1[tid + 32]; 
        vmem2[tid] = val2 += vmem2[tid + 32]; 
        vmem3[tid] = val3 += vmem3[tid + 32];

        vmem1[tid] = val1 += vmem1[tid + 16]; 
        vmem2[tid] = val2 += vmem2[tid + 16]; 
        vmem3[tid] = val3 += vmem3[tid + 16];

        vmem1[tid] = val1 += vmem1[tid + 8]; 
        vmem2[tid] = val2 += vmem2[tid + 8]; 
        vmem3[tid] = val3 += vmem3[tid + 8];

        vmem1[tid] = val1 += vmem1[tid + 4]; 
        vmem2[tid] = val2 += vmem2[tid + 4]; 
        vmem3[tid] = val3 += vmem3[tid + 4];

        vmem1[tid] = val1 += vmem1[tid + 2]; 
        vmem2[tid] = val2 += vmem2[tid + 2]; 
        vmem3[tid] = val3 += vmem3[tid + 2];

        vmem1[tid] = val1 += vmem1[tid + 1]; 
        vmem2[tid] = val2 += vmem2[tid + 1]; 
        vmem3[tid] = val3 += vmem3[tid + 1];
    }
}

void reduce2(float val1, float val2, __local float* smem1, __local float* smem2, int tid)
{
    smem1[tid] = val1;
    smem2[tid] = val2;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 128) 
    { 
        smem1[tid] = val1 += smem1[tid + 128]; 
        smem2[tid] = val2 += smem2[tid + 128];  
    } 
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 64) 
    { 
        smem1[tid] = val1 += smem1[tid + 64]; 
        smem2[tid] = val2 += smem2[tid + 64]; 
    } 
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 32)
    {
        volatile __local float* vmem1 = smem1;
        volatile __local float* vmem2 = smem2;

        vmem1[tid] = val1 += vmem1[tid + 32]; 
        vmem2[tid] = val2 += vmem2[tid + 32]; 

        vmem1[tid] = val1 += vmem1[tid + 16]; 
        vmem2[tid] = val2 += vmem2[tid + 16]; 

        vmem1[tid] = val1 += vmem1[tid + 8]; 
        vmem2[tid] = val2 += vmem2[tid + 8]; 

        vmem1[tid] = val1 += vmem1[tid + 4]; 
        vmem2[tid] = val2 += vmem2[tid + 4]; 

        vmem1[tid] = val1 += vmem1[tid + 2]; 
        vmem2[tid] = val2 += vmem2[tid + 2]; 

        vmem1[tid] = val1 += vmem1[tid + 1]; 
        vmem2[tid] = val2 += vmem2[tid + 1]; 
    }
}

void reduce1(float val1, __local float* smem1, int tid)
{
    smem1[tid] = val1;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 128) 
    { 
        smem1[tid] = val1 += smem1[tid + 128]; 
    } 
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 64) 
    { 
        smem1[tid] = val1 += smem1[tid + 64]; 
    } 
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 32)
    {
        volatile __local float* vmem1 = smem1;

        vmem1[tid] = val1 += vmem1[tid + 32]; 
        vmem1[tid] = val1 += vmem1[tid + 16]; 
        vmem1[tid] = val1 += vmem1[tid + 8]; 
        vmem1[tid] = val1 += vmem1[tid + 4];
        vmem1[tid] = val1 += vmem1[tid + 2]; 
        vmem1[tid] = val1 += vmem1[tid + 1]; 
    }
}

#define SCALE (1.0f / (1 << 20))

// Image read mode
__constant sampler_t sampler    = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

__kernel void lkSparse_C1_D5(image2d_t I, image2d_t J,
    __global const float2* prevPts, int prevPtsStep, __global float2* nextPts, int nextPtsStep, __global uchar* status/*, __global float* err*/, const int level, const int rows, const int cols, int PATCH_X, int PATCH_Y, int cn, int c_winSize_x, int c_winSize_y, int c_iters, char calcErr, char GET_MIN_EIGENVALS)
{
    __local float smem1[256];
    __local float smem2[256];
    __local float smem3[256];

	int c_halfWin_x = (c_winSize_x - 1) / 2;
	int c_halfWin_y = (c_winSize_y - 1) / 2;

    const int tid = get_local_id(1) * get_local_size(0) + get_local_id(0);

    float2 prevPt = prevPts[get_group_id(0)];
    prevPt.x *= (1.0f / (1 << level));
    prevPt.y *= (1.0f / (1 << level));

    if (prevPt.x < 0 || prevPt.x >= cols || prevPt.y < 0 || prevPt.y >= rows)
    {
        if (level == 0 && tid == 0)
        {
            status[get_group_id(0)] = 0;

            //if (calcErr) 
            //    err[get_group_id(0)] = 0;
        }

        return;
    }
    
    prevPt.x -= c_halfWin_x;
    prevPt.y -= c_halfWin_y;
    
    // extract the patch from the first image, compute covariation matrix of derivatives
    
    float A11 = 0;
    float A12 = 0;
    float A22 = 0;

    float I_patch[21][21];
    float dIdx_patch[21][21];
    float dIdy_patch[21][21];

    for (int yBase = get_local_id(1), i = 0; yBase < c_winSize_y; yBase += get_local_size(1), ++i)
    {                
        for (int xBase = get_local_id(0), j = 0; xBase < c_winSize_x; xBase += get_local_size(0), ++j)
        {
            float x = (prevPt.x + xBase + 0.5f);
            float y = (prevPt.y + yBase + 0.5f);

            I_patch[i][j] = read_imagef(I, sampler, (float2)(x, y)).x;
            
            float dIdx = 3.0f * read_imagef(I, sampler, (float2)(x + 1, y - 1)).x + 10.0f * read_imagef(I, sampler, (float2)(x + 1, y)).x + 3.0f * read_imagef(I, sampler, (float2)(x + 1, y + 1)).x -
                             (3.0f * read_imagef(I, sampler, (float2)(x - 1, y - 1)).x + 10.0f * read_imagef(I, sampler, (float2)(x - 1, y)).x + 3.0f * read_imagef(I, sampler, (float2)(x - 1, y + 1)).x);

            float dIdy = 3.0f * read_imagef(I, sampler, (float2)(x - 1, y + 1)).x + 10.0f * read_imagef(I, sampler, (float2)(x, y + 1)).x + 3.0f * read_imagef(I, sampler, (float2)(x + 1, y + 1)).x -
                            (3.0f * read_imagef(I, sampler, (float2)(x - 1, y - 1)).x + 10.0f * read_imagef(I, sampler, (float2)(x, y - 1)).x + 3.0f * read_imagef(I, sampler, (float2)(x + 1, y - 1)).x);

            dIdx_patch[i][j] = dIdx;
            dIdy_patch[i][j] = dIdy;
            
            A11 += dIdx * dIdx;
            A12 += dIdx * dIdy;
            A22 += dIdy * dIdy;
        }
    }

    reduce3(A11, A12, A22, smem1, smem2, smem3, tid);
    barrier(CLK_LOCAL_MEM_FENCE);

    A11 = smem1[0];
    A12 = smem2[0];
    A22 = smem3[0];
    
    float D = A11 * A22 - A12 * A12;

    //if (calcErr && GET_MIN_EIGENVALS && tid == 0) 
    //    err[get_group_id(0)] = minEig;

    if (D < 1.192092896e-07f)
    {
        if (level == 0 && tid == 0)
            status[get_group_id(0)] = 0;

        return;
    }

    D = 1.f / D;

    A11 *= D;
    A12 *= D;
    A22 *= D;

    float2 nextPt = nextPts[get_group_id(0)];
    nextPt.x *= 2.0f;
    nextPt.y *= 2.0f; 
    
    nextPt.x -= c_halfWin_x;
    nextPt.y -= c_halfWin_y;

    for (int k = 0; k < c_iters; ++k)
    {
        if (nextPt.x < -c_halfWin_x || nextPt.x >= cols || nextPt.y < -c_halfWin_y || nextPt.y >= rows)
        {
            if (tid == 0 && level == 0)
                status[get_group_id(0)] = 0;
            return;
        }

        float b1 = 0;
        float b2 = 0;
        
        for (int y = get_local_id(1), i = 0; y < c_winSize_y; y += get_local_size(1), ++i)
        {
            for (int x = get_local_id(0), j = 0; x < c_winSize_x; x += get_local_size(0), ++j)
            {
				float a = (nextPt.x + x + 0.5f);
				float b = (nextPt.y + y + 0.5f);
				
                float I_val = I_patch[i][j];
                float J_val = read_imagef(J, sampler, (float2)(a, b)).x;

                float diff = (J_val - I_val) * 32.0f;

                b1 += diff * dIdx_patch[i][j];
                b2 += diff * dIdy_patch[i][j];
            }
        }
        
        reduce2(b1, b2, smem1, smem2, tid);
        barrier(CLK_LOCAL_MEM_FENCE);

        b1 = smem1[0];
        b2 = smem2[0];

        float2 delta;
        delta.x = A12 * b2 - A22 * b1;
        delta.y = A12 * b1 - A11 * b2;
            
        nextPt.x += delta.x;
        nextPt.y += delta.y;

        if (fabs(delta.x) < 0.01f && fabs(delta.y) < 0.01f)
            break;
    }

    float errval = 0.0f;
    if (calcErr)
    {
        for (int y = get_local_id(1), i = 0; y < c_winSize_y; y += get_local_size(1), ++i)
        {
            for (int x = get_local_id(0), j = 0; x < c_winSize_x; x += get_local_size(0), ++j)
            {
				float a = (nextPt.x + x + 0.5f);
				float b = (nextPt.y + y + 0.5f);
				
                float I_val = I_patch[i][j];
                float J_val = read_imagef(J, sampler, (float2)(a, b)).x;

                float diff = J_val - I_val;

                errval += fabs((float)diff);
            }
        }

        reduce1(errval, smem1, tid);
    }

    if (tid == 0)
    {
        nextPt.x += c_halfWin_x;
        nextPt.y += c_halfWin_y;

        nextPts[get_group_id(0)] = nextPt;

        //if (calcErr && !GET_MIN_EIGENVALS)
        //    err[get_group_id(0)] = errval;
    }
}
__kernel void lkSparse_C4_D5(image2d_t I, image2d_t J,
    __global const float2* prevPts, int prevPtsStep, __global float2* nextPts, int nextPtsStep, __global uchar* status/*, __global float* err*/, const int level, const int rows, const int cols, int PATCH_X, int PATCH_Y, int cn, int c_winSize_x, int c_winSize_y, int c_iters, char calcErr, char GET_MIN_EIGENVALS)
{
    __local float smem1[256];
    __local float smem2[256];
    __local float smem3[256];

	int c_halfWin_x = (c_winSize_x - 1) / 2;
	int c_halfWin_y = (c_winSize_y - 1) / 2;

    const int tid = get_local_id(1) * get_local_size(0) + get_local_id(0);

    float2 prevPt = prevPts[get_group_id(0)];
    prevPt.x *= (1.0f / (1 << level));
    prevPt.y *= (1.0f / (1 << level));

    if (prevPt.x < 0 || prevPt.x >= cols || prevPt.y < 0 || prevPt.y >= rows)
    {
        if (level == 0 && tid == 0)
        {
            status[get_group_id(0)] = 0;

            //if (calcErr) 
            //    err[get_group_id(0)] = 0;
        }

        return;
    }
    
    prevPt.x -= c_halfWin_x;
    prevPt.y -= c_halfWin_y;
    
    // extract the patch from the first image, compute covariation matrix of derivatives
    
    float A11 = 0;
    float A12 = 0;
    float A22 = 0;

    float4 I_patch[21][21];
    float4 dIdx_patch[21][21];
    float4 dIdy_patch[21][21];

    for (int yBase = get_local_id(1), i = 0; yBase < c_winSize_y; yBase += get_local_size(1), ++i)
    {                
        for (int xBase = get_local_id(0), j = 0; xBase < c_winSize_x; xBase += get_local_size(0), ++j)
        {
            float x = (prevPt.x + xBase + 0.5f);
            float y = (prevPt.y + yBase + 0.5f);

            I_patch[i][j] = read_imagef(I, sampler, (float2)(x, y)).x;
            
            float4 dIdx = 3.0f * read_imagef(I, sampler, (float2)(x + 1, y - 1)).x + 10.0f * read_imagef(I, sampler, (float2)(x + 1, y)).x + 3.0f * read_imagef(I, sampler, (float2)(x + 1, y + 1)).x -
                             (3.0f * read_imagef(I, sampler, (float2)(x - 1, y - 1)).x + 10.0f * read_imagef(I, sampler, (float2)(x - 1, y)).x + 3.0f * read_imagef(I, sampler, (float2)(x - 1, y + 1)).x);

            float4 dIdy = 3.0f * read_imagef(I, sampler, (float2)(x - 1, y + 1)).x + 10.0f * read_imagef(I, sampler, (float2)(x, y + 1)).x + 3.0f * read_imagef(I, sampler, (float2)(x + 1, y + 1)).x -
                            (3.0f * read_imagef(I, sampler, (float2)(x - 1, y - 1)).x + 10.0f * read_imagef(I, sampler, (float2)(x, y - 1)).x + 3.0f * read_imagef(I, sampler, (float2)(x + 1, y - 1)).x);

            dIdx_patch[i][j] = dIdx;
            dIdy_patch[i][j] = dIdy;
            
            A11 += (dIdx * dIdx).x + (dIdx * dIdx).y + (dIdx * dIdx).z;
            A12 += (dIdx * dIdy).x + (dIdx * dIdy).y + (dIdx * dIdy).z;
            A22 += (dIdy * dIdy).x + (dIdy * dIdy).y + (dIdy * dIdy).z;
        }
    }

    reduce3(A11, A12, A22, smem1, smem2, smem3, tid);
    barrier(CLK_LOCAL_MEM_FENCE);

    A11 = smem1[0];
    A12 = smem2[0];
    A22 = smem3[0];
    
    float D = A11 * A22 - A12 * A12;

    //if (calcErr && GET_MIN_EIGENVALS && tid == 0) 
    //    err[get_group_id(0)] = minEig;

    if (D < 1.192092896e-07f)
    {
        if (level == 0 && tid == 0)
            status[get_group_id(0)] = 0;

        return;
    }

    D = 1.f / D;

    A11 *= D;
    A12 *= D;
    A22 *= D;

    float2 nextPt = nextPts[get_group_id(0)];
    nextPt.x *= 2.0f;
    nextPt.y *= 2.0f; 
    
    nextPt.x -= c_halfWin_x;
    nextPt.y -= c_halfWin_y;

    for (int k = 0; k < c_iters; ++k)
    {
        if (nextPt.x < -c_halfWin_x || nextPt.x >= cols || nextPt.y < -c_halfWin_y || nextPt.y >= rows)
        {
            if (tid == 0 && level == 0)
                status[get_group_id(0)] = 0;
            return;
        }

        float b1 = 0;
        float b2 = 0;
        
        for (int y = get_local_id(1), i = 0; y < c_winSize_y; y += get_local_size(1), ++i)
        {
            for (int x = get_local_id(0), j = 0; x < c_winSize_x; x += get_local_size(0), ++j)
            {
				float a = (nextPt.x + x + 0.5f);
				float b = (nextPt.y + y + 0.5f);
				
                float4 I_val = I_patch[i][j];
                float4 J_val = read_imagef(J, sampler, (float2)(a, b)).x;

                float4 diff = (J_val - I_val) * 32.0f;

                b1 += (diff * dIdx_patch[i][j]).x + (diff * dIdx_patch[i][j]).y + (diff * dIdx_patch[i][j]).z;
                b2 += (diff * dIdy_patch[i][j]).x + (diff * dIdy_patch[i][j]).y + (diff * dIdy_patch[i][j]).z;
            }
        }
        
        reduce2(b1, b2, smem1, smem2, tid);
        barrier(CLK_LOCAL_MEM_FENCE);

        b1 = smem1[0];
        b2 = smem2[0];

        float2 delta;
        delta.x = A12 * b2 - A22 * b1;
        delta.y = A12 * b1 - A11 * b2;
            
        nextPt.x += delta.x;
        nextPt.y += delta.y;

        if (fabs(delta.x) < 0.01f && fabs(delta.y) < 0.01f)
            break;
    }

    float errval = 0.0f;
    if (calcErr)
    {
        for (int y = get_local_id(1), i = 0; y < c_winSize_y; y += get_local_size(1), ++i)
        {
            for (int x = get_local_id(0), j = 0; x < c_winSize_x; x += get_local_size(0), ++j)
            {
				float a = (nextPt.x + x + 0.5f);
				float b = (nextPt.y + y + 0.5f);
				
                float4 I_val = I_patch[i][j];
                float4 J_val = read_imagef(J, sampler, (float2)(a, b)).x;

                float4 diff = J_val - I_val;

                errval += fabs(diff.x) + fabs(diff.y) + fabs(diff.z);
            }
        }

        reduce1(errval, smem1, tid);
    }

    if (tid == 0)
    {
        nextPt.x += c_halfWin_x;
        nextPt.y += c_halfWin_y;

        nextPts[get_group_id(0)] = nextPt;

        //if (calcErr && !GET_MIN_EIGENVALS)
        //    err[get_group_id(0)] = errval;
    }
}

__kernel void lkDense_C1_D0(image2d_t I, image2d_t J, __global float* u, int uStep, __global float* v, int vStep, __global const float* prevU, int prevUStep, __global const float* prevV, int prevVStep, 
    const int rows, const int cols, /*__global float* err, int errStep, int cn,*/ int c_winSize_x, int c_winSize_y, int c_iters, char calcErr)
{
	int c_halfWin_x = (c_winSize_x - 1) / 2;
	int c_halfWin_y = (c_winSize_y - 1) / 2;

    const int patchWidth  = get_local_size(0) + 2 * c_halfWin_x;
    const int patchHeight = get_local_size(1) + 2 * c_halfWin_y;

    __local int smem[8192];

    __local int* I_patch = smem;
    __local int* dIdx_patch = I_patch + patchWidth * patchHeight;
    __local int* dIdy_patch = dIdx_patch + patchWidth * patchHeight;

    const int xBase = get_group_id(0) * get_local_size(0);
    const int yBase = get_group_id(1) * get_local_size(1);

	sampler_t sampleri    = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
	
    for (int i = get_local_id(1); i < patchHeight; i += get_local_size(1))
    {
        for (int j = get_local_id(0); j < patchWidth; j += get_local_size(0))
        {
            float x = xBase - c_halfWin_x + j + 0.5f;
            float y = yBase - c_halfWin_y + i + 0.5f;

            I_patch[i * patchWidth + j] = read_imagei(I, sampleri, (float2)(x, y)).x;

            // Sharr Deriv

            dIdx_patch[i * patchWidth + j] = 3 * read_imagei(I, sampleri, (float2)(x+1, y-1)).x + 10 * read_imagei(I, sampleri, (float2)(x+1, y)).x + 3 * read_imagei(I, sampleri, (float2)(x+1, y+1)).x -
                                            (3 * read_imagei(I, sampleri, (float2)(x-1, y-1)).x + 10 * read_imagei(I, sampleri, (float2)(x-1, y)).x + 3 * read_imagei(I, sampleri, (float2)(x-1, y+1)).x);

            dIdy_patch[i * patchWidth + j] = 3 * read_imagei(I, sampleri, (float2)(x-1, y+1)).x + 10 * read_imagei(I, sampleri, (float2)(x, y+1)).x + 3 * read_imagei(I, sampleri, (float2)(x+1, y+1)).x -
                                            (3 * read_imagei(I, sampleri, (float2)(x-1, y-1)).x + 10 * read_imagei(I, sampleri, (float2)(x, y-1)).x + 3 * read_imagei(I, sampleri, (float2)(x+1, y-1)).x);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // extract the patch from the first image, compute covariation matrix of derivatives
    
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= cols || y >= rows)
        return;

    int A11i = 0;
    int A12i = 0;
    int A22i = 0;

    for (int i = 0; i < c_winSize_y; ++i)
    {                
        for (int j = 0; j < c_winSize_x; ++j)
        {
            int dIdx = dIdx_patch[(get_local_id(1) + i) * patchWidth + (get_local_id(0) + j)];
            int dIdy = dIdy_patch[(get_local_id(1) + i) * patchWidth + (get_local_id(0) + j)];
            
            A11i += dIdx * dIdx;
            A12i += dIdx * dIdy;
            A22i += dIdy * dIdy;
        }
    }
    
    float A11 = A11i;
    float A12 = A12i;
    float A22 = A22i;

    float D = A11 * A22 - A12 * A12;
    
    //if (calcErr && GET_MIN_EIGENVALS)
    //    (err + y * errStep)[x] = minEig;

    if (D < 1.192092896e-07f)
    {
        //if (calcErr)
        //    err(y, x) = 3.402823466e+38f;

        return;
    }

    D = 1.f / D;

    A11 *= D;
    A12 *= D;
    A22 *= D;

    float2 nextPt;
    nextPt.x = x + prevU[y/2 * prevUStep / 4 + x/2] * 2.0f;
    nextPt.y = y + prevV[y/2 * prevVStep / 4 + x/2] * 2.0f;

    for (int k = 0; k < c_iters; ++k)
    {
        if (nextPt.x < 0 || nextPt.x >= cols || nextPt.y < 0 || nextPt.y >= rows)
        {
            //if (calcErr)
            //    err(y, x) = 3.402823466e+38f;

            return;
        }

        int b1 = 0;
        int b2 = 0;

        for (int i = 0; i < c_winSize_y; ++i)
        {
            for (int j = 0; j < c_winSize_x; ++j)
            {
                int iI = I_patch[(get_local_id(1) + i) * patchWidth + get_local_id(0) + j];
                int iJ = read_imagei(J, sampler, (float2)(nextPt.x - c_halfWin_x + j + 0.5f, nextPt.y - c_halfWin_y + i + 0.5f)).x;

                int diff = (iJ - iI) * 32;

                int dIdx = dIdx_patch[(get_local_id(1) + i) * patchWidth + (get_local_id(0) + j)];
                int dIdy = dIdy_patch[(get_local_id(1) + i) * patchWidth + (get_local_id(0) + j)];

                b1 += diff * dIdx;
                b2 += diff * dIdy;
            }
        }

        float2 delta;
        delta.x = A12 * b2 - A22 * b1;
        delta.y = A12 * b1 - A11 * b2;
            
        nextPt.x += delta.x;
        nextPt.y += delta.y;

        if (fabs(delta.x) < 0.01f && fabs(delta.y) < 0.01f)
            break;
    }

    u[y * uStep / 4 + x] = nextPt.x - x;
    v[y * vStep / 4 + x] = nextPt.y - y;

    if (calcErr)
    {
        int errval = 0;

        for (int i = 0; i < c_winSize_y; ++i)
        {
            for (int j = 0; j < c_winSize_x; ++j)
            {
                int iI = I_patch[(get_local_id(1) + i) * patchWidth + get_local_id(0) + j];
                int iJ = read_imagei(J, sampler, (float2)(nextPt.x - c_halfWin_x + j + 0.5f, nextPt.y - c_halfWin_y + i + 0.5f)).x;

                errval += abs(iJ - iI);
            }
        }

        //err[y * errStep / 4 + x] = static_cast<float>(errval) / (c_winSize_x * c_winSize_y);
    }
}
