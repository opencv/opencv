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
//    Yao Wang, bitwangyaoyao@gmail.com
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

#define	BUFFER	64
void reduce3(float val1, float val2, float val3, __local float* smem1, __local float* smem2, __local float* smem3, int tid)
{
    smem1[tid] = val1;
    smem2[tid] = val2;
    smem3[tid] = val3;
    barrier(CLK_LOCAL_MEM_FENCE);

#if	BUFFER > 128
    if (tid < 128)
    {
        smem1[tid] = val1 += smem1[tid + 128];
        smem2[tid] = val2 += smem2[tid + 128];
        smem3[tid] = val3 += smem3[tid + 128];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if	BUFFER > 64
    if (tid < 64)
    {
        smem1[tid] = val1 += smem1[tid + 64];
        smem2[tid] = val2 += smem2[tid + 64];
        smem3[tid] = val3 += smem3[tid + 64];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

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

#if	BUFFER > 128
    if (tid < 128)
    {
        smem1[tid] = val1 += smem1[tid + 128];
        smem2[tid] = val2 += smem2[tid + 128];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if	BUFFER > 64
    if (tid < 64)
    {
        smem1[tid] = val1 += smem1[tid + 64];
        smem2[tid] = val2 += smem2[tid + 64];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

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

#if	BUFFER > 128
    if (tid < 128)
    {
        smem1[tid] = val1 += smem1[tid + 128];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if	BUFFER > 64
    if (tid < 64)
    {
        smem1[tid] = val1 += smem1[tid + 64];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

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
#define	THRESHOLD	0.01f
#define	DIMENSION	21

// Image read mode
__constant sampler_t sampler    = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

void SetPatch(image2d_t I, float x, float y,
                                float* Pch, float* Dx, float* Dy,
                                float* A11, float* A12, float* A22)
{
            *Pch = read_imagef(I, sampler, (float2)(x, y)).x;

            float dIdx = 3.0f * read_imagef(I, sampler, (float2)(x + 1, y - 1)).x + 10.0f * read_imagef(I, sampler, (float2)(x + 1, y)).x + 3.0f * read_imagef(I, sampler, (float2)(x + 1, y + 1)).x -
                             (3.0f * read_imagef(I, sampler, (float2)(x - 1, y - 1)).x + 10.0f * read_imagef(I, sampler, (float2)(x - 1, y)).x + 3.0f * read_imagef(I, sampler, (float2)(x - 1, y + 1)).x);

            float dIdy = 3.0f * read_imagef(I, sampler, (float2)(x - 1, y + 1)).x + 10.0f * read_imagef(I, sampler, (float2)(x, y + 1)).x + 3.0f * read_imagef(I, sampler, (float2)(x + 1, y + 1)).x -
                            (3.0f * read_imagef(I, sampler, (float2)(x - 1, y - 1)).x + 10.0f * read_imagef(I, sampler, (float2)(x, y - 1)).x + 3.0f * read_imagef(I, sampler, (float2)(x + 1, y - 1)).x);


            *Dx = dIdx;
            *Dy = dIdy;

            *A11 += dIdx * dIdx;
            *A12 += dIdx * dIdy;
            *A22 += dIdy * dIdy;
}

void GetPatch(image2d_t J, float x, float y,
                                float* Pch, float* Dx, float* Dy,
                                float* b1, float* b2)
{
                float J_val = read_imagef(J, sampler, (float2)(x, y)).x;
                float diff = (J_val - *Pch) * 32.0f;
                *b1 += diff**Dx;
                *b2 += diff**Dy;
}

void GetError(image2d_t J, const float x, const float y, const float* Pch, float* errval)
{
        float diff = read_imagef(J, sampler, (float2)(x,y)).x-*Pch;
        *errval += fabs(diff);
}

void SetPatch4(image2d_t I, const float x, const float y,
                                float4* Pch, float4* Dx, float4* Dy,
                                float* A11, float* A12, float* A22)
{
            *Pch = read_imagef(I, sampler, (float2)(x, y));

            float4 dIdx = 3.0f * read_imagef(I, sampler, (float2)(x + 1, y - 1)) + 10.0f * read_imagef(I, sampler, (float2)(x + 1, y)) + 3.0f * read_imagef(I, sampler, (float2)(x + 1, y + 1)) -
                             (3.0f * read_imagef(I, sampler, (float2)(x - 1, y - 1)) + 10.0f * read_imagef(I, sampler, (float2)(x - 1, y)) + 3.0f * read_imagef(I, sampler, (float2)(x - 1, y + 1)));

            float4 dIdy = 3.0f * read_imagef(I, sampler, (float2)(x - 1, y + 1)) + 10.0f * read_imagef(I, sampler, (float2)(x, y + 1)) + 3.0f * read_imagef(I, sampler, (float2)(x + 1, y + 1)) -
                            (3.0f * read_imagef(I, sampler, (float2)(x - 1, y - 1)) + 10.0f * read_imagef(I, sampler, (float2)(x, y - 1)) + 3.0f * read_imagef(I, sampler, (float2)(x + 1, y - 1)));


            *Dx = dIdx;
            *Dy = dIdy;
                        float4 sqIdx = dIdx * dIdx;
                        *A11 += sqIdx.x + sqIdx.y + sqIdx.z;
                        sqIdx = dIdx * dIdy;
                        *A12 += sqIdx.x + sqIdx.y + sqIdx.z;
                        sqIdx = dIdy * dIdy;
                        *A22 += sqIdx.x + sqIdx.y + sqIdx.z;
}

void GetPatch4(image2d_t J, const float x, const float y,
                                const float4* Pch, const float4* Dx, const float4* Dy,
                                float* b1, float* b2)
{
                float4 J_val = read_imagef(J, sampler, (float2)(x, y));
                float4 diff = (J_val - *Pch) * 32.0f;
                                float4 xdiff = diff* *Dx;
                                *b1 += xdiff.x + xdiff.y + xdiff.z;
                                xdiff = diff* *Dy;
                                *b2 += xdiff.x + xdiff.y + xdiff.z;
}

void GetError4(image2d_t J, const float x, const float y, const float4* Pch, float* errval)
{
        float4 diff = read_imagef(J, sampler, (float2)(x,y))-*Pch;
        *errval += fabs(diff.x) + fabs(diff.y) + fabs(diff.z);
}


__kernel void lkSparse_C1_D5(image2d_t I, image2d_t J,
    __global const float2* prevPts, int prevPtsStep, __global float2* nextPts, int nextPtsStep, __global uchar* status, __global float* err,
        const int level, const int rows, const int cols, int PATCH_X, int PATCH_Y, int cn, int c_winSize_x, int c_winSize_y, int c_iters, char calcErr)
{
    __local float smem1[BUFFER];
    __local float smem2[BUFFER];
    __local float smem3[BUFFER];

        unsigned int xid=get_local_id(0);
        unsigned int yid=get_local_id(1);
        unsigned int gid=get_group_id(0);
        unsigned int xsize=get_local_size(0);
        unsigned int ysize=get_local_size(1);
        int xBase, yBase, i, j, k;

        float2 c_halfWin = (float2)((c_winSize_x - 1)>>1, (c_winSize_y - 1)>>1);

    const int tid = mad24(yid, xsize, xid);

    float2 prevPt = prevPts[gid] / (1 << level);

    if (prevPt.x < 0 || prevPt.x >= cols || prevPt.y < 0 || prevPt.y >= rows)
    {
        if (tid == 0 && level == 0)
        {
            status[gid] = 0;
        }

        return;
    }
    prevPt -= c_halfWin;

    // extract the patch from the first image, compute covariation matrix of derivatives

    float A11 = 0;
    float A12 = 0;
    float A22 = 0;

    float I_patch[3][3];
    float dIdx_patch[3][3];
    float dIdy_patch[3][3];

        yBase=yid;
        {
                xBase=xid;
                SetPatch(I, prevPt.x + xBase + 0.5f, prevPt.y + yBase + 0.5f,
                                        &I_patch[0][0], &dIdx_patch[0][0], &dIdy_patch[0][0],
                                        &A11, &A12, &A22);


                xBase+=xsize;
                SetPatch(I, prevPt.x + xBase + 0.5f, prevPt.y + yBase + 0.5f,
                                        &I_patch[0][1], &dIdx_patch[0][1], &dIdy_patch[0][1],
                                        &A11, &A12, &A22);

                xBase+=xsize;
                if(xBase<c_winSize_x)
                SetPatch(I, prevPt.x + xBase + 0.5f, prevPt.y + yBase + 0.5f,
                                        &I_patch[0][2], &dIdx_patch[0][2], &dIdy_patch[0][2],
                                        &A11, &A12, &A22);
        }
        yBase+=ysize;
        {
                xBase=xid;
                SetPatch(I, prevPt.x + xBase + 0.5f, prevPt.y + yBase + 0.5f,
                                        &I_patch[1][0], &dIdx_patch[1][0], &dIdy_patch[1][0],
                                        &A11, &A12, &A22);


                xBase+=xsize;
                SetPatch(I, prevPt.x + xBase + 0.5f, prevPt.y + yBase + 0.5f,
                                        &I_patch[1][1], &dIdx_patch[1][1], &dIdy_patch[1][1],
                                        &A11, &A12, &A22);

                xBase+=xsize;
                if(xBase<c_winSize_x)
                SetPatch(I, prevPt.x + xBase + 0.5f, prevPt.y + yBase + 0.5f,
                                        &I_patch[1][2], &dIdx_patch[1][2], &dIdy_patch[1][2],
                                        &A11, &A12, &A22);
        }
        yBase+=ysize;
        if(yBase<c_winSize_y)
        {
                xBase=xid;
                SetPatch(I, prevPt.x + xBase + 0.5f, prevPt.y + yBase + 0.5f,
                                        &I_patch[2][0], &dIdx_patch[2][0], &dIdy_patch[2][0],
                                        &A11, &A12, &A22);


                xBase+=xsize;
                SetPatch(I, prevPt.x + xBase + 0.5f, prevPt.y + yBase + 0.5f,
                                        &I_patch[2][1], &dIdx_patch[2][1], &dIdy_patch[2][1],
                                        &A11, &A12, &A22);

                xBase+=xsize;
                if(xBase<c_winSize_x)
                SetPatch(I, prevPt.x + xBase + 0.5f, prevPt.y + yBase + 0.5f,
                                        &I_patch[2][2], &dIdx_patch[2][2], &dIdy_patch[2][2],
                                        &A11, &A12, &A22);
        }
    reduce3(A11, A12, A22, smem1, smem2, smem3, tid);
    barrier(CLK_LOCAL_MEM_FENCE);

    A11 = smem1[0];
    A12 = smem2[0];
    A22 = smem3[0];

    float D = A11 * A22 - A12 * A12;

    if (D < 1.192092896e-07f)
    {
        if (tid == 0 && level == 0)
            status[gid] = 0;

        return;
    }

    A11 /= D;
    A12 /= D;
    A22 /= D;

    prevPt = nextPts[gid] * 2.0f - c_halfWin;

    for (k = 0; k < c_iters; ++k)
    {
        if (prevPt.x < -c_halfWin.x || prevPt.x >= cols || prevPt.y < -c_halfWin.y || prevPt.y >= rows)
        {
            if (tid == 0 && level == 0)
                status[gid] = 0;
            return;
        }

        float b1 = 0;
        float b2 = 0;

                yBase=yid;
                {
                        xBase=xid;
                        GetPatch(J, prevPt.x + xBase + 0.5f, prevPt.y + yBase + 0.5f,
                                                &I_patch[0][0], &dIdx_patch[0][0], &dIdy_patch[0][0],
                                                &b1, &b2);


                        xBase+=xsize;
                        GetPatch(J, prevPt.x + xBase + 0.5f, prevPt.y + yBase + 0.5f,
                                                &I_patch[0][1], &dIdx_patch[0][1], &dIdy_patch[0][1],
                                                &b1, &b2);

                        xBase+=xsize;
                        if(xBase<c_winSize_x)
                        GetPatch(J, prevPt.x + xBase + 0.5f, prevPt.y + yBase + 0.5f,
                                                &I_patch[0][2], &dIdx_patch[0][2], &dIdy_patch[0][2],
                                                &b1, &b2);
                }
                yBase+=ysize;
                {
                        xBase=xid;
                        GetPatch(J, prevPt.x + xBase + 0.5f, prevPt.y + yBase + 0.5f,
                                                &I_patch[1][0], &dIdx_patch[1][0], &dIdy_patch[1][0],
                                                &b1, &b2);


                        xBase+=xsize;
                        GetPatch(J, prevPt.x + xBase + 0.5f, prevPt.y + yBase + 0.5f,
                                                &I_patch[1][1], &dIdx_patch[1][1], &dIdy_patch[1][1],
                                                &b1, &b2);

                        xBase+=xsize;
                        if(xBase<c_winSize_x)
                        GetPatch(J, prevPt.x + xBase + 0.5f, prevPt.y + yBase + 0.5f,
                                                &I_patch[1][2], &dIdx_patch[1][2], &dIdy_patch[1][2],
                                                &b1, &b2);
                }
                yBase+=ysize;
                if(yBase<c_winSize_y)
                {
                        xBase=xid;
                        GetPatch(J, prevPt.x + xBase + 0.5f, prevPt.y + yBase + 0.5f,
                                                &I_patch[2][0], &dIdx_patch[2][0], &dIdy_patch[2][0],
                                                &b1, &b2);


                        xBase+=xsize;
                        GetPatch(J, prevPt.x + xBase + 0.5f, prevPt.y + yBase + 0.5f,
                                                &I_patch[2][1], &dIdx_patch[2][1], &dIdy_patch[2][1],
                                                &b1, &b2);

                        xBase+=xsize;
                        if(xBase<c_winSize_x)
                        GetPatch(J, prevPt.x + xBase + 0.5f, prevPt.y + yBase + 0.5f,
                                                &I_patch[2][2], &dIdx_patch[2][2], &dIdy_patch[2][2],
                                                &b1, &b2);
                }

        reduce2(b1, b2, smem1, smem2, tid);
        barrier(CLK_LOCAL_MEM_FENCE);

        b1 = smem1[0];
        b2 = smem2[0];

        float2 delta;
        delta.x = A12 * b2 - A22 * b1;
        delta.y = A12 * b1 - A11 * b2;

                prevPt += delta;

        if (fabs(delta.x) < THRESHOLD && fabs(delta.y) < THRESHOLD)
            break;
    }

    D = 0.0f;
    if (calcErr)
    {
                yBase=yid;
                {
                        xBase=xid;
                        GetError(J, prevPt.x + xBase + 0.5f, prevPt.y + yBase + 0.5f,
                                                &I_patch[0][0], &D);


                        xBase+=xsize;
                        GetError(J, prevPt.x + xBase + 0.5f, prevPt.y + yBase + 0.5f,
                                                &I_patch[0][1], &D);

                        xBase+=xsize;
                        if(xBase<c_winSize_x)
                        GetError(J, prevPt.x + xBase + 0.5f, prevPt.y + yBase + 0.5f,
                                                &I_patch[0][2], &D);
                }
                yBase+=ysize;
                {
                        xBase=xid;
                        GetError(J, prevPt.x + xBase + 0.5f, prevPt.y + yBase + 0.5f,
                                                &I_patch[1][0], &D);


                        xBase+=xsize;
                        GetError(J, prevPt.x + xBase + 0.5f, prevPt.y + yBase + 0.5f,
                                                &I_patch[1][1], &D);

                        xBase+=xsize;
                        if(xBase<c_winSize_x)
                        GetError(J, prevPt.x + xBase + 0.5f, prevPt.y + yBase + 0.5f,
                                                &I_patch[1][2], &D);
                }
                yBase+=ysize;
                if(yBase<c_winSize_y)
                {
                        xBase=xid;
                        GetError(J, prevPt.x + xBase + 0.5f, prevPt.y + yBase + 0.5f,
                                                &I_patch[2][0], &D);


                        xBase+=xsize;
                        GetError(J, prevPt.x + xBase + 0.5f, prevPt.y + yBase + 0.5f,
                                                &I_patch[2][1], &D);

                        xBase+=xsize;
                        if(xBase<c_winSize_x)
                        GetError(J, prevPt.x + xBase + 0.5f, prevPt.y + yBase + 0.5f,
                                                &I_patch[2][2], &D);
                }

        reduce1(D, smem1, tid);
    }

    if (tid == 0)
    {
                prevPt += c_halfWin;

        nextPts[gid] = prevPt;

        if (calcErr)
            err[gid] = smem1[0] / (c_winSize_x * c_winSize_y);
    }

}

__kernel void lkSparse_C4_D5(image2d_t I, image2d_t J,
    __global const float2* prevPts, int prevPtsStep, __global float2* nextPts, int nextPtsStep, __global uchar* status, __global float* err,
        const int level, const int rows, const int cols, int PATCH_X, int PATCH_Y, int cn, int c_winSize_x, int c_winSize_y, int c_iters, char calcErr)
{
    __local float smem1[BUFFER];
    __local float smem2[BUFFER];
    __local float smem3[BUFFER];

        unsigned int xid=get_local_id(0);
        unsigned int yid=get_local_id(1);
        unsigned int gid=get_group_id(0);
        unsigned int xsize=get_local_size(0);
        unsigned int ysize=get_local_size(1);
        int xBase, yBase, i, j, k;

        float2 c_halfWin = (float2)((c_winSize_x - 1)>>1, (c_winSize_y - 1)>>1);

    const int tid = mad24(yid, xsize, xid);

    float2 nextPt = prevPts[gid]/(1<<level);

    if (nextPt.x < 0 || nextPt.x >= cols || nextPt.y < 0 || nextPt.y >= rows)
    {
        if (tid == 0 && level == 0)
        {
            status[gid] = 0;
        }

        return;
    }

        nextPt -= c_halfWin;

    // extract the patch from the first image, compute covariation matrix of derivatives

    float A11 = 0;
    float A12 = 0;
    float A22 = 0;

    float4 I_patch[8];
    float4 dIdx_patch[8];
    float4 dIdy_patch[8];
        float4 I_add,Dx_add,Dy_add;

        yBase=yid;
        {
                xBase=xid;
                SetPatch4(I, nextPt.x + xBase + 0.5f, nextPt.y + yBase + 0.5f,
                                        &I_patch[0], &dIdx_patch[0], &dIdy_patch[0],
                                        &A11, &A12, &A22);


                xBase+=xsize;
                SetPatch4(I, nextPt.x + xBase + 0.5f, nextPt.y + yBase + 0.5f,
                                        &I_patch[1], &dIdx_patch[1], &dIdy_patch[1],
                                        &A11, &A12, &A22);

                xBase+=xsize;
                if(xBase<c_winSize_x)
                SetPatch4(I, nextPt.x + xBase + 0.5f, nextPt.y + yBase + 0.5f,
                                        &I_patch[2], &dIdx_patch[2], &dIdy_patch[2],
                                        &A11, &A12, &A22);

        }
        yBase+=ysize;
        {
                xBase=xid;
                SetPatch4(I, nextPt.x + xBase + 0.5f, nextPt.y + yBase + 0.5f,
                                        &I_patch[3], &dIdx_patch[3], &dIdy_patch[3],
                                        &A11, &A12, &A22);


                xBase+=xsize;
                SetPatch4(I, nextPt.x + xBase + 0.5f, nextPt.y + yBase + 0.5f,
                                        &I_patch[4], &dIdx_patch[4], &dIdy_patch[4],
                                        &A11, &A12, &A22);

                xBase+=xsize;
                if(xBase<c_winSize_x)
                SetPatch4(I, nextPt.x + xBase + 0.5f, nextPt.y + yBase + 0.5f,
                                        &I_patch[5], &dIdx_patch[5], &dIdy_patch[5],
                                        &A11, &A12, &A22);
        }
        yBase+=ysize;
        if(yBase<c_winSize_y)
        {
                xBase=xid;
                SetPatch4(I, nextPt.x + xBase + 0.5f, nextPt.y + yBase + 0.5f,
                                        &I_patch[6], &dIdx_patch[6], &dIdy_patch[6],
                                        &A11, &A12, &A22);


                xBase+=xsize;
                SetPatch4(I, nextPt.x + xBase + 0.5f, nextPt.y + yBase + 0.5f,
                                        &I_patch[7], &dIdx_patch[7], &dIdy_patch[7],
                                        &A11, &A12, &A22);

                xBase+=xsize;
                if(xBase<c_winSize_x)
                SetPatch4(I, nextPt.x + xBase + 0.5f, nextPt.y + yBase + 0.5f,
                                        &I_add, &Dx_add, &Dy_add,
                                        &A11, &A12, &A22);
        }

    reduce3(A11, A12, A22, smem1, smem2, smem3, tid);
    barrier(CLK_LOCAL_MEM_FENCE);

    A11 = smem1[0];
    A12 = smem2[0];
    A22 = smem3[0];

    float D = A11 * A22 - A12 * A12;

    if (D < 1.192092896e-07f)
    {
        if (tid == 0 && level == 0)
            status[gid] = 0;

        return;
    }

    A11 /= D;
    A12 /= D;
    A22 /= D;

        nextPt = nextPts[gid] * 2.0f - c_halfWin;

    for (k = 0; k < c_iters; ++k)
    {
        if (nextPt.x < -c_halfWin.x || nextPt.x >= cols || nextPt.y < -c_halfWin.y || nextPt.y >= rows)
        {
            if (tid == 0 && level == 0)
                status[gid] = 0;
            return;
        }

        float b1 = 0;
        float b2 = 0;

                yBase=yid;
                {
                        xBase=xid;
                        GetPatch4(J, nextPt.x + xBase + 0.5f, nextPt.y + yBase + 0.5f,
                                                &I_patch[0], &dIdx_patch[0], &dIdy_patch[0],
                                                &b1, &b2);


                        xBase+=xsize;
                        GetPatch4(J, nextPt.x + xBase + 0.5f, nextPt.y + yBase + 0.5f,
                                                &I_patch[1], &dIdx_patch[1], &dIdy_patch[1],
                                                &b1, &b2);

                        xBase+=xsize;
                        if(xBase<c_winSize_x)
                        GetPatch4(J, nextPt.x + xBase + 0.5f, nextPt.y + yBase + 0.5f,
                                                &I_patch[2], &dIdx_patch[2], &dIdy_patch[2],
                                                &b1, &b2);
                }
                yBase+=ysize;
                {
                        xBase=xid;
                        GetPatch4(J, nextPt.x + xBase + 0.5f, nextPt.y + yBase + 0.5f,
                                                &I_patch[3], &dIdx_patch[3], &dIdy_patch[3],
                                                &b1, &b2);


                        xBase+=xsize;
                        GetPatch4(J, nextPt.x + xBase + 0.5f, nextPt.y + yBase + 0.5f,
                                                &I_patch[4], &dIdx_patch[4], &dIdy_patch[4],
                                                &b1, &b2);

                        xBase+=xsize;
                        if(xBase<c_winSize_x)
                        GetPatch4(J, nextPt.x + xBase + 0.5f, nextPt.y + yBase + 0.5f,
                                                &I_patch[5], &dIdx_patch[5], &dIdy_patch[5],
                                                &b1, &b2);
                }
                yBase+=ysize;
                if(yBase<c_winSize_y)
                {
                        xBase=xid;
                        GetPatch4(J, nextPt.x + xBase + 0.5f, nextPt.y + yBase + 0.5f,
                                                &I_patch[6], &dIdx_patch[6], &dIdy_patch[6],
                                                &b1, &b2);


                        xBase+=xsize;
                        GetPatch4(J, nextPt.x + xBase + 0.5f, nextPt.y + yBase + 0.5f,
                                                &I_patch[7], &dIdx_patch[7], &dIdy_patch[7],
                                                &b1, &b2);

                        xBase+=xsize;
                        if(xBase<c_winSize_x)
                        GetPatch4(J, nextPt.x + xBase + 0.5f, nextPt.y + yBase + 0.5f,
                                                &I_add, &Dx_add, &Dy_add,
                                                &b1, &b2);
                }


        reduce2(b1, b2, smem1, smem2, tid);
        barrier(CLK_LOCAL_MEM_FENCE);

        b1 = smem1[0];
        b2 = smem2[0];

        float2 delta;
        delta.x = A12 * b2 - A22 * b1;
        delta.y = A12 * b1 - A11 * b2;

                nextPt +=delta;

        if (fabs(delta.x) < THRESHOLD && fabs(delta.y) < THRESHOLD)
            break;
    }

    D = 0.0f;
    if (calcErr)
    {
                yBase=yid;
                {
                        xBase=xid;
                        GetError4(J, nextPt.x + xBase + 0.5f, nextPt.y + yBase + 0.5f,
                                                &I_patch[0], &D);


                        xBase+=xsize;
                        GetError4(J, nextPt.x + xBase + 0.5f, nextPt.y + yBase + 0.5f,
                                                &I_patch[1], &D);

                        xBase+=xsize;
                        if(xBase<c_winSize_x)
                        GetError4(J, nextPt.x + xBase + 0.5f, nextPt.y + yBase + 0.5f,
                                                &I_patch[2], &D);
                }
                yBase+=ysize;
                {
                        xBase=xid;
                        GetError4(J, nextPt.x + xBase + 0.5f, nextPt.y + yBase + 0.5f,
                                                &I_patch[3], &D);


                        xBase+=xsize;
                        GetError4(J, nextPt.x + xBase + 0.5f, nextPt.y + yBase + 0.5f,
                                                &I_patch[4], &D);

                        xBase+=xsize;
                        if(xBase<c_winSize_x)
                        GetError4(J, nextPt.x + xBase + 0.5f, nextPt.y + yBase + 0.5f,
                                                &I_patch[5], &D);
                }
                yBase+=ysize;
                if(yBase<c_winSize_y)
                {
                        xBase=xid;
                        GetError4(J, nextPt.x + xBase + 0.5f, nextPt.y + yBase + 0.5f,
                                                &I_patch[6], &D);


                        xBase+=xsize;
                        GetError4(J, nextPt.x + xBase + 0.5f, nextPt.y + yBase + 0.5f,
                                                &I_patch[7], &D);

                        xBase+=xsize;
                        if(xBase<c_winSize_x)
                        GetError4(J, nextPt.x + xBase + 0.5f, nextPt.y + yBase + 0.5f,
                                                &I_add, &D);
                }

        reduce1(D, smem1, tid);
    }

    if (tid == 0)
    {
                nextPt += c_halfWin;
        nextPts[gid] = nextPt;

        if (calcErr)
            err[gid] = smem1[0] / (3 * c_winSize_x * c_winSize_y);
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
