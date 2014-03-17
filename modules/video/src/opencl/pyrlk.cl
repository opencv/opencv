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
//    Xiaopeng Fu, fuxiaopeng2222@163.com
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

#define	BUFFER	64
#define	BUFFER2	BUFFER>>1
#ifndef WAVE_SIZE
#define WAVE_SIZE 1
#endif
#ifdef CPU

inline void reduce3(float val1, float val2, float val3,  __local float* smem1,  __local float* smem2,  __local float* smem3, int tid)
{
    smem1[tid] = val1;
    smem2[tid] = val2;
    smem3[tid] = val3;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = BUFFER2; i > 0; i >>= 1)
    {
        if(tid < i)
        {
            smem1[tid] += smem1[tid + i];
            smem2[tid] += smem2[tid + i];
            smem3[tid] += smem3[tid + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

inline void reduce2(float val1, float val2, volatile __local float* smem1, volatile __local float* smem2, int tid)
{
    smem1[tid] = val1;
    smem2[tid] = val2;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = BUFFER2; i > 0; i >>= 1)
    {
        if(tid < i)
        {
            smem1[tid] += smem1[tid + i];
            smem2[tid] += smem2[tid + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

inline void reduce1(float val1, volatile __local float* smem1, int tid)
{
    smem1[tid] = val1;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = BUFFER2; i > 0; i >>= 1)
    {
        if(tid < i)
        {
            smem1[tid] += smem1[tid + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
#else
inline void reduce3(float val1, float val2, float val3,
             __local volatile float* smem1, __local volatile float* smem2, __local volatile float* smem3, int tid)
{
    smem1[tid] = val1;
    smem2[tid] = val2;
    smem3[tid] = val3;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 32)
    {
        smem1[tid] += smem1[tid + 32];
        smem2[tid] += smem2[tid + 32];
        smem3[tid] += smem3[tid + 32];
#if WAVE_SIZE < 32
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 16)
    {
#endif
        smem1[tid] += smem1[tid + 16];
        smem2[tid] += smem2[tid + 16];
        smem3[tid] += smem3[tid + 16];
#if WAVE_SIZE <16
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 8)
    {
#endif
        smem1[tid] += smem1[tid + 8];
        smem2[tid] += smem2[tid + 8];
        smem3[tid] += smem3[tid + 8];

        smem1[tid] += smem1[tid + 4];
        smem2[tid] += smem2[tid + 4];
        smem3[tid] += smem3[tid + 4];

        smem1[tid] += smem1[tid + 2];
        smem2[tid] += smem2[tid + 2];
        smem3[tid] += smem3[tid + 2];

        smem1[tid] += smem1[tid + 1];
        smem2[tid] += smem2[tid + 1];
        smem3[tid] += smem3[tid + 1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

inline void reduce2(float val1, float val2, __local volatile float* smem1, __local volatile float* smem2, int tid)
{
    smem1[tid] = val1;
    smem2[tid] = val2;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 32)
    {
        smem1[tid] += smem1[tid + 32];
        smem2[tid] += smem2[tid + 32];
#if WAVE_SIZE < 32
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 16)
    {
#endif
        smem1[tid] += smem1[tid + 16];
        smem2[tid] += smem2[tid + 16];
#if WAVE_SIZE <16
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 8)
    {
#endif
        smem1[tid] += smem1[tid + 8];
        smem2[tid] += smem2[tid + 8];

        smem1[tid] += smem1[tid + 4];
        smem2[tid] += smem2[tid + 4];

        smem1[tid] += smem1[tid + 2];
        smem2[tid] += smem2[tid + 2];

        smem1[tid] += smem1[tid + 1];
        smem2[tid] += smem2[tid + 1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

inline void reduce1(float val1, __local volatile float* smem1, int tid)
{
    smem1[tid] = val1;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 32)
    {
        smem1[tid] += smem1[tid + 32];
#if WAVE_SIZE < 32
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 16)
    {
#endif
        smem1[tid] += smem1[tid + 16];
#if WAVE_SIZE <16
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 8)
    {
#endif
        smem1[tid] += smem1[tid + 8];
        smem1[tid] += smem1[tid + 4];
        smem1[tid] += smem1[tid + 2];
        smem1[tid] += smem1[tid + 1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}
#endif

#define SCALE (1.0f / (1 << 20))
#define	THRESHOLD	0.01f

// Image read mode
__constant sampler_t sampler    = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

inline void SetPatch(image2d_t I, float x, float y,
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

inline void GetPatch(image2d_t J, float x, float y,
              float* Pch, float* Dx, float* Dy,
              float* b1, float* b2)
{
    float J_val = read_imagef(J, sampler, (float2)(x, y)).x;
    float diff = (J_val - *Pch) * 32.0f;
    *b1 += diff**Dx;
    *b2 += diff**Dy;
}

inline void GetError(image2d_t J, const float x, const float y, const float* Pch, float* errval)
{
    float diff = read_imagef(J, sampler, (float2)(x,y)).x-*Pch;
    *errval += fabs(diff);
}

#define	GRIDSIZE	3
__kernel void lkSparse(image2d_t I, image2d_t J,
                       __global const float2* prevPts, __global float2* nextPts, __global uchar* status, __global float* err,
                       const int level, const int rows, const int cols, int PATCH_X, int PATCH_Y, int c_winSize_x, int c_winSize_y, int c_iters, char calcErr)
{
    __local float smem1[BUFFER];
    __local float smem2[BUFFER];
    __local float smem3[BUFFER];

    unsigned int xid=get_local_id(0);
    unsigned int yid=get_local_id(1);
    unsigned int gid=get_group_id(0);
    unsigned int xsize=get_local_size(0);
    unsigned int ysize=get_local_size(1);
    int xBase, yBase, k;

    float2 c_halfWin = (float2)((c_winSize_x - 1)>>1, (c_winSize_y - 1)>>1);

    const int tid = mad24(yid, xsize, xid);

    float2 prevPt = prevPts[gid] / (float2)(1 << level);

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

    float I_patch[GRIDSIZE][GRIDSIZE];
    float dIdx_patch[GRIDSIZE][GRIDSIZE];
    float dIdy_patch[GRIDSIZE][GRIDSIZE];

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

    A11 = smem1[0];
    A12 = smem2[0];
    A22 = smem3[0];
    barrier(CLK_LOCAL_MEM_FENCE);

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
            break;
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

        b1 = smem1[0];
        b2 = smem2[0];
        barrier(CLK_LOCAL_MEM_FENCE);

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
            err[gid] = smem1[0] / (float)(c_winSize_x * c_winSize_y);
    }
}