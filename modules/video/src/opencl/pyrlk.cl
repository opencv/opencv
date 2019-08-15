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

#define GRIDSIZE    3
#define LSx 8
#define LSy 8
// define local memory sizes
#define LM_W (LSx*GRIDSIZE+2)
#define LM_H (LSy*GRIDSIZE+2)
#define BUFFER  (LSx*LSy)
#define BUFFER2 BUFFER>>1

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

inline void reduce2(float val1, float val2, __local float* smem1, __local float* smem2, int tid)
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

inline void reduce1(float val1, __local float* smem1, int tid)
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
             __local float* smem1, __local float* smem2, __local float* smem3, int tid)
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
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 16)
    {
        smem1[tid] += smem1[tid + 16];
        smem2[tid] += smem2[tid + 16];
        smem3[tid] += smem3[tid + 16];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 8)
    {
        smem1[tid] += smem1[tid + 8];
        smem2[tid] += smem2[tid + 8];
        smem3[tid] += smem3[tid + 8];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 4)
    {
        smem1[tid] += smem1[tid + 4];
        smem2[tid] += smem2[tid + 4];
        smem3[tid] += smem3[tid + 4];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid == 0)
    {
        smem1[0] = (smem1[0] + smem1[1]) + (smem1[2] + smem1[3]);
        smem2[0] = (smem2[0] + smem2[1]) + (smem2[2] + smem2[3]);
        smem3[0] = (smem3[0] + smem3[1]) + (smem3[2] + smem3[3]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

inline void reduce2(float val1, float val2, __local float* smem1, __local float* smem2, int tid)
{
    smem1[tid] = val1;
    smem2[tid] = val2;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 32)
    {
        smem1[tid] += smem1[tid + 32];
        smem2[tid] += smem2[tid + 32];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 16)
    {
        smem1[tid] += smem1[tid + 16];
        smem2[tid] += smem2[tid + 16];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 8)
    {
        smem1[tid] += smem1[tid + 8];
        smem2[tid] += smem2[tid + 8];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 4)
    {
        smem1[tid] += smem1[tid + 4];
        smem2[tid] += smem2[tid + 4];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid == 0)
    {
        smem1[0] = (smem1[0] + smem1[1]) + (smem1[2] + smem1[3]);
        smem2[0] = (smem2[0] + smem2[1]) + (smem2[2] + smem2[3]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

inline void reduce1(float val1, __local float* smem1, int tid)
{
    smem1[tid] = val1;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 32)
    {
        smem1[tid] += smem1[tid + 32];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 16)
    {
        smem1[tid] += smem1[tid + 16];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 8)
    {
        smem1[tid] += smem1[tid + 8];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 4)
    {
        smem1[tid] += smem1[tid + 4];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid == 0)
    {
        smem1[0] = (smem1[0] + smem1[1]) + (smem1[2] + smem1[3]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}
#endif

#define SCALE (1.0f / (1 << 20))
#define	THRESHOLD	0.01f

// Image read mode
__constant sampler_t sampler    = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

// macro to get pixel value from local memory

#define VAL(_y,_x,_yy,_xx)    (IPatchLocal[mad24(((_y) + (_yy)), LM_W, ((_x) + (_xx)))])
inline void SetPatch(local float* IPatchLocal, int TileY, int TileX,
              float* Pch, float* Dx, float* Dy,
              float* A11, float* A12, float* A22, float w)
{
    int xid=get_local_id(0);
    int yid=get_local_id(1);
    int xBase = mad24(TileX, LSx, (xid + 1));
    int yBase = mad24(TileY, LSy, (yid + 1));

    *Pch = VAL(yBase,xBase,0,0);

    *Dx = mad((VAL(yBase,xBase,-1,1) + VAL(yBase,xBase,+1,1) - VAL(yBase,xBase,-1,-1) - VAL(yBase,xBase,+1,-1)), 3.0f, (VAL(yBase,xBase,0,1) - VAL(yBase,xBase,0,-1)) * 10.0f) * w;
    *Dy = mad((VAL(yBase,xBase,1,-1) + VAL(yBase,xBase,1,+1) - VAL(yBase,xBase,-1,-1) - VAL(yBase,xBase,-1,+1)), 3.0f, (VAL(yBase,xBase,1,0) - VAL(yBase,xBase,-1,0)) * 10.0f) * w;

    *A11 = mad(*Dx, *Dx, *A11);
    *A12 = mad(*Dx, *Dy, *A12);
    *A22 = mad(*Dy, *Dy, *A22);
}
#undef VAL

inline void GetPatch(image2d_t J, float x, float y,
              float* Pch, float* Dx, float* Dy,
              float* b1, float* b2)
{
    float diff = read_imagef(J, sampler, (float2)(x,y)).x-*Pch;
    *b1 = mad(diff, *Dx, *b1);
    *b2 = mad(diff, *Dy, *b2);
}

inline void GetError(image2d_t J, const float x, const float y, const float* Pch, float* errval, float w)
{
    float diff = ((((read_imagef(J, sampler, (float2)(x,y)).x * 16384) + 256) / 512) - (((*Pch * 16384) + 256) /512)) * w;
    *errval += fabs(diff);
}


//macro to read pixel value into local memory.
#define READI(_y,_x) IPatchLocal[mad24(mad24((_y), LSy, yid), LM_W, mad24((_x), LSx, xid))] = read_imagef(I, sampler, (float2)(mad((float)(_x), (float)LSx, Point.x + xid - 0.5f), mad((float)(_y), (float)LSy, Point.y + yid - 0.5f))).x;
void ReadPatchIToLocalMem(image2d_t I, float2 Point, local float* IPatchLocal)
{
    int xid=get_local_id(0);
    int yid=get_local_id(1);
    //read (3*LSx)*(3*LSy) window. each macro call read LSx*LSy pixels block
    READI(0,0);READI(0,1);READI(0,2);
    READI(1,0);READI(1,1);READI(1,2);
    READI(2,0);READI(2,1);READI(2,2);
    if(xid<2)
    {// read last 2 columns border. each macro call reads 2*LSy pixels block
        READI(0,3);
        READI(1,3);
        READI(2,3);
    }

    if(yid<2)
    {// read last 2 row. each macro call reads LSx*2 pixels block
        READI(3,0);READI(3,1);READI(3,2);
    }

    if(yid<2 && xid<2)
    {// read right bottom 2x2 corner. one macro call reads 2*2 pixels block
        READI(3,3);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}
#undef READI

__attribute__((reqd_work_group_size(LSx, LSy, 1)))
__kernel void lkSparse(image2d_t I, image2d_t J,
                       __global const float2* prevPts, __global float2* nextPts, __global uchar* status, __global float* err,
                       const int level, const int rows, const int cols, int PATCH_X, int PATCH_Y, int c_winSize_x, int c_winSize_y, int c_iters, char calcErr)
{
    __local float smem1[BUFFER];
    __local float smem2[BUFFER];
    __local float smem3[BUFFER];

    int xid=get_local_id(0);
    int yid=get_local_id(1);
    int gid=get_group_id(0);
    int xsize=get_local_size(0);
    int ysize=get_local_size(1);
    int k;

#ifdef CPU
    float wx0 = 1.0f;
    float wy0 = 1.0f;
    int xBase = mad24(xsize, 2, xid);
    int yBase = mad24(ysize, 2, yid);
    float wx1 = (xBase < c_winSize_x) ? 1 : 0;
    float wy1 = (yBase < c_winSize_y) ? 1 : 0;
#else
#if WSX == 1
    float wx0 = 1.0f;
    int xBase = mad24(xsize, 2, xid);
    float wx1 = (xBase < c_winSize_x) ? 1 : 0;
#else
    int xBase = mad24(xsize, 1, xid);
    float wx0 = (xBase < c_winSize_x) ? 1 : 0;
    float wx1 = 0.0f;
#endif
#if WSY == 1
    float wy0 = 1.0f;
    int yBase = mad24(ysize, 2, yid);
    float wy1 = (yBase < c_winSize_y) ? 1 : 0;
#else
    int yBase = mad24(ysize, 1, yid);
    float wy0 = (yBase < c_winSize_y) ? 1 : 0;
    float wy1 = 0.0f;
#endif
#endif

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

    // local memory to read image with border to calc sobels
    local float IPatchLocal[LM_W*LM_H];
    ReadPatchIToLocalMem(I,prevPt,IPatchLocal);

    {
        SetPatch(IPatchLocal, 0, 0,
                 &I_patch[0][0], &dIdx_patch[0][0], &dIdy_patch[0][0],
                 &A11, &A12, &A22,1);


        SetPatch(IPatchLocal, 0, 1,
                 &I_patch[0][1], &dIdx_patch[0][1], &dIdy_patch[0][1],
                 &A11, &A12, &A22,wx0);

        SetPatch(IPatchLocal, 0, 2,
                    &I_patch[0][2], &dIdx_patch[0][2], &dIdy_patch[0][2],
                    &A11, &A12, &A22,wx1);
    }
    {
        SetPatch(IPatchLocal, 1, 0,
                 &I_patch[1][0], &dIdx_patch[1][0], &dIdy_patch[1][0],
                 &A11, &A12, &A22,wy0);


        SetPatch(IPatchLocal, 1,1,
                 &I_patch[1][1], &dIdx_patch[1][1], &dIdy_patch[1][1],
                 &A11, &A12, &A22,wx0*wy0);

        SetPatch(IPatchLocal, 1,2,
                    &I_patch[1][2], &dIdx_patch[1][2], &dIdy_patch[1][2],
                    &A11, &A12, &A22,wx1*wy0);
    }
    {
        SetPatch(IPatchLocal, 2,0,
                 &I_patch[2][0], &dIdx_patch[2][0], &dIdy_patch[2][0],
                 &A11, &A12, &A22,wy1);


        SetPatch(IPatchLocal, 2,1,
                 &I_patch[2][1], &dIdx_patch[2][1], &dIdy_patch[2][1],
                 &A11, &A12, &A22,wx0*wy1);

        SetPatch(IPatchLocal, 2,2,
                    &I_patch[2][2], &dIdx_patch[2][2], &dIdy_patch[2][2],
                    &A11, &A12, &A22,wx1*wy1);
    }


    reduce3(A11, A12, A22, smem1, smem2, smem3, tid);

    A11 = smem1[0];
    A12 = smem2[0];
    A22 = smem3[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    float D = mad(A11, A22, - A12 * A12);

    if (D < 1.192092896e-07f)
    {
        if (tid == 0 && level == 0)
            status[gid] = 0;

        return;
    }

    A11 /= D;
    A12 /= D;
    A22 /= D;

    prevPt = mad(nextPts[gid], 2.0f, - c_halfWin);

    float2 offset0 = (float2)(xid + 0.5f, yid + 0.5f);
    float2 offset1 = (float2)(xsize, ysize);
    float2 loc0 = prevPt + offset0;
    float2 loc1 = loc0 + offset1;
    float2 loc2 = loc1 + offset1;

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

        {
            GetPatch(J, loc0.x, loc0.y,
                     &I_patch[0][0], &dIdx_patch[0][0], &dIdy_patch[0][0],
                     &b1, &b2);


            GetPatch(J, loc1.x, loc0.y,
                     &I_patch[0][1], &dIdx_patch[0][1], &dIdy_patch[0][1],
                     &b1, &b2);

            GetPatch(J, loc2.x, loc0.y,
                        &I_patch[0][2], &dIdx_patch[0][2], &dIdy_patch[0][2],
                        &b1, &b2);
        }
        {
            GetPatch(J, loc0.x, loc1.y,
                     &I_patch[1][0], &dIdx_patch[1][0], &dIdy_patch[1][0],
                     &b1, &b2);


            GetPatch(J, loc1.x, loc1.y,
                     &I_patch[1][1], &dIdx_patch[1][1], &dIdy_patch[1][1],
                     &b1, &b2);

            GetPatch(J, loc2.x, loc1.y,
                        &I_patch[1][2], &dIdx_patch[1][2], &dIdy_patch[1][2],
                        &b1, &b2);
        }
        {
            GetPatch(J, loc0.x, loc2.y,
                     &I_patch[2][0], &dIdx_patch[2][0], &dIdy_patch[2][0],
                     &b1, &b2);


            GetPatch(J, loc1.x, loc2.y,
                     &I_patch[2][1], &dIdx_patch[2][1], &dIdy_patch[2][1],
                     &b1, &b2);

            GetPatch(J, loc2.x, loc2.y,
                        &I_patch[2][2], &dIdx_patch[2][2], &dIdy_patch[2][2],
                        &b1, &b2);
        }

        reduce2(b1, b2, smem1, smem2, tid);

        b1 = smem1[0];
        b2 = smem2[0];
        barrier(CLK_LOCAL_MEM_FENCE);

        float2 delta;
        delta.x = mad(A12, b2, - A22 * b1) * 32.0f;
        delta.y = mad(A12, b1, - A11 * b2) * 32.0f;

        prevPt += delta;
        loc0 += delta;
        loc1 += delta;
        loc2 += delta;

        if (fabs(delta.x) < THRESHOLD && fabs(delta.y) < THRESHOLD)
            break;
    }

    D = 0.0f;
    if (calcErr)
    {
        {
            GetError(J, loc0.x, loc0.y, &I_patch[0][0], &D, 1);
            GetError(J, loc1.x, loc0.y, &I_patch[0][1], &D, wx0);
        }
        {
            GetError(J, loc0.x, loc1.y, &I_patch[1][0], &D, wy0);
            GetError(J, loc1.x, loc1.y, &I_patch[1][1], &D, wx0*wy0);
        }
        if(xBase < c_winSize_x)
        {
            GetError(J, loc2.x, loc0.y, &I_patch[0][2], &D, wx1);
            GetError(J, loc2.x, loc1.y, &I_patch[1][2], &D, wx1*wy0);
        }
        if(yBase < c_winSize_y)
        {
            GetError(J, loc0.x, loc2.y, &I_patch[2][0], &D, wy1);
            GetError(J, loc1.x, loc2.y, &I_patch[2][1], &D, wx0*wy1);
            if(xBase < c_winSize_x)
                GetError(J, loc2.x, loc2.y, &I_patch[2][2], &D, wx1*wy1);
        }

        reduce1(D, smem1, tid);
    }

    if (tid == 0)
    {
        prevPt += c_halfWin;

        nextPts[gid] = prevPt;

        if (calcErr)
            err[gid] = smem1[0] / (float)(32 * c_winSize_x * c_winSize_y);
    }
}
