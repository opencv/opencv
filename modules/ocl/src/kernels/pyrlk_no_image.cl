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
//    Sen Liu, sen@multicorewareinc.com
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

#define	BUFFER	256
void reduce3(float val1, float val2, float val3, __local float *smem1, __local float *smem2, __local float *smem3, int tid)
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
                smem1[tid] = val1 += smem1[tid + 32];
                smem2[tid] = val2 += smem2[tid + 32];
                smem3[tid] = val3 += smem3[tid + 32];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (tid < 16)
        {
                smem1[tid] = val1 += smem1[tid + 16];
                smem2[tid] = val2 += smem2[tid + 16];
                smem3[tid] = val3 += smem3[tid + 16];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (tid < 8)
        {
                volatile __local float *vmem1 = smem1;
                volatile __local float *vmem2 = smem2;
                volatile __local float *vmem3 = smem3;

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

void reduce2(float val1, float val2, __local float *smem1, __local float *smem2, int tid)
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
                smem1[tid] = val1 += smem1[tid + 32];
                smem2[tid] = val2 += smem2[tid + 32];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (tid < 16)
        {
                smem1[tid] = val1 += smem1[tid + 16];
                smem2[tid] = val2 += smem2[tid + 16];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (tid < 8)
        {
                volatile __local float *vmem1 = smem1;
                volatile __local float *vmem2 = smem2;

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

void reduce1(float val1, __local float *smem1, int tid)
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
                smem1[tid] = val1 += smem1[tid + 32];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (tid < 16)
        {
                volatile __local float *vmem1 = smem1;

                vmem1[tid] = val1 += vmem1[tid + 16];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (tid < 8)
        {
                volatile __local float *vmem1 = smem1;

                vmem1[tid] = val1 += vmem1[tid + 8];
                vmem1[tid] = val1 += vmem1[tid + 4];
                vmem1[tid] = val1 += vmem1[tid + 2];
                vmem1[tid] = val1 += vmem1[tid + 1];
        }
}

#define SCALE (1.0f / (1 << 20))
#define	THRESHOLD	0.01f
#define	DIMENSION	21

float readImage2Df_C1(__global const float *image,  const float x,  const float y,  const int rows,  const int cols, const int elemCntPerRow)
{
        float2 coor = (float2)(x, y);

        int i0 = clamp((int)floor(coor.x), 0, cols - 1);
        int j0 = clamp((int)floor(coor.y), 0, rows - 1);
        int i1 = clamp((int)floor(coor.x) + 1, 0, cols - 1);
        int j1 = clamp((int)floor(coor.y) + 1, 0, rows - 1);
        float a = coor.x - floor(coor.x);
        float b = coor.y - floor(coor.y);

        return (1 - a) * (1 - b) * image[mad24(j0, elemCntPerRow, i0)]
               + a * (1 - b) * image[mad24(j0, elemCntPerRow, i1)]
               + (1 - a) * b * image[mad24(j1, elemCntPerRow, i0)]
               + a * b * image[mad24(j1, elemCntPerRow, i1)];
}

__kernel void lkSparse_C1_D5(__global const float *I, __global const float *J,
                             __global const float2 *prevPts, int prevPtsStep, __global float2 *nextPts, int nextPtsStep, __global uchar *status, __global float *err,
                             const int level, const int rows, const int cols, const int elemCntPerRow,
                             int PATCH_X, int PATCH_Y, int cn, int c_winSize_x, int c_winSize_y, int c_iters, char calcErr)
{
        __local float smem1[BUFFER];
        __local float smem2[BUFFER];
        __local float smem3[BUFFER];

        float2 c_halfWin = (float2)((c_winSize_x - 1) >> 1, (c_winSize_y - 1) >> 1);

        const int tid = mad24(get_local_id(1), get_local_size(0), get_local_id(0));

        float2 prevPt = prevPts[get_group_id(0)] * (1.0f / (1 << level));

        if (prevPt.x < 0 || prevPt.x >= cols || prevPt.y < 0 || prevPt.y >= rows)
        {
                if (tid == 0 && level == 0)
                {
                        status[get_group_id(0)] = 0;
                }

                return;
        }

        prevPt -= c_halfWin;

        // extract the patch from the first image, compute covariation matrix of derivatives

        float A11 = 0;
        float A12 = 0;
        float A22 = 0;

        float I_patch[1][3];
        float dIdx_patch[1][3];
        float dIdy_patch[1][3];

        for (int yBase = get_local_id(1), i = 0; yBase < c_winSize_y; yBase += get_local_size(1), ++i)
        {
                for (int xBase = get_local_id(0), j = 0; xBase < c_winSize_x; xBase += get_local_size(0), ++j)
                {
                        float x = (prevPt.x + xBase);
                        float y = (prevPt.y + yBase);

                        I_patch[i][j] = readImage2Df_C1(I, x, y, rows, cols, elemCntPerRow);
                        float dIdx = 3.0f * readImage2Df_C1(I, x + 1, y - 1, rows, cols, elemCntPerRow) + 10.0f * readImage2Df_C1(I, x + 1, y, rows, cols, elemCntPerRow) + 3.0f * readImage2Df_C1(I, x + 1, y + 1, rows, cols, elemCntPerRow) -
                                     (3.0f * readImage2Df_C1(I, x - 1, y - 1, rows, cols, elemCntPerRow) + 10.0f * readImage2Df_C1(I, x - 1, y, rows, cols, elemCntPerRow) + 3.0f * readImage2Df_C1(I, x - 1, y + 1, rows, cols, elemCntPerRow));

                        float dIdy = 3.0f * readImage2Df_C1(I, x - 1, y + 1, rows, cols, elemCntPerRow) + 10.0f * readImage2Df_C1(I, x, y + 1, rows, cols, elemCntPerRow) + 3.0f * readImage2Df_C1(I, x + 1, y + 1, rows, cols, elemCntPerRow) -
                                     (3.0f * readImage2Df_C1(I, x - 1, y - 1, rows, cols, elemCntPerRow) + 10.0f * readImage2Df_C1(I, x, y - 1, rows, cols, elemCntPerRow) + 3.0f * readImage2Df_C1(I, x + 1, y - 1, rows, cols, elemCntPerRow));

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

        if (D < 1.192092896e-07f)
        {
                if (tid == 0 && level == 0)
                {
                        status[get_group_id(0)] = 0;
                }

                return;
        }

        D = 1.f / D;

        A11 *= D;
        A12 *= D;
        A22 *= D;

        float2 nextPt = nextPts[get_group_id(0)];
        nextPt = nextPt * 2.0f - c_halfWin;

        for (int k = 0; k < c_iters; ++k)
        {
                if (nextPt.x < -c_halfWin.x || nextPt.x >= cols || nextPt.y < -c_halfWin.y || nextPt.y >= rows)
                {
                        if (tid == 0 && level == 0)
                        {
                                status[get_group_id(0)] = 0;
                        }

                        return;
                }

                float b1 = 0;
                float b2 = 0;

                for (int y = get_local_id(1), i = 0; y < c_winSize_y; y += get_local_size(1), ++i)
                {
                        for (int x = get_local_id(0), j = 0; x < c_winSize_x; x += get_local_size(0), ++j)
                        {
                                float diff = (readImage2Df_C1(J, nextPt.x + x, nextPt.y + y, rows, cols, elemCntPerRow) - I_patch[i][j]) * 32.0f;

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

                nextPt += delta;

                //if (fabs(delta.x) < THRESHOLD && fabs(delta.y) < THRESHOLD)
                //    break;
        }

        float errval = 0.0f;

        if (calcErr)
        {
                for (int y = get_local_id(1), i = 0; y < c_winSize_y; y += get_local_size(1), ++i)
                {
                        for (int x = get_local_id(0), j = 0; x < c_winSize_x; x += get_local_size(0), ++j)
                        {
                                float diff = readImage2Df_C1(J, nextPt.x + x, nextPt.y + y, rows, cols, elemCntPerRow) - I_patch[i][j];

                                errval += fabs(diff);
                        }
                }

                reduce1(errval, smem1, tid);
        }

        if (tid == 0)
        {
                nextPt += c_halfWin;

                nextPts[get_group_id(0)] = nextPt;

                if (calcErr)
                {
                        err[get_group_id(0)] = smem1[0] / (c_winSize_x * c_winSize_y);
                }
        }
}

float4 readImage2Df_C4(__global const float4 *image,  const float x,  const float y,  const int rows,  const int cols, const int elemCntPerRow)
{
        float2 coor = (float2)(x, y);

        int i0 = clamp((int)floor(coor.x), 0, cols - 1);
        int j0 = clamp((int)floor(coor.y), 0, rows - 1);
        int i1 = clamp((int)floor(coor.x) + 1, 0, cols - 1);
        int j1 = clamp((int)floor(coor.y) + 1, 0, rows - 1);
        float a = coor.x - floor(coor.x);
        float b = coor.y - floor(coor.y);

        return (1 - a) * (1 - b) * image[mad24(j0, elemCntPerRow, i0)]
               + a * (1 - b) * image[mad24(j0, elemCntPerRow, i1)]
               + (1 - a) * b * image[mad24(j1, elemCntPerRow, i0)]
               + a * b * image[mad24(j1, elemCntPerRow, i1)];
}

__kernel void lkSparse_C4_D5(__global const float *I, __global const float *J,
                             __global const float2 *prevPts, int prevPtsStep, __global float2 *nextPts, int nextPtsStep, __global uchar *status, __global float *err,
                             const int level, const int rows, const int cols, const int elemCntPerRow,
                             int PATCH_X, int PATCH_Y, int cn, int c_winSize_x, int c_winSize_y, int c_iters, char calcErr)
{
        __local float smem1[BUFFER];
        __local float smem2[BUFFER];
        __local float smem3[BUFFER];

        float2 c_halfWin = (float2)((c_winSize_x - 1) >> 1, (c_winSize_y - 1) >> 1);

        const int tid = mad24(get_local_id(1), get_local_size(0), get_local_id(0));

        float2 prevPt = prevPts[get_group_id(0)] * (1.0f / (1 << level));

        if (prevPt.x < 0 || prevPt.x >= cols || prevPt.y < 0 || prevPt.y >= rows)
        {
                if (tid == 0 && level == 0)
                {
                        status[get_group_id(0)] = 0;
                }

                return;
        }

        prevPt -= c_halfWin;

        // extract the patch from the first image, compute covariation matrix of derivatives

        float A11 = 0;
        float A12 = 0;
        float A22 = 0;

        float4 I_patch[1][3];
        float4 dIdx_patch[1][3];
        float4 dIdy_patch[1][3];

        __global float4 *ptrI = (__global float4 *)I;

        for (int yBase = get_local_id(1), i = 0; yBase < c_winSize_y; yBase += get_local_size(1), ++i)
        {
                for (int xBase = get_local_id(0), j = 0; xBase < c_winSize_x; xBase += get_local_size(0), ++j)
                {
                        float x = (prevPt.x + xBase);
                        float y = (prevPt.y + yBase);

                        I_patch[i][j] = readImage2Df_C4(ptrI, x, y, rows, cols, elemCntPerRow);

                        float4 dIdx = 3.0f * readImage2Df_C4(ptrI, x + 1, y - 1, rows, cols, elemCntPerRow) + 10.0f * readImage2Df_C4(ptrI, x + 1, y, rows, cols, elemCntPerRow) + 3.0f * readImage2Df_C4(ptrI, x + 1, y + 1, rows, cols, elemCntPerRow) -
                                      (3.0f * readImage2Df_C4(ptrI, x - 1, y - 1, rows, cols, elemCntPerRow) + 10.0f * readImage2Df_C4(ptrI, x - 1, y, rows, cols, elemCntPerRow) + 3.0f * readImage2Df_C4(ptrI, x - 1, y + 1, rows, cols, elemCntPerRow));

                        float4 dIdy = 3.0f * readImage2Df_C4(ptrI, x - 1, y + 1, rows, cols, elemCntPerRow) + 10.0f * readImage2Df_C4(ptrI, x, y + 1, rows, cols, elemCntPerRow) + 3.0f * readImage2Df_C4(ptrI, x + 1, y + 1, rows, cols, elemCntPerRow) -
                                      (3.0f * readImage2Df_C4(ptrI, x - 1, y - 1, rows, cols, elemCntPerRow) + 10.0f * readImage2Df_C4(ptrI, x, y - 1, rows, cols, elemCntPerRow) + 3.0f * readImage2Df_C4(ptrI, x + 1, y - 1, rows, cols, elemCntPerRow));

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
        //pD[get_group_id(0)] = D;

        if (D < 1.192092896e-07f)
        {
                if (tid == 0 && level == 0)
                {
                        status[get_group_id(0)] = 0;
                }

                return;
        }

        D = 1.f / D;

        A11 *= D;
        A12 *= D;
        A22 *= D;

        float2 nextPt = nextPts[get_group_id(0)];

        nextPt = nextPt * 2.0f - c_halfWin;

        __global float4 *ptrJ = (__global float4 *)J;

        for (int k = 0; k < c_iters; ++k)
        {
                if (nextPt.x < -c_halfWin.x || nextPt.x >= cols || nextPt.y < -c_halfWin.y || nextPt.y >= rows)
                {
                        if (tid == 0 && level == 0)
                        {
                                status[get_group_id(0)] = 0;
                        }

                        return;
                }

                float b1 = 0;
                float b2 = 0;

                for (int y = get_local_id(1), i = 0; y < c_winSize_y; y += get_local_size(1), ++i)
                {
                        for (int x = get_local_id(0), j = 0; x < c_winSize_x; x += get_local_size(0), ++j)
                        {
                                float4 diff = (readImage2Df_C4(ptrJ, nextPt.x + x, nextPt.y + y, rows, cols, elemCntPerRow) - I_patch[i][j]) * 32.0f;

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

                nextPt += delta;

                //if (fabs(delta.x) < THRESHOLD && fabs(delta.y) < THRESHOLD)
                //    break;
        }

        float errval = 0.0f;

        if (calcErr)
        {
                for (int y = get_local_id(1), i = 0; y < c_winSize_y; y += get_local_size(1), ++i)
                {
                        for (int x = get_local_id(0), j = 0; x < c_winSize_x; x += get_local_size(0), ++j)
                        {
                                float4 diff = readImage2Df_C4(ptrJ, nextPt.x + x, nextPt.y + y, rows, cols, elemCntPerRow) - I_patch[i][j];

                                errval += fabs(diff.x) + fabs(diff.y) + fabs(diff.z);
                        }
                }

                reduce1(errval, smem1, tid);
        }

        if (tid == 0)
        {
                nextPt += c_halfWin;
                nextPts[get_group_id(0)] = nextPt;

                if (calcErr)
                {
                        err[get_group_id(0)] = smem1[0] / (3 * c_winSize_x * c_winSize_y);
                }
        }
}

int readImage2Di_C1(__global const int *image, float2 coor,  int2 size, const int elemCntPerRow)
{
        int i = clamp((int)floor(coor.x), 0, size.x - 1);
        int j = clamp((int)floor(coor.y), 0, size.y - 1);
        return image[mad24(j, elemCntPerRow, i)];
}

__kernel void lkDense_C1_D0(__global const int *I, __global const int *J, __global float *u, int uStep, __global float *v, int vStep, __global const float *prevU, int prevUStep, __global const float *prevV, int prevVStep,
                            const int rows, const int cols, /*__global float* err, int errStep, int cn,*/
                            const int elemCntPerRow, int c_winSize_x, int c_winSize_y, int c_iters, char calcErr)
{
        int c_halfWin_x = (c_winSize_x - 1) / 2;
        int c_halfWin_y = (c_winSize_y - 1) / 2;

        const int patchWidth  = get_local_size(0) + 2 * c_halfWin_x;
        const int patchHeight = get_local_size(1) + 2 * c_halfWin_y;

        __local int smem[8192];

        __local int *I_patch = smem;
        __local int *dIdx_patch = I_patch + patchWidth * patchHeight;
        __local int *dIdy_patch = dIdx_patch + patchWidth * patchHeight;

        const int xBase = get_group_id(0) * get_local_size(0);
        const int yBase = get_group_id(1) * get_local_size(1);
        int2 size = (int2)(cols, rows);

        for (int i = get_local_id(1); i < patchHeight; i += get_local_size(1))
        {
                for (int j = get_local_id(0); j < patchWidth; j += get_local_size(0))
                {
                        float x = xBase - c_halfWin_x + j + 0.5f;
                        float y = yBase - c_halfWin_y + i + 0.5f;

                        I_patch[i * patchWidth + j] = readImage2Di_C1(I, (float2)(x, y), size, elemCntPerRow);

                        // Sharr Deriv

                        dIdx_patch[i * patchWidth + j] = 3 * readImage2Di_C1(I, (float2)(x + 1, y - 1), size, elemCntPerRow) + 10 * readImage2Di_C1(I, (float2)(x + 1, y), size, elemCntPerRow) + 3 * readImage2Di_C1(I, (float2)(x + 1, y + 1), size, elemCntPerRow) -
                                                         (3 * readImage2Di_C1(I, (float2)(x - 1, y - 1), size, elemCntPerRow) + 10 * readImage2Di_C1(I, (float2)(x - 1, y), size, elemCntPerRow) + 3 * readImage2Di_C1(I, (float2)(x - 1, y + 1), size, elemCntPerRow));

                        dIdy_patch[i * patchWidth + j] = 3 * readImage2Di_C1(I, (float2)(x - 1, y + 1), size, elemCntPerRow) + 10 * readImage2Di_C1(I, (float2)(x, y + 1), size, elemCntPerRow) + 3 * readImage2Di_C1(I, (float2)(x + 1, y + 1), size, elemCntPerRow) -
                                                         (3 * readImage2Di_C1(I, (float2)(x - 1, y - 1), size, elemCntPerRow) + 10 * readImage2Di_C1(I, (float2)(x, y - 1), size, elemCntPerRow) + 3 * readImage2Di_C1(I, (float2)(x + 1, y - 1), size, elemCntPerRow));
                }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // extract the patch from the first image, compute covariation matrix of derivatives

        const int x = get_global_id(0);
        const int y = get_global_id(1);

        if (x >= cols || y >= rows)
        {
                return;
        }

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
        nextPt.x = x + prevU[y / 2 * prevUStep / 4 + x / 2] * 2.0f;
        nextPt.y = y + prevV[y / 2 * prevVStep / 4 + x / 2] * 2.0f;

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
                                int iJ = readImage2Di_C1(J, (float2)(nextPt.x - c_halfWin_x + j + 0.5f, nextPt.y - c_halfWin_y + i + 0.5f), size, elemCntPerRow);

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
                {
                        break;
                }
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
                                int iJ = readImage2Di_C1(J, (float2)(nextPt.x - c_halfWin_x + j + 0.5f, nextPt.y - c_halfWin_y + i + 0.5f), size, elemCntPerRow);

                                errval += abs(iJ - iI);
                        }
                }

                //err[y * errStep / 4 + x] = static_cast<float>(errval) / (c_winSize_x * c_winSize_y);
        }
}
