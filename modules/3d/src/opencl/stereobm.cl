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

//////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// stereoBM //////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

#define MAX_VAL 32767

#ifndef WSZ
#define WSZ     2
#endif

#define WSZ2    (WSZ / 2)

#ifdef DEFINE_KERNEL_STEREOBM

#define DISPARITY_SHIFT     4
#define FILTERED            ((MIN_DISP - 1) << DISPARITY_SHIFT)

void calcDisp(__local short * cost, __global short * disp, int uniquenessRatio,
              __local int * bestDisp, __local int * bestCost, int d, int x, int y, int cols, int rows)
{
    int best_disp = *bestDisp, best_cost = *bestCost;
    barrier(CLK_LOCAL_MEM_FENCE);

    short c = cost[0];
    int thresh = best_cost + (best_cost * uniquenessRatio / 100);
    bool notUniq = ( (c <= thresh) && (d < (best_disp - 1) || d > (best_disp + 1) ) );

    if (notUniq)
        *bestCost = FILTERED;
    barrier(CLK_LOCAL_MEM_FENCE);

    if( *bestCost != FILTERED && x < cols - WSZ2 - MIN_DISP && y < rows - WSZ2 && d == best_disp)
    {
        int d_aprox = 0;
        int yp =0, yn = 0;
        if ((0 < best_disp) && (best_disp < NUM_DISP - 1))
        {
            yp = cost[-2 * BLOCK_SIZE_Y];
            yn = cost[2 * BLOCK_SIZE_Y];
            d_aprox = yp + yn - 2 * c + abs(yp - yn);
        }
        disp[0] = (short)(((best_disp + MIN_DISP)*256 + (d_aprox != 0 ? (yp - yn) * 256 / d_aprox : 0) + 15) >> 4);
    }
}

short calcCostBorder(__global const uchar * leftptr, __global const uchar * rightptr, int x, int y, int nthread,
                     short * costbuf, int *h, int cols, int d, short cost)
{
    int head = (*h) % WSZ;
    __global const uchar * left, * right;
    int idx = mad24(y + WSZ2 * (2 * nthread - 1), cols, x + WSZ2 * (1 - 2 * nthread));
    left = leftptr + idx;
    right = rightptr + (idx - d);

    short costdiff = 0;
    if (0 == nthread)
    {
        #pragma unroll
        for (int i = 0; i < WSZ; i++)
        {
            costdiff += abs( left[0] - right[0] );
            left += cols;
            right += cols;
        }
    }
    else // (1 == nthread)
    {
        #pragma unroll
        for (int i = 0; i < WSZ; i++)
        {
            costdiff += abs(left[i] - right[i]);
        }
    }
    cost += costdiff - costbuf[head];
    costbuf[head] = costdiff;
    *h = head + 1;
    return cost;
}

short calcCostInside(__global const uchar * leftptr, __global const uchar * rightptr, int x, int y,
                     int cols, int d, short cost_up_left, short cost_up, short cost_left)
{
    __global const uchar * left, * right;
    int idx = mad24(y - WSZ2 - 1, cols, x - WSZ2 - 1);
    left = leftptr + idx;
    right = rightptr + (idx - d);
    int idx2 = WSZ*cols;

    uchar corrner1 = abs(left[0] - right[0]),
          corrner2 = abs(left[WSZ] - right[WSZ]),
          corrner3 = abs(left[idx2] - right[idx2]),
          corrner4 = abs(left[idx2 + WSZ] - right[idx2 + WSZ]);

    return cost_up + cost_left - cost_up_left + corrner1 -
        corrner2 - corrner3 + corrner4;
}

__kernel void stereoBM(__global const uchar * leftptr,
                       __global const uchar * rightptr,
                       __global uchar * dispptr, int disp_step, int disp_offset,
                       int rows, int cols,                                              // rows, cols of left and right images, not disp
                       int textureThreshold, int uniquenessRatio)
{
    int lz = get_local_id(0);
    int gx = get_global_id(1) * BLOCK_SIZE_X;
    int gy = get_global_id(2) * BLOCK_SIZE_Y;

    int nthread = lz / NUM_DISP;
    int disp_idx = lz % NUM_DISP;

    __global short * disp;
    __global const uchar * left, * right;

    __local short costFunc[2 * BLOCK_SIZE_Y * NUM_DISP];

    __local short * cost;
    __local int best_disp[2];
    __local int best_cost[2];
    best_cost[nthread] = MAX_VAL;
    best_disp[nthread] = -1;
    barrier(CLK_LOCAL_MEM_FENCE);

    short costbuf[WSZ];
    int head = 0;

    int shiftX = WSZ2 + NUM_DISP + MIN_DISP - 1;
    int shiftY = WSZ2;

    int x = gx + shiftX, y = gy + shiftY, lx = 0, ly = 0;

    int costIdx = disp_idx * 2 * BLOCK_SIZE_Y + (BLOCK_SIZE_Y - 1);
    cost = costFunc + costIdx;

    int tempcost = 0;
    if (x < cols - WSZ2 - MIN_DISP && y < rows - WSZ2)
    {
        if (0 == nthread)
        {
            #pragma unroll
            for (int i = 0; i < WSZ; i++)
            {
                int idx = mad24(y - WSZ2, cols, x - WSZ2 + i);
                left = leftptr + idx;
                right = rightptr + (idx - disp_idx);
                short costdiff = 0;
                for(int j = 0; j < WSZ; j++)
                {
                    costdiff += abs( left[0] - right[0] );
                    left += cols;
                    right += cols;
                }
                costbuf[i] = costdiff;
            }
        }
        else // (1 == nthread)
        {
            #pragma unroll
            for (int i = 0; i < WSZ; i++)
            {
                int idx = mad24(y - WSZ2 + i, cols, x - WSZ2);
                left = leftptr + idx;
                right = rightptr + (idx - disp_idx);
                short costdiff = 0;
                for (int j = 0; j < WSZ; j++)
                {
                    costdiff += abs( left[j] - right[j]);
                }
                tempcost += costdiff;
                costbuf[i] = costdiff;
            }
        }
    }
    if (nthread == 1)
    {
        cost[0] = tempcost;
        atomic_min(best_cost + 1, tempcost);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (best_cost[1] == tempcost)
         atomic_max(best_disp + 1, disp_idx);
    barrier(CLK_LOCAL_MEM_FENCE);

    int dispIdx = mad24(gy, disp_step, mad24((int)sizeof(short), gx, disp_offset));
    disp = (__global short *)(dispptr + dispIdx);
    calcDisp(cost, disp, uniquenessRatio, best_disp + 1, best_cost + 1, disp_idx, x, y, cols, rows);
    barrier(CLK_LOCAL_MEM_FENCE);

    lx = 1 - nthread;
    ly = nthread;

    for (int i = 0; i < BLOCK_SIZE_Y * BLOCK_SIZE_X / 2; i++)
    {
        x = (lx < BLOCK_SIZE_X) ? gx + shiftX + lx : cols;
        y = (ly < BLOCK_SIZE_Y) ? gy + shiftY + ly : rows;

        best_cost[nthread] = MAX_VAL;
        best_disp[nthread] = -1;
        barrier(CLK_LOCAL_MEM_FENCE);

        costIdx = mad24(2 * BLOCK_SIZE_Y, disp_idx, (BLOCK_SIZE_Y - 1 - ly + lx));
        if (0 > costIdx)
            costIdx = BLOCK_SIZE_Y - 1;
        cost = costFunc + costIdx;
        if (x < cols - WSZ2 - MIN_DISP && y < rows - WSZ2)
        {
            tempcost = (ly * (1 - nthread) + lx * nthread == 0) ?
                calcCostBorder(leftptr, rightptr, x, y, nthread, costbuf, &head, cols, disp_idx, cost[2*nthread-1]) :
                calcCostInside(leftptr, rightptr, x, y, cols, disp_idx, cost[0], cost[1], cost[-1]);
        }
        cost[0] = tempcost;
        atomic_min(best_cost + nthread, tempcost);
        barrier(CLK_LOCAL_MEM_FENCE);

        if (best_cost[nthread] == tempcost)
            atomic_max(best_disp + nthread, disp_idx);
        barrier(CLK_LOCAL_MEM_FENCE);

        dispIdx = mad24(gy + ly, disp_step, mad24((int)sizeof(short), (gx + lx), disp_offset));
        disp = (__global short *)(dispptr + dispIdx);
        calcDisp(cost, disp, uniquenessRatio, best_disp + nthread, best_cost + nthread, disp_idx, x, y, cols, rows);

        barrier(CLK_LOCAL_MEM_FENCE);

        if (lx + nthread - 1 == ly)
        {
            lx = (lx + nthread + 1) * (1 - nthread);
            ly = (ly + 1) * nthread;
        }
        else
        {
            lx += nthread;
            ly = ly - nthread + 1;
        }
    }
}
#endif //DEFINE_KERNEL_STEREOBM

//////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////// Norm Prefiler ////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void prefilter_norm(__global unsigned char *input, __global unsigned char *output,
                               int rows, int cols, int prefilterCap, int scale_g, int scale_s)
{
    // prefilterCap in range 1..63, checked in StereoBMImpl::compute

    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < cols && y < rows)
    {
        int cov1 =                                   input[   max(y-1, 0)   * cols + x] * 1 +
                  input[y * cols + max(x-1,0)] * 1 + input[      y          * cols + x] * 4 + input[y * cols + min(x+1, cols-1)] * 1 +
                                                     input[min(y+1, rows-1) * cols + x] * 1;
        int cov2 = 0;
        for(int i = -WSZ2; i < WSZ2+1; i++)
            for(int j = -WSZ2; j < WSZ2+1; j++)
                cov2 += input[clamp(y+i, 0, rows-1) * cols + clamp(x+j, 0, cols-1)];

        int res = (cov1*scale_g - cov2*scale_s)>>10;
        res = clamp(res, -prefilterCap, prefilterCap) + prefilterCap;
        output[y * cols + x] = res;
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////// Sobel Prefiler ////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void prefilter_xsobel(__global unsigned char *input, __global unsigned char *output,
                               int rows, int cols, int prefilterCap)
{
    // prefilterCap in range 1..63, checked in StereoBMImpl::compute
    int x = get_global_id(0);
    int y = get_global_id(1);
    if(x < cols && y < rows)
    {
        if (0 < x && !((y == rows-1) & (rows%2==1) ) )
        {
            int cov = input[ ((y > 0) ? y-1 : y+1)  * cols + (x-1)] * (-1) + input[ ((y > 0) ? y-1 : y+1)  * cols + ((x<cols-1) ? x+1 : x-1)] * (1) +
                      input[              (y)       * cols + (x-1)] * (-2) + input[        (y)             * cols + ((x<cols-1) ? x+1 : x-1)] * (2) +
                      input[((y<rows-1)?(y+1):(y-1))* cols + (x-1)] * (-1) + input[((y<rows-1)?(y+1):(y-1))* cols + ((x<cols-1) ? x+1 : x-1)] * (1);

            cov = clamp(cov, -prefilterCap, prefilterCap) + prefilterCap;
            output[y * cols + x] = cov;
        }
        else
            output[y * cols + x] = prefilterCap;
    }
}