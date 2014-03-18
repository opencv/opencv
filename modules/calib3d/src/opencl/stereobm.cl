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

#ifdef csize

#define MAX_VAL 32767

void calcDisp(__local short * cost, __global short * disp, int uniquenessRatio, int mindisp, int ndisp, int w,
              __local int * bestDisp, __local int * bestCost, int d, int x, int y, int cols, int rows, int wsz2)
{
    short FILTERED = (mindisp - 1)<<4;
    int best_disp = *bestDisp, best_cost = *bestCost, best_disp_back = ndisp - best_disp - 1;

    short c = cost[0];

    int thresh = best_cost + (best_cost * uniquenessRatio/100);
    bool notUniq = ( (c <= thresh) && (d < (best_disp_back - 1) || d > (best_disp_back + 1) ) );

    if(notUniq)
        *bestCost = FILTERED;
    barrier(CLK_LOCAL_MEM_FENCE);

    if( *bestCost != FILTERED && x < cols-wsz2-mindisp && y < rows-wsz2 && d == best_disp_back)
    {
        int y3 = (best_disp_back > 0) ? cost[-w] : cost[w],
            y2 = c,
            y1 = (best_disp_back < ndisp-1) ? cost[w] : cost[-w];
        int d_aprox = y3+y1-2*y2 + abs(y3-y1);
        disp[0] = (short)(((best_disp_back + mindisp)*256 + (d_aprox != 0 ? (y3-y1)*256/d_aprox : 0) + 15) >> 4);
    }
}

int calcLocalIdx(int x, int y, int d, int w)
{
    return d*2*w + (w - 1 - y + x);
}

void calcNewCoordinates(int * x, int * y, int nthread)
{
    int oldX = *x - (1-nthread), oldY = *y;
    *x = (oldX == oldY) ? (0*nthread + (oldX + 2)*(1-nthread) ) : (oldX+1)*(1-nthread) + (oldX+1)*nthread;
    *y = (oldX == oldY) ? (0*(1-nthread) + (oldY + 1)*nthread) : oldY + 1*(1-nthread);
}

short calcCostBorder(__global const uchar * leftptr, __global const uchar * rightptr, int x, int y, int nthread,
                     int wsz2, short * costbuf, int * h, int cols, int d, short cost, int winsize)
{
    int head = (*h)%wsz;
    __global const uchar * left, * right;
    int idx = mad24(y+wsz2*(2*nthread-1), cols, x+wsz2*(1-2*nthread));
    left = leftptr + idx;
    right = rightptr + (idx - d);
    int shift = 1*nthread + cols*(1-nthread);

    short costdiff = 0;
    for(int i = 0; i < winsize; i++)
    {
        costdiff += abs( left[0] - right[0] );
        left += shift;
        right += shift;
    }
    cost += costdiff - costbuf[head];
    costbuf[head] = costdiff;
    (*h) = (*h)%wsz + 1;
    return cost;
}

short calcCostInside(__global const uchar * leftptr, __global const uchar * rightptr, int x, int y,
                     int wsz2, int cols, int d, short cost_up_left, short cost_up, short cost_left,
                     int winsize)
{
    __global const uchar * left, * right;
    int idx = mad24(y-wsz2-1, cols, x-wsz2-1);
    left = leftptr + idx;
    right = rightptr + (idx - d);
    int idx2 = winsize*cols;

    uchar corrner1 = abs(left[0] - right[0]),
          corrner2 = abs(left[winsize] - right[winsize]),
          corrner3 = abs(left[idx2] - right[idx2]),
          corrner4 = abs(left[idx2 + winsize] - right[idx2 + winsize]);

    return cost_up + cost_left - cost_up_left + corrner1 -
        corrner2 - corrner3 + corrner4;
}

__kernel void stereoBM(__global const uchar * leftptr, __global const uchar * rightptr, __global uchar * dispptr,
                       int disp_step, int disp_offset, int rows, int cols, int mindisp, int ndisp,
                       int preFilterCap, int textureTreshold, int uniquenessRatio, int sizeX, int sizeY, int winsize)
{
    int gx = get_global_id(0)*sizeX;
    int gy = get_global_id(1)*sizeY;
    int lz = get_local_id(2);

    int nthread = lz/ndisp;
    int d = lz%ndisp;
    int wsz2 = wsz/2;

    __global short * disp;
    __global const uchar * left, * right;

    __local short costFunc[csize];
    __local short * cost;
    __local int best_disp[2];
    __local int best_cost[2];
    best_cost[nthread] = MAX_VAL;

    short costbuf[wsz];
    int head = 0;

    int shiftX = wsz2 + ndisp + mindisp - 1;
    int shiftY = wsz2;

    int x = gx + shiftX, y = gy + shiftY, lx = 0, ly = 0;

    int costIdx = calcLocalIdx(lx, ly, d, sizeY);
    cost = costFunc + costIdx;

    short tempcost = 0;
    if(x < cols-wsz2-mindisp && y < rows-wsz2)
    {
        int shift = 1*nthread + cols*(1-nthread);
        for(int i = 0; i < winsize; i++)
        {
            int idx = mad24(y-wsz2+i*nthread, cols, x-wsz2+i*(1-nthread));
            left = leftptr + idx;
            right = rightptr + (idx - d);
            short costdiff = 0;
            for(int j = 0; j < winsize; j++)
            {
                costdiff += abs( left[0] - right[0] );
                left += shift;
                right += shift;
            }
            if(nthread==1)
            {
                tempcost += costdiff;
            }
            costbuf[head] = costdiff;
            head++;
        }
    }
    if(nthread==1)
    {
        cost[0] = tempcost;
        atomic_min(best_cost+nthread, tempcost);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(best_cost[1] == tempcost)
        best_disp[1] = ndisp - d - 1;
    barrier(CLK_LOCAL_MEM_FENCE);

    int dispIdx = mad24(gy, disp_step, disp_offset + gx*(int)sizeof(short));
    disp = (__global short *)(dispptr + dispIdx);
    calcDisp(cost, disp, uniquenessRatio, mindisp, ndisp, 2*sizeY,
        best_disp + 1, best_cost+1, d, x, y, cols, rows, wsz2);
    barrier(CLK_LOCAL_MEM_FENCE);

    lx = 1 - nthread;
    ly = nthread;

    for(int i = 0; i < sizeY*sizeX/2; i++)
    {
        x = (lx < sizeX) ? gx + shiftX + lx : cols;
        y = (ly < sizeY) ? gy + shiftY + ly : rows;

        best_cost[nthread] = MAX_VAL;
        barrier(CLK_LOCAL_MEM_FENCE);

        costIdx = calcLocalIdx(lx, ly, d, sizeY);
        cost = costFunc + costIdx;

        if(x < cols-wsz2-mindisp && y < rows-wsz2 )
        {
            tempcost = ( ly*(1-nthread) + lx*nthread == 0 ) ?
                calcCostBorder(leftptr, rightptr, x, y, nthread, wsz2, costbuf, &head, cols, d,
                    cost[2*nthread-1], winsize) :
                calcCostInside(leftptr, rightptr, x, y, wsz2, cols, d,
                    cost[0], cost[1], cost[-1], winsize);
        }
        cost[0] = tempcost;
        atomic_min(best_cost + nthread, tempcost);
        barrier(CLK_LOCAL_MEM_FENCE);

        if(best_cost[nthread] == tempcost)
            best_disp[nthread] = ndisp - d - 1;
        barrier(CLK_LOCAL_MEM_FENCE);

        int dispIdx = mad24(gy+ly, disp_step, disp_offset + (gx+lx)*(int)sizeof(short));
        disp = (__global short *)(dispptr + dispIdx);

        calcDisp(cost, disp, uniquenessRatio, mindisp, ndisp, 2*sizeY,
            best_disp + nthread, best_cost + nthread, d, x, y, cols, rows, wsz2);
        barrier(CLK_LOCAL_MEM_FENCE);

        calcNewCoordinates(&lx, &ly, nthread);
    }
}

#endif

//////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////// Norm Prefiler ////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void prefilter_norm(__global unsigned char *input, __global unsigned char *output,
                               int rows, int cols, int prefilterCap, int winsize, int scale_g, int scale_s)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int wsz2 = winsize/2;

    if(x < cols && y < rows)
    {
        int cov1 =                                   input[   max(y-1, 0)   * cols + x] * 1 +
                  input[y * cols + max(x-1,0)] * 1 + input[      y          * cols + x] * 4 + input[y * cols + min(x+1, cols-1)] * 1 +
                                                     input[min(y+1, rows-1) * cols + x] * 1;
        int cov2 = 0;
        for(int i = -wsz2; i < wsz2+1; i++)
            for(int j = -wsz2; j < wsz2+1; j++)
                cov2 += input[clamp(y+i, 0, rows-1) * cols + clamp(x+j, 0, cols-1)];

        int res = (cov1*scale_g - cov2*scale_s)>>10;
        res = min(clamp(res, -prefilterCap, prefilterCap) + prefilterCap, 255);
        output[y * cols + x] = res & 0xFF;
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////// Sobel Prefiler ////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void prefilter_xsobel(__global unsigned char *input, __global unsigned char *output,
                               int rows, int cols, int prefilterCap)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if(x < cols && y < rows)
    {
            output[y * cols + x] = min(prefilterCap, 255) & 0xFF;
    }

    if(x < cols && y < rows && x > 0 && !((y == rows-1)&(rows%2==1) ) )
    {
        int cov = input[ ((y > 0) ? y-1 : y+1)  * cols + (x-1)] * (-1) + input[ ((y > 0) ? y-1 : y+1)  * cols + ((x<cols-1) ? x+1 : x-1)] * (1) +
                  input[              (y)       * cols + (x-1)] * (-2) + input[        (y)             * cols + ((x<cols-1) ? x+1 : x-1)] * (2) +
                  input[((y<rows-1)?(y+1):(y-1))* cols + (x-1)] * (-1) + input[((y<rows-1)?(y+1):(y-1))* cols + ((x<cols-1) ? x+1 : x-1)] * (1);

        cov = min(clamp(cov, -prefilterCap, prefilterCap) + prefilterCap, 255);
        output[y * cols + x] = cov & 0xFF;
    }
}
