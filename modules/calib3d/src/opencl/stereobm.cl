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

#pragma OPENCL EXTENSION cl_amd_printf : enable

//////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// stereoBM //////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef csize

#define MAX_VAL 32767

void calcDisp(__local short * costFunc, __global short * disp, int uniquenessRatio/*, int textureTreshold, short textsum*/,
              int mindisp, int ndisp, int w, __local short * dispbuf, int d)
{
    short FILTERED = (mindisp - 1)<<4;
    short best_disp = FILTERED, best_cost = MAX_VAL-1;
    __local short * cost;

    cost = &costFunc[0];
    dispbuf[d] = d;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int lsize = tsize/2 >> 1; lsize > 0; lsize >>= 1)
    {
        short lid1 = dispbuf[d], lid2 = dispbuf[d+lsize],
            cost1 = cost[lid1*w], cost2 = cost[lid2*w];
        if (d < lsize)
        {
           dispbuf[d] = (cost1 < cost2) ? lid1 : (cost1==cost2) ? max(lid1, lid2) : lid2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    best_disp = ndisp - dispbuf[0] - 1;
    best_cost = costFunc[(ndisp-best_disp-1)*w];

    int thresh = best_cost + (best_cost * uniquenessRatio/100);
    dispbuf[d] = ( (cost[d*w] <= thresh) && (d < (ndisp - best_disp - 2) || d > (ndisp - best_disp) ) ) ? FILTERED : best_disp;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int lsize = tsize/2 >> 1; lsize > 0; lsize >>= 1)
    {
        short val1 = dispbuf[d], val2 = dispbuf[d+lsize];
        if (d < lsize)
        {
           dispbuf[d] = min(val1, val2);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

//    best_disp = (textsum < textureTreshold) ? FILTERED : best_disp;

    if( dispbuf[0] != FILTERED )
    {
        cost = &costFunc[0] + (ndisp - best_disp - 1)*w;
        int y3 = ((ndisp - best_disp - 1) > 0) ? cost[-w] : cost[w],
            y2 = cost[0],
            y1 = ((ndisp - best_disp - 1) < ndisp-1) ? cost[w] : cost[-w];
        int d = y3+y1-2*y2 + abs(y3-y1);
        disp[0] = (short)(((ndisp - best_disp - 1 + mindisp)*256 + (d != 0 ? (y3-y1)*256/d : 0) + 15) >> 4);
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
                     int wsz2, short * costbuf, int * h, int cols, int d, short cost)
{
    int head = (*h)%wsz;
    __global const uchar * left, * right;
    int idx = mad24(y+wsz2*(2*nthread-1), cols, x+wsz2*(1-2*nthread));
    left = leftptr + idx;
    right = rightptr + (idx - d);

    short costdiff = 0;
    for(int i = 0; i < wsz; i++)
    {
            costdiff += abs( left[0] - right[0] );
            left += 1*nthread + cols*(1-nthread);
            right += 1*nthread + cols*(1-nthread);// maybe use ? operator
    }
    cost += costdiff - costbuf[head];
    costbuf[head] = costdiff;
    (*h) = (*h)%wsz + 1;
    return cost;
}

short calcCostInside(__global const uchar * leftptr, __global const uchar * rightptr, int x, int y,
                     int wsz2, int cols, int d, short cost_up_left, short cost_up, short cost_left)
{
    __global const uchar * left, * right;
    int idx = mad24(y-wsz2-1, cols, x-wsz2-1);
    left = leftptr + idx;
    right = rightptr + (idx - d);

    return cost_up + cost_left - cost_up_left + abs(left[0] - right[0]) -
        abs(left[wsz] - right[wsz]) - abs(left[(wsz)*cols] - right[(wsz)*cols]) +
        abs(left[(wsz)*cols + wsz] - right[(wsz)*cols + wsz]);
}

__kernel void stereoBM_opt(__global const uchar * leftptr, __global const uchar * rightptr, __global uchar * dispptr,
                       int disp_step, int disp_offset, int rows, int cols, int mindisp, int ndisp,
                       int preFilterCap, int textureTreshold, int uniquenessRatio, int sizeX, int sizeY)
{
    int gx = get_global_id(0)*sizeX;
    int gy = get_global_id(1)*sizeY;
    int lz = get_local_id(2);

    int nthread = lz/32;// only 0 or 1
    int d = lz%32;// 1 .. 32
    int wsz2 = wsz/2;

    __global short * disp;
    __global const uchar * left, * right;

    __local short dispbuf[tsize];
    __local short costFunc[csize];
    __local short * cost;

    short costbuf[wsz];
    int head = 0;

    int shiftX = wsz2 + ndisp + mindisp - 1;
    int shiftY = wsz2;

    int x = gx + shiftX, y = gy + shiftY, lx = 0, ly = 0;

    int costIdx = calcLocalIdx(lx, ly, d, sizeY);
    cost = costFunc + costIdx;

    short tempcost = 0;
    for(int i = 0; i < wsz; i++)
    {
        int idx = mad24(y-wsz2+i*nthread, cols, x-wsz2+i*(1-nthread));
        left = leftptr + idx;
        right = rightptr + (idx - d);
        short costdiff = 0;

        for(int j = 0; j < wsz; j++)
        {
            costdiff += abs( left[0] - right[0] );
            left += 1*nthread + cols*(1-nthread);
            right += 1*nthread + cols*(1-nthread);// maybe use ? operator
        }
        if(nthread==1)
        {
            tempcost += costdiff;
        }
        costbuf[head] = costdiff;
        head++;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    cost[0] = tempcost;

    if(x < cols-wsz2-mindisp && y < rows-wsz2 && nthread == 1)
    {
        int dispIdx = mad24(gy, disp_step, disp_offset + gx*(int)sizeof(short));
        disp = (__global short *)(dispptr + dispIdx);
        calcDisp(&costFunc[sizeY - 1 + lx - ly], disp, uniquenessRatio, /*textureTreshold, textsum,*/
            mindisp, ndisp, 2*sizeY, &dispbuf[nthread*tsize/2], d);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    lx = 1 - nthread;
    ly = nthread;

    while(lx < sizeX && ly < sizeY )
    {
        x = gx + shiftX + lx;
        y = gy + shiftY + ly;

        costIdx = calcLocalIdx(lx, ly, d, sizeY);
        cost = costFunc + costIdx;
        cost[0] = ( ly*(1-nthread) + lx*nthread == 0 ) ?
            calcCostBorder(leftptr, rightptr, x, y, nthread, wsz2, costbuf, &head, cols, d,
                costFunc[calcLocalIdx(lx-1*(1-nthread), ly-1*nthread, d, sizeY)]) :
            calcCostInside(leftptr, rightptr, x, y, wsz2, cols, d,
                costFunc[calcLocalIdx(lx-1, ly-1, d, sizeY)],
                costFunc[calcLocalIdx(lx, ly-1, d, sizeY)],
                costFunc[calcLocalIdx(lx-1, ly, d, sizeY)]);
        barrier(CLK_LOCAL_MEM_FENCE);

        if(x < cols-mindisp-wsz2 && y < rows-wsz2)
        {
            int dispIdx = mad24(gy+ly, disp_step, disp_offset + (gx+lx)*(int)sizeof(short));
            disp = (__global short *)(dispptr + dispIdx);
            calcDisp(&costFunc[sizeY - 1 - ly + lx], disp, uniquenessRatio, //textureTreshold, textsum,
                mindisp, ndisp, 2*sizeY, &dispbuf[nthread*tsize/2], d);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        calcNewCoordinates(&lx, &ly, nthread);
    }
}

#endif

#ifdef SIZE

__kernel void stereoBM_BF(__global const uchar * left, __global const uchar * right, __global uchar * dispptr,
                       int disp_step, int disp_offset, int rows, int cols, int mindisp, int ndisp,
                       int preFilterCap, int winsize, int textureTreshold, int uniquenessRatio)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int wsz2 = winsize/2;
    short FILTERED = (mindisp - 1)<<4;

    if(x < cols && y < rows )

    {
        int dispIdx = mad24(y, disp_step, disp_offset + x*(int)sizeof(short) );
        __global short * disp = (__global short*)(dispptr + dispIdx);
        disp[0] = FILTERED;
        if( (x > mindisp+ndisp+wsz2-2) && (y > wsz2-1) && (x < cols-wsz2-mindisp) && (y < rows - wsz2))
        {
            int cost[SIZE];
            int textsum = 0;

            for(int d = mindisp; d < ndisp+mindisp; d++)
            {
                cost[(ndisp-1) - (d - mindisp)] = 0;
                for(int i = -wsz2; i < wsz2+1; i++)
                    for(int j = -wsz2; j < wsz2+1; j++)
                    {
                        textsum += (d == mindisp) ? abs( left[ (y+i) * cols + x + j] - preFilterCap ) : 0;
                        cost[(ndisp-1) - (d - mindisp)] += abs(left[(y+i) * cols + x+j] - right[(y+i) * cols + x+j-d] );
                    }
            }

            int best_disp = -1, best_cost = INT_MAX;
            for(int d = ndisp + mindisp - 1; d > mindisp-1; d--)
            {
                best_cost = (cost[d-mindisp] < best_cost) ? cost[d-mindisp] : best_cost;
                best_disp = (best_cost == cost[d-mindisp]) ? (d) : best_disp;
            }

            int thresh = best_cost + (best_cost * uniquenessRatio/100);
            for(int d = mindisp; (d < ndisp + mindisp) && (uniquenessRatio > 0); d++)
            {
                best_disp = ( (cost[d-mindisp] <= thresh) && (d < best_disp-1 || d > best_disp + 1) ) ? FILTERED : best_disp;
            }

            disp[0] = textsum < textureTreshold ? (FILTERED) : (best_disp == FILTERED) ? (short)(best_disp) : (short)(best_disp);

            if( best_disp != FILTERED )
            {
                int y1 = (best_disp > mindisp) ? cost[best_disp-mindisp-1] : cost[best_disp-mindisp+1],
                    y2 = cost[best_disp-mindisp],
                    y3 = (best_disp < mindisp+ndisp-1) ? cost[best_disp-mindisp+1] : cost[best_disp-mindisp-1];
                int _d = y3+y1-2*y2 + abs(y3-y1);
                disp[0] = (short)(((ndisp - (best_disp-mindisp) - 1 + mindisp)*256 + (_d != 0 ? (y3-y1)*256/_d : 0) + 15) >> 4);
            }
        }
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
