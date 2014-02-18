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

__kernel void stereoBM_opt(__global const uchar * leftptr, __global const uchar * rightptr, __global uchar * dispptr,
                       int disp_step, int disp_offset, int rows, int cols, int mindisp, int ndisp,
                       int preFilterCap, int nthreads, int textureTreshold, int uniquenessRatio)
{
    int x = get_global_id(0);
    int total_y = get_global_id(1);
    int z = get_local_id(2);
    int d = get_local_id(1);
    int gy = get_group_id(1), y = gy*ndisp + z*ndisp/nthreads;
    int wsz2 = wsz/2;
    short FILTERED = (mindisp - 1)<<4;
    __local short costFunc[csize];
    short textsum[tsize];
    __local short * cost = &costFunc[0] + d + ndisp*ndisp/nthreads*z;
    __global const uchar * left, * right;
    int dispIdx = mad24(total_y, disp_step, disp_offset + x*(int)sizeof(short) );
    __global short * disp = (__global short*)(dispptr + dispIdx);
    if( x < cols && total_y < rows)
    {
        disp[0] = FILTERED;
    }

    short costbuf[wsz];
    short textbuf[wsz];
    int head = 0;

    if( (x > ndisp+mindisp+wsz2-2) && (x < cols - wsz2 - mindisp) && (y < rows-wsz2) )
    {
        cost += (y < wsz2) ? ndisp*wsz2 : 0;
        y = (y<wsz2) ? wsz2 : y;
        cost[0] = 0;
        textsum[y%ndisp] = 0;
        #pragma unroll
        for(int i = 0; (i < wsz); i++)
        {
            left = leftptr + mad24(y-wsz2+i, cols, x-wsz2);
            right = rightptr + mad24(y-wsz2+i, cols, x-wsz2-d-mindisp);

            int costdiff = 0, textdiff = 0;
            #pragma unroll
            for(int j = 0; j < wsz; j++)
            {
                costdiff += abs( left[0] - right[0] );
                textdiff += abs( left[0] - preFilterCap );
                left++; right++;
            }
            cost[0] += costdiff;
            textsum[y%ndisp] += textdiff;
            costbuf[head] = costdiff;
            textbuf[head] = textdiff;
            head++;
        }
        y++;

        for(; y < gy*ndisp + (z+1)*ndisp/nthreads; y++)
        {
            head = head%wsz;
            cost += ndisp;
            cost[0] = cost[-ndisp];
            textsum[y%ndisp] = textsum[(y-1)%ndisp];
            left = leftptr + mad24(y+wsz2, cols, x - wsz2);
            right = rightptr + mad24(y+wsz2, cols, x - wsz2 - d - mindisp);

            int costdiff = 0, textdiff = 0;
            #pragma unroll
            for(int i = 0; i < wsz; i++)
            {
                costdiff +=
                    abs( left[0] - right[0] );
                textdiff += abs( left[0] - preFilterCap );
                left++; right++;
            }
            cost[0] += costdiff - costbuf[head];
            textsum[y%ndisp] += textdiff - textbuf[head];
            costbuf[head] = costdiff;
            textbuf[head] = textdiff;
            head++;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        cost = &costFunc[0] + d*ndisp;
        short best_disp = FILTERED, best_cost = MAX_VAL-1;
        #pragma unroll
        for(int i = 0; i < tsize; i++)
        {
            short c = cost[0];
            best_cost = (c < best_cost) ? c : best_cost;
            best_disp = (best_cost == c) ? ndisp - i - 1 : best_disp;
            cost++;
        }

        cost = &costFunc[0] + d*ndisp;
        int thresh = best_cost + (best_cost * uniquenessRatio/100);
        #pragma unroll
        for(int i = 0; (i < tsize) && (uniquenessRatio > 0); i++)
        {
            best_disp = ( (cost[0] <= thresh) && (i < (ndisp - best_disp - 2) || i > (ndisp - best_disp) ) ) ?
                FILTERED : best_disp;
            cost++;
        }

        best_disp = (total_y >= rows-wsz2) || (total_y < wsz2) || (textsum[d] < textureTreshold) ? FILTERED : best_disp;

        if( best_disp != FILTERED )
        {
            cost = &costFunc[0] + (ndisp - best_disp - 1) + ndisp*d;
            int y3 = ((ndisp - best_disp - 1) > 0) ? cost[-1] : cost[1],
                y2 = cost[0],
                y1 = ((ndisp - best_disp - 1) < ndisp-1) ? cost[1] : cost[-1];
            d = y3+y1-2*y2 + abs(y3-y1);
            if( x < cols && total_y < rows)
            {
                disp[0] = (short)(((ndisp - best_disp - 1 + mindisp)*256 + (d != 0 ? (y3-y1)*256/d : 0) + 15) >> 4);
            }
        }
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
    output[y * cols + x] = min(prefilterCap, 255) & 0xFF;
    if(x < cols && y < rows && x > 0 && !((y == rows-1)&(rows%2==1) ) )
    {
        int cov = input[ ((y > 0) ? y-1 : y+1)  * cols + (x-1)] * (-1) + input[ ((y > 0) ? y-1 : y+1)  * cols + ((x<cols-1) ? x+1 : x-1)] * (1) +
                  input[              (y)       * cols + (x-1)] * (-2) + input[        (y)             * cols + ((x<cols-1) ? x+1 : x-1)] * (2) +
                  input[((y<rows-1)?(y+1):(y-1))* cols + (x-1)] * (-1) + input[((y<rows-1)?(y+1):(y-1))* cols + ((x<cols-1) ? x+1 : x-1)] * (1);

        cov = min(clamp(cov, -prefilterCap, prefilterCap) + prefilterCap, 255);
        output[y * cols + x] = cov & 0xFF;
    }
}
