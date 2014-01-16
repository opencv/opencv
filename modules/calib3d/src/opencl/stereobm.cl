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

#ifdef SIZE

__kernel void stereoBM(__global const uchar * left, __global const uchar * right, __global uchar * dispptr,
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
                cost[d-mindisp] = 0;
                for(int i = -wsz2; i < wsz2+1; i++)
                    for(int j = -wsz2; j < wsz2+1; j++)
                    {
                        textsum += abs( left[min( y+i, rows-1 ) * cols + min( x+j, cols-1 )] - preFilterCap );
                        cost[d-mindisp] += abs( left[min( y+i, rows-1 ) * cols + min( x+j, cols-1 )]
                            - right[min( y+i, rows-1 ) * cols + min( x+j-d, cols-1 )] );
                    }
            }

            int best_disp = mindisp, best_cost = cost[0];
            for(int d = mindisp; d < ndisp+mindisp; d++)
            {
                best_cost = (cost[d-mindisp] < best_cost) ? cost[d-mindisp] : best_cost;
                best_disp = (best_cost == cost[d-mindisp]) ? d : best_disp;
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
                float a = (y3 - ((best_disp+1)*(y2-y1) + best_disp*y1 - (best_disp-1)*y2)/(best_disp - (best_disp-1)) )/
                    ((best_disp+1)*((best_disp+1) - (best_disp-1) - best_disp) + (best_disp-1)*best_disp);
                float b = (y2 - y1)/(best_disp - (best_disp-1)) - a*((best_disp-1)+best_disp);
                disp[0] = (y1 == y2 || y2 == y3) ? (short)(best_disp*16) : (short)(-b/(2*a)*16);
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
                cov2 += input[min( max( (y+i),0 ),rows-1 ) * cols + min( max( (x+j),0 ),cols-1 )];

        int res = (cov1*scale_g - cov2*scale_s)>>10;
        res = min(min(max(-prefilterCap, res), prefilterCap) + prefilterCap, 255);
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
    if(x < cols && y < rows-1 && x > 0)
    {
        int cov = input[((y > 0) ? y-1 : y+1) * cols + (x-1)] * (-1) + input[((y > 0) ? y-1 : y+1) * cols + ((x<cols-1) ? x+1 : x-1)] * (1) +
                  input[              (y)     * cols + (x-1)] * (-2) + input[       (y)            * cols + ((x<cols-1) ? x+1 : x-1)] * (2) +
                  input[             (y+1)    * cols + (x-1)] * (-1) + input[      (y+1)           * cols + ((x<cols-1) ? x+1 : x-1)] * (1);

        cov = min(min(max(-prefilterCap, cov), prefilterCap) + prefilterCap, 255);
        output[y * cols + x] = cov & 0xFF;
    }
}
