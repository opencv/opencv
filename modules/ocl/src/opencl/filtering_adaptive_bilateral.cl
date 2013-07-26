///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2012, AMD Inc, all rights reserved.
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


///////////////////////////////////////////////////////////////////////////////
// Author:  Harris Gasparakis
//  
// Description:
//
// This sample program demonstrates how to easily use custom openCL together with 
// openCL accelerated openCV (available as of openCV version 2.4.3).
//
// While typically opencl programs can include extensive plumbing code to initialize
// opencl context, queues, memory buffers, and the like, this demo inherits
// all the plumbing from cv::ocl.  It also inherits all the video feed plumbing 
// from opencv. With just a few lines of code, the stage is set to call
// a custom opencl kernel, which is implemented in-place in this file.
// 
// This executable links with opencv binaries, so you need to have opencv
// built, of course with OpenCL enabled in cmake.
//

#ifndef DEMO_OCL_OPENCVCL_CL
#define DEMO_OCL_OPENCVCL_CL


#ifdef BORDER_REPLICATE
//BORDER_REPLICATE:     aaaaaa|abcdefgh|hhhhhhh
#define ADDR_L(i, l_edge, r_edge)  ((i) <  (l_edge) ? (l_edge)   : (i))
#define ADDR_R(i, r_edge, addr)    ((i) >= (r_edge) ? (r_edge)-1 : (addr))
#define ADDR_H(i, t_edge, b_edge)  ((i) <  (t_edge) ? (t_edge)   :(i))
#define ADDR_B(i, b_edge, addr)    ((i) >= (b_edge) ? (b_edge)-1 :(addr))
#endif

#ifdef BORDER_REFLECT
//BORDER_REFLECT:       fedcba|abcdefgh|hgfedcb
#define ADDR_L(i, l_edge, r_edge)  ((i) <  (l_edge) ? -(i)-1               : (i))
#define ADDR_R(i, r_edge, addr)    ((i) >= (r_edge) ? -(i)-1+((r_edge)<<1) : (addr))
#define ADDR_H(i, t_edge, b_edge)  ((i) <  (t_edge) ? -(i)-1 : (i))
#define ADDR_B(i, b_edge, addr)    ((i) >= (b_edge) ? -(i)-1+((b_edge)<<1) : (addr))
#endif

#ifdef BORDER_REFLECT_101
//BORDER_REFLECT_101:   gfedcb|abcdefgh|gfedcba
#define ADDR_L(i, l_edge, r_edge)  ((i) <  (l_edge) ? -(i)                 : (i))
#define ADDR_R(i, r_edge, addr)    ((i) >= (r_edge) ? -(i)-2+((r_edge)<<1) : (addr))
#define ADDR_H(i, t_edge, b_edge)  ((i) <  (t_edge) ? -(i)                 : (i))
#define ADDR_B(i, b_edge, addr)    ((i) >= (b_edge) ? -(i)-2+((b_edge)<<1) : (addr))
#endif

//blur function does not support BORDER_WRAP
#ifdef BORDER_WRAP
//BORDER_WRAP:          cdefgh|abcdefgh|abcdefg
#define ADDR_L(i, l_edge, r_edge)  ((i) <  (l_edge) ? (i)+(r_edge) : (i))
#define ADDR_R(i, r_edge, addr)    ((i) >= (r_edge) ? (i)-(r_edge) : (addr))
#define ADDR_H(i, t_edge, b_edge)  ((i) <  (t_edge) ? (i)+(b_edge) : (i))
#define ADDR_B(i, b_edge, addr)    ((i) >= (b_edge) ? (i)-(b_edge) : (addr))
#endif

__kernel void 
    edgeEnhancingFilter(
    __global const uchar4 * restrict src, 
    __global uchar4 *dst, 
    float alpha,                  
    int src_offset, 
    int src_whole_rows, 
    int src_whole_cols, 
    int src_step,                
    int dst_offset, 
    int dst_rows, 
    int dst_cols, 
    int dst_step)
{																			
    int col = get_local_id(0);												
    const int gX = get_group_id(0);											
    const int gY = get_group_id(1);											

    int src_x_off = (src_offset % src_step) >> 2;							
    int src_y_off = src_offset / src_step;									
    int dst_x_off = (dst_offset % dst_step) >> 2;							
    int dst_y_off = dst_offset / dst_step;									

    int startX = gX * (THREADS-ksX+1) - anX + src_x_off;					
    int startY = (gY * (1+EXTRA)) - anY + src_y_off;

    int dst_startX = gX * (THREADS-ksX+1) + dst_x_off;						
    int dst_startY = (gY * (1+EXTRA)) + dst_y_off;									

    int posX = dst_startX - dst_x_off + col;								
    int posY = (gY * (1+EXTRA))	;											

#if LDATATYPESIZE == 1														
    __local uchar4 data[ksY+EXTRA][THREADS];									
#elif	LDATATYPESIZE == 4													
    __local uint4 data[ksY+EXTRA][THREADS];									
#endif																		

    float4 tmp_sum[1+EXTRA];
    for(int tmpint = 0; tmpint < 1+EXTRA; tmpint++)
    {
        tmp_sum[tmpint] = (float4)(0,0,0,0);
    }

#ifdef BORDER_CONSTANT														
    bool con;															
    uchar4 ss;															
    for(int j = 0;	j < ksY+EXTRA; j++)															
    {																	
        con = (startX+col >= 0 && startX+col < src_whole_cols && startY+j >= 0 && startY+j < src_whole_rows);

        int cur_col = clamp(startX + col, 0, src_whole_cols);			
        if(con)															
        {
#if LDATATYPESIZE ==4														
            ss = convert_uint4(src[(startY+j)*(src_step>>2) + cur_col]);
#elif 	LDATATYPESIZE==1													
            ss = src[(startY+j)*(src_step>>2) + cur_col];				
#endif																		
        }

        data[j][col] = con ? ss : 0;							
    }																	
#else																		
    for(int j= 0; j < ksY+EXTRA; j++)														
    {																	
        int selected_row;												
        int selected_col;												
        selected_row = ADDR_H(startY+j, 0, src_whole_rows);				
        selected_row = ADDR_B(startY+j, src_whole_rows, selected_row);	

        selected_col = ADDR_L(startX+col, 0, src_whole_cols);			
        selected_col = ADDR_R(startX+col, src_whole_cols, selected_col);

#if LDATATYPESIZE ==4														
        data[j][col] = convert_uint4(src[selected_row * (src_step>>2) + selected_col]);	
#elif 	LDATATYPESIZE==1													
        data[j][col] = src[selected_row * (src_step>>2) + selected_col];
#endif																		
    }																	
#endif																		

    barrier(CLK_LOCAL_MEM_FENCE);	

    float4 var;	

#if VAR_PER_CHANNEL
    float4 weight;
    float4 totalWeight = (float4)(0,0,0,0);
#else
    float weight;
    float totalWeight = 0;
#endif

    int4 currValCenter;
    int4 currWRTCenter;

    int4 sumVal = 0;
    int4 sumValSqr = 0;

    if(col < (THREADS-(ksX-1)))											
    {		
        int4 currVal;															

        int howManyAll = (2*anX+1)*(ksY+EXTRA);

        //find variance of all data	
        int startLMj = 0;																				
        int endLMj =  ksY+EXTRA-1;
#if CALCVAR
        // Top row: don't sum the very last element						
        for(int j = startLMj; j < endLMj; j++)							
        {																																	
            for(int i=-anX; i<=anX; i++)								
            {				
                currVal	= convert_int4(data[j][col+anX+i])	;

                sumVal += currVal;		
                sumValSqr += mul24(currVal, currVal);	
            }															
        }															
        var = convert_float4( ( (sumValSqr * howManyAll)- mul24(sumVal , sumVal) ) ) /  ( (float)(howManyAll*howManyAll) ) ;
#else
        var = (float4)(900.0, 900.0, 900.0, 0.0);
#endif

        for(int extraCnt = 0; extraCnt <= EXTRA; extraCnt++)
        {

            // top row: include the very first element, even on first time	
            startLMj = extraCnt;										
            // go all the way, unless this is the last local mem chunk,     
            // then stay within limits - 1										
            endLMj =  extraCnt + ksY;

            // Top row: don't sum the very last element		
            currValCenter = convert_int4( data[ (startLMj + endLMj)/2][col+anX] );

            for(int j = startLMj; j < endLMj; j++)							
            {																

                for(int i=-anX; i<=anX; i++)								
                {				
#if FIXED_WEIGHT
#if VAR_PER_CHANNEL
                    weight.x = 1.0f;
                    weight.y = 1.0f;
                    weight.z = 1.0f;
                    weight.w = 1.0f;
#else
                    weight = 1.0f;
#endif
#else
                    currVal	= convert_int4(data[j][col+anX+i])	;
                    currWRTCenter = currVal-currValCenter;

#if VAR_PER_CHANNEL
                    weight.x = var.x / ( var.x + (float) mul24(currWRTCenter.x , currWRTCenter.x) ) ;
                    weight.y = var.y / ( var.y + (float) mul24(currWRTCenter.y , currWRTCenter.y)  ) ;
                    weight.z = var.z / ( var.z + (float) mul24(currWRTCenter.z , currWRTCenter.z)  ) ;
                    weight.w = 0;
#else
                    weight = 1.0f/(1.0f+( mul24(currWRTCenter.x, currWRTCenter.x) + mul24(currWRTCenter.y, currWRTCenter.y) +  mul24(currWRTCenter.z, currWRTCenter.z))/(var.x+var.y+var.z));
#endif
#endif //FIXED_WEIGHT
                    tmp_sum[extraCnt] += convert_float4(data[j][col+anX+i]) * weight;
                    totalWeight += weight;
                }															
            }	

            tmp_sum[extraCnt] /= totalWeight;
            tmp_sum[extraCnt].w = 0;


            if(posX >= 0 && posX < dst_cols && (posY+extraCnt) >= 0 && (posY+extraCnt) < dst_rows)
            {										
                dst[(dst_startY+extraCnt) * (dst_step>>2)+ dst_startX + col] = convert_uchar4(tmp_sum[extraCnt]);
            }	

#if VAR_PER_CHANNEL
            totalWeight = (float4)(0,0,0,0);
#else
            totalWeight = 0;
#endif
        }																
    }																		
}		

#endif

