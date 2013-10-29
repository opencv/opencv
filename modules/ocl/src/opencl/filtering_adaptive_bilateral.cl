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
// Copyright (C) 2010-2013, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Harris Gasparakis, harris.gasparakis@amd.com
//    Xiaopeng Fu, fuxiaopeng2222@163.com
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

#ifdef BORDER_CONSTANT
#define ELEM(i,l_edge,r_edge,elem1,elem2) (i)<(l_edge) | (i) >= (r_edge) ? (elem1) : (elem2)
#elif defined BORDER_REPLICATE
#define EXTRAPOLATE(x, maxV) \
    { \
        x = max(min(x, maxV - 1), 0); \
    }
#elif defined BORDER_WRAP
#define EXTRAPOLATE(x, maxV) \
    { \
        if (x < 0) \
            x -= ((x - maxV + 1) / maxV) * maxV; \
        if (x >= maxV) \
            x %= maxV; \
    }
#elif defined(BORDER_REFLECT) || defined(BORDER_REFLECT_101)
#define EXTRAPOLATE_(x, maxV, delta) \
    { \
        if (maxV == 1) \
            x = 0; \
        else \
            do \
            { \
                if ( x < 0 ) \
                    x = -x - 1 + delta; \
                else \
                    x = maxV - 1 - (x - maxV) - delta; \
            } \
            while (x >= maxV || x < 0); \
    }
#ifdef BORDER_REFLECT
#define EXTRAPOLATE(x, maxV) EXTRAPOLATE_(x, maxV, 0)
#else
#define EXTRAPOLATE(x, maxV) EXTRAPOLATE_(x, maxV, 1)
#endif
#else
#error No extrapolation method
#endif

__kernel void
edgeEnhancingFilter_C4_D0(
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
    int dst_step,
    __global const float* lut,
    int lut_step)
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

    __local uchar4 data[ksY+EXTRA][THREADS];

    float4 tmp_sum[1+EXTRA];
    for(int tmpint = 0; tmpint < 1+EXTRA; tmpint++)
        tmp_sum[tmpint] = (float4)(0,0,0,0);

#ifdef BORDER_CONSTANT
    bool con;
    uchar4 ss;
    for(int j = 0;	j < ksY+EXTRA; j++)
    {
        con = (startX+col >= 0 && startX+col < src_whole_cols && startY+j >= 0 && startY+j < src_whole_rows);
        int cur_col = clamp(startX + col, 0, src_whole_cols);
        if (con)
            ss = src[(startY+j)*(src_step>>2) + cur_col];

        data[j][col] = con ? ss : (uchar4)0;
    }
#else
    for(int j= 0; j < ksY+EXTRA; j++)
    {
        int selected_row = startY+j, selected_col = startX+col;
        EXTRAPOLATE(selected_row, src_whole_rows)
        EXTRAPOLATE(selected_col, src_whole_cols)

        data[j][col] = src[selected_row * (src_step>>2) + selected_col];
    }
#endif

    barrier(CLK_LOCAL_MEM_FENCE);

    float4 var[1+EXTRA];

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
        int howManyAll = (2*anX+1)*(ksY);

        //find variance of all data
        int startLMj;
        int endLMj ;
#if CALCVAR
        // Top row: don't sum the very last element
        for(int extraCnt = 0; extraCnt <=EXTRA; extraCnt++)
        {
            startLMj = extraCnt;
            endLMj =  ksY+extraCnt-1;
            sumVal =0;
            sumValSqr=0;
            for(int j = startLMj; j < endLMj; j++)
                for(int i=-anX; i<=anX; i++)
                {
                    currVal = convert_int4(data[j][col+anX+i]);

                    sumVal += currVal;
                    sumValSqr += mul24(currVal, currVal);
                }

            var[extraCnt] = convert_float4( ( (sumValSqr * howManyAll)- mul24(sumVal , sumVal) ) ) /  ( (float)(howManyAll*howManyAll) ) ;
#else
        var[extraCnt] = (float4)(900.0, 900.0, 900.0, 0.0);
#endif
        }

        for(int extraCnt = 0; extraCnt <= EXTRA; extraCnt++)
        {

            // top row: include the very first element, even on first time
            startLMj = extraCnt;
            // go all the way, unless this is the last local mem chunk,
            // then stay within limits - 1
            endLMj =  extraCnt + ksY;

            // Top row: don't sum the very last element
            currValCenter = convert_int4( data[ (startLMj + endLMj)/2][col+anX] );

            for(int j = startLMj, lut_j = 0; j < endLMj; j++, lut_j++)
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
                    currVal = convert_int4(data[j][col+anX+i]);
                    currWRTCenter = currVal-currValCenter;

#if VAR_PER_CHANNEL
                    weight = var[extraCnt] / (var[extraCnt] + convert_float4(currWRTCenter * currWRTCenter)) *
                        (float4)(lut[lut_j*lut_step+anX+i]);
#else
                    weight = 1.0f/(1.0f+( mul24(currWRTCenter.x, currWRTCenter.x) + mul24(currWRTCenter.y, currWRTCenter.y) +
                        mul24(currWRTCenter.z, currWRTCenter.z))/(var.x+var.y+var.z));
#endif
#endif
                    tmp_sum[extraCnt] += convert_float4(data[j][col+anX+i]) * weight;
                    totalWeight += weight;
                }
            }

            tmp_sum[extraCnt] /= totalWeight;

            if(posX >= 0 && posX < dst_cols && (posY+extraCnt) >= 0 && (posY+extraCnt) < dst_rows)
                dst[(dst_startY+extraCnt) * (dst_step>>2)+ dst_startX + col] = convert_uchar4(tmp_sum[extraCnt]);

#if VAR_PER_CHANNEL
            totalWeight = (float4)(0,0,0,0);
#else
            totalWeight = 0;
#endif
        }
    }
}


__kernel void
edgeEnhancingFilter_C1_D0(
    __global const uchar * restrict src,
    __global uchar *dst,
    float alpha,
    int src_offset,
    int src_whole_rows,
    int src_whole_cols,
    int src_step,
    int dst_offset,
    int dst_rows,
    int dst_cols,
    int dst_step,
    __global const float * lut,
    int lut_step)
{
    int col = get_local_id(0);
    const int gX = get_group_id(0);
    const int gY = get_group_id(1);

    int src_x_off = (src_offset % src_step);
    int src_y_off = src_offset / src_step;
    int dst_x_off = (dst_offset % dst_step);
    int dst_y_off = dst_offset / dst_step;

    int startX = gX * (THREADS-ksX+1) - anX + src_x_off;
    int startY = (gY * (1+EXTRA)) - anY + src_y_off;

    int dst_startX = gX * (THREADS-ksX+1) + dst_x_off;
    int dst_startY = (gY * (1+EXTRA)) + dst_y_off;

    int posX = dst_startX - dst_x_off + col;
    int posY = (gY * (1+EXTRA))	;

    __local uchar data[ksY+EXTRA][THREADS];

    float tmp_sum[1+EXTRA];
    for(int tmpint = 0; tmpint < 1+EXTRA; tmpint++)
    {
        tmp_sum[tmpint] = (float)(0);
    }

#ifdef BORDER_CONSTANT
    bool con;
    uchar ss;
    for(int j = 0;	j < ksY+EXTRA; j++)
    {
        con = (startX+col >= 0 && startX+col < src_whole_cols && startY+j >= 0 && startY+j < src_whole_rows);

        int cur_col = clamp(startX + col, 0, src_whole_cols);
        if(con)
        {
            ss = src[(startY+j)*(src_step) + cur_col];
        }

        data[j][col] = con ? ss : 0;
    }
#else
    for(int j= 0; j < ksY+EXTRA; j++)
    {
        int selected_row = startY+j, selected_col = startX+col;
        EXTRAPOLATE(selected_row, src_whole_rows)
        EXTRAPOLATE(selected_col, src_whole_cols)

        data[j][col] = src[selected_row * (src_step) + selected_col];
    }
#endif

    barrier(CLK_LOCAL_MEM_FENCE);

    float var[1+EXTRA];

    float weight;
    float totalWeight = 0;

    int currValCenter;
    int currWRTCenter;

    int sumVal = 0;
    int sumValSqr = 0;

    if(col < (THREADS-(ksX-1)))
    {
        int currVal;

        int howManyAll = (2*anX+1)*(ksY);

        //find variance of all data
        int startLMj;
        int endLMj;
#if CALCVAR
        // Top row: don't sum the very last element
        for(int extraCnt=0; extraCnt<=EXTRA; extraCnt++)
        {
            startLMj = extraCnt;
            endLMj =  ksY+extraCnt-1;
            sumVal = 0;
            sumValSqr =0;
            for(int j = startLMj; j < endLMj; j++)
            {
                for(int i=-anX; i<=anX; i++)
                {
                    currVal	= (uint)(data[j][col+anX+i])	;

                    sumVal += currVal;
                    sumValSqr += mul24(currVal, currVal);
                }
            }
            var[extraCnt] = (float)( ( (sumValSqr * howManyAll)- mul24(sumVal , sumVal) ) ) /  ( (float)(howManyAll*howManyAll) ) ;
#else
        var[extraCnt] = (float)(900.0);
#endif
        }

        for(int extraCnt = 0; extraCnt <= EXTRA; extraCnt++)
        {

            // top row: include the very first element, even on first time
            startLMj = extraCnt;
            // go all the way, unless this is the last local mem chunk,
            // then stay within limits - 1
            endLMj =  extraCnt + ksY;

            // Top row: don't sum the very last element
            currValCenter = (int)( data[ (startLMj + endLMj)/2][col+anX] );

            for(int j = startLMj, lut_j = 0; j < endLMj; j++, lut_j++)
            {
                for(int i=-anX; i<=anX; i++)
                {
#if FIXED_WEIGHT
                    weight = 1.0f;
#else
                    currVal	= (int)(data[j][col+anX+i])	;
                    currWRTCenter = currVal-currValCenter;

                    weight = var[extraCnt] / (var[extraCnt] + (float)mul24(currWRTCenter,currWRTCenter)) * lut[lut_j*lut_step+anX+i] ;
#endif
                    tmp_sum[extraCnt] += (float)(data[j][col+anX+i] * weight);
                    totalWeight += weight;
                }
            }

            tmp_sum[extraCnt] /= totalWeight;


            if(posX >= 0 && posX < dst_cols && (posY+extraCnt) >= 0 && (posY+extraCnt) < dst_rows)
            {
                dst[(dst_startY+extraCnt) * (dst_step)+ dst_startX + col] = (uchar)(tmp_sum[extraCnt]);
            }

            totalWeight = 0;
        }
    }
}
