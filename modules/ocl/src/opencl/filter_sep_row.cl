//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
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
//

#define READ_TIMES_ROW ((2*(RADIUSX+LSIZE0)-1)/LSIZE0) //for c4 only
#define READ_TIMES_COL ((2*(RADIUSY+LSIZE1)-1)/LSIZE1)
//#pragma OPENCL EXTENSION cl_amd_printf : enable
#define RADIUS 1
#if CN ==1
#define ALIGN (((RADIUS)+3)>>2<<2)
#elif CN==2
#define ALIGN (((RADIUS)+1)>>1<<1)
#elif CN==3
#define ALIGN (((RADIUS)+3)>>2<<2)
#elif CN==4
#define ALIGN (RADIUS)
#endif

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

/**********************************************************************************
These kernels are written for separable filters such as Sobel, Scharr, GaussianBlur.
Now(6/29/2011) the kernels only support 8U data type and the anchor of the convovle
kernel must be in the center. ROI is not supported either.
For channels =1,2,4, each kernels read 4 elements(not 4 pixels), and for channels =3,
the kernel read 4 pixels, save them to LDS and read the data needed from LDS to
calculate the result.
The length of the convovle kernel supported is related to the LSIZE0 and the MAX size
of LDS, which is HW related.
For channels = 1,3 the RADIUS is no more than LSIZE0*2
For channels = 2, the RADIUS is no more than LSIZE0
For channels = 4, arbitary RADIUS is supported unless the LDS is not enough
Niko
6/29/2011
The info above maybe obsolete.
***********************************************************************************/

__kernel __attribute__((reqd_work_group_size(LSIZE0,LSIZE1,1))) void row_filter_C1_D0
    (__global uchar * restrict src,
     __global float * dst,
     int dst_cols, int dst_rows,
     int src_whole_cols, int src_whole_rows,
     int src_step_in_pixel,
     int src_offset_x, int src_offset_y,
     int dst_step_in_pixel, int radiusy,
     __constant float * mat_kernel __attribute__((max_constant_size(4*(2*RADIUSX+1)))))
{
    int x = get_global_id(0)<<2;
    int y = get_global_id(1);
    int l_x = get_local_id(0);
    int l_y = get_local_id(1);

    int start_x = x+src_offset_x - RADIUSX & 0xfffffffc;
    int offset = src_offset_x - RADIUSX & 3;
    int start_y = y + src_offset_y - radiusy;
    int start_addr = mad24(start_y, src_step_in_pixel, start_x);
    int i;
    float4 sum;
    uchar4 temp[READ_TIMES_ROW];

    __local uchar4 LDS_DAT[LSIZE1][READ_TIMES_ROW*LSIZE0+1];
#ifdef BORDER_CONSTANT
    int end_addr = mad24(src_whole_rows - 1, src_step_in_pixel, src_whole_cols);

    // read pixels from src
    for (i = 0; i < READ_TIMES_ROW; i++)
    {
        int current_addr = start_addr+i*LSIZE0*4;
        current_addr = ((current_addr < end_addr) && (current_addr > 0)) ? current_addr : 0;
        temp[i] = *(__global uchar4*)&src[current_addr];
    }

    // judge if read out of boundary
    for (i = 0; i<READ_TIMES_ROW; i++)
    {
        temp[i].x = ELEM(start_x+i*LSIZE0*4,0,src_whole_cols,0,temp[i].x);
        temp[i].y = ELEM(start_x+i*LSIZE0*4+1,0,src_whole_cols,0,temp[i].y);
        temp[i].z = ELEM(start_x+i*LSIZE0*4+2,0,src_whole_cols,0,temp[i].z);
        temp[i].w = ELEM(start_x+i*LSIZE0*4+3,0,src_whole_cols,0,temp[i].w);
        temp[i]   = ELEM(start_y,0,src_whole_rows,(uchar4)0,temp[i]);
    }
#else
    int not_all_in_range = (start_x<0) | (start_x + READ_TIMES_ROW*LSIZE0*4+4>src_whole_cols)| (start_y<0) | (start_y >= src_whole_rows);
    int4 index[READ_TIMES_ROW];
    int4 addr;
    int s_y;

    if (not_all_in_range)
    {
        // judge if read out of boundary
        for (i = 0; i < READ_TIMES_ROW; i++)
        {
            index[i] = (int4)(start_x+i*LSIZE0*4) + (int4)(0, 1, 2, 3);
            EXTRAPOLATE(index[i].x, src_whole_cols);
            EXTRAPOLATE(index[i].y, src_whole_cols);
            EXTRAPOLATE(index[i].z, src_whole_cols);
            EXTRAPOLATE(index[i].w, src_whole_cols);
        }
        s_y = start_y;
        EXTRAPOLATE(s_y, src_whole_rows);

        // read pixels from src
        for (i = 0; i<READ_TIMES_ROW; i++)
        {
            addr = mad24((int4)s_y,(int4)src_step_in_pixel,index[i]);
            temp[i].x = src[addr.x];
            temp[i].y = src[addr.y];
            temp[i].z = src[addr.z];
            temp[i].w = src[addr.w];
        }
    }
    else
    {
        // read pixels from src
        for (i = 0; i<READ_TIMES_ROW; i++)
            temp[i] = *(__global uchar4*)&src[start_addr+i*LSIZE0*4];
    }
#endif

    // save pixels to lds
    for (i = 0; i<READ_TIMES_ROW; i++)
        LDS_DAT[l_y][l_x+i*LSIZE0]=temp[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    // read pixels from lds and calculate the result
    sum =convert_float4(vload4(0,(__local uchar*)&LDS_DAT[l_y][l_x]+RADIUSX+offset))*mat_kernel[RADIUSX];
    for (i=1; i<=RADIUSX; i++)
    {
        temp[0] = vload4(0, (__local uchar*)&LDS_DAT[l_y][l_x] + RADIUSX + offset - i);
        temp[1] = vload4(0, (__local uchar*)&LDS_DAT[l_y][l_x] + RADIUSX + offset + i);
        sum += convert_float4(temp[0]) * mat_kernel[RADIUSX-i] + convert_float4(temp[1]) * mat_kernel[RADIUSX+i];
    }

    start_addr = mad24(y,dst_step_in_pixel,x);

    // write the result to dst
    if ((x+3<dst_cols) & (y<dst_rows))
        *(__global float4*)&dst[start_addr] = sum;
    else if ((x+2<dst_cols) && (y<dst_rows))
    {
        dst[start_addr] = sum.x;
        dst[start_addr+1] = sum.y;
        dst[start_addr+2] = sum.z;
    }
    else if ((x+1<dst_cols) && (y<dst_rows))
    {
        dst[start_addr] = sum.x;
        dst[start_addr+1] = sum.y;
    }
    else if (x<dst_cols && y<dst_rows)
        dst[start_addr] = sum.x;
}

__kernel __attribute__((reqd_work_group_size(LSIZE0,LSIZE1,1))) void row_filter_C4_D0
    (__global uchar4 * restrict src,
     __global float4 * dst,
     int dst_cols, int dst_rows,
     int src_whole_cols, int src_whole_rows,
     int src_step_in_pixel,
     int src_offset_x, int src_offset_y,
     int dst_step_in_pixel, int radiusy,
     __constant float * mat_kernel __attribute__((max_constant_size(4*(2*RADIUSX+1)))))
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int l_x = get_local_id(0);
    int l_y = get_local_id(1);
    int start_x = x+src_offset_x-RADIUSX;
    int start_y = y+src_offset_y-radiusy;
    int start_addr = mad24(start_y,src_step_in_pixel,start_x);
    int i;
    float4 sum;
    uchar4 temp[READ_TIMES_ROW];

    __local uchar4 LDS_DAT[LSIZE1][READ_TIMES_ROW*LSIZE0+1];
#ifdef BORDER_CONSTANT
    int end_addr = mad24(src_whole_rows - 1,src_step_in_pixel,src_whole_cols);

    // read pixels from src
    for (i = 0; i<READ_TIMES_ROW; i++)
    {
        int current_addr = start_addr+i*LSIZE0;
        current_addr = ((current_addr < end_addr) && (current_addr > 0)) ? current_addr : 0;
        temp[i] = src[current_addr];
    }

    //judge if read out of boundary
    for (i = 0; i<READ_TIMES_ROW; i++)
    {
        temp[i]= ELEM(start_x+i*LSIZE0,0,src_whole_cols,(uchar4)0,temp[i]);
        temp[i]= ELEM(start_y,0,src_whole_rows,(uchar4)0,temp[i]);
    }
#else
    int index[READ_TIMES_ROW];
    int s_x,s_y;

    // judge if read out of boundary
    for (i = 0; i<READ_TIMES_ROW; i++)
    {
        s_x = start_x+i*LSIZE0;
        EXTRAPOLATE(s_x, src_whole_cols);
        s_y = start_y;
        EXTRAPOLATE(s_y, src_whole_rows);
        index[i]=mad24(s_y,src_step_in_pixel,s_x);
    }

    //read pixels from src
    for (i = 0; i<READ_TIMES_ROW; i++)
        temp[i] = src[index[i]];
#endif

    //save pixels to lds
    for (i = 0; i<READ_TIMES_ROW; i++)
        LDS_DAT[l_y][l_x+i*LSIZE0]=temp[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    //read pixels from lds and calculate the result
    sum =convert_float4(LDS_DAT[l_y][l_x+RADIUSX])*mat_kernel[RADIUSX];
    for (i=1; i<=RADIUSX; i++)
    {
        temp[0]=LDS_DAT[l_y][l_x+RADIUSX-i];
        temp[1]=LDS_DAT[l_y][l_x+RADIUSX+i];
        sum += convert_float4(temp[0])*mat_kernel[RADIUSX-i]+convert_float4(temp[1])*mat_kernel[RADIUSX+i];
    }
    //write the result to dst
    if (x<dst_cols && y<dst_rows)
    {
        start_addr = mad24(y,dst_step_in_pixel,x);
        dst[start_addr] = sum;
    }
}

__kernel __attribute__((reqd_work_group_size(LSIZE0,LSIZE1,1))) void row_filter_C1_D5
    (__global float * restrict src,
     __global float * dst,
     int dst_cols, int dst_rows,
     int src_whole_cols, int src_whole_rows,
     int src_step_in_pixel,
     int src_offset_x, int src_offset_y,
     int dst_step_in_pixel, int radiusy,
     __constant float * mat_kernel __attribute__((max_constant_size(4*(2*RADIUSX+1)))))
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int l_x = get_local_id(0);
    int l_y = get_local_id(1);
    int start_x = x+src_offset_x-RADIUSX;
    int start_y = y+src_offset_y-radiusy;
    int start_addr = mad24(start_y,src_step_in_pixel,start_x);
    int i;
    float sum;
    float temp[READ_TIMES_ROW];

    __local float LDS_DAT[LSIZE1][READ_TIMES_ROW*LSIZE0+1];
#ifdef BORDER_CONSTANT
    int end_addr = mad24(src_whole_rows - 1,src_step_in_pixel,src_whole_cols);

    // read pixels from src
    for (i = 0; i<READ_TIMES_ROW; i++)
    {
        int current_addr = start_addr+i*LSIZE0;
        current_addr = ((current_addr < end_addr) && (current_addr > 0)) ? current_addr : 0;
        temp[i] = src[current_addr];
    }

    // judge if read out of boundary
    for (i = 0; i<READ_TIMES_ROW; i++)
    {
        temp[i]= ELEM(start_x+i*LSIZE0,0,src_whole_cols,(float)0,temp[i]);
        temp[i]= ELEM(start_y,0,src_whole_rows,(float)0,temp[i]);
    }
#else
    int index[READ_TIMES_ROW];
    int s_x,s_y;
    // judge if read out of boundary
    for (i = 0; i<READ_TIMES_ROW; i++)
    {
        s_x = start_x + i*LSIZE0, s_y = start_y;
        EXTRAPOLATE(s_x, src_whole_cols);
        EXTRAPOLATE(s_y, src_whole_rows);

        index[i]=mad24(s_y, src_step_in_pixel, s_x);
    }
    // read pixels from src
    for (i = 0; i<READ_TIMES_ROW; i++)
        temp[i] = src[index[i]];
#endif

    //save pixels to lds
    for (i = 0; i<READ_TIMES_ROW; i++)
        LDS_DAT[l_y][l_x+i*LSIZE0]=temp[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    // read pixels from lds and calculate the result
    sum =LDS_DAT[l_y][l_x+RADIUSX]*mat_kernel[RADIUSX];
    for (i=1; i<=RADIUSX; i++)
    {
        temp[0]=LDS_DAT[l_y][l_x+RADIUSX-i];
        temp[1]=LDS_DAT[l_y][l_x+RADIUSX+i];
        sum += temp[0]*mat_kernel[RADIUSX-i]+temp[1]*mat_kernel[RADIUSX+i];
    }

    // write the result to dst
    if (x<dst_cols && y<dst_rows)
    {
        start_addr = mad24(y,dst_step_in_pixel,x);
        dst[start_addr] = sum;
    }
}

__kernel __attribute__((reqd_work_group_size(LSIZE0,LSIZE1,1))) void row_filter_C4_D5
    (__global float4 * restrict src,
     __global float4 * dst,
     int dst_cols, int dst_rows,
     int src_whole_cols, int src_whole_rows,
     int src_step_in_pixel,
     int src_offset_x, int src_offset_y,
     int dst_step_in_pixel, int radiusy,
     __constant float * mat_kernel __attribute__((max_constant_size(4*(2*RADIUSX+1)))))
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int l_x = get_local_id(0);
    int l_y = get_local_id(1);
    int start_x = x+src_offset_x-RADIUSX;
    int start_y = y+src_offset_y-radiusy;
    int start_addr = mad24(start_y,src_step_in_pixel,start_x);
    int i;
    float4 sum;
    float4 temp[READ_TIMES_ROW];

    __local float4 LDS_DAT[LSIZE1][READ_TIMES_ROW*LSIZE0+1];
#ifdef BORDER_CONSTANT
    int end_addr = mad24(src_whole_rows - 1,src_step_in_pixel,src_whole_cols);

    // read pixels from src
    for (i = 0; i<READ_TIMES_ROW; i++)
    {
        int current_addr = start_addr+i*LSIZE0;
        current_addr = ((current_addr < end_addr) && (current_addr > 0)) ? current_addr : 0;
        temp[i] = src[current_addr];
    }

    // judge if read out of boundary
    for (i = 0; i<READ_TIMES_ROW; i++)
    {
        temp[i]= ELEM(start_x+i*LSIZE0,0,src_whole_cols,(float4)0,temp[i]);
        temp[i]= ELEM(start_y,0,src_whole_rows,(float4)0,temp[i]);
    }
#else
    int index[READ_TIMES_ROW];
    int s_x,s_y;

    // judge if read out of boundary
    for (i = 0; i<READ_TIMES_ROW; i++)
    {
        s_x = start_x + i*LSIZE0, s_y = start_y;
        EXTRAPOLATE(s_x, src_whole_cols);
        EXTRAPOLATE(s_y, src_whole_rows);

        index[i]=mad24(s_y,src_step_in_pixel,s_x);
    }
    // read pixels from src
    for (i = 0; i<READ_TIMES_ROW; i++)
        temp[i] = src[index[i]];
#endif

    // save pixels to lds
    for (i = 0; i<READ_TIMES_ROW; i++)
        LDS_DAT[l_y][l_x+i*LSIZE0]=temp[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    // read pixels from lds and calculate the result
    sum =LDS_DAT[l_y][l_x+RADIUSX]*mat_kernel[RADIUSX];
    for (i=1; i<=RADIUSX; i++)
    {
        temp[0]=LDS_DAT[l_y][l_x+RADIUSX-i];
        temp[1]=LDS_DAT[l_y][l_x+RADIUSX+i];
        sum += temp[0]*mat_kernel[RADIUSX-i]+temp[1]*mat_kernel[RADIUSX+i];
    }

    // write the result to dst
    if (x<dst_cols && y<dst_rows)
    {
        start_addr = mad24(y,dst_step_in_pixel,x);
        dst[start_addr] = sum;
    }
}
