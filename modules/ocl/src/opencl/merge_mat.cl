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
// @Authors
//    Jia Haipeng, jiahaipeng95@gmail.com
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

#if defined (DOUBLE_SUPPORT)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif


///////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////optimized code using vector roi//////////////////////////
////////////vector fuction name format: merge_vector_C(channels number)D_(data type depth)//////
////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void merge_vector_C2_D0(__global uchar *mat_dst,  int dst_step,  int dst_offset,
                                 __global uchar *mat_src0, int src0_step, int src0_offset,
                                 __global uchar *mat_src1, int src1_step, int src1_offset,
                                 int rows, int cols, int dst_step1)

{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        x = x << 1;

        #define dst_align  ((dst_offset & 3) >> 1)
        int src0_index = mad24(y, src0_step, src0_offset + x - dst_align);
        int src1_index = mad24(y, src1_step, src1_offset + x - dst_align);

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + (x << 1) & (int)0xfffffffc);

        __global uchar4 * dst  = (__global uchar4 *)(mat_dst + dst_index);
        __global uchar  * src0 = mat_src0 + src0_index;
        __global uchar  * src1 = src0     + 1;
        __global uchar  * src2 = mat_src1 + src1_index;
        __global uchar  * src3 = src2     + 1;

        uchar4 dst_data = *dst;
        uchar  data_0   = *(src0);
        uchar  data_1   = *(src1);
        uchar  data_2   = *(src2);
        uchar  data_3   = *(src3);

        uchar4 tmp_data = (uchar4)(data_0, data_2, data_1, data_3);

        tmp_data.xy = dst_index + 0 >= dst_start ? tmp_data.xy : dst_data.xy;
        tmp_data.zw = dst_index + 2 <  dst_end   ? tmp_data.zw : dst_data.zw;

        *dst = tmp_data;
    }
}
__kernel void merge_vector_C2_D1(__global char *mat_dst,  int dst_step,  int dst_offset,
                                 __global char *mat_src0, int src0_step, int src0_offset,
                                 __global char *mat_src1, int src1_step, int src1_offset,
                                 int rows, int cols, int dst_step1)

{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        x = x << 1;

        #define dst_align  ((dst_offset & 3) >> 1)
        int src0_index = mad24(y, src0_step, src0_offset + x - dst_align);
        int src1_index = mad24(y, src1_step, src1_offset + x - dst_align);

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + (x << 1) & (int)0xfffffffc);

        __global char4 * dst  = (__global char4 *)(mat_dst + dst_index);
        __global char  * src0 = mat_src0 + src0_index;
        __global char  * src1 = src0     + 1;
        __global char  * src2 = mat_src1 + src1_index;
        __global char  * src3 = src2     + 1;

        char4 dst_data = *dst;
        char  data_0   = *(src0);
        char  data_1   = *(src1);
        char  data_2   = *(src2);
        char  data_3   = *(src3);

        char4 tmp_data = (char4)(data_0, data_2, data_1, data_3);

        tmp_data.xy = dst_index + 0 >= dst_start ? tmp_data.xy : dst_data.xy;
        tmp_data.zw = dst_index + 2 <  dst_end   ? tmp_data.zw : dst_data.zw;

        *dst = tmp_data;
    }
}
__kernel void merge_vector_C2_D2(__global ushort *mat_dst,  int dst_step,  int dst_offset,
                                 __global ushort *mat_src0, int src0_step, int src0_offset,
                                 __global ushort *mat_src1, int src1_step, int src1_offset,
                                 int rows, int cols, int dst_step1)

{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        int src0_index = mad24(y, src0_step, src0_offset);
        int src1_index = mad24(y, src1_step, src1_offset);

        int dst_index  = mad24(y, dst_step , dst_offset);

        __global ushort*  src0 = (__global ushort * )((__global uchar *)mat_src0 + src0_index + (x << 1));
        __global ushort*  src1 = (__global ushort * )((__global uchar *)mat_src1 + src1_index + (x << 1));
        __global ushort2* dist = (__global ushort2 *)((__global uchar *)mat_dst  + dst_index  + (x << 2));

        ushort  src0_data = *src0;
        ushort  src1_data = *src1;

        *dist = (ushort2)(src0_data, src1_data);

    }
}
__kernel void merge_vector_C2_D3(__global short *mat_dst,  int dst_step,  int dst_offset,
                                 __global short *mat_src0, int src0_step, int src0_offset,
                                 __global short *mat_src1, int src1_step, int src1_offset,
                                 int rows, int cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        int src0_index = mad24(y, src0_step, src0_offset);
        int src1_index = mad24(y, src1_step, src1_offset);

        int dst_index  = mad24(y, dst_step , dst_offset);

        __global short*  src0 = (__global short * )((__global uchar *)mat_src0 + src0_index + (x << 1));
        __global short*  src1 = (__global short * )((__global uchar *)mat_src1 + src1_index + (x << 1));
        __global short2* dist = (__global short2 *)((__global uchar *)mat_dst  + dst_index   + (x << 2));

        short  src0_data = *src0;
        short  src1_data = *src1;

        *dist = (short2)(src0_data, src1_data);
    }
}

__kernel void merge_vector_C2_D4(__global int *mat_dst,  int dst_step,  int dst_offset,
                                 __global int *mat_src0, int src0_step, int src0_offset,
                                 __global int *mat_src1, int src1_step, int src1_offset,
                                 int rows, int cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        int src0_index = mad24(y, src0_step, src0_offset);
        int src1_index = mad24(y, src1_step, src1_offset);
        int dst_index  = mad24(y, dst_step , dst_offset);

        int src0 = *((__global int *)((__global uchar *)mat_src0 + src0_index + (x << 2)));
        int src1 = *((__global int *)((__global uchar *)mat_src1 + src1_index + (x << 2)));

        *((__global int2 *)((__global uchar *)mat_dst  + dst_index + (x << 3))) = (int2)(src0, src1);
    }
}
__kernel void merge_vector_C2_D5(__global float *mat_dst,  int dst_step,  int dst_offset,
                                 __global float *mat_src0, int src0_step, int src0_offset,
                                 __global float *mat_src1, int src1_step, int src1_offset,
                                 int rows, int cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        int src0_index = mad24(y, src0_step, src0_offset);
        int src1_index = mad24(y, src1_step, src1_offset);
        int dst_index  = mad24(y, dst_step , dst_offset);

        float src0 = *((__global float *)((__global uchar *)mat_src0 + src0_index + (x << 2)));
        float src1 = *((__global float *)((__global uchar *)mat_src1 + src1_index + (x << 2)));

        *((__global float2 *)((__global uchar *)mat_dst  + dst_index + (x << 3))) = (float2)(src0, src1);
    }
}

#if defined (DOUBLE_SUPPORT)
__kernel void merge_vector_C2_D6(__global double *mat_dst,  int dst_step,  int dst_offset,
                                 __global double *mat_src0, int src0_step, int src0_offset,
                                 __global double *mat_src1, int src1_step, int src1_offset,
                                 int rows, int cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        int src0_index = mad24(y, src0_step, src0_offset);
        int src1_index = mad24(y, src1_step, src1_offset);
        int dst_index  = mad24(y, dst_step , dst_offset);

        double src0 = *((__global double *)((__global uchar *)mat_src0 + src0_index + (x << 3)));
        double src1 = *((__global double *)((__global uchar *)mat_src1 + src1_index + (x << 3)));

        *((__global double2 *)((__global uchar *)mat_dst  + dst_index + (x << 4))) = (double2)(src0, src1);
    }
}
#endif

__kernel void merge_vector_C3_D0(__global uchar *mat_dst,  int dst_step,  int dst_offset,
                                 __global uchar *mat_src0, int src0_step, int src0_offset,
                                 __global uchar *mat_src1, int src1_step, int src1_offset,
                                 __global uchar *mat_src2, int src2_step, int src2_offset, int offset_cols,
                                 int rows, int cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        x = x << 2;

        int src0_index = mad24(y, src0_step, x + src0_offset - offset_cols);
        int src1_index = mad24(y, src1_step, x + src1_offset - offset_cols);
        int src2_index = mad24(y, src2_step, x + src2_offset - offset_cols);

        int dst_start = mad24(y, dst_step, dst_offset);
        int dst_end   = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index = mad24(y, dst_step, dst_offset + 3 * x - offset_cols * 3);

        uchar data0_0 = *(mat_src0 + src0_index + 0);
        uchar data0_1 = *(mat_src0 + src0_index + 1);
        uchar data0_2 = *(mat_src0 + src0_index + 2);
        uchar data0_3 = *(mat_src0 + src0_index + 3);

        uchar data1_0 = *(mat_src1 + src1_index + 0);
        uchar data1_1 = *(mat_src1 + src1_index + 1);
        uchar data1_2 = *(mat_src1 + src1_index + 2);
        uchar data1_3 = *(mat_src1 + src1_index + 3);

        uchar data2_0 = *(mat_src2 + src2_index + 0);
        uchar data2_1 = *(mat_src2 + src2_index + 1);
        uchar data2_2 = *(mat_src2 + src2_index + 2);
        uchar data2_3 = *(mat_src2 + src2_index + 3);

        uchar4 tmp_data0 = (uchar4)(data0_0, data1_0, data2_0, data0_1);
        uchar4 tmp_data1 = (uchar4)(data1_1, data2_1, data0_2, data1_2);
        uchar4 tmp_data2 = (uchar4)(data2_2, data0_3, data1_3, data2_3);

        uchar4 dst_data0 = *((__global uchar4*)(mat_dst + dst_index + 0));
        uchar4 dst_data1 = *((__global uchar4*)(mat_dst + dst_index + 4));
        uchar4 dst_data2 = *((__global uchar4*)(mat_dst + dst_index + 8));

        tmp_data0.x = ((dst_index + 0  >= dst_start) && (dst_index + 0  < dst_end)) ? tmp_data0.x : dst_data0.x;
        tmp_data0.y = ((dst_index + 1  >= dst_start) && (dst_index + 1  < dst_end)) ? tmp_data0.y : dst_data0.y;
        tmp_data0.z = ((dst_index + 2  >= dst_start) && (dst_index + 2  < dst_end)) ? tmp_data0.z : dst_data0.z;
        tmp_data0.w = ((dst_index + 3  >= dst_start) && (dst_index + 3  < dst_end)) ? tmp_data0.w : dst_data0.w;

        tmp_data1.x = ((dst_index + 4  >= dst_start) && (dst_index + 4  < dst_end)) ? tmp_data1.x : dst_data1.x;
        tmp_data1.y = ((dst_index + 5  >= dst_start) && (dst_index + 5  < dst_end)) ? tmp_data1.y : dst_data1.y;
        tmp_data1.z = ((dst_index + 6  >= dst_start) && (dst_index + 6  < dst_end)) ? tmp_data1.z : dst_data1.z;
        tmp_data1.w = ((dst_index + 7  >= dst_start) && (dst_index + 7  < dst_end)) ? tmp_data1.w : dst_data1.w;

        tmp_data2.x = ((dst_index + 8  >= dst_start) && (dst_index + 8  < dst_end)) ? tmp_data2.x : dst_data2.x;
        tmp_data2.y = ((dst_index + 9  >= dst_start) && (dst_index + 9  < dst_end)) ? tmp_data2.y : dst_data2.y;
        tmp_data2.z = ((dst_index + 10 >= dst_start) && (dst_index + 10 < dst_end)) ? tmp_data2.z : dst_data2.z;
        tmp_data2.w = ((dst_index + 11 >= dst_start) && (dst_index + 11 < dst_end)) ? tmp_data2.w : dst_data2.w;

        *((__global uchar4*)(mat_dst + dst_index + 0)) = tmp_data0;
        *((__global uchar4*)(mat_dst + dst_index + 4)) = tmp_data1;
        *((__global uchar4*)(mat_dst + dst_index + 8)) = tmp_data2;
    }
}
__kernel void merge_vector_C3_D1(__global char *mat_dst,  int dst_step,  int dst_offset,
                                 __global char *mat_src0, int src0_step, int src0_offset,
                                 __global char *mat_src1, int src1_step, int src1_offset,
                                 __global char *mat_src2, int src2_step, int src2_offset, int offset_cols,
                                 int rows, int cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        x = x << 2;

        int src0_index = mad24(y, src0_step, x + src0_offset - offset_cols);
        int src1_index = mad24(y, src1_step, x + src1_offset - offset_cols);
        int src2_index = mad24(y, src2_step, x + src2_offset - offset_cols);

        int dst_start = mad24(y, dst_step, dst_offset);
        int dst_end   = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index = mad24(y, dst_step, dst_offset + 3 * x - offset_cols * 3);

        char data0_0 = *(mat_src0 + src0_index + 0);
        char data0_1 = *(mat_src0 + src0_index + 1);
        char data0_2 = *(mat_src0 + src0_index + 2);
        char data0_3 = *(mat_src0 + src0_index + 3);

        char data1_0 = *(mat_src1 + src1_index + 0);
        char data1_1 = *(mat_src1 + src1_index + 1);
        char data1_2 = *(mat_src1 + src1_index + 2);
        char data1_3 = *(mat_src1 + src1_index + 3);

        char data2_0 = *(mat_src2 + src2_index + 0);
        char data2_1 = *(mat_src2 + src2_index + 1);
        char data2_2 = *(mat_src2 + src2_index + 2);
        char data2_3 = *(mat_src2 + src2_index + 3);

        char4 tmp_data0 = (char4)(data0_0, data1_0, data2_0, data0_1);
        char4 tmp_data1 = (char4)(data1_1, data2_1, data0_2, data1_2);
        char4 tmp_data2 = (char4)(data2_2, data0_3, data1_3, data2_3);

        char4 dst_data0 = *((__global char4*)(mat_dst + dst_index + 0));
        char4 dst_data1 = *((__global char4*)(mat_dst + dst_index + 4));
        char4 dst_data2 = *((__global char4*)(mat_dst + dst_index + 8));

        tmp_data0.x = ((dst_index + 0  >= dst_start) && (dst_index + 0  < dst_end)) ? tmp_data0.x : dst_data0.x;
        tmp_data0.y = ((dst_index + 1  >= dst_start) && (dst_index + 1  < dst_end)) ? tmp_data0.y : dst_data0.y;
        tmp_data0.z = ((dst_index + 2  >= dst_start) && (dst_index + 2  < dst_end)) ? tmp_data0.z : dst_data0.z;
        tmp_data0.w = ((dst_index + 3  >= dst_start) && (dst_index + 3  < dst_end)) ? tmp_data0.w : dst_data0.w;

        tmp_data1.x = ((dst_index + 4  >= dst_start) && (dst_index + 4  < dst_end)) ? tmp_data1.x : dst_data1.x;
        tmp_data1.y = ((dst_index + 5  >= dst_start) && (dst_index + 5  < dst_end)) ? tmp_data1.y : dst_data1.y;
        tmp_data1.z = ((dst_index + 6  >= dst_start) && (dst_index + 6  < dst_end)) ? tmp_data1.z : dst_data1.z;
        tmp_data1.w = ((dst_index + 7  >= dst_start) && (dst_index + 7  < dst_end)) ? tmp_data1.w : dst_data1.w;

        tmp_data2.x = ((dst_index + 8  >= dst_start) && (dst_index + 8  < dst_end)) ? tmp_data2.x : dst_data2.x;
        tmp_data2.y = ((dst_index + 9  >= dst_start) && (dst_index + 9  < dst_end)) ? tmp_data2.y : dst_data2.y;
        tmp_data2.z = ((dst_index + 10 >= dst_start) && (dst_index + 10 < dst_end)) ? tmp_data2.z : dst_data2.z;
        tmp_data2.w = ((dst_index + 11 >= dst_start) && (dst_index + 11 < dst_end)) ? tmp_data2.w : dst_data2.w;

        *((__global char4*)(mat_dst + dst_index + 0)) = tmp_data0;
        *((__global char4*)(mat_dst + dst_index + 4)) = tmp_data1;
        *((__global char4*)(mat_dst + dst_index + 8)) = tmp_data2;
    }
}
__kernel void merge_vector_C3_D2(__global ushort *mat_dst,  int dst_step,  int dst_offset,
                                 __global ushort *mat_src0, int src0_step, int src0_offset,
                                 __global ushort *mat_src1, int src1_step, int src1_offset,
                                 __global ushort *mat_src2, int src2_step, int src2_offset, int offset_cols,
                                 int rows, int cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        x = x << 1;

        int src0_index = mad24(y, src0_step, (x << 1) + src0_offset - offset_cols);
        int src1_index = mad24(y, src1_step, (x << 1) + src1_offset - offset_cols);
        int src2_index = mad24(y, src2_step, (x << 1) + src2_offset - offset_cols);

        int dst_start = mad24(y, dst_step, dst_offset);
        int dst_end   = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index = mad24(y, dst_step, dst_offset + 6 * x - offset_cols * 6);

        ushort data0_0 = *((__global ushort *)((__global char *)mat_src0 + src0_index + 0));
        ushort data0_1 = *((__global ushort *)((__global char *)mat_src0 + src0_index + 2));

        ushort data1_0 = *((__global ushort *)((__global char *)mat_src1 + src1_index + 0));
        ushort data1_1 = *((__global ushort *)((__global char *)mat_src1 + src1_index + 2));

        ushort data2_0 = *((__global ushort *)((__global char *)mat_src2 + src2_index + 0));
        ushort data2_1 = *((__global ushort *)((__global char *)mat_src2 + src2_index + 2));

        ushort2 tmp_data0 = (ushort2)(data0_0, data1_0);
        ushort2 tmp_data1 = (ushort2)(data2_0, data0_1);
        ushort2 tmp_data2 = (ushort2)(data1_1, data2_1);

        ushort2 dst_data0 = *((__global ushort2*)((__global char *)mat_dst + dst_index + 0));
        ushort2 dst_data1 = *((__global ushort2*)((__global char *)mat_dst + dst_index + 4));
        ushort2 dst_data2 = *((__global ushort2*)((__global char *)mat_dst + dst_index + 8));

        tmp_data0.x = ((dst_index + 0  >= dst_start) && (dst_index + 0  < dst_end)) ? tmp_data0.x : dst_data0.x;
        tmp_data0.y = ((dst_index + 2  >= dst_start) && (dst_index + 2  < dst_end)) ? tmp_data0.y : dst_data0.y;

        tmp_data1.x = ((dst_index + 4  >= dst_start) && (dst_index + 4  < dst_end)) ? tmp_data1.x : dst_data1.x;
        tmp_data1.y = ((dst_index + 6  >= dst_start) && (dst_index + 6  < dst_end)) ? tmp_data1.y : dst_data1.y;

        tmp_data2.x = ((dst_index + 8  >= dst_start) && (dst_index + 8  < dst_end)) ? tmp_data2.x : dst_data2.x;
        tmp_data2.y = ((dst_index + 10 >= dst_start) && (dst_index + 10 < dst_end)) ? tmp_data2.y : dst_data2.y;

        *((__global ushort2*)((__global char *)mat_dst + dst_index + 0)) = tmp_data0;
        *((__global ushort2*)((__global char *)mat_dst + dst_index + 4)) = tmp_data1;
        *((__global ushort2*)((__global char *)mat_dst + dst_index + 8)) = tmp_data2;
    }
}
__kernel void merge_vector_C3_D3(__global short *mat_dst,  int dst_step,  int dst_offset,
                                 __global short *mat_src0, int src0_step, int src0_offset,
                                 __global short *mat_src1, int src1_step, int src1_offset,
                                 __global short *mat_src2, int src2_step, int src2_offset, int offset_cols,
                                 int rows, int cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        x = x << 1;

        int src0_index = mad24(y, src0_step, (x << 1) + src0_offset - offset_cols);
        int src1_index = mad24(y, src1_step, (x << 1) + src1_offset - offset_cols);
        int src2_index = mad24(y, src2_step, (x << 1) + src2_offset - offset_cols);

        int dst_start = mad24(y, dst_step, dst_offset);
        int dst_end   = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index = mad24(y, dst_step, dst_offset + 6 * x - offset_cols * 6);

        short data0_0 = *((__global short *)((__global char *)mat_src0 + src0_index + 0));
        short data0_1 = *((__global short *)((__global char *)mat_src0 + src0_index + 2));

        short data1_0 = *((__global short *)((__global char *)mat_src1 + src1_index + 0));
        short data1_1 = *((__global short *)((__global char *)mat_src1 + src1_index + 2));

        short data2_0 = *((__global short *)((__global char *)mat_src2 + src2_index + 0));
        short data2_1 = *((__global short *)((__global char *)mat_src2 + src2_index + 2));

        short2 tmp_data0 = (short2)(data0_0, data1_0);
        short2 tmp_data1 = (short2)(data2_0, data0_1);
        short2 tmp_data2 = (short2)(data1_1, data2_1);

        short2 dst_data0 = *((__global short2*)((__global char *)mat_dst + dst_index + 0));
        short2 dst_data1 = *((__global short2*)((__global char *)mat_dst + dst_index + 4));
        short2 dst_data2 = *((__global short2*)((__global char *)mat_dst + dst_index + 8));

        tmp_data0.x = ((dst_index + 0  >= dst_start) && (dst_index + 0  < dst_end)) ? tmp_data0.x : dst_data0.x;
        tmp_data0.y = ((dst_index + 2  >= dst_start) && (dst_index + 2  < dst_end)) ? tmp_data0.y : dst_data0.y;

        tmp_data1.x = ((dst_index + 4  >= dst_start) && (dst_index + 4  < dst_end)) ? tmp_data1.x : dst_data1.x;
        tmp_data1.y = ((dst_index + 6  >= dst_start) && (dst_index + 6  < dst_end)) ? tmp_data1.y : dst_data1.y;

        tmp_data2.x = ((dst_index + 8  >= dst_start) && (dst_index + 8  < dst_end)) ? tmp_data2.x : dst_data2.x;
        tmp_data2.y = ((dst_index + 10 >= dst_start) && (dst_index + 10 < dst_end)) ? tmp_data2.y : dst_data2.y;

        *((__global short2*)((__global char *)mat_dst + dst_index + 0)) = tmp_data0;
        *((__global short2*)((__global char *)mat_dst + dst_index + 4)) = tmp_data1;
        *((__global short2*)((__global char *)mat_dst + dst_index + 8)) = tmp_data2;
    }
}
__kernel void merge_vector_C3_D4(__global int *mat_dst,  int dst_step,  int dst_offset,
                                 __global int *mat_src0, int src0_step, int src0_offset,
                                 __global int *mat_src1, int src1_step, int src1_offset,
                                 __global int *mat_src2, int src2_step, int src2_offset, int offset_cols,
                                 int rows, int cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        int src0_index = mad24(y, src0_step, src0_offset);
        int src1_index = mad24(y, src1_step, src1_offset);
        int src2_index = mad24(y, src2_step, src2_offset);

        int dst_index  = mad24(y, dst_step , dst_offset);

        __global int* src0 = (__global int * )((__global uchar *)mat_src0 + src0_index + (x << 2));
        __global int* src1 = (__global int * )((__global uchar *)mat_src1 + src1_index + (x << 2));
        __global int* src2 = (__global int * )((__global uchar *)mat_src2 + src2_index + (x << 2));

        __global int* dist0 = (__global int *)((__global uchar *)mat_dst  + dst_index  + 3 * (x << 2));
        __global int* dist1 = dist0 + 1;
        __global int* dist2 = dist0 + 2;

        int  src0_data = *src0;
        int  src1_data = *src1;
        int  src2_data = *src2;

        *dist0 = src0_data;
        *dist1 = src1_data;
        *dist2 = src2_data;
    }
}
__kernel void merge_vector_C3_D5(__global float *mat_dst,  int dst_step,  int dst_offset,
                                 __global float *mat_src0, int src0_step, int src0_offset,
                                 __global float *mat_src1, int src1_step, int src1_offset,
                                 __global float *mat_src2, int src2_step, int src2_offset, int offset_cols,
                                 int rows, int cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        int src0_index = mad24(y, src0_step, src0_offset);
        int src1_index = mad24(y, src1_step, src1_offset);
        int src2_index = mad24(y, src2_step, src2_offset);

        int dst_index  = mad24(y, dst_step , dst_offset);

        __global float* src0 = (__global float * )((__global uchar *)mat_src0 + src0_index + (x << 2));
        __global float* src1 = (__global float * )((__global uchar *)mat_src1 + src1_index + (x << 2));
        __global float* src2 = (__global float * )((__global uchar *)mat_src2 + src2_index + (x << 2));

        __global float* dist0 = (__global float *)((__global uchar *)mat_dst  + dst_index  + 3 * (x << 2));
        __global float* dist1 = dist0 + 1;
        __global float* dist2 = dist0 + 2;

        float  src0_data = *src0;
        float  src1_data = *src1;
        float  src2_data = *src2;

        *dist0 = src0_data;
        *dist1 = src1_data;
        *dist2 = src2_data;
    }
}

#if defined (DOUBLE_SUPPORT)
__kernel void merge_vector_C3_D6(__global double *mat_dst,  int dst_step,  int dst_offset,
                                 __global double *mat_src0, int src0_step, int src0_offset,
                                 __global double *mat_src1, int src1_step, int src1_offset,
                                 __global double *mat_src2, int src2_step, int src2_offset, int offset_cols,
                                 int rows, int cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        int src0_index = mad24(y, src0_step, src0_offset);
        int src1_index = mad24(y, src1_step, src1_offset);
        int src2_index = mad24(y, src2_step, src2_offset);

        int dst_index  = mad24(y, dst_step , dst_offset);

        __global double* src0 = (__global double * )((__global uchar *)mat_src0 + src0_index + (x << 3));
        __global double* src1 = (__global double * )((__global uchar *)mat_src1 + src1_index + (x << 3));
        __global double* src2 = (__global double * )((__global uchar *)mat_src2 + src2_index + (x << 3));

        __global double* dist0 = (__global double *)((__global uchar *)mat_dst  + dst_index  + 3 * (x << 3));
        __global double* dist1 = dist0 + 1;
        __global double* dist2 = dist0 + 2;

        double  src0_data = *src0;
        double  src1_data = *src1;
        double  src2_data = *src2;

        *dist0 = src0_data;
        *dist1 = src1_data;
        *dist2 = src2_data;
    }
}
#endif
__kernel void merge_vector_C4_D0(__global uchar *mat_dst,  int dst_step,  int dst_offset,
                                 __global uchar *mat_src0, int src0_step, int src0_offset,
                                 __global uchar *mat_src1, int src1_step, int src1_offset,
                                 __global uchar *mat_src2, int src2_step, int src2_offset,
                                 __global uchar *mat_src3, int src3_step, int src3_offset,
                                 int rows, int cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        int src0_index = mad24(y, src0_step, src0_offset);
        int src1_index = mad24(y, src1_step, src1_offset);
        int src2_index = mad24(y, src2_step, src2_offset);
        int src3_index = mad24(y, src3_step, src3_offset);
        int dst_index  = mad24(y, dst_step , dst_offset);

        uchar src0 = *(mat_src0 + src0_index + x );
        uchar src1 = *(mat_src1 + src1_index + x);
        uchar src2 = *(mat_src2 + src2_index + x);
        uchar src3 = *(mat_src3 + src3_index + x);

        *((__global uchar4 *)(mat_dst  + dst_index + (x << 2))) = (uchar4)(src0, src1, src2, src3);
    }
}
__kernel void merge_vector_C4_D1(__global char *mat_dst,  int dst_step,  int dst_offset,
                                 __global char *mat_src0, int src0_step, int src0_offset,
                                 __global char *mat_src1, int src1_step, int src1_offset,
                                 __global char *mat_src2, int src2_step, int src2_offset,
                                 __global char *mat_src3, int src3_step, int src3_offset,
                                 int rows, int cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        int src0_index = mad24(y, src0_step, src0_offset);
        int src1_index = mad24(y, src1_step, src1_offset);
        int src2_index = mad24(y, src2_step, src2_offset);
        int src3_index = mad24(y, src3_step, src3_offset);
        int dst_index  = mad24(y, dst_step , dst_offset);

        char src0 = *(mat_src0 + src0_index + x );
        char src1 = *(mat_src1 + src1_index + x);
        char src2 = *(mat_src2 + src2_index + x);
        char src3 = *(mat_src3 + src3_index + x);

        *((__global char4 *)(mat_dst  + dst_index + (x << 2))) = (char4)(src0, src1, src2, src3);
    }
}
__kernel void merge_vector_C4_D2(__global ushort *mat_dst,  int dst_step,  int dst_offset,
                                 __global ushort *mat_src0, int src0_step, int src0_offset,
                                 __global ushort *mat_src1, int src1_step, int src1_offset,
                                 __global ushort *mat_src2, int src2_step, int src2_offset,
                                 __global ushort *mat_src3, int src3_step, int src3_offset,
                                 int rows, int cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        int src0_index = mad24(y, src0_step, src0_offset);
        int src1_index = mad24(y, src1_step, src1_offset);
        int src2_index = mad24(y, src2_step, src2_offset);
        int src3_index = mad24(y, src3_step, src3_offset);
        int dst_index  = mad24(y, dst_step , dst_offset);

        ushort src0 = *((__global ushort *)((__global uchar *)mat_src0 + src0_index + (x << 1)));
        ushort src1 = *((__global ushort *)((__global uchar *)mat_src1 + src1_index + (x << 1)));
        ushort src2 = *((__global ushort *)((__global uchar *)mat_src2 + src2_index + (x << 1)));
        ushort src3 = *((__global ushort *)((__global uchar *)mat_src3 + src3_index + (x << 1)));

        *((__global ushort4 *)((__global uchar *)mat_dst  + dst_index + (x << 3))) = (ushort4)(src0, src1, src2, src3);
    }
}
__kernel void merge_vector_C4_D3(__global short *mat_dst,  int dst_step,  int dst_offset,
                                 __global short *mat_src0, int src0_step, int src0_offset,
                                 __global short *mat_src1, int src1_step, int src1_offset,
                                 __global short *mat_src2, int src2_step, int src2_offset,
                                 __global short *mat_src3, int src3_step, int src3_offset,
                                 int rows, int cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        int src0_index = mad24(y, src0_step, src0_offset);
        int src1_index = mad24(y, src1_step, src1_offset);
        int src2_index = mad24(y, src2_step, src2_offset);
        int src3_index = mad24(y, src3_step, src3_offset);
        int dst_index  = mad24(y, dst_step , dst_offset);

        short src0 = *((__global short *)((__global uchar *)mat_src0 + src0_index + (x << 1)));
        short src1 = *((__global short *)((__global uchar *)mat_src1 + src1_index + (x << 1)));
        short src2 = *((__global short *)((__global uchar *)mat_src2 + src2_index + (x << 1)));
        short src3 = *((__global short *)((__global uchar *)mat_src3 + src3_index + (x << 1)));

        *((__global short4 *)((__global uchar *)mat_dst  + dst_index + (x << 3))) = (short4)(src0, src1, src2, src3);
    }
}
__kernel void merge_vector_C4_D4(__global int *mat_dst,  int dst_step,  int dst_offset,
                                 __global int *mat_src0, int src0_step, int src0_offset,
                                 __global int *mat_src1, int src1_step, int src1_offset,
                                 __global int *mat_src2, int src2_step, int src2_offset,
                                 __global int *mat_src3, int src3_step, int src3_offset,
                                 int rows, int cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        int src0_index = mad24(y, src0_step, src0_offset);
        int src1_index = mad24(y, src1_step, src1_offset);
        int src2_index = mad24(y, src2_step, src2_offset);
        int src3_index = mad24(y, src3_step, src3_offset);
        int dst_index  = mad24(y, dst_step , dst_offset);

        int src0 = *((__global int *)((__global uchar *)mat_src0 + src0_index + (x << 2)));
        int src1 = *((__global int *)((__global uchar *)mat_src1 + src1_index + (x << 2)));
        int src2 = *((__global int *)((__global uchar *)mat_src2 + src2_index + (x << 2)));
        int src3 = *((__global int *)((__global uchar *)mat_src3 + src3_index + (x << 2)));

        *((__global int4 *)((__global uchar *)mat_dst  + dst_index + (x << 4))) = (int4)(src0, src1, src2, src3);
    }
}
__kernel void merge_vector_C4_D5(__global float *mat_dst,  int dst_step,  int dst_offset,
                                 __global float *mat_src0, int src0_step, int src0_offset,
                                 __global float *mat_src1, int src1_step, int src1_offset,
                                 __global float *mat_src2, int src2_step, int src2_offset,
                                 __global float *mat_src3, int src3_step, int src3_offset,
                                 int rows, int cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        int src0_index = mad24(y, src0_step, src0_offset);
        int src1_index = mad24(y, src1_step, src1_offset);
        int src2_index = mad24(y, src2_step, src2_offset);
        int src3_index = mad24(y, src3_step, src3_offset);
        int dst_index  = mad24(y, dst_step , dst_offset);

        float src0 = *((__global float *)((__global uchar *)mat_src0 + src0_index + (x << 2)));
        float src1 = *((__global float *)((__global uchar *)mat_src1 + src1_index + (x << 2)));
        float src2 = *((__global float *)((__global uchar *)mat_src2 + src2_index + (x << 2)));
        float src3 = *((__global float *)((__global uchar *)mat_src3 + src3_index + (x << 2)));

        *((__global float4 *)((__global uchar *)mat_dst  + dst_index + (x << 4))) = (float4)(src0, src1, src2, src3);
    }
}

#if defined (DOUBLE_SUPPORT)
__kernel void merge_vector_C4_D6(__global double *mat_dst,  int dst_step,  int dst_offset,
                                 __global double *mat_src0, int src0_step, int src0_offset,
                                 __global double *mat_src1, int src1_step, int src1_offset,
                                 __global double *mat_src2, int src2_step, int src2_offset,
                                 __global double *mat_src3, int src3_step, int src3_offset,
                                 int rows, int cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        int src0_index = mad24(y, src0_step, src0_offset);
        int src1_index = mad24(y, src1_step, src1_offset);
        int src2_index = mad24(y, src2_step, src2_offset);
        int src3_index = mad24(y, src3_step, src3_offset);
        int dst_index  = mad24(y, dst_step , dst_offset);

        double src0 = *((__global double *)((__global uchar *)mat_src0 + src0_index + (x << 3)));
        double src1 = *((__global double *)((__global uchar *)mat_src1 + src1_index + (x << 3)));
        double src2 = *((__global double *)((__global uchar *)mat_src2 + src2_index + (x << 3)));
        double src3 = *((__global double *)((__global uchar *)mat_src3 + src3_index + (x << 3)));

        *((__global double4 *)((__global uchar *)mat_dst  + dst_index + (x << 5))) = (double4)(src0, src1, src2, src3);
    }
}
#endif
///////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////optimized code using vector  no roi//////////////////////////
////////////vector fuction name format: merge_vector_C(channels number)D_(data type depth)//////
////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void merge_vector_C2_D0_1(int rows, int cols,
                                   __global uchar *mat_dst,  int dst_step,
                                   __global uchar *mat_src0, int src0_step,
                                   __global uchar *mat_src1, int src1_step)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        __global uchar4  *src0_y = (__global uchar4 * )(mat_src0 + y * src0_step);
        __global uchar4  *src1_y = (__global uchar4 * )(mat_src1 + y * src1_step);
        __global uchar8 *dst_y  = (__global uchar8 *)(mat_dst  + y * dst_step);

        uchar4 value1 = src0_y[x];
        uchar4 value2 = src1_y[x];

        uchar8 value;
        value.even = value1;
        value.odd = value2;

        dst_y[x] = value;
    }
}
__kernel void merge_vector_C2_D1_1(int rows, int cols,
                                   __global char *mat_dst,  int dst_step,
                                   __global char *mat_src0, int src0_step,
                                   __global char *mat_src1, int src1_step)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        __global char4  *src0_y = (__global char4 * )(mat_src0 + y * src0_step);
        __global char4  *src1_y = (__global char4 * )(mat_src1 + y * src1_step);
        __global char8 *dst_y  = (__global char8 *)(mat_dst  + y * dst_step);

        char4 value1 = src0_y[x];
        char4 value2 = src1_y[x];

        char8 value;
        value.even = value1;
        value.odd = value2;

        dst_y[x] = value;
    }
}
__kernel void merge_vector_C2_D2_1(int rows, int cols,
                                   __global ushort *mat_dst,  int dst_step,
                                   __global ushort *mat_src0, int src0_step,
                                   __global ushort *mat_src1, int src1_step)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        __global ushort2  *src0_y = (__global ushort2 *)((__global uchar *)mat_src0 + y * src0_step);
        __global ushort2  *src1_y = (__global ushort2 *)((__global uchar *)mat_src1 + y * src1_step);
        __global ushort4  *dst_y  = (__global ushort4 *)((__global uchar *)mat_dst  + y * dst_step);

        ushort2 value1 = src0_y[x];
        ushort2 value2 = src1_y[x];

        ushort4 value;
        value.even = value1;
        value.odd = value2;

        dst_y[x] = value;
    }
}
__kernel void merge_vector_C2_D3_1(int rows, int cols,
                                   __global short *mat_dst,  int dst_step,
                                   __global short *mat_src0, int src0_step,
                                   __global short *mat_src1, int src1_step)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        __global short2  *src0_y = (__global short2 *)((__global uchar *)mat_src0 + y * src0_step);
        __global short2  *src1_y = (__global short2 *)((__global uchar *)mat_src1 + y * src1_step);
        __global short4 *dst_y   = (__global short4 *)((__global uchar *)mat_dst  + y * dst_step);

        short2 value1 = src0_y[x];
        short2 value2 = src1_y[x];

        short4 value;
        value.even = value1;
        value.odd = value2;

        dst_y[x] = value;
    }
}

__kernel void merge_vector_C2_D4_1(int rows, int cols,
                                   __global int *mat_dst,  int dst_step,
                                   __global int *mat_src0, int src0_step,
                                   __global int *mat_src1, int src1_step)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        __global int  *src0_y = (__global int *)((__global uchar *)mat_src0 + y * src0_step);
        __global int  *src1_y = (__global int *)((__global uchar *)mat_src1 + y * src1_step);
        __global int2  *dst_y  = (__global int2 *)((__global uchar *)mat_dst  + y * dst_step);

        int value1 = src0_y[x];
        int value2 = src1_y[x];

        int2 value;
        value.even = value1;
        value.odd = value2;

        dst_y[x] = value;
    }
}
__kernel void merge_vector_C2_D5_1(int rows, int cols,
                                   __global float *mat_dst,  int dst_step,
                                   __global float *mat_src0, int src0_step,
                                   __global float *mat_src1, int src1_step)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        __global float  *src0_y = (__global float *)((__global uchar *)mat_src0 + y * src0_step);
        __global float  *src1_y = (__global float *)((__global uchar *)mat_src1 + y * src1_step);
        __global float2  *dst_y  = (__global float2 *)((__global uchar *)mat_dst  + y * dst_step);

        float value1 = src0_y[x];
        float value2 = src1_y[x];

        dst_y[x] = (float2)(value1, value2);
    }
}

#if defined (DOUBLE_SUPPORT)
__kernel void merge_vector_C2_D6_1(int rows, int cols,
                                   __global double *mat_dst,  int dst_step,
                                   __global double *mat_src0, int src0_step,
                                   __global double *mat_src1, int src1_step)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        __global double  *src0_y = (__global double *)((__global uchar *)mat_src0 + y * src0_step);
        __global double  *src1_y = (__global double *)((__global uchar *)mat_src1 + y * src1_step);
        __global double2 *dst_y  = (__global double2 *)((__global uchar *)mat_dst  + y * dst_step);

        double value1 = src0_y[x];
        double value2 = src1_y[x];

        dst_y[x] = (double2)(value1, value2);
    }
}
#endif

__kernel void merge_vector_C3_D0_1(int rows, int cols,
                                   __global uchar *mat_dst,  int dst_step,
                                   __global uchar *mat_src0, int src0_step,
                                   __global uchar *mat_src1, int src1_step,
                                   __global uchar *mat_src2, int src2_step)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        __global uchar4  *src0_y = (__global uchar4 * )(mat_src0 + y * src0_step);
        __global uchar4  *src1_y = (__global uchar4 * )(mat_src1 + y * src1_step);
        __global uchar4  *src2_y = (__global uchar4 * )(mat_src2 + y * src0_step);

        __global uchar4 *dst_y  = (__global uchar4 *)(mat_dst  + y * dst_step);

        uchar4 value0 = src0_y[x];
        uchar4 value1 = src1_y[x];
        uchar4 value2 = src2_y[x];

        dst_y[3 * x + 0] = (uchar4)(value0.s0, value1.s0, value2.s0,
                                    value0.s1);

        dst_y[3 * x + 1] = (uchar4)(value1.s1, value2.s1,
                                    value0.s2, value1.s2);

        dst_y[3 * x + 2] = (uchar4)(value2.s2,
                                    value0.s3, value1.s3, value2.s3);

    }
}
__kernel void merge_vector_C3_D1_1(int rows, int cols,
                                   __global char *mat_dst,  int dst_step,
                                   __global char *mat_src0, int src0_step,
                                   __global char *mat_src1, int src1_step,
                                   __global char *mat_src2, int src2_step)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        __global char4  *src0_y = (__global char4 * )(mat_src0 + y * src0_step);
        __global char4  *src1_y = (__global char4 * )(mat_src1 + y * src1_step);
        __global char4  *src2_y = (__global char4 * )(mat_src2 + y * src0_step);

        __global char4 *dst_y  = (__global char4 *)(mat_dst  + y * dst_step);

        char4 value0 = src0_y[x];
        char4 value1 = src1_y[x];
        char4 value2 = src2_y[x];

        dst_y[3 * x + 0] = (char4)(value0.s0, value1.s0, value2.s0,
                                   value0.s1);

        dst_y[3 * x + 1] = (char4)(value1.s1, value2.s1,
                                     value0.s2, value1.s2);

        dst_y[3 * x + 2] = (char4)(value2.s2,
                                     value0.s3, value1.s3, value2.s3);

        /* for test do not delete
        dst_y[3 * x + 0] = (char8)(value0.s0, value1.s0, value2.s0,
                                    value0.s1, value1.s1, value2.s1,
                                    value0.s2, value1.s2);

        dst_y[3 * x + 1] = (char8)(value2.s2,
                                    value0.s3, value1.s3, value2.s3,
                                    value0.s4, value1.s4, value2.s4,
                                    value0.s5);

        dst_y[3 * x + 2] = (char8)(value1.s5, value2.s5,
                                    value0.s6, value1.s6, value2.s6,
                                    value0.s7, value1.s7, value2.s7);
                                    */
    }
}
__kernel void merge_vector_C3_D2_1(int rows, int cols,
                                   __global ushort *mat_dst,  int dst_step,
                                   __global ushort *mat_src0, int src0_step,
                                   __global ushort *mat_src1, int src1_step,
                                   __global ushort *mat_src2, int src2_step)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        __global ushort2  *src0_y = (__global ushort2 * )((__global char *)mat_src0 + y * src0_step);
        __global ushort2  *src1_y = (__global ushort2 * )((__global char *)mat_src1 + y * src1_step);
        __global ushort2  *src2_y = (__global ushort2 * )((__global char *)mat_src2 + y * src0_step);

        __global ushort2 *dst_y  = (__global ushort2 *)((__global char *)mat_dst  + y * dst_step);

        ushort2 value0 = src0_y[x];
        ushort2 value1 = src1_y[x];
        ushort2 value2 = src2_y[x];

        dst_y[3 * x + 0] = (ushort2)(value0.x, value1.x);
        dst_y[3 * x + 1] = (ushort2)(value2.x, value0.y);
        dst_y[3 * x + 2] = (ushort2)(value1.y, value2.y);

    }
}
__kernel void merge_vector_C3_D3_1(int rows, int cols,
                                   __global short *mat_dst,  int dst_step,
                                   __global short *mat_src0, int src0_step,
                                   __global short *mat_src1, int src1_step,
                                   __global short *mat_src2, int src2_step)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        __global short2  *src0_y = (__global short2 * )((__global char *)mat_src0 + y * src0_step);
        __global short2  *src1_y = (__global short2 * )((__global char *)mat_src1 + y * src1_step);
        __global short2  *src2_y = (__global short2 * )((__global char *)mat_src2 + y * src0_step);

        __global short2 *dst_y  = (__global short2 *)((__global char *)mat_dst  + y * dst_step);

        short2 value0 = src0_y[x];
        short2 value1 = src1_y[x];
        short2 value2 = src2_y[x];

        dst_y[3 * x + 0] = (short2)(value0.x, value1.x);
        dst_y[3 * x + 1] = (short2)(value2.x, value0.y);
        dst_y[3 * x + 2] = (short2)(value1.y, value2.y);

        /*
        dst_y[3 * x + 0] = (short4)(value0.s0, value1.s0, value2.s0,
                                    value0.s1);

        dst_y[3 * x + 1] = (short4)(value1.s1, value2.s1,
                                    value0.s2, value1.s2);

        dst_y[3 * x + 2] = (short4)(value2.s2,
                                    value0.s3, value1.s3, value2.s3);
                                    */
    }
}
__kernel void merge_vector_C3_D4_1(int rows, int cols,
                                   __global int *mat_dst,  int dst_step,
                                   __global int *mat_src0, int src0_step,
                                   __global int *mat_src1, int src1_step,
                                   __global int *mat_src2, int src2_step)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        __global int  *src0_y = (__global int * )((__global char *)mat_src0 + y * src0_step);
        __global int  *src1_y = (__global int * )((__global char *)mat_src1 + y * src1_step);
        __global int  *src2_y = (__global int * )((__global char *)mat_src2 + y * src0_step);

        __global int *dst_y  = (__global int *)((__global char *)mat_dst  + y * dst_step);

        int value0 = src0_y[x];
        int value1 = src1_y[x];
        int value2 = src2_y[x];

        dst_y[3 * x + 0] = value0;
        dst_y[3 * x + 1] = value1;
        dst_y[3 * x + 2] = value2;

        /*for test do not delete
        dst_y[3 * x + 0] = (int2)(value0.x, value1.x);
        dst_y[3 * x + 1] = (int2)(value2.x, value0.y);
        dst_y[3 * x + 2] = (int2)(value1.y, value2.y);
        */
    }
}
__kernel void merge_vector_C3_D5_1(int rows, int cols,
                                   __global float *mat_dst,  int dst_step,
                                   __global float *mat_src0, int src0_step,
                                   __global float *mat_src1, int src1_step,
                                   __global float *mat_src2, int src2_step)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        __global float  *src0_y = (__global float * )((__global char *)mat_src0 + y * src0_step);
        __global float  *src1_y = (__global float * )((__global char *)mat_src1 + y * src1_step);
        __global float  *src2_y = (__global float * )((__global char *)mat_src2 + y * src0_step);

        __global float *dst_y  = (__global float *)((__global char *)mat_dst  + y * dst_step);

        float value0 = src0_y[x];
        float value1 = src1_y[x];
        float value2 = src2_y[x];

        dst_y[3 * x + 0] = value0;
        dst_y[3 * x + 1] = value1;
        dst_y[3 * x + 2] = value2;
    }
}

#if defined (DOUBLE_SUPPORT)
__kernel void merge_vector_C3_D6_1(int rows, int cols,
                                   __global double *mat_dst,  int dst_step,
                                   __global double *mat_src0, int src0_step,
                                   __global double *mat_src1, int src1_step,
                                   __global double *mat_src2, int src2_step)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        __global double  *src0_y = (__global double * )((__global char *)mat_src0 + y * src0_step);
        __global double  *src1_y = (__global double * )((__global char *)mat_src1 + y * src1_step);
        __global double  *src2_y = (__global double * )((__global char *)mat_src2 + y * src0_step);

        __global double *dst_y  = (__global double *)((__global char *)mat_dst  + y * dst_step);

        double value0 = src0_y[x];
        double value1 = src1_y[x];
        double value2 = src2_y[x];

        dst_y[3 * x + 0] = value0;
        dst_y[3 * x + 1] = value1;
        dst_y[3 * x + 2] = value2;
    }
}
#endif
__kernel void merge_vector_C4_D0_1(int rows, int cols,
                                   __global uchar *mat_dst,  int dst_step,
                                   __global uchar *mat_src0, int src0_step,
                                   __global uchar *mat_src1, int src1_step,
                                   __global uchar *mat_src2, int src2_step,
                                   __global uchar *mat_src3, int src3_step)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        __global uchar4  *src0_y = (__global uchar4 * )(mat_src0 + y * src0_step);
        __global uchar4  *src1_y = (__global uchar4 * )(mat_src1 + y * src1_step);
        __global uchar4  *src2_y = (__global uchar4 * )(mat_src2 + y * src0_step);
        __global uchar4  *src3_y = (__global uchar4 * )(mat_src3 + y * src1_step);

        __global uchar16 *dst_y  = (__global uchar16 *)(mat_dst  + y * dst_step);

        uchar4 value0 = src0_y[x];
        uchar4 value1 = src1_y[x];
        uchar4 value2 = src2_y[x];
        uchar4 value3 = src3_y[x];

        dst_y[x] = (uchar16)(value0.x, value1.x, value2.x, value3.x,
                             value0.y, value1.y, value2.y, value3.y,
                             value0.z, value1.z, value2.z, value3.z,
                             value0.w, value1.w, value2.w, value3.w);
    }
}

__kernel void merge_vector_C4_D1_1(int rows, int cols,
                                   __global char *mat_dst,  int dst_step,
                                   __global char *mat_src0, int src0_step,
                                   __global char *mat_src1, int src1_step,
                                   __global char *mat_src2, int src2_step,
                                   __global char *mat_src3, int src3_step)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        __global char4  *src0_y = (__global char4 * )(mat_src0 + y * src0_step);
        __global char4  *src1_y = (__global char4 * )(mat_src1 + y * src1_step);
        __global char4  *src2_y = (__global char4 * )(mat_src2 + y * src0_step);
        __global char4  *src3_y = (__global char4 * )(mat_src3 + y * src1_step);

        __global char16 *dst_y  = (__global char16 *)(mat_dst  + y * dst_step);

        char4 value0 = src0_y[x];
        char4 value1 = src1_y[x];
        char4 value2 = src2_y[x];
        char4 value3 = src3_y[x];

        dst_y[x] = (char16)(value0.x, value1.x, value2.x, value3.x,
                            value0.y, value1.y, value2.y, value3.y,
                            value0.z, value1.z, value2.z, value3.z,
                            value0.w, value1.w, value2.w, value3.w);
    }
}
__kernel void merge_vector_C4_D2_1(int rows, int cols,
                                   __global ushort *mat_dst,  int dst_step,
                                   __global ushort *mat_src0, int src0_step,
                                   __global ushort *mat_src1, int src1_step,
                                   __global ushort *mat_src2, int src2_step,
                                   __global ushort *mat_src3, int src3_step)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        __global ushort2  *src0_y = (__global ushort2 * )((__global uchar*)mat_src0 + y * src0_step);
        __global ushort2  *src1_y = (__global ushort2 * )((__global uchar*)mat_src1 + y * src1_step);
        __global ushort2  *src2_y = (__global ushort2 * )((__global uchar*)mat_src2 + y * src0_step);
        __global ushort2  *src3_y = (__global ushort2 * )((__global uchar*)mat_src3 + y * src1_step);

        __global ushort8 *dst_y  = (__global ushort8 *)((__global uchar*)mat_dst  + y * dst_step);

        ushort2 value0 = src0_y[x];
        ushort2 value1 = src1_y[x];
        ushort2 value2 = src2_y[x];
        ushort2 value3 = src3_y[x];

        dst_y[x] = (ushort8)(value0.x, value1.x, value2.x, value3.x,
                             value0.y, value1.y, value2.y, value3.y);
    }
}
__kernel void merge_vector_C4_D3_1(int rows, int cols,
                                   __global short *mat_dst,  int dst_step,
                                   __global short *mat_src0, int src0_step,
                                   __global short *mat_src1, int src1_step,
                                   __global short *mat_src2, int src2_step,
                                   __global short *mat_src3, int src3_step)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        __global short2  *src0_y = (__global short2 * )((__global uchar*)mat_src0 + y * src0_step);
        __global short2  *src1_y = (__global short2 * )((__global uchar*)mat_src1 + y * src1_step);
        __global short2  *src2_y = (__global short2 * )((__global uchar*)mat_src2 + y * src0_step);
        __global short2  *src3_y = (__global short2 * )((__global uchar*)mat_src3 + y * src1_step);

        __global short8 *dst_y  = (__global short8 *)((__global uchar*)mat_dst  + y * dst_step);

        short2 value0 = src0_y[x];
        short2 value1 = src1_y[x];
        short2 value2 = src2_y[x];
        short2 value3 = src3_y[x];

        dst_y[x] = (short8)(value0.x, value1.x, value2.x, value3.x,
                            value0.y, value1.y, value2.y, value3.y);
    }
}
__kernel void merge_vector_C4_D4_1(int rows, int cols,
                                   __global int *mat_dst,  int dst_step,
                                   __global int *mat_src0, int src0_step,
                                   __global int *mat_src1, int src1_step,
                                   __global int *mat_src2, int src2_step,
                                   __global int *mat_src3, int src3_step)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        __global int *src0_y = (__global int * )((__global uchar*)mat_src0 + y * src0_step);
        __global int *src1_y = (__global int * )((__global uchar*)mat_src1 + y * src1_step);
        __global int *src2_y = (__global int * )((__global uchar*)mat_src2 + y * src0_step);
        __global int *src3_y = (__global int * )((__global uchar*)mat_src3 + y * src1_step);

        __global int4 *dst_y  = (__global int4 *)((__global uchar*)mat_dst  + y * dst_step);

        int value0 = src0_y[x];
        int value1 = src1_y[x];
        int value2 = src2_y[x];
        int value3 = src3_y[x];

        dst_y[x] = (int4)(value0, value1, value2, value3);
    }
}
__kernel void merge_vector_C4_D5_1(int rows, int cols,
                                   __global float *mat_dst,  int dst_step,
                                   __global float *mat_src0, int src0_step,
                                   __global float *mat_src1, int src1_step,
                                   __global float *mat_src2, int src2_step,
                                   __global float *mat_src3, int src3_step)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        __global float *src0_y = (__global float * )((__global uchar*)mat_src0 + y * src0_step);
        __global float *src1_y = (__global float * )((__global uchar*)mat_src1 + y * src1_step);
        __global float *src2_y = (__global float * )((__global uchar*)mat_src2 + y * src0_step);
        __global float *src3_y = (__global float * )((__global uchar*)mat_src3 + y * src1_step);

        __global float4 *dst_y  = (__global float4 *)((__global uchar*)mat_dst  + y * dst_step);

        float value0 = src0_y[x];
        float value1 = src1_y[x];
        float value2 = src2_y[x];
        float value3 = src3_y[x];

        dst_y[x] = (float4)(value0, value1, value2, value3);
    }
}

#if defined (DOUBLE_SUPPORT)
__kernel void merge_vector_C4_D6_1(int rows, int cols,
                                   __global double *mat_dst,  int dst_step,
                                   __global double *mat_src0, int src0_step,
                                   __global double *mat_src1, int src1_step,
                                   __global double *mat_src2, int src2_step,
                                   __global double *mat_src3, int src3_step)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ((x < cols) && (y < rows))
    {
        __global double *src0_y = (__global double * )((__global uchar*)mat_src0 + y * src0_step);
        __global double *src1_y = (__global double * )((__global uchar*)mat_src1 + y * src1_step);
        __global double *src2_y = (__global double * )((__global uchar*)mat_src2 + y * src0_step);
        __global double *src3_y = (__global double * )((__global uchar*)mat_src3 + y * src1_step);

        __global double4 *dst_y  = (__global double4 *)((__global uchar*)mat_dst  + y * dst_step);

        double value0 = src0_y[x];
        double value1 = src1_y[x];
        double value2 = src2_y[x];
        double value3 = src3_y[x];

        dst_y[x] = (double4)(value0, value1, value2, value3);
    }
}
#endif
