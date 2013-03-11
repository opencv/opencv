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

#if defined (DOUBLE_SUPPORT)
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#elif defined (cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#endif
#endif
/**************************************sub with scalar without mask**************************************/
__kernel void arithm_s_sub_C1_D0 (__global   uchar *src1, int src1_step, int src1_offset,
                                  __global   uchar *dst,  int dst_step,  int dst_offset,
                                  int4 src2, int rows, int cols, int dst_step1, int isMatSubScalar)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        x = x << 2;

        #define dst_align (dst_offset & 3)
        int src1_index = mad24(y, src1_step, x + src1_offset - dst_align);

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + x & (int)0xfffffffc);

        uchar4 src1_data = vload4(0, src1 + src1_index);
        int4 src2_data = (int4)(src2.x, src2.x, src2.x, src2.x);

        uchar4 data = *((__global uchar4 *)(dst + dst_index));
        int4 tmp = convert_int4_sat(src1_data) - src2_data;
        tmp = isMatSubScalar ? tmp : -tmp;
        uchar4 tmp_data = convert_uchar4_sat(tmp);

        data.x = ((dst_index + 0 >= dst_start) && (dst_index + 0 < dst_end)) ? tmp_data.x : data.x;
        data.y = ((dst_index + 1 >= dst_start) && (dst_index + 1 < dst_end)) ? tmp_data.y : data.y;
        data.z = ((dst_index + 2 >= dst_start) && (dst_index + 2 < dst_end)) ? tmp_data.z : data.z;
        data.w = ((dst_index + 3 >= dst_start) && (dst_index + 3 < dst_end)) ? tmp_data.w : data.w;

        *((__global uchar4 *)(dst + dst_index)) = data;
    }
}
__kernel void arithm_s_sub_C1_D2 (__global   ushort *src1, int src1_step, int src1_offset,
                                  __global   ushort *dst,  int dst_step,  int dst_offset,
                                  int4 src2, int rows, int cols, int dst_step1, int isMatSubScalar)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        x = x << 1;

        #define dst_align ((dst_offset >> 1) & 1)
        int src1_index = mad24(y, src1_step, (x << 1) + src1_offset - (dst_align << 1));

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + (x << 1) & (int)0xfffffffc);

        ushort2 src1_data = vload2(0, (__global ushort *)((__global char *)src1 + src1_index));
        int2 src2_data = (int2)(src2.x, src2.x);

        ushort2 data = *((__global ushort2 *)((__global uchar *)dst + dst_index));
        int2    tmp = convert_int2_sat(src1_data) - src2_data;
        tmp = isMatSubScalar ? tmp : -tmp;
        ushort2 tmp_data = convert_ushort2_sat(tmp);

        data.x = (dst_index + 0 >= dst_start) ? tmp_data.x : data.x;
        data.y = (dst_index + 2 <  dst_end  ) ? tmp_data.y : data.y;

        *((__global ushort2 *)((__global uchar *)dst + dst_index)) = data;
    }
}
__kernel void arithm_s_sub_C1_D3 (__global   short *src1, int src1_step, int src1_offset,
                                  __global   short *dst,  int dst_step,  int dst_offset,
                                  int4 src2, int rows, int cols, int dst_step1, int isMatSubScalar)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        x = x << 1;

        #define dst_align ((dst_offset >> 1) & 1)
        int src1_index = mad24(y, src1_step, (x << 1) + src1_offset - (dst_align << 1));

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + (x << 1) & (int)0xfffffffc);

        short2 src1_data = vload2(0, (__global short *)((__global char *)src1 + src1_index));
        int2 src2_data = (int2)(src2.x, src2.x);
        short2 data = *((__global short2 *)((__global uchar *)dst + dst_index));

        int2   tmp = convert_int2_sat(src1_data) - src2_data;
        tmp = isMatSubScalar ? tmp : -tmp;
        short2 tmp_data = convert_short2_sat(tmp);

        data.x = (dst_index + 0 >= dst_start) ? tmp_data.x : data.x;
        data.y = (dst_index + 2 <  dst_end  ) ? tmp_data.y : data.y;

        *((__global short2 *)((__global uchar *)dst + dst_index)) = data;
    }
}
__kernel void arithm_s_sub_C1_D4 (__global   int *src1, int src1_step, int src1_offset,
                                  __global   int *dst,  int dst_step,  int dst_offset,
                                  int4 src2, int rows, int cols, int dst_step1, int isMatSubScalar)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 2) + src1_offset);
        int dst_index  = mad24(y, dst_step,  (x << 2) + dst_offset);

        int src_data1 = *((__global int *)((__global char *)src1 + src1_index));
        int src_data2 = src2.x;

        long tmp = (long)src_data1 - (long)src_data2;
        tmp = isMatSubScalar ? tmp : -tmp;
        int data = convert_int_sat(tmp);

        *((__global int *)((__global char *)dst + dst_index)) = data;
    }
}
__kernel void arithm_s_sub_C1_D5 (__global   float *src1, int src1_step, int src1_offset,
                                  __global   float *dst,  int dst_step,  int dst_offset,
                                  float4 src2, int rows, int cols, int dst_step1, int isMatSubScalar)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 2) + src1_offset);
        int dst_index  = mad24(y, dst_step,  (x << 2) + dst_offset);

        float src_data1 = *((__global float *)((__global char *)src1 + src1_index));
        float src_data2 = src2.x;

        float tmp = src_data1 - src_data2;
        tmp = isMatSubScalar ? tmp : -tmp;

        *((__global float *)((__global char *)dst + dst_index)) = tmp;
    }
}

#if defined (DOUBLE_SUPPORT)
__kernel void arithm_s_sub_C1_D6 (__global   double *src1, int src1_step, int src1_offset,
                                  __global   double *dst,  int dst_step,  int dst_offset,
                                  double4 src2, int rows, int cols, int dst_step1, int isMatSubScalar)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 3) + src1_offset);
        int dst_index  = mad24(y, dst_step,  (x << 3) + dst_offset);

        double src_data1 = *((__global double *)((__global char *)src1 + src1_index));
        double src2_data = src2.x;

        double data = src_data1 - src2_data;
        data = isMatSubScalar ? data : -data;

        *((__global double *)((__global char *)dst + dst_index)) = data;
    }
}
#endif

__kernel void arithm_s_sub_C2_D0 (__global   uchar *src1, int src1_step, int src1_offset,
                                  __global   uchar *dst,  int dst_step,  int dst_offset,
                                  int4 src2, int rows, int cols, int dst_step1, int isMatSubScalar)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        x = x << 1;

        #define dst_align ((dst_offset >> 1) & 1)
        int src1_index = mad24(y, src1_step, (x << 1) + src1_offset - (dst_align << 1));

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + (x << 1) & (int)0xfffffffc);

        uchar4 src1_data = vload4(0, src1 + src1_index);
        int4 src2_data = (int4)(src2.x, src2.y, src2.x, src2.y);

        uchar4 data = *((__global uchar4 *)(dst + dst_index));
        int4 tmp = convert_int4_sat(src1_data) - src2_data;
        tmp = isMatSubScalar ? tmp : -tmp;
        uchar4 tmp_data = convert_uchar4_sat(tmp);

        data.xy = (dst_index + 0 >= dst_start) ? tmp_data.xy : data.xy;
        data.zw = (dst_index + 2 <  dst_end  ) ? tmp_data.zw : data.zw;

        *((__global uchar4 *)(dst + dst_index)) = data;
    }
}
__kernel void arithm_s_sub_C2_D2 (__global   ushort *src1, int src1_step, int src1_offset,
                                  __global   ushort *dst,  int dst_step,  int dst_offset,
                                  int4 src2, int rows, int cols, int dst_step1, int isMatSubScalar)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 2) + src1_offset);
        int dst_index  = mad24(y, dst_step,  (x << 2) + dst_offset);

        ushort2 src_data1 = *((__global ushort2 *)((__global char *)src1 + src1_index));
        int2 src_data2    = (int2)(src2.x, src2.y);
        ushort2 dst_data  = *((__global ushort2 *)((__global char *)dst  + dst_index));

        int2    tmp = convert_int2_sat(src_data1) - src_data2;
        tmp = isMatSubScalar ? tmp : -tmp;
        ushort2 data = convert_ushort2_sat(tmp);

        *((__global ushort2 *)((__global char *)dst + dst_index)) = data;
    }
}
__kernel void arithm_s_sub_C2_D3 (__global   short *src1, int src1_step, int src1_offset,
                                  __global   short *dst,  int dst_step,  int dst_offset,
                                  int4 src2, int rows, int cols, int dst_step1, int isMatSubScalar)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 2) + src1_offset);
        int dst_index  = mad24(y, dst_step,  (x << 2) + dst_offset);

        short2 src_data1 = *((__global short2 *)((__global char *)src1 + src1_index));
        int2 src_data2 = (int2)(src2.x, src2.y);
        short2 dst_data  = *((__global short2 *)((__global char *)dst  + dst_index));

        int2    tmp = convert_int2_sat(src_data1) - src_data2;
        tmp = isMatSubScalar ? tmp : -tmp;
        short2 data = convert_short2_sat(tmp);

        *((__global short2 *)((__global char *)dst + dst_index)) = data;
    }
}
__kernel void arithm_s_sub_C2_D4 (__global   int *src1, int src1_step, int src1_offset,
                                  __global   int *dst,  int dst_step,  int dst_offset,
                                  int4 src2, int rows, int cols, int dst_step1, int isMatSubScalar)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 3) + src1_offset);
        int dst_index  = mad24(y, dst_step,  (x << 3) + dst_offset);

        int2 src_data1 = *((__global int2 *)((__global char *)src1 + src1_index));
        int2 src_data2 = (int2)(src2.x, src2.y);
        int2 dst_data  = *((__global int2 *)((__global char *)dst  + dst_index));

        long2 tmp = convert_long2_sat(src_data1) - convert_long2_sat(src_data2);
        tmp = isMatSubScalar ? tmp : -tmp;
        int2 data = convert_int2_sat(tmp);

        *((__global int2 *)((__global char *)dst + dst_index)) = data;
    }
}
__kernel void arithm_s_sub_C2_D5 (__global   float *src1, int src1_step, int src1_offset,
                                  __global   float *dst,  int dst_step,  int dst_offset,
                                  float4 src2, int rows, int cols, int dst_step1, int isMatSubScalar)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 3) + src1_offset);
        int dst_index  = mad24(y, dst_step,  (x << 3) + dst_offset);

        float2 src_data1 = *((__global float2 *)((__global char *)src1 + src1_index));
        float2 src_data2 = (float2)(src2.x, src2.y);
        float2 dst_data  = *((__global float2 *)((__global char *)dst  + dst_index));

        float2 tmp = src_data1 - src_data2;
        tmp = isMatSubScalar ? tmp : -tmp;

        *((__global float2 *)((__global char *)dst + dst_index)) = tmp;
    }
}

#if defined (DOUBLE_SUPPORT)
__kernel void arithm_s_sub_C2_D6 (__global   double *src1, int src1_step, int src1_offset,
                                  __global   double *dst,  int dst_step,  int dst_offset,
                                  double4 src2, int rows, int cols, int dst_step1, int isMatSubScalar)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 4) + src1_offset);
        int dst_index  = mad24(y, dst_step,  (x << 4) + dst_offset);

        double2 src_data1 = *((__global double2 *)((__global char *)src1 + src1_index));
        double2 src_data2 = (double2)(src2.x, src2.y);
        double2 dst_data  = *((__global double2 *)((__global char *)dst  + dst_index));

        double2 data = src_data1 - src_data2;
        data = isMatSubScalar ? data : -data;

        *((__global double2 *)((__global char *)dst + dst_index)) = data;
    }
}
#endif
__kernel void arithm_s_sub_C3_D0 (__global   uchar *src1, int src1_step, int src1_offset,
                                  __global   uchar *dst,  int dst_step,  int dst_offset,
                                  int4 src2, int rows, int cols, int dst_step1, int isMatSubScalar)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        x = x << 2;

        #define dst_align (((dst_offset % dst_step) / 3 ) & 3)
        int src1_index = mad24(y, src1_step, (x * 3) + src1_offset - (dst_align * 3));

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + (x * 3) - (dst_align * 3));

        uchar4 src1_data_0 = vload4(0, src1 + src1_index + 0);
        uchar4 src1_data_1 = vload4(0, src1 + src1_index + 4);
        uchar4 src1_data_2 = vload4(0, src1 + src1_index + 8);

        int4 src2_data_0 = (int4)(src2.x, src2.y, src2.z, src2.x);
        int4 src2_data_1 = (int4)(src2.y, src2.z, src2.x, src2.y);
        int4 src2_data_2 = (int4)(src2.z, src2.x, src2.y, src2.z);

        uchar4 data_0 = *((__global uchar4 *)(dst + dst_index + 0));
        uchar4 data_1 = *((__global uchar4 *)(dst + dst_index + 4));
        uchar4 data_2 = *((__global uchar4 *)(dst + dst_index + 8));

        int4 tmp_0 = convert_int4_sat(src1_data_0) - src2_data_0;
        int4 tmp_1 = convert_int4_sat(src1_data_1) - src2_data_1;
        int4 tmp_2 = convert_int4_sat(src1_data_2) - src2_data_2;

        tmp_0 = isMatSubScalar ? tmp_0 : -tmp_0;
        tmp_1 = isMatSubScalar ? tmp_1 : -tmp_1;
        tmp_2 = isMatSubScalar ? tmp_2 : -tmp_2;

        uchar4 tmp_data_0 = convert_uchar4_sat(tmp_0);
        uchar4 tmp_data_1 = convert_uchar4_sat(tmp_1);
        uchar4 tmp_data_2 = convert_uchar4_sat(tmp_2);

        data_0.xyz = ((dst_index + 0 >= dst_start)) ? tmp_data_0.xyz : data_0.xyz;
        data_0.w   = ((dst_index + 3 >= dst_start) && (dst_index + 3 < dst_end))
                     ? tmp_data_0.w : data_0.w;

        data_1.xy  = ((dst_index + 3 >= dst_start) && (dst_index + 3 < dst_end))
                     ? tmp_data_1.xy : data_1.xy;
        data_1.zw  = ((dst_index + 6 >= dst_start) && (dst_index + 6 < dst_end))
                     ? tmp_data_1.zw : data_1.zw;

        data_2.x   = ((dst_index + 6 >= dst_start) && (dst_index + 6 < dst_end))
                     ? tmp_data_2.x : data_2.x;
        data_2.yzw = ((dst_index + 9 >= dst_start) && (dst_index + 9 < dst_end))
                     ? tmp_data_2.yzw : data_2.yzw;

        *((__global uchar4 *)(dst + dst_index + 0)) = data_0;
        *((__global uchar4 *)(dst + dst_index + 4)) = data_1;
        *((__global uchar4 *)(dst + dst_index + 8)) = data_2;
    }
}
__kernel void arithm_s_sub_C3_D2 (__global   ushort *src1, int src1_step, int src1_offset,
                                  __global   ushort *dst,  int dst_step,  int dst_offset,
                                  int4 src2, int rows, int cols, int dst_step1, int isMatSubScalar)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        x = x << 1;

        #define dst_align (((dst_offset % dst_step) / 6 ) & 1)
        int src1_index = mad24(y, src1_step, (x * 6) + src1_offset - (dst_align * 6));

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + (x * 6) - (dst_align * 6));

        ushort2 src1_data_0 = vload2(0, (__global ushort *)((__global char *)src1 + src1_index + 0));
        ushort2 src1_data_1 = vload2(0, (__global ushort *)((__global char *)src1 + src1_index + 4));
        ushort2 src1_data_2 = vload2(0, (__global ushort *)((__global char *)src1 + src1_index + 8));

        int2 src2_data_0 = (int2)(src2.x, src2.y);
        int2 src2_data_1 = (int2)(src2.z, src2.x);
        int2 src2_data_2 = (int2)(src2.y, src2.z);

        ushort2 data_0 = *((__global ushort2 *)((__global char *)dst + dst_index + 0));
        ushort2 data_1 = *((__global ushort2 *)((__global char *)dst + dst_index + 4));
        ushort2 data_2 = *((__global ushort2 *)((__global char *)dst + dst_index + 8));

        int2 tmp_0 = convert_int2_sat(src1_data_0) - src2_data_0;
        int2 tmp_1 = convert_int2_sat(src1_data_1) - src2_data_1;
        int2 tmp_2 = convert_int2_sat(src1_data_2) - src2_data_2;

        tmp_0 = isMatSubScalar ? tmp_0 : -tmp_0;
        tmp_1 = isMatSubScalar ? tmp_1 : -tmp_1;
        tmp_2 = isMatSubScalar ? tmp_2 : -tmp_2;

        ushort2 tmp_data_0 = convert_ushort2_sat(tmp_0);
        ushort2 tmp_data_1 = convert_ushort2_sat(tmp_1);
        ushort2 tmp_data_2 = convert_ushort2_sat(tmp_2);

        data_0.xy = ((dst_index + 0 >= dst_start)) ? tmp_data_0.xy : data_0.xy;

        data_1.x  = ((dst_index + 0 >= dst_start) && (dst_index + 0 < dst_end))
                     ? tmp_data_1.x : data_1.x;
        data_1.y  = ((dst_index + 6 >= dst_start) && (dst_index + 6 < dst_end))
                     ? tmp_data_1.y : data_1.y;

        data_2.xy = ((dst_index + 6 >= dst_start) && (dst_index + 6 < dst_end))
                     ? tmp_data_2.xy : data_2.xy;

       *((__global ushort2 *)((__global char *)dst + dst_index + 0))= data_0;
       *((__global ushort2 *)((__global char *)dst + dst_index + 4))= data_1;
       *((__global ushort2 *)((__global char *)dst + dst_index + 8))= data_2;
    }
}
__kernel void arithm_s_sub_C3_D3 (__global   short *src1, int src1_step, int src1_offset,
                                  __global   short *dst,  int dst_step,  int dst_offset,
                                  int4 src2, int rows, int cols, int dst_step1, int isMatSubScalar)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        x = x << 1;

        #define dst_align (((dst_offset % dst_step) / 6 ) & 1)
        int src1_index = mad24(y, src1_step, (x * 6) + src1_offset - (dst_align * 6));

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + (x * 6) - (dst_align * 6));

        short2 src1_data_0 = vload2(0, (__global short *)((__global char *)src1 + src1_index + 0));
        short2 src1_data_1 = vload2(0, (__global short *)((__global char *)src1 + src1_index + 4));
        short2 src1_data_2 = vload2(0, (__global short *)((__global char *)src1 + src1_index + 8));

        int2 src2_data_0 = (int2)(src2.x, src2.y);
        int2 src2_data_1 = (int2)(src2.z, src2.x);
        int2 src2_data_2 = (int2)(src2.y, src2.z);

        short2 data_0 = *((__global short2 *)((__global char *)dst + dst_index + 0));
        short2 data_1 = *((__global short2 *)((__global char *)dst + dst_index + 4));
        short2 data_2 = *((__global short2 *)((__global char *)dst + dst_index + 8));

        int2 tmp_0 = convert_int2_sat(src1_data_0) - src2_data_0;
        int2 tmp_1 = convert_int2_sat(src1_data_1) - src2_data_1;
        int2 tmp_2 = convert_int2_sat(src1_data_2) - src2_data_2;

        tmp_0 = isMatSubScalar ? tmp_0 : -tmp_0;
        tmp_1 = isMatSubScalar ? tmp_1 : -tmp_1;
        tmp_2 = isMatSubScalar ? tmp_2 : -tmp_2;

        short2 tmp_data_0 = convert_short2_sat(tmp_0);
        short2 tmp_data_1 = convert_short2_sat(tmp_1);
        short2 tmp_data_2 = convert_short2_sat(tmp_2);

        data_0.xy = ((dst_index + 0 >= dst_start)) ? tmp_data_0.xy : data_0.xy;

        data_1.x  = ((dst_index + 0 >= dst_start) && (dst_index + 0 < dst_end))
                     ? tmp_data_1.x : data_1.x;
        data_1.y  = ((dst_index + 6 >= dst_start) && (dst_index + 6 < dst_end))
                     ? tmp_data_1.y : data_1.y;

        data_2.xy = ((dst_index + 6 >= dst_start) && (dst_index + 6 < dst_end))
                     ? tmp_data_2.xy : data_2.xy;

       *((__global short2 *)((__global char *)dst + dst_index + 0))= data_0;
       *((__global short2 *)((__global char *)dst + dst_index + 4))= data_1;
       *((__global short2 *)((__global char *)dst + dst_index + 8))= data_2;
    }
}
__kernel void arithm_s_sub_C3_D4 (__global   int *src1, int src1_step, int src1_offset,
                                  __global   int *dst,  int dst_step,  int dst_offset,
                                  int4 src2, int rows, int cols, int dst_step1, int isMatSubScalar)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x * 12) + src1_offset);
        int dst_index  = mad24(y, dst_step, dst_offset + (x * 12));

        int src1_data_0 = *((__global int *)((__global char *)src1 + src1_index + 0));
        int src1_data_1 = *((__global int *)((__global char *)src1 + src1_index + 4));
        int src1_data_2 = *((__global int *)((__global char *)src1 + src1_index + 8));

        int src2_data_0 = src2.x;
        int src2_data_1 = src2.y;
        int src2_data_2 = src2.z;

        int data_0 = *((__global int *)((__global char *)dst + dst_index + 0));
        int data_1 = *((__global int *)((__global char *)dst + dst_index + 4));
        int data_2 = *((__global int *)((__global char *)dst + dst_index + 8));

        long tmp_0 = (long)src1_data_0 - (long)src2_data_0;
        long tmp_1 = (long)src1_data_1 - (long)src2_data_1;
        long tmp_2 = (long)src1_data_2 - (long)src2_data_2;

        tmp_0 = isMatSubScalar ? tmp_0 : -tmp_0;
        tmp_1 = isMatSubScalar ? tmp_1 : -tmp_1;
        tmp_2 = isMatSubScalar ? tmp_2 : -tmp_2;

        int tmp_data_0 = convert_int_sat(tmp_0);
        int tmp_data_1 = convert_int_sat(tmp_1);
        int tmp_data_2 = convert_int_sat(tmp_2);

       *((__global int *)((__global char *)dst + dst_index + 0))= tmp_data_0;
       *((__global int *)((__global char *)dst + dst_index + 4))= tmp_data_1;
       *((__global int *)((__global char *)dst + dst_index + 8))= tmp_data_2;
    }
}
__kernel void arithm_s_sub_C3_D5 (__global   float *src1, int src1_step, int src1_offset,
                                  __global   float *dst,  int dst_step,  int dst_offset,
                                  float4 src2, int rows, int cols, int dst_step1, int isMatSubScalar)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x * 12) + src1_offset);
        int dst_index  = mad24(y, dst_step, dst_offset + (x * 12));

        float src1_data_0 = *((__global float *)((__global char *)src1 + src1_index + 0));
        float src1_data_1 = *((__global float *)((__global char *)src1 + src1_index + 4));
        float src1_data_2 = *((__global float *)((__global char *)src1 + src1_index + 8));

        float src2_data_0 = src2.x;
        float src2_data_1 = src2.y;
        float src2_data_2 = src2.z;

        float data_0 = *((__global float *)((__global char *)dst + dst_index + 0));
        float data_1 = *((__global float *)((__global char *)dst + dst_index + 4));
        float data_2 = *((__global float *)((__global char *)dst + dst_index + 8));

        float tmp_0 = src1_data_0 - src2_data_0;
        float tmp_1 = src1_data_1 - src2_data_1;
        float tmp_2 = src1_data_2 - src2_data_2;

        tmp_0 = isMatSubScalar ? tmp_0 : -tmp_0;
        tmp_1 = isMatSubScalar ? tmp_1 : -tmp_1;
        tmp_2 = isMatSubScalar ? tmp_2 : -tmp_2;

       *((__global float *)((__global char *)dst + dst_index + 0))= tmp_0;
       *((__global float *)((__global char *)dst + dst_index + 4))= tmp_1;
       *((__global float *)((__global char *)dst + dst_index + 8))= tmp_2;
    }
}

#if defined (DOUBLE_SUPPORT)
__kernel void arithm_s_sub_C3_D6 (__global   double *src1, int src1_step, int src1_offset,
                                  __global   double *dst,  int dst_step,  int dst_offset,
                                  double4 src2, int rows, int cols, int dst_step1, int isMatSubScalar)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x * 24) + src1_offset);
        int dst_index  = mad24(y, dst_step, dst_offset + (x * 24));

        double src1_data_0 = *((__global double *)((__global char *)src1 + src1_index + 0 ));
        double src1_data_1 = *((__global double *)((__global char *)src1 + src1_index + 8 ));
        double src1_data_2 = *((__global double *)((__global char *)src1 + src1_index + 16));

        double src2_data_0 = src2.x;
        double src2_data_1 = src2.y;
        double src2_data_2 = src2.z;

        double data_0 = *((__global double *)((__global char *)dst + dst_index + 0 ));
        double data_1 = *((__global double *)((__global char *)dst + dst_index + 8 ));
        double data_2 = *((__global double *)((__global char *)dst + dst_index + 16));

        double tmp_data_0 = src1_data_0 - src2_data_0;
        double tmp_data_1 = src1_data_1 - src2_data_1;
        double tmp_data_2 = src1_data_2 - src2_data_2;

        tmp_data_0 = isMatSubScalar ? tmp_data_0 : -tmp_data_0;
        tmp_data_1 = isMatSubScalar ? tmp_data_1 : -tmp_data_1;
        tmp_data_2 = isMatSubScalar ? tmp_data_2 : -tmp_data_2;

       *((__global double *)((__global char *)dst + dst_index + 0 ))= tmp_data_0;
       *((__global double *)((__global char *)dst + dst_index + 8 ))= tmp_data_1;
       *((__global double *)((__global char *)dst + dst_index + 16))= tmp_data_2;
    }
}
#endif
__kernel void arithm_s_sub_C4_D0 (__global   uchar *src1, int src1_step, int src1_offset,
                                  __global   uchar *dst,  int dst_step,  int dst_offset,
                                  int4 src2, int rows, int cols, int dst_step1, int isMatSubScalar)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 2) + src1_offset);
        int dst_index  = mad24(y, dst_step,  (x << 2) + dst_offset);

        uchar4 src_data1 = *((__global uchar4 *)(src1 + src1_index));

        int4 tmp = convert_int4_sat(src_data1) - src2;
        tmp = isMatSubScalar ? tmp : -tmp;
        uchar4 data = convert_uchar4_sat(tmp);

        *((__global uchar4 *)(dst + dst_index)) = data;
    }
}
__kernel void arithm_s_sub_C4_D2 (__global   ushort *src1, int src1_step, int src1_offset,
                                  __global   ushort *dst,  int dst_step,  int dst_offset,
                                  int4 src2, int rows, int cols, int dst_step1, int isMatSubScalar)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 3) + src1_offset);
        int dst_index  = mad24(y, dst_step,  (x << 3) + dst_offset);

        ushort4 src_data1 = *((__global ushort4 *)((__global char *)src1 + src1_index));

        int4 tmp = convert_int4_sat(src_data1) - src2;
        tmp = isMatSubScalar ? tmp : -tmp;
        ushort4 data = convert_ushort4_sat(tmp);

        *((__global ushort4 *)((__global char *)dst + dst_index)) = data;
    }
}
__kernel void arithm_s_sub_C4_D3 (__global   short *src1, int src1_step, int src1_offset,
                                  __global   short *dst,  int dst_step,  int dst_offset,
                                  int4 src2, int rows, int cols, int dst_step1, int isMatSubScalar)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 3) + src1_offset);
        int dst_index  = mad24(y, dst_step,  (x << 3) + dst_offset);

        short4 src_data1 = *((__global short4 *)((__global char *)src1 + src1_index));

        int4 tmp = convert_int4_sat(src_data1) - src2;
        tmp = isMatSubScalar ? tmp : -tmp;
        short4 data = convert_short4_sat(tmp);

        *((__global short4 *)((__global char *)dst + dst_index)) = data;
    }
}
__kernel void arithm_s_sub_C4_D4 (__global   int *src1, int src1_step, int src1_offset,
                                  __global   int *dst,  int dst_step,  int dst_offset,
                                  int4 src2, int rows, int cols, int dst_step1, int isMatSubScalar)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 4) + src1_offset);
        int dst_index  = mad24(y, dst_step,  (x << 4) + dst_offset);

        int4 src_data1 = *((__global int4 *)((__global char *)src1 + src1_index));

        long4 tmp = convert_long4_sat(src_data1) - convert_long4_sat(src2);
        tmp = isMatSubScalar ? tmp : -tmp;
        int4 data = convert_int4_sat(tmp);

        *((__global int4 *)((__global char *)dst + dst_index)) = data;
    }
}
__kernel void arithm_s_sub_C4_D5 (__global   float *src1, int src1_step, int src1_offset,
                                  __global   float *dst,  int dst_step,  int dst_offset,
                                  float4 src2, int rows, int cols, int dst_step1, int isMatSubScalar)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 4) + src1_offset);
        int dst_index  = mad24(y, dst_step,  (x << 4) + dst_offset);

        float4 src_data1 = *((__global float4 *)((__global char *)src1 + src1_index));

        float4 tmp = src_data1 - src2;
        tmp = isMatSubScalar ? tmp : -tmp;

        *((__global float4 *)((__global char *)dst + dst_index)) = tmp;
    }
}

#if defined (DOUBLE_SUPPORT)
__kernel void arithm_s_sub_C4_D6 (__global   double *src1, int src1_step, int src1_offset,
                                  __global   double *dst,  int dst_step,  int dst_offset,
                                  double4 src2, int rows, int cols, int dst_step1, int isMatSubScalar)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 5) + src1_offset);
        int dst_index  = mad24(y, dst_step,  (x << 5) + dst_offset);

        double4 src_data1 = *((__global double4 *)((__global char *)src1 + src1_index));

        double4 data = src_data1 - src2;
        data = isMatSubScalar ? data : -data;

        *((__global double4 *)((__global char *)dst + dst_index)) = data;
    }
}
#endif
