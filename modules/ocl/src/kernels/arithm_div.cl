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
#pragma OPENCL EXTENSION cl_khr_fp64:enable
typedef double F ;
typedef double4 F4;
#define convert_F4 convert_double4
#define convert_F  double
#else
typedef float F;
typedef float4 F4;
#define convert_F4 convert_float4
#define convert_F  float
#endif

uchar round2_uchar(F v){

    uchar v1 = convert_uchar_sat(round(v));
    //uchar v2 = convert_uchar_sat(v+(v>=0 ? 0.5 : -0.5));

    return v1;//(((v-v1)==0.5) && (v1%2==0)) ? v1 : v2;
}

ushort round2_ushort(F v){

    ushort v1 = convert_ushort_sat(round(v));
    //ushort v2 = convert_ushort_sat(v+(v>=0 ? 0.5 : -0.5));

    return v1;//(((v-v1)==0.5) && (v1%2==0)) ? v1 : v2;
}
short round2_short(F v){

    short v1 = convert_short_sat(round(v));
    //short v2 = convert_short_sat(v+(v>=0 ? 0.5 : -0.5));

    return v1;//(((v-v1)==0.5) && (v1%2==0)) ? v1 : v2;
}
int round2_int(F v){

    int v1 = convert_int_sat(round(v));
    //int v2 = convert_int_sat(v+(v>=0 ? 0.5 : -0.5));

    return v1;//(((v-v1)==0.5) && (v1%2==0)) ? v1 : v2;
}
///////////////////////////////////////////////////////////////////////////////////////
////////////////////////////divide///////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
/**********************************div*********************************************/
__kernel void arithm_div_D0 (__global uchar *src1, int src1_step, int src1_offset,
                             __global uchar *src2, int src2_step, int src2_offset,
                             __global uchar *dst,  int dst_step,  int dst_offset,
                             int rows, int cols, int dst_step1, F scalar)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        x = x << 2;

        #define dst_align (dst_offset & 3)
        int src1_index = mad24(y, src1_step, x + src1_offset - dst_align);
        int src2_index = mad24(y, src2_step, x + src2_offset - dst_align);

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + x & (int)0xfffffffc);

        uchar4 src1_data = vload4(0, src1 + src1_index);
        uchar4 src2_data = vload4(0, src2 + src2_index);
        uchar4 dst_data  = *((__global uchar4 *)(dst + dst_index));

        F4 tmp      = convert_F4(src1_data) * scalar;

        uchar4 tmp_data;
        tmp_data.x = ((tmp.x == 0) || (src2_data.x == 0)) ? 0 : round2_uchar(tmp.x / (F)src2_data.x);
        tmp_data.y = ((tmp.y == 0) || (src2_data.y == 0)) ? 0 : round2_uchar(tmp.y / (F)src2_data.y);
        tmp_data.z = ((tmp.z == 0) || (src2_data.z == 0)) ? 0 : round2_uchar(tmp.z / (F)src2_data.z);
        tmp_data.w = ((tmp.w == 0) || (src2_data.w == 0)) ? 0 : round2_uchar(tmp.w / (F)src2_data.w);

        dst_data.x = ((dst_index + 0 >= dst_start) && (dst_index + 0 < dst_end)) ? tmp_data.x : dst_data.x;
        dst_data.y = ((dst_index + 1 >= dst_start) && (dst_index + 1 < dst_end)) ? tmp_data.y : dst_data.y;
        dst_data.z = ((dst_index + 2 >= dst_start) && (dst_index + 2 < dst_end)) ? tmp_data.z : dst_data.z;
        dst_data.w = ((dst_index + 3 >= dst_start) && (dst_index + 3 < dst_end)) ? tmp_data.w : dst_data.w;

        *((__global uchar4 *)(dst + dst_index)) = dst_data;
    }
}

__kernel void arithm_div_D2 (__global ushort *src1, int src1_step, int src1_offset,
                             __global ushort *src2, int src2_step, int src2_offset,
                             __global ushort *dst,  int dst_step,  int dst_offset,
                             int rows, int cols, int dst_step1, F scalar)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        x = x << 2;

        #define dst_align ((dst_offset >> 1) & 3)
        int src1_index = mad24(y, src1_step, (x << 1) + src1_offset - (dst_align << 1));
        int src2_index = mad24(y, src2_step, (x << 1) + src2_offset - (dst_align << 1));

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + (x << 1) & (int)0xfffffff8);

        ushort4 src1_data = vload4(0, (__global ushort *)((__global char *)src1 + src1_index));
        ushort4 src2_data = vload4(0, (__global ushort *)((__global char *)src2 + src2_index));
        ushort4 dst_data = *((__global ushort4 *)((__global char *)dst + dst_index));

        F4 tmp   = convert_F4(src1_data) * scalar;

        ushort4 tmp_data;
        tmp_data.x = ((tmp.x == 0) || (src2_data.x == 0)) ? 0 : round2_ushort(tmp.x / (F)src2_data.x);
        tmp_data.y = ((tmp.y == 0) || (src2_data.y == 0)) ? 0 : round2_ushort(tmp.y / (F)src2_data.y);
        tmp_data.z = ((tmp.z == 0) || (src2_data.z == 0)) ? 0 : round2_ushort(tmp.z / (F)src2_data.z);
        tmp_data.w = ((tmp.w == 0) || (src2_data.w == 0)) ? 0 : round2_ushort(tmp.w / (F)src2_data.w);

        dst_data.x = ((dst_index + 0 >= dst_start) && (dst_index + 0 < dst_end)) ? tmp_data.x : dst_data.x;
        dst_data.y = ((dst_index + 2 >= dst_start) && (dst_index + 2 < dst_end)) ? tmp_data.y : dst_data.y;
        dst_data.z = ((dst_index + 4 >= dst_start) && (dst_index + 4 < dst_end)) ? tmp_data.z : dst_data.z;
        dst_data.w = ((dst_index + 6 >= dst_start) && (dst_index + 6 < dst_end)) ? tmp_data.w : dst_data.w;

        *((__global ushort4 *)((__global char *)dst + dst_index)) = dst_data;
    }
}
__kernel void arithm_div_D3 (__global short *src1, int src1_step, int src1_offset,
                             __global short *src2, int src2_step, int src2_offset,
                             __global short *dst,  int dst_step,  int dst_offset,
                             int rows, int cols, int dst_step1, F scalar)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        x = x << 2;

        #define dst_align ((dst_offset >> 1) & 3)
        int src1_index = mad24(y, src1_step, (x << 1) + src1_offset - (dst_align << 1));
        int src2_index = mad24(y, src2_step, (x << 1) + src2_offset - (dst_align << 1));

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + (x << 1) & (int)0xfffffff8);

        short4 src1_data = vload4(0, (__global short *)((__global char *)src1 + src1_index));
        short4 src2_data = vload4(0, (__global short *)((__global char *)src2 + src2_index));
        short4 dst_data = *((__global short4 *)((__global char *)dst + dst_index));

        F4 tmp   = convert_F4(src1_data) * scalar;

        short4 tmp_data;
        tmp_data.x = ((tmp.x == 0) || (src2_data.x == 0)) ? 0 : round2_short(tmp.x / (F)src2_data.x);
        tmp_data.y = ((tmp.y == 0) || (src2_data.y == 0)) ? 0 : round2_short(tmp.y / (F)src2_data.y);
        tmp_data.z = ((tmp.z == 0) || (src2_data.z == 0)) ? 0 : round2_short(tmp.z / (F)src2_data.z);
        tmp_data.w = ((tmp.w == 0) || (src2_data.w == 0)) ? 0 : round2_short(tmp.w / (F)src2_data.w);


        dst_data.x = ((dst_index + 0 >= dst_start) && (dst_index + 0 < dst_end)) ? tmp_data.x : dst_data.x;
        dst_data.y = ((dst_index + 2 >= dst_start) && (dst_index + 2 < dst_end)) ? tmp_data.y : dst_data.y;
        dst_data.z = ((dst_index + 4 >= dst_start) && (dst_index + 4 < dst_end)) ? tmp_data.z : dst_data.z;
        dst_data.w = ((dst_index + 6 >= dst_start) && (dst_index + 6 < dst_end)) ? tmp_data.w : dst_data.w;

        *((__global short4 *)((__global char *)dst + dst_index)) = dst_data;
    }
}

__kernel void arithm_div_D4 (__global int *src1, int src1_step, int src1_offset,
                             __global int *src2, int src2_step, int src2_offset,
                             __global int *dst,  int dst_step,  int dst_offset,
                             int rows, int cols, int dst_step1, F scalar)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 2) + src1_offset);
        int src2_index = mad24(y, src2_step, (x << 2) + src2_offset);
        int dst_index  = mad24(y, dst_step,  (x << 2) + dst_offset);

        int data1 = *((__global int *)((__global char *)src1 + src1_index));
        int data2 = *((__global int *)((__global char *)src2 + src2_index));

        F tmp  = (convert_F)(data1) * scalar;
        int tmp_data = (tmp == 0 || data2 == 0) ? 0 : round2_int(tmp / (convert_F)(data2));

        *((__global int *)((__global char *)dst + dst_index)) =tmp_data;
    }
}

__kernel void arithm_div_D5 (__global float *src1, int src1_step, int src1_offset,
                             __global float *src2, int src2_step, int src2_offset,
                             __global float *dst,  int dst_step,  int dst_offset,
                             int rows, int cols, int dst_step1, F scalar)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 2) + src1_offset);
        int src2_index = mad24(y, src2_step, (x << 2) + src2_offset);
        int dst_index  = mad24(y, dst_step,  (x << 2) + dst_offset);

        float data1 = *((__global float *)((__global char *)src1 + src1_index));
        float data2 = *((__global float *)((__global char *)src2 + src2_index));

        F tmp  = (convert_F)(data1) * scalar;
        float tmp_data = (tmp == 0 || data2 == 0) ? 0 : convert_float(tmp / (convert_F)(data2));

        *((__global float *)((__global char *)dst + dst_index)) = tmp_data;
    }
}

#if defined (DOUBLE_SUPPORT)
__kernel void arithm_div_D6 (__global double *src1, int src1_step, int src1_offset,
                             __global double *src2, int src2_step, int src2_offset,
                             __global double *dst,  int dst_step,  int dst_offset,
                             int rows, int cols, int dst_step1, double scalar)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 3) + src1_offset);
        int src2_index = mad24(y, src2_step, (x << 3) + src2_offset);
        int dst_index  = mad24(y, dst_step,  (x << 3) + dst_offset);

        double data1 = *((__global double *)((__global char *)src1 + src1_index));
        double data2 = *((__global double *)((__global char *)src2 + src2_index));

        double tmp  = data1 * scalar;
        double tmp_data = (tmp == 0 || data2 == 0) ? 0 : (tmp / data2);

        *((__global double *)((__global char *)dst + dst_index)) = tmp_data;
    }
}
#endif
/************************************div with scalar************************************/
__kernel void arithm_s_div_D0 (__global uchar *src, int src_step, int src_offset,
                               __global uchar *dst,  int dst_step,  int dst_offset,
                               int rows, int cols, int dst_step1, F scalar)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        x = x << 2;

        #define dst_align (dst_offset & 3)
        int src_index = mad24(y, src_step, x + src_offset - dst_align);

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + x & (int)0xfffffffc);

        uchar4 src_data = vload4(0, src + src_index);
        uchar4 dst_data  = *((__global uchar4 *)(dst + dst_index));

        uchar4 tmp_data;
        tmp_data.x = ((scalar == 0) || (src_data.x == 0)) ? 0 : round2_uchar(scalar / (F)src_data.x);
        tmp_data.y = ((scalar == 0) || (src_data.y == 0)) ? 0 : round2_uchar(scalar / (F)src_data.y);
        tmp_data.z = ((scalar == 0) || (src_data.z == 0)) ? 0 : round2_uchar(scalar / (F)src_data.z);
        tmp_data.w = ((scalar == 0) || (src_data.w == 0)) ? 0 : round2_uchar(scalar / (F)src_data.w);

        dst_data.x = ((dst_index + 0 >= dst_start) && (dst_index + 0 < dst_end)) ? tmp_data.x : dst_data.x;
        dst_data.y = ((dst_index + 1 >= dst_start) && (dst_index + 1 < dst_end)) ? tmp_data.y : dst_data.y;
        dst_data.z = ((dst_index + 2 >= dst_start) && (dst_index + 2 < dst_end)) ? tmp_data.z : dst_data.z;
        dst_data.w = ((dst_index + 3 >= dst_start) && (dst_index + 3 < dst_end)) ? tmp_data.w : dst_data.w;

        *((__global uchar4 *)(dst + dst_index)) = dst_data;
    }
}

__kernel void arithm_s_div_D2 (__global ushort *src, int src_step, int src_offset,
                               __global ushort *dst,  int dst_step,  int dst_offset,
                               int rows, int cols, int dst_step1, F scalar)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        x = x << 2;

        #define dst_align ((dst_offset >> 1) & 3)
        int src_index = mad24(y, src_step, (x << 1) + src_offset - (dst_align << 1));

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + (x << 1) & (int)0xfffffff8);

        ushort4 src_data = vload4(0, (__global ushort *)((__global char *)src + src_index));
        ushort4 dst_data = *((__global ushort4 *)((__global char *)dst + dst_index));

        ushort4 tmp_data;
        tmp_data.x = ((scalar == 0) || (src_data.x == 0)) ? 0 : round2_ushort(scalar / (F)src_data.x);
        tmp_data.y = ((scalar == 0) || (src_data.y == 0)) ? 0 : round2_ushort(scalar / (F)src_data.y);
        tmp_data.z = ((scalar == 0) || (src_data.z == 0)) ? 0 : round2_ushort(scalar / (F)src_data.z);
        tmp_data.w = ((scalar == 0) || (src_data.w == 0)) ? 0 : round2_ushort(scalar / (F)src_data.w);

        dst_data.x = ((dst_index + 0 >= dst_start) && (dst_index + 0 < dst_end)) ? tmp_data.x : dst_data.x;
        dst_data.y = ((dst_index + 2 >= dst_start) && (dst_index + 2 < dst_end)) ? tmp_data.y : dst_data.y;
        dst_data.z = ((dst_index + 4 >= dst_start) && (dst_index + 4 < dst_end)) ? tmp_data.z : dst_data.z;
        dst_data.w = ((dst_index + 6 >= dst_start) && (dst_index + 6 < dst_end)) ? tmp_data.w : dst_data.w;

        *((__global ushort4 *)((__global char *)dst + dst_index)) = dst_data;
    }
}
__kernel void arithm_s_div_D3 (__global short *src, int src_step, int src_offset,
                               __global short *dst,  int dst_step,  int dst_offset,
                               int rows, int cols, int dst_step1, F scalar)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        x = x << 2;

        #define dst_align ((dst_offset >> 1) & 3)
        int src_index = mad24(y, src_step, (x << 1) + src_offset - (dst_align << 1));

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + (x << 1) & (int)0xfffffff8);

        short4 src_data = vload4(0, (__global short *)((__global char *)src + src_index));
        short4 dst_data = *((__global short4 *)((__global char *)dst + dst_index));

        short4 tmp_data;
        tmp_data.x = ((scalar == 0) || (src_data.x == 0)) ? 0 : round2_short(scalar / (F)src_data.x);
        tmp_data.y = ((scalar == 0) || (src_data.y == 0)) ? 0 : round2_short(scalar / (F)src_data.y);
        tmp_data.z = ((scalar == 0) || (src_data.z == 0)) ? 0 : round2_short(scalar / (F)src_data.z);
        tmp_data.w = ((scalar == 0) || (src_data.w == 0)) ? 0 : round2_short(scalar / (F)src_data.w);


        dst_data.x = ((dst_index + 0 >= dst_start) && (dst_index + 0 < dst_end)) ? tmp_data.x : dst_data.x;
        dst_data.y = ((dst_index + 2 >= dst_start) && (dst_index + 2 < dst_end)) ? tmp_data.y : dst_data.y;
        dst_data.z = ((dst_index + 4 >= dst_start) && (dst_index + 4 < dst_end)) ? tmp_data.z : dst_data.z;
        dst_data.w = ((dst_index + 6 >= dst_start) && (dst_index + 6 < dst_end)) ? tmp_data.w : dst_data.w;

        *((__global short4 *)((__global char *)dst + dst_index)) = dst_data;
    }
}

__kernel void arithm_s_div_D4 (__global int *src, int src_step, int src_offset,
                               __global int *dst,  int dst_step,  int dst_offset,
                               int rows, int cols, int dst_step1, F scalar)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src_index = mad24(y, src_step, (x << 2) + src_offset);
        int dst_index  = mad24(y, dst_step,  (x << 2) + dst_offset);

        int data = *((__global int *)((__global char *)src + src_index));

        int tmp_data = (scalar == 0 || data == 0) ? 0 : round2_int(scalar / (convert_F)(data));

        *((__global int *)((__global char *)dst + dst_index)) =tmp_data;
    }
}

__kernel void arithm_s_div_D5 (__global float *src, int src_step, int src_offset,
                               __global float *dst,  int dst_step,  int dst_offset,
                               int rows, int cols, int dst_step1, F scalar)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src_index = mad24(y, src_step, (x << 2) + src_offset);
        int dst_index  = mad24(y, dst_step,  (x << 2) + dst_offset);

        float data = *((__global float *)((__global char *)src + src_index));

        float tmp_data = (scalar == 0 || data == 0) ? 0 : convert_float(scalar / (convert_F)(data));

        *((__global float *)((__global char *)dst + dst_index)) = tmp_data;
    }
}

#if defined (DOUBLE_SUPPORT)
__kernel void arithm_s_div_D6 (__global double *src, int src_step, int src_offset,
                               __global double *dst,  int dst_step,  int dst_offset,
                               int rows, int cols, int dst_step1, double scalar)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src_index = mad24(y, src_step, (x << 3) + src_offset);
        int dst_index  = mad24(y, dst_step,  (x << 3) + dst_offset);

        double data = *((__global double *)((__global char *)src + src_index));

        double tmp_data = (scalar == 0 || data == 0) ? 0 : (scalar / data);

        *((__global double *)((__global char *)dst + dst_index)) = tmp_data;
    }
}
#endif


