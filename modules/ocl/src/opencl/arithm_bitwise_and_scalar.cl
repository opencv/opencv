////////////////////////////////////////////////////////////////////////////////////////
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
//    Jiang Liyuan, jlyuan001.good@163.com
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
//
#if defined (DOUBLE_SUPPORT)
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#elif defined (cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#endif
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////BITWISE_AND////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
/**************************************and with scalar without mask**************************************/
__kernel void arithm_s_bitwise_and_C1_D0 (
        __global   uchar *src1, int src1_step, int src1_offset,
        __global   uchar *dst,  int dst_step,  int dst_offset,
        uchar4 src2, int rows, int cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        x = x << 2;
        
#ifdef dst_align
#undef dst_align
#endif
#define dst_align (dst_offset & 3)
        int src1_index = mad24(y, src1_step, x + src1_offset - dst_align);

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + x & (int)0xfffffffc);

        uchar4 src1_data = vload4(0, src1 + src1_index);
        uchar4 src2_data = (uchar4)(src2.x, src2.x, src2.x, src2.x);

        uchar4 data = *((__global uchar4 *)(dst + dst_index));
        uchar4 tmp_data = src1_data & src2_data;

        data.x = ((dst_index + 0 >= dst_start) && (dst_index + 0 < dst_end)) ? tmp_data.x : data.x;
        data.y = ((dst_index + 1 >= dst_start) && (dst_index + 1 < dst_end)) ? tmp_data.y : data.y;
        data.z = ((dst_index + 2 >= dst_start) && (dst_index + 2 < dst_end)) ? tmp_data.z : data.z;
        data.w = ((dst_index + 3 >= dst_start) && (dst_index + 3 < dst_end)) ? tmp_data.w : data.w;

        *((__global uchar4 *)(dst + dst_index)) = data;
    }
}


__kernel void arithm_s_bitwise_and_C1_D1 (
        __global   char *src1, int src1_step, int src1_offset,
        __global   char *dst,  int dst_step,  int dst_offset,
        char4 src2, int rows, int cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        x = x << 2;
        
#ifdef dst_align
#undef dst_align
#endif
#define dst_align (dst_offset & 3)
        int src1_index = mad24(y, src1_step, x + src1_offset - dst_align);

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + x & (int)0xfffffffc);

        char4 src1_data = vload4(0, src1 + src1_index);
        char4 src2_data = (char4)(src2.x, src2.x, src2.x, src2.x);

        char4 data = *((__global char4 *)(dst + dst_index));
        char4 tmp_data = src1_data & src2_data;

        data.x = ((dst_index + 0 >= dst_start) && (dst_index + 0 < dst_end)) ? tmp_data.x : data.x;
        data.y = ((dst_index + 1 >= dst_start) && (dst_index + 1 < dst_end)) ? tmp_data.y : data.y;
        data.z = ((dst_index + 2 >= dst_start) && (dst_index + 2 < dst_end)) ? tmp_data.z : data.z;
        data.w = ((dst_index + 3 >= dst_start) && (dst_index + 3 < dst_end)) ? tmp_data.w : data.w;

        *((__global char4 *)(dst + dst_index)) = data;
    }
}

__kernel void arithm_s_bitwise_and_C1_D2 (
        __global   ushort *src1, int src1_step, int src1_offset,
        __global   ushort *dst,  int dst_step,  int dst_offset,
        ushort4 src2, int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        x = x << 1;
        
#ifdef dst_align
#undef dst_align
#endif
#define dst_align ((dst_offset >> 1) & 1)
        int src1_index = mad24(y, src1_step, (x << 1) + src1_offset - (dst_align << 1));

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + (x << 1) & (int)0xfffffffc);

        ushort2 src1_data = vload2(0, (__global ushort *)((__global char *)src1 + src1_index));
        ushort2 src2_data = (ushort2)(src2.x, src2.x);

        ushort2 data = *((__global ushort2 *)((__global uchar *)dst + dst_index));
        ushort2 tmp_data = src1_data & src2_data;

        data.x = (dst_index + 0 >= dst_start) ? tmp_data.x : data.x;
        data.y = (dst_index + 2 <  dst_end  ) ? tmp_data.y : data.y;

        *((__global ushort2 *)((__global uchar *)dst + dst_index)) = data;
    }
}
__kernel void arithm_s_bitwise_and_C1_D3 (
        __global   short *src1, int src1_step, int src1_offset,
        __global   short *dst,  int dst_step,  int dst_offset,
        short4 src2, int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        x = x << 1;
        
#ifdef dst_align
#undef dst_align
#endif
#define dst_align ((dst_offset >> 1) & 1)
        int src1_index = mad24(y, src1_step, (x << 1) + src1_offset - (dst_align << 1));

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + (x << 1) & (int)0xfffffffc);

        short2 src1_data = vload2(0, (__global short *)((__global char *)src1 + src1_index));
        short2 src2_data = (short2)(src2.x, src2.x);
        short2 data = *((__global short2 *)((__global uchar *)dst + dst_index));

        short2 tmp_data = src1_data & src2_data;

        data.x = (dst_index + 0 >= dst_start) ? tmp_data.x : data.x;
        data.y = (dst_index + 2 <  dst_end  ) ? tmp_data.y : data.y;

        *((__global short2 *)((__global uchar *)dst + dst_index)) = data;
    }
}
__kernel void arithm_s_bitwise_and_C1_D4 (
        __global   int *src1, int src1_step, int src1_offset,
        __global   int *dst,  int dst_step,  int dst_offset,
        int4 src2, int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 2) + src1_offset);
        int dst_index  = mad24(y, dst_step,  (x << 2) + dst_offset);

        int src_data1 = *((__global int *)((__global char *)src1 + src1_index));
        int src_data2 = src2.x;

        int data = src_data1 & src_data2;

        *((__global int *)((__global char *)dst + dst_index)) = data;
    }
}
__kernel void arithm_s_bitwise_and_C1_D5 (
        __global   char *src1, int src1_step, int src1_offset,
        __global   char *dst,  int dst_step,  int dst_offset,
        char16 src2, int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 2) + src1_offset);

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step,  (x << 2) + dst_offset);

        char4 src1_data = *((__global char4 *)((__global char *)src1 + src1_index));
        char4 src2_data = (char4)(src2.s0, src2.s1, src2.s2, src2.s3);

        char4 data  = *((__global char4 *)((__global char *)dst  + dst_index));
        char4 tmp_data = src1_data & src2_data;

        data.x = ((dst_index + 0 >= dst_start) && (dst_index + 0 < dst_end)) ? tmp_data.x : data.x;
        data.y = ((dst_index + 1 >= dst_start) && (dst_index + 1 < dst_end)) ? tmp_data.y : data.y;
        data.z = ((dst_index + 2 >= dst_start) && (dst_index + 2 < dst_end)) ? tmp_data.z : data.z;
        data.w = ((dst_index + 3 >= dst_start) && (dst_index + 3 < dst_end)) ? tmp_data.w : data.w;

        *((__global char4 *)((__global char *)dst + dst_index)) = data;
    }
}
#if defined (DOUBLE_SUPPORT)
__kernel void arithm_s_bitwise_and_C1_D6 (
        __global short *src1, int src1_step, int src1_offset,
        __global short *dst,  int dst_step,  int dst_offset,
        short16 src2, int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 3) + src1_offset);
        int dst_index  = mad24(y, dst_step,  (x << 3) + dst_offset);

        short4 src1_data = *((__global short4 *)((__global char *)src1 + src1_index));
        short4 src2_data = (short4)(src2.s0, src2.s1, src2.s2, src2.s3);

        short4 tmp_data = src1_data & src2_data;

        *((__global short4 *)((__global char *)dst + dst_index)) = tmp_data;
    }
}
#endif
__kernel void arithm_s_bitwise_and_C2_D0 (
        __global   uchar *src1, int src1_step, int src1_offset,
        __global   uchar *dst,  int dst_step,  int dst_offset,
        uchar4 src2, int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        x = x << 1;
        
#ifdef dst_align
#undef dst_align
#endif
#define dst_align ((dst_offset >> 1) & 1)
        int src1_index = mad24(y, src1_step, (x << 1) + src1_offset - (dst_align << 1));

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + (x << 1) & (int)0xfffffffc);

        uchar4 src1_data = vload4(0, src1 + src1_index);
        uchar4 src2_data = (uchar4)(src2.x, src2.y, src2.x, src2.y);

        uchar4 data = *((__global uchar4 *)(dst + dst_index));
        uchar4 tmp_data = src1_data & src2_data;


        data.xy = (dst_index + 0 >= dst_start) ? tmp_data.xy : data.xy;
        data.zw = (dst_index + 2 <  dst_end  ) ? tmp_data.zw : data.zw;

        *((__global uchar4 *)(dst + dst_index)) = data;
    }
}


__kernel void arithm_s_bitwise_and_C2_D1 (
        __global   char *src1, int src1_step, int src1_offset,
        __global   char *dst,  int dst_step,  int dst_offset,
        char4 src2, int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        x = x << 1;
        
#ifdef dst_align
#undef dst_align
#endif
#define dst_align ((dst_offset >> 1) & 1)
        int src1_index = mad24(y, src1_step, (x << 1) + src1_offset - (dst_align << 1));

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + (x << 1) & (int)0xfffffffc);

        char4 src1_data = vload4(0, src1 + src1_index);
        char4 src2_data = (char4)(src2.x, src2.y, src2.x, src2.y);

        char4 data = *((__global char4 *)(dst + dst_index));
        char4 tmp_data = src1_data & src2_data;

        data.xy = (dst_index + 0 >= dst_start) ? tmp_data.xy : data.xy;
        data.zw = (dst_index + 2 <  dst_end  ) ? tmp_data.zw : data.zw;

        *((__global char4 *)(dst + dst_index)) = data;
    }
}

__kernel void arithm_s_bitwise_and_C2_D2 (
        __global   ushort *src1, int src1_step, int src1_offset,
        __global   ushort *dst,  int dst_step,  int dst_offset,
        ushort4 src2, int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 2) + src1_offset);
        int dst_index  = mad24(y, dst_step,  (x << 2) + dst_offset);

        ushort2 src_data1 = *((__global ushort2 *)((__global char *)src1 + src1_index));
        ushort2 src_data2 = (ushort2)(src2.x, src2.y);

        ushort2 data = src_data1 & src_data2;

        *((__global ushort2 *)((__global char *)dst + dst_index)) = data;
    }
}
__kernel void arithm_s_bitwise_and_C2_D3 (
        __global   short *src1, int src1_step, int src1_offset,
        __global   short *dst,  int dst_step,  int dst_offset,
        short4 src2, int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 2) + src1_offset);
        int dst_index  = mad24(y, dst_step,  (x << 2) + dst_offset);

        short2 src_data1 = *((__global short2 *)((__global char *)src1 + src1_index));
        short2 src_data2 = (short2)(src2.x, src2.y);

        short2 data = src_data1 & src_data2;

        *((__global short2 *)((__global char *)dst + dst_index)) = data;
    }
}
__kernel void arithm_s_bitwise_and_C2_D4 (
        __global   int *src1, int src1_step, int src1_offset,
        __global   int *dst,  int dst_step,  int dst_offset,
        int4 src2, int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 3) + src1_offset);
        int dst_index  = mad24(y, dst_step,  (x << 3) + dst_offset);

        int2 src_data1 = *((__global int2 *)((__global char *)src1 + src1_index));
        int2 src_data2 = (int2)(src2.x, src2.y);

        int2 data = src_data1 & src_data2;
        *((__global int2 *)((__global char *)dst + dst_index)) = data;
    }
}
__kernel void arithm_s_bitwise_and_C2_D5 (
        __global   char *src1, int src1_step, int src1_offset,
        __global   char *dst,  int dst_step,  int dst_offset,
        char16 src2, int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 3) + src1_offset);
        int dst_index  = mad24(y, dst_step,  (x << 3) + dst_offset);

        char8 src1_data = *((__global char8 *)((__global char *)src1 + src1_index));
        char8 src2_data = (char8)(src2.s0, src2.s1, src2.s2, src2.s3, src2.s4, src2.s5, src2.s6, src2.s7);

        char8 tmp_data = src1_data & src2_data;

        *((__global char8 *)((__global char *)dst + dst_index)) = tmp_data;
    }
}
#if defined (DOUBLE_SUPPORT)
__kernel void arithm_s_bitwise_and_C2_D6 (
        __global short *src1, int src1_step, int src1_offset,
        __global short *dst,  int dst_step,  int dst_offset,
        short16 src2, int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 4) + src1_offset);
        int dst_index  = mad24(y, dst_step,  (x << 4) + dst_offset);

        short8 src1_data = *((__global short8 *)((__global char *)src1 + src1_index));
        short8 src2_data = (short8)(src2.s0, src2.s1, src2.s2, src2.s3, src2.s4, src2.s5, src2.s6, src2.s7);

        short8 tmp_data = src1_data & src2_data;

        *((__global short8 *)((__global char *)dst + dst_index)) = tmp_data;
    }
}
#endif

__kernel void arithm_s_bitwise_and_C4_D0 (
        __global   uchar *src1, int src1_step, int src1_offset,
        __global   uchar *dst,  int dst_step,  int dst_offset,
        uchar4 src2, int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 2) + src1_offset);
        int dst_index  = mad24(y, dst_step,  (x << 2) + dst_offset);

        uchar4 src_data1 = *((__global uchar4 *)(src1 + src1_index));

        uchar4 data = src_data1 & src2;

        *((__global uchar4 *)(dst + dst_index)) = data;
    }
}


__kernel void arithm_s_bitwise_and_C4_D1 (
        __global   char *src1, int src1_step, int src1_offset,
        __global   char *dst,  int dst_step,  int dst_offset,
        char4 src2, int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 2) + src1_offset);
        int dst_index  = mad24(y, dst_step,  (x << 2) + dst_offset);

        char4 src_data1 = *((__global char4 *)(src1 + src1_index));

        char4 data = src_data1 & src2;

        *((__global char4 *)(dst + dst_index)) = data;
    }
}

__kernel void arithm_s_bitwise_and_C4_D2 (
        __global   ushort *src1, int src1_step, int src1_offset,
        __global   ushort *dst,  int dst_step,  int dst_offset,
        ushort4 src2, int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 3) + src1_offset);
        int dst_index  = mad24(y, dst_step,  (x << 3) + dst_offset);

        ushort4 src_data1 = *((__global ushort4 *)((__global char *)src1 + src1_index));

        ushort4 data = src_data1 & src2;

        *((__global ushort4 *)((__global char *)dst + dst_index)) = data;
    }
}
__kernel void arithm_s_bitwise_and_C4_D3 (
        __global   short *src1, int src1_step, int src1_offset,
        __global   short *dst,  int dst_step,  int dst_offset,
        short4 src2, int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 3) + src1_offset);
        int dst_index  = mad24(y, dst_step,  (x << 3) + dst_offset);

        short4 src_data1 = *((__global short4 *)((__global char *)src1 + src1_index));

        short4 data = src_data1 & src2;

        *((__global short4 *)((__global char *)dst + dst_index)) = data;
    }
}
__kernel void arithm_s_bitwise_and_C4_D4 (
        __global   int *src1, int src1_step, int src1_offset,
        __global   int *dst,  int dst_step,  int dst_offset,
        int4 src2, int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 4) + src1_offset);
        int dst_index  = mad24(y, dst_step,  (x << 4) + dst_offset);

        int4 src_data1 = *((__global int4 *)((__global char *)src1 + src1_index));

        int4 data = src_data1 & src2;

        *((__global int4 *)((__global char *)dst + dst_index)) = data;
    }
}
__kernel void arithm_s_bitwise_and_C4_D5 (
        __global   char *src1, int src1_step, int src1_offset,
        __global   char *dst,  int dst_step,  int dst_offset,
        char16 src2, int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 4) + src1_offset);
        int dst_index  = mad24(y, dst_step,  (x << 4) + dst_offset);

        char16 src1_data = *((__global char16 *)((__global char *)src1 + src1_index));
        char16 src2_data = (char16)(src2.s0, src2.s1, src2.s2, src2.s3, src2.s4, src2.s5, src2.s6, src2.s7,
                                    src2.s8, src2.s9, src2.sa, src2.sb, src2.sc, src2.sd, src2.se, src2.sf);

        char16 tmp_data = src1_data & src2_data;

        *((__global char16 *)((__global char *)dst + dst_index)) = tmp_data;
    }
}
#if defined (DOUBLE_SUPPORT)
__kernel void arithm_s_bitwise_and_C4_D6 (
        __global short *src1, int src1_step, int src1_offset,
        __global short *dst,  int dst_step,  int dst_offset,
        short16 src2, int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 5) + src1_offset);
        int dst_index  = mad24(y, dst_step,  (x << 5) + dst_offset);

        short4 src1_data_0 = *((__global short4 *)((__global char *)src1 + src1_index + 0));
        short4 src1_data_1 = *((__global short4 *)((__global char *)src1 + src1_index + 8));
        short4 src1_data_2 = *((__global short4 *)((__global char *)src1 + src1_index + 16));
        short4 src1_data_3 = *((__global short4 *)((__global char *)src1 + src1_index + 24));

        short4 src2_data_0 = (short4)(src2.s0, src2.s1, src2.s2, src2.s3);
        short4 src2_data_1 = (short4)(src2.s4, src2.s5, src2.s6, src2.s7);
        short4 src2_data_2 = (short4)(src2.s8, src2.s9, src2.sa, src2.sb);
        short4 src2_data_3 = (short4)(src2.sc, src2.sd, src2.se, src2.sf);

        short4 tmp_data_0 = src1_data_0 & src2_data_0;
        short4 tmp_data_1 = src1_data_1 & src2_data_1;
        short4 tmp_data_2 = src1_data_2 & src2_data_2;
        short4 tmp_data_3 = src1_data_3 & src2_data_3;

        *((__global short4 *)((__global char *)dst + dst_index + 0 ))= tmp_data_0;
        *((__global short4 *)((__global char *)dst + dst_index + 8 ))= tmp_data_1;
        *((__global short4 *)((__global char *)dst + dst_index + 16))= tmp_data_2;
        *((__global short4 *)((__global char *)dst + dst_index + 24))= tmp_data_3;

    }
}
#endif
