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
//M*/
#if defined (DOUBLE_SUPPORT)
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#elif defined (cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#endif
#endif
//////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////BITWISE_OR////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
/**************************************bitwise_or with mask**************************************/
__kernel void arithm_bitwise_or_with_mask_C1_D0 (
        __global uchar *src1, int src1_step, int src1_offset,
        __global uchar *src2, int src2_step, int src2_offset,
        __global uchar *mask, int mask_step, int mask_offset,
        __global uchar *dst,  int dst_step,  int dst_offset,
        int rows, int cols, int dst_step1)
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
        int src2_index = mad24(y, src2_step, x + src2_offset - dst_align);
        int mask_index = mad24(y, mask_step, x + mask_offset - dst_align);

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + x & (int)0xfffffffc);

        uchar4 src1_data = vload4(0, src1 + src1_index);
        uchar4 src2_data = vload4(0, src2 + src2_index);
        uchar4 mask_data = vload4(0, mask + mask_index);

        uchar4 data = *((__global uchar4 *)(dst + dst_index));
        uchar4 tmp_data = src1_data | src2_data;

        data.x = ((mask_data.x) && (dst_index + 0 >= dst_start) && (dst_index + 0 < dst_end)) ? tmp_data.x : data.x;
        data.y = ((mask_data.y) && (dst_index + 1 >= dst_start) && (dst_index + 1 < dst_end)) ? tmp_data.y : data.y;
        data.z = ((mask_data.z) && (dst_index + 2 >= dst_start) && (dst_index + 2 < dst_end)) ? tmp_data.z : data.z;
        data.w = ((mask_data.w) && (dst_index + 3 >= dst_start) && (dst_index + 3 < dst_end)) ? tmp_data.w : data.w;

        *((__global uchar4 *)(dst + dst_index)) = data;
    }
}



__kernel void arithm_bitwise_or_with_mask_C1_D1 (
        __global char *src1, int src1_step, int src1_offset,
        __global char *src2, int src2_step, int src2_offset,
        __global uchar *mask, int mask_step, int mask_offset,
        __global char *dst,  int dst_step,  int dst_offset,
        int rows, int cols, int dst_step1)
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
        int src2_index = mad24(y, src2_step, x + src2_offset - dst_align);
        int mask_index = mad24(y, mask_step, x + mask_offset - dst_align);

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + x & (int)0xfffffffc);

        char4 src1_data = vload4(0, src1 + src1_index);
        char4 src2_data = vload4(0, src2 + src2_index);
        uchar4 mask_data = vload4(0, mask + mask_index);

        char4 data = *((__global char4 *)(dst + dst_index));
        char4 tmp_data = src1_data | src2_data;

        data.x = convert_char((mask_data.x) && (dst_index + 0 >= dst_start) && (dst_index + 0 < dst_end)) ? tmp_data.x : data.x;
        data.y = convert_char((mask_data.y) && (dst_index + 1 >= dst_start) && (dst_index + 1 < dst_end)) ? tmp_data.y : data.y;
        data.z = convert_char((mask_data.z) && (dst_index + 2 >= dst_start) && (dst_index + 2 < dst_end)) ? tmp_data.z : data.z;
        data.w = convert_char((mask_data.w) && (dst_index + 3 >= dst_start) && (dst_index + 3 < dst_end)) ? tmp_data.w : data.w;

        *((__global char4 *)(dst + dst_index)) = data;
    }
}



__kernel void arithm_bitwise_or_with_mask_C1_D2 (
        __global ushort *src1, int src1_step, int src1_offset,
        __global ushort *src2, int src2_step, int src2_offset,
        __global uchar  *mask, int mask_step, int mask_offset,
        __global ushort *dst,  int dst_step,  int dst_offset,
        int rows, int cols, int dst_step1)
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
        int src2_index = mad24(y, src2_step, (x << 1) + src2_offset - (dst_align << 1));
        int mask_index = mad24(y, mask_step, x + mask_offset - dst_align);

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + (x << 1) & (int)0xfffffffc);

        ushort2 src1_data = vload2(0, (__global ushort *)((__global char *)src1 + src1_index));
        ushort2 src2_data = vload2(0, (__global ushort *)((__global char *)src2 + src2_index));
        uchar2  mask_data = vload2(0, mask + mask_index);

        ushort2 data = *((__global ushort2 *)((__global uchar *)dst + dst_index));
        ushort2 tmp_data = src1_data | src2_data;

        data.x = convert_ushort((mask_data.x) && (dst_index + 0 >= dst_start) && (dst_index + 0 < dst_end)) ? tmp_data.x : data.x;
        data.y = convert_ushort((mask_data.y) && (dst_index + 2 >= dst_start) && (dst_index + 2 < dst_end)) ? tmp_data.y : data.y;

        *((__global ushort2 *)((__global uchar *)dst + dst_index)) = data;
    }
}



__kernel void arithm_bitwise_or_with_mask_C1_D3 (
        __global short *src1, int src1_step, int src1_offset,
        __global short *src2, int src2_step, int src2_offset,
        __global uchar *mask, int mask_step, int mask_offset,
        __global short *dst,  int dst_step,  int dst_offset,
        int rows, int cols, int dst_step1)
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
        int src2_index = mad24(y, src2_step, (x << 1) + src2_offset - (dst_align << 1));
        int mask_index = mad24(y, mask_step, x + mask_offset - dst_align);

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + (x << 1) & (int)0xfffffffc);

        short2 src1_data = vload2(0, (__global short *)((__global char *)src1 + src1_index));
        short2 src2_data = vload2(0, (__global short *)((__global char *)src2 + src2_index));
        uchar2  mask_data = vload2(0, mask + mask_index);

        short2 data = *((__global short2 *)((__global uchar *)dst + dst_index));
        short2 tmp_data = src1_data | src2_data;

        data.x = convert_short((mask_data.x) && (dst_index + 0 >= dst_start) && (dst_index + 0 < dst_end)) ? tmp_data.x : data.x;
        data.y = convert_short((mask_data.y) && (dst_index + 2 >= dst_start) && (dst_index + 2 < dst_end)) ? tmp_data.y : data.y;

        *((__global short2 *)((__global uchar *)dst + dst_index)) = data;
    }
}



__kernel void arithm_bitwise_or_with_mask_C1_D4 (
        __global int   *src1, int src1_step, int src1_offset,
        __global int   *src2, int src2_step, int src2_offset,
        __global uchar *mask, int mask_step, int mask_offset,
        __global int   *dst,  int dst_step,  int dst_offset,
        int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 2) + src1_offset);
        int src2_index = mad24(y, src2_step, (x << 2) + src2_offset);
        int mask_index = mad24(y, mask_step,  x       + mask_offset);
        int dst_index  = mad24(y, dst_step,  (x << 2) + dst_offset);

        uchar mask_data = *(mask + mask_index);

        int src_data1 = *((__global int *)((__global char *)src1 + src1_index));
        int src_data2 = *((__global int *)((__global char *)src2 + src2_index));
        int dst_data  = *((__global int *)((__global char *)dst  + dst_index));

        int data = src_data1 | src_data2;
        data = mask_data ? data : dst_data;

        *((__global int *)((__global char *)dst + dst_index)) = data;
    }
}



__kernel void arithm_bitwise_or_with_mask_C1_D5 (
        __global char *src1, int src1_step, int src1_offset,
        __global char *src2, int src2_step, int src2_offset,
        __global uchar *mask, int mask_step, int mask_offset,
        __global char *dst,  int dst_step,  int dst_offset,
        int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 2) + src1_offset);
        int src2_index = mad24(y, src2_step, (x << 2) + src2_offset);
        int mask_index = mad24(y, mask_step,  x       + mask_offset);
        int dst_index  = mad24(y, dst_step,  (x << 2) + dst_offset);

        uchar mask_data = *(mask + mask_index);

        char4 src_data1 = *((__global char4 *)((__global char *)src1 + src1_index));
        char4 src_data2 = *((__global char4 *)((__global char *)src2 + src2_index));
        char4 dst_data  = *((__global char4 *)((__global char *)dst  + dst_index));

        char4 data = src_data1 | src_data2;
        data = mask_data ? data : dst_data;

        *((__global char4 *)((__global char *)dst + dst_index)) = data;
    }
}


#if defined (DOUBLE_SUPPORT)
__kernel void arithm_bitwise_or_with_mask_C1_D6 (
        __global char *src1, int src1_step, int src1_offset,
        __global char *src2, int src2_step, int src2_offset,
        __global uchar *mask, int mask_step, int mask_offset,
        __global char *dst,  int dst_step,  int dst_offset,
        int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 3) + src1_offset);
        int src2_index = mad24(y, src2_step, (x << 3) + src2_offset);
        int mask_index = mad24(y, mask_step,  x       + mask_offset);
        int dst_index  = mad24(y, dst_step,  (x << 3) + dst_offset);

        uchar mask_data = *(mask + mask_index);

        char8 src_data1 = *((__global char8 *)((__global char *)src1 + src1_index));
        char8 src_data2 = *((__global char8 *)((__global char *)src2 + src2_index));
        char8 dst_data  = *((__global char8 *)((__global char *)dst  + dst_index));

        char8 data = src_data1 | src_data2;
        data = mask_data ? data : dst_data;

        *((__global char8 *)((__global char *)dst + dst_index)) = data;
    }

}
#endif


__kernel void arithm_bitwise_or_with_mask_C2_D0 (
        __global uchar *src1, int src1_step, int src1_offset,
        __global uchar *src2, int src2_step, int src2_offset,
        __global uchar *mask, int mask_step, int mask_offset,
        __global uchar *dst,  int dst_step,  int dst_offset,
        int rows, int cols, int dst_step1)
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
        int src2_index = mad24(y, src2_step, (x << 1) + src2_offset - (dst_align << 1));
        int mask_index = mad24(y, mask_step, x + mask_offset - dst_align);

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + (x << 1) & (int)0xfffffffc);

        uchar4 src1_data = vload4(0, src1 + src1_index);
        uchar4 src2_data = vload4(0, src2 + src2_index);
        uchar2 mask_data = vload2(0, mask + mask_index);

        uchar4 data = *((__global uchar4 *)(dst + dst_index));
        uchar4 tmp_data = src1_data | src2_data;

        data.xy = ((mask_data.x) && (dst_index + 0 >= dst_start)) ? tmp_data.xy : data.xy;
        data.zw = ((mask_data.y) && (dst_index + 2 <  dst_end  )) ? tmp_data.zw : data.zw;

        *((__global uchar4 *)(dst + dst_index)) = data;
    }
}


__kernel void arithm_bitwise_or_with_mask_C2_D1 (
        __global char *src1, int src1_step, int src1_offset,
        __global char *src2, int src2_step, int src2_offset,
        __global uchar *mask, int mask_step, int mask_offset,
        __global char *dst,  int dst_step,  int dst_offset,
        int rows, int cols, int dst_step1)
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
        int src2_index = mad24(y, src2_step, (x << 1) + src2_offset - (dst_align << 1));
        int mask_index = mad24(y, mask_step, x + mask_offset - dst_align);

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + (x << 1) & (int)0xfffffffc);

        char4 src1_data = vload4(0, src1 + src1_index);
        char4 src2_data = vload4(0, src2 + src2_index);
        uchar2 mask_data = vload2(0, mask + mask_index);

        char4 data = *((__global char4 *)(dst + dst_index));
        char4 tmp_data = src1_data | src2_data;

        data.xy = ((mask_data.x) && (dst_index + 0 >= dst_start)) ? tmp_data.xy : data.xy;
        data.zw = ((mask_data.y) && (dst_index + 2 <  dst_end  )) ? tmp_data.zw : data.zw;

        *((__global char4 *)(dst + dst_index)) = data;
    }
}

__kernel void arithm_bitwise_or_with_mask_C2_D2 (
        __global ushort *src1, int src1_step, int src1_offset,
        __global ushort *src2, int src2_step, int src2_offset,
        __global uchar  *mask, int mask_step, int mask_offset,
        __global ushort *dst,  int dst_step,  int dst_offset,
        int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 2) + src1_offset);
        int src2_index = mad24(y, src2_step, (x << 2) + src2_offset);
        int mask_index = mad24(y, mask_step,  x       + mask_offset);
        int dst_index  = mad24(y, dst_step,  (x << 2) + dst_offset);

        uchar mask_data = *(mask + mask_index);

        ushort2 src_data1 = *((__global ushort2 *)((__global char *)src1 + src1_index));
        ushort2 src_data2 = *((__global ushort2 *)((__global char *)src2 + src2_index));
        ushort2 dst_data  = *((__global ushort2 *)((__global char *)dst  + dst_index));

        ushort2 data = src_data1 | src_data2;
        data = mask_data ? data : dst_data;

        *((__global ushort2 *)((__global char *)dst + dst_index)) = data;
    }
}
__kernel void arithm_bitwise_or_with_mask_C2_D3 (
        __global short *src1, int src1_step, int src1_offset,
        __global short *src2, int src2_step, int src2_offset,
        __global uchar *mask, int mask_step, int mask_offset,
        __global short *dst,  int dst_step,  int dst_offset,
        int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 2) + src1_offset);
        int src2_index = mad24(y, src2_step, (x << 2) + src2_offset);
        int mask_index = mad24(y, mask_step,  x       + mask_offset);
        int dst_index  = mad24(y, dst_step,  (x << 2) + dst_offset);

        uchar mask_data = *(mask + mask_index);

        short2 src_data1 = *((__global short2 *)((__global char *)src1 + src1_index));
        short2 src_data2 = *((__global short2 *)((__global char *)src2 + src2_index));
        short2 dst_data  = *((__global short2 *)((__global char *)dst  + dst_index));

        short2 data = src_data1 | src_data2;
        data = mask_data ? data : dst_data;

        *((__global short2 *)((__global char *)dst + dst_index)) = data;
    }
}
__kernel void arithm_bitwise_or_with_mask_C2_D4 (
        __global int   *src1, int src1_step, int src1_offset,
        __global int   *src2, int src2_step, int src2_offset,
        __global uchar *mask, int mask_step, int mask_offset,
        __global int    *dst,  int dst_step,  int dst_offset,
        int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 3) + src1_offset);
        int src2_index = mad24(y, src2_step, (x << 3) + src2_offset);
        int mask_index = mad24(y, mask_step,  x       + mask_offset);
        int dst_index  = mad24(y, dst_step,  (x << 3) + dst_offset);

        uchar mask_data = *(mask + mask_index);

        int2 src_data1 = *((__global int2 *)((__global char *)src1 + src1_index));
        int2 src_data2 = *((__global int2 *)((__global char *)src2 + src2_index));
        int2 dst_data  = *((__global int2 *)((__global char *)dst  + dst_index));

        int2 data = src_data1 | src_data2;
        data = mask_data ? data : dst_data;

        *((__global int2 *)((__global char *)dst + dst_index)) = data;
    }
}
__kernel void arithm_bitwise_or_with_mask_C2_D5 (
        __global char *src1, int src1_step, int src1_offset,
        __global char *src2, int src2_step, int src2_offset,
        __global uchar *mask, int mask_step, int mask_offset,
        __global char *dst,  int dst_step,  int dst_offset,
        int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 3) + src1_offset);
        int src2_index = mad24(y, src2_step, (x << 3) + src2_offset);
        int mask_index = mad24(y, mask_step,  x       + mask_offset);
        int dst_index  = mad24(y, dst_step,  (x << 3) + dst_offset);

        uchar mask_data = *(mask + mask_index);

        char8 src_data1 = *((__global char8 *)((__global char *)src1 + src1_index));
        char8 src_data2 = *((__global char8 *)((__global char *)src2 + src2_index));
        char8 dst_data  = *((__global char8 *)((__global char *)dst  + dst_index));

        char8 data = src_data1 | src_data2;
        data = mask_data ? data : dst_data;

        *((__global char8 *)((__global char *)dst + dst_index)) = data;
    }
}
#if defined (DOUBLE_SUPPORT)
__kernel void arithm_bitwise_or_with_mask_C2_D6 (
        __global char *src1, int src1_step, int src1_offset,
        __global char *src2, int src2_step, int src2_offset,
        __global uchar *mask, int mask_step, int mask_offset,
        __global char *dst,  int dst_step,  int dst_offset,
        int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 4) + src1_offset);
        int src2_index = mad24(y, src2_step, (x << 4) + src2_offset);
        int mask_index = mad24(y, mask_step,  x       + mask_offset);
        int dst_index  = mad24(y, dst_step,  (x << 4) + dst_offset);

        uchar mask_data = *(mask + mask_index);

        char16 src_data1 = *((__global char16 *)((__global char *)src1 + src1_index));
        char16 src_data2 = *((__global char16 *)((__global char *)src2 + src2_index));
        char16 dst_data  = *((__global char16 *)((__global char *)dst  + dst_index));

        char16 data = src_data1 | src_data2;
        data = mask_data ? data : dst_data;

        *((__global char16 *)((__global char *)dst + dst_index)) = data;
    }
}
#endif


__kernel void arithm_bitwise_or_with_mask_C3_D0 (
        __global uchar *src1, int src1_step, int src1_offset,
        __global uchar *src2, int src2_step, int src2_offset,
        __global uchar *mask, int mask_step, int mask_offset,
        __global uchar *dst,  int dst_step,  int dst_offset,
        int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        x = x << 2;
        
#ifdef dst_align
#undef dst_align
#endif
#define dst_align (((dst_offset % dst_step) / 3 ) & 3)
        int src1_index = mad24(y, src1_step, (x * 3) + src1_offset - (dst_align * 3));
        int src2_index = mad24(y, src2_step, (x * 3) + src2_offset - (dst_align * 3));
        int mask_index = mad24(y, mask_step, x + mask_offset - dst_align);

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + (x * 3) - (dst_align * 3));

        uchar4 src1_data_0 = vload4(0, src1 + src1_index + 0);
        uchar4 src1_data_1 = vload4(0, src1 + src1_index + 4);
        uchar4 src1_data_2 = vload4(0, src1 + src1_index + 8);

        uchar4 src2_data_0 = vload4(0, src2 + src2_index + 0);
        uchar4 src2_data_1 = vload4(0, src2 + src2_index + 4);
        uchar4 src2_data_2 = vload4(0, src2 + src2_index + 8);

        uchar4 mask_data = vload4(0, mask + mask_index);

        uchar4 data_0 = *((__global uchar4 *)(dst + dst_index + 0));
        uchar4 data_1 = *((__global uchar4 *)(dst + dst_index + 4));
        uchar4 data_2 = *((__global uchar4 *)(dst + dst_index + 8));

        uchar4 tmp_data_0 =  src1_data_0 | src2_data_0;
        uchar4 tmp_data_1 =  src1_data_1 | src2_data_1;
        uchar4 tmp_data_2 =  src1_data_2 | src2_data_2;

        data_0.xyz = ((mask_data.x) && (dst_index + 0 >= dst_start)) ? tmp_data_0.xyz : data_0.xyz;
        data_0.w   = ((mask_data.y) && (dst_index + 3 >= dst_start) && (dst_index + 3 < dst_end))
                     ? tmp_data_0.w : data_0.w;

        data_1.xy  = ((mask_data.y) && (dst_index + 3 >= dst_start) && (dst_index + 3 < dst_end))
                     ? tmp_data_1.xy : data_1.xy;
        data_1.zw  = ((mask_data.z) && (dst_index + 6 >= dst_start) && (dst_index + 6 < dst_end))
                     ? tmp_data_1.zw : data_1.zw;

        data_2.x   = ((mask_data.z) && (dst_index + 6 >= dst_start) && (dst_index + 6 < dst_end))
                     ? tmp_data_2.x : data_2.x;
        data_2.yzw = ((mask_data.w) && (dst_index + 9 >= dst_start) && (dst_index + 9 < dst_end))
                     ? tmp_data_2.yzw : data_2.yzw;

        *((__global uchar4 *)(dst + dst_index + 0)) = data_0;
        *((__global uchar4 *)(dst + dst_index + 4)) = data_1;
        *((__global uchar4 *)(dst + dst_index + 8)) = data_2;
    }
}


__kernel void arithm_bitwise_or_with_mask_C3_D1 (
        __global char *src1, int src1_step, int src1_offset,
        __global char *src2, int src2_step, int src2_offset,
        __global uchar *mask, int mask_step, int mask_offset,
        __global char *dst,  int dst_step,  int dst_offset,
        int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        x = x << 2;
        
#ifdef dst_align
#undef dst_align
#endif
#define dst_align (((dst_offset % dst_step) / 3 ) & 3)
        int src1_index = mad24(y, src1_step, (x * 3) + src1_offset - (dst_align * 3));
        int src2_index = mad24(y, src2_step, (x * 3) + src2_offset - (dst_align * 3));
        int mask_index = mad24(y, mask_step, x + mask_offset - dst_align);

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + (x * 3) - (dst_align * 3));

        char4 src1_data_0 = vload4(0, src1 + src1_index + 0);
        char4 src1_data_1 = vload4(0, src1 + src1_index + 4);
        char4 src1_data_2 = vload4(0, src1 + src1_index + 8);

        char4 src2_data_0 = vload4(0, src2 + src2_index + 0);
        char4 src2_data_1 = vload4(0, src2 + src2_index + 4);
        char4 src2_data_2 = vload4(0, src2 + src2_index + 8);

        uchar4 mask_data = vload4(0, mask + mask_index);

        char4 data_0 = *((__global char4 *)(dst + dst_index + 0));
        char4 data_1 = *((__global char4 *)(dst + dst_index + 4));
        char4 data_2 = *((__global char4 *)(dst + dst_index + 8));

        char4 tmp_data_0 =  src1_data_0 | src2_data_0;
        char4 tmp_data_1 =  src1_data_1 | src2_data_1;
        char4 tmp_data_2 =  src1_data_2 | src2_data_2;

        data_0.xyz = ((mask_data.x) && (dst_index + 0 >= dst_start)) ? tmp_data_0.xyz : data_0.xyz;
        data_0.w   = ((mask_data.y) && (dst_index + 3 >= dst_start) && (dst_index + 3 < dst_end))
                     ? tmp_data_0.w : data_0.w;

        data_1.xy  = ((mask_data.y) && (dst_index + 3 >= dst_start) && (dst_index + 3 < dst_end))
                     ? tmp_data_1.xy : data_1.xy;
        data_1.zw  = ((mask_data.z) && (dst_index + 6 >= dst_start) && (dst_index + 6 < dst_end))
                     ? tmp_data_1.zw : data_1.zw;

        data_2.x   = ((mask_data.z) && (dst_index + 6 >= dst_start) && (dst_index + 6 < dst_end))
                     ? tmp_data_2.x : data_2.x;
        data_2.yzw = ((mask_data.w) && (dst_index + 9 >= dst_start) && (dst_index + 9 < dst_end))
                     ? tmp_data_2.yzw : data_2.yzw;

        *((__global char4 *)(dst + dst_index + 0)) = data_0;
        *((__global char4 *)(dst + dst_index + 4)) = data_1;
        *((__global char4 *)(dst + dst_index + 8)) = data_2;
    }
}

__kernel void arithm_bitwise_or_with_mask_C3_D2 (
        __global ushort *src1, int src1_step, int src1_offset,
        __global ushort *src2, int src2_step, int src2_offset,
        __global uchar  *mask, int mask_step, int mask_offset,
        __global ushort *dst,  int dst_step,  int dst_offset,
        int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        x = x << 1;
        
#ifdef dst_align
#undef dst_align
#endif
#define dst_align (((dst_offset % dst_step) / 6 ) & 1)
        int src1_index = mad24(y, src1_step, (x * 6) + src1_offset - (dst_align * 6));
        int src2_index = mad24(y, src2_step, (x * 6) + src2_offset - (dst_align * 6));
        int mask_index = mad24(y, mask_step, x + mask_offset - dst_align);

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + (x * 6) - (dst_align * 6));

        ushort2 src1_data_0 = vload2(0, (__global ushort *)((__global char *)src1 + src1_index + 0));
        ushort2 src1_data_1 = vload2(0, (__global ushort *)((__global char *)src1 + src1_index + 4));
        ushort2 src1_data_2 = vload2(0, (__global ushort *)((__global char *)src1 + src1_index + 8));

        ushort2 src2_data_0 = vload2(0, (__global ushort *)((__global char *)src2 + src2_index + 0));
        ushort2 src2_data_1 = vload2(0, (__global ushort *)((__global char *)src2 + src2_index + 4));
        ushort2 src2_data_2 = vload2(0, (__global ushort *)((__global char *)src2 + src2_index + 8));

        uchar2 mask_data = vload2(0, mask + mask_index);

        ushort2 data_0 = *((__global ushort2 *)((__global char *)dst + dst_index + 0));
        ushort2 data_1 = *((__global ushort2 *)((__global char *)dst + dst_index + 4));
        ushort2 data_2 = *((__global ushort2 *)((__global char *)dst + dst_index + 8));

        ushort2 tmp_data_0 =  src1_data_0 | src2_data_0 ;
        ushort2 tmp_data_1 =  src1_data_1 | src2_data_1 ;
        ushort2 tmp_data_2 =  src1_data_2 | src2_data_2 ;

        data_0.xy = ((mask_data.x) && (dst_index + 0 >= dst_start)) ? tmp_data_0.xy : data_0.xy;

        data_1.x  = ((mask_data.x) && (dst_index + 0 >= dst_start) && (dst_index + 0 < dst_end))
                    ? tmp_data_1.x : data_1.x;
        data_1.y  = ((mask_data.y) && (dst_index + 6 >= dst_start) && (dst_index + 6 < dst_end))
                    ? tmp_data_1.y : data_1.y;

        data_2.xy = ((mask_data.y) && (dst_index + 6 >= dst_start) && (dst_index + 6 < dst_end))
                    ? tmp_data_2.xy : data_2.xy;

        *((__global ushort2 *)((__global char *)dst + dst_index + 0))= data_0;
        *((__global ushort2 *)((__global char *)dst + dst_index + 4))= data_1;
        *((__global ushort2 *)((__global char *)dst + dst_index + 8))= data_2;
    }
}
__kernel void arithm_bitwise_or_with_mask_C3_D3 (
        __global short *src1, int src1_step, int src1_offset,
        __global short *src2, int src2_step, int src2_offset,
        __global uchar  *mask, int mask_step, int mask_offset,
        __global short *dst,  int dst_step,  int dst_offset,
        int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        x = x << 1;
        
#ifdef dst_align
#undef dst_align
#endif
#define dst_align (((dst_offset % dst_step) / 6 ) & 1)
        int src1_index = mad24(y, src1_step, (x * 6) + src1_offset - (dst_align * 6));
        int src2_index = mad24(y, src2_step, (x * 6) + src2_offset - (dst_align * 6));
        int mask_index = mad24(y, mask_step, x + mask_offset - dst_align);

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + (x * 6) - (dst_align * 6));

        short2 src1_data_0 = vload2(0, (__global short *)((__global char *)src1 + src1_index + 0));
        short2 src1_data_1 = vload2(0, (__global short *)((__global char *)src1 + src1_index + 4));
        short2 src1_data_2 = vload2(0, (__global short *)((__global char *)src1 + src1_index + 8));

        short2 src2_data_0 = vload2(0, (__global short *)((__global char *)src2 + src2_index + 0));
        short2 src2_data_1 = vload2(0, (__global short *)((__global char *)src2 + src2_index + 4));
        short2 src2_data_2 = vload2(0, (__global short *)((__global char *)src2 + src2_index + 8));

        uchar2 mask_data = vload2(0, mask + mask_index);

        short2 data_0 = *((__global short2 *)((__global char *)dst + dst_index + 0));
        short2 data_1 = *((__global short2 *)((__global char *)dst + dst_index + 4));
        short2 data_2 = *((__global short2 *)((__global char *)dst + dst_index + 8));

        short2 tmp_data_0 =  src1_data_0 | src2_data_0 ;
        short2 tmp_data_1 =  src1_data_1 | src2_data_1 ;
        short2 tmp_data_2 =  src1_data_2 | src2_data_2 ;

        data_0.xy = ((mask_data.x) && (dst_index + 0 >= dst_start)) ? tmp_data_0.xy : data_0.xy;

        data_1.x  = ((mask_data.x) && (dst_index + 0 >= dst_start) && (dst_index + 0 < dst_end))
                    ? tmp_data_1.x : data_1.x;
        data_1.y  = ((mask_data.y) && (dst_index + 6 >= dst_start) && (dst_index + 6 < dst_end))
                    ? tmp_data_1.y : data_1.y;

        data_2.xy = ((mask_data.y) && (dst_index + 6 >= dst_start) && (dst_index + 6 < dst_end))
                    ? tmp_data_2.xy : data_2.xy;

        *((__global short2 *)((__global char *)dst + dst_index + 0))= data_0;
        *((__global short2 *)((__global char *)dst + dst_index + 4))= data_1;
        *((__global short2 *)((__global char *)dst + dst_index + 8))= data_2;
    }
}
__kernel void arithm_bitwise_or_with_mask_C3_D4 (
        __global int   *src1, int src1_step, int src1_offset,
        __global int   *src2, int src2_step, int src2_offset,
        __global uchar *mask, int mask_step, int mask_offset,
        __global int   *dst,  int dst_step,  int dst_offset,
        int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x * 12) + src1_offset);
        int src2_index = mad24(y, src2_step, (x * 12) + src2_offset);
        int mask_index = mad24(y, mask_step, x + mask_offset);
        int dst_index  = mad24(y, dst_step, dst_offset + (x * 12));

        int src1_data_0 = *((__global int *)((__global char *)src1 + src1_index + 0));
        int src1_data_1 = *((__global int *)((__global char *)src1 + src1_index + 4));
        int src1_data_2 = *((__global int *)((__global char *)src1 + src1_index + 8));

        int src2_data_0 = *((__global int *)((__global char *)src2 + src2_index + 0));
        int src2_data_1 = *((__global int *)((__global char *)src2 + src2_index + 4));
        int src2_data_2 = *((__global int *)((__global char *)src2 + src2_index + 8));

        uchar mask_data = * (mask + mask_index);

        int data_0 = *((__global int *)((__global char *)dst + dst_index + 0));
        int data_1 = *((__global int *)((__global char *)dst + dst_index + 4));
        int data_2 = *((__global int *)((__global char *)dst + dst_index + 8));

        int tmp_data_0 =  src1_data_0 |  src2_data_0 ;
        int tmp_data_1 =  src1_data_1 |  src2_data_1 ;
        int tmp_data_2 =  src1_data_2 |  src2_data_2 ;

        data_0 = mask_data ? tmp_data_0 : data_0;
        data_1 = mask_data ? tmp_data_1 : data_1;
        data_2 = mask_data ? tmp_data_2 : data_2;

        *((__global int *)((__global char *)dst + dst_index + 0))= data_0;
        *((__global int *)((__global char *)dst + dst_index + 4))= data_1;
        *((__global int *)((__global char *)dst + dst_index + 8))= data_2;
    }
}
__kernel void arithm_bitwise_or_with_mask_C3_D5 (
        __global char *src1, int src1_step, int src1_offset,
        __global char *src2, int src2_step, int src2_offset,
        __global uchar *mask, int mask_step, int mask_offset,
        __global char *dst,  int dst_step,  int dst_offset,
        int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x * 12) + src1_offset);
        int src2_index = mad24(y, src2_step, (x * 12) + src2_offset);
        int mask_index = mad24(y, mask_step, x + mask_offset);
        int dst_index  = mad24(y, dst_step, dst_offset + (x * 12));

        char4 src1_data_0 = *((__global char4 *)((__global char *)src1 + src1_index + 0));
        char4 src1_data_1 = *((__global char4 *)((__global char *)src1 + src1_index + 4));
        char4 src1_data_2 = *((__global char4 *)((__global char *)src1 + src1_index + 8));

        char4 src2_data_0 = *((__global char4 *)((__global char *)src2 + src2_index + 0));
        char4 src2_data_1 = *((__global char4 *)((__global char *)src2 + src2_index + 4));
        char4 src2_data_2 = *((__global char4 *)((__global char *)src2 + src2_index + 8));

        uchar mask_data = * (mask + mask_index);

        char4 data_0 = *((__global char4 *)((__global char *)dst + dst_index + 0));
        char4 data_1 = *((__global char4 *)((__global char *)dst + dst_index + 4));
        char4 data_2 = *((__global char4 *)((__global char *)dst + dst_index + 8));

        char4 tmp_data_0 = src1_data_0 | src2_data_0;
        char4 tmp_data_1 = src1_data_1 | src2_data_1;
        char4 tmp_data_2 = src1_data_2 | src2_data_2;

        data_0 = mask_data ? tmp_data_0 : data_0;
        data_1 = mask_data ? tmp_data_1 : data_1;
        data_2 = mask_data ? tmp_data_2 : data_2;

        *((__global char4 *)((__global char *)dst + dst_index + 0))= data_0;
        *((__global char4 *)((__global char *)dst + dst_index + 4))= data_1;
        *((__global char4 *)((__global char *)dst + dst_index + 8))= data_2;
    }
}
#if defined (DOUBLE_SUPPORT)
__kernel void arithm_bitwise_or_with_mask_C3_D6 (
        __global char *src1, int src1_step, int src1_offset,
        __global char *src2, int src2_step, int src2_offset,
        __global uchar  *mask, int mask_step, int mask_offset,
        __global char *dst,  int dst_step,  int dst_offset,
        int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x * 24) + src1_offset);
        int src2_index = mad24(y, src2_step, (x * 24) + src2_offset);
        int mask_index = mad24(y, mask_step, x + mask_offset);
        int dst_index  = mad24(y, dst_step, dst_offset + (x * 24));

        char8 src1_data_0 = *((__global char8 *)((__global char *)src1 + src1_index + 0 ));
        char8 src1_data_1 = *((__global char8 *)((__global char *)src1 + src1_index + 8 ));
        char8 src1_data_2 = *((__global char8 *)((__global char *)src1 + src1_index + 16));

        char8 src2_data_0 = *((__global char8 *)((__global char *)src2 + src2_index + 0 ));
        char8 src2_data_1 = *((__global char8 *)((__global char *)src2 + src2_index + 8 ));
        char8 src2_data_2 = *((__global char8 *)((__global char *)src2 + src2_index + 16));

        uchar mask_data = * (mask + mask_index);

        char8 data_0 = *((__global char8 *)((__global char *)dst + dst_index + 0 ));
        char8 data_1 = *((__global char8 *)((__global char *)dst + dst_index + 8 ));
        char8 data_2 = *((__global char8 *)((__global char *)dst + dst_index + 16));

        char8 tmp_data_0 = src1_data_0 | src2_data_0;
        char8 tmp_data_1 = src1_data_1 | src2_data_1;
        char8 tmp_data_2 = src1_data_2 | src2_data_2;

        data_0 = mask_data ? tmp_data_0 : data_0;
        data_1 = mask_data ? tmp_data_1 : data_1;
        data_2 = mask_data ? tmp_data_2 : data_2;

        *((__global char8 *)((__global char *)dst + dst_index + 0 ))= data_0;
        *((__global char8 *)((__global char *)dst + dst_index + 8 ))= data_1;
        *((__global char8 *)((__global char *)dst + dst_index + 16))= data_2;
    }
}
#endif


__kernel void arithm_bitwise_or_with_mask_C4_D0 (
        __global uchar *src1, int src1_step, int src1_offset,
        __global uchar *src2, int src2_step, int src2_offset,
        __global uchar *mask, int mask_step, int mask_offset,
        __global uchar *dst,  int dst_step,  int dst_offset,
        int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 2) + src1_offset);
        int src2_index = mad24(y, src2_step, (x << 2) + src2_offset);
        int mask_index = mad24(y, mask_step,  x       + mask_offset);
        int dst_index  = mad24(y, dst_step,  (x << 2) + dst_offset);

        uchar mask_data = *(mask + mask_index);

        uchar4 src_data1 = *((__global uchar4 *)(src1 + src1_index));
        uchar4 src_data2 = *((__global uchar4 *)(src2 + src2_index));
        uchar4 dst_data  = *((__global uchar4 *)(dst  + dst_index));

        uchar4 data = src_data1 | src_data2;
        data = mask_data ? data : dst_data;

        *((__global uchar4 *)(dst + dst_index)) = data;
    }
}


__kernel void arithm_bitwise_or_with_mask_C4_D1 (
        __global char *src1, int src1_step, int src1_offset,
        __global char *src2, int src2_step, int src2_offset,
        __global uchar *mask, int mask_step, int mask_offset,
        __global char *dst,  int dst_step,  int dst_offset,
        int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 2) + src1_offset);
        int src2_index = mad24(y, src2_step, (x << 2) + src2_offset);
        int mask_index = mad24(y, mask_step,  x       + mask_offset);
        int dst_index  = mad24(y, dst_step,  (x << 2) + dst_offset);

        uchar mask_data = *(mask + mask_index);

        char4 src_data1 = *((__global char4 *)(src1 + src1_index));
        char4 src_data2 = *((__global char4 *)(src2 + src2_index));
        char4 dst_data  = *((__global char4 *)(dst  + dst_index));

        char4 data = src_data1 | src_data2;
        data = mask_data ? data : dst_data;

        *((__global char4 *)(dst + dst_index)) = data;
    }
}

__kernel void arithm_bitwise_or_with_mask_C4_D2 (
        __global ushort *src1, int src1_step, int src1_offset,
        __global ushort *src2, int src2_step, int src2_offset,
        __global uchar  *mask, int mask_step, int mask_offset,
        __global ushort *dst,  int dst_step,  int dst_offset,
        int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 3) + src1_offset);
        int src2_index = mad24(y, src2_step, (x << 3) + src2_offset);
        int mask_index = mad24(y, mask_step,  x       + mask_offset);
        int dst_index  = mad24(y, dst_step,  (x << 3) + dst_offset);

        uchar mask_data = *(mask + mask_index);

        ushort4 src_data1 = *((__global ushort4 *)((__global char *)src1 + src1_index));
        ushort4 src_data2 = *((__global ushort4 *)((__global char *)src2 + src2_index));
        ushort4 dst_data  = *((__global ushort4 *)((__global char *)dst  + dst_index));

        ushort4 data = src_data1 | src_data2;
        data = mask_data ? data : dst_data;

        *((__global ushort4 *)((__global char *)dst + dst_index)) = data;
    }
}
__kernel void arithm_bitwise_or_with_mask_C4_D3 (
        __global short *src1, int src1_step, int src1_offset,
        __global short *src2, int src2_step, int src2_offset,
        __global uchar *mask, int mask_step, int mask_offset,
        __global short *dst,  int dst_step,  int dst_offset,
        int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 3) + src1_offset);
        int src2_index = mad24(y, src2_step, (x << 3) + src2_offset);
        int mask_index = mad24(y, mask_step,  x       + mask_offset);
        int dst_index  = mad24(y, dst_step,  (x << 3) + dst_offset);

        uchar mask_data = *(mask + mask_index);

        short4 src_data1 = *((__global short4 *)((__global char *)src1 + src1_index));
        short4 src_data2 = *((__global short4 *)((__global char *)src2 + src2_index));
        short4 dst_data  = *((__global short4 *)((__global char *)dst  + dst_index));

        short4 data = src_data1 | src_data2;
        data = mask_data ? data : dst_data;

        *((__global short4 *)((__global char *)dst + dst_index)) = data;
    }
}
__kernel void arithm_bitwise_or_with_mask_C4_D4 (
        __global int   *src1, int src1_step, int src1_offset,
        __global int   *src2, int src2_step, int src2_offset,
        __global uchar *mask, int mask_step, int mask_offset,
        __global int   *dst,  int dst_step,  int dst_offset,
        int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 4) + src1_offset);
        int src2_index = mad24(y, src2_step, (x << 4) + src2_offset);
        int mask_index = mad24(y, mask_step,  x       + mask_offset);
        int dst_index  = mad24(y, dst_step,  (x << 4) + dst_offset);

        uchar mask_data = *(mask + mask_index);

        int4 src_data1 = *((__global int4 *)((__global char *)src1 + src1_index));
        int4 src_data2 = *((__global int4 *)((__global char *)src2 + src2_index));
        int4 dst_data  = *((__global int4 *)((__global char *)dst  + dst_index));

        int4 data = src_data1 | src_data2;
        data = mask_data ? data : dst_data;

        *((__global int4 *)((__global char *)dst + dst_index)) = data;
    }
}
__kernel void arithm_bitwise_or_with_mask_C4_D5 (
        __global char *src1, int src1_step, int src1_offset,
        __global char *src2, int src2_step, int src2_offset,
        __global uchar *mask, int mask_step, int mask_offset,
        __global char *dst,  int dst_step,  int dst_offset,
        int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 4) + src1_offset);
        int src2_index = mad24(y, src2_step, (x << 4) + src2_offset);
        int mask_index = mad24(y, mask_step,  x       + mask_offset);
        int dst_index  = mad24(y, dst_step,  (x << 4) + dst_offset);

        uchar mask_data = *(mask + mask_index);

        char16 src_data1 = *((__global char16 *)((__global char *)src1 + src1_index));
        char16 src_data2 = *((__global char16 *)((__global char *)src2 + src2_index));
        char16 dst_data  = *((__global char16 *)((__global char *)dst  + dst_index));

        char16 data = src_data1 | src_data2;
        data = mask_data ? data : dst_data;

        *((__global char16 *)((__global char *)dst + dst_index)) = data;
    }
}
#if defined (DOUBLE_SUPPORT)
__kernel void arithm_bitwise_or_with_mask_C4_D6 (
        __global char *src1, int src1_step, int src1_offset,
        __global char *src2, int src2_step, int src2_offset,
        __global uchar  *mask, int mask_step, int mask_offset,
        __global char *dst,  int dst_step,  int dst_offset,
        int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 5) + src1_offset);
        int src2_index = mad24(y, src2_step, (x << 5) + src2_offset);
        int mask_index = mad24(y, mask_step,  x       + mask_offset);
        int dst_index  = mad24(y, dst_step,  (x << 5) + dst_offset);

        uchar mask_data = *(mask + mask_index);

        char8 src_data1_0 = *((__global char8 *)((__global char *)src1 + src1_index + 0));
        char8 src_data1_1 = *((__global char8 *)((__global char *)src1 + src1_index + 8));
        char8 src_data1_2 = *((__global char8 *)((__global char *)src1 + src1_index + 16));
        char8 src_data1_3 = *((__global char8 *)((__global char *)src1 + src1_index + 24));

        char8 src_data2_0 = *((__global char8 *)((__global char *)src2 + src2_index + 0));
        char8 src_data2_1 = *((__global char8 *)((__global char *)src2 + src2_index + 8));
        char8 src_data2_2 = *((__global char8 *)((__global char *)src2 + src2_index + 16));
        char8 src_data2_3 = *((__global char8 *)((__global char *)src2 + src2_index + 24));

        char8 dst_data_0  = *((__global char8 *)((__global char *)dst  + dst_index + 0));
        char8 dst_data_1  = *((__global char8 *)((__global char *)dst  + dst_index + 8));
        char8 dst_data_2  = *((__global char8 *)((__global char *)dst  + dst_index + 16));
        char8 dst_data_3  = *((__global char8 *)((__global char *)dst  + dst_index + 24));

        char8 data_0 = src_data1_0 | src_data2_0;
        char8 data_1 = src_data1_1 | src_data2_1;
        char8 data_2 = src_data1_2 | src_data2_2;
        char8 data_3 = src_data1_3 | src_data2_3;

        data_0 = mask_data ? data_0 : dst_data_0;
        data_1 = mask_data ? data_1 : dst_data_1;
        data_2 = mask_data ? data_2 : dst_data_2;
        data_3 = mask_data ? data_3 : dst_data_3;

        *((__global char8 *)((__global char *)dst + dst_index + 0)) = data_0;
        *((__global char8 *)((__global char *)dst + dst_index + 8)) = data_1;
        *((__global char8 *)((__global char *)dst + dst_index + 16)) = data_2;
        *((__global char8 *)((__global char *)dst + dst_index + 24)) = data_3;
    }
}
#endif
