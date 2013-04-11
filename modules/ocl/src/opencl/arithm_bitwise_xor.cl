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
//     and/or other GpuMaterials provided with the distribution.
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
////////////////////////////////////////////BITWISE_XOR////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
/**************************************bitwise_xor without mask**************************************/
__kernel void arithm_bitwise_xor_D0 (__global uchar *src1, int src1_step, int src1_offset,
                                     __global uchar *src2, int src2_step, int src2_offset,
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

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + x & (int)0xfffffffc);

        int src1_index_fix = src1_index < 0 ? 0 : src1_index;
        int src2_index_fix = src2_index < 0 ? 0 : src2_index;
        uchar4 src1_data = vload4(0, src1 + src1_index_fix);
        uchar4 src2_data = vload4(0, src2 + src2_index_fix);

        if(src1_index < 0)
        {
            uchar4 tmp;
            tmp.xyzw = (src1_index == -2) ? src1_data.zwxy:src1_data.yzwx;
            src1_data.xyzw = (src1_index == -1) ? src1_data.wxyz:tmp.xyzw;
        }
        if(src2_index < 0)
        {
            uchar4 tmp;
            tmp.xyzw = (src2_index == -2) ? src2_data.zwxy:src2_data.yzwx;
            src2_data.xyzw = (src2_index == -1) ? src2_data.wxyz:tmp.xyzw;
        }
        uchar4 dst_data = *((__global uchar4 *)(dst + dst_index));
        uchar4 tmp_data = src1_data ^ src2_data;

        dst_data.x = ((dst_index + 0 >= dst_start) && (dst_index + 0 < dst_end)) ? tmp_data.x : dst_data.x;
        dst_data.y = ((dst_index + 1 >= dst_start) && (dst_index + 1 < dst_end)) ? tmp_data.y : dst_data.y;
        dst_data.z = ((dst_index + 2 >= dst_start) && (dst_index + 2 < dst_end)) ? tmp_data.z : dst_data.z;
        dst_data.w = ((dst_index + 3 >= dst_start) && (dst_index + 3 < dst_end)) ? tmp_data.w : dst_data.w;

        *((__global uchar4 *)(dst + dst_index)) = dst_data;
    }
}


__kernel void arithm_bitwise_xor_D1 (__global char *src1, int src1_step, int src1_offset,
                                     __global char *src2, int src2_step, int src2_offset,
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

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + x & (int)0xfffffffc);

        int src1_index_fix = src1_index < 0 ? 0 : src1_index;
        int src2_index_fix = src2_index < 0 ? 0 : src2_index;
        char4 src1_data = vload4(0, src1 + src1_index_fix);
        char4 src2_data = vload4(0, src2 + src2_index_fix);

        if(src1_index < 0)
        {
            char4 tmp;
            tmp.xyzw = (src1_index == -2) ? src1_data.zwxy:src1_data.yzwx;
            src1_data.xyzw = (src1_index == -1) ? src1_data.wxyz:tmp.xyzw;
        }
        if(src2_index < 0)
        {
            char4 tmp;
            tmp.xyzw = (src2_index == -2) ? src2_data.zwxy:src2_data.yzwx;
            src2_data.xyzw = (src2_index == -1) ? src2_data.wxyz:tmp.xyzw;
        }
        char4 dst_data = *((__global char4 *)(dst + dst_index));
        char4 tmp_data = src1_data ^ src2_data;

        dst_data.x = ((dst_index + 0 >= dst_start) && (dst_index + 0 < dst_end)) ? tmp_data.x : dst_data.x;
        dst_data.y = ((dst_index + 1 >= dst_start) && (dst_index + 1 < dst_end)) ? tmp_data.y : dst_data.y;
        dst_data.z = ((dst_index + 2 >= dst_start) && (dst_index + 2 < dst_end)) ? tmp_data.z : dst_data.z;
        dst_data.w = ((dst_index + 3 >= dst_start) && (dst_index + 3 < dst_end)) ? tmp_data.w : dst_data.w;

        *((__global char4 *)(dst + dst_index)) = dst_data;
    }
}


__kernel void arithm_bitwise_xor_D2 (__global ushort *src1, int src1_step, int src1_offset,
                                     __global ushort *src2, int src2_step, int src2_offset,
                                     __global ushort *dst,  int dst_step,  int dst_offset,
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
#define dst_align ((dst_offset >> 1) & 3)
        int src1_index = mad24(y, src1_step, (x << 1) + src1_offset - (dst_align << 1));
        int src2_index = mad24(y, src2_step, (x << 1) + src2_offset - (dst_align << 1));

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + (x << 1) & (int)0xfffffff8);

        int src1_index_fix = src1_index < 0 ? 0 : src1_index;
        int src2_index_fix = src2_index < 0 ? 0 : src2_index;
        ushort4 src1_data = vload4(0, (__global ushort *)((__global char *)src1 + src1_index_fix));
        ushort4 src2_data = vload4(0, (__global ushort *)((__global char *)src2 + src2_index_fix));

        if(src1_index < 0)
        {
            ushort4 tmp;
            tmp.xyzw = (src1_index == -2) ? src1_data.zwxy:src1_data.yzwx;
            src1_data.xyzw = (src1_index == -1) ? src1_data.wxyz:tmp.xyzw;
        }
        if(src2_index < 0)
        {
            ushort4 tmp;
            tmp.xyzw = (src2_index == -2) ? src2_data.zwxy:src2_data.yzwx;
            src2_data.xyzw = (src2_index == -1) ? src2_data.wxyz:tmp.xyzw;
        }
        ushort4 dst_data = *((__global ushort4 *)((__global char *)dst + dst_index));
        ushort4 tmp_data = src1_data ^ src2_data;

        dst_data.x = ((dst_index + 0 >= dst_start) && (dst_index + 0 < dst_end)) ? tmp_data.x : dst_data.x;
        dst_data.y = ((dst_index + 2 >= dst_start) && (dst_index + 2 < dst_end)) ? tmp_data.y : dst_data.y;
        dst_data.z = ((dst_index + 4 >= dst_start) && (dst_index + 4 < dst_end)) ? tmp_data.z : dst_data.z;
        dst_data.w = ((dst_index + 6 >= dst_start) && (dst_index + 6 < dst_end)) ? tmp_data.w : dst_data.w;

        *((__global ushort4 *)((__global char *)dst + dst_index)) = dst_data;
    }
}



__kernel void arithm_bitwise_xor_D3 (__global short *src1, int src1_step, int src1_offset,
                                     __global short *src2, int src2_step, int src2_offset,
                                     __global short *dst,  int dst_step,  int dst_offset,
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
#define dst_align ((dst_offset >> 1) & 3)
        int src1_index = mad24(y, src1_step, (x << 1) + src1_offset - (dst_align << 1));
        int src2_index = mad24(y, src2_step, (x << 1) + src2_offset - (dst_align << 1));

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + (x << 1) & (int)0xfffffff8);

        int src1_index_fix = src1_index < 0 ? 0 : src1_index;
        int src2_index_fix = src2_index < 0 ? 0 : src2_index;
        short4 src1_data = vload4(0, (__global short *)((__global char *)src1 + src1_index_fix));
        short4 src2_data = vload4(0, (__global short *)((__global char *)src2 + src2_index_fix));

        short4 dst_data = *((__global short4 *)((__global char *)dst + dst_index));

        if(src1_index < 0)
        {
            short4 tmp;
            tmp.xyzw = (src1_index == -2) ? src1_data.zwxy:src1_data.yzwx;
            src1_data.xyzw = (src1_index == -1) ? src1_data.wxyz:tmp.xyzw;
        }
        if(src2_index < 0)
        {
            short4 tmp;
            tmp.xyzw = (src2_index == -2) ? src2_data.zwxy:src2_data.yzwx;
            src2_data.xyzw = (src2_index == -1) ? src2_data.wxyz:tmp.xyzw;
        }



        short4 tmp_data = src1_data ^ src2_data;

        dst_data.x = ((dst_index + 0 >= dst_start) && (dst_index + 0 < dst_end)) ? tmp_data.x : dst_data.x;
        dst_data.y = ((dst_index + 2 >= dst_start) && (dst_index + 2 < dst_end)) ? tmp_data.y : dst_data.y;
        dst_data.z = ((dst_index + 4 >= dst_start) && (dst_index + 4 < dst_end)) ? tmp_data.z : dst_data.z;
        dst_data.w = ((dst_index + 6 >= dst_start) && (dst_index + 6 < dst_end)) ? tmp_data.w : dst_data.w;

        *((__global short4 *)((__global char *)dst + dst_index)) = dst_data;
    }
}



__kernel void arithm_bitwise_xor_D4 (__global int *src1, int src1_step, int src1_offset,
                                     __global int *src2, int src2_step, int src2_offset,
                                     __global int *dst,  int dst_step,  int dst_offset,
                                     int rows, int cols, int dst_step1)
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
        int tmp  = data1 ^ data2;

        *((__global int *)((__global char *)dst + dst_index)) = tmp;
    }
}

__kernel void arithm_bitwise_xor_D5 (__global char *src1, int src1_step, int src1_offset,
                                     __global char *src2, int src2_step, int src2_offset,
                                     __global char *dst,  int dst_step,  int dst_offset,
                                     int rows, int cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 2) + src1_offset);
        int src2_index = mad24(y, src2_step, (x << 2) + src2_offset);
        int dst_index  = mad24(y, dst_step,  (x << 2) + dst_offset);

        char4 data1 = *((__global char4 *)((__global char *)src1 + src1_index));
        char4 data2 = *((__global char4 *)((__global char *)src2 + src2_index));
        char4 tmp = data1 ^ data2;

        *((__global char4 *)((__global char *)dst + dst_index)) = tmp;
    }
}
#if defined (DOUBLE_SUPPORT)
__kernel void arithm_bitwise_xor_D6 (__global char *src1, int src1_step, int src1_offset,
                                     __global char *src2, int src2_step, int src2_offset,
                                     __global char *dst,  int dst_step,  int dst_offset,
                                     int rows, int cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 3) + src1_offset);
        int src2_index = mad24(y, src2_step, (x << 3) + src2_offset);
        int dst_index  = mad24(y, dst_step,  (x << 3) + dst_offset);

        char8 data1 = *((__global char8 *)((__global char *)src1 + src1_index));
        char8 data2 = *((__global char8 *)((__global char *)src2 + src2_index));

        *((__global char8 *)((__global char *)dst + dst_index)) = data1 ^ data2;
    }
}
#endif
