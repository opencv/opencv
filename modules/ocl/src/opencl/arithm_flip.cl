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
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#elif defined (cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#endif
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////flip rows///////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void arithm_flip_rows_D0 (__global uchar *src, int src_step, int src_offset,
                                   __global uchar *dst, int dst_step, int dst_offset,
                                   int rows, int cols, int thread_rows, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < thread_rows)
    {
        x = x << 2;

#ifdef dst_align
#undef dst_align
#endif
#define dst_align (dst_offset & 3)
        int src_index_0 = mad24(y,            src_step, x + src_offset - dst_align);
        int src_index_1 = mad24(rows - y - 1, src_step, x + src_offset - dst_align);

        int dst_start_0  = mad24(y,            dst_step, dst_offset);
        int dst_start_1  = mad24(rows - y - 1, dst_step, dst_offset);
        int dst_end_0    = mad24(y,            dst_step, dst_offset + dst_step1);
        int dst_end_1    = mad24(rows - y - 1, dst_step, dst_offset + dst_step1);
        int dst_index_0  = mad24(y,            dst_step, dst_offset + x & (int)0xfffffffc);
        int dst_index_1  = mad24(rows - y - 1, dst_step, dst_offset + x & (int)0xfffffffc);
        int src1_index_fix = src_index_0 < 0 ? 0 : src_index_0;
        int src2_index_fix = src_index_1 < 0 ? 0 : src_index_1;
        uchar4 src_data_0 = vload4(0, src + src1_index_fix);
        uchar4 src_data_1 = vload4(0, src + src2_index_fix);
        if(src_index_0 < 0)
        {
            uchar4 tmp;
            tmp.xyzw = (src_index_0 == -2) ? src_data_0.zwxy:src_data_0.yzwx;
            src_data_0.xyzw = (src_index_0 == -1) ? src_data_0.wxyz:tmp.xyzw;
        }
        if(src_index_1 < 0)
        {
            uchar4 tmp;
            tmp.xyzw = (src_index_1 == -2) ? src_data_1.zwxy:src_data_1.yzwx;
            src_data_1.xyzw = (src_index_1 == -1) ? src_data_1.wxyz:tmp.xyzw;
        }

        uchar4 dst_data_0 = *((__global uchar4 *)(dst + dst_index_0));
        uchar4 dst_data_1 = *((__global uchar4 *)(dst + dst_index_1));

        dst_data_0.x =  (dst_index_0 + 0 >= dst_start_0)                                   ? src_data_1.x : dst_data_0.x;
        dst_data_0.y = ((dst_index_0 + 1 >= dst_start_0) && (dst_index_0 + 1 < dst_end_0)) ? src_data_1.y : dst_data_0.y;
        dst_data_0.z = ((dst_index_0 + 2 >= dst_start_0) && (dst_index_0 + 2 < dst_end_0)) ? src_data_1.z : dst_data_0.z;
        dst_data_0.w =  (dst_index_0 + 3 < dst_end_0)                                      ? src_data_1.w : dst_data_0.w;

        dst_data_1.x =  (dst_index_1 + 0 >= dst_start_1)                                   ? src_data_0.x : dst_data_1.x;
        dst_data_1.y = ((dst_index_1 + 1 >= dst_start_1) && (dst_index_1 + 1 < dst_end_1)) ? src_data_0.y : dst_data_1.y;
        dst_data_1.z = ((dst_index_1 + 2 >= dst_start_1) && (dst_index_1 + 2 < dst_end_1)) ? src_data_0.z : dst_data_1.z;
        dst_data_1.w =  (dst_index_1 + 3 < dst_end_1)                                      ? src_data_0.w : dst_data_1.w;

        *((__global uchar4 *)(dst + dst_index_0)) = dst_data_0;
        *((__global uchar4 *)(dst + dst_index_1)) = dst_data_1;
    }
}
__kernel void arithm_flip_rows_D1 (__global char *src, int src_step, int src_offset,
                                   __global char *dst, int dst_step, int dst_offset,
                                   int rows, int cols, int thread_rows, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < thread_rows)
    {
        x = x << 2;

#ifdef dst_align
#undef dst_align
#endif
#define dst_align (dst_offset & 3)
        int src_index_0 = mad24(y,            src_step, x + src_offset - dst_align);
        int src_index_1 = mad24(rows - y - 1, src_step, x + src_offset - dst_align);

        int dst_start_0  = mad24(y,            dst_step, dst_offset);
        int dst_start_1  = mad24(rows - y - 1, dst_step, dst_offset);
        int dst_end_0    = mad24(y,            dst_step, dst_offset + dst_step1);
        int dst_end_1    = mad24(rows - y - 1, dst_step, dst_offset + dst_step1);
        int dst_index_0  = mad24(y,            dst_step, dst_offset + x & (int)0xfffffffc);
        int dst_index_1  = mad24(rows - y - 1, dst_step, dst_offset + x & (int)0xfffffffc);

        char4 src_data_0 = vload4(0, src + src_index_0);
        char4 src_data_1 = vload4(0, src + src_index_1);

        char4 dst_data_0 = *((__global char4 *)(dst + dst_index_0));
        char4 dst_data_1 = *((__global char4 *)(dst + dst_index_1));

        dst_data_0.x =  (dst_index_0 + 0 >= dst_start_0)                                   ? src_data_1.x : dst_data_0.x;
        dst_data_0.y = ((dst_index_0 + 1 >= dst_start_0) && (dst_index_0 + 1 < dst_end_0)) ? src_data_1.y : dst_data_0.y;
        dst_data_0.z = ((dst_index_0 + 2 >= dst_start_0) && (dst_index_0 + 2 < dst_end_0)) ? src_data_1.z : dst_data_0.z;
        dst_data_0.w =  (dst_index_0 + 3 < dst_end_0)                                      ? src_data_1.w : dst_data_0.w;

        dst_data_1.x =  (dst_index_1 + 0 >= dst_start_1)                                   ? src_data_0.x : dst_data_1.x;
        dst_data_1.y = ((dst_index_1 + 1 >= dst_start_1) && (dst_index_1 + 1 < dst_end_1)) ? src_data_0.y : dst_data_1.y;
        dst_data_1.z = ((dst_index_1 + 2 >= dst_start_1) && (dst_index_1 + 2 < dst_end_1)) ? src_data_0.z : dst_data_1.z;
        dst_data_1.w =  (dst_index_1 + 3 < dst_end_1)                                      ? src_data_0.w : dst_data_1.w;

        *((__global char4 *)(dst + dst_index_0)) = dst_data_0;
        *((__global char4 *)(dst + dst_index_1)) = dst_data_1;
    }
}
__kernel void arithm_flip_rows_D2 (__global ushort *src, int src_step, int src_offset,
                                   __global ushort *dst, int dst_step, int dst_offset,
                                   int rows, int cols, int thread_rows, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < thread_rows)
    {
        x = x << 2;

#ifdef dst_align
#undef dst_align
#endif
#define dst_align (((dst_offset >> 1) & 3) << 1)
        int src_index_0 = mad24(y,            src_step, (x << 1) + src_offset - dst_align);
        int src_index_1 = mad24(rows - y - 1, src_step, (x << 1) + src_offset - dst_align);

        int dst_start_0  = mad24(y,            dst_step, dst_offset);
        int dst_start_1  = mad24(rows - y - 1, dst_step, dst_offset);
        int dst_end_0    = mad24(y,            dst_step, dst_offset + dst_step1);
        int dst_end_1    = mad24(rows - y - 1, dst_step, dst_offset + dst_step1);
        int dst_index_0  = mad24(y,            dst_step, dst_offset + (x << 1) & (int)0xfffffff8);
        int dst_index_1  = mad24(rows - y - 1, dst_step, dst_offset + (x << 1) & (int)0xfffffff8);

        ushort4 src_data_0 = vload4(0, (__global ushort *)((__global char *)src + src_index_0));
        ushort4 src_data_1 = vload4(0, (__global ushort *)((__global char *)src + src_index_1));

        ushort4 dst_data_0 = *((__global ushort4 *)((__global char *)dst + dst_index_0));
        ushort4 dst_data_1 = *((__global ushort4 *)((__global char *)dst + dst_index_1));

        dst_data_0.x =  (dst_index_0 + 0 >= dst_start_0)                                   ? src_data_1.x : dst_data_0.x;
        dst_data_0.y = ((dst_index_0 + 2 >= dst_start_0) && (dst_index_0 + 2 < dst_end_0)) ? src_data_1.y : dst_data_0.y;
        dst_data_0.z = ((dst_index_0 + 4 >= dst_start_0) && (dst_index_0 + 4 < dst_end_0)) ? src_data_1.z : dst_data_0.z;
        dst_data_0.w =  (dst_index_0 + 6 < dst_end_0)                                      ? src_data_1.w : dst_data_0.w;

        dst_data_1.x =  (dst_index_1 + 0 >= dst_start_1)                                   ? src_data_0.x : dst_data_1.x;
        dst_data_1.y = ((dst_index_1 + 2 >= dst_start_1) && (dst_index_1 + 2 < dst_end_1)) ? src_data_0.y : dst_data_1.y;
        dst_data_1.z = ((dst_index_1 + 4 >= dst_start_1) && (dst_index_1 + 4 < dst_end_1)) ? src_data_0.z : dst_data_1.z;
        dst_data_1.w =  (dst_index_1 + 6 < dst_end_1)                                      ? src_data_0.w : dst_data_1.w;

        *((__global ushort4 *)((__global char *)dst + dst_index_0)) = dst_data_0;
        *((__global ushort4 *)((__global char *)dst + dst_index_1)) = dst_data_1;
    }
}
__kernel void arithm_flip_rows_D3 (__global short *src, int src_step, int src_offset,
                                   __global short *dst, int dst_step, int dst_offset,
                                   int rows, int cols, int thread_rows, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < thread_rows)
    {
        x = x << 2;

#ifdef dst_align
#undef dst_align
#endif
#define dst_align (((dst_offset >> 1) & 3) << 1)
        int src_index_0 = mad24(y,            src_step, (x << 1) + src_offset - dst_align);
        int src_index_1 = mad24(rows - y - 1, src_step, (x << 1) + src_offset - dst_align);

        int dst_start_0  = mad24(y,            dst_step, dst_offset);
        int dst_start_1  = mad24(rows - y - 1, dst_step, dst_offset);
        int dst_end_0    = mad24(y,            dst_step, dst_offset + dst_step1);
        int dst_end_1    = mad24(rows - y - 1, dst_step, dst_offset + dst_step1);
        int dst_index_0  = mad24(y,            dst_step, dst_offset + (x << 1) & (int)0xfffffff8);
        int dst_index_1  = mad24(rows - y - 1, dst_step, dst_offset + (x << 1) & (int)0xfffffff8);

        short4 src_data_0 = vload4(0, (__global short *)((__global char *)src + src_index_0));
        short4 src_data_1 = vload4(0, (__global short *)((__global char *)src + src_index_1));

        short4 dst_data_0 = *((__global short4 *)((__global char *)dst + dst_index_0));
        short4 dst_data_1 = *((__global short4 *)((__global char *)dst + dst_index_1));

        dst_data_0.x =  (dst_index_0 + 0 >= dst_start_0)                                   ? src_data_1.x : dst_data_0.x;
        dst_data_0.y = ((dst_index_0 + 2 >= dst_start_0) && (dst_index_0 + 2 < dst_end_0)) ? src_data_1.y : dst_data_0.y;
        dst_data_0.z = ((dst_index_0 + 4 >= dst_start_0) && (dst_index_0 + 4 < dst_end_0)) ? src_data_1.z : dst_data_0.z;
        dst_data_0.w =  (dst_index_0 + 6 < dst_end_0)                                      ? src_data_1.w : dst_data_0.w;

        dst_data_1.x =  (dst_index_1 + 0 >= dst_start_1)                                   ? src_data_0.x : dst_data_1.x;
        dst_data_1.y = ((dst_index_1 + 2 >= dst_start_1) && (dst_index_1 + 2 < dst_end_1)) ? src_data_0.y : dst_data_1.y;
        dst_data_1.z = ((dst_index_1 + 4 >= dst_start_1) && (dst_index_1 + 4 < dst_end_1)) ? src_data_0.z : dst_data_1.z;
        dst_data_1.w =  (dst_index_1 + 6 < dst_end_1)                                      ? src_data_0.w : dst_data_1.w;

        *((__global short4 *)((__global char *)dst + dst_index_0)) = dst_data_0;
        *((__global short4 *)((__global char *)dst + dst_index_1)) = dst_data_1;
    }
}

__kernel void arithm_flip_rows_D4 (__global int *src, int src_step, int src_offset,
                                   __global int *dst, int dst_step, int dst_offset,
                                   int rows, int cols, int thread_rows, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < thread_rows)
    {
        int src_index_0 = mad24(y,            src_step, (x << 2) + src_offset);
        int src_index_1 = mad24(rows - y - 1, src_step, (x << 2) + src_offset);

        int dst_index_0 = mad24(y,            dst_step, (x << 2) + dst_offset);
        int dst_index_1 = mad24(rows - y - 1, dst_step, (x << 2) + dst_offset);

        int data0 = *((__global int *)((__global char *)src + src_index_0));
        int data1 = *((__global int *)((__global char *)src + src_index_1));

        *((__global int *)((__global char *)dst + dst_index_0)) = data1;
        *((__global int *)((__global char *)dst + dst_index_1)) = data0;
    }
}
__kernel void arithm_flip_rows_D5 (__global float *src, int src_step, int src_offset,
                                   __global float *dst, int dst_step, int dst_offset,
                                   int rows, int cols, int thread_rows, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < thread_rows)
    {
        int src_index_0 = mad24(y,            src_step, (x << 2) + src_offset);
        int src_index_1 = mad24(rows - y - 1, src_step, (x << 2) + src_offset);

        int dst_index_0 = mad24(y,            dst_step, (x << 2) + dst_offset);
        int dst_index_1 = mad24(rows - y - 1, dst_step, (x << 2) + dst_offset);

        float data0 = *((__global float *)((__global char *)src + src_index_0));
        float data1 = *((__global float *)((__global char *)src + src_index_1));

        *((__global float *)((__global char *)dst + dst_index_0)) = data1;
        *((__global float *)((__global char *)dst + dst_index_1)) = data0;
    }
}

#if defined (DOUBLE_SUPPORT)
__kernel void arithm_flip_rows_D6 (__global double *src, int src_step, int src_offset,
                                   __global double *dst, int dst_step, int dst_offset,
                                   int rows, int cols, int thread_rows, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < thread_rows)
    {
        int src_index_0 = mad24(y,            src_step, (x << 3) + src_offset);
        int src_index_1 = mad24(rows - y - 1, src_step, (x << 3) + src_offset);

        int dst_index_0 = mad24(y,            dst_step, (x << 3) + dst_offset);
        int dst_index_1 = mad24(rows - y - 1, dst_step, (x << 3) + dst_offset);

        double data0 = *((__global double *)((__global char *)src + src_index_0));
        double data1 = *((__global double *)((__global char *)src + src_index_1));

        *((__global double *)((__global char *)dst + dst_index_0)) = data1;
        *((__global double *)((__global char *)dst + dst_index_1)) = data0;
    }
}
#endif
//////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////flip cols///////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void arithm_flip_cols_C1_D0 (__global uchar *src, int src_step, int src_offset,
                                      __global uchar *dst, int dst_step, int dst_offset,
                                      int rows, int cols, int thread_cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < thread_cols && y < rows)
    {
        int src_index_0 = mad24(y, src_step, (x)           + src_offset);
        int dst_index_1 = mad24(y, dst_step, (cols - x -1) + dst_offset);
        uchar data0 = *(src + src_index_0);
        *(dst + dst_index_1) = data0;

        int src_index_1 = mad24(y, src_step, (cols - x -1) + src_offset);
        int dst_index_0 = mad24(y, dst_step, (x)           + dst_offset);
        uchar data1 = *(src + src_index_1);
        *(dst + dst_index_0) = data1;
    }
}
__kernel void arithm_flip_cols_C1_D1 (__global char *src, int src_step, int src_offset,
                                      __global char *dst, int dst_step, int dst_offset,
                                      int rows, int cols, int thread_cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < thread_cols && y < rows)
    {
        int src_index_0 = mad24(y, src_step, (x)           + src_offset);
        int src_index_1 = mad24(y, src_step, (cols - x -1) + src_offset);

        int dst_index_0 = mad24(y, dst_step, (x)           + dst_offset);
        int dst_index_1 = mad24(y, dst_step, (cols - x -1) + dst_offset);

        char data0 = *(src + src_index_0);
        char data1 = *(src + src_index_1);

        *(dst + dst_index_0) = data1;
        *(dst + dst_index_1) = data0;
    }
}
__kernel void arithm_flip_cols_C1_D2 (__global ushort *src, int src_step, int src_offset,
                                      __global ushort *dst, int dst_step, int dst_offset,
                                      int rows, int cols, int thread_cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < thread_cols && y < rows)
    {
        int src_index_0 = mad24(y, src_step, (x << 1)             + src_offset);
        int src_index_1 = mad24(y, src_step, ((cols - x -1) << 1) + src_offset);

        int dst_index_0 = mad24(y, dst_step, (x << 1)             + dst_offset);
        int dst_index_1 = mad24(y, dst_step, ((cols - x -1) << 1) + dst_offset);

        ushort data0 = *((__global ushort *)((__global char *)src + src_index_0));
        ushort data1 = *((__global ushort *)((__global char *)src + src_index_1));

        *((__global ushort *)((__global char *)dst + dst_index_0)) = data1;
        *((__global ushort *)((__global char *)dst + dst_index_1)) = data0;
    }
}
__kernel void arithm_flip_cols_C1_D3 (__global short *src, int src_step, int src_offset,
                                      __global short *dst, int dst_step, int dst_offset,
                                      int rows, int cols, int thread_cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < thread_cols && y < rows)
    {
        int src_index_0 = mad24(y, src_step, (x << 1)             + src_offset);
        int src_index_1 = mad24(y, src_step, ((cols - x -1) << 1) + src_offset);

        int dst_index_0 = mad24(y, dst_step, (x << 1)             + dst_offset);
        int dst_index_1 = mad24(y, dst_step, ((cols - x -1) << 1) + dst_offset);

        short data0 = *((__global short *)((__global char *)src + src_index_0));
        short data1 = *((__global short *)((__global char *)src + src_index_1));

        *((__global short *)((__global char *)dst + dst_index_0)) = data1;
        *((__global short *)((__global char *)dst + dst_index_1)) = data0;
    }
}
__kernel void arithm_flip_cols_C1_D4 (__global int *src, int src_step, int src_offset,
                                      __global int *dst, int dst_step, int dst_offset,
                                      int rows, int cols, int thread_cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < thread_cols && y < rows)
    {
        int src_index_0 = mad24(y, src_step, (x << 2)             + src_offset);
        int src_index_1 = mad24(y, src_step, ((cols - x -1) << 2) + src_offset);

        int dst_index_0 = mad24(y, dst_step, (x << 2)             + dst_offset);
        int dst_index_1 = mad24(y, dst_step, ((cols - x -1) << 2) + dst_offset);

        int data0 = *((__global int *)((__global char *)src + src_index_0));
        int data1 = *((__global int *)((__global char *)src + src_index_1));

        *((__global int *)((__global char *)dst + dst_index_0)) = data1;
        *((__global int *)((__global char *)dst + dst_index_1)) = data0;
    }
}
__kernel void arithm_flip_cols_C1_D5 (__global float *src, int src_step, int src_offset,
                                      __global float *dst, int dst_step, int dst_offset,
                                      int rows, int cols, int thread_cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < thread_cols && y < rows)
    {
        int src_index_0 = mad24(y, src_step, (x << 2)             + src_offset);
        int src_index_1 = mad24(y, src_step, ((cols - x -1) << 2) + src_offset);

        int dst_index_0 = mad24(y, dst_step, (x << 2)             + dst_offset);
        int dst_index_1 = mad24(y, dst_step, ((cols - x -1) << 2) + dst_offset);

        float data0 = *((__global float *)((__global char *)src + src_index_0));
        float data1 = *((__global float *)((__global char *)src + src_index_1));

        *((__global float *)((__global char *)dst + dst_index_0)) = data1;
        *((__global float *)((__global char *)dst + dst_index_1)) = data0;
    }
}

#if defined (DOUBLE_SUPPORT)
__kernel void arithm_flip_cols_C1_D6 (__global double *src, int src_step, int src_offset,
                                      __global double *dst, int dst_step, int dst_offset,
                                      int rows, int cols, int thread_cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < thread_cols && y < rows)
    {
        int src_index_0 = mad24(y, src_step, (x << 3)             + src_offset);
        int src_index_1 = mad24(y, src_step, ((cols - x -1) << 3) + src_offset);

        int dst_index_0 = mad24(y, dst_step, (x << 3)             + dst_offset);
        int dst_index_1 = mad24(y, dst_step, ((cols - x -1) << 3) + dst_offset);

        double data0 = *((__global double *)((__global char *)src + src_index_0));
        double data1 = *((__global double *)((__global char *)src + src_index_1));

        *((__global double *)((__global char *)dst + dst_index_0)) = data1;
        *((__global double *)((__global char *)dst + dst_index_1)) = data0;
    }
}
#endif
__kernel void arithm_flip_cols_C2_D0 (__global uchar *src, int src_step, int src_offset,
                                      __global uchar *dst, int dst_step, int dst_offset,
                                      int rows, int cols, int thread_cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < thread_cols && y < rows)
    {
        int src_index_0 = mad24(y, src_step, (x << 1)             + src_offset);
        int src_index_1 = mad24(y, src_step, ((cols - x -1) << 1) + src_offset);

        int dst_index_0 = mad24(y, dst_step, (x << 1)             + dst_offset);
        int dst_index_1 = mad24(y, dst_step, ((cols - x -1) << 1) + dst_offset);

        uchar2 data0 = *((__global uchar2 *)((__global char *)src + src_index_0));
        uchar2 data1 = *((__global uchar2 *)((__global char *)src + src_index_1));

        *((__global uchar2 *)((__global char *)dst + dst_index_0)) = data1;
        *((__global uchar2 *)((__global char *)dst + dst_index_1)) = data0;
    }
}
__kernel void arithm_flip_cols_C2_D1 (__global char *src, int src_step, int src_offset,
                                      __global char *dst, int dst_step, int dst_offset,
                                      int rows, int cols, int thread_cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < thread_cols && y < rows)
    {
        int src_index_0 = mad24(y, src_step, (x << 1)             + src_offset);
        int src_index_1 = mad24(y, src_step, ((cols - x -1) << 1) + src_offset);

        int dst_index_0 = mad24(y, dst_step, (x << 1)             + dst_offset);
        int dst_index_1 = mad24(y, dst_step, ((cols - x -1) << 1) + dst_offset);

        char2 data0 = *((__global char2 *)((__global char *)src + src_index_0));
        char2 data1 = *((__global char2 *)((__global char *)src + src_index_1));

        *((__global char2 *)((__global char *)dst + dst_index_0)) = data1;
        *((__global char2 *)((__global char *)dst + dst_index_1)) = data0;
    }
}
__kernel void arithm_flip_cols_C2_D2 (__global ushort *src, int src_step, int src_offset,
                                      __global ushort *dst, int dst_step, int dst_offset,
                                      int rows, int cols, int thread_cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < thread_cols && y < rows)
    {
        int src_index_0 = mad24(y, src_step, (x << 2)             + src_offset);
        int src_index_1 = mad24(y, src_step, ((cols - x -1) << 2) + src_offset);

        int dst_index_0 = mad24(y, dst_step, (x << 2)             + dst_offset);
        int dst_index_1 = mad24(y, dst_step, ((cols - x -1) << 2) + dst_offset);

        ushort2 data0 = *((__global ushort2 *)((__global char *)src + src_index_0));
        ushort2 data1 = *((__global ushort2 *)((__global char *)src + src_index_1));

        *((__global ushort2 *)((__global char *)dst + dst_index_0)) = data1;
        *((__global ushort2 *)((__global char *)dst + dst_index_1)) = data0;
    }
}
__kernel void arithm_flip_cols_C2_D3 (__global short *src, int src_step, int src_offset,
                                      __global short *dst, int dst_step, int dst_offset,
                                      int rows, int cols, int thread_cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < thread_cols && y < rows)
    {
        int src_index_0 = mad24(y, src_step, (x << 2)             + src_offset);
        int src_index_1 = mad24(y, src_step, ((cols - x -1) << 2) + src_offset);

        int dst_index_0 = mad24(y, dst_step, (x << 2)             + dst_offset);
        int dst_index_1 = mad24(y, dst_step, ((cols - x -1) << 2) + dst_offset);

        short2 data0 = *((__global short2 *)((__global char *)src + src_index_0));
        short2 data1 = *((__global short2 *)((__global char *)src + src_index_1));

        *((__global short2 *)((__global char *)dst + dst_index_0)) = data1;
        *((__global short2 *)((__global char *)dst + dst_index_1)) = data0;
    }
}
__kernel void arithm_flip_cols_C2_D4 (__global int *src, int src_step, int src_offset,
                                      __global int *dst, int dst_step, int dst_offset,
                                      int rows, int cols, int thread_cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < thread_cols && y < rows)
    {
        int src_index_0 = mad24(y, src_step, (x << 3)             + src_offset);
        int src_index_1 = mad24(y, src_step, ((cols - x -1) << 3) + src_offset);

        int dst_index_0 = mad24(y, dst_step, (x << 3)             + dst_offset);
        int dst_index_1 = mad24(y, dst_step, ((cols - x -1) << 3) + dst_offset);

        int2 data0 = *((__global int2 *)((__global char *)src + src_index_0));
        int2 data1 = *((__global int2 *)((__global char *)src + src_index_1));

        *((__global int2 *)((__global char *)dst + dst_index_0)) = data1;
        *((__global int2 *)((__global char *)dst + dst_index_1)) = data0;
    }
}
__kernel void arithm_flip_cols_C2_D5 (__global float *src, int src_step, int src_offset,
                                      __global float *dst, int dst_step, int dst_offset,
                                      int rows, int cols, int thread_cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < thread_cols && y < rows)
    {
        int src_index_0 = mad24(y, src_step, (x << 3)             + src_offset);
        int src_index_1 = mad24(y, src_step, ((cols - x -1) << 3) + src_offset);

        int dst_index_0 = mad24(y, dst_step, (x << 3)             + dst_offset);
        int dst_index_1 = mad24(y, dst_step, ((cols - x -1) << 3) + dst_offset);

        float2 data0 = *((__global float2 *)((__global char *)src + src_index_0));
        float2 data1 = *((__global float2 *)((__global char *)src + src_index_1));

        *((__global float2 *)((__global char *)dst + dst_index_0)) = data1;
        *((__global float2 *)((__global char *)dst + dst_index_1)) = data0;
    }
}

#if defined (DOUBLE_SUPPORT)
__kernel void arithm_flip_cols_C2_D6 (__global double *src, int src_step, int src_offset,
                                      __global double *dst, int dst_step, int dst_offset,
                                      int rows, int cols, int thread_cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < thread_cols && y < rows)
    {
        int src_index_0 = mad24(y, src_step, (x << 4)             + src_offset);
        int src_index_1 = mad24(y, src_step, ((cols - x -1) << 4) + src_offset);

        int dst_index_0 = mad24(y, dst_step, (x << 4)             + dst_offset);
        int dst_index_1 = mad24(y, dst_step, ((cols - x -1) << 4) + dst_offset);

        double2 data0 = *((__global double2 *)((__global char *)src + src_index_0));
        double2 data1 = *((__global double2 *)((__global char *)src + src_index_1));

        *((__global double2 *)((__global char *)dst + dst_index_0)) = data1;
        *((__global double2 *)((__global char *)dst + dst_index_1)) = data0;
    }
}
#endif

__kernel void arithm_flip_cols_C3_D0 (__global uchar *src, int src_step, int src_offset,
                                      __global uchar *dst, int dst_step, int dst_offset,
                                      int rows, int cols, int thread_cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < thread_cols && y < rows)
    {
        int src_index_0 = mad24(y, src_step, (x) * 3           + src_offset);
        int src_index_1 = mad24(y, src_step, (cols - x -1) * 3 + src_offset);

        int dst_index_0 = mad24(y, dst_step, (x) * 3           + dst_offset);
        int dst_index_1 = mad24(y, dst_step, (cols - x -1) * 3 + dst_offset);

        uchar data0_0 = *(src + src_index_0 + 0);
        uchar data0_1 = *(src + src_index_0 + 1);
        uchar data0_2 = *(src + src_index_0 + 2);

        uchar data1_0 = *(src + src_index_1 + 0);
        uchar data1_1 = *(src + src_index_1 + 1);
        uchar data1_2 = *(src + src_index_1 + 2);

        *(dst + dst_index_0 + 0 ) = data1_0;
        *(dst + dst_index_0 + 1 ) = data1_1;
        *(dst + dst_index_0 + 2 ) = data1_2;

        *(dst + dst_index_1 + 0) = data0_0;
        *(dst + dst_index_1 + 1) = data0_1;
        *(dst + dst_index_1 + 2) = data0_2;
    }
}
__kernel void arithm_flip_cols_C3_D1 (__global char *src, int src_step, int src_offset,
                                      __global char *dst, int dst_step, int dst_offset,
                                      int rows, int cols, int thread_cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < thread_cols && y < rows)
    {
        int src_index_0 = mad24(y, src_step, (x) * 3           + src_offset);
        int src_index_1 = mad24(y, src_step, (cols - x -1) * 3 + src_offset);

        int dst_index_0 = mad24(y, dst_step, (x) * 3           + dst_offset);
        int dst_index_1 = mad24(y, dst_step, (cols - x -1) * 3 + dst_offset);

        char data0_0 = *(src + src_index_0 + 0);
        char data0_1 = *(src + src_index_0 + 1);
        char data0_2 = *(src + src_index_0 + 2);

        char data1_0 = *(src + src_index_1 + 0);
        char data1_1 = *(src + src_index_1 + 1);
        char data1_2 = *(src + src_index_1 + 2);

        *(dst + dst_index_0 + 0 ) = data1_0;
        *(dst + dst_index_0 + 1 ) = data1_1;
        *(dst + dst_index_0 + 2 ) = data1_2;

        *(dst + dst_index_1 + 0) = data0_0;
        *(dst + dst_index_1 + 1) = data0_1;
        *(dst + dst_index_1 + 2) = data0_2;
    }
}
__kernel void arithm_flip_cols_C3_D2 (__global ushort *src, int src_step, int src_offset,
                                      __global ushort *dst, int dst_step, int dst_offset,
                                      int rows, int cols, int thread_cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < thread_cols && y < rows)
    {
        int src_index_0 = mad24(y, src_step, (x * 3 << 1)             + src_offset);
        int src_index_1 = mad24(y, src_step, ((cols - x -1) * 3 << 1) + src_offset);

        int dst_index_0 = mad24(y, dst_step, (x * 3 << 1)             + dst_offset);
        int dst_index_1 = mad24(y, dst_step, ((cols - x -1) * 3 << 1) + dst_offset);

        ushort data0_0 = *((__global ushort *)((__global char *)src + src_index_0 + 0));
        ushort data0_1 = *((__global ushort *)((__global char *)src + src_index_0 + 2));
        ushort data0_2 = *((__global ushort *)((__global char *)src + src_index_0 + 4));

        ushort data1_0 = *((__global ushort *)((__global char *)src + src_index_1 + 0));
        ushort data1_1 = *((__global ushort *)((__global char *)src + src_index_1 + 2));
        ushort data1_2 = *((__global ushort *)((__global char *)src + src_index_1 + 4));

        *((__global ushort *)((__global char *)dst + dst_index_0 + 0)) = data1_0;
        *((__global ushort *)((__global char *)dst + dst_index_0 + 2)) = data1_1;
        *((__global ushort *)((__global char *)dst + dst_index_0 + 4)) = data1_2;

        *((__global ushort *)((__global char *)dst + dst_index_1 + 0)) = data0_0;
        *((__global ushort *)((__global char *)dst + dst_index_1 + 2)) = data0_1;
        *((__global ushort *)((__global char *)dst + dst_index_1 + 4)) = data0_2;
    }
}
__kernel void arithm_flip_cols_C3_D3 (__global short *src, int src_step, int src_offset,
                                      __global short *dst, int dst_step, int dst_offset,
                                      int rows, int cols, int thread_cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < thread_cols && y < rows)
    {
        int src_index_0 = mad24(y, src_step, (x * 3 << 1)             + src_offset);
        int src_index_1 = mad24(y, src_step, ((cols - x -1) * 3 << 1) + src_offset);

        int dst_index_0 = mad24(y, dst_step, (x * 3 << 1)             + dst_offset);
        int dst_index_1 = mad24(y, dst_step, ((cols - x -1) * 3 << 1) + dst_offset);

        short data0_0 = *((__global short *)((__global char *)src + src_index_0 + 0));
        short data0_1 = *((__global short *)((__global char *)src + src_index_0 + 2));
        short data0_2 = *((__global short *)((__global char *)src + src_index_0 + 4));

        short data1_0 = *((__global short *)((__global char *)src + src_index_1 + 0));
        short data1_1 = *((__global short *)((__global char *)src + src_index_1 + 2));
        short data1_2 = *((__global short *)((__global char *)src + src_index_1 + 4));

        *((__global short *)((__global char *)dst + dst_index_0 + 0)) = data1_0;
        *((__global short *)((__global char *)dst + dst_index_0 + 2)) = data1_1;
        *((__global short *)((__global char *)dst + dst_index_0 + 4)) = data1_2;

        *((__global short *)((__global char *)dst + dst_index_1 + 0)) = data0_0;
        *((__global short *)((__global char *)dst + dst_index_1 + 2)) = data0_1;
        *((__global short *)((__global char *)dst + dst_index_1 + 4)) = data0_2;
    }
}
__kernel void arithm_flip_cols_C3_D4 (__global int *src, int src_step, int src_offset,
                                      __global int *dst, int dst_step, int dst_offset,
                                      int rows, int cols, int thread_cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < thread_cols && y < rows)
    {
        int src_index_0 = mad24(y, src_step, (x * 3 << 2)             + src_offset);
        int src_index_1 = mad24(y, src_step, ((cols - x -1) * 3 << 2) + src_offset);

        int dst_index_0 = mad24(y, dst_step, (x * 3 << 2)             + dst_offset);
        int dst_index_1 = mad24(y, dst_step, ((cols - x -1) * 3 << 2) + dst_offset);

        int data0_0 = *((__global int *)((__global char *)src + src_index_0 + 0));
        int data0_1 = *((__global int *)((__global char *)src + src_index_0 + 4));
        int data0_2 = *((__global int *)((__global char *)src + src_index_0 + 8));

        int data1_0 = *((__global int *)((__global char *)src + src_index_1 + 0));
        int data1_1 = *((__global int *)((__global char *)src + src_index_1 + 4));
        int data1_2 = *((__global int *)((__global char *)src + src_index_1 + 8));

        *((__global int *)((__global char *)dst + dst_index_0 + 0)) = data1_0;
        *((__global int *)((__global char *)dst + dst_index_0 + 4)) = data1_1;
        *((__global int *)((__global char *)dst + dst_index_0 + 8)) = data1_2;

        *((__global int *)((__global char *)dst + dst_index_1 + 0)) = data0_0;
        *((__global int *)((__global char *)dst + dst_index_1 + 4)) = data0_1;
        *((__global int *)((__global char *)dst + dst_index_1 + 8)) = data0_2;
    }
}
__kernel void arithm_flip_cols_C3_D5 (__global float *src, int src_step, int src_offset,
                                      __global float *dst, int dst_step, int dst_offset,
                                      int rows, int cols, int thread_cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < thread_cols && y < rows)
    {
        int src_index_0 = mad24(y, src_step, (x * 3 << 2)             + src_offset);
        int src_index_1 = mad24(y, src_step, ((cols - x -1) * 3 << 2) + src_offset);

        int dst_index_0 = mad24(y, dst_step, (x * 3 << 2)             + dst_offset);
        int dst_index_1 = mad24(y, dst_step, ((cols - x -1) * 3 << 2) + dst_offset);

        float data0_0 = *((__global float *)((__global char *)src + src_index_0 + 0));
        float data0_1 = *((__global float *)((__global char *)src + src_index_0 + 4));
        float data0_2 = *((__global float *)((__global char *)src + src_index_0 + 8));

        float data1_0 = *((__global float *)((__global char *)src + src_index_1 + 0));
        float data1_1 = *((__global float *)((__global char *)src + src_index_1 + 4));
        float data1_2 = *((__global float *)((__global char *)src + src_index_1 + 8));

        *((__global float *)((__global char *)dst + dst_index_0 + 0)) = data1_0;
        *((__global float *)((__global char *)dst + dst_index_0 + 4)) = data1_1;
        *((__global float *)((__global char *)dst + dst_index_0 + 8)) = data1_2;

        *((__global float *)((__global char *)dst + dst_index_1 + 0)) = data0_0;
        *((__global float *)((__global char *)dst + dst_index_1 + 4)) = data0_1;
        *((__global float *)((__global char *)dst + dst_index_1 + 8)) = data0_2;
    }
}

#if defined (DOUBLE_SUPPORT)
__kernel void arithm_flip_cols_C3_D6 (__global double *src, int src_step, int src_offset,
                                      __global double *dst, int dst_step, int dst_offset,
                                      int rows, int cols, int thread_cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < thread_cols && y < rows)
    {
        int src_index_0 = mad24(y, src_step, (x * 3 << 3)             + src_offset);
        int src_index_1 = mad24(y, src_step, ((cols - x -1) * 3 << 3) + src_offset);

        int dst_index_0 = mad24(y, dst_step, (x * 3 << 3)             + dst_offset);
        int dst_index_1 = mad24(y, dst_step, ((cols - x -1) * 3 << 3) + dst_offset);

        double data0_0 = *((__global double *)((__global char *)src + src_index_0 + 0));
        double data0_1 = *((__global double *)((__global char *)src + src_index_0 + 8));
        double data0_2 = *((__global double *)((__global char *)src + src_index_0 + 16));

        double data1_0 = *((__global double *)((__global char *)src + src_index_1 + 0));
        double data1_1 = *((__global double *)((__global char *)src + src_index_1 + 8));
        double data1_2 = *((__global double *)((__global char *)src + src_index_1 + 16));

        *((__global double *)((__global char *)dst + dst_index_0 + 0 )) = data1_0;
        *((__global double *)((__global char *)dst + dst_index_0 + 8 )) = data1_1;
        *((__global double *)((__global char *)dst + dst_index_0 + 16)) = data1_2;

        *((__global double *)((__global char *)dst + dst_index_1 + 0 )) = data0_0;
        *((__global double *)((__global char *)dst + dst_index_1 + 8 )) = data0_1;
        *((__global double *)((__global char *)dst + dst_index_1 + 16)) = data0_2;
    }
}
#endif
__kernel void arithm_flip_cols_C4_D0 (__global uchar *src, int src_step, int src_offset,
                                      __global uchar *dst, int dst_step, int dst_offset,
                                      int rows, int cols, int thread_cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < thread_cols && y < rows)
    {
        int src_index_0 = mad24(y, src_step, (x << 2)             + src_offset);
        int src_index_1 = mad24(y, src_step, ((cols - x -1) << 2) + src_offset);

        int dst_index_0 = mad24(y, dst_step, (x << 2)             + dst_offset);
        int dst_index_1 = mad24(y, dst_step, ((cols - x -1) << 2) + dst_offset);

        uchar4 data0 = *((__global uchar4 *)(src + src_index_0));
        uchar4 data1 = *((__global uchar4 *)(src + src_index_1));

        *((__global uchar4 *)(dst + dst_index_0)) = data1;
        *((__global uchar4 *)(dst + dst_index_1)) = data0;
    }
}
__kernel void arithm_flip_cols_C4_D1 (__global char *src, int src_step, int src_offset,
                                      __global char *dst, int dst_step, int dst_offset,
                                      int rows, int cols, int thread_cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < thread_cols && y < rows)
    {
        int src_index_0 = mad24(y, src_step, (x << 2)             + src_offset);
        int src_index_1 = mad24(y, src_step, ((cols - x -1) << 2) + src_offset);

        int dst_index_0 = mad24(y, dst_step, (x << 2)             + dst_offset);
        int dst_index_1 = mad24(y, dst_step, ((cols - x -1) << 2) + dst_offset);

        char4 data0 = *((__global char4 *)(src + src_index_0));
        char4 data1 = *((__global char4 *)(src + src_index_1));

        *((__global char4 *)(dst + dst_index_0)) = data1;
        *((__global char4 *)(dst + dst_index_1)) = data0;
    }
}
__kernel void arithm_flip_cols_C4_D2 (__global ushort *src, int src_step, int src_offset,
                                      __global ushort *dst, int dst_step, int dst_offset,
                                      int rows, int cols, int thread_cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < thread_cols && y < rows)
    {
        int src_index_0 = mad24(y, src_step, (x << 3)             + src_offset);
        int src_index_1 = mad24(y, src_step, ((cols - x -1) << 3) + src_offset);

        int dst_index_0 = mad24(y, dst_step, (x << 3)             + dst_offset);
        int dst_index_1 = mad24(y, dst_step, ((cols - x -1) << 3) + dst_offset);

        ushort4 data0 = *((__global ushort4 *)((__global char *)src + src_index_0));
        ushort4 data1 = *((__global ushort4 *)((__global char *)src + src_index_1));

        *((__global ushort4 *)((__global char *)dst + dst_index_0)) = data1;
        *((__global ushort4 *)((__global char *)dst + dst_index_1)) = data0;
    }
}
__kernel void arithm_flip_cols_C4_D3 (__global short *src, int src_step, int src_offset,
                                      __global short *dst, int dst_step, int dst_offset,
                                      int rows, int cols, int thread_cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < thread_cols && y < rows)
    {
        int src_index_0 = mad24(y, src_step, (x << 3)             + src_offset);
        int src_index_1 = mad24(y, src_step, ((cols - x -1) << 3) + src_offset);

        int dst_index_0 = mad24(y, dst_step, (x << 3)             + dst_offset);
        int dst_index_1 = mad24(y, dst_step, ((cols - x -1) << 3) + dst_offset);

        short4 data0 = *((__global short4 *)((__global char *)src + src_index_0));
        short4 data1 = *((__global short4 *)((__global char *)src + src_index_1));

        *((__global short4 *)((__global char *)dst + dst_index_0)) = data1;
        *((__global short4 *)((__global char *)dst + dst_index_1)) = data0;
    }
}

__kernel void arithm_flip_cols_C4_D4 (__global int *src, int src_step, int src_offset,
                                      __global int *dst, int dst_step, int dst_offset,
                                      int rows, int cols, int thread_cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < thread_cols && y < rows)
    {
        int src_index_0 = mad24(y, src_step, (x << 4)             + src_offset);
        int src_index_1 = mad24(y, src_step, ((cols - x -1) << 4) + src_offset);

        int dst_index_0 = mad24(y, dst_step, (x << 4)             + dst_offset);
        int dst_index_1 = mad24(y, dst_step, ((cols - x -1) << 4) + dst_offset);

        int4 data0 = *((__global int4 *)((__global char *)src + src_index_0));
        int4 data1 = *((__global int4 *)((__global char *)src + src_index_1));

        *((__global int4 *)((__global char *)dst + dst_index_0)) = data1;
        *((__global int4 *)((__global char *)dst + dst_index_1)) = data0;
    }
}
__kernel void arithm_flip_cols_C4_D5 (__global float *src, int src_step, int src_offset,
                                      __global float *dst, int dst_step, int dst_offset,
                                      int rows, int cols, int thread_cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < thread_cols && y < rows)
    {
        int src_index_0 = mad24(y, src_step, (x << 4)             + src_offset);
        int src_index_1 = mad24(y, src_step, ((cols - x -1) << 4) + src_offset);

        int dst_index_0 = mad24(y, dst_step, (x << 4)             + dst_offset);
        int dst_index_1 = mad24(y, dst_step, ((cols - x -1) << 4) + dst_offset);

        float4 data0 = *((__global float4 *)((__global char *)src + src_index_0));
        float4 data1 = *((__global float4 *)((__global char *)src + src_index_1));

        *((__global float4 *)((__global char *)dst + dst_index_0)) = data1;
        *((__global float4 *)((__global char *)dst + dst_index_1)) = data0;
    }
}
#if defined (DOUBLE_SUPPORT)
__kernel void arithm_flip_cols_C4_D6 (__global double *src, int src_step, int src_offset,
                                      __global double *dst, int dst_step, int dst_offset,
                                      int rows, int cols, int thread_cols, int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < thread_cols && y < rows)
    {
        int src_index_0 = mad24(y, src_step, (x << 5)             + src_offset);
        int src_index_1 = mad24(y, src_step, ((cols - x -1) << 5) + src_offset);

        int dst_index_0 = mad24(y, dst_step, (x << 5)             + dst_offset);
        int dst_index_1 = mad24(y, dst_step, ((cols - x -1) << 5) + dst_offset);

        double4 data0 = *((__global double4 *)((__global char *)src + src_index_0));
        double4 data1 = *((__global double4 *)((__global char *)src + src_index_1));

        *((__global double4 *)((__global char *)dst + dst_index_0)) = data1;
        *((__global double4 *)((__global char *)dst + dst_index_1)) = data0;
    }
}
#endif
