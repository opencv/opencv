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
//////////////////////////////////optimized code using vector ////////////////////////////////
////////////vector fuction name format: split_vector_C(channels number)_D(data type depth)//////
////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void split_vector_C4_D0 (__global uchar *mat_src,  int src_step,  int src_offset,
                                  __global uchar *mat_dst0, int dst0_step, int dst0_offset,
                                  __global uchar *mat_dst1, int dst1_step, int dst1_offset,
                                    __global uchar *mat_dst2, int dst2_step, int dst2_offset,
                                  __global uchar *mat_dst3, int dst3_step, int dst3_offset,
                                  int rows, int cols, int dst_step1)

{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if((x  < cols) && (y < rows))
    {
        x = x << 2;

        int src_idx  = mad24(y, src_step, src_offset + (x << 2));

        int dst0_start = mad24(y, dst0_step, dst0_offset);
        int dst0_end   = mad24(y, dst0_step, dst0_offset + dst_step1);
        int dst0_idx   = mad24(y, dst0_step, dst0_offset + x) & (int)0xfffffffc;

        int dst1_start = mad24(y, dst1_step, dst1_offset);
        int dst1_end   = mad24(y, dst1_step, dst1_offset + dst_step1);
        int dst1_idx   = mad24(y, dst1_step, dst1_offset + x) & (int)0xfffffffc;

        int dst2_start = mad24(y, dst2_step, dst2_offset);
        int dst2_end   = mad24(y, dst2_step, dst2_offset + dst_step1);
        int dst2_idx   = mad24(y, dst2_step, dst2_offset + x) & (int)0xfffffffc;

        int dst3_start = mad24(y, dst3_step, dst3_offset);
        int dst3_end   = mad24(y, dst3_step, dst3_offset + dst_step1);
        int dst3_idx   = mad24(y, dst3_step, dst3_offset + x) & (int)0xfffffffc;

        uchar4 data_0 = *((global uchar4 *)(mat_src + (src_idx - 12 >= 0 ? src_idx - 12 : src_idx)));
        uchar4 data_1 = *((global uchar4 *)(mat_src + (src_idx - 8  >= 0 ? src_idx - 8  : src_idx)));
        uchar4 data_2 = *((global uchar4 *)(mat_src + (src_idx - 4  >= 0 ? src_idx - 4  : src_idx)));
        uchar4 data_3 = *((global uchar4 *)(mat_src + src_idx + 0 ));

        int total_bytes = src_offset + rows * src_step;
        uchar4 data_4 = *((global uchar4 *)(mat_src + (src_idx + 4  < total_bytes ? src_idx + 4  : src_idx)));
        uchar4 data_5 = *((global uchar4 *)(mat_src + (src_idx + 8  < total_bytes ? src_idx + 8  : src_idx)));
        uchar4 data_6 = *((global uchar4 *)(mat_src + (src_idx + 12 < total_bytes ? src_idx + 12 : src_idx)));

        uchar4 tmp_data0=1, tmp_data1=2, tmp_data2, tmp_data3;

        if((dst0_offset & 3) == 3)
            tmp_data0 = (uchar4)(data_0.x, data_1.x, data_2.x, data_3.x);
        if((dst0_offset & 3) == 2)
            tmp_data0 = (uchar4)(data_1.x, data_2.x, data_3.x, data_4.x);
        if((dst0_offset & 3) == 1)
            tmp_data0 = (uchar4)(data_2.x, data_3.x, data_4.x, data_5.x);
        if((dst0_offset & 3) == 0)
            tmp_data0 = (uchar4)(data_3.x, data_4.x, data_5.x, data_6.x);

        if((dst1_offset & 3) == 3)
            tmp_data1 = (uchar4)(data_0.y, data_1.y, data_2.y, data_3.y);
        if((dst1_offset & 3) == 2)
            tmp_data1 = (uchar4)(data_1.y, data_2.y, data_3.y, data_4.y);
        if((dst1_offset & 3) == 1)
            tmp_data1 = (uchar4)(data_2.y, data_3.y, data_4.y, data_5.y);
        if((dst1_offset & 3) == 0)
            tmp_data1 = (uchar4)(data_3.y, data_4.y, data_5.y, data_6.y);

        if((dst2_offset & 3) == 3)
            tmp_data2 = (uchar4)(data_0.z, data_1.z, data_2.z, data_3.z);
        if((dst2_offset & 3) == 2)
            tmp_data2 = (uchar4)(data_1.z, data_2.z, data_3.z, data_4.z);
        if((dst2_offset & 3) == 1)
            tmp_data2 = (uchar4)(data_2.z, data_3.z, data_4.z, data_5.z);
        if((dst2_offset & 3) == 0)
            tmp_data2 = (uchar4)(data_3.z, data_4.z, data_5.z, data_6.z);

        if((dst3_offset & 3) == 3)
            tmp_data3 = (uchar4)(data_0.w, data_1.w, data_2.w, data_3.w);
        if((dst3_offset & 3) == 2)
            tmp_data3 = (uchar4)(data_1.w, data_2.w, data_3.w, data_4.w);
        if((dst3_offset & 3) == 1)
            tmp_data3 = (uchar4)(data_2.w, data_3.w, data_4.w, data_5.w);
        if((dst3_offset & 3) == 0)
            tmp_data3 = (uchar4)(data_3.w, data_4.w, data_5.w, data_6.w);

        uchar4 dst0_data  = *((__global uchar4 *)(mat_dst0 + dst0_idx));
        uchar4 dst1_data  = *((__global uchar4 *)(mat_dst1 + dst1_idx));
        uchar4 dst2_data  = *((__global uchar4 *)(mat_dst2 + dst2_idx));
        uchar4 dst3_data  = *((__global uchar4 *)(mat_dst3 + dst3_idx));

        tmp_data0.x = ((dst0_idx + 0 >= dst0_start) && (dst0_idx + 0 < dst0_end)) ? tmp_data0.x : dst0_data.x;
        tmp_data0.y = ((dst0_idx + 1 >= dst0_start) && (dst0_idx + 1 < dst0_end)) ? tmp_data0.y : dst0_data.y;
        tmp_data0.z = ((dst0_idx + 2 >= dst0_start) && (dst0_idx + 2 < dst0_end)) ? tmp_data0.z : dst0_data.z;
        tmp_data0.w = ((dst0_idx + 3 >= dst0_start) && (dst0_idx + 3 < dst0_end)) ? tmp_data0.w : dst0_data.w;

        tmp_data1.x = ((dst1_idx + 0 >= dst1_start) && (dst1_idx + 0 < dst1_end)) ? tmp_data1.x : dst1_data.x;
        tmp_data1.y = ((dst1_idx + 1 >= dst1_start) && (dst1_idx + 1 < dst1_end)) ? tmp_data1.y : dst1_data.y;
        tmp_data1.z = ((dst1_idx + 2 >= dst1_start) && (dst1_idx + 2 < dst1_end)) ? tmp_data1.z : dst1_data.z;
        tmp_data1.w = ((dst1_idx + 3 >= dst1_start) && (dst1_idx + 3 < dst1_end)) ? tmp_data1.w : dst1_data.w;

        tmp_data2.x = ((dst2_idx + 0 >= dst2_start) && (dst2_idx + 0 < dst2_end)) ? tmp_data2.x : dst2_data.x;
        tmp_data2.y = ((dst2_idx + 1 >= dst2_start) && (dst2_idx + 1 < dst2_end)) ? tmp_data2.y : dst2_data.y;
        tmp_data2.z = ((dst2_idx + 2 >= dst2_start) && (dst2_idx + 2 < dst2_end)) ? tmp_data2.z : dst2_data.z;
        tmp_data2.w = ((dst2_idx + 3 >= dst2_start) && (dst2_idx + 3 < dst2_end)) ? tmp_data2.w : dst2_data.w;

        tmp_data3.x = ((dst3_idx + 0 >= dst3_start) && (dst3_idx + 0 < dst3_end)) ? tmp_data3.x : dst3_data.x;
        tmp_data3.y = ((dst3_idx + 1 >= dst3_start) && (dst3_idx + 1 < dst3_end)) ? tmp_data3.y : dst3_data.y;
        tmp_data3.z = ((dst3_idx + 2 >= dst3_start) && (dst3_idx + 2 < dst3_end)) ? tmp_data3.z : dst3_data.z;
        tmp_data3.w = ((dst3_idx + 3 >= dst3_start) && (dst3_idx + 3 < dst3_end)) ? tmp_data3.w : dst3_data.w;

        *((__global uchar4 *)(mat_dst0 + dst0_idx)) = tmp_data0;
        *((__global uchar4 *)(mat_dst1 + dst1_idx)) = tmp_data1;
        *((__global uchar4 *)(mat_dst2 + dst2_idx)) = tmp_data2;
        *((__global uchar4 *)(mat_dst3 + dst3_idx)) = tmp_data3;
    }
}

__kernel void split_vector_C3_D0 (__global uchar *mat_src,  int src_step,  int src_offset,
                                  __global uchar *mat_dst0, int dst0_step, int dst0_offset,
                                  __global uchar *mat_dst1, int dst1_step, int dst1_offset,
                                    __global uchar *mat_dst2, int dst2_step, int dst2_offset,
                                  int rows, int cols, int dst_step1)

{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if((x  < cols) && (y < rows))
    {
        x = x << 2;

        int src_idx  = mad24(y, src_step, src_offset);

        int dst0_start = mad24(y, dst0_step, dst0_offset);
        int dst0_end   = mad24(y, dst0_step, dst0_offset + dst_step1);
        int dst0_idx   = mad24(y, dst0_step, dst0_offset + x & (int)0xfffffffc);

        int dst1_start = mad24(y, dst1_step, dst1_offset);
        int dst1_end   = mad24(y, dst1_step, dst1_offset + dst_step1);
        int dst1_idx   = mad24(y, dst1_step, dst1_offset + x  & (int)0xfffffffc);

        int dst2_start = mad24(y, dst2_step, dst2_offset);
        int dst2_end   = mad24(y, dst2_step, dst2_offset + dst_step1);
        int dst2_idx   = mad24(y, dst2_step, dst2_offset + x & (int)0xfffffffc);

        uchar4 dst0_data  = *((__global uchar4 *)(mat_dst0 + dst0_idx));
        uchar4 dst1_data  = *((__global uchar4 *)(mat_dst1 + dst1_idx));
        uchar4 dst2_data  = *((__global uchar4 *)(mat_dst2 + dst2_idx));

        uchar4 tmp_data0, tmp_data1, tmp_data2;

        uchar src_data_0  =  *(mat_src + src_idx + 3 * x - 9);
        uchar src_data_1  =  *(mat_src + src_idx + 3 * x - 8);
        uchar src_data_2  =  *(mat_src + src_idx + 3 * x - 7);

        uchar src_data_3  =  *(mat_src + src_idx + 3 * x - 6);
        uchar src_data_4  =  *(mat_src + src_idx + 3 * x - 5);
        uchar src_data_5  =  *(mat_src + src_idx + 3 * x - 4);

        uchar src_data_6  =  *(mat_src + src_idx + 3 * x - 3);
        uchar src_data_7  =  *(mat_src + src_idx + 3 * x - 2);
        uchar src_data_8  =  *(mat_src + src_idx + 3 * x - 1);

        uchar src_data_9  =  *(mat_src + src_idx + 3 * x + 0);
        uchar src_data_10 =  *(mat_src + src_idx + 3 * x + 1);
        uchar src_data_11 =  *(mat_src + src_idx + 3 * x + 2);

        uchar src_data_12 =  *(mat_src + src_idx + 3 * x + 3);
        uchar src_data_13 =  *(mat_src + src_idx + 3 * x + 4);
        uchar src_data_14 =  *(mat_src + src_idx + 3 * x + 5);

        uchar src_data_15 =  *(mat_src + src_idx + 3 * x + 6);
        uchar src_data_16 =  *(mat_src + src_idx + 3 * x + 7);
        uchar src_data_17 =  *(mat_src + src_idx + 3 * x + 8);

        uchar src_data_18 =  *(mat_src + src_idx + 3 * x + 9);
        uchar src_data_19 =  *(mat_src + src_idx + 3 * x + 10);
        uchar src_data_20 =  *(mat_src + src_idx + 3 * x + 11);

        uchar data[7] = {src_data_0, src_data_3, src_data_6, src_data_9, src_data_12, src_data_15, src_data_18};
        int index = 3 - dst0_offset & 3;
        tmp_data0 = (uchar4)(data[index], data[index + 1], data[index + 2], data[index + 3]);

        uchar4 data0, data1, data2;

        data0     = (uchar4)(src_data_1, src_data_4, src_data_7, src_data_10);
        data1     = (dst1_offset & 3) == 2 ? (uchar4)(src_data_4, src_data_7, src_data_10, src_data_13)  : data0;
        data2     = (dst1_offset & 3) == 1 ? (uchar4)(src_data_7, src_data_10, src_data_13, src_data_16) : data1;
        tmp_data1 = (dst1_offset & 3) == 0 ? (uchar4)(src_data_10, src_data_13, src_data_16, src_data_19): data2;

        data0     = (uchar4)(src_data_2, src_data_5, src_data_8, src_data_11);
        data1     = (dst2_offset & 3) == 2 ? (uchar4)(src_data_5, src_data_8, src_data_11, src_data_14)   : data0;
        data2     = (dst2_offset & 3) == 1 ? (uchar4)(src_data_8, src_data_11, src_data_14, src_data_17)  : data1;
        tmp_data2 = (dst2_offset & 3) == 0 ? (uchar4)(src_data_11, src_data_14, src_data_17, src_data_20) : data2;

        tmp_data0.x = ((dst0_idx + 0 >= dst0_start) && (dst0_idx + 0 < dst0_end)) ? tmp_data0.x : dst0_data.x;
        tmp_data0.y = ((dst0_idx + 1 >= dst0_start) && (dst0_idx + 1 < dst0_end)) ? tmp_data0.y : dst0_data.y;
        tmp_data0.z = ((dst0_idx + 2 >= dst0_start) && (dst0_idx + 2 < dst0_end)) ? tmp_data0.z : dst0_data.z;
        tmp_data0.w = ((dst0_idx + 3 >= dst0_start) && (dst0_idx + 3 < dst0_end)) ? tmp_data0.w : dst0_data.w;

        tmp_data1.x = ((dst1_idx + 0 >= dst1_start) && (dst1_idx + 0 < dst1_end)) ? tmp_data1.x : dst1_data.x;
        tmp_data1.y = ((dst1_idx + 1 >= dst1_start) && (dst1_idx + 1 < dst1_end)) ? tmp_data1.y : dst1_data.y;
        tmp_data1.z = ((dst1_idx + 2 >= dst1_start) && (dst1_idx + 2 < dst1_end)) ? tmp_data1.z : dst1_data.z;
        tmp_data1.w = ((dst1_idx + 3 >= dst1_start) && (dst1_idx + 3 < dst1_end)) ? tmp_data1.w : dst1_data.w;

        tmp_data2.x = ((dst2_idx + 0 >= dst2_start) && (dst2_idx + 0 < dst2_end)) ? tmp_data2.x : dst2_data.x;
        tmp_data2.y = ((dst2_idx + 1 >= dst2_start) && (dst2_idx + 1 < dst2_end)) ? tmp_data2.y : dst2_data.y;
        tmp_data2.z = ((dst2_idx + 2 >= dst2_start) && (dst2_idx + 2 < dst2_end)) ? tmp_data2.z : dst2_data.z;
        tmp_data2.w = ((dst2_idx + 3 >= dst2_start) && (dst2_idx + 3 < dst2_end)) ? tmp_data2.w : dst2_data.w;

        *((__global uchar4 *)(mat_dst0 + dst0_idx)) = tmp_data0;
        *((__global uchar4 *)(mat_dst1 + dst1_idx)) = tmp_data1;
        *((__global uchar4 *)(mat_dst2 + dst2_idx)) = tmp_data2;
    }
}

__kernel void split_vector_C2_D0 (__global uchar *mat_src,  int src_step,  int src_offset,
                                  __global uchar *mat_dst0, int dst0_step, int dst0_offset,
                                  __global uchar *mat_dst1, int dst1_step, int dst1_offset,
                                  int rows, int cols, int dst_step1)

{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if((x  < cols) && (y < rows))
    {
        x = x << 2;

        #define dst0_align ((dst0_offset & 3) << 1)
        #define dst1_align ((dst1_offset & 3) << 1)
        int src_idx_0  = mad24(y, src_step, src_offset - dst0_align + (x << 1));
        int src_idx_1  = mad24(y, src_step, src_offset - dst1_align + (x << 1));

        int dst0_start = mad24(y, dst0_step, dst0_offset);
        int dst0_end   = mad24(y, dst0_step, dst0_offset + dst_step1);
        int dst0_idx   = mad24(y, dst0_step, dst0_offset + x & (int)0xfffffffc);

        int dst1_start = mad24(y, dst1_step, dst1_offset);
        int dst1_end   = mad24(y, dst1_step, dst1_offset + dst_step1);
        int dst1_idx   = mad24(y, dst1_step, dst1_offset + x & (int)0xfffffffc);

        int src1_index_fix = src_idx_0 < 0 ? 0 : src_idx_0;
        int src2_index_fix = src_idx_1 < 0 ? 0 : src_idx_1;
        uchar8 src_data_0 = vload8(0, mat_src + src_idx_0);
        uchar8 src_data_1 = vload8(0, mat_src + src_idx_1);
        if(src_idx_0 == -6)
            src_data_0.s01234567 = src_data_0.s67012345;
        if(src_idx_0 == -4)
            src_data_0.s01234567 = src_data_0.s45670123;
        if(src_idx_0 == -2)
            src_data_0.s01234567 = src_data_0.s23456701;
        if(src_idx_1 == -6)
            src_data_1.s01234567 = src_data_1.s67012345;
        if(src_idx_1 == -4)
            src_data_1.s01234567 = src_data_1.s45670123;
        if(src_idx_1 == -2)
            src_data_1.s01234567 = src_data_1.s23456701;

        uchar4 dst0_data  = *((__global uchar4 *)(mat_dst0 + dst0_idx));
        uchar4 dst1_data  = *((__global uchar4 *)(mat_dst1 + dst1_idx));

        uchar4 tmp_data0, tmp_data1;

        tmp_data0.x = ((dst0_idx + 0 >= dst0_start) && (dst0_idx + 0 < dst0_end)) ? src_data_0.s0 : dst0_data.x;
        tmp_data0.y = ((dst0_idx + 1 >= dst0_start) && (dst0_idx + 1 < dst0_end)) ? src_data_0.s2 : dst0_data.y;
        tmp_data0.z = ((dst0_idx + 2 >= dst0_start) && (dst0_idx + 2 < dst0_end)) ? src_data_0.s4 : dst0_data.z;
        tmp_data0.w = ((dst0_idx + 3 >= dst0_start) && (dst0_idx + 3 < dst0_end)) ? src_data_0.s6 : dst0_data.w;

        tmp_data1.x = ((dst1_idx + 0 >= dst1_start) && (dst1_idx + 0 < dst1_end)) ? src_data_1.s1 : dst1_data.x;
        tmp_data1.y = ((dst1_idx + 1 >= dst1_start) && (dst1_idx + 1 < dst1_end)) ? src_data_1.s3 : dst1_data.y;
        tmp_data1.z = ((dst1_idx + 2 >= dst1_start) && (dst1_idx + 2 < dst1_end)) ? src_data_1.s5 : dst1_data.z;
        tmp_data1.w = ((dst1_idx + 3 >= dst1_start) && (dst1_idx + 3 < dst1_end)) ? src_data_1.s7 : dst1_data.w;

        *((__global uchar4 *)(mat_dst0 + dst0_idx)) = tmp_data0;
        *((__global uchar4 *)(mat_dst1 + dst1_idx)) = tmp_data1;
    }
}

__kernel void split_vector_C4_D1 (__global char *mat_src,  int src_step,  int src_offset,
                                  __global char *mat_dst0, int dst0_step, int dst0_offset,
                                  __global char *mat_dst1, int dst1_step, int dst1_offset,
                                    __global char *mat_dst2, int dst2_step, int dst2_offset,
                                  __global char *mat_dst3, int dst3_step, int dst3_offset,
                                  int rows, int cols, int dst_step1)

{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if((x  < cols) && (y < rows))
    {
        x = x << 2;

        int src_idx  = mad24(y, src_step, src_offset + (x << 2));

        int dst0_start = mad24(y, dst0_step, dst0_offset);
        int dst0_end   = mad24(y, dst0_step, dst0_offset + dst_step1);
        int dst0_idx   = mad24(y, dst0_step, dst0_offset + x & (int)0xfffffffc);

        int dst1_start = mad24(y, dst1_step, dst1_offset);
        int dst1_end   = mad24(y, dst1_step, dst1_offset + dst_step1);
        int dst1_idx   = mad24(y, dst1_step, dst1_offset + x & (int)0xfffffffc);

        int dst2_start = mad24(y, dst2_step, dst2_offset);
        int dst2_end   = mad24(y, dst2_step, dst2_offset + dst_step1);
        int dst2_idx   = mad24(y, dst2_step, dst2_offset + x & (int)0xfffffffc);

        int dst3_start = mad24(y, dst3_step, dst3_offset);
        int dst3_end   = mad24(y, dst3_step, dst3_offset + dst_step1);
        int dst3_idx   = mad24(y, dst3_step, dst3_offset + x & (int)0xfffffffc);

        char4 data_0 = *((global char4 *)(mat_src + src_idx - 12));
        char4 data_1 = *((global char4 *)(mat_src + src_idx - 8 ));
        char4 data_2 = *((global char4 *)(mat_src + src_idx - 4 ));
        char4 data_3 = *((global char4 *)(mat_src + src_idx + 0 ));
        char4 data_4 = *((global char4 *)(mat_src + src_idx + 4 ));
        char4 data_5 = *((global char4 *)(mat_src + src_idx + 8 ));
        char4 data_6 = *((global char4 *)(mat_src + src_idx + 12));

        char4 tmp_data0=1, tmp_data1=2, tmp_data2, tmp_data3;

        if((dst0_offset & 3) == 3)
            tmp_data0 = (char4)(data_0.x, data_1.x, data_2.x, data_3.x);
        if((dst0_offset & 3) == 2)
            tmp_data0 = (char4)(data_1.x, data_2.x, data_3.x, data_4.x);
        if((dst0_offset & 3) == 1)
            tmp_data0 = (char4)(data_2.x, data_3.x, data_4.x, data_5.x);
        if((dst0_offset & 3) == 0)
            tmp_data0 = (char4)(data_3.x, data_4.x, data_5.x, data_6.x);

        if((dst1_offset & 3) == 3)
            tmp_data1 = (char4)(data_0.y, data_1.y, data_2.y, data_3.y);
        if((dst1_offset & 3) == 2)
            tmp_data1 = (char4)(data_1.y, data_2.y, data_3.y, data_4.y);
        if((dst1_offset & 3) == 1)
            tmp_data1 = (char4)(data_2.y, data_3.y, data_4.y, data_5.y);
        if((dst1_offset & 3) == 0)
            tmp_data1 = (char4)(data_3.y, data_4.y, data_5.y, data_6.y);

        if((dst2_offset & 3) == 3)
            tmp_data2 = (char4)(data_0.z, data_1.z, data_2.z, data_3.z);
        if((dst2_offset & 3) == 2)
            tmp_data2 = (char4)(data_1.z, data_2.z, data_3.z, data_4.z);
        if((dst2_offset & 3) == 1)
            tmp_data2 = (char4)(data_2.z, data_3.z, data_4.z, data_5.z);
        if((dst2_offset & 3) == 0)
            tmp_data2 = (char4)(data_3.z, data_4.z, data_5.z, data_6.z);

        if((dst3_offset & 3) == 3)
            tmp_data3 = (char4)(data_0.w, data_1.w, data_2.w, data_3.w);
        if((dst3_offset & 3) == 2)
            tmp_data3 = (char4)(data_1.w, data_2.w, data_3.w, data_4.w);
        if((dst3_offset & 3) == 1)
            tmp_data3 = (char4)(data_2.w, data_3.w, data_4.w, data_5.w);
        if((dst3_offset & 3) == 0)
            tmp_data3 = (char4)(data_3.w, data_4.w, data_5.w, data_6.w);

        char4 dst0_data  = *((__global char4 *)(mat_dst0 + dst0_idx));
        char4 dst1_data  = *((__global char4 *)(mat_dst1 + dst1_idx));
        char4 dst2_data  = *((__global char4 *)(mat_dst2 + dst2_idx));
        char4 dst3_data  = *((__global char4 *)(mat_dst3 + dst3_idx));

        tmp_data0.x = ((dst0_idx + 0 >= dst0_start) && (dst0_idx + 0 < dst0_end)) ? tmp_data0.x : dst0_data.x;
        tmp_data0.y = ((dst0_idx + 1 >= dst0_start) && (dst0_idx + 1 < dst0_end)) ? tmp_data0.y : dst0_data.y;
        tmp_data0.z = ((dst0_idx + 2 >= dst0_start) && (dst0_idx + 2 < dst0_end)) ? tmp_data0.z : dst0_data.z;
        tmp_data0.w = ((dst0_idx + 3 >= dst0_start) && (dst0_idx + 3 < dst0_end)) ? tmp_data0.w : dst0_data.w;

        tmp_data1.x = ((dst1_idx + 0 >= dst1_start) && (dst1_idx + 0 < dst1_end)) ? tmp_data1.x : dst1_data.x;
        tmp_data1.y = ((dst1_idx + 1 >= dst1_start) && (dst1_idx + 1 < dst1_end)) ? tmp_data1.y : dst1_data.y;
        tmp_data1.z = ((dst1_idx + 2 >= dst1_start) && (dst1_idx + 2 < dst1_end)) ? tmp_data1.z : dst1_data.z;
        tmp_data1.w = ((dst1_idx + 3 >= dst1_start) && (dst1_idx + 3 < dst1_end)) ? tmp_data1.w : dst1_data.w;

        tmp_data2.x = ((dst2_idx + 0 >= dst2_start) && (dst2_idx + 0 < dst2_end)) ? tmp_data2.x : dst2_data.x;
        tmp_data2.y = ((dst2_idx + 1 >= dst2_start) && (dst2_idx + 1 < dst2_end)) ? tmp_data2.y : dst2_data.y;
        tmp_data2.z = ((dst2_idx + 2 >= dst2_start) && (dst2_idx + 2 < dst2_end)) ? tmp_data2.z : dst2_data.z;
        tmp_data2.w = ((dst2_idx + 3 >= dst2_start) && (dst2_idx + 3 < dst2_end)) ? tmp_data2.w : dst2_data.w;

        tmp_data3.x = ((dst3_idx + 0 >= dst3_start) && (dst3_idx + 0 < dst3_end)) ? tmp_data3.x : dst3_data.x;
        tmp_data3.y = ((dst3_idx + 1 >= dst3_start) && (dst3_idx + 1 < dst3_end)) ? tmp_data3.y : dst3_data.y;
        tmp_data3.z = ((dst3_idx + 2 >= dst3_start) && (dst3_idx + 2 < dst3_end)) ? tmp_data3.z : dst3_data.z;
        tmp_data3.w = ((dst3_idx + 3 >= dst3_start) && (dst3_idx + 3 < dst3_end)) ? tmp_data3.w : dst3_data.w;

        *((__global char4 *)(mat_dst0 + dst0_idx)) = tmp_data0;
        *((__global char4 *)(mat_dst1 + dst1_idx)) = tmp_data1;
        *((__global char4 *)(mat_dst2 + dst2_idx)) = tmp_data2;
        *((__global char4 *)(mat_dst3 + dst3_idx)) = tmp_data3;
    }
}

__kernel void split_vector_C3_D1 (__global char *mat_src,  int src_step,  int src_offset,
                                  __global char *mat_dst0, int dst0_step, int dst0_offset,
                                  __global char *mat_dst1, int dst1_step, int dst1_offset,
                                    __global char *mat_dst2, int dst2_step, int dst2_offset,
                                  int rows, int cols, int dst_step1)

{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if((x  < cols) && (y < rows))
    {
        x = x << 2;

        int src_idx  = mad24(y, src_step, src_offset);

        int dst0_start = mad24(y, dst0_step, dst0_offset);
        int dst0_end   = mad24(y, dst0_step, dst0_offset + dst_step1);
        int dst0_idx   = mad24(y, dst0_step, dst0_offset + x & (int)0xfffffffc);

        int dst1_start = mad24(y, dst1_step, dst1_offset);
        int dst1_end   = mad24(y, dst1_step, dst1_offset + dst_step1);
        int dst1_idx   = mad24(y, dst1_step, dst1_offset + x  & (int)0xfffffffc);

        int dst2_start = mad24(y, dst2_step, dst2_offset);
        int dst2_end   = mad24(y, dst2_step, dst2_offset + dst_step1);
        int dst2_idx   = mad24(y, dst2_step, dst2_offset + x & (int)0xfffffffc);

        char4 dst0_data  = *((__global char4 *)(mat_dst0 + dst0_idx));
        char4 dst1_data  = *((__global char4 *)(mat_dst1 + dst1_idx));
        char4 dst2_data  = *((__global char4 *)(mat_dst2 + dst2_idx));

        char4 tmp_data0, tmp_data1, tmp_data2;

        char src_data_0  =  *(mat_src + src_idx + 3 * x - 9);
        char src_data_1  =  *(mat_src + src_idx + 3 * x - 8);
        char src_data_2  =  *(mat_src + src_idx + 3 * x - 7);

        char src_data_3  =  *(mat_src + src_idx + 3 * x - 6);
        char src_data_4  =  *(mat_src + src_idx + 3 * x - 5);
        char src_data_5  =  *(mat_src + src_idx + 3 * x - 4);

        char src_data_6  =  *(mat_src + src_idx + 3 * x - 3);
        char src_data_7  =  *(mat_src + src_idx + 3 * x - 2);
        char src_data_8  =  *(mat_src + src_idx + 3 * x - 1);

        char src_data_9  =  *(mat_src + src_idx + 3 * x + 0);
        char src_data_10 =  *(mat_src + src_idx + 3 * x + 1);
        char src_data_11 =  *(mat_src + src_idx + 3 * x + 2);

        char src_data_12 =  *(mat_src + src_idx + 3 * x + 3);
        char src_data_13 =  *(mat_src + src_idx + 3 * x + 4);
        char src_data_14 =  *(mat_src + src_idx + 3 * x + 5);

        char src_data_15 =  *(mat_src + src_idx + 3 * x + 6);
        char src_data_16 =  *(mat_src + src_idx + 3 * x + 7);
        char src_data_17 =  *(mat_src + src_idx + 3 * x + 8);

        char src_data_18 =  *(mat_src + src_idx + 3 * x + 9);
        char src_data_19 =  *(mat_src + src_idx + 3 * x + 10);
        char src_data_20 =  *(mat_src + src_idx + 3 * x + 11);

        char data[7] = {src_data_0, src_data_3, src_data_6, src_data_9, src_data_12, src_data_15, src_data_18};
        int index = 3 - dst0_offset & 3;
        tmp_data0 = (char4)(data[index], data[index + 1], data[index + 2], data[index + 3]);

        char4 data0, data1, data2;

        data0     = (char4)(src_data_1, src_data_4, src_data_7, src_data_10);
        data1     = (dst1_offset & 3) == 2 ? (char4)(src_data_4, src_data_7, src_data_10, src_data_13)  : data0;
        data2     = (dst1_offset & 3) == 1 ? (char4)(src_data_7, src_data_10, src_data_13, src_data_16) : data1;
        tmp_data1 = (dst1_offset & 3) == 0 ? (char4)(src_data_10, src_data_13, src_data_16, src_data_19): data2;

        data0     = (char4)(src_data_2, src_data_5, src_data_8, src_data_11);
        data1     = (dst2_offset & 3) == 2 ? (char4)(src_data_5, src_data_8, src_data_11, src_data_14)   : data0;
        data2     = (dst2_offset & 3) == 1 ? (char4)(src_data_8, src_data_11, src_data_14, src_data_17)  : data1;
        tmp_data2 = (dst2_offset & 3) == 0 ? (char4)(src_data_11, src_data_14, src_data_17, src_data_20) : data2;

        tmp_data0.x = ((dst0_idx + 0 >= dst0_start) && (dst0_idx + 0 < dst0_end)) ? tmp_data0.x : dst0_data.x;
        tmp_data0.y = ((dst0_idx + 1 >= dst0_start) && (dst0_idx + 1 < dst0_end)) ? tmp_data0.y : dst0_data.y;
        tmp_data0.z = ((dst0_idx + 2 >= dst0_start) && (dst0_idx + 2 < dst0_end)) ? tmp_data0.z : dst0_data.z;
        tmp_data0.w = ((dst0_idx + 3 >= dst0_start) && (dst0_idx + 3 < dst0_end)) ? tmp_data0.w : dst0_data.w;

        tmp_data1.x = ((dst1_idx + 0 >= dst1_start) && (dst1_idx + 0 < dst1_end)) ? tmp_data1.x : dst1_data.x;
        tmp_data1.y = ((dst1_idx + 1 >= dst1_start) && (dst1_idx + 1 < dst1_end)) ? tmp_data1.y : dst1_data.y;
        tmp_data1.z = ((dst1_idx + 2 >= dst1_start) && (dst1_idx + 2 < dst1_end)) ? tmp_data1.z : dst1_data.z;
        tmp_data1.w = ((dst1_idx + 3 >= dst1_start) && (dst1_idx + 3 < dst1_end)) ? tmp_data1.w : dst1_data.w;

        tmp_data2.x = ((dst2_idx + 0 >= dst2_start) && (dst2_idx + 0 < dst2_end)) ? tmp_data2.x : dst2_data.x;
        tmp_data2.y = ((dst2_idx + 1 >= dst2_start) && (dst2_idx + 1 < dst2_end)) ? tmp_data2.y : dst2_data.y;
        tmp_data2.z = ((dst2_idx + 2 >= dst2_start) && (dst2_idx + 2 < dst2_end)) ? tmp_data2.z : dst2_data.z;
        tmp_data2.w = ((dst2_idx + 3 >= dst2_start) && (dst2_idx + 3 < dst2_end)) ? tmp_data2.w : dst2_data.w;

        *((__global char4 *)(mat_dst0 + dst0_idx)) = tmp_data0;
        *((__global char4 *)(mat_dst1 + dst1_idx)) = tmp_data1;
        *((__global char4 *)(mat_dst2 + dst2_idx)) = tmp_data2;
    }
}

__kernel void split_vector_C2_D1 (__global char *mat_src,  int src_step,  int src_offset,
                                  __global char *mat_dst0, int dst0_step, int dst0_offset,
                                  __global char *mat_dst1, int dst1_step, int dst1_offset,
                                  int rows, int cols, int dst_step1)

{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if((x  < cols) && (y < rows))
    {
        x = x << 2;

        #define dst0_align ((dst0_offset & 3) << 1)
        #define dst1_align ((dst1_offset & 3) << 1)
        int src_idx_0  = mad24(y, src_step, src_offset - dst0_align + (x << 1));
        int src_idx_1  = mad24(y, src_step, src_offset - dst1_align + (x << 1));

        int dst0_start = mad24(y, dst0_step, dst0_offset);
        int dst0_end   = mad24(y, dst0_step, dst0_offset + dst_step1);
        int dst0_idx   = mad24(y, dst0_step, dst0_offset + x & (int)0xfffffffc);

        int dst1_start = mad24(y, dst1_step, dst1_offset);
        int dst1_end   = mad24(y, dst1_step, dst1_offset + dst_step1);
        int dst1_idx   = mad24(y, dst1_step, dst1_offset + x & (int)0xfffffffc);
    int src1_index_fix = src_idx_0 < 0 ? 0 : src_idx_0;
        int src2_index_fix = src_idx_1 < 0 ? 0 : src_idx_1;
        char8 src_data_0 = vload8(0, mat_src + src_idx_0);
        char8 src_data_1 = vload8(0, mat_src + src_idx_1);
        if(src_idx_0 == -6)
            src_data_0.s01234567 = src_data_0.s67012345;
        if(src_idx_0 == -4)
            src_data_0.s01234567 = src_data_0.s45670123;
        if(src_idx_0 == -2)
            src_data_0.s01234567 = src_data_0.s23456701;
        if(src_idx_1 == -6)
            src_data_1.s01234567 = src_data_1.s67012345;
        if(src_idx_1 == -4)
            src_data_1.s01234567 = src_data_1.s45670123;
        if(src_idx_1 == -2)
            src_data_1.s01234567 = src_data_1.s23456701;
        char4 dst0_data  = *((__global char4 *)(mat_dst0 + dst0_idx));
        char4 dst1_data  = *((__global char4 *)(mat_dst1 + dst1_idx));

        char4 tmp_data0, tmp_data1;

        tmp_data0.x = ((dst0_idx + 0 >= dst0_start) && (dst0_idx + 0 < dst0_end)) ? src_data_0.s0 : dst0_data.x;
        tmp_data0.y = ((dst0_idx + 1 >= dst0_start) && (dst0_idx + 1 < dst0_end)) ? src_data_0.s2 : dst0_data.y;
        tmp_data0.z = ((dst0_idx + 2 >= dst0_start) && (dst0_idx + 2 < dst0_end)) ? src_data_0.s4 : dst0_data.z;
        tmp_data0.w = ((dst0_idx + 3 >= dst0_start) && (dst0_idx + 3 < dst0_end)) ? src_data_0.s6 : dst0_data.w;

        tmp_data1.x = ((dst1_idx + 0 >= dst1_start) && (dst1_idx + 0 < dst1_end)) ? src_data_1.s1 : dst1_data.x;
        tmp_data1.y = ((dst1_idx + 1 >= dst1_start) && (dst1_idx + 1 < dst1_end)) ? src_data_1.s3 : dst1_data.y;
        tmp_data1.z = ((dst1_idx + 2 >= dst1_start) && (dst1_idx + 2 < dst1_end)) ? src_data_1.s5 : dst1_data.z;
        tmp_data1.w = ((dst1_idx + 3 >= dst1_start) && (dst1_idx + 3 < dst1_end)) ? src_data_1.s7 : dst1_data.w;

        *((__global char4 *)(mat_dst0 + dst0_idx)) = tmp_data0;
        *((__global char4 *)(mat_dst1 + dst1_idx)) = tmp_data1;
    }
}

__kernel void split_vector_C4_D2 (__global ushort *mat_src,  int src_step,  int src_offset,
                                  __global ushort *mat_dst0, int dst0_step, int dst0_offset,
                                  __global ushort *mat_dst1, int dst1_step, int dst1_offset,
                                    __global ushort *mat_dst2, int dst2_step, int dst2_offset,
                                  __global ushort *mat_dst3, int dst3_step, int dst3_offset,
                                  int rows, int cols, int dst_step1)

{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if((x  < cols) && (y < rows))
    {
        x = x << 1;

        int src_idx_0  = mad24(y, src_step, src_offset + (x << 3) - 8);
        int src_idx_1  = mad24(y, src_step, src_offset + (x << 3) + 8);

        int dst0_start = mad24(y, dst0_step, dst0_offset);
        int dst0_end   = mad24(y, dst0_step, dst0_offset + dst_step1);
        int dst0_idx   = mad24(y, dst0_step, dst0_offset + (x << 1) & (int)0xfffffffc);

        int dst1_start = mad24(y, dst1_step, dst1_offset);
        int dst1_end   = mad24(y, dst1_step, dst1_offset + dst_step1);
        int dst1_idx   = mad24(y, dst1_step, dst1_offset + (x << 1) & (int)0xfffffffc);

        int dst2_start = mad24(y, dst2_step, dst2_offset);
        int dst2_end   = mad24(y, dst2_step, dst2_offset + dst_step1);
        int dst2_idx   = mad24(y, dst2_step, dst2_offset + (x << 1) & (int)0xfffffffc);

        int dst3_start = mad24(y, dst3_step, dst3_offset);
        int dst3_end   = mad24(y, dst3_step, dst3_offset + dst_step1);
        int dst3_idx   = mad24(y, dst3_step, dst3_offset + (x << 1) & (int)0xfffffffc);

    int src1_index_fix = src_idx_0 < 0 ? 0 : src_idx_0;
        ushort8 src_data0 = vload8(0,(__global ushort *)((__global char *)mat_src + src_idx_0));
             if(src_idx_0 == -6)
            src_data0.s01234567 = src_data0.s67012345;
        if(src_idx_0 == -4)
            src_data0.s01234567 = src_data0.s45670123;
        if(src_idx_0 == -2)
            src_data0.s01234567 = src_data0.s23456701;
        ushort4 src_data1 = *((__global ushort4 *)((__global char *)mat_src + src_idx_1));

        ushort2 dst0_data  = *((__global ushort2 *)((__global char *)mat_dst0 + dst0_idx));
        ushort2 dst1_data  = *((__global ushort2 *)((__global char *)mat_dst1 + dst1_idx));
        ushort2 dst2_data  = *((__global ushort2 *)((__global char *)mat_dst2 + dst2_idx));
        ushort2 dst3_data  = *((__global ushort2 *)((__global char *)mat_dst3 + dst3_idx));

        ushort2 tmp_data0, tmp_data1, tmp_data2, tmp_data3;

        tmp_data0 = (dst0_offset & 3) == 0 ? (ushort2)(src_data0.s4, src_data1.s0) : (ushort2)(src_data0.s0, src_data0.s4);
        tmp_data1 = (dst1_offset & 3) == 0 ? (ushort2)(src_data0.s5, src_data1.s1) : (ushort2)(src_data0.s1, src_data0.s5);
        tmp_data2 = (dst2_offset & 3) == 0 ? (ushort2)(src_data0.s6, src_data1.s2) : (ushort2)(src_data0.s2, src_data0.s6);
        tmp_data3 = (dst3_offset & 3) == 0 ? (ushort2)(src_data0.s7, src_data1.s3) : (ushort2)(src_data0.s3, src_data0.s7);

        tmp_data0.x = ((dst0_idx + 0 >= dst0_start) && (dst0_idx + 0 < dst0_end)) ? tmp_data0.x : dst0_data.x;
        tmp_data0.y = ((dst0_idx + 2 >= dst0_start) && (dst0_idx + 2 < dst0_end)) ? tmp_data0.y : dst0_data.y;

        tmp_data1.x = ((dst1_idx + 0 >= dst1_start) && (dst1_idx + 0 < dst1_end)) ? tmp_data1.x : dst1_data.x;
        tmp_data1.y = ((dst1_idx + 2 >= dst1_start) && (dst1_idx + 2 < dst1_end)) ? tmp_data1.y : dst1_data.y;

        tmp_data2.x = ((dst2_idx + 0 >= dst2_start) && (dst2_idx + 0 < dst2_end)) ? tmp_data2.x : dst2_data.x;
        tmp_data2.y = ((dst2_idx + 2 >= dst2_start) && (dst2_idx + 2 < dst2_end)) ? tmp_data2.y : dst2_data.y;

        tmp_data3.x = ((dst3_idx + 0 >= dst3_start) && (dst3_idx + 0 < dst3_end)) ? tmp_data3.x : dst3_data.x;
        tmp_data3.y = ((dst3_idx + 2 >= dst3_start) && (dst3_idx + 2 < dst3_end)) ? tmp_data3.y : dst3_data.y;

        *((global ushort2 *)((__global char *)mat_dst0 + dst0_idx)) = tmp_data0;
        *((global ushort2 *)((__global char *)mat_dst1 + dst1_idx)) = tmp_data1;
        *((global ushort2 *)((__global char *)mat_dst2 + dst2_idx)) = tmp_data2;
        *((global ushort2 *)((__global char *)mat_dst3 + dst3_idx)) = tmp_data3;
    }
}

__kernel void split_vector_C3_D2 (__global ushort *mat_src,  int src_step,  int src_offset,
                                  __global ushort *mat_dst0, int dst0_step, int dst0_offset,
                                  __global ushort *mat_dst1, int dst1_step, int dst1_offset,
                                    __global ushort *mat_dst2, int dst2_step, int dst2_offset,
                                  int rows, int cols, int dst_step1)

{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if((x  < cols) && (y < rows))
    {
        x = x << 1;

        int src_idx  = mad24(y, src_step, src_offset);

        int dst0_start = mad24(y, dst0_step, dst0_offset);
        int dst0_end   = mad24(y, dst0_step, dst0_offset + dst_step1);
        int dst0_idx   = mad24(y, dst0_step, dst0_offset + (x << 1) & (int)0xfffffffc);

        int dst1_start = mad24(y, dst1_step, dst1_offset);
        int dst1_end   = mad24(y, dst1_step, dst1_offset + dst_step1);
        int dst1_idx   = mad24(y, dst1_step, dst1_offset + (x << 1) & (int)0xfffffffc);

        int dst2_start = mad24(y, dst2_step, dst2_offset);
        int dst2_end   = mad24(y, dst2_step, dst2_offset + dst_step1);
        int dst2_idx   = mad24(y, dst2_step, dst2_offset + (x << 1) & (int)0xfffffffc);

        ushort2 dst0_data  = *((__global ushort2 *)((__global char *)mat_dst0 + dst0_idx));
        ushort2 dst1_data  = *((__global ushort2 *)((__global char *)mat_dst1 + dst1_idx));
        ushort2 dst2_data  = *((__global ushort2 *)((__global char *)mat_dst2 + dst2_idx));

        ushort2 tmp_data0, tmp_data1, tmp_data2;

        ushort src_data_0 = ((__global ushort *)((__global char *)mat_src + src_idx))[3 * x - 3];
        ushort src_data_1 = ((__global ushort *)((__global char *)mat_src + src_idx))[3 * x - 2];
        ushort src_data_2 = ((__global ushort *)((__global char *)mat_src + src_idx))[3 * x - 1];
        ushort src_data_3 = ((__global ushort *)((__global char *)mat_src + src_idx))[3 * x + 0];
        ushort src_data_4 = ((__global ushort *)((__global char *)mat_src + src_idx))[3 * x + 1];
        ushort src_data_5 = ((__global ushort *)((__global char *)mat_src + src_idx))[3 * x + 2];
        ushort src_data_6 = ((__global ushort *)((__global char *)mat_src + src_idx))[3 * x + 3];
        ushort src_data_7 = ((__global ushort *)((__global char *)mat_src + src_idx))[3 * x + 4];
        ushort src_data_8 = ((__global ushort *)((__global char *)mat_src + src_idx))[3 * x + 5];

        tmp_data0 = (dst0_offset & 3) == 0 ? (ushort2)(src_data_3, src_data_6) : (ushort2)(src_data_0, src_data_3);
        tmp_data1 = (dst1_offset & 3) == 0 ? (ushort2)(src_data_4, src_data_7) : (ushort2)(src_data_1, src_data_4);
        tmp_data2 = (dst2_offset & 3) == 0 ? (ushort2)(src_data_5, src_data_8) : (ushort2)(src_data_2, src_data_5);

        tmp_data0.x = ((dst0_idx + 0 >= dst0_start) && (dst0_idx + 0 < dst0_end)) ? tmp_data0.x : dst0_data.x;
        tmp_data0.y = ((dst0_idx + 2 >= dst0_start) && (dst0_idx + 2 < dst0_end)) ? tmp_data0.y : dst0_data.y;

        tmp_data1.x = ((dst1_idx + 0 >= dst1_start) && (dst1_idx + 0 < dst1_end)) ? tmp_data1.x : dst1_data.x;
        tmp_data1.y = ((dst1_idx + 2 >= dst1_start) && (dst1_idx + 2 < dst1_end)) ? tmp_data1.y : dst1_data.y;

        tmp_data2.x = ((dst2_idx + 0 >= dst2_start) && (dst2_idx + 0 < dst2_end)) ? tmp_data2.x : dst2_data.x;
        tmp_data2.y = ((dst2_idx + 2 >= dst2_start) && (dst2_idx + 2 < dst2_end)) ? tmp_data2.y : dst2_data.y;

        *((__global ushort2 *)((__global char *)mat_dst0 + dst0_idx)) = tmp_data0;
        *((__global ushort2 *)((__global char *)mat_dst1 + dst1_idx)) = tmp_data1;
        *((__global ushort2 *)((__global char *)mat_dst2 + dst2_idx)) = tmp_data2;
    }
}

__kernel void split_vector_C2_D2 (__global ushort *mat_src,  int src_step,  int src_offset,
                                  __global ushort *mat_dst0, int dst0_step, int dst0_offset,
                                  __global ushort *mat_dst1, int dst1_step, int dst1_offset,
                                  int rows, int cols, int dst_step1)

{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if((x  < cols) && (y < rows))
    {
        x = x << 1;

        #define dst0_align ((dst0_offset & 3) << 1)
        #define dst1_align ((dst1_offset & 3) << 1)
        int src_idx_0  = mad24(y, src_step, src_offset - dst0_align + (x << 2));
        int src_idx_1  = mad24(y, src_step, src_offset - dst1_align + (x << 2));

        int dst0_start = mad24(y, dst0_step, dst0_offset);
        int dst0_end   = mad24(y, dst0_step, dst0_offset + dst_step1);
        int dst0_idx   = mad24(y, dst0_step, dst0_offset + (x << 1) & (int)0xfffffffc);

        int dst1_start = mad24(y, dst1_step, dst1_offset);
        int dst1_end   = mad24(y, dst1_step, dst1_offset + dst_step1);
        int dst1_idx   = mad24(y, dst1_step, dst1_offset + (x << 1) & (int)0xfffffffc);

        int src1_index_fix = src_idx_0 < 0 ? 0 : src_idx_0;
        int src2_index_fix = src_idx_1 < 0 ? 0 : src_idx_1;
        ushort4 src_data_0 = vload4(0, (__global ushort *)((__global char *)mat_src + src1_index_fix));
        ushort4 src_data_1 = vload4(0, (__global ushort *)((__global char *)mat_src + src2_index_fix));
        if(src_idx_0 < 0)
        {
            ushort4 tmp;
            tmp.xyzw = (src_idx_0 == -2) ? src_data_0.zwxy : src_data_0.yzwx;
            src_data_0.xyzw = (src_idx_1 == -1) ? src_data_0.wxyz:tmp.xyzw;
        }
        if(src_idx_1 < 0)
        {
            ushort4 tmp;
            tmp.xyzw = (src_idx_1 == -2) ? src_data_1.zwxy : src_data_1.yzwx;
            src_data_1.xyzw = (src_idx_1 == -1) ? src_data_1.wxyz : tmp.xyzw;
        }

        ushort2 dst0_data  = *((__global ushort2 *)((__global char *)mat_dst0 + dst0_idx));
        ushort2 dst1_data  = *((__global ushort2 *)((__global char *)mat_dst1 + dst1_idx));

        ushort2 tmp_data0, tmp_data1;

        tmp_data0.x = ((dst0_idx + 0 >= dst0_start) && (dst0_idx + 0 < dst0_end)) ? src_data_0.x : dst0_data.x;
        tmp_data0.y = ((dst0_idx + 2 >= dst0_start) && (dst0_idx + 2 < dst0_end)) ? src_data_0.z : dst0_data.y;

        tmp_data1.x = ((dst1_idx + 0 >= dst1_start) && (dst1_idx + 0 < dst1_end)) ? src_data_1.y : dst1_data.x;
        tmp_data1.y = ((dst1_idx + 2 >= dst1_start) && (dst1_idx + 2 < dst1_end)) ? src_data_1.w : dst1_data.y;

        *((global ushort2 *)((__global char *)mat_dst0 + dst0_idx)) = tmp_data0;
        *((global ushort2 *)((__global char *)mat_dst1 + dst1_idx)) = tmp_data1;
    }
}
__kernel void split_vector_C4_D3 (__global short *mat_src,  int src_step,  int src_offset,
                                  __global short *mat_dst0, int dst0_step, int dst0_offset,
                                  __global short *mat_dst1, int dst1_step, int dst1_offset,
                                    __global short *mat_dst2, int dst2_step, int dst2_offset,
                                  __global short *mat_dst3, int dst3_step, int dst3_offset,
                                  int rows, int cols, int dst_step1)

{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if((x  < cols) && (y < rows))
    {
        x = x << 1;

        int src_idx_0  = mad24(y, src_step, src_offset + (x << 3) - 8);
        int src_idx_1  = mad24(y, src_step, src_offset + (x << 3) + 8);

        int dst0_start = mad24(y, dst0_step, dst0_offset);
        int dst0_end   = mad24(y, dst0_step, dst0_offset + dst_step1);
        int dst0_idx   = mad24(y, dst0_step, dst0_offset + (x << 1) & (int)0xfffffffc);

        int dst1_start = mad24(y, dst1_step, dst1_offset);
        int dst1_end   = mad24(y, dst1_step, dst1_offset + dst_step1);
        int dst1_idx   = mad24(y, dst1_step, dst1_offset + (x << 1) & (int)0xfffffffc);

        int dst2_start = mad24(y, dst2_step, dst2_offset);
        int dst2_end   = mad24(y, dst2_step, dst2_offset + dst_step1);
        int dst2_idx   = mad24(y, dst2_step, dst2_offset + (x << 1) & (int)0xfffffffc);

        int dst3_start = mad24(y, dst3_step, dst3_offset);
        int dst3_end   = mad24(y, dst3_step, dst3_offset + dst_step1);
        int dst3_idx   = mad24(y, dst3_step, dst3_offset + (x << 1) & (int)0xfffffffc);
        int src1_index_fix = src_idx_0 < 0 ? 0 : src_idx_0;
        short8 src_data0 = vload8(0,(__global short *)((__global char *)mat_src + src_idx_0));

        if(src_idx_0 == -6)
            src_data0.s01234567 = src_data0.s67012345;
        if(src_idx_0 == -4)
            src_data0.s01234567 = src_data0.s45670123;
        if(src_idx_0 == -2)
            src_data0.s01234567 = src_data0.s23456701;

        short4 src_data1 = *((__global short4 *)((__global char *)mat_src + src_idx_1));

        short2 dst0_data  = *((__global short2 *)((__global char *)mat_dst0 + dst0_idx));
        short2 dst1_data  = *((__global short2 *)((__global char *)mat_dst1 + dst1_idx));
        short2 dst2_data  = *((__global short2 *)((__global char *)mat_dst2 + dst2_idx));
        short2 dst3_data  = *((__global short2 *)((__global char *)mat_dst3 + dst3_idx));

        short2 tmp_data0, tmp_data1, tmp_data2, tmp_data3;

        tmp_data0 = (dst0_offset & 3) == 0 ? (short2)(src_data0.s4, src_data1.s0) : (short2)(src_data0.s0, src_data0.s4);
        tmp_data1 = (dst1_offset & 3) == 0 ? (short2)(src_data0.s5, src_data1.s1) : (short2)(src_data0.s1, src_data0.s5);
        tmp_data2 = (dst2_offset & 3) == 0 ? (short2)(src_data0.s6, src_data1.s2) : (short2)(src_data0.s2, src_data0.s6);
        tmp_data3 = (dst3_offset & 3) == 0 ? (short2)(src_data0.s7, src_data1.s3) : (short2)(src_data0.s3, src_data0.s7);

        tmp_data0.x = ((dst0_idx + 0 >= dst0_start) && (dst0_idx + 0 < dst0_end)) ? tmp_data0.x : dst0_data.x;
        tmp_data0.y = ((dst0_idx + 2 >= dst0_start) && (dst0_idx + 2 < dst0_end)) ? tmp_data0.y : dst0_data.y;

        tmp_data1.x = ((dst1_idx + 0 >= dst1_start) && (dst1_idx + 0 < dst1_end)) ? tmp_data1.x : dst1_data.x;
        tmp_data1.y = ((dst1_idx + 2 >= dst1_start) && (dst1_idx + 2 < dst1_end)) ? tmp_data1.y : dst1_data.y;

        tmp_data2.x = ((dst2_idx + 0 >= dst2_start) && (dst2_idx + 0 < dst2_end)) ? tmp_data2.x : dst2_data.x;
        tmp_data2.y = ((dst2_idx + 2 >= dst2_start) && (dst2_idx + 2 < dst2_end)) ? tmp_data2.y : dst2_data.y;

        tmp_data3.x = ((dst3_idx + 0 >= dst3_start) && (dst3_idx + 0 < dst3_end)) ? tmp_data3.x : dst3_data.x;
        tmp_data3.y = ((dst3_idx + 2 >= dst3_start) && (dst3_idx + 2 < dst3_end)) ? tmp_data3.y : dst3_data.y;

        *((global short2 *)((__global char *)mat_dst0 + dst0_idx)) = tmp_data0;
        *((global short2 *)((__global char *)mat_dst1 + dst1_idx)) = tmp_data1;
        *((global short2 *)((__global char *)mat_dst2 + dst2_idx)) = tmp_data2;
        *((global short2 *)((__global char *)mat_dst3 + dst3_idx)) = tmp_data3;
    }
}
__kernel void split_vector_C3_D3 (__global short *mat_src,  int src_step,  int src_offset,
                                  __global short *mat_dst0, int dst0_step, int dst0_offset,
                                  __global short *mat_dst1, int dst1_step, int dst1_offset,
                                    __global short *mat_dst2, int dst2_step, int dst2_offset,
                                  int rows, int cols, int dst_step1)

{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if((x  < cols) && (y < rows))
    {
        x = x << 1;

        int src_idx  = mad24(y, src_step, src_offset);

        int dst0_start = mad24(y, dst0_step, dst0_offset);
        int dst0_end   = mad24(y, dst0_step, dst0_offset + dst_step1);
        int dst0_idx   = mad24(y, dst0_step, dst0_offset + (x << 1) & (int)0xfffffffc);

        int dst1_start = mad24(y, dst1_step, dst1_offset);
        int dst1_end   = mad24(y, dst1_step, dst1_offset + dst_step1);
        int dst1_idx   = mad24(y, dst1_step, dst1_offset + (x << 1) & (int)0xfffffffc);

        int dst2_start = mad24(y, dst2_step, dst2_offset);
        int dst2_end   = mad24(y, dst2_step, dst2_offset + dst_step1);
        int dst2_idx   = mad24(y, dst2_step, dst2_offset + (x << 1) & (int)0xfffffffc);

        short2 dst0_data  = *((__global short2 *)((__global char *)mat_dst0 + dst0_idx));
        short2 dst1_data  = *((__global short2 *)((__global char *)mat_dst1 + dst1_idx));
        short2 dst2_data  = *((__global short2 *)((__global char *)mat_dst2 + dst2_idx));

        short2 tmp_data0, tmp_data1, tmp_data2;

        short src_data_0 = ((__global short *)((__global char *)mat_src + src_idx))[3 * x - 3];
        short src_data_1 = ((__global short *)((__global char *)mat_src + src_idx))[3 * x - 2];
        short src_data_2 = ((__global short *)((__global char *)mat_src + src_idx))[3 * x - 1];
        short src_data_3 = ((__global short *)((__global char *)mat_src + src_idx))[3 * x + 0];
        short src_data_4 = ((__global short *)((__global char *)mat_src + src_idx))[3 * x + 1];
        short src_data_5 = ((__global short *)((__global char *)mat_src + src_idx))[3 * x + 2];
        short src_data_6 = ((__global short *)((__global char *)mat_src + src_idx))[3 * x + 3];
        short src_data_7 = ((__global short *)((__global char *)mat_src + src_idx))[3 * x + 4];
        short src_data_8 = ((__global short *)((__global char *)mat_src + src_idx))[3 * x + 5];

        tmp_data0 = (dst0_offset & 3) == 0 ? (short2)(src_data_3, src_data_6) : (short2)(src_data_0, src_data_3);
        tmp_data1 = (dst1_offset & 3) == 0 ? (short2)(src_data_4, src_data_7) : (short2)(src_data_1, src_data_4);
        tmp_data2 = (dst2_offset & 3) == 0 ? (short2)(src_data_5, src_data_8) : (short2)(src_data_2, src_data_5);

        tmp_data0.x = ((dst0_idx + 0 >= dst0_start) && (dst0_idx + 0 < dst0_end)) ? tmp_data0.x : dst0_data.x;
        tmp_data0.y = ((dst0_idx + 2 >= dst0_start) && (dst0_idx + 2 < dst0_end)) ? tmp_data0.y : dst0_data.y;

        tmp_data1.x = ((dst1_idx + 0 >= dst1_start) && (dst1_idx + 0 < dst1_end)) ? tmp_data1.x : dst1_data.x;
        tmp_data1.y = ((dst1_idx + 2 >= dst1_start) && (dst1_idx + 2 < dst1_end)) ? tmp_data1.y : dst1_data.y;

        tmp_data2.x = ((dst2_idx + 0 >= dst2_start) && (dst2_idx + 0 < dst2_end)) ? tmp_data2.x : dst2_data.x;
        tmp_data2.y = ((dst2_idx + 2 >= dst2_start) && (dst2_idx + 2 < dst2_end)) ? tmp_data2.y : dst2_data.y;

        *((__global short2 *)((__global char *)mat_dst0 + dst0_idx)) = tmp_data0;
        *((__global short2 *)((__global char *)mat_dst1 + dst1_idx)) = tmp_data1;
        *((__global short2 *)((__global char *)mat_dst2 + dst2_idx)) = tmp_data2;
    }
}


__kernel void split_vector_C2_D3 (__global short *mat_src,  int src_step,  int src_offset,
                                  __global short *mat_dst0, int dst0_step, int dst0_offset,
                                  __global short *mat_dst1, int dst1_step, int dst1_offset,
                                  int rows, int cols, int dst_step1)

{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if((x  < cols) && (y < rows))
    {
        x = x << 1;

        #define dst0_align ((dst0_offset & 3) << 1)
        #define dst1_align ((dst1_offset & 3) << 1)
        int src_idx_0  = mad24(y, src_step, src_offset - dst0_align + (x << 2));
        int src_idx_1  = mad24(y, src_step, src_offset - dst1_align + (x << 2));

        int dst0_start = mad24(y, dst0_step, dst0_offset);
        int dst0_end   = mad24(y, dst0_step, dst0_offset + dst_step1);
        int dst0_idx   = mad24(y, dst0_step, dst0_offset + (x << 1) & (int)0xfffffffc);

        int dst1_start = mad24(y, dst1_step, dst1_offset);
        int dst1_end   = mad24(y, dst1_step, dst1_offset + dst_step1);
        int dst1_idx   = mad24(y, dst1_step, dst1_offset + (x << 1) & (int)0xfffffffc);
        int src1_index_fix = src_idx_0 < 0 ? 0 : src_idx_0;
        int src2_index_fix = src_idx_1 < 0 ? 0 : src_idx_1;
        short4 src_data_0 = vload4(0, (__global short *)((__global char *)mat_src + src_idx_0));
        short4 src_data_1 = vload4(0, (__global short *)((__global char *)mat_src + src_idx_1));
        if(src_idx_0 < 0)
        {
            short4 tmp;
            tmp.xyzw = (src_idx_0 == -2) ? src_data_0.zwxy : src_data_0.yzwx;
            src_data_0.xyzw = (src_idx_0 == -1) ? src_data_0.wxyz:tmp.xyzw;
        }
        if(src_idx_1< 0)
        {
            short4 tmp;
            tmp.xyzw = ( src_idx_1== -2) ? src_data_1.zwxy : src_data_1.yzwx;
            src_data_1.xyzw = ( src_idx_1== -1) ? src_data_1.wxyz : tmp.xyzw;
        }


        short2 dst0_data  = *((__global short2 *)((__global char *)mat_dst0 + dst0_idx));
        short2 dst1_data  = *((__global short2 *)((__global char *)mat_dst1 + dst1_idx));

        short2 tmp_data0, tmp_data1;

        tmp_data0.x = ((dst0_idx + 0 >= dst0_start) && (dst0_idx + 0 < dst0_end)) ? src_data_0.x : dst0_data.x;
        tmp_data0.y = ((dst0_idx + 2 >= dst0_start) && (dst0_idx + 2 < dst0_end)) ? src_data_0.z : dst0_data.y;

        tmp_data1.x = ((dst1_idx + 0 >= dst1_start) && (dst1_idx + 0 < dst1_end)) ? src_data_1.y : dst1_data.x;
        tmp_data1.y = ((dst1_idx + 2 >= dst1_start) && (dst1_idx + 2 < dst1_end)) ? src_data_1.w : dst1_data.y;

        *((global short2 *)((__global char *)mat_dst0 + dst0_idx)) = tmp_data0;
        *((global short2 *)((__global char *)mat_dst1 + dst1_idx)) = tmp_data1;
    }
}
__kernel void split_vector_C4_D4 (__global int *mat_src,  int src_step,  int src_offset,
                                  __global int *mat_dst0, int dst0_step, int dst0_offset,
                                  __global int *mat_dst1, int dst1_step, int dst1_offset,
                                    __global int *mat_dst2, int dst2_step, int dst2_offset,
                                  __global int *mat_dst3, int dst3_step, int dst3_offset,
                                  int rows, int cols, int dst_step1)

{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if((x  < cols) && (y < rows))
    {
        int src_idx  = mad24(y, src_step,  src_offset);
        int dst0_idx = mad24(y, dst0_step, dst0_offset);
        int dst1_idx = mad24(y, dst1_step, dst1_offset);
        int dst2_idx = mad24(y, dst2_step, dst2_offset);
        int dst3_idx = mad24(y, dst3_step, dst3_offset);

        int4 src_data = ((__global int4 *)((__global char *)mat_src + src_idx))[x];

        ((__global int *)((__global char *)mat_dst0 + dst0_idx))[x] = src_data.x;
        ((__global int *)((__global char *)mat_dst1 + dst1_idx))[x] = src_data.y;
        ((__global int *)((__global char *)mat_dst2 + dst2_idx))[x] = src_data.z;
        ((__global int *)((__global char *)mat_dst3 + dst3_idx))[x] = src_data.w;
    }
}
__kernel void split_vector_C3_D4 (__global int *mat_src,  int src_step,  int src_offset,
                                  __global int *mat_dst0, int dst0_step, int dst0_offset,
                                  __global int *mat_dst1, int dst1_step, int dst1_offset,
                                    __global int *mat_dst2, int dst2_step, int dst2_offset,
                                  int rows, int cols, int dst_step1)

{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if((x  < cols) && (y < rows))
    {
        int src_idx  = mad24(y, src_step,  src_offset);
        int dst0_idx = mad24(y, dst0_step, dst0_offset);
        int dst1_idx = mad24(y, dst1_step, dst1_offset);
        int dst2_idx = mad24(y, dst2_step, dst2_offset);

        int src_data_0 = ((__global int *)((__global char *)mat_src + src_idx))[3 * x + 0];
        int src_data_1 = ((__global int *)((__global char *)mat_src + src_idx))[3 * x + 1];
        int src_data_2 = ((__global int *)((__global char *)mat_src + src_idx))[3 * x + 2];

        ((__global int *)((__global char *)mat_dst0 + dst0_idx))[x] = src_data_0;
        ((__global int *)((__global char *)mat_dst1 + dst1_idx))[x] = src_data_1;
        ((__global int *)((__global char *)mat_dst2 + dst2_idx))[x] = src_data_2;
    }
}

__kernel void split_vector_C2_D4 (__global int *mat_src,  int src_step,  int src_offset,
                                  __global int *mat_dst0, int dst0_step, int dst0_offset,
                                  __global int *mat_dst1, int dst1_step, int dst1_offset,
                                  int rows, int cols, int dst_step1)

{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if((x  < cols) && (y < rows))
    {
        int src_idx  = mad24(y, src_step,  src_offset);
        int dst0_idx = mad24(y, dst0_step, dst0_offset);
        int dst1_idx = mad24(y, dst1_step, dst1_offset);

        int2 src_data = ((__global int2 *)((__global char *)mat_src + src_idx))[x];

        ((__global int *)((__global char *)mat_dst0 + dst0_idx))[x] = src_data.x;
        ((__global int *)((__global char *)mat_dst1 + dst1_idx))[x] = src_data.y;
    }
}

__kernel void split_vector_C4_D5 (__global float *mat_src,  int src_step,  int src_offset,
                                  __global float *mat_dst0, int dst0_step, int dst0_offset,
                                  __global float *mat_dst1, int dst1_step, int dst1_offset,
                                    __global float *mat_dst2, int dst2_step, int dst2_offset,
                                  __global float *mat_dst3, int dst3_step, int dst3_offset,
                                  int rows, int cols, int dst_step1)

{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if((x  < cols) && (y < rows))
    {
        int src_idx  = mad24(y, src_step,  src_offset);
        int dst0_idx = mad24(y, dst0_step, dst0_offset);
        int dst1_idx = mad24(y, dst1_step, dst1_offset);
        int dst2_idx = mad24(y, dst2_step, dst2_offset);
        int dst3_idx = mad24(y, dst3_step, dst3_offset);

        float4 src_data = ((__global float4 *)((__global char *)mat_src + src_idx))[x];

        ((__global float *)((__global char *)mat_dst0 + dst0_idx))[x] = src_data.x;
        ((__global float *)((__global char *)mat_dst1 + dst1_idx))[x] = src_data.y;
        ((__global float *)((__global char *)mat_dst2 + dst2_idx))[x] = src_data.z;
        ((__global float *)((__global char *)mat_dst3 + dst3_idx))[x] = src_data.w;
    }
}

__kernel void split_vector_C3_D5 (__global float *mat_src,  int src_step,  int src_offset,
                                  __global float *mat_dst0, int dst0_step, int dst0_offset,
                                  __global float *mat_dst1, int dst1_step, int dst1_offset,
                                    __global float *mat_dst2, int dst2_step, int dst2_offset,
                                  int rows, int cols, int dst_step1)

{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if((x  < cols) && (y < rows))
    {
        int src_idx  = mad24(y, src_step,  src_offset);
        int dst0_idx = mad24(y, dst0_step, dst0_offset);
        int dst1_idx = mad24(y, dst1_step, dst1_offset);
        int dst2_idx = mad24(y, dst2_step, dst2_offset);

        float src_data_0 = ((__global float *)((__global char *)mat_src + src_idx))[3 * x + 0];
        float src_data_1 = ((__global float *)((__global char *)mat_src + src_idx))[3 * x + 1];
        float src_data_2 = ((__global float *)((__global char *)mat_src + src_idx))[3 * x + 2];

        ((__global float *)((__global char *)mat_dst0 + dst0_idx))[x] = src_data_0;
        ((__global float *)((__global char *)mat_dst1 + dst1_idx))[x] = src_data_1;
        ((__global float *)((__global char *)mat_dst2 + dst2_idx))[x] = src_data_2;
    }
}

__kernel void split_vector_C2_D5 (__global float *mat_src,  int src_step,  int src_offset,
                                  __global float *mat_dst0, int dst0_step, int dst0_offset,
                                  __global float *mat_dst1, int dst1_step, int dst1_offset,
                                  int rows, int cols, int dst_step1)

{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if((x  < cols) && (y < rows))
    {
        int src_idx  = mad24(y, src_step,  src_offset);
        int dst0_idx = mad24(y, dst0_step, dst0_offset);
        int dst1_idx = mad24(y, dst1_step, dst1_offset);

        float2 src_data = ((__global float2 *)((__global char *)mat_src + src_idx))[x];

        ((__global float *)((__global char *)mat_dst0 + dst0_idx))[x] = src_data.x;
        ((__global float *)((__global char *)mat_dst1 + dst1_idx))[x] = src_data.y;
    }
}

#if defined (DOUBLE_SUPPORT)
__kernel void split_vector_C4_D6 (__global double *mat_src,  int src_step,  int src_offset,
                                  __global double *mat_dst0, int dst0_step, int dst0_offset,
                                  __global double *mat_dst1, int dst1_step, int dst1_offset,
                                    __global double *mat_dst2, int dst2_step, int dst2_offset,
                                  __global double *mat_dst3, int dst3_step, int dst3_offset,
                                  int rows, int cols, int dst_step1)

{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if((x  < cols) && (y < rows))
    {
        int src_idx  = mad24(y, src_step,  src_offset);
        int dst0_idx = mad24(y, dst0_step, dst0_offset);
        int dst1_idx = mad24(y, dst1_step, dst1_offset);
        int dst2_idx = mad24(y, dst2_step, dst2_offset);
        int dst3_idx = mad24(y, dst3_step, dst3_offset);

        double4 src_data = ((__global double4 *)((__global char *)mat_src + src_idx))[x];

        ((__global double *)((__global char *)mat_dst0 + dst0_idx))[x] = src_data.x;
        ((__global double *)((__global char *)mat_dst1 + dst1_idx))[x] = src_data.y;
        ((__global double *)((__global char *)mat_dst2 + dst2_idx))[x] = src_data.z;
        ((__global double *)((__global char *)mat_dst3 + dst3_idx))[x] = src_data.w;
    }
}

__kernel void split_vector_C3_D6 (__global double *mat_src,  int src_step,  int src_offset,
                                  __global double *mat_dst0, int dst0_step, int dst0_offset,
                                  __global double *mat_dst1, int dst1_step, int dst1_offset,
                                    __global double *mat_dst2, int dst2_step, int dst2_offset,
                                  int rows, int cols, int dst_step1)

{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if((x  < cols) && (y < rows))
    {
        int src_idx  = mad24(y, src_step,  src_offset);
        int dst0_idx = mad24(y, dst0_step, dst0_offset);
        int dst1_idx = mad24(y, dst1_step, dst1_offset);
        int dst2_idx = mad24(y, dst2_step, dst2_offset);

        double src_data_0 = ((__global double *)((__global char *)mat_src + src_idx))[3 * x + 0];
        double src_data_1 = ((__global double *)((__global char *)mat_src + src_idx))[3 * x + 1];
        double src_data_2 = ((__global double *)((__global char *)mat_src + src_idx))[3 * x + 2];

        ((__global double *)((__global char *)mat_dst0 + dst0_idx))[x] = src_data_0;
        ((__global double *)((__global char *)mat_dst1 + dst1_idx))[x] = src_data_1;
        ((__global double *)((__global char *)mat_dst2 + dst2_idx))[x] = src_data_2;
    }
}

__kernel void split_vector_C2_D6 (__global double *mat_src,  int src_step,  int src_offset,
                                  __global double *mat_dst0, int dst0_step, int dst0_offset,
                                  __global double *mat_dst1, int dst1_step, int dst1_offset,
                                  int rows, int cols, int dst_step1)

{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if((x  < cols) && (y < rows))
    {
        int src_idx  = mad24(y, src_step,  src_offset);
        int dst0_idx = mad24(y, dst0_step, dst0_offset);
        int dst1_idx = mad24(y, dst1_step, dst1_offset);

        double2 src_data = ((__global double2 *)((__global char *)mat_src + src_idx))[x];

        ((__global double *)((__global char *)mat_dst0 + dst0_idx))[x] = src_data.x;
        ((__global double *)((__global char *)mat_dst1 + dst1_idx))[x] = src_data.y;
    }
}
#endif
