
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
// the use of this softwareif advised of the possibility of such damage.
//
//M*/

#if defined (DOUBLE_SUPPORT)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////magnitudeSqr//////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void magnitudeSqr_C1_D5 (__global float *src1,int src1_step,int src1_offset,
                           __global float *src2, int src2_step,int src2_offset,
                           __global float *dst,  int dst_step,int dst_offset,
                           int rows,  int cols,int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)


    {

        x = x << 2;

        #define dst_align ((dst_offset >> 2) & 3)

        int src1_index = mad24(y, src1_step, (x << 2) + src1_offset - (dst_align << 2));
        int src2_index = mad24(y, src2_step, (x << 2) + src2_offset - (dst_align << 2));

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + (x << 2) -(dst_align << 2));

        float4 src1_data = vload4(0, (__global float  *)((__global char *)src1 + src1_index));
        float4 src2_data = vload4(0, (__global float *)((__global char *)src2 + src2_index));
        float4 dst_data = *((__global float4 *)((__global char *)dst + dst_index));

        float4   tmp_data  ;
      tmp_data.x = src1_data.x * src1_data.x + src2_data.x * src2_data.x;

      tmp_data.y = src1_data.y * src1_data.y + src2_data.y * src2_data.y;

      tmp_data.z = src1_data.z * src1_data.z + src2_data.z * src2_data.z;

      tmp_data.w = src1_data.w * src1_data.w + src2_data.w * src2_data.w;




        dst_data.x = ((dst_index + 0 >= dst_start) && (dst_index + 0 < dst_end)) ? tmp_data.x : dst_data.x;
        dst_data.y = ((dst_index + 4 >= dst_start) && (dst_index + 4 < dst_end)) ? tmp_data.y : dst_data.y;
        dst_data.z = ((dst_index + 8 >= dst_start) && (dst_index + 8 < dst_end)) ? tmp_data.z : dst_data.z;
        dst_data.w = ((dst_index + 12 >= dst_start) && (dst_index + 12 < dst_end)) ? tmp_data.w : dst_data.w;

        *((__global float4 *)((__global char *)dst + dst_index)) = dst_data;
    }

}


#if defined (DOUBLE_SUPPORT)

__kernel void magnitudeSqr_C2_D5 (__global float *src1,int src1_step,int src1_offset,
                           __global float *dst,  int dst_step,int dst_offset,
                           int rows,  int cols,int dst_step1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)


    {

        x = x << 2;

        #define dst_align ((dst_offset >> 2) & 3)

        int src1_index = mad24(y, src1_step, (x << 3) + src1_offset - (dst_align << 3));

        int dst_start  = mad24(y, dst_step, dst_offset);
        int dst_end    = mad24(y, dst_step, dst_offset + dst_step1);
        int dst_index  = mad24(y, dst_step, dst_offset + (x << 2) -(dst_align << 2));

        float8 src1_data = vload8(0, (__global float  *)((__global char *)src1 + src1_index));
        float4 dst_data = *((__global float4 *)((__global char *)dst + dst_index));

        float4   tmp_data  ;
      tmp_data.x = src1_data.s0 * src1_data.s0 + src1_data.s1 * src1_data.s1;

      tmp_data.y = src1_data.s2 * src1_data.s2 + src1_data.s3 * src1_data.s3;

      tmp_data.z = src1_data.s4 * src1_data.s4 + src1_data.s5 * src1_data.s5;

      tmp_data.w = src1_data.s6 * src1_data.s6 + src1_data.s7 * src1_data.s7;




        dst_data.x = ((dst_index + 0 >= dst_start) && (dst_index + 0 < dst_end)) ? tmp_data.x : dst_data.x;
        dst_data.y = ((dst_index + 4 >= dst_start) && (dst_index + 4 < dst_end)) ? tmp_data.y : dst_data.y;
        dst_data.z = ((dst_index + 8 >= dst_start) && (dst_index + 8 < dst_end)) ? tmp_data.z : dst_data.z;
        dst_data.w = ((dst_index + 12 >= dst_start) && (dst_index + 12 < dst_end)) ? tmp_data.w : dst_data.w;

        *((__global float4 *)((__global char *)dst + dst_index)) = dst_data;
    }

}
#endif
