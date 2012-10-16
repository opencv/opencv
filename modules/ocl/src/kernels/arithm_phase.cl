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
//    Jia Haipeng, jiahaipeng95@gmail.com
//
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
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#define CV_PI 3.1415926535898
/**************************************phase inradians**************************************/
__kernel void arithm_phase_inradians_D5 (__global float *src1, int src1_step, int src1_offset,
                                         __global float *src2, int src2_step, int src2_offset,
                                         __global float *dst,  int dst_step,  int dst_offset,
                                         int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 2) + src1_offset);
        int src2_index = mad24(y, src2_step, (x << 2) + src2_offset);
        int dst_index  = mad24(y, dst_step,  (x << 2) + dst_offset);

        float data1 = *((__global float *)((__global char *)src1 + src1_index));
        float data2 = *((__global float *)((__global char *)src2 + src2_index));
        float tmp = atan2(data2,data1);
        
        *((__global float *)((__global char *)dst + dst_index)) = tmp;
    }

}


#if defined (DOUBLE_SUPPORT)
__kernel void arithm_phase_inradians_D6 (__global double *src1, int src1_step, int src1_offset,
                                         __global double *src2, int src2_step, int src2_offset,
                                         __global double *dst,  int dst_step,  int dst_offset,
                                         int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 3) + src1_offset);
        int src2_index = mad24(y, src2_step, (x << 3) + src2_offset);
        int dst_index  = mad24(y, dst_step,  (x << 3) + dst_offset);

        double data1 = *((__global double *)((__global char *)src1 + src1_index));
        double data2 = *((__global double *)((__global char *)src2 + src2_index));
        
        *((__global double *)((__global char *)dst + dst_index)) = atan2(data2,data1);
    }

}
#endif

/**************************************phase indegrees**************************************/
__kernel void arithm_phase_indegrees_D5 (__global float *src1, int src1_step, int src1_offset,
                                         __global float *src2, int src2_step, int src2_offset,
                                         __global float *dst,  int dst_step,  int dst_offset,
                                         int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 2) + src1_offset);
        int src2_index = mad24(y, src2_step, (x << 2) + src2_offset);
        int dst_index  = mad24(y, dst_step,  (x << 2) + dst_offset);

        float data1 = *((__global float *)((__global char *)src1 + src1_index));
        float data2 = *((__global float *)((__global char *)src2 + src2_index));
        float tmp = atan2(data2,data1);
        float tmp_data = 180*tmp/CV_PI;
        
        *((__global float *)((__global char *)dst + dst_index)) = tmp_data;
    }

}


#if defined (DOUBLE_SUPPORT)
__kernel void arithm_phase_indegrees_D6 (__global double *src1, int src1_step, int src1_offset,
                                         __global double *src2, int src2_step, int src2_offset,
                                         __global double *dst,  int dst_step,  int dst_offset,
                                         int rows, int cols, int dst_step1)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, (x << 3) + src1_offset);
        int src2_index = mad24(y, src2_step, (x << 3) + src2_offset);
        int dst_index  = mad24(y, dst_step,  (x << 3) + dst_offset);

        double data1 = *((__global double *)((__global char *)src1 + src1_index));
        double data2 = *((__global double *)((__global char *)src2 + src2_index));
        double tmp = atan2(data2,data1);
        double tmp_data = 180*tmp/CV_PI;
        
        *((__global double *)((__global char *)dst + dst_index)) = tmp_data;
    }

}
#endif
