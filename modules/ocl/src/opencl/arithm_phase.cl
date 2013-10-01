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
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#elif defined (cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#endif
#endif

#define CV_PI 3.1415926535898
#define CV_2PI 2*3.1415926535898

/**************************************phase inradians**************************************/

__kernel void arithm_phase_inradians_D5(__global float *src1, int src1_step1, int src1_offset1,
                                         __global float *src2, int src2_step1, int src2_offset1,
                                         __global float *dst,  int dst_step1,  int dst_offset1,
                                         int cols, int rows)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step1, x + src1_offset1);
        int src2_index = mad24(y, src2_step1, x + src2_offset1);
        int dst_index  = mad24(y, dst_step1, x + dst_offset1);

        float data1 = src1[src1_index];
        float data2 = src2[src2_index];
        float tmp = atan2(data2, data1);

        if (tmp < 0)
            tmp += CV_2PI;

        dst[dst_index] = tmp;
    }
}


#if defined (DOUBLE_SUPPORT)
__kernel void arithm_phase_inradians_D6(__global double *src1, int src1_step1, int src1_offset1,
                                         __global double *src2, int src2_step1, int src2_offset1,
                                         __global double *dst,  int dst_step1,  int dst_offset1,
                                         int cols, int rows)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step1, x + src1_offset1);
        int src2_index = mad24(y, src2_step1, x + src2_offset1);
        int dst_index  = mad24(y, dst_step1, x + dst_offset1);

        double data1 = src1[src1_index];
        double data2 = src2[src2_index];
        double tmp = atan2(data2, data1);

        if (tmp < 0)
            tmp += CV_2PI;

        dst[dst_index] = tmp;
    }
}

#endif

/**************************************phase indegrees**************************************/

__kernel void arithm_phase_indegrees_D5(__global float *src1, int src1_step1, int src1_offset1,
                                         __global float *src2, int src2_step1, int src2_offset1,
                                         __global float *dst,  int dst_step1,  int dst_offset1,
                                         int cols, int rows)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step1, x + src1_offset1);
        int src2_index = mad24(y, src2_step1, x + src2_offset1);
        int dst_index  = mad24(y, dst_step1, x + dst_offset1);

        float data1 = src1[src1_index];
        float data2 = src2[src2_index];
        float tmp = atan2(data2, data1);
        tmp = 180 * tmp / CV_PI;

        if (tmp < 0)
            tmp += 360;

        dst[dst_index] = tmp;
    }
}


#if defined (DOUBLE_SUPPORT)
__kernel void arithm_phase_indegrees_D6 (__global double *src1, int src1_step1, int src1_offset1,
                                         __global double *src2, int src2_step1, int src2_offset1,
                                         __global double *dst,  int dst_step1,  int dst_offset1,
                                         int cols, int rows)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step1, x + src1_offset1);
        int src2_index = mad24(y, src2_step1, x + src2_offset1);
        int dst_index  = mad24(y, dst_step1, x + dst_offset1);

        double data1 = src1[src1_index];
        double data2 = src2[src2_index];
        double tmp = atan2(src2[src2_index], src1[src1_index]);

        tmp = 180 * tmp / CV_PI;
        if (tmp < 0)
            tmp += 360;

        dst[dst_index] = tmp;
    }
}
#endif
