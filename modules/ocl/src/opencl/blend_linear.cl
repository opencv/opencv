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
// Copyright (C) 2010-2012, MulticoreWare Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Liu Liujun, liujun@multicorewareinc.com
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

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

__kernel void blendLinear(__global const T * src1, int src1_offset, int src1_step,
                          __global const T * src2, int src2_offset, int src2_step,
                          __global const float * weight1, int weight1_offset, int weight1_step,
                          __global const float * weight2, int weight2_offset, int weight2_step,
                          __global T * dst, int dst_offset, int dst_step,
                          int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, src1_offset + x);
        int src2_index = mad24(y, src2_step, src2_offset + x);
        int weight1_index = mad24(y, weight1_step, weight1_offset + x);
        int weight2_index = mad24(y, weight2_step, weight2_offset + x);
        int dst_index = mad24(y, dst_step, dst_offset + x);

        FT w1 = (FT)(weight1[weight1_index]), w2 = (FT)(weight2[weight2_index]);
        FT den = w1 + w2 + (FT)(1e-5f);
        FT num = w1 * convertToFT(src1[src1_index]) + w2 * convertToFT(src2[src2_index]);

        dst[dst_index] = convertToT(num / den);
    }
}
