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

#define noconvert

__kernel void blendLinear(__global const uchar * src1ptr, int src1_step, int src1_offset,
                          __global const uchar * src2ptr, int src2_step, int src2_offset,
                          __global const uchar * weight1, int weight1_step, int weight1_offset,
                          __global const uchar * weight2, int weight2_step, int weight2_offset,
                          __global uchar * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < dst_cols && y < dst_rows)
    {
        int src1_index = mad24(y, src1_step, src1_offset + x * cn * (int)sizeof(T));
        int src2_index = mad24(y, src2_step, src2_offset + x * cn * (int)sizeof(T));
        int weight1_index = mad24(y, weight1_step, weight1_offset + x * (int)sizeof(float));
        int weight2_index = mad24(y, weight2_step, weight2_offset + x * (int)sizeof(float));
        int dst_index = mad24(y, dst_step, dst_offset + x * cn * (int)sizeof(T));

        float w1 = *(__global const float *)(weight1 + weight1_index),
              w2 = *(__global const float *)(weight2 + weight2_index);
        float den = w1 + w2 + 1e-5f;

        __global const T * src1 = (__global const T *)(src1ptr + src1_index);
        __global const T * src2 = (__global const T *)(src2ptr + src2_index);
        __global T * dst = (__global T *)(dstptr + dst_index);

        #pragma unroll
        for (int i = 0; i < cn; ++i)
        {
            float num = w1 * convert_float(src1[i]) + w2 * convert_float(src2[i]);
            dst[i] = convertToT(num / den);
        }
    }
}
