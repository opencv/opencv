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
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Copyright (c) 2016-2017 Fabian David Tschopp, all rights reserved.
// Third party copyrights are property of their respective owners.
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
// This software is provided by the copyright holders and contributors "as is" and
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

#if defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if NUM == 8
    #define load(src, index) vload8(0, src + index)
    #define store(vec, dst, index) vstore8(vec, 0, dst + index)
    #define float_type float8
    #define convert_f convert_float8
    #define BATCH_NORM batch_norm8
#elif NUM == 4
    #define load(src, index) vload4(0, src + index)
    #define store(vec, dst, index) vstore4(vec, 0, dst + index)
    #define float_type float4
    #define convert_f convert_float4
    #define BATCH_NORM batch_norm4
#elif NUM == 1
    #define load(src, index) src[index]
    #define store(vec, dst, index) dst[index] = vec
    #define float_type float
    #define convert_f convert_float
    #define BATCH_NORM batch_norm1
#endif

__kernel void BATCH_NORM(__global const Dtype* src,
                         const int rows,
                         const int cols,
                         const int channels,
                         __global const float* weight,
                         __global const float* bias,
                         __global Dtype* dst)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * NUM;
    int index = x * cols + y;

    if (x >= rows || y >= cols)
        return;

    float w = weight[x % channels];
    float b = bias[x % channels];
    float_type src_vec = convert_f(load(src, index));
    float_type dst_vec = src_vec * w + (float_type)b;
    store(convert_T(dst_vec), dst, index);
}
