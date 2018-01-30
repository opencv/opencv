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

#define Dtype float
#define Dtype4 float4
#define Dtype8 float8

#if NUM == 8
    #define load(src, index) vload8(0, src + index)
    #define store(vec, dst, index) vstore8(vec, 0, dst + index)
    #define vec_type Dtype8
    #define CALC_MEAN calc_mean8
    #define MVN mvn8
#elif NUM == 4
    #define load(src, index) vload4(0, src + index)
    #define store(vec, dst, index) vstore4(vec, 0, dst + index)
    #define vec_type Dtype4
    #define CALC_MEAN calc_mean4
    #define MVN mvn4
#elif NUM == 1
    #define load(src, index) src[index]
    #define store(vec, dst, index) dst[index] = vec
    #define vec_type Dtype
    #define CALC_MEAN calc_mean1
    #define MVN mvn1
#endif

__kernel void CALC_MEAN(__global const Dtype* src,
                        const int rows,
                        const int cols,
                        __global Dtype* mean,
                        __global Dtype* dst)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * NUM;
    int index = x * cols + y;

    if (x >= rows || y >= cols)
        return;

    Dtype mean_val = mean[x];
    vec_type src_vec = load(src, index);
    vec_type dst_vec = native_powr(src_vec - (vec_type)mean_val, 2);
    store(dst_vec, dst, index);
}

__kernel void MVN(__global const Dtype* src,
                  const int rows,
                  const int cols,
                  const Dtype eps,
                  __global const Dtype* mean,
                  __global const Dtype* dev,
                  __global const Dtype* bnorm_weight,
                  __global const Dtype* bnorm_bias,
                  const int channels,
                  const float relu_slope,
                  __global Dtype* dst)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * NUM;
    int index = x * cols + y;

    if (x >= rows || y >= cols)
        return;

    Dtype mean_val = mean[x];
    Dtype dev_val = sqrt(dev[x]);
    Dtype alpha;
#ifdef NORM_VARIANCE
    alpha = 1 / (eps + dev_val);
#else
    alpha = 1;
#endif

    Dtype w = 1.f, b = 0.f;
#ifdef FUSE_BATCH_NORM
    w = bnorm_weight[x % channels];
    b = bnorm_bias[x % channels];
#endif

    vec_type src_vec = load(src, index) - (vec_type)mean_val;
    vec_type dst_vec = src_vec * alpha;
    dst_vec = dst_vec * w + (vec_type)b;

#ifdef FUSE_RELU
    vec_type new_val = dst_vec * relu_slope;
    dst_vec = select(new_val, dst_vec, dst_vec > (vec_type)0.f);
#endif

    store(dst_vec, dst, index);
}
