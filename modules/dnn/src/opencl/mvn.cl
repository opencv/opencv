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

#define Dtype  float
#define Dtype4 float4
#define Dtype8 float8

#if NUM == 8
    #define load(src, index) vload8(0, src + index)
    #define store(vec, dst, index) vstore8(vec, 0, dst + index)
    #define vec_type Dtype8
    #define CALC_MEAN calc_mean8
    #define MVN mvn8
    #define MVN_GROUP mvn_group8
    #define MEAN_FUSE mean_fuse8
    #define MVN_FUSE mvn_fuse8
#elif NUM == 4
    #define load(src, index) vload4(0, src + index)
    #define store(vec, dst, index) vstore4(vec, 0, dst + index)
    #define vec_type Dtype4
    #define CALC_MEAN calc_mean4
    #define MVN mvn4
    #define MVN_GROUP mvn_group4
    #define MEAN_FUSE mean_fuse4
    #define MVN_FUSE mvn_fuse4
#elif NUM == 1
    #define load(src, index) src[index]
    #define store(vec, dst, index) dst[index] = vec
    #define vec_type Dtype
    #define CALC_MEAN calc_mean1
    #define MVN mvn1
    #define MVN_GROUP mvn_group1
    #define MEAN_FUSE mean_fuse1
    #define MVN_FUSE mvn_fuse1
#endif

#ifdef KERNEL_MEAN

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
    vec_type dst_vec = src_vec - (vec_type)mean_val;
    dst_vec = dst_vec * dst_vec;
    store(dst_vec, dst, index);
}

#elif defined KERNEL_MVN

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
    Dtype dev_val = dev[x];
    Dtype alpha;
#ifdef NORM_VARIANCE
    alpha = 1 / sqrt(eps + dev_val);
#else
    alpha = 1;
#endif

#ifdef LAYER_NORM
    vec_type w = load(bnorm_weight, y), b = load(bnorm_bias, y);
#else

    Dtype w = 1.f, b = 0.f;
#ifdef FUSE_BATCH_NORM
    w = bnorm_weight[x % channels];
    b = bnorm_bias[x % channels];
#endif

#endif // LAYER_NORM

    vec_type src_vec = load(src, index) - (vec_type)mean_val;
    vec_type dst_vec = src_vec * alpha;
    dst_vec = dst_vec * w + (vec_type)b;

#ifdef FUSE_RELU
    vec_type new_val = dst_vec * relu_slope;
    dst_vec = select(new_val, dst_vec, dst_vec > (vec_type)0.f);
#endif

    store(dst_vec, dst, index);
}

#elif defined KERNEL_MVN_GROUP

__kernel void MVN_GROUP(__global const Dtype* src,
                            const int rows,
                            const int cols,
                            const Dtype eps,
                            __global const Dtype* mean,
                            __global const Dtype* dev,
                            __global const Dtype* weight,
                            __global const Dtype* bias,
                            const int channels,
                            const int num_groups,
                            const float relu_slope,
                            __global Dtype* dst)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * NUM;
    int index = x * cols + y;

    if (x >= rows || y >= cols)
        return;

    int group_size = channels / num_groups;
    int step = norm_size / group_size;
    int channel_index = x % num_groups * group_size + y / step
    Dtype mean_val = mean[x];
    Dtype dev_val = dev[x];
    Dtype alpha;
#ifdef NORM_VARIANCE
    alpha = 1 / sqrt(eps + dev_val);
#else
    alpha = 1;
#endif

    Dtype w = weight[channel_index], b = bias[channel_index];

    vec_type src_vec = load(src, index) - (vec_type)mean_val;
    vec_type dst_vec = src_vec * alpha;
    dst_vec = dst_vec * w + (vec_type)b;

#ifdef FUSE_RELU
    vec_type new_val = dst_vec * relu_slope;
    dst_vec = select(new_val, dst_vec, dst_vec > (vec_type)0.f);
#endif

    store(dst_vec, dst, index);
}

#elif defined KERNEL_MEAN_FUSE

__kernel void MEAN_FUSE(__global const T * A,
                        unsigned int A_col_size,
                        float alpha,
                        __global T4 * mean,
                        __global Dtype * tmp)
{
    unsigned int row_gid = get_group_id(0);
    unsigned int lid = get_local_id(0);
    const __global T *src0_read = A + row_gid * 4 * A_col_size;
    __global Dtype *dst0_read = tmp + row_gid * 4 * A_col_size;
    Dtype4 dot0, dot1, dot2, dot3;
    dot0 = dot1 = dot2 = dot3 = (Dtype4)(0.f);

    unsigned int i = lid;
    const Dtype4 b0 = (Dtype4)1.f;
    while( i < A_col_size / 4)
    {
        const T4 a0 = vload4(i, src0_read);
        const T4 a1 = vload4(i, src0_read + A_col_size);
        const T4 a2 = vload4(i, src0_read + 2 * A_col_size);
        const T4 a3 = vload4(i, src0_read + 3 * A_col_size);

        dot0 += convert_float4(a0);
        dot1 += convert_float4(a1);
        dot2 += convert_float4(a2);
        dot3 += convert_float4(a3);

        i += LOCAL_SIZE;
    }

    __local Dtype4 work[LOCAL_SIZE];
    work[lid].s0 = dot(dot0, b0);
    work[lid].s1 = dot(dot1, b0);
    work[lid].s2 = dot(dot2, b0);
    work[lid].s3 = dot(dot3, b0);

    for(unsigned int stride=LOCAL_SIZE/2 ; stride>0 ; stride>>=1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if(lid < stride)
            work[lid] += work[lid+stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(lid == 0)
    {
        mean[row_gid] = convert_T(alpha * work[0]);
    }

    Dtype4 sum = work[0] * alpha;
    i = lid;
    while( i < A_col_size / 4)
    {
        const T4 a0 = vload4(i, src0_read);
        const T4 a1 = vload4(i, src0_read + A_col_size);
        const T4 a2 = vload4(i, src0_read + 2 * A_col_size);
        const T4 a3 = vload4(i, src0_read + 3 * A_col_size);

        dot0 = convert_float4(a0) - (Dtype4)sum.x;
        dot1 = convert_float4(a1) - (Dtype4)sum.y;
        dot2 = convert_float4(a2) - (Dtype4)sum.z;
        dot3 = convert_float4(a3) - (Dtype4)sum.w;
        dot0 = dot0 * dot0;
        dot1 = dot1 * dot1;
        dot2 = dot2 * dot2;
        dot3 = dot3 * dot3;

        vstore4(dot0, i, dst0_read);
        vstore4(dot1, i, dst0_read + A_col_size);
        vstore4(dot2, i, dst0_read + 2 * A_col_size);
        vstore4(dot3, i, dst0_read + 3 * A_col_size);

        i += LOCAL_SIZE;
    }
}

#elif defined KERNEL_MVN_FUSE

__kernel void MVN_FUSE(__global const Dtype * tmp,
                       __global const T * A,
                       __global const T4 * mean,
                       unsigned int A_col_size,
                       const float alpha_val,
                       const float eps,
                       const float relu_slope,
                       __global const Dtype4 * bnorm_weight,
                       __global const Dtype4 * bnorm_bias,
                       __global T * B)
{
    unsigned int row_gid = get_group_id(0);
    unsigned int lid = get_local_id(0);
    const __global Dtype *src0_read = tmp + row_gid * 4 * A_col_size;
    const __global T *src1_read = A + row_gid * 4 * A_col_size;
    __global T *dst0_read = B + row_gid * 4 * A_col_size;
    Dtype4 dot0, dot1, dot2, dot3;
    dot0 = dot1 = dot2 = dot3 = (Dtype4)(0.f);

    unsigned int i = lid;
    const Dtype4 b0 = (Dtype4)1.f;
    while( i < A_col_size / 4)
    {
        const Dtype4 a0 = vload4(i, src0_read);
        const Dtype4 a1 = vload4(i, src0_read + A_col_size);
        const Dtype4 a2 = vload4(i, src0_read + 2 * A_col_size);
        const Dtype4 a3 = vload4(i, src0_read + 3 * A_col_size);

        dot0 += a0;
        dot1 += a1;
        dot2 += a2;
        dot3 += a3;

        i += LOCAL_SIZE;
    }

    __local Dtype4 work[LOCAL_SIZE];
    work[lid].s0 = dot(dot0, b0);
    work[lid].s1 = dot(dot1, b0);
    work[lid].s2 = dot(dot2, b0);
    work[lid].s3 = dot(dot3, b0);

    for(unsigned int stride=LOCAL_SIZE/2 ; stride>0 ; stride>>=1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if(lid < stride)
            work[lid] += work[lid+stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    Dtype4 mean_val = convert_float4(mean[row_gid]);
    Dtype4 dev_val = sqrt(work[0] * alpha_val + (Dtype4)eps);
    Dtype4 alpha = (Dtype4)1.f / dev_val;

    Dtype4 w = (Dtype4)1.f;
    Dtype4 b = (Dtype4)0.f;
#ifdef FUSE_BATCH_NORM
    w = bnorm_weight[row_gid];
    b = bnorm_bias[row_gid];
#endif

    i = lid;
    while( i < A_col_size / 4)
    {
        const T4 a0 = vload4(i, src1_read);
        const T4 a1 = vload4(i, src1_read + A_col_size);
        const T4 a2 = vload4(i, src1_read + 2 * A_col_size);
        const T4 a3 = vload4(i, src1_read + 3 * A_col_size);

        dot0 = (convert_float4(a0) - (Dtype4)mean_val.x) * alpha.x;
        dot1 = (convert_float4(a1) - (Dtype4)mean_val.y) * alpha.y;
        dot2 = (convert_float4(a2) - (Dtype4)mean_val.z) * alpha.z;
        dot3 = (convert_float4(a3) - (Dtype4)mean_val.w) * alpha.w;

        dot0 = dot0 * w.x + (Dtype4)b.x;
        dot1 = dot1 * w.y + (Dtype4)b.y;
        dot2 = dot2 * w.z + (Dtype4)b.z;
        dot3 = dot3 * w.w + (Dtype4)b.w;

#ifdef FUSE_RELU
        Dtype4 new0 = dot0 * relu_slope;
        dot0 = select(new0, dot0, dot0 > (Dtype4)0.f);

        Dtype4 new1 = dot1 * relu_slope;
        dot1 = select(new1, dot1, dot1 > (Dtype4)0.f);

        Dtype4 new2 = dot2 * relu_slope;
        dot2 = select(new2, dot2, dot2 > (Dtype4)0.f);

        Dtype4 new3 = dot3 * relu_slope;
        dot3 = select(new3, dot3, dot3 > (Dtype4)0.f);
#endif

        vstore4(convert_T(dot0), i, dst0_read);
        vstore4(convert_T(dot1), i, dst0_read + A_col_size);
        vstore4(convert_T(dot2), i, dst0_read + 2 * A_col_size);
        vstore4(convert_T(dot3), i, dst0_read + 3 * A_col_size);

        i += LOCAL_SIZE;
    }
}

#else
#error "Configuration error!"
#endif
