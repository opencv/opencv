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

#define CONCAT(A,B) A##_##B
#define TEMPLATE(name,type) CONCAT(name,type)

#if defined(cl_intel_subgroups)
#pragma OPENCL EXTENSION  cl_intel_subgroups : enable
#endif

#if defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__kernel void TEMPLATE(softmax_forward_slm,Dtype)(const int num, const int channels,
                                   const int spatial_dim,
                                   __global Dtype* scale,
                                   __global const Dtype* data,
                                   __global Dtype* out,
                                   __local Dtype *out_tmp,
                                   __local Dtype *scale_tmp,
                                   __local Dtype *group_tmp) {

  int n = get_global_id(1);
  for (int index = get_global_id(0), s = 0; index < spatial_dim * get_local_size(0); index +=
      get_global_size(0), ++s) {
    Dtype maxval = -DTYPE_MAX;
    for (int c = get_global_id(0); c < channels; c += get_global_size(0)) {
      Dtype tmp = data[(n * channels + c) * spatial_dim + s];
      maxval = max((Dtype)tmp, (Dtype)maxval);
    }
    maxval = sub_group_reduce_max(maxval);
    //if (get_sub_group_local_id() == 0)
    group_tmp[get_sub_group_id() * spatial_dim + s] = maxval;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int index = get_global_id(0); index < spatial_dim * get_max_sub_group_size(); index +=
      get_global_size(0)) {
    int s = index / get_max_sub_group_size();
    Dtype maxval = sub_group_reduce_max(group_tmp[get_sub_group_local_id() * spatial_dim + s]);
    //if (get_sub_group_local_id() == 0)
    scale_tmp[s] = maxval;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int index = get_global_id(0); index < channels * spatial_dim;
      index += get_global_size(0)) {
    int s = index % spatial_dim;
#ifdef LOG_SOFTMAX
    out_tmp[index] = data[n * channels * spatial_dim + index] - scale_tmp[s];
#else
    out_tmp[index] = exp(data[n * channels * spatial_dim + index] - scale_tmp[s]);
#endif
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int index = get_global_id(0), s = 0; index < spatial_dim * get_local_size(0); index +=
      get_global_size(0), ++s) {
    Dtype sum = 0;
    for (int c = get_global_id(0); c < channels; c += get_global_size(0)) {
#ifdef LOG_SOFTMAX
      sum += exp(out_tmp[c * spatial_dim + s]);
#else
      sum += out_tmp[c * spatial_dim + s];
#endif
    }
    sum = sub_group_reduce_add(sum);
    group_tmp[get_sub_group_id() * spatial_dim + s] = sum;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int index = get_global_id(0); index < spatial_dim * get_max_sub_group_size(); index +=
      get_global_size(0)) {
    int s = index / get_max_sub_group_size();
    Dtype sum = sub_group_reduce_add(group_tmp[get_sub_group_local_id() * spatial_dim + s]);
    //if (get_sub_group_local_id() == 0)
    scale_tmp[s] = sum;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int index = get_global_id(0); index < channels * spatial_dim;
      index += get_global_size(0)) {
    int s = index % spatial_dim;
#ifdef LOG_SOFTMAX
    Dtype v = out_tmp[index] - log(scale_tmp[s]);
#else
    Dtype v = out_tmp[index] / scale_tmp[s];
#endif
    out[n * channels * spatial_dim + index] = v;
  }
}

__kernel void TEMPLATE(softmax_forward,Dtype)(const int num, const int channels,
                                   const int spatial_dim,
                                   __global Dtype* scale,
                                   __global const Dtype* data,
                                   __global Dtype* out) {

  int n = get_global_id(1);
  __global Dtype *group_tmp = scale + spatial_dim * num + n * get_max_sub_group_size() * spatial_dim;
  for (int index = get_global_id(0), s = 0; index < spatial_dim * get_local_size(0); index +=
      get_global_size(0), ++s) {
    Dtype maxval = -DTYPE_MAX;
    for (int c = get_global_id(0); c < channels; c += get_global_size(0)) {
      Dtype tmp = data[(n * channels + c) * spatial_dim + s];
      maxval = max((Dtype)tmp, (Dtype)maxval);
    }
    maxval = sub_group_reduce_max(maxval);
    //if (get_sub_group_local_id() == 0)
    group_tmp[get_sub_group_id() * spatial_dim + s] = maxval;
  }
  barrier(CLK_GLOBAL_MEM_FENCE);

  for (int index = get_global_id(0); index < spatial_dim * get_max_sub_group_size(); index +=
      get_global_size(0)) {
    int s = index / get_max_sub_group_size();
    Dtype maxval = sub_group_reduce_max(group_tmp[get_sub_group_local_id() * spatial_dim + s]);
    //if (get_sub_group_local_id() == 0)
    scale[n * spatial_dim + s] = maxval;
  }

  barrier(CLK_GLOBAL_MEM_FENCE);

  for (int index = get_global_id(0); index < channels * spatial_dim;
      index += get_global_size(0)) {
    int s = index % spatial_dim;
#ifdef LOG_SOFTMAX
    out[n * channels * spatial_dim + index] = data[n * channels * spatial_dim + index] - scale[n * spatial_dim + s];
#else
    out[n * channels * spatial_dim + index] = exp(data[n * channels * spatial_dim + index] - scale[n * spatial_dim + s]);
#endif
  }
  barrier(CLK_GLOBAL_MEM_FENCE);

  for (int index = get_global_id(0), s = 0; index < spatial_dim * get_local_size(0); index +=
      get_global_size(0), ++s) {
    Dtype sum = 0;
    for (int c = get_global_id(0); c < channels; c += get_global_size(0)) {
#ifdef LOG_SOFTMAX
      sum += exp(out[n * channels * spatial_dim + c * spatial_dim + s]);
#else
      sum += out[n * channels * spatial_dim + c * spatial_dim + s];
#endif
    }
    sum = sub_group_reduce_add(sum);
    group_tmp[get_sub_group_id() * spatial_dim + s] = sum;
  }
  barrier(CLK_GLOBAL_MEM_FENCE);

  for (int index = get_global_id(0); index < spatial_dim * get_max_sub_group_size(); index +=
      get_global_size(0)) {
    int s = index / get_max_sub_group_size();
    Dtype sum = sub_group_reduce_add(group_tmp[get_sub_group_local_id() * spatial_dim + s]);
    //if (get_sub_group_local_id() == 0)
    scale[n * spatial_dim + s] = sum;
  }
  barrier(CLK_GLOBAL_MEM_FENCE);

  for (int index = get_global_id(0); index < channels * spatial_dim;
      index += get_global_size(0)) {
    int s = index % spatial_dim;
#ifdef LOG_SOFTMAX
    Dtype v = out[n * channels * spatial_dim + index] - log(scale[n * spatial_dim + s]);
#else
    Dtype v = out[n * channels * spatial_dim + index] / scale[n * spatial_dim + s];
#endif
    out[n * channels * spatial_dim + index] = v;
  }
}
