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
#define Dtype float

#if defined(cl_intel_subgroups)
#pragma OPENCL EXTENSION  cl_intel_subgroups : enable
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
    float maxval = -FLT_MAX;
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
    out_tmp[index] = exp(data[n * channels * spatial_dim + index] - scale_tmp[s]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int index = get_global_id(0), s = 0; index < spatial_dim * get_local_size(0); index +=
      get_global_size(0), ++s) {
    Dtype sum = 0;
    for (int c = get_global_id(0); c < channels; c += get_global_size(0)) {
      sum += out_tmp[c * spatial_dim + s];
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
    out[n * channels * spatial_dim + index] = out_tmp[index] / scale_tmp[s];
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
    float maxval = -FLT_MAX;
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
    out[n * channels * spatial_dim + index] = exp(data[n * channels * spatial_dim + index] - scale[n * spatial_dim + s]);
  }
  barrier(CLK_GLOBAL_MEM_FENCE);

  for (int index = get_global_id(0), s = 0; index < spatial_dim * get_local_size(0); index +=
      get_global_size(0), ++s) {
    Dtype sum = 0;
    for (int c = get_global_id(0); c < channels; c += get_global_size(0)) {
      sum += out[n * channels * spatial_dim + c * spatial_dim + s];
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
    out[n * channels * spatial_dim + index] /= scale[n * spatial_dim + s];
  }
}

// Copied from caffe.pb.h, must keep consistent with the original definition
#ifndef __SOFTMAX_LOSS_CL__
#define __SOFTMAX_LOSS_CL__
enum LossParameter_NormalizationMode {
  LossParameter_NormalizationMode_FULL = 0,
  LossParameter_NormalizationMode_VALID = 1,
  LossParameter_NormalizationMode_BATCH_SIZE = 2,
  LossParameter_NormalizationMode_NONE = 3
};
#endif
// Copied from softmax_loss_layer.cpp, must keep consistent with the original implementation
Dtype TEMPLATE(get_normalizer, Dtype)(
    enum LossParameter_NormalizationMode normalization_mode, int valid_count,
    int outer_num_, int inner_num_) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = (Dtype)(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = (Dtype)(outer_num_ * inner_num_);
      } else {
        normalizer = (Dtype)(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = (Dtype)(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = (Dtype)(1);
      break;
    default:
      normalizer = (Dtype)(0);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return fmax((Dtype)(1.0), normalizer);
}

Dtype TEMPLATE(asum, Dtype)(int n, __global const Dtype *data, __local Dtype *sum_tmp) {
  Dtype sum = 0;
  for(int i = get_global_id(0); i < n; i += get_global_size(0)) {
    sum += data[i];
  }
  sum = sub_group_reduce_add(sum);
  sum_tmp[get_sub_group_id()] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (get_sub_group_id() == 0)
    sum = sub_group_reduce_add(sum_tmp[get_sub_group_local_id()]);
  return sum;
}

__kernel void TEMPLATE(softmax_loss_forward_asum, Dtype)(
    int n, int outer_num_, int inner_num_,
    int compute_count_sum, int normalization_type,
    __global const Dtype *loss,
    __global const Dtype *counts, __global Dtype *out) {
    __local Dtype sum_tmp[16];

    Dtype loss_sum = TEMPLATE(asum, Dtype)(n, loss, sum_tmp);
    Dtype counts_sum = -1;
    if (compute_count_sum)
      counts_sum = TEMPLATE(asum, Dtype)(n, counts, sum_tmp);

    if (get_global_id(0) == 0)
      out[0] = loss_sum / TEMPLATE(get_normalizer, Dtype)(normalization_type, counts_sum, outer_num_, inner_num_);
}

__kernel void TEMPLATE(softmax_loss_forward,Dtype)(
    int n, __global const Dtype* prob_data, __global const Dtype* label,
    __global Dtype* loss,
    const int num, const int dim, const int spatial_dim,
    const int has_ignore_label_, const int ignore_label_,
    __global Dtype* counts) {

  for (int index = get_global_id(0); index < n;
      index += get_global_size(0)) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = (int) (label[n * spatial_dim + s]);
    if (has_ignore_label_ == 1 && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      loss[index] = -log((Dtype)(
          max((Dtype) (prob_data[n * dim + label_value * spatial_dim + s]),
              (Dtype) FLT_MIN)));
      counts[index] = 1;
    }
  }
}

__kernel void TEMPLATE(softmax_loss_backward,Dtype)(const int nthreads,
                                                    __global const Dtype* top,
                                                    __global const Dtype* label,
                                                    __global Dtype* bottom_diff,
                                                    const int num,
                                                    const int dim,
                                                    const int spatial_dim,
                                                    const int has_ignore_label_,
                                                    const int ignore_label_,
                                                    __global Dtype* counts) {
  const int channels = dim / spatial_dim;
  for (int index = get_global_id(0); index < nthreads; index +=
      get_global_size(0)) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = (int) (label[n * spatial_dim + s]);
    if (has_ignore_label_ == 1 && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
      counts[index] = 1;
    }
  }
}
