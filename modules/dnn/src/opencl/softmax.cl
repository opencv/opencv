/*************************************************************************************
 * Copyright (c) 2015, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation and/or
 *  other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************************/

__kernel void kernel_channel_max(const int num, const int channels,
    const int spatial_dim, __global const T* data, __global T* out) {
  int index = get_global_id(0);
  if(index < num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    T maxval = -FLT_MAX;
    for (int c = 0; c < channels; ++c) {
      maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
    }
    out[index] = maxval;
  }
}

__kernel void kernel_channel_subtract(const int count,
    const int num, const int channels,
    const int spatial_dim, __global const T* channel_max, __global T* data) {
  int index = get_global_id(0);
  if(index < count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] -= channel_max[n * spatial_dim + s];
  }
}

__kernel void kernel_channel_sum(const int num, const int channels,
    const int spatial_dim, __global const T* data, __global T* channel_sum) {
  int index = get_global_id(0);
  if(index < num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    T sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    channel_sum[index] = sum;
  }
}

__kernel void kernel_channel_div(const int count,
    const int num, const int channels,
    const int spatial_dim, __global const T* channel_sum, __global T* data) {
  int index = get_global_id(0);
  if(index < count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    T v = data[index] / channel_sum[n * spatial_dim + s];
#ifdef LOG_SOFTMAX
    v = log(v);
#endif
    data[index] = v;
  }
}
