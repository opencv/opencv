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

#if defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if defined KERNEL_MAX_POOL

__kernel void
#ifdef HAVE_MASK
    TEMPLATE(max_pool_forward_mask, Dtype)
#else
    TEMPLATE(max_pool_forward, Dtype)
#endif
(
    const int nthreads, __global const Dtype* bottom_data,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width,
    __global Dtype* top_data
#ifdef HAVE_MASK
    , __global long* mask
#endif
)
{
  int index = get_global_id(0);
  if (index >= nthreads)
    return;

  const int pw = index % pooled_width;
  const int xx = index / pooled_width;
  const int ph = xx % pooled_height;
  const int ch = xx / pooled_height;
  int hstart = ph * STRIDE_H - PAD_T;
  int wstart = pw * STRIDE_W - PAD_L;
  Dtype maxval = -FLT_MAX;
  int maxidx = -1;
  int in_offset = ch * height * width;
  for (int h = 0; h < KERNEL_H; ++h)
  {
    int off_y = hstart + h;
    if (off_y >= 0 && off_y < height)
    {
      for (int w = 0; w < KERNEL_W; ++w)
      {
        int off_x = wstart + w;
        if (off_x >= 0 && off_x < width)
        {
          Dtype val = bottom_data[in_offset + off_y * width + off_x];
          maxidx = (val > maxval) ? (off_y * width + off_x) : maxidx;
          maxval = fmax(val, maxval);
        }
      }
    }
  }
  top_data[index] = maxval;
#ifdef HAVE_MASK
  mask[index] = maxidx;
#endif
}

#elif defined KERNEL_AVE_POOL

__kernel void TEMPLATE(ave_pool_forward, Dtype)(
    const int nthreads, __global const Dtype* bottom_data,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width,
    __global Dtype* top_data)
{
  int index = get_global_id(0);
  if (index >= nthreads)
    return;

  const int pw = index % pooled_width;
  const int xx = index / pooled_width;
  const int ph = xx % pooled_height;
  const int ch = xx / pooled_height;
  int hstart = ph * STRIDE_H - PAD_T;
  int wstart = pw * STRIDE_W - PAD_L;
  int hend = min(hstart + KERNEL_H, height + PAD_B);
  int wend = min(wstart + KERNEL_W, width + PAD_R);
  int pool_size;
#ifdef AVE_POOL_PADDING_AREA
  pool_size = (hend - hstart) * (wend - wstart);
  hstart = max(hstart, (int)0);
  wstart = max(wstart, (int)0);
  hend = min(hend, height);
  wend = min(wend, width);
#else
  hstart = max(hstart, (int)0);
  wstart = max(wstart, (int)0);
  hend = min(hend, height);
  wend = min(wend, width);
  pool_size = (hend - hstart) * (wend - wstart);
#endif
  Dtype aveval = 0;
  int in_offset = ch * height * width;
  for (int h = hstart; h < hend; ++h)
  {
    for (int w = wstart; w < wend; ++w)
    {
      aveval += bottom_data[in_offset + h * width + w];
    }
  }
  top_data[index] = aveval / pool_size;
}

#elif defined KERNEL_STO_POOL

__kernel void TEMPLATE(sto_pool_forward_test,Dtype)(
    const int nthreads, __global const Dtype* bottom_data,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width,
    __global Dtype* top_data)
{
  for (int index = get_global_id(0); index < nthreads;
      index += get_global_size(0))
  {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    const int hstart = ph * STRIDE_H;
    const int hend = min(hstart + KERNEL_H, height);
    const int wstart = pw * STRIDE_W;
    const int wend = min(wstart + KERNEL_W, width);
    // We set cumsum to be 0 to avoid divide-by-zero problems
    Dtype cumsum = FLT_MIN;
    Dtype cumvalues = 0.;
    __global const Dtype* bottom_slice = bottom_data
        + (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        Dtype v = bottom_slice[h * width + w];
        cumsum += v;
        cumvalues += v * v;
      }
    }
    top_data[index] = cumvalues / cumsum;
  }
}

#endif // KERNEL_*
