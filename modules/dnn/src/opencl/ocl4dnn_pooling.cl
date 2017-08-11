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

void TEMPLATE(max_pool_forward_impl, Dtype)(
    const int nthreads, __global const Dtype* bottom_data, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w,
    __global Dtype* top_data,
    const int use_mask, __global int* mask, __global Dtype* top_mask, bool no_mask) {
  for (int index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, (int)0);
    wstart = max(wstart, (int)0);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    __global const Dtype* bottom_slice = bottom_data
        + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_slice[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_slice[maxidx];
        }
      }
    }
    top_data[index] = maxval;
    if (!no_mask) {
      if (use_mask == 1) {
        mask[index] = maxidx;
      } else {
        top_mask[index] = maxidx;
      }
    }
  }
}

__kernel void TEMPLATE(max_pool_forward_no_mask, Dtype)(
    const int nthreads, __global const Dtype* bottom_data, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w,
    __global Dtype* top_data) {

    TEMPLATE(max_pool_forward_impl, Dtype)(
      nthreads, bottom_data, num, channels, height, width,
      pooled_height, pooled_width, kernel_h,
      kernel_w, stride_h, stride_w, pad_h, pad_w, top_data, 0, NULL, NULL, true
    );
}

__kernel void TEMPLATE(max_pool_forward, Dtype)(
    const int nthreads, __global const Dtype* bottom_data, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w,
    __global Dtype* top_data,
    const int use_mask, __global int* mask, __global Dtype* top_mask) {

    TEMPLATE(max_pool_forward_impl, Dtype)(
      nthreads, bottom_data, num, channels, height, width,
      pooled_height, pooled_width, kernel_h,
      kernel_w, stride_h, stride_w, pad_h, pad_w, top_data, use_mask, mask, top_mask, false
    );
}

__kernel void TEMPLATE(ave_pool_forward, Dtype)(
    const int nthreads, __global const Dtype* const bottom_data, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, __global Dtype* top_data) {
  for (int index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    {
      const int pw = index % pooled_width;
      const int ph = (index / pooled_width) % pooled_height;
      const int c = (index / pooled_width / pooled_height) % channels;
      const int n = index / pooled_width / pooled_height / channels;
      int hstart = ph * stride_h - pad_h;
      int wstart = pw * stride_w - pad_w;
      int hend = min(hstart + kernel_h, height + pad_h);
      int wend = min(wstart + kernel_w, width + pad_w);
      const int pool_size = (hend - hstart) * (wend - wstart);
      hstart = max(hstart, (int)0);
      wstart = max(wstart, (int)0);
      hend = min(hend, height);
      wend = min(wend, width);
      Dtype aveval = 0;
      __global const Dtype* bottom_slice = bottom_data
          + (n * channels + c) * height * width;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          aveval += bottom_slice[h * width + w];
        }
      }
      top_data[index] = aveval / pool_size;
    }
  }
}

// !!!!!!!!
// The following macros are duplicated in pooling_layer.cu since we currently
// don't have a simple solution for sharing between cl and cpp files.
// You must keep both locations in sync!
#define SIMD_SIZE 8
#define TILE_HEIGHT 7
#define TILE_WIDTH (SIMD_SIZE - 2)
// !!!!!!!!

#define KERNEL_SIZE 3.0f
#define POOL_SIZE (KERNEL_SIZE * KERNEL_SIZE)
#define ONE_OVER_POOL_SIZE (1.0f / POOL_SIZE)

__attribute__((intel_reqd_sub_group_size(SIMD_SIZE)))
__kernel void TEMPLATE(ave_pool_forward_opt,Dtype)(
    __global const float* const bottom_data,
    const int height,
    const int width,
    __global float* top_data)
{
    int local_id = get_local_id(0);

    int startx = get_global_id(0);
    int tile_y = get_global_id(1);
    int channel = get_global_id(2);

    int realx = startx / SIMD_SIZE * TILE_WIDTH + (startx - startx / SIMD_SIZE * SIMD_SIZE) % TILE_WIDTH;

    int offset = height * width * channel;

    __global const float* const input_image = bottom_data + offset;
    __global float* const output_image = top_data + offset;

#ifdef PRINT
    bool print = (realx == 0);
#endif
    float input[3];

    int starty = tile_y * TILE_HEIGHT;
    int endy = min(height - 1, starty + TILE_HEIGHT - 1);

    // Read 3 lines of 16 floats
    // The 3 lines start one float before the current (to the left) and one line up:
    //
    // 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
    // 0 X 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
    // 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
    //
    // In the diagram above X represents the current work item
    int base = starty * width + realx;
    __global const float* base_addr = input_image + base - 1;

    input[0] = *(base_addr - width);
    input[1] = *base_addr;

    const float zero = 0;

    int first = 0;
    int second = 1;
    int third = 2;

    for (int y = starty  ; y <= endy ; y++) {
      base_addr += width;

      input[third] = *base_addr;

      float res, sum;
      if (y == 0) {
        sum = input[second] + input[third];
      }
      else if (y == height - 1) {
        sum = input[first] + input[second];
      }
      else {
        sum = input[first] + input[second] + input[third];
      }

      float sum1 = intel_sub_group_shuffle_down(sum, zero , 1);
      float sum2 = intel_sub_group_shuffle_down(sum, zero , 2);

      if (realx == 0) {
        res = (sum1 + sum2) * ONE_OVER_POOL_SIZE;
      }
      else if (realx == width - 1) {
        res = (sum + sum1) * ONE_OVER_POOL_SIZE;
      }
      else {
        res = (sum + sum1 + sum2) * ONE_OVER_POOL_SIZE;
      }

      int idx = y * width + realx;

      if (local_id < TILE_WIDTH && realx < width) {
        output_image[idx] = res;
      }

      if (first == 0) {
        first = 1;
        second = 2;
        third = 0;
      }
      else if (first == 1) {
        first = 2;
        second = 0;
        third = 1;
      }
      else {
        first = 0;
        second = 1;
        third = 2;
      }
    }
}

#undef SIMD_SIZE
#undef TILE_HEIGHT
#undef TILE_WIDTH
#undef KERNEL_SIZE
#undef POOL_SIZE
#undef ONE_OVER_POOL_SIZE

__kernel void TEMPLATE(sto_pool_forward_train,Dtype)(
    const int nthreads, __global const Dtype* bottom_data, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w,
    __global Dtype* rand_idx,
    __global Dtype* top_data) {
  for (int index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    const int hstart = ph * stride_h;
    const int hend = min(hstart + kernel_h, height);
    const int wstart = pw * stride_w;
    const int wend = min(wstart + kernel_w, width);
    Dtype cumsum = 0.;
    __global const Dtype* bottom_slice = bottom_data
        + (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
      }
    }
    const float thres = rand_idx[index] * cumsum;
    // Second pass: get value, and set index.
    cumsum = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
        if (cumsum >= thres) {
          rand_idx[index] = ((n * channels + c) * height + h) * width + w;
          top_data[index] = bottom_slice[h * width + w];
          h = hend;
          w = wend;
        }
      }
    }
  }
}

__kernel void TEMPLATE(sto_pool_forward_test,Dtype)(
    const int nthreads, __global const Dtype* const bottom_data, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w,
    __global Dtype* top_data) {
  for (int index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    const int hstart = ph * stride_h;
    const int hend = min(hstart + kernel_h, height);
    const int wstart = pw * stride_w;
    const int wend = min(wstart + kernel_w, width);
    // We set cumsum to be 0 to avoid divide-by-zero problems
    Dtype cumsum = FLT_MIN;
    Dtype cumvalues = 0.;
    __global const Dtype* bottom_slice = bottom_data
        + (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
        cumvalues += bottom_slice[h * width + w] * bottom_slice[h * width + w];
      }
    }
    top_data[index] = cumvalues / cumsum;
  }
}

__kernel void TEMPLATE(max_pool_backward,Dtype)(const int nthreads,
                                                __global const Dtype* top_diff,
                                                const int use_mask,
                                                __global const int* mask,
                                                __global const Dtype* top_mask,
                                                const int num,
                                                const int channels,
                                                const int height,
                                                const int width,
                                                const int pooled_height,
                                                const int pooled_width,
                                                const int kernel_h,
                                                const int kernel_w,
                                                const int stride_h,
                                                const int stride_w,
                                                const int pad_h,
                                                const int pad_w,
                                                __global Dtype* bottom_diff) {
  for (int index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart =
        (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int pwstart =
        (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const int offset = (n * channels + c) * pooled_height * pooled_width;
    __global const Dtype* top_diff_slice = top_diff + offset;
    if (use_mask == 1) {
      __global const int* mask_slice = mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_slice[ph * pooled_width + pw];
          }
        }
      }
    } else {
      __global const Dtype* top_mask_slice = top_mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (top_mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_slice[ph * pooled_width + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

__kernel void TEMPLATE(ave_pool_backward,Dtype)(const int nthreads,
                                                __global const Dtype* top_diff,
                                                const int num,
                                                const int channels,
                                                const int height,
                                                const int width,
                                                const int pooled_height,
                                                const int pooled_width,
                                                const int kernel_h,
                                                const int kernel_w,
                                                const int stride_h,
                                                const int stride_w,
                                                const int pad_h,
                                                const int pad_w,
                                                __global Dtype* bottom_diff) {
  for (int index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    // find out the local index
    // find out the local offset
    const int w = index % width + pad_w;
    const int h = (index / width) % height + pad_h;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0.0;
    __global const Dtype* const top_diff_slice = top_diff
        + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        int pool_size = (hend - hstart) * (wend - wstart);
        gradient += top_diff_slice[ph * pooled_width + pw] / pool_size;
      }
    }
    bottom_diff[index] = gradient;
  }
}

__kernel void TEMPLATE(sto_pool_backward,Dtype)(
    const int nthreads, __global const Dtype* rand_idx,
    __global const Dtype* const top_diff, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, __global Dtype* bottom_diff) {
  for (int index = get_global_id(0); index < nthreads; index +=
      get_global_size(0)) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0.0;
    __global const Dtype* rand_idx_slice = rand_idx
        + (n * channels + c) * pooled_height * pooled_width;
    __global const Dtype* top_diff_slice = top_diff
        + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        gradient += top_diff_slice[ph * pooled_width + pw]
            * (Dtype)(index == (int) (rand_idx_slice[ph * pooled_width + pw])?1.0:0.0);
      }
    }
    bottom_diff[index] = gradient;
  }
}
