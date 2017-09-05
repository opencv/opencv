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

__kernel void TEMPLATE(lrn_compute_output,Dtype)(const int nthreads,
                                                 __global const Dtype* in,
                                                 __global const Dtype* scale,
                                                 const Dtype negative_beta,
                                                 __global Dtype* out) {
  for (int index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    out[index] = in[index] * pow(scale[index], negative_beta);
  }
}

__kernel void TEMPLATE(lrn_fill_scale,Dtype)(const int nthreads, __global const Dtype* in,
                             const int num, const int channels,
                             const int height, const int width, const int size,
                             const Dtype alpha_over_size, const Dtype k,
                             __global Dtype* const scale) {
  for (int index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * channels * height + h) * width + w;
    const int step = height * width;
    __global const Dtype* in_off = in + offset;
    __global Dtype* scale_off = scale + offset;
    int head = 0;
    const int pre_pad = (size - 1) / 2;
    const int post_pad = size - pre_pad - 1;
    Dtype accum_scale = 0;
    // fill the scale at [n, :, h, w]
    // accumulate values
    while (head < post_pad && head < channels) {
      accum_scale += in_off[head * step] * in_off[head * step];
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_scale += in_off[head * step] * in_off[head * step];
      if (head - size >= 0) {
        accum_scale -= in_off[(head - size) * step]
            * in_off[(head - size) * step];
      }
      scale_off[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        accum_scale -= in_off[(head - size) * step]
            * in_off[(head - size) * step];
      }
      scale_off[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
      ++head;
    }
  }
}

__kernel void TEMPLATE(lrn_compute_diff,Dtype)(const int nthreads,
                               __global const Dtype* bottom_data,
                               __global const Dtype* top_data,
                               __global const Dtype* scale,
                               __global const Dtype* top_diff, const int num,
                               const int channels, const int height,
                               const int width, const int size,
                               const Dtype negative_beta,
                               const Dtype cache_ratio,
                               __global Dtype* bottom_diff) {
  for (int index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * channels * height + h) * width + w;
    const int step = height * width;
    __global const Dtype* bottom_off = bottom_data + offset;
    __global const Dtype* top_off = top_data + offset;
    __global const Dtype* scale_off = scale + offset;
    __global const Dtype* top_diff_off = top_diff + offset;
    __global Dtype* bottom_diff_off = bottom_diff + offset;
    int head = 0;
    const int pre_pad = size - (size + 1) / 2;
    const int post_pad = size - pre_pad - 1;
    Dtype accum_ratio = 0;
    // accumulate values
    while (head < post_pad && head < channels) {
      accum_ratio += top_diff_off[head * step] * top_off[head * step]
          / scale_off[head * step];
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_ratio += top_diff_off[head * step] * top_off[head * step]
          / scale_off[head * step];
      if (head - size >= 0) {
        accum_ratio -= top_diff_off[(head - size) * step]
            * top_off[(head - size) * step] / scale_off[(head - size) * step];
      }
      bottom_diff_off[(head - post_pad) * step] = top_diff_off[(head - post_pad)
          * step] * pow(scale_off[(head - post_pad) * step], negative_beta)
          - cache_ratio * bottom_off[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        accum_ratio -= top_diff_off[(head - size) * step]
            * top_off[(head - size) * step] / scale_off[(head - size) * step];
      }
      bottom_diff_off[(head - post_pad) * step] = top_diff_off[(head - post_pad)
          * step] * pow(scale_off[(head - post_pad) * step], negative_beta)
          - cache_ratio * bottom_off[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
  }
}

// This kernel is used in LRNOptimizedLayer.
// It is based on ave_pool_forward kernel with some modifications on the input and output.
__kernel void TEMPLATE(lrn_within_channel_forward,Dtype)(
    const int nthreads, __global const Dtype* const bottom_data, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, const Dtype power, const Dtype scale, __global Dtype* top_data) {
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
          Dtype tmp_val = bottom_slice[h * width + w];
          aveval += (tmp_val * tmp_val);
        }
      }
          Dtype tmp_aveval = aveval / pool_size;
          top_data[index] = bottom_data[index] * pow(tmp_aveval * scale + 1, power);
    }
  }
}


#define SIMD_SIZE 16
#define TILE_HEIGHT 32
#define TILE_WIDTH (SIMD_SIZE - 2)

#define KERNEL_SIZE 3.0f
#define POOL_SIZE (KERNEL_SIZE * KERNEL_SIZE)
#define ONE_OVER_POOL_SIZE (1.0f / POOL_SIZE)

__attribute__((intel_reqd_sub_group_size(SIMD_SIZE)))
__kernel void TEMPLATE(lrn_within_channel_forward_opt,Dtype)(
    __global const float* const bottom_data,
    const int height,
    const int width,
    const float power,
    const float scale,
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

      input[0] = as_float(intel_sub_group_block_read((const __global uint*)(base_addr - width)));
      input[1] = as_float(intel_sub_group_block_read((const __global uint*)(base_addr)));

      input[0] = input[0] * input[0];
      input[1] = input[1] * input[1];

      float zero = 0;

      int first = 0;
      int second = 1;
      int third = 2;

      for (int y = starty  ; y <= endy ; y++) {
        base_addr += width;
        input[third] = as_float(intel_sub_group_block_read((const __global uint*)(base_addr)));

        input[third] = input[third] * input[third];

        float tmp_aveval, sum;

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
          tmp_aveval = (sum1 + sum2) * ONE_OVER_POOL_SIZE;
        }
        else if (realx == width - 1) {
          tmp_aveval = (sum + sum1) * ONE_OVER_POOL_SIZE;
        }
        else {
          tmp_aveval = (sum + sum1 + sum2) * ONE_OVER_POOL_SIZE;
       }

        int idx = y * width + realx;

        float res = input_image[idx] * native_powr(tmp_aveval * scale + 1, power);

        if (local_id < TILE_WIDTH && realx < width) {
          output_image[idx] = res;
        }

        first = (first + 1) % 3;
        second = (second + 1) % 3;
        third = (third + 1) % 3;
      }
}

#undef SIMD_SIZE
#undef TILE_HEIGHT
#undef TILE_WIDTH
#undef KERNEL_SIZE
#undef POOL_SIZE
#undef ONE_OVER_POOL_SIZE

// This kernel is used in LRNOptimizedLayer.
// It is based on ave_pool_backward kernel with some modifications on the input and output.
__kernel void TEMPLATE(lrn_within_channel_backward,Dtype)(const int nthreads,
    __global const Dtype* top_data, __global const Dtype* top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    const Dtype power, const Dtype scale, __global const Dtype* bottom_data,
                      __global Dtype* top_bottom_ratio , __global Dtype* bottom_diff){

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
      __global const Dtype* top_diff_slice = top_diff
                 + (n * channels + c) * pooled_height * pooled_width;

      __global const Dtype* top_data_slice = top_data
                 + (n * channels + c) * pooled_height * pooled_width;

      __global const Dtype* bottom_data_slice = bottom_data
                 + (n * channels + c) * height * width;
      __global Dtype* top_bottom_ratio_slice = top_bottom_ratio
                 + (n * channels + c) * pooled_height * pooled_width;

      if (( h - pad_h) * width +  w - pad_w < pooled_width * pooled_height){
          gradient += top_data_slice[(h - pad_h) * width + w - pad_w] /
                      bottom_data_slice[(h - pad_h) * width + w - pad_w] *
                      top_diff_slice[(h - pad_h) * width + w - pad_w];
      }

      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
            // figure out the pooling size
            int hstart = ph * stride_h - pad_h;
            int wstart = pw * stride_w - pad_w;
            int hend = min(hstart + kernel_h, height + pad_h);
            int wend = min(wstart + kernel_w, width + pad_w);
            int pool_size = (hend - hstart) * (wend - wstart);
            Dtype tmp_value = 2 * scale * power * bottom_data[index] / pool_size;
            gradient += top_diff_slice[ph * pooled_width + pw] *
                      top_bottom_ratio_slice[ph * pooled_width + pw] *
                          tmp_value * bottom_data_slice[ph *pooled_width + pw];
        }
      }
      bottom_diff[index] = gradient;
  }
}

// compute power of top and bottom ratio
__kernel void TEMPLATE(lrn_within_channel_backward_ratio,Dtype)(
                       const int nthreads, __global const Dtype* bottom_data,
                       Dtype power, __global const Dtype* top_data,
                       __global Dtype* top_bottom_ratio ){
  for (int index = get_global_id(0); index < nthreads;
              index += get_global_size(0)) {
        top_bottom_ratio[index] = pow(top_data[index] / bottom_data[index],
                                                     (power - 1) / power);
  }
}

#define SIMD_WIDTH 16
#define TILE_W SIMD_WIDTH
#define TILE_H 8

#ifndef BEIGNET
__attribute__((intel_reqd_sub_group_size(SIMD_WIDTH)))
#endif
// Fuse pooling max layer into LRN across channel layer.
// Currently, only support non-padding, non-dilation mode and pool_w/h == pool_stride_w + 1.
// This kernel only get better performance on those Intel platforms with edram.
__kernel void TEMPLATE(lrn_fuse_pool_max,Dtype)(
                             __global const Dtype* in,
                             const int channels,
                             const int height, const int width,
                             const int tiled_height, int tiled_width,
                             const int size,
                             const Dtype alpha_over_size, const Dtype k,
                             __global Dtype* const out,
                             const Dtype negative_beta,
                             const int pool_h, const int pool_w, const int pool_stride_h, int pool_stride_w,
                             const int pooled_height, const int pooled_width,
                             const int tile_pooled_block_h, const int tile_pooled_block_w) {
  // find out the local offset
  const int block_x = get_global_id(0) % tiled_width;
  const int block_y = (get_global_id(0) / tiled_width) % tiled_height;
  const int n = get_global_id(0) / (tiled_width * tiled_height);

  const int w = block_x * tile_pooled_block_w * pool_stride_w;
  const int h = block_y * tile_pooled_block_h * pool_stride_h;
  const int offset = (n * channels * height + h) * width + w;
  const int out_h = block_y * tile_pooled_block_h;
  const int out_w = block_x * tile_pooled_block_w;
  const int out_offset = (n * channels * pooled_height + out_h) * pooled_width + out_w + get_local_id(1);
  const int step = height * width;
  const int out_step = pooled_height * pooled_width;
  __global const Dtype* in_off = in + offset + get_local_id(1);
  __global Dtype* out_off = out + out_offset;
  Dtype scale_val;
  int head = 0;
  const int pre_pad = (size - 1) / 2;
  const int post_pad = size - pre_pad - 1;
  Dtype accum_scale[TILE_H] = {0};
  if (w + get_local_id(1) >= width)
    return;

  while ( head < channels + post_pad ) {
    int ph = 0;
    int cur_out_h = 0;
    Dtype output_val = -FLT_MAX;
    // fill the scale at [n, :, h, w]
    // accumulate values
    for( int lrn_out_h = 0; lrn_out_h < TILE_H && (lrn_out_h + h) < height; lrn_out_h++) {
      Dtype prev_val = accum_scale[lrn_out_h];
      // add
      if (head < channels) {
        prev_val += in_off[head * step + width * lrn_out_h] * in_off[head * step + width * lrn_out_h];
      }
      // subtract
      if (head - size >= 0) {
        prev_val -= in_off[(head - size) * step + width * lrn_out_h] * in_off[(head - size) * step + width * lrn_out_h];
      }
      // compute output.
      if (head >= post_pad) {
        scale_val = k + prev_val * alpha_over_size;
        Dtype tmp = -FLT_MAX;
        //if (w + get_local_id(1) < width)
          tmp = in_off[(head - post_pad) * step + width * lrn_out_h] * native_powr(scale_val, negative_beta);

        Dtype h_max_val = -FLT_MAX;
        int index = (get_local_id(1) * pool_stride_w) % SIMD_WIDTH;
        for(int i = 0; i < pool_w; i++) {
          Dtype val = intel_sub_group_shuffle(tmp, index);
          if (h_max_val < val && (index + w < width))
            h_max_val = val;

          index = (index + 1) % SIMD_WIDTH;
        }
        // update output value.
        output_val = (output_val > h_max_val) ?
                      output_val : h_max_val;
        // time to write previous output and move to next value
        if (lrn_out_h - cur_out_h + 1 == pool_h) {
          if (get_local_id(1) < tile_pooled_block_w && (out_w + get_local_id(1)) < pooled_width) {
            out_off[(head - post_pad) * out_step + ph * pooled_width] = output_val;

            output_val = h_max_val;
          }
          ++ph;
          cur_out_h += pool_stride_h;
        }
      }
      accum_scale[lrn_out_h] = prev_val;
    }
    // Handle the incomplete pool box
    // an incomplete tiling box and we are not hitting the end of the pooled output.
    if (head >= post_pad &&
        ph < tile_pooled_block_h &&
        ph + out_h < pooled_height &&
        get_local_id(1) < tile_pooled_block_w &&
        (out_w + get_local_id(1)) < pooled_width) {
      out_off[(head - post_pad) * out_step + ph * pooled_width] = output_val;
    }
    head++;
  }
}

#undef TILE_W
#undef TILE_H

__kernel void TEMPLATE(lrn_full_no_scale,Dtype)(const int nthreads, __global const Dtype* in,
                             const int num, const int channels,
                             const int height, const int width, const int size,
                             const Dtype alpha_over_size, const Dtype k,
                             __global Dtype* const out,
                             const Dtype negative_beta) {
  for (int index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * channels * height + h) * width + w;
    const int step = height * width;
    __global const Dtype* in_off = in + offset;
    __global Dtype* out_off = out + offset;
    Dtype scale_val;
    int head = 0;
    const int pre_pad = (size - 1) / 2;
    const int post_pad = size - pre_pad - 1;
    Dtype accum_scale = 0;
    // fill the scale at [n, :, h, w]
    // accumulate values
    while (head < post_pad && head < channels) {
      accum_scale += in_off[head * step] * in_off[head * step];
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_scale += in_off[head * step] * in_off[head * step];
      if (head - size >= 0) {
        accum_scale -= in_off[(head - size) * step]
            * in_off[(head - size) * step];
      }
      scale_val = k + accum_scale * alpha_over_size;
      out_off[(head - post_pad) * step] = in_off[(head - post_pad) * step] * (Dtype)native_powr((float)scale_val, (float)negative_beta);
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        accum_scale -= in_off[(head - size) * step]
            * in_off[(head - size) * step];
      }
      scale_val = k + accum_scale * alpha_over_size;
      out_off[(head - post_pad) * step] = in_off[(head - post_pad) * step] * (Dtype)native_powr((float)scale_val, (float)negative_beta);
      ++head;
    }
  }
}

__kernel void TEMPLATE(lrn_full,Dtype)(const int nthreads, __global const Dtype* in,
                             const int num, const int channels,
                             const int height, const int width, const int size,
                             const Dtype alpha_over_size, const Dtype k,
                             __global Dtype* const scale,
                             __global Dtype* const out,
                             const Dtype negative_beta) {
  for (int index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * channels * height + h) * width + w;
    const int step = height * width;
    __global const Dtype* in_off = in + offset;
    __global Dtype* out_off = out + offset;
    __global Dtype* scale_off = scale + offset;
    Dtype scale_val;
    int head = 0;
    const int pre_pad = (size - 1) / 2;
    const int post_pad = size - pre_pad - 1;
    Dtype accum_scale = 0;
    // fill the scale at [n, :, h, w]
    // accumulate values
    while (head < post_pad && head < channels) {
      accum_scale += in_off[head * step] * in_off[head * step];
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_scale += in_off[head * step] * in_off[head * step];
      if (head - size >= 0) {
        accum_scale -= in_off[(head - size) * step]
            * in_off[(head - size) * step];
      }
      scale_val = k + accum_scale * alpha_over_size;
      scale_off[(head - post_pad) * step] = scale_val;
      out_off[(head - post_pad) * step] = in_off[(head - post_pad) * step] * (Dtype)native_powr((float)scale_val, (float)negative_beta);
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        accum_scale -= in_off[(head - size) * step]
            * in_off[(head - size) * step];
      }
      scale_val = k + accum_scale * alpha_over_size;
      scale_off[(head - post_pad) * step] = scale_val;
      out_off[(head - post_pad) * step] = in_off[(head - post_pad) * step] * (Dtype)native_powr((float)scale_val, (float)negative_beta);
      ++head;
    }
  }
}
