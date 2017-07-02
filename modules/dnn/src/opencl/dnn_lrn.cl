#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define __constant
#define __local
#define get_global_id(x) 0
#define get_global_size(x) 0
#define get_local_id(x) 0
#define get_local_size(x) 0
#define FLT_MAX 0
#define FLT_MIN 0
#define cl_khr_fp64
#define cl_amd_fp64
#ifndef DISABLE_DOUBLE_SUPPORT
#define DOUBLE_SUPPORT_AVAILABLE
#endif //DISABLE_DOUBLE_SUPPORT
#define CLK_LOCAL_MEM_FENCE
#define CLK_GLOBAL_MEM_FENCE
#define Dtype float
#define barrier(x)
#define atomic_cmpxchg(x, y, z) x
#define signbit(x) x
#define int_tp long
#define uint_tp unsigned long
#define int_tpc long
#define uint_tpc unsigned long
#endif

#define CONCAT(A,B) A##_##B
#define TEMPLATE(name,type) CONCAT(name,type)

#define TYPE_FLOAT 1
#define TYPE_DOUBLE 2

#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#ifndef DISABLE_DOUBLE_SUPPORT
#define DOUBLE_SUPPORT_AVAILABLE
#endif //DISABLE_DOUBLE_SUPPORT
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#ifndef DISABLE_DOUBLE_SUPPORT
#define DOUBLE_SUPPORT_AVAILABLE
#endif //DISABLE_DOUBLE_SUPPORT
#endif

#if defined(cl_khr_int64_base_atomics)
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#define ATOMICS_64_AVAILABLE
#endif

#if defined(cl_khr_int32_base_atomics)
#pragma OPENCL EXTENSION cl_khr_int32_base_atomics : enable
#define ATOMICS_32_AVAILABLE
#endif

#if defined(cl_khr_global_int32_base_atomics)
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#define ATOMICS_32_AVAILABLE
#endif

// Types used for parameters, offset computations and so on
#define int_tp int
#define uint_tp unsigned int

// Definitions used to cast the types above as needed
#define int_tpc int
#define uint_tpc unsigned int

#define Dtype float

__kernel void TEMPLATE(lrn_compute_output,Dtype)(const int_tp nthreads,
                                                 __global const Dtype* in,
                                                 __global const Dtype* scale,
                                                 const Dtype negative_beta,
                                                 __global Dtype* out) {
  for (int_tp index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    out[index] = in[index] * pow(scale[index], negative_beta);
  }
}

__kernel void TEMPLATE(lrn_fill_scale,Dtype)(const int_tp nthreads, __global const Dtype* in,
                             const int_tp num, const int_tp channels,
                             const int_tp height, const int_tp width, const int_tp size,
                             const Dtype alpha_over_size, const Dtype k,
                             __global Dtype* const scale) {
  for (int_tp index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    // find out the local offset
    const int_tp w = index % width;
    const int_tp h = (index / width) % height;
    const int_tp n = index / width / height;
    const int_tp offset = (n * channels * height + h) * width + w;
    const int_tp step = height * width;
    __global const Dtype* in_off = in + offset;
    __global Dtype* scale_off = scale + offset;
    int_tp head = 0;
    const int_tp pre_pad = (size - 1) / 2;
    const int_tp post_pad = size - pre_pad - 1;
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

__kernel void TEMPLATE(lrn_compute_diff,Dtype)(const int_tp nthreads,
                               __global const Dtype* bottom_data,
                               __global const Dtype* top_data,
                               __global const Dtype* scale,
                               __global const Dtype* top_diff, const int_tp num,
                               const int_tp channels, const int_tp height,
                               const int_tp width, const int_tp size,
                               const Dtype negative_beta,
                               const Dtype cache_ratio,
                               __global Dtype* bottom_diff) {
  for (int_tp index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    // find out the local offset
    const int_tp w = index % width;
    const int_tp h = (index / width) % height;
    const int_tp n = index / width / height;
    const int_tp offset = (n * channels * height + h) * width + w;
    const int_tp step = height * width;
    __global const Dtype* bottom_off = bottom_data + offset;
    __global const Dtype* top_off = top_data + offset;
    __global const Dtype* scale_off = scale + offset;
    __global const Dtype* top_diff_off = top_diff + offset;
    __global Dtype* bottom_diff_off = bottom_diff + offset;
    int_tp head = 0;
    const int_tp pre_pad = size - (size + 1) / 2;
    const int_tp post_pad = size - pre_pad - 1;
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
    const int_tp nthreads, __global const Dtype* const bottom_data, const int_tp num,
    const int_tp channels, const int_tp height, const int_tp width,
    const int_tp pooled_height, const int_tp pooled_width, const int_tp kernel_h,
    const int_tp kernel_w, const int_tp stride_h, const int_tp stride_w, const int_tp pad_h,
    const int_tp pad_w, const Dtype power, const Dtype scale, __global Dtype* top_data) {
  for (int_tp index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    {
      const int_tp pw = index % pooled_width;
      const int_tp ph = (index / pooled_width) % pooled_height;
      const int_tp c = (index / pooled_width / pooled_height) % channels;
      const int_tp n = index / pooled_width / pooled_height / channels;
      int_tp hstart = ph * stride_h - pad_h;
      int_tp wstart = pw * stride_w - pad_w;
      int_tp hend = min(hstart + kernel_h, height + pad_h);
      int_tp wend = min(wstart + kernel_w, width + pad_w);
      const int_tp pool_size = (hend - hstart) * (wend - wstart);
      hstart = max(hstart, (int_tp)0);
      wstart = max(wstart, (int_tp)0);
      hend = min(hend, height);
      wend = min(wend, width);
      Dtype aveval = 0;
      __global const Dtype* bottom_slice = bottom_data
          + (n * channels + c) * height * width;
      for (int_tp h = hstart; h < hend; ++h) {
        for (int_tp w = wstart; w < wend; ++w) {
          Dtype tmp_val = bottom_slice[h * width + w];
          aveval += (tmp_val * tmp_val);
        }
      }
          Dtype tmp_aveval = aveval / pool_size;
          top_data[index] = bottom_data[index] * pow(tmp_aveval * scale + 1, power);
    }
  }
}


// !!!!!!!!
// The following macros are duplicated in lrn_optimized_layer.cu since we currently
// don't have a simple solution for sharing between cl and cpp files.
// You must keep both locations in sync!
#define SIMD_SIZE 16
#define TILE_HEIGHT 32
#define TILE_WIDTH (SIMD_SIZE - 2)
// !!!!!!!!

#define KERNEL_SIZE 3.0f
#define POOL_SIZE (KERNEL_SIZE * KERNEL_SIZE)
#define ONE_OVER_POOL_SIZE (1.0f / POOL_SIZE)

__attribute__((intel_reqd_sub_group_size(SIMD_SIZE)))
__kernel void TEMPLATE(lrn_within_channel_forward_opt,Dtype)(
    __global const float* const bottom_data,
    const int_tp height,
    const int_tp width,
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
__kernel void TEMPLATE(lrn_within_channel_backward,Dtype)(const int_tp nthreads,
    __global const Dtype* top_data, __global const Dtype* top_diff,
    const int_tp num, const int_tp channels, const int_tp height,
    const int_tp width, const int_tp pooled_height, const int_tp pooled_width,
    const int_tp kernel_h, const int_tp kernel_w, const int_tp stride_h,
    const int_tp stride_w, const int_tp pad_h, const int_tp pad_w,
    const Dtype power, const Dtype scale, __global const Dtype* bottom_data,
                      __global Dtype* top_bottom_ratio , __global Dtype* bottom_diff){

  for (int_tp index = get_global_id(0); index < nthreads;
              index += get_global_size(0)) {
       // find out the local index
       // find out the local offset
      const int_tp w = index % width + pad_w;
      const int_tp h = (index / width) % height + pad_h;
      const int_tp c = (index / width / height) % channels;
      const int_tp n = index / width / height / channels;
      const int_tp phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
      const int_tp phend = min(h / stride_h + 1, pooled_height);
      const int_tp pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
      const int_tp pwend = min(w / stride_w + 1, pooled_width);
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

      for (int_tp ph = phstart; ph < phend; ++ph) {
        for (int_tp pw = pwstart; pw < pwend; ++pw) {
            // figure out the pooling size
            int_tp hstart = ph * stride_h - pad_h;
            int_tp wstart = pw * stride_w - pad_w;
            int_tp hend = min(hstart + kernel_h, height + pad_h);
            int_tp wend = min(wstart + kernel_w, width + pad_w);
            int_tp pool_size = (hend - hstart) * (wend - wstart);
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
                       const int_tp nthreads, __global const Dtype* bottom_data,
                       Dtype power, __global const Dtype* top_data,
                       __global Dtype* top_bottom_ratio ){
  for (int_tp index = get_global_id(0); index < nthreads;
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
                             const int_tp channels,
                             const int_tp height, const int_tp width,
                             const int_tp tiled_height, int_tp tiled_width,
                             const int_tp size,
                             const Dtype alpha_over_size, const Dtype k,
                             __global Dtype* const out,
                             const Dtype negative_beta,
                             const int_tp pool_h, const int_tp pool_w, const int_tp pool_stride_h, int_tp pool_stride_w,
                             const int_tp pooled_height, const int_tp pooled_width,
                             const int_tp tile_pooled_block_h, const int_tp tile_pooled_block_w) {
  // find out the local offset
  const int_tp block_x = get_global_id(0) % tiled_width;
  const int_tp block_y = (get_global_id(0) / tiled_width) % tiled_height;
  const int_tp n = get_global_id(0) / (tiled_width * tiled_height);
  
  const int_tp w = block_x * tile_pooled_block_w * pool_stride_w;
  const int_tp h = block_y * tile_pooled_block_h * pool_stride_h;
  const int_tp offset = (n * channels * height + h) * width + w;
  const int_tp out_h = block_y * tile_pooled_block_h;
  const int_tp out_w = block_x * tile_pooled_block_w;
  const int_tp out_offset = (n * channels * pooled_height + out_h) * pooled_width + out_w + get_local_id(1);
  const int_tp step = height * width;
  const int_tp out_step = pooled_height * pooled_width;
  __global const Dtype* in_off = in + offset + get_local_id(1);
  __global Dtype* out_off = out + out_offset;
  Dtype scale_val;
  int_tp head = 0;
  const int_tp pre_pad = (size - 1) / 2;
  const int_tp post_pad = size - pre_pad - 1;
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

__kernel void TEMPLATE(lrn_full_no_scale,Dtype)(const int_tp nthreads, __global const Dtype* in,
                             const int_tp num, const int_tp channels,
                             const int_tp height, const int_tp width, const int_tp size,
                             const Dtype alpha_over_size, const Dtype k,
                             __global Dtype* const out,
                             const Dtype negative_beta) {
  for (int_tp index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    // find out the local offset
    const int_tp w = index % width;
    const int_tp h = (index / width) % height;
    const int_tp n = index / width / height;
    const int_tp offset = (n * channels * height + h) * width + w;
    const int_tp step = height * width;
    __global const Dtype* in_off = in + offset;
    __global Dtype* out_off = out + offset;
    Dtype scale_val;
    int_tp head = 0;
    const int_tp pre_pad = (size - 1) / 2;
    const int_tp post_pad = size - pre_pad - 1;
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

__kernel void TEMPLATE(lrn_full,Dtype)(const int_tp nthreads, __global const Dtype* in,
                             const int_tp num, const int_tp channels,
                             const int_tp height, const int_tp width, const int_tp size,
                             const Dtype alpha_over_size, const Dtype k,
                             __global Dtype* const scale,
                             __global Dtype* const out,
                             const Dtype negative_beta) {
  for (int_tp index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    // find out the local offset
    const int_tp w = index % width;
    const int_tp h = (index / width) % height;
    const int_tp n = index / width / height;
    const int_tp offset = (n * channels * height + h) * width + w;
    const int_tp step = height * width;
    __global const Dtype* in_off = in + offset;
    __global Dtype* out_off = out + offset;
    __global Dtype* scale_off = scale + offset;
    Dtype scale_val;
    int_tp head = 0;
    const int_tp pre_pad = (size - 1) / 2;
    const int_tp post_pad = size - pre_pad - 1;
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
