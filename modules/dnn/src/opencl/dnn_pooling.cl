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

#if defined(cl_intel_subgroups)
#pragma OPENCL EXTENSION  cl_intel_subgroups : enable
#endif

void TEMPLATE(max_pool_forward_impl, Dtype)(
    const int_tp nthreads, __global const Dtype* bottom_data, const int_tp num,
    const int_tp channels, const int_tp height, const int_tp width,
    const int_tp pooled_height, const int_tp pooled_width, const int_tp kernel_h,
    const int_tp kernel_w, const int_tp stride_h, const int_tp stride_w, const int_tp pad_h,
    const int_tp pad_w,
    __global Dtype* top_data,
    const int use_mask, __global int_tp* mask, __global Dtype* top_mask, bool no_mask) {
  for (int_tp index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    const int_tp pw = index % pooled_width;
    const int_tp ph = (index / pooled_width) % pooled_height;
    const int_tp c = (index / pooled_width / pooled_height) % channels;
    const int_tp n = index / pooled_width / pooled_height / channels;
    int_tp hstart = ph * stride_h - pad_h;
    int_tp wstart = pw * stride_w - pad_w;
    const int_tp hend = min(hstart + kernel_h, height);
    const int_tp wend = min(wstart + kernel_w, width);
    hstart = max(hstart, (int_tp)0);
    wstart = max(wstart, (int_tp)0);
    Dtype maxval = -FLT_MAX;
    int_tp maxidx = -1;
    __global const Dtype* bottom_slice = bottom_data
        + (n * channels + c) * height * width;
    for (int_tp h = hstart; h < hend; ++h) {
      for (int_tp w = wstart; w < wend; ++w) {
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
    const int_tp nthreads, __global const Dtype* bottom_data, const int_tp num,
    const int_tp channels, const int_tp height, const int_tp width,
    const int_tp pooled_height, const int_tp pooled_width, const int_tp kernel_h,
    const int_tp kernel_w, const int_tp stride_h, const int_tp stride_w, const int_tp pad_h,
    const int_tp pad_w,
    __global Dtype* top_data) {

    TEMPLATE(max_pool_forward_impl, Dtype)(
      nthreads, bottom_data, num, channels, height, width,
      pooled_height, pooled_width, kernel_h,
      kernel_w, stride_h, stride_w, pad_h, pad_w, top_data, 0, NULL, NULL, true
    );
}

__kernel void TEMPLATE(max_pool_forward, Dtype)(
    const int_tp nthreads, __global const Dtype* bottom_data, const int_tp num,
    const int_tp channels, const int_tp height, const int_tp width,
    const int_tp pooled_height, const int_tp pooled_width, const int_tp kernel_h,
    const int_tp kernel_w, const int_tp stride_h, const int_tp stride_w, const int_tp pad_h,
    const int_tp pad_w,
    __global Dtype* top_data,
    const int use_mask, __global int_tp* mask, __global Dtype* top_mask) {

    TEMPLATE(max_pool_forward_impl, Dtype)(
      nthreads, bottom_data, num, channels, height, width,
      pooled_height, pooled_width, kernel_h,
      kernel_w, stride_h, stride_w, pad_h, pad_w, top_data, use_mask, mask, top_mask, false
    );
}

__kernel void TEMPLATE(ave_pool_forward, Dtype)(
    const int_tp nthreads, __global const Dtype* const bottom_data, const int_tp num,
    const int_tp channels, const int_tp height, const int_tp width,
    const int_tp pooled_height, const int_tp pooled_width, const int_tp kernel_h,
    const int_tp kernel_w, const int_tp stride_h, const int_tp stride_w, const int_tp pad_h,
    const int_tp pad_w, __global Dtype* top_data) {
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
    const int_tp height,
    const int_tp width,
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
    const int_tp nthreads, __global const Dtype* bottom_data, const int_tp num,
    const int_tp channels, const int_tp height, const int_tp width,
    const int_tp pooled_height, const int_tp pooled_width, const int_tp kernel_h,
    const int_tp kernel_w, const int_tp stride_h, const int_tp stride_w,
    __global Dtype* rand_idx,
    __global Dtype* top_data) {
  for (int_tp index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    const int_tp pw = index % pooled_width;
    const int_tp ph = (index / pooled_width) % pooled_height;
    const int_tp c = (index / pooled_width / pooled_height) % channels;
    const int_tp n = index / pooled_width / pooled_height / channels;
    const int_tp hstart = ph * stride_h;
    const int_tp hend = min(hstart + kernel_h, height);
    const int_tp wstart = pw * stride_w;
    const int_tp wend = min(wstart + kernel_w, width);
    Dtype cumsum = 0.;
    __global const Dtype* bottom_slice = bottom_data
        + (n * channels + c) * height * width;
    // First pass: get sum
    for (int_tp h = hstart; h < hend; ++h) {
      for (int_tp w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
      }
    }
    const float thres = rand_idx[index] * cumsum;
    // Second pass: get value, and set index.
    cumsum = 0;
    for (int_tp h = hstart; h < hend; ++h) {
      for (int_tp w = wstart; w < wend; ++w) {
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
    const int_tp nthreads, __global const Dtype* const bottom_data, const int_tp num,
    const int_tp channels, const int_tp height, const int_tp width,
    const int_tp pooled_height, const int_tp pooled_width, const int_tp kernel_h,
    const int_tp kernel_w, const int_tp stride_h, const int_tp stride_w,
    __global Dtype* top_data) {
  for (int_tp index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    const int_tp pw = index % pooled_width;
    const int_tp ph = (index / pooled_width) % pooled_height;
    const int_tp c = (index / pooled_width / pooled_height) % channels;
    const int_tp n = index / pooled_width / pooled_height / channels;
    const int_tp hstart = ph * stride_h;
    const int_tp hend = min(hstart + kernel_h, height);
    const int_tp wstart = pw * stride_w;
    const int_tp wend = min(wstart + kernel_w, width);
    // We set cumsum to be 0 to avoid divide-by-zero problems
    Dtype cumsum = FLT_MIN;
    Dtype cumvalues = 0.;
    __global const Dtype* bottom_slice = bottom_data
        + (n * channels + c) * height * width;
    // First pass: get sum
    for (int_tp h = hstart; h < hend; ++h) {
      for (int_tp w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
        cumvalues += bottom_slice[h * width + w] * bottom_slice[h * width + w];
      }
    }
    top_data[index] = cumvalues / cumsum;
  }
}

__kernel void TEMPLATE(max_pool_backward,Dtype)(const int_tp nthreads,
                                                __global const Dtype* top_diff,
                                                const int use_mask,
                                                __global const int_tp* mask,
                                                __global const Dtype* top_mask,
                                                const int_tp num,
                                                const int_tp channels,
                                                const int_tp height,
                                                const int_tp width,
                                                const int_tp pooled_height,
                                                const int_tp pooled_width,
                                                const int_tp kernel_h,
                                                const int_tp kernel_w,
                                                const int_tp stride_h,
                                                const int_tp stride_w,
                                                const int_tp pad_h,
                                                const int_tp pad_w,
                                                __global Dtype* bottom_diff) {
  for (int_tp index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    // find out the local index
    // find out the local offset
    const int_tp w = index % width;
    const int_tp h = (index / width) % height;
    const int_tp c = (index / width / height) % channels;
    const int_tp n = index / width / height / channels;
    const int_tp phstart =
        (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int_tp phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int_tp pwstart =
        (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int_tp pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const int_tp offset = (n * channels + c) * pooled_height * pooled_width;
    __global const Dtype* top_diff_slice = top_diff + offset;
    if (use_mask == 1) {
      __global const int_tp* mask_slice = mask + offset;
      for (int_tp ph = phstart; ph < phend; ++ph) {
        for (int_tp pw = pwstart; pw < pwend; ++pw) {
          if (mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_slice[ph * pooled_width + pw];
          }
        }
      }
    } else {
      __global const Dtype* top_mask_slice = top_mask + offset;
      for (int_tp ph = phstart; ph < phend; ++ph) {
        for (int_tp pw = pwstart; pw < pwend; ++pw) {
          if (top_mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_slice[ph * pooled_width + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

__kernel void TEMPLATE(ave_pool_backward,Dtype)(const int_tp nthreads,
                                                __global const Dtype* top_diff,
                                                const int_tp num,
                                                const int_tp channels,
                                                const int_tp height,
                                                const int_tp width,
                                                const int_tp pooled_height,
                                                const int_tp pooled_width,
                                                const int_tp kernel_h,
                                                const int_tp kernel_w,
                                                const int_tp stride_h,
                                                const int_tp stride_w,
                                                const int_tp pad_h,
                                                const int_tp pad_w,
                                                __global Dtype* bottom_diff) {
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
    __global const Dtype* const top_diff_slice = top_diff
        + (n * channels + c) * pooled_height * pooled_width;
    for (int_tp ph = phstart; ph < phend; ++ph) {
      for (int_tp pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int_tp hstart = ph * stride_h - pad_h;
        int_tp wstart = pw * stride_w - pad_w;
        int_tp hend = min(hstart + kernel_h, height + pad_h);
        int_tp wend = min(wstart + kernel_w, width + pad_w);
        int_tp pool_size = (hend - hstart) * (wend - wstart);
        gradient += top_diff_slice[ph * pooled_width + pw] / pool_size;
      }
    }
    bottom_diff[index] = gradient;
  }
}

__kernel void TEMPLATE(sto_pool_backward,Dtype)(
    const int_tp nthreads, __global const Dtype* rand_idx,
    __global const Dtype* const top_diff, const int_tp num,
    const int_tp channels, const int_tp height, const int_tp width,
    const int_tp pooled_height, const int_tp pooled_width,
    const int_tp kernel_h, const int_tp kernel_w, const int_tp stride_h,
    const int_tp stride_w, __global Dtype* bottom_diff) {
  for (int_tp index = get_global_id(0); index < nthreads; index +=
      get_global_size(0)) {
    // find out the local index
    // find out the local offset
    const int_tp w = index % width;
    const int_tp h = (index / width) % height;
    const int_tp c = (index / width / height) % channels;
    const int_tp n = index / width / height / channels;
    const int_tp phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int_tp phend = min(h / stride_h + 1, pooled_height);
    const int_tp pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int_tp pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0.0;
    __global const Dtype* rand_idx_slice = rand_idx
        + (n * channels + c) * pooled_height * pooled_width;
    __global const Dtype* top_diff_slice = top_diff
        + (n * channels + c) * pooled_height * pooled_width;
    for (int_tp ph = phstart; ph < phend; ++ph) {
      for (int_tp pw = pwstart; pw < pwend; ++pw) {
        gradient += top_diff_slice[ph * pooled_width + pw]
            * (Dtype)(index == (int_tp) (rand_idx_slice[ph * pooled_width + pw])?1.0:0.0);
      }
    }
    bottom_diff[index] = gradient;
  }
}

