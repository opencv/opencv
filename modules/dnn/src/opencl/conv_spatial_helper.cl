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

// Types used for parameters, offset computations and so on
#define int_tp int
#define uint_tp unsigned int
#define Dtype float

__kernel void TEMPLATE(copyImage, Dtype)
    (__global Dtype* image_data,
     int_tp image_offset,
     const int_tp channels, const int_tp height, const int_tp width,
     const int_tp adjustedHeight, const int_tp adjustedWidth,
     const int_tp pad_h, const int_tp pad_w,
     __global Dtype* output_image,
     const int_tp output_offset,
     const int_tp batch_size) {

  uint_tp sX = get_global_id(0);
  uint_tp sY = get_global_id(1);
  uint_tp sZ = get_global_id(2);

  int_tp in_y = sY - pad_h;
  int_tp in_x = sX - pad_w;

  int_tp batch_offset = 0;
  int_tp adjusted_batch_offset = 0;
  for(uint_tp batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    int_tp dst_offset = adjusted_batch_offset + output_offset + sZ*adjustedHeight*adjustedWidth + sY*adjustedWidth +sX;
    int_tp src_offset = batch_offset + image_offset + sZ*height*width + in_y*width + in_x;
    if((in_y >= 0 && in_y < height && in_x >= 0 && in_x < width))
      output_image[dst_offset] = image_data[src_offset];
    else
      output_image[dst_offset] = 0;
    batch_offset += height * width * channels;
    adjusted_batch_offset += adjustedHeight * adjustedWidth * channels;
  }
}

__kernel void TEMPLATE(copyWeightsSwizzled, Dtype)
    (__global Dtype* weightIn,
     __global Dtype* weightOut,
     const int_tp kernel_w,
     const int_tp kernel_h,
     const int_tp channels,
     const int_tp outputs,
     const int_tp swizzleFactor) {

  uint_tp sX = get_global_id(0);

  //Original location

  //Output location
  int_tp outputSublayer = channels / swizzleFactor;
  int_tp outputSublayerIndex = channels % swizzleFactor;

  int_tp filter = sX / (kernel_w*kernel_h*channels);
  int_tp kernel_X = sX % kernel_w;
  int_tp kernel_Y = (sX / kernel_w) % kernel_h;
  int_tp kernel_C = (sX / (kernel_w * kernel_h)) % channels;

  int_tp FP = filter / swizzleFactor;
  int_tp F1 = filter % swizzleFactor;

  weightOut[FP*(kernel_w*kernel_h*channels*swizzleFactor) + kernel_C*(kernel_w*kernel_h*swizzleFactor) + kernel_Y*(kernel_w*swizzleFactor) + kernel_X*swizzleFactor + F1]
  = weightIn[filter*(kernel_w*kernel_h*channels) + kernel_C*(kernel_w*kernel_h) + kernel_Y*kernel_w + kernel_X];
}
