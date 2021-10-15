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

#ifdef HALF_SUPPORT
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16:enable
#endif
#endif

#define CONCAT(A,B) A##_##B
#define TEMPLATE(name,type) CONCAT(name,type)

__kernel void TEMPLATE(copyWeightsSwizzled, Dtype)
    (__global Dtype* weightIn,
     __global Dtype* weightOut,
     const int kernel_w,
     const int kernel_h,
     const int channels,
     const int outputs,
     const int swizzleFactor) {

  unsigned int sX = get_global_id(0);

  //Original location

  //Output location
  //int outputSublayer = channels / swizzleFactor;
  //int outputSublayerIndex = channels % swizzleFactor;

  int filter = sX / (kernel_w*kernel_h*channels);
  int kernel_X = sX % kernel_w;
  int kernel_Y = (sX / kernel_w) % kernel_h;
  int kernel_C = (sX / (kernel_w * kernel_h)) % channels;

  int FP = filter / swizzleFactor;
  int F1 = filter % swizzleFactor;

  int idxOut = FP*(kernel_w*kernel_h*channels*swizzleFactor) + kernel_C*(kernel_w*kernel_h*swizzleFactor) + kernel_Y*(kernel_w*swizzleFactor) + kernel_X*swizzleFactor + F1;
  int idxIn = filter*(kernel_w*kernel_h*channels) + kernel_C*(kernel_w*kernel_h) + kernel_Y*kernel_w + kernel_X;

  // idxIn is not valid if (filter >= outputs) - no data for these elements. Output alignment gaps are filled by zeros
  Dtype v = (filter < outputs) ? weightIn[idxIn] : (Dtype)0;
  weightOut[idxOut] = v;
}
