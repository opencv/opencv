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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#ifndef __OPENCV_DNN_LAYERS_LAYERS_COMMON_HPP__
#define __OPENCV_DNN_LAYERS_LAYERS_COMMON_HPP__
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>

#define CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
// dispatched AVX/AVX2 optimizations
#include "./layers_common.simd.hpp"
#include "layers/layers_common.simd_declarations.hpp"
#undef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

#ifdef HAVE_OPENCL
#include "../ocl4dnn/include/ocl4dnn.hpp"
#endif

namespace cv
{
namespace dnn
{

void getConvolutionKernelParams(const LayerParams &params, int &kernelH, int &kernelW, int &padT, int &padL, int &padB, int &padR,
                                int &strideH, int &strideW, int &dilationH, int &dilationW, cv::String& padMode);

void getPoolingKernelParams(const LayerParams &params, int &kernelH, int &kernelW, bool &globalPooling,
                            int &padT, int &padL, int &padB, int &padR, int &strideH, int &strideW, cv::String& padMode);

void getConvPoolOutParams(const Size& inp, const Size &kernel,
                          const Size &stride, const String &padMode,
                          const Size &dilation, Size& out);


void getConvPoolPaddings(const Size& inp, const Size& out,
                         const Size &kernel, const Size &stride,
                         const String &padMode, const Size &dilation, int &padT, int &padL, int &padB, int &padR);

}
}

#endif
