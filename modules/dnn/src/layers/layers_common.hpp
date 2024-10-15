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
void getConvolutionKernelParams(const LayerParams &params, std::vector<size_t>& kernel, std::vector<size_t>& pads_begin,
                                std::vector<size_t>& pads_end, std::vector<size_t>& strides, std::vector<size_t>& dilations,
                                cv::String &padMode, std::vector<size_t>& adjust_pads, bool& useWinograd);

void getPoolingKernelParams(const LayerParams &params, std::vector<size_t>& kernel, std::vector<bool>& globalPooling,
                            std::vector<size_t>& pads_begin, std::vector<size_t>& pads_end, std::vector<size_t>& strides, cv::String &padMode);

void getConvPoolOutParams(const std::vector<int>& inp, const std::vector<size_t>& kernel,
                          const std::vector<size_t>& stride, const String &padMode,
                          const std::vector<size_t>& dilation, std::vector<int>& out);

void getConvPoolPaddings(const std::vector<int>& inp, const std::vector<size_t>& kernel,
                          const std::vector<size_t>& strides, const String &padMode,
                          std::vector<size_t>& pads_begin, std::vector<size_t>& pads_end);

// Used in quantized model. It will return the (Max_element - Min_element)/127.
double getWeightScale(const Mat& weightsMat);

// Several ONNX operations take list of integer's or float's,
// e.g. to specify list of axes (Squeeze, Unsqueeze, Transpose, Reduce*, ...),
// coordinates, repetitions etc. (Slice, Tile, ...), scale factors (Resize, ...).
// Here are helper functions to extract this data
void tensorToIntVec(const Mat& tensor, std::vector<int>& vec);
void tensorToFloatVec(const Mat& tensor, std::vector<float>& vec);

// tensor to mat shape
MatShape tensorToShape(const Mat& shapeTensor);

// inputs and outputs are both vector<Mat>'s or both are vector<UMat>'s.
// the function does the following:
//
// 1. resizes output vector to 1-element vector
// 2. outputs[0].fit(shape, inputs[0].type())
// 3. temp = inputs[0].reshape(shape);
// 4. temp.copyTo(outputs[0]) // detect in-place case and do nothing in this case
//
// the function helps to implement DL operations
// 'Reshape', 'Flatten', 'Squeeze', 'Unsqueeze', 'Identity'.
void reshapeAndCopyFirst(InputArrayOfArrays inputs,
                         OutputArrayOfArrays outputs,
                         const MatShape& shape);

}
}

#endif
