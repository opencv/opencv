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

enum OnnxDataType
{
    ONNX_UNDEFINED  = 0,
    ONNX_FLOAT      = 1,   // float
    ONNX_UINT8      = 2,   // uint8_t
    ONNX_INT8       = 3,   // int8_t
    ONNX_UINT16     = 4,   // uint16_t
    ONNX_INT16      = 5,   // int16_t
    ONNX_INT32      = 6,   // int32_t
    ONNX_INT64      = 7,   // int64_t
    ONNX_STRING     = 8,   // string
    ONNX_BOOL       = 9,   // bool
    ONNX_FLOAT16    = 10,
    ONNX_DOUBLE     = 11,
    ONNX_UINT32     = 12,
    ONNX_UINT64     = 13,
    ONNX_BFLOAT16   = 14
};

inline int onnxDataTypeToCV(OnnxDataType dt)
{
    switch (dt)
    {
    case ONNX_UINT8:      return CV_8U;
    case ONNX_INT8:       return CV_8S;
    case ONNX_UINT16:     return CV_16U;
    case ONNX_INT16:      return CV_16S;
    case ONNX_UINT32:
#ifdef CV_32U
        return CV_32U;
#else
        return CV_32S;
#endif
    case ONNX_INT32:      return CV_32S;
    case ONNX_UINT64:
#ifdef CV_64U
        return CV_64U;
#else
        return CV_32S;
#endif
    case ONNX_INT64:
#ifdef CV_64S
        return CV_64S;
#else
        return CV_32S;
#endif
    case ONNX_FLOAT:      return CV_32F;
    case ONNX_DOUBLE:     return CV_64F;
    case ONNX_FLOAT16:    return CV_16F;
    case ONNX_BFLOAT16:
#ifdef CV_16BF
        return CV_16BF;
#else
        return CV_16F;
#endif
    case ONNX_BOOL:
#ifdef CV_Bool
        return CV_Bool;
#else
        return CV_8U;
#endif
    default:
        // Fallback to default ONNX FLOAT if value is unknown.
        return CV_32F;
    }
}

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
}
}

#endif
