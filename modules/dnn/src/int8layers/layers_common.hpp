// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_DNN_LAYERS_LAYERS_COMMON_HPP__
#define __OPENCV_DNN_LAYERS_LAYERS_COMMON_HPP__
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>

#define CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
// dispatched AVX/AVX2 optimizations
#include "./layers_common.simd.hpp"
#include "int8layers/layers_common.simd_declarations.hpp"
#undef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

#include "./layers_rvp052.hpp"

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
}
}

#endif
