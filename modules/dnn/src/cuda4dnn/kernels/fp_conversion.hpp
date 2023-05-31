// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_FP_CONVERSION_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_FP_CONVERSION_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    void fp32_to_fp16(const csl::Stream& stream, csl::Span<half> output, csl::View<float> input);
    void fp16_to_fp32(const csl::Stream& stream, csl::Span<float> output, csl::View<half> input);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_FP_CONVERSION_HPP */
