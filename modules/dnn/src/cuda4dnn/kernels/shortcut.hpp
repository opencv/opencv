// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_SHORTCUT_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_SHORTCUT_HPP

#include "../csl/stream.hpp"
#include "../csl/tensor.hpp"

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    template <class T>
    void input_shortcut(const csl::Stream& stream, csl::TensorSpan<T> output, csl::TensorView<T> input, csl::TensorView<T> from);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_SHORTCUT_HPP */
