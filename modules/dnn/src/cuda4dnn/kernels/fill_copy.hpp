// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_FILL_COPY_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_FILL_COPY_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    template <class T>
    void fill(const csl::Stream& stream, csl::Span<T> output, T value);

    template <class T>
    void copy(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_FILL_COPY_HPP */
