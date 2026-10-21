// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_ONE_HOT_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_ONE_HOT_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

#include <cstddef>

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    template <class T, class TIdx>
    void one_hot(const csl::Stream& stream, csl::Span<T> output, csl::View<TIdx> indices,
                 std::size_t depth, std::size_t inner, T off_value, T on_value);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_ONE_HOT_HPP */
