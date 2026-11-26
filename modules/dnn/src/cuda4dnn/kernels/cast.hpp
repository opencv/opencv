// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_CAST_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_CAST_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

#include <cstdint>

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

template <class TOut, class TIn>
void cast(const csl::Stream& stream, csl::Span<TOut> output, csl::View<TIn> input);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_CAST_HPP */
