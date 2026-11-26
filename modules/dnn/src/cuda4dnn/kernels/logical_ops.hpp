// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_LOGICAL_OPS_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_LOGICAL_OPS_HPP

#include "../csl/stream.hpp"
#include "../csl/tensor.hpp"

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    template <class T>
    void compare_equal(const csl::Stream& stream, csl::TensorSpan<bool> output, csl::TensorView<T> x, csl::TensorView<T> y);

    template <class T>
    void compare_greater(const csl::Stream& stream, csl::TensorSpan<bool> output, csl::TensorView<T> x, csl::TensorView<T> y);

    template <class T>
    void compare_greater_equal(const csl::Stream& stream, csl::TensorSpan<bool> output, csl::TensorView<T> x, csl::TensorView<T> y);

    template <class T>
    void compare_less(const csl::Stream& stream, csl::TensorSpan<bool> output, csl::TensorView<T> x, csl::TensorView<T> y);

    template <class T>
    void compare_less_equal(const csl::Stream& stream, csl::TensorSpan<bool> output, csl::TensorView<T> x, csl::TensorView<T> y);

    void logical_and(const csl::Stream& stream, csl::TensorSpan<bool> output, csl::TensorView<bool> x, csl::TensorView<bool> y);

    void logical_or(const csl::Stream& stream, csl::TensorSpan<bool> output, csl::TensorView<bool> x, csl::TensorView<bool> y);

    void logical_xor(const csl::Stream& stream, csl::TensorSpan<bool> output, csl::TensorView<bool> x, csl::TensorView<bool> y);

    void logical_not(const csl::Stream& stream, csl::TensorSpan<bool> output, csl::TensorView<bool> input);

    template <class T>
    void where(const csl::Stream& stream, csl::TensorSpan<T> output, csl::TensorView<bool> cond, csl::TensorView<T> x, csl::TensorView<T> y);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_LOGICAL_OPS_HPP */
