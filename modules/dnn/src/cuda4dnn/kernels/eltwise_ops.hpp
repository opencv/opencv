// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_ELTWISE_OPS_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_ELTWISE_OPS_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

#include <cstddef>

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    template <class T>
    void eltwise_max_2(const csl::Stream& stream, csl::Span<T> output, csl::View<T> x, csl::View<T> y);

    template <class T>
    void eltwise_sum_2(const csl::Stream& stream, csl::Span<T> output, csl::View<T> x, csl::View<T> y);

    template <class T>
    void eltwise_sum_coeff_2(const csl::Stream& stream, csl::Span<T> output, T coeff_x, csl::View<T> x, T coeff_y, csl::View<T> y);

    template <class T>
    void eltwise_prod_2(const csl::Stream& stream, csl::Span<T> output, csl::View<T> x, csl::View<T> y);

    template <class T>
    void eltwise_div_2(const csl::Stream& stream, csl::Span<T> output, csl::View<T> x, csl::View<T> y);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_ELTWISE_OPS_HPP */
