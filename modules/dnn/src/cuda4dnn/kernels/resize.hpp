// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_RESIZE_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_RESIZE_HPP

#include "../csl/stream.hpp"
#include "../csl/tensor.hpp"
#include "../csl/span.hpp"

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    // separable gather over host-built per-axis taps (idx < 0 skips a tap); a pixel whose row or
    // column is flagged out of bounds takes extrapolation_value (tf_crop_and_resize)
    template <class T>
    void resize_separable(const csl::Stream& stream, csl::TensorSpan<T> output, csl::TensorView<T> input,
                          csl::View<int> idx_y, csl::View<float> w_y, csl::View<int> oob_y, int taps_y,
                          csl::View<int> idx_x, csl::View<float> w_x, csl::View<int> oob_x, int taps_x,
                          float extrapolation_value);

    template <class T>
    void resize_nn(const csl::Stream& stream, csl::TensorSpan<T> output, csl::TensorView<T> input, float scale_y, float scale_x, bool round, bool half_pixel_centers);

    template <class T>
    void resize_bilinear(const csl::Stream& stream, csl::TensorSpan<T> output, csl::TensorView<T> input, float scale_y, float scale_x, bool half_pixel_centers);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_RESIZE_HPP */
