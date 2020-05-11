// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_REGION_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_REGION_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

#include <cstddef>

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    template <class T>
    void region(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input, csl::View<T> bias,
        T object_prob_cutoff, T class_prob_cutoff,
        std::size_t boxes_per_cell, std::size_t box_size,
        std::size_t rows, std::size_t cols, T scale_x_y,
        std::size_t height_norm, std::size_t width_norm,
        bool if_true_sigmoid_else_softmax);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_REGION_HPP */
