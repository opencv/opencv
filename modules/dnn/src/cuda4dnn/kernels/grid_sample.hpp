// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_GRID_SAMPLE_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_GRID_SAMPLE_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

struct GridSampleParams {
    int channels{}, height{}, width{};
    int output_height{}, output_width{};
    int mode{1};      // 0 = nearest, 1 = bilinear, 2 = bicubic
    int padding{0};   // 0 = zeros,   1 = border,   2 = reflection
    bool align_corners{false};
    float cubic_alpha{-0.75f};
};

/* ONNX GridSample over 4-D input. `input` is [N, C, H, W], `grid` is
 * [N, H_out, W_out, 2] in CV_32F holding normalised (x, y), and the output is
 * [N, C, H_out, W_out].
 *
 * Mirrors gridSampleComputeRows() in gridsample_layer.cpp. That function has a fast
 * path for fully-interior taps and a general `fetch` path; the two agree by
 * construction, so this kernel only implements the general one. */
template <class T>
void grid_sample(const csl::Stream& stream,
    csl::Span<T> output, csl::View<T> input, csl::View<float> grid,
    const GridSampleParams& params);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_GRID_SAMPLE_HPP */
