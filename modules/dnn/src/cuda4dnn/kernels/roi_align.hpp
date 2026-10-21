// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_ROI_ALIGN_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_ROI_ALIGN_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

#include <cstddef>
#include <cstdint>

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

struct RoiAlignParams {
    int channels{}, height{}, width{};
    int output_height{}, output_width{};
    int sampling_ratio{};
    float spatial_scale{1.f};
    /* half_pixel subtracts 0.5 from the scaled ROI corners; output_half_pixel does not
     * but clamps a degenerate ROI up to one unit. They are the two ONNX
     * coordinate_transformation_mode values and they move independently. */
    float offset{0.f};
    bool clamp_malformed_roi{false};
    bool max_mode{false};
};

/* ONNX RoiAlign. `rois` is [num_rois, 4] as (x1, y1, x2, y2) in CV_32F; `batch_indices`
 * is [num_rois] of either int32 or int64, selected by `batch_index_element_size`.
 * Output is [num_rois, channels, output_height, output_width].
 *
 * Mirrors the CPU RoiAlignForwardInvoker element for element, including its rule that a
 * sample falling fully outside [-1, extent] contributes nothing at all. */
template <class T>
void roi_align(const csl::Stream& stream,
    csl::Span<T> output, csl::View<T> input, csl::View<float> rois,
    const void* batch_indices, std::size_t batch_index_element_size,
    const RoiAlignParams& params);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_ROI_ALIGN_HPP */
