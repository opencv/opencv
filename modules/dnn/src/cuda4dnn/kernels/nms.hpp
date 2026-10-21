// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_NMS_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_NMS_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

#include <cstddef>
#include <cstdint>

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    constexpr int kNmsMaxBoxes = 16384;

    std::size_t nms_workspace(const csl::Stream& stream, int batch, int classes, int num_boxes);

    int nms(const csl::Stream& stream, csl::Span<std::int64_t> output, csl::View<float> boxes,
            csl::View<float> scores, int batch, int classes, int num_boxes, bool center_point_box,
            int max_output, float iou_threshold, float score_threshold,
            unsigned char* workspace, std::size_t workspace_bytes);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_NMS_HPP */
