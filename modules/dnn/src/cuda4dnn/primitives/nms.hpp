// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_NMS_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_NMS_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/memory.hpp"

#include "../kernels/nms.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <memory>
#include <vector>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    class NonMaxSuppressionOp final : public CUDABackendNode {
    public:
        NonMaxSuppressionOp(csl::Stream stream_, bool center_point_box_, int default_max_output_,
                            float default_iou_threshold_, float default_score_threshold_,
                            std::shared_ptr<int> selected_)
            : stream(std::move(stream_)), center_point_box(center_point_box_),
              default_max_output(default_max_output_), default_iou_threshold(default_iou_threshold_),
              default_score_threshold(default_score_threshold_), selected(std::move(selected_))
        {
        }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_UNUSED(workspace);
            CV_Assert(inputs.size() >= 2 && outputs.size() == 1);

            auto scalarInput = [&](std::size_t i, double fallback) {
                if (i >= inputs.size() || inputs[i].empty())
                    return fallback;
                Mat m, m64;
                inputs[i].copyTo(m);
                CV_Assert(m.total() == 1);
                m.convertTo(m64, CV_64F);
                return m64.ptr<double>()[0];
            };
            const int max_output = static_cast<int>(scalarInput(2, default_max_output));
            const float iou_threshold = static_cast<float>(scalarInput(3, default_iou_threshold));
            const float score_threshold = static_cast<float>(scalarInput(4, default_score_threshold));

            const MatShape boxes_shape = cv::dnn::shape(inputs[0]);
            const MatShape scores_shape = cv::dnn::shape(inputs[1]);
            CV_Assert(boxes_shape.dims == 3 && boxes_shape[2] == 4 && scores_shape.dims == 3);
            const int batch = boxes_shape[0], num_boxes = boxes_shape[1], classes = scores_shape[1];

            *selected = 0;
            if (outputs[0].total() == 0 || num_boxes == 0 || classes == 0)
                return;

            auto boxes = csl::viewOf<float>(inputs[0]);
            auto scores = csl::viewOf<float>(inputs[1]);
            auto output = csl::spanOf<std::int64_t>(outputs[0]);

            const std::size_t bytes = kernels::nms_workspace(stream, batch, classes, num_boxes);
            if (scratch.size() < bytes)
                scratch.reset(bytes);
            *selected = kernels::nms(stream, output, boxes, scores, batch, classes, num_boxes, center_point_box,
                                     max_output, iou_threshold, score_threshold, scratch.get().get(), bytes);
        }

    private:
        csl::Stream stream;
        bool center_point_box;
        int default_max_output;
        float default_iou_threshold, default_score_threshold;
        std::shared_ptr<int> selected;
        csl::ManagedPtr<unsigned char> scratch;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_NMS_HPP */
