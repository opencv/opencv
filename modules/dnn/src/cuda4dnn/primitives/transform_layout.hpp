// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_TRANSFORM_LAYOUT_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_TRANSFORM_LAYOUT_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"

#include "../kernels/transform_layout.hpp"
#include "../kernels/fill_copy.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class TransformLayoutOp final : public CUDABackendNode {
    public:
        TransformLayoutOp(csl::Stream stream_, int target_layout_, int C0_, int original_layout_)
            : stream(std::move(stream_)), target_layout(target_layout_), C0(C0_), original_layout(original_layout_)
        {
        }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_UNUSED(workspace);
            CV_Assert(inputs.size() == 1 && outputs.size() == 1);

            auto input = csl::viewOf<T>(inputs[0]);
            auto output = csl::spanOf<T>(outputs[0]);

            MatShape inShape = cv::dnn::shape(inputs[0]);
            MatShape outShape = cv::dnn::shape(outputs[0]);

            int in_layout = inShape.layout;
            if (in_layout == DATA_LAYOUT_UNKNOWN)
                in_layout = original_layout;

            if (in_layout == target_layout)
            {
                kernels::copy<T>(stream, output, input);
                return;
            }

            CV_Assert(original_layout == DATA_LAYOUT_NCHW);

            if (target_layout == DATA_LAYOUT_BLOCK)
            {
                const int N = inShape[0];
                const int C = inShape.channels();
                const std::int64_t planesize = static_cast<std::int64_t>(input.size()) / (static_cast<std::int64_t>(N) * C);
                const int C1 = (C + C0 - 1) / C0;
                kernels::transform_layout_to_block<T>(stream, output, input, C, C0, C1, planesize);
            }
            else
            {
                const int N = inShape[0];
                const int C0_ = inShape.back();
                const int C1 = inShape[1];
                const int C = outShape.channels();
                const std::int64_t planesize = static_cast<std::int64_t>(output.size()) / (static_cast<std::int64_t>(N) * C);
                kernels::transform_layout_from_block<T>(stream, output, input, C, C0_, C1, planesize);
            }
        }

    private:
        csl::Stream stream;
        int target_layout;
        int C0;
        int original_layout;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_TRANSFORM_LAYOUT_HPP */
