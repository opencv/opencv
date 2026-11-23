// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SIZE_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SIZE_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"

#include "../kernels/fill_copy.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    class SizeOp final : public CUDABackendNode {
    public:
        SizeOp(csl::Stream stream_) : stream(std::move(stream_)) { }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_UNUSED(workspace);
            CV_Assert(inputs.size() == 1 && outputs.size() == 1);

            auto output = csl::spanOf<std::int64_t>(outputs[0]);
            CV_Assert(output.size() == 1);

            const std::int64_t n = static_cast<std::int64_t>(inputs[0].total());
            kernels::fill<std::int64_t>(stream, output, n);
        }

    private:
        csl::Stream stream;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SIZE_HPP */
