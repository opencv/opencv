// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CAST_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CAST_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../kernels/cast.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class TOut, class TIn>
    class CastOp final : public CUDABackendNode {
    public:
        CastOp(csl::Stream stream_) : stream(std::move(stream_)) { }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_UNUSED(workspace);
            CV_Assert(inputs.size() == 1 && outputs.size() == 1);

            auto input = csl::viewOf<TIn>(inputs[0]);
            auto output = csl::spanOf<TOut>(outputs[0]);
            kernels::cast<TOut, TIn>(stream, output, input);
        }

    private:
        csl::Stream stream;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CAST_HPP */
