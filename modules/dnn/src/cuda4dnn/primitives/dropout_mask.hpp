// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_DROPOUT_MASK_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_DROPOUT_MASK_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/tensor_ops.hpp"

#include "../kernels/fill_copy.hpp"

#include <vector>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    // Inference-mode Dropout with its mask output(s): y = x, and every mask is all true.
    template <class T>
    class DropoutMaskOp final : public CUDABackendNode {
    public:
        DropoutMaskOp(csl::Stream stream_) : stream(std::move(stream_)) { }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_UNUSED(workspace);
            CV_Assert(!inputs.empty() && outputs.size() >= 2);

            auto input = csl::viewOf<T>(inputs[0]);
            auto output = csl::spanOf<T>(outputs[0]);
            if (input.get() != output.get())
                csl::tensor_ops::copy(stream, output, input);

            for (std::size_t i = 1; i < outputs.size(); i++)
                kernels::fill<bool>(stream, csl::spanOf<bool>(outputs[i]), true);
        }

    private:
        csl::Stream stream;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_DROPOUT_MASK_HPP */
