// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CONSTANT_OF_SHAPE_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CONSTANT_OF_SHAPE_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../kernels/fill_copy.hpp"

#include <opencv2/core.hpp>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class ConstantOfShapeOp final : public CUDABackendNode {
    public:
        ConstantOfShapeOp(csl::Stream stream_, const Mat& value)
            : stream(std::move(stream_))
        {
            CV_Assert(value.total() == 1);
            fillValue = *reinterpret_cast<const T*>(value.data);
        }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_UNUSED(inputs);
            CV_UNUSED(workspace);
            CV_Assert(outputs.size() == 1);

            auto output = csl::spanOf<T>(outputs[0]);
            kernels::fill<T>(stream, output, fillValue);
        }

    private:
        csl::Stream stream;
        T fillValue;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CONSTANT_OF_SHAPE_HPP */
