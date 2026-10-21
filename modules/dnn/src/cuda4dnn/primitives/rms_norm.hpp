// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_RMS_NORM_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_RMS_NORM_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"

#include "../kernels/rms_norm.hpp"

#include <opencv2/core.hpp>
#include <opencv2/dnn/shape_utils.hpp>

#include <cstddef>
#include <utility>
#include <vector>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class RMSNormOp final : public CUDABackendNode {
    public:
        RMSNormOp(csl::Stream stream_, int axis_, float epsilon_)
            : stream(std::move(stream_)), axis(axis_), epsilon(epsilon_) { }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_UNUSED(workspace);
            CV_Assert(inputs.size() >= 2 && outputs.size() == 1);

            auto input = csl::viewOf<T>(inputs[0]);
            auto scale = csl::viewOf<T>(inputs[1]);
            auto output = csl::spanOf<T>(outputs[0]);

            MatShape inShape = cv::dnn::shape(inputs[0]);
            const int a = normalize_axis(axis, inShape.dims);
            CV_Assert(a >= 0 && a < inShape.dims);

            std::size_t loops = 1, norm_size = 1;
            for (int i = 0; i < a; i++) loops *= static_cast<std::size_t>(inShape[i]);
            for (int i = a; i < inShape.dims; i++) norm_size *= static_cast<std::size_t>(inShape[i]);

            kernels::rms_norm<T>(stream, output, input, scale, loops, norm_size, epsilon);
        }

    private:
        csl::Stream stream;
        int axis;
        float epsilon;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_RMS_NORM_HPP */
