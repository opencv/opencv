// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_GRID_SAMPLE_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_GRID_SAMPLE_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"

#include "../kernels/grid_sample.hpp"

#include <opencv2/core.hpp>

#include <utility>
#include <vector>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class GridSampleOp final : public CUDABackendNode {
    public:
        GridSampleOp(csl::Stream stream_, const kernels::GridSampleParams& params_)
            : stream(std::move(stream_)), params(params_) { }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_UNUSED(workspace);
            CV_Assert(inputs.size() == 2 && outputs.size() == 1);

            auto input = csl::viewOf<T>(inputs[0]);
            auto output = csl::spanOf<T>(outputs[0]);
            auto grid = csl::viewOf<float>(inputs[1]);

            MatShape inShape = cv::dnn::shape(inputs[0]);
            MatShape outShape = cv::dnn::shape(outputs[0]);
            CV_Assert(inShape.dims == 4 && outShape.dims == 4);

            kernels::GridSampleParams p = params;
            p.channels      = inShape[1];
            p.height        = inShape[2];
            p.width         = inShape[3];
            p.output_height = outShape[2];
            p.output_width  = outShape[3];

            kernels::grid_sample<T>(stream, output, input, grid, p);
        }

    private:
        csl::Stream stream;
        kernels::GridSampleParams params;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_GRID_SAMPLE_HPP */
