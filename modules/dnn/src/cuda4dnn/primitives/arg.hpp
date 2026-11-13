// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_ARG_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_ARG_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"

#include "../kernels/argmax.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class ArgOp final : public CUDABackendNode {
    public:
        ArgOp(csl::Stream stream_, int axis_, bool is_argmax_, bool select_last_index_)
            : stream(std::move(stream_)), axis(axis_), is_argmax(is_argmax_), select_last_index(select_last_index_)
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
            auto output = csl::spanOf<std::int64_t>(outputs[0]);

            MatShape inShape = cv::dnn::shape(inputs[0]);
            const int rank = inShape.dims;
            const int ax = normalize_axis(axis, rank);

            const int axis_size = inShape[ax];
            int outer_size = 1;
            for (int i = 0; i < ax; i++)
                outer_size *= inShape[i];
            int inner_size = 1;
            for (int i = ax + 1; i < rank; i++)
                inner_size *= inShape[i];

            kernels::arg_min_max<T>(stream, output, input, outer_size, axis_size, inner_size, is_argmax, select_last_index);
        }

    private:
        csl::Stream stream;
        int axis;
        bool is_argmax;
        bool select_last_index;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_ARG_HPP */
