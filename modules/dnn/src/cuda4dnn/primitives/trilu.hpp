// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_TRILU_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_TRILU_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"

#include "../kernels/trilu.hpp"

#include <opencv2/core.hpp>

#include <utility>
#include <vector>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class TriluOp final : public CUDABackendNode {
    public:
        TriluOp(csl::Stream stream_, int k_, bool upper_)
            : stream(std::move(stream_)), k(k_), upper(upper_) { }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_UNUSED(workspace);
            CV_Assert(inputs.size() >= 1 && outputs.size() == 1);

            auto input = csl::viewOf<T>(inputs[0]);
            auto output = csl::spanOf<T>(outputs[0]);

            MatShape inShape = cv::dnn::shape(inputs[0]);
            const int rank = inShape.dims;
            CV_Assert(rank >= 1);

            const int width = inShape[rank - 1];
            const int height = (rank >= 2) ? inShape[rank - 2] : 1;

            int loops = 1;
            for (int i = 0; i < rank - 2; i++) loops *= inShape[i];

            kernels::trilu<T>(stream, output, input, loops, height, width, k, upper);
        }

    private:
        csl::Stream stream;
        int k;
        bool upper;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_TRILU_HPP */
