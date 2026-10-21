// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_BROADCAST_COPY_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_BROADCAST_COPY_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"

#include "../kernels/broadcast_copy.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    /* backs both Tile and Expand; shapes are read per forward() since Expand's target shape is an input */
    template <class T>
    class BroadcastCopyOp final : public CUDABackendNode {
    public:
        BroadcastCopyOp(csl::Stream stream_) : stream(std::move(stream_)) { }

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
            MatShape outShape = cv::dnn::shape(outputs[0]);
            CV_Assert(inShape.dims <= outShape.dims);

            /* ONNX broadcasts right-aligned, so a lower-rank input is padded on the
             * left with unit axes before the coordinate mapping is applied. */
            const int rank = outShape.dims;
            const int pad = rank - inShape.dims;
            std::vector<std::int64_t> in_dims(rank), out_dims(rank);
            for (int axis = 0; axis < rank; axis++) {
                in_dims[axis] = (axis < pad) ? 1 : static_cast<std::int64_t>(inShape[axis - pad]);
                out_dims[axis] = static_cast<std::int64_t>(outShape[axis]);
            }

            kernels::broadcast_copy<T>(stream, output, input, in_dims, out_dims);
        }

    private:
        csl::Stream stream;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_BROADCAST_COPY_HPP */
