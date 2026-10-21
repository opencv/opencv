// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_RANGE_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_RANGE_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"

#include "../kernels/range.hpp"

#include <opencv2/core.hpp>

#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class RangeOp final : public CUDABackendNode {
    public:
        RangeOp(csl::Stream stream_) : stream(std::move(stream_)) { }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_UNUSED(workspace);
            CV_Assert(inputs.size() == 3 && outputs.size() == 1);
            CV_Assert(inputs[0].total() == 1 && inputs[2].total() == 1);
            CV_Assert(inputs[0].u && inputs[2].u);

            // start/delta are dereferenced inside the kernel: no host round trip, no stall
            auto devPtr = [](const UMat& m) {
                return reinterpret_cast<const T*>(
                    reinterpret_cast<const uchar*>(m.u->handle) + m.offset);
            };

            auto output = csl::spanOf<T>(outputs[0]);
            kernels::range<T>(stream, output, devPtr(inputs[0]), devPtr(inputs[2]));
        }

    private:
        csl::Stream stream;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_RANGE_HPP */
