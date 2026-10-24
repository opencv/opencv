// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_LOGICAL_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_LOGICAL_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"

#include "../kernels/logical_ops.hpp"

#include <opencv2/core.hpp>

#include <vector>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    enum class CompareOpType {
        EQUAL,
        GREATER,
        GREATER_EQUAL,
        LESS,
        LESS_EQUAL,
    };

    template <class T>
    class CompareOp final : public CUDABackendNode {
    public:
        CompareOp(csl::Stream stream_, CompareOpType op_) : stream(std::move(stream_)), op(op_) { }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_UNUSED(workspace);
            CV_Assert(inputs.size() == 2 && outputs.size() == 1);

            auto x = csl::viewOf<T>(inputs[0]);
            auto y = csl::viewOf<T>(inputs[1]);
            auto output = csl::spanOf<bool>(outputs[0]);

            switch (op)
            {
            case CompareOpType::EQUAL: kernels::compare_equal<T>(stream, output, x, y); break;
            case CompareOpType::GREATER: kernels::compare_greater<T>(stream, output, x, y); break;
            case CompareOpType::GREATER_EQUAL: kernels::compare_greater_equal<T>(stream, output, x, y); break;
            case CompareOpType::LESS: kernels::compare_less<T>(stream, output, x, y); break;
            case CompareOpType::LESS_EQUAL: kernels::compare_less_equal<T>(stream, output, x, y); break;
            }
        }

    private:
        csl::Stream stream;
        CompareOpType op;
    };

    enum class LogicalOpType {
        AND,
        OR,
        XOR,
    };

    template <class T>
    class LogicalOp final : public CUDABackendNode {
    public:
        LogicalOp(csl::Stream stream_, LogicalOpType op_) : stream(std::move(stream_)), op(op_) { }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_UNUSED(workspace);
            CV_Assert(inputs.size() == 2 && outputs.size() == 1);

            auto x = csl::viewOf<T>(inputs[0]);
            auto y = csl::viewOf<T>(inputs[1]);
            auto output = csl::spanOf<T>(outputs[0]);

            switch (op)
            {
            case LogicalOpType::AND: kernels::logical_and(stream, output, x, y); break;
            case LogicalOpType::OR: kernels::logical_or(stream, output, x, y); break;
            case LogicalOpType::XOR: kernels::logical_xor(stream, output, x, y); break;
            }
        }

    private:
        csl::Stream stream;
        LogicalOpType op;
    };

    template <class T>
    class WhereOp final : public CUDABackendNode {
    public:
        WhereOp(csl::Stream stream_) : stream(std::move(stream_)) { }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_UNUSED(workspace);
            CV_Assert(inputs.size() == 3 && outputs.size() == 1);

            auto cond = csl::viewOf<bool>(inputs[0]);
            auto x = csl::viewOf<T>(inputs[1]);
            auto y = csl::viewOf<T>(inputs[2]);
            auto output = csl::spanOf<T>(outputs[0]);

            kernels::where<T>(stream, output, cond, x, y);
        }

    private:
        csl::Stream stream;
    };

    template <class T>
    class LogicalNotOp final : public CUDABackendNode {
    public:
        LogicalNotOp(csl::Stream stream_) : stream(std::move(stream_)) { }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_UNUSED(workspace);
            CV_Assert(inputs.size() == 1 && outputs.size() == 1);

            auto input = csl::viewOf<T>(inputs[0]);
            auto output = csl::spanOf<T>(outputs[0]);

            kernels::logical_not(stream, output, input);
        }

    private:
        csl::Stream stream;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_LOGICAL_HPP */
