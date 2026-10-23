// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SOFTMAX_KERNEL_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SOFTMAX_KERNEL_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"

#include "../kernels/softmax.hpp"
#include "../kernels/scale_shift.hpp"

#include <opencv2/core.hpp>

#include <limits>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class SoftmaxKernelOp final : public CUDABackendNode {
    public:
        /* flatten_tail: pre-opset-13 Softmax, which normalizes over the flattened dims [axis, rank) */
        SoftmaxKernelOp(csl::Stream stream_, int axis_, bool log_softmax_, float scale_ = 1.f, bool flatten_tail_ = false)
            : stream(std::move(stream_)), axis(axis_), log_softmax(log_softmax_), scale(scale_), flatten_tail(flatten_tail_)
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
            auto output = csl::spanOf<T>(outputs[0]);

            MatShape inShape = cv::dnn::shape(inputs[0]);
            const int rank = inShape.dims;
            const int ax = normalize_axis(axis, rank);

            const int axis_end = flatten_tail ? rank : ax + 1;
            std::size_t axis_total = 1, inner_total = 1, outer_total = 1;
            for (int i = ax; i < axis_end; i++)
                axis_total *= inShape[i];
            for (int i = axis_end; i < rank; i++)
                inner_total *= inShape[i];
            for (int i = 0; i < ax; i++)
                outer_total *= inShape[i];
            // the kernels take int extents; the int64 offsets inside them cover the full product
            const std::size_t int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
            CV_CheckLE(axis_total, int_max, "DNN/CUDA: softmax axis is too large");
            CV_CheckLE(inner_total, int_max, "DNN/CUDA: softmax inner size is too large");
            CV_CheckLE(outer_total, int_max, "DNN/CUDA: softmax outer size is too large");
            const int axis_size = static_cast<int>(axis_total);
            const int inner_size = static_cast<int>(inner_total);
            const int outer_size = static_cast<int>(outer_total);

            if (scale != 1.f) {
                // softmax(s*x) does not reduce to a rescale, so fold the scale in before max/exp
                kernels::scale1_with_bias1<T>(stream, output, input,
                                              static_cast<T>(scale), static_cast<T>(0.f));
                input = csl::viewOf<T>(outputs[0]);
            }

            kernels::softmax<T>(stream, output, input, axis_size, outer_size, inner_size, log_softmax);
        }

    private:
        csl::Stream stream;
        int axis;
        bool log_softmax;
        float scale;
        bool flatten_tail;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SOFTMAX_KERNEL_HPP */
