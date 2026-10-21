// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_TOPK_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_TOPK_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/memory.hpp"

#include "../kernels/topk.hpp"

#include <opencv2/core.hpp>
#include <opencv2/dnn/shape_utils.hpp>

#include <cstdint>
#include <utility>
#include <vector>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class TopKOp final : public CUDABackendNode {
    public:
        TopKOp(csl::Stream stream_, int axis_, bool largest_)
            : stream(std::move(stream_)), axis(axis_), largest(largest_) { }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_UNUSED(workspace);
            CV_Assert(inputs.size() >= 1 && outputs.size() == 2);

            auto input = csl::viewOf<T>(inputs[0]);
            auto values = csl::spanOf<T>(outputs[0]);
            auto indices = csl::spanOf<std::int64_t>(outputs[1]);

            MatShape inShape = cv::dnn::shape(inputs[0]);
            MatShape outShape = cv::dnn::shape(outputs[0]);
            const int rank = inShape.dims;
            const int a = normalize_axis(axis, rank);
            CV_Assert(a >= 0 && a < rank);

            int outer = 1, inner = 1;
            for (int i = 0; i < a; i++) outer *= inShape[i];
            for (int i = a + 1; i < rank; i++) inner *= inShape[i];

            const int dim_axis = inShape[a];
            /* K is whatever shape inference settled on; reading it back from the output
             * avoids duplicating the attribute-vs-input resolution the layer already did. */
            const int k = outShape[a];

            if (inner == 1 && static_cast<std::int64_t>(k) * dim_axis > sort_threshold) {
                const std::size_t bytes = kernels::topk_sort_workspace<T>(stream, outer, dim_axis, largest);
                if (scratch.size() < bytes)
                    scratch.reset(bytes);
                kernels::topk_sort<T>(stream, values, indices, input, outer, dim_axis, k, largest,
                                      scratch.get().get(), bytes);
                return;
            }

            kernels::topk<T>(stream, values, indices, input, outer, dim_axis, inner, k, largest);
        }

    private:
        static constexpr std::int64_t sort_threshold = std::int64_t(1) << 16;

        csl::Stream stream;
        int axis;
        bool largest;
        csl::ManagedPtr<unsigned char> scratch;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_TOPK_HPP */
