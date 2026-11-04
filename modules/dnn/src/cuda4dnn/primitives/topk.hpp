// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_TOPK_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_TOPK_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/workspace.hpp"

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
        TopKOp(csl::Stream stream_, int axis_, bool largest_, const MatShape& inShape, int k_)
            : stream(std::move(stream_)), largest(largest_), k(k_)
        {
            const int a = normalize_axis(axis_, inShape.dims);
            CV_Assert(a >= 0 && a < inShape.dims);
            for (int i = 0; i < a; i++) outer *= inShape[i];
            for (int i = a + 1; i < inShape.dims; i++) inner *= inShape[i];
            dim_axis = inShape[a];
            // the sort path's scratch comes from the shared workspace, sized once per shape
            if (inner == 1 && static_cast<std::int64_t>(k) * dim_axis > sort_threshold)
                sort_bytes = kernels::topk_sort_workspace<T>(stream, outer, dim_axis, largest);
        }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_Assert(inputs.size() >= 1 && outputs.size() == 2);

            auto input = csl::viewOf<T>(inputs[0]);
            auto values = csl::spanOf<T>(outputs[0]);
            auto indices = csl::spanOf<std::int64_t>(outputs[1]);

            if (sort_bytes) {
                auto scratch = csl::WorkspaceAllocator(workspace).get_span<unsigned char>(sort_bytes);
                kernels::topk_sort<T>(stream, values, indices, input, outer, dim_axis, k, largest,
                                      scratch.data().get(), sort_bytes);
                return;
            }

            kernels::topk<T>(stream, values, indices, input, outer, dim_axis, inner, k, largest);
        }

        std::size_t get_workspace_memory_in_bytes() const noexcept override { return sort_bytes; }

    private:
        static constexpr std::int64_t sort_threshold = std::int64_t(1) << 16;

        csl::Stream stream;
        bool largest;
        int k, outer = 1, inner = 1, dim_axis = 1;
        std::size_t sort_bytes = 0;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_TOPK_HPP */
