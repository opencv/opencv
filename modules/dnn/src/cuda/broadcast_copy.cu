// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "types.hpp"
#include "grid_stride_range.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include "../cuda4dnn/kernels/broadcast_copy.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {
        struct BroadcastLayout {
            int rank{};
            std::int64_t out_dims[kMaxBroadcastRank]{};
            std::int64_t in_dims[kMaxBroadcastRank]{};
            std::int64_t in_strides[kMaxBroadcastRank]{};
        };

        template <class T>
        __global__ void broadcast_copy(Span<T> output, View<T> input, BroadcastLayout layout)
        {
            for (auto id : grid_stride_range(output.size())) {
                /* fold each output coordinate straight into the input offset; the modulo covers Tile and Expand */
                std::int64_t remaining = id;
                std::int64_t input_index = 0;
                for (int axis = layout.rank - 1; axis >= 0; --axis) {
                    const std::int64_t coord = remaining % layout.out_dims[axis];
                    remaining /= layout.out_dims[axis];
                    input_index += (coord % layout.in_dims[axis]) * layout.in_strides[axis];
                }
                output[id] = input.data().get()[input_index];
            }
        }
    }

    template <class T>
    void broadcast_copy(const Stream& stream,
        Span<T> output, View<T> input,
        const std::vector<std::int64_t>& in_dims,
        const std::vector<std::int64_t>& out_dims)
    {
        CV_Assert(in_dims.size() == out_dims.size());
        CV_Assert(in_dims.size() <= kMaxBroadcastRank);

        raw::BroadcastLayout layout;
        layout.rank = static_cast<int>(in_dims.size());

        std::int64_t stride = 1;
        for (int axis = layout.rank - 1; axis >= 0; --axis) {
            CV_Assert(in_dims[axis] > 0 && out_dims[axis] > 0);
            /* Tile repeats the axis and Expand broadcasts it; anything else is a shape
             * the caller should not have accepted. */
            CV_Assert(out_dims[axis] % in_dims[axis] == 0);
            layout.out_dims[axis] = out_dims[axis];
            layout.in_dims[axis] = in_dims[axis];
            layout.in_strides[axis] = stride;
            stride *= in_dims[axis];
        }

        if (output.size() == 0)
            return;

        auto kernel = raw::broadcast_copy<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, input, layout);
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void broadcast_copy(const Stream&, Span<__half>, View<__half>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
#endif
    template void broadcast_copy(const Stream&, Span<float>, View<float>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void broadcast_copy(const Stream&, Span<int8_t>, View<int8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void broadcast_copy(const Stream&, Span<uint8_t>, View<uint8_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void broadcast_copy(const Stream&, Span<int32_t>, View<int32_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void broadcast_copy(const Stream&, Span<int64_t>, View<int64_t>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);
    template void broadcast_copy(const Stream&, Span<bool>, View<bool>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
