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

#include "../cuda4dnn/kernels/one_hot.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {
        template <class T, class TIdx>
        __global__ void one_hot(Span<T> output, View<TIdx> indices, std::int64_t depth, std::int64_t inner,
                                T off_value, T on_value)
        {
            for (auto id : grid_stride_range(output.size())) {
                const std::int64_t i = id;
                const std::int64_t lo = i % inner;
                const std::int64_t outer = i / inner;
                const std::int64_t d = outer % depth;
                const std::int64_t hi = outer / depth;
                // only [-depth, depth-1] is valid; anything else matches no d and stays all off_value
                std::int64_t v = static_cast<std::int64_t>(indices.data().get()[hi * inner + lo]);
                if (v < 0)
                    v += depth;
                output[id] = (v == d) ? on_value : off_value;
            }
        }
    }

    template <class T, class TIdx>
    void one_hot(const Stream& stream, Span<T> output, View<TIdx> indices,
                 std::size_t depth, std::size_t inner, T off_value, T on_value)
    {
        CV_Assert(depth > 0 && inner > 0);
        CV_Assert(output.size() == indices.size() * depth);
        if (output.size() == 0)
            return;

        auto kernel = raw::one_hot<T, TIdx>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, indices, static_cast<std::int64_t>(depth),
                      static_cast<std::int64_t>(inner), off_value, on_value);
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void one_hot(const Stream&, Span<__half>, View<int32_t>, std::size_t, std::size_t, __half, __half);
    template void one_hot(const Stream&, Span<__half>, View<int64_t>, std::size_t, std::size_t, __half, __half);
#endif
    template void one_hot(const Stream&, Span<float>, View<int32_t>, std::size_t, std::size_t, float, float);
    template void one_hot(const Stream&, Span<float>, View<int64_t>, std::size_t, std::size_t, float, float);
    template void one_hot(const Stream&, Span<int8_t>, View<int32_t>, std::size_t, std::size_t, int8_t, int8_t);
    template void one_hot(const Stream&, Span<int8_t>, View<int64_t>, std::size_t, std::size_t, int8_t, int8_t);
    template void one_hot(const Stream&, Span<uint8_t>, View<int32_t>, std::size_t, std::size_t, uint8_t, uint8_t);
    template void one_hot(const Stream&, Span<uint8_t>, View<int64_t>, std::size_t, std::size_t, uint8_t, uint8_t);
    template void one_hot(const Stream&, Span<int32_t>, View<int32_t>, std::size_t, std::size_t, int32_t, int32_t);
    template void one_hot(const Stream&, Span<int32_t>, View<int64_t>, std::size_t, std::size_t, int32_t, int32_t);
    template void one_hot(const Stream&, Span<int64_t>, View<int32_t>, std::size_t, std::size_t, int64_t, int64_t);
    template void one_hot(const Stream&, Span<int64_t>, View<int64_t>, std::size_t, std::size_t, int64_t, int64_t);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
