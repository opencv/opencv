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

#include "../cuda4dnn/kernels/transform_layout.hpp"

#include <cstdint>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {
        // plane [N, C, planesize] -> block [N, C1, planesize, C0], last block zero padded
        template <class T>
        __global__ void transform_layout_to_block(Span<T> output, View<T> input, int C, int C0, int C1, std::int64_t planesize)
        {
            const T* in = input.data().get();
            for (auto id : grid_stride_range(output.size())) {
                std::int64_t idx = id;
                std::int64_t c0 = idx % C0;
                std::int64_t rem = idx / C0;
                std::int64_t p = rem % planesize;
                std::int64_t rem2 = rem / planesize;
                std::int64_t c1 = rem2 % C1;
                std::int64_t n = rem2 / C1;
                std::int64_t c = c1 * C0 + c0;
                output[id] = (c < C) ? in[(n * C + c) * planesize + p] : static_cast<T>(0);
            }
        }

        // block [N, C1, planesize, C0] -> plane [N, C, planesize]
        template <class T>
        __global__ void transform_layout_from_block(Span<T> output, View<T> input, int C, int C0, int C1, std::int64_t planesize)
        {
            const T* in = input.data().get();
            for (auto id : grid_stride_range(output.size())) {
                std::int64_t idx = id;
                std::int64_t p = idx % planesize;
                std::int64_t rem = idx / planesize;
                std::int64_t c = rem % C;
                std::int64_t n = rem / C;
                std::int64_t c1 = c / C0;
                std::int64_t c0 = c % C0;
                output[id] = in[((n * C1 + c1) * planesize + p) * C0 + c0];
            }
        }
    }

    template <class T>
    void transform_layout_to_block(const Stream& stream, Span<T> output, View<T> input, int C, int C0, int C1, std::int64_t planesize)
    {
        auto kernel = raw::transform_layout_to_block<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, input, C, C0, C1, planesize);
    }

    template <class T>
    void transform_layout_from_block(const Stream& stream, Span<T> output, View<T> input, int C, int C0, int C1, std::int64_t planesize)
    {
        auto kernel = raw::transform_layout_from_block<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, input, C, C0, C1, planesize);
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void transform_layout_to_block(const Stream&, Span<__half>, View<__half>, int, int, int, std::int64_t);
    template void transform_layout_from_block(const Stream&, Span<__half>, View<__half>, int, int, int, std::int64_t);
#endif
    template void transform_layout_to_block(const Stream&, Span<float>, View<float>, int, int, int, std::int64_t);
    template void transform_layout_from_block(const Stream&, Span<float>, View<float>, int, int, int, std::int64_t);
    template void transform_layout_to_block(const Stream&, Span<int8_t>, View<int8_t>, int, int, int, std::int64_t);
    template void transform_layout_from_block(const Stream&, Span<int8_t>, View<int8_t>, int, int, int, std::int64_t);
    template void transform_layout_to_block(const Stream&, Span<uint8_t>, View<uint8_t>, int, int, int, std::int64_t);
    template void transform_layout_from_block(const Stream&, Span<uint8_t>, View<uint8_t>, int, int, int, std::int64_t);
    template void transform_layout_to_block(const Stream&, Span<int32_t>, View<int32_t>, int, int, int, std::int64_t);
    template void transform_layout_from_block(const Stream&, Span<int32_t>, View<int32_t>, int, int, int, std::int64_t);
    template void transform_layout_to_block(const Stream&, Span<int64_t>, View<int64_t>, int, int, int, std::int64_t);
    template void transform_layout_from_block(const Stream&, Span<int64_t>, View<int64_t>, int, int, int, std::int64_t);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
