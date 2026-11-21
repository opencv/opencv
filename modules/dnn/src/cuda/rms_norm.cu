// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cub/block/block_reduce.cuh>

#include "types.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include "../cuda4dnn/kernels/rms_norm.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {

        template <class T, int BlockSize>
        __global__ void rms_norm(Span<T> output, View<T> input, View<T> scale,
                                 std::size_t loops, std::size_t norm_size, float epsilon)
        {
            using BlockReduce = cub::BlockReduce<float, BlockSize>;
            __shared__ typename BlockReduce::TempStorage reduce_storage;
            __shared__ float inv_stddev;

            const T* in = input.data().get();
            const T* sc = scale.data().get();
            T* out = output.data().get();

            for (std::size_t row = blockIdx.x; row < loops; row += gridDim.x) {
                const std::int64_t base = static_cast<std::int64_t>(row) * norm_size;

                float thread_sum_sq = 0.f;
                for (std::size_t j = threadIdx.x; j < norm_size; j += BlockSize) {
                    const float v = static_cast<float>(in[base + j]);
                    thread_sum_sq += v * v;
                }

                const float block_sum_sq = BlockReduce(reduce_storage).Sum(thread_sum_sq);

                if (threadIdx.x == 0) {
                    // recenter=false collapses the variance to the mean square
                    const float mean_square = block_sum_sq / static_cast<float>(norm_size);
                    inv_stddev = 1.f / sqrtf(fmaxf(0.f, mean_square) + epsilon);
                }
                __syncthreads();  // publishes inv_stddev and frees reduce_storage for the next row

                for (std::size_t j = threadIdx.x; j < norm_size; j += BlockSize) {
                    const float v = static_cast<float>(in[base + j]);
                    const float s = static_cast<float>(sc[j]);
                    out[base + j] = static_cast<T>(s * v * inv_stddev);
                }
                __syncthreads();
            }
        }
    }

    template <class T>
    void rms_norm(const Stream& stream,
        Span<T> output, View<T> input, View<T> scale,
        std::size_t loops, std::size_t norm_size, float epsilon)
    {
        CV_Assert(loops > 0 && norm_size > 0);
        CV_Assert(scale.size() >= norm_size);

        constexpr int block_size = 256;
        constexpr int max_blocks = 65535;
        const int grid_size = static_cast<int>(std::min<std::size_t>(max_blocks, loops));

        raw::rms_norm<T, block_size><<<grid_size, block_size, 0, stream.get()>>>(
            output, input, scale, loops, norm_size, epsilon);
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void rms_norm(const Stream&, Span<__half>, View<__half>, View<__half>, std::size_t, std::size_t, float);
#endif
    template void rms_norm(const Stream&, Span<float>, View<float>, View<float>, std::size_t, std::size_t, float);
    template void rms_norm(const Stream&, Span<int8_t>, View<int8_t>, View<int8_t>, std::size_t, std::size_t, float);
    template void rms_norm(const Stream&, Span<uint8_t>, View<uint8_t>, View<uint8_t>, std::size_t, std::size_t, float);
    template void rms_norm(const Stream&, Span<int32_t>, View<int32_t>, View<int32_t>, std::size_t, std::size_t, float);
    template void rms_norm(const Stream&, Span<int64_t>, View<int64_t>, View<int64_t>, std::size_t, std::size_t, float);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
