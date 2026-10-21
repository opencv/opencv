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

#include "../cuda4dnn/kernels/cum_scan.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {

        /* fp16 accumulates in fp32; integral types must accumulate in themselves, since
         * routing an int64 scan through float would silently lose precision. */
        template <class T> struct work_type      { using type = T; };
        template <> struct work_type<__half>     { using type = float; };

        template <class T>
        __global__ void cum_scan(Span<T> output, View<T> input,
                                 std::int64_t outer_size, std::int64_t target_size, std::int64_t inner_size,
                                 bool is_prod, bool exclusive, bool reverse)
        {
            using W = typename work_type<T>::type;

            const T* src = input.data().get();
            T* dst = output.data().get();

            const std::int64_t outer_step_length = target_size * inner_size;

            /* Same index arithmetic as the CPU scan; target_step is signed so that the
             * reverse direction needs no separate code path. */
            const std::int64_t target_start = reverse ? target_size - 1 : 0;
            const std::int64_t target_stop  = reverse ? -1 : target_size;
            const std::int64_t target_delta = reverse ? -1 : 1;
            const std::int64_t target_step  = target_delta * inner_size;
            const std::int64_t exclusive_delta = exclusive ? target_step : 0;

            const std::int64_t nrows = outer_size * inner_size;

            for (auto id : grid_stride_range(static_cast<device::size_type>(nrows))) {
                const std::int64_t outer_idx = static_cast<std::int64_t>(id) / inner_size;
                const std::int64_t inner_idx = static_cast<std::int64_t>(id) - outer_idx * inner_size;

                const std::int64_t target_offset = outer_idx * outer_step_length;
                const std::int64_t first = target_offset + target_start * inner_size + inner_idx;

                dst[first] = exclusive ? static_cast<T>(is_prod ? W(1) : W(0)) : src[first];

                for (std::int64_t t = target_start + target_delta; t != target_stop; t += target_delta) {
                    const std::int64_t off = target_offset + t * inner_size + inner_idx;
                    const W prev = static_cast<W>(dst[off - target_step]);
                    const W cur  = static_cast<W>(src[off - exclusive_delta]);
                    dst[off] = static_cast<T>(is_prod ? (prev * cur) : (prev + cur));
                }
            }
        }
    }

    template <class T>
    void cum_scan(const Stream& stream,
        Span<T> output, View<T> input,
        std::size_t outer_size, std::size_t target_size, std::size_t inner_size,
        bool is_prod, bool exclusive, bool reverse)
    {
        CV_Assert(outer_size > 0 && target_size > 0 && inner_size > 0);

        const std::size_t nrows = outer_size * inner_size;
        if (nrows == 0)
            return;

        auto kernel = raw::cum_scan<T>;
        auto policy = make_policy(kernel, nrows, 0, stream);
        launch_kernel(kernel, policy, output, input,
                      static_cast<std::int64_t>(outer_size),
                      static_cast<std::int64_t>(target_size),
                      static_cast<std::int64_t>(inner_size),
                      is_prod, exclusive, reverse);
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void cum_scan(const Stream&, Span<__half>, View<__half>, std::size_t, std::size_t, std::size_t, bool, bool, bool);
#endif
    template void cum_scan(const Stream&, Span<float>, View<float>, std::size_t, std::size_t, std::size_t, bool, bool, bool);
    template void cum_scan(const Stream&, Span<int32_t>, View<int32_t>, std::size_t, std::size_t, std::size_t, bool, bool, bool);
    template void cum_scan(const Stream&, Span<int64_t>, View<int64_t>, std::size_t, std::size_t, std::size_t, bool, bool, bool);
    template void cum_scan(const Stream&, Span<int8_t>, View<int8_t>, std::size_t, std::size_t, std::size_t, bool, bool, bool);
    template void cum_scan(const Stream&, Span<uint8_t>, View<uint8_t>, std::size_t, std::size_t, std::size_t, bool, bool, bool);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
