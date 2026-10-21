// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "types.hpp"
#include "grid_stride_range.hpp"
#include "math.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include "../cuda4dnn/kernels/trilu.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cstdint>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {

        template <class T>
        __global__ void trilu(Span<T> output, View<T> input,
                              int height, int width, int m, int k, bool upper)
        {
            const T* in = input.data().get();
            T* out = output.data().get();

            for (auto id : grid_stride_range(output.size())) {
                const int c = id % width;
                const int l = (id / width) % height;

                bool zero = false;
                /* Rows at or past min(height, width) are untouched by the CPU path. */
                if (l < m) {
                    if (upper) {
                        // zeroes columns [0, min(l + k - 1, width - 1)]
                        const int cmax = device::min<int>(l + k - 1, width - 1);
                        zero = (c <= cmax);
                    } else {
                        // zeroes columns [max(l + k + 1, 0), width - 1]
                        const int cmin = device::max<int>(l + k + 1, 0);
                        zero = (c >= cmin);
                    }
                }

                out[id] = zero ? static_cast<T>(0.f) : in[id];
            }
        }
    }

    template <class T>
    void trilu(const Stream& stream,
        Span<T> output, View<T> input,
        int loops, int height, int width, int k, bool upper)
    {
        CV_Assert(loops > 0 && height > 0 && width > 0);
        CV_UNUSED(loops);

        if (output.size() == 0)
            return;

        const int m = std::min(height, width);

        auto kernel = raw::trilu<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, input, height, width, m, k, upper);
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void trilu(const Stream&, Span<__half>, View<__half>, int, int, int, int, bool);
#endif
    template void trilu(const Stream&, Span<float>, View<float>, int, int, int, int, bool);
    template void trilu(const Stream&, Span<int32_t>, View<int32_t>, int, int, int, int, bool);
    template void trilu(const Stream&, Span<int64_t>, View<int64_t>, int, int, int, int, bool);
    template void trilu(const Stream&, Span<int8_t>, View<int8_t>, int, int, int, int, bool);
    template void trilu(const Stream&, Span<uint8_t>, View<uint8_t>, int, int, int, int, bool);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
