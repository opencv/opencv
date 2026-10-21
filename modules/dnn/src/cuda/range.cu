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

#include "../cuda4dnn/kernels/range.hpp"

#include <cstdint>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {
        __device__ inline double as_double(float v)         { return static_cast<double>(v); }
        __device__ inline double as_double(__half v)        { return static_cast<double>(__half2float(v)); }
        __device__ inline double as_double(std::int8_t v)   { return static_cast<double>(v); }
        __device__ inline double as_double(std::uint8_t v)  { return static_cast<double>(v); }
        __device__ inline double as_double(std::int32_t v)  { return static_cast<double>(v); }
        __device__ inline double as_double(std::int64_t v)  { return static_cast<double>(v); }

        template <class T>
        __global__ void range(Span<T> output, const T* start_p, const T* delta_p)
        {
            const double start = as_double(*start_p);
            const double delta = as_double(*delta_p);
            for (auto i : grid_stride_range(output.size()))
                output[i] = static_cast<T>(start + static_cast<double>(i) * delta);
        }
    }

    template <class T>
    void range(const Stream& stream, Span<T> output, const T* start, const T* delta)
    {
        if (output.size() == 0)
            return;
        auto kernel = raw::range<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, start, delta);
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void range(const Stream&, Span<__half>, const __half*, const __half*);
#endif
    template void range(const Stream&, Span<float>, const float*, const float*);
    template void range(const Stream&, Span<int8_t>, const int8_t*, const int8_t*);
    template void range(const Stream&, Span<uint8_t>, const uint8_t*, const uint8_t*);
    template void range(const Stream&, Span<int32_t>, const int32_t*, const int32_t*);
    template void range(const Stream&, Span<int64_t>, const int64_t*, const int64_t*);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
