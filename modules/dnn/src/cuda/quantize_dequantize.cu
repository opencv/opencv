// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>

#include "grid_stride_range.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include "../cuda4dnn/kernels/quantize_dequantize.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {

        template <class T>
        __global__ void dequantize_linear(const T* input, const float* scale, const T* zero_point,
                                           float* output, std::size_t sz_a, std::size_t slice_size, std::size_t n)
        {
            for (auto i : grid_stride_range(n)) {
                std::size_t a_idx = (i / slice_size) % sz_a;
                T zp = zero_point ? zero_point[a_idx] : T(0);
                output[i] = (static_cast<float>(input[i]) - static_cast<float>(zp)) * scale[a_idx];
            }
        }

        template <class T> __device__ __forceinline__ T saturate_quantized(int v);
        template <> __device__ __forceinline__ std::int8_t saturate_quantized<std::int8_t>(int v) {
            return static_cast<std::int8_t>(::min(::max(v, -128), 127));
        }
        template <> __device__ __forceinline__ std::uint8_t saturate_quantized<std::uint8_t>(int v) {
            return static_cast<std::uint8_t>(::min(::max(v, 0), 255));
        }

        template <class T>
        __global__ void quantize_linear(const float* input, const float* scale, const T* zero_point,
                                         T* output, std::size_t sz_a, std::size_t slice_size, std::size_t n)
        {
            for (auto i : grid_stride_range(n)) {
                std::size_t a_idx = (i / slice_size) % sz_a;
                int zp = zero_point ? static_cast<int>(zero_point[a_idx]) : 0;
                int rounded = __float2int_rn(input[i] / scale[a_idx]);
                output[i] = saturate_quantized<T>(rounded + zp);
            }
        }

    }

    template <class T>
    void dequantize_linear(const Stream& stream, Span<float> output, View<T> input,
                            View<float> scale, View<T> zero_point,
                            std::size_t sz_a, std::size_t slice_size)
    {
        CV_Assert(output.size() == input.size());
        CV_Assert(zero_point.empty() || zero_point.size() == scale.size());

        auto kernel = raw::dequantize_linear<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, input.data().get(), scale.data().get(),
                      zero_point.empty() ? nullptr : zero_point.data().get(),
                      output.data().get(), sz_a, slice_size, output.size());
    }

    template <class T>
    void quantize_linear(const Stream& stream, Span<T> output, View<float> input,
                          View<float> scale, View<T> zero_point,
                          std::size_t sz_a, std::size_t slice_size)
    {
        CV_Assert(output.size() == input.size());
        CV_Assert(zero_point.empty() || zero_point.size() == scale.size());

        auto kernel = raw::quantize_linear<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, input.data().get(), scale.data().get(),
                      zero_point.empty() ? nullptr : zero_point.data().get(),
                      output.data().get(), sz_a, slice_size, output.size());
    }

    template void dequantize_linear<std::int8_t>(const Stream&, Span<float>, View<std::int8_t>, View<float>, View<std::int8_t>, std::size_t, std::size_t);
    template void dequantize_linear<std::uint8_t>(const Stream&, Span<float>, View<std::uint8_t>, View<float>, View<std::uint8_t>, std::size_t, std::size_t);
    template void quantize_linear<std::int8_t>(const Stream&, Span<std::int8_t>, View<float>, View<float>, View<std::int8_t>, std::size_t, std::size_t);
    template void quantize_linear<std::uint8_t>(const Stream&, Span<std::uint8_t>, View<float>, View<float>, View<std::uint8_t>, std::size_t, std::size_t);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
