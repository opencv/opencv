// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>

#include <cub/block/block_reduce.cuh>

#include "grid_stride_range.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include "../cuda4dnn/kernels/quantize_dequantize.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
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

        // 256 threads, as the module's other block reductions (rms_norm.cu, topk.cu, reduce.cu, mvn.cu)
        constexpr int DYNAMIC_QUANTIZE_BLOCK = 256;

        struct MinOp { __device__ float operator()(float a, float b) const { return fminf(a, b); } };
        struct MaxOp { __device__ float operator()(float a, float b) const { return fmaxf(a, b); } };

        // one (min, max) pair per block; zero starts both, since the range must include it
        __global__ void dynamic_quantize_minmax(const float* input, size_type n, float* pmin, float* pmax)
        {
            using BlockReduce = cub::BlockReduce<float, DYNAMIC_QUANTIZE_BLOCK>;
            __shared__ typename BlockReduce::TempStorage storage;

            float lo = 0.f, hi = 0.f;
            for (size_type i = blockIdx.x * DYNAMIC_QUANTIZE_BLOCK + threadIdx.x; i < n; i += gridDim.x * DYNAMIC_QUANTIZE_BLOCK) {
                lo = fminf(lo, input[i]);
                hi = fmaxf(hi, input[i]);
            }
            lo = BlockReduce(storage).Reduce(lo, MinOp());
            __syncthreads();
            hi = BlockReduce(storage).Reduce(hi, MaxOp());
            if (threadIdx.x == 0) {
                pmin[blockIdx.x] = lo;
                pmax[blockIdx.x] = hi;
            }
        }

        // folds the partials and derives the scale and zero point exactly as the CPU layer does
        __global__ void dynamic_quantize_params(const float* pmin, const float* pmax, int nparts,
                                                float* scale, std::uint8_t* zero_point)
        {
            using BlockReduce = cub::BlockReduce<float, DYNAMIC_QUANTIZE_BLOCK>;
            __shared__ typename BlockReduce::TempStorage storage;

            float lo = 0.f, hi = 0.f;
            for (int i = threadIdx.x; i < nparts; i += DYNAMIC_QUANTIZE_BLOCK) {
                lo = fminf(lo, pmin[i]);
                hi = fmaxf(hi, pmax[i]);
            }
            lo = BlockReduce(storage).Reduce(lo, MinOp());
            __syncthreads();
            hi = BlockReduce(storage).Reduce(hi, MaxOp());
            if (threadIdx.x == 0) {
                const float qmin = 0.f, qmax = 255.f;
                float y_scale = (hi - lo) / (qmax - qmin);
                if (y_scale == 0.f)
                    y_scale = 1.f;
                const int zp = __float2int_rn(qmin - lo / y_scale);
                scale[0] = y_scale;
                zero_point[0] = saturate_quantized<std::uint8_t>(zp);
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

    std::size_t dynamic_quantize_partials(std::size_t n)
    {
        const std::size_t block = raw::DYNAMIC_QUANTIZE_BLOCK;
        return std::max<std::size_t>(1, std::min<std::size_t>((n + block - 1) / block, block));
    }

    void dynamic_quantize_scale(const Stream& stream, Span<float> scale, Span<std::uint8_t> zero_point,
                                View<float> input, Span<float> scratch)
    {
        CV_Assert(scale.size() == 1 && zero_point.size() == 1);
        const std::size_t nparts = dynamic_quantize_partials(input.size());
        CV_Assert(scratch.size() >= 2 * nparts);
        float* pmin = scratch.data().get();
        float* pmax = pmin + nparts;

        auto minmax = raw::dynamic_quantize_minmax;
        launch_kernel(minmax, execution_policy(nparts, raw::DYNAMIC_QUANTIZE_BLOCK, stream),
                      input.data().get(), input.size(), pmin, pmax);
        auto params = raw::dynamic_quantize_params;
        launch_kernel(params, execution_policy(1, raw::DYNAMIC_QUANTIZE_BLOCK, stream),
                      pmin, pmax, static_cast<int>(nparts), scale.data().get(), zero_point.data().get());
    }

    template void dequantize_linear<std::int8_t>(const Stream&, Span<float>, View<std::int8_t>, View<float>, View<std::int8_t>, std::size_t, std::size_t);
    template void dequantize_linear<std::uint8_t>(const Stream&, Span<float>, View<std::uint8_t>, View<float>, View<std::uint8_t>, std::size_t, std::size_t);
    template void quantize_linear<std::int8_t>(const Stream&, Span<std::int8_t>, View<float>, View<float>, View<std::int8_t>, std::size_t, std::size_t);
    template void quantize_linear<std::uint8_t>(const Stream&, Span<std::uint8_t>, View<float>, View<float>, View<std::uint8_t>, std::size_t, std::size_t);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
