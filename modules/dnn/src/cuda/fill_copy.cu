// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "grid_stride_range.hpp"
#include "execution.hpp"
#include "vector_traits.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {
        template <class T, std::size_t N>
        __global__ void fill_vec(Span<T> output, T value) {
            using vector_type = get_vector_type_t<T, N>;
            auto output_vPtr = vector_type::get_pointer(output.data());
            for (auto i : grid_stride_range(output.size() / vector_type::size())) {
                vector_type vec;
                for (int j = 0; j < vector_type::size(); j++)
                    vec.data[j] = value;
                v_store(output_vPtr[i], vec);
            }
        }

        template <class T, std::size_t N>
        __global__ void  copy_vec(Span<T> output, View<T> input) {
            using vector_type = get_vector_type_t<T, N>;
            auto input_vPtr = vector_type::get_pointer(input.data());
            auto output_vPtr = vector_type::get_pointer(output.data());
            for (auto i : grid_stride_range(output.size() / vector_type::size())) {
                vector_type vec;
                v_load(vec, input_vPtr[i]);
                v_store(output_vPtr[i], vec);
            }
        }
    }

    template <class T, std::size_t N> static
    void launch_vectorized_fill(const Stream& stream, Span<T> output, T value) {
        CV_Assert(is_fully_aligned<T>(output, N));

        auto kernel = raw::fill_vec<T, N>;
        auto policy = make_policy(kernel, output.size() / N, 0, stream);
        launch_kernel(kernel, policy, output, value);
    }

    template <class T>
    void fill(const Stream& stream, Span<T> output, T value) {
        if (is_fully_aligned<T>(output, 4)) {
            launch_vectorized_fill<T, 4>(stream, output, value);
        } else if (is_fully_aligned<T>(output, 2)) {
            launch_vectorized_fill<T, 2>(stream, output, value);
        } else {
            launch_vectorized_fill<T, 1>(stream, output, value);
        }
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void fill(const Stream&, Span<__half>, __half);
#endif
    template void fill(const Stream&, Span<float>, float);
    template void fill(const Stream&, Span<int8_t>, int8_t);
    template void fill(const Stream&, Span<uint8_t>, uint8_t);
    template void fill(const Stream&, Span<int>, int);
    template void fill(const Stream&, Span<int64_t>, int64_t);
    template void fill(const Stream&, Span<bool>, bool);

    template <class T, std::size_t N> static
    void launch_vectorized_copy(const Stream& stream, Span<T> output, View<T> input) {
        CV_Assert(is_fully_aligned<T>(output, N));
        CV_Assert(is_fully_aligned<T>(input, N));

        auto kernel = raw::copy_vec<T, N>;
        auto policy = make_policy(kernel, output.size() / N, 0, stream);
        launch_kernel(kernel, policy, output, input);
    }

    template <class T>
    void copy(const Stream& stream, Span<T> output, View<T> input) {
        if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(input, 4)) {
            launch_vectorized_copy<T, 4>(stream, output, input);
        } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(input, 2)) {
            launch_vectorized_copy<T, 2>(stream, output, input);
        } else {
            launch_vectorized_copy<T, 1>(stream, output, input);
        }
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void copy(const Stream&, Span<__half>, View<__half>);
#endif
    template void copy(const Stream&, Span<float>, View<float>);
    template void copy(const Stream&, Span<int8_t>, View<int8_t>);
    template void copy(const Stream&, Span<uint8_t>, View<uint8_t>);
    template void copy(const Stream&, Span<int32_t>, View<int32_t>);
    template void copy(const Stream&, Span<int64_t>, View<int64_t>);
    template void copy(const Stream&, Span<bool>, View<bool>);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
