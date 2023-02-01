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
        template <std::size_t N>
        __global__ void fp32_to_fp16(Span<__half> output, View<float> input) {
            using output_vector_type = get_vector_type_t<__half, N>;
            using input_vector_type = get_vector_type_t<float, N>;

            auto output_vPtr = output_vector_type::get_pointer(output.data());
            auto input_vPtr = input_vector_type::get_pointer(input.data());

            for (auto i : grid_stride_range(output.size() / output_vector_type::size())) {
                input_vector_type in_vec;
                v_load(in_vec, input_vPtr[i]);

                output_vector_type out_vec;
                for (int j = 0; j < output_vector_type::size(); j++)
                    out_vec.data[j] = __float2half(in_vec.data[j]);

                v_store(output_vPtr[i], out_vec);
            }
        }

        template <std::size_t N>
        __global__ void fp16_to_fp32(Span<float> output, View<__half> input) {
            using output_vector_type = get_vector_type_t<float, N>;
            using input_vector_type = get_vector_type_t<__half, N>;

            auto output_vPtr = output_vector_type::get_pointer(output.data());
            auto input_vPtr = input_vector_type::get_pointer(input.data());

            for (auto i : grid_stride_range(output.size() / output_vector_type::size())) {
                input_vector_type in_vec;
                v_load(in_vec, input_vPtr[i]);

                output_vector_type out_vec;
                for (int j = 0; j < output_vector_type::size(); j++)
                    out_vec.data[j] = __half2float(in_vec.data[j]);

                v_store(output_vPtr[i], out_vec);
            }
        }
    }

    template <std::size_t N> static
    void launch_vectorized_fp32_to_fp16(const Stream& stream, Span<__half> output, View<float> input) {
        CV_Assert(is_fully_aligned<__half>(output, N));
        CV_Assert(is_fully_aligned<float>(input, N));

        auto kernel = raw::fp32_to_fp16<N>;
        auto policy = make_policy(kernel, output.size() / N, 0, stream);
        launch_kernel(kernel, policy, output, input);
    }

    void fp32_to_fp16(const Stream& stream, Span<__half> output, View<float> input) {
        if (is_fully_aligned<__half>(output, 4) && is_fully_aligned<float>(input, 4)) {
            launch_vectorized_fp32_to_fp16<4>(stream, output, input);
        } else if (is_fully_aligned<__half>(output, 2) && is_fully_aligned<float>(input, 2)) {
            launch_vectorized_fp32_to_fp16<2>(stream, output, input);
        } else {
            launch_vectorized_fp32_to_fp16<1>(stream, output, input);
        }
    }

    template <std::size_t N> static
    void launch_vectorized_fp16_to_fp32(const Stream& stream, Span<float> output, View<__half> input) {
        CV_Assert(is_fully_aligned<float>(output, N));
        CV_Assert(is_fully_aligned<__half>(input, N));

        auto kernel = raw::fp16_to_fp32<N>;
        auto policy = make_policy(kernel, output.size() / N, 0, stream);
        launch_kernel(kernel, policy, output, input);
    }

    void fp16_to_fp32(const Stream& stream, Span<float> output, View<__half> input) {
        if (is_fully_aligned<float>(output, 4) && is_fully_aligned<__half>(input, 4)) {
            launch_vectorized_fp16_to_fp32<4>(stream, output, input);
        } else if (is_fully_aligned<float>(output, 2) && is_fully_aligned<__half>(input, 2)) {
            launch_vectorized_fp16_to_fp32<2>(stream, output, input);
        } else {
            launch_vectorized_fp16_to_fp32<1>(stream, output, input);
        }
    }

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
