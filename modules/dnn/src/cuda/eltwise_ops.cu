// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "math.hpp"
#include "grid_stride_range.hpp"
#include "execution.hpp"
#include "vector_traits.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include <opencv2/core.hpp>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {
        template <class T, std::size_t N>
        __global__ void eltwise_max_2_vec(Span<T> output, View<T> x, View<T> y) {
            using vector_type = get_vector_type_t<T, N>;

            auto output_vPtr = vector_type::get_pointer(output.data());
            auto x_vPtr = vector_type::get_pointer(x.data());
            auto y_vPtr = vector_type::get_pointer(y.data());

            for (auto i : grid_stride_range(output.size() / vector_type::size())) {
                vector_type vec_x, vec_y;
                v_load(vec_x, x_vPtr[i]);
                v_load(vec_y, y_vPtr[i]);

                for (int j = 0; j < vector_type::size(); j++) {
                    using device::max;
                    vec_x.data[j] = max(vec_x.data[j], vec_y.data[j]);
                }

                v_store(output_vPtr[i], vec_x);
            }
        }

        template <class T, std::size_t N>
        __global__ void eltwise_sum_2_vec(Span<T> output, View<T> x, View<T> y) {
            using vector_type = get_vector_type_t<T, N>;

            auto output_vPtr = vector_type::get_pointer(output.data());
            auto x_vPtr = vector_type::get_pointer(x.data());
            auto y_vPtr = vector_type::get_pointer(y.data());

            for (auto i : grid_stride_range(output.size() / vector_type::size())) {
                vector_type vec_x, vec_y;
                v_load(vec_x, x_vPtr[i]);
                v_load(vec_y, y_vPtr[i]);

                for (int j = 0; j < vector_type::size(); j++)
                    vec_x.data[j] = vec_x.data[j] + vec_y.data[j];

                v_store(output_vPtr[i], vec_x);
            }
        }

        template <class T, std::size_t N>
        __global__ void eltwise_sum_coeff_2_vec(Span<T> output, T coeff_x, View<T> x, T coeff_y, View<T> y) {
            using vector_type = get_vector_type_t<T, N>;

            auto output_vPtr = vector_type::get_pointer(output.data());
            auto x_vPtr = vector_type::get_pointer(x.data());
            auto y_vPtr = vector_type::get_pointer(y.data());

            for (auto i : grid_stride_range(output.size() / vector_type::size())) {
                vector_type vec_x, vec_y;
                v_load(vec_x, x_vPtr[i]);
                v_load(vec_y, y_vPtr[i]);

                for (int j = 0; j < vector_type::size(); j++)
                    vec_x.data[j] = coeff_x * vec_x.data[j] + coeff_y * vec_y.data[j];

                v_store(output_vPtr[i], vec_x);
            }
        }

        template <class T, std::size_t N>
        __global__ void eltwise_prod_2_vec(Span<T> output, View<T> x, View<T> y) {
            using vector_type = get_vector_type_t<T, N>;

            auto output_vPtr = vector_type::get_pointer(output.data());
            auto x_vPtr = vector_type::get_pointer(x.data());
            auto y_vPtr = vector_type::get_pointer(y.data());

            for (auto i : grid_stride_range(output.size() / vector_type::size())) {
                vector_type vec_x, vec_y;
                v_load(vec_x, x_vPtr[i]);
                v_load(vec_y, y_vPtr[i]);

                for (int j = 0; j < vector_type::size(); j++)
                    vec_x.data[j] = vec_x.data[j] * vec_y.data[j];

                v_store(output_vPtr[i], vec_x);
            }
        }

        template <class T, std::size_t N>
        __global__ void eltwise_div_2_vec(Span<T> output, View<T> x, View<T> y) {
            using vector_type = get_vector_type_t<T, N>;

            auto output_vPtr = vector_type::get_pointer(output.data());
            auto x_vPtr = vector_type::get_pointer(x.data());
            auto y_vPtr = vector_type::get_pointer(y.data());

            for (auto i : grid_stride_range(output.size() / vector_type::size())) {
                vector_type vec_x, vec_y;
                v_load(vec_x, x_vPtr[i]);
                v_load(vec_y, y_vPtr[i]);

                for (int j = 0; j < vector_type::size(); j++)
                    vec_x.data[j] = vec_x.data[j] / vec_y.data[j];

                v_store(output_vPtr[i], vec_x);
            }
        }
    }

    template <class T, std::size_t N>
    void launch_vectorized_eltwise_max_2(const Stream& stream, Span<T> output, View<T> x, View<T> y) {
        CV_Assert(is_fully_aligned<T>(output, N));
        CV_Assert(is_fully_aligned<T>(x, N));
        CV_Assert(is_fully_aligned<T>(y, N));

        auto kernel = raw::eltwise_max_2_vec<T, N>;
        auto policy = make_policy(kernel, output.size() / N, 0, stream);
        launch_kernel(kernel, policy, output, x, y);
    }

    template <class T>
    void eltwise_max_2(const Stream& stream, Span<T> output, View<T> x, View<T> y) {
        CV_Assert(x.size() == y.size());
        CV_Assert(x.size() == output.size());

        if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(x, 4) && is_fully_aligned<T>(y, 4)) {
            launch_vectorized_eltwise_max_2<T, 4>(stream, output, x, y);
        } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(x, 2) && is_fully_aligned<T>(y, 2)) {
            launch_vectorized_eltwise_max_2<T, 2>(stream, output, x, y);
        } else {
            launch_vectorized_eltwise_max_2<T, 1>(stream, output, x, y);
        }
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void eltwise_max_2(const Stream& stream, Span<__half> output, View<__half> x, View<__half> y);
#endif
    template void eltwise_max_2(const Stream& stream, Span<float> output, View<float> x, View<float> y);

    template <class T, std::size_t N>
    void launch_vectorized_eltwise_sum_2(const Stream& stream, Span<T> output, View<T> x, View<T> y) {
        CV_Assert(is_fully_aligned<T>(output, N));
        CV_Assert(is_fully_aligned<T>(x, N));
        CV_Assert(is_fully_aligned<T>(y, N));

        auto kernel = raw::eltwise_sum_2_vec<T, N>;
        auto policy = make_policy(kernel, output.size() / N, 0, stream);
        launch_kernel(kernel, policy, output, x, y);
    }

    template <class T>
    void eltwise_sum_2(const Stream& stream, Span<T> output, View<T> x, View<T> y) {
        CV_Assert(x.size() == y.size());
        CV_Assert(x.size() == output.size());

        if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(x, 4) && is_fully_aligned<T>(y, 4)) {
            launch_vectorized_eltwise_sum_2<T, 4>(stream, output, x, y);
        } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(x, 2) && is_fully_aligned<T>(y, 2)) {
            launch_vectorized_eltwise_sum_2<T, 2>(stream, output, x, y);
        } else {
            launch_vectorized_eltwise_sum_2<T, 1>(stream, output, x, y);
        }
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void eltwise_sum_2(const Stream& stream, Span<__half> output, View<__half> x, View<__half> y);
#endif
    template void eltwise_sum_2(const Stream& stream, Span<float> output, View<float> x, View<float> y);

    template <class T, std::size_t N>
    void launch_vectorized_eltwise_sum_coeff_2(const Stream& stream, Span<T> output, T coeff_x, View<T> x, T coeff_y, View<T> y) {
        CV_Assert(is_fully_aligned<T>(output, N));
        CV_Assert(is_fully_aligned<T>(x, N));
        CV_Assert(is_fully_aligned<T>(y, N));

        auto kernel = raw::eltwise_sum_coeff_2_vec<T, N>;
        auto policy = make_policy(kernel, output.size() / N, 0, stream);
        launch_kernel(kernel, policy, output, coeff_x, x, coeff_y, y);
    }

    template <class T>
    void eltwise_sum_coeff_2(const Stream& stream, Span<T> output, T coeff_x, View<T> x, T coeff_y, View<T> y) {
        CV_Assert(x.size() == y.size());
        CV_Assert(x.size() == output.size());

        if (static_cast<float>(coeff_x) == 1.0f && static_cast<float>(coeff_y) == 1.0f) {
            eltwise_sum_2(stream, output, x, y);
            return;
        }

        if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(x, 4) && is_fully_aligned<T>(y, 4)) {
            launch_vectorized_eltwise_sum_coeff_2<T, 4>(stream, output, coeff_x, x, coeff_y, y);
        } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(x, 2) && is_fully_aligned<T>(y, 2)) {
            launch_vectorized_eltwise_sum_coeff_2<T, 2>(stream, output, coeff_x, x, coeff_y, y);
        } else {
            launch_vectorized_eltwise_sum_coeff_2<T, 1>(stream, output, coeff_x, x, coeff_y, y);
        }
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void eltwise_sum_coeff_2(const Stream&, Span<__half>, __half, View<__half>, __half, View<__half>);
#endif
    template void eltwise_sum_coeff_2(const Stream&, Span<float>, float, View<float>, float, View<float>);

    template <class T, std::size_t N>
    void launch_vectorized_eltwise_prod_2(const Stream& stream, Span<T> output, View<T> x, View<T> y) {
        CV_Assert(is_fully_aligned<T>(output, N));
        CV_Assert(is_fully_aligned<T>(x, N));
        CV_Assert(is_fully_aligned<T>(y, N));

        auto kernel = raw::eltwise_prod_2_vec<T, N>;
        auto policy = make_policy(kernel, output.size() / N, 0, stream);
        launch_kernel(kernel, policy, output, x, y);
    }

    template <class T>
    void eltwise_prod_2(const Stream& stream, Span<T> output, View<T> x, View<T> y) {
        CV_Assert(x.size() == y.size());
        CV_Assert(x.size() == output.size());

        if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(x, 4) && is_fully_aligned<T>(y, 4)) {
            launch_vectorized_eltwise_prod_2<T, 4>(stream, output, x, y);
        } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(x, 2) && is_fully_aligned<T>(y, 2)) {
            launch_vectorized_eltwise_prod_2<T, 2>(stream, output, x, y);
        } else {
            launch_vectorized_eltwise_prod_2<T, 1>(stream, output, x, y);
        }
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void eltwise_prod_2(const Stream& stream, Span<__half> output, View<__half> x, View<__half> y);
#endif
    template void eltwise_prod_2(const Stream& stream, Span<float> output, View<float> x, View<float> y);

    template <class T, std::size_t N>
    void launch_vectorized_eltwise_div_2(const Stream& stream, Span<T> output, View<T> x, View<T> y) {
        CV_Assert(is_fully_aligned<T>(output, N));
        CV_Assert(is_fully_aligned<T>(x, N));
        CV_Assert(is_fully_aligned<T>(y, N));

        auto kernel = raw::eltwise_div_2_vec<T, N>;
        auto policy = make_policy(kernel, output.size() / N, 0, stream);
        launch_kernel(kernel, policy, output, x, y);
    }

    template <class T>
    void eltwise_div_2(const Stream& stream, Span<T> output, View<T> x, View<T> y) {
        CV_Assert(x.size() == y.size());
        CV_Assert(x.size() == output.size());

        if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(x, 4) && is_fully_aligned<T>(y, 4)) {
            launch_vectorized_eltwise_div_2<T, 4>(stream, output, x, y);
        } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(x, 2) && is_fully_aligned<T>(y, 2)) {
            launch_vectorized_eltwise_div_2<T, 2>(stream, output, x, y);
        } else {
            launch_vectorized_eltwise_div_2<T, 1>(stream, output, x, y);
        }
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void eltwise_div_2(const Stream& stream, Span<__half> output, View<__half> x, View<__half> y);
#endif
    template void eltwise_div_2(const Stream& stream, Span<float> output, View<float> x, View<float> y);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
