// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "functors.hpp"
#include "types.hpp"
#include "vector_traits.hpp"
#include "grid_stride_range.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include "../cuda4dnn/kernels/scale_shift.hpp"

#include <opencv2/core.hpp>

#include <cstddef>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn  { namespace kernels {

namespace raw {
    template <class T, class ActivationOp, std::size_t N>
    __global__ void generic_op_vec(Span<T> output, View<T> input, const typename ActivationOp::Params params) {
        using vector_type = get_vector_type_t<T, N>;

        auto output_vPtr = vector_type::get_pointer(output.data());
        auto input_vPtr = vector_type::get_pointer(input.data());

        ActivationOp activation_op(params);

        for (auto i : grid_stride_range(output.size() / vector_type::size())) {
            vector_type vec;
            v_load(vec, input_vPtr[i]);
            for (int j = 0; j < vector_type::size(); j++)
                vec.data[j] = activation_op(vec.data[j]);
            v_store(output_vPtr[i], vec);
        }
    }

    template <class T, std::size_t N>
    __global__ void axiswise_relu_vec(Span<T> output, View<T> input, size_type inner_size, View<T> slope) {
        using vector_type = get_vector_type_t<T, N>;

        auto output_vPtr = vector_type::get_pointer(output.data());
        auto input_vPtr = vector_type::get_pointer(input.data());

        for (auto i : grid_stride_range(output.size() / vector_type::size())) {
            const index_type c = (i / inner_size) % slope.size();

            vector_type vec;
            v_load(vec, input_vPtr[i]);
            for (int j = 0; j < vector_type::size(); j++)
                vec.data[j] = vec.data[j] > T(0) ? vec.data[j] : vec.data[j] * slope[c];
            v_store(output_vPtr[i], vec);
        }
    }

} /* namespace raw */

template <class T, class ActivationOp, std::size_t N> static
void launch_vectorized_generic_op(const Stream& stream, Span<T> output, View<T> input, const typename ActivationOp::Params& params) {
    CV_Assert(is_fully_aligned<T>(output, N));
    CV_Assert(is_fully_aligned<T>(input, N));

    auto kernel = raw::generic_op_vec<T, ActivationOp, N>;
    auto policy = make_policy(kernel, output.size() / N, 0, stream);
    launch_kernel(kernel, policy, output, input, params);
}

template <class T, class ActivationOp> static
void generic_op(const Stream& stream, Span<T> output, View<T> input, const typename ActivationOp::Params& params = {}) {
    CV_Assert(input.size() == output.size());

    if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(input, 4)) {
        launch_vectorized_generic_op<T, ActivationOp, 4>(stream, output, input, params);
    } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(input, 2)) {
        launch_vectorized_generic_op<T, ActivationOp, 2>(stream, output, input, params);
    } else {
        launch_vectorized_generic_op<T, ActivationOp, 1>(stream, output, input, params);
    }
}

template <class T>
void relu(const Stream& stream, Span<T> output, View<T> input, T slope) {
    generic_op<T, ReLUFunctor<T>>(stream, output, input, {slope});
}

template <class T>
void clipped_relu(const Stream& stream, Span<T> output, View<T> input, T floor, T ceiling) {
    CV_Assert(static_cast<double>(floor) <= static_cast<double>(ceiling));
    generic_op<T, ClippedReLUFunctor<T>>(stream, output, input, {floor, ceiling});
}

template <class T>
void tanh(const Stream& stream, Span<T> output, View<T> input) {
    generic_op<T, TanHFunctor<T>>(stream, output, input);
}

template <class T>
void swish(const Stream& stream, Span<T> output, View<T> input) {
    generic_op<T, SwishFunctor<T>>(stream, output, input);
}

template <class T>
void mish(const Stream& stream, Span<T> output, View<T> input) {
    generic_op<T, MishFunctor<T>>(stream, output, input);
}

template <class T>
void sigmoid(const Stream& stream, Span<T> output, View<T> input) {
    generic_op<T, SigmoidFunctor<T>>(stream, output, input);
}

template <class T>
void elu(const Stream& stream, Span<T> output, View<T> input) {
    generic_op<T, ELUFunctor<T>>(stream, output, input);
}

template <class T>
void bnll(const Stream& stream, Span<T> output, View<T> input) {
    generic_op<T, BNLLFunctor<T>>(stream, output, input);
}

template <class T>
void ceil(const Stream& stream, Span<T> output, View<T> input) {
    generic_op<T, CeilFunctor<T>>(stream, output, input);
}

template <class T>
void floor(const Stream& stream, Span<T> output, View<T> input) {
    generic_op<T, FloorFunctor<T>>(stream, output, input);
}

template <class T>
void log(const Stream& stream, Span<T> output, View<T> input) {
    generic_op<T, LogFunctor<T>>(stream, output, input);
}

template <class T>
void rint(const Stream& stream, Span<T> output, View<T> input) {
    generic_op<T, RintFunctor<T>>(stream, output, input);
}

template <class T>
void sqrt(const Stream& stream, Span<T> output, View<T> input) {
    generic_op<T, SqrtFunctor<T>>(stream, output, input);
}

template <class T>
void not_k(const Stream& stream, Span<T> output, View<T> input) {
    generic_op<T, NotFunctor<T>>(stream, output, input);
}

template <class T>
void abs(const Stream& stream, Span<T> output, View<T> input) {
    generic_op<T, AbsFunctor<T>>(stream, output, input);
}

template <class T>
void power(const Stream& stream, Span<T> output, View<T> input, T exp, T scale, T shift) {
    CV_Assert(input.size() == output.size());

    if (static_cast<float>(exp) == 1.0f) {
        scale1_with_bias1(stream, output, input, scale, shift);
        return;
    }

    generic_op<T, PowerFunctor<T>>(stream, output, input, {exp, scale, shift});
}

template <class T>
void exp(const Stream& stream, Span<T> output, View<T> input, T normScale, T normShift) {
    generic_op<T, ExpFunctor<T>>(stream, output, input, {normScale, normShift});
}

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
template void relu<__half>(const Stream&, Span<__half>, View<__half>, __half);
template void clipped_relu<__half>(const Stream&, Span<__half>, View<__half>, __half, __half);
template void tanh<__half>(const Stream&, Span<__half>, View<__half>);
template void swish<__half>(const Stream&, Span<__half>, View<__half>);
template void mish<__half>(const Stream&, Span<__half>, View<__half>);
template void sigmoid<__half>(const Stream&, Span<__half>, View<__half>);
template void elu<__half>(const Stream&, Span<__half>, View<__half>);
template void abs<__half>(const Stream& stream, Span<__half> output, View<__half> input);
template void bnll<__half>(const Stream&, Span<__half>, View<__half>);
template void ceil<__half>(const Stream&, Span<__half>, View<__half>);
template void floor<__half>(const Stream&, Span<__half>, View<__half>);
template void log<__half>(const Stream&, Span<__half>, View<__half>);
template void rint<__half>(const Stream&, Span<__half>, View<__half>);
template void sqrt<__half>(const Stream&, Span<__half>, View<__half>);
template void not_k<__half>(const Stream&, Span<__half>, View<__half>);
template void power<__half>(const Stream&, Span<__half>, View<__half>, __half, __half, __half);
template void exp<__half>(const Stream&, Span<__half>, View<__half>, __half, __half);
#endif


template void relu<float>(const Stream&, Span<float>, View<float>, float);
template void clipped_relu<float>(const Stream&, Span<float>, View<float>, float, float);
template void tanh<float>(const Stream&, Span<float>, View<float>);
template void swish<float>(const Stream&, Span<float>, View<float>);
template void mish<float>(const Stream&, Span<float>, View<float>);
template void sigmoid<float>(const Stream&, Span<float>, View<float>);
template void elu<float>(const Stream&, Span<float>, View<float>);
template void abs<float>(const Stream& stream, Span<float> output, View<float> input);
template void bnll<float>(const Stream&, Span<float>, View<float>);
template void ceil<float>(const Stream&, Span<float>, View<float>);
template void floor<float>(const Stream&, Span<float>, View<float>);
template void log<float>(const Stream&, Span<float>, View<float>);
template void rint<float>(const Stream&, Span<float>, View<float>);
template void sqrt<float>(const Stream&, Span<float>, View<float>);
template void not_k<float>(const Stream&, Span<float>, View<float>);
template void power<float>(const Stream&, Span<float>, View<float>, float, float, float);
template void exp<float>(const Stream&, Span<float>, View<float>, float, float);

template <class T, std::size_t N> static
void launch_vectorized_axiswise_relu(const Stream& stream, Span<T> output, View<T> input, std::size_t inner_size, View<T> slope) {
    CV_Assert(is_fully_aligned<T>(output, N));
    CV_Assert(is_fully_aligned<T>(input, N));
    CV_Assert(inner_size % N == 0);

    auto kernel = raw::axiswise_relu_vec<T, N>;
    auto policy = make_policy(kernel, output.size() / N, 0, stream);
    launch_kernel(kernel, policy, output, input, inner_size / N, slope);
}

template <class T>
void axiswise_relu(const Stream& stream, Span<T> output, View<T> input, std::size_t inner_size, View<T> slope) {
    CV_Assert(input.size() == output.size());

    if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(input, 4) && inner_size % 4 == 0) {
        launch_vectorized_axiswise_relu<T, 4>(stream, output, input, inner_size, slope);
    } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(input, 2) && inner_size % 2 == 0) {
        launch_vectorized_axiswise_relu<T, 2>(stream, output, input, inner_size, slope);
    } else {
        launch_vectorized_axiswise_relu<T, 1>(stream, output, input, inner_size, slope);
    }
}

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void axiswise_relu<__half>(const Stream&, Span<__half>, View<__half>, std::size_t, View<__half>);
#endif
    template void axiswise_relu<float>(const Stream&, Span<float>, View<float>, std::size_t, View<float>);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
