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
    template <class T, class Functor, std::size_t N, class ...FunctorArgs>
    __global__ void generic_op_vec(Span<T> output, View<T> input, FunctorArgs ...functorArgs) {
        using vector_type = get_vector_type_t<T, N>;

        auto output_vPtr = vector_type::get_pointer(output.data());
        auto input_vPtr = vector_type::get_pointer(input.data());

        Functor functor(functorArgs...);

        for (auto i : grid_stride_range(output.size() / vector_type::size())) {
            vector_type vec;
            v_load(vec, input_vPtr[i]);
            for (int j = 0; j < vector_type::size(); j++)
                vec.data[j] = functor(vec.data[j]);
            v_store(output_vPtr[i], vec);
        }
    }

    template <class T, std::size_t N>
    __global__ void axiswise_relu_vec(Span<T> output, View<T> input, size_type inner_size, View<T> slope) {
        using vector_type = get_vector_type_t<T, N>;

        auto output_vPtr = vector_type::get_pointer(output.data());
        auto input_vPtr = vector_type::get_pointer(input.data());

        inner_size /= vector_type::size();
        for (auto i : grid_stride_range(output.size() / vector_type::size())) {
            const index_type c = (i / inner_size) % static_cast<size_type>(slope.size());

            vector_type vec;
            v_load(vec, input_vPtr[i]);
            for (int j = 0; j < vector_type::size(); j++)
                vec.data[j] = vec.data[j] > T(0) ? vec.data[j] : vec.data[j] * slope[c];
            v_store(output_vPtr[i], vec);
        }
    }

} /* namespace raw */

template <class T, template <class> class Activation, std::size_t N, class ...ActivationArgs> static
void launch_vectorized_generic_op(const Stream& stream, Span<T> output, View<T> input, ActivationArgs ...activationArgs) {
    CV_Assert(is_fully_aligned<T>(output, N));
    CV_Assert(is_fully_aligned<T>(input, N));

    auto kernel = raw::generic_op_vec<T, Activation<T>, N, ActivationArgs...>;
    auto policy = make_policy(kernel, output.size() / N, 0, stream);
    launch_kernel(kernel, policy, output, input, activationArgs...);
}

template <class T, template <class> class Activation, class ...ActivationArgs> static
void generic_op(const Stream& stream, Span<T> output, View<T> input, ActivationArgs ...activationArgs) {
    CV_Assert(input.size() == output.size());

    if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(input, 4)) {
        launch_vectorized_generic_op<T, Activation, 4>(stream, output, input, activationArgs...);
    } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(input, 2)) {
        launch_vectorized_generic_op<T, Activation, 2>(stream, output, input, activationArgs...);
    } else {
        launch_vectorized_generic_op<T, Activation, 1>(stream, output, input, activationArgs...);
    }
}

template <class T>
void abs(const Stream& stream, Span<T> output, View<T> input) {
    generic_op<T, abs_functor>(stream, output, input);
}

template <class T>
void tanh(const Stream& stream, Span<T> output, View<T> input) {
    generic_op<T, tanh_functor>(stream, output, input);
}

template <class T>
void swish(const Stream& stream, Span<T> output, View<T> input) {
    generic_op<T, swish_functor>(stream, output, input);
}

template <class T>
void mish(const Stream& stream, Span<T> output, View<T> input) {
    generic_op<T, mish_functor>(stream, output, input);
}

template <class T>
void sigmoid(const Stream& stream, Span<T> output, View<T> input) {
    generic_op<T, sigmoid_functor>(stream, output, input);
}

template <class T>
void bnll(const Stream& stream, Span<T> output, View<T> input) {
    generic_op<T, bnll_functor>(stream, output, input);
}

template <class T>
void elu(const Stream& stream, Span<T> output, View<T> input) {
    generic_op<T, elu_functor>(stream, output, input);
}

template <class T>
void relu(const Stream& stream, Span<T> output, View<T> input, T slope) {
    generic_op<T, relu_functor>(stream, output, input, slope);
}

template <class T>
void clipped_relu(const Stream& stream, Span<T> output, View<T> input, T floor, T ceiling) {
    CV_Assert(static_cast<double>(floor) <= static_cast<double>(ceiling));
    generic_op<T, clipped_relu_functor>(stream, output, input, floor, ceiling);
}

template <class T>
void power(const Stream& stream, Span<T> output, View<T> input, T exp, T scale, T shift) {
    CV_Assert(input.size() == output.size());

    if (static_cast<float>(exp) == 1.0f) {
        scale1_with_bias1(stream, output, input, scale, shift);
        return;
    }

    generic_op<T, power_functor>(stream, output, input, exp, scale, shift);
}

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
template void abs<__half>(const Stream& stream, Span<__half> output, View<__half> input);
template void tanh<__half>(const Stream&, Span<__half>, View<__half>);
template void swish<__half>(const Stream&, Span<__half>, View<__half>);
template void mish<__half>(const Stream&, Span<__half>, View<__half>);
template void sigmoid<__half>(const Stream&, Span<__half>, View<__half>);
template void bnll<__half>(const Stream&, Span<__half>, View<__half>);
template void elu<__half>(const Stream&, Span<__half>, View<__half>);
template void relu<__half>(const Stream&, Span<__half>, View<__half>, __half);
template void clipped_relu<__half>(const Stream&, Span<__half>, View<__half>, __half, __half);
template void power<__half>(const Stream&, Span<__half>, View<__half>, __half, __half, __half);
#endif

template void abs<float>(const Stream& stream, Span<float> output, View<float> input);
template void tanh<float>(const Stream&, Span<float>, View<float>);
template void swish<float>(const Stream&, Span<float>, View<float>);
template void mish<float>(const Stream&, Span<float>, View<float>);
template void sigmoid<float>(const Stream&, Span<float>, View<float>);
template void bnll<float>(const Stream&, Span<float>, View<float>);
template void elu<float>(const Stream&, Span<float>, View<float>);
template void relu<float>(const Stream&, Span<float>, View<float>, float);
template void clipped_relu<float>(const Stream&, Span<float>, View<float>, float, float);
template void power<float>(const Stream&, Span<float>, View<float>, float, float, float);

template <class T, std::size_t N> static
void launch_vectorized_axiswise_relu(const Stream& stream, Span<T> output, View<T> input, std::size_t inner_size, View<T> slope) {
    CV_Assert(is_fully_aligned<T>(output, N));
    CV_Assert(is_fully_aligned<T>(input, N));
    CV_Assert(inner_size % N == 0);

    auto kernel = raw::axiswise_relu_vec<T, N>;
    auto policy = make_policy(kernel, output.size() / N, 0, stream);
    launch_kernel(kernel, policy, output, input, inner_size, slope);
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
