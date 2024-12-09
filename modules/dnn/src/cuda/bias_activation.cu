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

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

namespace raw {
    template <class T, class ActivationOp, std::size_t N>
    __global__ void biasN_generic_op_inplace_vec(Span<T> inplace_output, size_type inner_size, View<T> bias, const typename ActivationOp::Params params) {
        using vector_type = get_vector_type_t<T, N>;

        auto inplace_output_vPtr = vector_type::get_pointer(inplace_output.data());

        ActivationOp activation_op(params);

        for (auto i : grid_stride_range(inplace_output.size() / vector_type::size())) {
            const index_type bias_idx = (i / inner_size) % bias.size();

            vector_type vec;
            v_load(vec, inplace_output_vPtr[i]);
            for(int j = 0; j < vec.size(); j++)
                vec.data[j] = activation_op(vec.data[j] + bias[bias_idx]);
            v_store(inplace_output_vPtr[i], vec);
        }
    }

} /* namespace raw */

template <class T, class ActivationOp, std::size_t N> static
void launch_vectorized_biasN_generic_op_inplace(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias, const typename ActivationOp::Params& params) {
    CV_Assert(inplace_output.size() % inner_size == 0);
    CV_Assert(is_fully_aligned<T>(inplace_output, N));
    CV_Assert(inner_size % N == 0);

    auto kernel = raw::biasN_generic_op_inplace_vec<T, ActivationOp, N>;
    auto policy = make_policy(kernel, inplace_output.size() / N, 0, stream);
    launch_kernel(kernel, policy, inplace_output, inner_size / N, bias, params);
}

template <class T, class ActivationOp> static
void biasN_generic_op_inplace(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias, const typename ActivationOp::Params& params = {}) {
    if (is_fully_aligned<T>(inplace_output, 4) && inner_size % 4 == 0) {
        launch_vectorized_biasN_generic_op_inplace<T, ActivationOp, 4>(stream, inplace_output, inner_size, bias, params);
    } else if (is_fully_aligned<T>(inplace_output, 2) && inner_size % 2 == 0) {
        launch_vectorized_biasN_generic_op_inplace<T, ActivationOp, 2>(stream, inplace_output, inner_size, bias, params);
    } else {
        launch_vectorized_biasN_generic_op_inplace<T, ActivationOp, 1>(stream, inplace_output, inner_size, bias, params);
    }
}

template <class T>
void biasN_relu_inplace(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias, T slope) {
    biasN_generic_op_inplace<T, ReLUFunctor<T>>(stream, inplace_output, inner_size, bias, {slope});
}

template <class T>
void biasN_clipped_relu_inplace(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias, T floor, T ceil) {
    CV_Assert(static_cast<double>(floor) <= static_cast<double>(ceil));
    biasN_generic_op_inplace<T, ClippedReLUFunctor<T>>(stream, inplace_output, inner_size, bias, {floor, ceil});
}

template <class T>
void biasN_tanh_inplace(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias) {
    biasN_generic_op_inplace<T, TanHFunctor<T>>(stream, inplace_output, inner_size, bias);
}

template <class T>
void biasN_swish_inplace(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias) {
    biasN_generic_op_inplace<T, SwishFunctor<T>>(stream, inplace_output, inner_size, bias);
}

template <class T>
void biasN_mish_inplace(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias) {
    biasN_generic_op_inplace<T, MishFunctor<T>>(stream, inplace_output, inner_size, bias);
}

template <class T>
void biasN_sigmoid_inplace(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias) {
    biasN_generic_op_inplace<T, SigmoidFunctor<T>>(stream, inplace_output, inner_size, bias);
}

template <class T>
void biasN_power_inplace(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias, T power, T scale, T shift) {
    biasN_generic_op_inplace<T, PowerFunctor<T>>(stream, inplace_output, inner_size, bias, {power, scale, shift});
}

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
template void biasN_relu_inplace<__half>(const Stream&, Span<__half>, std::size_t, View<__half>, __half);
template void biasN_clipped_relu_inplace<__half>(const Stream&, Span<__half>, std::size_t, View<__half>, __half, __half);
template void biasN_tanh_inplace<__half>(const Stream&, Span<__half>, std::size_t, View<__half>);
template void biasN_swish_inplace<__half>(const Stream&, Span<__half>, std::size_t, View<__half>);
template void biasN_mish_inplace<__half>(const Stream&, Span<__half>, std::size_t, View<__half>);
template void biasN_sigmoid_inplace<__half>(const Stream&, Span<__half>, std::size_t, View<__half>);
template void biasN_power_inplace<__half>(const Stream&, Span<__half>, std::size_t, View<__half>, __half, __half, __half);
#endif

template void biasN_relu_inplace<float>(const Stream&, Span<float>, std::size_t, View<float>, float);
template void biasN_clipped_relu_inplace<float>(const Stream&, Span<float>, std::size_t, View<float>, float, float);
template void biasN_tanh_inplace<float>(const Stream&, Span<float>, std::size_t, View<float>);
template void biasN_swish_inplace<float>(const Stream&, Span<float>, std::size_t, View<float>);
template void biasN_mish_inplace<float>(const Stream&, Span<float>, std::size_t, View<float>);
template void biasN_sigmoid_inplace<float>(const Stream&, Span<float>, std::size_t, View<float>);
template void biasN_power_inplace<float>(const Stream&, Span<float>, std::size_t, View<float>, float, float, float);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
