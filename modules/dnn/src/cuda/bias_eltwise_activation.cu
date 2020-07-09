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
    template <class T, class EltwiseOp, class ActivationOp, std::size_t N>
    __global__ void biasN_eltwise_op_generic_op_inplace_vec(Span<T> inplace_output, size_type inner_size, View<T> bias, View<T> eltwise, const typename EltwiseOp::Params eltwise_params, const typename ActivationOp::Params act_params) {
        using vector_type = get_vector_type_t<T, N>;

        auto inplace_output_vPtr = vector_type::get_pointer(inplace_output.data());
        auto eltwise_vPtr = vector_type::get_pointer(eltwise.data());

        EltwiseOp eltwise_op(eltwise_params);
        ActivationOp activation_op(act_params);

        for (auto i : grid_stride_range(inplace_output.size() / vector_type::size())) {
            const index_type bias_idx = (i / inner_size) % bias.size();

            vector_type output_vec, eltwise_vec;
            v_load(output_vec, inplace_output_vPtr[i]);
            v_load(eltwise_vec, eltwise_vPtr[i]);
            for(int j = 0; j < output_vec.size(); j++)
                output_vec.data[j] = activation_op(eltwise_op(output_vec.data[j] + bias[bias_idx], eltwise_vec.data[j]));
            v_store(inplace_output_vPtr[i], output_vec);
        }
    }
}

template <class T, class EltwiseOp, class ActivationOp, std::size_t N> static
void launch_vectorized_biasN_eltwise_op_generic_op_inplace(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias, View<T> eltwise, const typename EltwiseOp::Params& eltwise_params, const typename ActivationOp::Params& act_params) {
    CV_Assert(is_fully_aligned<T>(inplace_output, N));
    CV_Assert(inplace_output.size() % bias.size() == 0);
    CV_Assert(is_fully_aligned<T>(eltwise, N));
    CV_Assert(inner_size % N == 0);

    auto kernel = raw::biasN_eltwise_op_generic_op_inplace_vec<T, EltwiseOp, ActivationOp, N>;
    auto policy = make_policy(kernel, inplace_output.size() / N, 0, stream);
    launch_kernel(kernel, policy, inplace_output, inner_size / N, bias, eltwise, eltwise_params, act_params);
}

template <class T, class EltwiseOp, class ActivationOp> static
void biasN_eltwise_op_generic_op_inplace(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias, View<T> eltwise, const typename EltwiseOp::Params& eltwise_params = {}, const typename ActivationOp::Params& act_params = {}) {
    CV_Assert(inplace_output.size() == eltwise.size());

    if (is_fully_aligned<T>(inplace_output, 4) && is_fully_aligned<T>(eltwise, 4) && inner_size % 4 == 0) {
        launch_vectorized_biasN_eltwise_op_generic_op_inplace<T, EltwiseOp, ActivationOp, 4>(stream, inplace_output, inner_size, bias, eltwise, eltwise_params, act_params);
    } else if (is_fully_aligned<T>(inplace_output, 2) && is_fully_aligned<T>(eltwise, 2) && inner_size % 2 == 0) {
        launch_vectorized_biasN_eltwise_op_generic_op_inplace<T, EltwiseOp, ActivationOp, 2>(stream, inplace_output, inner_size, bias, eltwise, eltwise_params, act_params);
    } else {
        launch_vectorized_biasN_eltwise_op_generic_op_inplace<T, EltwiseOp, ActivationOp, 1>(stream, inplace_output, inner_size, bias, eltwise, eltwise_params, act_params);
    }
}

template <class T>
void biasN_eltwise_sum_2_identity_inplace(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias, View<T> eltwise) {
    biasN_eltwise_op_generic_op_inplace<T, SumFunctor<T>, IdentityFunctor<T>>(stream, inplace_output, inner_size, bias, eltwise);
}

template <class T>
void biasN_eltwise_sum_2_relu_inplace(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias, View<T> eltwise, T slope) {
    biasN_eltwise_op_generic_op_inplace<T, SumFunctor<T>, ReLUFunctor<T>>(stream, inplace_output, inner_size, bias, eltwise, {}, {slope});
}

template <class T>
void biasN_eltwise_sum_2_clipped_relu_inplace(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias, View<T> eltwise, T floor, T ceiling) {
    CV_Assert(static_cast<double>(floor) <= static_cast<double>(ceiling));
    biasN_eltwise_op_generic_op_inplace<T, SumFunctor<T>, ClippedReLUFunctor<T>>(stream, inplace_output, inner_size, bias, eltwise, {}, {floor, ceiling});
}

template <class T>
void biasN_eltwise_sum_2_tanh_inplace(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias, View<T> eltwise) {
    biasN_eltwise_op_generic_op_inplace<T, SumFunctor<T>, TanHFunctor<T>>(stream, inplace_output, inner_size, bias, eltwise);
}

template <class T>
void biasN_eltwise_sum_2_swish_inplace(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias, View<T> eltwise) {
    biasN_eltwise_op_generic_op_inplace<T, SumFunctor<T>, SwishFunctor<T>>(stream, inplace_output, inner_size, bias, eltwise);
}

template <class T>
void biasN_eltwise_sum_2_mish_inplace(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias, View<T> eltwise) {
    biasN_eltwise_op_generic_op_inplace<T, SumFunctor<T>, MishFunctor<T>>(stream, inplace_output, inner_size, bias, eltwise);
}

template <class T>
void biasN_eltwise_sum_2_sigmoid_inplace(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias, View<T> eltwise) {
    biasN_eltwise_op_generic_op_inplace<T, SumFunctor<T>, SigmoidFunctor<T>>(stream, inplace_output, inner_size, bias, eltwise);
}

template <class T>
void biasN_eltwise_sum_2_power_inplace(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias, View<T> eltwise, T exp, T scale, T shift) {
    biasN_eltwise_op_generic_op_inplace<T, SumFunctor<T>, PowerFunctor<T>>(stream, inplace_output, inner_size, bias, eltwise, {}, {exp, scale, shift});
}

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
template void biasN_eltwise_sum_2_identity_inplace<__half>(const Stream&, Span<__half>, std::size_t, View<__half>, View<__half>);
template void biasN_eltwise_sum_2_relu_inplace<__half>(const Stream&, Span<__half>, std::size_t, View<__half>, View<__half>, __half);
template void biasN_eltwise_sum_2_clipped_relu_inplace<__half>(const Stream&, Span<__half>, std::size_t, View<__half>, View<__half>, __half, __half);
template void biasN_eltwise_sum_2_tanh_inplace<__half>(const Stream&, Span<__half>, std::size_t, View<__half>, View<__half>);
template void biasN_eltwise_sum_2_swish_inplace<__half>(const Stream&, Span<__half>, std::size_t, View<__half>, View<__half>);
template void biasN_eltwise_sum_2_mish_inplace<__half>(const Stream&, Span<__half>, std::size_t, View<__half>, View<__half>);
template void biasN_eltwise_sum_2_sigmoid_inplace<__half>(const Stream&, Span<__half>, std::size_t, View<__half>, View<__half>);
template void biasN_eltwise_sum_2_power_inplace<__half>(const Stream&, Span<__half>, std::size_t, View<__half>, View<__half>, __half, __half, __half);
#endif

template void biasN_eltwise_sum_2_identity_inplace<float>(const Stream&, Span<float>, std::size_t, View<float>, View<float>);
template void biasN_eltwise_sum_2_relu_inplace<float>(const Stream&, Span<float>, std::size_t, View<float>, View<float>, float);
template void biasN_eltwise_sum_2_clipped_relu_inplace<float>(const Stream&, Span<float>, std::size_t, View<float>, View<float>, float, float);
template void biasN_eltwise_sum_2_tanh_inplace<float>(const Stream&, Span<float>, std::size_t, View<float>, View<float>);
template void biasN_eltwise_sum_2_swish_inplace<float>(const Stream&, Span<float>, std::size_t, View<float>, View<float>);
template void biasN_eltwise_sum_2_mish_inplace<float>(const Stream&, Span<float>, std::size_t, View<float>, View<float>);
template void biasN_eltwise_sum_2_sigmoid_inplace<float>(const Stream&, Span<float>, std::size_t, View<float>, View<float>);
template void biasN_eltwise_sum_2_power_inplace<float>(const Stream&, Span<float>, std::size_t, View<float>, View<float>, float, float, float);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
