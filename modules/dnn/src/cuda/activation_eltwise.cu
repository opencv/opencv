// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "functors.hpp"
#include "vector_traits.hpp"
#include "grid_stride_range.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

namespace raw {

    template <class T, class ActivationOp, class EltwiseOp, std::size_t N>
    __global__ void generic_op_eltwise_op_inplace_vec(Span<T> inplace_output, View<T> eltwise, const typename ActivationOp::Params act_params, const typename EltwiseOp::Params eltwise_params) {
        using vector_type = get_vector_type_t<T, N>;

        auto inplace_output_vPtr = vector_type::get_pointer(inplace_output.data());
        auto eltwise_vPtr = vector_type::get_pointer(eltwise.data());

        ActivationOp activation_op(act_params);
        EltwiseOp eltwise_op(eltwise_params);

        for (auto i : grid_stride_range(inplace_output.size() / vector_type::size())) {
            vector_type output_vec, eltwise_vec;
            v_load(output_vec, inplace_output_vPtr[i]);
            v_load(eltwise_vec, eltwise_vPtr[i]);
            for(int j = 0; j < output_vec.size(); j++)
                output_vec.data[j] = eltwise_op(activation_op(output_vec.data[j]), eltwise_vec.data[j]);
            v_store(inplace_output_vPtr[i], output_vec);
        }
    }
}

template <class T, class ActivationOp, class EltwiseOp, std::size_t N> static
void launch_vectorized_generic_op_eltwise_op_inplace(const Stream& stream, Span<T> inplace_output, View<T> eltwise, const typename ActivationOp::Params& act_params, const typename EltwiseOp::Params& eltwise_params) {
    CV_Assert(is_fully_aligned<T>(inplace_output, N));
    CV_Assert(is_fully_aligned<T>(eltwise, N));

    auto kernel = raw::generic_op_eltwise_op_inplace_vec<T, ActivationOp, EltwiseOp, N>;
    auto policy = make_policy(kernel, inplace_output.size() / N, 0, stream);
    launch_kernel(kernel, policy, inplace_output, eltwise, act_params, eltwise_params);
}

template <class T, class ActivationOp, class EltwiseOp> static
void generic_op_eltwise_op_inplace(const Stream& stream, Span<T> inplace_output, View<T> eltwise, const typename ActivationOp::Params& act_params = {}, const typename EltwiseOp::Params& eltwise_params = {}) {
    CV_Assert(inplace_output.size() == eltwise.size());

    if (is_fully_aligned<T>(inplace_output, 4) && is_fully_aligned<T>(eltwise, 4)) {
        launch_vectorized_generic_op_eltwise_op_inplace<T, ActivationOp, EltwiseOp, 4>(stream, inplace_output, eltwise, act_params, eltwise_params);
    } else if (is_fully_aligned<T>(inplace_output, 2) && is_fully_aligned<T>(eltwise, 2)) {
        launch_vectorized_generic_op_eltwise_op_inplace<T, ActivationOp, EltwiseOp, 2>(stream, inplace_output, eltwise, act_params, eltwise_params);
    } else {
        launch_vectorized_generic_op_eltwise_op_inplace<T, ActivationOp, EltwiseOp, 1>(stream, inplace_output, eltwise, act_params, eltwise_params);
    }
}

template <class T>
void relu_eltwise_sum_2_inplace(const Stream& stream, Span<T> inplace_output, View<T> eltwise, T slope) {
    generic_op_eltwise_op_inplace<T, ReLUFunctor<T>, SumFunctor<T>>(stream, inplace_output, eltwise, {slope});
}

template <class T>
void clipped_relu_eltwise_sum_2_inplace(const Stream& stream, Span<T> inplace_output, View<T> eltwise, T floor, T ceiling) {
    CV_Assert(static_cast<double>(floor) <= static_cast<double>(ceiling));
    generic_op_eltwise_op_inplace<T, ClippedReLUFunctor<T>, SumFunctor<T>>(stream, inplace_output, eltwise, {floor, ceiling});
}

template <class T>
void tanh_eltwise_sum_2_inplace(const Stream& stream, Span<T> inplace_output, View<T> eltwise) {
    generic_op_eltwise_op_inplace<T, TanHFunctor<T>, SumFunctor<T>>(stream, inplace_output, eltwise);
}

template <class T>
void swish_eltwise_sum_2_inplace(const Stream& stream, Span<T> inplace_output, View<T> eltwise) {
    generic_op_eltwise_op_inplace<T, SwishFunctor<T>, SumFunctor<T>>(stream, inplace_output, eltwise);
}

template <class T>
void mish_eltwise_sum_2_inplace(const Stream& stream, Span<T> inplace_output, View<T> eltwise) {
    generic_op_eltwise_op_inplace<T, MishFunctor<T>, SumFunctor<T>>(stream, inplace_output, eltwise);
}

template <class T>
void sigmoid_eltwise_sum_2_inplace(const Stream& stream, Span<T> inplace_output, View<T> eltwise) {
    generic_op_eltwise_op_inplace<T, SigmoidFunctor<T>, SumFunctor<T>>(stream, inplace_output, eltwise);
}

template <class T>
void power_eltwise_sum_2_inplace(const Stream& stream, Span<T> inplace_output, View<T> eltwise, T exp, T scale, T shift) {
    generic_op_eltwise_op_inplace<T, PowerFunctor<T>, SumFunctor<T>>(stream, inplace_output, eltwise, {exp, scale, shift});
}

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
template void relu_eltwise_sum_2_inplace<__half>(const Stream&, Span<__half>, View<__half>, __half);
template void clipped_relu_eltwise_sum_2_inplace<__half>(const Stream&, Span<__half>, View<__half>, __half, __half);
template void tanh_eltwise_sum_2_inplace<__half>(const Stream&, Span<__half>, View<__half>);
template void swish_eltwise_sum_2_inplace<__half>(const Stream&, Span<__half>, View<__half>);
template void mish_eltwise_sum_2_inplace<__half>(const Stream&, Span<__half>, View<__half>);
template void sigmoid_eltwise_sum_2_inplace<__half>(const Stream&, Span<__half>, View<__half>);
template void power_eltwise_sum_2_inplace<__half>(const Stream&, Span<__half>, View<__half>, __half, __half, __half);
#endif

template void relu_eltwise_sum_2_inplace<float>(const Stream&, Span<float>, View<float>, float);
template void clipped_relu_eltwise_sum_2_inplace<float>(const Stream&, Span<float>, View<float>, float, float);
template void tanh_eltwise_sum_2_inplace<float>(const Stream&, Span<float>, View<float>);
template void swish_eltwise_sum_2_inplace<float>(const Stream&, Span<float>, View<float>);
template void mish_eltwise_sum_2_inplace<float>(const Stream&, Span<float>, View<float>);
template void sigmoid_eltwise_sum_2_inplace<float>(const Stream&, Span<float>, View<float>);
template void power_eltwise_sum_2_inplace<float>(const Stream&, Span<float>, View<float>, float, float, float);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
