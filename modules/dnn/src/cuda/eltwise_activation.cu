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
    __global__ void eltwise_op_generic_op_vec(Span<T> output, View<T> x, View<T> y, const typename EltwiseOp::Params eltwise_params, const typename ActivationOp::Params act_params) {
        using vector_type = get_vector_type_t<T, N>;

        auto output_vPtr = vector_type::get_pointer(output.data());
        auto x_vPtr = vector_type::get_pointer(x.data());
        auto y_vPtr = vector_type::get_pointer(y.data());

        EltwiseOp eltwise_op(eltwise_params);
        ActivationOp activation_op(act_params);

        for (auto i : grid_stride_range(output.size() / vector_type::size())) {
            vector_type vec_x, vec_y;
            v_load(vec_x, x_vPtr[i]);
            v_load(vec_y, y_vPtr[i]);
            for(int j = 0; j < vec_x.size(); j++)
                vec_x.data[j] = activation_op(eltwise_op(vec_x.data[j], vec_y.data[j]));
            v_store(output_vPtr[i], vec_x);
        }
    }
}

template <class T, class EltwiseOp, class ActivationOp, std::size_t N> static
void launch_vectorized_eltwise_op_generic_op(const Stream& stream, Span<T> output, View<T> x, View<T> y, const typename EltwiseOp::Params& eltwise_params, const typename ActivationOp::Params& act_params) {
    CV_Assert(is_fully_aligned<T>(output, N));
    CV_Assert(is_fully_aligned<T>(x, N));
    CV_Assert(is_fully_aligned<T>(y, N));

    auto kernel = raw::eltwise_op_generic_op_vec<T, EltwiseOp, ActivationOp, N>;
    auto policy = make_policy(kernel, output.size() / N, 0, stream);
    launch_kernel(kernel, policy, output, x, y, eltwise_params, act_params);
}

template <class T, class EltwiseOp, class ActivationOp> static
void eltwise_op_generic_op(const Stream& stream, Span<T> output, View<T> x, View<T> y, const typename EltwiseOp::Params& eltwise_params = {}, const typename ActivationOp::Params& act_params = {}) {
    CV_Assert(output.size() == x.size());
    CV_Assert(output.size() == y.size());

    if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(x, 4) && is_fully_aligned<T>(y, 4)) {
        launch_vectorized_eltwise_op_generic_op<T, EltwiseOp, ActivationOp, 4>(stream, output, x, y, eltwise_params, act_params);
    } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(x, 2) && is_fully_aligned<T>(y, 4)) {
        launch_vectorized_eltwise_op_generic_op<T, EltwiseOp, ActivationOp, 2>(stream, output, x, y, eltwise_params, act_params);
    } else {
        launch_vectorized_eltwise_op_generic_op<T, EltwiseOp, ActivationOp, 1>(stream, output, x, y, eltwise_params, act_params);
    }
}

template <class T>
void eltwise_sum_2_relu(const Stream& stream, Span<T> output, View<T> x, View<T> y, T slope) {
    eltwise_op_generic_op<T, SumFunctor<T>, ReLUFunctor<T>>(stream, output, x, y, {}, {slope});
}

template <class T>
void eltwise_sum_2_clipped_relu(const Stream& stream, Span<T> output, View<T> x, View<T> y, T floor, T ceiling) {
    CV_Assert(static_cast<double>(floor) <= static_cast<double>(ceiling));
    eltwise_op_generic_op<T, SumFunctor<T>, ClippedReLUFunctor<T>>(stream, output, x, y, {}, {floor, ceiling});
}

template <class T>
void eltwise_sum_2_tanh(const Stream& stream, Span<T> output, View<T> x, View<T> y) {
    eltwise_op_generic_op<T, SumFunctor<T>, TanHFunctor<T>>(stream, output, x, y);
}

template <class T>
void eltwise_sum_2_swish(const Stream& stream, Span<T> output, View<T> x, View<T> y) {
    eltwise_op_generic_op<T, SumFunctor<T>, SwishFunctor<T>>(stream, output, x, y);
}

template <class T>
void eltwise_sum_2_mish(const Stream& stream, Span<T> output, View<T> x, View<T> y) {
    eltwise_op_generic_op<T, SumFunctor<T>, MishFunctor<T>>(stream, output, x, y);
}

template <class T>
void eltwise_sum_2_sigmoid(const Stream& stream, Span<T> output, View<T> x, View<T> y) {
    eltwise_op_generic_op<T, SumFunctor<T>, SigmoidFunctor<T>>(stream, output, x, y);
}

template <class T>
void eltwise_sum_2_power(const Stream& stream, Span<T> output, View<T> x, View<T> y, T exp, T scale, T shift) {
    eltwise_op_generic_op<T, SumFunctor<T>, PowerFunctor<T>>(stream, output, x, y, {}, {exp, scale, shift});
}

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
template void eltwise_sum_2_relu<__half>(const Stream&, Span<__half>, View<__half>, View<__half>, __half);
template void eltwise_sum_2_clipped_relu<__half>(const Stream&, Span<__half>, View<__half>, View<__half>, __half, __half);
template void eltwise_sum_2_tanh<__half>(const Stream&, Span<__half>, View<__half>, View<__half>);
template void eltwise_sum_2_swish<__half>(const Stream&, Span<__half>, View<__half>, View<__half>);
template void eltwise_sum_2_mish<__half>(const Stream&, Span<__half>, View<__half>, View<__half>);
template void eltwise_sum_2_sigmoid<__half>(const Stream&, Span<__half>, View<__half>, View<__half>);
template void eltwise_sum_2_power<__half>(const Stream&, Span<__half>, View<__half>, View<__half>, __half, __half, __half);
#endif

template void eltwise_sum_2_relu<float>(const Stream&, Span<float>, View<float>, View<float>, float);
template void eltwise_sum_2_clipped_relu<float>(const Stream&, Span<float>, View<float>, View<float>, float, float);
template void eltwise_sum_2_tanh<float>(const Stream&, Span<float>, View<float>, View<float>);
template void eltwise_sum_2_swish<float>(const Stream&, Span<float>, View<float>, View<float>);
template void eltwise_sum_2_mish<float>(const Stream&, Span<float>, View<float>, View<float>);
template void eltwise_sum_2_sigmoid<float>(const Stream&, Span<float>, View<float>, View<float>);
template void eltwise_sum_2_power<float>(const Stream&, Span<float>, View<float>, View<float>, float, float, float);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
