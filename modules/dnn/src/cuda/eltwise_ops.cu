// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "functors.hpp"
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
    template <class T, class Functor, std::size_t N, class ...FunctorArgs>
    __global__ void eltwise_op_vec(Span<T> output, View<T> x, View<T> y, FunctorArgs ...functorArgs) {
        using vector_type = get_vector_type_t<T, N>;

        auto output_vPtr = vector_type::get_pointer(output.data());
        auto x_vPtr = vector_type::get_pointer(x.data());
        auto y_vPtr = vector_type::get_pointer(y.data());

        Functor functor(functorArgs...);

        for (auto i : grid_stride_range(output.size() / vector_type::size())) {
            vector_type vec_x, vec_y;
            v_load(vec_x, x_vPtr[i]);
            v_load(vec_y, y_vPtr[i]);
            for (int j = 0; j < vector_type::size(); j++)
                vec_x.data[j] = functor(vec_x.data[j], vec_y.data[j]);
            v_store(output_vPtr[i], vec_x);
        }
    }
}

template <class T, template <class> class EltwiseOp, std::size_t N, class ...EltwiseOpArgs> static
void launch_vectorized_eltwise_op(const Stream& stream, Span<T> output, View<T> x, View<T> y, EltwiseOpArgs ...eltwiseOpArgs) {
    CV_Assert(x.size() == y.size());
    CV_Assert(x.size() == output.size());
    CV_Assert(is_fully_aligned<T>(output, N));
    CV_Assert(is_fully_aligned<T>(x, N));
    CV_Assert(is_fully_aligned<T>(y, N));

    auto kernel = raw::eltwise_op_vec<T, EltwiseOp<T>, N, EltwiseOpArgs...>;
    auto policy = make_policy(kernel, output.size() / N, 0, stream);
    launch_kernel(kernel, policy, output, x, y, eltwiseOpArgs...);
}

template <class T, template <class> class EltwiseOp, class ...EltwiseOpArgs> static
void eltwise_op(const Stream& stream, Span<T> output, View<T> x, View<T> y, EltwiseOpArgs ...eltwiseOpArgs) {
    CV_Assert(x.size() == y.size());
    CV_Assert(x.size() == output.size());

    if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(x, 4) && is_fully_aligned<T>(y, 4)) {
        launch_vectorized_eltwise_op<T, EltwiseOp, 4>(stream, output, x, y, eltwiseOpArgs...);
    } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(x, 2) && is_fully_aligned<T>(y, 2)) {
        launch_vectorized_eltwise_op<T, EltwiseOp, 2>(stream, output, x, y, eltwiseOpArgs...);
    } else {
        launch_vectorized_eltwise_op<T, EltwiseOp, 1>(stream, output, x, y, eltwiseOpArgs...);
    }
}

template <class T>
void eltwise_max_2(const Stream& stream, Span<T> output, View<T> x, View<T> y) {
    eltwise_op<T, max_functor>(stream, output, x, y);
}

template <class T>
void eltwise_sum_2(const Stream& stream, Span<T> output, View<T> x, View<T> y) {
    eltwise_op<T, sum_functor>(stream, output, x, y);
}

template <class T>
void eltwise_sum_coeff_2(const Stream& stream, Span<T> output, T coeff_x, View<T> x, T coeff_y, View<T> y) {
    eltwise_op<T, scaled_sum_functor>(stream, output, x, y, coeff_x, coeff_y);
}

template <class T>
void eltwise_prod_2(const Stream& stream, Span<T> output, View<T> x, View<T> y) {
    eltwise_op<T, product_functor>(stream, output, x, y);
}

template <class T>
void eltwise_div_2(const Stream& stream, Span<T> output, View<T> x, View<T> y) {
    eltwise_op<T, div_functor>(stream, output, x, y);
}

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void eltwise_div_2(const Stream& stream, Span<__half> output, View<__half> x, View<__half> y);
    template void eltwise_prod_2(const Stream& stream, Span<__half> output, View<__half> x, View<__half> y);
    template void eltwise_sum_coeff_2(const Stream&, Span<__half>, __half, View<__half>, __half, View<__half>);
    template void eltwise_sum_2(const Stream& stream, Span<__half> output, View<__half> x, View<__half> y);
    template void eltwise_max_2(const Stream& stream, Span<__half> output, View<__half> x, View<__half> y);
#endif
    template void eltwise_div_2(const Stream& stream, Span<float> output, View<float> x, View<float> y);
    template void eltwise_prod_2(const Stream& stream, Span<float> output, View<float> x, View<float> y);
    template void eltwise_sum_coeff_2(const Stream&, Span<float>, float, View<float>, float, View<float>);
    template void eltwise_sum_2(const Stream& stream, Span<float> output, View<float> x, View<float> y);
    template void eltwise_max_2(const Stream& stream, Span<float> output, View<float> x, View<float> y);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
