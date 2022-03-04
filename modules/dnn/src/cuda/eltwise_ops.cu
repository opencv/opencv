// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "array.hpp"
#include "functors.hpp"
#include "grid_stride_range.hpp"
#include "execution.hpp"
#include "vector_traits.hpp"
#include "kernel_dispatcher.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"
#include "../cuda4dnn/csl/tensor.hpp"

#include <opencv2/core.hpp>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

namespace raw {
    template <class T, class EltwiseOp, std::size_t N>
    __global__ void eltwise_op_vec(Span<T> output, View<T> x, View<T> y, const typename EltwiseOp::Params params) {
        using vector_type = get_vector_type_t<T, N>;

        auto output_vPtr = vector_type::get_pointer(output.data());
        auto x_vPtr = vector_type::get_pointer(x.data());
        auto y_vPtr = vector_type::get_pointer(y.data());

        EltwiseOp eltwise_op(params);

        for (auto i : grid_stride_range(output.size() / vector_type::size())) {
            vector_type vec_x, vec_y;
            v_load(vec_x, x_vPtr[i]);
            v_load(vec_y, y_vPtr[i]);
            for (int j = 0; j < vector_type::size(); j++)
                vec_x.data[j] = eltwise_op(vec_x.data[j], vec_y.data[j]);
            v_store(output_vPtr[i], vec_x);
        }
    }

    template <class T, class EltwiseOp, std::size_t Rank>
    __global__ void eltwise_op_bcast(
        Span<T> output, array<size_type, Rank> out_strides,
        View<T> x, array<size_type, Rank> x_strides, array<bool, Rank> x_bcast,
        View<T> y, array<size_type, Rank> y_strides, array<bool, Rank> y_bcast,
        const typename EltwiseOp::Params params) {
        EltwiseOp eltwise_op(params);

        for (auto i : grid_stride_range(output.size())) {
            index_type out_index = i / out_strides[0];
            index_type x_index = x_bcast[0] ? 0 : out_index * x_strides[0];
            index_type y_index = y_bcast[0] ? 0 : out_index * y_strides[0];

            for (int j = 1; j < Rank; j++)
            {
                out_index = (i % out_strides[j - 1]) / out_strides[j];
                if (!x_bcast[j])
                    x_index += out_index * x_strides[j];
                if (!y_bcast[j])
                    y_index += out_index * y_strides[j];
            }

            output[i] = eltwise_op(x[x_index], y[y_index]);
        }
    }
}

template <class T, class EltwiseOp, std::size_t N> static
void launch_vectorized_eltwise_op(const Stream& stream, Span<T> output, View<T> x, View<T> y, const typename EltwiseOp::Params& params) {
    CV_Assert(x.size() == y.size());
    CV_Assert(x.size() == output.size());
    CV_Assert(is_fully_aligned<T>(output, N));
    CV_Assert(is_fully_aligned<T>(x, N));
    CV_Assert(is_fully_aligned<T>(y, N));

    auto kernel = raw::eltwise_op_vec<T, EltwiseOp, N>;
    auto policy = make_policy(kernel, output.size() / N, 0, stream);
    launch_kernel(kernel, policy, output, x, y, params);
}

template <class T, class EltwiseOp, std::size_t Rank> static
void launch_eltwise_op_bcast(
    const Stream& stream,
    Span<T> output, const std::vector<std::size_t>& outStride,
    View<T> x, const std::vector<std::size_t>& inStride1, const std::vector<int>& inBcast1,
    View<T> y, const std::vector<std::size_t>& inStride2, const std::vector<int>& inBcast2,
    const typename EltwiseOp::Params& params)
{
    CV_Assert(outStride.size() == Rank);
    CV_Assert(inStride1.size() == Rank);
    CV_Assert(inStride2.size() == Rank);
    CV_Assert(inBcast1.size() == Rank);
    CV_Assert(inBcast2.size() == Rank);

    array<size_type, Rank> outStride_k, inStride1_k, inStride2_k;
    outStride_k.assign(std::begin(outStride), std::end(outStride));
    inStride1_k.assign(std::begin(inStride1), std::end(inStride1));
    inStride2_k.assign(std::begin(inStride2), std::end(inStride2));

    array<bool, Rank> inBcast1_k, inBcast2_k;
    inBcast1_k.assign(std::begin(inBcast1), std::end(inBcast1));
    inBcast2_k.assign(std::begin(inBcast2), std::end(inBcast2));

    auto kernel = raw::eltwise_op_bcast<T, EltwiseOp, Rank>;
    auto policy = make_policy(kernel, output.size(), 0, stream);
    launch_kernel(kernel, policy, output, outStride_k, x, inStride1_k, inBcast1_k, y, inStride2_k, inBcast2_k, params);
}

GENERATE_KERNEL_DISPATCHER_2TP(eltwise_op_bcast_dispatcher, launch_eltwise_op_bcast);

template <class T, class EltwiseOp> static
void eltwise_op(const Stream& stream, TensorSpan<T> output, TensorView<T> x, TensorView<T> y, const typename EltwiseOp::Params& params = {}) {
    if (is_shape_same(output, x) && is_shape_same(output, y))
    {
        /* no broadcasting; use fast path */
        CV_Assert(x.size() == y.size());
        CV_Assert(x.size() == output.size());

        if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(x, 4) && is_fully_aligned<T>(y, 4)) {
            launch_vectorized_eltwise_op<T, EltwiseOp, 4>(stream, output, x, y, params);
        } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(x, 2) && is_fully_aligned<T>(y, 2)) {
            launch_vectorized_eltwise_op<T, EltwiseOp, 2>(stream, output, x, y, params);
        } else {
            launch_vectorized_eltwise_op<T, EltwiseOp, 1>(stream, output, x, y, params);
        }
    }
    else
    {
        CV_Assert(is_shape_compatible(output, x));
        CV_Assert(is_shape_compatible(output, y));

        /* matching singleton axes in both input tensors can be eliminated
         *
         * Reasoning:
         * ----------
         * Singleton axes do not contribute towards address calculation. They are redundant
         * unless there is broadcasting. If both input tensors have singleton axis at a
         * specified position, there is no broadcasting on that axis.
         *
         * Example:
         * ---------
         * x: [1, 256, 32, 32] -> [256, 32, 32]
         * y: [1, 256, 1, 1] -> [256, 1, 1]
         */
        for (int r = 0; r < output.rank(); r++)
        {
            while (x.get_axis_size(r) == 1 && y.get_axis_size(r) == 1) {
                CV_Assert(output.get_axis_size(r) == 1);

                x.squeeze(r);
                y.squeeze(r);
                output.squeeze(r);
            }
        }

        auto inShape1 = x.shape_as_vector();
        auto inShape2 = y.shape_as_vector();
        auto outShape = output.shape_as_vector();

        /* contiguous axes that do not broadcast can be merged into one axis
         *
         * Example:
         * ---------
         * x: [32, 8, 8] -> [32, 64]
         * y: [1, 8, 8] -> [1, 64]
         */
        for (int i = 0; i < inShape1.size(); i++) {
            /* check if axis `i` requires any broadcasting */
            if (inShape1[i] == inShape2[i]) {
                /* loop invariant: `i` is the first axis in the contiguous axis sequence */

                int j = i + 1; /* `j` is the axis which we will attempt to merge */
                while (j < inShape1.size() && inShape1[j] == inShape2[j]) {
                    CV_Assert(outShape[j] == inShape1[j]);

                    /* `j` axis is also used fully; merge `i` and `j` */
                    auto new_size = inShape1[i] * inShape1[j];
                    inShape1[i] = new_size;
                    inShape2[i] = new_size;

                    /* delete axis `j` */
                    inShape1.erase(std::begin(inShape1) + j);
                    inShape2.erase(std::begin(inShape2) + j);
                    outShape.erase(std::begin(outShape) + j);

                    /* optimizations should not break the invariants */
                    CV_Assert(inShape1.size() == outShape.size());
                    CV_Assert(inShape2.size() == outShape.size());
                    CV_Assert(inShape1[i] == outShape[i]);
                    CV_Assert(inShape2[i] == outShape[i]);
                }
            }
        }

        /* contiguous broadcasting axes on the same tensor can be merged into one axis
         *
         * Example:
         * ---------
         * x: [256, 8, 8] -> [256, 64]
         * y: [256, 1, 1] -> [256, 1]
         */
        for (int i = 0; i < inShape1.size(); i++) {
            /* check if axis `i` requires any broadcasting in tensor 1 */
            if (inShape1[i] == 1 && inShape2[i] != 1) {
                /* loop invariant: `i` is the first axis in the contiguous axis sequence */

                int j = i + 1; /* `j` is the axis which we will attempt to merge */
                while (j < inShape1.size() && inShape1[j] == 1 && inShape2[j] != 1) {
                    CV_Assert(outShape[j] == inShape2[j]);

                    /* `j` axis is also used fully; merge `i` and `j` */
                    inShape1[i] = 1;
                    inShape2[i] = inShape2[i] * inShape2[j];
                    outShape[i] = inShape2[i];

                    /* delete axis `j` */
                    inShape1.erase(std::begin(inShape1) + j);
                    inShape2.erase(std::begin(inShape2) + j);
                    outShape.erase(std::begin(outShape) + j);

                    /* optimizations should not break the invariants */
                    CV_Assert(inShape1.size() == outShape.size());
                    CV_Assert(inShape2.size() == outShape.size());
                    CV_Assert(inShape1[i] == 1);
                    CV_Assert(inShape2[i] == outShape[i]);
                }
            }

            /* check if axis `i` requires any broadcasting in tensor 2 */
            if (inShape1[i] != 1 && inShape2[i] == 1) {
                /* loop invariant: `i` is the first axis in the contiguous axis sequence */

                int j = i + 1; /* `j` is the axis which we will attempt to merge */
                while (j < inShape1.size() && inShape1[j] != 1 && inShape2[j] == 1) {
                    CV_Assert(outShape[j] == inShape1[j]);

                    /* `j` axis is also used fully; merge `i` and `j` */
                    inShape1[i] = inShape1[i] * inShape1[j];
                    inShape2[i] = 1;
                    outShape[i] = inShape1[i];

                    /* delete axis `j` */
                    inShape1.erase(std::begin(inShape1) + j);
                    inShape2.erase(std::begin(inShape2) + j);
                    outShape.erase(std::begin(outShape) + j);

                    /* optimizations should not break the invariants */
                    CV_Assert(inShape1.size() == outShape.size());
                    CV_Assert(inShape2.size() == outShape.size());
                    CV_Assert(inShape1[i] == outShape[i]);
                    CV_Assert(inShape2[i] == 1);
                }
            }
        }

        auto rank = outShape.size();

        std::vector<std::size_t> inStride1(rank), inStride2(rank), outStride(rank);
        inStride1.back() = 1;
        inStride2.back() = 1;
        outStride.back() = 1;
        /* garbage, ..., garbage, 1 */

        std::copy(std::begin(inShape1) + 1, std::end(inShape1), std::begin(inStride1));
        std::copy(std::begin(inShape2) + 1, std::end(inShape2), std::begin(inStride2));
        std::copy(std::begin(outShape) + 1, std::end(outShape), std::begin(outStride));
        /* dim[0], dim[1], ..., dim[-1], 1 */

        std::partial_sum(inStride1.rbegin(), inStride1.rend(), inStride1.rbegin(), std::multiplies<std::size_t>());
        std::partial_sum(inStride2.rbegin(), inStride2.rend(), inStride2.rbegin(), std::multiplies<std::size_t>());
        std::partial_sum(outStride.rbegin(), outStride.rend(), outStride.rbegin(), std::multiplies<std::size_t>());
        /* stride[0], stride[1], ..., stride[-2], 1 */

        std::vector<int> inBcast1(rank), inBcast2(rank);
        std::transform(std::begin(inShape1), std::end(inShape1), std::begin(inBcast1), [](std::size_t sz) { return sz == 1; });
        std::transform(std::begin(inShape2), std::end(inShape2), std::begin(inBcast2), [](std::size_t sz) { return sz == 1; });

        CV_Assert(1 <= rank && rank <= CSL_MAX_TENSOR_RANK);
        eltwise_op_bcast_dispatcher<T, EltwiseOp, 1, CSL_MAX_TENSOR_RANK>(rank, stream, output, outStride, x, inStride1, inBcast1, y, inStride2, inBcast2, params);
    }
}

template <class T>
void eltwise_max_2(const Stream& stream, TensorSpan<T> output, TensorView<T> x, TensorView<T> y) {
    eltwise_op<T, MaxFunctor<T>>(stream, output, x, y);
}

template <class T>
void eltwise_min_2(const Stream& stream, TensorSpan<T> output, TensorView<T> x, TensorView<T> y) {
    eltwise_op<T, MinFunctor<T>>(stream, output, x, y);
}

template <class T>
void eltwise_sum_2(const Stream& stream, TensorSpan<T> output, TensorView<T> x, TensorView<T> y) {
    eltwise_op<T, SumFunctor<T>>(stream, output, x, y);
}

template <class T>
void eltwise_sum_coeff_2(const Stream& stream, TensorSpan<T> output, T coeff_x, TensorView<T> x, T coeff_y, TensorView<T> y) {
    eltwise_op<T, ScaledSumFunctor<T>>(stream, output, x, y, {coeff_x, coeff_y});
}

template <class T>
void eltwise_prod_2(const Stream& stream, TensorSpan<T> output, TensorView<T> x, TensorView<T> y) {
    eltwise_op<T, ProductFunctor<T>>(stream, output, x, y);
}

template <class T>
void eltwise_div_2(const Stream& stream, TensorSpan<T> output, TensorView<T> x, TensorView<T> y) {
    eltwise_op<T, DivFunctor<T>>(stream, output, x, y);
}

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void eltwise_div_2(const Stream& stream, TensorSpan<__half> output, TensorView<__half> x, TensorView<__half> y);
    template void eltwise_prod_2(const Stream& stream, TensorSpan<__half> output, TensorView<__half> x, TensorView<__half> y);
    template void eltwise_sum_coeff_2(const Stream&, TensorSpan<__half>, __half, TensorView<__half>, __half, TensorView<__half>);
    template void eltwise_sum_2(const Stream& stream, TensorSpan<__half> output, TensorView<__half> x, TensorView<__half> y);
    template void eltwise_max_2(const Stream& stream, TensorSpan<__half> output, TensorView<__half> x, TensorView<__half> y);
    template void eltwise_min_2(const Stream& stream, TensorSpan<__half> output, TensorView<__half> x, TensorView<__half> y);
#endif
    template void eltwise_div_2(const Stream& stream, TensorSpan<float> output, TensorView<float> x, TensorView<float> y);
    template void eltwise_prod_2(const Stream& stream, TensorSpan<float> output, TensorView<float> x, TensorView<float> y);
    template void eltwise_sum_coeff_2(const Stream&, TensorSpan<float>, float, TensorView<float>, float, TensorView<float>);
    template void eltwise_sum_2(const Stream& stream, TensorSpan<float> output, TensorView<float> x, TensorView<float> y);
    template void eltwise_max_2(const Stream& stream, TensorSpan<float> output, TensorView<float> x, TensorView<float> y);
    template void eltwise_min_2(const Stream& stream, TensorSpan<float> output, TensorView<float> x, TensorView<float> y);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
