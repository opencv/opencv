// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "array.hpp"
#include "types.hpp"
#include "grid_stride_range.hpp"
#include "execution.hpp"
#include "kernel_dispatcher.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/tensor.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {
        template <class T, std::size_t Rank>
        __global__ void permute(
            array<index_type, Rank> axis_order,
            Span<T> output, array<size_type, Rank> outStrides,
            View<T> input, array<size_type, Rank> inStrides)
        {
            for (auto i : grid_stride_range(input.size())) {
                index_type oldPosition = 0;
                index_type newPosition = i;

                for (int j = 0; j < Rank; j++)
                {
                    auto order = axis_order[j];
                    oldPosition += (newPosition / outStrides[j]) * inStrides[order];
                    newPosition %= outStrides[j];
                }

                output[i] = input[oldPosition];
            }
        }
    }

    template <class T, std::size_t Rank> static
    void launch_permute_kernel(
        const Stream& stream,
        const std::vector<std::size_t>& order,
        Span<T> output, const std::vector<std::size_t>& outStride,
        View<T> input, const std::vector<std::size_t>& inStride)
    {
        CV_Assert(order.size() == Rank);
        CV_Assert(outStride.size() == Rank);
        CV_Assert(inStride.size() == Rank);

        array<index_type, Rank> order_k;
        order_k.assign(std::begin(order), std::end(order));

        array<size_type, Rank> outStride_k, inStride_k;
        outStride_k.assign(std::begin(outStride), std::end(outStride));
        inStride_k.assign(std::begin(inStride), std::end(inStride));

        auto kernel = raw::permute<T, Rank>;
        auto policy = make_policy(kernel, input.size(), 0, stream);
        launch_kernel(kernel, policy, order_k, output, outStride_k, input, inStride_k);
    }

    GENERATE_KERNEL_DISPATCHER(permute_dispatcher, launch_permute_kernel);

    template <class T>
    void permute(
        const Stream& stream,
        TensorSpan<T> output, TensorView<T> input,
        std::vector<std::size_t> order)
    {
        CV_Assert(output.rank() == input.rank());
        CV_Assert(input.rank() == order.size());
        CV_Assert(input.size() == output.size());

        /* squeezable axes at the beginning of both tensors which aren't permuted can be eliminated
         *
         * Reasoning:
         * ----------
         * Suppose an item's indices in the input tensor is [i1, i2, ...]. The indices in the
         * output tensor will be some permutation of the input tensor indices. Let the output
         * tensor indices be [o1, o2, ...]. The permutation operation essentially copies items
         * from the input tensor to new locations in the output tensor as dictated by the indices.
         *
         * If the size of the first axis of the input and output tensor is one and these axes are
         * not involved in any permutation, i.e. order[0] = 0, the input and output indicies for
         * all the elements will be of the form be [0, i2, ...] and [0, o2, ...] respectively.
         * The first index does not contribute to the element's address calculation and hence does
         * nothing apart from eating up few cycles.
         */
        while (order[0] == 0 && input.get_axis_size(0) == 1 && output.get_axis_size(0) == 1) {
            /* remove the axes */
            input.squeeze(0);
            output.squeeze(0);

            /* when we remove axis zero, the axis index will be one less than the previous index
             * for the remaining axes
             */
            order.erase(order.begin());
            for (auto& axis : order)
                axis--;

            /* optimizations should not break the invariants */
            CV_Assert(output.rank() == input.rank());
            CV_Assert(input.rank() == order.size());
            CV_Assert(input.size() == output.size());
        }

        auto rank = output.rank();
        auto inShape = input.shape_as_vector();
        auto outShape = output.shape_as_vector();

        std::vector<std::size_t> inStride(rank), outStride(rank);
        inStride.back() = 1;
        outStride.back() = 1;
        /* garbage, ..., garbage, 1 */

        std::copy(std::begin(inShape) + 1, std::end(inShape), std::begin(inStride));
        std::copy(std::begin(outShape) + 1, std::end(outShape), std::begin(outStride));
        /* dim[0], dim[1], ..., dim[-1], 1 */

        std::partial_sum(inStride.rbegin(), inStride.rend(), inStride.rbegin(), std::multiplies<std::size_t>());
        std::partial_sum(outStride.rbegin(), outStride.rend(), outStride.rbegin(), std::multiplies<std::size_t>());
        /* stride[0], stride[1], ..., stride[-2], 1 */

        CV_Assert(2 <= rank && rank <= CSL_MAX_TENSOR_RANK);
        permute_dispatcher<T, 2, CSL_MAX_TENSOR_RANK>(rank, stream, order, output, outStride, input, inStride);
    }

    template void permute(const Stream&, TensorSpan<__half>, TensorView<__half>, std::vector<std::size_t>);
    template void permute(const Stream&, TensorSpan<float>, TensorView<float>, std::vector<std::size_t>);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
