// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "array.hpp"
#include "math.hpp"
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
#include <utility>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {
        template <class T, std::size_t Rank>
        __global__ void copy_with_reflection101(
            Span<T> output, array<size_type, Rank> out_strides, array<index_type, Rank> start, array<index_type, Rank> end,
            View<T> input, array<size_type, Rank> in_strides)
        {
            for (auto i : grid_stride_range(output.size())) {
                /* compute output axis indices corresponding to element 'i' */
                array<index_type, Rank> out_index;
                out_index[0] = i / out_strides[0];
                for (int j = 1; j < Rank; j++)
                    out_index[j] = (i % out_strides[j - 1]) / out_strides[j];

                /* compute input axis indices corresponding to output axis indices */
                array<index_type, Rank> in_index;
                for (int j = 0; j < Rank; j++) {
                    /* if out_index < start, the point is in the left reflection region
                     * the reflected value's index is the absolute value of the difference
                     *
                     * otherwise, if the value is in the copy region, out_index - start gives the input index
                     */
                    using device::abs;
                    in_index[j] = abs(out_index[j] - start[j]);

                    /* if out_index >= end, it's in the right reflection region */
                    if (out_index[j] >= end[j])
                        in_index[j] = (end[j] - start[j]) - (out_index[j] - end[j]) - 2;
                }

                /* compute input element number from input axis indices */
                index_type iidx = 0;
                for (int j = 0; j < Rank; j++)
                    iidx += in_index[j] * in_strides[j];

                output[i] = input[iidx];
            }
        }
    }

    template <class T, std::size_t Rank> static
    void launch_copy_with_reflection101(
        const Stream& stream,
        Span<T> output, const std::vector<std::size_t>& outStride,
        View<T> input, const std::vector<std::size_t>& inStride,
        const std::vector<std::pair<std::size_t, std::size_t>>& ranges)
    {
        CV_Assert(outStride.size() == Rank);
        CV_Assert(inStride.size() == Rank);
        CV_Assert(ranges.size() == Rank);

        array<size_type, Rank> outStride_k, inStride_k;
        outStride_k.assign(std::begin(outStride), std::end(outStride));
        inStride_k.assign(std::begin(inStride), std::end(inStride));

        array<index_type, Rank> start_k, end_k;
        for (int i = 0; i < Rank; i++) {
            start_k[i] = ranges[i].first;
            end_k[i] = ranges[i].second;
        }

        auto kernel = raw::copy_with_reflection101<T, Rank>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, outStride_k, start_k, end_k, input, inStride_k);
    }

    GENERATE_KERNEL_DISPATCHER(copy_with_reflection101_dispatcher, launch_copy_with_reflection101);

    template <class T>
    void copy_with_reflection101(
        const Stream& stream,
        TensorSpan<T> output, TensorView<T> input,
        std::vector<std::pair<std::size_t, std::size_t>> ranges)
    {
        CV_Assert(output.rank() == input.rank());
        CV_Assert(output.rank() == ranges.size());

        /* squeezable axes at the beginning of both tensors can be eliminated
         *
         * Reasoning:
         * ----------
         * Suppose an item's indices in the input tensor is [i1, i2, ...]. The indices in the
         * output tensor will be [i1 + off1, i2 + off2, ...]. The rest of the elements in the output are padding.
         * The padding operation essentially copies items from the input tensor to new locations in the output tensor
         * and pads the remaining.
         *
         * If the size of the first axis of the input and output tensor is unity, the input and output indices
         * for all the elements will be of the form be [0, i2, ...] and [0, i2 + off2, ...] respectively. Note that
         * there cannot be extra padding since the axes have unit size. The first index does not contribute to the
         * element's address calculation and hence does nothing apart from eating up few cycles.
         */
        while (input.get_axis_size(0) == 1 && output.get_axis_size(0) == 1) {
            CV_Assert(ranges[0].first == 0 && ranges[0].second == 1);

            input.squeeze(0);
            output.squeeze(0);
            ranges.erase(std::begin(ranges));

            CV_Assert(output.rank() == input.rank());
            CV_Assert(output.rank() == ranges.size());
        }

        auto inShape = input.shape_as_vector();
        auto outShape = output.shape_as_vector();

        /* contiguous axes which do not have any padding can be combined into one axis
         *
         * Reasoning:
         * ----------
         * Suppose an item's indices in the input tensor is [i1, i2, i3, ...]. Let the first two axes not have any
         * padding. The indices in the output tensor will be [i1, i2, i3 + off3, ...].
         *
         * Each axis in the contiguous unpadded axes sequence will add an offset of iN * strideN. In the above example,
         * the two axes add a total offset of `i1 * stride1 + i2 * stride2`. We can merge the two axes into one axis with
         * a size of `size1 * size2`. The new offset added will be `i12 * stride2` as the kernel iterates through `i12`.
         * Note that `i12` is actually `(i1 * size2 + i2)` in the original tensor.
         */
        for (int i = 0; i < inShape.size(); i++) {
            /* check if axis `i` requires any padding */
            if (ranges[i].first == 0 && ranges[i].second == inShape[i]) {
                /* loop invariant: `i` is the first axis in the contiguous unpadded axis sequence */
                CV_Assert(inShape[i] == outShape[i]);

                /* we now iterate through the axes which follow and try to merge */
                int j = i + 1; /* `j` is the axis which we will attempt to merge */
                while (j < inShape.size() && ranges[j].first == 0 && ranges[j].second == inShape[j]) {
                    CV_Assert(inShape[j] == outShape[j]);

                    /* `j` is also unpadded; merge `i` and `j` */
                    auto new_size = inShape[i] * inShape[j];
                    inShape[i] = new_size;
                    outShape[i] = new_size;
                    ranges[i].second = new_size;

                    /* delete axis `j` */
                    inShape.erase(std::begin(inShape) + j);
                    outShape.erase(std::begin(outShape) + j);
                    ranges.erase(std::begin(ranges) + j);

                    /* optimizations should not break the invariants */
                    CV_Assert(inShape.size() == outShape.size());
                    CV_Assert(inShape.size() == ranges.size());
                    CV_Assert(inShape[i] == outShape[i]);
                    CV_Assert(ranges[i].first == 0 && ranges[i].second == inShape[i]);
                }
            }
        }

        auto rank = inShape.size();

        std::vector<std::size_t> inStride(rank), outStride(rank);
        inStride.back() = 1;
        outStride.back() = 1;
        /* garbage, ..., garbage, 1 */

        std::copy(std::begin(inShape) + 1, std::end(inShape), std::begin(inStride));
        std::copy(std::begin(outShape) + 1, std::end(outShape), std::begin(outStride));
        /* dim[0], dim[1], ..., dim[-1], 1 */

        std::partial_sum(inStride.rbegin(), inStride.rend(), inStride.rbegin(), std::multiplies<int>());
        std::partial_sum(outStride.rbegin(), outStride.rend(), outStride.rbegin(), std::multiplies<int>());
        /* stride[0], stride[1], ..., stride[-2], 1 */

        CV_Assert(1 <= rank && rank <= CSL_MAX_TENSOR_RANK);
        copy_with_reflection101_dispatcher<T, 1, CSL_MAX_TENSOR_RANK>(rank, stream, output, outStride, input, inStride, ranges);
    }

    template void copy_with_reflection101(const Stream&, TensorSpan<__half>, TensorView<__half>, std::vector<std::pair<std::size_t, std::size_t>> ranges);
    template void copy_with_reflection101(const Stream&, TensorSpan<float>, TensorView<float>, std::vector<std::pair<std::size_t, std::size_t>> ranges);

}}}} /* namespace namespace cv::dnn::cuda4dnn::kernels */
