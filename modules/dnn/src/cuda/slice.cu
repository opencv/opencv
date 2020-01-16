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
#include <iostream>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {
        template <class T, std::size_t Rank>
        __global__ void slice(
            Span<T> output, array<size_type, Rank> out_strides,
            View<T> input, array<size_type, Rank> in_strides, array<index_type, Rank> in_offset)
        {
            for (auto i : grid_stride_range(output.size())) {
                index_type out_index = i / out_strides[0];
                index_type in_index = in_offset[0] + out_index;
                index_type iidx = in_index * in_strides[0];
                for (int j = 1; j < Rank; j++) {
                    out_index = (i % out_strides[j - 1]) / out_strides[j];
                    in_index = in_offset[j] + out_index;
                    iidx += in_index * in_strides[j];
                }

                output[i] = input[iidx];
            }
        }
    }

    template <class T, std::size_t Rank> static
    void launch_slice(
        const Stream& stream,
        Span<T> output, const std::vector<std::size_t>& outStride,
        View<T> input, const std::vector<std::size_t>& inStride, const std::vector<std::size_t>& inOffset)
    {
        CV_Assert(outStride.size() == Rank);
        CV_Assert(inStride.size() == Rank);
        CV_Assert(inOffset.size() == Rank);

        array<size_type, Rank> outStride_k, inStride_k;
        outStride_k.assign(std::begin(outStride), std::end(outStride));
        inStride_k.assign(std::begin(inStride), std::end(inStride));

        array<index_type, Rank> inOffset_k;
        inOffset_k.assign(std::begin(inOffset), std::end(inOffset));

        auto kernel = raw::slice<T, Rank>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, outStride_k, input, inStride_k, inOffset_k);
    }

    GENERATE_KERNEL_DISPATCHER(slice_dispatcher, launch_slice);

    template <class T>
    void slice(const Stream& stream,
        TensorSpan<T> output, TensorView<T> input,
        std::vector<std::size_t> offsets)
    {
        CV_Assert(output.rank() == input.rank());
        CV_Assert(output.rank() == offsets.size());

        /* squeezable axes at the beginning of both tensors can be eliminated
         *
         * Reasoning:
         * ----------
         * Suppose an item's indices in the output tensor is [o1, o2, ...]. The indices in the input
         * tensor will be [o1 + off1, o2 + off2, ...]. The rest of the elements in the input are ignored.
         *
         * If the size of the first axis of the input and output tensor is unity, the input and output indices
         * for all the elements will be of the form be [0, o2 + off2, ...] and [0, o2, ...] respectively. Note that
         * there cannot be any ignored items since the axes have unit size. The first index does not contribute to the
         * element's address calculation and hence does nothing apart from eating up few cycles.
         */
        while (input.get_axis_size(0) == 1 && output.get_axis_size(0) == 1) {
            CV_Assert(offsets[0] == 0);

            input.squeeze(0);
            output.squeeze(0);
            offsets.erase(std::begin(offsets));

            CV_Assert(output.rank() == input.rank());
            CV_Assert(output.rank() == offsets.size());
        }

        auto inShape = input.shape_as_vector();
        auto outShape = output.shape_as_vector();

        /* contiguous axes which do not undergo slicing can be combined into one axis
         *
         * Reasoning:
         * ----------
         * Suppose an item's indices in the output tensor is [o1, o2, o3, ...]. Let the first two axes not undergo any
         * slicing. The indices in the input tensor will be [o1, o2, o3 + off3, ...].
         *
         * Each axis in the contiguous unsliced axes sequence will add an offset of iN * strideN. In the above example,
         * the two axes add a total offset of `o1 * stride1 + o2 * stride2`. We can merge the two axes into one axis with
         * a size of `size1 * size2`. The new offset added will be o12 * stride2` as the kernel iterates through `o12`.
         * Note that `o12` is actually `(o1 * size2 + o2)` in the original tensor.
         */
        for (int i = 0; i < inShape.size(); i++) {
            /* check if axis `i` requires any slicing */
            if (offsets[i] == 0 && inShape[i] == outShape[i]) {
                /* loop invariant: `i` is the first axis in the contiguous unsliced axis sequence */

                int j = i + 1; /* `j` is the axis which we will attempt to merge */
                while (j < inShape.size() && offsets[j] == 0 && inShape[j] == outShape[j]) {
                    /* `j` axis is also unsliced; merge `i` and `j` */
                    auto new_size = inShape[i] * inShape[j];
                    inShape[i] = new_size;
                    outShape[i] = new_size;
                    offsets[i] = 0; /* redundant */

                    /* delete axis `j` */
                    inShape.erase(std::begin(inShape) + j);
                    outShape.erase(std::begin(outShape) + j);
                    offsets.erase(std::begin(offsets) + j);

                    /* optimizations should not break the invariants */
                    CV_Assert(inShape.size() == outShape.size());
                    CV_Assert(inShape.size() == offsets.size());
                    CV_Assert(inShape[i] == outShape[i]);
                    CV_Assert(offsets[i] == 0);
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

        std::partial_sum(inStride.rbegin(), inStride.rend(), inStride.rbegin(), std::multiplies<std::size_t>());
        std::partial_sum(outStride.rbegin(), outStride.rend(), outStride.rbegin(), std::multiplies<std::size_t>());
        /* stride[0], stride[1], ..., stride[-2], 1 */

        CV_Assert(1 <= rank && rank <= CSL_MAX_TENSOR_RANK);
        slice_dispatcher<T, 1, CSL_MAX_TENSOR_RANK>(rank, stream, output, outStride, input, inStride, offsets);
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void slice(const Stream&, TensorSpan<__half>, TensorView<__half>, std::vector<std::size_t>);
#endif
    template void slice(const Stream&, TensorSpan<float>, TensorView<float>, std::vector<std::size_t>);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
