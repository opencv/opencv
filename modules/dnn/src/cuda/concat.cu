// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "array.hpp"
#include "types.hpp"
#include "vector_traits.hpp"
#include "grid_stride_range.hpp"
#include "execution.hpp"
#include "kernel_dispatcher.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/tensor.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include "../cuda4dnn/kernels/fill_copy.hpp"

#include <cstddef>
#include <vector>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {
        template <class T, std::size_t N>
        __global__ void concat_vec(
            Span<T> output, size_type output_axis_size, index_type output_axis_offset,
            View<T> input, size_type input_axis_size, size_type concat_size)
        {
            using vector_type = get_vector_type_t<T, N>;

            auto output_vPtr = vector_type::get_pointer(output.data());
            auto input_vPtr = vector_type::get_pointer(input.data());

            /* we need to copy all the elements of input to some location in the output
             * we copy blocks of size `total_concat_size` to some location in the output
             */
            const auto total_concat_size = concat_size * input_axis_size;

            for (auto in_idx : grid_stride_range(input.size() / vector_type::size())) {
                const index_type idx = in_idx * vector_type::size();
                const index_type concat_num = idx / total_concat_size;
                const index_type concat_index = idx % total_concat_size;
                const index_type top_index = concat_index +
                    (concat_num * output_axis_size + output_axis_offset) * concat_size;

                const auto out_idx = top_index / vector_type::size();

                vector_type vec;
                v_load(vec, input_vPtr[in_idx]);
                v_store(output_vPtr[out_idx], vec);
            }
        }

        template <class T, std::size_t Rank>
        __global__ void concat_with_offsets(
            Span<T> output, array<size_type, Rank> out_strides, array<index_type, Rank> out_offset,
            View<T> input, array<size_type, Rank> in_strides)
        {
            for (auto i : grid_stride_range(input.size())) {
                index_type in_index = i / in_strides[0];
                index_type out_index = out_offset[0] + in_index;
                index_type oidx = out_index * out_strides[0];
                for (int j = 1; j < Rank; j++) {
                    in_index = (i % in_strides[j - 1]) / in_strides[j];
                    out_index = out_offset[j] + in_index;
                    oidx += out_index * out_strides[j];
                }

                output[oidx] = input[i];
            }
        }
    }

    template <class T, std::size_t N> static
    void launch_vectorized_concat(const Stream& stream,
        Span<T> output, size_type output_axis_size, index_type output_axis_offset,
        View<T> input, size_type input_axis_size, size_type concat_size)
    {
        CV_Assert(is_fully_aligned<T>(output, N));
        CV_Assert(is_fully_aligned<T>(input, N));
        /* more assertions are required to fully check for vectorization possibility; check concat() */

        auto kernel = raw::concat_vec<T, N>;
        auto policy = make_policy(kernel, input.size() / N, 0, stream);
        launch_kernel(kernel, policy, output, output_axis_size, output_axis_offset, input, input_axis_size, concat_size);
    }

    template <class T>
    void concat(
        const Stream& stream,
        TensorSpan<T> output, std::size_t output_axis_offset,
        TensorView<T> input, std::size_t axis)
    {
        CV_Assert(output.rank() == input.rank());
        CV_Assert(output_axis_offset < output.get_axis_size(axis));

        /* if axes preceding the concat axis are all singleton, the concat blocks are contiguous
         * in the output and we can copy each block directly
         */
        if (output.size_range(0, axis) == 1)
        {
            auto stride = output.size_range(axis + 1, output.rank());
            auto sliced_output = Span<T>(output.get() + output_axis_offset * stride, input.size());
            kernels::copy<T>(stream, sliced_output, input);
            return;
        }

        /* let's call the axis of interest as the channel axis for the purpose of the following discussion
         * even though it can be any axis
         *
         * for each batch item:
         *    we move all the channels from the input (which together, for a single batch item, is contiguous)
         *    of a batch item to its corresponding contiguous place in the output
         *
         * for a valid vector operation:
         * - the size of each copy block must be aligned
         * - input must be aligned
         * - all the destination locations in the output must be aligned
         */
        std::size_t concat_size = output.size_range(axis + 1, output.rank());

        std::size_t input_axis_size = input.get_axis_size(axis);
        std::size_t output_axis_size = output.get_axis_size(axis);

        std::size_t copy_block_size = concat_size * input_axis_size;
        std::size_t copy_block_stride = concat_size * output_axis_size;
        std::size_t starting_offset = output_axis_offset * concat_size;

        /* in a nutshell, all this concat operation does is copy several blocks of size `copy_block_size`
         * to the output starting from `starting_offset` with blocks in the output strided by `copy_block_stride`
         */

        bool is_aligned_4 = copy_block_size % 4 == 0 && copy_block_stride % 4 == 0 && starting_offset % 4 == 0;
        bool is_aligned_2 = copy_block_size % 2 == 0 && copy_block_stride % 2 == 0 && starting_offset % 2 == 0;

        if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(input, 4) && is_aligned_4) {
            launch_vectorized_concat<T, 4>(stream, output, output_axis_size, output_axis_offset, input, input_axis_size, concat_size);
        } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(input, 2) && is_aligned_2) {
            launch_vectorized_concat<T, 2>(stream, output, output_axis_size, output_axis_offset, input, input_axis_size, concat_size);
        } else {
            launch_vectorized_concat<T, 1>(stream, output, output_axis_size, output_axis_offset, input, input_axis_size, concat_size);
        }
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void concat<__half>(const Stream&, TensorSpan<__half>, std::size_t, TensorView<__half>, std::size_t);
#endif
    template void concat<float>(const Stream&, TensorSpan<float>, std::size_t, TensorView<float>,  std::size_t);

    template <class T, std::size_t Rank> static
    void launch_concat_with_offsets(
        const Stream& stream,
        Span<T> output, const std::vector<std::size_t>& outStride, const std::vector<std::size_t>& outOffset,
        View<T> input, const std::vector<std::size_t>& inStride)
    {
        CV_Assert(outStride.size() == Rank);
        CV_Assert(outOffset.size() == Rank);
        CV_Assert(inStride.size() == Rank);

        array<size_type, Rank> outStride_k, inStride_k;
        outStride_k.assign(std::begin(outStride), std::end(outStride));
        inStride_k.assign(std::begin(inStride), std::end(inStride));

        array<index_type, Rank> outOffset_k;
        outOffset_k.assign(std::begin(outOffset), std::end(outOffset));

        auto kernel = raw::concat_with_offsets<T, Rank>;
        auto policy = make_policy(kernel, input.size(), 0, stream);
        launch_kernel(kernel, policy, output, outStride_k, outOffset_k, input, inStride_k);
    }

    GENERATE_KERNEL_DISPATCHER(concat_with_offsets_dispatcher, launch_concat_with_offsets);

    template <class T>
    void concat_with_offsets(
        const Stream& stream,
        TensorSpan<T> output, TensorView<T> input,
        std::vector<std::size_t> offsets)
    {
        CV_Assert(output.rank() == input.rank());
        CV_Assert(output.rank() == offsets.size());

        /* squeezable axes at the beginning of both tensors can be eliminated
         *
         * Reasoning:
         * ----------
         * Suppose an item's indices in the input tensor is [i1, i2, ...]. The indices in the output
         * tensor will be [i1 + off1, i2 + off2, ...]. The concat operation essentially copies items
         * from the input tensor to new locations in the output tensor.
         *
         * If the size of the first axis of the input and output tensor is unity, the input and output
         * indices for all the elements will be of the form be [0, i2, ...] and [0, i2 + off2, ...]
         * respectively. The first index does not contribute to the element's address calculation and
         * hence does nothing apart from eating up few cycles.
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

        /* contiguous axes that undergo full copy can be combined into one axis
         *
         * Reasoning:
         * ----------
         * Suppose an item's indices in the input tensor is [i1, i2, i3, ...]. Let the first two axes not undergo any
         * concatenation. The indices in the output tensor will be [i1, i2, i3 + off3, ...].
         *
         * Each axis in the contiguous axes sequence will add an offset of iN * strideN. In the above example,
         * the two axes add a total offset of `i1 * stride1 + i2 * stride2`. We can merge the two axes into one axis with
         * a size of `size1 * size2`. The new offset added will be i12 * stride2` as the kernel iterates through `i12`.
         * Note that `i12` is actually `(i1 * size2 + i2)` in the original tensor.
         */
        for (int i = 0; i < inShape.size(); i++) {
            /* check if axis `i` requires any slicing */
            if (offsets[i] == 0 && inShape[i] == outShape[i]) {
                /* loop invariant: `i` is the first axis in the contiguous unsliced axis sequence */

                int j = i + 1; /* `j` is the axis which we will attempt to merge */
                while (j < inShape.size() && offsets[j] == 0 && inShape[j] == outShape[j]) {
                    /* `j` axis is also copied fully; merge `i` and `j` */
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

        std::partial_sum(inStride.rbegin(), inStride.rend(), inStride.rbegin(), std::multiplies<int>());
        std::partial_sum(outStride.rbegin(), outStride.rend(), outStride.rbegin(), std::multiplies<int>());
        /* stride[0], stride[1], ..., stride[-2], 1 */

        CV_Assert(1 <= rank && rank <= CSL_MAX_TENSOR_RANK);
        concat_with_offsets_dispatcher<T, 1, CSL_MAX_TENSOR_RANK>(rank, stream, output, outStride, offsets, input, inStride);
    }

    template void concat_with_offsets(const Stream&, TensorSpan<__half>, TensorView<__half>, std::vector<std::size_t>);
    template void concat_with_offsets(const Stream&, TensorSpan<float>, TensorView<float>, std::vector<std::size_t>);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
