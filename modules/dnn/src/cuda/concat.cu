// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "array.hpp"
#include "types.hpp"
#include "vector_traits.hpp"
#include "grid_stride_loop.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/tensor.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include <cstddef>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl  { namespace kernels {

    using index_type = gpu::index_type;
    using size_type = gpu::size_type;

    namespace raw {
        /* Reference: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/concat_layer.cu */
        template <class T, std::size_t N>
        __global__ void concat_vec(
            span<T> output, size_type output_axis_size, index_type output_axis_offset,
            view<T> input, size_type input_axis_size, size_type concat_size)
        {
            using vector_type = typename get_vector_type<T, N>::type;

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

        template <class T, std::size_t N>
        using array = utils::array<T, N>;

        template <class T, std::size_t N>
        __global__ void concat_with_offsets(
            span<T> output, array<size_type, N> out_strides, array<index_type, N> out_offset,
            view<T> input, array<size_type, N> in_strides)
        {
            for (auto i : grid_stride_range(input.size())) {
                index_type in_index = i / in_strides[0];
                index_type out_index = out_offset[0] + in_index;
                index_type oidx = out_index * out_strides[0];
                for (int j = 1; j < N; j++) {
                    in_index = (i % in_strides[j - 1]) / in_strides[j];
                    out_index = out_offset[j] + in_index;
                    oidx += out_index * out_strides[j];
                }

                output[oidx] = input[i];
            }
        }
    }

    template <class T, std::size_t N>
    void launch_vectorized_concat(const Stream& stream,
        span<T> output, size_type output_axis_size, index_type output_axis_offset,
        view<T> input, size_type input_axis_size, size_type concat_size)
    {
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
        /* let's call the axis of interest as the channel axis for the purpose of the following discussion
         * even though it can be any axis
         *
         * for each batch item:
         *    we move all the channels from the input (which together for a single batch item is contiguous)
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

    template void concat<__half>(const Stream&, TensorSpan<__half>, std::size_t, TensorView<__half>, std::size_t);
    template void concat<float>(const Stream&, TensorSpan<float>, std::size_t, TensorView<float>,  std::size_t);
    template void concat<double>(const Stream&, TensorSpan<double>, std::size_t, TensorView<double>, std::size_t);

    template <class T, std::size_t N> static
    void launch_concat_with_offsets_kernel(
        const Stream& stream,
        span<T> output, const std::vector<std::size_t>& outStride, const std::vector<std::size_t>& outOffset,
        view<T> input, const std::vector<std::size_t>& inStride)
    {
        CV_Assert(outStride.size() == N);
        CV_Assert(outOffset.size() == N);
        CV_Assert(inStride.size() == N);

        utils::array<size_type, N> outStride_k, inStride_k;
        outStride_k.assign(std::begin(outStride), std::end(outStride));
        inStride_k.assign(std::begin(inStride), std::end(inStride));

        utils::array<index_type, N> outOffset_k;
        outOffset_k.assign(std::begin(outOffset), std::end(outOffset));

        auto kernel = raw::concat_with_offsets<T, N>;
        auto policy = make_policy(kernel, input.size(), 0, stream);
        launch_kernel(kernel, policy, output, outStride_k, outOffset_k, input, inStride_k);
    }

    template <class T>
    void concat_with_offsets(
        const Stream& stream,
        TensorSpan<T> output, TensorView<T> input,
        const std::vector<std::size_t>& offsets)
    {
        CV_Assert(output.rank() == input.rank());
        CV_Assert(output.rank() >= 3 && output.rank() <= 5);

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

        std::partial_sum(inStride.rbegin(), inStride.rend(), inStride.rbegin(), std::multiplies<int>());
        std::partial_sum(outStride.rbegin(), outStride.rend(), outStride.rbegin(), std::multiplies<int>());
        /* stride[0], stride[1], ..., stride[-2], 1 */

        if (offsets.size() != rank) {
            auto diff = rank - offsets.size();
            outStride.erase(outStride.begin(), outStride.begin() + diff);
            inStride.erase(inStride.begin(), inStride.begin() + diff);
        }

        if (rank == 5) {
            launch_concat_with_offsets_kernel<T, 5>(stream, output, outStride, offsets, input, inStride);
        } else if (rank == 4) {
            launch_concat_with_offsets_kernel<T, 4>(stream, output, outStride, offsets, input, inStride);
        } else if (rank == 3) {
            launch_concat_with_offsets_kernel<T, 3>(stream, output, outStride, offsets, input, inStride);
        }
    }

    template void concat_with_offsets(const Stream&, TensorSpan<__half>, TensorView<__half>, const std::vector<std::size_t>&);
    template void concat_with_offsets(const Stream&, TensorSpan<float>, TensorView<float>, const std::vector<std::size_t>&);
    template void concat_with_offsets(const Stream&, TensorSpan<double>, TensorView<double>, const std::vector<std::size_t>&);

}}}}} /*  cv::dnn::cuda4dnn::csl::kernels */
