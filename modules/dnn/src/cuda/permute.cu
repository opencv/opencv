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

        template <class T, int TILE_SIZE, std::size_t N>
        __global__ void transpose(Span<T> output, View<T> input, size_type in_width, size_type out_width)
        {
            using vector_type = get_vector_type_t<T, N>;

            __shared__ T tile[TILE_SIZE][TILE_SIZE + 1];

            /* blockDim.y = TILE_SIZE, blockDim.x = TILE_SIZE/N */
            const index_type in_x = blockIdx.x * TILE_SIZE + threadIdx.x * N;
            const index_type in_y = blockIdx.y * TILE_SIZE + threadIdx.y;

            /* Every valid input location has a corresponding output location and vice versa.
             * Hence, if we do not load values into the shared memory for a given location, we
             * also won't read them for storing in the output.
             */
            if (in_x < in_width && in_y < out_width)
            {
                vector_type vec;
                auto input_vPtr = vector_type::get_pointer(input.data());
                v_load(vec, input_vPtr[(in_y * in_width + in_x) / N]);

                for (int i = 0; i < vector_type::size(); i++)
                    tile[threadIdx.y][threadIdx.x * N + i] = vec.data[i];
            }

            __syncthreads();

            /* Note that `blockDim.x * N` is equal to `blockDim.y`. Since there are an equal
             * number of them, we can interchange `threadIdx.x` and `threadIdx.y` without changing
             * result. The advantage of interchanging is that consecutive output indices map to
             * consecutive threads. This would allow writes across threds in a warp to be coalesced.
             */
            const index_type out_x = blockIdx.y * TILE_SIZE + threadIdx.x * N;
            const index_type out_y = blockIdx.x * TILE_SIZE + threadIdx.y;

            if (out_x < out_width && out_y < in_width)
            {
                vector_type vec;
                for (int i = 0; i < vector_type::size(); i++)
                    vec.data[i] = tile[threadIdx.x * N + i][threadIdx.y];

                auto output_vPtr = vector_type::get_pointer(output.data());
                v_store(output_vPtr[(out_y * out_width + out_x) / N], vec);
            }
        }
    }

    template <class T, std::size_t N> static
    void launch_transpose_kernel(const Stream& stream, Span<T> output, View<T> input, size_type in_width, size_type out_width)
    {
        CV_Assert(is_fully_aligned<T>(output, N));
        CV_Assert(is_fully_aligned<T>(input, N));
        CV_Assert(in_width % N == 0);
        CV_Assert(out_width % N == 0);

        constexpr int TILE_SIZE = 32;
        constexpr int TILE_SIZE_X = TILE_SIZE/N, TILE_SIZE_Y = TILE_SIZE;
        auto kernel = raw::transpose<T, TILE_SIZE, N>;

        dim3 grid_size((in_width/N + TILE_SIZE_X - 1)/TILE_SIZE_X, (out_width + TILE_SIZE_Y - 1)/TILE_SIZE_Y);
        dim3 block_size(TILE_SIZE_X, TILE_SIZE_Y);
        auto policy = execution_policy(grid_size, block_size, stream);

        launch_kernel(kernel, policy, output, input, in_width, out_width);
    }

    template <class T>
    void transpose(const Stream& stream, Span<T> output, View<T> input, std::size_t in_width, std::size_t out_width)
    {
        if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(input, 4) && in_width % 4 == 0 && out_width % 4 == 0) {
            launch_transpose_kernel<T, 4>(stream, output, input, in_width, out_width);
        } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(input, 2) && in_width % 2 == 0 && out_width % 2 == 0) {
            launch_transpose_kernel<T, 2>(stream, output, input, in_width, out_width);
        } else {
            launch_transpose_kernel<T, 1>(stream, output, input, in_width, out_width);
        }
    }

    template void transpose(const Stream&, Span<__half>, View<__half>, std::size_t, std::size_t);
    template void transpose(const Stream&, Span<float>, View<float>, std::size_t, std::size_t);

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

        auto rank = output.rank();
        auto inShape = input.shape_as_vector();
        auto outShape = output.shape_as_vector();

        /* singleton axes do not contribute towards address calculation
         *
         * Reasoning:
         * ----------
         * Suppose an item's indices in the input tensor is [i1, i2, ...]. The indices in the
         * output tensor will be some permutation of the input tensor indices. Let the output
         * tensor indices be [o1, o2, ...]. The permutation operation essentially copies items
         * from the input tensor to new locations in the output tensor as dictated by the indices.
         *
         * If the size of the nth axis (say i2) of the input is one the input and output indicies for
         * all the elements will be of the form be [i1, 0, ...] and [..., 0, ...] respectively.
         * The index does not contribute to the element's address calculation and hence would give
         * identical result if it weren't there.
         */
        for (int i = 0; i < rank; i++)
        {
            /* index `i` corresponds to the axis index in the output; order[i] has the corresponding axis index in the input */
            while (i < rank && outShape[i] == 1)
            {
                int in_i = order[i];
                CV_Assert(inShape[in_i] == 1);

                /* delete axis `i` */
                inShape.erase(std::begin(inShape) + in_i);
                outShape.erase(std::begin(outShape) + i);

                /* deletion of an axis reduces an axis in the input tensor which would cause the indices
                 * of the axes that come after the deleted axis to reduce by one
                 */
                order.erase(order.begin() + i);
                for (auto& axis : order)
                    if (axis > in_i)
                        axis--;

                rank--;

                /* optimizations should not break the invariants */
                CV_Assert(rank == order.size());
                CV_Assert(inShape.size() == order.size());
                CV_Assert(outShape.size() == order.size());
                CV_Assert(input.size() == output.size());
            }
        }

        /* contiguous axes whose relative ordering stays same before and after permutation can be merged into one axis
         * example: in permute order 0 2 3 1, axes 2 and 3 can be grouped into a single axis
         *
         * Reasoning:
         * ----------
         * Suppose an item's indices in the input tensor is [i0, i1, i2, i3, ...]. Let the permutation order be [0, 3, 1, 2, ...].
         * Note that i1 and i2 are adjacent axes in the same order in input as well as output. The indices in the output tensor
         * will be [i0, i3, i1, i2, ...].
         *
         * Each axis in the contiguous axes sequence will add an offset of iN * strideN. In the above example,
         * the two axes add a total offset of `i1 * (size2 * stride2) + i2 * stride2` which is `(i1 * size2 + i2) * stride2`,
         * in both input and output. Note stride2 can be different in the input and output. We can merge the two axes into one axis
         * with a size of `size1 * size2`. The new offset added will be `i12 * stride12` as the kernel iterates through `i12`. Note
         * that `i12` is actually `(i1 * size2 + i2)` and `stride12` is `stride2`.
         */
         for (int i = 0; i < rank; i++) {
            /* the indices used in the loops such as `i` and `j` are axis indices in the output tensor */
            /* the corresponding input axis indices are `order[i]` and `order[j]`*/

            /* loop invariant: `i` is the first axis in the contiguous unpermuted axis sequence */

            int j = i + 1; /* `j` is the axis which we will attempt to merge */
            while (j < rank && (order[i] + 1) == order[j]) {
                /* axis `i` and axis `j` do not change relative order */

                auto in_i = order[i], in_j = order[j];

                auto new_size = inShape[in_i] * inShape[in_j];
                inShape[in_i] = new_size;
                outShape[i] = new_size;

                /* delete axis `j` */
                inShape.erase(std::begin(inShape) + in_j);
                outShape.erase(std::begin(outShape) + j);

                /* deletion of an axis reduces an axis in the input tensor which would cause the indices
                 * of the axes that come after the deleted axis to reduce by one
                 */
                order.erase(order.begin() + j);
                for (auto& axis : order)
                    if (axis > order[i])
                        axis--;

                rank--;

                /* optimizations should not break the invariants */
                CV_Assert(rank == order.size());
                CV_Assert(inShape.size() == order.size());
                CV_Assert(outShape.size() == order.size());
                CV_Assert(input.size() == output.size());
            }
        }

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

        const bool is_in_order = [&order] {
            for (int i = 0; i < order.size(); i++)
                if (order[i] != i)
                    return false;
            return true;
        }();

        if (is_in_order)
        {
            kernels::copy<T>(stream, output, input);
        }
        else if(rank == 2)
        {
            /* use the more efficient transpose kernel */
            transpose<T>(stream, output, input, inShape[1], outShape[1]);
        }
        else
        {
            CV_Assert(3 <= rank && rank <= CSL_MAX_TENSOR_RANK);
            permute_dispatcher<T, 3, CSL_MAX_TENSOR_RANK>(rank, stream, order, output, outStride, input, inStride);
        }
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void permute(const Stream&, TensorSpan<__half>, TensorView<__half>, std::vector<std::size_t>);
#endif
    template void permute(const Stream&, TensorSpan<float>, TensorView<float>, std::vector<std::size_t>);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
