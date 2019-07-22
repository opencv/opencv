// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "array.hpp"
#include "types.hpp"
#include "grid_stride_loop.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/tensor.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include <opencv2/core.hpp>

#include <cstddef>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl  { namespace kernels {

    using index_type = gpu::index_type;
    using size_type = gpu::size_type;

    namespace raw {
        template <class T, std::size_t N>
        using array = utils::array<T, N>;

        template <class T, std::size_t N>
        __global__ void permute(
            array<index_type, N> axis_order,
            span<T> output, array<size_type, N> outStrides,
            view<T> input, array<size_type, N> inStrides)
        {
            for (auto i : grid_stride_range(input.size())) {
                index_type oldPosition = 0;
                index_type newPosition = i;

                for (int j = 0; j < N; j++)
                {
                    auto order = axis_order[j];
                    oldPosition += (newPosition / outStrides[j]) * inStrides[order];
                    newPosition %= outStrides[j];
                }

                output[i] = input[oldPosition];
            }
        }
    }

    template <class T, std::size_t N> static
    void launch_permute_kernel(
        const Stream& stream,
        const std::vector<std::size_t>& order,
        span<T> output, const std::vector<std::size_t>& outStride,
        view<T> input, const std::vector<std::size_t>& inStride)
    {
        CV_Assert(order.size() == N);
        CV_Assert(outStride.size() == N);
        CV_Assert(inStride.size() == N);

        utils::array<index_type, N> order_k;
        order_k.assign(std::begin(order), std::end(order));

        utils::array<size_type, N> outStride_k, inStride_k;
        outStride_k.assign(std::begin(outStride), std::end(outStride));
        inStride_k.assign(std::begin(inStride), std::end(inStride));

        auto kernel = raw::permute<T, N>;
        auto policy = make_policy(kernel, input.size(), 0, stream);
        launch_kernel(kernel, policy, order_k, output, outStride_k, input, inStride_k);
    }

    template <class T>
    void permute(
        const Stream& stream,
        TensorSpan<T> output, TensorView<T> input,
        const std::vector<std::size_t>& order)
    {
        CV_Assert(output.rank() == input.rank());
        CV_Assert(input.rank() >= order.size());
        CV_Assert(input.size() == output.size());
        CV_Assert(order.size() >= 3 && order.size() <= 5);
        CV_Assert(get_effective_rank(input) <= order.size());
        CV_Assert(get_effective_rank(output) <= order.size());

        int rank = output.rank();
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

        if (order.size() != rank) {
            auto diff = rank - order.size();
            outStride.erase(outStride.begin(), outStride.begin() + diff);
            inStride.erase(inStride.begin(), inStride.begin() + diff);
        }

        if (rank == 5) {
            launch_permute_kernel<T, 5>(stream, order, output, outStride, input, inStride);
        } else if (rank == 4) {
            launch_permute_kernel<T, 4>(stream, order, output, outStride, input, inStride);
        } else if (rank == 3) {
            launch_permute_kernel<T, 3>(stream, order, output, outStride, input, inStride);
        }
    }

    template void permute(const Stream&, TensorSpan<__half>, TensorView<__half>, const std::vector<std::size_t>&);
    template void permute(const Stream&, TensorSpan<float>, TensorView<float>, const std::vector<std::size_t>&);
    template void permute(const Stream&, TensorSpan<double>, TensorView<double>, const std::vector<std::size_t>&);

}}}}} /*  cv::dnn::cuda4dnn::csl::kernels */
