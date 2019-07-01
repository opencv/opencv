// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "array.hpp"

#include "../cuda4dnn/csl/kernels.hpp"
#include "../cuda4dnn/csl/kernel_utils.hpp"
#include "../cuda4dnn/csl/tensor.hpp"
#include "../cuda4dnn/csl/stream.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <cuda_runtime.h>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl  { namespace kernels {

    namespace raw {
        template <class T, std::size_t N>
        using array = utils::array<T, N>;

        template <class T, std::size_t N>
        __global__ void permute(
            array<int, N> axis_order,
            span<T> output, array<int, N> outStrides,
            view<T> input, array<int, N> inStrides)
        {
            for (auto i : grid_stride_range(input.size())) {
                int oldPosition = 0;
                int newPosition = i;

                for (int j = 0; j < N; j++)
                {
                    int order = axis_order[j];
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

        utils::array<int, N> order_k, outStride_k, inStride_k;
        order_k.assign(std::begin(order), std::end(order));
        outStride_k.assign(std::begin(outStride), std::end(outStride));
        inStride_k.assign(std::begin(inStride), std::end(inStride));

        auto kernel = raw::permute<T, N>;
        auto policy = make_policy(kernel, 0, stream);
        launch_kernel(kernel, policy, order_k, output, outStride_k, input, inStride_k);
    }

    template <class T>
    void permute(
        const Stream& stream,
        TensorSpan<T> output, TensorView<T> input,
        const std::vector<std::size_t>& order)
    {
        CV_Assert(output.rank == input.rank);
        CV_Assert(input.size() == output.size());
        CV_Assert(order.size() >= 3 && order.size() <= 5);
        CV_Assert(input.rank >= order.size());
        CV_Assert(output.rank >= order.size());
        CV_Assert(get_effective_rank(input) <= order.size());
        CV_Assert(get_effective_rank(output) <= order.size());

        int rank = output.rank;
        auto inShape = input.shape();
        auto outShape = output.shape();

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

    template void permute(const Stream&, TensorSpan<float>, TensorView<float>, const std::vector<std::size_t>&);
    template void permute(const Stream&, TensorSpan<double>, TensorView<double>, const std::vector<std::size_t>&);

}}}}} /*  cv::dnn::cuda4dnn::csl::kernels */
