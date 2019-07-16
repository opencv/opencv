// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "array.hpp"
#include "math.hpp"
#include "types.hpp"
#include "grid_stride_loop.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/tensor.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include <opencv2/core.hpp>

#include <cuda_runtime.h>

#include <cstddef>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl  { namespace kernels {
    using index_type = gpu::index_type;
    using size_type = gpu::size_type;

    namespace raw {
        template <class T, std::size_t N>
        using array = utils::array<T, N>;

        template <class T, std::size_t N>
        __global__ void copy_with_reflection101(
            span<T> output, array<size_type, N> out_strides, array<index_type, N> start, array<index_type, N> end,
            view<T> input, array<size_type, N> in_strides)
        {
            for (auto i : grid_stride_range(output.size())) {
                /* compute output axis indices corresponding to element 'i' */
                array<index_type, N> out_index;
                out_index[0] = i / out_strides[0];
                for (int j = 1; j < N; j++)
                    out_index[j] = (i % out_strides[j - 1]) / out_strides[j];

                /* compute input axis indices corresponding to output axis indices */
                array<index_type, N> in_index;
                for (int j = 0; j < N; j++) {
                    /* if out_index < start, the point is in the left reflection region
                     * the reflected value's index is the absolute value of the difference
                     *
                     * otherwise, if the value is in the copy region, out_index - start gives the input index
                     */
                    using utils::abs;
                    in_index[j] = abs(out_index[j] - start[j]);

                    /* if out_index >= end, it's in the right reflection region */
                    if (out_index[j] >= end[j])
                        in_index[j] = (end[j] - start[j]) - (out_index[j] - end[j]) - 2;
                }

                /* compute input element number from input axis indices */
                index_type iidx = 0;
                for (int j = 0; j < N; j++)
                    iidx += in_index[j] * in_strides[j];

                output[i] = input[iidx];
            }
        }
    }

    template <class T, std::size_t N> static
    void launch_copy_with_reflection101_kernel(
        const Stream& stream,
        span<T> output, const std::vector<std::size_t>& outStride,
        view<T> input, const std::vector<std::size_t>& inStride,
        const std::vector<std::pair<std::size_t, std::size_t>>& ranges)
    {
        CV_Assert(outStride.size() == N);
        CV_Assert(inStride.size() == N);
        CV_Assert(ranges.size() == N);

        utils::array<size_type, N> outStride_k, inStride_k;
        outStride_k.assign(std::begin(outStride), std::end(outStride));
        inStride_k.assign(std::begin(inStride), std::end(inStride));

        utils::array<index_type, N> start_k, end_k;
        for (int i = 0; i < N; i++) {
            start_k[i] = ranges[i].first;
            end_k[i] = ranges[i].second;
        }

        auto kernel = raw::copy_with_reflection101<T, N>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, outStride_k, start_k, end_k, input, inStride_k);
    }

    template <class T>
    void copy_with_reflection101(
        const Stream& stream,
        TensorSpan<T> output, TensorView<T> input,
        const std::vector<std::pair<std::size_t, std::size_t>>& ranges)
    {
        CV_Assert(output.rank == input.rank);
        CV_Assert(output.rank >= 3 && output.rank <= 5);
        CV_Assert(ranges.size() > 0 && ranges.size() < 5);

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

        if (ranges.size() != rank) {
            auto diff = rank - ranges.size();
            outStride.erase(outStride.begin(), outStride.begin() + diff);
            inStride.erase(inStride.begin(), inStride.begin() + diff);
        }

        if (ranges.size() == 4) {
            launch_copy_with_reflection101_kernel<T, 4>(stream, output, outStride, input, inStride, ranges);
        } else if (ranges.size() == 3) {
            launch_copy_with_reflection101_kernel<T, 3>(stream, output, outStride, input, inStride, ranges);
        } else if (ranges.size() == 2) {
            launch_copy_with_reflection101_kernel<T, 2>(stream, output, outStride, input, inStride, ranges);
        } else if (ranges.size() == 1) {
            launch_copy_with_reflection101_kernel<T, 1>(stream, output, outStride, input, inStride, ranges);
        }
    }

    template void copy_with_reflection101(const Stream&, TensorSpan<float>, TensorView<float>, const std::vector<std::pair<std::size_t, std::size_t>>& ranges);
    template void copy_with_reflection101(const Stream&, TensorSpan<double>, TensorView<double>, const std::vector<std::pair<std::size_t, std::size_t>>& ranges);

}}}}} /* cv::dnn::cuda4dnn::csl::kernels */
