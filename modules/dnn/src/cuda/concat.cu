// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>

#include "array.hpp"

#include "../cuda4dnn/csl/kernels.hpp"
#include "../cuda4dnn/csl/kernel_utils.hpp"
#include "../cuda4dnn/csl/tensor.hpp"
#include "../cuda4dnn/csl/pointer.hpp"
#include "../cuda4dnn/csl/stream.hpp"

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl  { namespace kernels {

    namespace raw {
        /* Reference: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/concat_layer.cu */
        template <class T>
        __global__ void concat(
            std::size_t n,
            DevicePtr<T> output, std::size_t output_concat_axis_size, std::size_t output_offset_concat_axis,
            DevicePtr<const T> input, std::size_t concat_size, std::size_t input_concat_axis_size)
        {
            for (auto idx : grid_stride_range(n)) {
                const auto total_concat_size = concat_size * input_concat_axis_size;
                const auto concat_num = idx / total_concat_size;
                const auto concat_index = idx % total_concat_size;
                const auto top_index = concat_index +
                    (concat_num * output_concat_axis_size + output_offset_concat_axis) * concat_size;

                output[top_index] = input[idx];
            }
        }

        template <class T, std::size_t N>
        __global__ void concat_with_axis_offset(
            std::size_t n,
            DevicePtr<T> output, utils::array<int, N> outStrides, utils::array<int, N> outOffset,
            DevicePtr<const T> input, utils::array<int, N> inStrides)
        {
            using utils::array;

            for (auto i : grid_stride_range(n)) {
                /* compute input indices corresponding to element 'i' */
                array<int, N> in_index;
                in_index[0] = i / inStrides[0];
                for (int j = 1; j < N; j++)
                    in_index[j] = (i % inStrides[j - 1]) / inStrides[j];

                /* compute output indices corresponding to element 'i' */
                array<int, N> out_index;
                for (int j = 0; j < N; j++)
                    out_index[j] = outOffset[j] + in_index[j];

                /* compute output element index from output indices */
                int oidx = 0;
                for (int j = 0; j < N; j++)
                    oidx += out_index[j] * outStrides[j];

                output[oidx] = input[i];
            }
        }
    }

    template <class T>
    void concat(
        const Stream& stream,
        TensorSpan<T> output, TensorView<T> input,
        std::size_t concat_size, std::size_t input_concat_axis_size,
        std::size_t output_concat_axis_size, std::size_t output_offset_concat_axis)
    {
        auto policy = make_policy(raw::concat<T>, 0, stream);
        launch_kernel(raw::concat<T>, policy,
            input.size(),
            output.get(), output_concat_axis_size, output_offset_concat_axis,
            input.get(), concat_size, input_concat_axis_size);
    }

    template void concat<float>(
        const Stream& stream,
        TensorSpan<float> output, TensorView<float> input,
        std::size_t concat_size, std::size_t input_concat_axis_size,
        std::size_t output_concat_axis_size, std::size_t output_offset_concat_axis);

    template void concat<double>(
        const Stream& stream,
        TensorSpan<double> output, TensorView<double> input,
        std::size_t concat_size, std::size_t input_concat_axis_size,
        std::size_t output_concat_axis_size, std::size_t output_offset_concat_axis);

    template <class T, std::size_t N> static
    void launch_concat_with_axis_offset_kernel(
        const Stream& stream,
        std::size_t n,
        DevicePtr<T> output, const std::vector<std::size_t>& outStride, const std::vector<std::size_t>& outOffset,
        DevicePtr<const T> input, const std::vector<std::size_t>& inStride)
    {
        CV_Assert(outStride.size() == N);
        CV_Assert(outOffset.size() == N);
        CV_Assert(inStride.size() == N);

        utils::array<int, N> outStride_k, outOffset_k, inStride_k;
        outStride_k.assign(std::begin(outStride), std::end(outStride));
        outOffset_k.assign(std::begin(outOffset), std::end(outOffset));
        inStride_k.assign(std::begin(inStride), std::end(inStride));

        auto kernel = raw::concat_with_axis_offset<T, N>;
        auto policy = make_policy(kernel, 0, stream);
        launch_kernel(kernel, policy, n, output, outStride_k, outOffset_k, input, inStride_k);
    }

    template <class T>
    void concat_with_axis_offset(
        const Stream& stream,
        TensorSpan<T> output, TensorView<T> input,
        const std::vector<std::size_t>& offset)
    {
        CV_Assert(output.rank == input.rank);
        CV_Assert(output.rank >= 3 && output.rank <= 5);

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

        if (offset.size() != rank) {
            auto diff = rank - offset.size();
            outStride.erase(outStride.begin(), outStride.begin() + diff);
            inStride.erase(inStride.begin(), inStride.begin() + diff);
        }

        if (rank == 5) {
            launch_concat_with_axis_offset_kernel<T, 5>(stream, input.size(), output.get(), outStride, offset, input.get(), inStride);
        } else if (rank == 4) {
            launch_concat_with_axis_offset_kernel<T, 4>(stream, input.size(), output.get(), outStride, offset, input.get(), inStride);
        } else if (rank == 3) {
            launch_concat_with_axis_offset_kernel<T, 3>(stream, input.size(), output.get(), outStride, offset, input.get(), inStride);
        }
    }

    template void concat_with_axis_offset(const Stream&, TensorSpan<float>, TensorView<float>, const std::vector<std::size_t>&);
    template void concat_with_axis_offset(const Stream&, TensorSpan<double>, TensorView<double>, const std::vector<std::size_t>&);

}}}}} /*  cv::dnn::cuda4dnn::csl::kernels */
