// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "math.hpp"
#include "array.hpp"
#include "limits.hpp"
#include "types.hpp"
#include "grid_stride_loop.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/tensor.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl  { namespace kernels {

    using index_type = gpu::index_type;
    using size_type = gpu::size_type;

    namespace raw {

        template <class T, std::size_t N>
        using array = utils::array<T, N>;

        template <class T, std::size_t Order>
        __global__ void max_pooling_with_indices(
            span<T> output, span<T> indices, view<T> input, size_type channels,
            array<size_type, Order> out_spatial_dims, array<size_type, Order> in_spatial_dims,
            array<size_type, Order> window_size, array<size_type, Order> strides, array<size_type, Order> padding_left)
        {
            /* every element in the output is mapped to a window in the input and each thread processes several windows */
            for (auto idx : grid_stride_range(output.size())) {
                size_type out_spatial_size = 1;
                array<index_type, Order> window_idx;
                for (int i = Order - 1; i >= 0; i--) {
                    window_idx[i] = (idx / out_spatial_size) % out_spatial_dims[i];
                    out_spatial_size *= out_spatial_dims[i];
                }

                const index_type n = idx / (out_spatial_size * channels);
                const index_type c = (idx / out_spatial_size) % channels;

                array<index_type, Order> start;
                for(int i = 0; i < Order; i++)
                    start[i] = window_idx[i] * strides[i] - padding_left[i];

                array<index_type, Order> end;
                for (int i = 0; i < Order; i++) {
                    using utils::min;
                    end[i] = min<index_type>(start[i] + window_size[i], in_spatial_dims[i]);
                }

                for (int i = 0; i < Order; i++) {
                    using utils::max;
                    start[i] = max(start[i], 0);
                }

                T max_value = gpu::numeric_limits<T>::lowest();
                index_type max_idx = -1;

                size_type in_spatial_size = 1;
                for (int i = 0; i < Order; i++)
                    in_spatial_size *= in_spatial_dims[i];

                const auto outer_offset =  (n * channels + c) * in_spatial_size;
                if (Order == 2) {
                    array<index_type, Order> idx;
                    for (idx[0] = start[0]; idx[0] != end[0]; idx[0]++) {
                        for (idx[1] = start[1]; idx[1] != end[1]; idx[1]++) {
                            index_type offset = 0;
                            index_type stride = 1;
                            for (int i = Order - 1; i >= 0; i--) {
                                offset += stride * idx[i];
                                stride *= in_spatial_dims[i];
                            }

                            if (input[outer_offset + offset] > max_value) {
                                max_idx = offset;
                                max_value = input[outer_offset + offset];
                            }
                        }
                    }
                } else if(Order == 3) {
                    array<index_type, Order> idx;
                    for (idx[0] = start[0]; idx[0] != end[0]; idx[0]++) {
                        for (idx[1] = start[1]; idx[1] != end[1]; idx[1]++) {
                            for (idx[2] = start[2]; idx[2] != end[2]; idx[2]++) {
                                index_type offset = 0;
                                index_type stride = 1;
                                for (int i = Order - 1; i >= 0; i--) {
                                    offset += stride * idx[i];
                                    stride *= in_spatial_dims[i];
                                }

                                if (input[outer_offset + offset] > max_value) {
                                    max_idx = offset;
                                    max_value = input[outer_offset + offset];
                                }
                            }
                        }
                    }
                }

                output[idx] = max_value;
                indices[idx] = max_idx;
            }
        }

        template <class T, std::size_t Order>
        __global__ void max_unpooling(
            span<T> output, view<T> input, view<T> indices, size_type channels,
            array<size_type, Order> out_spatial_dims, array<size_type, Order> in_spatial_dims,
            array<size_type, Order> window_size, array<size_type, Order> strides, array<size_type, Order> padding_left)
        {
            /* the output has already been zero filled */
            /* Every input value represents a window in the output. The max unpooling operation
             * copies the input value to exactly one location in the output window which is given
             * by the indices tensor.
             */
            for (auto idx : grid_stride_range(input.size())) {
                size_type in_spatial_size = 1;
                array<index_type, Order> window_idx;
                for (int i = Order - 1; i >= 0; i--) {
                    window_idx[i] = (idx / in_spatial_size) % in_spatial_dims[i];
                    in_spatial_size *= in_spatial_dims[i];
                }

                const index_type n = idx / (in_spatial_size * channels);
                const index_type c = (idx / in_spatial_size) % channels;

                array<index_type, Order> start;
                for (int i = 0; i < Order; i++) {
                    using utils::min;
                    using utils::max;
                    start[i] = max(0, min(window_idx[i] * strides[i] - padding_left[i], out_spatial_dims[i] - 1));
                }

                size_type out_spatial_size = 1;
                for (int i = 0; i < Order; i++)
                    out_spatial_size *= out_spatial_dims[i];

                index_type outer_offset = (n * channels + c) * out_spatial_size;
                output[outer_offset + static_cast<index_type>(indices[idx])] = input[idx];
            }
        }
    }

    template <class T, std::size_t Order> static
    void launch_max_pooling_kernel(
        const Stream& stream,
        span<T> output, span<T> indices, view<T> input, std::size_t channels,
        const std::vector<std::size_t>& out_spatial_dims, const std::vector<std::size_t>& in_spatial_dims,
        const std::vector<std::size_t>& window_size,
        const std::vector<std::size_t>& strides, const std::vector<std::size_t>& padding_left)
    {
        CV_Assert(out_spatial_dims.size() == Order);
        CV_Assert(in_spatial_dims.size() == Order);
        CV_Assert(window_size.size() == Order);
        CV_Assert(strides.size() == Order);
        CV_Assert(padding_left.size() == Order);
        CV_Assert(indices.size() == output.size());

        utils::array<size_type, Order> out_spatial_dims_k, in_spatial_dims_k;
        out_spatial_dims_k.assign(std::begin(out_spatial_dims), std::end(out_spatial_dims));
        in_spatial_dims_k.assign(std::begin(in_spatial_dims), std::end(in_spatial_dims));

        utils::array<size_type, Order> window_size_k, strides_k, padding_left_k;
        window_size_k.assign(std::begin(window_size), std::end(window_size));
        strides_k.assign(std::begin(strides), std::end(strides));
        padding_left_k.assign(std::begin(padding_left), std::end(padding_left));

        auto kernel = raw::max_pooling_with_indices<T, Order>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, indices, input, channels,
            out_spatial_dims_k, in_spatial_dims_k, window_size_k, strides_k, padding_left_k);
    }

    template <class T>
    void max_pooling_with_indices(
        const Stream& stream,
        TensorSpan<T> output, TensorSpan<T> indices, TensorView<T> input,
        const std::vector<std::size_t>& window_size, const std::vector<std::size_t>& strides,
        const std::vector<std::size_t>& padding_left)
    {
        auto order = window_size.size();
        CV_Assert(strides.size() == order);
        CV_Assert(padding_left.size() == order);
        CV_Assert(2 <= order && order <= 3);

        CV_Assert(is_shape_same(output, indices));
        CV_Assert(output.rank() == order + 2);
        CV_Assert(input.rank() == order + 2);

        CV_Assert(input.get_axis_size(1) == output.get_axis_size(1));

        std::vector<std::size_t> out_spatial_dims(order), in_spatial_dims(order);
        for (int i = 0; i < order; i++) {
            in_spatial_dims[i] = input.get_axis_size(2 + i);
            out_spatial_dims[i] = output.get_axis_size(2 + i);
        }

        std::size_t channels = input.get_axis_size(1);
        if (order == 3) {
            launch_max_pooling_kernel<T, 3>(stream, output, indices, input, channels,
                out_spatial_dims, in_spatial_dims, window_size, strides, padding_left);
        } else if (order == 2) {
            launch_max_pooling_kernel<T, 2>(stream, output, indices, input, channels,
                out_spatial_dims, in_spatial_dims, window_size, strides, padding_left);
        }
    }

    template void max_pooling_with_indices(const Stream&,
        TensorSpan<__half>, TensorSpan<__half>, TensorView<__half>,
        const std::vector<std::size_t>&, const std::vector<std::size_t>&,
        const std::vector<std::size_t>&);

    template void max_pooling_with_indices(const Stream&,
        TensorSpan<float>, TensorSpan<float>, TensorView<float>,
        const std::vector<std::size_t>&, const std::vector<std::size_t>&,
        const std::vector<std::size_t>&);

    template void max_pooling_with_indices(const Stream&,
        TensorSpan<double>, TensorSpan<double>, TensorView<double>,
        const std::vector<std::size_t>&, const std::vector<std::size_t>&,
        const std::vector<std::size_t>&);

    template <class T, std::size_t Order> static
    void launch_max_unpooling_kernel(
        const Stream& stream,
        span<T> output, view<T> input, view<T> indices, std::size_t channels,
        const std::vector<std::size_t>& out_spatial_dims, const std::vector<std::size_t>& in_spatial_dims,
        const std::vector<std::size_t>& window_size,
        const std::vector<std::size_t>& strides, const std::vector<std::size_t>& padding_left)
    {
        CV_Assert(out_spatial_dims.size() == Order);
        CV_Assert(in_spatial_dims.size() == Order);
        CV_Assert(window_size.size() == Order);
        CV_Assert(strides.size() == Order);
        CV_Assert(padding_left.size() == Order);
        CV_Assert(indices.size() == input.size());

        utils::array<size_type, Order> out_spatial_dims_k, in_spatial_dims_k;
        out_spatial_dims_k.assign(std::begin(out_spatial_dims), std::end(out_spatial_dims));
        in_spatial_dims_k.assign(std::begin(in_spatial_dims), std::end(in_spatial_dims));

        utils::array<size_type, Order> window_size_k, strides_k, padding_left_k;
        window_size_k.assign(std::begin(window_size), std::end(window_size));
        strides_k.assign(std::begin(strides), std::end(strides));
        padding_left_k.assign(std::begin(padding_left), std::end(padding_left));

        auto kernel = raw::max_unpooling<T, Order>;
        auto policy = make_policy(kernel, input.size(), 0, stream);
        launch_kernel(kernel, policy, output, input, indices, channels,
            out_spatial_dims_k, in_spatial_dims_k, window_size_k, strides_k, padding_left_k);
    }

    template <class T>
    void max_unpooling(
        const Stream& stream,
        TensorSpan<T> output, TensorView<T> input, TensorView<T> indices,
        const std::vector<std::size_t>& window_size, const std::vector<std::size_t>& strides,
        const std::vector<std::size_t>& padding_left)
    {
        auto order = window_size.size();
        CV_Assert(strides.size() == order);
        CV_Assert(padding_left.size() == order);
        CV_Assert(2 <= order && order <= 3);

        CV_Assert(is_shape_same(input, indices));
        CV_Assert(output.rank() == order + 2);
        CV_Assert(input.rank() == order + 2);

        CV_Assert(input.get_axis_size(1) == output.get_axis_size(1));

        std::vector<std::size_t> out_spatial_dims(order), in_spatial_dims(order);
        for (int i = 0; i < order; i++) {
            in_spatial_dims[i] = input.get_axis_size(2 + i);
            out_spatial_dims[i] = output.get_axis_size(2 + i);
        }

        std::size_t channels = input.get_axis_size(1);
        if (order == 3) {
            launch_max_unpooling_kernel<T, 3>(stream, output, input, indices, channels,
                out_spatial_dims, in_spatial_dims, window_size, strides, padding_left);
        } else if (order == 2) {
            launch_max_unpooling_kernel<T, 2>(stream, output, input, indices, channels,
                out_spatial_dims, in_spatial_dims, window_size, strides, padding_left);
        }
    }

    template void max_unpooling(const Stream&,
        TensorSpan<__half>, TensorView<__half>, TensorView<__half>,
        const std::vector<std::size_t>&, const std::vector<std::size_t>&,
        const std::vector<std::size_t>&);

    template void max_unpooling(const Stream&,
        TensorSpan<float>, TensorView<float>, TensorView<float>,
        const std::vector<std::size_t>&, const std::vector<std::size_t>&,
        const std::vector<std::size_t>&);

    template void max_unpooling(const Stream&,
        TensorSpan<double>, TensorView<double>, TensorView<double>,
        const std::vector<std::size_t>&, const std::vector<std::size_t>&,
        const std::vector<std::size_t>&);


}}}}} /*  cv::dnn::cuda4dnn::csl::kernels */
