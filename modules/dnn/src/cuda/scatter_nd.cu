// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "array.hpp"
#include "types.hpp"
#include "grid_stride_range.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"
#include "../cuda4dnn/csl/tensor.hpp"

#include "../cuda4dnn/kernels/scatter_nd.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {
        template <class T, class TIdx>
        __global__ void scatter_nd(
            Span<T> output, View<TIdx> indices, View<T> updates,
            size_type num_indices, std::int64_t last_index_dimension,
            array<std::int64_t, CSL_MAX_TENSOR_RANK> element_counts,
            array<std::int64_t, CSL_MAX_TENSOR_RANK> input_dims,
            std::int64_t num_updates_elements)
        {
            for (auto id : grid_stride_range(num_indices)) {
                std::int64_t data_offset = 0;

                std::int64_t indices_start = last_index_dimension * id;
                for (std::int64_t j = 0; j < last_index_dimension; ++j) {
                    std::int64_t index = static_cast<std::int64_t>(indices.data().get()[indices_start + j]);

                    std::int64_t element_count_dim = element_counts[j];
                    std::int64_t dim_value = input_dims[j];

                    if (index >= 0) {
                        if (index >= dim_value) {
                            index = dim_value - 1;
                        }
                    } else {
                        if (index < -dim_value) {
                            index = 0;
                        } else {
                            index += dim_value;
                        }
                    }

                    data_offset += (index * element_count_dim);
                }

                const T* updates_data_base = updates.data().get() + num_updates_elements * static_cast<std::int64_t>(id);
                T* output_data_base = output.data().get() + data_offset;

                for (std::int64_t i = 0; i < num_updates_elements; ++i) {
                    output_data_base[i] = updates_data_base[i];
                }
            }
        }
    }

    template <class T, class TIdx>
    void scatter_nd(const Stream& stream,
        Span<T> output, View<TIdx> indices, View<T> updates,
        std::size_t num_indices, std::int64_t last_index_dimension,
        const std::vector<std::int64_t>& element_counts,
        const std::vector<std::int64_t>& input_dims,
        std::size_t num_updates_elements)
    {
        CV_Assert(last_index_dimension <= CSL_MAX_TENSOR_RANK);
        CV_Assert(element_counts.size() == static_cast<std::size_t>(last_index_dimension));
        CV_Assert(input_dims.size() == static_cast<std::size_t>(last_index_dimension));

        if (num_indices == 0)
            return;

        array<std::int64_t, CSL_MAX_TENSOR_RANK> element_counts_k, input_dims_k;
        element_counts_k.assign(std::begin(element_counts), std::end(element_counts));
        input_dims_k.assign(std::begin(input_dims), std::end(input_dims));

        auto kernel = raw::scatter_nd<T, TIdx>;
        auto policy = make_policy(kernel, num_indices, 0, stream);
        launch_kernel(kernel, policy, output, indices, updates,
                      static_cast<size_type>(num_indices), last_index_dimension,
                      element_counts_k, input_dims_k,
                      static_cast<std::int64_t>(num_updates_elements));
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void scatter_nd(const Stream&, Span<__half>, View<std::int32_t>, View<__half>, std::size_t, std::int64_t, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, std::size_t);
    template void scatter_nd(const Stream&, Span<__half>, View<std::int64_t>, View<__half>, std::size_t, std::int64_t, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, std::size_t);
#endif
    template void scatter_nd(const Stream&, Span<float>, View<std::int32_t>, View<float>, std::size_t, std::int64_t, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, std::size_t);
    template void scatter_nd(const Stream&, Span<float>, View<std::int64_t>, View<float>, std::size_t, std::int64_t, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, std::size_t);
    template void scatter_nd(const Stream&, Span<int8_t>, View<std::int32_t>, View<int8_t>, std::size_t, std::int64_t, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, std::size_t);
    template void scatter_nd(const Stream&, Span<int8_t>, View<std::int64_t>, View<int8_t>, std::size_t, std::int64_t, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, std::size_t);
    template void scatter_nd(const Stream&, Span<uint8_t>, View<std::int32_t>, View<uint8_t>, std::size_t, std::int64_t, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, std::size_t);
    template void scatter_nd(const Stream&, Span<uint8_t>, View<std::int64_t>, View<uint8_t>, std::size_t, std::int64_t, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, std::size_t);
    template void scatter_nd(const Stream&, Span<int32_t>, View<std::int32_t>, View<int32_t>, std::size_t, std::int64_t, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, std::size_t);
    template void scatter_nd(const Stream&, Span<int32_t>, View<std::int64_t>, View<int32_t>, std::size_t, std::int64_t, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, std::size_t);
    template void scatter_nd(const Stream&, Span<int64_t>, View<std::int32_t>, View<int64_t>, std::size_t, std::int64_t, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, std::size_t);
    template void scatter_nd(const Stream&, Span<int64_t>, View<std::int64_t>, View<int64_t>, std::size_t, std::int64_t, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, std::size_t);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
