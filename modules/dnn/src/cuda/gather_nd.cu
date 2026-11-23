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

#include "../cuda4dnn/kernels/gather_nd.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {
        template <class T, class TIdx>
        __global__ void gather_nd(
            Span<T> output, View<TIdx> indices, View<T> input,
            std::int64_t last_index_dimension,
            array<std::int64_t, CSL_MAX_TENSOR_RANK> element_counts,
            array<std::int64_t, CSL_MAX_TENSOR_RANK> input_dims,
            std::int64_t inner_size,
            std::int64_t batch_stride,
            std::int64_t num_slices_per_batch)
        {
            for (auto id : grid_stride_range(output.size())) {
                const std::int64_t slice = static_cast<std::int64_t>(id) / inner_size;
                const std::int64_t inner = static_cast<std::int64_t>(id) - slice * inner_size;

                std::int64_t data_offset = 0;
                const std::int64_t indices_start = last_index_dimension * slice;
                for (std::int64_t j = 0; j < last_index_dimension; ++j) {
                    std::int64_t index = static_cast<std::int64_t>(indices.data().get()[indices_start + j]);

                    const std::int64_t dim_value = input_dims[j];
                    if (index >= 0) {
                        if (index >= dim_value)
                            index = dim_value - 1;
                    } else {
                        if (index < -dim_value)
                            index = 0;
                        else
                            index += dim_value;
                    }

                    data_offset += index * element_counts[j];
                }

                if (batch_stride > 0)
                    data_offset += (slice / num_slices_per_batch) * batch_stride;

                output[id] = input.data().get()[data_offset + inner];
            }
        }
    }

    template <class T, class TIdx>
    void gather_nd(const Stream& stream,
        Span<T> output, View<TIdx> indices, View<T> input,
        std::int64_t last_index_dimension,
        const std::vector<std::int64_t>& element_counts,
        const std::vector<std::int64_t>& input_dims,
        std::int64_t inner_size,
        std::int64_t batch_stride,
        std::int64_t num_slices_per_batch)
    {
        CV_Assert(last_index_dimension <= CSL_MAX_TENSOR_RANK);
        CV_Assert(element_counts.size() == static_cast<std::size_t>(last_index_dimension));
        CV_Assert(input_dims.size() == static_cast<std::size_t>(last_index_dimension));
        CV_Assert(inner_size > 0);
        CV_Assert(num_slices_per_batch > 0);

        if (output.size() == 0)
            return;

        array<std::int64_t, CSL_MAX_TENSOR_RANK> element_counts_k, input_dims_k;
        element_counts_k.assign(std::begin(element_counts), std::end(element_counts));
        input_dims_k.assign(std::begin(input_dims), std::end(input_dims));

        auto kernel = raw::gather_nd<T, TIdx>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, indices, input,
                      last_index_dimension, element_counts_k, input_dims_k,
                      inner_size, batch_stride, num_slices_per_batch);
    }

#define CV_GATHER_ND_INST(T) \
    template void gather_nd(const Stream&, Span<T>, View<std::int32_t>, View<T>, std::int64_t, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, std::int64_t, std::int64_t, std::int64_t); \
    template void gather_nd(const Stream&, Span<T>, View<std::int64_t>, View<T>, std::int64_t, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, std::int64_t, std::int64_t, std::int64_t);

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    CV_GATHER_ND_INST(__half)
#endif
    CV_GATHER_ND_INST(float)
    CV_GATHER_ND_INST(int8_t)
    CV_GATHER_ND_INST(uint8_t)
    CV_GATHER_ND_INST(int32_t)
    CV_GATHER_ND_INST(int64_t)

#undef CV_GATHER_ND_INST

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
