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

#include "../cuda4dnn/kernels/scatter_elements.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {
        template <class T, class TIdx>
        __global__ void scatter_elements(
            Span<T> output, View<TIdx> indices, View<T> updates,
            int rank, int axis,
            array<std::int64_t, CSL_MAX_TENSOR_RANK> indices_dims,
            array<std::int64_t, CSL_MAX_TENSOR_RANK> output_strides,
            std::int64_t axis_dim_size)
        {
            for (auto id : grid_stride_range(updates.size())) {
                std::int64_t rem = static_cast<std::int64_t>(id);
                std::int64_t offset = 0;

                for (int d = rank - 1; d >= 0; --d) {
                    const std::int64_t dim = indices_dims[d];
                    const std::int64_t coord = rem % dim;
                    rem /= dim;

                    std::int64_t use = coord;
                    if (d == axis) {
                        std::int64_t idx = static_cast<std::int64_t>(indices.data().get()[id]);
                        if (idx < 0)
                            idx += axis_dim_size;
                        if (idx < 0)
                            idx = 0;
                        if (idx >= axis_dim_size)
                            idx = axis_dim_size - 1;
                        use = idx;
                    }
                    offset += use * output_strides[d];
                }

                output[offset] = updates.data().get()[id];
            }
        }
    }

    template <class T, class TIdx>
    void scatter_elements(const Stream& stream,
        Span<T> output, View<TIdx> indices, View<T> updates,
        int rank, int axis,
        const std::vector<std::int64_t>& indices_dims,
        const std::vector<std::int64_t>& output_strides,
        std::int64_t axis_dim_size)
    {
        CV_Assert(rank <= CSL_MAX_TENSOR_RANK);
        CV_Assert(static_cast<int>(indices_dims.size()) == rank);
        CV_Assert(static_cast<int>(output_strides.size()) == rank);

        if (updates.size() == 0)
            return;

        array<std::int64_t, CSL_MAX_TENSOR_RANK> dims_k, strides_k;
        dims_k.assign(std::begin(indices_dims), std::end(indices_dims));
        strides_k.assign(std::begin(output_strides), std::end(output_strides));

        auto kernel = raw::scatter_elements<T, TIdx>;
        auto policy = make_policy(kernel, updates.size(), 0, stream);
        launch_kernel(kernel, policy, output, indices, updates, rank, axis,
                      dims_k, strides_k, axis_dim_size);
    }

#define CV_SCATTER_ELEM_INST(T) \
    template void scatter_elements(const Stream&, Span<T>, View<std::int32_t>, View<T>, int, int, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, std::int64_t); \
    template void scatter_elements(const Stream&, Span<T>, View<std::int64_t>, View<T>, int, int, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, std::int64_t);

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    CV_SCATTER_ELEM_INST(__half)
#endif
    CV_SCATTER_ELEM_INST(float)
    CV_SCATTER_ELEM_INST(int8_t)
    CV_SCATTER_ELEM_INST(uint8_t)
    CV_SCATTER_ELEM_INST(int32_t)
    CV_SCATTER_ELEM_INST(int64_t)

#undef CV_SCATTER_ELEM_INST

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
