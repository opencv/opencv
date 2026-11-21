// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_GATHER_ND_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_GATHER_ND_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"

#include "../kernels/gather_nd.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class GatherNDOp final : public CUDABackendNode {
    public:
        GatherNDOp(csl::Stream stream_, int batch_dims_)
            : stream(std::move(stream_)), batch_dims(batch_dims_)
        {
        }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_UNUSED(workspace);
            CV_Assert(inputs.size() == 2 && outputs.size() == 1);

            auto data = csl::viewOf<T>(inputs[0]);
            auto output = csl::spanOf<T>(outputs[0]);

            const MatShape dataShape = cv::dnn::shape(inputs[0]);
            const MatShape indicesShape = cv::dnn::shape(inputs[1]);
            const int r = dataShape.dims;
            const int q = indicesShape.dims;
            CV_Assert(r >= 1 && q >= 1);

            const std::int64_t last_index_dimension = indicesShape[q - 1];
            CV_Assert(last_index_dimension >= 1 && batch_dims + last_index_dimension <= r);

            std::vector<std::int64_t> data_strides(r);
            data_strides[r - 1] = 1;
            for (int i = r - 2; i >= 0; --i)
                data_strides[i] = data_strides[i + 1] * dataShape[i + 1];

            std::vector<std::int64_t> element_counts(last_index_dimension);
            std::vector<std::int64_t> input_dims(last_index_dimension);
            for (std::int64_t j = 0; j < last_index_dimension; ++j) {
                element_counts[j] = data_strides[batch_dims + j];
                input_dims[j] = dataShape[batch_dims + static_cast<int>(j)];
            }

            std::int64_t indices_total = 1;
            for (int i = 0; i < q; i++)
                indices_total *= indicesShape[i];
            const std::int64_t outer_size = indices_total / last_index_dimension;

            // an empty indices or output tensor is a legal no-op, not an error
            if (outer_size <= 0 || output.size() == 0)
                return;

            const std::int64_t inner_size = static_cast<std::int64_t>(output.size()) / outer_size;

            std::int64_t num_slices_per_batch = 1;
            for (int d = batch_dims; d < q - 1; ++d)
                num_slices_per_batch *= indicesShape[d];
            if (num_slices_per_batch <= 0)
                num_slices_per_batch = 1;

            const std::int64_t batch_stride = batch_dims > 0 ? data_strides[batch_dims - 1] : 0;

            const int idepth = inputs[1].depth();
            if (idepth == CV_32S) {
                kernels::gather_nd<T, std::int32_t>(stream, output, csl::viewOf<std::int32_t>(inputs[1]),
                    data, last_index_dimension, element_counts, input_dims,
                    inner_size, batch_stride, num_slices_per_batch);
            } else if (idepth == CV_64S) {
                kernels::gather_nd<T, std::int64_t>(stream, output, csl::viewOf<std::int64_t>(inputs[1]),
                    data, last_index_dimension, element_counts, input_dims,
                    inner_size, batch_stride, num_slices_per_batch);
            } else {
                CV_Error(Error::StsNotImplemented, "GatherND: indices must be CV_32S or CV_64S");
            }
        }

    private:
        csl::Stream stream;
        int batch_dims;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_GATHER_ND_HPP */
