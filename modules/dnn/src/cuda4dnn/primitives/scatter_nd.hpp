// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SCATTER_ND_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SCATTER_ND_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"

#include "../kernels/scatter_nd.hpp"
#include "../kernels/fill_copy.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T, class TIdx>
    class ScatterNDOp final : public CUDABackendNode {
    public:
        ScatterNDOp(csl::Stream stream_)
            : stream(std::move(stream_))
        {
        }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_UNUSED(workspace);
            CV_Assert(inputs.size() == 3 && outputs.size() == 1);

            if (inputs[0].total() == 0)
                return;

            auto data = csl::viewOf<T>(inputs[0]);
            auto output = csl::spanOf<T>(outputs[0]);

            kernels::copy<T>(stream, output, data);

            if (inputs[1].total() == 0 || inputs[2].total() == 0)
                return;

            auto indices = csl::viewOf<TIdx>(inputs[1]);
            auto updates = csl::viewOf<T>(inputs[2]);

            MatShape dataShape = cv::dnn::shape(inputs[0]);
            MatShape indShape = cv::dnn::shape(inputs[1]);
            const int r = dataShape.dims;
            const int q = indShape.dims;
            const std::int64_t last_index_dimension = indShape[q - 1];

            std::vector<std::int64_t> element_counts(last_index_dimension), input_dims(last_index_dimension);
            for (std::int64_t j = 0; j < last_index_dimension; j++) {
                std::int64_t stride = 1;
                for (int i = static_cast<int>(j) + 1; i < r; i++)
                    stride *= dataShape[i];
                element_counts[j] = stride;
                input_dims[j] = dataShape[static_cast<int>(j)];
            }

            std::size_t num_indices = 1;
            for (int i = 0; i < q - 1; i++)
                num_indices *= static_cast<std::size_t>(indShape[i]);

            std::size_t num_updates_elements = 1;
            for (int i = static_cast<int>(last_index_dimension); i < r; i++)
                num_updates_elements *= static_cast<std::size_t>(dataShape[i]);

            kernels::scatter_nd<T, TIdx>(stream, output, indices, updates,
                                   num_indices, last_index_dimension, element_counts, input_dims, num_updates_elements);
        }

    private:
        csl::Stream stream;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SCATTER_ND_HPP */
