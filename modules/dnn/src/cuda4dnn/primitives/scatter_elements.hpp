// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SCATTER_ELEMENTS_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SCATTER_ELEMENTS_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"

#include "../kernels/scatter_elements.hpp"
#include "../kernels/fill_copy.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <utility>
#include <vector>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class ScatterElementsOp final : public CUDABackendNode {
    public:
        ScatterElementsOp(csl::Stream stream_, int axis_)
            : stream(std::move(stream_)), axis(axis_)
        {
        }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_UNUSED(workspace);
            CV_Assert(inputs.size() == 3 && outputs.size() == 1);

            auto data = csl::viewOf<T>(inputs[0]);
            auto updates = csl::viewOf<T>(inputs[2]);
            auto output = csl::spanOf<T>(outputs[0]);

            // ScatterElements writes a sparse subset, so the untouched elements come from data
            if (output.get() != data.get())
                kernels::copy<T>(stream, output, data);

            const MatShape dataShape = cv::dnn::shape(inputs[0]);
            const MatShape idxShape = cv::dnn::shape(inputs[1]);
            const int rank = dataShape.dims;
            CV_Assert(rank == idxShape.dims && rank >= 1);

            const int axis_ = normalize_axis(axis, rank);

            std::vector<std::int64_t> output_strides(rank);
            output_strides[rank - 1] = 1;
            for (int i = rank - 2; i >= 0; --i)
                output_strides[i] = output_strides[i + 1] * dataShape[i + 1];

            std::vector<std::int64_t> indices_dims(rank);
            for (int i = 0; i < rank; i++)
                indices_dims[i] = idxShape[i];

            const std::int64_t axis_dim_size = dataShape[axis_];
            const int idepth = inputs[1].depth();

            if (idepth == CV_32S) {
                kernels::scatter_elements<T, std::int32_t>(stream, output,
                    csl::viewOf<std::int32_t>(inputs[1]), updates,
                    rank, axis_, indices_dims, output_strides, axis_dim_size);
            } else if (idepth == CV_64S) {
                kernels::scatter_elements<T, std::int64_t>(stream, output,
                    csl::viewOf<std::int64_t>(inputs[1]), updates,
                    rank, axis_, indices_dims, output_strides, axis_dim_size);
            } else {
                CV_Error(Error::StsNotImplemented, "ScatterElements: indices must be CV_32S or CV_64S");
            }
        }

    private:
        csl::Stream stream;
        int axis;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SCATTER_ELEMENTS_HPP */
