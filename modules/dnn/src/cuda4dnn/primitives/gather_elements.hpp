// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_GATHER_ELEMENTS_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_GATHER_ELEMENTS_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"

#include "../kernels/gather_elements.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class GatherElementsOp final : public CUDABackendNode {
    public:
        GatherElementsOp(csl::Stream stream_, int axis_)
            : stream(std::move(stream_)), axis(axis_)
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

            MatShape dataShape = cv::dnn::shape(inputs[0]);
            MatShape indicesShape = cv::dnn::shape(inputs[1]);
            const int dataDims = dataShape.dims;
            const int axis_ = normalize_axis(axis, dataDims);

            std::int64_t inner_size = 1;
            for (int i = axis_ + 1; i < dataDims; i++)
                inner_size *= dataShape[i];

            const std::int64_t axis_size_data = dataShape[axis_];
            const std::int64_t axis_size_indices = indicesShape[axis_];

            csl::device::fast_divmod axis_inner_size(static_cast<int>(axis_size_indices * inner_size));
            csl::device::fast_divmod inner_size_divmod(static_cast<int>(inner_size));

            const std::size_t index_element_size = inputs[1].elemSize();
            const void* indices = reinterpret_cast<const uchar*>(inputs[1].u->handle) + inputs[1].offset;

            kernels::gather_elements<T>(stream, output, data, indices, index_element_size,
                                        axis_size_data, inner_size, axis_inner_size, inner_size_divmod);
        }

    private:
        csl::Stream stream;
        int axis;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_GATHER_ELEMENTS_HPP */
