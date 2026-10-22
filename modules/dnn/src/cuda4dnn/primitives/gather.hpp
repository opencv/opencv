// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_GATHER_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_GATHER_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/pointer.hpp"
#include "../csl/workspace.hpp"

#include "../kernels/gather.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class GatherOp final : public CUDABackendNode {
    public:
        GatherOp(csl::Stream stream_, int axis_, int* error_flag_)
            : stream(std::move(stream_)), axis(axis_), error_flag(error_flag_)
        {
            CV_Assert(error_flag);
        }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_Assert(inputs.size() == 2 && outputs.size() == 1);
            CV_CheckType(inputs[1].depth(), inputs[1].depth() == CV_32S || inputs[1].depth() == CV_64S,
                         "Gather: indices must be int32 or int64");

            auto data = csl::viewOf<T>(inputs[0]);
            auto output = csl::spanOf<T>(outputs[0]);

            MatShape dataShape = cv::dnn::shape(inputs[0]);
            const int dataDims = dataShape.dims;
            const int axis_ = normalize_axis(axis, dataDims);

            std::int64_t block_size = 1;
            for (int i = axis_ + 1; i < dataDims; i++)
                block_size *= dataShape[i];
            const std::int64_t indices_max = dataShape[axis_];
            const std::int64_t input_block_size = block_size * indices_max;
            const std::size_t num_indices = inputs[1].total();
            const std::int64_t output_block_size = block_size * static_cast<std::int64_t>(num_indices);

            CV_Assert(output_block_size <= std::numeric_limits<int>::max());
            csl::device::fast_divmod divmod_output_block_size(static_cast<int>(output_block_size));
            csl::device::fast_divmod divmod_block_size(static_cast<int>(block_size));

            const std::size_t index_element_size = inputs[1].elemSize();
            const void* indices = reinterpret_cast<const uchar*>(inputs[1].u->handle) + inputs[1].offset;

            // the net checks the flag after the forward, so no per-call sync is needed
            kernels::gather<T>(stream, output, data, indices, index_element_size,
                               input_block_size, indices_max, divmod_output_block_size, divmod_block_size,
                               error_flag);
        }

    private:
        csl::Stream stream;
        int axis;
        int* error_flag;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_GATHER_HPP */
