// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CONCAT_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CONCAT_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/pointer.hpp"

#include "../kernels/fill.hpp"
#include "../kernels/concat.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class ConcatOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        ConcatOp(csl::Stream stream_, std::size_t concat_axis, bool zero_padding)
            : stream(std::move(stream_)), concat_axis{ concat_axis }, zero_padding{ zero_padding }
        {
        }

        void forward(
            const std::vector<cv::Ptr<BackendWrapper>>& inputs,
            const std::vector<cv::Ptr<BackendWrapper>>& outputs,
            csl::Workspace& workspace) override
        {
            CV_Assert(outputs.size() == 1);

            auto output_wrapper = outputs[0].dynamicCast<wrapper_type>();
            auto output = output_wrapper->getSpan();

            if(zero_padding)
            {
                auto output_shape = output_wrapper->getShape();

                kernels::fill<T>(stream, output, 0.0);

                std::size_t output_concat_axis_offset = 0;
                for (int i = 0; i < inputs.size(); i++)
                {
                    auto input_wrapper = inputs[i].dynamicCast<wrapper_type>();
                    auto input = input_wrapper->getView();
                    auto input_shape = input_wrapper->getShape();

                    std::vector<std::size_t> offsets(input_shape.size());
                    for (int j = 0; j < offsets.size(); j++)
                        offsets[j] = (output_shape[j] - input_shape[j]) / 2;
                    offsets[concat_axis] = output_concat_axis_offset;

                    kernels::concat_with_offsets(stream, output, input, offsets);

                    output_concat_axis_offset += input.get_axis_size(concat_axis);
                }
            }
            else
            {
                std::size_t output_axis_offset = 0;
                for (int i = 0; i < inputs.size(); i++)
                {
                    auto input_wrapper = inputs[i].dynamicCast<wrapper_type>();
                    auto input = input_wrapper->getView();

                    kernels::concat(stream, output, output_axis_offset, input, concat_axis);

                    output_axis_offset += input.get_axis_size(concat_axis);
                }
            }
        }

    private:
        csl::Stream stream;
        std::size_t concat_axis;
        bool zero_padding;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CONCAT_HPP */
