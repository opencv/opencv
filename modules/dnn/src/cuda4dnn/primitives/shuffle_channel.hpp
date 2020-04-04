// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SHUFFLE_CHANNEL_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SHUFFLE_CHANNEL_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/tensor_ops.hpp"

#include "../kernels/permute.hpp"

#include <opencv2/core.hpp>

#include <vector>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class ShuffleChannelOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        ShuffleChannelOp(csl::Stream stream_, std::size_t group_)
            : stream(std::move(stream_)), group{ group_ } { }

        void forward(
            const std::vector<cv::Ptr<BackendWrapper>>& inputs,
            const std::vector<cv::Ptr<BackendWrapper>>& outputs,
            csl::Workspace& workspace) override
        {
            CV_Assert(inputs.size() == 1 && outputs.size() == 1);

            auto input_wrapper = inputs[0].dynamicCast<wrapper_type>();
            auto input = input_wrapper->getView();

            auto output_wrapper = outputs[0].dynamicCast<wrapper_type>();
            auto output = output_wrapper->getSpan();

            if (group == 1) {
                /* permute is redundant; check else branch to know why */
                if (input.get() != output.get()) {
                    input.reshape_as(output);
                    csl::tensor_ops::copy(stream, output, input);
                }
            } else {
                const std::size_t permute_input_shape[] = {
                   input.get_axis_size(0),
                   group,
                   input.get_axis_size(1) / group,
                   input.get_axis_size(2) * input.get_axis_size(3)
                };

                constexpr std::size_t order[] = { 0, 2, 1, 3 };

                const std::size_t permute_output_shape[] = {
                    permute_input_shape[order[0]],
                    permute_input_shape[order[1]],
                    permute_input_shape[order[2]],
                    permute_input_shape[order[3]],
                };

                input.reshape(std::begin(permute_input_shape), std::end(permute_input_shape));
                output.reshape(std::begin(permute_output_shape), std::end(permute_output_shape));
                kernels::permute(stream, output, input, { std::begin(order), std::end(order) });
            }
        }

    private:
        csl::Stream stream;
        std::size_t group;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SHUFFLE_CHANNEL_HPP */
