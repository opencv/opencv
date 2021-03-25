// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_REORG_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_REORG_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../kernels/permute.hpp"

#include <opencv2/core.hpp>

#include <vector>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class ReorgOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        ReorgOp(csl::Stream stream_, std::size_t stride_)
            : stream(std::move(stream_)), stride{ stride_ } { }

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

            const std::size_t permute_input_shape[] = {
               input.get_axis_size(0),
               input.get_axis_size(1) * input.get_axis_size(2) / (stride * stride),
               stride,
               input.get_axis_size(3),
               stride
            };

            constexpr std::size_t order[] = { 0, 2, 4, 1, 3 };

            const std::size_t permute_output_shape[] = {
                permute_input_shape[order[0]],
                permute_input_shape[order[1]],
                permute_input_shape[order[2]],
                permute_input_shape[order[3]],
                permute_input_shape[order[4]]
            };

            input.unsqueeze();
            input.reshape(std::begin(permute_input_shape), std::end(permute_input_shape));

            output.unsqueeze();
            output.reshape(std::begin(permute_output_shape), std::end(permute_output_shape));

            kernels::permute(stream, output, input, { std::begin(order), std::end(order) });
        }

    private:
        csl::Stream stream;
        std::size_t stride;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_REORG_HPP */
