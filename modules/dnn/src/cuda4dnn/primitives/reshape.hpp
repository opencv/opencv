// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_RESHAPE_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_RESHAPE_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/tensor.hpp"
#include "../csl/tensor_ops.hpp"

#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class ReshapeOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        ReshapeOp(csl::Stream stream_) : stream(std::move(stream_)) { }

        void forward(
            const std::vector<cv::Ptr<BackendWrapper>>& inputs,
            const std::vector<cv::Ptr<BackendWrapper>>& outputs,
            csl::Workspace& workspace) override
        {
            /* sometimes the output shape is passed as extra inputs; hence, >= instead of == */
            CV_Assert(inputs.size() >= outputs.size());

            for (int i = 0; i < outputs.size(); i++)
            {
                auto input_wrapper = inputs[i].dynamicCast<wrapper_type>();
                auto input = input_wrapper->getView();

                auto output_wrapper = outputs[i].dynamicCast<wrapper_type>();
                auto output = output_wrapper->getSpan();

                if (input.get() != output.get())
                {
                    while (input.rank() < output.rank())
                        input.unsqueeze();

                    while (output.rank() < input.rank())
                        output.unsqueeze();

                    input.reshape_as(output);
                    csl::tensor_ops::copy(stream, output, input);
                }
            }
        }

    private:
        csl::Stream stream;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_RESHAPE_HPP */
