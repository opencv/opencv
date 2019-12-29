// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SOFTMAX_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SOFTMAX_HPP

#include "../../op_cuda.hpp"

#include "../csl/cudnn.hpp"
#include "../csl/tensor_ops.hpp"

#include <cstddef>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class SoftmaxOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        SoftmaxOp(csl::cudnn::Handle handle, std::size_t axis_, bool log_)
            : cudnnHandle(std::move(handle)), channel_axis{ axis_ }, log{ log_ }
        {
        }

        void forward(
            const std::vector<cv::Ptr<BackendWrapper>>& inputs,
            const std::vector<cv::Ptr<BackendWrapper>>& outputs,
            csl::Workspace& workspace) override
        {
            for (int i = 0; i < inputs.size(); i++)
            {
                auto input_wrapper = inputs[i].dynamicCast<wrapper_type>();
                auto input = input_wrapper->getView();

                auto output_wrapper = outputs[i].dynamicCast<wrapper_type>();
                auto output = output_wrapper->getSpan();

                csl::tensor_ops::softmax<T>(cudnnHandle, output, input, channel_axis, log);
            }
        }

    private:
        csl::cudnn::Handle cudnnHandle;
        std::size_t channel_axis;
        bool log;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SOFTMAX_HPP */
