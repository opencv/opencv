// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_PERMUTE_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_PERMUTE_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/tensor_ops.hpp"

#include "../kernels/permute.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class PermuteOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        PermuteOp(csl::Stream stream_, std::vector<std::size_t> order_)
            : stream(std::move(stream_)), order(std::move(order_)) { }

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

                auto needsPermute = [&] {
                    for (int i = 0; i < order.size(); i++)
                        if (order[i] != i)
                            return true;
                    return false;
                }();

                if (needsPermute)
                {
                    kernels::permute(stream, output, input, order);
                }
                else
                {
                    if (input.get() != output.get())
                        csl::tensor_ops::copy(stream, output, input);
                }
            }
        }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_UNUSED(workspace);
            CV_Assert(inputs.size() == outputs.size());

            bool needsPermute = false;
            for (std::size_t i = 0; i < order.size(); i++)
                needsPermute = needsPermute || order[i] != i;

            for (std::size_t i = 0; i < inputs.size(); i++)
            {
                auto input = csl::viewOf<T>(inputs[i]);
                auto output = csl::spanOf<T>(outputs[i]);

                if (needsPermute)
                    kernels::permute(stream, output, input, order);
                else if (input.get() != output.get())
                    csl::tensor_ops::copy(stream, output, input);
            }
        }

    private:
        csl::Stream stream;
        std::vector<std::size_t> order;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_PERMUTE_HPP */
