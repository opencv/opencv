// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_PRIMITIVES_SCALE_SHIFT_HPP
#define OPENCV_DNN_CUDA4DNN_PRIMITIVES_SCALE_SHIFT_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/tensor.hpp"
#include "../csl/kernels.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class ScaleShiftOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        ScaleShiftOp(csl::Stream stream_, std::size_t axis, const cv::Mat& weights, const cv::Mat& bias)
            : stream(std::move(stream_)), axis{ axis }
        {
            if (!weights.empty())
            {
                weightsTensor = csl::makeTensorHeader<T>(weights);
                csl::copyMatToTensor<T>(weightsTensor, weights, stream);
            }

            if (!bias.empty())
            {
                biasTensor = csl::makeTensorHeader<T>(bias);
                csl::copyMatToTensor<T>(biasTensor, bias, stream);
            }
        }

        void forward(
            std::vector<cv::Ptr<BackendWrapper>>& inputs,
            std::vector<cv::Ptr<BackendWrapper>>& outputs,
            csl::Workspace& workspace) override
        {
            CV_Assert(outputs.size() == 1);

            auto input_wrapper = inputs[0].dynamicCast<wrapper_type>();
            auto input = input_wrapper->getView();

            auto output_wrapper = outputs[0].dynamicCast<wrapper_type>();
            auto output = output_wrapper->getSpan();

            csl::TensorView<T> weights;
            if (weightsTensor.empty() && biasTensor.empty())
            {
                CV_Assert(inputs.size() == 2);

                /* no explicit scale/shift values provided; use the second input as weights */
                auto wrapper = inputs[1].dynamicCast<wrapper_type>();
                weights = wrapper->getView();
            }
            else if (!weightsTensor.empty())
            {
                weights = csl::TensorSpan<T>(weightsTensor);
            }

            csl::TensorView<T> bias;
            if (!biasTensor.empty())
                bias = csl::TensorSpan<T>(biasTensor);

            const auto numParams = !weights.empty() ? weights.size() : bias.size();
            CV_Assert(numParams != 0);
            if (!weightsTensor.empty() && !biasTensor.empty())
            {
                CV_CheckEQ(weights.size(), bias.size(), "weights and bias size are not equal");
            }

            auto input_shape = input_wrapper->getShape();

            /* the weights/bias might require broadcasting to scale/shift */
            const int end_axis = [&] {
                for (int endAxis = axis + 1; endAxis <= input_shape.size(); ++endAxis)
                {
                    std::size_t size = 1;
                    for (int i = axis; i < endAxis; i++)
                        size *= input_shape[i];

                    if (size == numParams)
                        return endAxis;
                }
                CV_Assert(0 /* invalid weights matrix */);
            }();

            std::size_t inner_size = 1;
            for (int i = end_axis; i < input_shape.size(); i++)
                inner_size *= input_shape[i];

            if (!weights.empty() && !bias.empty())
                csl::kernels::scaleN_with_biasN<T>(stream, output, input, inner_size, weights, bias);
            else if (!weights.empty())
                csl::kernels::scaleN<T>(stream, output, input, inner_size, weights);
            else
                csl::kernels::biasN<T>(stream, output, input, inner_size, bias);
        }

    private:
        csl::Stream stream;
        csl::Tensor<T> weightsTensor, biasTensor;
        std::size_t axis;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_CUDA4DNN_PRIMITIVES_SCALE_SHIFT_HPP */
