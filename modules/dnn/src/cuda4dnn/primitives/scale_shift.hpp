// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SCALE_SHIFT_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SCALE_SHIFT_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/tensor.hpp"

#include "../kernels/scale_shift.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    struct ScaleShiftConfiguration {
        enum class OpMode {
            NONE,
            TRAINABLE, /* use a pretrained blob */
            UNTRAINABLE /* use another input */
        };

        OpMode scaleMode;
        OpMode shiftMode;

        std::size_t axis;
    };

    template <class T>
    class ScaleShiftOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        ScaleShiftOp(csl::Stream stream_, const ScaleShiftConfiguration& config, const cv::Mat& weights, const cv::Mat& bias)
            : stream(std::move(stream_)), axis{ config.axis }
        {
            scaleMode = config.scaleMode;
            if (scaleMode == ScaleShiftConfiguration::OpMode::TRAINABLE)
            {
                CV_Assert(!weights.empty());
                weightsTensor = csl::makeTensorHeader<T>(weights);
                csl::copyMatToTensor<T>(weights, weightsTensor, stream);
            }

            shiftMode = config.shiftMode;
            if (shiftMode == ScaleShiftConfiguration::OpMode::TRAINABLE)
            {
                CV_Assert(!bias.empty());
                biasTensor = csl::makeTensorHeader<T>(bias);
                csl::copyMatToTensor<T>(bias, biasTensor, stream);
            }

            CV_Assert(scaleMode != ScaleShiftConfiguration::OpMode::NONE ||
                      shiftMode != ScaleShiftConfiguration::OpMode::NONE);

            if (scaleMode == ScaleShiftConfiguration::OpMode::UNTRAINABLE &&
                shiftMode == ScaleShiftConfiguration::OpMode::UNTRAINABLE)
            {
                CV_Error(cv::Error::StsNotImplemented, "scale and shift both in untrainable mode is not supported");
            }
        }

        void forward(
            const std::vector<cv::Ptr<BackendWrapper>>& inputs,
            const std::vector<cv::Ptr<BackendWrapper>>& outputs,
            csl::Workspace& workspace) override
        {
            CV_Assert(outputs.size() == 1);

            auto input_wrapper = inputs[0].dynamicCast<wrapper_type>();
            auto input = input_wrapper->getView();

            auto output_wrapper = outputs[0].dynamicCast<wrapper_type>();
            auto output = output_wrapper->getSpan();

            /* number of batches in the weights/bias
             * trainable mode: same for all batches
             * untrainable mode: could be different for different batch samples
             */
            std::size_t parameter_batch_size = 1;

            csl::TensorView<T> weights;
            if (scaleMode == ScaleShiftConfiguration::OpMode::TRAINABLE)
            {
                CV_Assert(!weightsTensor.empty());
                weights = csl::TensorView<T>(weightsTensor);
            }
            else if (scaleMode == ScaleShiftConfiguration::OpMode::UNTRAINABLE)
            {
                CV_Assert(inputs.size() == 2);
                auto wrapper = inputs[1].dynamicCast<wrapper_type>();
                weights = wrapper->getView();

                parameter_batch_size = weights.get_axis_size(0);
                CV_Assert(parameter_batch_size == input.get_axis_size(0));
            }

            csl::TensorView<T> bias;
            if (shiftMode == ScaleShiftConfiguration::OpMode::TRAINABLE)
            {
                CV_Assert(!biasTensor.empty());
                bias = csl::TensorView<T>(biasTensor);
            }
            else if (shiftMode == ScaleShiftConfiguration::OpMode::UNTRAINABLE)
            {
                CV_Assert(inputs.size() == 2);
                auto wrapper = inputs[1].dynamicCast<wrapper_type>();
                bias = wrapper->getView();

                parameter_batch_size = bias.get_axis_size(0);
                CV_Assert(parameter_batch_size == input.get_axis_size(0));
            }

            CV_Assert(!weights.empty() || !bias.empty());
            if (!weights.empty() && !bias.empty())
            {
                CV_CheckEQ(weights.size(), bias.size(), "different broadcasting options for weights and bias is not supported");
            }

            const auto num_parameters = !weights.empty() ? weights.size() : bias.size();
            const auto mid_size = num_parameters / parameter_batch_size;

            /* the scale shift operation might require broadcasting */
            const int end_axis = [&] {
                if (num_parameters == 1) {
                    return static_cast<int>(axis + 1);
                }
                for (int endAxis = axis + 1; endAxis <= input.rank(); endAxis++) {
                    if (input.size_range(axis, endAxis) == mid_size)
                        return endAxis;
                }
                CV_Assert(0 /* failed to find a broadcast config */);
            }();

            std::size_t inner_size = input.size_range(end_axis, input.rank());

            if (!weights.empty() && !bias.empty())
                kernels::scaleN_with_biasN<T>(stream, output, input, inner_size, weights, bias);
            else if (!weights.empty())
                kernels::scaleN<T>(stream, output, input, inner_size, weights);
            else
                kernels::biasN<T>(stream, output, input, inner_size, bias);
        }

    private:
        csl::Stream stream;
        csl::Tensor<T> weightsTensor, biasTensor;
        std::size_t axis;

        ScaleShiftConfiguration::OpMode scaleMode, shiftMode;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SCALE_SHIFT_HPP */
