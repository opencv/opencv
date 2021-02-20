// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_ACTIVATION_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_ACTIVATION_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/tensor.hpp"

#include "../kernels/activations.hpp"

#include <opencv2/core.hpp>

#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class ReLUOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        ReLUOp(csl::Stream stream_, T slope_)
            : stream(std::move(stream_)), slope{ slope_ } { }

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

                kernels::relu<T>(stream, output, input, slope);
            }
        }

    private:
        csl::Stream stream;
        const T slope;
    };

    template <class T>
    class ClippedReLUOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        ClippedReLUOp(csl::Stream stream_, T min_, T max_)
            : stream(std::move(stream_)), min{ min_ }, max{ max_ } { }

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

                kernels::clipped_relu<T>(stream, output, input, min, max);
            }
        }

    private:
        csl::Stream stream;
        const T min, max;
    };

    template <class T>
    class ChannelwiseReLUOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        ChannelwiseReLUOp(csl::Stream stream_, const Mat& slope)
            : stream(std::move(stream_))
        {
            CV_Assert(!slope.empty());
            slopeTensor = csl::makeTensorHeader<T>(slope);
            csl::copyMatToTensor<T>(slope, slopeTensor, stream);
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

                CV_Assert(input.get_axis_size(1) == slopeTensor.size());
                std::size_t inner_size = input.size_range(2, input.rank());
                kernels::axiswise_relu<T>(stream, output, input, inner_size, slopeTensor);
            }
        }

    private:
        csl::Stream stream;
        csl::Tensor<T> slopeTensor;
    };

    template <class T>
    class TanHOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        TanHOp(csl::Stream stream_) : stream(std::move(stream_)) { }

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

                kernels::tanh<T>(stream, output, input);
            }
        }

    private:
        csl::Stream stream;
    };

    template <class T>
    class SwishOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        SwishOp(csl::Stream stream_) : stream(std::move(stream_)) { }

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

                kernels::swish<T>(stream, output, input);
            }
        }

    private:
        csl::Stream stream;
    };

    template <class T>
    class MishOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        MishOp(csl::Stream stream_) : stream(std::move(stream_)) { }

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

                kernels::mish<T>(stream, output, input);
            }
        }

    private:
        csl::Stream stream;
    };

    template <class T>
    class SigmoidOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        SigmoidOp(csl::Stream stream_) : stream(std::move(stream_)) { }

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

                kernels::sigmoid<T>(stream, output, input);
            }
        }

    private:
        csl::Stream stream;
    };

    template <class T>
    class ELUOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        ELUOp(csl::Stream stream_) : stream(std::move(stream_)) { }

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

                kernels::elu<T>(stream, output, input);
            }
        }

    private:
        csl::Stream stream;
    };

    template <class T>
    class AbsValOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        AbsValOp(csl::Stream stream_) : stream(std::move(stream_)) { }

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

                kernels::abs<T>(stream, output, input);
            }
        }

    private:
        csl::Stream stream;
    };

    template <class T>
    class BNLLOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        BNLLOp(csl::Stream stream_) : stream(std::move(stream_)) { }

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

                kernels::bnll<T>(stream, output, input);
            }
        }

    private:
        csl::Stream stream;
    };

    template <class T>
    class PowerOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        PowerOp(csl::Stream stream_, T exp_, T scale_, T shift_)
            : stream(std::move(stream_)), exp{ exp_ }, scale{ scale_ }, shift{ shift_ } { }

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

                kernels::power<T>(stream, output, input, exp, scale, shift);
            }
        }

    private:
        csl::Stream stream;
        const T exp, scale, shift;
    };

    template <class T>
    class ExpOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        ExpOp(csl::Stream stream_, T nScale_, T nShift_)
            : stream(std::move(stream_)), normScale{ nScale_ }, normShift{ nShift_ } { }

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

                kernels::exp<T>(stream, output, input, normScale, normShift);
            }
        }

    private:
        csl::Stream stream;
        const T normScale, normShift;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_ACTIVATION_HPP */
