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

    template <template<class> class Op, class T>
    struct BaseOp : public CUDABackendNode
    {
    protected:
        using wrapper_type = GetCUDABackendWrapperType<T>;

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

                static_cast<const Op<T>*>(this)->calculate(output, input);
            }
        }
    };

    template <class T>
    class ReLUOp final : public BaseOp<ReLUOp, T> {
    public:
        ReLUOp(csl::Stream stream_, T slope_)
                : stream(std::move(stream_)), slope{ slope_ } { }

        void calculate(csl::TensorSpan<T> output, csl::TensorView<T> input) const
        {
            kernels::relu<T>(stream, output, input, slope);
        }

    private:
        csl::Stream stream;
        const T slope;
    };

    template <class T>
    class ClippedReLUOp final : public BaseOp<ClippedReLUOp, T> {
    public:
        ClippedReLUOp(csl::Stream stream_, T min_, T max_)
            : stream(std::move(stream_)), min{ min_ }, max{ max_ } { }

        void calculate(csl::TensorSpan<T> output, csl::TensorView<T> input) const
        {
            kernels::clipped_relu<T>(stream, output, input, min, max);
        }

    private:
        csl::Stream stream;
        const T min, max;
    };

    template <class T>
    class ChannelwiseReLUOp final : public BaseOp<ChannelwiseReLUOp, T> {
    public:
        ChannelwiseReLUOp(csl::Stream stream_, const Mat& slope)
                : stream(std::move(stream_))
        {
            CV_Assert(!slope.empty());
            slopeTensor = csl::makeTensorHeader<T>(slope);
            csl::copyMatToTensor<T>(slope, slopeTensor, stream);
        }

        void calculate(csl::TensorSpan<T> output, csl::TensorView<T> input) const
        {
            CV_Assert(input.get_axis_size(1) == slopeTensor.size());
            std::size_t inner_size = input.size_range(2, input.rank());
            kernels::axiswise_relu<T>(stream, output, input, inner_size, slopeTensor);
        }

    private:
        csl::Stream stream;
        csl::Tensor<T> slopeTensor;
    };

    template <class T>
    class TanHOp final : public BaseOp<TanHOp, T> {
    public:
        TanHOp(csl::Stream stream_) : stream(std::move(stream_)) { }

        void calculate(csl::TensorSpan<T> output, csl::TensorView<T> input) const
        {
            kernels::tanh<T>(stream, output, input);
        }

    private:
        csl::Stream stream;
    };

    template <class T>
    class SwishOp final : public BaseOp<SwishOp, T> {
    public:
        SwishOp(csl::Stream stream_) : stream(std::move(stream_)) { }

        void calculate(csl::TensorSpan<T> output, csl::TensorView<T> input) const
        {
            kernels::swish<T>(stream, output, input);
        }

    private:
        csl::Stream stream;
    };

    template <class T>
    class MishOp final : public BaseOp<MishOp, T> {
    public:
        MishOp(csl::Stream stream_) : stream(std::move(stream_)) { }

        void calculate(csl::TensorSpan<T> output, csl::TensorView<T> input) const
        {
            kernels::mish<T>(stream, output, input);
        }

    private:
        csl::Stream stream;
    };

    template <class T>
    class SigmoidOp final : public BaseOp<SigmoidOp, T> {
    public:
        SigmoidOp(csl::Stream stream_) : stream(std::move(stream_)) { }

        void calculate(csl::TensorSpan<T> output, csl::TensorView<T> input) const
        {
            kernels::sigmoid<T>(stream, output, input);
        }

    private:
        csl::Stream stream;
    };

    template <class T>
    class ELUOp final : public BaseOp<ELUOp, T> {
    public:
        ELUOp(csl::Stream stream_) : stream(std::move(stream_)) { }

        void calculate(csl::TensorSpan<T> output, csl::TensorView<T> input) const
        {
            kernels::elu<T>(stream, output, input);
        }

    private:
        csl::Stream stream;
    };

    template <class T>
    class AbsValOp final : public BaseOp<AbsValOp, T> {
    public:
        AbsValOp(csl::Stream stream_) : stream(std::move(stream_)) { }

        void calculate(csl::TensorSpan<T> output, csl::TensorView<T> input) const
        {
            kernels::abs<T>(stream, output, input);
        }

    private:
        csl::Stream stream;
    };

    template <class T>
    class BNLLOp final : public BaseOp<BNLLOp, T> {
    public:
        BNLLOp(csl::Stream stream_) : stream(std::move(stream_)) { }

        void calculate(csl::TensorSpan<T> output, csl::TensorView<T> input) const
        {
            kernels::bnll<T>(stream, output, input);
        }

    private:
        csl::Stream stream;
    };

    template <class T>
    class CeilOp final : public BaseOp<CeilOp, T> {
    public:
        CeilOp(csl::Stream stream_) : stream(std::move(stream_)) { }

        void calculate(csl::TensorSpan<T> output, csl::TensorView<T> input) const
        {
            kernels::ceil<T>(stream, output, input);
        }

    private:
        csl::Stream stream;
    };

    template <class T>
    class FloorOp final : public BaseOp<FloorOp, T> {
    public:
        FloorOp(csl::Stream stream_) : stream(std::move(stream_)) { }

        void calculate(csl::TensorSpan<T> output, csl::TensorView<T> input) const
        {
            kernels::floor<T>(stream, output, input);
        }

    private:
        csl::Stream stream;
    };

    template <class T>
    class LogOp final : public BaseOp<LogOp, T> {
    public:
        LogOp(csl::Stream stream_) : stream(std::move(stream_)) { }

        void calculate(csl::TensorSpan<T> output, csl::TensorView<T> input) const
        {
            kernels::log<T>(stream, output, input);
        }

    private:
        csl::Stream stream;
    };

    template <class T>
    class RoundOp final : public BaseOp<RoundOp, T> {
    public:
        RoundOp(csl::Stream stream_) : stream(std::move(stream_)) { }

        void calculate(csl::TensorSpan<T> output, csl::TensorView<T> input) const
        {
            kernels::rint<T>(stream, output, input);
        }

    private:
        csl::Stream stream;
    };

    template <class T>
    class SqrtOp final : public BaseOp<SqrtOp, T> {
    public:
        SqrtOp(csl::Stream stream_) : stream(std::move(stream_)) { }

        void calculate(csl::TensorSpan<T> output, csl::TensorView<T> input) const
        {
            kernels::sqrt<T>(stream, output, input);
        }

    private:
        csl::Stream stream;
    };

    template <class T>
    class NotOp final : public BaseOp<NotOp, T> {
    public:
        NotOp(csl::Stream stream_) : stream(std::move(stream_)) { }

        void calculate(csl::TensorSpan<T> output, csl::TensorView<T> input) const
        {
            kernels::not_k<T>(stream, output, input);
        }

    private:
        csl::Stream stream;
    };

    template <class T>
    class PowerOp final : public BaseOp<PowerOp, T> {
    public:
        PowerOp(csl::Stream stream_, T exp_, T scale_, T shift_)
            : stream(std::move(stream_)), exp{ exp_ }, scale{ scale_ }, shift{ shift_ } { }

        void calculate(csl::TensorSpan<T> output, csl::TensorView<T> input) const
        {
            kernels::power<T>(stream, output, input, exp, scale, shift);
        }

    private:
        csl::Stream stream;
        const T exp, scale, shift;
    };

    template <class T>
    class ExpOp final : public BaseOp<ExpOp, T> {
    public:
        ExpOp(csl::Stream stream_, T nScale_, T nShift_)
            : stream(std::move(stream_)), normScale{ nScale_ }, normShift{ nShift_ } { }

        void calculate(csl::TensorSpan<T> output, csl::TensorView<T> input) const
        {
            kernels::exp<T>(stream, output, input, normScale, normShift);
        }

    private:
        csl::Stream stream;
        const T normScale, normShift;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_ACTIVATION_HPP */
