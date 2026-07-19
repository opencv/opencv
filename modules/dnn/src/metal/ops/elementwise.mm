// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"

#ifdef HAVE_METAL

#include "elementwise.hpp"
#include "operation.hpp"
#include "../runtime/context.hpp"
#include "../runtime/device.hpp"
#include "../runtime/tensor.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

namespace {

class MetalUnaryImpl final : public metal::Operation
{
public:
    MetalUnaryImpl(const Ptr<MetalBackendWrapper>& input,
                   const Ptr<MetalBackendWrapper>& output,
                   const char* kernelName,
                   float value0,
                   float value1,
                   float value2)
        : input_(input), output_(output), kernelName_(kernelName)
    {
        CV_CheckTrue(input_ != nullptr, "Metal unary input must not be null");
        CV_CheckTrue(output_ != nullptr, "Metal unary output must not be null");
        CV_CheckTrue(kernelName != nullptr && kernelName[0] != '\0',
                     "Metal unary kernel name must not be empty");
        CV_CheckTypeEQ(input_->tensor()->type(), CV_32F, "Metal input must be CV_32F");
        CV_CheckTypeEQ(output_->tensor()->type(), CV_32F, "Metal output must be CV_32F");
        CV_CheckEQ(input_->tensor()->total(), output_->tensor()->total(),
                   "Metal unary input and output sizes must match");
        parameters_.count = static_cast<uint32_t>(output_->tensor()->total());
        parameters_.value0 = value0;
        parameters_.value1 = value1;
        parameters_.value2 = value2;
    }

    void forward(metal::Context& context) CV_OVERRIDE
    {
        if (!pipeline_)
            pipeline_ = metal::Device::instance()->pipelineState(kernelName_);

        const auto& inputBuffer = input_->tensor()->bufferForRead();
        const auto& outputBuffer = output_->tensor()->bufferForWrite();
        id<MTLComputeCommandEncoder> encoder = context.computeEncoder();
        [encoder setComputePipelineState:pipeline_];
        [encoder setBuffer:context.use(inputBuffer) offset:0 atIndex:0];
        [encoder setBuffer:context.use(outputBuffer) offset:0 atIndex:1];
        [encoder setBytes:&parameters_ length:sizeof(parameters_) atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(parameters_.count, 1, 1)
            threadsPerThreadgroup:context.threadsPerThreadgroup1D(
                pipeline_, parameters_.count)];
        context.didDispatch();
    }

private:
    struct Parameters
    {
        uint32_t count;
        float value0;
        float value1;
        float value2;
    } parameters_;
    Ptr<MetalBackendWrapper> input_;
    Ptr<MetalBackendWrapper> output_;
    std::string kernelName_;
    id<MTLComputePipelineState> pipeline_ = nil;
};

static std::unique_ptr<metal::Operation> makeUnaryOperation(
        const std::vector<Ptr<BackendWrapper> >& inputs,
        const std::vector<Ptr<BackendWrapper> >& outputs,
        const char* kernelName,
        float value0 = 0.0f,
        float value1 = 0.0f,
        float value2 = 0.0f)
{
    return std::unique_ptr<metal::Operation>(new MetalUnaryImpl(
        inputs[0].dynamicCast<MetalBackendWrapper>(),
        outputs[0].dynamicCast<MetalBackendWrapper>(), kernelName,
        value0, value1, value2));
}

class MetalPReLUImpl final : public metal::Operation
{
public:
    MetalPReLUImpl(const Ptr<MetalBackendWrapper>& input,
                   const Ptr<MetalBackendWrapper>& output,
                   const Mat& slope,
                   bool channelWise)
        : input_(input), output_(output), channelWise_(channelWise)
    {
        CV_CheckTrue(input_ != nullptr, "Metal PReLU input must not be null");
        CV_CheckTrue(output_ != nullptr, "Metal PReLU output must not be null");
        CV_CheckTypeEQ(input_->tensor()->type(), CV_32F, "Metal input must be CV_32F");
        CV_CheckTypeEQ(output_->tensor()->type(), CV_32F, "Metal output must be CV_32F");
        CV_CheckTypeEQ(slope.type(), CV_32F, "Metal PReLU slope must be CV_32F");
        CV_CheckTrue(slope.isContinuous(), "Metal PReLU slope must be continuous");
        CV_CheckGT(slope.total(), static_cast<size_t>(0),
                   "Metal PReLU slope must not be empty");
        CV_CheckEQ(input_->tensor()->total(), output_->tensor()->total(),
                   "Metal PReLU input and output sizes must match");

        const MatShape& inputShape = input_->tensor()->shape();
        CV_CheckFalse(inputShape.empty(), "Metal PReLU input shape must not be empty");
        parameters_.count = static_cast<uint32_t>(output_->tensor()->total());
        parameters_.slopeCount = static_cast<uint32_t>(slope.total());
        if (channelWise_)
        {
            const size_t channels = inputShape.size() == 1
                ? static_cast<size_t>(inputShape[0])
                : static_cast<size_t>(inputShape[1]);
            CV_CheckEQ(slope.total(), channels,
                       "Metal ChannelsPReLU slope count must match input channels");
            size_t innerSize = 1;
            for (size_t i = 2; i < inputShape.size(); ++i)
                innerSize *= static_cast<size_t>(inputShape[i]);
            parameters_.innerSize = static_cast<uint32_t>(innerSize);
        }
        else
        {
            CV_CheckTrue(parameters_.count % parameters_.slopeCount == 0,
                         "Metal PReLU slope must tile the input tensor");
            parameters_.innerSize = 0;
        }

        slopeHost_ = slope.reshape(1, 1).clone();
        slopeTensor_ = metal::Tensor::create(slopeHost_);
    }

    void forward(metal::Context& context) CV_OVERRIDE
    {
        if (!pipeline_)
        {
            pipeline_ = metal::Device::instance()->pipelineState(
                channelWise_ ? "kernel_channels_prelu_f32" : "kernel_prelu_f32");
        }

        const auto& inputBuffer = input_->tensor()->bufferForRead();
        const auto& slopeBuffer = slopeTensor_->bufferForRead();
        const auto& outputBuffer = output_->tensor()->bufferForWrite();
        id<MTLComputeCommandEncoder> encoder = context.computeEncoder();
        [encoder setComputePipelineState:pipeline_];
        [encoder setBuffer:context.use(inputBuffer) offset:0 atIndex:0];
        [encoder setBuffer:context.use(outputBuffer) offset:0 atIndex:1];
        [encoder setBuffer:context.use(slopeBuffer) offset:0 atIndex:2];
        [encoder setBytes:&parameters_ length:sizeof(parameters_) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(parameters_.count, 1, 1)
            threadsPerThreadgroup:context.threadsPerThreadgroup1D(
                pipeline_, parameters_.count)];
        context.didDispatch();
    }

private:
    struct Parameters
    {
        uint32_t count;
        uint32_t slopeCount;
        uint32_t innerSize;
    } parameters_;
    Ptr<MetalBackendWrapper> input_;
    Ptr<MetalBackendWrapper> output_;
    bool channelWise_;
    Mat slopeHost_;
    std::shared_ptr<metal::Tensor> slopeTensor_;
    id<MTLComputePipelineState> pipeline_ = nil;
};

static std::unique_ptr<metal::Operation> makePReLUOperation(
        const std::vector<Ptr<BackendWrapper> >& inputs,
        const std::vector<Ptr<BackendWrapper> >& outputs,
        const Mat& slope,
        bool channelWise)
{
    return std::unique_ptr<metal::Operation>(new MetalPReLUImpl(
        inputs[0].dynamicCast<MetalBackendWrapper>(),
        outputs[0].dynamicCast<MetalBackendWrapper>(), slope, channelWise));
}


}  // namespace

CV__DNN_INLINE_NS_END

namespace metal {

Ptr<BackendNode> GeluOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                const std::vector<Ptr<BackendWrapper> >& outputs)
{
    return makeMetalBackendNode(makeUnaryOperation(inputs, outputs, "kernel_gelu_f32"));
}

Ptr<BackendNode> GeluApproximationOp::create(
    const std::vector<Ptr<BackendWrapper> >& inputs,
    const std::vector<Ptr<BackendWrapper> >& outputs)
{
    return makeMetalBackendNode(
        makeUnaryOperation(inputs, outputs, "kernel_gelu_approximation_f32"));
}

Ptr<BackendNode> ReLUOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                const std::vector<Ptr<BackendWrapper> >& outputs,
                                float negativeSlope)
{
    return makeMetalBackendNode(
        makeUnaryOperation(inputs, outputs, "kernel_relu_f32", negativeSlope));
}

Ptr<BackendNode> ReLU6Op::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                 const std::vector<Ptr<BackendWrapper> >& outputs,
                                 float minValue, float maxValue)
{
    return makeMetalBackendNode(
        makeUnaryOperation(inputs, outputs, "kernel_relu6_f32", minValue, maxValue));
}

Ptr<BackendNode> SigmoidOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                   const std::vector<Ptr<BackendWrapper> >& outputs)
{
    return makeMetalBackendNode(makeUnaryOperation(inputs, outputs, "kernel_sigmoid_f32"));
}

Ptr<BackendNode> SwishOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                 const std::vector<Ptr<BackendWrapper> >& outputs)
{
    return makeMetalBackendNode(makeUnaryOperation(inputs, outputs, "kernel_swish_f32"));
}

Ptr<BackendNode> TanHOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                const std::vector<Ptr<BackendWrapper> >& outputs)
{
    return makeMetalBackendNode(makeUnaryOperation(inputs, outputs, "kernel_tanh_f32"));
}

Ptr<BackendNode> MishOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                const std::vector<Ptr<BackendWrapper> >& outputs)
{
    return makeMetalBackendNode(makeUnaryOperation(inputs, outputs, "kernel_mish_f32"));
}

Ptr<BackendNode> ELUOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                               const std::vector<Ptr<BackendWrapper> >& outputs,
                               float alpha)
{
    return makeMetalBackendNode(
        makeUnaryOperation(inputs, outputs, "kernel_elu_f32", alpha));
}

Ptr<BackendNode> AbsValOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                  const std::vector<Ptr<BackendWrapper> >& outputs)
{
    return makeMetalBackendNode(makeUnaryOperation(inputs, outputs, "kernel_abs_f32"));
}

Ptr<BackendNode> BNLLOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                const std::vector<Ptr<BackendWrapper> >& outputs)
{
    return makeMetalBackendNode(makeUnaryOperation(inputs, outputs, "kernel_bnll_f32"));
}

Ptr<BackendNode> CeilOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                const std::vector<Ptr<BackendWrapper> >& outputs)
{
    return makeMetalBackendNode(makeUnaryOperation(inputs, outputs, "kernel_ceil_f32"));
}

Ptr<BackendNode> FloorOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                 const std::vector<Ptr<BackendWrapper> >& outputs)
{
    return makeMetalBackendNode(makeUnaryOperation(inputs, outputs, "kernel_floor_f32"));
}

Ptr<BackendNode> LogOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                               const std::vector<Ptr<BackendWrapper> >& outputs)
{
    return makeMetalBackendNode(makeUnaryOperation(inputs, outputs, "kernel_log_f32"));
}

Ptr<BackendNode> RoundOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                 const std::vector<Ptr<BackendWrapper> >& outputs)
{
    return makeMetalBackendNode(makeUnaryOperation(inputs, outputs, "kernel_round_f32"));
}

Ptr<BackendNode> SqrtOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                const std::vector<Ptr<BackendWrapper> >& outputs)
{
    return makeMetalBackendNode(makeUnaryOperation(inputs, outputs, "kernel_sqrt_f32"));
}

Ptr<BackendNode> AcosOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                const std::vector<Ptr<BackendWrapper> >& outputs)
{
    return makeMetalBackendNode(makeUnaryOperation(inputs, outputs, "kernel_acos_f32"));
}

Ptr<BackendNode> AcoshOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                 const std::vector<Ptr<BackendWrapper> >& outputs)
{
    return makeMetalBackendNode(makeUnaryOperation(inputs, outputs, "kernel_acosh_f32"));
}

Ptr<BackendNode> AsinOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                const std::vector<Ptr<BackendWrapper> >& outputs)
{
    return makeMetalBackendNode(makeUnaryOperation(inputs, outputs, "kernel_asin_f32"));
}

Ptr<BackendNode> AsinhOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                 const std::vector<Ptr<BackendWrapper> >& outputs)
{
    return makeMetalBackendNode(makeUnaryOperation(inputs, outputs, "kernel_asinh_f32"));
}

Ptr<BackendNode> AtanOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                const std::vector<Ptr<BackendWrapper> >& outputs)
{
    return makeMetalBackendNode(makeUnaryOperation(inputs, outputs, "kernel_atan_f32"));
}

Ptr<BackendNode> AtanhOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                 const std::vector<Ptr<BackendWrapper> >& outputs)
{
    return makeMetalBackendNode(makeUnaryOperation(inputs, outputs, "kernel_atanh_f32"));
}

Ptr<BackendNode> CosOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                               const std::vector<Ptr<BackendWrapper> >& outputs)
{
    return makeMetalBackendNode(makeUnaryOperation(inputs, outputs, "kernel_cos_f32"));
}

Ptr<BackendNode> CoshOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                const std::vector<Ptr<BackendWrapper> >& outputs)
{
    return makeMetalBackendNode(makeUnaryOperation(inputs, outputs, "kernel_cosh_f32"));
}

Ptr<BackendNode> ErfOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                               const std::vector<Ptr<BackendWrapper> >& outputs)
{
    return makeMetalBackendNode(makeUnaryOperation(inputs, outputs, "kernel_erf_f32"));
}

Ptr<BackendNode> HardSwishOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                     const std::vector<Ptr<BackendWrapper> >& outputs)
{
    return makeMetalBackendNode(
        makeUnaryOperation(inputs, outputs, "kernel_hard_swish_f32"));
}

Ptr<BackendNode> SinOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                               const std::vector<Ptr<BackendWrapper> >& outputs)
{
    return makeMetalBackendNode(makeUnaryOperation(inputs, outputs, "kernel_sin_f32"));
}

Ptr<BackendNode> SinhOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                const std::vector<Ptr<BackendWrapper> >& outputs)
{
    return makeMetalBackendNode(makeUnaryOperation(inputs, outputs, "kernel_sinh_f32"));
}

Ptr<BackendNode> SoftplusOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                    const std::vector<Ptr<BackendWrapper> >& outputs)
{
    return makeMetalBackendNode(
        makeUnaryOperation(inputs, outputs, "kernel_softplus_f32"));
}

Ptr<BackendNode> SoftsignOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                    const std::vector<Ptr<BackendWrapper> >& outputs)
{
    return makeMetalBackendNode(
        makeUnaryOperation(inputs, outputs, "kernel_softsign_f32"));
}

Ptr<BackendNode> TanOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                               const std::vector<Ptr<BackendWrapper> >& outputs)
{
    return makeMetalBackendNode(makeUnaryOperation(inputs, outputs, "kernel_tan_f32"));
}

Ptr<BackendNode> CeluOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                const std::vector<Ptr<BackendWrapper> >& outputs,
                                float alpha)
{
    return makeMetalBackendNode(
        makeUnaryOperation(inputs, outputs, "kernel_celu_f32", alpha));
}

Ptr<BackendNode> HardSigmoidOp::create(
    const std::vector<Ptr<BackendWrapper> >& inputs,
    const std::vector<Ptr<BackendWrapper> >& outputs,
    float alpha,
    float beta)
{
    return makeMetalBackendNode(
        makeUnaryOperation(inputs, outputs, "kernel_hard_sigmoid_f32", alpha, beta));
}

Ptr<BackendNode> SeluOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                const std::vector<Ptr<BackendWrapper> >& outputs,
                                float alpha,
                                float gamma)
{
    return makeMetalBackendNode(
        makeUnaryOperation(inputs, outputs, "kernel_selu_f32", alpha, gamma));
}

Ptr<BackendNode> ThresholdedReluOp::create(
    const std::vector<Ptr<BackendWrapper> >& inputs,
    const std::vector<Ptr<BackendWrapper> >& outputs,
    float alpha)
{
    return makeMetalBackendNode(
        makeUnaryOperation(inputs, outputs, "kernel_thresholded_relu_f32", alpha));
}

Ptr<BackendNode> PowerOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                 const std::vector<Ptr<BackendWrapper> >& outputs,
                                 float power,
                                 float scale,
                                 float shift)
{
    return makeMetalBackendNode(
        makeUnaryOperation(inputs, outputs, "kernel_power_f32", power, scale, shift));
}

Ptr<BackendNode> ExpOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                               const std::vector<Ptr<BackendWrapper> >& outputs,
                               float scale,
                               float shift)
{
    return makeMetalBackendNode(
        makeUnaryOperation(inputs, outputs, "kernel_exp_f32", scale, shift));
}

Ptr<BackendNode> SignOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                const std::vector<Ptr<BackendWrapper> >& outputs)
{
    return makeMetalBackendNode(makeUnaryOperation(inputs, outputs, "kernel_sign_f32"));
}

Ptr<BackendNode> ShrinkOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                  const std::vector<Ptr<BackendWrapper> >& outputs,
                                  float bias,
                                  float lambd)
{
    return makeMetalBackendNode(
        makeUnaryOperation(inputs, outputs, "kernel_shrink_f32", bias, lambd));
}

Ptr<BackendNode> ReciprocalOp::create(
    const std::vector<Ptr<BackendWrapper> >& inputs,
    const std::vector<Ptr<BackendWrapper> >& outputs)
{
    return makeMetalBackendNode(
        makeUnaryOperation(inputs, outputs, "kernel_reciprocal_f32"));
}

Ptr<BackendNode> ChannelsPReLUOp::create(
    const std::vector<Ptr<BackendWrapper> >& inputs,
    const std::vector<Ptr<BackendWrapper> >& outputs,
    const Mat& slope)
{
    return makeMetalBackendNode(makePReLUOperation(inputs, outputs, slope, true));
}

Ptr<BackendNode> PReLUOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                 const std::vector<Ptr<BackendWrapper> >& outputs,
                                 const Mat& slope)
{
    return makeMetalBackendNode(makePReLUOperation(inputs, outputs, slope, false));
}

}  // namespace metal
}}  // namespace cv::dnn

#endif  // HAVE_METAL
