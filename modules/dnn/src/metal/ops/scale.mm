// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"

#ifdef HAVE_METAL

#include "scale.hpp"
#include "operation.hpp"
#include "../runtime/context.hpp"
#include "../runtime/device.hpp"
#include "../runtime/tensor.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

class MetalAffineImpl final : public metal::Operation
{
public:
    MetalAffineImpl(
        const Ptr<MetalBackendWrapper>& input,
        const Ptr<MetalBackendWrapper>& output,
        const Ptr<MetalBackendWrapper>& dynamicParameter,
        const Mat& scale,
        const Mat& bias,
        int axis,
        metal::AffineConfiguration::DynamicParameter dynamicParameterMode)
        : input_(input), output_(output), dynamicParameter_(dynamicParameter),
          dynamicParameterMode_(dynamicParameterMode)
    {
        const MatShape& inputShape = input_->tensor()->shape();
        CV_CheckFalse(inputShape.empty(), "Metal affine input shape must not be empty");
        CV_CheckTypeEQ(input_->tensor()->type(), CV_32F, "Metal input must be CV_32F");
        CV_CheckTypeEQ(output_->tensor()->type(), CV_32F, "Metal output must be CV_32F");
        const uint32_t elementCount = static_cast<uint32_t>(output_->tensor()->total());
        if (axis < 0)
            axis += static_cast<int>(inputShape.size());
        CV_CheckGE(axis, 0, "Metal affine axis is out of range");
        CV_CheckLT(axis, static_cast<int>(inputShape.size()),
                   "Metal affine axis is out of range");

        const bool hasDynamicParameter = !dynamicParameter_.empty();
        if (hasDynamicParameter)
        {
            CV_CheckTypeEQ(dynamicParameter_->tensor()->type(), CV_32F,
                           "Metal affine dynamic parameters must be CV_32F");
        }

        const size_t parameterCount = hasDynamicParameter
            ? dynamicParameter_->tensor()->total()
            : (!scale.empty() ? scale.total() : bias.total());
        CV_CheckGT(parameterCount, static_cast<size_t>(0),
                   "Metal affine requires scale or bias parameters");
        if (!scale.empty() && !bias.empty())
            CV_CheckEQ(scale.total(), bias.total(), "Metal affine parameter sizes must match");

        size_t covered = 1;
        int endAxis = axis;
        for (; endAxis < static_cast<int>(inputShape.size()) && covered < parameterCount; ++endAxis)
            covered *= static_cast<size_t>(inputShape[endAxis]);
        CV_CheckEQ(covered, parameterCount,
                   "Metal affine parameters do not match the input shape");

        size_t innerSize = 1;
        for (int i = endAxis; i < static_cast<int>(inputShape.size()); ++i)
            innerSize *= static_cast<size_t>(inputShape[i]);

        if (dynamicParameterMode_ != metal::AffineConfiguration::DynamicParameter::SCALE)
        {
            scaleHost_.create(1, static_cast<int>(parameterCount), CV_32F);
            if (scale.empty())
                scaleHost_.setTo(Scalar(1));
            else
                scale.reshape(1, 1).copyTo(scaleHost_);
            scaleTensor_ = metal::Tensor::create(scaleHost_);
        }
        if (dynamicParameterMode_ != metal::AffineConfiguration::DynamicParameter::BIAS)
        {
            biasHost_.create(1, static_cast<int>(parameterCount), CV_32F);
            if (bias.empty())
                biasHost_.setTo(Scalar(0));
            else
                bias.reshape(1, 1).copyTo(biasHost_);
            biasTensor_ = metal::Tensor::create(biasHost_);
        }

        parameters_.count = elementCount;
        parameters_.parameterCount = static_cast<uint32_t>(parameterCount);
        parameters_.innerSize = static_cast<uint32_t>(innerSize);
    }

    void forward(metal::Context& context) CV_OVERRIDE
    {
        const bool useContiguousPipeline = parameters_.innerSize >= 4 && parameters_.innerSize % 4 == 0;
        const std::shared_ptr<metal::Tensor>& inputTensor = input_->tensor();
        const std::shared_ptr<metal::Tensor>& outputTensor = output_->tensor();
        const std::shared_ptr<metal::Tensor> dynamicParameterTensor =
            dynamicParameter_.empty() ? std::shared_ptr<metal::Tensor>()
                                      : dynamicParameter_->tensor();
        const std::shared_ptr<metal::Tensor>& scaleTensor =
            dynamicParameterMode_ == metal::AffineConfiguration::DynamicParameter::SCALE
                ? dynamicParameterTensor : scaleTensor_;
        const std::shared_ptr<metal::Tensor>& biasTensor =
            dynamicParameterMode_ == metal::AffineConfiguration::DynamicParameter::BIAS
                ? dynamicParameterTensor : biasTensor_;
        const auto& inputBuffer = inputTensor->bufferForRead();
        const auto& scaleBuffer = scaleTensor->bufferForRead();
        const auto& biasBuffer = biasTensor->bufferForRead();
        const auto& outputBuffer = outputTensor->bufferForWrite();
        id<MTLComputeCommandEncoder> encoder = context.computeEncoder();
        [encoder setBuffer:context.use(inputBuffer) offset:0 atIndex:0];
        [encoder setBuffer:context.use(scaleBuffer) offset:0 atIndex:1];
        [encoder setBuffer:context.use(biasBuffer) offset:0 atIndex:2];
        [encoder setBuffer:context.use(outputBuffer) offset:0 atIndex:3];
        [encoder setBytes:&parameters_ length:sizeof(parameters_) atIndex:4];
        if (useContiguousPipeline)
        {
            if (!contiguousPipeline_)
            {
                contiguousPipeline_ = metal::Device::instance()->pipelineState("kernel_affine_contiguous_f32");
            }

            const size_t executionWidth =
                static_cast<size_t>([contiguousPipeline_ threadExecutionWidth]);
            const size_t pipelineLimit =
                static_cast<size_t>([contiguousPipeline_ maxTotalThreadsPerThreadgroup]);
            const size_t preferredLimit =
                std::min(std::max(static_cast<size_t>(256), executionWidth), pipelineLimit);
            const size_t maximumThreads = preferredLimit - preferredLimit % executionWidth;
            const size_t requiredThreads =
                (static_cast<size_t>(parameters_.innerSize) / 4 + executionWidth - 1) /
                executionWidth * executionWidth;
            const size_t threadsPerThreadgroup = std::min(requiredThreads, maximumThreads);
            const size_t threadgroupCount = parameters_.count / parameters_.innerSize;
            [encoder setComputePipelineState:contiguousPipeline_];
            [encoder dispatchThreadgroups:MTLSizeMake(threadgroupCount, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadsPerThreadgroup, 1, 1)];
        }
        else
        {
            if (!pipeline_)
            {
                pipeline_ = metal::Device::instance()->pipelineState("kernel_affine_f32");
            }

            [encoder setComputePipelineState:pipeline_];
            [encoder dispatchThreads:MTLSizeMake(parameters_.count, 1, 1)
                threadsPerThreadgroup:context.threadsPerThreadgroup1D(
                    pipeline_, parameters_.count)];
        }
        context.didDispatch();
    }

private:
    struct Parameters
    {
        uint32_t count;
        uint32_t parameterCount;
        uint32_t innerSize;
    } parameters_;
    Ptr<MetalBackendWrapper> input_;
    Ptr<MetalBackendWrapper> output_;
    Ptr<MetalBackendWrapper> dynamicParameter_;
    metal::AffineConfiguration::DynamicParameter dynamicParameterMode_;
    id<MTLComputePipelineState> pipeline_ = nil;
    id<MTLComputePipelineState> contiguousPipeline_ = nil;
    Mat scaleHost_;
    Mat biasHost_;
    std::shared_ptr<metal::Tensor> scaleTensor_;
    std::shared_ptr<metal::Tensor> biasTensor_;
};

static std::unique_ptr<metal::Operation> makeAffineOperation(
        const std::vector<Ptr<BackendWrapper> >& inputs,
        const std::vector<Ptr<BackendWrapper> >& outputs,
        const metal::AffineConfiguration& config)
{
    const bool hasDynamicParameter =
        config.dynamicParameter != metal::AffineConfiguration::DynamicParameter::NONE;
    Ptr<MetalBackendWrapper> dynamicParameter;
    if (hasDynamicParameter)
        dynamicParameter = inputs[1].dynamicCast<MetalBackendWrapper>();
    return std::unique_ptr<metal::Operation>(new MetalAffineImpl(
        inputs[0].dynamicCast<MetalBackendWrapper>(),
        outputs[0].dynamicCast<MetalBackendWrapper>(),
        dynamicParameter, config.scale, config.bias, config.axis, config.dynamicParameter));
}

CV__DNN_INLINE_NS_END

namespace metal {

Ptr<BackendNode> AffineOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                  const std::vector<Ptr<BackendWrapper> >& outputs,
                                  const AffineConfiguration& config)
{
    return makeMetalBackendNode(makeAffineOperation(inputs, outputs, config));
}

}  // namespace metal
}}  // namespace cv::dnn

#endif  // HAVE_METAL
