// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"

#ifdef HAVE_METAL

#include "deconvolution.hpp"
#include "operation.hpp"
#include "../runtime/context.hpp"
#include "../runtime/device.hpp"
#include "../runtime/tensor.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

namespace {

class MetalDeconvolutionImpl final : public metal::Operation
{
public:
    MetalDeconvolutionImpl(
            const Ptr<MetalBackendWrapper>& input,
            const Ptr<MetalBackendWrapper>& output,
            const Ptr<MetalBackendWrapper>& dynamicWeights,
            const Ptr<MetalBackendWrapper>& dynamicBias,
            const Mat& weights,
            const Mat& bias,
            size_t groups,
            size_t kernelHeight,
            size_t kernelWidth,
            size_t strideHeight,
            size_t strideWidth,
            size_t dilationHeight,
            size_t dilationWidth,
            size_t padTop,
            size_t padLeft,
            metal::DeconvolutionConfiguration::PaddingMode paddingMode)
        : input_(input), output_(output), dynamicWeights_(dynamicWeights),
          dynamicBias_(dynamicBias)
    {
        CV_CheckTrue(!input_.empty() && !output_.empty(),
                     "Metal Deconv2D requires Metal tensor wrappers");
        const MatShape& inputShape = input_->tensor()->shape();
        const MatShape& outputShape = output_->tensor()->shape();
        CV_CheckEQ(inputShape.size(), static_cast<size_t>(4),
                   "Metal Deconv2D expects 4D NCHW input");
        CV_CheckEQ(outputShape.size(), static_cast<size_t>(4),
                   "Metal Deconv2D expects 4D NCHW output");
        CV_CheckTypeEQ(input_->tensor()->type(), CV_32F, "Metal tensor must be CV_32F");
        CV_CheckTypeEQ(output_->tensor()->type(), CV_32F, "Metal tensor must be CV_32F");

        const bool hasDynamicWeights = !dynamicWeights_.empty();
        CV_CheckNE(hasDynamicWeights, !weights.empty(),
                   "Metal Deconv2D requires exactly one static or dynamic weight source");
        MatShape weightShape;
        if (hasDynamicWeights)
        {
            CV_CheckTypeEQ(dynamicWeights_->tensor()->type(), CV_32F,
                           "Metal tensor must be CV_32F");
            weightShape = dynamicWeights_->tensor()->shape();
        }
        else
        {
            CV_CheckTypeEQ(weights.type(), CV_32F,
                           "Metal Deconv2D weights must be CV_32F");
            CV_CheckTrue(weights.isContinuous(),
                         "Metal Deconv2D weights must be continuous");
            weightShape = cv::dnn::shape(weights);
        }
        CV_CheckEQ(weightShape.size(), static_cast<size_t>(4),
                   "Metal Deconv2D weights must use IOHW layout");
        CV_CheckGT(groups, static_cast<size_t>(0),
                   "Metal Deconv2D groups must be positive");
        CV_CheckEQ(static_cast<size_t>(inputShape[1]) % groups, static_cast<size_t>(0),
                   "Metal Deconv2D input channels must be divisible by groups");
        CV_CheckEQ(static_cast<size_t>(outputShape[1]) % groups, static_cast<size_t>(0),
                   "Metal Deconv2D output channels must be divisible by groups");
        CV_CheckEQ(weightShape[0], inputShape[1],
                   "Metal Deconv2D input channels mismatch");
        CV_CheckEQ(weightShape[1], outputShape[1] / static_cast<int>(groups),
                   "Metal Deconv2D output channels per group mismatch");
        CV_CheckEQ(weightShape[2], static_cast<int>(kernelHeight),
                   "Metal Deconv2D kernel height mismatch");
        CV_CheckEQ(weightShape[3], static_cast<int>(kernelWidth),
                   "Metal Deconv2D kernel width mismatch");
        CV_CheckGT(strideHeight, static_cast<size_t>(0),
                   "Metal Deconv2D stride height must be positive");
        CV_CheckGT(strideWidth, static_cast<size_t>(0),
                   "Metal Deconv2D stride width must be positive");
        CV_CheckGT(dilationHeight, static_cast<size_t>(0),
                   "Metal Deconv2D dilation height must be positive");
        CV_CheckGT(dilationWidth, static_cast<size_t>(0),
                   "Metal Deconv2D dilation width must be positive");

        const size_t outputChannels = static_cast<size_t>(outputShape[1]);
        if (!dynamicBias_.empty())
        {
            CV_CheckTypeEQ(dynamicBias_->tensor()->type(), CV_32F,
                           "Metal tensor must be CV_32F");
            CV_CheckEQ(dynamicBias_->tensor()->total(), outputChannels,
                       "Metal Deconv2D bias size mismatch");
        }
        else
        {
            biasHost_.create(1, outputShape[1], CV_32F);
            if (bias.empty())
                biasHost_.setTo(Scalar(0));
            else
            {
                CV_CheckTypeEQ(bias.type(), CV_32F,
                               "Metal Deconv2D bias must be CV_32F");
                CV_CheckEQ(bias.total(), outputChannels,
                           "Metal Deconv2D bias size mismatch");
                bias.reshape(1, 1).copyTo(biasHost_);
            }
            biasTensor_ = metal::Tensor::create(biasHost_);
        }

        const size_t inputChannels = static_cast<size_t>(inputShape[1]);
        const size_t inputChannelsPerGroup = inputChannels / groups;
        const size_t outputChannelsPerGroup = outputChannels / groups;
        useOC4_ = !hasDynamicWeights && outputChannelsPerGroup % 4 == 0;
        if (useOC4_)
        {
            packedWeightsHost_.create(1, static_cast<int>(weights.total()), CV_32F);
            const float* source = weights.ptr<float>();
            float* destination = packedWeightsHost_.ptr<float>();
            const size_t kernelSize = kernelHeight * kernelWidth;
            const size_t outputBlocksPerGroup = outputChannelsPerGroup / 4;
            for (size_t group = 0; group < groups; ++group)
            {
                for (size_t localInputChannel = 0;
                     localInputChannel < inputChannelsPerGroup; ++localInputChannel)
                {
                    const size_t inputChannel =
                        group * inputChannelsPerGroup + localInputChannel;
                    for (size_t outputBlock = 0;
                         outputBlock < outputBlocksPerGroup; ++outputBlock)
                    {
                        const size_t globalOutputBlock =
                            group * outputBlocksPerGroup + outputBlock;
                        for (size_t kernelIndex = 0; kernelIndex < kernelSize; ++kernelIndex)
                        {
                            for (size_t lane = 0; lane < 4; ++lane)
                            {
                                const size_t localOutputChannel = outputBlock * 4 + lane;
                                const size_t sourceIndex =
                                    (inputChannel * outputChannelsPerGroup + localOutputChannel) *
                                    kernelSize + kernelIndex;
                                const size_t destinationIndex =
                                    ((globalOutputBlock * inputChannelsPerGroup +
                                      localInputChannel) * kernelSize + kernelIndex) * 4 + lane;
                                destination[destinationIndex] = source[sourceIndex];
                            }
                        }
                    }
                }
            }
            packedWeightsTensor_ = metal::Tensor::create(packedWeightsHost_);
        }
        else if (!hasDynamicWeights)
        {
            weightsHost_ = weights.clone();
            weightsTensor_ = metal::Tensor::create(weightsHost_);
        }

        size_t effectivePadTop = padTop;
        size_t effectivePadLeft = padLeft;
        if (paddingMode == metal::DeconvolutionConfiguration::PaddingMode::VALID)
        {
            effectivePadTop = 0;
            effectivePadLeft = 0;
        }
        else if (paddingMode == metal::DeconvolutionConfiguration::PaddingMode::SAME)
        {
            effectivePadTop = strideHeight <= kernelHeight
                ? (kernelHeight - 1 -
                   (static_cast<size_t>(outputShape[2] - 1) + strideHeight) % strideHeight) / 2
                : 0;
            effectivePadLeft = strideWidth <= kernelWidth
                ? (kernelWidth - 1 -
                   (static_cast<size_t>(outputShape[3] - 1) + strideWidth) % strideWidth) / 2
                : 0;
        }

        parameters_.batch = static_cast<uint32_t>(inputShape[0]);
        parameters_.inputChannels = static_cast<uint32_t>(inputShape[1]);
        parameters_.inputHeight = static_cast<uint32_t>(inputShape[2]);
        parameters_.inputWidth = static_cast<uint32_t>(inputShape[3]);
        parameters_.outputChannels = static_cast<uint32_t>(outputShape[1]);
        parameters_.outputHeight = static_cast<uint32_t>(outputShape[2]);
        parameters_.outputWidth = static_cast<uint32_t>(outputShape[3]);
        parameters_.kernelHeight = static_cast<uint32_t>(kernelHeight);
        parameters_.kernelWidth = static_cast<uint32_t>(kernelWidth);
        parameters_.strideHeight = static_cast<uint32_t>(strideHeight);
        parameters_.strideWidth = static_cast<uint32_t>(strideWidth);
        parameters_.dilationHeight = static_cast<uint32_t>(dilationHeight);
        parameters_.dilationWidth = static_cast<uint32_t>(dilationWidth);
        parameters_.padTop = static_cast<uint32_t>(effectivePadTop);
        parameters_.padLeft = static_cast<uint32_t>(effectivePadLeft);
        parameters_.groups = static_cast<uint32_t>(groups);
    }

    void forward(metal::Context& context) CV_OVERRIDE
    {
        const std::shared_ptr<metal::Tensor>& inputTensor = input_->tensor();
        const std::shared_ptr<metal::Tensor>& outputTensor = output_->tensor();
        const std::shared_ptr<metal::Tensor>& weightTensor =
            dynamicWeights_.empty() ? weightsTensor_ : dynamicWeights_->tensor();
        const std::shared_ptr<metal::Tensor>& biasTensor =
            dynamicBias_.empty() ? biasTensor_ : dynamicBias_->tensor();
        const std::shared_ptr<metal::Buffer>& inputBuffer = inputTensor->bufferForRead();
        const std::shared_ptr<metal::Buffer>& weightsBuffer =
            useOC4_ ? packedWeightsTensor_->bufferForRead() : weightTensor->bufferForRead();
        const std::shared_ptr<metal::Buffer>& biasBuffer = biasTensor->bufferForRead();
        const std::shared_ptr<metal::Buffer>& outputBuffer = outputTensor->bufferForWrite();

        id<MTLComputeCommandEncoder> encoder = context.computeEncoder();
        [encoder setBuffer:context.use(inputBuffer) offset:0 atIndex:0];
        [encoder setBuffer:context.use(weightsBuffer) offset:0 atIndex:1];
        [encoder setBuffer:context.use(biasBuffer) offset:0 atIndex:2];
        [encoder setBuffer:context.use(outputBuffer) offset:0 atIndex:3];
        [encoder setBytes:&parameters_ length:sizeof(parameters_) atIndex:4];
        if (useOC4_)
        {
            if (!oc4Pipeline_)
            {
                oc4Pipeline_ = metal::Device::instance()->pipelineState(
                    "kernel_deconv2d_oc4_f32");
            }
            const size_t count = outputTensor->total() / 4;
            [encoder setComputePipelineState:oc4Pipeline_];
            [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
                threadsPerThreadgroup:context.threadsPerThreadgroup1D(oc4Pipeline_, count)];
        }
        else
        {
            if (!pipeline_)
            {
                pipeline_ = metal::Device::instance()->pipelineState(
                    "kernel_deconv2d_f32");
            }
            const size_t count = outputTensor->total();
            [encoder setComputePipelineState:pipeline_];
            [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
                threadsPerThreadgroup:context.threadsPerThreadgroup1D(pipeline_, count)];
        }
        context.didDispatch();
    }

private:
    struct Parameters
    {
        uint32_t batch;
        uint32_t inputChannels;
        uint32_t inputHeight;
        uint32_t inputWidth;
        uint32_t outputChannels;
        uint32_t outputHeight;
        uint32_t outputWidth;
        uint32_t kernelHeight;
        uint32_t kernelWidth;
        uint32_t strideHeight;
        uint32_t strideWidth;
        uint32_t dilationHeight;
        uint32_t dilationWidth;
        uint32_t padTop;
        uint32_t padLeft;
        uint32_t groups;
    } parameters_;

    Ptr<MetalBackendWrapper> input_;
    Ptr<MetalBackendWrapper> output_;
    Ptr<MetalBackendWrapper> dynamicWeights_;
    Ptr<MetalBackendWrapper> dynamicBias_;
    id<MTLComputePipelineState> pipeline_ = nil;
    id<MTLComputePipelineState> oc4Pipeline_ = nil;
    Mat weightsHost_;
    Mat packedWeightsHost_;
    Mat biasHost_;
    std::shared_ptr<metal::Tensor> weightsTensor_;
    std::shared_ptr<metal::Tensor> packedWeightsTensor_;
    std::shared_ptr<metal::Tensor> biasTensor_;
    bool useOC4_ = false;
};

std::unique_ptr<metal::Operation> makeDeconvolutionOperation(
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs,
        const metal::DeconvolutionConfiguration& config)
{
    CV_CheckEQ(outputs.size(), static_cast<size_t>(1),
               "Metal Deconv2D requires one output");
    const bool hasDynamicWeights = config.weights.empty();
    if (hasDynamicWeights)
    {
        CV_CheckGE(inputs.size(), static_cast<size_t>(2),
                   "Metal Deconv2D with dynamic weights expects a weight input");
        CV_CheckLE(inputs.size(), static_cast<size_t>(3),
                   "Metal Deconv2D supports at most input, weights and bias");
    }
    else
    {
        CV_CheckEQ(inputs.size(), static_cast<size_t>(1),
                   "Metal Deconv2D with static weights expects one input");
    }
    Ptr<MetalBackendWrapper> dynamicWeights;
    Ptr<MetalBackendWrapper> dynamicBias;
    if (hasDynamicWeights)
    {
        dynamicWeights = inputs[1].dynamicCast<MetalBackendWrapper>();
        CV_CheckTrue(!dynamicWeights.empty(),
                     "Metal Deconv2D requires a Metal weight tensor wrapper");
        if (inputs.size() == 3)
        {
            dynamicBias = inputs[2].dynamicCast<MetalBackendWrapper>();
            CV_CheckTrue(!dynamicBias.empty(),
                         "Metal Deconv2D requires a Metal bias tensor wrapper");
        }
    }
    return std::unique_ptr<metal::Operation>(new MetalDeconvolutionImpl(
        inputs[0].dynamicCast<MetalBackendWrapper>(),
        outputs[0].dynamicCast<MetalBackendWrapper>(),
        dynamicWeights, dynamicBias, config.weights, config.bias, config.groups,
        config.kernelHeight, config.kernelWidth,
        config.strideHeight, config.strideWidth,
        config.dilationHeight, config.dilationWidth,
        config.padTop, config.padLeft, config.paddingMode));
}

}  // namespace

CV__DNN_INLINE_NS_END

namespace metal {

Ptr<BackendNode> DeconvolutionOp::create(
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs,
        const DeconvolutionConfiguration& config)
{
    return makeMetalBackendNode(makeDeconvolutionOperation(inputs, outputs, config));
}

}  // namespace metal
}}  // namespace cv::dnn

#endif  // HAVE_METAL
