// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"

#ifdef HAVE_METAL

#include "convolution.hpp"
#include "operation.hpp"
#include "../runtime/context.hpp"
#include "../runtime/device.hpp"
#include "../runtime/tensor.hpp"
#include "../runtime/buffer.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

static const float WINOGRAD_WEIGHT_TRANSFORM[8][3] = {
    {1.0f, 0.0f, 0.0f},
    {-2.0f / 9.0f, -2.0f / 9.0f, -2.0f / 9.0f},
    {-2.0f / 9.0f, 2.0f / 9.0f, -2.0f / 9.0f},
    {1.0f / 90.0f, 1.0f / 45.0f, 2.0f / 45.0f},
    {1.0f / 90.0f, -1.0f / 45.0f, 2.0f / 45.0f},
    {32.0f / 45.0f, 16.0f / 45.0f, 8.0f / 45.0f},
    {32.0f / 45.0f, -16.0f / 45.0f, 8.0f / 45.0f},
    {0.0f, 0.0f, 1.0f},
};

class MetalConvolutionImpl final : public metal::Operation
{
public:
    MetalConvolutionImpl(
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
            size_t padLeft)
        : input_(input), output_(output), dynamicWeights_(dynamicWeights),
          dynamicBias_(dynamicBias)
    {
        const MatShape& inputShape = input_->tensor()->shape();
        const MatShape& outputShape = output_->tensor()->shape();
        CV_CheckEQ(inputShape.size(), static_cast<size_t>(4),
                   "Metal Conv2D expects 4D NCHW input");
        CV_CheckEQ(outputShape.size(), static_cast<size_t>(4),
                   "Metal Conv2D expects 4D NCHW output");
        CV_CheckTypeEQ(input_->tensor()->type(), CV_32F, "Metal tensor must be CV_32F");
        CV_CheckTypeEQ(output_->tensor()->type(), CV_32F, "Metal tensor must be CV_32F");
        const bool hasDynamicWeights = !dynamicWeights_.empty();
        CV_CheckNE(hasDynamicWeights, !weights.empty(),
                   "Metal Conv2D requires exactly one static or dynamic weight source");
        MatShape weightShape;
        if (hasDynamicWeights)
        {
            CV_CheckTypeEQ(dynamicWeights_->tensor()->type(), CV_32F, "Metal tensor must be CV_32F");
            weightShape = dynamicWeights_->tensor()->shape();
        }
        else
        {
            CV_CheckTypeEQ(weights.type(), CV_32F, "Metal Conv2D weights must be CV_32F");
            CV_CheckTrue(weights.isContinuous(), "Metal Conv2D weights must be continuous");
            weightShape = cv::dnn::shape(weights);
        }
        CV_CheckEQ(weightShape.size(), static_cast<size_t>(4),
                   "Metal Conv2D weights must use OIHW layout");
        CV_CheckGT(groups, static_cast<size_t>(0), "Metal Conv2D groups must be positive");
        CV_CheckEQ(static_cast<size_t>(inputShape[1]) % groups, static_cast<size_t>(0),
                   "Metal Conv2D input channels must be divisible by groups");
        CV_CheckEQ(static_cast<size_t>(outputShape[1]) % groups, static_cast<size_t>(0),
                   "Metal Conv2D output channels must be divisible by groups");
        CV_CheckEQ(weightShape[0], outputShape[1], "Metal Conv2D output channels mismatch");
        CV_CheckEQ(weightShape[1], inputShape[1] / static_cast<int>(groups),
                   "Metal Conv2D input channels per group mismatch");
        CV_CheckEQ(weightShape[2], static_cast<int>(kernelHeight),
                   "Metal Conv2D kernel height mismatch");
        CV_CheckEQ(weightShape[3], static_cast<int>(kernelWidth),
                   "Metal Conv2D kernel width mismatch");

        if (!dynamicBias_.empty())
        {
            CV_CheckTypeEQ(dynamicBias_->tensor()->type(), CV_32F, "Metal tensor must be CV_32F");
            CV_CheckEQ(dynamicBias_->tensor()->total(), static_cast<size_t>(outputShape[1]),
                       "Metal Conv2D bias size mismatch");
        }
        else
        {
            biasHost_.create(1, outputShape[1], CV_32F);
            if (bias.empty())
                biasHost_.setTo(Scalar(0));
            else
            {
                CV_CheckTypeEQ(bias.type(), CV_32F, "Metal Conv2D bias must be CV_32F");
                CV_CheckEQ(bias.total(), static_cast<size_t>(outputShape[1]),
                           "Metal Conv2D bias size mismatch");
                bias.reshape(1, 1).copyTo(biasHost_);
            }
            biasTensor_ = metal::Tensor::create(biasHost_);
        }

        const size_t inputChannels = static_cast<size_t>(inputShape[1]);
        const size_t outputChannels = static_cast<size_t>(outputShape[1]);
        const size_t inputChannelsPerGroup = inputChannels / groups;
        const size_t outputChannelsPerGroup = outputChannels / groups;
        useDepthwise3x3_ = !hasDynamicWeights &&
                           groups == inputChannels && groups == outputChannels &&
                           kernelHeight == 3 && kernelWidth == 3 &&
                           dilationHeight == 1 && dilationWidth == 1 &&
                           strideHeight <= 2 && strideWidth <= 2;
        const size_t inputArea =
            static_cast<size_t>(inputShape[0]) * inputShape[2] * inputShape[3];
        useWinograd_ = !hasDynamicWeights && !useDepthwise3x3_ &&
                       groups == 1 && kernelHeight == 3 &&
                       kernelWidth == 3 && strideHeight == 1 && strideWidth == 1 &&
                       dilationHeight == 1 && dilationWidth == 1 && inputChannels % 32 == 0 &&
                       outputChannels % 32 == 0 && inputArea >= 4096 &&
                       inputChannels + outputChannels >= 256;
        const size_t outputPixels =
            static_cast<size_t>(outputShape[0]) * outputShape[2] * outputShape[3];
        const size_t kernelSize = kernelHeight * kernelWidth;
        const bool preferOC4ForWideSpatial =
            outputPixels > 16384 && outputChannelsPerGroup <= 64 && kernelSize > 1;
        const bool implicitGemmHasEnoughReuse =
            kernelSize == 1 || outputChannelsPerGroup >= 256 || kernelSize >= 25;
        useImplicitGemm_ = !hasDynamicWeights && !useDepthwise3x3_ && !useWinograd_ &&
                           inputChannelsPerGroup >= 8 && outputChannelsPerGroup >= 8 &&
                           outputPixels * outputChannelsPerGroup >= 16384 &&
                           implicitGemmHasEnoughReuse && !preferOC4ForWideSpatial;
        useOC4_ = !hasDynamicWeights && !useDepthwise3x3_ && !useWinograd_ &&
                  !useImplicitGemm_ &&
                  outputChannelsPerGroup % 4 == 0;
        if (useWinograd_)
        {
            const size_t transformedWeightCount = 64 * inputChannels * outputChannels;
            CV_CheckLE(transformedWeightCount, static_cast<size_t>(INT_MAX),
                       "Metal Winograd transformed weights exceed Mat capacity");
            transformedWeightsHost_.create(1, static_cast<int>(transformedWeightCount), CV_32F);
            const float* source = weights.ptr<float>();
            float* destination = transformedWeightsHost_.ptr<float>();
            for (size_t outputChannel = 0; outputChannel < outputChannels; ++outputChannel)
            {
                for (size_t inputChannel = 0; inputChannel < inputChannels; ++inputChannel)
                {
                    float temporary[8][3] = {};
                    for (size_t y = 0; y < 8; ++y)
                    {
                        for (size_t x = 0; x < 3; ++x)
                        {
                            for (size_t k = 0; k < 3; ++k)
                            {
                                const size_t sourceIndex =
                                    ((outputChannel * inputChannels + inputChannel) * 3 + k) * 3 + x;
                                temporary[y][x] +=
                                    WINOGRAD_WEIGHT_TRANSFORM[y][k] * source[sourceIndex];
                            }
                        }
                    }
                    for (size_t y = 0; y < 8; ++y)
                    {
                        for (size_t x = 0; x < 8; ++x)
                        {
                            float value = 0.0f;
                            for (size_t k = 0; k < 3; ++k)
                                value += temporary[y][k] * WINOGRAD_WEIGHT_TRANSFORM[x][k];
                            destination[((y * 8 + x) * outputChannels + outputChannel) *
                                        inputChannels + inputChannel] = value;
                        }
                    }
                }
            }
            transformedWeightsTensor_ = metal::Tensor::create(transformedWeightsHost_);
            const size_t tilesX = (static_cast<size_t>(outputShape[3]) + 5) / 6;
            const size_t tilesY = (static_cast<size_t>(outputShape[2]) + 5) / 6;
            winogradTileCount_ = static_cast<size_t>(outputShape[0]) * tilesX * tilesY;
            transformedInputBuffer_ =
                metal::Buffer::create(64 * winogradTileCount_ * inputChannels * sizeof(float));
            transformedOutputBuffer_ =
                metal::Buffer::create(64 * winogradTileCount_ * outputChannels * sizeof(float));
        }
        else if (useOC4_)
        {
            packedWeightsHost_.create(1, static_cast<int>(weights.total()), CV_32F);
            const float* source = weights.ptr<float>();
            float* destination = packedWeightsHost_.ptr<float>();
            const size_t kernelSize = kernelHeight * kernelWidth;
            const size_t outputBlockCount = outputChannels / 4;
            for (size_t outputBlock = 0; outputBlock < outputBlockCount; ++outputBlock)
            {
                for (size_t inputChannel = 0; inputChannel < inputChannelsPerGroup;
                     ++inputChannel)
                {
                    for (size_t kernelIndex = 0; kernelIndex < kernelSize; ++kernelIndex)
                    {
                        for (size_t lane = 0; lane < 4; ++lane)
                        {
                            const size_t outputChannel = outputBlock * 4 + lane;
                            const size_t sourceIndex =
                                (outputChannel * inputChannelsPerGroup + inputChannel) *
                                kernelSize + kernelIndex;
                            const size_t destinationIndex =
                                ((outputBlock * inputChannelsPerGroup + inputChannel) *
                                 kernelSize + kernelIndex) * 4 + lane;
                            destination[destinationIndex] = source[sourceIndex];
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
        parameters_.padTop = static_cast<uint32_t>(padTop);
        parameters_.padLeft = static_cast<uint32_t>(padLeft);
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
        if (useWinograd_)
        {
            if (!winogradInputPipeline_)
            {
                winogradInputPipeline_ = metal::Device::instance()->pipelineState("kernel_conv2d_winograd_input_f32");
            }

            const size_t inputCount = winogradTileCount_ * parameters_.inputChannels;
            const auto& inputBuffer = inputTensor->bufferForRead();
            id<MTLComputeCommandEncoder> encoder = context.computeEncoder();
            [encoder setComputePipelineState:winogradInputPipeline_];
            [encoder setBuffer:context.use(inputBuffer) offset:0 atIndex:0];
            [encoder setBuffer:context.use(transformedInputBuffer_) offset:0 atIndex:1];
            [encoder setBytes:&parameters_ length:sizeof(parameters_) atIndex:4];
            [encoder dispatchThreads:MTLSizeMake(inputCount, 1, 1)
                threadsPerThreadgroup:context.threadsPerThreadgroup1D(
                    winogradInputPipeline_, inputCount)];
            context.didDispatch();

            if (!winogradGemmPipeline_)
            {
                winogradGemmPipeline_ = metal::Device::instance()->pipelineState("kernel_conv2d_winograd_gemm_f32");
            }

            const size_t gemmThreadgroups = 64 * ((winogradTileCount_ + 31) / 32) *
                                            ((parameters_.outputChannels + 31) / 32);
            const auto& transformedWeightsBuffer = transformedWeightsTensor_->bufferForRead();
            encoder = context.computeEncoder();
            [encoder setComputePipelineState:winogradGemmPipeline_];
            [encoder setBuffer:context.use(transformedInputBuffer_) offset:0 atIndex:0];
            [encoder setBuffer:context.use(transformedWeightsBuffer) offset:0 atIndex:1];
            [encoder setBuffer:context.use(transformedOutputBuffer_) offset:0 atIndex:2];
            [encoder setBytes:&parameters_ length:sizeof(parameters_) atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake(gemmThreadgroups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            context.didDispatch();

            if (!winogradOutputPipeline_)
            {
                winogradOutputPipeline_ = metal::Device::instance()->pipelineState("kernel_conv2d_winograd_output_f32");
            }

            const size_t outputCount = winogradTileCount_ * parameters_.outputChannels;
            const auto& biasBuffer = biasTensor->bufferForRead();
            const auto& outputBuffer = outputTensor->bufferForWrite();
            encoder = context.computeEncoder();
            [encoder setComputePipelineState:winogradOutputPipeline_];
            [encoder setBuffer:context.use(transformedOutputBuffer_) offset:0 atIndex:0];
            [encoder setBuffer:context.use(biasBuffer) offset:0 atIndex:1];
            [encoder setBuffer:context.use(outputBuffer) offset:0 atIndex:2];
            [encoder setBytes:&parameters_ length:sizeof(parameters_) atIndex:4];
            [encoder dispatchThreads:MTLSizeMake(outputCount, 1, 1)
                threadsPerThreadgroup:context.threadsPerThreadgroup1D(
                    winogradOutputPipeline_, outputCount)];
            context.didDispatch();
        }
        else
        {
            const auto& inputBuffer = inputTensor->bufferForRead();
            const auto& weightsBuffer = useOC4_ ? packedWeightsTensor_->bufferForRead()
                                                : weightTensor->bufferForRead();
            const auto& biasBuffer = biasTensor->bufferForRead();
            const auto& outputBuffer = outputTensor->bufferForWrite();
            id<MTLComputeCommandEncoder> encoder = context.computeEncoder();
            [encoder setBuffer:context.use(inputBuffer) offset:0 atIndex:0];
            [encoder setBuffer:context.use(weightsBuffer) offset:0 atIndex:1];
            [encoder setBuffer:context.use(biasBuffer) offset:0 atIndex:2];
            [encoder setBuffer:context.use(outputBuffer) offset:0 atIndex:3];
            [encoder setBytes:&parameters_ length:sizeof(parameters_) atIndex:4];
            if (useDepthwise3x3_)
            {
                if (!depthwise3x3Pipeline_)
                {
                    depthwise3x3Pipeline_ = metal::Device::instance()->pipelineState("kernel_depthwise_conv2d_3x3_f32");
                }

                const size_t tilesX = (parameters_.outputWidth + 7) / 8;
                const size_t tilesY = (parameters_.outputHeight + 7) / 8;
                const size_t threadgroupCount =
                    static_cast<size_t>(parameters_.batch) * parameters_.outputChannels *
                    tilesX * tilesY;
                [encoder setComputePipelineState:depthwise3x3Pipeline_];
                [encoder dispatchThreadgroups:MTLSizeMake(threadgroupCount, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            }
            else if (useImplicitGemm_)
            {
                if (!implicitGemmPipeline_)
                {
                    implicitGemmPipeline_ = metal::Device::instance()->pipelineState("kernel_conv2d_implicit_gemm_f32");
                }

                const size_t outputPixels = static_cast<size_t>(parameters_.batch) *
                                            parameters_.outputHeight * parameters_.outputWidth;
                const size_t outputChannelsPerGroup =
                    parameters_.outputChannels / parameters_.groups;
                const size_t threadgroupCount = parameters_.groups * ((outputPixels + 31) / 32) *
                                                ((outputChannelsPerGroup + 31) / 32);
                [encoder setComputePipelineState:implicitGemmPipeline_];
                [encoder dispatchThreadgroups:MTLSizeMake(threadgroupCount, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
            else if (useOC4_)
            {
                if (!oc4Pipeline_)
                {
                    oc4Pipeline_ = metal::Device::instance()->pipelineState("kernel_conv2d_oc4_f32");
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
                    pipeline_ = metal::Device::instance()->pipelineState("kernel_conv2d_f32");
                }

                const size_t count = outputTensor->total();
                [encoder setComputePipelineState:pipeline_];
                [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
                    threadsPerThreadgroup:context.threadsPerThreadgroup1D(pipeline_, count)];
            }
            context.didDispatch();
        }
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
    id<MTLComputePipelineState> implicitGemmPipeline_ = nil;
    id<MTLComputePipelineState> winogradInputPipeline_ = nil;
    id<MTLComputePipelineState> winogradGemmPipeline_ = nil;
    id<MTLComputePipelineState> winogradOutputPipeline_ = nil;
    id<MTLComputePipelineState> depthwise3x3Pipeline_ = nil;
    Mat weightsHost_;
    Mat packedWeightsHost_;
    Mat transformedWeightsHost_;
    Mat biasHost_;
    std::shared_ptr<metal::Tensor> weightsTensor_;
    std::shared_ptr<metal::Tensor> packedWeightsTensor_;
    std::shared_ptr<metal::Tensor> transformedWeightsTensor_;
    std::shared_ptr<metal::Tensor> biasTensor_;
    std::shared_ptr<metal::Buffer> transformedInputBuffer_;
    std::shared_ptr<metal::Buffer> transformedOutputBuffer_;
    size_t winogradTileCount_ = 0;
    bool useOC4_ = false;
    bool useImplicitGemm_ = false;
    bool useWinograd_ = false;
    bool useDepthwise3x3_ = false;
};

static std::unique_ptr<metal::Operation> makeConvolutionOperation(
        const std::vector<Ptr<BackendWrapper> >& inputs,
        const std::vector<Ptr<BackendWrapper> >& outputs,
        const metal::ConvolutionConfiguration& config)
{
    const bool hasDynamicWeights = config.weights.empty();
    if (hasDynamicWeights)
    {
        CV_CheckGE(inputs.size(), static_cast<size_t>(2),
                   "Metal Conv2D with dynamic weights expects a weight input");
        CV_CheckLE(inputs.size(), static_cast<size_t>(3),
                   "Metal Conv2D supports at most input, weights and bias");
    }
    else
    {
        CV_CheckEQ(inputs.size(), static_cast<size_t>(1),
                   "Metal Conv2D with static weights expects one input");
    }
    Ptr<MetalBackendWrapper> dynamicWeights;
    Ptr<MetalBackendWrapper> dynamicBias;
    if (hasDynamicWeights)
    {
        dynamicWeights = inputs[1].dynamicCast<MetalBackendWrapper>();
        if (inputs.size() == 3)
            dynamicBias = inputs[2].dynamicCast<MetalBackendWrapper>();
    }
    return std::unique_ptr<metal::Operation>(new MetalConvolutionImpl(
        inputs[0].dynamicCast<MetalBackendWrapper>(),
        outputs[0].dynamicCast<MetalBackendWrapper>(),
        dynamicWeights, dynamicBias,
        config.weights, config.bias, config.groups,
        config.kernelHeight, config.kernelWidth,
        config.strideHeight, config.strideWidth,
        config.dilationHeight, config.dilationWidth,
        config.padTop, config.padLeft));
}

CV__DNN_INLINE_NS_END

namespace metal {

Ptr<BackendNode> ConvolutionOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                       const std::vector<Ptr<BackendWrapper> >& outputs,
                                       const ConvolutionConfiguration& config)
{
    return makeMetalBackendNode(makeConvolutionOperation(inputs, outputs, config));
}

}  // namespace metal
}}  // namespace cv::dnn

#endif  // HAVE_METAL
