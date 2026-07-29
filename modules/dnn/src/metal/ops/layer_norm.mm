// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"

#ifdef HAVE_METAL

#include "layer_norm.hpp"
#include "operation.hpp"
#include "../runtime/context.hpp"
#include "../runtime/device.hpp"
#include "../runtime/tensor.hpp"

#include <limits>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

namespace {

class MetalLayerNormImpl final : public metal::Operation
{
public:
    MetalLayerNormImpl(const std::vector<Ptr<BackendWrapper>>& inputs,
                       const std::vector<Ptr<BackendWrapper>>& outputs,
                       const metal::LayerNormConfiguration& config)
        : input_(inputs[0].dynamicCast<MetalBackendWrapper>()),
          output_(outputs[0].dynamicCast<MetalBackendWrapper>()),
          hasBias_(config.hasBias)
    {
        CV_CheckEQ(outputs.size(), static_cast<size_t>(1),
                   "Metal LayerNorm requires exactly one output");
        CV_CheckTrue(!input_.empty() && !output_.empty(),
                     "Metal LayerNorm requires Metal tensor wrappers");

        const MatShape& inputShape = input_->tensor()->shape();
        CV_CheckFalse(inputShape.empty(), "Metal LayerNorm input shape must not be empty");
        CV_CheckTypeEQ(input_->tensor()->type(), CV_32F, "Metal input must be CV_32F");
        CV_CheckTypeEQ(output_->tensor()->type(), CV_32F, "Metal output must be CV_32F");

        int axis = config.axis;
        if (axis < 0)
            axis += static_cast<int>(inputShape.size());
        CV_CheckGE(axis, 0, "Metal LayerNorm axis is out of range");
        CV_CheckLT(axis, static_cast<int>(inputShape.size()),
                   "Metal LayerNorm axis is out of range");

        size_t normalizationSize = 1;
        for (size_t i = static_cast<size_t>(axis); i < inputShape.size(); ++i)
            normalizationSize *= static_cast<size_t>(inputShape[i]);
        const size_t total = input_->tensor()->total();
        CV_CheckGT(normalizationSize, static_cast<size_t>(0),
                   "Metal LayerNorm size must be positive");
        CV_CheckEQ(total % normalizationSize, static_cast<size_t>(0),
                   "Metal LayerNorm shape is invalid");
        CV_CheckLE(normalizationSize,
                   static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
                   "Metal LayerNorm size exceeds uint32 capacity");
        CV_CheckLE(total / normalizationSize,
                   static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
                   "Metal LayerNorm row count exceeds uint32 capacity");

        size_t inputIndex = 1;
        if (config.scale.empty())
        {
            CV_CheckLT(inputIndex, inputs.size(), "Metal LayerNorm scale input is missing");
            scale_ = inputs[inputIndex++].dynamicCast<MetalBackendWrapper>();
            CV_CheckTrue(!scale_.empty(), "Metal LayerNorm scale must be a Metal tensor");
            CV_CheckTypeEQ(scale_->tensor()->type(), CV_32F,
                           "Metal LayerNorm scale must be CV_32F");
            CV_CheckEQ(scale_->tensor()->total(), normalizationSize,
                       "Metal LayerNorm scale shape does not match input");
        }
        else
        {
            CV_CheckTypeEQ(config.scale.type(), CV_32F,
                           "Metal LayerNorm scale must be CV_32F");
            CV_CheckEQ(config.scale.total(), normalizationSize,
                       "Metal LayerNorm scale shape does not match input");
            config.scale.reshape(1, 1).copyTo(scaleHost_);
            scaleTensor_ = metal::Tensor::create(scaleHost_);
        }

        if (hasBias_)
        {
            if (config.bias.empty())
            {
                CV_CheckLT(inputIndex, inputs.size(), "Metal LayerNorm bias input is missing");
                bias_ = inputs[inputIndex].dynamicCast<MetalBackendWrapper>();
                CV_CheckTrue(!bias_.empty(), "Metal LayerNorm bias must be a Metal tensor");
                CV_CheckTypeEQ(bias_->tensor()->type(), CV_32F,
                               "Metal LayerNorm bias must be CV_32F");
                CV_CheckEQ(bias_->tensor()->total(), normalizationSize,
                           "Metal LayerNorm bias shape does not match input");
            }
            else
            {
                CV_CheckTypeEQ(config.bias.type(), CV_32F,
                               "Metal LayerNorm bias must be CV_32F");
                CV_CheckEQ(config.bias.total(), normalizationSize,
                           "Metal LayerNorm bias shape does not match input");
                config.bias.reshape(1, 1).copyTo(biasHost_);
                biasTensor_ = metal::Tensor::create(biasHost_);
            }
        }

        parameters_.rowCount = static_cast<uint32_t>(total / normalizationSize);
        parameters_.normalizationSize = static_cast<uint32_t>(normalizationSize);
        parameters_.epsilon = config.epsilon;
    }

    void forward(metal::Context& context) CV_OVERRIDE
    {
        if (!pipeline_)
            pipeline_ = metal::Device::instance()->pipelineState("kernel_layer_norm_f32");

        const std::shared_ptr<metal::Tensor>& inputTensor = input_->tensor();
        const std::shared_ptr<metal::Tensor>& outputTensor = output_->tensor();
        const std::shared_ptr<metal::Tensor> scaleTensor =
            scale_.empty() ? scaleTensor_ : scale_->tensor();
        const std::shared_ptr<metal::Tensor> biasTensor = hasBias_
            ? (bias_.empty() ? biasTensor_ : bias_->tensor()) : scaleTensor;

        const auto& inputBuffer = inputTensor->bufferForRead();
        const auto& scaleBuffer = scaleTensor->bufferForRead();
        const auto& biasBuffer = biasTensor->bufferForRead();
        const auto& outputBuffer = outputTensor->bufferForWrite();

        const size_t executionWidth = static_cast<size_t>([pipeline_ threadExecutionWidth]);
        const size_t pipelineLimit =
            static_cast<size_t>([pipeline_ maxTotalThreadsPerThreadgroup]);
        const size_t preferredLimit =
            std::min(std::max(static_cast<size_t>(256), executionWidth), pipelineLimit);
        const size_t requiredThreads =
            (static_cast<size_t>(parameters_.normalizationSize) + executionWidth - 1) /
            executionWidth * executionWidth;
        const size_t threadsPerThreadgroup = std::min(requiredThreads, preferredLimit);

        id<MTLComputeCommandEncoder> encoder = context.computeEncoder();
        [encoder setComputePipelineState:pipeline_];
        [encoder setBuffer:context.use(inputBuffer) offset:0 atIndex:0];
        [encoder setBuffer:context.use(scaleBuffer) offset:0 atIndex:1];
        [encoder setBuffer:context.use(biasBuffer) offset:0 atIndex:2];
        [encoder setBuffer:context.use(outputBuffer) offset:0 atIndex:3];
        [encoder setBytes:&parameters_ length:sizeof(parameters_) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(parameters_.rowCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadsPerThreadgroup, 1, 1)];
        context.didDispatch();
    }

private:
    struct Parameters
    {
        uint32_t rowCount;
        uint32_t normalizationSize;
        float epsilon;
    } parameters_;

    Ptr<MetalBackendWrapper> input_;
    Ptr<MetalBackendWrapper> output_;
    Ptr<MetalBackendWrapper> scale_;
    Ptr<MetalBackendWrapper> bias_;
    bool hasBias_ = false;
    id<MTLComputePipelineState> pipeline_ = nil;
    Mat scaleHost_;
    Mat biasHost_;
    std::shared_ptr<metal::Tensor> scaleTensor_;
    std::shared_ptr<metal::Tensor> biasTensor_;
};

}  // namespace

CV__DNN_INLINE_NS_END

namespace metal {

Ptr<BackendNode> LayerNormOp::create(
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs,
        const LayerNormConfiguration& config)
{
    return makeMetalBackendNode(
        std::unique_ptr<Operation>(new MetalLayerNormImpl(inputs, outputs, config)));
}

}  // namespace metal
}}  // namespace cv::dnn

#endif  // HAVE_METAL
