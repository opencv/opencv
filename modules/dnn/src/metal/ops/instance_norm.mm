// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"

#ifdef HAVE_METAL

#include "instance_norm.hpp"
#include "operation.hpp"
#include "../runtime/context.hpp"
#include "../runtime/device.hpp"
#include "../runtime/tensor.hpp"

#include <limits>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

namespace {

class MetalInstanceNormImpl final : public metal::Operation
{
public:
    MetalInstanceNormImpl(const std::vector<Ptr<BackendWrapper>>& inputs,
                          const std::vector<Ptr<BackendWrapper>>& outputs,
                          const metal::InstanceNormConfiguration& config)
        : input_(inputs[0].dynamicCast<MetalBackendWrapper>()),
          scale_(inputs[1].dynamicCast<MetalBackendWrapper>()),
          bias_(inputs[2].dynamicCast<MetalBackendWrapper>()),
          output_(outputs[0].dynamicCast<MetalBackendWrapper>())
    {
        CV_CheckTrue(!input_.empty() && !scale_.empty() && !bias_.empty() &&
                     !output_.empty(),
                     "Metal InstanceNorm requires Metal tensor wrappers");
        CV_CheckTypeEQ(input_->tensor()->type(), CV_32F,
                       "Metal InstanceNorm input must be CV_32F");
        CV_CheckTypeEQ(scale_->tensor()->type(), CV_32F,
                       "Metal InstanceNorm scale must be CV_32F");
        CV_CheckTypeEQ(bias_->tensor()->type(), CV_32F,
                       "Metal InstanceNorm bias must be CV_32F");
        CV_CheckTypeEQ(output_->tensor()->type(), CV_32F,
                       "Metal InstanceNorm output must be CV_32F");

        const MatShape& inputShape = input_->tensor()->shape();
        CV_CheckGE(inputShape.size(), static_cast<size_t>(3),
                   "Metal InstanceNorm input rank must be at least three");
        CV_CheckNE(inputShape.layout, DATA_LAYOUT_BLOCK,
                   "Metal InstanceNorm does not support blocked layout");
        CV_CheckEQ(input_->tensor()->total(), output_->tensor()->total(),
                   "Metal InstanceNorm input and output sizes must match");

        const size_t batch = static_cast<size_t>(inputShape[0]);
        const size_t channels = static_cast<size_t>(inputShape[1]);
        const size_t groupCount = batch * channels;
        const size_t total = input_->tensor()->total();
        CV_CheckGT(groupCount, static_cast<size_t>(0),
                   "Metal InstanceNorm group count must be positive");
        CV_CheckEQ(total % groupCount, static_cast<size_t>(0),
                   "Metal InstanceNorm input shape is invalid");
        CV_CheckEQ(scale_->tensor()->total(), channels,
                   "Metal InstanceNorm scale size must match channels");
        CV_CheckEQ(bias_->tensor()->total(), channels,
                   "Metal InstanceNorm bias size must match channels");

        const size_t normalizationSize = total / groupCount;
        CV_CheckLE(groupCount,
                   static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
                   "Metal InstanceNorm group count exceeds uint32 capacity");
        CV_CheckLE(channels,
                   static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
                   "Metal InstanceNorm channel count exceeds uint32 capacity");
        CV_CheckLE(normalizationSize,
                   static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
                   "Metal InstanceNorm size exceeds uint32 capacity");

        parameters_.groupCount = static_cast<uint32_t>(groupCount);
        parameters_.channels = static_cast<uint32_t>(channels);
        parameters_.normalizationSize = static_cast<uint32_t>(normalizationSize);
        parameters_.epsilon = config.epsilon;
    }

    void forward(metal::Context& context) CV_OVERRIDE
    {
        if (!pipeline_)
            pipeline_ = metal::Device::instance()->pipelineState(
                "kernel_instance_norm_f32");

        const auto& inputBuffer = input_->tensor()->bufferForRead();
        const auto& scaleBuffer = scale_->tensor()->bufferForRead();
        const auto& biasBuffer = bias_->tensor()->bufferForRead();
        const auto& outputBuffer = output_->tensor()->bufferForWrite();

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
        [encoder dispatchThreadgroups:MTLSizeMake(parameters_.groupCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadsPerThreadgroup, 1, 1)];
        context.didDispatch();
    }

private:
    struct Parameters
    {
        uint32_t groupCount;
        uint32_t channels;
        uint32_t normalizationSize;
        float epsilon;
    } parameters_;

    Ptr<MetalBackendWrapper> input_;
    Ptr<MetalBackendWrapper> scale_;
    Ptr<MetalBackendWrapper> bias_;
    Ptr<MetalBackendWrapper> output_;
    id<MTLComputePipelineState> pipeline_ = nil;
};

}  // namespace

CV__DNN_INLINE_NS_END

namespace metal {

Ptr<BackendNode> InstanceNormOp::create(
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs,
        const InstanceNormConfiguration& config)
{
    CV_CheckEQ(inputs.size(), static_cast<size_t>(3),
               "Metal InstanceNorm requires input, scale and bias");
    CV_CheckEQ(outputs.size(), static_cast<size_t>(1),
               "Metal InstanceNorm requires one output");
    return makeMetalBackendNode(std::unique_ptr<Operation>(
        new MetalInstanceNormImpl(inputs, outputs, config)));
}

}  // namespace metal
}}  // namespace cv::dnn

#endif  // HAVE_METAL
