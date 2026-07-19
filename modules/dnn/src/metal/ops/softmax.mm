// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"

#ifdef HAVE_METAL

#include "softmax.hpp"
#include "operation.hpp"
#include "../runtime/context.hpp"
#include "../runtime/device.hpp"
#include "../runtime/tensor.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

class MetalSoftmaxImpl final : public metal::Operation
{
public:
    MetalSoftmaxImpl(
        const Ptr<MetalBackendWrapper>& input,
                     const Ptr<MetalBackendWrapper>& output,
                     int axis,
                     bool logSoftmax,
                     float scale)
        : input_(input), output_(output)
    {
        const MatShape& inputShape = input_->tensor()->shape();
        if (axis < 0)
            axis += static_cast<int>(inputShape.size());
        CV_CheckGE(axis, 0, "Metal Softmax axis is out of range");
        CV_CheckLT(axis, static_cast<int>(inputShape.size()), "Metal Softmax axis is out of range");
        size_t innerSize = 1;
        for (size_t d = static_cast<size_t>(axis + 1); d < inputShape.size(); ++d)
            innerSize *= static_cast<size_t>(inputShape[d]);
        CV_CheckTypeEQ(input_->tensor()->type(), CV_32F, "Metal input must be CV_32F");
        CV_CheckTypeEQ(output_->tensor()->type(), CV_32F, "Metal output must be CV_32F");
        parameters_.count = static_cast<uint32_t>(output_->tensor()->total());
        parameters_.channels = static_cast<uint32_t>(inputShape[axis]);
        parameters_.innerSize = static_cast<uint32_t>(innerSize);
        parameters_.logSoftmax = logSoftmax ? 1u : 0u;
        parameters_.scale = scale;
    }

    void forward(metal::Context& context) CV_OVERRIDE
    {
        const std::shared_ptr<metal::Tensor>& inputTensor = input_->tensor();
        const std::shared_ptr<metal::Tensor>& outputTensor = output_->tensor();
        const bool canUseContiguousPipeline =
            parameters_.innerSize == 1 && parameters_.logSoftmax == 0 &&
            parameters_.channels <= 4096;
        id<MTLComputePipelineState> pipeline = nil;
        size_t executionWidth = 0;
        size_t pipelineLimit = 0;
        size_t requiredThreads = 0;
        bool useContiguousPipeline = false;
        if (canUseContiguousPipeline)
        {
            if (!contiguousPipeline_)
            {
                contiguousPipeline_ = metal::Device::instance()->pipelineState("kernel_softmax_contiguous_f32");
            }

            pipeline = contiguousPipeline_;
            executionWidth = static_cast<size_t>([pipeline threadExecutionWidth]);
            pipelineLimit = static_cast<size_t>([pipeline maxTotalThreadsPerThreadgroup]);
            requiredThreads = (static_cast<size_t>(parameters_.channels) + 3) / 4;
            requiredThreads =
                (requiredThreads + executionWidth - 1) / executionWidth * executionWidth;
            useContiguousPipeline = requiredThreads <= pipelineLimit;
        }

        if (!useContiguousPipeline)
        {
            if (!pipeline_)
            {
                pipeline_ = metal::Device::instance()->pipelineState("kernel_softmax_f32");
            }

            pipeline = pipeline_;
            executionWidth = static_cast<size_t>([pipeline threadExecutionWidth]);
            pipelineLimit = static_cast<size_t>([pipeline maxTotalThreadsPerThreadgroup]);
            requiredThreads =
                (static_cast<size_t>(parameters_.channels) + executionWidth - 1) / executionWidth *
                executionWidth;
        }
        const size_t preferredLimit =
            std::min(std::max(static_cast<size_t>(256), executionWidth), pipelineLimit);
        const size_t maximumThreads = preferredLimit - preferredLimit % executionWidth;
        const size_t threadsPerThreadgroup = useContiguousPipeline
            ? requiredThreads : std::min(requiredThreads, maximumThreads);
        const size_t threadgroupCount = parameters_.count / parameters_.channels;
        const auto& inputBuffer = inputTensor->bufferForRead();
        const auto& outputBuffer = outputTensor->bufferForWrite();
        id<MTLComputeCommandEncoder> encoder = context.computeEncoder();
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:context.use(inputBuffer) offset:0 atIndex:0];
        [encoder setBuffer:context.use(outputBuffer) offset:0 atIndex:1];
        [encoder setBytes:&parameters_ length:sizeof(parameters_) atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(threadgroupCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadsPerThreadgroup, 1, 1)];
        context.didDispatch();
    }

private:
    struct Parameters
    {
        uint32_t count;
        uint32_t channels;
        uint32_t innerSize;
        uint32_t logSoftmax;
        float scale;
    } parameters_;
    Ptr<MetalBackendWrapper> input_;
    Ptr<MetalBackendWrapper> output_;
    id<MTLComputePipelineState> pipeline_ = nil;
    id<MTLComputePipelineState> contiguousPipeline_ = nil;
};


static std::unique_ptr<metal::Operation> makeSoftmaxOperation(
        const std::vector<Ptr<BackendWrapper> >& inputs,
        const std::vector<Ptr<BackendWrapper> >& outputs,
        const metal::SoftmaxConfiguration& config)
{
    return std::unique_ptr<metal::Operation>(new MetalSoftmaxImpl(
        inputs[0].dynamicCast<MetalBackendWrapper>(),
        outputs[0].dynamicCast<MetalBackendWrapper>(),
        config.axis, config.logSoftmax, config.scale));
}

CV__DNN_INLINE_NS_END

namespace metal {

Ptr<BackendNode> SoftmaxOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                   const std::vector<Ptr<BackendWrapper> >& outputs,
                                   const SoftmaxConfiguration& config)
{
    return makeMetalBackendNode(makeSoftmaxOperation(inputs, outputs, config));
}

}  // namespace metal
}}  // namespace cv::dnn

#endif  // HAVE_METAL
