// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"

#ifdef HAVE_METAL

#include "topk.hpp"
#include "operation.hpp"
#include "../runtime/context.hpp"
#include "../runtime/device.hpp"
#include "../runtime/tensor.hpp"

#include <limits>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

namespace {

class MetalTopKImpl final : public metal::Operation
{
public:
    MetalTopKImpl(const Ptr<MetalBackendWrapper>& input,
                  const Ptr<MetalBackendWrapper>& outputValues,
                  const Ptr<MetalBackendWrapper>& outputIndices,
                  const metal::TopKConfiguration& config)
        : input_(input), outputValues_(outputValues), outputIndices_(outputIndices),
          largest_(config.largest)
    {
        CV_CheckTrue(!input_.empty() && !outputValues_.empty() && !outputIndices_.empty(),
                     "Metal TopK requires Metal tensor wrappers");
        CV_CheckTypeEQ(input_->tensor()->type(), CV_32F, "Metal TopK input must be CV_32F");
        CV_CheckTypeEQ(outputValues_->tensor()->type(), CV_32F,
                       "Metal TopK values must be CV_32F");
        CV_CheckTypeEQ(outputIndices_->tensor()->type(), CV_64S,
                       "Metal TopK indices must be CV_64S");

        const MatShape& inputShape = input_->tensor()->shape();
        CV_CheckFalse(inputShape.empty(), "Metal TopK input shape must not be empty");
        int axis = config.axis;
        if (axis < 0)
            axis += static_cast<int>(inputShape.size());
        CV_CheckGE(axis, 0, "Metal TopK axis is out of range");
        CV_CheckLT(axis, static_cast<int>(inputShape.size()),
                   "Metal TopK axis is out of range");
        CV_CheckGT(config.k, 0, "Metal TopK K must be positive");
        CV_CheckLE(config.k, inputShape[axis], "Metal TopK K is out of range");

        size_t outerSize = 1;
        for (int i = 0; i < axis; ++i)
            outerSize *= static_cast<size_t>(inputShape[i]);
        size_t innerSize = 1;
        for (size_t i = static_cast<size_t>(axis + 1); i < inputShape.size(); ++i)
            innerSize *= static_cast<size_t>(inputShape[i]);
        const size_t rowCount = outerSize * innerSize;
        CV_CheckLE(rowCount, static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
                   "Metal TopK row count exceeds uint32 capacity");
        CV_CheckLE(innerSize, static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
                   "Metal TopK inner size exceeds uint32 capacity");
        CV_CheckEQ(outputValues_->tensor()->total(),
                   rowCount * static_cast<size_t>(config.k),
                   "Metal TopK values shape is invalid");
        CV_CheckEQ(outputIndices_->tensor()->total(), outputValues_->tensor()->total(),
                   "Metal TopK output shapes must match");

        parameters_.rowCount = static_cast<uint32_t>(rowCount);
        parameters_.axisSize = static_cast<uint32_t>(inputShape[axis]);
        parameters_.innerSize = static_cast<uint32_t>(innerSize);
        parameters_.k = static_cast<uint32_t>(config.k);
    }

    void forward(metal::Context& context) CV_OVERRIDE
    {
        if (!pipeline_)
        {
            pipeline_ = metal::Device::instance()->pipelineState(
                largest_ ? "kernel_topk_largest_f32" : "kernel_topk_smallest_f32");
        }

        const std::shared_ptr<metal::Tensor>& inputTensor = input_->tensor();
        const std::shared_ptr<metal::Tensor>& outputValuesTensor = outputValues_->tensor();
        const std::shared_ptr<metal::Tensor>& outputIndicesTensor = outputIndices_->tensor();
        const auto& inputBuffer = inputTensor->bufferForRead();
        const auto& outputValuesBuffer = outputValuesTensor->bufferForWrite();
        const auto& outputIndicesBuffer = outputIndicesTensor->bufferForWrite();

        const size_t pipelineLimit = std::min(
            static_cast<size_t>(256),
            static_cast<size_t>([pipeline_ maxTotalThreadsPerThreadgroup]));
        size_t maximumPowerOfTwo = 1;
        while (maximumPowerOfTwo <= pipelineLimit / 2)
            maximumPowerOfTwo *= 2;
        size_t threadsPerThreadgroup = 1;
        while (threadsPerThreadgroup < parameters_.axisSize &&
               threadsPerThreadgroup < maximumPowerOfTwo)
        {
            threadsPerThreadgroup *= 2;
        }

        id<MTLComputeCommandEncoder> encoder = context.computeEncoder();
        [encoder setComputePipelineState:pipeline_];
        [encoder setBuffer:context.use(inputBuffer) offset:0 atIndex:0];
        [encoder setBuffer:context.use(outputValuesBuffer) offset:0 atIndex:1];
        [encoder setBuffer:context.use(outputIndicesBuffer) offset:0 atIndex:2];
        [encoder setBytes:&parameters_ length:sizeof(parameters_) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(parameters_.rowCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadsPerThreadgroup, 1, 1)];
        context.didDispatch();
    }

private:
    struct Parameters
    {
        uint32_t rowCount;
        uint32_t axisSize;
        uint32_t innerSize;
        uint32_t k;
    } parameters_;

    Ptr<MetalBackendWrapper> input_;
    Ptr<MetalBackendWrapper> outputValues_;
    Ptr<MetalBackendWrapper> outputIndices_;
    bool largest_ = true;
    id<MTLComputePipelineState> pipeline_ = nil;
};

static std::unique_ptr<metal::Operation> makeTopKOperation(
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs,
        const metal::TopKConfiguration& config)
{
    CV_CheckEQ(inputs.size(), static_cast<size_t>(1),
               "Metal TopK requires exactly one input");
    CV_CheckEQ(outputs.size(), static_cast<size_t>(2),
               "Metal TopK requires values and indices outputs");
    return std::unique_ptr<metal::Operation>(new MetalTopKImpl(
        inputs[0].dynamicCast<MetalBackendWrapper>(),
        outputs[0].dynamicCast<MetalBackendWrapper>(),
        outputs[1].dynamicCast<MetalBackendWrapper>(), config));
}

}  // namespace

CV__DNN_INLINE_NS_END

namespace metal {

Ptr<BackendNode> TopKOp::create(const std::vector<Ptr<BackendWrapper>>& inputs,
                                const std::vector<Ptr<BackendWrapper>>& outputs,
                                const TopKConfiguration& config)
{
    return makeMetalBackendNode(makeTopKOperation(inputs, outputs, config));
}

}  // namespace metal
}}  // namespace cv::dnn

#endif  // HAVE_METAL
