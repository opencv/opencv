// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"

#ifdef HAVE_METAL

#include "permute.hpp"
#include "operation.hpp"
#include "../runtime/context.hpp"
#include "../runtime/device.hpp"
#include "../runtime/tensor.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

class MetalPermuteImpl final : public metal::Operation
{
public:
    MetalPermuteImpl(
        const Ptr<MetalBackendWrapper>& input,
                     const Ptr<MetalBackendWrapper>& output,
                     const std::vector<size_t>& order)
        : input_(input), output_(output)
    {
        const MatShape& inputShape = input_->tensor()->shape();
        const MatShape& outputShape = output_->tensor()->shape();
        CV_CheckEQ(order.size(), inputShape.size(), "Metal Permute order rank mismatch");
        CV_CheckEQ(outputShape.size(), inputShape.size(), "Metal Permute output rank mismatch");
        CV_CheckLE(order.size(), static_cast<size_t>(8), "Metal Permute supports rank up to 8");
        CV_CheckTypeEQ(input_->tensor()->type(), CV_32F, "Metal tensor must be CV_32F");
        CV_CheckTypeEQ(output_->tensor()->type(), CV_32F, "Metal tensor must be CV_32F");

        parameters_ = Parameters();
        parameters_.count = static_cast<uint32_t>(output_->tensor()->total());
        parameters_.rank = static_cast<uint32_t>(order.size());
        uint32_t inputStride = 1, outputStride = 1;
        for (int d = static_cast<int>(order.size()) - 1; d >= 0; --d)
        {
            CV_CheckLT(order[d], order.size(), "Metal Permute order is invalid");
            parameters_.order[d] = static_cast<uint32_t>(order[d]);
            parameters_.inputStride[d] = inputStride;
            parameters_.outputStride[d] = outputStride;
            inputStride *= static_cast<uint32_t>(inputShape[d]);
            outputStride *= static_cast<uint32_t>(outputShape[d]);
        }
    }

    void forward(metal::Context& context) CV_OVERRIDE
    {
        if (!pipeline_)
        {
            pipeline_ = metal::Device::instance()->pipelineState("kernel_permute_f32");
        }

        const std::shared_ptr<metal::Tensor>& inputTensor = input_->tensor();
        const std::shared_ptr<metal::Tensor>& outputTensor = output_->tensor();
        const auto& inputBuffer = inputTensor->bufferForRead();
        const auto& outputBuffer = outputTensor->bufferForWrite();
        id<MTLComputeCommandEncoder> encoder = context.computeEncoder();
        [encoder setComputePipelineState:pipeline_];
        [encoder setBuffer:context.use(inputBuffer) offset:0 atIndex:0];
        [encoder setBuffer:context.use(outputBuffer) offset:0 atIndex:1];
        [encoder setBytes:&parameters_ length:sizeof(parameters_) atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(parameters_.count, 1, 1)
            threadsPerThreadgroup:context.threadsPerThreadgroup1D(pipeline_, parameters_.count)];
        context.didDispatch();
    }

private:
    struct Parameters
    {
        uint32_t count;
        uint32_t rank;
        uint32_t order[8];
        uint32_t inputStride[8];
        uint32_t outputStride[8];
    } parameters_;
    Ptr<MetalBackendWrapper> input_;
    Ptr<MetalBackendWrapper> output_;
    id<MTLComputePipelineState> pipeline_ = nil;
};


static std::unique_ptr<metal::Operation> makePermuteOperation(
        const std::vector<Ptr<BackendWrapper> >& inputs,
                                  const std::vector<Ptr<BackendWrapper> >& outputs,
                                  const std::vector<size_t>& order)
{
    return std::unique_ptr<metal::Operation>(new MetalPermuteImpl(
        inputs[0].dynamicCast<MetalBackendWrapper>(),
        outputs[0].dynamicCast<MetalBackendWrapper>(), order));
}

CV__DNN_INLINE_NS_END

namespace metal {

Ptr<BackendNode> PermuteOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                   const std::vector<Ptr<BackendWrapper> >& outputs,
                                   const std::vector<size_t>& order)
{
    return makeMetalBackendNode(makePermuteOperation(inputs, outputs, order));
}


}  // namespace metal
}}  // namespace cv::dnn

#endif  // HAVE_METAL
