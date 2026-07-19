// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"

#ifdef HAVE_METAL

#include "eltwise.hpp"
#include "operation.hpp"
#include "../runtime/context.hpp"
#include "../runtime/device.hpp"
#include "../runtime/tensor.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

class MetalEltwiseAddImpl final : public metal::Operation
{
public:
    MetalEltwiseAddImpl(
        const std::vector<Ptr<MetalBackendWrapper> >& inputs,
                        const Ptr<MetalBackendWrapper>& output)
        : inputs_(inputs), output_(output)
    {
        const size_t count = output_->tensor()->total();
        CV_CheckTypeEQ(output_->tensor()->type(), CV_32F, "Metal tensor must be CV_32F");
        for (size_t i = 0; i < inputs_.size(); ++i)
        {
            CV_CheckTypeEQ(inputs_[i]->tensor()->type(), CV_32F, "Metal tensor must be CV_32F");
            CV_CheckEQ(inputs_[i]->tensor()->total(), count,
                       "Metal Eltwise Add does not support broadcast inputs");
        }
        parameters_.count = static_cast<uint32_t>(count);
    }

    void forward(metal::Context& context) CV_OVERRIDE
    {
        if (!pipeline_)
        {
            pipeline_ = metal::Device::instance()->pipelineState("kernel_add_f32");
        }

        const std::shared_ptr<metal::Tensor>& outputTensor = output_->tensor();
        for (size_t i = 1; i < inputs_.size(); ++i)
        {
            const std::shared_ptr<metal::Tensor>& lhs =
                i == 1 ? inputs_[0]->tensor() : outputTensor;
            const std::shared_ptr<metal::Tensor>& rhs = inputs_[i]->tensor();
            const auto& lhsBuffer = lhs->bufferForRead();
            const auto& rhsBuffer = rhs->bufferForRead();
            const auto& outputBuffer = outputTensor->bufferForWrite();
            id<MTLComputeCommandEncoder> encoder = context.computeEncoder();
            [encoder setComputePipelineState:pipeline_];
            [encoder setBuffer:context.use(lhsBuffer) offset:0 atIndex:0];
            [encoder setBuffer:context.use(rhsBuffer) offset:0 atIndex:1];
            [encoder setBuffer:context.use(outputBuffer) offset:0 atIndex:2];
            [encoder setBytes:&parameters_ length:sizeof(parameters_) atIndex:3];
            [encoder dispatchThreads:MTLSizeMake(parameters_.count, 1, 1)
                threadsPerThreadgroup:context.threadsPerThreadgroup1D(
                    pipeline_, parameters_.count)];
            context.didDispatch();
        }
    }

private:
    struct Parameters { uint32_t count; } parameters_;
    std::vector<Ptr<MetalBackendWrapper> > inputs_;
    Ptr<MetalBackendWrapper> output_;
    id<MTLComputePipelineState> pipeline_ = nil;
};


static std::unique_ptr<metal::Operation> makeEltwiseAddOperation(
        const std::vector<Ptr<BackendWrapper> >& inputs,
                                     const std::vector<Ptr<BackendWrapper> >& outputs)
{
    std::vector<Ptr<MetalBackendWrapper>> metalInputs(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i)
        metalInputs[i] = inputs[i].dynamicCast<MetalBackendWrapper>();
    return std::unique_ptr<metal::Operation>(new MetalEltwiseAddImpl(
        metalInputs,
        outputs[0].dynamicCast<MetalBackendWrapper>()));
}


CV__DNN_INLINE_NS_END

namespace metal {

Ptr<BackendNode> EltwiseAddOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                      const std::vector<Ptr<BackendWrapper> >& outputs)
{
    return makeMetalBackendNode(makeEltwiseAddOperation(inputs, outputs));
}

}  // namespace metal
}}  // namespace cv::dnn

#endif  // HAVE_METAL
