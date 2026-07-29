// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"

#ifdef HAVE_METAL

#include "reshape.hpp"
#include "operation.hpp"
#include "../runtime/context.hpp"
#include "../runtime/device.hpp"
#include "../runtime/tensor.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

class MetalReshapeImpl final : public metal::Operation
{
public:
    MetalReshapeImpl(
        const Ptr<MetalBackendWrapper>& input,
                  const Ptr<MetalBackendWrapper>& output)
        : input_(input), output_(output)
    {
        CV_CheckTypeEQ(input_->tensor()->type(), CV_32F, "Metal input must be CV_32F");
        CV_CheckTypeEQ(output_->tensor()->type(), CV_32F, "Metal output must be CV_32F");
        parameters_.count = static_cast<uint32_t>(output_->tensor()->total());
    }

    void forward(metal::Context& context) CV_OVERRIDE
    {
        if (!pipeline_)
        {
            pipeline_ = metal::Device::instance()->pipelineState("kernel_reshape_f32");
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
    struct Parameters { uint32_t count; } parameters_;
    Ptr<MetalBackendWrapper> input_;
    Ptr<MetalBackendWrapper> output_;
    id<MTLComputePipelineState> pipeline_ = nil;
};


static std::unique_ptr<metal::Operation> makeReshapeOperation(
        const std::vector<Ptr<BackendWrapper> >& inputs,
                               const std::vector<Ptr<BackendWrapper> >& outputs)
{
    return std::unique_ptr<metal::Operation>(new MetalReshapeImpl(
        inputs[0].dynamicCast<MetalBackendWrapper>(),
        outputs[0].dynamicCast<MetalBackendWrapper>()));
}

CV__DNN_INLINE_NS_END

namespace metal {

Ptr<BackendNode> ReshapeOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                const std::vector<Ptr<BackendWrapper> >& outputs)
{
    return makeMetalBackendNode(makeReshapeOperation(inputs, outputs));
}


}  // namespace metal
}}  // namespace cv::dnn

#endif  // HAVE_METAL
