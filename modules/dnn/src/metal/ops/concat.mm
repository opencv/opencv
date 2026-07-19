// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"

#ifdef HAVE_METAL

#include "concat.hpp"
#include "operation.hpp"
#include "../runtime/context.hpp"
#include "../runtime/device.hpp"
#include "../runtime/tensor.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

class MetalConcatImpl final : public metal::Operation
{
public:
    MetalConcatImpl(
        const std::vector<Ptr<MetalBackendWrapper> >& inputs,
                    const Ptr<MetalBackendWrapper>& output,
                    int axis)
        : inputs_(inputs), output_(output)
    {
        const MatShape& outputShape = output_->tensor()->shape();
        if (axis < 0)
            axis += static_cast<int>(outputShape.size());
        CV_CheckGE(axis, 0, "Metal Concat axis is out of range");
        CV_CheckLT(axis, static_cast<int>(outputShape.size()), "Metal Concat axis is out of range");
        CV_CheckTypeEQ(output_->tensor()->type(), CV_32F, "Metal tensor must be CV_32F");

        size_t innerSize = 1;
        for (size_t d = static_cast<size_t>(axis + 1); d < outputShape.size(); ++d)
            innerSize *= static_cast<size_t>(outputShape[d]);

        uint32_t axisOffset = 0;
        parameters_.resize(inputs_.size());
        for (size_t i = 0; i < inputs_.size(); ++i)
        {
            CV_CheckTypeEQ(inputs_[i]->tensor()->type(), CV_32F, "Metal tensor must be CV_32F");
            const std::shared_ptr<metal::Tensor>& inputTensor = inputs_[i]->tensor();
            Parameters& parameters = parameters_[i];
            parameters.count = static_cast<uint32_t>(inputTensor->total());
            parameters.innerSize = static_cast<uint32_t>(innerSize);
            parameters.inputAxisSize = static_cast<uint32_t>(inputTensor->shape()[axis]);
            parameters.outputAxisSize = static_cast<uint32_t>(outputShape[axis]);
            parameters.axisOffset = axisOffset;
            axisOffset += parameters.inputAxisSize;
        }
    }

    void forward(metal::Context& context) CV_OVERRIDE
    {
        if (!pipeline_)
        {
            pipeline_ = metal::Device::instance()->pipelineState("kernel_concat_f32");
        }

        for (size_t i = 0; i < inputs_.size(); ++i)
        {
            const std::shared_ptr<metal::Tensor>& inputTensor = inputs_[i]->tensor();
            const std::shared_ptr<metal::Tensor>& outputTensor = output_->tensor();
            const Parameters& parameters = parameters_[i];

            const auto& inputBuffer = inputTensor->bufferForRead();
            const auto& outputBuffer = outputTensor->bufferForWrite();
            id<MTLComputeCommandEncoder> encoder = context.computeEncoder();
            [encoder setComputePipelineState:pipeline_];
            [encoder setBuffer:context.use(inputBuffer) offset:0 atIndex:0];
            [encoder setBuffer:context.use(outputBuffer) offset:0 atIndex:1];
            [encoder setBytes:&parameters length:sizeof(parameters) atIndex:2];
            [encoder dispatchThreads:MTLSizeMake(parameters.count, 1, 1)
                threadsPerThreadgroup:context.threadsPerThreadgroup1D(
                    pipeline_, parameters.count)];
            context.didDispatch();
        }
    }

private:
    struct Parameters
    {
        uint32_t count;
        uint32_t innerSize;
        uint32_t inputAxisSize;
        uint32_t outputAxisSize;
        uint32_t axisOffset;
    };
    std::vector<Ptr<MetalBackendWrapper> > inputs_;
    Ptr<MetalBackendWrapper> output_;
    id<MTLComputePipelineState> pipeline_ = nil;
    std::vector<Parameters> parameters_;
};


static std::unique_ptr<metal::Operation> makeConcatOperation(
        const std::vector<Ptr<BackendWrapper> >& inputs,
                                 const std::vector<Ptr<BackendWrapper> >& outputs,
                                 int axis)
{
    std::vector<Ptr<MetalBackendWrapper>> metalInputs(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i)
        metalInputs[i] = inputs[i].dynamicCast<MetalBackendWrapper>();
    return std::unique_ptr<metal::Operation>(new MetalConcatImpl(
        metalInputs,
        outputs[0].dynamicCast<MetalBackendWrapper>(), axis));
}

CV__DNN_INLINE_NS_END

namespace metal {

Ptr<BackendNode> ConcatOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                  const std::vector<Ptr<BackendWrapper> >& outputs, int axis)
{
    return makeMetalBackendNode(makeConcatOperation(inputs, outputs, axis));
}


}  // namespace metal
}}  // namespace cv::dnn

#endif  // HAVE_METAL
