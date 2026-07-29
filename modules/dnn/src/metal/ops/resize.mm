// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"

#ifdef HAVE_METAL

#include "resize.hpp"
#include "operation.hpp"
#include "../runtime/context.hpp"
#include "../runtime/device.hpp"
#include "../runtime/tensor.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

class MetalResizeImpl final : public metal::Operation
{
public:
    MetalResizeImpl(
        const Ptr<MetalBackendWrapper>& input,
                    const Ptr<MetalBackendWrapper>& output,
                    int interpolation,
                    bool alignCorners,
                    bool halfPixelCenters)
        : input_(input), output_(output)
    {
        const MatShape& inputShape = input_->tensor()->shape();
        const MatShape& outputShape = output_->tensor()->shape();
        CV_CheckEQ(inputShape.size(), static_cast<size_t>(4),
                   "Metal Resize expects 4D NCHW input");
        CV_CheckEQ(outputShape.size(), static_cast<size_t>(4),
                   "Metal Resize expects 4D NCHW output");
        CV_CheckTypeEQ(input_->tensor()->type(), CV_32F, "Metal tensor must be CV_32F");
        CV_CheckTypeEQ(output_->tensor()->type(), CV_32F, "Metal tensor must be CV_32F");

        parameters_.inputHeight = static_cast<uint32_t>(inputShape[2]);
        parameters_.inputWidth = static_cast<uint32_t>(inputShape[3]);
        parameters_.outputHeight = static_cast<uint32_t>(outputShape[2]);
        parameters_.outputWidth = static_cast<uint32_t>(outputShape[3]);
        parameters_.interpolation = static_cast<uint32_t>(interpolation);
        parameters_.alignCorners = alignCorners ? 1u : 0u;
        parameters_.halfPixelCenters = halfPixelCenters ? 1u : 0u;
        parameters_.scaleHeight = alignCorners && outputShape[2] > 1
            ? static_cast<float>(inputShape[2] - 1) / static_cast<float>(outputShape[2] - 1)
            : static_cast<float>(inputShape[2]) / static_cast<float>(outputShape[2]);
        parameters_.scaleWidth = alignCorners && outputShape[3] > 1
            ? static_cast<float>(inputShape[3] - 1) / static_cast<float>(outputShape[3] - 1)
            : static_cast<float>(inputShape[3]) / static_cast<float>(outputShape[3]);
    }

    void forward(metal::Context& context) CV_OVERRIDE
    {
        if (!pipeline_)
        {
            pipeline_ = metal::Device::instance()->pipelineState("kernel_resize2d_f32");
        }

        const std::shared_ptr<metal::Tensor>& inputTensor = input_->tensor();
        const std::shared_ptr<metal::Tensor>& outputTensor = output_->tensor();
        const size_t count = outputTensor->total();
        const auto& inputBuffer = inputTensor->bufferForRead();
        const auto& outputBuffer = outputTensor->bufferForWrite();
        id<MTLComputeCommandEncoder> encoder = context.computeEncoder();
        [encoder setComputePipelineState:pipeline_];
        [encoder setBuffer:context.use(inputBuffer) offset:0 atIndex:0];
        [encoder setBuffer:context.use(outputBuffer) offset:0 atIndex:1];
        [encoder setBytes:&parameters_ length:sizeof(parameters_) atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
            threadsPerThreadgroup:context.threadsPerThreadgroup1D(pipeline_, count)];
        context.didDispatch();
    }

private:
    struct Parameters
    {
        uint32_t inputHeight;
        uint32_t inputWidth;
        uint32_t outputHeight;
        uint32_t outputWidth;
        uint32_t interpolation;
        uint32_t alignCorners;
        uint32_t halfPixelCenters;
        float scaleHeight;
        float scaleWidth;
    } parameters_;
    Ptr<MetalBackendWrapper> input_;
    Ptr<MetalBackendWrapper> output_;
    id<MTLComputePipelineState> pipeline_ = nil;
};


static std::unique_ptr<metal::Operation> makeResizeOperation(
        const std::vector<Ptr<BackendWrapper> >& inputs,
        const std::vector<Ptr<BackendWrapper> >& outputs,
        const metal::ResizeConfiguration& config)
{
    return std::unique_ptr<metal::Operation>(new MetalResizeImpl(
        inputs[0].dynamicCast<MetalBackendWrapper>(),
        outputs[0].dynamicCast<MetalBackendWrapper>(),
        static_cast<int>(config.interpolation), config.alignCorners,
        config.halfPixelCenters));
}

CV__DNN_INLINE_NS_END

namespace metal {

Ptr<BackendNode> ResizeOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                  const std::vector<Ptr<BackendWrapper> >& outputs,
                                  const ResizeConfiguration& config)
{
    return makeMetalBackendNode(makeResizeOperation(inputs, outputs, config));
}


}  // namespace metal
}}  // namespace cv::dnn

#endif  // HAVE_METAL
