// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"

#ifdef HAVE_METAL

#include "fully_connected.hpp"
#include "operation.hpp"
#include "../runtime/context.hpp"
#include "../runtime/device.hpp"
#include "../runtime/tensor.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

class MetalFullyConnectedImpl final : public metal::Operation
{
public:
    MetalFullyConnectedImpl(
        const Ptr<MetalBackendWrapper>& input,
                            const Ptr<MetalBackendWrapper>& output,
                            const Mat& weights,
                            const Mat& bias,
                            int axis)
        : input_(input), output_(output)
    {
        CV_CheckTypeEQ(input_->tensor()->type(), CV_32F, "Metal tensor must be CV_32F");
        CV_CheckTypeEQ(output_->tensor()->type(), CV_32F, "Metal tensor must be CV_32F");
        CV_CheckTypeEQ(weights.type(), CV_32F,
                       "Metal FullyConnected weights must be CV_32F");
        CV_CheckEQ(weights.dims, 2, "Metal FullyConnected weights must be 2D");

        const MatShape& inputShape = input_->tensor()->shape();
        if (axis < 0)
            axis += static_cast<int>(inputShape.size());
        CV_CheckGE(axis, 0, "Metal FullyConnected axis is out of range");
        CV_CheckLT(axis, static_cast<int>(inputShape.size()),
                   "Metal FullyConnected axis is out of range");
        size_t innerSize = 1;
        for (size_t d = static_cast<size_t>(axis); d < inputShape.size(); ++d)
            innerSize *= static_cast<size_t>(inputShape[d]);
        CV_CheckEQ(static_cast<size_t>(weights.cols), innerSize,
                   "Metal FullyConnected input size mismatch");

        weightsHost_ = weights.clone();
        biasHost_.create(1, weights.rows, CV_32F);
        if (bias.empty())
            biasHost_.setTo(Scalar(0));
        else
        {
            CV_CheckTypeEQ(bias.type(), CV_32F, "Metal FullyConnected bias must be CV_32F");
            CV_CheckEQ(bias.total(), static_cast<size_t>(weights.rows),
                       "Metal FullyConnected bias size mismatch");
            bias.reshape(1, 1).copyTo(biasHost_);
        }
        weightsTensor_ = metal::Tensor::create(weightsHost_);
        biasTensor_ = metal::Tensor::create(biasHost_);

        parameters_.outputCount = static_cast<uint32_t>(output_->tensor()->total());
        parameters_.innerSize = static_cast<uint32_t>(innerSize);
        parameters_.outputChannels = static_cast<uint32_t>(weights.rows);
    }

    void forward(metal::Context& context) CV_OVERRIDE
    {
        if (!pipeline_)
        {
            pipeline_ = metal::Device::instance()->pipelineState("kernel_fully_connected_f32");
        }

        const std::shared_ptr<metal::Tensor>& inputTensor = input_->tensor();
        const std::shared_ptr<metal::Tensor>& outputTensor = output_->tensor();
        const auto& inputBuffer = inputTensor->bufferForRead();
        const auto& weightsBuffer = weightsTensor_->bufferForRead();
        const auto& biasBuffer = biasTensor_->bufferForRead();
        const auto& outputBuffer = outputTensor->bufferForWrite();
        id<MTLComputeCommandEncoder> encoder = context.computeEncoder();
        [encoder setComputePipelineState:pipeline_];
        [encoder setBuffer:context.use(inputBuffer) offset:0 atIndex:0];
        [encoder setBuffer:context.use(weightsBuffer) offset:0 atIndex:1];
        [encoder setBuffer:context.use(biasBuffer) offset:0 atIndex:2];
        [encoder setBuffer:context.use(outputBuffer) offset:0 atIndex:3];
        [encoder setBytes:&parameters_ length:sizeof(parameters_) atIndex:4];
        [encoder dispatchThreads:MTLSizeMake(parameters_.outputCount, 1, 1)
            threadsPerThreadgroup:context.threadsPerThreadgroup1D(
                pipeline_, parameters_.outputCount)];
        context.didDispatch();
    }

private:
    struct Parameters
    {
        uint32_t outputCount;
        uint32_t innerSize;
        uint32_t outputChannels;
    } parameters_;
    Ptr<MetalBackendWrapper> input_;
    Ptr<MetalBackendWrapper> output_;
    Mat weightsHost_;
    Mat biasHost_;
    std::shared_ptr<metal::Tensor> weightsTensor_;
    std::shared_ptr<metal::Tensor> biasTensor_;
    id<MTLComputePipelineState> pipeline_ = nil;
};


static std::unique_ptr<metal::Operation> makeFullyConnectedOperation(
        const std::vector<Ptr<BackendWrapper> >& inputs,
        const std::vector<Ptr<BackendWrapper> >& outputs,
        const metal::FullyConnectedConfiguration& config)
{
    return std::unique_ptr<metal::Operation>(new MetalFullyConnectedImpl(
        inputs[0].dynamicCast<MetalBackendWrapper>(),
        outputs[0].dynamicCast<MetalBackendWrapper>(),
        config.weights, config.bias, config.axis));
}

CV__DNN_INLINE_NS_END

namespace metal {

Ptr<BackendNode> FullyConnectedOp::create(
    const std::vector<Ptr<BackendWrapper> >& inputs,
    const std::vector<Ptr<BackendWrapper> >& outputs,
    const FullyConnectedConfiguration& config)
{
    return makeMetalBackendNode(makeFullyConnectedOperation(inputs, outputs, config));
}


}  // namespace metal
}}  // namespace cv::dnn

#endif  // HAVE_METAL
