// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"

#ifdef HAVE_METAL

#include "pooling.hpp"
#include "operation.hpp"
#include "../runtime/context.hpp"
#include "../runtime/device.hpp"
#include "../runtime/tensor.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

namespace {

class MetalAvgPool2DImpl final : public metal::Operation
{
public:
    MetalAvgPool2DImpl(const Ptr<MetalBackendWrapper>& input,
                       const Ptr<MetalBackendWrapper>& output,
                       size_t kernelHeight,
                       size_t kernelWidth,
                       size_t strideHeight,
                       size_t strideWidth,
                       size_t padTop,
                       size_t padBottom,
                       size_t padLeft,
                       size_t padRight,
                       bool includePadding)
        : input_(input), output_(output)
    {
        const MatShape& inputShape = input_->tensor()->shape();
        const MatShape& outputShape = output_->tensor()->shape();
        CV_CheckEQ(inputShape.size(), static_cast<size_t>(4),
                   "Metal average pooling expects a 4D NCHW input");
        CV_CheckEQ(outputShape.size(), static_cast<size_t>(4),
                   "Metal average pooling expects a 4D NCHW output");
        CV_CheckTypeEQ(input_->tensor()->type(), CV_32F, "Metal tensor must be CV_32F");
        CV_CheckTypeEQ(output_->tensor()->type(), CV_32F, "Metal tensor must be CV_32F");
        outputCount_ = static_cast<uint32_t>(output_->tensor()->total());

        parameters_.inputHeight = static_cast<uint32_t>(inputShape[2]);
        parameters_.inputWidth = static_cast<uint32_t>(inputShape[3]);
        parameters_.outputHeight = static_cast<uint32_t>(outputShape[2]);
        parameters_.outputWidth = static_cast<uint32_t>(outputShape[3]);
        parameters_.kernelHeight = static_cast<uint32_t>(kernelHeight);
        parameters_.kernelWidth = static_cast<uint32_t>(kernelWidth);
        parameters_.strideHeight = static_cast<uint32_t>(strideHeight);
        parameters_.strideWidth = static_cast<uint32_t>(strideWidth);
        parameters_.padTop = static_cast<uint32_t>(padTop);
        parameters_.padBottom = static_cast<uint32_t>(padBottom);
        parameters_.padLeft = static_cast<uint32_t>(padLeft);
        parameters_.padRight = static_cast<uint32_t>(padRight);
        parameters_.includePadding = includePadding ? 1u : 0u;

        CV_CheckLE(static_cast<size_t>(parameters_.inputHeight) + parameters_.padBottom,
                   static_cast<size_t>(INT32_MAX),
                   "Metal average pooling bottom padded input exceeds the supported range");
        CV_CheckLE(static_cast<size_t>(parameters_.inputWidth) + parameters_.padRight,
                   static_cast<size_t>(INT32_MAX),
                   "Metal average pooling right padded input exceeds the supported range");
    }

    void forward(metal::Context& context) CV_OVERRIDE
    {
        if (!pipeline_)
        {
            pipeline_ = metal::Device::instance()->pipelineState("kernel_avg_pool2d_f32");
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
        [encoder dispatchThreads:MTLSizeMake(outputCount_, 1, 1)
            threadsPerThreadgroup:context.threadsPerThreadgroup1D(pipeline_, outputCount_)];
        context.didDispatch();
    }

private:
    struct Parameters
    {
        uint32_t inputHeight;
        uint32_t inputWidth;
        uint32_t outputHeight;
        uint32_t outputWidth;
        uint32_t kernelHeight;
        uint32_t kernelWidth;
        uint32_t strideHeight;
        uint32_t strideWidth;
        uint32_t padTop;
        uint32_t padBottom;
        uint32_t padLeft;
        uint32_t padRight;
        uint32_t includePadding;
    };

    Ptr<MetalBackendWrapper> input_;
    Ptr<MetalBackendWrapper> output_;
    id<MTLComputePipelineState> pipeline_ = nil;
    Parameters parameters_;
    uint32_t outputCount_ = 0;
};

static std::unique_ptr<metal::Operation> makeAvgPool2DOperation(
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs,
        const metal::AvgPool2DConfiguration& config)
{
    return std::unique_ptr<metal::Operation>(new MetalAvgPool2DImpl(
        inputs[0].dynamicCast<MetalBackendWrapper>(),
        outputs[0].dynamicCast<MetalBackendWrapper>(),
        config.kernelHeight, config.kernelWidth,
        config.strideHeight, config.strideWidth,
        config.padTop, config.padBottom, config.padLeft, config.padRight,
        config.includePadding));
}

class MetalMaxPoolingImpl final : public metal::Operation
{
public:
    MetalMaxPoolingImpl(const Ptr<MetalBackendWrapper>& input,
                        const Ptr<MetalBackendWrapper>& output,
                        const Ptr<MetalBackendWrapper>& indices,
                        size_t kernelHeight,
                        size_t kernelWidth,
                        size_t strideHeight,
                        size_t strideWidth,
                        size_t padTop,
                        size_t padLeft)
        : input_(input), output_(output), indices_(indices)
    {
        const MatShape& inputShape = input_->tensor()->shape();
        const MatShape& outputShape = output_->tensor()->shape();
        CV_CheckEQ(inputShape.size(), static_cast<size_t>(4),
                   "Metal max pooling expects 4D NCHW input");
        CV_CheckEQ(outputShape.size(), static_cast<size_t>(4),
                   "Metal max pooling expects 4D NCHW output");
        CV_CheckTypeEQ(input_->tensor()->type(), CV_32F, "Metal tensor must be CV_32F");
        CV_CheckTypeEQ(output_->tensor()->type(), CV_32F, "Metal tensor must be CV_32F");
        if (indices_.empty())
        {
            indicesHost_.create(static_cast<int>(outputShape.size()), outputShape.data(), CV_64S);
            indices_ = Ptr<MetalBackendWrapper>(new MetalBackendWrapper(indicesHost_));
        }
        else
        {
            CV_CheckTypeEQ(indices_->tensor()->type(), CV_64S,
                           "Metal max pooling indices must be CV_64S");
        }

        parameters_.inputHeight = static_cast<uint32_t>(inputShape[2]);
        parameters_.inputWidth = static_cast<uint32_t>(inputShape[3]);
        parameters_.outputHeight = static_cast<uint32_t>(outputShape[2]);
        parameters_.outputWidth = static_cast<uint32_t>(outputShape[3]);
        parameters_.kernelHeight = static_cast<uint32_t>(kernelHeight);
        parameters_.kernelWidth = static_cast<uint32_t>(kernelWidth);
        parameters_.strideHeight = static_cast<uint32_t>(strideHeight);
        parameters_.strideWidth = static_cast<uint32_t>(strideWidth);
        parameters_.padTop = static_cast<uint32_t>(padTop);
        parameters_.padLeft = static_cast<uint32_t>(padLeft);
    }

    void forward(metal::Context& context) CV_OVERRIDE
    {
        if (!pipeline_)
        {
            pipeline_ = metal::Device::instance()->pipelineState("kernel_max_pool2d_f32");
        }

        const std::shared_ptr<metal::Tensor>& inputTensor = input_->tensor();
        const std::shared_ptr<metal::Tensor>& outputTensor = output_->tensor();
        const std::shared_ptr<metal::Tensor>& indicesTensor = indices_->tensor();
        const size_t count = outputTensor->total();
        const auto& inputBuffer = inputTensor->bufferForRead();
        const auto& outputBuffer = outputTensor->bufferForWrite();
        const auto& indicesBuffer = indicesTensor->bufferForWrite();
        id<MTLComputeCommandEncoder> encoder = context.computeEncoder();
        [encoder setComputePipelineState:pipeline_];
        [encoder setBuffer:context.use(inputBuffer) offset:0 atIndex:0];
        [encoder setBuffer:context.use(outputBuffer) offset:0 atIndex:1];
        [encoder setBuffer:context.use(indicesBuffer) offset:0 atIndex:2];
        [encoder setBytes:&parameters_ length:sizeof(parameters_) atIndex:3];
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
        uint32_t kernelHeight;
        uint32_t kernelWidth;
        uint32_t strideHeight;
        uint32_t strideWidth;
        uint32_t padTop;
        uint32_t padLeft;
    } parameters_;
    Ptr<MetalBackendWrapper> input_;
    Ptr<MetalBackendWrapper> output_;
    Ptr<MetalBackendWrapper> indices_;
    Mat indicesHost_;
    id<MTLComputePipelineState> pipeline_ = nil;
};

static std::unique_ptr<metal::Operation> makeMaxPoolingOperation(
        const std::vector<Ptr<BackendWrapper> >& inputs,
        const std::vector<Ptr<BackendWrapper> >& outputs,
        const metal::MaxPoolingConfiguration& config)
{
    Ptr<MetalBackendWrapper> input = inputs[0].dynamicCast<MetalBackendWrapper>();
    Ptr<MetalBackendWrapper> output = outputs[0].dynamicCast<MetalBackendWrapper>();
    Ptr<MetalBackendWrapper> indices;
    if (outputs.size() == 2)
    {
        indices = outputs[1].dynamicCast<MetalBackendWrapper>();
    }
    return std::unique_ptr<metal::Operation>(new MetalMaxPoolingImpl(
        input, output, indices, config.kernelHeight, config.kernelWidth,
        config.strideHeight, config.strideWidth, config.padTop, config.padLeft));
}

}  // namespace

CV__DNN_INLINE_NS_END

namespace metal {

Ptr<BackendNode> AvgPool2DOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                     const std::vector<Ptr<BackendWrapper> >& outputs,
                                     const AvgPool2DConfiguration& config)
{
    return makeMetalBackendNode(makeAvgPool2DOperation(inputs, outputs, config));
}

Ptr<BackendNode> MaxPoolingOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                      const std::vector<Ptr<BackendWrapper> >& outputs,
                                      const MaxPoolingConfiguration& config)
{
    return makeMetalBackendNode(makeMaxPoolingOperation(inputs, outputs, config));
}

}  // namespace metal
}}  // namespace cv::dnn

#endif  // HAVE_METAL
