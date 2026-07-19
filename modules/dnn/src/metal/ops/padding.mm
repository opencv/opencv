// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"

#ifdef HAVE_METAL

#include "padding.hpp"
#include "operation.hpp"
#include "../runtime/context.hpp"
#include "../runtime/device.hpp"
#include "../runtime/tensor.hpp"

#include <limits>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

namespace {

constexpr size_t PADDING_MAX_DIMS = MatShape::MAX_DIMS;

class MetalPaddingImpl final : public metal::Operation
{
public:
    MetalPaddingImpl(const Ptr<MetalBackendWrapper>& input,
                     const Ptr<MetalBackendWrapper>& output,
                     const metal::PaddingConfiguration& config)
        : input_(input), output_(output)
    {
        CV_CheckTrue(!input_.empty() && !output_.empty(),
                     "Metal Padding requires Metal tensor wrappers");
        CV_CheckTypeEQ(input_->tensor()->type(), output_->tensor()->type(),
                       "Metal Padding input and output types must match");
        const MatShape& inputShape = input_->tensor()->shape();
        const MatShape& outputShape = output_->tensor()->shape();
        CV_CheckFalse(inputShape.empty(), "Metal Padding input shape must not be empty");
        CV_CheckEQ(inputShape.size(), outputShape.size(),
                   "Metal Padding input and output ranks must match");
        CV_CheckLE(inputShape.size(), PADDING_MAX_DIMS,
                   "Metal Padding input has too many dimensions");
        CV_CheckEQ(config.paddings.size(), inputShape.size(),
                   "Metal Padding requires one padding pair per dimension");
        CV_CheckLE(output_->tensor()->total(),
                   static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
                   "Metal Padding tensor size exceeds uint32 capacity");

        if (config.type != metal::PaddingType::CONSTANT)
        {
            CV_CheckEQ(inputShape.size(), static_cast<size_t>(4),
                       "Metal reflection and edge padding expect 4D NCHW input");
            CV_CheckEQ(config.paddings[0].first + config.paddings[0].second, 0,
                       "Metal reflection and edge padding only support spatial dimensions");
            CV_CheckEQ(config.paddings[1].first + config.paddings[1].second, 0,
                       "Metal reflection and edge padding only support spatial dimensions");
            CV_CheckLE(config.paddings[2].first, inputShape[2], "Metal top padding is too large");
            CV_CheckLE(config.paddings[2].second, inputShape[2], "Metal bottom padding is too large");
            CV_CheckLE(config.paddings[3].first, inputShape[3], "Metal left padding is too large");
            CV_CheckLE(config.paddings[3].second, inputShape[3], "Metal right padding is too large");
        }

        parameters_ = {};
        parameters_.count = static_cast<uint32_t>(output_->tensor()->total());
        parameters_.rank = static_cast<uint32_t>(inputShape.size());
        parameters_.elementSize = static_cast<uint32_t>(
            output_->tensor()->byteSize() / output_->tensor()->total());
        parameters_.mode = config.type == metal::PaddingType::CONSTANT ? 0 :
                           config.type == metal::PaddingType::REFLECT ? 1 : 2;

        size_t stride = 1;
        for (int i = static_cast<int>(inputShape.size()) - 1; i >= 0; --i)
        {
            CV_CheckGT(inputShape[i], 0, "Metal Padding dimensions must be positive");
            CV_CheckEQ(outputShape[i], inputShape[i] + config.paddings[i].first +
                                               config.paddings[i].second,
                       "Metal Padding output shape mismatch");
            CV_CheckLE(stride, static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
                       "Metal Padding input stride exceeds uint32 capacity");
            parameters_.inputShape[i] = static_cast<uint32_t>(inputShape[i]);
            parameters_.outputShape[i] = static_cast<uint32_t>(outputShape[i]);
            parameters_.inputStrides[i] = static_cast<uint32_t>(stride);
            parameters_.paddingBefore[i] = static_cast<uint32_t>(config.paddings[i].first);
            stride *= static_cast<size_t>(inputShape[i]);
        }

        fillHost_.create(1, 1, input_->tensor()->type());
        fillHost_.setTo(Scalar::all(config.value));
        fillTensor_ = metal::Tensor::create(fillHost_);
    }

    void forward(metal::Context& context) CV_OVERRIDE
    {
        if (!pipeline_)
        {
            const char* suffix = nullptr;
            switch (parameters_.elementSize)
            {
                case 1: suffix = "_u8"; break;
                case 2: suffix = "_u16"; break;
                case 4: suffix = "_u32"; break;
                case 8: suffix = "_u64"; break;
                case 16: suffix = "_u128"; break;
                default: suffix = "_bytes"; break;
            }
            pipeline_ = metal::Device::instance()->pipelineState(
                std::string("kernel_padding") + suffix);
        }

        const std::shared_ptr<metal::Buffer>& inputBuffer = input_->tensor()->bufferForRead();
        const std::shared_ptr<metal::Buffer>& outputBuffer = output_->tensor()->bufferForWrite();
        const std::shared_ptr<metal::Buffer>& fillBuffer = fillTensor_->bufferForRead();
        id<MTLComputeCommandEncoder> encoder = context.computeEncoder();
        [encoder setComputePipelineState:pipeline_];
        [encoder setBuffer:context.use(inputBuffer) offset:0 atIndex:0];
        [encoder setBuffer:context.use(outputBuffer) offset:0 atIndex:1];
        [encoder setBuffer:context.use(fillBuffer) offset:0 atIndex:2];
        [encoder setBytes:&parameters_ length:sizeof(parameters_) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(parameters_.count, 1, 1)
            threadsPerThreadgroup:context.threadsPerThreadgroup1D(
                pipeline_, parameters_.count)];
        context.didDispatch();
    }

private:
    struct Parameters
    {
        uint32_t count;
        uint32_t rank;
        uint32_t elementSize;
        uint32_t mode;
        uint32_t inputShape[PADDING_MAX_DIMS];
        uint32_t outputShape[PADDING_MAX_DIMS];
        uint32_t inputStrides[PADDING_MAX_DIMS];
        uint32_t paddingBefore[PADDING_MAX_DIMS];
    } parameters_;

    Ptr<MetalBackendWrapper> input_;
    Ptr<MetalBackendWrapper> output_;
    Mat fillHost_;
    std::shared_ptr<metal::Tensor> fillTensor_;
    id<MTLComputePipelineState> pipeline_ = nil;
};

}  // namespace

CV__DNN_INLINE_NS_END

namespace metal {

Ptr<BackendNode> PaddingOp::create(
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs,
        const PaddingConfiguration& config)
{
    CV_CheckEQ(inputs.size(), static_cast<size_t>(1),
               "Metal Padding requires one input");
    CV_CheckEQ(outputs.size(), static_cast<size_t>(1),
               "Metal Padding requires one output");
    return makeMetalBackendNode(std::unique_ptr<Operation>(
        new MetalPaddingImpl(inputs[0].dynamicCast<MetalBackendWrapper>(),
                             outputs[0].dynamicCast<MetalBackendWrapper>(), config)));
}

}  // namespace metal
}}  // namespace cv::dnn

#endif  // HAVE_METAL
