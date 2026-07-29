// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"

#ifdef HAVE_METAL

#include "depth_space_ops.hpp"
#include "operation.hpp"
#include "../runtime/context.hpp"
#include "../runtime/device.hpp"
#include "../runtime/tensor.hpp"

#include <limits>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

namespace {

class MetalDepthSpaceOpsImpl final : public metal::Operation
{
public:
    MetalDepthSpaceOpsImpl(const Ptr<MetalBackendWrapper>& input,
                           const Ptr<MetalBackendWrapper>& output,
                           metal::DepthSpaceOperation operation,
                           int blockSize)
        : input_(input), output_(output), operation_(operation)
    {
        CV_CheckTrue(!input_.empty() && !output_.empty(),
                     "Metal depth-space operation requires Metal tensor wrappers");
        CV_CheckGT(blockSize, 0, "Metal depth-space block size must be positive");
        const MatShape& inputShape = input_->tensor()->shape();
        const MatShape& outputShape = output_->tensor()->shape();
        CV_CheckEQ(inputShape.size(), static_cast<size_t>(4),
                   "Metal depth-space operation expects 4D NCHW input");
        CV_CheckEQ(outputShape.size(), static_cast<size_t>(4),
                   "Metal depth-space operation expects 4D NCHW output");
        CV_CheckTypeEQ(input_->tensor()->type(), output_->tensor()->type(),
                       "Metal depth-space input and output types must match");
        CV_CheckEQ(input_->tensor()->total(), output_->tensor()->total(),
                   "Metal depth-space input and output element counts must match");
        CV_CheckLE(output_->tensor()->total(),
                   static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
                   "Metal depth-space tensor size exceeds uint32 capacity");

        const int blockArea = blockSize * blockSize;
        if (operation_ == metal::DepthSpaceOperation::SPACE_TO_DEPTH)
        {
            CV_CheckEQ(inputShape[2] % blockSize, 0,
                       "Metal SpaceToDepth input height must be divisible by block size");
            CV_CheckEQ(inputShape[3] % blockSize, 0,
                       "Metal SpaceToDepth input width must be divisible by block size");
            CV_CheckEQ(outputShape[1], inputShape[1] * blockArea,
                       "Metal SpaceToDepth output channels mismatch");
            CV_CheckEQ(outputShape[2], inputShape[2] / blockSize,
                       "Metal SpaceToDepth output height mismatch");
            CV_CheckEQ(outputShape[3], inputShape[3] / blockSize,
                       "Metal SpaceToDepth output width mismatch");
        }
        else
        {
            CV_CheckEQ(inputShape[1] % blockArea, 0,
                       "Metal DepthToSpace input channels must be divisible by block area");
            CV_CheckEQ(outputShape[1], inputShape[1] / blockArea,
                       "Metal DepthToSpace output channels mismatch");
            CV_CheckEQ(outputShape[2], inputShape[2] * blockSize,
                       "Metal DepthToSpace output height mismatch");
            CV_CheckEQ(outputShape[3], inputShape[3] * blockSize,
                       "Metal DepthToSpace output width mismatch");
        }

        parameters_.count = static_cast<uint32_t>(output_->tensor()->total());
        parameters_.inputChannels = static_cast<uint32_t>(inputShape[1]);
        parameters_.inputHeight = static_cast<uint32_t>(inputShape[2]);
        parameters_.inputWidth = static_cast<uint32_t>(inputShape[3]);
        parameters_.outputChannels = static_cast<uint32_t>(outputShape[1]);
        parameters_.outputHeight = static_cast<uint32_t>(outputShape[2]);
        parameters_.outputWidth = static_cast<uint32_t>(outputShape[3]);
        parameters_.blockSize = static_cast<uint32_t>(blockSize);
        parameters_.elementSize = static_cast<uint32_t>(
            output_->tensor()->byteSize() / output_->tensor()->total());
        parameters_.mode = operation_ == metal::DepthSpaceOperation::DEPTH_TO_SPACE_CRD ? 1 : 0;
    }

    void forward(metal::Context& context) CV_OVERRIDE
    {
        if (!pipeline_)
        {
            const bool depthToSpace =
                operation_ != metal::DepthSpaceOperation::SPACE_TO_DEPTH;
            const char* prefix = depthToSpace ? "kernel_depth_to_space" :
                                                "kernel_space_to_depth";
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
                std::string(prefix) + suffix);
        }

        const std::shared_ptr<metal::Buffer>& inputBuffer =
            input_->tensor()->bufferForRead();
        const std::shared_ptr<metal::Buffer>& outputBuffer =
            output_->tensor()->bufferForWrite();
        id<MTLComputeCommandEncoder> encoder = context.computeEncoder();
        [encoder setComputePipelineState:pipeline_];
        [encoder setBuffer:context.use(inputBuffer) offset:0 atIndex:0];
        [encoder setBuffer:context.use(outputBuffer) offset:0 atIndex:1];
        [encoder setBytes:&parameters_ length:sizeof(parameters_) atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(parameters_.count, 1, 1)
            threadsPerThreadgroup:context.threadsPerThreadgroup1D(
                pipeline_, parameters_.count)];
        context.didDispatch();
    }

private:
    struct Parameters
    {
        uint32_t count;
        uint32_t inputChannels;
        uint32_t inputHeight;
        uint32_t inputWidth;
        uint32_t outputChannels;
        uint32_t outputHeight;
        uint32_t outputWidth;
        uint32_t blockSize;
        uint32_t elementSize;
        uint32_t mode;
    } parameters_;

    Ptr<MetalBackendWrapper> input_;
    Ptr<MetalBackendWrapper> output_;
    metal::DepthSpaceOperation operation_;
    id<MTLComputePipelineState> pipeline_ = nil;
};

}  // namespace

CV__DNN_INLINE_NS_END

namespace metal {

Ptr<BackendNode> DepthSpaceOpsOp::create(
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs,
        DepthSpaceOperation operation,
        int blockSize)
{
    CV_CheckEQ(inputs.size(), static_cast<size_t>(1),
               "Metal depth-space operation requires one input");
    CV_CheckEQ(outputs.size(), static_cast<size_t>(1),
               "Metal depth-space operation requires one output");
    return makeMetalBackendNode(std::unique_ptr<Operation>(
        new MetalDepthSpaceOpsImpl(
            inputs[0].dynamicCast<MetalBackendWrapper>(),
            outputs[0].dynamicCast<MetalBackendWrapper>(), operation, blockSize)));
}

}  // namespace metal
}}  // namespace cv::dnn

#endif  // HAVE_METAL
