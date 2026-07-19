// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"

#ifdef HAVE_METAL

#include "slice.hpp"
#include "operation.hpp"
#include "../runtime/context.hpp"
#include "../runtime/device.hpp"
#include "../runtime/tensor.hpp"

#include <limits>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

namespace {

constexpr size_t SLICE_MAX_DIMS = MatShape::MAX_DIMS;

class MetalSliceImpl final : public metal::Operation
{
public:
    MetalSliceImpl(const Ptr<MetalBackendWrapper>& input,
                   const std::vector<Ptr<MetalBackendWrapper>>& outputs,
                   const metal::SliceConfiguration& config)
        : input_(input), outputs_(outputs)
    {
        CV_CheckTrue(!input_.empty(), "Metal Slice requires a Metal input wrapper");
        CV_CheckFalse(outputs_.empty(), "Metal Slice requires at least one output");
        CV_CheckEQ(config.ranges.size(), outputs_.size(),
                   "Metal Slice requires one range set per output");

        const MatShape& inputShape = input_->tensor()->shape();
        CV_CheckFalse(inputShape.empty(), "Metal Slice input shape must not be empty");
        CV_CheckLE(inputShape.size(), SLICE_MAX_DIMS,
                   "Metal Slice input has too many dimensions");

        std::vector<bool> flipped(inputShape.size(), false);
        for (int dim : config.flippedDimensions)
        {
            CV_CheckGE(dim, 0, "Metal Slice flipped dimension is out of range");
            CV_CheckLT(dim, static_cast<int>(inputShape.size()),
                       "Metal Slice flipped dimension is out of range");
            flipped[dim] = true;
        }

        std::vector<size_t> inputStrides(inputShape.size(), 1);
        for (int i = static_cast<int>(inputShape.size()) - 2; i >= 0; --i)
            inputStrides[i] = inputStrides[i + 1] * static_cast<size_t>(inputShape[i + 1]);

        parameters_.reserve(outputs_.size());
        for (size_t outputIndex = 0; outputIndex < outputs_.size(); ++outputIndex)
        {
            CV_CheckTrue(!outputs_[outputIndex].empty(),
                         "Metal Slice requires Metal output wrappers");
            CV_CheckTypeEQ(input_->tensor()->type(), outputs_[outputIndex]->tensor()->type(),
                           "Metal Slice input and output types must match");
            const MatShape& outputShape = outputs_[outputIndex]->tensor()->shape();
            CV_CheckEQ(inputShape.size(), outputShape.size(),
                       "Metal Slice input and output ranks must match");
            CV_CheckEQ(config.ranges[outputIndex].size(), inputShape.size(),
                       "Metal Slice requires one range per dimension");
            CV_CheckLE(outputs_[outputIndex]->tensor()->total(),
                       static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
                       "Metal Slice tensor size exceeds uint32 capacity");

            Parameters params = {};
            params.count = static_cast<uint32_t>(outputs_[outputIndex]->tensor()->total());
            params.rank = static_cast<uint32_t>(inputShape.size());
            params.elementSize = static_cast<uint32_t>(
                outputs_[outputIndex]->tensor()->byteSize() /
                outputs_[outputIndex]->tensor()->total());
            for (size_t dim = 0; dim < inputShape.size(); ++dim)
            {
                int step = 1;
                if (outputIndex < config.steps.size() && dim < config.steps[outputIndex].size())
                    step = config.steps[outputIndex][dim];
                else if (config.steps.size() == 1 && dim < config.steps[0].size())
                    step = config.steps[0][dim];
                CV_CheckGT(step, 0, "Metal Slice steps must be positive after finalization");
                CV_CheckLE(inputStrides[dim],
                           static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
                           "Metal Slice input stride exceeds uint32 capacity");
                const Range& range = config.ranges[outputIndex][dim];
                CV_CheckGE(range.start, 0, "Metal Slice range start is out of bounds");
                CV_CheckLE(range.end, inputShape[dim], "Metal Slice range end is out of bounds");
                CV_CheckEQ(outputShape[dim], (range.size() + step - 1) / step,
                           "Metal Slice output shape mismatch");
                params.outputShape[dim] = static_cast<uint32_t>(outputShape[dim]);
                params.inputStrides[dim] = static_cast<uint32_t>(inputStrides[dim]);
                params.starts[dim] = static_cast<uint32_t>(range.start);
                params.steps[dim] = static_cast<uint32_t>(step);
                params.flipped[dim] = flipped[dim] ? 1 : 0;
            }
            parameters_.push_back(params);
        }
    }

    void forward(metal::Context& context) CV_OVERRIDE
    {
        if (!pipeline_)
        {
            const char* suffix = nullptr;
            switch (parameters_[0].elementSize)
            {
                case 1: suffix = "_u8"; break;
                case 2: suffix = "_u16"; break;
                case 4: suffix = "_u32"; break;
                case 8: suffix = "_u64"; break;
                case 16: suffix = "_u128"; break;
                default: suffix = "_bytes"; break;
            }
            pipeline_ = metal::Device::instance()->pipelineState(
                std::string("kernel_slice") + suffix);
        }

        const std::shared_ptr<metal::Buffer>& inputBuffer = input_->tensor()->bufferForRead();
        for (size_t i = 0; i < outputs_.size(); ++i)
        {
            const std::shared_ptr<metal::Buffer>& outputBuffer =
                outputs_[i]->tensor()->bufferForWrite();
            id<MTLComputeCommandEncoder> encoder = context.computeEncoder();
            [encoder setComputePipelineState:pipeline_];
            [encoder setBuffer:context.use(inputBuffer) offset:0 atIndex:0];
            [encoder setBuffer:context.use(outputBuffer) offset:0 atIndex:1];
            [encoder setBytes:&parameters_[i] length:sizeof(parameters_[i]) atIndex:2];
            [encoder dispatchThreads:MTLSizeMake(parameters_[i].count, 1, 1)
                threadsPerThreadgroup:context.threadsPerThreadgroup1D(
                    pipeline_, parameters_[i].count)];
            context.didDispatch();
        }
    }

private:
    struct Parameters
    {
        uint32_t count;
        uint32_t rank;
        uint32_t elementSize;
        uint32_t reserved;
        uint32_t outputShape[SLICE_MAX_DIMS];
        uint32_t inputStrides[SLICE_MAX_DIMS];
        uint32_t starts[SLICE_MAX_DIMS];
        uint32_t steps[SLICE_MAX_DIMS];
        uint32_t flipped[SLICE_MAX_DIMS];
    };

    Ptr<MetalBackendWrapper> input_;
    std::vector<Ptr<MetalBackendWrapper>> outputs_;
    std::vector<Parameters> parameters_;
    id<MTLComputePipelineState> pipeline_ = nil;
};

}  // namespace

CV__DNN_INLINE_NS_END

namespace metal {

Ptr<BackendNode> SliceOp::create(
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs,
        const SliceConfiguration& config)
{
    CV_CheckFalse(inputs.empty(), "Metal Slice requires at least one input");
    std::vector<Ptr<MetalBackendWrapper>> metalOutputs(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i)
        metalOutputs[i] = outputs[i].dynamicCast<MetalBackendWrapper>();
    return makeMetalBackendNode(std::unique_ptr<Operation>(
        new MetalSliceImpl(inputs[0].dynamicCast<MetalBackendWrapper>(),
                           metalOutputs, config)));
}

}  // namespace metal
}}  // namespace cv::dnn

#endif  // HAVE_METAL
