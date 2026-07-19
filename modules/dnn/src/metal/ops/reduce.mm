// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"

#ifdef HAVE_METAL

#include "reduce.hpp"
#include "operation.hpp"
#include "../runtime/context.hpp"
#include "../runtime/device.hpp"
#include "../runtime/tensor.hpp"

#include <limits>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

namespace {

constexpr size_t REDUCE_MAX_DIMS = MatShape::MAX_DIMS;

class MetalReduceImpl final : public metal::Operation
{
public:
    MetalReduceImpl(const Ptr<MetalBackendWrapper>& input,
                    const Ptr<MetalBackendWrapper>& output,
                    const metal::ReduceConfiguration& config)
        : input_(input), output_(output), type_(config.type)
    {
        CV_CheckTrue(!input_.empty() && !output_.empty(),
                     "Metal Reduce requires Metal tensor wrappers");
        CV_CheckTypeEQ(input_->tensor()->type(), CV_32F,
                       "Metal Reduce input must be CV_32F");
        CV_CheckTypeEQ(output_->tensor()->type(), CV_32F,
                       "Metal Reduce output must be CV_32F");

        const MatShape& inputShape = input_->tensor()->shape();
        CV_CheckLE(inputShape.size(), REDUCE_MAX_DIMS,
                   "Metal Reduce input has too many dimensions");
        CV_CheckFalse(inputShape.empty(), "Metal Reduce input shape must not be empty");

        std::vector<bool> reduced(inputShape.size(), false);
        if (config.axes.empty() && !config.noopWithEmptyAxes)
        {
            std::fill(reduced.begin(), reduced.end(), true);
        }
        else
        {
            for (int axis : config.axes)
            {
                if (axis < 0)
                    axis += static_cast<int>(inputShape.size());
                CV_CheckGE(axis, 0, "Metal Reduce axis is out of range");
                CV_CheckLT(axis, static_cast<int>(inputShape.size()),
                           "Metal Reduce axis is out of range");
                CV_CheckFalse(reduced[axis], "Metal Reduce axes must be unique");
                reduced[axis] = true;
            }
        }

        std::vector<size_t> inputStrides(inputShape.size(), 1);
        for (int i = static_cast<int>(inputShape.size()) - 2; i >= 0; --i)
        {
            inputStrides[i] = inputStrides[i + 1] *
                              static_cast<size_t>(inputShape[i + 1]);
        }

        size_t outputCount = 1;
        size_t reductionCount = 1;
        bool foundReducedDimension = false;
        for (size_t i = 0; i < inputShape.size(); ++i)
        {
            CV_CheckGT(inputShape[i], 0, "Metal Reduce dimensions must be positive");
            CV_CheckLE(inputStrides[i],
                       static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
                       "Metal Reduce input stride exceeds uint32 capacity");
            if (reduced[i])
            {
                foundReducedDimension = true;
                const uint32_t dim = parameters_.reductionDims++;
                parameters_.reductionShape[dim] = static_cast<uint32_t>(inputShape[i]);
                parameters_.reductionStrides[dim] = static_cast<uint32_t>(inputStrides[i]);
                reductionCount *= static_cast<size_t>(inputShape[i]);
            }
            else
            {
                if (foundReducedDimension)
                    parameters_.contiguousReduction = 0;
                const uint32_t dim = parameters_.outputDims++;
                parameters_.outputShape[dim] = static_cast<uint32_t>(inputShape[i]);
                parameters_.outputStrides[dim] = static_cast<uint32_t>(inputStrides[i]);
                outputCount *= static_cast<size_t>(inputShape[i]);
            }
        }

        CV_CheckLE(outputCount, static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
                   "Metal Reduce output size exceeds uint32 capacity");
        CV_CheckLE(reductionCount,
                   static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
                   "Metal Reduce reduction size exceeds uint32 capacity");
        CV_CheckEQ(output_->tensor()->total(), outputCount,
                   "Metal Reduce output shape is invalid");
        CV_CheckEQ(input_->tensor()->total(), outputCount * reductionCount,
                   "Metal Reduce input shape is invalid");
        parameters_.outputCount = static_cast<uint32_t>(outputCount);
        parameters_.reductionCount = static_cast<uint32_t>(reductionCount);
    }

    void forward(metal::Context& context) CV_OVERRIDE
    {
        if (!pipeline_)
            pipeline_ = metal::Device::instance()->pipelineState(pipelineName());

        const auto& inputBuffer = input_->tensor()->bufferForRead();
        const auto& outputBuffer = output_->tensor()->bufferForWrite();

        const size_t pipelineLimit = std::min(
            static_cast<size_t>(256),
            static_cast<size_t>([pipeline_ maxTotalThreadsPerThreadgroup]));
        size_t maximumPowerOfTwo = 1;
        while (maximumPowerOfTwo <= pipelineLimit / 2)
            maximumPowerOfTwo *= 2;
        size_t threadsPerThreadgroup = 1;
        while (threadsPerThreadgroup < parameters_.reductionCount &&
               threadsPerThreadgroup < maximumPowerOfTwo)
        {
            threadsPerThreadgroup *= 2;
        }

        id<MTLComputeCommandEncoder> encoder = context.computeEncoder();
        [encoder setComputePipelineState:pipeline_];
        [encoder setBuffer:context.use(inputBuffer) offset:0 atIndex:0];
        [encoder setBuffer:context.use(outputBuffer) offset:0 atIndex:1];
        [encoder setBytes:&parameters_ length:sizeof(parameters_) atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(parameters_.outputCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadsPerThreadgroup, 1, 1)];
        context.didDispatch();
    }

private:
    const char* pipelineName() const
    {
        switch (type_)
        {
            case metal::ReduceType::MAX:         return "kernel_reduce_max_f32";
            case metal::ReduceType::MIN:         return "kernel_reduce_min_f32";
            case metal::ReduceType::MEAN:        return "kernel_reduce_mean_f32";
            case metal::ReduceType::SUM:         return "kernel_reduce_sum_f32";
            case metal::ReduceType::L1:          return "kernel_reduce_l1_f32";
            case metal::ReduceType::L2:          return "kernel_reduce_l2_f32";
            case metal::ReduceType::PROD:        return "kernel_reduce_prod_f32";
            case metal::ReduceType::SUM_SQUARE:  return "kernel_reduce_sum_square_f32";
            case metal::ReduceType::LOG_SUM:     return "kernel_reduce_log_sum_f32";
            case metal::ReduceType::LOG_SUM_EXP: return "kernel_reduce_log_sum_exp_f32";
            default: CV_Error(Error::StsBadArg, "Unsupported Metal Reduce operation");
        }
    }

    struct Parameters
    {
        uint32_t outputCount = 0;
        uint32_t reductionCount = 0;
        uint32_t outputDims = 0;
        uint32_t reductionDims = 0;
        uint32_t contiguousReduction = 1;
        uint32_t outputShape[REDUCE_MAX_DIMS] = {};
        uint32_t outputStrides[REDUCE_MAX_DIMS] = {};
        uint32_t reductionShape[REDUCE_MAX_DIMS] = {};
        uint32_t reductionStrides[REDUCE_MAX_DIMS] = {};
    } parameters_;

    Ptr<MetalBackendWrapper> input_;
    Ptr<MetalBackendWrapper> output_;
    metal::ReduceType type_;
    id<MTLComputePipelineState> pipeline_ = nil;
};

static std::unique_ptr<metal::Operation> makeReduceOperation(
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs,
        const metal::ReduceConfiguration& config)
{
    CV_CheckEQ(inputs.size(), static_cast<size_t>(1),
               "Metal Reduce requires exactly one input");
    CV_CheckEQ(outputs.size(), static_cast<size_t>(1),
               "Metal Reduce requires exactly one output");
    return std::unique_ptr<metal::Operation>(new MetalReduceImpl(
        inputs[0].dynamicCast<MetalBackendWrapper>(),
        outputs[0].dynamicCast<MetalBackendWrapper>(), config));
}

}  // namespace

CV__DNN_INLINE_NS_END

namespace metal {

Ptr<BackendNode> ReduceOp::create(const std::vector<Ptr<BackendWrapper>>& inputs,
                                  const std::vector<Ptr<BackendWrapper>>& outputs,
                                  const ReduceConfiguration& config)
{
    return makeMetalBackendNode(makeReduceOperation(inputs, outputs, config));
}

}  // namespace metal
}}  // namespace cv::dnn

#endif  // HAVE_METAL
