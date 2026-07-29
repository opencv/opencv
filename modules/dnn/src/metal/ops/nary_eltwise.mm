// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"

#ifdef HAVE_METAL

#include "nary_eltwise.hpp"
#include "operation.hpp"
#include "../runtime/context.hpp"
#include "../runtime/device.hpp"
#include "../runtime/tensor.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

class MetalNaryCopyImpl final : public metal::Operation
{
public:
    MetalNaryCopyImpl(const Ptr<MetalBackendWrapper>& input,
                      const Ptr<MetalBackendWrapper>& output)
        : input_(input), output_(output)
    {
        CV_CheckTypeEQ(input_->tensor()->type(), CV_32F, "Metal input must be CV_32F");
        CV_CheckTypeEQ(output_->tensor()->type(), CV_32F, "Metal output must be CV_32F");
        count_ = static_cast<uint32_t>(output_->tensor()->total());
    }

    void forward(metal::Context& context) CV_OVERRIDE
    {
        if (!pipeline_)
        {
            pipeline_ = metal::Device::instance()->pipelineState("kernel_nary_copy_f32");
        }

        const std::shared_ptr<metal::Tensor>& inputTensor = input_->tensor();
        const std::shared_ptr<metal::Tensor>& outputTensor = output_->tensor();
        const auto& inputBuffer = inputTensor->bufferForRead();
        const auto& outputBuffer = outputTensor->bufferForWrite();
        id<MTLComputeCommandEncoder> encoder = context.computeEncoder();
        [encoder setComputePipelineState:pipeline_];
        [encoder setBuffer:context.use(inputBuffer) offset:0 atIndex:0];
        [encoder setBuffer:context.use(outputBuffer) offset:0 atIndex:1];
        [encoder setBytes:&count_ length:sizeof(count_) atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(count_, 1, 1)
            threadsPerThreadgroup:context.threadsPerThreadgroup1D(pipeline_, count_)];
        context.didDispatch();
    }

private:
    Ptr<MetalBackendWrapper> input_;
    Ptr<MetalBackendWrapper> output_;
    id<MTLComputePipelineState> pipeline_ = nil;
    uint32_t count_ = 0;
};

static std::unique_ptr<metal::Operation> makeNaryCopyOperation(
        const std::vector<Ptr<BackendWrapper> >& inputs,
        const std::vector<Ptr<BackendWrapper> >& outputs)
{
    return std::unique_ptr<metal::Operation>(new MetalNaryCopyImpl(
        inputs[0].dynamicCast<MetalBackendWrapper>(),
        outputs[0].dynamicCast<MetalBackendWrapper>()));
}

class MetalNaryEltwiseImpl final : public metal::Operation
{
public:
    MetalNaryEltwiseImpl(
        const std::vector<Ptr<MetalBackendWrapper> >& inputs,
                         const Ptr<MetalBackendWrapper>& output,
                         int operation)
        : inputs_(inputs), output_(output), operation_(operation)
    {
        CV_CheckLE(output_->tensor()->shape().size(), static_cast<size_t>(8),
                   "Metal NaryEltwise supports rank up to 8");
        CV_CheckTypeEQ(output_->tensor()->type(), CV_32F, "Metal tensor must be CV_32F");
        for (size_t i = 0; i < inputs_.size(); ++i)
            CV_CheckTypeEQ(inputs_[i]->tensor()->type(), CV_32F, "Metal tensor must be CV_32F");
        const MatShape& outputShape = output_->tensor()->shape();
        parameters_.resize(inputs_.size() - 1);
        for (size_t i = 1; i < inputs_.size(); ++i)
        {
            const MatShape& lhsShape = i == 1 ? inputs_[0]->tensor()->shape() : outputShape;
            fillParameters(lhsShape, inputs_[i]->tensor()->shape(), outputShape,
                           parameters_[i - 1]);
        }
    }

    void forward(metal::Context& context) CV_OVERRIDE
    {
        if (!pipeline_)
        {
            pipeline_ = metal::Device::instance()->pipelineState("kernel_nary_eltwise_f32");
        }

        const std::shared_ptr<metal::Tensor>& outputTensor = output_->tensor();
        for (size_t i = 1; i < inputs_.size(); ++i)
        {
            const std::shared_ptr<metal::Tensor>& lhs =
                i == 1 ? inputs_[0]->tensor() : outputTensor;
            const std::shared_ptr<metal::Tensor>& rhs = inputs_[i]->tensor();
            const Parameters& parameters = parameters_[i - 1];

            const auto& lhsBuffer = lhs->bufferForRead();
            const auto& rhsBuffer = rhs->bufferForRead();
            const auto& outputBuffer = outputTensor->bufferForWrite();
            id<MTLComputeCommandEncoder> encoder = context.computeEncoder();
            [encoder setComputePipelineState:pipeline_];
            [encoder setBuffer:context.use(lhsBuffer) offset:0 atIndex:0];
            [encoder setBuffer:context.use(rhsBuffer) offset:0 atIndex:1];
            [encoder setBuffer:context.use(outputBuffer) offset:0 atIndex:2];
            [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
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
        uint32_t rank;
        uint32_t operation;
        uint32_t lhsShape[8];
        uint32_t rhsShape[8];
        uint32_t outputShape[8];
        uint32_t lhsStride[8];
        uint32_t rhsStride[8];
        uint32_t outputStride[8];
    };

    void fillParameters(const MatShape& lhsShape,
                        const MatShape& rhsShape,
                        const MatShape& outputShape,
                        Parameters& parameters) const
    {
        parameters = Parameters();
        parameters.count = static_cast<uint32_t>(output_->tensor()->total());
        parameters.rank = static_cast<uint32_t>(outputShape.size());
        parameters.operation = static_cast<uint32_t>(operation_);
        const size_t rank = outputShape.size();
        CV_CheckLE(lhsShape.size(), rank, "Metal NaryEltwise lhs rank exceeds output rank");
        CV_CheckLE(rhsShape.size(), rank, "Metal NaryEltwise rhs rank exceeds output rank");

        for (size_t d = 0; d < rank; ++d)
        {
            const int lhsIndex = static_cast<int>(d) - static_cast<int>(rank - lhsShape.size());
            const int rhsIndex = static_cast<int>(d) - static_cast<int>(rank - rhsShape.size());
            parameters.lhsShape[d] = lhsIndex >= 0 ? static_cast<uint32_t>(lhsShape[lhsIndex]) : 1u;
            parameters.rhsShape[d] = rhsIndex >= 0 ? static_cast<uint32_t>(rhsShape[rhsIndex]) : 1u;
            parameters.outputShape[d] = static_cast<uint32_t>(outputShape[d]);
        }

        uint32_t lhsStride = 1, rhsStride = 1, outputStride = 1;
        for (int d = static_cast<int>(rank) - 1; d >= 0; --d)
        {
            parameters.lhsStride[d] = lhsStride;
            parameters.rhsStride[d] = rhsStride;
            parameters.outputStride[d] = outputStride;
            lhsStride *= parameters.lhsShape[d];
            rhsStride *= parameters.rhsShape[d];
            outputStride *= parameters.outputShape[d];
        }
    }
    std::vector<Ptr<MetalBackendWrapper> > inputs_;
    Ptr<MetalBackendWrapper> output_;
    int operation_;
    id<MTLComputePipelineState> pipeline_ = nil;
    std::vector<Parameters> parameters_;
};


static std::unique_ptr<metal::Operation> makeNaryEltwiseOperation(
        const std::vector<Ptr<BackendWrapper> >& inputs,
        const std::vector<Ptr<BackendWrapper> >& outputs,
        metal::NaryOperation operation)
{
    std::vector<Ptr<MetalBackendWrapper>> metalInputs(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i)
        metalInputs[i] = inputs[i].dynamicCast<MetalBackendWrapper>();
    return std::unique_ptr<metal::Operation>(new MetalNaryEltwiseImpl(
        metalInputs,
        outputs[0].dynamicCast<MetalBackendWrapper>(),
        static_cast<int>(operation)));
}

CV__DNN_INLINE_NS_END

namespace metal {

Ptr<BackendNode> NaryEltwiseOp::create(const std::vector<Ptr<BackendWrapper> >& inputs,
                                       const std::vector<Ptr<BackendWrapper> >& outputs,
                                       NaryOperation operation)
{
    if (inputs.size() == 1)
        return makeMetalBackendNode(makeNaryCopyOperation(inputs, outputs));

    return makeMetalBackendNode(makeNaryEltwiseOperation(inputs, outputs, operation));
}

}  // namespace metal
}}  // namespace cv::dnn

#endif  // HAVE_METAL
