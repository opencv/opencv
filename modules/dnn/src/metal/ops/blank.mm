// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"

#ifdef HAVE_METAL

#include "blank.hpp"
#include "operation.hpp"
#include "../runtime/context.hpp"
#include "../runtime/device.hpp"
#include "../runtime/tensor.hpp"

#include <limits>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

namespace {

constexpr size_t BLANK_COPY_BYTES_PER_THREAD = 16;

class MetalBlankImpl final : public metal::Operation
{
public:
    MetalBlankImpl(const std::vector<Ptr<BackendWrapper>>& inputs,
                   const std::vector<Ptr<BackendWrapper>>& outputs)
    {
        CV_CheckFalse(inputs.empty(), "Metal Blank requires at least one input");
        CV_CheckFalse(outputs.empty(), "Metal Blank requires at least one output");

        inputs_.reserve(outputs.size());
        outputs_.reserve(outputs.size());
        for (size_t i = 0; i < outputs.size(); ++i)
        {
            Ptr<MetalBackendWrapper> input =
                inputs[i < inputs.size() ? i : 0].dynamicCast<MetalBackendWrapper>();
            Ptr<MetalBackendWrapper> output =
                outputs[i].dynamicCast<MetalBackendWrapper>();
            CV_CheckTrue(!input.empty() && !output.empty(),
                         "Metal Blank requires Metal tensor wrappers");
            CV_CheckTypeEQ(output->tensor()->type(), input->tensor()->type(),
                           "Metal Blank input and output types must match");
            CV_CheckEQ(output->tensor()->byteSize(), input->tensor()->byteSize(),
                       "Metal Blank input and output sizes must match");
            CV_CheckLE(output->tensor()->byteSize(),
                       static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
                       "Metal Blank tensor size exceeds uint32 capacity");
            inputs_.push_back(input);
            outputs_.push_back(output);
        }
    }

    void forward(metal::Context& context) CV_OVERRIDE
    {
        for (size_t i = 0; i < outputs_.size(); ++i)
        {
            const std::shared_ptr<metal::Tensor>& inputTensor = inputs_[i]->tensor();
            const std::shared_ptr<metal::Tensor>& outputTensor = outputs_[i]->tensor();
            const std::shared_ptr<metal::Buffer>& inputBuffer = inputTensor->bufferForRead();
            const std::shared_ptr<metal::Buffer>& outputBuffer = outputTensor->bufferForWrite();

            // The classic allocator normally aliases Blank output storage to its input.
            // Preserve that zero-dispatch path, while still handling separately allocated
            // outputs required by retained blobs or multiple consumers.
            if (inputBuffer == outputBuffer)
                continue;

            if (!pipeline_)
                pipeline_ = metal::Device::instance()->pipelineState("kernel_blank_copy");

            Parameters parameters;
            parameters.byteCount = static_cast<uint32_t>(outputTensor->byteSize());
            const size_t threadCount =
                (parameters.byteCount + BLANK_COPY_BYTES_PER_THREAD - 1) /
                BLANK_COPY_BYTES_PER_THREAD;
            id<MTLComputeCommandEncoder> encoder = context.computeEncoder();
            [encoder setComputePipelineState:pipeline_];
            [encoder setBuffer:context.use(inputBuffer) offset:0 atIndex:0];
            [encoder setBuffer:context.use(outputBuffer) offset:0 atIndex:1];
            [encoder setBytes:&parameters length:sizeof(parameters) atIndex:2];
            [encoder dispatchThreads:MTLSizeMake(threadCount, 1, 1)
                threadsPerThreadgroup:context.threadsPerThreadgroup1D(
                    pipeline_, threadCount)];
            context.didDispatch();
        }
    }

private:
    struct Parameters
    {
        uint32_t byteCount;
    };

    std::vector<Ptr<MetalBackendWrapper>> inputs_;
    std::vector<Ptr<MetalBackendWrapper>> outputs_;
    id<MTLComputePipelineState> pipeline_ = nil;
};

}  // namespace

CV__DNN_INLINE_NS_END

namespace metal {

Ptr<BackendNode> BlankOp::create(const std::vector<Ptr<BackendWrapper>>& inputs,
                                 const std::vector<Ptr<BackendWrapper>>& outputs)
{
    return makeMetalBackendNode(
        std::unique_ptr<Operation>(new MetalBlankImpl(inputs, outputs)));
}

}  // namespace metal
}}  // namespace cv::dnn

#endif  // HAVE_METAL
