// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"

#ifdef HAVE_METAL

#include "lrn.hpp"
#include "operation.hpp"
#include "../runtime/context.hpp"
#include "../runtime/device.hpp"
#include "../runtime/tensor.hpp"

#include <limits>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

namespace {

class MetalLRNImpl final : public metal::Operation
{
public:
    MetalLRNImpl(const std::vector<Ptr<MetalBackendWrapper>>& inputs,
                 const std::vector<Ptr<MetalBackendWrapper>>& outputs,
                 const metal::LRNConfiguration& config)
        : inputs_(inputs), outputs_(outputs)
    {
        CV_CheckFalse(inputs_.empty(), "Metal LRN requires at least one input");
        CV_CheckEQ(inputs_.size(), outputs_.size(),
                   "Metal LRN input and output counts must match");
        CV_CheckGT(config.localSize, 0, "Metal LRN local size must be positive");
        CV_CheckEQ(config.localSize % 2, 1, "Metal LRN local size must be odd");

        parameters_.reserve(inputs_.size());
        for (size_t i = 0; i < inputs_.size(); ++i)
        {
            CV_CheckTrue(!inputs_[i].empty() && !outputs_[i].empty(),
                         "Metal LRN requires Metal tensor wrappers");
            CV_CheckTypeEQ(inputs_[i]->tensor()->type(), CV_32F,
                           "Metal LRN input must be CV_32F");
            CV_CheckTypeEQ(outputs_[i]->tensor()->type(), CV_32F,
                           "Metal LRN output must be CV_32F");
            const MatShape& inputShape = inputs_[i]->tensor()->shape();
            CV_CheckEQ(inputShape.size(), static_cast<size_t>(4),
                       "Metal LRN expects 4D NCHW input");
            CV_CheckTrue(inputShape == outputs_[i]->tensor()->shape(),
                         "Metal LRN input and output shapes must match");
            CV_CheckLE(inputs_[i]->tensor()->total(),
                       static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
                       "Metal LRN tensor size exceeds uint32 capacity");

            Parameters params = {};
            params.count = static_cast<uint32_t>(inputs_[i]->tensor()->total());
            params.channels = static_cast<uint32_t>(inputShape[1]);
            params.height = static_cast<uint32_t>(inputShape[2]);
            params.width = static_cast<uint32_t>(inputShape[3]);
            params.localSize = static_cast<uint32_t>(config.localSize);
            params.alpha = config.alpha /
                (config.normBySize ? static_cast<float>(config.type == metal::LRNType::ACROSS_CHANNELS ?
                                                        config.localSize :
                                                        config.localSize * config.localSize) : 1.0f);
            params.beta = config.beta;
            params.bias = config.bias;
            params.type = config.type == metal::LRNType::ACROSS_CHANNELS ? 0 : 1;
            parameters_.push_back(params);
        }
    }

    void forward(metal::Context& context) CV_OVERRIDE
    {
        if (!pipeline_)
            pipeline_ = metal::Device::instance()->pipelineState("kernel_lrn_f32");

        for (size_t i = 0; i < inputs_.size(); ++i)
        {
            const std::shared_ptr<metal::Buffer>& inputBuffer =
                inputs_[i]->tensor()->bufferForRead();
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
        uint32_t channels;
        uint32_t height;
        uint32_t width;
        uint32_t localSize;
        float alpha;
        float beta;
        float bias;
        uint32_t type;
    };

    std::vector<Ptr<MetalBackendWrapper>> inputs_;
    std::vector<Ptr<MetalBackendWrapper>> outputs_;
    std::vector<Parameters> parameters_;
    id<MTLComputePipelineState> pipeline_ = nil;
};

}  // namespace

CV__DNN_INLINE_NS_END

namespace metal {

Ptr<BackendNode> LRNOp::create(
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs,
        const LRNConfiguration& config)
{
    std::vector<Ptr<MetalBackendWrapper>> metalInputs(inputs.size());
    std::vector<Ptr<MetalBackendWrapper>> metalOutputs(outputs.size());
    for (size_t i = 0; i < inputs.size(); ++i)
        metalInputs[i] = inputs[i].dynamicCast<MetalBackendWrapper>();
    for (size_t i = 0; i < outputs.size(); ++i)
        metalOutputs[i] = outputs[i].dynamicCast<MetalBackendWrapper>();
    return makeMetalBackendNode(std::unique_ptr<Operation>(
        new MetalLRNImpl(metalInputs, metalOutputs, config)));
}

}  // namespace metal
}}  // namespace cv::dnn

#endif  // HAVE_METAL
