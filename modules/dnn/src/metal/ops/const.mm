// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"

#ifdef HAVE_METAL

#include "const.hpp"
#include "operation.hpp"
#include "../runtime/context.hpp"
#include "../runtime/device.hpp"
#include "../runtime/tensor.hpp"

#include <limits>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

namespace {

constexpr size_t CONST_COPY_BYTES_PER_THREAD = 16;

class MetalConstImpl final : public metal::Operation
{
public:
    MetalConstImpl(const std::vector<Ptr<BackendWrapper>>& inputs,
                   const std::vector<Ptr<BackendWrapper>>& outputs,
                   const Mat& data)
    {
        CV_CheckEQ(inputs.size(), static_cast<size_t>(0),
                   "Metal Const does not accept inputs");
        CV_CheckEQ(outputs.size(), static_cast<size_t>(1),
                   "Metal Const requires one output");
        CV_CheckFalse(data.empty(), "Metal Const data must not be empty");
        CV_CheckTrue(data.isContinuous(), "Metal Const data must be continuous");

        output_ = outputs[0].dynamicCast<MetalBackendWrapper>();
        CV_CheckTrue(!output_.empty(), "Metal Const requires a Metal tensor wrapper");
        CV_CheckTypeEQ(output_->tensor()->type(), data.type(),
                       "Metal Const data and output types must match");
        CV_CheckEQ(output_->tensor()->byteSize(), data.total() * data.elemSize(),
                   "Metal Const data and output sizes must match");
        CV_CheckLE(output_->tensor()->byteSize(),
                   static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
                   "Metal Const tensor size exceeds uint32 capacity");

        data_ = data;
        dataTensor_ = metal::Tensor::create(data_);
    }

    void forward(metal::Context& context) CV_OVERRIDE
    {
        const std::shared_ptr<metal::Buffer>& dataBuffer = dataTensor_->bufferForRead();
        const std::shared_ptr<metal::Tensor>& outputTensor = output_->tensor();
        const std::shared_ptr<metal::Buffer>& outputBuffer = outputTensor->bufferForWrite();

        if (!pipeline_)
            pipeline_ = metal::Device::instance()->pipelineState("kernel_const_copy");

        Parameters parameters;
        parameters.byteCount = static_cast<uint32_t>(outputTensor->byteSize());
        const size_t threadCount =
            (parameters.byteCount + CONST_COPY_BYTES_PER_THREAD - 1) /
            CONST_COPY_BYTES_PER_THREAD;
        id<MTLComputeCommandEncoder> encoder = context.computeEncoder();
        [encoder setComputePipelineState:pipeline_];
        [encoder setBuffer:context.use(dataBuffer) offset:0 atIndex:0];
        [encoder setBuffer:context.use(outputBuffer) offset:0 atIndex:1];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(threadCount, 1, 1)
            threadsPerThreadgroup:context.threadsPerThreadgroup1D(
                pipeline_, threadCount)];
        context.didDispatch();
    }

private:
    struct Parameters
    {
        uint32_t byteCount;
    };

    Mat data_;
    std::shared_ptr<metal::Tensor> dataTensor_;
    Ptr<MetalBackendWrapper> output_;
    id<MTLComputePipelineState> pipeline_ = nil;
};

}  // namespace

CV__DNN_INLINE_NS_END

namespace metal {

Ptr<BackendNode> ConstOp::create(const std::vector<Ptr<BackendWrapper>>& inputs,
                                 const std::vector<Ptr<BackendWrapper>>& outputs,
                                 const Mat& data)
{
    return makeMetalBackendNode(
        std::unique_ptr<Operation>(new MetalConstImpl(inputs, outputs, data)));
}

}  // namespace metal
}}  // namespace cv::dnn

#endif  // HAVE_METAL
