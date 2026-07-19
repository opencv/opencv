// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"

#ifdef HAVE_METAL

#include "matmul.hpp"
#include "operation.hpp"
#include "../runtime/context.hpp"
#include "../runtime/device.hpp"
#include "../runtime/tensor.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

class MetalMatMulImpl final : public metal::Operation
{
public:
    MetalMatMulImpl(const Ptr<MetalBackendWrapper>& inputA,
                    const Ptr<MetalBackendWrapper>& dynamicInputB,
                    const Ptr<MetalBackendWrapper>& output,
                    const Mat& staticInputB,
                    const Mat& bias,
                    bool transA,
                    bool transB,
                    float alpha,
                    float beta)
        : inputA_(inputA), dynamicInputB_(dynamicInputB), output_(output)
    {
        CV_CheckTypeEQ(inputA_->tensor()->type(), CV_32F, "Metal tensor must be CV_32F");
        CV_CheckTypeEQ(output_->tensor()->type(), CV_32F, "Metal tensor must be CV_32F");

        const bool hasDynamicInputB = !dynamicInputB_.empty();
        CV_CheckNE(hasDynamicInputB, !staticInputB.empty(),
                   "Metal MatMul requires exactly one static or dynamic B input");

        MatShape inputBShape;
        size_t inputBTotal;
        if (hasDynamicInputB)
        {
            CV_CheckTypeEQ(dynamicInputB_->tensor()->type(), CV_32F, "Metal tensor must be CV_32F");
            inputBShape = dynamicInputB_->tensor()->shape();
            inputBTotal = dynamicInputB_->tensor()->total();
        }
        else
        {
            CV_CheckTypeEQ(staticInputB.type(), CV_32F,
                           "Metal MatMul static B input must be CV_32F");
            CV_CheckTrue(staticInputB.isContinuous(),
                         "Metal MatMul static B input must be continuous");
            staticInputBHost_ = staticInputB;
            staticInputBTensor_ = metal::Tensor::create(staticInputBHost_);
            inputBShape = cv::dnn::shape(staticInputB);
            inputBTotal = staticInputB.total();
        }

        const MatShape& inputAShape = inputA_->tensor()->shape();
        const MatShape& outputShape = output_->tensor()->shape();
        CV_CheckGE(inputAShape.size(), static_cast<size_t>(2),
                   "Metal MatMul requires rank 2 or greater");
        CV_CheckEQ(inputBShape.size(), inputAShape.size(),
                   "Metal MatMul inputs must have equal rank");
        CV_CheckEQ(outputShape.size(), inputAShape.size(),
                   "Metal MatMul output rank mismatch");

        const size_t rank = inputAShape.size();
        for (size_t d = 0; d + 2 < rank; ++d)
        {
            CV_CheckEQ(inputAShape[d], inputBShape[d],
                       "Metal MatMul does not support batch broadcasting");
            CV_CheckEQ(outputShape[d], inputAShape[d],
                       "Metal MatMul output batch shape mismatch");
        }

        const size_t aRows = static_cast<size_t>(inputAShape[rank - 2]);
        const size_t aColumns = static_cast<size_t>(inputAShape[rank - 1]);
        const size_t bRows = static_cast<size_t>(inputBShape[rank - 2]);
        const size_t bColumns = static_cast<size_t>(inputBShape[rank - 1]);
        const size_t m = transA ? aColumns : aRows;
        const size_t kA = transA ? aRows : aColumns;
        const size_t kB = transB ? bColumns : bRows;
        const size_t n = transB ? bRows : bColumns;
        CV_CheckEQ(kA, kB, "Metal MatMul inner dimensions mismatch");
        CV_CheckEQ(outputShape[rank - 2], static_cast<int>(m),
                   "Metal MatMul output rows mismatch");
        CV_CheckEQ(outputShape[rank - 1], static_cast<int>(n),
                   "Metal MatMul output columns mismatch");

        const size_t batchCount = inputA_->tensor()->total() / (aRows * aColumns);
        CV_CheckEQ(inputBTotal / (bRows * bColumns),
                   batchCount, "Metal MatMul batch count mismatch");
        CV_CheckEQ(output_->tensor()->total(), batchCount * m * n,
                   "Metal MatMul output size mismatch");

        biasHost_.create(1, static_cast<int>(n), CV_32F);
        if (bias.empty())
            biasHost_.setTo(Scalar(0));
        else
        {
            CV_CheckTypeEQ(bias.type(), CV_32F, "Metal MatMul bias must be CV_32F");
            CV_CheckEQ(bias.total(), n, "Metal MatMul bias size mismatch");
            bias.reshape(1, 1).copyTo(biasHost_);
        }
        biasTensor_ = metal::Tensor::create(biasHost_);

        parameters_.outputCount = static_cast<uint32_t>(output_->tensor()->total());
        parameters_.m = static_cast<uint32_t>(m);
        parameters_.n = static_cast<uint32_t>(n);
        parameters_.k = static_cast<uint32_t>(kA);
        parameters_.aRows = static_cast<uint32_t>(aRows);
        parameters_.aColumns = static_cast<uint32_t>(aColumns);
        parameters_.bRows = static_cast<uint32_t>(bRows);
        parameters_.bColumns = static_cast<uint32_t>(bColumns);
        parameters_.transA = transA ? 1u : 0u;
        parameters_.transB = transB ? 1u : 0u;
        parameters_.alpha = alpha;
        parameters_.beta = beta;
        const size_t minimumOutputElements = ((1u << 20) + kA - 1) / kA;
        useTiledKernel_ = !transA && !transB && m >= 16 && n >= 16 && n <= 256 && kA >= 16 &&
                          output_->tensor()->total() >= minimumOutputElements;
    }

    void forward(metal::Context& context) CV_OVERRIDE
    {
        if (useTiledKernel_)
        {
            if (!tiledPipeline_)
                tiledPipeline_ = metal::Device::instance()->pipelineState("kernel_matmul_tiled_f32");
        }
        else if (!pipeline_)
        {
            pipeline_ = metal::Device::instance()->pipelineState("kernel_matmul_f32");
        }

        const std::shared_ptr<metal::Tensor>& inputBTensor =
            dynamicInputB_.empty() ? staticInputBTensor_ : dynamicInputB_->tensor();
        const auto& inputABuffer = inputA_->tensor()->bufferForRead();
        const auto& inputBBuffer = inputBTensor->bufferForRead();
        const auto& biasBuffer = biasTensor_->bufferForRead();
        const auto& outputBuffer = output_->tensor()->bufferForWrite();
        id<MTLComputeCommandEncoder> encoder = context.computeEncoder();
        [encoder setComputePipelineState:useTiledKernel_ ? tiledPipeline_ : pipeline_];
        [encoder setBuffer:context.use(inputABuffer) offset:0 atIndex:0];
        [encoder setBuffer:context.use(inputBBuffer) offset:0 atIndex:1];
        [encoder setBuffer:context.use(biasBuffer) offset:0 atIndex:2];
        [encoder setBuffer:context.use(outputBuffer) offset:0 atIndex:3];
        [encoder setBytes:&parameters_ length:sizeof(parameters_) atIndex:4];
        if (useTiledKernel_)
        {
            const uint32_t batchCount = parameters_.outputCount / (parameters_.m * parameters_.n);
            [encoder dispatchThreadgroups:MTLSizeMake((parameters_.n + 31) / 32,
                                                       (parameters_.m + 15) / 16,
                                                       batchCount)
                threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        }
        else
        {
            [encoder dispatchThreads:MTLSizeMake(parameters_.outputCount, 1, 1)
                threadsPerThreadgroup:context.threadsPerThreadgroup1D(
                    pipeline_, parameters_.outputCount)];
        }
        context.didDispatch();
    }

private:
    struct Parameters
    {
        uint32_t outputCount;
        uint32_t m;
        uint32_t n;
        uint32_t k;
        uint32_t aRows;
        uint32_t aColumns;
        uint32_t bRows;
        uint32_t bColumns;
        uint32_t transA;
        uint32_t transB;
        float alpha;
        float beta;
    } parameters_;
    Ptr<MetalBackendWrapper> inputA_;
    Ptr<MetalBackendWrapper> dynamicInputB_;
    Ptr<MetalBackendWrapper> output_;
    Mat staticInputBHost_;
    Mat biasHost_;
    std::shared_ptr<metal::Tensor> staticInputBTensor_;
    std::shared_ptr<metal::Tensor> biasTensor_;
    id<MTLComputePipelineState> pipeline_ = nil;
    id<MTLComputePipelineState> tiledPipeline_ = nil;
    bool useTiledKernel_ = false;
};

static std::unique_ptr<metal::Operation> makeMatMulOperation(
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs,
        const metal::MatMulConfiguration& config)
{
    const bool hasDynamicInputB = config.weights.empty();
    CV_CheckEQ(inputs.size(), hasDynamicInputB ? static_cast<size_t>(2) : static_cast<size_t>(1),
               "Metal MatMul input count mismatch");
    CV_CheckEQ(outputs.size(), static_cast<size_t>(1), "Metal MatMul expects one output");
    Ptr<MetalBackendWrapper> dynamicInputB;
    if (hasDynamicInputB)
        dynamicInputB = inputs[1].dynamicCast<MetalBackendWrapper>();
    return std::unique_ptr<metal::Operation>(new MetalMatMulImpl(
        inputs[0].dynamicCast<MetalBackendWrapper>(), dynamicInputB,
        outputs[0].dynamicCast<MetalBackendWrapper>(), config.weights, config.bias,
        config.transA, config.transB, config.alpha, config.beta));
}

CV__DNN_INLINE_NS_END

namespace metal {

Ptr<BackendNode> MatMulOp::create(const std::vector<Ptr<BackendWrapper>>& inputs,
                                  const std::vector<Ptr<BackendWrapper>>& outputs,
                                  const MatMulConfiguration& config)
{
    return makeMetalBackendNode(makeMatMulOperation(inputs, outputs, config));
}

}  // namespace metal
}}  // namespace cv::dnn

#endif  // HAVE_METAL
