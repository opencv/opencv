// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"

#ifdef HAVE_METAL

#include "gemm.hpp"
#include "operation.hpp"
#include "../runtime/context.hpp"
#include "../runtime/device.hpp"
#include "../runtime/tensor.hpp"

#include <limits>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

namespace {

enum class GemmBiasMode : uint32_t
{
    NONE = 0,
    SCALAR = 1,
    COLUMN = 2,
    ROW = 3,
    MATRIX = 4
};

static GemmBiasMode resolveBiasMode(const MatShape& shape, size_t total,
                                    int realRank, size_t rows, size_t columns)
{
    CV_CheckGE(realRank, 0, "Metal Gemm bias rank must be known");
    CV_CheckLE(realRank, 2, "Metal Gemm bias rank must not exceed two");
    if (total == 1)
        return GemmBiasMode::SCALAR;
    if (realRank == 1)
    {
        CV_CheckEQ(total, columns, "Metal Gemm 1D bias size mismatch");
        return GemmBiasMode::COLUMN;
    }

    CV_CheckGE(shape.size(), static_cast<size_t>(2),
               "Metal Gemm 2D bias shape is unavailable");
    const size_t biasRows = static_cast<size_t>(shape[shape.size() - 2]);
    const size_t biasColumns = static_cast<size_t>(shape.back());
    if (biasRows == 1 && biasColumns == columns)
        return GemmBiasMode::COLUMN;
    if (biasRows == rows && biasColumns == 1)
        return GemmBiasMode::ROW;
    CV_CheckEQ(biasRows, rows, "Metal Gemm bias row count mismatch");
    CV_CheckEQ(biasColumns, columns, "Metal Gemm bias column count mismatch");
    return GemmBiasMode::MATRIX;
}

class MetalGemmImpl final : public metal::Operation
{
public:
    MetalGemmImpl(const Ptr<MetalBackendWrapper>& inputA,
                  const Ptr<MetalBackendWrapper>& dynamicInputB,
                  const Ptr<MetalBackendWrapper>& dynamicBias,
                  const Ptr<MetalBackendWrapper>& output,
                  const metal::GemmConfiguration& config)
        : inputA_(inputA), dynamicInputB_(dynamicInputB), dynamicBias_(dynamicBias),
          output_(output)
    {
        CV_CheckTrue(!inputA_.empty() && !output_.empty(),
                     "Metal Gemm requires Metal tensor wrappers");
        CV_CheckTypeEQ(inputA_->tensor()->type(), CV_32F,
                       "Metal Gemm input A must be CV_32F");
        CV_CheckTypeEQ(output_->tensor()->type(), CV_32F,
                       "Metal Gemm output must be CV_32F");

        const bool hasDynamicInputB = !dynamicInputB_.empty();
        CV_CheckNE(hasDynamicInputB, !config.weights.empty(),
                   "Metal Gemm requires exactly one static or dynamic B input");

        MatShape inputBShape;
        size_t inputBTotal = 0;
        if (hasDynamicInputB)
        {
            CV_CheckTypeEQ(dynamicInputB_->tensor()->type(), CV_32F,
                           "Metal Gemm input B must be CV_32F");
            inputBShape = dynamicInputB_->tensor()->shape();
            inputBTotal = dynamicInputB_->tensor()->total();
        }
        else
        {
            CV_CheckTypeEQ(config.weights.type(), CV_32F,
                           "Metal Gemm constant B must be CV_32F");
            staticInputBHost_ = config.weights.isContinuous()
                ? config.weights : config.weights.clone();
            staticInputBTensor_ = metal::Tensor::create(staticInputBHost_);
            inputBShape = cv::dnn::shape(staticInputBHost_);
            inputBTotal = staticInputBHost_.total();
        }

        const MatShape& inputAShape = inputA_->tensor()->shape();
        const MatShape& outputShape = output_->tensor()->shape();
        CV_CheckGE(inputAShape.size(), static_cast<size_t>(2),
                   "Metal Gemm input A rank must be at least two");
        CV_CheckEQ(inputBShape.size(), static_cast<size_t>(2),
                   "Metal Gemm input B must be two dimensional");

        const size_t rankA = inputAShape.size();
        const size_t aRows = static_cast<size_t>(inputAShape[rankA - 2]);
        const size_t aColumns = static_cast<size_t>(inputAShape[rankA - 1]);
        const size_t bRows = static_cast<size_t>(inputBShape[0]);
        const size_t bColumns = static_cast<size_t>(inputBShape[1]);
        const size_t m = config.transA ? aColumns : aRows;
        const size_t kA = config.transA ? aRows : aColumns;
        const size_t kB = config.transB ? bColumns : bRows;
        const size_t n = config.transB ? bRows : bColumns;
        CV_CheckEQ(kA, kB, "Metal Gemm inner dimensions mismatch");
        CV_CheckEQ(inputBTotal, bRows * bColumns, "Metal Gemm input B size mismatch");

        const size_t matrixElementsA = aRows * aColumns;
        const size_t batchCount = inputA_->tensor()->total() / matrixElementsA;
        CV_CheckEQ(output_->tensor()->total(), batchCount * m * n,
                   "Metal Gemm output size mismatch");
        CV_CheckFalse(outputShape.empty(), "Metal Gemm output shape must not be empty");
        CV_CheckEQ(static_cast<size_t>(outputShape.back()), n,
                   "Metal Gemm output column count mismatch");
        if (!config.flattenA)
        {
            CV_CheckFalse(config.transA,
                          "Metal Gemm flatten_a=false does not support transA");
            CV_CheckEQ(outputShape.size(), inputAShape.size(),
                       "Metal Gemm output rank mismatch");
        }

        GemmBiasMode biasMode = GemmBiasMode::NONE;
        if (config.hasBias)
        {
            const bool hasDynamicBias = !dynamicBias_.empty();
            CV_CheckNE(hasDynamicBias, !config.bias.empty(),
                       "Metal Gemm requires exactly one static or dynamic C input");
            MatShape biasShape;
            size_t biasTotal = 0;
            if (hasDynamicBias)
            {
                CV_CheckTypeEQ(dynamicBias_->tensor()->type(), CV_32F,
                               "Metal Gemm input C must be CV_32F");
                biasShape = dynamicBias_->tensor()->shape();
                biasTotal = dynamicBias_->tensor()->total();
            }
            else
            {
                CV_CheckTypeEQ(config.bias.type(), CV_32F,
                               "Metal Gemm constant C must be CV_32F");
                biasHost_ = config.bias.isContinuous() ? config.bias : config.bias.clone();
                biasTensor_ = metal::Tensor::create(biasHost_);
                biasShape = cv::dnn::shape(biasHost_);
                biasTotal = biasHost_.total();
            }
            const int biasRank = config.realBiasRank >= 0
                ? config.realBiasRank : static_cast<int>(biasShape.size());
            biasMode = resolveBiasMode(biasShape, biasTotal, biasRank, m, n);
        }
        else
        {
            biasHost_ = Mat::zeros(1, 1, CV_32F);
            biasTensor_ = metal::Tensor::create(biasHost_);
        }

        const size_t uint32Max = std::numeric_limits<uint32_t>::max();
        CV_CheckLE(output_->tensor()->total(), uint32Max,
                   "Metal Gemm output exceeds uint32 capacity");
        CV_CheckLE(aRows, uint32Max, "Metal Gemm A rows exceed uint32 capacity");
        CV_CheckLE(aColumns, uint32Max, "Metal Gemm A columns exceed uint32 capacity");
        CV_CheckLE(bRows, uint32Max, "Metal Gemm B rows exceed uint32 capacity");
        CV_CheckLE(bColumns, uint32Max, "Metal Gemm B columns exceed uint32 capacity");

        parameters_.outputCount = static_cast<uint32_t>(output_->tensor()->total());
        parameters_.m = static_cast<uint32_t>(m);
        parameters_.n = static_cast<uint32_t>(n);
        parameters_.k = static_cast<uint32_t>(kA);
        parameters_.aRows = static_cast<uint32_t>(aRows);
        parameters_.aColumns = static_cast<uint32_t>(aColumns);
        parameters_.bRows = static_cast<uint32_t>(bRows);
        parameters_.bColumns = static_cast<uint32_t>(bColumns);
        parameters_.biasMode = static_cast<uint32_t>(biasMode);
        parameters_.transA = config.transA ? 1u : 0u;
        parameters_.transB = config.transB ? 1u : 0u;
        parameters_.alpha = config.alpha;
        parameters_.beta = config.beta;

        const size_t operationCount = output_->tensor()->total() * kA;
        useTiledKernel_ = !config.transA && !config.transB &&
                          m >= 16 && n >= 16 && kA >= 16 &&
                          operationCount >= (static_cast<size_t>(1) << 20);
    }

    void forward(metal::Context& context) CV_OVERRIDE
    {
        if (!pipeline_)
        {
            pipeline_ = metal::Device::instance()->pipelineState(
                useTiledKernel_ ? "kernel_gemm_tiled_f32" : "kernel_gemm_f32");
        }

        const std::shared_ptr<metal::Tensor>& inputBTensor = dynamicInputB_.empty()
            ? staticInputBTensor_ : dynamicInputB_->tensor();
        const std::shared_ptr<metal::Tensor>& biasTensor = dynamicBias_.empty()
            ? biasTensor_ : dynamicBias_->tensor();
        const auto& inputABuffer = inputA_->tensor()->bufferForRead();
        const auto& inputBBuffer = inputBTensor->bufferForRead();
        const auto& biasBuffer = biasTensor->bufferForRead();
        const auto& outputBuffer = output_->tensor()->bufferForWrite();

        id<MTLComputeCommandEncoder> encoder = context.computeEncoder();
        [encoder setComputePipelineState:pipeline_];
        [encoder setBuffer:context.use(inputABuffer) offset:0 atIndex:0];
        [encoder setBuffer:context.use(inputBBuffer) offset:0 atIndex:1];
        [encoder setBuffer:context.use(biasBuffer) offset:0 atIndex:2];
        [encoder setBuffer:context.use(outputBuffer) offset:0 atIndex:3];
        [encoder setBytes:&parameters_ length:sizeof(parameters_) atIndex:4];
        if (useTiledKernel_)
        {
            const uint32_t batchCount = parameters_.outputCount /
                (parameters_.m * parameters_.n);
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
        uint32_t biasMode;
        uint32_t transA;
        uint32_t transB;
        float alpha;
        float beta;
    } parameters_;

    Ptr<MetalBackendWrapper> inputA_;
    Ptr<MetalBackendWrapper> dynamicInputB_;
    Ptr<MetalBackendWrapper> dynamicBias_;
    Ptr<MetalBackendWrapper> output_;
    Mat staticInputBHost_;
    Mat biasHost_;
    std::shared_ptr<metal::Tensor> staticInputBTensor_;
    std::shared_ptr<metal::Tensor> biasTensor_;
    id<MTLComputePipelineState> pipeline_ = nil;
    bool useTiledKernel_ = false;
};

}  // namespace

CV__DNN_INLINE_NS_END

namespace metal {

Ptr<BackendNode> GemmOp::create(const std::vector<Ptr<BackendWrapper>>& inputs,
                                const std::vector<Ptr<BackendWrapper>>& outputs,
                                const GemmConfiguration& config)
{
    CV_CheckEQ(outputs.size(), static_cast<size_t>(1),
               "Metal Gemm expects one output");
    size_t inputIndex = 0;
    CV_CheckGT(inputs.size(), inputIndex, "Metal Gemm expects input A");
    Ptr<MetalBackendWrapper> inputA = inputs[inputIndex++].dynamicCast<MetalBackendWrapper>();

    Ptr<MetalBackendWrapper> dynamicInputB;
    if (config.weights.empty())
    {
        CV_CheckGT(inputs.size(), inputIndex, "Metal Gemm expects dynamic input B");
        dynamicInputB = inputs[inputIndex++].dynamicCast<MetalBackendWrapper>();
    }

    Ptr<MetalBackendWrapper> dynamicBias;
    if (config.hasBias && config.bias.empty())
    {
        CV_CheckGT(inputs.size(), inputIndex, "Metal Gemm expects dynamic input C");
        dynamicBias = inputs[inputIndex++].dynamicCast<MetalBackendWrapper>();
    }
    CV_CheckEQ(inputIndex, inputs.size(), "Metal Gemm input count mismatch");

    return makeMetalBackendNode(std::unique_ptr<Operation>(new MetalGemmImpl(
        inputA, dynamicInputB, dynamicBias,
        outputs[0].dynamicCast<MetalBackendWrapper>(), config)));
}

}  // namespace metal
}}  // namespace cv::dnn

#endif  // HAVE_METAL
