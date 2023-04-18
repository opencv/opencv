// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../op_timvx.hpp"

#include <algorithm>
#include <stdlib.h>

namespace cv
{
namespace dnn
{

class SoftMaxLayerInt8Impl CV_FINAL : public SoftmaxLayerInt8
{
public:
    float input_sc;
    int input_zp;

    SoftMaxLayerInt8Impl(const LayerParams& params)
    {
        axisRaw = params.get<int>("axis", 1);
        logSoftMax = params.get<bool>("log_softmax", false);

        input_sc = params.get<float>("input_scale");
        input_zp = params.get<int>("input_zeropoint");

        output_sc = params.get<float>("scales");
        output_zp = params.get<int>("zeropoints");
        setParamsFrom(params);
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        bool inplace = Layer::getMemoryShapes(inputs, requiredOutputs, outputs, internals);
        MatShape shape = inputs[0];
        int cAxis = normalize_axis(axisRaw, shape.size());
        shape[cAxis] = 1;
        internals.assign(1, shape);
        return inplace;
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV ||
            (backendId == DNN_BACKEND_TIMVX && haveTimVX());
    }

    virtual bool tryFuse(Ptr<Layer>& top) CV_OVERRIDE
    {
        Ptr<DequantizeLayer> dequantize_layer = top.dynamicCast<DequantizeLayer>();
        return !dequantize_layer.empty() && preferableTarget != DNN_TARGET_OPENCL_FP16;
    }

   virtual Ptr<BackendNode> initTimVX(void* timVXInfo_,
                                       const std::vector<Ptr<BackendWrapper> > &inputsWrapper,
                                       const std::vector<Ptr<BackendWrapper> > &outputsWrapper,
                                       bool isLast) CV_OVERRIDE
    {
#ifdef HAVE_TIMVX
        // tvGraph Initialization.
        auto timVxInfo = reinterpret_cast<TimVXInfo *>(timVXInfo_);
        CV_Assert(timVxInfo);
        Ptr<TimVXGraph> tvGraph = timVxInfo->getGraph();
        CV_Assert(tvGraph);
        Ptr<tim::vx::Graph> graph = tvGraph->graph;

        std::vector<int> inputsIndex, outputsIndex;
        int input_index, output_index;

        // input Tensor
        CV_Assert(inputsWrapper.size() == 1);
        Ptr<TimVXBackendWrapper> inputWrapper = inputsWrapper[0].dynamicCast<TimVXBackendWrapper>();
        const Mat &src = inputWrapper->getMat();

        // convert axis from OpenCV NCHW toTimVX WHCN.
        int axis = normalize_axis(axisRaw, src.dims);
        int tvAxis = src.dims - 1 - axis;
        if(tvAxis < 0)
            tvAxis = 0; // default value is 0.

        if (inputWrapper->isTensor())
        {
            input_index = tvGraph->getTensorIndex(inputWrapper->getTensor());
            if (input_index == -1)
            {
                // Copy To New inputWrapper
                Mat tmp = inputWrapper->getMat();
                inputWrapper = Ptr<TimVXBackendWrapper>(new TimVXBackendWrapper(tmp));
            }
        }

        if (!inputWrapper->isTensor())
        {
            Ptr<tim::vx::Quantization> tvInputQuant = Ptr<tim::vx::Quantization>(
                    new tim::vx::Quantization(tim::vx::QuantType::ASYMMETRIC, input_sc, input_zp));
            inputWrapper->createTensor(graph,tim::vx::TensorAttribute::INPUT, tvInputQuant);
            input_index = tvGraph->addWrapper(inputWrapper);
        }
        inputsIndex.push_back(input_index);

        // output tensor
        CV_Assert(outputsWrapper.size() == 1);
        Ptr<TimVXBackendWrapper> outputWrapper = outputsWrapper[0].dynamicCast<TimVXBackendWrapper>();
        Mat dstMat = outputWrapper->getMat();
        Ptr<tim::vx::Quantization> outputQuant = Ptr<tim::vx::Quantization>(
                new tim::vx::Quantization(tim::vx::QuantType::ASYMMETRIC, output_sc, output_zp));

        Ptr<tim::vx::Tensor> outputTensor;

        if (isLast)
        {
            auto shapeType = getShapeTypeFromMat(outputWrapper->getMat());

            // For Graph Output tensor, we need to set tensor shape before createTensor().
            outputWrapper->setTensorShape(shapeType);
            if (dstMat.type() == CV_32F)
                outputWrapper->createTensor(graph, tim::vx::TensorAttribute::OUTPUT);
            else
                outputWrapper->createTensor(graph, tim::vx::TensorAttribute::OUTPUT, outputQuant);
        }
        else
        {
            if (dstMat.type() == CV_32F)
                outputWrapper->createTensor(graph, tim::vx::TensorAttribute::TRANSIENT);
            else
                outputWrapper->createTensor(graph, tim::vx::TensorAttribute::TRANSIENT, outputQuant);
        }
        output_index = tvGraph->addWrapper(outputWrapper);
        outputsIndex.push_back(output_index);

        std::shared_ptr<tim::vx::Operation> tvSoftmax;

        if (logSoftMax)
        {
            tvSoftmax = graph->CreateOperation<tim::vx::ops::LogSoftmax>(tvAxis);

        }
        else
        {
            tvSoftmax = graph->CreateOperation<tim::vx::ops::Softmax>(1.0f, tvAxis);
        }

        Ptr<TimVXBackendNode> tvBackendNode = new TimVXBackendNode(tvGraph, tvSoftmax, inputsIndex, outputsIndex);

        return tvBackendNode;
#endif  // HAVE_TIMVX
        return Ptr<BackendNode>();
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs, internals;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        internals_arr.getMatVector(internals);

        const Mat &src = inputs[0];
        Mat &dst = outputs[0];

        int axis = normalize_axis(axisRaw, src.dims);
        size_t outerSize = src.total(0, axis), channels = src.size[axis],
               innerSize = src.total(axis + 1);

        CV_Assert(src.type() == CV_8S && (dst.type() == CV_8S || dst.type() == CV_32F));
        CV_Assert(src.isContinuous() && dst.isContinuous());

        size_t outerStep = src.total(axis);
        size_t cnStep = src.total(axis + 1);
        const int8_t *srcPtr = src.ptr<int8_t>();
        const float *expPtr = blobs[0].ptr<float>();

        if (dst.type() == CV_32F)
        {
            float *dstPtr = dst.ptr<float>();
            for (size_t outerDim = 0; outerDim < outerSize; outerDim++)
            {
                size_t srcOffset = outerDim * outerStep;
                std::vector<float> expSum(innerSize, 0.f);

                // sum exp along axis
                for (size_t cnDim = 0; cnDim < channels; cnDim++)
                {
                    const int offset = srcOffset + cnDim * cnStep;
                    for (size_t i = 0; i < innerSize; i++)
                        expSum[i] += expPtr[srcPtr[offset + i] + 128];
                }

                // divide by computed sum
                for (size_t cnDim = 0; cnDim < channels; cnDim++)
                {
                    const int offset = srcOffset + cnDim * cnStep;
                    for (size_t i = 0; i < innerSize; i++)
                        dstPtr[offset + i] = expPtr[srcPtr[offset + i] + 128]/expSum[i];
                }

                if (logSoftMax)
                {
                    for (size_t cnDim = 0; cnDim < channels; cnDim++)
                    {
                        const int offset = srcOffset + cnDim * cnStep;
                        for (size_t i = 0; i < innerSize; i++)
                            dstPtr[offset + i] = log(dstPtr[offset + i]);
                    }
                }
            }
        }
        else
        {
            const float inv_scale = 1.f/output_sc;
            int8_t *dstPtr = dst.ptr<int8_t>();
            for (size_t outerDim = 0; outerDim < outerSize; outerDim++)
            {
                size_t srcOffset = outerDim * outerStep;
                std::vector<float> expSum(innerSize, 0.f);

                // sum exp along axis
                for (size_t cnDim = 0; cnDim < channels; cnDim++)
                {
                    const int offset = srcOffset + cnDim * cnStep;
                    for (size_t i = 0; i < innerSize; i++)
                        expSum[i] += expPtr[srcPtr[offset + i] + 128];
                }

                // divide by computed sum and quantize to int8
                if (logSoftMax)
                {
                    for (size_t cnDim = 0; cnDim < channels; cnDim++)
                    {
                        const int offset = srcOffset + cnDim * cnStep;
                        for (size_t i = 0; i < innerSize; i++)
                            dstPtr[offset + i] = saturate_cast<int8_t>(output_zp + std::round(inv_scale*log(expPtr[srcPtr[offset + i] + 128]/expSum[i])));
                    }
                }
                else
                {
                    for (size_t cnDim = 0; cnDim < channels; cnDim++)
                    {
                        const int offset = srcOffset + cnDim * cnStep;
                        for (size_t i = 0; i < innerSize; i++)
                            dstPtr[offset + i] = saturate_cast<int8_t>(output_zp + std::round(inv_scale*(expPtr[srcPtr[offset + i] + 128]/expSum[i])));
                    }
                }
            }
        }
    }

    int64 getFLOPS(const std::vector<MatShape> &inputs,
                  const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_UNUSED(outputs); // suppress unused variable warning
        int64 flops = 0;

        for (int i = 0; i < inputs.size(); i++)
        {
            flops += 4*total(inputs[i]);
        }

        return flops;
    }

    int axisRaw;
};

Ptr<SoftmaxLayerInt8> SoftmaxLayerInt8::create(const LayerParams& params)
{
    return Ptr<SoftmaxLayerInt8>(new SoftMaxLayerInt8Impl(params));
}

}
}
