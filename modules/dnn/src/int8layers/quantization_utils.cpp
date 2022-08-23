// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../op_timvx.hpp"

namespace cv
{
namespace dnn
{

// Quantize FP32/FP16 Inputs to INT8
class QuantizeLayerImpl CV_FINAL : public QuantizeLayer
{
public:
    QuantizeLayerImpl(const LayerParams& params)
    {
        scale = params.get<float>("scales", 1.0f);
        zeropoint = params.get<int>("zeropoints", 0);
        setParamsFrom(params);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 1);
        Layer::getMemoryShapes(inputs, requiredOutputs, outputs, internals);
        return false;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
    }

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inputs_, OutputArrayOfArrays outputs_, OutputArrayOfArrays internals_)
    {
        std::vector<UMat> inputs, outputs;
        inputs_.getUMatVector(inputs);
        outputs_.getUMatVector(outputs);

        if (inputs_.depth() == CV_16S)
        {
            UMat inputFp32;
            convertFp16(inputs[0], inputFp32);
            inputs[0] = inputFp32;  // replace
        }

        inputs[0].convertTo(outputs[0], CV_8S, 1.f/scale, zeropoint);
        return true;
    }
#endif

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget),
                   forward_ocl(inputs_arr, outputs_arr, internals_arr))

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        inputs[0].convertTo(outputs[0], CV_8S, 1.f/scale, zeropoint);
    }
};

// Dequantize INT8 Inputs to FP32/FP16
class DequantizeLayerImpl CV_FINAL : public DequantizeLayer
{
public:
    DequantizeLayerImpl(const LayerParams& params)
    {
        scale = params.get<float>("scales", 1.0f);
        zeropoint = params.get<int>("zeropoints", 0);
        setParamsFrom(params);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 1);
        Layer::getMemoryShapes(inputs, requiredOutputs, outputs, internals);
        return false;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
    }

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inputs_, OutputArrayOfArrays outputs_, OutputArrayOfArrays internals_)
    {
        std::vector<UMat> inputs, outputs;
        inputs_.getUMatVector(inputs);
        outputs_.getUMatVector(outputs);

        UMat outputFp32;
        inputs[0].convertTo(outputFp32, CV_32F, scale, -(scale*zeropoint));

        if (outputs_.depth() == CV_16S)
            convertFp16(outputFp32, outputs[0]);
        else
            outputFp32.copyTo(outputs[0]);
        return true;
    }
#endif

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget),
                   forward_ocl(inputs_arr, outputs_arr, internals_arr))

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        inputs[0].convertTo(outputs[0], CV_32F, scale, -(scale*zeropoint));
    }
};

// Rescale/Requantize INT8 Inputs from (scale1, zeropoint1) to (scale2, zeropoint2)
class RequantizeLayerImpl CV_FINAL : public RequantizeLayer
{
public:
    bool isEltwise;
    RequantizeLayerImpl(const LayerParams& params)
    {
        scale = params.get<float>("scale", 1.f);
        shift = params.get<float>("shift", 0.f);
        isEltwise = params.get<bool>("isEltwise", false);
        setParamsFrom(params);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        if (backendId == DNN_BACKEND_TIMVX && haveTimVX() && !isEltwise)
        {
            return true;
        }
        return backendId == DNN_BACKEND_OPENCV;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 1);
        Layer::getMemoryShapes(inputs, requiredOutputs, outputs, internals);
        return false;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
    }

    virtual Ptr<BackendNode> initTimVX(void* timVXInfo_,
                                       const std::vector<Ptr<BackendWrapper> > &inputsWrapper,
                                       const std::vector<Ptr<BackendWrapper> > &outputsWrapper,
                                       bool isLast) CV_OVERRIDE
    {
#ifdef HAVE_TIMVX
        // preprocessing
        // Check if data is 8-bit.
        CV_Assert(inputsWrapper.size() == 1 && outputsWrapper.size() == 1);
        Ptr<TimVXBackendWrapper> inputWrapper = inputsWrapper[0].dynamicCast<TimVXBackendWrapper>();

        if (!inputWrapper->isTensor())
        {
            return Ptr<BackendNode>();
        }

         auto timVxInfo = reinterpret_cast<TimVXInfo *>(timVXInfo_);
         CV_Assert(timVxInfo);
         Ptr<TimVXGraph> tvGraph = timVxInfo->getGraph();
         CV_Assert(tvGraph);
         Ptr<tim::vx::Graph> graph = tvGraph->graph;

        std::vector<int> inputsIndex, outputsIndex;
        int input_index = -1, output_index = -1;

        // Input
        std::shared_ptr<tim::vx::Tensor> inputTensor = inputWrapper->getTensor();
        input_index = tvGraph->getTensorIndex(inputTensor);
        if (input_index == -1)
            return Ptr<BackendNode>();

        inputsIndex.push_back(input_index);

        Ptr<tim::vx::Quantization> inputQuant = inputWrapper->getTensorQuantization();

        tim::vx::QuantType quanType = inputQuant->Type();
        CV_Assert(quanType == tim::vx::QuantType::ASYMMETRIC);

        std::vector<float> scales = inputQuant->Scales();
        std::vector<int32_t> zeropoints = inputQuant->ZeroPoints();
        CV_Assert(!scales.empty() && !zeropoints.empty());
        int input_zp = int(zeropoints[0]);
        float input_scale = scales[0];

        float  tmpOut_sc = input_scale/scale;
        int tmpOut_zp = int(shift + scale * input_zp);

        // Output
        Ptr<TimVXBackendWrapper> outputWrapper = outputsWrapper[0].dynamicCast<TimVXBackendWrapper>();
        Ptr<tim::vx::Quantization> outputQuant = Ptr<tim::vx::Quantization>(
                new tim::vx::Quantization(tim::vx::QuantType::ASYMMETRIC, tmpOut_sc, tmpOut_zp));

        if (isLast)
        {
            auto shapeType = getShapeTypeFromMat(outputWrapper->getMat());

            // For Graph Output tensor, we need to set tensor shape before createTensor().
            outputWrapper->setTensorShape(shapeType);
            outputWrapper->createTensor(graph, tim::vx::TensorAttribute::OUTPUT, outputQuant);
        }
        else
        {
            outputWrapper->createTensor(graph, tim::vx::TensorAttribute::TRANSIENT, outputQuant);
        }
        output_index = tvGraph->addWrapper(outputWrapper);
        outputsIndex.push_back(output_index);

        std::shared_ptr<tim::vx::Operation> tvRequantize = graph->CreateOperation<tim::vx::ops::DataConvert>();

        Ptr<TimVXBackendNode> tvBackendNode = new TimVXBackendNode(tvGraph, tvRequantize, inputsIndex, outputsIndex);

        return tvBackendNode;
#endif  // HAVE_TIMVX
        return Ptr<BackendNode>();
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        inputs[0].convertTo(outputs[0], CV_8S, scale, shift);
    }
};

Ptr<QuantizeLayer> QuantizeLayer::create(const LayerParams& params)
{
    return Ptr<QuantizeLayer>(new QuantizeLayerImpl(params));
}

Ptr<DequantizeLayer> DequantizeLayer::create(const LayerParams& params)
{
    return Ptr<DequantizeLayer>(new DequantizeLayerImpl(params));
}

Ptr<RequantizeLayer> RequantizeLayer::create(const LayerParams& params)
{
    return Ptr<RequantizeLayer>(new RequantizeLayerImpl(params));
}

}
}
