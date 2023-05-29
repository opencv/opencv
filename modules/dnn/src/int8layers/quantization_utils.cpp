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

static void broadcast1D2TargetMat(Mat& data, const MatShape& targetShape, int axis)
{
    // The data is the 1-D scales or zeropoints.
    CV_Assert(axis >= 0 && targetShape.size() > axis && data.total() == targetShape[axis]);
    std::vector<int> broadcast_axes;
    for (int i = 0; i < targetShape.size(); i++)
    {
        if (i != axis)
            broadcast_axes.push_back(i);
    }

    MatShape subTargetShape = shape(data);

    // convert std::vector to 1D Mat.
    for (auto broadcast_axis : broadcast_axes)
    {
        subTargetShape[broadcast_axis] = targetShape[broadcast_axis];
        data = data.reshape(0, total(data, 0, broadcast_axis));
        Mat tmp = cv::repeat(data, 1, subTargetShape[broadcast_axis]);
        data = tmp.reshape(0, subTargetShape);
    }
}

static void broadcastScaleAndZeropoint(Mat& scalesMat, Mat& zeropointsMat, const std::vector<float>& scales,
                                       const std::vector<int>& zeropoints, const MatShape& targetShape, int axis)
{
    // broad cast the scales and zeropoint to the input shape.
    MatShape subTargetShape(targetShape.size(), 1);
    subTargetShape[axis] = scales.size();

    zeropointsMat.create(subTargetShape.size(), subTargetShape.data(), CV_32FC1);
    scalesMat.create(subTargetShape.size(), subTargetShape.data(), CV_32FC1);

    const int len = scales.size();
    // Deep copy the scales and zeropoint data and prevent the original data from being changed.

    float * scalePtr = scalesMat.ptr<float>(0);
    for (int i = 0; i < len; i++)
        scalePtr[i] = scales[i];

    float * zpPtr = zeropointsMat.ptr<float>(0);
    for (int i = 0; i < len; i++)
        zpPtr[i] = (float )zeropoints[i];

    broadcast1D2TargetMat(scalesMat, targetShape, axis);
    broadcast1D2TargetMat(zeropointsMat, targetShape, axis);
}

// Quantize FP32/FP16 Inputs to INT8
class QuantizeLayerImpl CV_FINAL : public QuantizeLayer
{
public:
    int axis;
    bool is1D;
    Mat scalesMat, zeropointsMat; // Saving the broadcasetd scales data.

    QuantizeLayerImpl(const LayerParams& params)
    {
        is1D = params.get<bool>("is1D", false);
        axis = params.get<int>("axis", 1);
        if (!is1D)
        {
            scales.push_back(params.get<float>("scales", 1.0f));
            zeropoints.push_back(params.get<int>("zeropoints", 0));
        }
        else
        {
            DictValue paramScales = params.get("scales");
            int i, n = paramScales.size();

            CV_Assert(n > 0);
            scales.resize(n, 0.);
            for (i = 0; i < n; i++)
                scales[i] = paramScales.get<float>(i);

            zeropoints.resize(n, 0);
            DictValue paramZp = params.get("zeropoints");
            n = paramZp.size();

            for (i = 0; i < n; i++)
                zeropoints[i] = paramZp.get<int>(i);
        }
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

        axis = normalize_axis(axis, shape(inputs[0]).size());

        if (is1D)
        {
            MatShape inputShape = shape(inputs[0]);
            broadcastScaleAndZeropoint(scalesMat, zeropointsMat, scales, zeropoints, inputShape, axis);
        }
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

        inputs[0].convertTo(outputs[0], CV_8S, 1.f/scales[0], zeropoints[0]);
        return true;
    }
#endif

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget) && !is1D,
                   forward_ocl(inputs_arr, outputs_arr, internals_arr))

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        if (outputs[0].depth() != CV_8S)
            outputs[0].convertTo(outputs[0], CV_8S);

        if (is1D)
        {
            Mat inputTmp;
            divide(inputs[0], scalesMat, inputTmp);
            subtract(inputTmp, zeropointsMat, inputTmp);

            inputTmp.convertTo(outputs[0], CV_8S);
        }
        else
            inputs[0].convertTo(outputs[0], CV_8S, 1.f/scales[0], zeropoints[0]);
    }
};

// Dequantize INT8 Inputs to FP32/FP16
class DequantizeLayerImpl CV_FINAL : public DequantizeLayer
{
public:
    int axis;
    bool is1D;
    Mat scalesMat, zeropointsMat; // Saving the broadcasetd scales data.

    DequantizeLayerImpl(const LayerParams& params)
    {
        is1D = params.get<bool>("is1D", false);
        axis = params.get<int>("axis", 1);

        if (!is1D)
        {
            scales.push_back(params.get<float>("scales", 1.0f));
            zeropoints.push_back(params.get<int>("zeropoints", 0));
        }
        else
        {
            DictValue paramScales = params.get("scales");
            int i, n = paramScales.size();

            CV_Assert(n > 0);
            scales.resize(n);
            for (i = 0; i < n; i++)
                scales[i] = paramScales.get<float>(i);

            zeropoints.resize(n, 0);
            DictValue paramZp = params.get("zeropoints");
            n = paramZp.size();

            for (i = 0; i < n; i++)
                zeropoints[i] = paramZp.get<int>(i);
        }

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

        axis = normalize_axis(axis, shape(inputs[0]).size());

        if (is1D)
        {
            MatShape inputShape = shape(inputs[0]);
            broadcastScaleAndZeropoint(scalesMat, zeropointsMat, scales, zeropoints, inputShape, axis);
        }
    }

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inputs_, OutputArrayOfArrays outputs_, OutputArrayOfArrays internals_)
    {
        std::vector<UMat> inputs, outputs;
        inputs_.getUMatVector(inputs);
        outputs_.getUMatVector(outputs);

        UMat outputFp32;
        inputs[0].convertTo(outputFp32, CV_32F, scales[0], -(scales[0]*zeropoints[0]));

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

        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget) && !is1D,
                   forward_ocl(inputs_arr, outputs_arr, internals_arr))

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        if (outputs[0].depth() != CV_32F)
            outputs[0].convertTo(outputs[0], CV_32F);

        if (is1D)
        {
            Mat inputTmp;
            inputs[0].convertTo(inputTmp, CV_32F);
            subtract(inputTmp, zeropointsMat, inputTmp);
            multiply(inputTmp, scalesMat, outputs[0]);
        }
        else
            inputs[0].convertTo(outputs[0], CV_32F, scales[0], -(scales[0]*zeropoints[0]));
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
