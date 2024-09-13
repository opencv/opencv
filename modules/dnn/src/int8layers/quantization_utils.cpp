// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../op_timvx.hpp"
#include "../ie_ngraph.hpp"

namespace cv
{
namespace dnn
{

static void broadcast1D2TargetMat(Mat& data, const MatShape& targetShape, int axis)
{
    // The data is the 1-D scales or zeropoints.
    CV_CheckGE(axis, 0, "Quantization axis must be non-negative.");
    CV_CheckGT((int)targetShape.size(),axis,"Quantization axis must be within the valid range of target shape dimensions.");
    CV_CheckEQ((int)data.total(), (int)targetShape[axis], "Data total size must match the size of the specified target dimension.");

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

static void block_repeat(InputArray src, const MatShape& srcShape, int axis, int repetitions, OutputArray dst)
{
    CV_Assert(src.getObj() != dst.getObj());
    CV_Check(axis, axis >= 0 && axis < src.dims(), "Axis out of range");
    CV_CheckGT(repetitions, 1, "More than one repetition expected");

    Mat src_mat = src.getMat();
    Mat dst_mat;

    if (src_mat.depth() != CV_32F)
        src_mat.convertTo(src_mat, CV_32F);

    MatShape sshape = srcShape;
    MatShape dshape = srcShape;

    size_t dtype_bytes = src_mat.elemSize();
    int chunk_size = dtype_bytes;
    int num_chunks = 1;

    dshape[axis] *= repetitions;

    for (int i = axis+1; i < sshape.size(); ++i)
        chunk_size*=sshape[i];

    for (int i = 0; i <= axis; ++i)
        num_chunks*=sshape[i];

    dst.create(dshape.size(), dshape.data(), src_mat.type());
    dst_mat = dst.getMat();

    CV_Assert(dst_mat.isContinuous());
    CV_Assert(src_mat.isContinuous());

    for (int i = 0; i < repetitions; ++i) {
        size_t src_offset = 0;
        size_t dst_offset = i * chunk_size;

        for (int j = 0; j < num_chunks; ++j) {
            memcpy(dst_mat.data + dst_offset, src_mat.data + src_offset, chunk_size);
            src_offset += chunk_size;
            dst_offset += chunk_size * repetitions;
        }
    }
}

template <typename T>
static void copyVecToMat(Mat& mat, const std::vector<T>& data){
    float * matPtr = mat.ptr<float>(0);
    const int len = data.size();

    for (int i = 0; i < len; i++)
        matPtr[i] = (float) data[i];
}

template <typename T>
static void broadcastBlockedMatrix(Mat& mat, const std::vector<T>& data, const MatShape& targetShape, int axis, int block_size){
    CV_Check(block_size, targetShape[axis] % block_size == 0 && block_size <= targetShape[axis], "Block size must be a divisor of the target dimension size and not exceed it.");

    MatShape subTargetShape(targetShape);
    subTargetShape[axis] = static_cast<int>(subTargetShape[axis] / block_size);

    block_repeat(data, subTargetShape, axis, block_size, mat);
}

template <typename T>
static void broadcastStandardMatrix(Mat& mat, const std::vector<T>& data, const MatShape& targetShape, int axis)
{
    MatShape subTargetShape(targetShape.size(), 1);
    subTargetShape[axis] = data.size();
    mat.create(subTargetShape.size(), subTargetShape.data(), CV_32FC1);

    copyVecToMat(mat,data);

    broadcast1D2TargetMat(mat, targetShape, axis);
}


static void broadcastScaleAndZeropoint(Mat& scalesMat, Mat& zeropointsMat, const std::vector<float>& scales,
                                       const std::vector<int>& zeropoints, const MatShape& targetShape, int axis, int block_size)
{
    // broad cast the scales and zeropoint to the input shape.

    if (block_size == 0)
    {
        broadcastStandardMatrix(zeropointsMat, zeropoints, targetShape, axis);
        broadcastStandardMatrix(scalesMat, scales, targetShape, axis);
    }
    else
    {
        broadcastBlockedMatrix(zeropointsMat, zeropoints, targetShape, axis, block_size);
        broadcastBlockedMatrix(scalesMat, scales, targetShape, axis, block_size);
    }
}

// Quantize FP32/FP16 Inputs to INT8
class QuantizeLayerImpl CV_FINAL : public QuantizeLayer
{
public:
    int axis;
    int block_size;
    bool is1D;
    Mat scalesMat, zeropointsMat; // Saving the broadcasted scales data.
    bool quantParamExternal = true;  // Indicates if the quantization parameters (scale and zero point) are provided as inputs to the node.

    QuantizeLayerImpl(const LayerParams& params)
    {
        is1D = params.get<bool>("is1D", false);
        axis = params.get<int>("axis", 1);
        block_size = params.get<int>("block_size", 0);

        if (!is1D)
        {
            scales.push_back(params.get<float>("scales", 1.0f));
            zeropoints.push_back(params.get<int>("zeropoints", 0));
        }
        else
        {
            DictValue paramScales = params.get("scales");
            int i, n = paramScales.size();

            CV_CheckGT(n, 0, "Scale missing.");
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
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Check(inputs.size(), inputs.size() >= 1 && inputs.size() <= 3, "Number of inputs must be between 1 and 3 inclusive.");
        CV_Assert(requiredOutputs <= 1);
        outputs.assign(1, inputs[0]);
        return false;
    }

    void getTypes(const std::vector<MatType>& inputs,
        const int requiredOutputs,
        const int requiredInternals,
        std::vector<MatType>& outputs,
        std::vector<MatType>& internals) const CV_OVERRIDE
    {
        outputs.assign(requiredOutputs, CV_8S);
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
            broadcastScaleAndZeropoint(scalesMat, zeropointsMat, scales, zeropoints, inputShape, axis, block_size);
        }
    }

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inputs_, OutputArrayOfArrays outputs_, OutputArrayOfArrays internals_)
    {
        std::vector<UMat> inputs, outputs;
        inputs_.getUMatVector(inputs);
        outputs_.getUMatVector(outputs);

        if (inputs_.depth() == CV_16F)
        {
            UMat inputFp32;
            inputs[0].convertTo(inputFp32, CV_32F);
            inputs[0] = inputFp32;  // replace
        }

        inputs[0].convertTo(outputs[0], CV_8S, 1.f/scales[0], zeropoints[0]);
        return true;
    }
#endif
    void processInputOutput(std::vector<Mat>& inputs, std::vector<Mat>& outputs)
    {
        CV_Check(inputs.size(), inputs.size() >= 1 && inputs.size() <= 3, "Number of inputs must be between 1 and 3 inclusive.");
        quantParamExternal &= inputs.size() > 1;

        // Scale and zeropoint taken as input
        if (quantParamExternal)
        {
            quantParamExternal = false;
            scalesMat = inputs[1];

            scalesMat.reshape(1, 1).copyTo(scales);

            if(scalesMat.total() > 1) is1D = true;


            if (inputs.size() > 2)
            {
                zeropointsMat = inputs[2];
                CV_CheckEQ((int)zeropointsMat.total(), (int)scalesMat.total(), "Scale and zero point elements number must match.");
                zeropointsMat.reshape(1, 1).copyTo(zeropoints);
            }

            if (is1D)
            {
                MatShape inputShape = shape(inputs[0]);
                broadcastScaleAndZeropoint(scalesMat, zeropointsMat, scales, zeropoints, inputShape, axis, block_size);
            }
        }

        if (outputs[0].depth() != CV_8S)
            outputs[0].convertTo(outputs[0], CV_8S);
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget) && !is1D,
                   forward_ocl(inputs_arr, outputs_arr, internals_arr))

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        processInputOutput(inputs, outputs);

        if (is1D)
        {
            Mat inputTmp;
            divide(inputs[0], scalesMat, inputTmp);
            add(inputTmp, zeropointsMat, inputTmp);

            inputTmp.convertTo(outputs[0], CV_8S);
        }
        else
            inputs[0].convertTo(outputs[0], CV_8S, 1.f/scales[0], zeropoints[0]);
    }

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        const auto input = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
        auto quantized = ngraphQuantize(input, scales[0], zeropoints[0]);
        return Ptr<BackendNode>(new InfEngineNgraphNode(quantized));
    }
#endif  // HAVE_DNN_NGRAPH
};

// Dequantize INT8 Inputs to FP32/FP16
class DequantizeLayerImpl CV_FINAL : public DequantizeLayer
{
public:
    int axis;
    int block_size;
    bool is1D;
    Mat scalesMat, zeropointsMat; // Saving the broadcasetd scales data.
    bool quantParamExternal = true;

    DequantizeLayerImpl(const LayerParams& params)
    {
        is1D = params.get<bool>("is1D", false);
        axis = params.get<int>("axis", 1);
        block_size = params.get<int>("block_size", 0);

        if (!is1D)
        {
            scales.push_back(params.get<float>("scales", 1.0f));
            zeropoints.push_back(params.get<int>("zeropoints", 0));
        }
        else
        {
            DictValue paramScales = params.get("scales");
            int i, n = paramScales.size();

            CV_CheckGT(n, 0, "Scale missing.");
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
        return backendId == DNN_BACKEND_OPENCV || backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Check(inputs.size(), inputs.size() >= 1 && inputs.size() <= 3, "Number of inputs must be between 1 and 3 inclusive.");
        CV_Assert(requiredOutputs <= 1);
        outputs.assign(1, inputs[0]);
        return false;
    }

    void getTypes(const std::vector<MatType>& inputs,
        const int requiredOutputs,
        const int requiredInternals,
        std::vector<MatType>& outputs,
        std::vector<MatType>& internals) const CV_OVERRIDE
    {
        if (preferableTarget == DNN_TARGET_OPENCL_FP16)
            outputs.assign(requiredOutputs, CV_16F);
        else
            outputs.assign(requiredOutputs, CV_32F);
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
            broadcastScaleAndZeropoint(scalesMat, zeropointsMat, scales, zeropoints, inputShape, axis, block_size);
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

        outputFp32.convertTo(outputs[0], outputs_.depth());
        return true;
    }
#endif

    void processInputOutput(std::vector<Mat>& inputs, std::vector<Mat>& outputs)
    {
        CV_Check(inputs.size(), inputs.size() >= 1 && inputs.size() <= 3, "Number of inputs must be between 1 and 3 inclusive.");

        quantParamExternal &= inputs.size() > 1;
        // Scale and zeropoint taken as input
        if (quantParamExternal)
        {
            quantParamExternal = false;
            scalesMat = inputs[1];

            scalesMat.reshape(1, 1).copyTo(scales);

            if(scalesMat.total() > 1) is1D = true;

            if (inputs.size() > 2)
            {
                zeropointsMat = inputs[2];
                CV_CheckEQ((int)zeropointsMat.total(), (int)scalesMat.total(), "Scale and zero point elements number must match.");
                zeropointsMat.reshape(1, 1).copyTo(zeropoints);
            }

            if (is1D)
            {
                MatShape inputShape = shape(inputs[0]);
                broadcastScaleAndZeropoint(scalesMat, zeropointsMat, scales, zeropoints, inputShape, axis, block_size);
            }
        }

        if (outputs[0].depth() != CV_32F)
            outputs[0].convertTo(outputs[0], CV_32F);
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget) && !is1D,
                   forward_ocl(inputs_arr, outputs_arr, internals_arr))

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        processInputOutput(inputs, outputs);

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

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        const auto input = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
        auto quantized = ngraphDequantize(input, scales[0], zeropoints[0]);
        return new InfEngineNgraphNode(quantized);
    }
#endif  // HAVE_DNN_NGRAPH
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
