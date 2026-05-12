// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../op_timvx.hpp"
#include "../ie_ngraph.hpp"

static_assert(sizeof(float) == 4, "float must be 4 bytes for int8 encoding");

namespace cv
{
namespace dnn
{

// Encode a float value as 4 raw bytes into a CV_8S Mat of shape (1, 4).
// This is used to pass float scale through the blob pipeline that only supports CV_8S dtype.
static inline void encodeFloatToInt8Mat(float value, Mat& dst)
{
    CV_Assert(dst.type() == CV_8S && dst.total() == 4);
    std::memcpy(dst.ptr(), &value, sizeof(float));
}

// Decode a float value from a CV_8S Mat of shape (1, 4).
static inline float decodeFloatFromInt8Mat(const Mat& src)
{
    CV_Assert(src.type() == CV_8S && src.total() == 4);
    float value;
    std::memcpy(&value, src.ptr(), sizeof(float));
    return value;
}

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

// Dynamic Quantize: compute scale/zp at runtime from activation min/max
class QuantizeDynamicLayerImpl CV_FINAL : public QuantizeDynamicLayer
{
public:
    int axis;

    QuantizeDynamicLayerImpl(const LayerParams& params)
    {
        axis = params.get<int>("axis", 1);
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
        outputs.resize(3);
        outputs[0] = inputs[0];          // quantized INT8 data (same shape as input)
        outputs[1] = MatShape({1, 4});   // scale: float encoded as 4 x int8 raw bytes
        outputs[2] = MatShape({1, 1});   // zeropoint (int8, dtype matches CV_8S)
        return false;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        // ONNX DynamicQuantizeLinear spec: quantize to uint8 range [0, 255]
        const int qmin = 0, qmax = 255;

        // Dynamically compute scale and zeropoint from activation range
        double rmin, rmax;
        cv::minMaxIdx(inputs[0], &rmin, &rmax);
        rmin = std::min(rmin, 0.0);
        rmax = std::max(rmax, 0.0);

        float sc = (float)((rmax == rmin) ? 1.0 : (rmax - rmin) / (qmax - qmin));
        // zp in uint8 space
        int zp_uint8 = (int)std::round(std::max(0.0, std::min(255.0, 0.0 - rmin / sc)));

        scales.resize(1); scales[0] = sc;
        zeropoints.resize(1); zeropoints[0] = zp_uint8;

        // output[0]: quantize using uint8 math, then subtract 128 to store as CV_8S
        // This matches getMatFromTensor's convention: int8_value = uint8_value - 128
        const float* inp = inputs[0].ptr<float>();
        int8_t* out = outputs[0].ptr<int8_t>();
        size_t total = inputs[0].total();
        float inv_sc = 1.f / sc;
        for (size_t i = 0; i < total; i++)
        {
            // Quantize to uint8 range with round-half-away-from-zero
            int y_uint8 = (int)std::round(std::max(0.0f, std::min(255.0f, inp[i] * inv_sc + (float)zp_uint8)));
            // Convert to int8 for CV_8S storage
            out[i] = (int8_t)(y_uint8 - 128);
        }

        // output[1]: scale encoded as 4 raw bytes in CV_8S blob (avoids dtype mismatch)
        // output[2]: zeropoint as int8 (CV_8S dtype matches, no encoding needed)
        encodeFloatToInt8Mat(sc, outputs[1]);
        outputs[2].at<int8_t>(0) = static_cast<int8_t>(zp_uint8 - 128);
    }
};

// Dynamic Dequantize: reads scale/zp from input tensors (produced by QuantizeDynamic)
class DequantizeDynamicLayerImpl CV_FINAL : public DequantizeDynamicLayer
{
public:
    DequantizeDynamicLayerImpl(const LayerParams& params)
    {
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
        // inputs[0] = INT8 data, inputs[1] = scale (1x1), inputs[2] = zeropoint (1x1)
        CV_Check(inputs.size(), inputs.size() >= 1 && inputs.size() <= 3,
                 "Number of inputs must be between 1 and 3 inclusive.");
        outputs.assign(1, inputs[0]);
        return false;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        float sc = 1.0f;
        int zp_int8 = 0;  // zero point in int8 space (i.e., uint8_value - 128)

        if (inputs.size() > 1)
        {
            // Decode scale from 4 raw bytes in CV_8S blob
            sc = decodeFloatFromInt8Mat(inputs[1]);
            if (inputs.size() > 2)
                zp_int8 = static_cast<int>(inputs[2].at<int8_t>(0));
        }
        else if (!scales.empty())
        {
            sc = scales[0];
            // zeropoints stores uint8 value; convert to int8 space
            zp_int8 = zeropoints.empty() ? -128 : (zeropoints[0] - 128);
        }

        // Dequantize: y_int8 = uint8_value - 128, zp_int8 = zp_uint8 - 128
        // x = (y_int8 - zp_int8) * sc
        inputs[0].convertTo(outputs[0], CV_32F, sc, -sc * zp_int8);
    }
};

Ptr<QuantizeDynamicLayer> QuantizeDynamicLayer::create(const LayerParams& params)
{
    return Ptr<QuantizeDynamicLayer>(new QuantizeDynamicLayerImpl(params));
}

Ptr<DequantizeDynamicLayer> DequantizeDynamicLayer::create(const LayerParams& params)
{
    return Ptr<DequantizeDynamicLayer>(new DequantizeDynamicLayerImpl(params));
}

}
}
