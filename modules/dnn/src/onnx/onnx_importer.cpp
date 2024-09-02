// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>

#include <opencv2/dnn/layer_reg.private.hpp>

#include <opencv2/core/utils/fp_control_utils.hpp>

#include <opencv2/core/utils/logger.defines.hpp>
#undef CV_LOG_STRIP_LEVEL
#define CV_LOG_STRIP_LEVEL CV_LOG_LEVEL_VERBOSE + 1
#include <opencv2/core/utils/logger.hpp>

#include <opencv2/core/utils/configuration.private.hpp>


#ifdef HAVE_PROTOBUF

#include <array>
#include <iostream>
#include <fstream>
#include <string>
#include <limits>
#include <algorithm>

#if defined _MSC_VER && _MSC_VER < 1910/*MSVS 2017*/
#pragma warning(push)
#pragma warning(disable: 4503)  // decorated name length exceeded, name was truncated
#endif

#if defined(__GNUC__) && __GNUC__ >= 5
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif
#include "opencv-onnx.pb.h"
#if defined(__GNUC__) && __GNUC__ >= 5
#pragma GCC diagnostic pop
#endif

#include "onnx_graph_simplifier.hpp"
#endif

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

extern bool DNN_DIAGNOSTICS_RUN;

#ifdef HAVE_PROTOBUF
class ONNXLayerHandler;

template <typename T>
static T getScalarFromMat(Mat m)
{
    CV_Assert(m.total() == 1);
    return m.at<T>(0);
}

class ONNXImporter
{
    FPDenormalsIgnoreHintScope fp_denormals_ignore_scope;

    opencv_onnx::ModelProto model_proto;
    struct LayerInfo {
        int layerId;
        int outputId;
        int depth;
        LayerInfo(int _layerId = 0, int _outputId = 0, int _depth = CV_32F)
            :layerId(_layerId), outputId(_outputId), depth(_depth) {}
    };

    struct TensorInfo {
        int real_ndims;
        TensorInfo(int _real_ndims = 0) : real_ndims(_real_ndims) {}
    };

    std::map<std::string, Mat> getGraphTensors(
                                    const opencv_onnx::GraphProto& graph_proto);
    Mat getBlob(const opencv_onnx::NodeProto& node_proto, int index);
    Mat getBlob(const std::string& input_name);
    Mat getIntBlob(const opencv_onnx::NodeProto& node_proto, int index);
    TensorInfo getBlobExtraInfo(const opencv_onnx::NodeProto& node_proto, int index);
    TensorInfo getBlobExtraInfo(const std::string& input_name);

    LayerParams getLayerParams(const opencv_onnx::NodeProto& node_proto);

    void addConstant(const std::string& name, const Mat& blob);
    void addLayer(LayerParams& layerParams,
                  const opencv_onnx::NodeProto& node_proto,
                  int num_inputs = std::numeric_limits<int>::max());
    void setParamsDtype(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);

    void lstm_extractConsts(LayerParams& layerParams, const opencv_onnx::NodeProto& lstm_proto, size_t idx, int* blobShape_, int size);
    void lstm_add_reshape(const std::string& input_name, const std::string& output_name, int* layerShape, size_t n);
    std::string lstm_add_slice(int index, const std::string& input_name, int* begin, int* end, size_t n);
    std::string lstm_fix_dims(LayerParams& layerParams, const opencv_onnx::NodeProto& lstm_proto,
                              int batch_size, int num_directions, int hidden_size, bool need_y, const std::string& y_name,
                              const int index);
    void lstm_add_transform(int num_directions, int batch_size, int hidden_size,
                            int index, const std::string& input_name, const std::string& output_name);
public:
    ONNXImporter(Net& net, const char *onnxFile);
    ONNXImporter(Net& net, const char* buffer, size_t sizeBuffer);

    void populateNet();

protected:
    std::unique_ptr<ONNXLayerHandler> layerHandler;
    Net& dstNet;

    opencv_onnx::GraphProto* graph_proto;
    std::string framework_name;

    std::map<std::string, Mat> constBlobs;
    std::map<std::string, TensorInfo> constBlobsExtraInfo;

    std::map<std::string, MatShape> outShapes;  // List of internal blobs shapes.
    bool hasDynamicShapes;  // Whether the model has inputs with dynamic shapes
    typedef std::map<std::string, MatShape>::iterator IterShape_t;

    std::map<std::string, LayerInfo> layer_id;
    typedef std::map<std::string, LayerInfo>::iterator IterLayerId_t;
    typedef std::map<std::string, LayerInfo>::const_iterator ConstIterLayerId_t;

    void handleNode(const opencv_onnx::NodeProto& node_proto);

private:
    friend class ONNXLayerHandler;
    typedef void (ONNXImporter::*ONNXImporterNodeParser)(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    typedef std::map<std::string, ONNXImporterNodeParser> DispatchMap;
    typedef std::map<std::string, DispatchMap> DomainDispatchMap;

    DomainDispatchMap domain_dispatch_map;
    std::string getLayerTypeDomain(const opencv_onnx::NodeProto& node_proto);
    const DispatchMap& getDispatchMap(const opencv_onnx::NodeProto& node_proto);
    void buildDispatchMap_ONNX_AI(int opset_version);
    void buildDispatchMap_COM_MICROSOFT(int opset_version);

    // Domain: 'ai.onnx' (default)
    // URL: https://github.com/onnx/onnx/blob/master/docs/Operators.md
    void parseArg                  (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseMaxUnpool            (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseMaxPool              (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseAveragePool          (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseGlobalPool           (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseReduce               (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseSlice                (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseSplit                (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseNeg                  (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseConstant             (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseLSTM                 (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseGRU                  (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseImageScaler          (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseClip                 (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseLeakyRelu            (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseRelu                 (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseElu                  (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseTanh                 (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseAbs                  (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parsePRelu                (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseLRN                  (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseInstanceNormalization(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseBatchNormalization   (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseGemm                 (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseMatMul               (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseConv                 (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseConvTranspose        (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseTranspose            (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseSqueeze              (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseFlatten              (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseUnsqueeze            (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseExpand               (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseReshape              (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parsePad                  (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseShape                (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseCast                 (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseConstantFill         (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseGather               (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseGatherElements       (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseConcat               (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseResize               (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseUpsample             (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseSoftMax              (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseDetectionOutput      (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseCumSum               (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseElementWise          (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseDepthSpaceOps        (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseRange                (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseScatter              (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseTile                 (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseLayerNorm            (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseTopK                 (LayerParams& LayerParams, const opencv_onnx::NodeProto& node_proto);
    void parseSimpleLayers         (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseEinsum               (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseHardmax              (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseGatherND             (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);

    // Domain: com.microsoft
    // URL: https://github.com/microsoft/onnxruntime/blob/master/docs/ContribOperators.md
    void parseQuantDequant         (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseQConv                (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseQMatMul              (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseQEltwise             (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseQLeakyRelu           (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseQSigmoid             (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseQAvgPool             (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseQConcat              (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseQGemm                (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseQSoftmax             (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);
    void parseAttention            (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);

    // '???' domain or '???' layer type
    void parseCustomLayer          (LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto);

    int onnx_opset;  // OperatorSetIdProto for 'onnx' domain
    std::map<std::string, int> onnx_opset_map;  // map from OperatorSetIdProto
    void parseOperatorSet();

    const std::string str_domain_ai_onnx = "ai.onnx";


    bool useLegacyNames;
    bool getParamUseLegacyNames()
    {
        bool param = utils::getConfigurationParameterBool("OPENCV_DNN_ONNX_USE_LEGACY_NAMES", false);
        return param;
    }
    std::string extractNodeName(const opencv_onnx::NodeProto& node_proto);
};


class ONNXLayerHandler : public detail::LayerHandler
{
public:
    explicit ONNXLayerHandler(ONNXImporter* importer_);

    void fillRegistry(const opencv_onnx::GraphProto& net);

protected:
    ONNXImporter* importer;
};

ONNXLayerHandler::ONNXLayerHandler(ONNXImporter* importer_) : importer(importer_){}

void ONNXLayerHandler::fillRegistry(const opencv_onnx::GraphProto &net)
{
    int layersSize = net.node_size();
    for (int li = 0; li < layersSize; li++) {
        const opencv_onnx::NodeProto &node_proto = net.node(li);
        const std::string& name = node_proto.output(0);
        const std::string& type = node_proto.op_type();
        const std::string& layer_type_domain = importer->getLayerTypeDomain(node_proto);
        const auto& dispatch = importer->getDispatchMap(node_proto);
        if (dispatch.find(type) == dispatch.end())
        {
            addMissing(name, cv::format("%s.%s", layer_type_domain.c_str(), type.c_str()));
        }
    }
    printMissing();
}

ONNXImporter::ONNXImporter(Net& net, const char *onnxFile)
    : layerHandler(DNN_DIAGNOSTICS_RUN ? new ONNXLayerHandler(this) : nullptr)
    , dstNet(net)
    , onnx_opset(0)
    , useLegacyNames(getParamUseLegacyNames())
{
    hasDynamicShapes = false;
    CV_Assert(onnxFile);
    CV_LOG_DEBUG(NULL, "DNN/ONNX: processing ONNX model from file: " << onnxFile);

    std::fstream input(onnxFile, std::ios::in | std::ios::binary);
    if (!input)
    {
        CV_Error(Error::StsBadArg, cv::format("Can't read ONNX file: %s", onnxFile));
    }

    if (!model_proto.ParseFromIstream(&input))
    {
        CV_Error(Error::StsUnsupportedFormat, cv::format("Failed to parse ONNX model: %s", onnxFile));
    }

    populateNet();
}

ONNXImporter::ONNXImporter(Net& net, const char* buffer, size_t sizeBuffer)
    : layerHandler(DNN_DIAGNOSTICS_RUN ? new ONNXLayerHandler(this) : nullptr)
    , dstNet(net)
    , onnx_opset(0)
    , useLegacyNames(getParamUseLegacyNames())
{
    hasDynamicShapes = false;
    CV_LOG_DEBUG(NULL, "DNN/ONNX: processing in-memory ONNX model (" << sizeBuffer << " bytes)");

    struct _Buf : public std::streambuf
            {
        _Buf(const char* buffer, size_t sizeBuffer)
        {
            char* p = const_cast<char*>(buffer);
            setg(p, p, p + sizeBuffer);
        }
            };

    _Buf buf(buffer, sizeBuffer);
    std::istream input(&buf);

    if (!model_proto.ParseFromIstream(&input))
        CV_Error(Error::StsUnsupportedFormat, "Failed to parse onnx model from in-memory byte array.");

    populateNet();
}


inline void replaceLayerParam(LayerParams& layerParams, const String& oldKey, const String& newKey)
{
    if (layerParams.has(oldKey)) {
        layerParams.set(newKey, layerParams.get(oldKey));
        layerParams.erase(oldKey);
    }
}

static
void dumpValueInfoProto(int i, const opencv_onnx::ValueInfoProto& valueInfoProto, const std::string& prefix)
{
    CV_Assert(valueInfoProto.has_name());
    CV_Assert(valueInfoProto.has_type());
    const opencv_onnx::TypeProto& typeProto = valueInfoProto.type();
    CV_Assert(typeProto.has_tensor_type());
    const opencv_onnx::TypeProto::Tensor& tensor = typeProto.tensor_type();
    CV_Assert(tensor.has_shape());
    const opencv_onnx::TensorShapeProto& tensorShape = tensor.shape();

    int dim_size = tensorShape.dim_size();
    CV_CheckGE(dim_size, 0, "");
    MatShape shape(dim_size);
    for (int j = 0; j < dim_size; ++j)
    {
        const opencv_onnx::TensorShapeProto_Dimension& dimension = tensorShape.dim(j);
        if (dimension.has_dim_param())
        {
            CV_LOG_DEBUG(NULL, "DNN/ONNX: " << prefix << "[" << i << "] dim[" << j << "] = <" << dimension.dim_param() << "> (dynamic)");
        }
        // https://github.com/onnx/onnx/blob/master/docs/DimensionDenotation.md#denotation-definition
        if (dimension.has_denotation())
        {
            CV_LOG_INFO(NULL, "DNN/ONNX: " << prefix << "[" << i << "] dim[" << j << "] denotation is '" << dimension.denotation() << "'");
        }
        shape[j] = dimension.dim_value();
    }
    CV_LOG_DEBUG(NULL, "DNN/ONNX: " << prefix << "[" << i << " as '" << valueInfoProto.name() << "'] shape=" << toString(shape));
}

static
void dumpTensorProto(int i, const opencv_onnx::TensorProto& tensorProto, const std::string& prefix)
{
    if (utils::logging::getLogLevel() < utils::logging::LOG_LEVEL_VERBOSE)
        return;
    int dim_size = tensorProto.dims_size();
    CV_CheckGE(dim_size, 0, "");
    MatShape shape(dim_size);
    for (int j = 0; j < dim_size; ++j)
    {
        int sz = static_cast<int>(tensorProto.dims(j));
        shape[j] = sz;
    }
    CV_LOG_VERBOSE(NULL, 0, "DNN/ONNX: " << prefix << "[" << i << " as '" << tensorProto.name() << "'] shape=" << toString(shape) << " data_type=" << (int)tensorProto.data_type());
}

void releaseONNXTensor(opencv_onnx::TensorProto& tensor_proto)
{
    if (!tensor_proto.raw_data().empty()) {
        delete tensor_proto.release_raw_data();
    }
}

void runLayer(LayerParams& params, const std::vector<Mat>& inputs,
              std::vector<Mat>& outputs)
{
    Ptr<Layer> layer = LayerFactory::createLayerInstance(params.type, params);
    CV_Assert((bool)layer);

    std::vector<MatShape> inpShapes(inputs.size());
    std::vector<MatType> inpTypes(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        inpShapes[i] = shape(inputs[i]);
        inpTypes[i] = inputs[i].type();
    }

    std::vector<MatShape> outShapes, internalShapes;
    std::vector<MatType> outTypes, internalTypes;
    layer->getMemoryShapes(inpShapes, 0, outShapes, internalShapes);
    layer->getTypes(inpTypes, outShapes.size(), internalShapes.size(), outTypes, internalTypes);

    std::vector<Mat> internals(internalShapes.size());
    outputs.resize(outShapes.size());
    for (size_t i = 0; i < outShapes.size(); ++i)
        outputs[i].create(outShapes[i], outTypes[i]);
    for (size_t i = 0; i < internalShapes.size(); ++i)
        internals[i].create(internalShapes[i], internalTypes[i]);

    layer->finalize(inputs, outputs);
    layer->forward(inputs, outputs, internals);
}

std::map<std::string, Mat> ONNXImporter::getGraphTensors(
                                        const opencv_onnx::GraphProto& graph_proto)
{
    std::map<std::string, Mat> layers_weights;

    for (int i = 0; i < graph_proto.initializer_size(); i++)
    {
        const opencv_onnx::TensorProto& tensor_proto = graph_proto.initializer(i);
        dumpTensorProto(i, tensor_proto, "initializer");
        Mat mat = getMatFromTensor(tensor_proto);
        releaseONNXTensor(const_cast<opencv_onnx::TensorProto&>(tensor_proto));  // drop already loaded data

        if (DNN_DIAGNOSTICS_RUN && mat.empty())
            continue;

        layers_weights.insert(std::make_pair(tensor_proto.name(), mat));
        constBlobsExtraInfo.insert(std::make_pair(tensor_proto.name(), TensorInfo(tensor_proto.dims_size())));
    }
    return layers_weights;
}

static DictValue parse(const ::google::protobuf::RepeatedField< ::google::protobuf::int64>& src) {
    std::vector<int32_t> dst(src.size());
    convertInt64ToInt32(src, dst, src.size());
    return DictValue::arrayInt(&dst[0], src.size());
}

static DictValue parseStr(const ::google::protobuf::RepeatedPtrField< ::std::string>& src) {
    return DictValue::arrayString(src.begin(), static_cast<int>(src.size()));
}

LayerParams ONNXImporter::getLayerParams(const opencv_onnx::NodeProto& node_proto)
{
    LayerParams lp;
    for(int i = 0; i < node_proto.attribute_size(); i++)
    {
        opencv_onnx::AttributeProto attribute_proto = node_proto.attribute(i);
        std::string attribute_name = attribute_proto.name();

        try
        {
            if(attribute_name == "kernel_shape")
            {
                CV_Assert(attribute_proto.ints_size() == 1 || attribute_proto.ints_size() == 2 || attribute_proto.ints_size() == 3);
                lp.set("kernel_size", parse(attribute_proto.ints()));
            }
            else if(attribute_name == "strides")
            {
                CV_Assert(attribute_proto.ints_size() == 1 || attribute_proto.ints_size() == 2 || attribute_proto.ints_size() == 3);
                lp.set("stride", parse(attribute_proto.ints()));
            }
            else if(attribute_name == "pads")
            {
                if (node_proto.op_type() == "Pad")
                {
                    // Padding layer.
                    // Paddings are in order begin0, begin1, .. beginN, end0, end1, ..., endN.
                    // We need to shuffle it to begin0, end0, begin1, end1, ...
                    CV_Assert(attribute_proto.ints_size() % 2 == 0);
                    const int dims = attribute_proto.ints_size() / 2;
                    std::vector<int32_t> paddings;
                    paddings.reserve(attribute_proto.ints_size());
                    for (int i = 0; i < dims; ++i)
                    {
                        paddings.push_back(attribute_proto.ints(i));
                        paddings.push_back(attribute_proto.ints(dims + i));
                    }
                    lp.set("paddings", DictValue::arrayInt(&paddings[0], paddings.size()));
                }
                else
                {
                    // Convolution or pooling.
                    CV_Assert(attribute_proto.ints_size() == 2 || attribute_proto.ints_size() == 4 || attribute_proto.ints_size() == 6);
                    lp.set("pad", parse(attribute_proto.ints()));
                }
            }
            else if(attribute_name == "auto_pad")
            {
                if (attribute_proto.s() == "SAME_UPPER" || attribute_proto.s() == "SAME_LOWER") {
                    lp.set("pad_mode",  "SAME");
                }
                else if (attribute_proto.s() == "VALID") {
                    lp.set("pad_mode", "VALID");
                }
            }
            else if(attribute_name == "dilations")
            {
                CV_Assert(attribute_proto.ints_size() == 1 || attribute_proto.ints_size() == 2 || attribute_proto.ints_size() == 3);
                lp.set("dilation", parse(attribute_proto.ints()));
            }
            else if(attribute_name == "activations" && node_proto.op_type() == "LSTM")
            {
                lp.set(attribute_name, parseStr(attribute_proto.strings()));
            }
            else if (attribute_proto.has_i())
            {
                ::google::protobuf::int64 src = attribute_proto.i();
                if (src < std::numeric_limits<int32_t>::min() || src > std::numeric_limits<int32_t>::max())
                    CV_Error(Error::StsOutOfRange, "Input is out of OpenCV 32S range");
                else
                    lp.set(attribute_name, saturate_cast<int32_t>(src));
            }
            else if (attribute_proto.has_f())
            {
                lp.set(attribute_name, attribute_proto.f());
            }
            else if (attribute_proto.has_s())
            {
                lp.set(attribute_name, attribute_proto.s());
            }
            else if (attribute_proto.floats_size() > 0)
            {
                lp.set(attribute_name, DictValue::arrayReal(
                    attribute_proto.floats().data(), attribute_proto.floats_size()));
            }
            else if (attribute_proto.ints_size() > 0)
            {
                lp.set(attribute_name, parse(attribute_proto.ints()));
            }
            else if (attribute_proto.has_t())
            {
                opencv_onnx::TensorProto tensor = attribute_proto.t();
                Mat blob = getMatFromTensor(tensor);
                lp.blobs.push_back(blob);
                lp.set("original_dims_of_mat", tensor.dims_size());
            }
            else if (attribute_proto.has_g())
            {
                CV_Error(Error::StsNotImplemented, cv::format("DNN/ONNX/Attribute[%s]: 'Graph' is not supported", attribute_name.c_str()));
            }
            else if (attribute_proto.graphs_size() > 0)
            {
                CV_Error(Error::StsNotImplemented,
                        cv::format("DNN/ONNX/Attribute[%s]: 'Graphs' (%d) in attributes is not supported",
                                attribute_name.c_str(), attribute_proto.graphs_size())
                );
            }
            else if (attribute_proto.strings_size() > 0)
            {
                std::string msg = cv::format("DNN/ONNX/Attribute[%s]: 'Strings' (%d) are not supported",
                        attribute_name.c_str(), attribute_proto.strings_size());
                CV_LOG_ERROR(NULL, msg);
                for (int i = 0; i < attribute_proto.strings_size(); i++)
                {
                    CV_LOG_ERROR(NULL, "    Attribute[" << attribute_name << "].string(" << i << ") = '" << attribute_proto.strings(i) << "'");
                }
                CV_Error(Error::StsNotImplemented, msg);
            }
            else if (attribute_proto.tensors_size() > 0)
            {
                CV_Error(Error::StsNotImplemented,
                        cv::format("DNN/ONNX/Attribute[%s]: 'Tensors' (%d) in attributes are not supported",
                                attribute_name.c_str(), attribute_proto.tensors_size())
                );
            }
            else
            {
                CV_Error(Error::StsNotImplemented, cv::format("DNN/ONNX/Attribute[%s]: unsupported attribute format", attribute_name.c_str()));
            }
        }
        catch (const cv::Exception& e)
        {
            CV_UNUSED(e);
            if (DNN_DIAGNOSTICS_RUN)
            {
                CV_LOG_ERROR(NULL, "DNN/ONNX: Potential problem with processing attributes for node " << node_proto.name() << " Attribute " << attribute_name.c_str()
                );
                continue;
            }
            throw;
        }
    }
    return lp;
}

Mat ONNXImporter::getBlob(const opencv_onnx::NodeProto& node_proto, int index)
{
    CV_Assert(index < node_proto.input_size());
    const std::string& input_name = node_proto.input(index);
    return getBlob(input_name);
}

Mat ONNXImporter::getBlob(const std::string& input_name)
{
    std::map<std::string, Mat>::const_iterator constBlob = constBlobs.find(input_name);
    if (constBlob == constBlobs.end())
    {
        CV_Error(Error::StsBadArg, std::string("Blob ") + input_name + " not found in const blobs");
    }
    return constBlob->second;
}

Mat ONNXImporter::getIntBlob(const opencv_onnx::NodeProto& node_proto, int index)
{
    Mat blob = getBlob(node_proto, index);
    if (blob.depth() == CV_32S)
        return blob;
    if (blob.depth() == CV_64S) {
        Mat blobInt32;
        blob.convertTo(blobInt32, CV_32S);
        return blobInt32;
    }
    CV_Error(Error::BadDepth, "blob should have integer type");
    return Mat();
}

ONNXImporter::TensorInfo ONNXImporter::getBlobExtraInfo(const opencv_onnx::NodeProto &node_proto, int index)
{
    CV_Assert(index < node_proto.input_size());
    const std::string& input_name = node_proto.input(index);
    return getBlobExtraInfo(input_name);
}

ONNXImporter::TensorInfo ONNXImporter::getBlobExtraInfo(const std::string& input_name)
{
    std::map<std::string, TensorInfo>::const_iterator constBlobExtraInfo = constBlobsExtraInfo.find(input_name);
    if (constBlobExtraInfo == constBlobsExtraInfo.end())
    {
        CV_Error(Error::StsBadArg, std::string("Blob ") + input_name + " not found in const blobs of extra info");
    }
    return constBlobExtraInfo->second;
}

void ONNXImporter::addLayer(LayerParams& layerParams,
                            const opencv_onnx::NodeProto& node_proto,
                            int num_inputs)
{
    int depth = layerParams.get<int>("depth", CV_32F);
    int id = dstNet.addLayer(layerParams.name, layerParams.type, depth, layerParams);
    for (int i = 0; i < node_proto.output_size(); ++i)
    {
        const std::string& output_name = node_proto.output(i);
        if (!output_name.empty())
        {
            layer_id.insert(std::make_pair(output_name, LayerInfo(id, i, depth)));
        }
    }

    std::vector<MatShape> layerInpShapes, layerOutShapes, layerInternalShapes;
    int inpNum = 0;
    num_inputs = std::min(node_proto.input_size(), num_inputs);
    for (int j = 0; j < num_inputs; j++)
    {
        const std::string& input_name = node_proto.input(j);
        IterLayerId_t layerId = layer_id.find(input_name);
        if (layerId != layer_id.end()) {
            dstNet.connect(layerId->second.layerId, layerId->second.outputId, id, inpNum);
            ++inpNum;
            // Collect input shapes.
            IterShape_t shapeIt = outShapes.find(input_name);
            CV_Assert(shapeIt != outShapes.end());
            layerInpShapes.push_back(shapeIt->second);
        }
    }
    // Compute shape of output blob for this layer.
    Ptr<Layer> layer = dstNet.getLayer(id);  // FIXIT: avoid instantiation of layers during the import stage
    layer->getMemoryShapes(layerInpShapes, 0, layerOutShapes, layerInternalShapes);
    for (int i = 0; i < node_proto.output_size() && i < (int)layerOutShapes.size(); ++i)
    {
        const std::string& output_name = node_proto.output(i);
        if (!output_name.empty())
        {
            outShapes[node_proto.output(i)] = layerOutShapes[i];
        }
    }
}

void ONNXImporter::addConstant(const std::string& name, const Mat& blob)
{
    CV_LOG_DEBUG(NULL, "DNN/ONNX: add constant '" << name << "' shape=" << toString(shape(blob)) << ": " << toString(blob));
    constBlobs.insert(std::make_pair(name, blob));
    outShapes.insert(std::make_pair(name, shape(blob)));
}

void ONNXImporter::parseOperatorSet()
{
    int ir_version = model_proto.has_ir_version() ? static_cast<int>(model_proto.ir_version()) : -1;
    if (ir_version < 3)
        return;

    int opset_size = model_proto.opset_import_size();
    if (opset_size <= 0)
    {
        CV_LOG_INFO(NULL, "DNN/ONNX: missing opset information")
        return;
    }

    for (int i = 0; i < opset_size; ++i)
    {
        const ::opencv_onnx::OperatorSetIdProto& opset_entry = model_proto.opset_import(i);
        const std::string& domain = opset_entry.has_domain() ? opset_entry.domain() : std::string();
        int version = opset_entry.has_version() ? opset_entry.version() : -1;
        if (domain.empty() || domain == str_domain_ai_onnx)
        {
            // ONNX opset covered by specification: https://github.com/onnx/onnx/blob/master/docs/Operators.md
            onnx_opset = std::max(onnx_opset, version);
            onnx_opset_map[str_domain_ai_onnx] = onnx_opset;
        }
        else
        {
            CV_LOG_DEBUG(NULL, "DNN/ONNX: using non-standard ONNX opset[" << i << "]: domain='" << domain << "' version=" << version);
            onnx_opset_map[domain] = onnx_opset;
        }
    }

    CV_LOG_INFO(NULL, "DNN/ONNX: ONNX opset version = " << onnx_opset);

    buildDispatchMap_ONNX_AI(onnx_opset);
    for (const auto& pair : onnx_opset_map)
    {
        if (pair.first == str_domain_ai_onnx)
        {
            continue;  // done above
        }
        else if (pair.first == "com.microsoft")
        {
            buildDispatchMap_COM_MICROSOFT(pair.second);
        }
        else
        {
            CV_LOG_INFO(NULL, "DNN/ONNX: unknown domain='" << pair.first << "' version=" << pair.second << ". No dispatch map, you may need to register 'custom' layers.");
        }
    }
}

static bool ifInt8Output(const String& layerType)
{
    // Contains all node types whose output should be int8 when it get int8 input.
    // ai.onnx opset 15
    static std::vector<String> input8output8List = {
            "QuantizeLinear",
            "QLinearAdd",
            "QLinearMul",
            "QLinearAveragePool",
            "QLinearGlobalAveragePool",
            "QLinearLeakyRelu",
            "QLinearSigmoid",
            "QLinearConcat",
            "QGemm",
            "QLinearSoftmax",
            "QLinearConv",
            "QLinearMatMul",
            "MaxPool",
            "ReduceMax",
            "ReduceMin",
            "Split",
            "Clip",
            "Abs",
            "Transpose",
            "Squeeze",
            "Flatten",
            "Unsqueeze",
            "Expand",
            "Reshape",
            "Pad",
            "Gather",
            "Concat",
            "Resize",
            "SpaceToDepth",
            "DepthToSpace",
            "Pow",
            "Add",
            "Sub",
            "Mul",
            "Div"
    };
    auto layerIt = std::find(input8output8List.begin(), input8output8List.end(), layerType);
    return layerIt != input8output8List.end();
}

void ONNXImporter::setParamsDtype(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    // If the current layer should output the same data type as the input, and it's input type is int8, we think the current
    // layer should also output int8.

    // Check if the layer has int8 input.
    const std::string& layer_type = node_proto.op_type();
    for (int i = 0; i < node_proto.input_size(); ++i)
    {
        if (layer_id.find(node_proto.input(i)) != layer_id.end())
        {
            LayerInfo layerInfo = layer_id.find(node_proto.input(i))->second;

            if (layerInfo.depth == CV_8S && ifInt8Output(layer_type))
            {
                layerParams.set("depth", CV_8S);
                return;
            }
        }
    }
    layerParams.set("depth", CV_32F);
}

void ONNXImporter::populateNet()
{
    CV_Assert(model_proto.has_graph());
    graph_proto = model_proto.mutable_graph();

    std::string framework_version;
    if (model_proto.has_producer_name())
        framework_name = model_proto.producer_name();
    if (model_proto.has_producer_version())
        framework_version = model_proto.producer_version();

    CV_LOG_INFO(NULL, "DNN/ONNX: loading ONNX"
            << (model_proto.has_ir_version() ? cv::format(" v%d", (int)model_proto.ir_version()) : cv::String())
            << " model produced by '" << framework_name << "'"
            << (framework_version.empty() ? cv::String() : cv::format(":%s", framework_version.c_str()))
            << ". Number of nodes = " << graph_proto->node_size()
            << ", initializers = " << graph_proto->initializer_size()
            << ", inputs = " << graph_proto->input_size()
            << ", outputs = " << graph_proto->output_size()
            );

    parseOperatorSet();

    simplifySubgraphs(*graph_proto);

    const int layersSize = graph_proto->node_size();
    CV_LOG_DEBUG(NULL, "DNN/ONNX: graph simplified to " << layersSize << " nodes");

    constBlobs = getGraphTensors(*graph_proto);  // scan GraphProto.initializer
    std::vector<String> netInputs;  // map with network inputs (without const blobs)
    // Add all the inputs shapes. It includes as constant blobs as network's inputs shapes.
    for (int i = 0; i < graph_proto->input_size(); ++i)
    {
        const opencv_onnx::ValueInfoProto& valueInfoProto = graph_proto->input(i);
        CV_Assert(valueInfoProto.has_name());
        const std::string& name = valueInfoProto.name();
        CV_Assert(valueInfoProto.has_type());
        const opencv_onnx::TypeProto& typeProto = valueInfoProto.type();
        CV_Assert(typeProto.has_tensor_type());
        const opencv_onnx::TypeProto::Tensor& tensor = typeProto.tensor_type();
        CV_Assert(tensor.has_shape());
        const opencv_onnx::TensorShapeProto& tensorShape = tensor.shape();

        int dim_size = tensorShape.dim_size();
        CV_CheckGE(dim_size, 0, "");  // some inputs are scalars (dims=0), e.g. in Test_ONNX_nets.Resnet34_kinetics test
        MatShape inpShape(dim_size);
        for (int j = 0; j < dim_size; ++j)
        {
            const opencv_onnx::TensorShapeProto_Dimension& dimension = tensorShape.dim(j);
            if (dimension.has_dim_param())
            {
                CV_LOG_DEBUG(NULL, "DNN/ONNX: input[" << i << "] dim[" << j << "] = <" << dimension.dim_param() << "> (dynamic)");
            }
            // https://github.com/onnx/onnx/blob/master/docs/DimensionDenotation.md#denotation-definition
            if (dimension.has_denotation())
            {
                CV_LOG_INFO(NULL, "DNN/ONNX: input[" << i << "] dim[" << j << "] denotation is '" << dimension.denotation() << "'");
            }
            inpShape[j] = dimension.dim_value();
            // NHW, NCHW(NHWC), NCDHW(NDHWC); do not set this flag if only N is dynamic
            if (dimension.has_dim_param() && !(j == 0 && inpShape.size() >= 3))
            {
                hasDynamicShapes = true;
            }
        }
        bool isInitialized = ((constBlobs.find(name) != constBlobs.end()));
        CV_LOG_IF_DEBUG(NULL, !isInitialized, "DNN/ONNX: input[" << i << " as '" << name << "'] shape=" << toString(inpShape));
        CV_LOG_IF_VERBOSE(NULL, 0, isInitialized, "DNN/ONNX: pre-initialized input[" << i << " as '" << name << "'] shape=" << toString(inpShape));
        if (dim_size > 0 && !hasDynamicShapes)  // FIXIT result is not reliable for models with multiple inputs
        {
            inpShape[0] = std::max(inpShape[0], 1); // It's OK to have undetermined batch size
        }
        outShapes[valueInfoProto.name()] = inpShape;
        // fill map: push layer name, layer id and output id
        if (!isInitialized)
        {
            netInputs.push_back(name);
            layer_id.insert(std::make_pair(name, LayerInfo(0, netInputs.size() - 1)));
        }
    }

    dstNet.setInputsNames(netInputs);
    if (!hasDynamicShapes)
    {
        for (int i = 0; i < netInputs.size(); ++i)
            dstNet.setInputShape(netInputs[i], outShapes[netInputs[i]]);
    }

    // dump outputs
    for (int i = 0; i < graph_proto->output_size(); ++i)
    {
        dumpValueInfoProto(i, graph_proto->output(i), "output");
    }

    if (DNN_DIAGNOSTICS_RUN) {
        CV_LOG_INFO(NULL, "DNN/ONNX: start diagnostic run!");
        layerHandler->fillRegistry(*graph_proto);
    }

    for(int li = 0; li < layersSize; li++)
    {
        const opencv_onnx::NodeProto& node_proto = graph_proto->node(li);
        handleNode(node_proto);
    }

    // register outputs
    for (int i = 0; i < graph_proto->output_size(); ++i)
    {
        const std::string& output_name = graph_proto->output(i).name();
        if (output_name.empty())
        {
            CV_LOG_ERROR(NULL, "DNN/ONNX: can't register output without name: " << i);
            continue;
        }
        ConstIterLayerId_t layerIt = layer_id.find(output_name);
        if (layerIt == layer_id.end())
        {
            CV_LOG_ERROR(NULL, "DNN/ONNX: can't find layer for output name: '" << output_name << "'. Does model imported properly?");
            continue;
        }

        const LayerInfo& li = layerIt->second;
        int outputId = dstNet.registerOutput(output_name, li.layerId, li.outputId); CV_UNUSED(outputId);
        // no need to duplicate message from engine: CV_LOG_DEBUG(NULL, "DNN/ONNX: registered output='" << output_name << "' with id=" << outputId);
    }

    CV_LOG_DEBUG(NULL, (DNN_DIAGNOSTICS_RUN ? "DNN/ONNX: diagnostic run completed!" : "DNN/ONNX: import completed!"));
}

std::string ONNXImporter::getLayerTypeDomain(const opencv_onnx::NodeProto& node_proto)
{
    if (!node_proto.has_domain())
        return str_domain_ai_onnx;
    const std::string& domain = node_proto.domain();
    if (domain.empty())
        return str_domain_ai_onnx;
    return domain;
}

const ONNXImporter::DispatchMap& ONNXImporter::getDispatchMap(const opencv_onnx::NodeProto& node_proto)
{
    static DispatchMap empty_map;
    const std::string& layer_type_domain = getLayerTypeDomain(node_proto);
    auto it = domain_dispatch_map.find(layer_type_domain);
    if (it == domain_dispatch_map.end())
    {
        return empty_map;
    }

    return it->second;
}

std::string ONNXImporter::extractNodeName(const opencv_onnx::NodeProto& node_proto)
{
    // We need to rework DNN outputs API, this is a workaround for #21698
    if (node_proto.has_name() && !node_proto.name().empty())
    {
        if (useLegacyNames)
            return node_proto.name();
        return cv::format("onnx_node!%s", node_proto.name().c_str());
    }
    for (int i = 0; i < node_proto.output_size(); ++i)
    {
        const std::string& name = node_proto.output(i);
        // There are two ways to leave an optional input or output unspecified:
        // the first, available only for trailing inputs and outputs, is to simply not provide that input;
        // the second method is to use an empty string in place of an input or output name.
        if (!name.empty())
        {
            if (useLegacyNames)
                return name.c_str();
            return cv::format("onnx_node_output_%d!%s", i, name.c_str());
        }
    }
    CV_Error(Error::StsAssert, "Couldn't deduce Node name.");
}

void ONNXImporter::handleNode(const opencv_onnx::NodeProto& node_proto)
{
    CV_Assert(node_proto.output_size() >= 1);
    const std::string& name = extractNodeName(node_proto);
    const std::string& layer_type = node_proto.op_type();
    const std::string& layer_type_domain = getLayerTypeDomain(node_proto);
    const auto& dispatch = getDispatchMap(node_proto);

    CV_LOG_INFO(NULL, "DNN/ONNX: processing node with " << node_proto.input_size() << " inputs and "
                                                         << node_proto.output_size() << " outputs: "
                                                         << cv::format("[%s]:(%s)", layer_type.c_str(), name.c_str())
                                                         << cv::format(" from %sdomain='", onnx_opset_map.count(layer_type_domain) == 1 ? "" : "undeclared ")
                                                         << layer_type_domain << "'"
    );

    if (dispatch.empty())
    {
        CV_LOG_WARNING(NULL, "DNN/ONNX: missing dispatch map for domain='" << layer_type_domain << "'");
    }

    LayerParams layerParams;
    try
    {
        // FIXIT not all cases can be repacked into "LayerParams". Importer should handle such cases directly for each "layer_type"
        layerParams = getLayerParams(node_proto);

        layerParams.name = name;
        layerParams.type = layer_type;
        layerParams.set("has_dynamic_shapes", hasDynamicShapes);

        setParamsDtype(layerParams, node_proto);

        DispatchMap::const_iterator iter = dispatch.find(layer_type);
        if (iter != dispatch.end())
        {
            CALL_MEMBER_FN(*this, iter->second)(layerParams, node_proto);
        }
        else
        {
            parseCustomLayer(layerParams, node_proto);
        }
    }
    catch (const cv::Exception& e)
    {
        if (DNN_DIAGNOSTICS_RUN)
        {
            CV_LOG_ERROR(NULL, "DNN/ONNX: Potential problem during processing node with " << node_proto.input_size() << " inputs and " << node_proto.output_size() << " outputs: "
                    << cv::format("[%s]:(%s)", layer_type.c_str(), name.c_str())
                    << " from domain='" << layer_type_domain << "'"
                    << "\n" << e.msg
            );
            cv::AutoLock lock(getLayerFactoryMutex());
            auto registeredLayers = getLayerFactoryImpl();
            if (registeredLayers.find(layerParams.type) != registeredLayers.end())
            {
                try
                {
                    Ptr<Layer> layer = LayerFactory::createLayerInstance(layerParams.type, layerParams);
                }
                catch (const std::exception& e)
                {
                    CV_LOG_ERROR(NULL, "DNN/ONNX: Layer of type " << layerParams.type << "(" << layer_type << ") cannot be created with parameters " << layerParams << ". Error: " << e.what()
                    );
                }
            }
        }
        else
        {
            CV_LOG_ERROR(NULL, "DNN/ONNX: ERROR during processing node with " << node_proto.input_size() << " inputs and " << node_proto.output_size() << " outputs: "
                    << cv::format("[%s]:(%s)", layer_type.c_str(), name.c_str())
                    << " from domain='" << layer_type_domain << "'"
            );
        }
        for (int i = 0; i < node_proto.input_size(); i++)
        {
            CV_LOG_INFO(NULL, "    Input[" << i << "] = '" << node_proto.input(i) << "'");
        }
        for (int i = 0; i < node_proto.output_size(); i++)
        {
            CV_LOG_INFO(NULL, "    Output[" << i << "] = '" << node_proto.output(i) << "'");
        }
        if (DNN_DIAGNOSTICS_RUN)
        {
            for (int i = 0; i < node_proto.output_size(); ++i)
            {
                layer_id.insert(std::make_pair(node_proto.output(i), LayerInfo(0, i)));
                outShapes[node_proto.output(i)] = outShapes[node_proto.input(0)];
            }
        }
        else
            CV_Error(Error::StsError, cv::format("Node [%s@%s]:(%s) parse error: %s", layer_type.c_str(), layer_type_domain.c_str(), name.c_str(), e.what()));
    }
}

void ONNXImporter::parseArg(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    const std::string& layer_type = node_proto.op_type();
    layerParams.type = "Arg";
    layerParams.set("op", layer_type == "ArgMax" ? "max" : "min");
    addLayer(layerParams, node_proto);
}

void setCeilMode(LayerParams& layerParams)
{
    // auto_pad attribute is deprecated and uses ceil
    if (layerParams.has("pad_mode"))
    {
        layerParams.set("ceil_mode", true);
    }
    else if (!layerParams.has("ceil_mode"))
    {
        layerParams.set("ceil_mode", false);
    }
}

void ONNXImporter::parseMaxUnpool(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    layerParams.type = "MaxUnpool";

    DictValue kernel_shape = layerParams.get("kernel_size");
    CV_Assert(kernel_shape.size() == 2);
    layerParams.set("pool_k_w", kernel_shape.get<int>(0));
    layerParams.set("pool_k_h", kernel_shape.get<int>(1));

    int pool_pad_w = 0, pool_pad_h = 0;
    if (layerParams.has("pad"))
    {
        DictValue pads = layerParams.get("pad");
        CV_CheckEQ(pads.size(), 2, "");
        pool_pad_w = pads.get<int>(0);
        pool_pad_h = pads.get<int>(1);
    }
    layerParams.set("pool_pad_w", pool_pad_w);
    layerParams.set("pool_pad_h", pool_pad_h);


    int pool_stride_w = 1, pool_stride_h = 1;
    if (layerParams.has("stride"))
    {
        DictValue strides = layerParams.get("stride");
        CV_CheckEQ(strides.size(), 2, "");
        pool_stride_w = strides.get<int>(0);
        pool_stride_h = strides.get<int>(1);
    }
    layerParams.set("pool_stride_w", pool_stride_w);
    layerParams.set("pool_stride_h", pool_stride_h);

    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseMaxPool(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    int depth = layerParams.get<int>("depth", CV_32F);
    layerParams.type = (depth == CV_8S) ? "PoolingInt8" : "Pooling";
    layerParams.set("pool", "MAX");
    setCeilMode(layerParams);
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseAveragePool(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    layerParams.type = "Pooling";
    layerParams.set("pool", "AVE");
    setCeilMode(layerParams);
    layerParams.set("ave_pool_padded_area", framework_name == "pytorch");
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseGlobalPool(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto_)
{
    opencv_onnx::NodeProto node_proto = node_proto_;
    const std::string& layer_type = node_proto.op_type();
    const std::string output_name = node_proto.output(0);

    CV_Assert(node_proto.input_size() == 1);
    layerParams.type = "Pooling";
    String pool;
    if (layer_type == "GlobalMaxPool")
        pool = "MAX";
    else if (layer_type == "GlobalAveragePool")
        pool = "AVE";
    else
        CV_Error(Error::StsNotImplemented, "Unsupported Pooling type of " + layer_type + " operation.");

    CV_Assert(!layerParams.has("axes"));
    layerParams.set("global_pooling", true);
    layerParams.set("pool", pool);
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseReduce(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    layerParams.type = "Reduce";
    const auto& op_type = node_proto.op_type();
    String reduce_type;
    if (op_type == "ReduceMax")
        reduce_type = "MAX";
    else if (op_type == "ReduceMean")
        reduce_type = "MEAN";
    else if (op_type == "ReduceMin")
        reduce_type = "MIN";
    else if (op_type == "ReduceProd")
        reduce_type = "PROD";
    else if (op_type == "ReduceSum")
        reduce_type = "SUM";
    else if (op_type == "ReduceL1")
        reduce_type = "L1";
    else if (op_type == "ReduceL2")
        reduce_type = "L2";
    else if (op_type == "ReduceLogSum")
        reduce_type = "LOG_SUM";
    else if (op_type == "ReduceLogSumExp")
        reduce_type = "LOG_SUM_EXP";
    else if (op_type == "ReduceSumSquare")
        reduce_type = "SUM_SQUARE";
    else
        CV_Error(Error::StsNotImplemented, "DNN/ONNX: " + op_type + " is not supported.");
    layerParams.set("reduce", reduce_type);

    int num_inputs = node_proto.input_size();
    CV_Check(num_inputs, num_inputs >= 1 && num_inputs <= 2, "DNN/ONNX: Reduce layers should have at least one input and at most two inputs");

    if (num_inputs >= 2)
        CV_CheckTrue(constBlobs.find(node_proto.input(1)) != constBlobs.end(), "Reduce layer doesn't support non contant axes");

    // "axes" is turned to one of the inputs since opset 18,
    // except for ReduceSum, which has "axes" input since opset 13.
    if (!layerParams.has("axes") && num_inputs == 2 && constBlobs.find(node_proto.input(1)) != constBlobs.end()) {
        Mat mat_axes = getIntBlob(node_proto, 1);
        int num_axes = (int)mat_axes.total();
        std::vector<int> axes(num_axes);
        for (int i = 0; i < num_axes; ++i)
            axes[i] = mat_axes.at<int>(i);
        layerParams.set("axes", DictValue::arrayInt(&axes[0], num_axes));
        if (constBlobs.find(node_proto.input(0)) != constBlobs.end()){
            std::vector<Mat> inputs, output;
            inputs.push_back(getBlob(node_proto, 0));
            runLayer(layerParams, inputs, output);
            CV_Assert(output.size() == 1);
            addConstant(node_proto.output(0), output[0]);
            return;
        }
    }

    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseSlice(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    MatShape inpShape;
    if (constBlobs.find(node_proto.input(0)) != constBlobs.end())
        inpShape = shape(getBlob(node_proto, 0));
    else {
        inpShape = outShapes[node_proto.input(0)];
    }
    int dims = inpShape.size();
    std::vector<int> begin(dims, 0);
    std::vector<int> end(dims, INT_MAX);
    std::vector<int> steps;
    int inp_size = node_proto.input_size();
    int axis = 0;
    bool has_axes = false;
    DictValue starts_, ends_, axes_, steps_;

    // opset = 1
    if (inp_size == 1)
    {
        starts_ = layerParams.get("starts");
        ends_ = layerParams.get("ends");
        CV_Assert(starts_.size() == ends_.size());
        if (layerParams.has("axes"))
        {
            axes_ = layerParams.get("axes");
            CV_Assert(axes_.size() == starts_.size());
            axis = axes_.getIntValue(0) < 0 ? axes_.getIntValue(0) + dims : axes_.getIntValue(0);
            has_axes = true;
        }
    }
    // opset > 1
    else
    {
        CV_Assert(inp_size >= 3);
        for (int i = 1; i < inp_size; ++i)
        {
            CV_Assert(constBlobs.find(node_proto.input(i)) != constBlobs.end());
        }
        Mat start_blob = getIntBlob(node_proto, 1);
        Mat end_blob = getIntBlob(node_proto, 2);
        CV_Assert(start_blob.total() == end_blob.total());
        starts_ = DictValue::arrayInt(start_blob.begin<int>(), start_blob.total());
        ends_ = DictValue::arrayInt(end_blob.begin<int>(), end_blob.total());

        if (inp_size > 3 && !getBlob(node_proto, 3).empty())
        {
            Mat axes_blob = getIntBlob(node_proto, 3);
            CV_Assert(axes_blob.total() == start_blob.total());
            axes_ = DictValue::arrayInt(axes_blob.begin<int>(), axes_blob.total());
            axis = axes_.getIntValue(0) < 0 ? axes_.getIntValue(0) + dims : axes_.getIntValue(0);
            has_axes = true;
        }

        if (inp_size == 5 && !getBlob(node_proto, 4).empty())
        {
            Mat step_blob = getIntBlob(node_proto, 4);
            CV_Assert(step_blob.total() == start_blob.total());
            steps_ = DictValue::arrayInt(step_blob.begin<int>(), step_blob.total());
            steps.resize(dims, 1);

            // Very strange application for Slice op with tensor reversing.
            // We just workaround it for 2d constants.
            if (constBlobs.find(node_proto.input(0)) != constBlobs.end() &&
                axis == 0 &&
                start_blob.at<int>(0) == -1 && step_blob.at<int>(0) == -1 &&
                end_blob.at<int>(0) == std::numeric_limits<int32_t>::min())
            {
                Mat inp = getBlob(node_proto, 0);
                if (inp.dims == 2)
                {
                    Mat flipped;
                    flip(inp, flipped, 0);
                    addConstant(node_proto.output(0), flipped);
                    return;
                }
            }
        }
    }

    if (!has_axes)
    {
        // make a default axes [0, 1, 2...]
        Mat axes_tmp(1, starts_.size(), CV_32S);
        std::iota(axes_tmp.begin<int>(), axes_tmp.end<int>(), 0);
        axes_ = DictValue::arrayInt(axes_tmp.begin<int>(), axes_tmp.total());
    }

    int cur_axe;
    std::vector<bool> flag(dims, false);
    Mat axes(1, starts_.size(), CV_32S);
    auto axes_ptr = axes.ptr<int>();
    // resize begin and end
    for (int i = 0; i < axes_.size(); ++i)
    {
        // dims should be added to the negative axes
        cur_axe = axes_.getIntValue(i) < 0 ? axes_.getIntValue(i) + dims : axes_.getIntValue(i);
        CV_CheckGE(cur_axe, 0, "Axes should be grater or equal to '-dims'.");
        CV_CheckLT(cur_axe, dims, "Axes should be less than 'dim'.");
        CV_CheckEQ(flag[cur_axe], false, "Axes shouldn't have duplicated values.");
        flag[cur_axe] = true;
        // change axis to the minimum axe
        if (cur_axe < axis) axis = cur_axe;
        axes_ptr[i] = cur_axe;
        begin[cur_axe] = starts_.getIntValue(i);
        end[cur_axe] = ends_.getIntValue(i);
    }

    layerParams.set("begin", DictValue::arrayInt(&begin[0], begin.size()));
    layerParams.set("end", DictValue::arrayInt(&end[0], end.size()));
    layerParams.set("axis", axis);

    if (!steps.empty())
    {
        for (int i = 0; i < axes.total(); ++i)
            steps[axes_ptr[i]] = steps_.getIntValue(i);
        layerParams.set("steps", DictValue::arrayInt(&steps[0], steps.size()));
    }

    if (constBlobs.find(node_proto.input(0)) != constBlobs.end())
    {
        Mat inp = getBlob(node_proto, 0);
        std::vector<Mat> inputs, sliced;
        inputs.push_back(inp);
        runLayer(layerParams, inputs, sliced);
        CV_Assert(sliced.size() == 1);
        addConstant(node_proto.output(0), sliced[0]);
        return;
    }
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseSplit(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    int axis = layerParams.get<int>("axis", 0);
    MatShape inpShape = outShapes[node_proto.input(0)];
    axis = normalize_axis(axis, inpShape.size());

    if (layerParams.has("split"))
    {
        DictValue splits = layerParams.get("split");
        const int numSplits = splits.size();

        if (numSplits == 1)
        {
            layerParams.set("num_split", 1);
        }
        else
        {
            CV_Assert(numSplits >= 1);

            std::vector<int> slicePoints(numSplits - 1, splits.get<int>(0));
            for (int i = 1; i < splits.size() - 1; ++i)
            {
                slicePoints[i] = slicePoints[i - 1] + splits.get<int>(i);
            }
            layerParams.set("slice_point", DictValue::arrayInt(&slicePoints[0], slicePoints.size()));
        }
    }
    else if (node_proto.input_size() == 2) // opset >= 13, the split will be stored at the second input, instead of the attribute.
    {
        CV_Assert(constBlobs.find(node_proto.input(1)) != constBlobs.end());
        Mat splitsBlob = getIntBlob(node_proto, 1);
        int splitSize = splitsBlob.total();
        if (splitSize == 1)
        {
            layerParams.set("num_split", 1);
        }
        else
        {
            std::vector<int> slicePoints(splitSize - 1, splitsBlob.at<int>(0));
            for (int i = 1; i < splitSize - 1; ++i)
            {
                slicePoints[i] = slicePoints[i - 1] + splitsBlob.at<int>(i);
            }
            layerParams.set("slice_point", DictValue::arrayInt(&slicePoints[0], slicePoints.size()));
        }
    }
    else
    {
        layerParams.set("num_split", node_proto.output_size());
    }
    int depth = layerParams.get<int>("depth", CV_32F);
    layerParams.type = (depth == CV_8S) ? "SliceInt8" : "Slice";
    layerParams.set("axis", axis);
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseNeg(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    layerParams.type = "Power";
    layerParams.set("scale", -1);
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseConstant(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    CV_Assert(node_proto.input_size() == 0);
    CV_Assert(layerParams.blobs.size() == 1);
    if (layerParams.has("original_dims_of_mat")) {
        int original_dims_of_mat = layerParams.get<int>("original_dims_of_mat");
        if (original_dims_of_mat == 0) {
            Mat& blob = layerParams.blobs[0];
            CV_Assert(blob.dims <= 2 && blob.total() == 1);
            blob = blob.reshape(1, 0, 0);
        }
        // add constant for constBlobsExtraInfo
        constBlobsExtraInfo.insert(std::make_pair(node_proto.output(0), TensorInfo(original_dims_of_mat)));
    }
    addConstant(node_proto.output(0), layerParams.blobs[0]);
}

void transformBlobs(std::vector<Mat>& blobs)
{
    Mat Wx = blobs[0];
    Mat Wh = blobs[1];
    Mat b = blobs[2];
    std::vector<Mat> cudaWorkaround;
    cudaWorkaround.push_back(Wx.clone());
    cudaWorkaround.push_back(Wh.clone());
    cudaWorkaround.push_back(b.clone());

    const int numHidden = Wh.size[2];

    Mat h0, c0;
    // check weather input is dynamic or not: hx, cx are given by user.
    // Resahpe if only they are given
    if (!blobs[3].empty()){
        h0 = blobs[3];
        h0 = h0.reshape(1, h0.size[0] * h0.size[1]);
    }
    if (!blobs[4].empty()){
        c0 = blobs[4];
        c0 = c0.reshape(1, c0.size[0] * c0.size[1]);
    }

    b = b.reshape(1, b.size[0]);
    Mat bx = b.colRange(0, b.cols / 2);
    Mat bh = b.colRange(b.cols / 2, b.cols);
    b = bx + bh;

    auto toIFOC = [] (Mat& in) {
        int first = in.size[0];
        int rest = in.total() / first / 4;
        // every weight blob contains weights for Input, Output, Forget and Cell gates
        Mat m = in.reshape(1, {first, 4, rest});
        Mat outputGate = m.col(1);
        Mat forgetGate = m.col(2);
        std::swap_ranges(outputGate.begin<float>(), outputGate.end<float>(), forgetGate.begin<float>());
    };

    toIFOC(Wx);
    toIFOC(Wh);
    toIFOC(b);

    Wx = Wx.reshape(1, Wx.size[0] * Wx.size[1]);
    Wh = Wh.reshape(1, Wh.size[0] * Wh.size[1]);

    blobs[0] = Wh;
    blobs[1] = Wx;
    blobs[2] = b.reshape(1, 1);

    if (!blobs[3].empty()){
        blobs[3] = h0;
    }
    if (!blobs[4].empty()){
        blobs[4] = c0;
    }

    if (blobs.size() == 5) {
        // so that future patch removing copies can leave all indexing as is
        blobs.insert(blobs.begin(), cudaWorkaround.begin(), cudaWorkaround.end());
        return;
    }

    Mat P = blobs[5];
    blobs[5] = P.colRange(0, numHidden);
    blobs[5] = blobs[5].clone().reshape(1, blobs[5].total());  // Single column.
    blobs[5] = Mat::diag(blobs[5]);

    blobs.push_back(P.colRange(numHidden, 2 * numHidden));
    blobs[6] = blobs[6].clone().reshape(1, blobs[6].total());  // Single column.
    blobs[6] = Mat::diag(blobs[6]);

    blobs.push_back(P.colRange(2 * numHidden, 3 * numHidden));
    blobs[7] = blobs[7].clone().reshape(1, blobs[7].total());  // Single column.
    blobs[7] = Mat::diag(blobs[7]);

    // so that future patch removing copies can leave all indexing as is
    blobs.insert(blobs.begin(), cudaWorkaround.begin(), cudaWorkaround.end());
}

void ONNXImporter::lstm_extractConsts(LayerParams& layerParams, const opencv_onnx::NodeProto& lstm_proto, size_t idx, int* blobShape_, int size)
{
        MatShape blobShape(blobShape_, blobShape_ + size);
        Mat blob;
        if (idx < lstm_proto.input_size() && !lstm_proto.input(idx).empty())
        {
            if ((idx == 5 || idx == 6) && (constBlobs.find(lstm_proto.input(idx)) == constBlobs.end()))
            {
                blob = Mat();
            }
            else
            {
                blob = getBlob(lstm_proto, idx);
                CV_Assert(shape(blob) == blobShape);
            }
        }
        else
        {
            blob = Mat(blobShape, CV_32FC1, 0.);
        }
        layerParams.blobs.push_back(blob);
}

void ONNXImporter::lstm_add_reshape(const std::string& input_name, const std::string& output_name, int* layerShape, size_t n)
{
    LayerParams reshapeLp;
    reshapeLp.name = cv::format("%s/reshape", input_name.c_str());
    reshapeLp.type = "Reshape";
    CV_Assert(layer_id.find(reshapeLp.name) == layer_id.end());

    reshapeLp.set("dim", DictValue::arrayInt(layerShape, n));

    opencv_onnx::NodeProto reshape_proto;
    reshape_proto.add_input(input_name);
    reshape_proto.add_output(output_name);
    addLayer(reshapeLp, reshape_proto);
}

std::string ONNXImporter::lstm_add_slice(int index, const std::string& input_name, int* begin, int* end, size_t n)
{
    LayerParams sliceLP;
    sliceLP.name = cv::format("%s/slice_%d", input_name.c_str(), index);
    sliceLP.type = "Slice";
    CV_Assert(layer_id.find(sliceLP.name) == layer_id.end());

    sliceLP.set("begin", DictValue::arrayInt(begin, n));
    sliceLP.set("end", DictValue::arrayInt(end, n));
    sliceLP.set("axis", 0);

    opencv_onnx::NodeProto slice_proto;
    slice_proto.add_input(input_name);
    slice_proto.add_output(sliceLP.name);
    addLayer(sliceLP, slice_proto);

    return slice_proto.output(0);
}

std::string ONNXImporter::lstm_fix_dims(LayerParams& layerParams, const opencv_onnx::NodeProto& lstm_proto,
                                        int batch_size, int num_directions, int hidden_size, bool need_y, const std::string& y_name,
                                        const int index)
{
    std::string reshape_output = cv::format("%s/reshape_%d", layerParams.name.c_str(), index);

    // reshape from Seq, Batch, Dirs*Hidden to Seq, Batch, Dirs, Hidden
    // to not confuse reshape with dynamic first dimension, zero means 'leave unchanged'
    int layerShape[] = {0, batch_size, num_directions, hidden_size};
    lstm_add_reshape(lstm_proto.output(index), reshape_output, layerShape, sizeof(layerShape) / sizeof(layerShape[0]));

    // permute from Seq, Batch, Dirs, Hidden to Seq, Dirs, Batch, Hidden
    LayerParams permuteLP;
    permuteLP.name = reshape_output + "/permute";
    permuteLP.type = "Permute";
    CV_Assert(layer_id.find(permuteLP.name) == layer_id.end());

    int order[] = {0, 2, 1, 3};
    permuteLP.set("order", DictValue::arrayInt(order, 4));

    opencv_onnx::NodeProto permute_proto;
    permute_proto.add_input(reshape_output);
    permute_proto.add_output((need_y && index == 0) ? y_name : static_cast<std::string>(permuteLP.name));
    addLayer(permuteLP, permute_proto);

    return permute_proto.output(0);
}

void ONNXImporter::lstm_add_transform(int num_directions, int batch_size, int hidden_size,
                                      int index, const std::string& input_name, const std::string& output_name)
{
    if (num_directions == 1)
    {
        // Slice: Yh = Y[-1, :, :, :]
        int begin[] = {-1}, end[] = {INT_MAX};
        std::string slice_output = lstm_add_slice(index, input_name, begin, end, sizeof(begin) / sizeof(begin[0]));

        // Reshape: 1x1xBxH -> 1xBxH
        int layerShape[] = {1, batch_size, hidden_size};
        lstm_add_reshape(slice_output, output_name, layerShape, sizeof(layerShape) / sizeof(layerShape[0]));
    }
    else
    {
        // Slice: SxDxBxH -> last sequence, first direction
        int begin0[] = {-1, 0}, end0[] = {INT_MAX, 1};
        std::string slice_0 = lstm_add_slice(0, input_name, begin0, end0, sizeof(begin0) / sizeof(begin0[0]));

        // Slice: SxDxBxH -> first sequence, last direction
        int begin1[] = {0, -1}, end1[] = {1, INT_MAX};
        std::string slice_1 = lstm_add_slice(1, input_name, begin1, end1, sizeof(begin1) / sizeof(begin1[0]));

        LayerParams concatLP;
        concatLP.name = cv::format("%s/concat", input_name.c_str());
        concatLP.type = "Concat";
        CV_Assert(layer_id.find(concatLP.name) == layer_id.end());

        concatLP.set("axis", 1); // 1x1xBxH -> 1x2xBxH

        opencv_onnx::NodeProto concat_proto;
        concat_proto.add_input(slice_0);
        concat_proto.add_input(slice_1);
        concat_proto.add_output(concatLP.name);
        addLayer(concatLP, concat_proto);

        // Reshape: 1x2xBxH -> 2xBxH
        int layerShape[] = {2, batch_size, hidden_size};
        lstm_add_reshape(concat_proto.output(0), output_name, layerShape, sizeof(layerShape) / sizeof(layerShape[0]));
    }
}

void ONNXImporter::parseLSTM(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto_)
{
    opencv_onnx::NodeProto lstm_proto = node_proto_;
    layerParams.name += "/lstm";

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#LSTM
    CV_Assert(lstm_proto.input_size() >= 3);
    for (size_t i = 1; i < 3; ++i)
    {
        const std::string& name = lstm_proto.input(i);
        CV_Assert(!name.empty() && constBlobs.count(name) == 1);
    }

    IterShape_t shapeIt = outShapes.find(lstm_proto.input(0));
    CV_Assert(shapeIt != outShapes.end());
    const MatShape x_shape = shapeIt->second;

    //if layout is 1, change batch and sequence dims
    const int layout = layerParams.get<int>("layout", 0);
    int batch_size, seq_length;
    if (layout == 1){
        batch_size = x_shape[0];
        seq_length = x_shape[1];
    }else{
        seq_length = x_shape[0];
        batch_size = x_shape[1];
    }
    const int input_size = x_shape[2];
    const int hidden_size = layerParams.get<int>("hidden_size");
    const int num_directions = constBlobs[lstm_proto.input(1)].size[0];

    int w_size[] = {num_directions, 4*hidden_size, input_size};
    lstm_extractConsts(layerParams, lstm_proto, 1, w_size, sizeof(w_size) / sizeof(w_size[0])); // W

    int r_size[] =  {num_directions, 4*hidden_size, hidden_size};
    lstm_extractConsts(layerParams, lstm_proto, 2, r_size, sizeof(r_size) / sizeof(r_size[0])); // R

    int b_size[] = {num_directions, 8*hidden_size};
    lstm_extractConsts(layerParams, lstm_proto, 3, b_size, sizeof(b_size) / sizeof(b_size[0])); // B

    if (4 < lstm_proto.input_size() && !lstm_proto.input(4).empty())
    {
        Mat blob = getIntBlob(lstm_proto, 4);
        CV_Assert(blob.total() == batch_size);
        for (MatIterator_<int32_t> it = blob.begin<int32_t>(); it != blob.end<int32_t>(); ++it)
        {
            CV_Assert(*it == seq_length);
        }
    }

    int h_size[] = {num_directions, batch_size, hidden_size};
    lstm_extractConsts(layerParams, lstm_proto, 5, h_size, sizeof(h_size) / sizeof(h_size[0])); // initial_h

    int c_size[] = {num_directions, batch_size, hidden_size};
    lstm_extractConsts(layerParams, lstm_proto, 6, c_size, sizeof(c_size) / sizeof(c_size[0])); // initial_c

    if (lstm_proto.input_size() > 7 && !lstm_proto.input(7).empty())
    {
        layerParams.set("use_peephole", true);
        int p_size[] = {num_directions, 3 * hidden_size};
        lstm_extractConsts(layerParams, lstm_proto, 7, p_size, sizeof(p_size) / sizeof(p_size[0])); // P
    }

    transformBlobs(layerParams.blobs);

    layerParams.set("is_onnx", true);
    layerParams.set("reverse", layerParams.get<String>("direction", "") == "reverse");
    layerParams.set("bidirectional", layerParams.get<String>("direction", "") == "bidirectional");

    bool need_yc = lstm_proto.output_size() > 2 && !lstm_proto.output(2).empty();
    bool need_yh = lstm_proto.output_size() > 1 && !lstm_proto.output(1).empty();
    bool need_y = lstm_proto.output_size() > 0 && !lstm_proto.output(0).empty();

    const std::string y_name = need_y ? lstm_proto.output(0) : "";
    const std::string yh_name = need_yh ? lstm_proto.output(1) : "";
    const std::string yc_name = need_yc ? lstm_proto.output(2) : "";

    layerParams.set("produce_cell_output", need_yc);

    lstm_proto.clear_output();
    if (need_y || need_yh)
    {
        // give random names to LSTMLayer's outputs because every output needs postprocessing
        lstm_proto.add_output(cv::format("%s_y", layerParams.name.c_str()));
    }
    if (need_yc)
    {
        lstm_proto.add_output(yc_name);
    }

    addLayer(layerParams, lstm_proto);

    std::string y_output = lstm_fix_dims(layerParams, lstm_proto, batch_size, num_directions, hidden_size, need_y,
                                         y_name, 0);
    if (need_yh)
    {
        lstm_add_transform(num_directions, batch_size, hidden_size, 0, y_output, yh_name);
    }
}

void ONNXImporter::parseGRU(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto_)
{
    opencv_onnx::NodeProto node_proto = node_proto_;
    const std::string output_name = node_proto.output(0);
    LayerParams gruParams = layerParams;
    gruParams.name += "/gru";

    // https://pytorch.org/docs/stable/generated/torch.nn.GRU.html?highlight=gru#
    CV_Assert(node_proto.input_size() == 6);
    Mat Wx = getBlob(node_proto, 1);
    Mat Wh = getBlob(node_proto, 2);
    Mat b = getBlob(node_proto, 3);
    Mat h0 = getBlob(node_proto, 5);

    Wx = Wx.reshape(1, Wx.size[0] * Wx.size[1]);
    Wh = Wh.reshape(1, Wh.size[0] * Wh.size[1]);
    h0 = h0.reshape(1, h0.size[0] * h0.size[1]);
    b = b.reshape(1, b.size[0]);

    gruParams.blobs.resize(4);
    gruParams.blobs[0] = Wh;
    gruParams.blobs[1] = Wx;
    gruParams.blobs[2] = b;
    gruParams.blobs[3] = h0;
    gruParams.set("bidirectional", gruParams.get<String>("direction", "") == "bidirectional");

    node_proto.set_output(0, gruParams.name);  // set different name so output shapes will be registered on that name
    addLayer(gruParams, node_proto);

    MatShape gruShape = outShapes[node_proto.output(0)];

    // Add fake 1 as it is done in ONNX
    gruShape.insert(gruShape.begin() + 1, 1);

    layerParams.type = "Reshape";
    layerParams.set("dim", DictValue::arrayInt(&gruShape[0], gruShape.size()));
    node_proto.set_input(0, gruParams.name);  // redirect input to GRU
    node_proto.set_output(0, output_name);  // keep origin GRU's name
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseImageScaler(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    const float scale = layerParams.has("scale") ? layerParams.get<float>("scale") : 1.0f;
    layerParams.erase("scale");

    if (layerParams.has("bias"))
    {
        layerParams.type = "Scale";
        layerParams.blobs.push_back(
                Mat(Size(1,  layerParams.get("bias").size()), CV_32FC1, scale));

        layerParams.set("bias_term", true);
        Mat bias(1, layerParams.get("bias").size(), CV_32FC1);
        for (int j = 0; j < bias.total(); j++) {
            bias.at<float>(0, j) = layerParams.get("bias").getRealValue(j);
        }
        layerParams.blobs.push_back(bias);
        layerParams.erase("bias");
    }
    else {
        layerParams.set("scale", scale);
        layerParams.type = "Power";
    }
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseClip(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    layerParams.type = "ReLU6";
    float min_value = -FLT_MAX, max_value = FLT_MAX;
    int input_size = node_proto.input_size();
    CV_Check(input_size, 1 <= input_size && input_size <= 3, "");

    if (input_size >= 2 && !node_proto.input(1).empty())
    {
        if (constBlobs.find(node_proto.input(1)) != constBlobs.end())
            min_value = getBlob(node_proto, 1).at<float>(0);
        else
            CV_Error(Error::StsNotImplemented, "Non-constant min values in Clip are not supported");
    }

    if (input_size == 3 && !node_proto.input(2).empty())
    {
        if (constBlobs.find(node_proto.input(2)) != constBlobs.end())
            max_value = getBlob(node_proto, 2).at<float>(0);
        else
            CV_Error(Error::StsNotImplemented, "Non-constant max values in Clip are not supported");
    }

    layerParams.set("min_value", layerParams.get<float>("min", min_value));
    layerParams.set("max_value", layerParams.get<float>("max", max_value));
    addLayer(layerParams, node_proto, 1);
}

void ONNXImporter::parseLeakyRelu(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    layerParams.type = "ReLU";
    layerParams.set("negative_slope", layerParams.get<float>("alpha", 0.01));
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseRelu(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    layerParams.type = "ReLU";
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseElu(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    layerParams.type = "ELU";
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseTanh(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    layerParams.type = "TanH";
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseAbs(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    layerParams.type = "AbsVal";
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parsePRelu(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    layerParams.type = "PReLU";
    layerParams.blobs.push_back(getBlob(node_proto, 1));
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseLRN(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    replaceLayerParam(layerParams, "size", "local_size");
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseInstanceNormalization(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto) {
    int num_inputs = node_proto.input_size();
    CV_CheckEQ(num_inputs, 3, "DNN/ONNXImporter - InstanceNorm: three inputs are required");

    bool found_input = constBlobs.find(node_proto.input(0)) != constBlobs.end();
    bool found_scale = constBlobs.find(node_proto.input(1)) != constBlobs.end();
    bool found_bias = constBlobs.find(node_proto.input(2)) != constBlobs.end();

    if (found_input && found_scale && found_bias) {
        std::vector<Mat> inputs, output;

        Mat input = getBlob(node_proto, 0);
        Mat scale = getBlob(node_proto, 1);
        Mat bias = getBlob(node_proto, 2);
        inputs.push_back(input);
        inputs.push_back(scale);
        inputs.push_back(bias);

        runLayer(layerParams, inputs, output);
        addConstant(node_proto.output(0), output[0]);
    } else {
        auto add_const_node = [&] (int i) {
            LayerParams const_params;
            const_params.name = node_proto.input(i);
            const_params.type = "Const";
            Mat blob = getBlob(node_proto, i);
            const_params.blobs.push_back(blob);

            opencv_onnx::NodeProto proto;
            proto.add_output(const_params.name);
            addLayer(const_params, proto);
        };
        if (found_input && layer_id.find(node_proto.input(0)) == layer_id.end()) { add_const_node(0); }
        if (found_scale && layer_id.find(node_proto.input(1)) == layer_id.end()) { add_const_node(1); }
        if (found_bias && layer_id.find(node_proto.input(2)) == layer_id.end()) { add_const_node(2); }
        addLayer(layerParams, node_proto);
    }
}

void ONNXImporter::parseBatchNormalization(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    if (node_proto.input_size() != 5)
        CV_Error(Error::StsNotImplemented,
                 "Expected input, scale, bias, mean and var");

    layerParams.type = "BatchNorm";
    replaceLayerParam(layerParams, "epsilon", "eps");
    replaceLayerParam(layerParams, "spatial", "use_global_stats");

    Mat meanData = getBlob(node_proto, 3);
    Mat stdData =  getBlob(node_proto, 4);

    layerParams.blobs.push_back(meanData);
    layerParams.blobs.push_back(stdData);

    if (!node_proto.input(1).empty()) {
        layerParams.set("has_weight", true);
        layerParams.blobs.push_back(getBlob(node_proto, 1));  // weightData
    } else {
        layerParams.set("has_weight", false);
    }

    if (!node_proto.input(2).empty()) {
        layerParams.set("has_bias", true);
        layerParams.blobs.push_back(getBlob(node_proto, 2)); // biasData
    } else {
        layerParams.set("has_bias", false);
    }
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseGemm(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto_)
{
    auto node_proto = node_proto_;
    layerParams.type = "Gemm";
    CV_CheckGE(node_proto.input_size(), 2, "DNN/ONNXImporter: Gemm requires at least two inputs");
    CV_CheckLE(node_proto.input_size(), 3, "DNN/ONNXImporter: Gemm have at most three inputs.");

    for (int i = 0; i < node_proto.input_size(); ++i) {
        if (i == 2) {
            layerParams.set("have_bias", true);
        }
        if (constBlobs.find(node_proto.input(i)) == constBlobs.end()) {
            continue;
        }

        if (i == 2 && constBlobsExtraInfo.find(node_proto.input(2)) != constBlobsExtraInfo.end()) {
            layerParams.set("real_ndims_C", getBlobExtraInfo(node_proto, 2).real_ndims);
        }

        Mat blob = getBlob(node_proto, i);

        if (i == 0) { // A, always as inputs without prepacking
            LayerParams const_A_params;
            const_A_params.name = layerParams.name + "/const_A";
            const_A_params.type = "Const";
            const_A_params.blobs.push_back(blob);

            opencv_onnx::NodeProto const_node_proto;
            const_node_proto.add_output(const_A_params.name);
            addLayer(const_A_params, const_node_proto);
            node_proto.set_input(0, const_A_params.name);
        } else { // B or C
            std::string const_params_name = i == 1 ? "B" : "C";

            layerParams.blobs.push_back(blob);
            layerParams.set(cv::format("const%s", const_params_name.c_str()), true);
        }
    }

    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseMatMul(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto_) {
    auto node_proto = node_proto_;
    CV_CheckGE(node_proto.input_size(), 2, "ONNXImporter/MatMul: two inputs required at least");
    CV_CheckLE(node_proto.input_size(), 3, "ONNXImporter/MatMul: three inputs required at most");

    for (int i = 0; i < node_proto.input_size(); i++) {
        if (constBlobs.find(node_proto.input(i)) == constBlobs.end()) {
            continue;
        }

        Mat blob = getBlob(node_proto, i);

        if (i == 0) {
            LayerParams const_params;
            const_params.name = node_proto.input(i);
            const_params.type = "Const";
            const_params.blobs.push_back(blob);

            opencv_onnx::NodeProto const_node_proto;
            const_node_proto.add_output(const_params.name);
            addLayer(const_params, const_node_proto);

            node_proto.set_input(i, const_params.name);
        } else {
            layerParams.blobs.push_back(blob);
        }

        if (i == 2 && constBlobsExtraInfo.find(node_proto.input(2)) != constBlobsExtraInfo.end()) {
            layerParams.set("real_ndims_C", getBlobExtraInfo(node_proto, 2).real_ndims);
        }
    }

    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseConv(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto_)
{
    opencv_onnx::NodeProto node_proto = node_proto_;
    CV_Assert(node_proto.input_size() >= 2);
    layerParams.type = "Convolution";
    for (int j = 1; j < node_proto.input_size(); j++) {
        if (constBlobs.find(node_proto.input(j)) != constBlobs.end())
        {
            layerParams.blobs.push_back(getBlob(node_proto, j));
        }
    }
    int outCn = layerParams.blobs.empty() ? outShapes[node_proto.input(1)][0] : layerParams.blobs[0].size[0];
    layerParams.set("num_output", outCn);

    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseConvTranspose(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    CV_Assert(node_proto.input_size() >= 2);
    layerParams.type = "Deconvolution";
    for (int j = 1; j < node_proto.input_size(); j++) {
        layerParams.blobs.push_back(getBlob(node_proto, j));
    }
    layerParams.set("num_output", layerParams.blobs[0].size[1] * layerParams.get<int>("group", 1));
    layerParams.set("bias_term", node_proto.input_size() == 3);

    if (!layerParams.has("kernel_size"))
        CV_Error(Error::StsNotImplemented,
                 "Required attribute 'kernel_size' is not present.");

    if (layerParams.has("output_shape"))
    {
        const DictValue& outShape = layerParams.get("output_shape");
        DictValue strides = layerParams.get("stride");
        DictValue kernel = layerParams.get("kernel_size");

        String padMode;
        std::vector<int> adjust_pads;
        if (layerParams.has("pad_mode"))
        {
            padMode = toUpperCase(layerParams.get<String>("pad_mode"));
            if (padMode != "SAME" && padMode != "VALID")
                CV_Error(Error::StsError, "Unsupported padding mode " + padMode);

            for (int i = 0; i < strides.size(); i++)
            {
                int sz = outShape.get<int>(2 + i);
                int stride = strides.get<int>(i);
                adjust_pads.push_back(padMode == "SAME"? (sz - 1) % stride :
                                                         (sz - kernel.get<int>(i)) % stride);
            }
            layerParams.set("adj", DictValue::arrayInt(&adjust_pads[0], adjust_pads.size()));
        }
    }
    else if (layerParams.has("output_padding"))
    {
        replaceLayerParam(layerParams, "output_padding", "adj");
    }
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseTranspose(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    int depth = layerParams.get<int>("depth", CV_32F);
    layerParams.type = (depth == CV_8S) ? "PermuteInt8" : "Permute";
    replaceLayerParam(layerParams, "perm", "order");
    if (!layerParams.has("order")) {
        MatShape inpShape = outShapes[node_proto.input(0)];
        size_t dims = inpShape.size();
        std::vector<int> perm(dims);
        for (size_t d = 0; d < dims; ++d)
        {
            perm[d] = static_cast<int>(dims - 1 - d);
        }
        layerParams.set("order", DictValue::arrayInt(perm.data(), perm.size()));
    }

    CV_Assert(node_proto.input_size() == 1);
    if (constBlobs.find(node_proto.input(0)) != constBlobs.end())
    {
        std::vector<Mat> inputs(1, getBlob(node_proto, 0)), transposed;
        runLayer(layerParams, inputs, transposed);
        CV_Assert(transposed.size() == 1);
        addConstant(node_proto.output(0), transposed[0]);
        return;
    }
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseSqueeze(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    CV_Assert(node_proto.input_size() <= 2);

    MatShape inpShape = outShapes[node_proto.input(0)];
    std::vector<bool> maskedAxes(inpShape.size(), false);
    if (layerParams.has("axes"))
    {
        DictValue axes_dict = layerParams.get("axes");
        for (int i = 0; i < axes_dict.size(); ++i)
        {
            int axis = axes_dict.getIntValue(i);
            axis = normalize_axis(axis, inpShape.size());
            CV_CheckLE(axis, static_cast<int>(inpShape.size()), "Squeeze axis");
            maskedAxes[axis] = inpShape[axis] == 1;
        }
    }
    else if (node_proto.input_size() == 2)
    {
        if (constBlobs.find(node_proto.input(1)) != constBlobs.end())
        {
            Mat axesMat = getIntBlob(node_proto, 1);
            size_t axesLen = axesMat.total();
            for (int i = 0; i < axesLen; i++)
            {
                int axis = axesMat.at<int>(i);
                axis = normalize_axis(axis, inpShape.size());
                CV_CheckLE(axis, static_cast<int>(inpShape.size()), "Squeeze axis");
                maskedAxes[axis] = inpShape[axis] == 1;
            }
        }
        else
            CV_Error(Error::StsNotImplemented, cv::format("ONNX/Squeeze: doesn't support non-constant 'axes' input"));
    }

    MatShape outShape;
    for (int i = 0; i < inpShape.size(); ++i)
    {
        if (!maskedAxes[i])
            outShape.push_back(inpShape[i]);
    }
    if (outShape.size() != inpShape.size())
    {
        layerParams.type = "Reshape";
        layerParams.set("dim", DictValue::arrayInt(&outShape[0], outShape.size()));
        if (hasDynamicShapes)
        {
            std::vector<int> dynamicAxes;
            std::vector<int> inputIndices;
            for (int index = 0; index < inpShape.size(); ++index)
            {
                if (!maskedAxes[index])
                    inputIndices.push_back(index);
            }
            for (int index = 0; index < outShape.size(); ++index)
                dynamicAxes.push_back(index);
            layerParams.set("dynamic_axes", DictValue::arrayInt(dynamicAxes.data(), dynamicAxes.size()));
            layerParams.set("input_indices", DictValue::arrayInt(inputIndices.data(), inputIndices.size()));
        }
    }
    else
        layerParams.type = "Identity";

    if (constBlobs.find(node_proto.input(0)) != constBlobs.end())
    {
        Mat inp = getBlob(node_proto, 0);
        Mat out = inp.reshape(1, outShape);
        out.dims = outShape.size();  // to workaround dims == 1
        addConstant(node_proto.output(0), out);
        return;
    }
    int depth = layerParams.get<int>("depth", CV_32F);
    layerParams.type += (depth == CV_8S) ? "Int8" : "";
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseFlatten(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto_)
{
    opencv_onnx::NodeProto node_proto = node_proto_;
    CV_CheckEQ(node_proto.input_size(), 1, "");
    int axis_ = layerParams.get<int>("axis", 1);
    if (constBlobs.find(node_proto.input(0)) != constBlobs.end())
    {
        Mat input = getBlob(node_proto, 0);
        if (constBlobsExtraInfo.find(node_proto.input(0)) != constBlobsExtraInfo.end())
        {
            constBlobsExtraInfo.insert(std::make_pair(node_proto.output(0), getBlobExtraInfo(node_proto, 0)));
        }
        int axis = normalize_axis(axis_, input.dims);

        int out_size[2] = {1, 1};
        for (int i = 0; i < axis; ++i)
        {
            out_size[0] *= input.size[i];
        }
        for (int i = axis; i < input.dims; ++i)
        {
            out_size[1] *= input.size[i];
        }

        Mat output = input.reshape(1, 2, out_size);
        addConstant(node_proto.output(0), output);
        return;
    }
    IterShape_t shapeIt = outShapes.find(node_proto.input(0));
    CV_Assert(shapeIt != outShapes.end());
    MatShape inpShape = shapeIt->second;
    int axis = normalize_axis(axis_, inpShape.size());

    if (axis == 0 || axis == inpShape.size())
    {
        LayerParams reshapeLp;
        reshapeLp.name = layerParams.name + "/reshape";
        reshapeLp.type = "Reshape";
        CV_Assert(layer_id.find(reshapeLp.name) == layer_id.end());

        inpShape.insert(axis == 0 ? inpShape.begin() : inpShape.end(), 1);
        reshapeLp.set("dim", DictValue::arrayInt(&inpShape[0], inpShape.size()));

        opencv_onnx::NodeProto proto;
        proto.add_input(node_proto.input(0));
        proto.add_output(reshapeLp.name);
        addLayer(reshapeLp, proto);
        node_proto.set_input(0, reshapeLp.name);
        axis += 1;
    }

    LayerParams first_pass;
    first_pass.name = layerParams.name + "/flatten";
    CV_Assert(layer_id.find(first_pass.name) == layer_id.end());
    first_pass.type = "Flatten";
    first_pass.set("axis", 0);
    first_pass.set("end_axis", axis - 1);

    opencv_onnx::NodeProto proto;
    proto.add_input(node_proto.input(0));
    proto.add_output(first_pass.name);
    addLayer(first_pass, proto);

    layerParams.set("axis", 1);
    node_proto.set_input(0, first_pass.name);
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseUnsqueeze(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    CV_Assert(node_proto.input_size() == 1 || node_proto.input_size() == 2);
    DictValue axes;
    if (node_proto.input_size() == 2)
    {
        Mat blob = getIntBlob(node_proto, 1);
        axes = DictValue::arrayInt(blob.ptr<int>(), blob.total());
    }
    else
        axes = layerParams.get("axes");

    if (constBlobs.find(node_proto.input(0)) != constBlobs.end())
    {
        // Constant input.
        Mat input = getBlob(node_proto, 0);
        int input_dims = input.dims;
        if (constBlobsExtraInfo.find(node_proto.input(0)) != constBlobsExtraInfo.end())
            if (getBlobExtraInfo(node_proto, 0).real_ndims == 1)
                input_dims = 1;

        std::vector<int> dims;
        for (int j = 0; j < input_dims; j++) {
            dims.push_back(input.size[j]);
        }
//        CV_Assert(axes.getIntValue(axes.size()-1) <= dims.size());
        for (int j = 0; j < axes.size(); j++) {
            int idx = axes.getIntValue(j);
            idx = idx < 0 ? idx + input_dims + 1 : idx;
            CV_Assert(0 <= idx && idx <= dims.size());
            dims.insert(dims.begin() + idx, 1);
        }

        Mat out = input.reshape(0, dims);
        addConstant(node_proto.output(0), out);
        return;
    }

    // Variable input.
    if (axes.size() != 1)
        CV_Error(Error::StsNotImplemented, "Multidimensional unsqueeze");

    int depth = layerParams.get<int>("depth", CV_32F);

    MatShape inpShape = outShapes[node_proto.input(0)];
    int axis = axes.getIntValue(0);
    axis = axis < 0 ? axis + (int)inpShape.size() + 1 : axis;
    CV_Assert(0 <= axis && axis <= inpShape.size());
    std::vector<int> outShape = inpShape;
    outShape.insert(outShape.begin() + axis, 1);
    layerParams.type = (depth == CV_8S) ? "ReshapeInt8" : "Reshape";
    layerParams.set("dim", DictValue::arrayInt(&outShape[0], outShape.size()));
    if (hasDynamicShapes)
    {
        std::vector<int> dynamicAxes;
        std::vector<int> inputIndices;
        for (int index = 0; index < outShape.size(); ++index) {
            if (index != axis)
                dynamicAxes.push_back(index);
        }
        for (int index = 0; index < inpShape.size(); ++index)
            inputIndices.push_back(index);
        layerParams.set("dynamic_axes", DictValue::arrayInt(dynamicAxes.data(), dynamicAxes.size()));
        layerParams.set("input_indices", DictValue::arrayInt(inputIndices.data(), inputIndices.size()));
    }
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseExpand(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    CV_CheckEQ(node_proto.input_size(), 2, "DNN/ONNXImporter-Expand: two inputs are required");
    // input shape must be constant and it is passed as param to the layer
    CV_CheckTrue(constBlobs.find(node_proto.input(1)) != constBlobs.end(),
                 "DNN/ONNXImporter-Expand: input shape must be constant");

    Mat mat_input_shape = getIntBlob(node_proto, 1);
    CV_CheckTypeEQ(mat_input_shape.depth(), CV_32S, "DNN/ONNXImporter-Expand: data type of input shape must be CV_32S");
    for (int i = 0; i < mat_input_shape.total(); ++i) {
        CV_Check(i, *(mat_input_shape.ptr<int>() + i) >= 0, "DNN/ONNXImporter-Expand: invalid shape dimension");
    }
    layerParams.set("shape", DictValue::arrayInt(mat_input_shape.ptr<int>(), mat_input_shape.total()));

    if (constBlobs.find(node_proto.input(0)) != constBlobs.end()) {
        bool const_input_1d = false;
        if (constBlobsExtraInfo.find(node_proto.input(0)) != constBlobsExtraInfo.end()) {
            if (getBlobExtraInfo(node_proto, 0).real_ndims == 1) {
                const_input_1d = true;
            }
        }
        layerParams.set("const_input_1d", const_input_1d);

        Mat input = getBlob(node_proto, 0);
        std::vector<Mat> inputs, expanded;
        inputs.push_back(input);
        runLayer(layerParams, inputs, expanded);
        CV_CheckEQ(expanded.size(), static_cast<size_t>(1), "DNN/Expand: only one output is expected when folding constant");
        addConstant(node_proto.output(0), expanded[0]);
        return;
    }

    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseReshape(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    CV_Assert(node_proto.input_size() == 2 || layerParams.has("shape"));
    int depth = layerParams.get<int>("depth", CV_32F);
    layerParams.type += (depth == CV_8S) ? "Int8" : "";

    if (node_proto.input_size() == 2) {
        Mat blob = getIntBlob(node_proto, 1);
        CV_Assert(blob.type() == CV_32SC1);

        layerParams.set("dim", DictValue::arrayInt<int*>(blob.ptr<int>(), blob.total()));

        if (layer_id.find(node_proto.input(0)) == layer_id.end()) {
            std::vector<Mat> inputs(1, getBlob(node_proto, 0)), outputs;
            runLayer(layerParams, inputs, outputs);
            addConstant(node_proto.output(0), outputs[0]);
            if (constBlobsExtraInfo.find(node_proto.input(0)) != constBlobsExtraInfo.end())
            {
                const int real_ndims_input0 = getBlobExtraInfo(node_proto, 0).real_ndims;
                if (real_ndims_input0 == 1 && blob.total() == 1 && blob.at<int>() == -1) // 1D tensor as input0 (data), and shape is -1
                    constBlobsExtraInfo.insert(std::make_pair(node_proto.output(0), TensorInfo(1)));
            }
            return;
        }
    }
    else {
        DictValue shape = layerParams.get("shape");
        std::vector<int> dim;
        for (int j = 0; j < shape.size(); j++) {
            dim.push_back(shape.getIntValue(j));
        }

        if (layer_id.find(node_proto.input(0)) == layer_id.end()) {
            Mat input = getBlob(node_proto, 0);
            Mat out = input.reshape(0, dim);
            addConstant(node_proto.output(0), out);
            return;
        }
        replaceLayerParam(layerParams, "shape", "dim");
    }
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parsePad(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    int depth = layerParams.get<int>("depth", CV_32F);
    layerParams.type = (depth == CV_8S) ? "PaddingInt8" : "Padding";
    replaceLayerParam(layerParams, "mode", "type");
    if (node_proto.input_size() == 3 || node_proto.input_size() == 2)
    {
        // Paddings are in order begin0, begin1, .. beginN, end0, end1, ..., endN.
        // We need to shuffle it to begin0, end0, begin1, end1, ...
        Mat paddings = getIntBlob(node_proto, 1).reshape(1, 2);
        paddings = paddings.t();
        layerParams.set("paddings", DictValue::arrayInt(paddings.ptr<int>(), paddings.total()));

        // check for non-null constant_value
        if (node_proto.input_size() == 3 && !node_proto.input(2).empty())
        {
            Mat value = getBlob(node_proto, 2);
            double padValue = 0;
            switch(value.depth())
            {
                case CV_32F: padValue = value.ptr<float>()[0];   break;
                case CV_32S: padValue = value.ptr<int32_t>()[0]; break;
                case CV_64S: padValue = value.ptr<int64_t>()[0]; break;
                case CV_8S:  padValue = value.ptr<int8_t>()[0];  break;
                default: CV_Error(Error::BadDepth, "Unsupported type");
            }
            layerParams.set<double>("value", (double)padValue);
        }
    }
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseShape(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    CV_Assert(node_proto.input_size() == 1);
    IterShape_t shapeIt = outShapes.find(node_proto.input(0));
    CV_Assert(shapeIt != outShapes.end());
    const MatShape& inpShape = shapeIt->second;

    bool isInput1D = false;
    if (constBlobsExtraInfo.find(node_proto.input(0)) != constBlobsExtraInfo.end())
        if (getBlobExtraInfo(node_proto, 0).real_ndims == 1)
            isInput1D = true;

    int dims = static_cast<int>(inpShape.size());
    if (isInput1D)
        dims = 1;
    Mat shapeMat(1, dims, CV_64S);
    bool isDynamicShape = false;
    for (int j = 0; j < dims; ++j)
    {
        int sz = inpShape[j];
        isDynamicShape |= (sz == 0);
        shapeMat.at<int64_t>(j) = sz;
    }
    shapeMat.dims = 1;  // FIXIT Mat 1D

    if (isDynamicShape)
    {
        CV_LOG_ERROR(NULL, "DNN/ONNX(Shape): dynamic 'zero' shapes are not supported, input " << toString(inpShape, node_proto.input(0)));
        CV_Assert(!isDynamicShape);  // not supported
    }
    addConstant(node_proto.output(0), shapeMat);
}

void ONNXImporter::parseCast(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    int type;
    switch (layerParams.get<int>("to"))
    {
        case opencv_onnx::TensorProto_DataType_FLOAT:   type = CV_32F; break;
        case opencv_onnx::TensorProto_DataType_UINT8:   type = CV_8U;  break;
        case opencv_onnx::TensorProto_DataType_UINT16:  type = CV_16U; break;
        case opencv_onnx::TensorProto_DataType_FLOAT16: type = CV_16F; break;
        case opencv_onnx::TensorProto_DataType_INT8:    type = CV_8S;  break;
        case opencv_onnx::TensorProto_DataType_INT16:   type = CV_16S; break;
        case opencv_onnx::TensorProto_DataType_INT32:   type = CV_32S; break;
        case opencv_onnx::TensorProto_DataType_INT64:   type = CV_64S; break;
        default: CV_Error(Error::BadDepth, "Unsupported type");
    }

    if (constBlobs.find(node_proto.input(0)) != constBlobs.end())
    {
        Mat blob = getBlob(node_proto, 0);
        if (constBlobsExtraInfo.find(node_proto.input(0)) != constBlobsExtraInfo.end())
        {
            constBlobsExtraInfo.insert(std::make_pair(node_proto.output(0), getBlobExtraInfo(node_proto, 0)));
        }
        Mat dst;
        blob.convertTo(dst, type);
        dst.dims = blob.dims;
        addConstant(node_proto.output(0), dst);
        return;
    }

    layerParams.type = "Cast";
    layerParams.set("outputType", type);
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseConstantFill(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    int depth = CV_32F;
    float fill_value;
    if (!layerParams.blobs.empty())
    {
        CV_Assert(!layerParams.has("value"));
        depth = layerParams.blobs[0].depth();
        Mat floats;
        layerParams.blobs[0].convertTo(floats, CV_32F);
        fill_value = floats.at<float>(0, 0);
    }
    else
        fill_value = layerParams.get("value", 0);

    MatShape inpShape = getIntBlob(node_proto, 0);
    for (int i = 0; i < inpShape.size(); i++)
        CV_CheckGT(inpShape[i], 0, "");
    Mat tensor(inpShape.size(), &inpShape[0], depth, Scalar(fill_value));
    addConstant(node_proto.output(0), tensor);
}

void ONNXImporter::parseGather(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    CV_CheckEQ(node_proto.input_size(), 2, "");

    // TODO: get rid of the type conversions and 1-d/0-d special-casing when the time comes
    if (constBlobs.find(node_proto.input(1)) != constBlobs.end())
    {
        int real_ndims = getBlobExtraInfo(node_proto.input(1)).real_ndims;
        layerParams.set("real_ndims", real_ndims);
        if (constBlobs.find(node_proto.input(0)) != constBlobs.end())
        {
            std::vector<Mat> inputs, output;

            Mat input = getBlob(node_proto, 0);
            inputs.push_back(input);

            Mat indices = getBlob(node_proto, 1);
            inputs.push_back(indices);

            runLayer(layerParams, inputs, output);
            //output.back().dims = std::max(input.dims - real_ndims, 1);
            addConstant(node_proto.output(0), output.back());
            return;
        }
    }

    for (int i = 0; i < node_proto.input_size(); ++i)
    {
        if (layer_id.find(node_proto.input(i)) == layer_id.end())
        {
            LayerParams constParams;
            constParams.name = node_proto.input(i);
            constParams.type = "Const";
            Mat blob = getBlob(node_proto, i);
            constParams.blobs.push_back(blob);

            opencv_onnx::NodeProto proto;
            proto.add_output(constParams.name);
            addLayer(constParams, proto);
        }
    }

    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseGatherElements(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    CV_CheckEQ(node_proto.input_size(), 2, "GatherElements: two inputs are required");

    size_t num_const = 0;
    for (size_t i = 0; i < node_proto.input_size(); ++i){
        if (constBlobs.find(node_proto.input(i)) != constBlobs.end())
            ++num_const;
    }

    if (num_const == node_proto.input_size())
    {
        std::vector<Mat> inputs, output;
        for (size_t i = 0; i < node_proto.input_size(); i++) {
            Mat blob = getBlob(node_proto, i);
            inputs.push_back(blob);
        }
        runLayer(layerParams, inputs, output);
        CV_Assert(output.size() == 1);
        addConstant(node_proto.output(0), output[0]);
        return;
    } else if (num_const > 0) {
        for (size_t i = 0; i < node_proto.input_size(); i++) {
            if (constBlobs.find(node_proto.input(i)) != constBlobs.end()) {
                Mat blob = getBlob(node_proto, i);

                LayerParams constParams;
                constParams.name = node_proto.input(i);
                constParams.type = "Const";
                constParams.blobs.push_back(blob);

                opencv_onnx::NodeProto proto;
                proto.add_output(constParams.name);
                addLayer(constParams, proto);
            }
        }
    }

    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseConcat(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    bool hasVariableInps = false;
    for (int i = 0; i < node_proto.input_size(); ++i)
    {
        if (layer_id.find(node_proto.input(i)) != layer_id.end())
        {
            hasVariableInps = true;
            break;
        }
    }
    if (constBlobsExtraInfo.find(node_proto.input(0)) != constBlobsExtraInfo.end())
    {
        constBlobsExtraInfo.insert(std::make_pair(node_proto.output(0), getBlobExtraInfo(node_proto, 0)));
    }

    if (!hasVariableInps)
    {
        std::vector<Mat> inputs(node_proto.input_size()), concatenated;
        // Due constant folding we can get inputs with different number of dimensions
        // Insert the missing dimension to inputs
        MatShape inputShape;
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            inputs[i] = getBlob(node_proto, (int)i);
            if (inputs[i].size.dims() > (int)inputShape.size())
            {
                inputShape = shape(inputs[i]);
            }
        }

        // Concat-1 has default value for axis is 1: https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Concat-1
        int axis = layerParams.get<int>("axis", 1);
        axis = normalize_axis(axis, inputShape.size());
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            inputShape[axis] = inputs[i].dims == (int)inputShape.size() ? inputs[i].size[axis] : 1;
            CV_CheckEQ((size_t)total(inputShape), inputs[i].total(), "");
            inputs[i] = inputs[i].reshape(1, inputShape);
        }
        runLayer(layerParams, inputs, concatenated);

        CV_Assert(concatenated.size() == 1);
        addConstant(node_proto.output(0), concatenated[0]);
        return;
    }
    else
    {
        for (int i = 0; i < node_proto.input_size(); ++i)
        {
            if (constBlobs.find(node_proto.input(i)) != constBlobs.end())
            {
                LayerParams constParams;
                constParams.name = node_proto.input(i);
                constParams.type = "Const";
                constParams.blobs.push_back(getBlob(node_proto, i));

                opencv_onnx::NodeProto proto;
                proto.add_output(constParams.name);
                addLayer(constParams, proto);
            }
        }
    }
    addLayer(layerParams, node_proto);
}

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Resize
void ONNXImporter::parseResize(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    for (int i = 1; i < node_proto.input_size(); i++)
        CV_Assert(layer_id.find(node_proto.input(i)) == layer_id.end());

    int depth = layerParams.get<int>("depth", CV_32F);
    layerParams.type += (depth == CV_8S) ? "Int8" : "";

    if (layerParams.has("coordinate_transformation_mode"))
    {
        String interp_mode = layerParams.get<String>("coordinate_transformation_mode");
        CV_Assert(interp_mode != "tf_crop_and_resize");

        bool halfPixel = interp_mode == "tf_half_pixel_for_nn" || interp_mode == "half_pixel" || interp_mode == "pytorch_half_pixel";

        layerParams.set("align_corners", interp_mode == "align_corners");
        layerParams.set("half_pixel_centers", halfPixel);
        if (layerParams.get<String>("mode") == "linear")
        {
            layerParams.set("mode", halfPixel ? "opencv_linear" : "bilinear");
        }
    }
    if (layerParams.get<String>("mode") == "linear" && framework_name == "pytorch")
        layerParams.set("mode", "opencv_linear");

    // opset-10: input = [X, scales]
    // opset-11: input = [X, roi, scales] or [x, roi, scales, sizes]
    // opset-13: may have empty input, [X, "", "", sizes] or [x, "", scales]
    int scalesInputId = node_proto.input_size() == 2 ? 1 : 2;
    const std::string& scale_name = node_proto.input(scalesInputId);
    Mat scales;
    if(!scale_name.empty())
        scales = getBlob(node_proto, scalesInputId);

    if (!scales.empty())
    {
        CV_CheckEQ(scales.total(), (size_t)4, "HCHW layout is expected");
        layerParams.set("zoom_factor_y", scales.at<float>(2));
        layerParams.set("zoom_factor_x", scales.at<float>(3));
    }
    else if (node_proto.input_size() >= 4)  // opset-11 [x, roi, scales, sizes] or opset-13: input = [X, "", "", sizes]
    {
        const std::string& inputSizes = node_proto.input(3);
        if (constBlobs.find(inputSizes) != constBlobs.end())
        {
            Mat shapes = getIntBlob(node_proto, 3);
            CV_CheckEQ(shapes.total(), (size_t)4, "HCHW layout is expected");
            layerParams.set("width", shapes.at<int>(3));
            layerParams.set("height", shapes.at<int>(2));
        }
        else
        {
            CV_Error(Error::StsNotImplemented, cv::format("ONNX/Resize: doesn't support dynamic non-constant 'sizes' input: %s", inputSizes.c_str()));
        }
    }
    else
    {
        CV_Error(Error::StsNotImplemented, "ONNX/Resize: can't find neither 'scale' nor destination sizes parameters");
    }
    replaceLayerParam(layerParams, "mode", "interpolation");
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseUpsample(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    //fused from Resize Subgraph
    if (layerParams.has("coordinate_transformation_mode"))
    {
        String interp_mode = layerParams.get<String>("coordinate_transformation_mode");
        CV_Assert(interp_mode != "tf_crop_and_resize");

        bool halfPixel = interp_mode == "tf_half_pixel_for_nn" || interp_mode == "half_pixel" || interp_mode == "pytorch_half_pixel";

        layerParams.set("align_corners", interp_mode == "align_corners");
        layerParams.set("half_pixel_centers", halfPixel);
        if (layerParams.get<String>("mode") == "linear")
        {
            layerParams.set("mode", halfPixel ? "opencv_linear" : "bilinear");
        }
    }
    if (layerParams.get<String>("mode") == "linear" && framework_name == "pytorch")
        layerParams.set("mode", "opencv_linear");

    layerParams.type = "Resize";
    if (layerParams.has("scales"))
    {
        // Pytorch layer
        DictValue scales = layerParams.get("scales");
        CV_Assert(scales.size() == 4);
        layerParams.set("zoom_factor_y", scales.getIntValue(2));
        layerParams.set("zoom_factor_x", scales.getIntValue(3));
    }
    else if (layerParams.has("height_scale") && layerParams.has("width_scale"))
    {
        // Caffe2 layer
        replaceLayerParam(layerParams, "height_scale", "zoom_factor_y");
        replaceLayerParam(layerParams, "width_scale", "zoom_factor_x");
    }
    else
    {
        // scales as input
        const std::string& input1 = node_proto.input(1);
        if (constBlobs.find(input1) != constBlobs.end())
        {
            Mat scales = getBlob(input1);
            CV_Assert(scales.total() == 4);
            layerParams.set("zoom_factor_y", scales.at<float>(2));
            layerParams.set("zoom_factor_x", scales.at<float>(3));
        }
    }
    replaceLayerParam(layerParams, "mode", "interpolation");
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseSoftMax(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    const std::string& layer_type = node_proto.op_type();
    int axis;
    if (onnx_opset != 0 && onnx_opset <= 11) {
        axis = layerParams.get<int>("axis", 1);
    } else {
        axis = layerParams.get<int>("axis", -1);
    }
    layerParams.set<int>("axis", axis);
    layerParams.type = "Softmax";
    layerParams.set("log_softmax", layer_type == "LogSoftmax");
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseDetectionOutput(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto_)
{
    opencv_onnx::NodeProto node_proto = node_proto_;
    CV_CheckEQ(node_proto.input_size(), 3, "");
    if (constBlobs.find(node_proto.input(2)) != constBlobs.end())
    {
        Mat priors = getBlob(node_proto, 2);

        LayerParams constParams;
        constParams.name = layerParams.name + "/priors";
        constParams.type = "Const";
        constParams.blobs.push_back(priors);

        opencv_onnx::NodeProto priorsProto;
        priorsProto.add_output(constParams.name);
        addLayer(constParams, priorsProto);

        node_proto.set_input(2, constParams.name);
    }
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseCumSum(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    layerParams.type = "CumSum";

    // Get axis.
    const std::string& input1 = node_proto.input(1);

    if (constBlobs.find(input1) != constBlobs.end())
    {
        Mat axis_blob = getIntBlob(node_proto, 1);
        CV_Assert(axis_blob.total() == 1u);
        layerParams.set("axis", axis_blob.at<int>(0));
    }

    addLayer(layerParams, node_proto);
}

// "Equal" "Greater" "Less" "Pow" "Add" "Sub" "Mul" "Div" "Sum" "Min" "Max" "GreaterOrEqual" "LessOrEqual" "And" "Or" "Xor"
void ONNXImporter::parseElementWise(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto_)
{
    opencv_onnx::NodeProto node_proto = node_proto_;
    String op_type = toLowerCase(node_proto.op_type());

    layerParams.type = "NaryEltwise";
    layerParams.set("operation", toLowerCase(node_proto.op_type()));
    if (node_proto.op_type() == "Mod") {
        if (layerParams.get<int>("fmod", 0)) {
            layerParams.set("operation", "fmod");
        };
    }

    auto pre_broadcast_transform = [](Mat& t, int t_real_ndims) {
        if (t.dims == 2 && t_real_ndims == 1 && t.size[1] == 1)
            transpose(t, t);
    };

    size_t consts = 0;
    for (size_t i = 0; i < node_proto.input_size(); ++i)
    {
        if (layer_id.find(node_proto.input(i)) == layer_id.end())
        {
            ++consts;
        }
    }

    if (consts == node_proto.input_size())
    {
        std::vector<Mat> inputs, output;
        for (size_t i = 0; i < node_proto.input_size(); ++i)
        {
            inputs.push_back(getBlob(node_proto, i));
        }
        runLayer(layerParams, inputs, output);
        CV_Assert(output.size() == 1);
        addConstant(node_proto.output(0), output[0]);
        return;
    }
    else if (consts > 0)
    {
        for (size_t i = 0; i < node_proto.input_size(); ++i)
        {
            if (layer_id.find(node_proto.input(i)) == layer_id.end())
            {
                Mat inp = getBlob(node_proto, i);
                // for cases like a tensor of shape (2,), it will be loaded as shape (2, 1) in OpenCV Mat,
                // but for correct broadcast, we need to make it of shape (1, 2)
                if (constBlobsExtraInfo.find(node_proto.input(i)) != constBlobsExtraInfo.end())
                    pre_broadcast_transform(inp, getBlobExtraInfo(node_proto, i).real_ndims);

                // carry the constant by adding a Const node
                LayerParams constParams;
                constParams.name = node_proto.input(i);
                constParams.type = "Const";
                // Non-constant propagated layers cannot output 1-d or 0-d tensors.
                inp.dims = std::max(inp.dims, 2);
                constParams.blobs.push_back(inp);

                opencv_onnx::NodeProto proto;
                proto.add_output(constParams.name);
                addLayer(constParams, proto);
            }
        }
    }

    // add element-wise layer
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseDepthSpaceOps(LayerParams &layerParams, const opencv_onnx::NodeProto& node_proto) {
    CV_CheckTrue(layerParams.has("blocksize"), "blocksize is required but not found");
    addLayer(layerParams, node_proto);
}

template<typename T>
Mat runRangeLayer(const Mat& startMat, const Mat& limitMat, const Mat& deltaMat)
{
    T start = startMat.at<T>(0);
    T limit = limitMat.at<T>(0);
    T delta = deltaMat.at<T>(0);

    int numberOfElements;
    if (startMat.depth() == CV_32S || startMat.depth() == CV_64S) {
        if (delta > 0)
            numberOfElements = (limit - start + delta - 1) / delta;
        else
            numberOfElements = (start - limit - delta - 1) / -delta;
    }
    else
    {
        numberOfElements = std::ceil((limit - start) / delta);
    }
    numberOfElements = std::max(numberOfElements, 0);

    Mat r(std::vector<int>{numberOfElements}, startMat.type());
    for (int i = 0; i < numberOfElements; i++)
    {
        r.at<T>(i) = start + (i * delta);
    }
    return r;
}

// Currently we only support range with all constant inputs
void ONNXImporter::parseRange(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    CV_Assert(node_proto.input_size() == 3); // 0 - start, 1 - limit, 2 - delta
    layerParams.type = "Range";

    std::vector<int> const_id;
    for (int i = 0; i < node_proto.input_size(); i++)
        if (layer_id.find(node_proto.input(i)) == layer_id.end())
            const_id.push_back(i);

    // only supports the case which all inputs are constant
    CV_Assert(const_id.size() == 3);

    Mat startMat = getBlob(node_proto, 0);
    Mat limitMat = getBlob(node_proto, 1);
    Mat deltaMat = getBlob(node_proto, 2);

    Mat result;
    switch (startMat.depth())
    {
    case CV_32F:
        result = runRangeLayer<float>(startMat, limitMat, deltaMat);
        break;
    case CV_32S:
        result = runRangeLayer<int32_t>(startMat, limitMat, deltaMat);
        break;
    case CV_64S:
        result = runRangeLayer<int64_t>(startMat, limitMat, deltaMat);
        break;
    default:
        CV_Error(cv::Error::BadDepth, "Unsupported type.");
    };

    addConstant(node_proto.output(0), result);
    constBlobsExtraInfo.insert(std::make_pair(node_proto.output(0), TensorInfo(1)));
}

void ONNXImporter::parseScatter(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    CV_CheckEQ(node_proto.input_size(), 3, "Scatter: three inputs are required.");
    layerParams.type = "Scatter";
    if (node_proto.op_type() == "ScatterND")
        layerParams.type = "ScatterND";

     size_t consts = 0;
    for (size_t i = 0; i < node_proto.input_size(); ++i)
        if (layer_id.find(node_proto.input(i)) == layer_id.end())
            ++consts;

    if (consts == node_proto.input_size())
    {
        std::vector<Mat> inputs, output;
        for (size_t i = 0; i < node_proto.input_size(); i++)
        {
            Mat blob = getBlob(node_proto, i);
            inputs.push_back(blob);
        }
        runLayer(layerParams, inputs, output);
        CV_Assert(output.size() == 1);
        addConstant(node_proto.output(0), output[0]);
        return;
    }
    else if (consts > 0)
    {
        for (size_t i = 0; i < node_proto.input_size(); i++)
        {
            if (layer_id.find(node_proto.input(i)) == layer_id.end())
            {
                Mat blob = getBlob(node_proto, i);

                LayerParams constParams;
                constParams.name = node_proto.input(i);
                constParams.type = "Const";
                constParams.blobs.push_back(blob);

                opencv_onnx::NodeProto proto;
                proto.add_output(constParams.name);
                addLayer(constParams, proto);
            }
        }
    }

    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseTile(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    // for Tile>1, only the case of 'repeats' being constant is supported.
    // 'repeats' is treated as a parameter instead of an input to determine shape in pre-run.

    CV_Assert(node_proto.input_size() == 2 || node_proto.input_size() == 3); // tile-1: 3 inputs, tile>1: 2 inputs
    bool is_opset_1 = node_proto.input_size() == 3;

    std::vector<size_t> const_input_idx;
    for (size_t i = 0; i < node_proto.input_size(); ++i)
        if (layer_id.find(node_proto.input(i)) == layer_id.end())
            const_input_idx.push_back(i);

    bool all_const = false;
    if (const_input_idx.size() == node_proto.input_size()) // all inputs are constant
    {
        all_const = true;
    }
    else if ((const_input_idx.size() == 1 && const_input_idx[0] == 1) || // tile>1
             (const_input_idx.size() == 2 && const_input_idx[0] == 1 && const_input_idx[1] == 2)) // tile-1
    {
        all_const = false;
    }
    else
    {
        if (!is_opset_1)
            CV_Error(Error::StsNotImplemented, "ONNX/Tile: repeats being non-constant is not supported.");
        else
            CV_Error(Error::StsNotImplemented, "ONNX/Tile: tiles or axis being non-constant are not supported.");
    }

    int input0_dims = 1;
    if (all_const)
        input0_dims = getBlob(node_proto, 0).dims;
    else
        input0_dims = (int)outShapes[node_proto.input(0)].size();

    // repeats, treated as paramenter
    std::vector<int> repeats_vec(input0_dims, 1);
    Mat input1_blob = getIntBlob(node_proto, 1);
    if (is_opset_1)
    {
        // input1 in tile-1: tiles, 1d tensor of shape [1]
        CV_CheckEQ(input1_blob.total(), 1ull, "ONNX/Tile: tiles must be a 0D tensor or 1D tensor of shape [1].");
        int tiles = input1_blob.at<int>(0);
        // input2 in tile-1: axis, 1d tensor of shape [1]
        Mat input2_blob = getIntBlob(node_proto, 2);
        CV_CheckEQ(input2_blob.total(), 1ull, "ONNX/Tile: axis must be a 0D tensor or 1D tensor of shape [1].");
        int axis = input2_blob.at<int>(0);
        repeats_vec[axis] = tiles;
    }
    else
    {
        // input1 in tile>1: repeats
        CV_CheckLE(input1_blob.dims, 2, "ONNX/Tile: repeats must be a 1D tensor."); // 1D tensor is represented as a 2D Mat
        for (int i = 0; i < input1_blob.total(); i++)
            repeats_vec[i] = input1_blob.at<int>(i);
    }
    layerParams.set("repeats", DictValue::arrayInt(repeats_vec.data(), (int)repeats_vec.size()));

    if (all_const)
    {
        std::vector<Mat> inputs, output;
        Mat input0_blob = getBlob(node_proto, 0);
        inputs.push_back(input0_blob);
        runLayer(layerParams, inputs, output);
        CV_Assert(output.size() == 1);
        addConstant(node_proto.output(0), output[0]);
        return;
    }
    else
    {
        addLayer(layerParams, node_proto);
    }
}

void ONNXImporter::parseLayerNorm(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    // validate axis and convert it if negative
    auto inputDims = static_cast<int>(outShapes[node_proto.input(0)].size());
    int axis = layerParams.get<int>("axis", -1);
    // axis: [-dims, dims)
    CV_CheckGE(axis, -inputDims, "DNN/ONNXImporter: axis of LayerNormalization is out of range");
    CV_CheckLT(axis,  inputDims, "DNN/ONNXImporter: axis of LayerNormalization is out of range");
    axis = (axis + inputDims) % inputDims;
    layerParams.set("axis", axis);

    // constants as constant inputs
    for (size_t i = 1; i < node_proto.input_size(); i++)
    {
        if (constBlobs.find(node_proto.input(i)) != constBlobs.end()) {
            Mat blob = getBlob(node_proto, i);
            layerParams.blobs.push_back(blob);
        }
    }

    // Remove additional outputs (Mean, InvStdDev)
    if (node_proto.output_size() > 1)
    {
        // remove from graph proto
        for (size_t i = 1; i < node_proto.output_size(); i++) {
            for (int j = graph_proto->output_size() - 1; j >= 0; j--) {
                if (graph_proto->output(j).name() == node_proto.output(i)) {
                    graph_proto->mutable_output()->DeleteSubrange(j, 1);
                    break;
                }
            }
        }
        // remove from node proto
        auto outputName = node_proto.output(0);
        opencv_onnx::NodeProto node_proto_ = node_proto;
        node_proto_.mutable_output()->DeleteSubrange(1, node_proto_.output_size() - 1);
        addLayer(layerParams, node_proto_);
    }
    else
    {
        addLayer(layerParams, node_proto);
    }
}

void ONNXImporter::parseTopK(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    // K needs to be constant in case of being input (since opset 10)
    if (node_proto.input_size() == 2) {
        bool K_const = constBlobs.find(node_proto.input(1)) != constBlobs.end();
        CV_CheckTrue(K_const, "OnnxImporter/TopK: K being non-constant is not supported");

        Mat input_K = getBlob(node_proto, 1);
        int K = input_K.at<int>(0);
        layerParams.set("k", K);
    }

    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseSimpleLayers(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    bool is_all_input_const = true;
    for (int i = 0; i < node_proto.input_size(); i++)
    {
        if (layer_id.find(node_proto.input(i)) != layer_id.end())
        {
            is_all_input_const = false;
            break;
        }
    }
    if (is_all_input_const && node_proto.output_size() == 1)
    {
        std::vector<Mat> input, output;
        for (int i = 0; i < node_proto.input_size(); i++)
            input.push_back(getBlob(node_proto, i));
        runLayer(layerParams, input, output);
        addConstant(node_proto.output(0), output[0]);
        return;
    }

    for (int j = 0; j < node_proto.input_size(); j++) {
        if (layer_id.find(node_proto.input(j)) == layer_id.end())
            layerParams.blobs.push_back(getBlob(node_proto, j));
    }
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseHardmax(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    layerParams.type = "Hardmax";
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseGatherND(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    std::cout << "Parsing GatherND" << std::endl;
    CV_Assert(node_proto.input_size() == 2);
    layerParams.type = "GatherND";
    int batch_dims = layerParams.get<int>("batch_dims", 0);
    layerParams.set("batch_dims", batch_dims);
    addLayer(layerParams, node_proto);
    std::cout << "Parsed GatherND" << std::endl;
}

void ONNXImporter::parseEinsum(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    std::vector<MatShape> einsumInpShapes;
    for (int j = 0; j < node_proto.input_size(); j++)
    {
        // create Const layer for constants and mark its shape
        std::vector<int> input_shape;
        if (layer_id.find(node_proto.input(j)) == layer_id.end()) {
            Mat blob = getBlob(node_proto, j);

            LayerParams const_params;
            const_params.name = node_proto.input(j);
            const_params.type = "Const";
            const_params.blobs.push_back(blob);

            opencv_onnx::NodeProto proto;
            proto.add_output(const_params.name);
            addLayer(const_params, proto);

            input_shape.resize(blob.dims);
            for (size_t i = 0; i < input_shape.size(); i++) {
                input_shape[i] = blob.size[i];
            }
        }

        // also try getting shape from inferred shapes
        if (input_shape.empty()) {
            const auto& inputLayerName = node_proto.input(j);
            auto it = outShapes.find(inputLayerName);
            if (it != outShapes.end()) {
                input_shape = it->second;
            }
        }

        if (input_shape.empty()) {
            CV_Error(Error::StsAssert, format("ERROR input shape of %s not found", node_proto.input(j).c_str()));
        } else {
            einsumInpShapes.emplace_back(input_shape);
        }
    }

    CV_CheckFalse(einsumInpShapes.empty(), "ERROR no inputs shapes");
    for (int i = 0; i < einsumInpShapes.size(); i++) {
        layerParams.set("inputShapes" + cv::format("%d", i), DictValue::arrayInt(einsumInpShapes[i].begin(), einsumInpShapes[i].size()));
    }

    // Check if of eqution is valid
    std::string equation = layerParams.get<std::string>("equation");
    CV_CheckFalse(equation.empty(), "Equation is empty");

    // Save number of inputs. We need it in layer initialization
    layerParams.set("inputSize", node_proto.input_size());

    // Save number of outputs. We need it in layer initialization
    layerParams.set("outputSize", node_proto.output_size());

    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseCustomLayer(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    const std::string& name = layerParams.name;
    std::string& layer_type = layerParams.type;
    const std::string& layer_type_domain = node_proto.has_domain() ? node_proto.domain() : std::string();
    if (!layer_type_domain.empty() && layer_type_domain != str_domain_ai_onnx)
    {
        // append ONNX domain name
        static bool DNN_CUSTOM_ONNX_TYPE_INCLUDE_DOMAIN_NAME = utils::getConfigurationParameterBool("OPENCV_DNN_CUSTOM_ONNX_TYPE_INCLUDE_DOMAIN_NAME", true);
        if (DNN_CUSTOM_ONNX_TYPE_INCLUDE_DOMAIN_NAME)
        {
            layer_type = layer_type_domain + "." + layer_type;
        }
    }

    CV_LOG_IF_INFO(NULL, !LayerFactory::isLayerRegistered(layer_type), "DNN/ONNX: unknown node type, try using custom handler for node with " << node_proto.input_size() << " inputs and " << node_proto.output_size() << " outputs: "
            << cv::format("[%s]:(%s)", layer_type.c_str(), name.c_str())
    );

    parseSimpleLayers(layerParams, node_proto);
}

void ONNXImporter::parseQuantDequant(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    CV_Assert(node_proto.input_size() == 2 || node_proto.input_size() == 3);
    layerParams.type = (node_proto.op_type() == "QuantizeLinear") ? "Quantize" : "Dequantize";
    int axis = layerParams.get<int>("axis", 1);
    // For QuantizeLinear and DequantizeLinear, the scale and zeropoint can be a Scalar (per-tensor quantized)
    // or 1-D tensor (per-channel quantized).
    bool is1D = false;

    if (layerParams.type == "Quantize")
        layerParams.set("depth", CV_8S);
    else // Dequantize
        layerParams.set("depth", CV_32F);

    // If scale is not defined as a constant blob, it is considered an external input.
    if(constBlobs.find(node_proto.input(1)) == constBlobs.end()){
        addLayer(layerParams, node_proto);
        return;
    }

    Mat scaleMat = getBlob(node_proto, 1);
    if(scaleMat.total() > 1) is1D = true;

    Mat zpMat;
    if (node_proto.input_size() == 3)
    {
        zpMat = getBlob(node_proto, 2);
        CV_Assert(zpMat.total() ==  scaleMat.total()); // zero point should has the same shape as scale.
    }

    if (is1D)
    {
        const int num = scaleMat.total();

        std::vector<int> zeropoints(num, 0);
        std::vector<float> scales(num, 0);

        for (int i = 0; i < num; i++)
        {
            scales[i] = scaleMat.at<float>(i);
            if (!zpMat.empty())
                zeropoints[i] = zpMat.depth() == CV_32S ?
                                zpMat.at<int>(i) : (int)zpMat.at<int8_t>(i);
        }

        layerParams.set("is1D", true);
        layerParams.set("axis", axis);
        layerParams.set("scales", DictValue::arrayReal(scales.data(), scales.size()));
        layerParams.set("zeropoints", DictValue::arrayInt(zeropoints.data(), zeropoints.size()));
    }
    else
    {
        int zeropoint = zpMat.empty() ? 0 : zpMat.depth() == CV_32S ?
                                            getScalarFromMat<int>(zpMat) : (int)getScalarFromMat<int8_t>(zpMat);
        float scale = getScalarFromMat<float>(scaleMat);

        layerParams.set("is1D", false);
        layerParams.set("scales", scale);
        layerParams.set("zeropoints", zeropoint);
    }

    if (constBlobs.find(node_proto.input(0)) != constBlobs.end()) // Variable input.
    {
        std::vector<Mat> inputs, outputs;
        inputs.push_back(getBlob(node_proto, 0));

        runLayer(layerParams, inputs, outputs);
        addConstant(node_proto.output(0), outputs[0]);
    }
    else
        addLayer(layerParams, node_proto);
}

void ONNXImporter::parseQConv(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto_)
{
    opencv_onnx::NodeProto node_proto = node_proto_;
    int ninputs = node_proto.input_size();
    CV_Assert(ninputs == 8 || ninputs == 9);

    float inp_sc = getScalarFromMat<float>(getBlob(node_proto, 1));
    int inp_zp = (int)getScalarFromMat<int8_t>(getBlob(node_proto, 2));

    if (layerParams.has("pad"))
    {
        bool asymmetricPadding = false;
        DictValue pads = layerParams.get("pad");
        const int dims = pads.size() / 2;

        for (int i = 0; i < dims; ++i)
        {
            if (pads.get<int>(i) != pads.get<int>(i + dims))
            {
                asymmetricPadding = true;
                break;
            }
        }
        if (asymmetricPadding && pads.size() == 4)
        {
            layerParams.erase("pad");
            std::vector<int> paddings(4, 0);
            for (int i = 0; i < dims; ++i)
            {
                paddings.push_back(pads.get<int>(i));
                paddings.push_back(pads.get<int>(dims + i));
            }
            LayerParams padLp;
            padLp.name = layerParams.name + "/pad";
            padLp.type = "PaddingInt8";
            padLp.set("paddings", DictValue::arrayInt(&paddings[0], paddings.size()));
            padLp.set("depth", CV_8S);
            padLp.set<double>("value", (double)inp_zp);

            opencv_onnx::NodeProto proto;
            proto.add_input(node_proto.input(0));
            proto.add_output(padLp.name);

            addLayer(padLp, proto);
            node_proto.set_input(0, padLp.name);
        }
    }

    Mat weights = getBlob(node_proto, 3);
    int outCn = weights.size[0];
    Mat w_scale = getBlob(node_proto, 4);
    CV_Assert(w_scale.total() == 1 || w_scale.total() == outCn);
    bool per_channel = w_scale.total() == outCn;
    Mat wt_sc = (w_scale.total() == outCn) ? w_scale : Mat(1, outCn, CV_32F, Scalar(w_scale.at<float>(0)));

    float out_sc = getScalarFromMat<float>(getBlob(node_proto, 6));
    int8_t out_zp = getScalarFromMat<int8_t>(getBlob(node_proto, 7));

    Mat bias = (ninputs == 9) ? getBlob(node_proto, 8) : Mat::zeros(1, outCn, CV_32S);

    Mat weights_2d = weights.reshape(1, outCn);
    Mat biasFused(1, outCn, CV_32S);
    Mat outputMultiplier(1, outCn, CV_32F);
    for (int i = 0; i < outCn; i++)
    {
        biasFused.at<int>(i) = bias.at<int>(i) - inp_zp*(cv::sum(weights_2d.row(i))[0]);
        outputMultiplier.at<float>(i) = (inp_sc * wt_sc.at<float>(i)) / out_sc;
    }

    layerParams.type = "ConvolutionInt8";
    layerParams.set("num_output", outCn);
    layerParams.set("input_zeropoint", inp_zp);
    layerParams.set("input_scale",inp_sc);
    layerParams.set("zeropoints", out_zp);
    layerParams.set("scales", out_sc);
    layerParams.set("per_channel", per_channel);
    layerParams.blobs.push_back(weights);
    layerParams.blobs.push_back(biasFused);
    layerParams.blobs.push_back(outputMultiplier);
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseQMatMul(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    int ninputs = node_proto.input_size();
    CV_Assert(ninputs == 8);

    if (constBlobs.find(node_proto.input(3)) == constBlobs.end())
        CV_Error(Error::StsNotImplemented, "Variable weights is not supported");

    int firstInpDims = outShapes[node_proto.input(0)].size();

    float inp_sc = getScalarFromMat<float>(getBlob(node_proto, 1));
    int8_t inp_zp = getScalarFromMat<int8_t>(getBlob(node_proto, 2));

    Mat weights = getBlob(node_proto, 3).t();
    int outCn = weights.size[0];
    int secondInpDims = weights.dims;

    Mat w_scale = getBlob(node_proto, 4);
    CV_Assert(w_scale.total() == 1 || w_scale.total() == outCn);
    bool per_channel = w_scale.total() == outCn ? true : false;
    Mat wt_sc = (w_scale.total() == outCn) ? w_scale : Mat(1, outCn, CV_32F, Scalar(w_scale.at<float>(0)));

    float out_sc = getScalarFromMat<float>(getBlob(node_proto, 6));
    int8_t out_zp = getScalarFromMat<int8_t>(getBlob(node_proto, 7));

    Mat bias(1, outCn, CV_32S);
    Mat outputMultiplier(1, outCn, CV_32F);
    for (int i = 0; i < outCn; i++)
    {
        bias.at<int>(i) = -inp_zp*(cv::sum(weights.row(i))[0]);
        outputMultiplier.at<float>(i) = (inp_sc * wt_sc.at<float>(i)) / out_sc;
    }

    layerParams.type = "InnerProductInt8";
    layerParams.set("num_output", outCn);
    layerParams.set("axis", firstInpDims - secondInpDims + 1);
    layerParams.set("input_scale", inp_sc);
    layerParams.set("input_zeropoint", inp_zp);
    layerParams.set("zeropoints", out_zp);
    layerParams.set("scales", out_sc);
    layerParams.set("per_channel", per_channel);

    layerParams.blobs.push_back(weights);
    layerParams.blobs.push_back(bias);
    layerParams.blobs.push_back(outputMultiplier);
    addLayer(layerParams, node_proto);
}

// A * B + C = Y, we require that the dimension of A is [m, k], and the dimension of B is [n, k].
// And the dim of output Y is [m, n]
void ONNXImporter::parseQGemm(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    int ninputs = node_proto.input_size();
    CV_Assert(ninputs == 8 || ninputs == 9);

    layerParams.type = "InnerProductInt8";

    if (constBlobs.find(node_proto.input(3)) == constBlobs.end())
        CV_Error(Error::StsNotImplemented, "Variable weights is not supported");

    Mat weights = getBlob(node_proto, 3);

    if (!layerParams.get<int>("transB", 0))
    {
        transpose(weights, weights);
    }

    CV_Assert(layerParams.get<float>("alpha", 1) == 1.0f);
    CV_Assert(layerParams.get<int>("transA", 0) == 0);

    int firstInpDims = outShapes[node_proto.input(0)].size();

    float inp_sc = getScalarFromMat<float>(getBlob(node_proto, 1));
    int8_t inp_zp = getScalarFromMat<int8_t>(getBlob(node_proto, 2));

    int outCn = weights.size[0];
    int secondInpDims = weights.dims;

    Mat w_scale = getBlob(node_proto, 4);
    CV_Assert(w_scale.total() == 1 || w_scale.total() == outCn);
    bool per_channel = w_scale.total() == outCn;
    Mat wt_sc = (w_scale.total() == outCn) ? w_scale : Mat(1, outCn, CV_32F, Scalar(w_scale.at<float>(0)));

    Mat w_zp = getBlob(node_proto, 5);
    int8_t* ptrZp = w_zp.ptr<int8_t>(0);

    for (int i = 0; i < w_zp.total(); i++)
    {
        if (ptrZp[i] != (int8_t)0)
            CV_Error(Error::StsUnsupportedFormat, "The zero-point non-zero case of W is not supported!");
    }

    float out_sc = getScalarFromMat<float>(getBlob(node_proto, 7));
    int8_t out_zp = ninputs == 9 ? getScalarFromMat<int8_t>(getBlob(node_proto, 8)) : 0;

    Mat bias;
    if (constBlobs.find(node_proto.input(6)) != constBlobs.end())
        bias = getBlob(node_proto, 6);
    if (bias.empty())
        bias = Mat::zeros(1, outCn, CV_32S);

    Mat biasFused(1, outCn, CV_32S);
    Mat outputMultiplier(1, outCn, CV_32F);
    for (int i = 0; i < outCn; i++)
    {
        biasFused.at<int>(i) = bias.at<int>(i) - inp_zp*(cv::sum(weights.row(i))[0]);
        outputMultiplier.at<float>(i) = (inp_sc * wt_sc.at<float>(i)) / out_sc;
    }

    layerParams.type = "InnerProductInt8";
    layerParams.set("num_output", outCn);
    layerParams.set("axis", firstInpDims - secondInpDims + 1);
    layerParams.set("input_scale", inp_sc);
    layerParams.set("input_zeropoint", inp_zp);
    layerParams.set("scales", out_sc);
    layerParams.set("zeropoints", out_zp);
    layerParams.set("per_channel", per_channel);

    layerParams.blobs.push_back(weights);
    layerParams.blobs.push_back(biasFused);
    layerParams.blobs.push_back(outputMultiplier);
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseQEltwise(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto_)
{
    opencv_onnx::NodeProto node_proto = node_proto_;
    CV_Assert(node_proto.input_size() == 7 || node_proto.input_size() == 8);
    std::string op = (node_proto.op_type() == "QLinearAdd") ? "sum" : "prod";
    int constId = -1;
    for (int i = 0; i < 4; i += 3)
    {
        if (constBlobs.find(node_proto.input(i)) != constBlobs.end())
            constId = i;
    }

    float inp_0_sc = getScalarFromMat<float>(getBlob(node_proto, 1));
    int8_t inp_0_zp = getScalarFromMat<int8_t>(getBlob(node_proto, 2));

    float inp_1_sc = getScalarFromMat<float>(getBlob(node_proto, 4));
    int8_t inp_1_zp = getScalarFromMat<int8_t>(getBlob(node_proto, 5));

    // Set 2nd input as the const input
    if (constId == 0)
    {
        cv::swap(inp_0_sc, inp_1_sc);
        cv::swap(inp_0_zp, inp_1_zp);
    }

    float out_sc = getScalarFromMat<float>(getBlob(node_proto, 6));

    int8_t out_zp = 0;
    if (node_proto.input_size() == 8)
        out_zp = getScalarFromMat<int8_t>(getBlob(node_proto, 7));

    std::vector<float> inp_scales = {inp_0_sc, inp_1_sc};
    std::vector<int8_t> inp_zps = {inp_0_zp, inp_1_zp};

    std::vector<float> coeffs;
    float offset;
    if (op == "sum")
    {
        coeffs = {inp_scales[0]/out_sc, inp_scales[1]/out_sc};
        offset = out_zp - coeffs[0]*inp_zps[0] - coeffs[1]*inp_zps[1];
    }
    else
    {
        coeffs = {inp_scales[0]/out_sc, inp_scales[1]};
        offset = out_zp;
    }

    if (constId != -1)
    {
        Mat blob = getBlob(node_proto, constId);
        if (blob.total() == 1)
        {
            float val = inp_scales[1] * (blob.at<int8_t>(0) - inp_zps[1]);
            float scale = inp_scales[0] / out_sc;
            if (op == "prod")
                scale *= val;

            float shift = out_zp - scale*inp_zps[0];
            if (op == "sum")
                shift += (val/out_sc);

            LayerParams rescaleParams;
            rescaleParams.name = layerParams.name;
            rescaleParams.type = "Requantize";
            rescaleParams.set("depth", CV_8S);
            rescaleParams.set("scale", scale);
            rescaleParams.set("shift", shift);
            rescaleParams.set("isEltwise", true);
            addLayer(rescaleParams, node_proto);
            return;
        }
        else
        {
            MatShape inpShape = outShapes[node_proto.input(3 - constId)];
            if (blob.dims == 2)
                blob = blob.t();

            if (shape(blob) == inpShape)
            {
                LayerParams constParams;
                constParams.name = layerParams.name + "/const";
                constParams.type = "ConstInt8";
                constParams.set("depth", CV_8S);
                constParams.set("scales", inp_1_sc);
                constParams.set("zeropoints", inp_1_zp);
                constParams.blobs.push_back(blob);

                int id = dstNet.addLayer(constParams.name, constParams.type, CV_8S, constParams);
                layer_id.insert(std::make_pair(constParams.name, LayerInfo(id, 0, CV_8S)));
                outShapes[constParams.name] = shape(blob);
                node_proto.set_input(constId, constParams.name);

                layerParams.type = "EltwiseInt8";
                layerParams.set("operation", op);
                layerParams.set("coeff", DictValue::arrayReal(coeffs.data(), coeffs.size()));
                layerParams.set("offset", offset);
            }
            else
            {
                layerParams.type = "ScaleInt8";
                layerParams.set("bias_term", op == "sum");
                int axis = 1;
                for (int i = 0; i < graph_proto->initializer_size(); i++)
                {
                    opencv_onnx::TensorProto tensor_proto = graph_proto->initializer(i);
                    if (tensor_proto.name() == node_proto.input(constId))
                    {
                        axis = inpShape.size() - tensor_proto.dims_size();
                        break;
                    }
                }
                layerParams.set("axis", axis);
                blob = blob.reshape(1, 1);
                Mat blob_dequantized;
                blob.convertTo(blob_dequantized, CV_32F, inp_scales[1], -(inp_scales[1] * inp_zps[1]));
                layerParams.blobs.push_back(blob_dequantized);
            }
        }
    }
    else if (outShapes[node_proto.input(0)] == outShapes[node_proto.input(3)])
    {
        layerParams.type = "EltwiseInt8";
        layerParams.set("operation", op);
        layerParams.set("coeff", DictValue::arrayReal(coeffs.data(), coeffs.size()));
        layerParams.set("offset", offset);
    }
    else
    {
        layerParams.type = "ScaleInt8";
        layerParams.set("bias_term", op == "sum");
    }

    layerParams.set("input_scales", DictValue::arrayReal(inp_scales.data(), inp_scales.size()));
    layerParams.set("input_zeropoints", DictValue::arrayInt(inp_zps.data(), inp_zps.size()));
    layerParams.set("scales", out_sc);
    layerParams.set("zeropoints", out_zp);

    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseQLeakyRelu(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    CV_Assert(node_proto.input_size() == 4 || node_proto.input_size() == 5);

    float slope = layerParams.get<float>("alpha");
    float inp_sc = getScalarFromMat<float>(getBlob(node_proto, 1));
    int8_t inp_zp = getScalarFromMat<int8_t>(getBlob(node_proto, 2));
    float out_sc = getScalarFromMat<float>(getBlob(node_proto, 3));
    int8_t out_zp = node_proto.input_size() == 4 ? 0 : getScalarFromMat<int8_t>(getBlob(node_proto, 4));

    Mat lookUpTable(1, 256, CV_8S);
    int8_t* table = lookUpTable.ptr<int8_t>();
    for (int i = -128; i < 128; i++)
    {
        float x = inp_sc*(i - inp_zp);
        float y = x >= 0.f ? x : slope*x;
        int quantized = out_zp + cvRound(y/out_sc);
        table[i+128] = saturate_cast<int8_t>(quantized);
    }

    layerParams.type = "ReLUInt8";
    layerParams.set("input_scale", inp_sc);
    layerParams.set("input_zeropoint", inp_zp);
    layerParams.set("scales", out_sc);
    layerParams.set("zeropoints", out_zp);
    layerParams.set("slope", slope);
    layerParams.blobs.push_back(lookUpTable);
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseQSigmoid(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    CV_Assert(node_proto.input_size() == 4 || node_proto.input_size() == 5);

    float inp_sc = getScalarFromMat<float>(getBlob(node_proto, 1));
    int8_t inp_zp = getScalarFromMat<int8_t>(getBlob(node_proto, 2));
    float out_sc = getScalarFromMat<float>(getBlob(node_proto, 3));
    int8_t out_zp = node_proto.input_size() == 4 ? 0 : getScalarFromMat<int8_t>(getBlob(node_proto, 4));

    Mat lookUpTable(1, 256, CV_8S);
    int8_t* table = lookUpTable.ptr<int8_t>();
    for (int i = -128; i < 128; i++)
    {
        float x = inp_sc*(i - inp_zp);
        float y = 1.f/(1.f + std::exp(-x));
        int quantized = out_zp + cvRound(y/out_sc);
        table[i+128] = saturate_cast<int8_t>(quantized);
    }

    layerParams.type = "SigmoidInt8";
    layerParams.set("input_scale", inp_sc);
    layerParams.set("input_zeropoint", inp_zp);
    layerParams.set("scales", out_sc);
    layerParams.set("zeropoints", out_zp);
    layerParams.blobs.push_back(lookUpTable);
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseQAvgPool(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    CV_Assert(node_proto.input_size() == 4 || node_proto.input_size() == 5);

    float inp_sc = getScalarFromMat<float>(getBlob(node_proto, 1));
    int8_t inp_zp = getScalarFromMat<int8_t>(getBlob(node_proto, 2));
    float out_sc = getScalarFromMat<float>(getBlob(node_proto, 3));
    int8_t out_zp = node_proto.input_size() == 4 ? 0 : getScalarFromMat<int8_t>(getBlob(node_proto, 4));

    layerParams.type = "PoolingInt8";
    layerParams.set("pool", "ave");
    layerParams.set("global_pooling", node_proto.op_type() == "QLinearGlobalAveragePool");
    layerParams.set("multiplier", inp_sc/out_sc);
    layerParams.set("input_scale", inp_sc);
    layerParams.set("input_zeropoint", inp_zp);
    layerParams.set("scales", out_sc);
    layerParams.set("zeropoints", out_zp);
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseQConcat(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto_)
{
    opencv_onnx::NodeProto node_proto = node_proto_;
    layerParams.type = "ConcatInt8";
    int num_inputs = node_proto.input_size();

    float out_scale = getScalarFromMat<float>(getBlob(node_proto, 0));
    int8_t out_zp = getScalarFromMat<int8_t>(getBlob(node_proto, 1));

    for (int i = 2; i < num_inputs; i += 3)
    {
        float inp_scale = getScalarFromMat<float>(getBlob(node_proto, i + 1));
        int8_t inp_zp = getScalarFromMat<int8_t>(getBlob(node_proto, i + 2));

        if (inp_scale != out_scale || inp_zp != out_zp)
        {
            float scale = inp_scale/out_scale;
            float shift = out_zp - scale*inp_zp;

            if (constBlobs.find(node_proto.input(i)) != constBlobs.end())
            {
                Mat blob = getBlob(node_proto, i);
                Mat blob_rescaled;
                blob.convertTo(blob_rescaled, CV_8S, scale, shift);
                constBlobs[node_proto.input(i)] = blob_rescaled;
            }
            else
            {
                LayerParams rescaleParams;
                rescaleParams.name = node_proto.input(i) + "/rescale";
                rescaleParams.type = "Requantize";
                rescaleParams.set("depth", CV_8S);
                rescaleParams.set("scale", scale);
                rescaleParams.set("shift", shift);
                rescaleParams.set("isEltwise", false);

                opencv_onnx::NodeProto proto;
                proto.add_input(node_proto.input(i));
                proto.add_output(rescaleParams.name);
                addLayer(rescaleParams, proto);
                node_proto.set_input(i, rescaleParams.name);
            }
        }
    }

    bool hasVariableInps = false;
    for (int i = 2; i < num_inputs; i += 3)
    {
        if (layer_id.find(node_proto.input(i)) != layer_id.end())
        {
            hasVariableInps = true;
            break;
        }
    }

    if (!hasVariableInps)
    {
        std::vector<Mat> inputs, concatenated;
        MatShape inputShape;
        for (size_t i = 2; i < num_inputs; i += 3)
        {
            Mat blob = getBlob(node_proto, i);
            if (blob.size.dims() > inputShape.size())
            {
                inputShape = shape(blob);
            }
            inputs.push_back(blob);
        }

        int axis = layerParams.get<int>("axis", 1);
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            MatShape targetShape = inputShape;
            targetShape[axis] = shape(inputs[i])[axis];
            CV_CheckEQ(total(targetShape), total(shape(inputs[i])), "");
            inputs[i] = inputs[i].reshape(0, targetShape);
        }
        runLayer(layerParams, inputs, concatenated);
        CV_Assert(concatenated.size() == 1);
        addConstant(layerParams.name, concatenated[0]);
        return;
    }
    else
    {
        for (int i = 2; i < num_inputs; i += 3)
        {
            if (constBlobs.find(node_proto.input(i)) != constBlobs.end())
            {
                LayerParams constParams;
                constParams.name = node_proto.input(i);
                constParams.type = "ConstInt8";
                constParams.blobs.push_back(getBlob(node_proto, i));
                constParams.set("depth", CV_8S);

                opencv_onnx::NodeProto proto;
                proto.add_output(constParams.name);
                addLayer(constParams, proto);
            }
        }
    }
    layerParams.set("scales", out_scale);
    layerParams.set("zeropoints", out_zp);
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseQSoftmax(LayerParams& layerParams, const opencv_onnx::NodeProto& node_proto)
{
    CV_CheckEQ(node_proto.input_size(), 5, "DNN/ONNX: QLinearSoftmax requires 5 inputs, X, X_scale, X_zero_point, Y_scale, Y_zero_point");

    int opset = layerParams.get<int>("opset");
    if (opset < 13) {
        layerParams.set("coerced_2d", true);
    }

    float x_scale = getScalarFromMat<float>(getBlob(node_proto, 1));
    int8_t x_zero_point = getScalarFromMat<int8_t>(getBlob(node_proto, 2));
    float y_scale = getScalarFromMat<float>(getBlob(node_proto, 3));
    int8_t y_zero_point = getScalarFromMat<int8_t>(getBlob(node_proto, 4));

    layerParams.type = "SoftmaxInt8";
    // layerParams also has "axis" and "opset" attrs
    layerParams.set("input_scale", x_scale);
    layerParams.set("input_zeropoint", x_zero_point);
    layerParams.set("scales", y_scale);
    layerParams.set("zeropoints", y_zero_point);
    addLayer(layerParams, node_proto);
}

void ONNXImporter::parseAttention(LayerParams& params, const opencv_onnx::NodeProto& node_proto) {
    CV_CheckTrue(params.has("num_heads"), "ONNXImporter/parseAttention: num_heads is required but missing");
    CV_CheckTrue(params.has("qkv_hidden_sizes"), "ONNXImporter/parseAttention: qkv_hidden_sizes is required but missing");

    auto param_qkv_hidden_sizes = params.get("qkv_hidden_sizes");
    CV_CheckEQ(param_qkv_hidden_sizes.size(), 3, "ONNXImporter/parseAttention: qkv_hidden_sizes is must and only have three elements");

    for (size_t i = 1; i < node_proto.input_size(); i++) {
        if (constBlobs.find(node_proto.input(i)) != constBlobs.end()) {
            Mat blob = getBlob(node_proto, i);
            params.blobs.push_back(blob);
        }
    }

    addLayer(params, node_proto);
}

// Domain: ai.onnx (default)
// URL: https://github.com/onnx/onnx/blob/master/docs/Operators.md
void ONNXImporter::buildDispatchMap_ONNX_AI(int opset_version)
{
    CV_UNUSED(opset_version);
    DispatchMap dispatch;

    dispatch["ArgMax"] = dispatch["ArgMin"] = &ONNXImporter::parseArg;
    dispatch["MaxUnpool"] = &ONNXImporter::parseMaxUnpool;
    dispatch["MaxPool"] = &ONNXImporter::parseMaxPool;
    dispatch["AveragePool"] = &ONNXImporter::parseAveragePool;
    dispatch["GlobalAveragePool"] = dispatch["GlobalMaxPool"] = &ONNXImporter::parseGlobalPool;
    dispatch["ReduceMax"] = dispatch["ReduceMin"] = dispatch["ReduceMean"] = dispatch["ReduceSum"] =
            dispatch["ReduceSumSquare"] = dispatch["ReduceProd"] = dispatch["ReduceL1"] =
            dispatch["ReduceL2"] = dispatch["ReduceLogSum"] = dispatch["ReduceLogSumExp"] = &ONNXImporter::parseReduce;
    dispatch["Slice"] = &ONNXImporter::parseSlice;
    dispatch["Split"] = &ONNXImporter::parseSplit;
    dispatch["Neg"] = &ONNXImporter::parseNeg;
    dispatch["Constant"] = &ONNXImporter::parseConstant;
    dispatch["LSTM"] = &ONNXImporter::parseLSTM;
    dispatch["GRU"] = &ONNXImporter::parseGRU;
    dispatch["ImageScaler"] = &ONNXImporter::parseImageScaler;
    dispatch["Clip"] = &ONNXImporter::parseClip;
    dispatch["LeakyRelu"] = &ONNXImporter::parseLeakyRelu;
    dispatch["Relu"] = &ONNXImporter::parseRelu;
    dispatch["Elu"] = &ONNXImporter::parseElu;
    dispatch["Tanh"] = &ONNXImporter::parseTanh;
    dispatch["Abs"] = &ONNXImporter::parseAbs;
    dispatch["PRelu"] = &ONNXImporter::parsePRelu;
    dispatch["LRN"] = &ONNXImporter::parseLRN;
    dispatch["InstanceNormalization"] = &ONNXImporter::parseInstanceNormalization;
    dispatch["BatchNormalization"] = &ONNXImporter::parseBatchNormalization;
    dispatch["Gemm"] = &ONNXImporter::parseGemm;
    dispatch["MatMul"] = &ONNXImporter::parseMatMul;
    dispatch["Conv"] = &ONNXImporter::parseConv;
    dispatch["ConvTranspose"] = &ONNXImporter::parseConvTranspose;
    dispatch["Transpose"] = &ONNXImporter::parseTranspose;
    dispatch["Squeeze"] = &ONNXImporter::parseSqueeze;
    dispatch["Flatten"] = &ONNXImporter::parseFlatten;
    dispatch["Unsqueeze"] = &ONNXImporter::parseUnsqueeze;
    dispatch["Expand"] = &ONNXImporter::parseExpand;
    dispatch["Reshape"] = &ONNXImporter::parseReshape;
    dispatch["Pad"] = &ONNXImporter::parsePad;
    dispatch["Shape"] = &ONNXImporter::parseShape;
    dispatch["Cast"] = &ONNXImporter::parseCast;
    dispatch["ConstantFill"] = dispatch["ConstantOfShape"] = &ONNXImporter::parseConstantFill;
    dispatch["Gather"] = &ONNXImporter::parseGather;
    dispatch["GatherElements"] = &ONNXImporter::parseGatherElements;
    dispatch["Concat"] = &ONNXImporter::parseConcat;
    dispatch["Resize"] = &ONNXImporter::parseResize;
    dispatch["Upsample"] = &ONNXImporter::parseUpsample;
    dispatch["SoftMax"] = dispatch["Softmax"] = dispatch["LogSoftmax"] = &ONNXImporter::parseSoftMax;
    dispatch["DetectionOutput"] = &ONNXImporter::parseDetectionOutput;
    dispatch["CumSum"] = &ONNXImporter::parseCumSum;
    dispatch["SpaceToDepth"] = dispatch["DepthToSpace"] = &ONNXImporter::parseDepthSpaceOps;
    dispatch["ScatterElements"] = dispatch["Scatter"] = dispatch["ScatterND"] = &ONNXImporter::parseScatter;
    dispatch["Tile"] = &ONNXImporter::parseTile;
    dispatch["LayerNormalization"] = &ONNXImporter::parseLayerNorm;
    dispatch["GroupNormalization"] = &ONNXImporter::parseInstanceNormalization;
    dispatch["TopK"] = &ONNXImporter::parseTopK;

    dispatch["Equal"] = dispatch["Greater"] = dispatch["Less"] = dispatch["Pow"] = dispatch["Add"] =
            dispatch["Sub"] = dispatch["Mul"] = dispatch["Div"] = dispatch["GreaterOrEqual"] =
            dispatch["LessOrEqual"] = dispatch["Mod"] = dispatch["And"] = dispatch["Or"] = dispatch["Xor"] = &ONNXImporter::parseElementWise;

    dispatch["Sum"] = dispatch["Min"] = dispatch["Max"] = dispatch["Mean"] = &ONNXImporter::parseElementWise;
    dispatch["Where"] = &ONNXImporter::parseElementWise;
    dispatch["Range"] = &ONNXImporter::parseRange;
    dispatch["Einsum"] = &ONNXImporter::parseEinsum;
    dispatch["Hardmax"] = &ONNXImporter::parseHardmax;
    dispatch["GatherND"] = &ONNXImporter::parseGatherND;

    std::vector<std::string> simpleLayers{"Acos", "Acosh", "Asin", "Asinh", "Atan", "Atanh", "Ceil", "Celu", "Cos",
                                          "Cosh", "Dropout", "Erf", "Exp", "Floor", "HardSigmoid", "HardSwish",
                                          "Identity", "Log", "Round", "Reciprocal", "Selu", "Sign", "Sigmoid", "Sin", "Sinh",
                                          "Softplus", "Softsign", "Shrink", "Sqrt", "Tan", "ThresholdedRelu", "Gelu",
                                          "GeluApproximation"};
    for (const auto& name : simpleLayers)
    {
        dispatch[name] = &ONNXImporter::parseSimpleLayers;
    }

    // ai.onnx: opset 10+
    dispatch["QuantizeLinear"] = dispatch["DequantizeLinear"] = &ONNXImporter::parseQuantDequant;
    dispatch["QLinearConv"] = &ONNXImporter::parseQConv;
    dispatch["QLinearMatMul"] = &ONNXImporter::parseQMatMul;

    // com.microsft: This operator is added for compatibility via onnx graph simplifier.
    //               Opset domain cannot be modified from onnx_graph_simplifier.cpp so this
    //               operator cannot be parsed if only added in buildDispatchMap_COM_MICROSOFT
    dispatch["Attention"] = &ONNXImporter::parseAttention;

    domain_dispatch_map[str_domain_ai_onnx] = dispatch;
}

// Domain: com.microsoft
// URL: https://github.com/microsoft/onnxruntime/blob/master/docs/ContribOperators.md
void ONNXImporter::buildDispatchMap_COM_MICROSOFT(int opset_version)
{
    CV_UNUSED(opset_version);
    DispatchMap dispatch;

    dispatch["QLinearAdd"] = dispatch["QLinearMul"] = &ONNXImporter::parseQEltwise;
    dispatch["QLinearAveragePool"] = dispatch["QLinearGlobalAveragePool"] = &ONNXImporter::parseQAvgPool;
    dispatch["QLinearLeakyRelu"] = &ONNXImporter::parseQLeakyRelu;
    dispatch["QLinearSigmoid"] = &ONNXImporter::parseQSigmoid;
    dispatch["QLinearConcat"] = &ONNXImporter::parseQConcat;
    dispatch["QGemm"] = &ONNXImporter::parseQGemm;
    dispatch["QLinearSoftmax"] = &ONNXImporter::parseQSoftmax;
    dispatch["Attention"] = &ONNXImporter::parseAttention;

    domain_dispatch_map["com.microsoft"] = dispatch;
}


Net readNetFromONNX(const String& onnxFile)
{
    return detail::readNetDiagnostic<ONNXImporter>(onnxFile.c_str());
}

Net readNetFromONNX(const char* buffer, size_t sizeBuffer)
{
    return detail::readNetDiagnostic<ONNXImporter>(buffer, sizeBuffer);
}

Net readNetFromONNX(const std::vector<uchar>& buffer)
{
    return readNetFromONNX(reinterpret_cast<const char*>(buffer.data()), buffer.size());
}

Mat readTensorFromONNX(const String& path)
{
    std::fstream input(path.c_str(), std::ios::in | std::ios::binary);
    if (!input)
    {
        CV_Error(Error::StsBadArg, cv::format("Can't read ONNX file: %s", path.c_str()));
    }

    opencv_onnx::TensorProto tensor_proto = opencv_onnx::TensorProto();
    if (!tensor_proto.ParseFromIstream(&input))
    {
        CV_Error(Error::StsUnsupportedFormat, cv::format("Failed to parse ONNX data: %s", path.c_str()));
    }
    Mat mat = getMatFromTensor(tensor_proto, false);
    releaseONNXTensor(tensor_proto);
    return mat;
}

#else  // HAVE_PROTOBUF

#define DNN_PROTOBUF_UNSUPPORTED() CV_Error(Error::StsError, "DNN/ONNX: Build OpenCV with Protobuf to import ONNX models")

Net readNetFromONNX(const String&) {
    DNN_PROTOBUF_UNSUPPORTED();
}

Net readNetFromONNX(const char*, size_t) {
    DNN_PROTOBUF_UNSUPPORTED();
}

Net readNetFromONNX(const std::vector<uchar>&) {
    DNN_PROTOBUF_UNSUPPORTED();
}

Mat readTensorFromONNX(const String&) {
    DNN_PROTOBUF_UNSUPPORTED();
}

#endif  // HAVE_PROTOBUF

CV__DNN_INLINE_NS_END
}} // namespace
