// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>

#include <opencv2/core/utils/logger.defines.hpp>
#undef CV_LOG_STRIP_LEVEL
#define CV_LOG_STRIP_LEVEL CV_LOG_LEVEL_DEBUG + 1
#include <opencv2/core/utils/logger.hpp>

#ifdef HAVE_PROTOBUF

#include <iostream>
#include <fstream>
#include <string>
#include <limits>
#include <algorithm>


#if defined(__GNUC__) && __GNUC__ >= 5
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif
#include "opencv-onnx.pb.h"
#if defined(__GNUC__) && __GNUC__ >= 5
#pragma GCC diagnostic pop
#endif

#include "onnx_graph_simplifier.hpp"

namespace cv {
namespace dnn {
CV__DNN_EXPERIMENTAL_NS_BEGIN


class ONNXImporter
{
    opencv_onnx::ModelProto model_proto;
    struct LayerInfo {
        int layerId;
        int outputId;
        LayerInfo(int _layerId = 0, int _outputId = 0) : layerId(_layerId), outputId(_outputId) {}
    };

    std::map<std::string, Mat> getGraphTensors(
                                    const opencv_onnx::GraphProto& graph_proto);
    Mat getBlob(const opencv_onnx::NodeProto& node_proto, int index);
    Mat getBlob(const std::string& input_name);

    LayerParams getLayerParams(const opencv_onnx::NodeProto& node_proto);
    bool isCeilMode(const LayerParams& layerParams);

    void addConstant(const std::string& name, const Mat& blob);
    void addLayer(LayerParams& layerParams,
                  const opencv_onnx::NodeProto& node_proto);

public:

    ONNXImporter(Net& net, const char *onnxFile)
        : dstNet(net)
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

    ONNXImporter(Net& net, const char* buffer, size_t sizeBuffer)
        : dstNet(net)
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

    void populateNet();

protected:
    Net& dstNet;

    opencv_onnx::GraphProto graph_proto;
    std::string framework_name;

    std::map<std::string, Mat> constBlobs;

    std::map<std::string, MatShape> outShapes;  // List of internal blobs shapes.
    bool hasDynamicShapes;  // Whether the model has inputs with dynamic shapes
    typedef std::map<std::string, MatShape>::iterator IterShape_t;

    std::map<std::string, LayerInfo> layer_id;
    typedef std::map<std::string, LayerInfo>::iterator IterLayerId_t;

    void handleNode(const opencv_onnx::NodeProto& node_proto);
};

inline void replaceLayerParam(LayerParams& layerParams, const String& oldKey, const String& newKey)
{
    if (layerParams.has(oldKey)) {
        layerParams.set(newKey, layerParams.get(oldKey));
        layerParams.erase(oldKey);
    }
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
    int ddepth = CV_32F;
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        inpShapes[i] = shape(inputs[i]);
        if (i > 0 && ddepth != inputs[i].depth())
            CV_Error(Error::StsNotImplemented, "Mixed input data types.");
        ddepth = inputs[i].depth();
    }

    std::vector<MatShape> outShapes, internalShapes;
    layer->getMemoryShapes(inpShapes, 0, outShapes, internalShapes);

    std::vector<Mat> internals(internalShapes.size());
    outputs.resize(outShapes.size());
    for (size_t i = 0; i < outShapes.size(); ++i)
        outputs[i].create(outShapes[i], ddepth);
    for (size_t i = 0; i < internalShapes.size(); ++i)
        internals[i].create(internalShapes[i], ddepth);

    layer->finalize(inputs, outputs);
    layer->forward(inputs, outputs, internals);
}

std::map<std::string, Mat> ONNXImporter::getGraphTensors(
                                        const opencv_onnx::GraphProto& graph_proto)
{
  opencv_onnx::TensorProto tensor_proto;
  std::map<std::string, Mat> layers_weights;

  for (int i = 0; i < graph_proto.initializer_size(); i++)
  {
    tensor_proto = graph_proto.initializer(i);
    Mat mat = getMatFromTensor(tensor_proto);
    releaseONNXTensor(tensor_proto);
    layers_weights.insert(std::make_pair(tensor_proto.name(), mat));
  }
  return layers_weights;
}

static DictValue parse(const ::google::protobuf::RepeatedField< ::google::protobuf::int64>& src) {
    std::vector<int32_t> dst(src.size());
    convertInt64ToInt32(src, dst, src.size());
    return DictValue::arrayInt(&dst[0], src.size());
}

LayerParams ONNXImporter::getLayerParams(const opencv_onnx::NodeProto& node_proto)
{
    LayerParams lp;
    for(int i = 0; i < node_proto.attribute_size(); i++)
    {
        opencv_onnx::AttributeProto attribute_proto = node_proto.attribute(i);
        std::string attribute_name = attribute_proto.name();

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

void ONNXImporter::addLayer(LayerParams& layerParams,
                            const opencv_onnx::NodeProto& node_proto)
{
    int id = dstNet.addLayer(layerParams.name, layerParams.type, layerParams);
    for (int i = 0; i < node_proto.output_size(); ++i)
    {
        layer_id.insert(std::make_pair(node_proto.output(i), LayerInfo(id, i)));
    }

    std::vector<MatShape> layerInpShapes, layerOutShapes, layerInternalShapes;
    int inpNum = 0;
    for (int j = 0; j < node_proto.input_size(); j++)
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
        outShapes[node_proto.output(i)] = layerOutShapes[i];
    }
}

void ONNXImporter::addConstant(const std::string& name, const Mat& blob)
{
    constBlobs.insert(std::make_pair(name, blob));
    outShapes.insert(std::make_pair(name, shape(blob)));
}

void ONNXImporter::populateNet()
{
    CV_Assert(model_proto.has_graph());
    graph_proto = model_proto.graph();

    std::string framework_version;
    if (model_proto.has_producer_name())
        framework_name = model_proto.producer_name();
    if (model_proto.has_producer_version())
        framework_version = model_proto.producer_version();

    CV_LOG_INFO(NULL, "DNN/ONNX: loading ONNX"
            << (model_proto.has_ir_version() ? cv::format(" v%d", (int)model_proto.ir_version()) : cv::String())
            << " model produced by '" << framework_name << "'"
            << (framework_version.empty() ? cv::String() : cv::format(":%s", framework_version.c_str()))
            << ". Number of nodes = " << graph_proto.node_size()
            << ", inputs = " << graph_proto.input_size()
            << ", outputs = " << graph_proto.output_size()
            );

    simplifySubgraphs(graph_proto);

    const int layersSize = graph_proto.node_size();
    CV_LOG_DEBUG(NULL, "DNN/ONNX: graph simplified to " << layersSize << " nodes");

    constBlobs = getGraphTensors(graph_proto);
    // Add all the inputs shapes. It includes as constant blobs as network's inputs shapes.
    for (int i = 0; i < graph_proto.input_size(); ++i)
    {
        const opencv_onnx::ValueInfoProto& valueInfoProto = graph_proto.input(i);
        CV_Assert(valueInfoProto.has_name());
        CV_Assert(valueInfoProto.has_type());
        opencv_onnx::TypeProto typeProto = valueInfoProto.type();
        CV_Assert(typeProto.has_tensor_type());
        opencv_onnx::TypeProto::Tensor tensor = typeProto.tensor_type();
        CV_Assert(tensor.has_shape());
        opencv_onnx::TensorShapeProto tensorShape = tensor.shape();

        MatShape inpShape(tensorShape.dim_size());
        for (int j = 0; j < inpShape.size(); ++j)
        {
            inpShape[j] = tensorShape.dim(j).dim_value();
            if (!tensorShape.dim(j).dim_param().empty())
                hasDynamicShapes = true;
        }
        if (!inpShape.empty() && !hasDynamicShapes)
        {
            inpShape[0] = std::max(inpShape[0], 1); // It's OK to have undetermined batch size
        }
        outShapes[valueInfoProto.name()] = inpShape;
    }

    // create map with network inputs (without const blobs)
    // fill map: push layer name, layer id and output id
    std::vector<String> netInputs;
    for (int j = 0; j < graph_proto.input_size(); j++)
    {
        const std::string& name = graph_proto.input(j).name();
        if (constBlobs.find(name) == constBlobs.end()) {
            netInputs.push_back(name);
            layer_id.insert(std::make_pair(name, LayerInfo(0, netInputs.size() - 1)));
        }
    }
    dstNet.setInputsNames(netInputs);

    for(int li = 0; li < layersSize; li++)
    {
        const opencv_onnx::NodeProto& node_proto = graph_proto.node(li);
        handleNode(node_proto);
    }

    CV_LOG_DEBUG(NULL, "DNN/ONNX: import completed!");
}

void ONNXImporter::handleNode(const opencv_onnx::NodeProto& node_proto_)
{
    opencv_onnx::NodeProto node_proto = node_proto_;  // TODO FIXIT

    CV_Assert(node_proto.output_size() >= 1);
    std::string name = node_proto.output(0);
    std::string layer_type = node_proto.op_type();
    CV_LOG_DEBUG(NULL, "DNN/ONNX: processing node with " << node_proto.input_size() << " inputs and " << node_proto.output_size() << " outputs: "
            << cv::format("[%s]:(%s)", layer_type.c_str(), name.c_str())
    );

    try
    {
        // FIXIT not all cases can be repacked into "LayerParams". Importer should handle such cases directly for each "layer_type"
        LayerParams layerParams = getLayerParams(node_proto);

        layerParams.name = name;
        layerParams.type = layer_type;
        layerParams.set("has_dynamic_shapes", hasDynamicShapes);

        if (layer_type == "MaxPool")
        {
            layerParams.type = "Pooling";
            layerParams.set("pool", "MAX");
            layerParams.set("ceil_mode", layerParams.has("pad_mode"));
        }
        else if (layer_type == "AveragePool")
        {
            layerParams.type = "Pooling";
            layerParams.set("pool", "AVE");
            layerParams.set("ceil_mode", layerParams.has("pad_mode"));
            layerParams.set("ave_pool_padded_area", framework_name == "pytorch");
        }
        else if (layer_type == "GlobalAveragePool" || layer_type == "GlobalMaxPool" ||
                layer_type == "ReduceMean" || layer_type == "ReduceSum" || layer_type == "ReduceMax")
        {
            CV_Assert(node_proto.input_size() == 1);
            layerParams.type = "Pooling";
            String pool;
            if (layer_type == "GlobalMaxPool" || layer_type == "ReduceMax")
                pool = "MAX";
            else if (layer_type == "ReduceSum")
                pool = "SUM";
            else
                pool = "AVE";
            layerParams.set("pool", pool);
            layerParams.set("global_pooling", !layerParams.has("axes"));
            if (layerParams.has("axes") && (layer_type == "ReduceMean" || layer_type == "ReduceSum" || layer_type == "ReduceMax"))
            {
                MatShape inpShape = outShapes[node_proto.input(0)];
                DictValue axes = layerParams.get("axes");
                bool keepdims = layerParams.get<int>("keepdims");
                MatShape targetShape;
                std::vector<bool> shouldDelete(inpShape.size(), false);
                for (int i = 0; i < axes.size(); i++) {
                    int axis = normalize_axis(axes.get<int>(i), inpShape.size());
                    shouldDelete[axis] = true;
                }
                for (int axis = 0; axis < inpShape.size(); ++axis){
                    if (!shouldDelete[axis])
                        targetShape.push_back(inpShape[axis]);
                    else if (keepdims)
                        targetShape.push_back(1);
                }

                if (inpShape.size() == 3 && axes.size() <= 2)
                {
                    int axis = normalize_axis(axes.get<int>(0), inpShape.size());
                    CV_CheckNE(axis, 0, "");

                    LayerParams reshapeLp;
                    reshapeLp.name = layerParams.name + "/reshape";
                    reshapeLp.type = "Reshape";
                    CV_Assert(layer_id.find(reshapeLp.name) == layer_id.end());
                    reshapeLp.set("axis", 0);
                    reshapeLp.set("num_axes", 1);
                    int newShape[] = {1, -1};
                    reshapeLp.set("dim", DictValue::arrayInt(&newShape[0], 2));

                    opencv_onnx::NodeProto proto;
                    proto.add_input(node_proto.input(0));
                    proto.add_output(reshapeLp.name);
                    addLayer(reshapeLp, proto);

                    LayerParams avgLp;
                    avgLp.name = layerParams.name + "/avg";
                    avgLp.type = "Pooling";
                    CV_Assert(layer_id.find(avgLp.name) == layer_id.end());
                    avgLp.set("pool", pool);
                    if (axes.size() == 2)
                    {
                        CV_CheckEQ(normalize_axis(axes.get<int>(0), inpShape.size()), 1, "Unsupported mode");
                        CV_CheckEQ(normalize_axis(axes.get<int>(1), inpShape.size()), 2, "Unsupported mode");
                        avgLp.set("global_pooling", true);
                    }
                    else
                    {
                        avgLp.set(axis == 2 ? "global_pooling_w" : "global_pooling_h", true);
                        avgLp.set(axis == 2 ? "kernel_h" : "kernel_w", 1);
                    }

                    node_proto.set_input(0, reshapeLp.name);
                    node_proto.set_output(0, avgLp.name);
                    addLayer(avgLp, node_proto);
                }
                else
                {
                    if (inpShape.size() != 4 && inpShape.size() != 5)
                        CV_Error(Error::StsNotImplemented, "Unsupported input shape of " + layer_type + " operation.");

                    CV_Assert(axes.size() <= inpShape.size() - 2);
                    std::vector<int> kernel_size(inpShape.size() - 2, 1);
                    if (axes.size() == 1 && (normalize_axis(axes.get<int>(0), inpShape.size()) <= 1))
                    {
                        int axis = normalize_axis(axes.get<int>(0), inpShape.size());
                        MatShape newShape = inpShape;
                        newShape[axis + 1] = total(newShape, axis + 1);
                        newShape.resize(axis + 2);
                        newShape.insert(newShape.begin(), 2 - axis, 1);

                        LayerParams reshapeLp;
                        reshapeLp.type = "Reshape";
                        reshapeLp.name = layerParams.name + "/reshape";
                        CV_Assert(layer_id.find(reshapeLp.name) == layer_id.end());
                        reshapeLp.set("dim", DictValue::arrayInt(&newShape[0], newShape.size()));

                        node_proto.set_output(0, reshapeLp.name);
                        addLayer(reshapeLp, node_proto);

                        kernel_size.resize(2);
                        kernel_size[0] = inpShape[axis];
                        node_proto.set_input(0, node_proto.output(0));
                    }
                    else
                    {
                        for (int i = 0; i < axes.size(); i++) {
                            int axis = normalize_axis(axes.get<int>(i), inpShape.size());
                            CV_Assert_N(axis >= 2 + i, axis < inpShape.size());
                            kernel_size[axis - 2] = inpShape[axis];
                        }
                    }

                    LayerParams poolLp = layerParams;
                    poolLp.name = layerParams.name + "/avg";
                    CV_Assert(layer_id.find(poolLp.name) == layer_id.end());
                    poolLp.set("kernel_size", DictValue::arrayInt(&kernel_size[0], kernel_size.size()));

                    node_proto.set_output(0, poolLp.name);
                    addLayer(poolLp, node_proto);
                }

                layerParams.type = "Reshape";
                layerParams.set("dim", DictValue::arrayInt(&targetShape[0], targetShape.size()));

                node_proto.set_input(0, node_proto.output(0));
                node_proto.set_output(0, layerParams.name);
            }
            else if (!layerParams.has("axes") && (layer_type == "ReduceMean" || layer_type == "ReduceSum" || layer_type == "ReduceMax"))
            {
                CV_CheckEQ(layerParams.get<int>("keepdims"), 0, "layer only supports keepdims = false");
                LayerParams reshapeLp;
                reshapeLp.name = layerParams.name + "/reshape";
                reshapeLp.type = "Reshape";
                CV_Assert(layer_id.find(reshapeLp.name) == layer_id.end());
                int newShape[] = {1, 1, 1, -1};
                reshapeLp.set("dim", DictValue::arrayInt(&newShape[0], 4));

                opencv_onnx::NodeProto proto;
                proto.add_input(node_proto.input(0));
                proto.add_output(reshapeLp.name);
                addLayer(reshapeLp, proto);

                LayerParams poolLp = layerParams;
                poolLp.name = layerParams.name + "/pool";
                CV_Assert(layer_id.find(poolLp.name) == layer_id.end());

                node_proto.set_input(0, reshapeLp.name);
                node_proto.set_output(0, poolLp.name);
                addLayer(poolLp, node_proto);

                layerParams.type = "Reshape";
                int targetShape[] = {1};
                layerParams.set("dim", DictValue::arrayInt(&targetShape[0], 1));

                node_proto.set_input(0, node_proto.output(0));
                node_proto.set_output(0, layerParams.name);
            }
        }
        else if (layer_type == "Slice")
        {
            int axis = 0;
            std::vector<int> begin;
            std::vector<int> end;
            std::vector<int> steps;
            int inp_size = node_proto.input_size();

            if (inp_size == 1)
            {
                if (layerParams.has("axes")) {
                    DictValue axes = layerParams.get("axes");
                    for (int i = 1; i < axes.size(); ++i) {
                        CV_Assert(axes.get<int>(i - 1) == axes.get<int>(i) - 1);
                    }
                    axis = axes.get<int>(0);
                }

                DictValue starts = layerParams.get("starts");
                DictValue ends = layerParams.get("ends");
                CV_Assert(starts.size() == ends.size());

                if (axis > 0) {
                    begin.resize(axis, 0);
                    end.resize(axis, -1);
                }
                for (int i = 0; i < starts.size(); ++i)
                {
                    begin.push_back(starts.get<int>(i));
                    int finish = ends.get<int>(i);
                    end.push_back((finish < 0) ? --finish : finish); // numpy doesn't include last dim
                }
            } else { // inp_size > 1
                CV_Assert(inp_size >= 3);
                for (int i = 1; i < inp_size; i++) {
                    CV_Assert(constBlobs.find(node_proto.input(i)) != constBlobs.end());
                }
                Mat start_blob = getBlob(node_proto, 1);
                Mat end_blob   = getBlob(node_proto, 2);
                CV_Assert(start_blob.total() == end_blob.total());

                if (inp_size > 3) {
                    Mat axes_blob = getBlob(node_proto, 3);
                    const int* axes = (int*)axes_blob.data;
                    for (int i = 1; i < axes_blob.total(); ++i) {
                        CV_Assert(axes[i - 1] == axes[i] - 1);
                    }
                    axis = axes[0];
                }

                const int* starts = start_blob.ptr<int>();
                const int* ends   = end_blob.ptr<int>();
                if (axis > 0) {
                    begin.resize(axis, 0);
                    end.resize(axis, -1);
                }
                std::copy(starts, starts + start_blob.total(), std::back_inserter(begin));
                for (int i = 0; i < end_blob.total(); ++i)
                {
                    int finish = ends[i];
                    end.push_back((finish < 0) ? --finish : finish); // numpy doesn't include last dim
                }

                if (inp_size == 5) {
                    CV_Assert(constBlobs.find(node_proto.input(4)) != constBlobs.end());
                    Mat step_blob = getBlob(node_proto, 4);
                    const int* steps_ptr = step_blob.ptr<int>();

                    if (axis > 0)
                        steps.resize(axis, 1);

                    std::copy(steps_ptr, steps_ptr + step_blob.total(), std::back_inserter(steps));

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
                            addConstant(layerParams.name, flipped);
                            return;
                        }
                    }
                }
            }
            layerParams.set("begin", DictValue::arrayInt(&begin[0], begin.size()));
            layerParams.set("end", DictValue::arrayInt(&end[0], end.size()));
            layerParams.set("axis", axis);

            if (!steps.empty())
                layerParams.set("steps", DictValue::arrayInt(&steps[0], steps.size()));

            if (constBlobs.find(node_proto.input(0)) != constBlobs.end())
            {
                Mat inp = getBlob(node_proto, 0);
                std::vector<Mat> inputs, sliced;
                inputs.push_back(inp);
                runLayer(layerParams, inputs, sliced);
                CV_Assert(sliced.size() == 1);
                addConstant(layerParams.name, sliced[0]);
                return;
            }
        }
        else if (layer_type == "Split")
        {
            if (layerParams.has("split"))
            {
                DictValue splits = layerParams.get("split");
                const int numSplits = splits.size();
                CV_Assert(numSplits > 1);

                std::vector<int> slicePoints(numSplits - 1, splits.get<int>(0));
                for (int i = 1; i < splits.size() - 1; ++i)
                {
                    slicePoints[i] = slicePoints[i - 1] + splits.get<int>(i - 1);
                }
                layerParams.set("slice_point", DictValue::arrayInt(&slicePoints[0], slicePoints.size()));
            }
            else
            {
                layerParams.set("num_split", node_proto.output_size());
            }
            layerParams.type = "Slice";
        }
        else if (layer_type == "Add" || layer_type == "Sum" || layer_type == "Sub")
        {
            bool isSub = layer_type == "Sub";
            CV_CheckEQ(node_proto.input_size(), 2, "");
            bool is_const_0 = layer_id.find(node_proto.input(0)) == layer_id.end();
            bool is_const_1 = layer_id.find(node_proto.input(1)) == layer_id.end();
            if (is_const_0 && is_const_1)
            {
                Mat blob_0 = getBlob(node_proto, 0);
                Mat blob_1 = getBlob(node_proto, 1);
                CV_Assert(blob_0.size == blob_1.size);
                Mat output = isSub ? (blob_0 - blob_1) : (blob_0 + blob_1);
                addConstant(layerParams.name, output);
                return;
            }
            else if (is_const_0 || is_const_1)
            {
                int const_blob_id = is_const_0 ? 0 : 1;
                Mat blob = getBlob(node_proto, const_blob_id);
                int blob_total = blob.total();
                if (blob_total == 1) {
                    layerParams.type = "Power";
                    layerParams.set("shift", (isSub ? -1 : 1) * blob.at<float>(0));
                }
                else {
                    MatShape inpShape = outShapes[node_proto.input(1 - const_blob_id)];
                    if (shape(blob) == inpShape)
                    {
                        LayerParams constParams;
                        constParams.name = layerParams.name + "/const";
                        constParams.type = "Const";
                        constParams.blobs.push_back((isSub ? -1 : 1) * blob);
                        int id = dstNet.addLayer(constParams.name, constParams.type, constParams);
                        layer_id.insert(std::make_pair(constParams.name, LayerInfo(id, 0)));
                        outShapes[constParams.name] = shape(blob);

                        layerParams.type = "Eltwise";
                        node_proto.set_input(const_blob_id, constParams.name);
                    }
                    else
                    {
                        layerParams.type = "Scale";
                        layerParams.set("bias_term", true);
                        int axis = 1;
                        for (int i = 0; i < graph_proto.initializer_size(); i++)
                        {
                            opencv_onnx::TensorProto tensor_proto = graph_proto.initializer(i);
                            if (tensor_proto.name() == node_proto.input(const_blob_id))
                            {
                                axis = inpShape.size() - tensor_proto.dims_size();
                                break;
                            }
                        }
                        layerParams.set("axis", axis);
                        blob = blob.reshape(1, 1);
                        layerParams.blobs.push_back((isSub ? -1 : 1) * blob);
                    }
                }
            }
            else if (outShapes[node_proto.input(0)] == outShapes[node_proto.input(1)])
            {
                layerParams.type = "Eltwise";
                if (isSub)
                {
                    static float subCoeffs[] = {1.f, -1.f};
                    layerParams.set("coeff", DictValue::arrayReal<float*>(subCoeffs, 2));
                }
            }
            else
            {
                if (isSub)
                {
                    LayerParams powerParams;
                    powerParams.name = layerParams.name + "/neg";
                    powerParams.type = "Power";
                    powerParams.set("scale", -1);

                    //Create Power layer
                    int id = dstNet.addLayer(powerParams.name, powerParams.type, powerParams);
                    //Connect to input
                    IterLayerId_t layerId = layer_id.find(node_proto.input(1));
                    CV_Assert(layerId != layer_id.end());
                    dstNet.connect(layerId->second.layerId, layerId->second.outputId, id, 0);
                    //Add shape
                    layer_id.insert(std::make_pair(powerParams.name, LayerInfo(id, 0)));
                    outShapes[powerParams.name] = outShapes[node_proto.input(1)];

                    //Replace input to Power
                    node_proto.set_input(1, powerParams.name);
                }
                layerParams.type = "Scale";
                layerParams.set("bias_term", true);
            }
        }
        else if (layer_type == "Pow")
        {
            if (layer_id.find(node_proto.input(1)) != layer_id.end())
                CV_Error(Error::StsNotImplemented, "Unsupported Pow op with variable power");

            Mat blob = getBlob(node_proto, 1);
            if (blob.total() != 1)
                CV_Error(Error::StsNotImplemented, "Pow op supports only scalar power");

            blob.convertTo(blob, CV_32F);
            layerParams.type = "Power";
            layerParams.set("power", blob.at<float>(0));
        }
        else if (layer_type == "Max")
        {
            layerParams.type = "Eltwise";
            layerParams.set("operation", "max");
        }
        else if (layer_type == "Neg")
        {
            layerParams.type = "Power";
            layerParams.set("scale", -1);
        }
        else if (layer_type == "Constant")
        {
            CV_Assert(node_proto.input_size() == 0);
            CV_Assert(layerParams.blobs.size() == 1);
            addConstant(layerParams.name, layerParams.blobs[0]);
            return;
        }
        else if (layer_type == "LSTM")
        {
            LayerParams lstmParams = layerParams;
            lstmParams.name += "/lstm";

            // https://pytorch.org/docs/stable/nn.html#lstm
            CV_Assert(node_proto.input_size() == 7);
            Mat Wx = getBlob(node_proto, 1);
            Mat Wh = getBlob(node_proto, 2);
            Mat b = getBlob(node_proto, 3);
            CV_CheckEQ(countNonZero(getBlob(node_proto, 5)), 0, "Unsupported non zero initial_h");
            CV_CheckEQ(countNonZero(getBlob(node_proto, 6)), 0, "Unsupported non zero initial_c");
            b = b.reshape(1, b.size[0]);

            const int numHidden = lstmParams.get<int>("hidden_size");
            const int numDirs = Wx.size[0];  // Is 1 for forward only and 2 for bidirectional LSTM.
            const int numFeatures = Wx.size[2];
            Mat bx = b.colRange(0, b.cols / 2);
            Mat bh = b.colRange(b.cols / 2, b.cols);
            b = bx + bh;

            // IFGO->IGFO
            for (int k = 0; k < numDirs; ++k)
            {
                float* WxData = Wx.ptr<float>(k);
                float* WhData = Wh.ptr<float>(k);
                float* biasData = b.ptr<float>(k);
                for (int j = 0; j < numHidden; ++j)
                {
                    for (int i = 0; i < numFeatures; ++i)
                    {
                        std::swap(WxData[(numHidden + j) * numFeatures + i],
                                  WxData[(numHidden * 2 + j) * numFeatures + i]);
                    }
                    for (int i = 0; i < numHidden; ++i)
                    {
                        std::swap(WhData[(numHidden + j) * numHidden + i],
                                  WhData[(numHidden * 2 + j) * numHidden + i]);
                    }
                    std::swap(biasData[numHidden + j], biasData[numHidden * 2 + j]);
                }
            }
            Wx = Wx.reshape(1, Wx.size[0] * Wx.size[1]);
            Wh = Wh.reshape(1, Wh.size[0] * Wh.size[1]);

            lstmParams.blobs.resize(3);
            lstmParams.blobs[0] = Wh;
            lstmParams.blobs[1] = Wx;
            lstmParams.blobs[2] = b;
            lstmParams.set("bidirectional", lstmParams.get<String>("direction", "") == "bidirectional");

            node_proto.set_output(0, lstmParams.name);  // set different name so output shapes will be registered on that name
            addLayer(lstmParams, node_proto);

            MatShape lstmShape = outShapes[node_proto.output(0)];

            // Add fake 1 as it is done in ONNX
            lstmShape.insert(lstmShape.begin() + 1, 1);

            layerParams.type = "Reshape";
            layerParams.set("dim", DictValue::arrayInt(&lstmShape[0], lstmShape.size()));
            node_proto.set_input(0, lstmParams.name);  // redirect input to LSTM
            node_proto.set_output(0, layerParams.name);  // keep origin LSTM's name
        }
        else if (layer_type == "ImageScaler")
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
        }
        else if (layer_type == "Clip")
        {
            layerParams.type = "ReLU6";
            replaceLayerParam(layerParams, "min", "min_value");
            replaceLayerParam(layerParams, "max", "max_value");

        }
        else if (layer_type == "LeakyRelu")
        {
            layerParams.type = "ReLU";
            replaceLayerParam(layerParams, "alpha", "negative_slope");
        }
        else if (layer_type == "Relu")
        {
            layerParams.type = "ReLU";
        }
        else if (layer_type == "Elu")
        {
            layerParams.type = "ELU";
        }
        else if (layer_type == "Tanh")
        {
            layerParams.type = "TanH";
        }
        else if (layer_type == "PRelu")
        {
            layerParams.type = "PReLU";
            layerParams.blobs.push_back(getBlob(node_proto, 1));
        }
        else if (layer_type == "LRN")
        {
            replaceLayerParam(layerParams, "size", "local_size");
        }
        else if (layer_type == "InstanceNormalization")
        {
            if (node_proto.input_size() != 3)
                CV_Error(Error::StsNotImplemented,
                         "Expected input, scale, bias");

            layerParams.blobs.resize(4);
            layerParams.blobs[2] = getBlob(node_proto, 1);  // weightData
            layerParams.blobs[3] = getBlob(node_proto, 2);  // biasData
            layerParams.set("has_bias", true);
            layerParams.set("has_weight", true);

            // Get number of channels in input
            int size = layerParams.blobs[2].total();
            layerParams.blobs[0] = Mat::zeros(size, 1, CV_32F); // mean
            layerParams.blobs[1] = Mat::ones(size, 1, CV_32F); // std

            LayerParams mvnParams;
            mvnParams.name = layerParams.name + "/MVN";
            mvnParams.type = "MVN";
            mvnParams.set("eps", layerParams.get<float>("epsilon"));
            layerParams.erase("epsilon");

            //Create MVN layer
            int id = dstNet.addLayer(mvnParams.name, mvnParams.type, mvnParams);
            //Connect to input
            IterLayerId_t layerId = layer_id.find(node_proto.input(0));
            CV_Assert(layerId != layer_id.end());
            dstNet.connect(layerId->second.layerId, layerId->second.outputId, id, 0);
            //Add shape
            layer_id.insert(std::make_pair(mvnParams.name, LayerInfo(id, 0)));
            outShapes[mvnParams.name] = outShapes[node_proto.input(0)];

            //Replace Batch Norm's input to MVN
            node_proto.set_input(0, mvnParams.name);
            layerParams.type = "BatchNorm";
        }
        else if (layer_type == "BatchNormalization")
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
        }
        else if (layer_type == "Gemm")
        {
            CV_Assert(node_proto.input_size() >= 2);
            layerParams.type = "InnerProduct";
            Mat weights = getBlob(node_proto, 1);
            int ind_num_out = 0;
            if (layerParams.has("transB") && !layerParams.get<int>("transB")) {
                transpose(weights, weights);
                ind_num_out = 1;
            }
            layerParams.blobs.push_back(weights);

            if (node_proto.input_size() == 3) {
                Mat bias = getBlob(node_proto, 2);
                layerParams.blobs.push_back(bias);
            }
            if (constBlobs.find(node_proto.input(0)) != constBlobs.end())
            {
                Mat inputBuf = getBlob(node_proto, 0);

                LayerParams constParams;
                constParams.name = node_proto.input(0);
                constParams.type = "Const";
                constParams.blobs.push_back(inputBuf);

                opencv_onnx::NodeProto proto;
                proto.add_output(constParams.name);
                addLayer(constParams, proto);
            }

            layerParams.set("num_output", layerParams.blobs[0].size[ind_num_out]);
            layerParams.set("bias_term", node_proto.input_size() == 3);
        }
        else if (layer_type == "MatMul")
        {
            CV_Assert(node_proto.input_size() == 2);
            layerParams.type = "InnerProduct";
            layerParams.set("bias_term", false);
            CV_Assert(constBlobs.find(node_proto.input(0)) == constBlobs.end());
            int firstInpDims = outShapes[node_proto.input(0)].size();
            int secondInpDims;

            if (constBlobs.find(node_proto.input(1)) != constBlobs.end())
            {
                Mat blob = getBlob(node_proto, 1);
                secondInpDims = blob.dims;
                layerParams.blobs.push_back(blob.t());
                layerParams.set("num_output", layerParams.blobs[0].size[0]);
            } else {
                secondInpDims = outShapes[node_proto.input(1)].size();
            }
            layerParams.set("axis", firstInpDims - secondInpDims + 1);
        }
        else if (layer_type == "Mul" || layer_type == "Div")
        {
            CV_Assert(node_proto.input_size() == 2);

            bool isDiv = layer_type == "Div";
            int constId = -1;
            bool haveVariables = false;
            for (int i = 0; i < 2; ++i)
            {
                if (constBlobs.find(node_proto.input(i)) != constBlobs.end())
                    constId = i;
                else
                    haveVariables = true;
            }
            if (constId != -1 && haveVariables)
            {
                Mat blob = getBlob(node_proto, constId);
                blob = blob.reshape(1, 1);
                if (blob.total() == 1) {
                    float coeff = isDiv ? 1.0 / blob.at<float>(0) : blob.at<float>(0);
                    layerParams.set("scale", coeff);
                    layerParams.type = "Power";
                }
                else {
                    if (isDiv)
                        divide(1.0, blob, blob);
                    layerParams.blobs.push_back(blob);
                    layerParams.type = "Scale";
                }
            }
            else if (!haveVariables)
            {
                Mat inp0 = getBlob(node_proto, 0);
                Mat inp1 = getBlob(node_proto, 1);

                if (inp0.size != inp1.size && (inp0.total() != 1 || inp1.total() != 1))
                    CV_Error_(Error::StsNotImplemented, ("Different shapes case is not supported with constant inputs: %s", layer_type.c_str()));

                if (inp0.total() == 1 && inp1.total() == 1 && inp0.dims != inp1.dims)
                {
                    if (inp0.dims < inp1.dims)
                    {
                        inp0 = inp0.reshape(1, inp1.dims, inp1.size);
                        inp0.dims = inp1.dims;
                    }
                    else
                    {
                        inp1 = inp1.reshape(1, inp0.dims, inp0.size);
                        inp1.dims = inp0.dims;
                    }
                }

                Mat out;
                if (inp0.total() != inp1.total())
                {
                    if (inp0.total() == 1)
                    {
                        float coeff = isDiv ? 1.0 / inp0.at<float>(0) : inp0.at<float>(0);
                        multiply(inp1, coeff, out);
                    }
                    else
                    {
                        float coeff = isDiv ? 1.0 / inp1.at<float>(0) : inp1.at<float>(0);
                        multiply(inp0, coeff, out);
                    }

                }
                else
                {
                    out = isDiv ? inp0 / inp1 : inp0.mul(inp1);
                }

                if (inp0.dims == 1 && inp1.dims == 1)
                    out.dims = 1;  // to workaround dims == 1
                addConstant(layerParams.name, out);
                return;
            }
            else if (outShapes[node_proto.input(0)] == outShapes[node_proto.input(1)])
            {
                layerParams.type = "Eltwise";
                layerParams.set("operation", isDiv ? "div" : "prod");
            }
            else
            {
                // Scale layer allocate output with the first input shape
                if (total(outShapes[node_proto.input(0)]) < total(outShapes[node_proto.input(1)]))
                {
                    opencv_onnx::NodeProto proto;
                    proto.add_input(node_proto.input(1));
                    proto.add_input(node_proto.input(0));
                    proto.add_output(layerParams.name);
                    node_proto = proto;
                }

                if (isDiv)
                {
                    LayerParams powerParams;
                    powerParams.name = layerParams.name + "/inv";
                    powerParams.type = "Power";
                    powerParams.set("power", -1);

                    //Create Power layer
                    int id = dstNet.addLayer(powerParams.name, powerParams.type, powerParams);
                    //Connect to input
                    IterLayerId_t layerId = layer_id.find(node_proto.input(1));
                    CV_Assert(layerId != layer_id.end());
                    dstNet.connect(layerId->second.layerId, layerId->second.outputId, id, 0);
                    //Add shape
                    layer_id.insert(std::make_pair(powerParams.name, LayerInfo(id, 0)));
                    outShapes[powerParams.name] = outShapes[node_proto.input(1)];

                    //Replace input to Power
                    node_proto.set_input(1, powerParams.name);
                }
                layerParams.type = "Scale";
            }
        }
        else if (layer_type == "Conv")
        {
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
        }
        else if (layer_type == "ConvTranspose")
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
        }
        else if (layer_type == "Transpose")
        {
            layerParams.type = "Permute";
            replaceLayerParam(layerParams, "perm", "order");

            CV_Assert(node_proto.input_size() == 1);
            if (constBlobs.find(node_proto.input(0)) != constBlobs.end())
            {
                std::vector<Mat> inputs(1, getBlob(node_proto, 0)), transposed;
                runLayer(layerParams, inputs, transposed);
                CV_Assert(transposed.size() == 1);
                addConstant(layerParams.name, transposed[0]);
                return;
            }
        }
        else if (layer_type == "Squeeze")
        {
            CV_Assert_N(node_proto.input_size() == 1, layerParams.has("axes"));
            DictValue axes_dict = layerParams.get("axes");
            MatShape inpShape = outShapes[node_proto.input(0)];

            std::vector<bool> maskedAxes(inpShape.size(), false);
            for (int i = 0; i < axes_dict.size(); ++i)
            {
                int axis = axes_dict.getIntValue(i);
                CV_CheckLE(axis, static_cast<int>(inpShape.size()), "Squeeze axis");
                maskedAxes[axis] = inpShape[axis] == 1;
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
                addConstant(layerParams.name, out);
                return;
            }
        }
        else if (layer_type == "Flatten")
        {
            CV_CheckEQ(node_proto.input_size(), 1, "");
            if (constBlobs.find(node_proto.input(0)) != constBlobs.end())
            {
                Mat input = getBlob(node_proto, 0);
                int axis = normalize_axis(layerParams.get<int>("axis", 1), input.dims);

                std::vector<int> out_size(&input.size[0], &input.size[0] + axis);
                out_size.push_back(input.total(axis));
                Mat output = input.reshape(1, out_size);
                addConstant(layerParams.name, output);
                return;
            }
        }
        else if (layer_type == "Unsqueeze")
        {
            CV_Assert(node_proto.input_size() == 1);
            DictValue axes = layerParams.get("axes");
            if (constBlobs.find(node_proto.input(0)) != constBlobs.end())
            {
                // Constant input.
                Mat input = getBlob(node_proto, 0);

                std::vector<int> dims;
                for (int j = 0; j < input.dims; j++) {
                    dims.push_back(input.size[j]);
                }
                CV_Assert(axes.getIntValue(axes.size()-1) <= dims.size());
                for (int j = 0; j < axes.size(); j++) {
                    dims.insert(dims.begin() + axes.getIntValue(j), 1);
                }

                Mat out = input.reshape(0, dims);
                addConstant(layerParams.name, out);
                return;
            }

            // Variable input.
            if (axes.size() != 1)
                CV_Error(Error::StsNotImplemented, "Multidimensional unsqueeze");

            MatShape inpShape = outShapes[node_proto.input(0)];
            int axis = axes.getIntValue(0);
            CV_Assert(0 <= axis && axis <= inpShape.size());
            std::vector<int> outShape = inpShape;
            outShape.insert(outShape.begin() + axis, 1);
            layerParams.type = "Reshape";
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
        }
        else if (layer_type == "Expand")
        {
            CV_CheckEQ(node_proto.input_size(), 2, "");
            const std::string& input0 = node_proto.input(0);
            const std::string& input1 = node_proto.input(1);
            Mat newShapeMat = getBlob(input1);
            MatShape targetShape(newShapeMat.ptr<int>(), newShapeMat.ptr<int>() + newShapeMat.total());

            MatShape inpShape;
            bool haveVariables = constBlobs.find(input0) == constBlobs.end();
            if (haveVariables)
            {
                IterShape_t shapeIt = outShapes.find(input0);
                CV_Assert(shapeIt != outShapes.end());
                inpShape = shapeIt->second;
            }
            else
            {
                inpShape = shape(getBlob(input0));
            }

            String srcName = input0;
            // Unsqueeze and repeat along new axis
            if (targetShape.size() == inpShape.size() + 1)
            {
                for (int i = 0; i < targetShape.size(); i++)
                {
                    if (targetShape[i] == -1 && i < inpShape.size())
                        targetShape[i] = inpShape[i];
                    else if (i < inpShape.size() && targetShape[i] != inpShape[i])
                        inpShape.insert(inpShape.begin() + i, 1);
                }
                if (haveVariables)
                {
                    LayerParams reshapeLp;
                    reshapeLp.name = layerParams.name + "/reshape";
                    reshapeLp.type = "Reshape";
                    CV_Assert(layer_id.find(reshapeLp.name) == layer_id.end());
                    reshapeLp.set("dim", DictValue::arrayInt(&inpShape[0], inpShape.size()));

                    opencv_onnx::NodeProto proto;
                    proto.add_input(node_proto.input(0));
                    proto.add_output(reshapeLp.name);
                    addLayer(reshapeLp, proto);
                    srcName = reshapeLp.name;
                }
            }
            CV_CheckEQ(inpShape.size(), targetShape.size(), "Unsupported Expand op with different dims");

            std::vector<int> broadcast_axes;
            for (int i = 0; i < targetShape.size(); i++)
            {
                if (targetShape[i] != inpShape[i])
                {
                    if (inpShape[i] == 1)
                        broadcast_axes.push_back(i);
                    else
                        CV_Error(Error::StsError, format("Could not be broadcast by axis: %d", i));
                }
            }

            if (!haveVariables)
            {
                if (broadcast_axes.size() != 1)
                    CV_Error(Error::StsNotImplemented, "Expand op doesn't support multiple axes for constant input");

                Mat input = getBlob(node_proto, 0);
                input = input.reshape(0, total(inpShape, 0, broadcast_axes[0]));
                Mat output = cv::repeat(input, 1, targetShape[broadcast_axes[0]]);
                output = output.reshape(0, targetShape);
                addConstant(layerParams.name, output);
                return;
            }

            if (broadcast_axes.size() == 2 &&
                broadcast_axes[0] == broadcast_axes[1] - 1 && broadcast_axes[1] == inpShape.size() - 1)
            {
                LayerParams constParams;
                constParams.name = layerParams.name + "/const";
                CV_Assert(layer_id.find(constParams.name) == layer_id.end());
                constParams.type = "Const";

                Mat inp = Mat::ones(newShapeMat.total(), newShapeMat.ptr<int>(), CV_32F);
                constParams.blobs.push_back(inp);

                opencv_onnx::NodeProto proto;
                proto.add_output(constParams.name);
                addLayer(constParams, proto);

                layerParams.type = "Scale";
                layerParams.set("bias_term", false);
                node_proto.set_input(0, constParams.name);
                node_proto.set_input(1, srcName);
            }
            else if (broadcast_axes.size() == 1 && broadcast_axes[0] <= 1)
            {
                String base_name = layerParams.name + "/copy_";
                std::vector<std::string> input_names;
                for (int j = 0; j < targetShape[broadcast_axes[0]]; j++)
                {
                    std::ostringstream ss;
                    ss << j;
                    LayerParams copyLP;
                    copyLP.name = base_name + ss.str();
                    copyLP.type = "Identity";
                    CV_Assert(layer_id.find(copyLP.name) == layer_id.end());
                    input_names.push_back(copyLP.name);

                    node_proto.set_input(0, srcName);
                    node_proto.set_output(0, copyLP.name);
                    addLayer(copyLP, node_proto);
                }
                node_proto.clear_input();
                for (int i = 0; i < input_names.size(); i++)
                {
                    node_proto.add_input(input_names[i]);
                }
                layerParams.set("axis", broadcast_axes[0]);
                layerParams.type = "Concat";
                node_proto.set_output(0, layerParams.name);
            }
            else
                CV_Error(Error::StsNotImplemented, "Unsupported Expand op");
        }
        else if (layer_type == "Reshape")
        {
            CV_Assert(node_proto.input_size() == 2 || layerParams.has("shape"));

            if (node_proto.input_size() == 2) {
                Mat blob = getBlob(node_proto, 1);
                CV_Assert(blob.type() == CV_32SC1);

                layerParams.set("dim", DictValue::arrayInt<int*>(
                            blob.ptr<int>(), blob.total() ));

                if (layer_id.find(node_proto.input(0)) == layer_id.end()) {
                    std::vector<Mat> inputs(1, getBlob(node_proto, 0)), outputs;
                    runLayer(layerParams, inputs, outputs);
                    addConstant(layerParams.name, outputs[0]);
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
                    addConstant(layerParams.name, out);
                    return;
                }
                replaceLayerParam(layerParams, "shape", "dim");
            }
        }
        else if (layer_type == "Pad")
        {
            layerParams.type = "Padding";
            replaceLayerParam(layerParams, "mode", "type");
            if (node_proto.input_size() == 3 || node_proto.input_size() == 2)
            {
                // Paddings are in order begin0, begin1, .. beginN, end0, end1, ..., endN.
                // We need to shuffle it to begin0, end0, begin1, end1, ...
                Mat paddings = getBlob(node_proto, 1).reshape(1, 2);
                paddings = paddings.t();
                layerParams.set("paddings", DictValue::arrayInt(paddings.ptr<int>(), paddings.total()));

                if (node_proto.input_size() == 3)
                {
                    Mat value = getBlob(node_proto, 2);
                    layerParams.set("value", value.at<float>(0));
                }
            }
        }
        else if (layer_type == "Shape")
        {
            CV_Assert(node_proto.input_size() == 1);
            IterShape_t shapeIt = outShapes.find(node_proto.input(0));
            CV_Assert(shapeIt != outShapes.end());
            const MatShape& inpShape = shapeIt->second;

            Mat shapeMat(inpShape.size(), 1, CV_32S);
            for (int j = 0; j < inpShape.size(); ++j)
                shapeMat.at<int>(j) = inpShape[j];
            shapeMat.dims = 1;

            addConstant(layerParams.name, shapeMat);
            return;
        }
        else if (layer_type == "Cast")
        {
            if (constBlobs.find(node_proto.input(0)) != constBlobs.end())
            {
                Mat blob = getBlob(node_proto, 0);
                int type;
                switch (layerParams.get<int>("to"))
                {
                    case opencv_onnx::TensorProto_DataType_FLOAT:   type = CV_32F; break;
                    case opencv_onnx::TensorProto_DataType_UINT8:   type = CV_8U; break;
                    case opencv_onnx::TensorProto_DataType_UINT16:  type = CV_16U; break;
                    case opencv_onnx::TensorProto_DataType_FLOAT16: type = CV_16S; break;
                    case opencv_onnx::TensorProto_DataType_INT8:
                    case opencv_onnx::TensorProto_DataType_INT16:
                    case opencv_onnx::TensorProto_DataType_INT32:
                    case opencv_onnx::TensorProto_DataType_INT64:   type = CV_32S; break;
                    default: type = blob.type();
                }
                Mat dst;
                blob.convertTo(dst, type);
                dst.dims = blob.dims;
                addConstant(layerParams.name, dst);
                return;
            }
            else
                layerParams.type = "Identity";
        }
        else if (layer_type == "ConstantOfShape" || layer_type == "ConstantFill")
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

            MatShape inpShape = getBlob(node_proto, 0);
            for (int i = 0; i < inpShape.size(); i++)
                CV_CheckGT(inpShape[i], 0, "");
            Mat tensor(inpShape.size(), &inpShape[0], depth, Scalar(fill_value));
            addConstant(layerParams.name, tensor);
            return;
        }
        else if (layer_type == "Gather")
        {
            CV_Assert(node_proto.input_size() == 2);
            Mat indexMat = getBlob(node_proto, 1);
            CV_Assert_N(indexMat.type() == CV_32S, indexMat.total() == 1);
            int index = indexMat.at<int>(0);
            int axis = layerParams.get<int>("axis", 0);

            if ((constBlobs.find(node_proto.input(0)) != constBlobs.end()))
            {
                Mat input = getBlob(node_proto, 0);
                Mat out;
                std::vector<cv::Range> ranges(input.dims, Range::all());
                ranges[axis] = Range(index, index + 1);

                out = input(ranges);
                MatShape outShape = shape(out);
                if (outShape.size() > 1)
                {
                    outShape.erase(outShape.begin() + axis);
                    out.reshape(0, outShape);
                } else {
                    out.dims = 1;
                }
                addConstant(layerParams.name, out);
                return;
            }
            else
            {
                IterShape_t shapeIt = outShapes.find(node_proto.input(0));
                CV_Assert(shapeIt != outShapes.end());
                MatShape inpShape = shapeIt->second;

                LayerParams sliceLp;
                sliceLp.type = "Slice";
                sliceLp.name = inpShape.size() > 1 ? layerParams.name + "/slice" : layerParams.name;
                std::vector<int> begin(inpShape.size(), 0);
                std::vector<int> end(inpShape.size(), -1);
                begin[axis] = index;
                end[axis] = index + 1;

                cv::dnn::DictValue paramBegin = cv::dnn::DictValue::arrayInt(begin.data(), begin.size());
                cv::dnn::DictValue paramEnd = cv::dnn::DictValue::arrayInt(end.data(), end.size());
                sliceLp.set("begin", paramBegin);
                sliceLp.set("end", paramEnd);
                sliceLp.set("has_dynamic_shapes", hasDynamicShapes);

                if (inpShape.size() > 1)
                {
                    opencv_onnx::NodeProto proto;
                    proto.add_input(node_proto.input(0));
                    proto.add_output(sliceLp.name);
                    addLayer(sliceLp, proto);

                    inpShape.erase(inpShape.begin() + axis);
                    layerParams.type = "Reshape";
                    layerParams.set("axis", 0);
                    layerParams.set("dim", DictValue::arrayInt(&inpShape[0], inpShape.size()));
                    if (hasDynamicShapes)
                    {
                        std::vector<int> dynamicAxes;
                        std::vector<int> inputIndices;
                        for (int index = 0; index < inpShape.size(); ++index)
                            dynamicAxes.push_back(index);
                        for (int index = 0; index < inpShape.size(); ++index)
                            inputIndices.push_back(index);
                        layerParams.set("dynamic_axes", DictValue::arrayInt(dynamicAxes.data(), dynamicAxes.size()));
                        layerParams.set("input_indices", DictValue::arrayInt(inputIndices.data(), inputIndices.size()));
                    }
                    node_proto.set_input(0, sliceLp.name);
                }
                else
                {
                    layerParams = sliceLp;
                }
            }
        }
        else if (layer_type == "Concat")
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

            if (!hasVariableInps)
            {
                std::vector<Mat> inputs(node_proto.input_size()), concatenated;
                // Due constant folding we can get inputs with different number of dimensions
                // Insert the missing dimension to inputs
                MatShape inputShape;
                for (size_t i = 0; i < inputs.size(); ++i)
                {
                    inputs[i] = getBlob(node_proto, i);
                    if (inputs[i].size.dims() > inputShape.size())
                    {
                        inputShape = shape(inputs[i]);
                    }
                }

                // Concat-1 has default value for axis is 1: https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Concat-1
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
        }
        else if (layer_type == "Resize")
        {
            for (int i = 1; i < node_proto.input_size(); i++)
                CV_Assert(layer_id.find(node_proto.input(i)) == layer_id.end());

            if (layerParams.has("coordinate_transformation_mode"))
            {
                String interp_mode = layerParams.get<String>("coordinate_transformation_mode");
                CV_Assert_N(interp_mode != "tf_crop_and_resize", interp_mode != "tf_half_pixel_for_nn");

                layerParams.set("align_corners", interp_mode == "align_corners");
                if (layerParams.get<String>("mode") == "linear")
                {
                    layerParams.set("mode", interp_mode == "pytorch_half_pixel" ?
                                            "opencv_linear" : "bilinear");
                }
            }
            if (layerParams.get<String>("mode") == "linear" && framework_name == "pytorch")
                layerParams.set("mode", "opencv_linear");

            // input = [X, scales], [X, roi, scales] or [x, roi, scales, sizes]
            int foundScaleId = hasDynamicShapes ? node_proto.input_size() - 1
                                                : node_proto.input_size() > 2 ? 2 : 1;

            Mat scales = getBlob(node_proto, foundScaleId);
            if (scales.total() == 4)
            {
                layerParams.set("zoom_factor_y", scales.at<float>(2));
                layerParams.set("zoom_factor_x", scales.at<float>(3));
            }
            else
            {
                const std::string& inputLast = node_proto.input(node_proto.input_size() - 1);
                if (constBlobs.find(inputLast) != constBlobs.end())
                {
                    Mat shapes = getBlob(inputLast);
                    CV_CheckEQ(shapes.size[0], 4, "");
                    CV_CheckEQ(shapes.size[1], 1, "");
                    CV_CheckDepth(shapes.depth(), shapes.depth() == CV_32S || shapes.depth() == CV_32F, "");
                    if (shapes.depth() == CV_32F)
                        shapes.convertTo(shapes, CV_32S);
                    layerParams.set("width", shapes.at<int>(3));
                    layerParams.set("height", shapes.at<int>(2));
                }
            }
            replaceLayerParam(layerParams, "mode", "interpolation");
        }
        else if (layer_type == "Upsample")
        {
            //fused from Resize Subgraph
            if (layerParams.has("coordinate_transformation_mode"))
            {
                String interp_mode = layerParams.get<String>("coordinate_transformation_mode");
                CV_Assert_N(interp_mode != "tf_crop_and_resize", interp_mode != "tf_half_pixel_for_nn");

                layerParams.set("align_corners", interp_mode == "align_corners");
                if (layerParams.get<String>("mode") == "linear")
                {
                    layerParams.set("mode", interp_mode == "pytorch_half_pixel" ?
                                            "opencv_linear" : "bilinear");
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
        }
        else if (layer_type == "SoftMax" || layer_type == "LogSoftmax")
        {
            layerParams.type = "Softmax";
            layerParams.set("log_softmax", layer_type == "LogSoftmax");
        }
        else if (layer_type == "DetectionOutput")
        {
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
        }
        else
        {
            for (int j = 0; j < node_proto.input_size(); j++) {
                if (layer_id.find(node_proto.input(j)) == layer_id.end())
                    layerParams.blobs.push_back(getBlob(node_proto, j));
            }
        }
        addLayer(layerParams, node_proto);
    }
    catch (const cv::Exception& e)
    {
        CV_LOG_ERROR(NULL, "DNN/ONNX: ERROR during processing node with " << node_proto.input_size() << " inputs and " << node_proto.output_size() << " outputs: "
                << cv::format("[%s]:(%s)", layer_type.c_str(), name.c_str())
        );
        for (int i = 0; i < node_proto.input_size(); i++)
        {
            CV_LOG_INFO(NULL, "    Input[" << i << "] = '" << node_proto.input(i) << "'");
        }
        for (int i = 0; i < node_proto.output_size(); i++)
        {
            CV_LOG_INFO(NULL, "    Output[" << i << "] = '" << node_proto.output(i) << "'");
        }
        CV_Error(Error::StsError, cv::format("Node [%s]:(%s) parse error: %s", layer_type.c_str(), name.c_str(), e.what()));
    }
}

Net readNetFromONNX(const String& onnxFile)
{
    Net net;
    ONNXImporter onnxImporter(net, onnxFile.c_str());
    return net;
}

Net readNetFromONNX(const char* buffer, size_t sizeBuffer)
{
    Net net;
    ONNXImporter onnxImporter(net, buffer, sizeBuffer);
    return net;
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
    Mat mat = getMatFromTensor(tensor_proto);
    releaseONNXTensor(tensor_proto);
    return mat;
}

CV__DNN_EXPERIMENTAL_NS_END
}} // namespace

#endif
