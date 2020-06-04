// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>

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
    Mat getBlob(const opencv_onnx::NodeProto& node_proto, const std::map<std::string, Mat>& constBlobs, int index);

    LayerParams getLayerParams(const opencv_onnx::NodeProto& node_proto);
    bool isCeilMode(const LayerParams& layerParams);

    void addLayer(Net& dstNet, LayerParams& layerParams,
                  const opencv_onnx::NodeProto& node_proto,
                  std::map<std::string, LayerInfo>& layer_id,
                  std::map<std::string, MatShape>& outShapes);

public:

    ONNXImporter(const char *onnxFile)
    {
        std::fstream input(onnxFile, std::ios::in | std::ios::binary);

        if (!model_proto.ParseFromIstream(&input))
            CV_Error(Error::StsUnsupportedFormat, "Failed to parse onnx model");
    }

    ONNXImporter(const char* buffer, size_t sizeBuffer)
    {
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
    }

    void populateNet(Net dstNet);
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
            CV_Assert(attribute_proto.ints_size() == 2 || attribute_proto.ints_size() == 3);
            lp.set("kernel_size", parse(attribute_proto.ints()));
        }
        else if(attribute_name == "strides")
        {
            CV_Assert(attribute_proto.ints_size() == 2 || attribute_proto.ints_size() == 3);
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
                CV_Assert(attribute_proto.ints_size() == 4 || attribute_proto.ints_size() == 6);
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
            CV_Assert(attribute_proto.ints_size() == 2 || attribute_proto.ints_size() == 3);
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
            lp.set(attribute_proto.name(), parse(attribute_proto.ints()));
        }
        else if (attribute_proto.has_t())
        {
            opencv_onnx::TensorProto tensor = attribute_proto.t();
            Mat blob = getMatFromTensor(tensor);
            lp.blobs.push_back(blob);
        }
        else if (attribute_proto.has_g() || attribute_proto.strings_size() > 0 ||
                    attribute_proto.tensors_size() > 0 || attribute_proto.graphs_size() > 0)
        {
                CV_Error(Error::StsNotImplemented, "Unexpected attribute type");
        }
        else
            CV_Error(Error::StsNotImplemented, "Unsupported attribute type");
    }
    return lp;
}

Mat ONNXImporter::getBlob(const opencv_onnx::NodeProto& node_proto,
                    const std::map<std::string, Mat>& constBlobs, int index)
{
    CV_Assert(index < node_proto.input_size());
    std::map<std::string, Mat>::const_iterator constBlob;
    constBlob = constBlobs.find(node_proto.input(index));
    if (constBlob == constBlobs.end()) {
        CV_Error(Error::StsObjectNotFound,
             "Blob " + node_proto.input(index) + " not found in const blobs");
    }
    return constBlob->second;
}

void ONNXImporter::addLayer(Net& dstNet, LayerParams& layerParams,
                            const opencv_onnx::NodeProto& node_proto,
                            std::map<std::string, LayerInfo>& layer_id,
                            std::map<std::string, MatShape>& outShapes)
{
    std::map<std::string, LayerInfo>::iterator layerId;
    std::map<std::string, MatShape>::iterator shapeIt;

    int id = dstNet.addLayer(layerParams.name, layerParams.type, layerParams);
    for (int i = 0; i < node_proto.output_size(); ++i)
    {
        layer_id.insert(std::make_pair(node_proto.output(i), LayerInfo(id, i)));
    }

    std::vector<MatShape> layerInpShapes, layerOutShapes, layerInternalShapes;
    int inpNum = 0;
    for (int j = 0; j < node_proto.input_size(); j++) {
        layerId = layer_id.find(node_proto.input(j));
        if (layerId != layer_id.end()) {
            dstNet.connect(layerId->second.layerId, layerId->second.outputId, id, inpNum);
            ++inpNum;
            // Collect input shapes.
            shapeIt = outShapes.find(node_proto.input(j));
            CV_Assert(shapeIt != outShapes.end());
            layerInpShapes.push_back(shapeIt->second);
        }
    }
    // Compute shape of output blob for this layer.
    Ptr<Layer> layer = dstNet.getLayer(id);
    layer->getMemoryShapes(layerInpShapes, 0, layerOutShapes, layerInternalShapes);
    for (int i = 0; i < node_proto.output_size() && i < (int)layerOutShapes.size(); ++i)
    {
        outShapes[node_proto.output(i)] = layerOutShapes[i];
    }
}

static void addConstant(const std::string& name,
                        const Mat& blob,
                        std::map<std::string, Mat>& constBlobs,
                        std::map<std::string, MatShape>& outShapes)
{
    constBlobs.insert(std::make_pair(name, blob));
    outShapes.insert(std::make_pair(name, shape(blob)));
}

void ONNXImporter::populateNet(Net dstNet)
{
    CV_Assert(model_proto.has_graph());
    opencv_onnx::GraphProto graph_proto = model_proto.graph();

    simplifySubgraphs(graph_proto);

    std::map<std::string, Mat> constBlobs = getGraphTensors(graph_proto);
    // List of internal blobs shapes.
    std::map<std::string, MatShape> outShapes;
    // Add all the inputs shapes. It includes as constant blobs as network's inputs shapes.
    for (int i = 0; i < graph_proto.input_size(); ++i)
    {
        opencv_onnx::ValueInfoProto valueInfoProto = graph_proto.input(i);
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
        }
        outShapes[valueInfoProto.name()] = inpShape;
    }

    std::string framework_name;
    if (model_proto.has_producer_name()) {
        framework_name = model_proto.producer_name();
    }

    // create map with network inputs (without const blobs)
    std::map<std::string, LayerInfo> layer_id;
    std::map<std::string, LayerInfo>::iterator layerId;
    std::map<std::string, MatShape>::iterator shapeIt;
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

    int layersSize = graph_proto.node_size();
    LayerParams layerParams;
    opencv_onnx::NodeProto node_proto;

    for(int li = 0; li < layersSize; li++)
    {
        node_proto = graph_proto.node(li);
        layerParams = getLayerParams(node_proto);
        CV_Assert(node_proto.output_size() >= 1);
        layerParams.name = node_proto.output(0);

        std::string layer_type = node_proto.op_type();
        layerParams.type = layer_type;


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
        else if (layer_type == "GlobalAveragePool" || layer_type == "GlobalMaxPool" || layer_type == "ReduceMean")
        {
            CV_Assert(node_proto.input_size() == 1);
            layerParams.type = "Pooling";
            layerParams.set("pool", layer_type == "GlobalMaxPool"? "MAX" : "AVE");
            layerParams.set("global_pooling", layer_type == "GlobalAveragePool" || layer_type == "GlobalMaxPool");

            if (layer_type == "ReduceMean")
            {
                if (layerParams.get<int>("keepdims") == 0 || !layerParams.has("axes"))
                    CV_Error(Error::StsNotImplemented, "Unsupported mode of ReduceMean operation.");

                MatShape inpShape = outShapes[node_proto.input(0)];
                DictValue axes = layerParams.get("axes");
                if (inpShape.size() == 3 && axes.size() <= 2)
                {
                    int axis = axes.get<int>(0);
                    CV_CheckNE(axis, 0, "");
                    outShapes[layerParams.name] = inpShape;
                    outShapes[layerParams.name][axis] = 1;

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
                    addLayer(dstNet, reshapeLp, proto, layer_id, outShapes);

                    LayerParams avgLp;
                    avgLp.name = layerParams.name + "/avg";
                    avgLp.type = "Pooling";
                    CV_Assert(layer_id.find(avgLp.name) == layer_id.end());
                    avgLp.set("pool", "ave");
                    if (axes.size() == 2)
                    {
                        CV_CheckEQ(axes.get<int>(0), 1, "Unsupported ReduceMean mode");
                        CV_CheckEQ(axes.get<int>(1), 2, "Unsupported ReduceMean mode");
                        avgLp.set("global_pooling", true);
                        outShapes[layerParams.name][axes.get<int>(1)] = 1;
                    }
                    else
                    {
                        avgLp.set(axis == 2 ? "global_pooling_w" : "global_pooling_h", true);
                        avgLp.set(axis == 2 ? "kernel_h" : "kernel_w", 1);
                    }

                    node_proto.set_input(0, reshapeLp.name);
                    node_proto.set_output(0, avgLp.name);
                    addLayer(dstNet, avgLp, node_proto, layer_id, outShapes);

                    layerParams.type = "Flatten";
                    layerParams.set("axis", 0);
                    layerParams.set("end_axis", 1);

                    node_proto.set_input(0, avgLp.name);
                    node_proto.set_output(0, layerParams.name);
                }
                else
                {
                    if (inpShape.size() != 4 && inpShape.size() != 5)
                    CV_Error(Error::StsNotImplemented, "Unsupported input shape of reduce_mean operation.");

                    CV_Assert(axes.size() <= inpShape.size() - 2);
                    std::vector<int> kernel_size(inpShape.size() - 2, 1);
                    for (int i = 0; i < axes.size(); i++) {
                        int axis = axes.get<int>(i);
                        CV_Assert_N(axis >= 2 + i, axis < inpShape.size());
                        kernel_size[axis - 2] = inpShape[axis];
                    }
                    layerParams.set("kernel_size", DictValue::arrayInt(&kernel_size[0], kernel_size.size()));
                }
            }
        }
        else if (layer_type == "Slice")
        {
            int axis = 0;
            std::vector<int> begin;
            std::vector<int> end;
            int inp_size = node_proto.input_size();

            if (inp_size == 1)
            {
                if (layerParams.has("steps"))
                {
                    DictValue steps = layerParams.get("steps");
                    for (int i = 0; i < steps.size(); ++i)
                    {
                        if (steps.get<int>(i) != 1)
                            CV_Error(Error::StsNotImplemented,
                                "Slice layer only supports steps = 1");
                    }
                }
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
            } else {
                CV_Assert(inp_size >= 3);
                for (int i = 1; i < inp_size; i++) {
                    CV_Assert(constBlobs.find(node_proto.input(i)) != constBlobs.end());
                }
                Mat start_blob = getBlob(node_proto, constBlobs, 1);
                Mat end_blob   = getBlob(node_proto, constBlobs, 2);
                CV_Assert(start_blob.total() == end_blob.total());

                if (inp_size > 3) {
                    Mat axes_blob = getBlob(node_proto, constBlobs, 3);
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
                    Mat step_blob = getBlob(node_proto, constBlobs, 4);

                    // Very strange application for Slice op with tensor reversing.
                    // We just workaround it for 2d constants.
                    if (constBlobs.find(node_proto.input(0)) != constBlobs.end() &&
                        axis == 0 &&
                        start_blob.at<int>(0) == -1 && step_blob.at<int>(0) == -1 &&
                        end_blob.at<int>(0) == std::numeric_limits<int32_t>::min())
                    {
                        Mat inp = getBlob(node_proto, constBlobs, 0);
                        if (inp.dims == 2)
                        {
                            Mat flipped;
                            flip(inp, flipped, 0);
                            addConstant(layerParams.name, flipped, constBlobs, outShapes);
                            continue;
                        }
                    }
                    CV_CheckEQ(countNonZero(step_blob != 1), 0, "Slice layer only supports steps = 1");
                }
            }
            layerParams.set("begin", DictValue::arrayInt(&begin[0], begin.size()));
            layerParams.set("end", DictValue::arrayInt(&end[0], end.size()));
            layerParams.set("axis", axis);

            if (constBlobs.find(node_proto.input(0)) != constBlobs.end())
            {
                Mat inp = getBlob(node_proto, constBlobs, 0);
                std::vector<Mat> inputs, sliced;
                inputs.push_back(inp);
                runLayer(layerParams, inputs, sliced);
                CV_Assert(sliced.size() == 1);
                addConstant(layerParams.name, sliced[0], constBlobs, outShapes);
                continue;
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
                Mat blob_0 = getBlob(node_proto, constBlobs, 0);
                Mat blob_1 = getBlob(node_proto, constBlobs, 1);
                CV_Assert(blob_0.size == blob_1.size);
                Mat output = isSub ? (blob_0 - blob_1) : (blob_0 + blob_1);
                addConstant(layerParams.name, output, constBlobs, outShapes);
                continue;
            }
            else if (is_const_0 || is_const_1)
            {
                int const_blob_id = is_const_0 ? 0 : 1;
                Mat blob = getBlob(node_proto, constBlobs, const_blob_id);
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
                        constParams.blobs.push_back(blob);
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
                    layerId = layer_id.find(node_proto.input(1));
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
            addConstant(layerParams.name, layerParams.blobs[0], constBlobs, outShapes);
            continue;
        }
        else if (layer_type == "LSTM")
        {
            LayerParams lstmParams = layerParams;
            lstmParams.name += "/lstm";

            // https://pytorch.org/docs/stable/nn.html#lstm
            CV_Assert(node_proto.input_size() == 7);
            Mat Wx = getBlob(node_proto, constBlobs, 1);
            Mat Wh = getBlob(node_proto, constBlobs, 2);
            Mat b = getBlob(node_proto, constBlobs, 3);
            CV_CheckEQ(countNonZero(getBlob(node_proto, constBlobs, 5)), 0, "Unsupported non zero initial_h");
            CV_CheckEQ(countNonZero(getBlob(node_proto, constBlobs, 6)), 0, "Unsupported non zero initial_c");
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
            addLayer(dstNet, lstmParams, node_proto, layer_id, outShapes);

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
            layerParams.blobs.push_back(getBlob(node_proto, constBlobs, 1));
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
            layerParams.blobs[2] = getBlob(node_proto, constBlobs, 1);  // weightData
            layerParams.blobs[3] = getBlob(node_proto, constBlobs, 2);  // biasData
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
            layerId = layer_id.find(node_proto.input(0));
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

            Mat meanData = getBlob(node_proto, constBlobs, 3);
            Mat stdData =  getBlob(node_proto, constBlobs, 4);

            layerParams.blobs.push_back(meanData);
            layerParams.blobs.push_back(stdData);

            if (!node_proto.input(1).empty()) {
                layerParams.set("has_weight", true);
                layerParams.blobs.push_back(getBlob(node_proto, constBlobs, 1));  // weightData
            } else {
                layerParams.set("has_weight", false);
            }

            if (!node_proto.input(2).empty()) {
                layerParams.set("has_bias", true);
                layerParams.blobs.push_back(getBlob(node_proto, constBlobs, 2)); // biasData
            } else {
                layerParams.set("has_bias", false);
            }
        }
        else if (layer_type == "Gemm")
        {
            CV_Assert(node_proto.input_size() >= 2);
            layerParams.type = "InnerProduct";
            Mat weights = getBlob(node_proto, constBlobs, 1);
            int ind_num_out = 0;
            if (layerParams.has("transB") && !layerParams.get<int>("transB")) {
                transpose(weights, weights);
                ind_num_out = 1;
            }
            layerParams.blobs.push_back(weights);

            if (node_proto.input_size() == 3) {
                Mat bias = getBlob(node_proto, constBlobs, 2);
                layerParams.blobs.push_back(bias);
            }

            layerParams.set("num_output", layerParams.blobs[0].size[ind_num_out]);
            layerParams.set("bias_term", node_proto.input_size() == 3);
        }
        else if (layer_type == "MatMul")
        {
            CV_Assert(node_proto.input_size() == 2);
            layerParams.type = "InnerProduct";
            layerParams.set("bias_term", false);

            if (constBlobs.find(node_proto.input(1)) != constBlobs.end())
            {
                Mat blob = getBlob(node_proto, constBlobs, 1);
                layerParams.blobs.push_back(blob.t());
                layerParams.set("num_output", layerParams.blobs[0].size[0]);
            }
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
                Mat blob = getBlob(node_proto, constBlobs, constId);
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
            else if (outShapes[node_proto.input(0)] == outShapes[node_proto.input(1)])
            {
                layerParams.type = "Eltwise";
                layerParams.set("operation", isDiv ? "div" : "prod");
            }
            else
            {
                if (isDiv)
                {
                    LayerParams powerParams;
                    powerParams.name = layerParams.name + "/inv";
                    powerParams.type = "Power";
                    powerParams.set("power", -1);

                    //Create Power layer
                    int id = dstNet.addLayer(powerParams.name, powerParams.type, powerParams);
                    //Connect to input
                    layerId = layer_id.find(node_proto.input(1));
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

            if (!haveVariables)
            {
                Mat inp0 = getBlob(node_proto, constBlobs, 0);
                Mat inp1 = getBlob(node_proto, constBlobs, 1);
                if (inp0.size != inp1.size)
                    CV_Error(Error::StsNotImplemented, "Constant multiply with different shapes");

                Mat out;
                if (isDiv)
                    divide(inp0, inp1, out);
                else
                    multiply(inp0, inp1, out);

                out = out.reshape(1, inp0.dims, inp0.size);
                out.dims = inp0.dims;  // to workaround dims == 1
                addConstant(layerParams.name, out, constBlobs, outShapes);
                continue;
            }
        }
        else if (layer_type == "Conv")
        {
            CV_Assert(node_proto.input_size() >= 2);
            layerParams.type = "Convolution";
            for (int j = 1; j < node_proto.input_size(); j++) {
                layerParams.blobs.push_back(getBlob(node_proto, constBlobs, j));
            }
            layerParams.set("num_output", layerParams.blobs[0].size[0]);
            layerParams.set("bias_term", node_proto.input_size() == 3);
        }
        else if (layer_type == "ConvTranspose")
        {
            CV_Assert(node_proto.input_size() >= 2);
            layerParams.type = "Deconvolution";
            for (int j = 1; j < node_proto.input_size(); j++) {
                layerParams.blobs.push_back(getBlob(node_proto, constBlobs, j));
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
                std::vector<Mat> inputs(1, getBlob(node_proto, constBlobs, 0)), transposed;
                runLayer(layerParams, inputs, transposed);
                CV_Assert(transposed.size() == 1);
                addConstant(layerParams.name, transposed[0], constBlobs, outShapes);
                continue;
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
            }
            else
                layerParams.type = "Identity";

            if (constBlobs.find(node_proto.input(0)) != constBlobs.end())
            {
                Mat inp = getBlob(node_proto, constBlobs, 0);
                Mat out = inp.reshape(1, outShape);
                out.dims = outShape.size();  // to workaround dims == 1
                addConstant(layerParams.name, out, constBlobs, outShapes);
                continue;
            }
        }
        else if (layer_type == "Flatten")
        {
            CV_CheckEQ(node_proto.input_size(), 1, "");
            if (constBlobs.find(node_proto.input(0)) != constBlobs.end())
            {
                Mat input = getBlob(node_proto, constBlobs, 0);
                int axis = clamp(layerParams.get<int>("axis", 1), input.dims);

                std::vector<int> out_size(&input.size[0], &input.size[0] + axis);
                out_size.push_back(input.total(axis));
                Mat output = input.reshape(1, out_size);
                addConstant(layerParams.name, output, constBlobs, outShapes);
                continue;
            }
        }
        else if (layer_type == "Unsqueeze")
        {
            CV_Assert(node_proto.input_size() == 1);
            DictValue axes = layerParams.get("axes");
            if (constBlobs.find(node_proto.input(0)) != constBlobs.end())
            {
                // Constant input.
                Mat input = getBlob(node_proto, constBlobs, 0);

                std::vector<int> dims;
                for (int j = 0; j < input.dims; j++) {
                    dims.push_back(input.size[j]);
                }
                CV_Assert(axes.getIntValue(axes.size()-1) <= dims.size());
                for (int j = 0; j < axes.size(); j++) {
                    dims.insert(dims.begin() + axes.getIntValue(j), 1);
                }

                Mat out = input.reshape(0, dims);
                addConstant(layerParams.name, out, constBlobs, outShapes);
                continue;
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
        }
        else if (layer_type == "Expand")
        {
            CV_CheckEQ(node_proto.input_size(), 2, "");
            CV_Assert(constBlobs.find(node_proto.input(1)) != constBlobs.end());
            Mat newShapeMat = getBlob(node_proto, constBlobs, 1);
            MatShape targetShape(newShapeMat.ptr<int>(), newShapeMat.ptr<int>() + newShapeMat.total());

            shapeIt = outShapes.find(node_proto.input(0));
            CV_Assert(shapeIt != outShapes.end());
            MatShape inpShape = shapeIt->second;
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
                addLayer(dstNet, constParams, proto, layer_id, outShapes);

                layerParams.type = "Scale";
                layerParams.set("bias_term", false);
                node_proto.set_input(0, constParams.name);
                node_proto.set_input(1, shapeIt->first);
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

                    node_proto.set_output(0, copyLP.name);
                    addLayer(dstNet, copyLP, node_proto, layer_id, outShapes);
                }
                node_proto.clear_input();
                for (int i = 0; i < input_names.size(); i++)
                {
                    node_proto.add_input(input_names[i]);
                }
                layerParams.set("axis", broadcast_axes[0]);
                layerParams.type = "Concat";
            }
            else
                CV_Error(Error::StsNotImplemented, "Unsupported Expand op");
        }
        else if (layer_type == "Reshape")
        {
            CV_Assert(node_proto.input_size() == 2 || layerParams.has("shape"));

            if (node_proto.input_size() == 2) {
                Mat blob = getBlob(node_proto, constBlobs, 1);
                CV_Assert(blob.type() == CV_32SC1);

                layerParams.set("dim", DictValue::arrayInt<int*>(
                            blob.ptr<int>(), blob.total() ));

                if (layer_id.find(node_proto.input(0)) == layer_id.end()) {
                    std::vector<Mat> inputs(1, getBlob(node_proto, constBlobs, 0)), outputs;
                    runLayer(layerParams, inputs, outputs);
                    addConstant(layerParams.name, outputs[0], constBlobs, outShapes);
                    continue;
                }
            }
            else {
                DictValue shape = layerParams.get("shape");
                std::vector<int> dim;
                for (int j = 0; j < shape.size(); j++) {
                    dim.push_back(shape.getIntValue(j));
                }

                if (layer_id.find(node_proto.input(0)) == layer_id.end()) {
                    Mat input = getBlob(node_proto, constBlobs, 0);
                    Mat out = input.reshape(0, dim);
                    addConstant(layerParams.name, out, constBlobs, outShapes);
                    continue;
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
                Mat paddings = getBlob(node_proto, constBlobs, 1).reshape(1, 2);
                paddings = paddings.t();
                layerParams.set("paddings", DictValue::arrayInt(paddings.ptr<int>(), paddings.total()));

                if (node_proto.input_size() == 3)
                {
                    Mat value = getBlob(node_proto, constBlobs, 2);
                    layerParams.set("value", value.at<float>(0));
                }
            }
        }
        else if (layer_type == "Shape")
        {
            CV_Assert(node_proto.input_size() == 1);
            shapeIt = outShapes.find(node_proto.input(0));
            CV_Assert(shapeIt != outShapes.end());
            MatShape inpShape = shapeIt->second;

            Mat shapeMat(inpShape.size(), 1, CV_32S);
            for (int j = 0; j < inpShape.size(); ++j)
                shapeMat.at<int>(j) = inpShape[j];
            shapeMat.dims = 1;

            addConstant(layerParams.name, shapeMat, constBlobs, outShapes);
            continue;
        }
        else if (layer_type == "Cast")
        {
            if (constBlobs.find(node_proto.input(0)) != constBlobs.end())
            {
                Mat blob = getBlob(node_proto, constBlobs, 0);
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
                blob.convertTo(blob, type);
                addConstant(layerParams.name, blob, constBlobs, outShapes);
                continue;
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

            MatShape inpShape = getBlob(node_proto, constBlobs, 0);
            for (int i = 0; i < inpShape.size(); i++)
                CV_CheckGT(inpShape[i], 0, "");
            Mat tensor(inpShape.size(), &inpShape[0], depth, Scalar(fill_value));
            addConstant(layerParams.name, tensor, constBlobs, outShapes);
            continue;
        }
        else if (layer_type == "Gather")
        {
            CV_Assert(node_proto.input_size() == 2);
            Mat input = getBlob(node_proto, constBlobs, 0);
            Mat indexMat = getBlob(node_proto, constBlobs, 1);
            CV_Assert_N(indexMat.type() == CV_32S, indexMat.total() == 1);
            int index = indexMat.at<int>(0);

            Mat out;
            if (layerParams.has("axis"))
            {
                int axis = layerParams.get<int>("axis");

                std::vector<cv::Range> ranges(input.dims, Range::all());
                ranges[axis] = Range(index, index + 1);

                out = input(ranges);
            }
            else
            {
                CV_Assert(index < input.total());
                const int dims = input.dims;
                input = input.reshape(1, 1);
                input.dims = 2;
                out = input.reshape(1, 1).colRange(index, index + 1);
                out.dims = dims;
            }
            addConstant(layerParams.name, out, constBlobs, outShapes);
            continue;
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
                for (size_t i = 0; i < inputs.size(); ++i)
                {
                    inputs[i] = getBlob(node_proto, constBlobs, i);
                }
                runLayer(layerParams, inputs, concatenated);

                CV_Assert(concatenated.size() == 1);
                addConstant(layerParams.name, concatenated[0], constBlobs, outShapes);
                continue;
            }
        }
        else if (layer_type == "Resize")
        {
            for (int i = 1; i < node_proto.input_size(); i++)
                CV_Assert(layer_id.find(node_proto.input(i)) == layer_id.end());

            String interp_mode = layerParams.get<String>("coordinate_transformation_mode");
            CV_Assert_N(interp_mode != "tf_crop_and_resize", interp_mode != "tf_half_pixel_for_nn");

            layerParams.set("align_corners", interp_mode == "align_corners");
            Mat shapes = getBlob(node_proto, constBlobs, node_proto.input_size() - 1);
            CV_CheckEQ(shapes.size[0], 4, "");
            CV_CheckEQ(shapes.size[1], 1, "");
            CV_CheckTypeEQ(shapes.depth(), CV_32S, "");
            int height = shapes.at<int>(2);
            int width  = shapes.at<int>(3);
            if (node_proto.input_size() == 3)
            {
                shapeIt = outShapes.find(node_proto.input(0));
                CV_Assert(shapeIt != outShapes.end());
                MatShape scales = shapeIt->second;
                height *= scales[2];
                width  *= scales[3];
            }
            layerParams.set("width", width);
            layerParams.set("height", height);

            if (layerParams.get<String>("mode") == "linear") {
                layerParams.set("mode", interp_mode == "pytorch_half_pixel" ?
                                        "opencv_linear" : "bilinear");
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
                Mat scales = getBlob(node_proto, constBlobs, 1);
                CV_Assert(scales.total() == 4);
                layerParams.set("zoom_factor_y", scales.at<float>(2));
                layerParams.set("zoom_factor_x", scales.at<float>(3));
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
                Mat priors = getBlob(node_proto, constBlobs, 2);

                LayerParams constParams;
                constParams.name = layerParams.name + "/priors";
                constParams.type = "Const";
                constParams.blobs.push_back(priors);

                opencv_onnx::NodeProto priorsProto;
                priorsProto.add_output(constParams.name);
                addLayer(dstNet, constParams, priorsProto, layer_id, outShapes);

                node_proto.set_input(2, constParams.name);
            }
        }
        else
        {
            for (int j = 0; j < node_proto.input_size(); j++) {
                if (layer_id.find(node_proto.input(j)) == layer_id.end())
                    layerParams.blobs.push_back(getBlob(node_proto, constBlobs, j));
            }
        }
        addLayer(dstNet, layerParams, node_proto, layer_id, outShapes);
    }
}

Net readNetFromONNX(const String& onnxFile)
{
    ONNXImporter onnxImporter(onnxFile.c_str());
    Net net;
    onnxImporter.populateNet(net);
    return net;
}

Net readNetFromONNX(const char* buffer, size_t sizeBuffer)
{
    ONNXImporter onnxImporter(buffer, sizeBuffer);
    Net net;
    onnxImporter.populateNet(net);
    return net;
}

Net readNetFromONNX(const std::vector<uchar>& buffer)
{
    return readNetFromONNX(reinterpret_cast<const char*>(buffer.data()), buffer.size());
}

Mat readTensorFromONNX(const String& path)
{
    opencv_onnx::TensorProto tensor_proto = opencv_onnx::TensorProto();
    std::fstream input(path.c_str(), std::ios::in | std::ios::binary);
    if (!tensor_proto.ParseFromIstream(&input)) {
        CV_Error(Error::StsUnsupportedFormat, "Failed to parse data");
    }
    Mat mat = getMatFromTensor(tensor_proto);
    releaseONNXTensor(tensor_proto);
    return mat;
}

CV__DNN_EXPERIMENTAL_NS_END
}} // namespace

#endif
