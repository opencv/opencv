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

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN


class ONNXImporter
{
    opencv_onnx::ModelProto model_proto;
    struct LayerInfo {
        int layerId;
        int outputId;
        LayerInfo(int _layerId, int _outputId) : layerId(_layerId), outputId(_outputId) {}
    };

    std::map<std::string, Mat> getGraphTensors(
                                    const opencv_onnx::GraphProto& graph_proto);
    Mat getBlob(const opencv_onnx::NodeProto& node_proto, const std::map<std::string, Mat>& constBlobs, int index);

    LayerParams getLayerParams(const opencv_onnx::NodeProto& node_proto);
    bool isCeilMode(const LayerParams& layerParams);

public:

    ONNXImporter(const char *onnxFile)
    {
        std::fstream input(onnxFile, std::ios::in | std::ios::binary);

        if (!model_proto.ParseFromIstream(&input))
            CV_Error(Error::StsUnsupportedFormat, "Failed to parse onnx model");
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

template<typename T1, typename T2>
void convertInt64ToInt32(const T1& src, T2& dst, int size)
{
    for (int i = 0; i < size; i++) {
        if (src[i] < std::numeric_limits<int32_t>::min() || src[i] > std::numeric_limits<int32_t>::max()) {
            CV_Error(Error::StsOutOfRange, "Input is out of OpenCV 32S range");
        }
        dst[i] = saturate_cast<int32_t>(src[i]);
    }
}

Mat getMatFromTensor(opencv_onnx::TensorProto& tensor_proto)
{
    CV_Assert(!tensor_proto.raw_data().empty() || !tensor_proto.float_data().empty()
                    || !tensor_proto.double_data().empty() || !tensor_proto.int64_data().empty());

    opencv_onnx::TensorProto_DataType datatype = tensor_proto.data_type();
    Mat blob;
    std::vector<int> sizes;
    for (int i = 0; i < tensor_proto.dims_size(); i++) {
            sizes.push_back(tensor_proto.dims(i));
    }
    if (sizes.empty())
        sizes.assign(1, 1);
    if (datatype == opencv_onnx::TensorProto_DataType_FLOAT) {

        if (!tensor_proto.float_data().empty()) {
            const ::google::protobuf::RepeatedField<float> field = tensor_proto.float_data();
            Mat(sizes, CV_32FC1, (void*)field.data()).copyTo(blob);
        }
        else {
            char* val = const_cast<char*>(tensor_proto.raw_data().c_str());
            Mat(sizes, CV_32FC1, val).copyTo(blob);
        }
    }
    else if (datatype == opencv_onnx::TensorProto_DataType_DOUBLE)
    {
        const ::google::protobuf::RepeatedField<double> field = tensor_proto.double_data();
        CV_Assert(!field.empty());
        Mat(sizes, CV_64FC1, (void*)field.data()).convertTo(blob, CV_32FC1);
    }
    else if (datatype == opencv_onnx::TensorProto_DataType_INT64)
    {
        blob.create(sizes, CV_32SC1);
        int32_t* dst = reinterpret_cast<int32_t*>(blob.data);

        if (!tensor_proto.int64_data().empty()) {
            ::google::protobuf::RepeatedField< ::google::protobuf::int64> src = tensor_proto.int64_data();
            convertInt64ToInt32(src, dst, blob.total());
        }
        else
        {
            char* val = const_cast<char*>(tensor_proto.raw_data().c_str());
            int64_t* src = reinterpret_cast<int64_t*>(val);
            convertInt64ToInt32(src, dst, blob.total());
        }
    }
    else
        CV_Error(Error::StsUnsupportedFormat, "Unsupported data type: " +
                        opencv_onnx::TensorProto_DataType_Name(datatype));
    if (tensor_proto.dims_size() == 0)
        blob.dims = 1;  // To force 1-dimensional cv::Mat for scalars.
    return blob;
}

void runLayer(Ptr<Layer> layer, const std::vector<Mat>& inputs,
              std::vector<Mat>& outputs)
{
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

LayerParams ONNXImporter::getLayerParams(const opencv_onnx::NodeProto& node_proto)
{
    LayerParams lp;
    for(int i = 0; i < node_proto.attribute_size(); i++)
    {
        opencv_onnx::AttributeProto attribute_proto = node_proto.attribute(i);
        std::string attribute_name = attribute_proto.name();

        if(attribute_name == "kernel_shape")
        {
            CV_Assert(attribute_proto.ints_size() == 2);
            lp.set("kernel_h", saturate_cast<int32_t>(attribute_proto.ints(0)));
            lp.set("kernel_w", saturate_cast<int32_t>(attribute_proto.ints(1)));
        }
        else if(attribute_name == "strides")
        {
            CV_Assert(attribute_proto.ints_size() == 2);
            lp.set("stride_h", saturate_cast<int32_t>(attribute_proto.ints(0)));
            lp.set("stride_w", saturate_cast<int32_t>(attribute_proto.ints(1)));
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
                CV_Assert(attribute_proto.ints_size() == 4);
                lp.set("pad_t", saturate_cast<int32_t>(attribute_proto.ints(0)));
                lp.set("pad_l", saturate_cast<int32_t>(attribute_proto.ints(1)));
                lp.set("pad_b", saturate_cast<int32_t>(attribute_proto.ints(2)));
                lp.set("pad_r", saturate_cast<int32_t>(attribute_proto.ints(3)));
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
            CV_Assert(attribute_proto.ints_size() == 2);
            lp.set("dilation_h",  saturate_cast<int32_t>(attribute_proto.ints(0)));
            lp.set("dilation_w",  saturate_cast<int32_t>(attribute_proto.ints(1)));
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
            const ::google::protobuf::RepeatedField< ::google::protobuf::int64> src = attribute_proto.ints();
            std::vector<int32_t> dst(attribute_proto.ints_size());
            convertInt64ToInt32(src, dst, attribute_proto.ints_size());
            lp.set(attribute_proto.name(), DictValue::arrayInt(&dst[0], attribute_proto.ints_size()));
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


bool ONNXImporter::isCeilMode(const LayerParams& layerParams) {
    if (!layerParams.has("pad_mode")) {
        if (layerParams.has("pad_h")) {
            return layerParams.get<int>("pad_h") != layerParams.get<int>("pad_b") ||
                        layerParams.get<int>("pad_w") != layerParams.get<int>("pad_r");
        }
        else
            return false; // all pads == 0
    }
    return true;
}

void ONNXImporter::populateNet(Net dstNet)
{
    CV_Assert(model_proto.has_graph());
    opencv_onnx::GraphProto graph_proto = model_proto.graph();
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
            layerParams.set("ceil_mode", isCeilMode(layerParams));
        }
        else if (layer_type == "AveragePool")
        {
            layerParams.type = "Pooling";
            layerParams.set("pool", "AVE");
            layerParams.set("ceil_mode", isCeilMode(layerParams));
            layerParams.set("ave_pool_padded_area", framework_name == "pytorch");
        }
        else if (layer_type == "GlobalAveragePool" || layer_type == "GlobalMaxPool")
        {
            layerParams.type = "Pooling";
            layerParams.set("pool", layer_type == "GlobalAveragePool" ? "AVE" : "MAX");
            layerParams.set("global_pooling", true);
        }
        else if (layer_type == "Add" || layer_type == "Sum")
        {
            if (layer_id.find(node_proto.input(1)) == layer_id.end())
            {
                Mat blob = getBlob(node_proto, constBlobs, 1);
                blob = blob.reshape(1, 1);
                if (blob.total() == 1) {
                    layerParams.type = "Power";
                    layerParams.set("shift", blob.at<float>(0));
                }
                else {
                    layerParams.type = "Scale";
                    layerParams.set("bias_term", true);
                    layerParams.blobs.push_back(blob);
                }
            }
            else {
                layerParams.type = "Eltwise";
            }
        }
        else if (layer_type == "Sub")
        {
            Mat blob = getBlob(node_proto, constBlobs, 1);
            if (blob.total() == 1) {
                layerParams.type = "Power";
                layerParams.set("shift", -blob.at<float>(0));
            }
            else {
                layerParams.type = "Scale";
                layerParams.set("has_bias", true);
                layerParams.blobs.push_back(-1.0f * blob.reshape(1, 1));
            }
        }
        else if (layer_type == "Div")
        {
            Mat blob = getBlob(node_proto, constBlobs, 1);
            CV_Assert_N(blob.type() == CV_32F, blob.total());
            if (blob.total() == 1)
            {
                layerParams.set("scale", 1.0f / blob.at<float>(0));
                layerParams.type = "Power";
            }
            else
            {
                layerParams.type = "Scale";
                divide(1.0, blob, blob);
                layerParams.blobs.push_back(blob);
                layerParams.set("bias_term", false);
            }
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
            constBlobs.insert(std::make_pair(layerParams.name, layerParams.blobs[0]));
            continue;
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
        else if (layer_type == "LeakyRelu")
        {
            layerParams.type = "ReLU";
            replaceLayerParam(layerParams, "alpha", "negative_slope");
        }
        else if (layer_type == "LRN")
        {
            replaceLayerParam(layerParams, "size", "local_size");
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
            Mat blob = getBlob(node_proto, constBlobs, 1);
            layerParams.blobs.push_back(blob.t());
            layerParams.set("bias_term", false);
            layerParams.set("num_output", layerParams.blobs[0].size[0]);
        }
        else if (layer_type == "Mul")
        {
            CV_Assert(node_proto.input_size() == 2);
            if (layer_id.find(node_proto.input(1)) == layer_id.end()) {
                Mat blob = getBlob(node_proto, constBlobs, 1);
                blob = blob.reshape(1, 1);
                if (blob.total() == 1) {
                    layerParams.set("scale", blob.at<float>(0));
                    layerParams.type = "Power";
                }
                else {
                    layerParams.blobs.push_back(blob);
                    layerParams.type = "Scale";
                }
            }
            else {
                layerParams.type = "Eltwise";
                layerParams.set("operation", "prod");
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
        }
        else if (layer_type == "Transpose")
        {
            layerParams.type = "Permute";
            replaceLayerParam(layerParams, "perm", "order");
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
                constBlobs.insert(std::make_pair(layerParams.name, out));
                continue;
            }

            // Variable input.
            if (axes.size() != 1)
                CV_Error(Error::StsNotImplemented, "Multidimensional unsqueeze");

            int dims[] = {1, -1};
            layerParams.type = "Reshape";
            layerParams.set("axis", axes.getIntValue(0));
            layerParams.set("num_axes", 1);
            layerParams.set("dim", DictValue::arrayInt(&dims[0], 2));
        }
        else if (layer_type == "Reshape")
        {
            CV_Assert(node_proto.input_size() == 2 || layerParams.has("shape"));

            if (node_proto.input_size() == 2) {
                Mat blob = getBlob(node_proto, constBlobs, 1);
                CV_Assert(blob.type() == CV_32SC1);

                if (layer_id.find(node_proto.input(0)) == layer_id.end()) {
                    Mat input = getBlob(node_proto, constBlobs, 0);
                    Mat out = input.reshape(0, static_cast<std::vector<int> >(blob));
                    constBlobs.insert(std::make_pair(layerParams.name, out));
                    continue;
                }
                layerParams.set("dim", DictValue::arrayInt<int*>(
                            blob.ptr<int>(), blob.total() ));
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
                    constBlobs.insert(std::make_pair(layerParams.name, out));
                    continue;
                }
                replaceLayerParam(layerParams, "shape", "dim");
            }
        }
        else if (layer_type == "Pad")
        {
            layerParams.type = "Padding";
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

            constBlobs.insert(std::make_pair(layerParams.name, shapeMat));
            continue;
        }
        else if (layer_type == "Gather")
        {
            CV_Assert(node_proto.input_size() == 2);
            CV_Assert(layerParams.has("axis"));
            Mat input = getBlob(node_proto, constBlobs, 0);
            Mat indexMat = getBlob(node_proto, constBlobs, 1);
            CV_Assert_N(indexMat.type() == CV_32S, indexMat.total() == 1);
            int index = indexMat.at<int>(0);
            int axis = layerParams.get<int>("axis");

            std::vector<cv::Range> ranges(input.dims, Range::all());
            ranges[axis] = Range(index, index + 1);

            Mat out = input(ranges);
            constBlobs.insert(std::make_pair(layerParams.name, out));
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
                Ptr<Layer> concat = ConcatLayer::create(layerParams);
                runLayer(concat, inputs, concatenated);

                CV_Assert(concatenated.size() == 1);
                constBlobs.insert(std::make_pair(layerParams.name, concatenated[0]));
                continue;
            }
        }
        else if (layer_type == "Upsample")
        {
            layerParams.type = "Resize";
            if (layerParams.has("scales"))
            {
                // Pytorch layer
                DictValue scales = layerParams.get("scales");
                CV_Assert(scales.size() == 4);
                layerParams.set("zoom_factor_y", scales.getIntValue(2));
                layerParams.set("zoom_factor_x", scales.getIntValue(3));
            }
            else
            {
                // Caffe2 layer
                replaceLayerParam(layerParams, "height_scale", "zoom_factor_y");
                replaceLayerParam(layerParams, "width_scale", "zoom_factor_x");
            }
            replaceLayerParam(layerParams, "mode", "interpolation");
        }
        else
        {
            for (int j = 0; j < node_proto.input_size(); j++) {
                if (layer_id.find(node_proto.input(j)) == layer_id.end())
                    layerParams.blobs.push_back(getBlob(node_proto, constBlobs, j));
            }
         }

         int id = dstNet.addLayer(layerParams.name, layerParams.type, layerParams);
         layer_id.insert(std::make_pair(layerParams.name, LayerInfo(id, 0)));


         std::vector<MatShape> layerInpShapes, layerOutShapes, layerInternalShapes;
         for (int j = 0; j < node_proto.input_size(); j++) {
             layerId = layer_id.find(node_proto.input(j));
             if (layerId != layer_id.end()) {
                 dstNet.connect(layerId->second.layerId, layerId->second.outputId, id, j);
                 // Collect input shapes.
                 shapeIt = outShapes.find(node_proto.input(j));
                 CV_Assert(shapeIt != outShapes.end());
                 layerInpShapes.push_back(shapeIt->second);
             }
         }

         // Compute shape of output blob for this layer.
         Ptr<Layer> layer = dstNet.getLayer(id);
         layer->getMemoryShapes(layerInpShapes, 0, layerOutShapes, layerInternalShapes);
         CV_Assert(!layerOutShapes.empty());
         outShapes[layerParams.name] = layerOutShapes[0];
     }
 }

Net readNetFromONNX(const String& onnxFile)
{
    ONNXImporter onnxImporter(onnxFile.c_str());
    Net net;
    onnxImporter.populateNet(net);
    return net;
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

CV__DNN_INLINE_NS_END
}} // namespace

#endif
