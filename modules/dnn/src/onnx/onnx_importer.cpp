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
                if (inpShape.size() != 4 && inpShape.size() != 5)
                    CV_Error(Error::StsNotImplemented, "Unsupported input shape of reduce_mean operation.");

                DictValue axes = layerParams.get("axes");
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
        else if (layer_type == "Slice")
        {
            if (layerParams.has("steps")) {
                DictValue steps = layerParams.get("steps");
                for (int i = 0; i < steps.size(); ++i) {
                    if (steps.get<int>(i) != 1)
                        CV_Error(Error::StsNotImplemented,
                                 "Slice layer only supports steps = 1");
                }
            }

            int axis = 0;
            if (layerParams.has("axes")) {
                DictValue axes = layerParams.get("axes");
                for (int i = 1; i < axes.size(); ++i) {
                    CV_Assert(axes.get<int>(i - 1) == axes.get<int>(i) - 1);
                }
                axis = axes.get<int>(0);
            }
            layerParams.set("axis", axis);

            DictValue starts = layerParams.get("starts");
            DictValue ends = layerParams.get("ends");
            CV_Assert(starts.size() == ends.size());

            std::vector<int> begin;
            std::vector<int> end;
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
            layerParams.set("begin", DictValue::arrayInt(&begin[0], begin.size()));
            layerParams.set("end", DictValue::arrayInt(&end[0], end.size()));
         }
        else if (layer_type == "Split")
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
            layerParams.type = "Slice";
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
        else if (layer_type == "Max")
        {
            layerParams.type = "Eltwise";
            layerParams.set("operation", "max");
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
            if (constBlobs.find(node_proto.input(1)) == constBlobs.end())
            {
                layerParams.type = "Eltwise";
                layerParams.set("operation", "div");
            }
            else
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
                constBlobs.insert(std::make_pair(layerParams.name, transposed[0]));
                continue;
            }
        }
        else if (layer_type == "ReduceL2")
        {
            CV_Assert_N(node_proto.input_size() == 1, layerParams.has("axes"));
            CV_Assert(graph_proto.node_size() > li + 1 && graph_proto.node(li + 1).op_type() == "Div");
            ++li;
            node_proto = graph_proto.node(li);
            layerParams.name = node_proto.output(0);
            layerParams.type = "Normalize";

            DictValue axes_dict = layerParams.get("axes");
            if (axes_dict.size() != 1)
                CV_Error(Error::StsNotImplemented, "Multidimensional reduceL2");
            int axis = axes_dict.getIntValue(0);
            layerParams.set("axis",axis);
            layerParams.set("end_axis", axis);
        }
        else if (layer_type == "Squeeze")
        {
            CV_Assert_N(node_proto.input_size() == 1, layerParams.has("axes"));
            DictValue axes_dict = layerParams.get("axes");
            if (axes_dict.size() != 1)
                CV_Error(Error::StsNotImplemented, "Multidimensional squeeze");

            int axis = axes_dict.getIntValue(0);
            layerParams.set("axis", axis - 1);
            layerParams.set("end_axis", axis);
            layerParams.type = "Flatten";
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

            MatShape inpShape = outShapes[node_proto.input(0)];
            int axis = axes.getIntValue(0);
            CV_Assert(0 <= axis && axis <= inpShape.size());
            std::vector<int> outShape = inpShape;
            outShape.insert(outShape.begin() + axis, 1);
            layerParams.type = "Reshape";
            layerParams.set("dim", DictValue::arrayInt(&outShape[0], outShape.size()));
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
                    constBlobs.insert(std::make_pair(layerParams.name, outputs[0]));
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
                runLayer(layerParams, inputs, concatenated);

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
        else if (layer_type == "LogSoftmax")
        {
            layerParams.type = "Softmax";
            layerParams.set("log_softmax", true);
        }
        else
        {
            for (int j = 0; j < node_proto.input_size(); j++) {
                if (layer_id.find(node_proto.input(j)) == layer_id.end())
                    layerParams.blobs.push_back(getBlob(node_proto, constBlobs, j));
            }
        }

        int id = dstNet.addLayer(layerParams.name, layerParams.type, layerParams);
        for (int i = 0; i < node_proto.output_size(); ++i)
        {
            layer_id.insert(std::make_pair(node_proto.output(i), LayerInfo(id, i)));
        }

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
        for (int i = 0; i < node_proto.output_size() && i < (int)layerOutShapes.size(); ++i)
        {
            outShapes[node_proto.output(i)] = layerOutShapes[i];
        }
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

CV__DNN_INLINE_NS_END
}} // namespace

#endif
