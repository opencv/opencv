// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"

#ifdef HAVE_PROTOBUF

#include <iostream>
#include <fstream>
#include <string>
#include <limits>


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
CV__DNN_EXPERIMENTAL_NS_BEGIN


class ONNXImporter
{
    opencv_onnx::ModelProto model_proto;

    std::map<std::string, Mat> getGraphTensors(
                                    const opencv_onnx::GraphProto& graph_proto);
    Mat getBlob(const opencv_onnx::NodeProto& node_proto, const std::map<std::string, Mat>& constBlobs, int index);

    LayerParams getLayerParams(const opencv_onnx::NodeProto& node_proto);

public:

    ONNXImporter(const char *onnxFile)
    {
        std::fstream input(onnxFile, std::ios::in | std::ios::binary);

        if (!model_proto.ParseFromIstream(&input))
            CV_Error(Error::StsUnsupportedFormat, "Failed to parse onnx model");
    }

    void populateNet(Net dstNet);
};


void releaseONNXTensor(opencv_onnx::TensorProto& tensor_proto)
{
    if (!tensor_proto.raw_data().empty()) {
        delete tensor_proto.release_raw_data();
    }
}

Mat getMatFromTensor(const opencv_onnx::TensorProto& tensor_proto)
{
    opencv_onnx::TensorProto_DataType datatype = tensor_proto.data_type();
    CV_Assert(!tensor_proto.raw_data().empty());
    char* val = const_cast<char*>(tensor_proto.raw_data().c_str());

    std::vector<int> sizes;
    for (int i = 0; i < tensor_proto.dims_size(); i++) {
            sizes.push_back(tensor_proto.dims(i));
    }
    Mat blob;
    if (datatype == opencv_onnx::TensorProto_DataType_FLOAT)
        Mat(sizes, CV_32FC1, val).copyTo(blob);

    else if (datatype == opencv_onnx::TensorProto_DataType_INT64)
    {
        blob.create(sizes, CV_32SC1);

        int64_t* src = reinterpret_cast<int64_t*>(val);
        int32_t* dst = reinterpret_cast<int32_t*>(blob.data);

        for (int i = 0; i < blob.total(); i++) {
            if (src[i] < std::numeric_limits<int32_t>::min() || src[i] > std::numeric_limits<int32_t>::max()) {
                CV_Error(Error::StsOutOfRange, "Input is out of OpenCV 32S range");
            }
            dst[i] = saturate_cast<int32_t>(src[i]);
        }
    }
    else
        CV_Error(Error::StsUnsupportedFormat,
        "Unsupported data type: " + opencv_onnx::TensorProto_DataType_Name(datatype));

    return blob;
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
            lp.set("kernel_h",  attribute_proto.ints(0));
            lp.set("kernel_w",  attribute_proto.ints(1));
        }
        else if(attribute_name == "strides")
        {
            CV_Assert(attribute_proto.ints_size() == 2);
            lp.set("stride_h",  attribute_proto.ints(0));
            lp.set("stride_w",  attribute_proto.ints(1));
        }
        else if(attribute_name == "pads")
        {
            CV_Assert(attribute_proto.ints_size() >= 2);
            lp.set("pad_h", attribute_proto.ints(0));
            lp.set("pad_w", attribute_proto.ints(1));
        }
        else if (attribute_proto.has_i())
        {
            lp.set(attribute_name, attribute_proto.i());
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
                (float*)attribute_proto.mutable_floats(), attribute_proto.floats_size()));
        }
        else if (attribute_proto.ints_size() > 0)
        {
                lp.set(attribute_proto.name(), DictValue::arrayInt(
                    (int*)attribute_proto.mutable_ints(), attribute_proto.ints_size()));
        }
        else if (attribute_proto.has_t() || attribute_proto.has_g() ||
                    attribute_proto.strings_size() > 0 ||
                        attribute_proto.tensors_size() > 0 ||
                            attribute_proto.graphs_size() > 0)
        {
                CV_Error(Error::StsNotImplemented, "Unexpected attribute type");
        }
    }
    return lp;
}

Mat ONNXImporter::getBlob(const opencv_onnx::NodeProto& node_proto,
                    const std::map<std::string, Mat>& constBlobs, int index)
{
    std::map<std::string, Mat>::const_iterator constBlob;
    constBlob = constBlobs.find(node_proto.input(index) );
    CV_Assert(constBlob != constBlobs.end());
    return constBlob->second;
}

void ONNXImporter::populateNet(Net dstNet)
{
    CV_Assert(model_proto.has_graph());

    opencv_onnx::GraphProto graph_proto = model_proto.graph();
    std::map<std::string, Mat> constBlobs = getGraphTensors(graph_proto);

    int layersSize = graph_proto.node_size();
    LayerParams layerParams;
    opencv_onnx::NodeProto node_proto;

    for(int i = 0; i < layersSize; i++)
    {
        node_proto = graph_proto.node(i);
        layerParams = getLayerParams(node_proto);
        if (node_proto.has_name() && !node_proto.name().empty())
            layerParams.name = node_proto.name();
        else
            layerParams.name = format("%s_%d", node_proto.op_type().c_str(), i);


         std::string layer_type = node_proto.op_type();
         layerParams.type = layer_type;

         if (layer_type == "MaxPool")
         {
             layerParams.type = "Pooling";
             layerParams.set("pool", "MAX");
         }
         else if (layer_type == "LRN")
         {
             if (layerParams.has("size")) {
                layerParams.set("local_size", layerParams.get<int>("size"));
                layerParams.erase("size");
            }
        }
        else if (layer_type == "Gemm")
        {
            layerParams.type = "InnerProduct";
            for (int i = 1; i < node_proto.input_size(); i++) {
                layerParams.blobs.push_back(getBlob(node_proto, constBlobs, i));
            }
            layerParams.set("num_output", layerParams.blobs[0].size[0]);
            layerParams.set("bias_term", layerParams.blobs.size() == 2);
         }
         else if (layer_type == "Conv")
         {
             layerParams.type = "Convolution";
             for (int i = 1; i < node_proto.input_size(); i++) {
                 layerParams.blobs.push_back(getBlob(node_proto, constBlobs, i));
             }
             layerParams.set("num_output", layerParams.blobs[0].size[0]);
             layerParams.set("bias_term", layerParams.blobs.size() == 2);
         }
         else if (layer_type == "Reshape")
         {
             Mat blob = getBlob(node_proto, constBlobs, 1);
             CV_Assert(blob.type() == CV_32SC1);
             layerParams.set("dim", DictValue::arrayInt<int*>(
                    blob.ptr<int>(), blob.total() ));
        }
        else
        {
            for (int i = 1; i < node_proto.input_size(); i++) {
                layerParams.blobs.push_back( getBlob(node_proto, constBlobs, i) );
            }
         }
         dstNet.addLayerToPrev(layerParams.name, layerParams.type, layerParams);
     }
 }

Net readNetFromONNX(const String &onnxFile)
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

CV__DNN_EXPERIMENTAL_NS_END
}} // namespace

#else
CV_Error(Error::StsNotImplemented, "Work is not supported without protobuf");

#endif
