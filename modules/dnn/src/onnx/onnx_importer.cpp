// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.


#include "../precomp.hpp"
#include "opencv-onnx.pb.h"

#ifdef HAVE_PROTOBUF
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <string>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#endif

namespace cv {
namespace dnn {
CV__DNN_EXPERIMENTAL_NS_BEGIN

#ifdef HAVE_PROTOBUF
using ::google::protobuf::RepeatedField;
using ::google::protobuf::RepeatedPtrField;
using ::google::protobuf::Message;
using ::google::protobuf::Descriptor;
using ::google::protobuf::FieldDescriptor;
using ::google::protobuf::Reflection;


namespace
{
  class ONNXImporter
  {
      onnx::ModelProto model_proto;

      std::map<std::string, Mat> getWeights(
                                        const onnx::GraphProto& graph_proto);
      LayerParams getLayerParams(const onnx::NodeProto& node_proto);

  public:

    ONNXImporter(const char *onnxFile)
    {
        std::fstream input(onnxFile, std::ios::in | std::ios::binary);
        if (!model_proto.ParseFromIstream(&input)) {
            CV_Error(Error::StsUnsupportedFormat, "Failed to parse onnx model");
        }
    }

    void populateNet(Net dstNet);
};
}

std::map<std::string, Mat> ONNXImporter::getWeights(
                                        const onnx::GraphProto& graph_proto)
{
  onnx::TensorProto tensor_proto;
  std::map<std::string, Mat> layers_weights;
  onnx::TensorProto_DataType datatype;
  for (int i = 0; i < graph_proto.initializer_size(); i++) {
    tensor_proto = graph_proto.initializer(i);
    datatype = tensor_proto.data_type();

    if (datatype == onnx::TensorProto_DataType_FLOAT) {
        int* sizes = new int [tensor_proto.dims_size()];
        for (int i = 0; i < tensor_proto.dims_size(); i++) {
            sizes[i] = tensor_proto.dims(i);
        }
        char* val = const_cast<char*>(tensor_proto.raw_data().c_str());
        Mat blob(tensor_proto.dims_size(), sizes, CV_32FC1, val);
        layers_weights.insert(std::pair<std::string, Mat>(
                                        tensor_proto.name(), blob.clone()));
        delete[] sizes;
    }
    else if (datatype == onnx::TensorProto_DataType_INT64) {
        int* sizes = new int [tensor_proto.dims_size()];
        for (int i = 0; i < tensor_proto.dims_size(); i++) {
            sizes[i] = tensor_proto.dims(i);
        }
        Mat blob(tensor_proto.dims_size(), sizes, CV_32SC1);

        char* val = const_cast<char*>(tensor_proto.raw_data().c_str());
        int64_t* src = reinterpret_cast<int64_t*>(val);
        int32_t* dst = reinterpret_cast<int32_t*>(blob.data);

        for (int i = 0; i < blob.total(); i++) {
            dst[i] = static_cast<int32_t>(src[i]);
        }
        layers_weights.insert(std::pair<std::string, Mat>(
                                        tensor_proto.name(), blob.clone()));
        delete[] sizes;
    }
    else {
        std::cout << "datatype = " << datatype << '\n';
        CV_Error(Error::StsUnsupportedFormat, "Failed to get weights");
    }
  }
  return layers_weights;
}

LayerParams ONNXImporter::getLayerParams(const onnx::NodeProto& node_proto)
{
  LayerParams lp;
  for(int i = 0; i < node_proto.attribute_size(); i++) {
    const onnx::AttributeProto& attribute_proto = node_proto.attribute(i);
    std::string attribute_name = attribute_proto.name();
    if (attribute_proto.has_i()) {
      lp.set(attribute_proto.name(), attribute_proto.i());
    } else if (attribute_proto.has_f()) {
      lp.set(attribute_proto.name(), attribute_proto.f());
    } else if (attribute_proto.has_s()) {
      lp.set(attribute_proto.name(), attribute_proto.s());
    }
    else if (attribute_proto.has_t()) {
      std::cout << "I have tensor with sizes "
                                    << attribute_proto.t().dims_size()<< '\n';
     // lp.set(attribute_proto.name(), attribute_proto.t());
    } else if (attribute_proto.has_g()) {
       std::cout << "I have graph" << '\n';
      //lp.set(attribute_proto.name(), attribute_proto.g());
    }
    for (int i = 0; i < attribute_proto.floats_size(); i++) {
      lp.set(attribute_proto.name(), attribute_proto.floats(i));
    }
    for (int i = 0; i < attribute_proto.ints_size(); i++) {
      if(attribute_name == "kernel_shape") {
        lp.set("kernel_h",  attribute_proto.ints(0));
        lp.set("kernel_w",  attribute_proto.ints(1));
      } else if(attribute_name == "strides") {
        lp.set("stride_h",  attribute_proto.ints(0));
        lp.set("stride_w",  attribute_proto.ints(1));
      } else if(attribute_name == "pads") {
        lp.set("pad_h",  attribute_proto.ints(0));
        lp.set("pad_w",  attribute_proto.ints(1));
      } else
      lp.set(attribute_proto.name(), attribute_proto.ints(i));
    }
    for (int i = 0; i < attribute_proto.strings_size(); i++) {
      lp.set(attribute_proto.name(), attribute_proto.strings(i));
    }
    for (int i = 0; i < attribute_proto.tensors_size(); i++) {
      std::cout << "I have " << attribute_proto.tensors_size()
                                                        << " tensors" << '\n';
      std::cout << "Tensor with sizes "
                            << attribute_proto.tensors(i).dims_size()<< '\n';
     // lp.set(attribute_proto.name(), attribute_proto.tensors(i));
    }
    for (int i = 0; i < attribute_proto.graphs_size(); i++) {
        std::cout << "I have " << attribute_proto.graphs_size()
                                                        << " graphs" << '\n';
     // lp.set(attribute_proto.name(), attribute_proto.graphs(i));
    }
  }
  std::cout << lp << '\n';
  return lp;
}

void ONNXImporter::populateNet(Net dstNet)
{
    CV_Assert(model_proto.has_graph());

    onnx::GraphProto graph_proto = model_proto.graph();
    std::map<std::string, Mat> weights = getWeights(graph_proto);
    std::map<std::string, Mat>::iterator weight;

    int layersSize = graph_proto.node_size();
    LayerParams layerParams;
    onnx::NodeProto node_proto;

    for(int i = 0; i < layersSize; i++) {
        node_proto = graph_proto.node(i);
        layerParams = getLayerParams(node_proto);
        if (node_proto.has_name() && !node_proto.name().empty()) {
            layerParams.name = node_proto.name();
        } else {
            std::ostringstream s;
            s << node_proto.op_type() << "_" << i;
            layerParams.name = s.str();
         }
         std::string layer_type = node_proto.op_type();

         if (layer_type == "MaxPool") {
            layerParams.type = "Pooling";
            layerParams.set("pool", "MAX");
        } else if (layer_type == "LRN") {
            layerParams.type = "LRN";
            if (layerParams.has("size")) {
                layerParams.replace("size", "local_size");
            //    layerParams.set("local_size", layerParams.get<int>("size"));
            }
        } else if (layer_type == "Gemm") {
            layerParams.type = "InnerProduct";
            for (int i = 1; i < node_proto.input_size(); i++) {
                weight = weights.find(node_proto.input(i));
                CV_Assert(weight != weights.end());
                layerParams.blobs.push_back(weight->second);
            }
            layerParams.set("num_output", layerParams.blobs[0].size[0]);
            int numLayerWeights = node_proto.input_size() - 1;
            if (numLayerWeights == 2) {
                layerParams.set("bias_term", true);
            } else {
                layerParams.set("bias_term", false);
            }
         } else if (layer_type == "Conv") {
            layerParams.type = "Convolution";
            for (int i = 1; i < node_proto.input_size(); i++) {
                weight = weights.find(node_proto.input(i));
                CV_Assert(weight != weights.end());
                layerParams.blobs.push_back(weight->second);
            }
            layerParams.set("num_output", layerParams.blobs[0].size[0]);
            int numLayerWeights = node_proto.input_size() - 1;
            if (numLayerWeights == 2) {
                layerParams.set("bias_term", true);
            } else {
                layerParams.set("bias_term", false);
            }
        } else if (layer_type == "Reshape") {
            layerParams.type = "Reshape";
            for (int i = 1; i < node_proto.input_size(); i++) {
                weight = weights.find(node_proto.input(i));
                CV_Assert(weight != weights.end());
                layerParams.set("dim", DictValue::arrayInt<int*>(
                       weight->second.ptr<int>(),  weight->second.total() ));
            }
        } else {
            layerParams.type = node_proto.op_type();
            for (int i = 1; i < node_proto.input_size(); i++) {
                weight = weights.find(node_proto.input(i));
                CV_Assert(weight != weights.end());
                layerParams.blobs.push_back(weight->second);
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

Mat readTensorFromONNX(const String& path) {
    onnx::TensorProto tensor_proto = onnx::TensorProto();
    std::fstream input(path.c_str(), std::ios::in | std::ios::binary);
    if (!tensor_proto.ParseFromIstream(&input)) {
        CV_Error(Error::StsUnsupportedFormat, "Failed to parse data");
    }
    onnx::TensorProto_DataType datatype = tensor_proto.data_type();

    if (datatype == onnx::TensorProto_DataType_FLOAT) {
        std::vector<int> sizes;
        for (int i = 0; i < tensor_proto.dims_size(); i++) {
           sizes.push_back(tensor_proto.dims(i));
        }
        char* val = const_cast<char*>(tensor_proto.raw_data().c_str());
        Mat mat(tensor_proto.dims_size(), sizes.data(), CV_32FC1, val);
        return mat.clone();
    }
    std::cout << "datatype = " <<  datatype << '\n';
    CV_Error(Error::StsUnsupportedFormat, "Failed to parse data");
}

#endif //HAVE_PROTOBUF

CV__DNN_EXPERIMENTAL_NS_END
}} // namespace
