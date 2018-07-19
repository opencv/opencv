/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "../precomp.hpp"
#include "../../misc/onnx/opencv-onnx.pb.h"

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

      std::map<std::string, cv::Mat> getWeights(const onnx::GraphProto& graph_proto);
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

std::map<std::string, cv::Mat> ONNXImporter::getWeights(const onnx::GraphProto& graph_proto)
{
  onnx::TensorProto tensor_proto;
  std::map<std::string, cv::Mat> layers_weights;
  onnx::TensorProto_DataType datatype;
  for (int i = 0; i < graph_proto.initializer_size(); i++) {
    tensor_proto = graph_proto.initializer(i);
    datatype = tensor_proto.data_type();

    if (datatype == onnx::TensorProto_DataType_FLOAT) {
       std::vector<int> sizes;
       for (int i = 0; i < tensor_proto.dims_size(); i++) {
          sizes.push_back(tensor_proto.dims(i));
       }
       char* val = const_cast<char*>(tensor_proto.raw_data().c_str());
       cv::Mat blob(tensor_proto.dims_size(), sizes.data(), CV_32FC1, val);
      // std::cout << "blob" << blob << '\n';
       layers_weights.insert(std::pair<std::string, cv::Mat>(tensor_proto.name(), blob.clone()));
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
      // if(attribute_name == "ratio") lp.set("dropout_ratio", attribute_proto.f());
      lp.set(attribute_proto.name(), attribute_proto.f());
    } else if (attribute_proto.has_s()) {
      lp.set(attribute_proto.name(), attribute_proto.s());
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
  }
  std::string layer_type = node_proto.op_type();
  if (layer_type == "MaxPool") {
    lp.type = "Pooling";
    lp.set("pool", "MAX");
  } else if (layer_type == "Gemm") {
    lp.type = "InnerProduct";
  } else if (layer_type == "Conv") {
    lp.type = "Convolution";
  } else {
    lp.type = node_proto.op_type();
  }
  return lp;
}

void ONNXImporter::populateNet(Net dstNet)
{
    if(model_proto.has_graph()) {
      onnx::GraphProto graph_proto = model_proto.graph();
      std::map<std::string, cv::Mat> weights = getWeights(graph_proto);
      std::map<std::string, cv::Mat>::iterator weight;

      int layersSize = graph_proto.node_size();
      LayerParams layerParams;
      onnx::NodeProto node_proto;

      for(int i = 0; i < layersSize; i++) {
        node_proto = graph_proto.node(i);
        layerParams = getLayerParams(node_proto);
        // if (node_proto.has_name()) {
        //     layerParams.name = node_proto.name();
        // } else {
            std::ostringstream s;
            s << i;
            std::string id(s.str());
            layerParams.name = node_proto.op_type() + "_" + id;
    //    }
        // std::cout << "__layer params__" << '\n' << layerParams << '\n';
        std::cout << layerParams.name << '\n';
        int numLayerWeights = 0;
        for (int i = 0; i < node_proto.input_size(); i++) {
            weight = weights.find(node_proto.input(i));
            if (weight != weights.end()) {
              layerParams.blobs.push_back(weight->second);
              numLayerWeights++;
            }
        }
        if (numLayerWeights == 2) {
            layerParams.set("bias_term", true);
            int num_output = static_cast<int>(layerParams.blobs[1].total());  // add num output if bias = false
            layerParams.set("num_output", num_output);
        } else {
            layerParams.set("bias_term", false);
        }
        dstNet.addLayerToPrev(layerParams.name, layerParams.type, layerParams);
    }
  }
}

Net readNetFromONNX(const String &onnxFile)
{
    ONNXImporter onnxImporter(onnxFile.c_str());
    Net net;
    onnxImporter.populateNet(net);
    return net;
}

#endif //HAVE_PROTOBUF

CV__DNN_EXPERIMENTAL_NS_END
}} // namespace
