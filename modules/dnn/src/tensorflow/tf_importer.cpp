// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Implementation of Tensorflow models parser
*/

#include "../precomp.hpp"

#include <opencv2/core/utils/logger.defines.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#undef CV_LOG_STRIP_LEVEL
#define CV_LOG_STRIP_LEVEL CV_LOG_LEVEL_DEBUG + 1
#include <opencv2/core/utils/logger.hpp>

#ifdef HAVE_PROTOBUF
#include "tf_io.hpp"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <queue>
#include "tf_graph_simplifier.hpp"
#endif

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

#if HAVE_PROTOBUF

using ::google::protobuf::RepeatedField;
using ::google::protobuf::RepeatedPtrField;
using ::google::protobuf::Message;
using ::google::protobuf::Descriptor;
using ::google::protobuf::FieldDescriptor;
using ::google::protobuf::Reflection;

namespace
{

static int toNCHW(int idx)
{
    CV_Assert(-4 <= idx && idx < 4);
    if (idx == 0) return 0;
    else if (idx > 0) return idx % 3 + 1;
    else return (4 + idx) % 3 + 1;
}

static int toNCDHW(int idx)
{
    CV_Assert(-5 <= idx && idx < 5);
    if (idx == 0) return 0;
    else if (idx > 0) return idx % 4 + 1;
    else return (5 + idx) % 4 + 1;
}

// This values are used to indicate layer output's data layout where it's possible.
enum DataLayout
{
    DATA_LAYOUT_NHWC,
    DATA_LAYOUT_NCHW,
    DATA_LAYOUT_NDHWC,
    DATA_LAYOUT_UNKNOWN,
    DATA_LAYOUT_PLANAR  // 2-dimensional outputs (matmul, flatten, reshape to 2d)
};

typedef std::vector<std::pair<String, int> > StrIntVector;

struct Pin
{
    Pin(const std::string &_name, int _blobIndex = 0) :
        name(_name), blobIndex(_blobIndex) {}

    Pin() :
        name(""), blobIndex(-1) {}

    std::string name;
    int blobIndex;
};

void blobShapeFromTensor(const tensorflow::TensorProto &tensor, MatShape& shape)
{
    shape.clear();
    if (tensor.has_tensor_shape())
    {
        const tensorflow::TensorShapeProto &_shape = tensor.tensor_shape();
        int i, n = _shape.dim_size();
        if (n)
        {
            shape.resize(n);

            for (i = 0; i < n; i++)
                shape[i] = (int)_shape.dim(i).size();
        }
        else
            shape.resize(1, 1);  // Scalar. // FIXIT: should be empty
    }
    else
    {
        CV_Error(Error::StsError, "Unknown shape of input tensor");
    }
}

template <typename T>
void parseTensor(const tensorflow::TensorProto &tensor, Mat &dstBlob)
{
    MatShape shape;
    blobShapeFromTensor(tensor, shape);
    int dims = (int)shape.size();

    if (dims == 4)
    {
        // REORDER blob NHWC to NCHW
        swap(shape[2], shape[3]); // NHCW
        swap(shape[1], shape[2]); // NCHW
    }

    dstBlob.create(shape, CV_32F);

    Mat tensorContent = getTensorContent(tensor, /*no copy*/false);
    int size = tensorContent.total();
    CV_Assert(size == (int)dstBlob.total());

    float *dstData = dstBlob.ptr<float>();
    const T *data = reinterpret_cast<const T*>(tensorContent.data);

    if (dims == 4)
    {
        int num = shape[0], channels = shape[1], height = shape[2], width = shape[3];
        int total = num*channels*height*width;
        for(int i_n = 0; i_n < shape[0]; i_n++) {
            for(int i_c = 0; i_c < shape[1]; i_c++) {
                for(int i_h = 0; i_h < shape[2]; i_h++) {
                    for(int i_w = 0; i_w < shape[3]; i_w++) {
                       int dst_i = channels*height*width*i_n + height*width*i_c + width*i_h + i_w;
                       int src_i = channels*height*width*i_n + i_c + channels*width*i_h + channels*i_w;

                       CV_Assert(dst_i < total);
                       CV_Assert(src_i < total);

                       dstData[dst_i] = data[src_i];
                    }
                }
            }
        }
    } else {
        for (int i = 0; i < size; i++)
            dstData[i] = data[i];
    }
}

void blobFromTensor(const tensorflow::TensorProto &tensor, Mat &dstBlob)
{
    switch (tensor.dtype()) {
        case tensorflow::DT_FLOAT:
        case tensorflow::DT_HALF:
            parseTensor<float>(tensor, dstBlob);
            break;
        case tensorflow::DT_DOUBLE:
            parseTensor<double>(tensor, dstBlob);
            break;
        default:
            CV_Error(Error::StsError, "Tensor's data type is not supported");
            break;
    }
}

#if 0
void printList(const tensorflow::AttrValue::ListValue &val)
{
    std::cout << "(";
    for (int i = 0; i < val.i_size(); i++)
        std::cout << " " << val.i(i);
    std::cout << " )";
}

void printTensorShape(const tensorflow::TensorShapeProto &shape)
{
    std::cout << "[ ";
    for (int d = 0; d < shape.dim_size(); d++)
        std::cout << shape.dim(d).name() <<
                     ":" << shape.dim(d).size() << " ";
    std::cout << "]";
}

void printTensor(const tensorflow::TensorProto &tensor)
{
    printTensorShape(tensor.tensor_shape());

    if (tensor.tensor_content().empty())
        return;

    switch (tensor.dtype())
    {
    case tensorflow::DT_FLOAT:
        {
            const float *data = reinterpret_cast<const float*>(tensor.tensor_content().c_str());
            int size = tensor.tensor_content().size() / sizeof(float);
            for (int i = 0; i < std::min(10, size); i++)
                std::cout << " " << data[i];
            if (size > 10)
                std::cout << " ... " << size - 10 << " more";
            break;
        }
    case tensorflow::DT_INT32:
        {
            const int *data = reinterpret_cast<const int*>(tensor.tensor_content().c_str());
            int size = tensor.tensor_content().size() / sizeof(int);
            for (int i = 0; i < std::min(10, size); i++)
                std::cout << " " << data[i];
            if (size > 10)
                std::cout << " ... " << size - 10 << " more";
            break;
        }
    default:
        CV_Error(Error::StsError, "Tensor type is not supported");
        break;
    }
}

void printLayerAttr(const tensorflow::NodeDef &layer)
{
    std::cout << std::endl << layer.name() << ":" << layer.op();
    for (int ii = 0; ii < layer.input_size(); ii++)
        std::cout << "(" << layer.input(ii) << ")";
    std::cout << std::endl;
    google::protobuf::Map<std::string, tensorflow::AttrValue> attr
            = layer.attr();
    for (google::protobuf::Map<std::string, tensorflow::AttrValue>::const_iterator ai = attr.begin();
         ai != attr.end(); ++ai)
    {
        std::cout << ai->first << ":";
        if (ai->first == "dtype" || ai->first == "T")
            std::cout << ai->second.i();
        else if (ai->first == "padding")
            std::cout << ai->second.s();
        else if (ai->first == "transpose_a" || ai->first == "transpose_b")
            std::cout << ai->second.b();
        //            else if (ai->first == "shape")
        //              printTensorShape(ai->second.shape());
        else if (ai->first == "strides" || ai->first == "ksize")
            printList(ai->second.list());
        else
            printTensor(ai->second.tensor());
        std::cout << std::endl;
    }
}
#endif

bool hasLayerAttr(const tensorflow::NodeDef &layer, const std::string &name)
{
    google::protobuf::Map<std::string, tensorflow::AttrValue> attr = layer.attr();
    return attr.find(name) != attr.end();
}

const tensorflow::AttrValue& getLayerAttr(const tensorflow::NodeDef &layer, const std::string &name)
{
    return layer.attr().at(name);
}

static DataLayout getDataLayout(const tensorflow::NodeDef& layer)
{
    if (hasLayerAttr(layer, "data_format"))
    {
        std::string format = getLayerAttr(layer, "data_format").s();
        if (format == "NHWC" || format == "channels_last")
            return DATA_LAYOUT_NHWC;
        else if (format == "NCHW" || format == "channels_first")
            return DATA_LAYOUT_NCHW;
        else if (format == "NDHWC")
            return DATA_LAYOUT_NDHWC;
        else
            CV_Error(Error::StsParseError, "Unknown data_format value: " + format);
    }
    return DATA_LAYOUT_UNKNOWN;
}

static inline std::string getNodeName(const std::string& tensorName)
{
    return tensorName.substr(0, tensorName.rfind(':'));
}

static inline
DataLayout getDataLayout(
        const std::string& layerName,
        const std::map<String, DataLayout>& data_layouts
)
{
    std::map<String, DataLayout>::const_iterator it = data_layouts.find(getNodeName(layerName));
    return it != data_layouts.end() ? it->second : DATA_LAYOUT_UNKNOWN;
}

static
bool hasAllOnes(const Mat &inputs, int startPos, int endPos)
{
    CV_CheckLE(inputs.dims, 2, "");
    CV_CheckGE(startPos, 0, "");
    CV_CheckLE(startPos, endPos, "");
    CV_CheckLT((size_t)endPos, inputs.total(), "");

    for (int i = startPos; i < endPos; i++)
    {
        if (inputs.at<int>(i) != 1 && inputs.at<int>(i) != -1)
            return false;
    }
    return true;
}

void setStrides(LayerParams &layerParams, const tensorflow::NodeDef &layer)
{
    if (hasLayerAttr(layer, "strides"))
    {
        const tensorflow::AttrValue& val = getLayerAttr(layer, "strides");
        int dimX, dimY, dimC, dimD;
        int layout = getDataLayout(layer);
        if (layout == DATA_LAYOUT_NCHW)
        {
            dimC = 1; dimY = 2; dimX = 3;
        }
        else if (layout == DATA_LAYOUT_NDHWC)
        {
            dimD = 1; dimY = 2; dimX = 3; dimC = 4;
        }
        else
        {
            dimY = 1; dimX = 2; dimC = 3;
        }
        if (!(val.list().i_size() == 4 || val.list().i_size() == 5) ||
            val.list().i(0) != 1 || val.list().i(dimC) != 1)
            CV_Error(Error::StsError, "Unsupported strides");
        if (layout == DATA_LAYOUT_NDHWC) {
            int strides[] = {static_cast<int>(val.list().i(dimD)),
                             static_cast<int>(val.list().i(dimY)),
                             static_cast<int>(val.list().i(dimX))};
            layerParams.set("stride",  DictValue::arrayInt(strides, 3));
        }
        else
        {
            layerParams.set("stride_h", static_cast<int>(val.list().i(dimY)));
            layerParams.set("stride_w", static_cast<int>(val.list().i(dimX)));
        }
    }
}

DictValue parseDims(const tensorflow::TensorProto &tensor) {
    MatShape shape;
    blobShapeFromTensor(tensor, shape);
    int dims = (int)shape.size();

    CV_Assert(tensor.dtype() == tensorflow::DT_INT32);
    CV_Assert(dims == 1);

    Mat values = getTensorContent(tensor);
    CV_Assert(values.type() == CV_32SC1);
    // TODO: add reordering shape if dims == 4
    return DictValue::arrayInt((int*)values.data, values.total());
}

void setKSize(LayerParams &layerParams, const tensorflow::NodeDef &layer)
{
    if (hasLayerAttr(layer, "ksize"))
    {
        const tensorflow::AttrValue& val = getLayerAttr(layer, "ksize");
        int dimX, dimY, dimC, dimD;
        int layout = getDataLayout(layer);
        if (layout == DATA_LAYOUT_NCHW)
        {
            dimC = 1; dimY = 2; dimX = 3;
        }
        else if (layout == DATA_LAYOUT_NDHWC)
        {
            dimD = 1; dimY = 2; dimX = 3; dimC = 4;
        }
        else
        {
            dimY = 1; dimX = 2; dimC = 3;
        }
        if (!(val.list().i_size() == 4 || val.list().i_size() == 5) ||
            val.list().i(0) != 1 || val.list().i(dimC) != 1)
            CV_Error(Error::StsError, "Unsupported ksize");

        if (layout == DATA_LAYOUT_NDHWC) {
            int kernel[] = {static_cast<int>(val.list().i(dimD)),
                            static_cast<int>(val.list().i(dimY)),
                            static_cast<int>(val.list().i(dimX))};
            layerParams.set("kernel_size",  DictValue::arrayInt(kernel, 3));
        }
        else
        {
            layerParams.set("kernel_h", static_cast<int>(val.list().i(dimY)));
            layerParams.set("kernel_w", static_cast<int>(val.list().i(dimX)));
        }
    }
    else
    {
        layerParams.set("kernel_h", 1);
        layerParams.set("kernel_w", 1);
    }
}

void setPadding(LayerParams &layerParams, const tensorflow::NodeDef &layer)
{
    if (hasLayerAttr(layer, "padding"))
        layerParams.set("pad_mode", getLayerAttr(layer, "padding").s());
}

Pin parsePin(const std::string &name)
{
    Pin pin(name);

    size_t delimiter_pos = name.find_first_of(':');
    if (delimiter_pos != std::string::npos)
    {
        pin.name = name.substr(0, delimiter_pos);
        std::istringstream(name.substr(delimiter_pos + 1)) >> pin.blobIndex;
    }

    return pin;
}

StrIntVector getNextLayers(const tensorflow::GraphDef& net, const String& layer_name, const String& type = "")
{
   StrIntVector layers;

   for (int li = 0; li < net.node_size(); li++)
   {
       const tensorflow::NodeDef& layer = net.node(li);
       for (int input_id = 0; input_id < layer.input_size(); input_id++) {
           String input_op_name = parsePin(layer.input(input_id)).name;
           bool type_ok = type.empty() ? true : type == layer.op();
           if (input_op_name == layer_name && type_ok)
               layers.push_back(std::make_pair(layer.name(), li));
       }
   }

   return layers;
}

void ExcludeLayer(tensorflow::GraphDef& net, const int layer_index, const int input_blob_index, bool remove_from_net = true) {
    String layer_name = net.node(layer_index).name();
    StrIntVector layers = getNextLayers(net, layer_name);

    String removed_layer_input = net.node(layer_index).input(input_blob_index);

    for (size_t i = 0; i < layers.size(); i++)
    {
        tensorflow::NodeDef* layer = net.mutable_node(layers[i].second);
        for (int input_id = 0; input_id < layer->input_size(); input_id++) {
                String input_op_name = layer->input(input_id);

                if (input_op_name == layer_name) {
                    layer->set_input(input_id, removed_layer_input);
                }
        }
    }

    if (remove_from_net)
        net.mutable_node()->DeleteSubrange(layer_index, 1);
}

class TFImporter
{
public:
    TFImporter(Net& net, const char *model, const char *config = NULL);
    TFImporter(Net& net, const char *dataModel, size_t lenModel,
               const char *dataConfig = NULL, size_t lenConfig = 0);
protected:
    Net& dstNet;
    void populateNet();

    void parseNode(const tensorflow::NodeDef& layer);

    DataLayout predictOutputDataLayout(const tensorflow::NodeDef& layer);

    void kernelFromTensor(const tensorflow::TensorProto &tensor, Mat &dstBlob);

    void connect(const std::map<String, int>& layers_name_id_map, Net& network, const Pin& outPin,
                 const int input_layer_id, const int input_blob_id);
    void connectToAllBlobs(const std::map<String, int>& layer_id, Net& network, const Pin& outPin,
                           const int input_layer_id, const int input_blobs_count);
    const tensorflow::TensorProto& getConstBlob(const tensorflow::NodeDef &layer, std::map<String, int> const_layers,
                                                int input_blob_index = -1, int* actual_inp_blob_idx = 0);


    // Binary serialized TensorFlow graph includes weights.
    tensorflow::GraphDef netBin;
    // Optional text definition of TensorFlow graph. More flexible than binary format
    // and may be used to build the network using binary format only as a weights storage.
    // This approach is similar to Caffe's `.prorotxt` and `.caffemodel`.
    tensorflow::GraphDef netTxt;

    std::vector<String> netInputsNames;
    std::vector<MatShape> netInputShapes;

    std::set<String> layers_to_ignore;
    std::map<String, DataLayout> data_layouts;

    // find all Const layers for params
    std::map<String, int> value_id;
    // A map with constant blobs which are shared between multiple layers.
    std::map<String, Mat> sharedWeights;

    std::map<String, int> layer_id;

private:
    void addPermuteLayer(const int* order, const std::string& permName, Pin& inpId);
};

TFImporter::TFImporter(Net& net, const char *model, const char *config)
    : dstNet(net)
{
    if (model && model[0])
    {
        CV_LOG_DEBUG(NULL, "DNN/TF: processing TensorFlow model from file: " << model);
        ReadTFNetParamsFromBinaryFileOrDie(model, &netBin);
    }
    if (config && config[0])
    {
        CV_LOG_DEBUG(NULL, "DNN/TF: processing TensorFlow config from file: " << config);
        ReadTFNetParamsFromTextFileOrDie(config, &netTxt);
    }

    populateNet();
}

TFImporter::TFImporter(
        Net& net,
        const char *dataModel, size_t lenModel,
        const char *dataConfig, size_t lenConfig
)
    : dstNet(net)
{
    if (dataModel != NULL && lenModel > 0)
    {
        CV_LOG_DEBUG(NULL, "DNN/TF: processing TensorFlow model from memory (" << lenModel << " bytes)");
        ReadTFNetParamsFromBinaryBufferOrDie(dataModel, lenModel, &netBin);
    }
    if (dataConfig != NULL && lenConfig > 0)
    {
        CV_LOG_DEBUG(NULL, "DNN/TF: processing TensorFlow config from memory (" << lenConfig << " bytes)");
        ReadTFNetParamsFromTextBufferOrDie(dataConfig, lenConfig, &netTxt);
    }
    populateNet();
}

void TFImporter::kernelFromTensor(const tensorflow::TensorProto &tensor, Mat &dstBlob)
{
    MatShape shape;
    blobShapeFromTensor(tensor, shape);
    int dims = (int)shape.size();

    // TODO: other blob types
    CV_Assert(tensor.dtype() == tensorflow::DT_FLOAT ||
              tensor.dtype() == tensorflow::DT_HALF);
    CV_Assert(dims == 4 || dims == 5);

    int out_c, input_c, depth, height, width;
    if (dims == 4)
    {
        // REORDER kernel HWIO to OIHW
        swap(shape[0], shape[2]); // IWHO
        swap(shape[1], shape[3]); // IOHW
        swap(shape[0], shape[1]); // OIHW
        depth = 1; height = shape[2]; width = shape[3];
    }
    else
    {
        // REORDER kernel DHWIO to OIDHW
        swap(shape[0], shape[4]); // OHWID
        swap(shape[1], shape[3]); // OIWHD
        swap(shape[2], shape[4]); // OIDHW
        depth = shape[2]; height = shape[3]; width = shape[4];
    }
    out_c = shape[0]; input_c = shape[1];

    dstBlob.create(shape, CV_32F);

    Mat tensorContent = getTensorContent(tensor, /*no copy*/false);
    int size = tensorContent.total();
    CV_Assert(size == (int)dstBlob.total());

    float *dstData = dstBlob.ptr<float>();
    const float *data = reinterpret_cast<const float*>(tensorContent.data);

    int total = out_c * input_c * depth * height * width;
    for (int i_oc = 0; i_oc < out_c; i_oc++) {
        for (int i_ic = 0; i_ic < input_c; i_ic++) {
            for (int i_d = 0; i_d < depth; i_d++) {
                for (int i_h = 0; i_h < height; i_h++) {
                    for (int i_w = 0; i_w < width; i_w++) {
                        int dst_i = input_c * depth * height * width * i_oc +
                                    depth * height * width * i_ic + height * width * i_d + width * i_h + i_w;
                        int src_i = out_c * input_c * width * height * i_d +
                                    out_c * input_c * width * i_h + out_c * input_c * i_w + out_c * i_ic + i_oc;
                        CV_Assert(dst_i < total);
                        CV_Assert(src_i < total);
                       dstData[dst_i] = data[src_i];
                   }
                }
            }
        }
    }
}

void TFImporter::connect(const std::map<String, int>& layers_name_id_map, Net& network, const Pin& outPin,
             const int input_layer_id, const int input_blob_id)
{
    std::map<String, int>::const_iterator it = layers_name_id_map.find(outPin.name);
    if (it == layers_name_id_map.end())
        CV_Error(Error::StsError, "Input layer not found: " + outPin.name);

    std::vector<String>::iterator inpNameIt = std::find(netInputsNames.begin(), netInputsNames.end(), outPin.name);
    int blobIndex;
    if (inpNameIt == netInputsNames.end())
        blobIndex = outPin.blobIndex;
    else
        blobIndex = inpNameIt - netInputsNames.begin();
    network.connect(it->second, blobIndex, input_layer_id, input_blob_id);
}

void TFImporter::connectToAllBlobs(const std::map<String, int>& layer_id, Net& network, const Pin& outPin,
                     const int input_layer_id, const int input_blobs_count)
{
    for (int input_blob_id = 0; input_blob_id < input_blobs_count; input_blob_id++)
        connect(layer_id, network, outPin, input_layer_id, input_blob_id);
}

const tensorflow::TensorProto& TFImporter::getConstBlob(const tensorflow::NodeDef &layer, std::map<String, int> const_layers,
                                              int input_blob_index, int* actual_inp_blob_idx) {
    if (input_blob_index == -1) {
        for(int i = 0; i < layer.input_size(); i++) {
            Pin input = parsePin(layer.input(i));
            if (const_layers.find(input.name) != const_layers.end()) {
                if (input_blob_index != -1)
                    CV_Error(Error::StsError, "More than one input is Const op");

                input_blob_index = i;
            }
        }
    }

    if (input_blob_index == -1)
        CV_Error(Error::StsError, "Const input blob for weights not found");

    Pin kernel_inp = parsePin(layer.input(input_blob_index));
    if (const_layers.find(kernel_inp.name) == const_layers.end())
        CV_Error(Error::StsError, "Input [" + layer.input(input_blob_index) +
                                  "] for node [" + layer.name() + "] not found");
    if (kernel_inp.blobIndex != 0)
        CV_Error(Error::StsError, "Unsupported kernel input");

    if(actual_inp_blob_idx) {
        *actual_inp_blob_idx = input_blob_index;
    }

    int nodeIdx = const_layers.at(kernel_inp.name);
    if (nodeIdx < netBin.node_size() && netBin.node(nodeIdx).name() == kernel_inp.name)
    {
        return netBin.node(nodeIdx).attr().at("value").tensor();
    }
    else
    {
        CV_Assert_N(nodeIdx < netTxt.node_size(),
                    netTxt.node(nodeIdx).name() == kernel_inp.name);
        return netTxt.node(nodeIdx).attr().at("value").tensor();
    }
}

static void addConstNodes(tensorflow::GraphDef& net, std::map<String, int>& const_layers,
                          std::set<String>& layers_to_ignore)
{
    CV_LOG_DEBUG(NULL, "DNN/TF: addConstNodes(): handling " << net.node_size() << " nodes...");
    for (int li = 0; li < net.node_size(); li++)
    {
        const tensorflow::NodeDef &layer = net.node(li);
        String name = layer.name();
        String type = layer.op();

        //CV_LOG_DEBUG(NULL, "DNN/TF: layer_id=" << li << " - '" << name << "' @ " << type);

        try
        {
            if (type == "Dequantize")
            {
                // Example of Dequantize node:
                //   name: "conv2d_1/bias"
                //   op: "Dequantize"
                //   input: "conv2d_1/bias_quantized_const" (tensor of dtype DT_QUINT8)
                //   input: "conv2d_1/bias_quantized_min"
                //   input: "conv2d_1/bias_quantized_max"
                //   attr { key: "T" value { type: DT_QUINT8 } }   (quantized type)
                //   attr { key: "mode" value { s: "MIN_FIRST" } } (quantization technique)
                CV_CheckEQ(layer.input_size(), 3, "Dequantize: 3 inputs is supported only");
                for (int i = 0; i < 3; ++i)
                    CV_Assert(const_layers.find(layer.input(i)) != const_layers.end());
                CV_Assert(hasLayerAttr(layer, "mode") &&
                          getLayerAttr(layer, "mode").s() == "MIN_FIRST");

                int tensorId = const_layers[layer.input(0)];
                int minId = const_layers[layer.input(1)];
                int maxId = const_layers[layer.input(2)];

                tensorflow::TensorProto* tensor = net.mutable_node(tensorId)
                                                    ->mutable_attr()->at("value")
                                                     .mutable_tensor();
                CV_CheckEQ((int)tensor->dtype(), (int)tensorflow::DT_QUINT8, "");

                Mat qMin = getTensorContent(net.node(minId).attr().at("value").tensor());
                Mat qMax = getTensorContent(net.node(maxId).attr().at("value").tensor());
                CV_CheckEQ(qMin.total(), (size_t)1, "");
                CV_CheckTypeEQ(qMin.type(), CV_32FC1, "");
                CV_CheckEQ(qMax.total(), (size_t)1, "");
                CV_CheckTypeEQ(qMax.type(), CV_32FC1, "");

                Mat content = getTensorContent(*tensor);

                float minVal = qMin.at<float>(0);
                float rangeScale = (qMax.at<float>(0) - minVal) / 255;
                CV_Assert(rangeScale >= 0);
                content.convertTo(content, CV_32FC1, rangeScale,
                                  rangeScale * cvRound(minVal / rangeScale));

                tensor->set_dtype(tensorflow::DT_FLOAT);
                tensor->set_tensor_content(content.data, content.total() * content.elemSize1());

                net.mutable_node(tensorId)->set_name(name);
                CV_Assert(const_layers.insert(std::make_pair(name, tensorId)).second);
                layers_to_ignore.insert(name);
                continue;
            }
            else if (type != "Const")
                continue;  // only Const parameters are supported

            if (layer.attr().find("value") != layer.attr().end())
            {
                CV_Assert(const_layers.insert(std::make_pair(name, li)).second);
            }
            layers_to_ignore.insert(name);
        }
        catch (const std::exception& e)
        {
            CV_LOG_ERROR(NULL, "DNN/TF: Can't handle node='" << name << "'. Exception: " << e.what());
            throw;
        }
    }
    CV_LOG_DEBUG(NULL, "DNN/TF: layers_to_ignore.size() = " << layers_to_ignore.size());
}

// If all inputs of specific layer have the same data layout we can say that
// this layer's output has this data layout too. Returns DATA_LAYOUT_UNKNOWN otherwise.
DataLayout TFImporter::predictOutputDataLayout(const tensorflow::NodeDef& layer)
{
    DataLayout layout = getDataLayout(layer);
    if (layout != DATA_LAYOUT_UNKNOWN)
    {
        CV_LOG_DEBUG(NULL, "DNN/TF: predictOutputDataLayout(" << layer.name() << " @ " << layer.op() << ") => " << (int)layout << " (from attrs)");
        return layout;
    }

    // Determine layout by layer's inputs
    for (int i = 0, n = layer.input_size(); i < n; ++i)
    {
        std::map<String, DataLayout>::const_iterator it = data_layouts.find(getNodeName(layer.input(i)));
        if (it != data_layouts.end())
        {
            if (layout != DATA_LAYOUT_UNKNOWN)
            {
                if (it->second != layout && it->second != DATA_LAYOUT_UNKNOWN)
                    return DATA_LAYOUT_UNKNOWN;
            }
            else
                layout = it->second;
        }
    }

    if (layout != DATA_LAYOUT_UNKNOWN)
    {
        CV_LOG_DEBUG(NULL, "DNN/TF: predictOutputDataLayout(" << layer.name() << " @ " << layer.op() << ") => " << (int)layout << " (from inputs)");
        return layout;
    }

    // Determine layout by layer's consumers recursively.
    std::map<String, DataLayout>::const_iterator it = data_layouts.find(layer.name());
    CV_Assert(it != data_layouts.end());
    return it->second;
}

void TFImporter::populateNet()
{
    CV_Assert(netBin.ByteSize() || netTxt.ByteSize());

    CV_LOG_INFO(NULL, "DNN/TF: parsing model"
        << (netBin.has_versions() ? cv::format(" produced by TF v%d (min_consumer=%d)", (int)netBin.versions().producer(), (int)netBin.versions().min_consumer()) : cv::String(" (N/A version info)"))
        << ". Number of nodes = " << netBin.node_size()
    );

    if (netTxt.ByteSize())
    {
        CV_LOG_INFO(NULL, "DNN/TF: parsing config"
            << (netTxt.has_versions() ? cv::format(" produced by TF v%d (min_consumer=%d)", (int)netTxt.versions().producer(), (int)netTxt.versions().min_consumer()) : cv::String(" (N/A version info)"))
            << ". Number of nodes = " << netTxt.node_size()
        );

        RemoveIdentityOps(netBin);
        CV_LOG_DEBUG(NULL, "DNN/TF: RemoveIdentityOps(model) => " << netBin.node_size() << " nodes");
        RemoveIdentityOps(netTxt);
        CV_LOG_DEBUG(NULL, "DNN/TF: RemoveIdentityOps(config) => " << netTxt.node_size() << " nodes");

        sortByExecutionOrder(netTxt);
        CV_LOG_DEBUG(NULL, "DNN/TF: sortByExecutionOrder(config) => " << netTxt.node_size() << " nodes");
    }
    else
    {
        removePhaseSwitches(netBin);
        CV_LOG_DEBUG(NULL, "DNN/TF: removePhaseSwitches(model) => " << netBin.node_size() << " nodes");

        RemoveIdentityOps(netBin);
        CV_LOG_DEBUG(NULL, "DNN/TF: RemoveIdentityOps(model) => " << netBin.node_size() << " nodes");

        simplifySubgraphs(netBin);
        CV_LOG_DEBUG(NULL, "DNN/TF: simplifySubgraphs(model) => " << netBin.node_size() << " nodes");
        sortByExecutionOrder(netBin);
        CV_LOG_DEBUG(NULL, "DNN/TF: sortByExecutionOrder(model) => " << netBin.node_size() << " nodes");
    }

    tensorflow::GraphDef& net = netTxt.ByteSize() != 0 ? netTxt : netBin;

    int layersSize = net.node_size();

    // Pre-fill data layouts where they are set explicitly.
    // Assuming that nodes are in topological order
    for (int i = layersSize - 1; i >= 0; --i)
    {
        const tensorflow::NodeDef& layer = net.node(i);
        std::string name = layer.name();

        CV_LOG_DEBUG(NULL, "DNN/TF: node(" << i << " - '" << name << "') propagating layout...");

        try
        {
            DataLayout layout = getDataLayout(layer);
            std::map<String, DataLayout>::iterator it = data_layouts.find(name);
            if (it != data_layouts.end())
            {
                if (layout != DATA_LAYOUT_UNKNOWN)
                {
                    if (it->second == DATA_LAYOUT_UNKNOWN)
                        it->second = layout;
                    else if (it->second != layout)
                    {
                        it->second = DATA_LAYOUT_UNKNOWN;
                        layout = DATA_LAYOUT_UNKNOWN;
                    }
                }
                else
                    layout = it->second;
            }
            else
                data_layouts[name] = layout;

            // Specify input layers to have the same data layout.
            for (int j = 0; j < layer.input_size(); ++j)
            {
                name = getNodeName(layer.input(j));
                it = data_layouts.find(name);
                if (it != data_layouts.end())
                {
                    if (layout != DATA_LAYOUT_UNKNOWN)
                    {
                        if (it->second == DATA_LAYOUT_UNKNOWN)
                            it->second = layout;
                        else if (it->second != layout)
                            it->second = DATA_LAYOUT_UNKNOWN;
                    }
                }
                else
                    data_layouts[name] = layout;
            }
        }
        catch (const std::exception& e)
        {
            CV_LOG_ERROR(NULL, "DNN/TF: Can't propagate layout for node='" << name << "'. Exception: " << e.what());
            throw;
        }
    }

    addConstNodes(netBin, value_id, layers_to_ignore);
    addConstNodes(netTxt, value_id, layers_to_ignore);


    for (int li = 0; li < layersSize; li++)
    {
        const tensorflow::NodeDef& layer = net.node(li);

        const std::string name = layer.name();
        const std::string type = layer.op();
        const int ninputs = layer.input_size();
        CV_LOG_DEBUG(NULL, "DNN/TF: (" << li << "/" << layersSize << ") Parse layer " << name << " @ " << type << " with " << ninputs << " inputs");

        parseNode(layer);
    }

    for (size_t i = 0; i < netInputsNames.size(); i++)
    {
        CV_LOG_DEBUG(NULL, "DNN/TF: Model input: " << i << " - '" << netInputsNames[i] << "'");
        CV_Assert(!netInputsNames[i].empty());
    }
    dstNet.setInputsNames(netInputsNames);
    CV_LOG_DEBUG(NULL, "DNN/TF: ===================== Import completed =====================");
}

void TFImporter::addPermuteLayer(const int* order, const std::string& permName, Pin& inpId)
{
    LayerParams permLP;
    permLP.set("order", DictValue::arrayInt<const int*>(order, 4));
    CV_Assert(layer_id.find(permName) == layer_id.end());
    int permId = dstNet.addLayer(permName, "Permute", permLP);
    layer_id[permName] = permId;
    connect(layer_id, dstNet, inpId, permId, 0);
    inpId = Pin(permName);
}

void TFImporter::parseNode(const tensorflow::NodeDef& layer_)
{
    tensorflow::NodeDef layer = layer_;

    tensorflow::GraphDef& net = netTxt.ByteSize() != 0 ? netTxt : netBin;

    /*const*/ std::string name = layer.name();
    /*const*/ std::string type = layer.op();
    /*const*/ int num_inputs = layer.input_size();

    try
    {
        LayerParams layerParams;

        if (layers_to_ignore.find(name) != layers_to_ignore.end())
        {
            CV_LOG_DEBUG(NULL, "DNN/TF:     ignored");
            return;
        }

        DataLayout predictedLayout = predictOutputDataLayout(layer);
        data_layouts[name] = predictedLayout;

        if (type == "Conv2D" || type == "SpaceToBatchND" || type == "DepthwiseConv2dNative" || type == "Pad" || type == "MirrorPad" || type == "Conv3D")
        {
            CV_CheckGT(num_inputs, 0, "");
            // The first node of dilated convolution subgraph.
            // Extract input node, dilation rate and paddings.
            std::string input = layer.input(0);
            StrIntVector next_layers;
            if (type == "SpaceToBatchND" || type == "Pad")
            {
                next_layers = getNextLayers(net, name, "Conv2D");
                if (next_layers.empty())
                    next_layers = getNextLayers(net, name, "DepthwiseConv2dNative");
            }

            if (type == "SpaceToBatchND")
            {
                // op: "SpaceToBatchND"
                // input: "input"
                // input: "SpaceToBatchND/block_shape"
                // input: "SpaceToBatchND/paddings"
                CV_CheckEQ(num_inputs, 3, "");

                DictValue dilation = parseDims(getConstBlob(layer, value_id, 1));
                CV_Assert(dilation.size() == 2);
                layerParams.set("dilation_h", dilation.get<int>(0));
                layerParams.set("dilation_w", dilation.get<int>(1));

                Mat paddings;
                parseTensor<int>(getConstBlob(layer, value_id, 2), paddings);

                // paddings is a 2x2 matrix: [[top, bot], [left, right]]
                layerParams.set("pad_h", paddings.at<float>(0));
                layerParams.set("pad_w", paddings.at<float>(2));

                CV_Assert(next_layers.size() == 1);
                layers_to_ignore.insert(next_layers[0].first);

                // FIXIT don't override, rewrite this code
                layer = net.node(next_layers[0].second);
                name = layer.name();
                type = layer.op();
                num_inputs = layer.input_size();
                CV_LOG_DEBUG(NULL, "DNN/TF:     switched to layer " << name << " @ " << type << ") with " << num_inputs << " inputs");
            }
            else if (type == "Pad" || type == "MirrorPad")
            {
                Mat paddings = getTensorContent(getConstBlob(layer, value_id, 1));
                CV_Assert(paddings.type() == CV_32SC1);
                if (paddings.total() == 8)
                {
                    // Perhaps, we have NHWC padding dimensions order.
                    //  N    H    W    C
                    // 0 1  2 3  4 5  6 7
                    std::swap(paddings.at<int32_t>(2), paddings.at<int32_t>(6));
                    std::swap(paddings.at<int32_t>(3), paddings.at<int32_t>(7));
                    //  N    C    W    H
                    // 0 1  2 3  4 5  6 7
                    std::swap(paddings.at<int32_t>(4), paddings.at<int32_t>(6));
                    std::swap(paddings.at<int32_t>(5), paddings.at<int32_t>(7));
                    //  N    C    H    W
                    // 0 1  2 3  4 5  6 7
                }

                if (next_layers.empty() || paddings.total() != 8 ||
                    paddings.at<int32_t>(4) != paddings.at<int32_t>(5) ||
                    paddings.at<int32_t>(6) != paddings.at<int32_t>(7) || type == "MirrorPad")
                {
                    // Just a single padding layer.
                    layerParams.set("paddings", DictValue::arrayInt<int*>((int*)paddings.data, paddings.total()));
                    if (type == "MirrorPad")
                        layerParams.set("type", "reflect");

                    int id = dstNet.addLayer(name, "Padding", layerParams);
                    layer_id[name] = id;

                    connect(layer_id, dstNet, parsePin(input), id, 0);
                    return;
                }
                else
                {
                    // Merge with subsequent convolutional layer.
                    CV_Assert(next_layers.size() == 1);

                    layerParams.set("pad_h", paddings.at<int32_t>(4));
                    layerParams.set("pad_w", paddings.at<int32_t>(6));

                    layers_to_ignore.insert(next_layers[0].first);

                    // FIXIT don't override, rewrite this code
                    layer = net.node(next_layers[0].second);
                    name = layer.name();
                    type = layer.op();
                    num_inputs = layer.input_size();
                    CV_LOG_DEBUG(NULL, "DNN/TF:     switched to layer " << name << " @ " << type << ") with " << num_inputs << " inputs");
                }
            }

            // For the object detection networks, TensorFlow Object Detection API
            // predicts deltas for bounding boxes in yxYX (ymin, xmin, ymax, xmax)
            // order. We can manage it at DetectionOutput layer parsing predictions
            // or shuffle last convolution's weights.
            bool locPredTransposed = hasLayerAttr(layer, "loc_pred_transposed") &&
                                     getLayerAttr(layer, "loc_pred_transposed").b();

            layerParams.set("bias_term", false);
            layerParams.blobs.resize(1);

            next_layers = getNextLayers(net, name, "BiasAdd");
            if (next_layers.size() == 1) {
                layerParams.set("bias_term", true);
                layerParams.blobs.resize(2);

                int weights_layer_index = next_layers[0].second;

                blobFromTensor(getConstBlob(net.node(weights_layer_index), value_id), layerParams.blobs[1]);
                ExcludeLayer(net, weights_layer_index, 0, false);
                layers_to_ignore.insert(next_layers[0].first);

                // Shuffle bias from yxYX to xyXY.
                if (locPredTransposed)
                {
                    const int numWeights = layerParams.blobs[1].total();
                    float* biasData = reinterpret_cast<float*>(layerParams.blobs[1].data);
                    CV_Assert(numWeights % 4 == 0);
                    for (int i = 0; i < numWeights; i += 2)
                    {
                        std::swap(biasData[i], biasData[i + 1]);
                    }
                }
            }

            int kernelTensorInpId = -1;
            const tensorflow::TensorProto& kernelTensor = getConstBlob(layer, value_id, -1, &kernelTensorInpId);
            const String kernelTensorName = layer.input(kernelTensorInpId);
            std::map<String, Mat>::iterator sharedWeightsIt = sharedWeights.find(kernelTensorName);
            if (sharedWeightsIt == sharedWeights.end())
            {
                kernelFromTensor(kernelTensor, layerParams.blobs[0]);
                releaseTensor(const_cast<tensorflow::TensorProto*>(&kernelTensor));

                int* kshape = layerParams.blobs[0].size.p;
                const int outCh = kshape[0];
                const int inCh = kshape[1];
                const int height = kshape[2];
                const int width = kshape[3];
                if (type == "DepthwiseConv2dNative")
                {
                    CV_Assert(!locPredTransposed);
                    const int chMultiplier = kshape[0];

                    Mat copy = layerParams.blobs[0].clone();
                    float* src = (float*)copy.data;
                    float* dst = (float*)layerParams.blobs[0].data;
                    for (int i = 0; i < chMultiplier; ++i)
                        for (int j = 0; j < inCh; ++j)
                            for (int s = 0; s < height * width; ++s)
                                {
                                    int src_i = (i * inCh + j) * height * width + s;
                                    int dst_i = (j * chMultiplier + i) * height* width + s;
                                    dst[dst_i] = src[src_i];
                                }
                    // TODO Use reshape instead
                    kshape[0] = inCh * chMultiplier;
                    kshape[1] = 1;
                    size_t* kstep = layerParams.blobs[0].step.p;
                    kstep[0] = kstep[1]; // fix steps too
                }

                // Shuffle output channels from yxYX to xyXY.
                if (locPredTransposed)
                {
                    const int slice = height * width * inCh;
                    for (int i = 0; i < outCh; i += 2)
                    {
                        cv::Mat src(1, slice, CV_32F, layerParams.blobs[0].ptr<float>(i));
                        cv::Mat dst(1, slice, CV_32F, layerParams.blobs[0].ptr<float>(i + 1));
                        std::swap_ranges(src.begin<float>(), src.end<float>(), dst.begin<float>());
                    }
                }
                sharedWeights[kernelTensorName] = layerParams.blobs[0];
            }
            else
            {
                layerParams.blobs[0] = sharedWeightsIt->second;
            }
            Mat weights = layerParams.blobs[0];
            layerParams.set("kernel_size",  DictValue::arrayInt(&weights.size[2], weights.dims - 2));

            layerParams.set("num_output", layerParams.blobs[0].size[0]);

            setStrides(layerParams, layer);
            if (!layerParams.has("pad_w") && !layerParams.has("pad_h"))
                setPadding(layerParams, layer);

            // The final node of dilated convolution subgraph.
            next_layers = getNextLayers(net, name, "BatchToSpaceND");
            if (!next_layers.empty())
            {
                CV_Assert(next_layers.size() == 1);
                ExcludeLayer(net, next_layers[0].second, 0, false);
                layers_to_ignore.insert(next_layers[0].first);
            }

            int id = dstNet.addLayer(name, "Convolution", layerParams);
            layer_id[name] = id;

            // one input only
            connect(layer_id, dstNet, parsePin(input), id, 0);


            if (getDataLayout(name, data_layouts) == DATA_LAYOUT_UNKNOWN)
                data_layouts[name] = DATA_LAYOUT_NHWC;
        }
        else if (type == "BiasAdd" || type == "Add" || type == "AddV2" || type == "Sub" || type=="AddN")
        {
            CV_CheckGT(num_inputs, 0, "");
            bool haveConst = false;
            for(int ii = 0; !haveConst && ii < num_inputs; ++ii)
            {
                Pin input = parsePin(layer.input(ii));
                haveConst = value_id.find(input.name) != value_id.end();
            }
            CV_Assert(!haveConst || num_inputs == 2);

            if (haveConst)
            {
                Mat values = getTensorContent(getConstBlob(layer, value_id));
                CV_Assert(values.type() == CV_32FC1);
                if (type == "Sub")
                    values *= -1.0f;

                int id;
                if (values.total() == 1)  // is a scalar.
                {
                    layerParams.set("shift", values.at<float>(0));
                    id = dstNet.addLayer(name, "Power", layerParams);
                }
                else  // is a vector
                {
                    layerParams.blobs.resize(1, values);
                    id = dstNet.addLayer(name, "Shift", layerParams);
                }
                layer_id[name] = id;

                // one input only
                connect(layer_id, dstNet, parsePin(layer.input(0)), id, 0);
            }
            else
            {
                layerParams.set("operation", "sum");
                if (type == "Sub")
                {
                    static float subCoeffs[] = {1.f, -1.f};
                    layerParams.set("coeff", DictValue::arrayReal<float*>(subCoeffs, 2));
                }

                int id = dstNet.addLayer(name, "Eltwise", layerParams);
                layer_id[name] = id;

                for (int ii = 0; ii < num_inputs; ii++)
                {
                    Pin inp = parsePin(layer.input(ii));
                    if (layer_id.find(inp.name) == layer_id.end())
                        CV_Error(Error::StsError, "Input layer not found: " + inp.name);
                    connect(layer_id, dstNet, inp, id, ii);
                }
            }
        }
        else if (type == "MatMul")
        {
            CV_CheckEQ(num_inputs, 2, "");

            // For the object detection networks, TensorFlow Object Detection API
            // predicts deltas for bounding boxes in yxYX (ymin, xmin, ymax, xmax)
            // order. We can manage it at DetectionOutput layer parsing predictions
            // or shuffle last Faster-RCNN's matmul weights.
            bool locPredTransposed = hasLayerAttr(layer, "loc_pred_transposed") &&
                                     getLayerAttr(layer, "loc_pred_transposed").b();

            layerParams.set("bias_term", false);
            layerParams.blobs.resize(1);

            StrIntVector next_layers = getNextLayers(net, name, "BiasAdd");  // FIXIT Use layers fusion instead
            if (next_layers.empty())
            {
                next_layers = getNextLayers(net, name, "Add");
            }
            if (next_layers.size() == 1) {
                layerParams.set("bias_term", true);
                layerParams.blobs.resize(2);

                int weights_layer_index = next_layers[0].second;
                blobFromTensor(getConstBlob(net.node(weights_layer_index), value_id), layerParams.blobs[1]);
                ExcludeLayer(net, weights_layer_index, 0, false);
                layers_to_ignore.insert(next_layers[0].first);

                if (locPredTransposed)
                {
                    const int numWeights = layerParams.blobs[1].total();
                    float* biasData = reinterpret_cast<float*>(layerParams.blobs[1].data);
                    CV_Assert(numWeights % 4 == 0);
                    for (int i = 0; i < numWeights; i += 2)
                    {
                        std::swap(biasData[i], biasData[i + 1]);
                    }
                }
            }

            int kernel_blob_index = -1;
            const tensorflow::TensorProto& kernelTensor = getConstBlob(layer, value_id, -1, &kernel_blob_index);
            const String kernelTensorName = layer.input(kernel_blob_index);
            std::map<String, Mat>::iterator sharedWeightsIt = sharedWeights.find(kernelTensorName);
            if (sharedWeightsIt == sharedWeights.end())
            {
                blobFromTensor(kernelTensor, layerParams.blobs[0]);
                releaseTensor(const_cast<tensorflow::TensorProto*>(&kernelTensor));
                sharedWeights[kernelTensorName] = layerParams.blobs[0];
            }
            else
            {
                layerParams.blobs[0] = sharedWeightsIt->second;
            }

            if (kernel_blob_index == 1) { // In this case output is computed by x*W formula - W should be transposed
                Mat data = layerParams.blobs[0].t();
                layerParams.blobs[0] = data.clone();
            }

            layerParams.set("num_output", layerParams.blobs[0].size[0]);
            if (locPredTransposed)
            {
                CV_Assert(layerParams.blobs[0].dims == 2);
                for (int i = 0; i < layerParams.blobs[0].size[0]; i += 2)
                {
                    cv::Mat src = layerParams.blobs[0].row(i);
                    cv::Mat dst = layerParams.blobs[0].row(i + 1);
                    std::swap_ranges(src.begin<float>(), src.end<float>(), dst.begin<float>());
                }
            }

            int id = dstNet.addLayer(name, "InnerProduct", layerParams);
            layer_id[name] = id;

            // one input only
            int input_blob_index = kernel_blob_index == 0 ? 1 : 0;
            connect(layer_id, dstNet, parsePin(layer.input(input_blob_index)), id, 0);
            data_layouts[name] = DATA_LAYOUT_PLANAR;
        }
        else if (type == "Reshape")
        {
            CV_CheckGT(num_inputs, 0, "");
            Pin inpId = parsePin(layer.input(0));
            DataLayout inpLayout = getDataLayout(layer.input(0), data_layouts);
            // There are two possible implementations: reshape an input using
            // predefined sizes or use a second input blob as a source of new shape.
            if (value_id.find(layer.input(1)) != value_id.end())
            {
                Mat newShape = getTensorContent(getConstBlob(layer, value_id, 1));
                int newShapeSize = newShape.total();
                bool hasSwap = false;
                if (newShapeSize == 4 && hasAllOnes(newShape, 0, 2))
                {
                    // NHWC->NCHW
                    std::swap(*newShape.ptr<int32_t>(0, 2), *newShape.ptr<int32_t>(0, 3));
                    std::swap(*newShape.ptr<int32_t>(0, 1), *newShape.ptr<int32_t>(0, 2));
                    hasSwap = true;
                }
                if (inpLayout == DATA_LAYOUT_NHWC)
                {
                    if (newShapeSize >= 2 || newShape.at<int>(1) == 1)
                    {
                        int order[] = {0, 2, 3, 1};  // From OpenCV's NCHW to NHWC.
                        addPermuteLayer(order, name + "/nhwc", inpId);
                        if (newShapeSize < 4)
                        {
                            inpLayout = DATA_LAYOUT_NCHW;
                        }
                        else
                        {
                            inpLayout = DATA_LAYOUT_NHWC;
                        }
                    }
                }
                layerParams.set("dim", DictValue::arrayInt<int*>(newShape.ptr<int>(), newShapeSize));

                int id = dstNet.addLayer(name, "Reshape", layerParams);
                layer_id[name] = id;

                // one input only
                connect(layer_id, dstNet, inpId, id, 0);
                inpId = Pin(name);

                if ((inpLayout == DATA_LAYOUT_NHWC || inpLayout == DATA_LAYOUT_UNKNOWN || inpLayout == DATA_LAYOUT_PLANAR) &&
                    newShapeSize == 4 && !hasSwap)
                {
                    int order[] = {0, 3, 1, 2};  // Transform back to OpenCV's NCHW.
                    addPermuteLayer(order, name + "/nchw", inpId);
                    inpLayout = DATA_LAYOUT_NCHW;
                }

                data_layouts[name] = newShapeSize == 2 ? DATA_LAYOUT_PLANAR : inpLayout;
            }
            else
            {
                int id = dstNet.addLayer(name, "Reshape", layerParams);
                layer_id[name] = id;
                connect(layer_id, dstNet, inpId, id, 0);
                connect(layer_id, dstNet, parsePin(layer.input(1)), id, 1);
                data_layouts[name] = inpLayout;
            }
        }
        else if (type == "Flatten" || type == "Squeeze")
        {
            CV_CheckGT(num_inputs, 0, "");
            Pin inpId = parsePin(layer.input(0));
            int inpLayout = getDataLayout(layer.input(0), data_layouts);
            if (type == "Squeeze")
            {
                CV_Assert(hasLayerAttr(layer, "squeeze_dims"));
                const tensorflow::AttrValue& dims = getLayerAttr(layer, "squeeze_dims");
                std::vector<int> dimsVector(dims.list().i_size());
                for (int i = 0; i < dimsVector.size(); ++i)
                    dimsVector[i] = dims.list().i(i);

                // Flatten layer can squeeze dimensions range into one.
                std::sort(dimsVector.begin(), dimsVector.end());
                for (int i = 1; i < dimsVector.size(); ++i)
                {
                    if (dimsVector[i] != dimsVector[i - 1] + 1)
                        CV_Error(Error::StsNotImplemented, "Unsupported squeeze configuration");
                }
                int start = dimsVector.front() - 1, end = dimsVector.back();
                if (start == -1 && end == 0)  // squeeze 0th dimension
                {
                    start = 0;
                    end = 1;
                }
                layerParams.set("axis", start);
                layerParams.set("end_axis", end);
            }
            if (inpLayout == DATA_LAYOUT_NHWC)
            {
                LayerParams permLP;
                int order[] = {0, 2, 3, 1};  // From OpenCV's NCHW to NHWC.
                permLP.set("order", DictValue::arrayInt<int*>(order, 4));

                std::string permName = name + "/nchw";
                CV_Assert(layer_id.find(permName) == layer_id.end());
                int permId = dstNet.addLayer(permName, "Permute", permLP);
                layer_id[permName] = permId;
                connect(layer_id, dstNet, inpId, permId, 0);
                inpId = Pin(permName);
            }
            int id = dstNet.addLayer(name, "Flatten", layerParams);
            layer_id[name] = id;
            connect(layer_id, dstNet, inpId, id, 0);
            data_layouts[name] = DATA_LAYOUT_PLANAR;
        }
        else if (type == "Transpose")
        {
            CV_CheckGT(num_inputs, 0, "");
            Mat perm = getTensorContent(getConstBlob(layer, value_id, 1));
            CV_Assert(perm.type() == CV_32SC1);
            int* permData = (int*)perm.data;
            if (perm.total() == 4)
            {
                // Only NHWC <-> NCHW permutations are allowed. OpenCV is always
                // keep NCHW layout this way.
                int inpLayout = getDataLayout(layer.input(0), data_layouts);
                std::string type = "Identity";
                if (inpLayout == DATA_LAYOUT_NHWC)
                {
                    if (permData[0] == 0 && permData[1] == 3 && permData[2] == 1 && permData[3] == 2)
                    {
                        // in TensorFlow: NHWC->NCHW
                        // in OpenCV: NCHW->NCHW
                        data_layouts[name] = DATA_LAYOUT_NCHW;
                    }
                    else if (permData[0] == 0 && permData[1] == 1 && permData[2] == 2 && permData[3] == 3)
                    {
                        // in TensorFlow: NHWC->NHWC
                        // in OpenCV: NCHW->NCHW
                        data_layouts[name] = DATA_LAYOUT_NHWC;
                    }
                    else if (permData[0] == 0 && permData[1] == 3 && permData[2] == 2 && permData[3] == 1)
                    {
                        // in TensorFlow: NHWC->NCWH
                        // in OpenCV: NCHW->NCWH
                        int permData[] = {0, 1, 3, 2};
                        layerParams.set("order", DictValue::arrayInt<int*>(permData, perm.total()));
                        data_layouts[name] = DATA_LAYOUT_NCHW;  // we keep track NCHW because channels position only matters
                        type = "Permute";
                    }
                    else
                        CV_Error(Error::StsParseError, "Only NHWC <-> NCHW permutations are allowed.");
                }
                else if (inpLayout == DATA_LAYOUT_NCHW)
                {
                    if (permData[0] == 0 && permData[1] == 2 && permData[2] == 3 && permData[3] == 1)
                    {
                        // in TensorFlow: NCHW->NHWC
                        // in OpenCV: NCHW->NCHW
                        data_layouts[name] = DATA_LAYOUT_NHWC;
                    }
                    else if (permData[0] == 0 && permData[1] == 1 && permData[2] == 2 && permData[3] == 3)
                    {
                        // in TensorFlow: NCHW->NCHW
                        // in OpenCV: NCHW->NCHW
                        data_layouts[name] = DATA_LAYOUT_NCHW;
                    }
                    else
                        CV_Error(Error::StsParseError, "Only NHWC <-> NCHW permutations are allowed.");
                }
                int id = dstNet.addLayer(name, type, layerParams);
                layer_id[name] = id;
                connect(layer_id, dstNet, parsePin(layer.input(0)), id, 0);
            }
            else
            {
                layerParams.set("order", DictValue::arrayInt<int*>(permData, perm.total()));

                int id = dstNet.addLayer(name, "Permute", layerParams);
                layer_id[name] = id;

                // one input only
                connect(layer_id, dstNet, parsePin(layer.input(0)), id, 0);
                data_layouts[name] = DATA_LAYOUT_UNKNOWN;
            }
        }
        else if (type == "Const")
        {
        }
        else if (type == "LRN")
        {
            CV_CheckGT(num_inputs, 0, "");
            if(hasLayerAttr(layer, "alpha")) {
                layerParams.set("alpha", getLayerAttr(layer, "alpha").f());
            }
            if(hasLayerAttr(layer, "beta")) {
                layerParams.set("beta", getLayerAttr(layer, "beta").f());
            }
            if(hasLayerAttr(layer, "depth_radius")) {
                int radius = (int)getLayerAttr(layer, "depth_radius").i();
                layerParams.set("local_size", 2*radius + 1);
            }
            if(hasLayerAttr(layer, "bias")) {
                layerParams.set("bias", getLayerAttr(layer, "bias").f());
            }
            layerParams.set("norm_by_size", false);

            int id = dstNet.addLayer(name, "LRN", layerParams);
            layer_id[name] = id;

            connectToAllBlobs(layer_id, dstNet, parsePin(layer.input(0)), id, num_inputs);
        }
        else if (type == "Concat" || type == "ConcatV2")
        {
            CV_CheckGT(num_inputs, 0, "");
            int axisId = (type == "Concat" ? 0 : num_inputs - 1);
            int axis = getConstBlob(layer, value_id, axisId).int_val().Get(0);

            if (getDataLayout(name, data_layouts) == DATA_LAYOUT_NHWC)
                axis = toNCHW(axis);
            else if (getDataLayout(name, data_layouts) == DATA_LAYOUT_NDHWC)
                axis = toNCDHW(axis);
            layerParams.set("axis", axis);

            // input(0) or input(n-1) is concat_dim
            int from = (type == "Concat" ? 1 : 0);
            int to = (type == "Concat" ? num_inputs : num_inputs - 1);

            for (int ii = from; ii < to; ii++)
            {
                Pin inp = parsePin(layer.input(ii));
                if (layer_id.find(inp.name) == layer_id.end())
                {
                    // There are constant inputs.
                    LayerParams lp;
                    lp.name = inp.name;
                    lp.type = "Const";
                    lp.blobs.resize(1);
                    blobFromTensor(getConstBlob(layer, value_id, ii), lp.blobs.back());
                    CV_Assert_N(!lp.blobs[0].empty(), lp.blobs[0].type() == CV_32F);

                    int constInpId = dstNet.addLayer(lp.name, lp.type, lp);
                    layer_id[lp.name] = constInpId;
                }
            }

            int id = dstNet.addLayer(name, "Concat", layerParams);
            layer_id[name] = id;

            for (int ii = from; ii < to; ii++)
            {
                Pin inp = parsePin(layer.input(ii));
                if (layer_id.find(inp.name) == layer_id.end())
                    CV_Error(Error::StsError, "Input layer not found: " + inp.name);
                connect(layer_id, dstNet, inp, id, ii - from);
            }
        }
        else if (type == "MaxPool" || type == "MaxPool3D")
        {
            CV_CheckGT(num_inputs, 0, "");
            layerParams.set("pool", "max");

            setKSize(layerParams, layer);
            setStrides(layerParams, layer);
            setPadding(layerParams, layer);
            // Test_TensorFlow_nets.EAST_text_detection/1, NGRAPH/CPU
            layerParams.set("ceil_mode", false);

            int id = dstNet.addLayer(name, "Pooling", layerParams);
            layer_id[name] = id;

            connectToAllBlobs(layer_id, dstNet, parsePin(layer.input(0)), id, num_inputs);
        }
        else if (type == "AvgPool" || type == "AvgPool3D")
        {
            CV_CheckGT(num_inputs, 0, "");
            layerParams.set("pool", "ave");
            layerParams.set("ave_pool_padded_area", false);
            setKSize(layerParams, layer);
            setStrides(layerParams, layer);
            setPadding(layerParams, layer);

            int id = dstNet.addLayer(name, "Pooling", layerParams);
            layer_id[name] = id;

            connectToAllBlobs(layer_id, dstNet, parsePin(layer.input(0)), id, num_inputs);
        }
        else if (type == "MaxPoolGrad")
        {
            CV_CheckEQ(num_inputs, 3, "");

            layerParams.set("pool_k_h", 0);
            layerParams.set("pool_k_w", 0);
            layerParams.set("pool_stride_h", 0);
            layerParams.set("pool_stride_w", 0);
            layerParams.set("pool_pad_h", 0);
            layerParams.set("pool_pad_w", 0);

            int id = dstNet.addLayer(name, "MaxUnpool", layerParams);
            layer_id[name] = id;

            connect(layer_id, dstNet, parsePin(layer.input(2)), id, 0);
            connect(layer_id, dstNet, parsePin(layer.input(1) + ":1"), id, 1);
            connect(layer_id, dstNet, parsePin(layer.input(0)), id, 2);
        }
        else if (type == "Placeholder")
        {
            if (!hasLayerAttr(layer, "dtype") ||
                getLayerAttr(layer, "dtype").type() != tensorflow::DT_BOOL)  // If input is not a train/test flag.
            {
                netInputsNames.push_back(name);
                layer_id[name] = 0;
            }
            tensorflow::TensorShapeProto shape;
            if (hasLayerAttr(layer, "shape"))
                shape = getLayerAttr(layer, "shape").shape();
            else if (hasLayerAttr(layer, "_output_shapes"))
            {
                tensorflow::AttrValue_ListValue list = getLayerAttr(layer, "_output_shapes").list();
                if (list.shape_size())
                    shape = list.shape()[0];
            }
            if (shape.dim_size())
            {
                MatShape dims(shape.dim_size());
                for (int i = 0; i < dims.size(); ++i)
                    dims[i] = shape.dim(i).size();
                if (dims.size() == 4 && predictedLayout == DATA_LAYOUT_NHWC)
                {
                    std::swap(dims[1], dims[3]);  // NHWC->NCWH
                    std::swap(dims[2], dims[3]);  // NCWH->NCHW
                    if (dims[0] == -1)  // It's OK to have undetermined batch size
                        dims[0] = 1;
                }
                bool hasNeg = false;
                for (int i = 0; i < dims.size() && !hasNeg; ++i)
                {
                    hasNeg = dims[i] < 0;
                }
                if (!hasNeg)
                    netInputShapes.push_back(dims);
            }
        }
        else if (type == "Split") {
            // TODO: determining axis index remapping by input dimensions order of input blob
            // TODO: slicing input may be Const op
            // TODO: slicing kernels for convolutions - in current implementation it is impossible
            // TODO: add parsing num of slices parameter
            CV_CheckEQ(num_inputs, 2, "");
            // num_split
            // 1st blob is dims tensor
            int axis = getConstBlob(layer, value_id, 0).int_val().Get(0);
            if (getDataLayout(name, data_layouts) == DATA_LAYOUT_NHWC)
                axis = toNCHW(axis);
            layerParams.set("axis", axis);

            if (hasLayerAttr(layer, "num_split"))
                layerParams.set("num_split", getLayerAttr(layer, "num_split").i());

            int id = dstNet.addLayer(name, "Slice", layerParams);
            layer_id[name] = id;

            // one input only
            connect(layer_id, dstNet, parsePin(layer.input(1)), id, 0);
        }
        else if (type == "Slice")
        {
            // op: "Slice"
            // input: "input_node"
            // input: "Slice/begin"
            // input: "Slice/size"
            CV_CheckEQ(num_inputs, 3, "");
            Mat begins = getTensorContent(getConstBlob(layer, value_id, 1));
            Mat sizes = getTensorContent(getConstBlob(layer, value_id, 2));
            CV_Assert_N(!begins.empty(), !sizes.empty());
            CV_CheckTypeEQ(begins.type(), CV_32SC1, "");
            CV_CheckTypeEQ(sizes.type(), CV_32SC1, "");

            if (begins.total() == 4 && getDataLayout(name, data_layouts) == DATA_LAYOUT_NHWC)
            {
                // Swap NHWC parameters' order to NCHW.
                std::swap(*begins.ptr<int32_t>(0, 2), *begins.ptr<int32_t>(0, 3));
                std::swap(*begins.ptr<int32_t>(0, 1), *begins.ptr<int32_t>(0, 2));
                std::swap(*sizes.ptr<int32_t>(0, 2), *sizes.ptr<int32_t>(0, 3));
                std::swap(*sizes.ptr<int32_t>(0, 1), *sizes.ptr<int32_t>(0, 2));
            }
            layerParams.set("begin", DictValue::arrayInt((int*)begins.data, begins.total()));
            layerParams.set("size", DictValue::arrayInt((int*)sizes.data, sizes.total()));

            int id = dstNet.addLayer(name, "Slice", layerParams);
            layer_id[name] = id;

            connect(layer_id, dstNet, parsePin(layer.input(0)), id, 0);
        }
        else if (type == "StridedSlice")
        {
            CV_CheckEQ(num_inputs, 4, "");
            Mat begins = getTensorContent(getConstBlob(layer, value_id, 1));
            Mat ends = getTensorContent(getConstBlob(layer, value_id, 2));
            Mat strides = getTensorContent(getConstBlob(layer, value_id, 3));
            CV_CheckTypeEQ(begins.type(), CV_32SC1, "");
            CV_CheckTypeEQ(ends.type(), CV_32SC1, "");
            CV_CheckTypeEQ(strides.type(), CV_32SC1, "");
            const int num = begins.total();
            CV_Assert_N(num == ends.total(), num == strides.total());

            int end_mask = getLayerAttr(layer, "end_mask").i();
            for (int i = 0; i < num; ++i)
            {
                if (ends.at<int>(i) < 0)
                    ends.at<int>(i) -= 1;
                if (end_mask & (1 << i))
                    ends.at<int>(i) = -1;
                if (strides.at<int>(i) != 1)
                    CV_Error(Error::StsNotImplemented,
                             format("StridedSlice with stride %d", strides.at<int>(i)));
            }
            if (begins.total() == 4 && getDataLayout(name, data_layouts) == DATA_LAYOUT_NHWC)
            {
                // Swap NHWC parameters' order to NCHW.
                std::swap(begins.at<int>(2), begins.at<int>(3));
                std::swap(begins.at<int>(1), begins.at<int>(2));
                std::swap(ends.at<int>(2), ends.at<int>(3));
                std::swap(ends.at<int>(1), ends.at<int>(2));
            }
            layerParams.set("begin", DictValue::arrayInt((int*)begins.data, begins.total()));
            layerParams.set("end", DictValue::arrayInt((int*)ends.data, ends.total()));

            int id = dstNet.addLayer(name, "Slice", layerParams);
            layer_id[name] = id;

            connect(layer_id, dstNet, parsePin(layer.input(0)), id, 0);
        }
        else if (type == "Mul" || type == "RealDiv")
        {
            CV_CheckGT(num_inputs, 0, "");
            int constId = -1;
            for(int ii = 0; ii < num_inputs; ++ii)
            {
                Pin input = parsePin(layer.input(ii));
                if (value_id.find(input.name) != value_id.end())
                {
                    constId = ii;
                    break;
                }
            }
            CV_Assert((constId != -1) || (num_inputs == 2));

            if (constId != -1)
            {
                // Multiplication by constant.
                CV_CheckEQ(num_inputs, 2, "");
                Mat scaleMat = getTensorContent(getConstBlob(layer, value_id));
                CV_Assert(scaleMat.type() == CV_32FC1);
                if (type == "RealDiv")
                {
                    if (constId == 0)
                        CV_Error(Error::StsNotImplemented, "Division of constant over variable");
                    scaleMat = 1.0f / scaleMat;
                }

                int id;
                if (scaleMat.total() == 1)  // is a scalar.
                {
                    // Try to match with a LeakyRelu:
                    // node {
                    //   name: "LeakyRelu/mul"
                    //   op: "Mul"
                    //   input: "LeakyRelu/alpha"
                    //   input: "input"
                    // }
                    // node {
                    //   name: "LeakyRelu/Maximum"
                    //   op: "Maximum"
                    //   input: "LeakyRelu/mul"
                    //   input: "input"
                    // }
                    StrIntVector next_layers = getNextLayers(net, name, "Maximum");
                    if (!next_layers.empty())
                    {
                        int maximumLayerIdx = next_layers[0].second;

                        CV_Assert(net.node(maximumLayerIdx).input_size() == 2);

                        // The input from the Mul layer can also be at index 1.
                        int mulInputIdx = (net.node(maximumLayerIdx).input(0) == name) ? 0 : 1;

                        ExcludeLayer(net, maximumLayerIdx, mulInputIdx, false);
                        layers_to_ignore.insert(next_layers[0].first);

                        layerParams.set("negative_slope", scaleMat.at<float>(0));
                        id = dstNet.addLayer(name, "ReLU", layerParams);
                    }
                    else
                    {
                        // Just a multiplication.
                        layerParams.set("scale", scaleMat.at<float>(0));
                        id = dstNet.addLayer(name, "Power", layerParams);
                    }
                }
                else  // is a vector
                {
                    layerParams.blobs.resize(1, scaleMat);

                   StrIntVector next_layers = getNextLayers(net, name, "Add");
                   if (!next_layers.empty())
                   {
                       layerParams.set("bias_term", true);
                       layerParams.blobs.resize(2);

                       int weights_layer_index = next_layers[0].second;
                       blobFromTensor(getConstBlob(net.node(weights_layer_index), value_id), layerParams.blobs.back());
                       ExcludeLayer(net, weights_layer_index, 0, false);
                       layers_to_ignore.insert(next_layers[0].first);
                   }

                    if (hasLayerAttr(layer, "axis"))
                        layerParams.set("axis", getLayerAttr(layer, "axis").i());

                    id = dstNet.addLayer(name, "Scale", layerParams);
                }
                layer_id[name] = id;

                Pin inp0 = parsePin(layer.input(0));
                if (layer_id.find(inp0.name) != layer_id.end())
                    // First operand is a constant.
                    connect(layer_id, dstNet, parsePin(layer.input(0)), id, 0);
                else
                    connect(layer_id, dstNet, parsePin(layer.input(1)), id, 0);
            }
            else
            {
                // Check if all the inputs have the same shape.
                bool equalInpShapes = true;
                bool isShapeOnes = false;
                MatShape outShape0;
                for (int ii = 0; ii < num_inputs && !netInputShapes.empty(); ii++)
                {
                    Pin pin = parsePin(layer.input(ii));
                    int inpId = layer_id.find(pin.name)->second;

                    // Get input shape
                    MatShape outShape;
                    std::vector<MatShape> inpShapes, outShapes;
                    dstNet.getLayerShapes(netInputShapes, inpId, inpShapes, outShapes);
                    CV_CheckGT(static_cast<int>(outShapes.size()), pin.blobIndex, "");
                    outShape = outShapes[pin.blobIndex];

                    if (ii == 0)
                    {
                        outShape0 = outShape;
                    }
                    else if (outShape != outShape0)
                    {
                        equalInpShapes = false;
                        isShapeOnes = isAllOnes(outShape, 2, outShape.size()) ||
                                      isAllOnes(outShape0, 2, outShape0.size());
                        break;
                    }
                }

                int id;
                if (equalInpShapes || netInputShapes.empty() || (!equalInpShapes && isShapeOnes))
                {
                    layerParams.set("operation", type == "RealDiv" ? "div" : "prod");
                    id = dstNet.addLayer(name, "Eltwise", layerParams);
                }
                else
                {
                    if (type == "RealDiv")
                        CV_Error(Error::StsNotImplemented, "Division of non equal tensors");
                    id = dstNet.addLayer(name, "Scale", layerParams);
                }

                layer_id[name] = id;

                for (int ii = 0; ii < num_inputs; ii++)
                {
                    Pin inp = parsePin(layer.input(ii));
                    if (layer_id.find(inp.name) == layer_id.end())
                        CV_Error(Error::StsError, "Input layer not found: " + inp.name);
                    connect(layer_id, dstNet, inp, id, ii);
                }
            }
        }
        else if (type == "FusedBatchNorm" || type == "FusedBatchNormV3")
        {
            // op: "FusedBatchNorm"
            // input: "input"
            // input: "BatchNorm/gamma"
            // input: "BatchNorm/beta"
            // input: "BatchNorm/moving_mean"
            // input: "BatchNorm/moving_variance"
            CV_CheckEQ(num_inputs, 5, "Expected gamma, beta, mean and std");
            Pin inpId = parsePin(layer.input(0));

            bool isTraining = hasLayerAttr(layer, "is_training") && getLayerAttr(layer, "is_training").b();

            layerParams.blobs.resize(2);

            const tensorflow::TensorProto& gammaTensor = getConstBlob(layer, value_id, 1);
            if (!gammaTensor.tensor_content().empty())
            {
                layerParams.blobs.resize(layerParams.blobs.size() + 1);
                layerParams.set("has_weight", true);
                blobFromTensor(gammaTensor, layerParams.blobs.back());
            }
            else
                layerParams.set("has_weight", false);

            const tensorflow::TensorProto& betaTensor = getConstBlob(layer, value_id, 2);
            if (!betaTensor.tensor_content().empty())
            {
                layerParams.blobs.resize(layerParams.blobs.size() + 1);
                layerParams.set("has_bias", true);
                blobFromTensor(betaTensor, layerParams.blobs.back());
            }
            else
                layerParams.set("has_bias", false);

            Mat mean, std;
            if (isTraining)
            {
                if (layerParams.blobs.size() == 2)
                    CV_Error(Error::StsNotImplemented, "Cannot determine number "
                             "of parameters for batch normalization layer.");
                mean = Mat::zeros(1, layerParams.blobs[2].total(), CV_32F);
                std = Mat::ones(1, layerParams.blobs[2].total(), CV_32F);

                // Add an extra layer: Mean-Variance normalization
                LayerParams mvnParams;
                std::string mvnName = name + "/MVN";
                CV_Assert(layer_id.find(mvnName) == layer_id.end());
                int mvnId = dstNet.addLayer(mvnName, "MVN", mvnParams);
                layer_id[mvnName] = mvnId;
                connect(layer_id, dstNet, inpId, mvnId, 0);
                inpId = Pin(mvnName);
            }
            else
            {
                blobFromTensor(getConstBlob(layer, value_id, 3), mean);
                blobFromTensor(getConstBlob(layer, value_id, 4), std);
            }
            layerParams.blobs[0] = mean;
            layerParams.blobs[1] = std;

            if (hasLayerAttr(layer, "epsilon"))
                layerParams.set("eps", getLayerAttr(layer, "epsilon").f());

            int id = dstNet.addLayer(name, "BatchNorm", layerParams);
            layer_id[name] = id;

            // one input only
            connect(layer_id, dstNet, inpId, id, 0);
        }
        else if (type == "Conv2DBackpropInput")
        {
            // op: "Conv2DBackpropInput"
            // input: "conv2d_transpose/output_shape"
            // input: "weights"
            // input: "input"
            CV_CheckEQ(num_inputs, 3, "Expected output shape, weights and input nodes");

            layerParams.set("bias_term", false);
            layerParams.blobs.resize(1);

            StrIntVector next_layers = getNextLayers(net, name, "BiasAdd");
            if (next_layers.size() == 1)
            {
                layerParams.set("bias_term", true);
                layerParams.blobs.resize(2);

                int weights_layer_index = next_layers[0].second;

                blobFromTensor(getConstBlob(net.node(weights_layer_index), value_id), layerParams.blobs[1]);
                ExcludeLayer(net, weights_layer_index, 0, false);
                layers_to_ignore.insert(next_layers[0].first);
            }

            kernelFromTensor(getConstBlob(layer, value_id, 1), layerParams.blobs[0]);

            const int* kshape = layerParams.blobs[0].size.p;
            const int kernelH = kshape[2];
            const int kernelW = kshape[3];
            layerParams.set("kernel_h", kernelH);
            layerParams.set("kernel_w", kernelW);
            layerParams.set("num_output", kshape[1]);

            setStrides(layerParams, layer);
            setPadding(layerParams, layer);

            // For convolution layer, output shape computes as
            // o = 1 + (i - k + 2*p) / s
            // i - input size, o - output size, k - kernel size, p - pad, s - stride
            // In TensorFlow, p == 0 is padMode == 'VALID' or p == (k - 1) / 2
            // considering that k is odd.
            // SAME:  o = 1 + (i - 1) / s
            // VALID: o = 1 + i / s
            // Deconvolution's layer output shape computes as
            // SAME:  o = 1 + (i - 1)*s
            // VALID: o = (i - 1)*s
            // If output_shape differs from formulas above then adjust padding is applied.

            const int strideY = layerParams.get<int>("stride_h");
            const int strideX = layerParams.get<int>("stride_w");
            Mat outShape = getTensorContent(getConstBlob(layer, value_id, 0));
            const int outH = outShape.at<int>(1);
            const int outW = outShape.at<int>(2);
            if (layerParams.get<String>("pad_mode") == "SAME")
            {
                layerParams.set("adj_w", (outW - 1) % strideX);
                layerParams.set("adj_h", (outH - 1) % strideY);
            }
            else if (layerParams.get<String>("pad_mode") == "VALID")
            {
                layerParams.set("adj_w", (outW - kernelW) % strideX);
                layerParams.set("adj_h", (outH - kernelH) % strideY);
            }
            int id = dstNet.addLayer(name, "Deconvolution", layerParams);
            layer_id[name] = id;

            // one input only
            connect(layer_id, dstNet, parsePin(layer.input(2)), id, 0);
        }
        else if (type == "BlockLSTM")
        {
            // op: "BlockLSTM"
            // input: "lstm_block_wrapper/ToInt64/x"  (ignore, number of time stamps)
            // input: "input"
            // input: "lstm_block_wrapper/zeros"      (ignore)
            // input: "lstm_block_wrapper/zeros"      (ignore)
            // input: "lstm_block_wrapper/kernel"
            // input: "lstm_block_wrapper/w_i_diag"
            // input: "lstm_block_wrapper/w_f_diag"
            // input: "lstm_block_wrapper/w_o_diag"
            // input: "lstm_block_wrapper/bias"
            CV_CheckEQ(num_inputs, 9, "Unexpected number of input nodes");

            if (hasLayerAttr(layer, "forget_bias"))
                layerParams.set("forget_bias", getLayerAttr(layer, "forget_bias").f());

            if (hasLayerAttr(layer, "forget_bias"))
            {
                float cellClip = getLayerAttr(layer, "cell_clip").f();
                // Cell clip disabled if it's negative.
                if (cellClip >= 0)
                {
                    layerParams.set("use_cell_clip", true);
                    layerParams.set("cell_clip", cellClip);
                }
            }

            Mat W, Wh, Wx, b;
            blobFromTensor(getConstBlob(layer, value_id, 4), W);
            blobFromTensor(getConstBlob(layer, value_id, 8), b);
            const int outSize = W.cols / 4;

            // IGFO->IFOG
            float* weightData = (float*)W.data;
            for (int i = 0; i < W.rows; ++i)
                for (int j = 0; j < outSize; ++j)
                {
                    std::swap(weightData[i * W.cols + 1 * outSize + j],
                              weightData[i * W.cols + 2 * outSize + j]);
                    std::swap(weightData[i * W.cols + 2 * outSize + j],
                              weightData[i * W.cols + 3 * outSize + j]);
                }
            Wx = W.rowRange(0, W.rows - outSize).t();
            Wh = W.rowRange(W.rows - outSize, W.rows).t();

            layerParams.blobs.resize(3);
            layerParams.blobs[0] = Wh;
            layerParams.blobs[1] = Wx;
            layerParams.blobs[2] = b;

            if (hasLayerAttr(layer, "use_peephole"))
            {
                bool usePeephole = getLayerAttr(layer, "use_peephole").b();
                if (usePeephole)
                {
                    layerParams.set("use_peephole", true);
                    layerParams.blobs.resize(6);
                    for (int i = 0; i < 3; ++i)
                    {
                        Mat w;
                        blobFromTensor(getConstBlob(layer, value_id, 5 + i), w);
                        w = w.reshape(1, w.total());  // Single column.
                        w = Mat::diag(w);  // Make a diagonal matrix.
                        layerParams.blobs[3 + i] = w;
                    }
                }
            }

            int id = dstNet.addLayer(name, "LSTM", layerParams);
            layer_id[name] = id;

            // one input only
            connect(layer_id, dstNet, parsePin(layer.input(1)), id, 0);
            data_layouts[name] = DATA_LAYOUT_UNKNOWN;
        }
        else if (type == "ResizeNearestNeighbor" || type == "ResizeBilinear" || type == "FusedResizeAndPadConv2D")
        {
            CV_CheckGT(num_inputs, 0, "");
            std::string convWeights = "";
            if (type == "FusedResizeAndPadConv2D")
            {
                // input: "mul_1"
                // input: "decoder/ResizeBilinear/size"
                // input: "decoder/decoder_conv0/Conv2D_dummy_paddings"
                // input: "decoder/decoder_conv0/weights"
                CV_CheckEQ(num_inputs, 4, "Number of input for FusedResizeAndPadConv2D");

                Mat paddings = getTensorContent(getConstBlob(layer, value_id, 2));
                CV_CheckEQ(countNonZero(paddings), 0, "Unsupported mode");

                convWeights = layer.input(3);
                layer.mutable_input()->DeleteSubrange(2, 2);  // FIXIT do NOT modify input model
                num_inputs = layer.input_size();
                name = name + "/resize";

                if (hasLayerAttr(layer, "resize_align_corners"))
                {
                    // FIXIT do NOT modify input model
                    layer.mutable_attr()->insert(
                        ::google::protobuf::MapPair<std::string, tensorflow::AttrValue>("align_corners",
                                                                                        getLayerAttr(layer, "resize_align_corners")));
                }
            }
            if (num_inputs == 2)
            {
                Mat outSize = getTensorContent(getConstBlob(layer, value_id, 1));
                CV_CheckTypeEQ(outSize.type(), CV_32SC1, ""); CV_CheckEQ(outSize.total(), (size_t)2, "");
                layerParams.set("height", outSize.at<int>(0, 0));
                layerParams.set("width", outSize.at<int>(0, 1));
            }
            else if (num_inputs == 3)
            {
                Mat factorHeight = getTensorContent(getConstBlob(layer, value_id, 1));
                Mat factorWidth = getTensorContent(getConstBlob(layer, value_id, 2));
                factorHeight.convertTo(factorHeight, CV_32F);
                factorWidth.convertTo(factorWidth, CV_32F);
                layerParams.set("zoom_factor_x", factorWidth.at<float>(0));
                layerParams.set("zoom_factor_y", factorHeight.at<float>(0));
            }
            else
                CV_Check(num_inputs, num_inputs == 2 || num_inputs == 3, "");

            if (type == "ResizeNearestNeighbor")
                layerParams.set("interpolation", "nearest");
            else
                layerParams.set("interpolation", "bilinear");

            if (hasLayerAttr(layer, "align_corners"))
                layerParams.set("align_corners", getLayerAttr(layer, "align_corners").b());

            if (hasLayerAttr(layer, "half_pixel_centers"))
                layerParams.set("half_pixel_centers", getLayerAttr(layer, "half_pixel_centers").b());

            int id = dstNet.addLayer(name, "Resize", layerParams);
            layer_id[name] = id;

            connect(layer_id, dstNet, parsePin(layer.input(0)), id, 0);

            // Step back to add convolution
            if (type == "FusedResizeAndPadConv2D")
            {
                tensorflow::NodeDef conv = layer_;
                conv.clear_input();
                conv.add_input(name);
                conv.add_input(convWeights);
                conv.set_op("Conv2D");
                parseNode(conv);
            }
        }
        else if (type == "L2Normalize")
        {
            // op: "L2Normalize"
            // input: "input"
            // input: "reduction_indices" (axis)
            CV_CheckEQ(num_inputs, 2, "");
            Mat reductionIndices = getTensorContent(getConstBlob(layer, value_id, 1));
            CV_Assert(reductionIndices.type() == CV_32SC1);

            const int numAxes = reductionIndices.total();
            if (getDataLayout(name, data_layouts) == DATA_LAYOUT_NHWC)
                for (int i = 0; i < numAxes; ++i)
                    reductionIndices.at<int>(i) = toNCHW(reductionIndices.at<int>(i));

            cv::sort(reductionIndices, reductionIndices, SORT_ASCENDING);
            for (int i = 1; i < numAxes; ++i)
            {
                CV_Assert(reductionIndices.at<int>(i) == reductionIndices.at<int>(i - 1) + 1);
                // Axes have the same sign.
                CV_Assert(reductionIndices.at<int>(i) * reductionIndices.at<int>(i - 1) >= 0);
            }
            layerParams.set("start_axis", reductionIndices.at<int>(0));
            layerParams.set("end_axis", reductionIndices.at<int>(numAxes - 1));

            int id = dstNet.addLayer(name, "Normalize", layerParams);
            layer_id[name] = id;
            connect(layer_id, dstNet, parsePin(layer.input(0)), id, 0);
        }
        else if (type == "PriorBox")
        {
            CV_CheckEQ(num_inputs, 2, "");
            if (hasLayerAttr(layer, "min_size"))
                layerParams.set("min_size", getLayerAttr(layer, "min_size").i());
            if (hasLayerAttr(layer, "max_size"))
                layerParams.set("max_size", getLayerAttr(layer, "max_size").i());
            if (hasLayerAttr(layer, "flip"))
                layerParams.set("flip", getLayerAttr(layer, "flip").b());
            if (hasLayerAttr(layer, "clip"))
                layerParams.set("clip", getLayerAttr(layer, "clip").b());
            if (hasLayerAttr(layer, "offset"))
                layerParams.set("offset", getLayerAttr(layer, "offset").f());
            if (hasLayerAttr(layer, "step"))
                layerParams.set("step", getLayerAttr(layer, "step").f());

            const std::string paramNames[] = {"variance", "aspect_ratio", "scales",
                                              "width", "height"};
            for (int i = 0; i < 5; ++i)
            {
                if (hasLayerAttr(layer, paramNames[i]))
                {
                    Mat values = getTensorContent(getLayerAttr(layer, paramNames[i]).tensor());
                    layerParams.set(paramNames[i],
                                    DictValue::arrayReal<float*>((float*)values.data, values.total()));
                }
            }
            int id = dstNet.addLayer(name, "PriorBox", layerParams);
            layer_id[name] = id;
            connect(layer_id, dstNet, parsePin(layer.input(0)), id, 0);
            connect(layer_id, dstNet, parsePin(layer.input(1)), id, 1);
            data_layouts[name] = DATA_LAYOUT_UNKNOWN;
        }
        else if (type == "Softmax")
        {
            CV_CheckGT(num_inputs, 0, "");
            if (hasLayerAttr(layer, "axis"))
                layerParams.set("axis", getLayerAttr(layer, "axis").i());

            int id = dstNet.addLayer(name, "Softmax", layerParams);
            layer_id[name] = id;
            connectToAllBlobs(layer_id, dstNet, parsePin(layer.input(0)), id, num_inputs);
        }
        else if (type == "CropAndResize")
        {
            // op: "CropAndResize"
            // input: "input"
            // input: "boxes"
            // input: "sizes"
            CV_CheckEQ(num_inputs, 3, "");

            Mat cropSize = getTensorContent(getConstBlob(layer, value_id, 2));
            CV_CheckTypeEQ(cropSize.type(), CV_32SC1, ""); CV_CheckEQ(cropSize.total(), (size_t)2, "");

            layerParams.set("height", cropSize.at<int>(0));
            layerParams.set("width", cropSize.at<int>(1));

            int id = dstNet.addLayer(name, "CropAndResize", layerParams);
            layer_id[name] = id;

            connect(layer_id, dstNet, parsePin(layer.input(0)), id, 0);
            connect(layer_id, dstNet, parsePin(layer.input(1)), id, 1);
        }
        else if (type == "Mean" || type == "Sum")
        {
            // Computes the mean of elements across dimensions of a tensor.
            // If keepdims is false (default) reduces input_tensor along the dimensions given in axis,
            // else the reduced dimensions are retained with length 1.
            // if indices = [1, 2] in NHWC layout we use global pooling: NxCxHxW --Pooling--> NxCx1x1
            // if keepdims is false we use Flatten after Pooling: out_shape = NxC
            // if indices = [0] we use a global pooling by indices.
            // To return correct shape, we use Reshape after Pooling. To determine input shape use Slice for input,
            // if keepdims is false we use Flatten after Slice.
            // Example: input_shape = NxCxHxW
            // determine out shape: NxCxHxW --Slice--> 1xCxHxW
            //                      out_shape = 1xCxHxW if keepDims else (1xCxHxW --Flatten--> CxHxW)
            // global pool: NxCxHxW --Flatten--> Nx(C*H*W) --Reshape--> 1x1xNx(C*H*W) --Pooling--> 1x1x1x(C*H*W) --Reshape--> out_shape
            CV_CheckGT(num_inputs, 0, "");

            Mat indices = getTensorContent(getConstBlob(layer, value_id, 1));
            CV_Assert(indices.type() == CV_32SC1);

            // There are two attributes, "keepdims" and a deprecated "keep_dims".
            bool keepDims = false;
            if (hasLayerAttr(layer, "keepdims"))
                keepDims = getLayerAttr(layer, "keepdims").b();
            else if (hasLayerAttr(layer, "keep_dims"))
                keepDims = getLayerAttr(layer, "keep_dims").b();

            if (indices.total() == 1 && indices.at<int>(0) == 0)
            {
                LayerParams flattenLp;
                std::string flattenName = name + "/flatten";
                CV_Assert(layer_id.find(flattenName) == layer_id.end());
                int flattenId = dstNet.addLayer(flattenName, "Flatten", flattenLp);
                layer_id[flattenName] = flattenId;
                connect(layer_id, dstNet, parsePin(layer.input(0)), flattenId, 0);

                LayerParams reshapeLp;
                std::string reshapeName = name + "/reshape";
                CV_Assert(layer_id.find(reshapeName) == layer_id.end());
                reshapeLp.set("axis", 0);
                reshapeLp.set("num_axes", 1);
                int newShape[] = {1, 1, -1};
                reshapeLp.set("dim", DictValue::arrayInt(&newShape[0], 3));

                int reshapeId = dstNet.addLayer(reshapeName, "Reshape", reshapeLp);
                layer_id[reshapeName] = reshapeId;
                connect(layer_id, dstNet, Pin(flattenName), reshapeId, 0);

                LayerParams avgLp;
                std::string avgName = name + "/avg";
                CV_Assert(layer_id.find(avgName) == layer_id.end());
                avgLp.set("pool", type == "Mean" ? "ave" : "sum");
                // pooling kernel H x 1
                avgLp.set("global_pooling_h", true);
                avgLp.set("kernel_w", 1);
                int avgId = dstNet.addLayer(avgName, "Pooling", avgLp);
                layer_id[avgName] = avgId;
                connect(layer_id, dstNet, Pin(reshapeName), avgId, 0);

                LayerParams sliceLp;
                std::string layerShapeName = name + "/slice";
                CV_Assert(layer_id.find(layerShapeName) == layer_id.end());
                sliceLp.set("axis", 0);
                int begin[] = {0};
                int size[] = {1};
                sliceLp.set("begin", DictValue::arrayInt(&begin[0], 1));
                sliceLp.set("size", DictValue::arrayInt(&size[0], 1));
                int sliceId = dstNet.addLayer(layerShapeName, "Slice", sliceLp);
                layer_id[layerShapeName] = sliceId;
                connect(layer_id, dstNet, Pin(layer.input(0)), sliceId, 0);

                if (!keepDims)
                {
                    LayerParams squeezeLp;
                    std::string squeezeName = name + "/squeeze";
                    CV_Assert(layer_id.find(squeezeName) == layer_id.end());
                    squeezeLp.set("axis", 0);
                    squeezeLp.set("end_axis", 1);
                    int squeezeId = dstNet.addLayer(squeezeName, "Flatten", squeezeLp);
                    layer_id[squeezeName] = squeezeId;
                    connect(layer_id, dstNet, Pin(layerShapeName), squeezeId, 0);
                    layerShapeName = squeezeName;
                }

                int id = dstNet.addLayer(name, "Reshape", layerParams);
                layer_id[name] = id;
                connect(layer_id, dstNet, Pin(avgName), id, 0);
                connect(layer_id, dstNet, Pin(layerShapeName), id, 1);
            } else if (indices.total() == 1) {
                int axis = toNCHW(indices.at<int>(0));
                if (axis == 2 || axis == 3)
                {
                    layerParams.set("pool", type == "Mean" ? "ave" : "sum");
                    layerParams.set(axis == 2 ? "kernel_w" : "kernel_h", 1);
                    layerParams.set(axis == 2 ? "global_pooling_h" : "global_pooling_w", true);
                    int id = dstNet.addLayer(name, "Pooling", layerParams);
                    layer_id[name] = id;
                    connect(layer_id, dstNet, parsePin(layer.input(0)), id, 0);

                    if (!keepDims)
                    {
                        // To keep correct order after squeeze dims we first need to change layout from NCHW to NHWC
                        LayerParams permLP;
                        int order[] = {0, 2, 3, 1};  // From OpenCV's NCHW to NHWC.
                        permLP.set("order", DictValue::arrayInt<int*>(order, 4));
                        std::string permName = name + "/nchw";
                        CV_Assert(layer_id.find(permName) == layer_id.end());
                        int permId = dstNet.addLayer(permName, "Permute", permLP);
                        layer_id[permName] = permId;
                        connect(layer_id, dstNet, Pin(name), permId, 0);

                        LayerParams squeezeLp;
                        std::string squeezeName = name + "/squeeze";
                        CV_Assert(layer_id.find(squeezeName) == layer_id.end());
                        squeezeLp.set("axis", indices.at<int>(0));
                        squeezeLp.set("end_axis", indices.at<int>(0) + 1);
                        int squeezeId = dstNet.addLayer(squeezeName, "Flatten", squeezeLp);
                        layer_id[squeezeName] = squeezeId;
                        connect(layer_id, dstNet, Pin(permName), squeezeId, 0);
                    }
                }
            } else {
                if (indices.total() != 2 || indices.at<int>(0) != 1 || indices.at<int>(1) != 2)
                    CV_Error(Error::StsNotImplemented, "Unsupported mode of reduce_mean or reduce_sum operation.");

                layerParams.set("pool", type == "Mean" ? "ave" : "sum");
                layerParams.set("global_pooling", true);
                int id = dstNet.addLayer(name, "Pooling", layerParams);
                layer_id[name] = id;
                connect(layer_id, dstNet, parsePin(layer.input(0)), id, 0);

                if (!keepDims)
                {
                    LayerParams flattenLp;
                    std::string flattenName = name + "/flatten";
                    CV_Assert(layer_id.find(flattenName) == layer_id.end());
                    int flattenId = dstNet.addLayer(flattenName, "Flatten", flattenLp);
                    layer_id[flattenName] = flattenId;
                    connect(layer_id, dstNet, Pin(name), flattenId, 0);
                }
            }
        }
        else if (type == "Pack")
        {
            // op: tf.stack(list of tensors, axis=0)
            // Join a list of inputs along a new axis.
            // The "axis" specifies the index of the new axis in the dimensions of the output.
            // Example: given a list with "N" tensors of shape (C, H, W):
            // if axis == 0 then the output tensor will have the shape (N, C, H, W),
            // if axis == 1 then the output tensor will have the shape (C, N, H, W).
            CV_CheckGT(num_inputs, 0, "");
            CV_Assert(hasLayerAttr(layer, "axis"));
            int dim = (int)getLayerAttr(layer, "axis").i();
            if (dim != 0)
                CV_Error(Error::StsNotImplemented, "Unsupported mode of pack operation.");

            CV_Assert(hasLayerAttr(layer, "N"));
            int num = (int)getLayerAttr(layer, "N").i();
            CV_CheckEQ(num_inputs, num, "");
            std::string base_name = name + "/reshape_";
            std::vector<int> reshape_ids;
            for (int i = 0; i < num; i++) {
                std::ostringstream ss;
                ss << i;
                std::string reshape_name = base_name + ss.str();
                LayerParams reshapeLP;
                reshapeLP.set("axis", dim);
                reshapeLP.set("num_axes", 1);
                int outShape[] = {1, -1};
                reshapeLP.set("dim", DictValue::arrayInt(&outShape[0], 2));
                int id = dstNet.addLayer(reshape_name, "Reshape", reshapeLP);
                layer_id[reshape_name] = id;
                reshape_ids.push_back(id);
                connect(layer_id, dstNet, parsePin(layer.input(i)), id, 0);
            }

            layerParams.set("axis", dim);
            int id = dstNet.addLayer(name, "Concat", layerParams);
            layer_id[name] = id;

            for (int li = 0; li < num; li++)
                dstNet.connect(reshape_ids[li], 0, id, li);
        }
        else if (type == "ClipByValue")
        {
            // op: "ClipByValue"
            // input: "input"
            // input: "mix"
            // input: "max"
            CV_CheckEQ(num_inputs, 3, "");

            Mat minValue = getTensorContent(getConstBlob(layer, value_id, 1));
            Mat maxValue = getTensorContent(getConstBlob(layer, value_id, 2));
            CV_CheckEQ(minValue.total(), (size_t)1, ""); CV_CheckTypeEQ(minValue.type(), CV_32FC1, "");
            CV_CheckEQ(maxValue.total(), (size_t)1, ""); CV_CheckTypeEQ(maxValue.type(), CV_32FC1, "");

            layerParams.set("min_value", minValue.at<float>(0));
            layerParams.set("max_value", maxValue.at<float>(0));

            int id = dstNet.addLayer(name, "ReLU6", layerParams);
            layer_id[name] = id;

            connect(layer_id, dstNet, parsePin(layer.input(0)), id, 0);
        }
        else if (type == "LeakyRelu")
        {
            CV_CheckGT(num_inputs, 0, "");
            CV_Assert(hasLayerAttr(layer, "alpha"));
            layerParams.set("negative_slope", getLayerAttr(layer, "alpha").f());

            int id = dstNet.addLayer(name, "ReLU", layerParams);
            layer_id[name] = id;
            connectToAllBlobs(layer_id, dstNet, parsePin(layer.input(0)), id, num_inputs);
        }
        else if (type == "Abs" || type == "Tanh" || type == "Sigmoid" ||
                 type == "Relu" || type == "Elu" || type == "Exp" ||
                 type == "Identity" || type == "Relu6")
        {
            CV_CheckGT(num_inputs, 0, "");
            std::string dnnType = type;
            if (type == "Abs") dnnType = "AbsVal";
            else if (type == "Tanh") dnnType = "TanH";
            else if (type == "Relu") dnnType = "ReLU";
            else if (type == "Relu6") dnnType = "ReLU6";
            else if (type == "Elu") dnnType = "ELU";

            int id = dstNet.addLayer(name, dnnType, layerParams);
            layer_id[name] = id;
            connectToAllBlobs(layer_id, dstNet, parsePin(layer.input(0)), id, num_inputs);
        }
        else
        {
            // Importer does not know how to map this TensorFlow's operation onto OpenCV's layer.
            // However we create a layer with the same type and rely that user defined a custom layer.

            // All the attributes are added to LayerParams.
            google::protobuf::Map<std::string, tensorflow::AttrValue> attr = layer.attr();
            for (google::protobuf::Map<std::string, tensorflow::AttrValue>::const_iterator ai = attr.begin();
                 ai != attr.end(); ++ai)
            {
                if (ai->second.value_case() == tensorflow::AttrValue::kS)  // string
                    layerParams.set(ai->first, ai->second.s());
                if (ai->second.value_case() == tensorflow::AttrValue::kI)  // int64
                    layerParams.set(ai->first, ai->second.i());
                if (ai->second.value_case() == tensorflow::AttrValue::kF)  // float
                    layerParams.set(ai->first, ai->second.f());
                if (ai->second.value_case() == tensorflow::AttrValue::kB)  // bool
                    layerParams.set(ai->first, ai->second.b());
            }

            // All the Const input nodes are added to layer's blobs.
            std::vector<std::string> inputsNames;
            for (int i = 0; i < num_inputs; ++i)
            {
                // Check if input is a Const node.
                if (value_id.find(layer.input(i)) != value_id.end())
                {
                    Mat blob = getTensorContent(getConstBlob(layer, value_id, i));
                    layerParams.blobs.push_back(blob);
                }
                else
                    inputsNames.push_back(layer.input(i));
            }
            int id = dstNet.addLayer(name, type, layerParams);
            layer_id[name] = id;

            for (int i = 0; i < inputsNames.size(); ++i)
            {
                connect(layer_id, dstNet, parsePin(inputsNames[i]), id, i);
            }
        }
    }
    catch (const std::exception& e)
    {
        CV_LOG_ERROR(NULL, "DNN/TF: Can't parse layer for node='" << name << "'. Exception: " << e.what());
        throw;
    }
}

} // namespace

#endif //HAVE_PROTOBUF

Net readNetFromTensorflow(const String &model, const String &config)
{
    Net net;
    TFImporter importer(net, model.c_str(), config.c_str());
    return net;
}

Net readNetFromTensorflow(const char* bufferModel, size_t lenModel,
                          const char* bufferConfig, size_t lenConfig)
{
    Net net;
    TFImporter importer(net, bufferModel, lenModel, bufferConfig, lenConfig);
    return net;
}

Net readNetFromTensorflow(const std::vector<uchar>& bufferModel, const std::vector<uchar>& bufferConfig)
{
    const char* bufferModelPtr = reinterpret_cast<const char*>(&bufferModel[0]);
    const char* bufferConfigPtr = bufferConfig.empty() ? NULL :
                                  reinterpret_cast<const char*>(&bufferConfig[0]);
    return readNetFromTensorflow(bufferModelPtr, bufferModel.size(),
                                 bufferConfigPtr, bufferConfig.size());
}

void writeTextGraph(const String& _model, const String& output)
{
    String model = _model;
    const std::string modelExt = model.substr(model.rfind('.') + 1);
    if (modelExt != "pb")
        CV_Error(Error::StsNotImplemented, "Only TensorFlow models support export to text file");

    tensorflow::GraphDef net;
    ReadTFNetParamsFromBinaryFileOrDie(model.c_str(), &net);

    sortByExecutionOrder(net);

    RepeatedPtrField<tensorflow::NodeDef>::iterator it;
    for (it = net.mutable_node()->begin(); it != net.mutable_node()->end(); ++it)
    {
        if (it->op() == "Const")
        {
            it->mutable_attr()->at("value").mutable_tensor()->clear_tensor_content();
        }
    }

    std::string content;
    google::protobuf::TextFormat::PrintToString(net, &content);

    std::ofstream ofs(output.c_str());
    ofs << content;
    ofs.close();
}

CV__DNN_INLINE_NS_END
}} // namespace
