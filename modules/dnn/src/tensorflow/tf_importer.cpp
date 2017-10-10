// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Implementation of Tensorflow models parser
*/

#include "../precomp.hpp"

#ifdef HAVE_PROTOBUF
#include "graph.pb.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include "tf_io.hpp"
#endif

namespace cv {
namespace dnn {
CV__DNN_EXPERIMENTAL_NS_BEGIN

#if HAVE_PROTOBUF

using ::google::protobuf::RepeatedField;
using ::google::protobuf::RepeatedPtrField;
using ::google::protobuf::Message;
using ::google::protobuf::Descriptor;
using ::google::protobuf::FieldDescriptor;
using ::google::protobuf::Reflection;

namespace
{

static int toNCHW[] = {0, 2, 3, 1};

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
            shape.resize(1, 1);  // Scalar.
    }
    else
    {
        CV_Error(Error::StsError, "Unknown shape of input tensor");
    }
}

static Mat getTensorContent(const tensorflow::TensorProto &tensor)
{
    std::string content = tensor.tensor_content();
    switch (tensor.dtype())
    {
        case tensorflow::DT_FLOAT:
        {
            if (!content.empty())
                return Mat(1, content.size() / sizeof(float), CV_32FC1, (void*)content.c_str()).clone();
            else
            {
                const RepeatedField<float>& field = tensor.float_val();
                CV_Assert(!field.empty());
                return Mat(1, field.size(), CV_32FC1, (void*)field.data()).clone();
            }
        }
        case tensorflow::DT_DOUBLE:
        {
            if (!content.empty())
                return Mat(1, content.size() / sizeof(double), CV_64FC1, (void*)content.c_str()).clone();
            else
            {
                const RepeatedField<double>& field = tensor.double_val();
                CV_Assert(!field.empty());
                return Mat(1, field.size(), CV_64FC1, (void*)field.data()).clone();
            }
        }
        case tensorflow::DT_INT32:
        {
            if (!content.empty())
                return Mat(1, content.size() / sizeof(int32_t), CV_32SC1, (void*)content.c_str()).clone();
            else
            {
                const RepeatedField<int32_t>& field = tensor.int_val();
                CV_Assert(!field.empty());
                return Mat(1, field.size(), CV_32SC1, (void*)field.data()).clone();
            }
        }
        case tensorflow::DT_HALF:
        {
            Mat halfs;
            if (!content.empty())
            {
                static const int kHalfSize = 2;
                halfs = Mat(1, content.size() / kHalfSize, CV_16UC1, (void*)content.c_str());
            }
            else
            {
                const RepeatedField<int32_t>& field = tensor.half_val();
                CV_Assert(!field.empty());
                Mat ints(1, field.size(), CV_32SC1, (void*)field.data());
                ints.convertTo(halfs, CV_16UC1);
            }
            // Reinterpret as a signed shorts just for a convertFp16 call.
            Mat halfsSigned(halfs.size(), CV_16SC1, halfs.data);
            Mat floats(halfs.size(), CV_32FC1);
            convertFp16(halfsSigned, floats);
            return floats;
        }
        default:
            CV_Error(Error::StsError, "Tensor's data type is not supported");
            break;
    }
    return Mat();
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

    Mat tensorContent = getTensorContent(tensor);
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

bool hasLayerAttr(const tensorflow::NodeDef &layer, const std::string &name)
{
    google::protobuf::Map<std::string, tensorflow::AttrValue> attr = layer.attr();
    return attr.find(name) != attr.end();
}

const tensorflow::AttrValue& getLayerAttr(const tensorflow::NodeDef &layer, const std::string &name)
{
    return layer.attr().at(name);
}

void setStrides(LayerParams &layerParams, const tensorflow::NodeDef &layer)
{
    if (hasLayerAttr(layer, "strides"))
    {
        const tensorflow::AttrValue& val = getLayerAttr(layer, "strides");
        if (val.list().i_size() != 4 ||
            val.list().i(0) != 1 || val.list().i(3) != 1)
            CV_Error(Error::StsError, "Unsupported strides");
        layerParams.set("stride_h", static_cast<int>(val.list().i(1)));
        layerParams.set("stride_w", static_cast<int>(val.list().i(2)));
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
        if (val.list().i_size() != 4 ||
            val.list().i(0) != 1 || val.list().i(3) != 1)
            CV_Error(Error::StsError, "Unsupported ksize");
        layerParams.set("kernel_h", static_cast<int>(val.list().i(1)));
        layerParams.set("kernel_w", static_cast<int>(val.list().i(2)));
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

void RemoveIdentityOps(tensorflow::GraphDef& net) {
    typedef std::map<String, String>  IdentityOpsMap;
    IdentityOpsMap identity_ops;

    std::vector<int> identity_ops_idx;

    int layersCount = net.node_size();
    for (int li = 0; li < layersCount; li++)
    {
        const tensorflow::NodeDef &layer = net.node(li);
        String type = layer.op();

        if (type == "Identity" || type == "Dropout") {
            identity_ops_idx.push_back(li);
            identity_ops[layer.name()] = layer.input(0);
        }
    }

    for (int li = 0; li < layersCount; li++)
    {
        tensorflow::NodeDef* layer = net.mutable_node(li);
        for (int input_id = 0; input_id < layer->input_size(); input_id++) {
            String input_op_name = layer->input(input_id);
            IdentityOpsMap::iterator it = identity_ops.find(input_op_name);

            if (it != identity_ops.end()) {
                layer->set_input(input_id, it->second);
            }
        }
    }

    std::sort(identity_ops_idx.begin(), identity_ops_idx.end());

    int removed_nodes = 0;
    for(size_t i = 0; i < identity_ops_idx.size(); i++) {
        int start_id = identity_ops_idx[i] - removed_nodes;
        net.mutable_node()->DeleteSubrange(start_id, 1);
        removed_nodes++;
    }
}

Pin parsePin(const std::string &name)
{
    Pin pin(name);

    size_t delimiter_pos = name.find_first_of(":");
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

class TFImporter : public Importer {
public:
    TFImporter(const char *model, const char *config = NULL);
    void populateNet(Net dstNet);
    ~TFImporter() {}

private:
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
};

TFImporter::TFImporter(const char *model, const char *config)
{
    if (model && model[0])
        ReadTFNetParamsFromBinaryFileOrDie(model, &netBin);
    if (config && config[0])
        ReadTFNetParamsFromTextFileOrDie(config, &netTxt);
}

void TFImporter::kernelFromTensor(const tensorflow::TensorProto &tensor, Mat &dstBlob)
{
    MatShape shape;
    blobShapeFromTensor(tensor, shape);
    int dims = (int)shape.size();

    // TODO: other blob types
    CV_Assert(tensor.dtype() == tensorflow::DT_FLOAT ||
              tensor.dtype() == tensorflow::DT_HALF);
    CV_Assert(dims == 4);

    // REORDER kernel HWIO to OIHW
    swap(shape[0], shape[2]); // IWHO
    swap(shape[1], shape[3]); // IOHW
    swap(shape[0], shape[1]); // OIHW

    dstBlob.create(shape, CV_32F);

    Mat tensorContent = getTensorContent(tensor);
    int size = tensorContent.total();
    CV_Assert(size == (int)dstBlob.total());

    float *dstData = dstBlob.ptr<float>();
    const float *data = reinterpret_cast<const float*>(tensorContent.data);

    int out_c = shape[0], input_c = shape[1], height = shape[2], width = shape[3];
    int total = out_c*input_c*height*width;
    for(int i_oc = 0; i_oc < out_c; i_oc++) {
        for(int i_ic = 0; i_ic < input_c; i_ic++) {
            for(int i_h = 0; i_h < height; i_h++) {
                for(int i_w = 0; i_w < width; i_w++) {
                    int dst_i = input_c*height*width*i_oc + height*width*i_ic + width*i_h + i_w;
                    int src_i = out_c*input_c*width*i_h + out_c*input_c*i_w + out_c*i_ic + i_oc;
                    CV_Assert(dst_i < total);
                    CV_Assert(src_i < total);
                   dstData[dst_i] = data[src_i];
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
    network.connect(it->second, outPin.blobIndex, input_layer_id, input_blob_id);
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
        CV_Error(Error::StsError, "Const kernel input not found");
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
        CV_Assert(nodeIdx < netTxt.node_size(),
                  netTxt.node(nodeIdx).name() == kernel_inp.name);
        return netTxt.node(nodeIdx).attr().at("value").tensor();
    }
}

static void addConstNodes(const tensorflow::GraphDef& net, std::map<String, int>& const_layers,
                          std::set<String>& layers_to_ignore)
{
    for (int li = 0; li < net.node_size(); li++)
    {
        const tensorflow::NodeDef &layer = net.node(li);
        String name = layer.name();
        String type = layer.op();

        if (type != "Const")
            continue;  // only Const parameters are supported

        if (layer.attr().find("value") != layer.attr().end())
        {
            CV_Assert(const_layers.insert(std::make_pair(name, li)).second);
        }
        layers_to_ignore.insert(name);
    }
}

void TFImporter::populateNet(Net dstNet)
{
    RemoveIdentityOps(netBin);
    RemoveIdentityOps(netTxt);

    std::set<String> layers_to_ignore;

    tensorflow::GraphDef& net = netTxt.ByteSize() != 0 ? netTxt : netBin;

    int layersSize = net.node_size();

    // find all Const layers for params
    std::map<String, int> value_id;
    addConstNodes(netBin, value_id, layers_to_ignore);
    addConstNodes(netTxt, value_id, layers_to_ignore);

    std::map<String, int> layer_id;

    for (int li = 0; li < layersSize; li++)
    {
        tensorflow::NodeDef layer = net.node(li);
        String name = layer.name();
        String type = layer.op();
        LayerParams layerParams;

        if(layers_to_ignore.find(name) != layers_to_ignore.end())
            continue;

        if (type == "Conv2D" || type == "SpaceToBatchND" || type == "DepthwiseConv2dNative")
        {
            // The first node of dilated convolution subgraph.
            // Extract input node, dilation rate and paddings.
            std::string input = layer.input(0);
            if (type == "SpaceToBatchND")
            {
                // op: "SpaceToBatchND"
                // input: "input"
                // input: "SpaceToBatchND/block_shape"
                // input: "SpaceToBatchND/paddings"
                CV_Assert(layer.input_size() == 3);

                DictValue dilation = parseDims(getConstBlob(layer, value_id, 1));
                CV_Assert(dilation.size() == 2 && dilation.get<int>(0) == dilation.get<int>(1));
                layerParams.set("dilation", dilation.get<int>(0));

                Mat paddings;
                parseTensor<int>(getConstBlob(layer, value_id, 2), paddings);

                // paddings is a 2x2 matrix: [[top, bot], [left, right]]
                layerParams.set("pad_h", paddings.at<float>(0));
                layerParams.set("pad_w", paddings.at<float>(2));

                StrIntVector next_layers = getNextLayers(net, name, "Conv2D");
                CV_Assert(next_layers.size() == 1);
                layer = net.node(next_layers[0].second);
                layers_to_ignore.insert(next_layers[0].first);
                name = layer.name();
                type = layer.op();
            }

            layerParams.set("bias_term", false);
            layerParams.blobs.resize(1);

            StrIntVector next_layers = getNextLayers(net, name, "BiasAdd");
            if (next_layers.size() == 1) {
                layerParams.set("bias_term", true);
                layerParams.blobs.resize(2);

                int weights_layer_index = next_layers[0].second;

                blobFromTensor(getConstBlob(net.node(weights_layer_index), value_id), layerParams.blobs[1]);
                ExcludeLayer(net, weights_layer_index, 0, false);
                layers_to_ignore.insert(next_layers[0].first);
            }

            kernelFromTensor(getConstBlob(layer, value_id), layerParams.blobs[0]);
            int* kshape = layerParams.blobs[0].size.p;
            if (type == "DepthwiseConv2dNative")
            {
                const int chMultiplier = kshape[0];
                const int inCh = kshape[1];
                const int height = kshape[2];
                const int width = kshape[3];

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
                kshape[0] = inCh * chMultiplier;
                kshape[1] = 1;
            }
            layerParams.set("kernel_h", kshape[2]);
            layerParams.set("kernel_w", kshape[3]);
            layerParams.set("num_output", kshape[0]);

            setStrides(layerParams, layer);
            setPadding(layerParams, layer);

            // The final node of dilated convolution subgraph.
            next_layers = getNextLayers(net, name, "BatchToSpaceND");
            if (!next_layers.empty())
            {
                layerParams.set("pad_mode", "");  // We use padding values.
                CV_Assert(next_layers.size() == 1);
                ExcludeLayer(net, next_layers[0].second, 0, false);
                layers_to_ignore.insert(next_layers[0].first);
            }

            int id = dstNet.addLayer(name, "Convolution", layerParams);
            layer_id[name] = id;

            // one input only
            connect(layer_id, dstNet, parsePin(input), id, 0);
        }
        else if (type == "BiasAdd" || type == "Add")
        {
            bool haveConst = false;
            for(int ii = 0; !haveConst && ii < layer.input_size(); ++ii)
            {
                Pin input = parsePin(layer.input(ii));
                haveConst = value_id.find(input.name) != value_id.end();
            }
            CV_Assert(!haveConst || layer.input_size() == 2);

            if (haveConst)
            {
                layerParams.blobs.resize(1);
                blobFromTensor(getConstBlob(layer, value_id), layerParams.blobs[0]);

                int id = dstNet.addLayer(name, "Shift", layerParams);
                layer_id[name] = id;

                // one input only
                connect(layer_id, dstNet, parsePin(layer.input(0)), id, 0);
            }
            else
            {
                layerParams.set("operation", "sum");
                int id = dstNet.addLayer(name, "Eltwise", layerParams);
                layer_id[name] = id;

                for (int ii = 0; ii < layer.input_size(); ii++)
                {
                    Pin inp = parsePin(layer.input(ii));
                    if (layer_id.find(inp.name) == layer_id.end())
                        CV_Error(Error::StsError, "Input layer not found: " + inp.name);
                    dstNet.connect(layer_id.at(inp.name), inp.blobIndex, id, ii);
                }
            }
        }
        else if (type == "MatMul")
        {
            CV_Assert(layer.input_size() == 2);

            layerParams.set("bias_term", false);
            layerParams.blobs.resize(1);

            StrIntVector next_layers = getNextLayers(net, name, "BiasAdd");
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
            }

            int kernel_blob_index = -1;
            blobFromTensor(getConstBlob(layer, value_id, -1, &kernel_blob_index), layerParams.blobs[0]);

            if (kernel_blob_index == 1) { // In this case output is computed by x*W formula - W should be transposed
                Mat data = layerParams.blobs[0].t();
                layerParams.blobs[0] = data.clone();
            }

            layerParams.set("num_output", layerParams.blobs[0].size[0]);

            int id = dstNet.addLayer(name, "InnerProduct", layerParams);
            layer_id[name] = id;

            // one input only
            int input_blob_index = kernel_blob_index == 0 ? 1 : 0;
            connect(layer_id, dstNet, parsePin(layer.input(input_blob_index)), id, 0);
        }
        else if (type == "Reshape")
        {
            layerParams.set("dim", parseDims(getConstBlob(layer, value_id, 1)));

            int id = dstNet.addLayer(name, "Reshape", layerParams);
            layer_id[name] = id;

            // one input only
            connect(layer_id, dstNet, parsePin(layer.input(0)), id, 0);
        }
        else if (type == "Flatten")
        {
            int id = dstNet.addLayer(name, "Flatten", layerParams);
            layer_id[name] = id;
            connect(layer_id, dstNet, parsePin(layer.input(0)), id, 0);
        }
        else if (type == "Transpose")
        {
            Mat perm = getTensorContent(getConstBlob(layer, value_id, 1));
            CV_Assert(perm.type() == CV_32SC1);
            int* permData = (int*)perm.data;
            if (perm.total() == 4)
            {
                for (int i = 0; i < 4; ++i)
                    permData[i] = toNCHW[permData[i]];
            }
            layerParams.set("order", DictValue::arrayInt<int*>(permData, perm.total()));

            int id = dstNet.addLayer(name, "Permute", layerParams);
            layer_id[name] = id;

            // one input only
            connect(layer_id, dstNet, parsePin(layer.input(0)), id, 0);
        }
        else if (type == "Const")
        {
        }
        else if (type == "LRN")
        {
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

            connectToAllBlobs(layer_id, dstNet, parsePin(layer.input(0)), id, layer.input_size());
        }
        else if (type == "Concat" || type == "ConcatV2")
        {
            int axisId = (type == "Concat" ? 0 : layer.input_size() - 1);
            int axis = getConstBlob(layer, value_id, axisId).int_val().Get(0);
            layerParams.set("axis", 0 <= axis && axis < 4 ? toNCHW[axis] : axis);

            int id = dstNet.addLayer(name, "Concat", layerParams);
            layer_id[name] = id;


            int from = (type == "Concat" ? 1 : 0);
            int to = (type == "Concat" ? layer.input_size() : layer.input_size() - 1);

            // input(0) or input(n-1) is concat_dim
            for (int ii = from; ii < to; ii++)
            {
                Pin inp = parsePin(layer.input(ii));
                if (layer_id.find(inp.name) == layer_id.end())
                    CV_Error(Error::StsError, "Input layer not found: " + inp.name);
                dstNet.connect(layer_id.at(inp.name), inp.blobIndex, id, ii - from);
            }
        }
        else if (type == "MaxPool")
        {
            layerParams.set("pool", "max");

            setKSize(layerParams, layer);
            setStrides(layerParams, layer);
            setPadding(layerParams, layer);

            int id = dstNet.addLayer(name, "Pooling", layerParams);
            layer_id[name] = id;

            connectToAllBlobs(layer_id, dstNet, parsePin(layer.input(0)), id, layer.input_size());
        }
        else if (type == "AvgPool")
        {
            layerParams.set("pool", "ave");

            setKSize(layerParams, layer);
            setStrides(layerParams, layer);
            setPadding(layerParams, layer);

            int id = dstNet.addLayer(name, "Pooling", layerParams);
            layer_id[name] = id;

            connectToAllBlobs(layer_id, dstNet, parsePin(layer.input(0)), id, layer.input_size());
        }
        else if (type == "Placeholder")
        {
            std::vector<String> netInputs(1);
            netInputs[0] = name;
            layer_id[name] = 0;
            dstNet.setInputsNames(netInputs);
        }
        else if (type == "Split") {
            // TODO: determing axis index remapping by input dimensions order of input blob
            // TODO: slicing input may be Const op
            // TODO: slicing kernels for convolutions - in current implenmentation it is impossible
            // TODO: add parsing num of slices parameter
            CV_Assert(layer.input_size() == 2);
            // num_split
            // 1st blob is dims tensor
            int axis = getConstBlob(layer, value_id, 0).int_val().Get(0);
            layerParams.set("axis", toNCHW[axis]);

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
            CV_Assert(layer.input_size() == 3);

            const tensorflow::TensorProto begins = getConstBlob(layer, value_id, 1);
            const tensorflow::TensorProto sizes = getConstBlob(layer, value_id, 2);
            std::string beginsData = begins.tensor_content();
            std::string sizesData = sizes.tensor_content();
            CV_Assert(begins.dtype() == tensorflow::DT_INT32);
            CV_Assert(sizes.dtype() == tensorflow::DT_INT32);
            CV_Assert(!beginsData.empty());
            CV_Assert(!sizesData.empty());
            CV_Assert(beginsData.size() == sizesData.size());

            layerParams.set("begin", DictValue::arrayInt((int*)beginsData.c_str(),
                                                         beginsData.size() / 4));
            layerParams.set("size", DictValue::arrayInt((int*)sizesData.c_str(),
                                                        sizesData.size() / 4));

            int id = dstNet.addLayer(name, "Slice", layerParams);
            layer_id[name] = id;

            connect(layer_id, dstNet, parsePin(layer.input(0)), id, 0);
        }
        else if (type == "Mul")
        {
            bool haveConst = false;
            for(int ii = 0; !haveConst && ii < layer.input_size(); ++ii)
            {
                Pin input = parsePin(layer.input(ii));
                haveConst = value_id.find(input.name) != value_id.end();
            }
            CV_Assert(!haveConst || layer.input_size() == 2);

            if (haveConst)
            {
                // Multiplication by constant.
                CV_Assert(layer.input_size() == 2);
                Mat scaleMat = getTensorContent(getConstBlob(layer, value_id));
                CV_Assert(scaleMat.type() == CV_32FC1);

                int id;
                if (scaleMat.total() == 1)  // is a scalar.
                {
                    layerParams.set("scale", scaleMat.at<float>(0));
                    id = dstNet.addLayer(name, "Power", layerParams);
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
                layerParams.set("operation", "prod");
                int id = dstNet.addLayer(name, "Eltwise", layerParams);
                layer_id[name] = id;

                for (int ii = 0; ii < layer.input_size(); ii++)
                {
                    Pin inp = parsePin(layer.input(ii));
                    if (layer_id.find(inp.name) == layer_id.end())
                        CV_Error(Error::StsError, "Input layer not found: " + inp.name);
                    dstNet.connect(layer_id.at(inp.name), inp.blobIndex, id, ii);
                }
            }
        }
        else if (type == "Pad")
        {
            Mat paddings = getTensorContent(getConstBlob(layer, value_id, 1));
            CV_Assert(paddings.type() == CV_32SC1);
            if (paddings.total() == 8)
            {
                // Perhabs, we have NHWC padding dimensions order.
                //  N    H    W    C
                // 0 1  2 3  4 5  6 7
                std::swap(*paddings.ptr<int32_t>(0, 2), *paddings.ptr<int32_t>(0, 6));
                std::swap(*paddings.ptr<int32_t>(0, 3), *paddings.ptr<int32_t>(0, 7));
                //  N    C    W    H
                // 0 1  2 3  4 5  6 7
                std::swap(*paddings.ptr<int32_t>(0, 4), *paddings.ptr<int32_t>(0, 6));
                std::swap(*paddings.ptr<int32_t>(0, 5), *paddings.ptr<int32_t>(0, 7));
                //  N    C    H    W
                // 0 1  2 3  4 5  6 7
            }
            layerParams.set("paddings", DictValue::arrayInt<int*>((int*)paddings.data, paddings.total()));

            int id = dstNet.addLayer(name, "Padding", layerParams);
            layer_id[name] = id;

            connect(layer_id, dstNet, parsePin(layer.input(0)), id, 0);
        }
        else if (type == "FusedBatchNorm")
        {
            // op: "FusedBatchNorm"
            // input: "input"
            // input: "BatchNorm/gamma"
            // input: "BatchNorm/beta"
            // input: "BatchNorm/moving_mean"
            // input: "BatchNorm/moving_variance"
            if (layer.input_size() != 5)
                CV_Error(Error::StsNotImplemented,
                         "Expected gamma, beta, mean and std");

            layerParams.blobs.resize(4);
            // gamma
            blobFromTensor(getConstBlob(layer, value_id, 1), layerParams.blobs[2]);
            // beta
            blobFromTensor(getConstBlob(layer, value_id, 2), layerParams.blobs[3]);
            // mean
            blobFromTensor(getConstBlob(layer, value_id, 3), layerParams.blobs[0]);
            // std
            blobFromTensor(getConstBlob(layer, value_id, 4), layerParams.blobs[1]);

            if (hasLayerAttr(layer, "epsilon"))
                layerParams.set("eps", getLayerAttr(layer, "epsilon").f());

            layerParams.set("has_weight", true);
            layerParams.set("has_bias", true);

            int id = dstNet.addLayer(name, "BatchNorm", layerParams);
            layer_id[name] = id;

            // one input only
            connect(layer_id, dstNet, parsePin(layer.input(0)), id, 0);
        }
        else if (type == "Conv2DBackpropInput")
        {
            // op: "Conv2DBackpropInput"
            // input: "conv2d_transpose/output_shape"
            // input: "weights"
            // input: "input"
            if (layer.input_size() != 3)
                CV_Error(Error::StsNotImplemented,
                         "Expected output shape, weights and input nodes");

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
            layerParams.set("kernel_h", kshape[2]);
            layerParams.set("kernel_w", kshape[3]);
            layerParams.set("num_output", kshape[1]);

            setStrides(layerParams, layer);
            setPadding(layerParams, layer);

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
            if (layer.input_size() != 9)
                CV_Error(Error::StsNotImplemented, "Unexpected number of input nodes");

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
        }
        else if (type == "ResizeNearestNeighbor")
        {
            Mat outSize = getTensorContent(getConstBlob(layer, value_id, 1));
            CV_Assert(outSize.type() == CV_32SC1, outSize.total() == 2);

            layerParams.set("height", outSize.at<int>(0, 0));
            layerParams.set("width", outSize.at<int>(0, 1));

            if (hasLayerAttr(layer, "align_corners"))
                layerParams.set("align_corners", getLayerAttr(layer, "align_corners").b());

            int id = dstNet.addLayer(name, "ResizeNearestNeighbor", layerParams);
            layer_id[name] = id;

            connect(layer_id, dstNet, parsePin(layer.input(0)), id, 0);
        }
        else if (type == "PriorBox")
        {
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
            if (hasLayerAttr(layer, "variance"))
            {
                Mat variance = getTensorContent(getLayerAttr(layer, "variance").tensor());
                layerParams.set("variance",
                                DictValue::arrayReal<float*>((float*)variance.data, variance.total()));
            }
            if (hasLayerAttr(layer, "aspect_ratio"))
            {
                Mat aspectRatios = getTensorContent(getLayerAttr(layer, "aspect_ratio").tensor());
                layerParams.set("aspect_ratio",
                               DictValue::arrayReal<float*>((float*)aspectRatios.data, aspectRatios.total()));
            }
            if (hasLayerAttr(layer, "scales"))
            {
                Mat scales = getTensorContent(getLayerAttr(layer, "scales").tensor());
                layerParams.set("scales",
                               DictValue::arrayReal<float*>((float*)scales.data, scales.total()));
            }
            int id = dstNet.addLayer(name, "PriorBox", layerParams);
            layer_id[name] = id;
            connect(layer_id, dstNet, parsePin(layer.input(0)), id, 0);
            connect(layer_id, dstNet, parsePin(layer.input(1)), id, 1);
        }
        else if (type == "DetectionOutput")
        {
            // op: "DetectionOutput"
            // input_0: "locations"
            // input_1: "classifications"
            // input_2: "prior_boxes"
            if (hasLayerAttr(layer, "num_classes"))
                layerParams.set("num_classes", getLayerAttr(layer, "num_classes").i());
            if (hasLayerAttr(layer, "share_location"))
                layerParams.set("share_location", getLayerAttr(layer, "share_location").b());
            if (hasLayerAttr(layer, "background_label_id"))
                layerParams.set("background_label_id", getLayerAttr(layer, "background_label_id").i());
            if (hasLayerAttr(layer, "nms_threshold"))
                layerParams.set("nms_threshold", getLayerAttr(layer, "nms_threshold").f());
            if (hasLayerAttr(layer, "top_k"))
                layerParams.set("top_k", getLayerAttr(layer, "top_k").i());
            if (hasLayerAttr(layer, "code_type"))
                layerParams.set("code_type", getLayerAttr(layer, "code_type").s());
            if (hasLayerAttr(layer, "keep_top_k"))
                layerParams.set("keep_top_k", getLayerAttr(layer, "keep_top_k").i());
            if (hasLayerAttr(layer, "confidence_threshold"))
                layerParams.set("confidence_threshold", getLayerAttr(layer, "confidence_threshold").f());
            if (hasLayerAttr(layer, "loc_pred_transposed"))
                layerParams.set("loc_pred_transposed", getLayerAttr(layer, "loc_pred_transposed").b());

            int id = dstNet.addLayer(name, "DetectionOutput", layerParams);
            layer_id[name] = id;
            for (int i = 0; i < 3; ++i)
                connect(layer_id, dstNet, parsePin(layer.input(i)), id, i);
        }
        else if (type == "Abs" || type == "Tanh" || type == "Sigmoid" ||
                 type == "Relu" || type == "Elu" || type == "Softmax" ||
                 type == "Identity" || type == "Relu6")
        {
            std::string dnnType = type;
            if (type == "Abs") dnnType = "AbsVal";
            else if (type == "Tanh") dnnType = "TanH";
            else if (type == "Relu") dnnType = "ReLU";
            else if (type == "Relu6") dnnType = "ReLU6";
            else if (type == "Elu") dnnType = "ELU";

            int id = dstNet.addLayer(name, dnnType, layerParams);
            layer_id[name] = id;
            connectToAllBlobs(layer_id, dstNet, parsePin(layer.input(0)), id, layer.input_size());
        }
        else
        {
            printLayerAttr(layer);
            CV_Error_(Error::StsError, ("Unknown layer type %s in op %s", type.c_str(), name.c_str()));
        }
    }
}

} // namespace

Ptr<Importer> createTensorflowImporter(const String &model)
{
    return Ptr<Importer>(new TFImporter(model.c_str()));
}

#else //HAVE_PROTOBUF

Ptr<Importer> createTensorflowImporter(const String&)
{
    CV_Error(cv::Error::StsNotImplemented, "libprotobuf required to import data from TensorFlow models");
    return Ptr<Importer>();
}

#endif //HAVE_PROTOBUF

Net readNetFromTensorflow(const String &model, const String &config)
{
    TFImporter importer(model.c_str(), config.c_str());
    Net net;
    importer.populateNet(net);
    return net;
}

CV__DNN_EXPERIMENTAL_NS_END
}} // namespace
