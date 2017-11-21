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

#ifdef HAVE_PROTOBUF
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include "caffe_io.hpp"
#else
#include "caffe_proto.hpp"
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

#else

namespace
{
class NodeWrapper : public pb::ProtobufNode
{
public:
    NodeWrapper(const pb::ProtobufNode& node) : pb::ProtobufNode(node) {}

    ProtobufNode get(const std::string& name) const
    {
        return operator[](name);
    }
};

template <typename T>
struct RepeatedField : public NodeWrapper
{
    RepeatedField(const pb::ProtobufNode& node) : NodeWrapper(node) {}

    Mat getMat(int dtype) const
    {
        Mat m(1, size(), dtype);
        copyTo(m.total() * m.elemSize1(), m.data);
        return m;
    }
};
}  // namespace

// Fake namespace to switch between Google's protobuf and OpenCV's one smoother.
namespace caffe
{

static const cv::String FLOAT16 = "FLOAT16";

class BlobShape
{
public:
    BlobShape(const pb::ProtobufNode& node) : pbNode(node) {}

    int dim_size() const { return pbNode["dim"].size(); }
    int dim(int idx) const { return (int)(int64_t)pbNode["dim"][idx]; }

private:
    pb::ProtobufNode pbNode;
};

class BlobProto
{
public:
    BlobProto(const pb::ProtobufNode& node) : pbNode(node) {}

    bool has_num() const { return pbNode.has("num"); }
    bool has_width() const { return pbNode.has("width"); }
    bool has_height() const { return pbNode.has("height"); }
    bool has_channels() const { return pbNode.has("channels"); }
    bool has_shape() const { return pbNode.has("shape"); }

    int num() const { return pbNode["num"]; }
    int width() const { return pbNode["width"]; }
    int height() const { return pbNode["height"]; }
    int channels() const { return pbNode["channels"]; }
    int data_size() const { return pbNode["data"].size(); }
    RepeatedField<float> data() const { return pbNode["data"]; }
    std::string raw_data_type() const { return pbNode["raw_data_type"]; }
    std::string raw_data() const { return pbNode["raw_data"]; }
    BlobShape shape() const { CV_Assert(has_shape()); return pbNode["shape"]; }

private:
    pb::ProtobufNode pbNode;
};

class LayerParameter : public NodeWrapper
{
public:
    LayerParameter(const pb::ProtobufNode& node) : NodeWrapper(node) {}

    int top_size() const { return get("top").size(); }
    int bottom_size() const { return get("bottom").size(); }
    int blobs_size() const { return get("blobs").size(); }
    std::string top(int idx) const { return get("top")[idx]; }
    std::string bottom(int idx) const { return get("bottom")[idx]; }
    BlobProto blobs(int idx) const { return get("blobs")[idx]; }

    std::string type() const
    {
        return has("type") ? (std::string)get("type") : "";
    }

    std::string name() const
    {
        CV_Assert(has("name"));
        return get("name");
    }
};

class NetParameter : public pb::ProtobufParser
{
public:
    NetParameter()
        : pb::ProtobufParser(caffe_proto, sizeof(caffe_proto), ".caffe.NetParameter", true) {}

    int layer_size() const
    {
        return has("layer") ? operator[]("layer").size() : operator[]("layers").size();
    }

    int input_size() const
    {
        return operator[]("input").size();
    }

    std::string input(int idx) const
    {
        return operator[]("input")[idx];
    }

    LayerParameter layer(int idx) const
    {
        return has("layer") ? operator[]("layer")[idx] : operator[]("layers")[idx];
    }
};

}  // namespace caffe
#endif  // HAVE_PROTOBUF

namespace
{

class CaffeImporter : public Importer
{
    caffe::NetParameter net;
    caffe::NetParameter netBinary;

public:

    CaffeImporter(const char *prototxt, const char *caffeModel)
    {
        CV_TRACE_FUNCTION();
#ifdef HAVE_PROTOBUF
        ReadNetParamsFromTextFileOrDie(prototxt, &net);

        if (caffeModel && caffeModel[0])
            ReadNetParamsFromBinaryFileOrDie(caffeModel, &netBinary);
#else
        net.parse(prototxt, true);
        if (caffeModel && caffeModel[0])
            netBinary.parse(caffeModel);
#endif  // HAVE_PROTOBUF
    }

    CaffeImporter(const char *dataProto, size_t lenProto,
                  const char *dataModel, size_t lenModel)
    {
        CV_TRACE_FUNCTION();

#ifdef HAVE_PROTOBUF
        ReadNetParamsFromTextBufferOrDie(dataProto, lenProto, &net);

        if (dataModel != NULL && lenModel > 0)
            ReadNetParamsFromBinaryBufferOrDie(dataModel, lenModel, &netBinary);
#else
        net.parse(dataProto, lenProto, true);
        if (dataModel != NULL && lenModel > 0)
            netBinary.parse(dataModel, lenModel);
#endif  // HAVE_PROTOBUF
    }

#ifdef HAVE_PROTOBUF
    static void addParam(const Message &msg, const FieldDescriptor *field, cv::dnn::LayerParams &params)
    {
        const Reflection *refl = msg.GetReflection();
        int type = field->cpp_type();
        bool isRepeated = field->is_repeated();
        const std::string &name = field->name();

        #define SET_UP_FILED(getter, arrayConstr, gtype)                                    \
            if (isRepeated) {                                                               \
                const RepeatedField<gtype> &v = refl->GetRepeatedField<gtype>(msg, field);  \
                params.set(name, DictValue::arrayConstr(v.begin(), (int)v.size()));                  \
            }                                                                               \
            else {                                                                          \
                params.set(name, refl->getter(msg, field));                               \
            }

        switch (type)
        {
        case FieldDescriptor::CPPTYPE_INT32:
            SET_UP_FILED(GetInt32, arrayInt, ::google::protobuf::int32);
            break;
        case FieldDescriptor::CPPTYPE_UINT32:
            SET_UP_FILED(GetUInt32, arrayInt, ::google::protobuf::uint32);
            break;
        case FieldDescriptor::CPPTYPE_INT64:
            SET_UP_FILED(GetInt32, arrayInt, ::google::protobuf::int64);
            break;
        case FieldDescriptor::CPPTYPE_UINT64:
            SET_UP_FILED(GetUInt32, arrayInt, ::google::protobuf::uint64);
            break;
        case FieldDescriptor::CPPTYPE_BOOL:
            SET_UP_FILED(GetBool, arrayInt, bool);
            break;
        case FieldDescriptor::CPPTYPE_DOUBLE:
            SET_UP_FILED(GetDouble, arrayReal, double);
            break;
        case FieldDescriptor::CPPTYPE_FLOAT:
            SET_UP_FILED(GetFloat, arrayReal, float);
            break;
        case FieldDescriptor::CPPTYPE_STRING:
            if (isRepeated) {
                const RepeatedPtrField<std::string> &v = refl->GetRepeatedPtrField<std::string>(msg, field);
                params.set(name, DictValue::arrayString(v.begin(), (int)v.size()));
            }
            else {
                params.set(name, refl->GetString(msg, field));
            }
            break;
        case FieldDescriptor::CPPTYPE_ENUM:
            if (isRepeated) {
                int size = refl->FieldSize(msg, field);
                std::vector<cv::String> buf(size);
                for (int i = 0; i < size; i++)
                    buf[i] = refl->GetRepeatedEnum(msg, field, i)->name();
                params.set(name, DictValue::arrayString(buf.begin(), size));
            }
            else {
                params.set(name, refl->GetEnum(msg, field)->name());
            }
            break;
        default:
            CV_Error(Error::StsError, "Unknown type \"" + String(field->type_name()) + "\" in prototxt");
            break;
        }
    }

    void extractLayerParams(const Message &msg, cv::dnn::LayerParams &params, bool isInternal = false)
    {
        const Descriptor *msgDesc = msg.GetDescriptor();
        const Reflection *msgRefl = msg.GetReflection();

        for (int fieldId = 0; fieldId < msgDesc->field_count(); fieldId++)
        {
            const FieldDescriptor *fd = msgDesc->field(fieldId);

            if (!isInternal && !ends_with_param(fd->name()))
                continue;

            bool hasData =  fd->is_required() ||
                            (fd->is_optional() && msgRefl->HasField(msg, fd)) ||
                            (fd->is_repeated() && msgRefl->FieldSize(msg, fd) > 0);
            if (!hasData)
                continue;

            if (fd->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE)
            {
                if (fd->is_repeated()) //Extract only first item!
                    extractLayerParams(msgRefl->GetRepeatedMessage(msg, fd, 0), params, true);
                else
                    extractLayerParams(msgRefl->GetMessage(msg, fd), params, true);
            }
            else
            {
                addParam(msg, fd, params);
            }
        }
    }

#else  // HAVE_PROTOBUF

    template <typename T>
    static void addParamValues(const pb::ProtobufNode& field, const std::string& name, cv::dnn::LayerParams &params)
    {
        const int numValues = field.size();

        std::vector<T> values(numValues);
        for (int i = 0; i < numValues; ++i)
        {
            field[i] >> values[i];
        }

        if (numValues == 1)
        {
            params.set(name, values[0]);
        }
        else
        {
            DictValue dict;
            if (std::numeric_limits<T>::is_integer)
            {
                dict = DictValue::arrayInt<T*>(&values[0], numValues);
            }
            else if (std::numeric_limits<T>::is_specialized)
            {
                dict = DictValue::arrayReal<T*>(&values[0], numValues);
            }
            else
                CV_Error(Error::StsNotImplemented, "");
            params.set(name, dict);
        }
    }

    static void addParam(const pb::ProtobufNode& msg, const std::string& paramName, cv::dnn::LayerParams &params)
    {
        CV_Assert(msg.type() == pb::PB_MESSAGE, msg.has(paramName));
        pb::ProtobufNode field = msg[paramName];

        switch (field.type())
        {
        case pb::PB_INT32:  addParamValues<int32_t>(field, paramName, params); break;
        case pb::PB_UINT32: addParamValues<uint32_t>(field, paramName, params); break;
        case pb::PB_INT64:  addParamValues<int64_t>(field, paramName, params); break;
        case pb::PB_UINT64:
        {
            CV_Assert(field.size() == 1);
            params.set(paramName, (int)(uint64_t)field);
            break;
        }
        case pb::PB_FLOAT:  addParamValues<float>(field, paramName, params); break;
        case pb::PB_DOUBLE: addParamValues<double>(field, paramName, params); break;
        case pb::PB_BOOL:
        {
            CV_Assert(field.size() == 1);
            params.set(paramName, (bool)field);
            break;
        }
        case pb::PB_STRING:
        {
            CV_Assert(field.size() == 1);
            params.set(paramName, (std::string)field);
            break;
        }
        case pb::PB_MESSAGE:
            CV_Error(Error::StsParseError, "Elementary type is expected got message");
            break;
        default:
            CV_Error(Error::StsParseError, format("Unknown type id [%d]", field.type()));
            break;
        }
    }

    void extractLayerParams(const pb::ProtobufNode& msg, cv::dnn::LayerParams &params, bool isInternal = false)
    {
        std::vector<std::string> readFields = msg.readFields();
        for (int i = 0; i < readFields.size(); ++i)
        {
            if (!isInternal && !ends_with_param(readFields[i]))
                continue;

            pb::ProtobufNode field = msg[readFields[i]];

            if (field.empty())
                continue;

            if (field.type() == pb::PB_MESSAGE)
            {
                extractLayerParams(field[0], params, true);
            }
            else
            {
                addParam(msg, readFields[i], params);
            }
        }
    }
#endif  // HAVE_PROTOBUF

    inline static bool ends_with_param(const std::string &str)
    {
        static const std::string _param("_param");
        return (str.size() >= _param.size()) && str.compare(str.size() - _param.size(), _param.size(), _param) == 0;
    }

    static void blobShapeFromProto(const caffe::BlobProto &pbBlob, MatShape& shape)
    {
        shape.clear();
        if (pbBlob.has_num() || pbBlob.has_channels() || pbBlob.has_height() || pbBlob.has_width())
        {
            shape.push_back(pbBlob.num());
            shape.push_back(pbBlob.channels());
            shape.push_back(pbBlob.height());
            shape.push_back(pbBlob.width());
        }
        else if (pbBlob.has_shape())
        {
            const caffe::BlobShape &_shape = pbBlob.shape();

            for (int i = 0; i < _shape.dim_size(); i++)
                shape.push_back((int)_shape.dim(i));
        }
        else
            shape.resize(1, 1);  // Is a scalar.
    }

    static void blobFromProto(const caffe::BlobProto &pbBlob, cv::Mat &dstBlob)
    {
        CV_TRACE_FUNCTION();

        MatShape shape;
        blobShapeFromProto(pbBlob, shape);

        dstBlob.create((int)shape.size(), &shape[0], CV_32F);
        float *dstData = dstBlob.ptr<float>();
        if (pbBlob.data_size())
        {
            // Single precision floats.
            CV_Assert(pbBlob.data_size() == (int)dstBlob.total());

#ifdef HAVE_PROTOBUF
            CV_DbgAssert(pbBlob.GetDescriptor()->FindFieldByLowercaseName("data")->cpp_type() == FieldDescriptor::CPPTYPE_FLOAT);
#endif

            const RepeatedField<float>& field = pbBlob.data();
            CV_Assert(!field.empty(), field.size() == dstBlob.total());
#ifdef HAVE_PROTOBUF
            Mat srcBlob(1, dstBlob.total(), CV_32FC1, (void*)field.data());
#else
            Mat srcBlob = field.getMat(CV_32FC1);
#endif  // HAVE_PROTOBUF
            CV_Assert(srcBlob.total() == dstBlob.total());
            memcpy(dstData, srcBlob.data, dstBlob.total() * dstBlob.elemSize1());
        }
        else
        {
            // Half precision floats.
            CV_Assert(pbBlob.raw_data_type() == caffe::FLOAT16);
            std::string raw_data = pbBlob.raw_data();

            CV_Assert(raw_data.size() / 2 == (int)dstBlob.total());

            Mat halfs((int)shape.size(), &shape[0], CV_16SC1, (void*)raw_data.c_str());
            convertFp16(halfs, dstBlob);
        }
    }

    void extractBinaryLayerParms(const caffe::LayerParameter& layer, LayerParams& layerParams)
    {
        const std::string &name = layer.name();

        int li;
        for (li = 0; li != netBinary.layer_size(); li++)
        {
            if (netBinary.layer(li).name() == name)
                break;
        }

        if (li == netBinary.layer_size() || netBinary.layer(li).blobs_size() == 0)
            return;

        const caffe::LayerParameter &binLayer = netBinary.layer(li);
        layerParams.blobs.resize(binLayer.blobs_size());
        for (int bi = 0; bi < binLayer.blobs_size(); bi++)
        {
            blobFromProto(binLayer.blobs(bi), layerParams.blobs[bi]);
        }
    }

    struct BlobNote
    {
        BlobNote(const std::string &_name, int _layerId, int _outNum) :
            name(_name), layerId(_layerId), outNum(_outNum) {}

        std::string name;
        int layerId, outNum;
    };

    std::vector<BlobNote> addedBlobs;
    std::map<String, int> layerCounter;

    void populateNet(Net dstNet)
    {
        CV_TRACE_FUNCTION();

        int layersSize = net.layer_size();
        layerCounter.clear();
        addedBlobs.clear();
        addedBlobs.reserve(layersSize + 1);

        //setup input layer names
        std::vector<String> netInputs(net.input_size());
        {
            for (int inNum = 0; inNum < net.input_size(); inNum++)
            {
                addedBlobs.push_back(BlobNote(net.input(inNum), 0, inNum));
                netInputs[inNum] = net.input(inNum);
            }
        }

        for (int li = 0; li < layersSize; li++)
        {
            const caffe::LayerParameter &layer = net.layer(li);
            String name = layer.name();
            String type = layer.type();
            LayerParams layerParams;

            extractLayerParams(layer, layerParams);
            extractBinaryLayerParms(layer, layerParams);

            int repetitions = layerCounter[name]++;
            if (repetitions)
                name += format("_%d", repetitions);

            if (type == "Input")
            {
                for (int outNum = 0; outNum < layer.top_size(); outNum++)
                {
                    addOutput(layer, 0, outNum);
                    addedBlobs.back().outNum = netInputs.size();
                    netInputs.push_back(addedBlobs.back().name);
                }
                continue;
            }

            int id = dstNet.addLayer(name, type, layerParams);

            for (int inNum = 0; inNum < layer.bottom_size(); inNum++)
                addInput(layer.bottom(inNum), id, inNum, dstNet);

            for (int outNum = 0; outNum < layer.top_size(); outNum++)
                addOutput(layer, id, outNum);
        }
        dstNet.setInputsNames(netInputs);

        addedBlobs.clear();
    }

    void addOutput(const caffe::LayerParameter &layer, int layerId, int outNum)
    {
        const std::string &name = layer.top(outNum);

        bool haveDups = false;
        for (int idx = (int)addedBlobs.size() - 1; idx >= 0; idx--)
        {
            if (addedBlobs[idx].name == name)
            {
                haveDups = true;
                break;
            }
        }

        if (haveDups)
        {
            bool isInplace = layer.bottom_size() > outNum && layer.bottom(outNum) == name;
            if (!isInplace)
                CV_Error(Error::StsBadArg, "Duplicate blobs produced by multiple sources");
        }

        addedBlobs.push_back(BlobNote(name, layerId, outNum));
    }

    void addInput(const std::string &name, int layerId, int inNum, Net &dstNet)
    {
        int idx;
        for (idx = (int)addedBlobs.size() - 1; idx >= 0; idx--)
        {
            if (addedBlobs[idx].name == name)
                break;
        }

        if (idx < 0)
        {
            CV_Error(Error::StsObjectNotFound, "Can't find output blob \"" + name + "\"");
            return;
        }

        dstNet.connect(addedBlobs[idx].layerId, addedBlobs[idx].outNum, layerId, inNum);
    }

    ~CaffeImporter()
    {

    }

};

}

Ptr<Importer> createCaffeImporter(const String &prototxt, const String &caffeModel)
{
    return Ptr<Importer>(new CaffeImporter(prototxt.c_str(), caffeModel.c_str()));
}

Net readNetFromCaffe(const String &prototxt, const String &caffeModel /*= String()*/)
{
    CaffeImporter caffeImporter(prototxt.c_str(), caffeModel.c_str());
    Net net;
    caffeImporter.populateNet(net);
    return net;
}

Net readNetFromCaffe(const char *bufferProto, size_t lenProto,
                     const char *bufferModel, size_t lenModel)
{
    CaffeImporter caffeImporter(bufferProto, lenProto, bufferModel, lenModel);
    Net net;
    caffeImporter.populateNet(net);
    return net;
}

CV__DNN_EXPERIMENTAL_NS_END
}} // namespace
