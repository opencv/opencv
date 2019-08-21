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
#endif

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

#ifdef HAVE_PROTOBUF
using ::google::protobuf::RepeatedField;
using ::google::protobuf::RepeatedPtrField;
using ::google::protobuf::Message;
using ::google::protobuf::Descriptor;
using ::google::protobuf::FieldDescriptor;
using ::google::protobuf::Reflection;

namespace
{

template<typename T>
static cv::String toString(const T &v)
{
    std::ostringstream ss;
    ss << v;
    return ss.str();
}

class CaffeImporter
{
    caffe::NetParameter net;
    caffe::NetParameter netBinary;

public:

    CaffeImporter(const char *pototxt, const char *caffeModel)
    {
        CV_TRACE_FUNCTION();

        ReadNetParamsFromTextFileOrDie(pototxt, &net);

        if (caffeModel && caffeModel[0])
            ReadNetParamsFromBinaryFileOrDie(caffeModel, &netBinary);
    }

    CaffeImporter(const char *dataProto, size_t lenProto,
                  const char *dataModel, size_t lenModel)
    {
        CV_TRACE_FUNCTION();

        ReadNetParamsFromTextBufferOrDie(dataProto, lenProto, &net);

        if (dataModel != NULL && lenModel > 0)
            ReadNetParamsFromBinaryBufferOrDie(dataModel, lenModel, &netBinary);
    }

    void extractCustomParams(const google::protobuf::UnknownFieldSet& unknownFields, cv::dnn::LayerParams &params)
    {
        const int numFields = unknownFields.field_count();
        for (int i = 0; i < numFields; ++i)
        {
            const google::protobuf::UnknownField& field = unknownFields.field(i);
            CV_Assert(field.type() == google::protobuf::UnknownField::TYPE_GROUP);
            std::string fieldName = field.group().field(0).length_delimited();
            std::string fieldValue = field.group().field(1).length_delimited();
            params.set(fieldName, fieldValue);
        }
    }

    void addParam(const Message &msg, const FieldDescriptor *field, cv::dnn::LayerParams &params)
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

    inline static bool ends_with_param(const std::string &str)
    {
        static const std::string _param("_param");
        return (str.size() >= _param.size()) && str.compare(str.size() - _param.size(), _param.size(), _param) == 0;
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

            const google::protobuf::UnknownFieldSet& unknownFields = msgRefl->GetUnknownFields(msg);
            bool hasData =  fd->is_required() ||
                            (fd->is_optional() && msgRefl->HasField(msg, fd)) ||
                            (fd->is_repeated() && msgRefl->FieldSize(msg, fd) > 0) ||
                            !unknownFields.empty();
            if (!hasData)
                continue;

            extractCustomParams(unknownFields, params);
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

    void blobShapeFromProto(const caffe::BlobProto &pbBlob, MatShape& shape)
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

    void blobFromProto(const caffe::BlobProto &pbBlob, cv::Mat &dstBlob)
    {
        MatShape shape;
        blobShapeFromProto(pbBlob, shape);

        dstBlob.create((int)shape.size(), &shape[0], CV_32F);
        if (pbBlob.data_size())
        {
            // Single precision floats.
            CV_Assert(pbBlob.data_size() == (int)dstBlob.total());

            CV_DbgAssert(pbBlob.GetDescriptor()->FindFieldByLowercaseName("data")->cpp_type() == FieldDescriptor::CPPTYPE_FLOAT);
            Mat(dstBlob.dims, &dstBlob.size[0], CV_32F, (void*)pbBlob.data().data()).copyTo(dstBlob);
        }
        else
        {
            CV_Assert(pbBlob.has_raw_data());
            const std::string& raw_data = pbBlob.raw_data();
            if (pbBlob.raw_data_type() == caffe::FLOAT16)
            {
                // Half precision floats.
                CV_Assert(raw_data.size() / 2 == (int)dstBlob.total());

                Mat halfs((int)shape.size(), &shape[0], CV_16SC1, (void*)raw_data.c_str());
                convertFp16(halfs, dstBlob);
            }
            else if (pbBlob.raw_data_type() == caffe::FLOAT)
            {
                CV_Assert(raw_data.size() / 4 == (int)dstBlob.total());
                Mat((int)shape.size(), &shape[0], CV_32FC1, (void*)raw_data.c_str()).copyTo(dstBlob);
            }
            else
                CV_Error(Error::StsNotImplemented, "Unexpected blob data type");
        }
    }

    void extractBinaryLayerParams(const caffe::LayerParameter& layer, LayerParams& layerParams)
    {
        const std::string &name = layer.name();

        int li;
        for (li = 0; li != netBinary.layer_size(); li++)
        {
            const caffe::LayerParameter& binLayer = netBinary.layer(li);
            // Break if the layer name is the same and the blobs are not cleared
            if (binLayer.name() == name && binLayer.blobs_size() != 0)
                break;
        }

        if (li == netBinary.layer_size())
            return;

        caffe::LayerParameter* binLayer = netBinary.mutable_layer(li);
        const int numBlobs = binLayer->blobs_size();
        layerParams.blobs.resize(numBlobs);
        for (int bi = 0; bi < numBlobs; bi++)
        {
            blobFromProto(binLayer->blobs(bi), layerParams.blobs[bi]);
        }
        binLayer->clear_blobs();
        CV_Assert(numBlobs == binLayer->blobs().ClearedCount());
        for (int bi = 0; bi < numBlobs; bi++)
        {
            delete binLayer->mutable_blobs()->ReleaseCleared();
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
            extractBinaryLayerParams(layer, layerParams);

            int repetitions = layerCounter[name]++;
            if (repetitions)
                name += String("_") + toString(repetitions);

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
            else if (type == "BatchNorm")
            {
                if (!layerParams.get<bool>("use_global_stats", true))
                {
                    CV_Assert_N(layer.bottom_size() == 1, layer.top_size() == 1);

                    LayerParams mvnParams;
                    mvnParams.set("eps", layerParams.get<float>("eps", 1e-5));
                    std::string mvnName = name + "/mvn";

                    int repetitions = layerCounter[mvnName]++;
                    if (repetitions)
                        mvnName += String("_") + toString(repetitions);

                    int mvnId = dstNet.addLayer(mvnName, "MVN", mvnParams);
                    addInput(layer.bottom(0), mvnId, 0, dstNet);
                    addOutput(layer, mvnId, 0);
                    net.mutable_layer(li)->set_bottom(0, layer.top(0));
                    layerParams.blobs[0].setTo(0);  // mean
                    layerParams.blobs[1].setTo(1);  // std
                }
            }
            else if (type == "Axpy")
            {
                CV_Assert_N(layer.bottom_size() == 3, layer.top_size() == 1);

                std::string scaleName = name + "/scale";
                int repetitions = layerCounter[scaleName]++;
                if (repetitions) {
                    scaleName += String("_") + toString(repetitions);
                }

                LayerParams scaleParams;
                scaleParams.set("axis", 1);
                scaleParams.set("has_bias", false);
                int scaleId = dstNet.addLayer(scaleName, "Scale", scaleParams);
                addInput(layer.bottom(2), scaleId, 0, dstNet);
                addInput(layer.bottom(0), scaleId, 1, dstNet);
                addOutput(layer, scaleId, 0);
                net.mutable_layer(li)->set_bottom(0, layer.top(0));
                net.mutable_layer(li)->mutable_bottom()->RemoveLast();
                type = "Eltwise";
            }
            else if ("ConvolutionDepthwise" == type)
            {
                type = "Convolution";
            }

            int id = dstNet.addLayer(name, type, layerParams);

            for (int inNum = 0; inNum < layer.bottom_size(); inNum++)
                addInput(layer.bottom(inNum), id, inNum, dstNet);

            for (int outNum = 0; outNum < layer.top_size(); outNum++)
                addOutput(layer, id, outNum);
        }
        dstNet.setInputsNames(netInputs);

        std::vector<MatShape> inp_shapes;
        if (net.input_shape_size() > 0 || (layersSize > 0 && net.layer(0).has_input_param() &&
            net.layer(0).input_param().shape_size() > 0)) {

            int size = (net.input_shape_size() > 0) ? net.input_shape_size() :
                                                      net.layer(0).input_param().shape_size();
            for (int inp_id = 0; inp_id < size; inp_id++)
            {
                const caffe::BlobShape &_input_shape = (net.input_shape_size() > 0) ?
                                                        net.input_shape(inp_id) :
                                                        net.layer(0).input_param().shape(inp_id);
                MatShape shape;
                for (int i = 0; i < _input_shape.dim_size(); i++) {
                    shape.push_back((int)_input_shape.dim(i));
                }
                inp_shapes.push_back(shape);
            }
        }
        else if (net.input_dim_size() > 0) {
            MatShape shape;
            for (int dim = 0; dim < net.input_dim_size(); dim++) {
                shape.push_back(net.input_dim(dim));
            }
            inp_shapes.push_back(shape);
        }

        for (int inp_id = 0; inp_id < inp_shapes.size(); inp_id++) {
            dstNet.setInput(Mat(inp_shapes[inp_id], CV_32F), netInputs[inp_id]);
        }

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
};

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

Net readNetFromCaffe(const std::vector<uchar>& bufferProto, const std::vector<uchar>& bufferModel)
{
    const char* bufferProtoPtr = reinterpret_cast<const char*>(&bufferProto[0]);
    const char* bufferModelPtr = bufferModel.empty() ? NULL :
                                 reinterpret_cast<const char*>(&bufferModel[0]);
    return readNetFromCaffe(bufferProtoPtr, bufferProto.size(),
                            bufferModelPtr, bufferModel.size());
}

#endif //HAVE_PROTOBUF

CV__DNN_INLINE_NS_END
}} // namespace
