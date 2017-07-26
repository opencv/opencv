// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"

#ifndef HAVE_PROTOBUF
#include <typeinfo>

#include "caffe_proto.hpp"

namespace cv {
namespace dnn {
CV__DNN_EXPERIMENTAL_NS_BEGIN

namespace
{

using namespace pb;

class CaffeImporter : public Importer
{
public:

    CaffeImporter(const std::string& prototxt, const std::string& caffeModel)
        : parserBin(caffe_proto, sizeof(caffe_proto), ".caffe.NetParameter", true),
          parserTxt(caffe_proto, sizeof(caffe_proto), ".caffe.NetParameter", true),
          textOnly(caffeModel.empty())
    {
        CV_TRACE_FUNCTION();

        CV_Assert(!prototxt.empty());
        parserTxt.parse(prototxt, true);
        if (!textOnly)
            parserBin.parse(caffeModel);
    }

    virtual void populateNet(Net dstNet)
    {
        CV_TRACE_FUNCTION();

        // Setup input layer names.
        ProtobufNode inputNode;
        // Text file is a first priority for naming layers and their types.
        if (parserTxt.has("input"))
        {
            inputNode = parserTxt["input"];
        }
        else if (!textOnly && parserBin.has("input"))
        {
            inputNode = parserBin["input"];
        }

        std::vector<String> netInputs;
        for (int i = 0; i < inputNode.size(); ++i)
        {
            std::string inputName = (std::string)inputNode;
            addedBlobs[inputName] = LayerPin(0, i);
            netInputs.push_back(inputName);
        }

        // Parse layers.
        // Deprecated format stores layers in 'layers' field of message type
        // V1LayerParameter. Modern models keep them in 'layer' field of message
        // type LayerParameter. See caffe.proto for details.
        CV_Assert(parserTxt.has("layer") || parserTxt.has("layers"));

        ProtobufNode layersBin;
        ProtobufNode layersTxt = parserTxt.has("layer") ? parserTxt["layer"] :
                                                          parserTxt["layers"];
        if (!textOnly)
        {
            CV_Assert(parserBin.has("layers") || parserBin.has("layer"));
            layersBin = parserBin.has("layer") ? parserBin["layer"] :
                                                 parserBin["layers"];
        }
        // Add every layer that mentioned in .prototxt file.
        for (int i = 0, n = layersTxt.size(); i < n; ++i)
        {
            if (layersTxt[i].has("type") && (std::string)layersTxt[i]["type"] == "Input")
            {
                CV_Assert(layersTxt[i].has("name"));
                std::string inputName = layersTxt[i]["name"];

                addedBlobs[inputName] = LayerPin(0, netInputs.size());
                netInputs.push_back(inputName);
                continue;
            }

            addLayer(layersTxt[i], layersBin, dstNet);
        }
        dstNet.setInputsNames(netInputs);

        addedBlobs.clear();
    }

    static bool hasTestPhase(const ProtobufNode& node)
    {
        std::string phase;
        for (int i = 0, n = node["include"].size(); i < n; ++i)
        {
            if (node["include"][i].has("phase"))
            {
                node["include"][i]["phase"] >> phase;
                if (phase == "TEST")
                {
                    return true;
                }
            }
        }
        for (int i = 0, n = node["exclude"].size(); i < n; ++i)
        {
            if (node["exclude"][i].has("phase"))
            {
                node["exclude"][i]["phase"] >> phase;
                if (phase == "TEST")
                {
                    return false;
                }
            }
        }
        if (node.has("phase"))
        {
            return (std::string)node["phase"] == "TEST";
        }
        return true;
    }

    void addLayer(const ProtobufNode& nodeTxt, const ProtobufNode& layersBin,
                  Net& net)
    {
        CV_TRACE_FUNCTION();

        if (!hasTestPhase(nodeTxt))
        {
            return;
        }

        CV_Assert(nodeTxt.has("name") && nodeTxt.has("type"));

        std::string name = nodeTxt["name"];
        std::string type = nodeTxt["type"];

        LayerParams param;
        param.name = name;
        // Set type string before extractLayerParams to let change it inside.
        param.type = type;

        extractLayerParams(nodeTxt, param);
        extractWeights(name, layersBin, param);

        const int layerId = net.addLayer(name, param.type, param);
        addInput(nodeTxt, layerId, net);
        addOutput(nodeTxt, layerId);
    }

    // Connect new layer with it's inputs.
    void addInput(const ProtobufNode& layer, int layerId, Net& net)
    {
        CV_TRACE_FUNCTION();

        std::string bottomName;
        std::map<std::string, LayerPin>::iterator it;

        ProtobufNode bottom = layer["bottom"];
        for (int i = 0, n = bottom.size(); i < n; ++i)
        {
            bottom[i] >> bottomName;
            it = addedBlobs.find(bottomName);
            if (it != addedBlobs.end())
            {
                net.connect(it->second.first, it->second.second, layerId, i);
            }
            else
            {
                CV_Error(Error::StsParseError, "Cannot find blob " + bottomName);
            }
        }
    }

    // Add top blobs produced by new layer.
    void addOutput(const ProtobufNode& layer, int layerId)
    {
        CV_TRACE_FUNCTION();

        std::string topName;

        ProtobufNode top = layer["top"];
        for (int i = 0, n = top.size(); i < n; ++i)
        {
            top[i] >> topName;
            if (addedBlobs.find(topName) != addedBlobs.end())
            {
                ProtobufNode bottom = layer["bottom"];
                bool isInplace = bottom.size() == top.size() &&
                                 (std::string)bottom[i] == topName;
                if (!isInplace)
                    CV_Error(Error::StsBadArg, "Duplicate blobs produced by multiple sources");
            }
            addedBlobs[topName] = LayerPin(layerId, i);
        }
    }

    static void extractWeights(const std::string& name,
                               const ProtobufNode& layersBin, LayerParams& lp)
    {
        CV_TRACE_FUNCTION();

        // Find binary layer by name;
        ProtobufNode layer;
        for (int i = 0, n = layersBin.size(); i < n; ++i)
        {
            if (layersBin[i].has("name") &&
                name == (std::string)layersBin[i]["name"])
            {
                layer = layersBin[i];
                break;
            }
        }
        if (layer.empty())
        {
            return;
        }

        // Parse blobs.
        ProtobufNode blobs = layer["blobs"];
        for (int i = 0, n = blobs.size(); i < n; ++i)
        {
            ProtobufNode blob = blobs[i];
            std::vector<int> blobShape;

            if (blob.has("num") || blob.has("channels") || blob.has("height") ||
                blob.has("width"))
            {
                CV_Assert(blob.has("num") && blob.has("channels") &&
                          blob.has("height") && blob.has("width"));
                blobShape.resize(4);
                blob["num"] >> blobShape[0];
                blob["channels"] >> blobShape[1];
                blob["height"] >> blobShape[2];
                blob["width"] >> blobShape[3];
            }
            else
            {
                if (blob.has("shape"))
                {
                    ProtobufNode shape = blob["shape"]["dim"];
                    blobShape.resize(shape.size());
                    for (int i = 0; i < blobShape.size(); ++i)
                    {
                        blobShape[i] = (int64_t)shape[i];
                    }
                }
                else  // Is a scalar
                    blobShape.resize(1, 1);
            }

            cv::Mat weights(blobShape, CV_32F);

            ProtobufNode blobData = blob["data"];
            CV_Assert(blobData.size() == weights.total());
            blobData.copyTo(weights.total() * sizeof(float), weights.data);
            lp.blobs.push_back(weights);
        }
    }

    template <typename T>
    static void setParam(const std::string& name, const ProtobufNode& node,
                         LayerParams& lp)
    {
        lp.set(name, (T)node[name]);
    }

    template <typename T>
    static void setParamArray(const std::string& name,
                              const ProtobufNode& node, LayerParams& lp)
    {
        if (node.has(name))
        {
            std::vector<T> param(node[name].size());
            for (int i = 0; i < param.size(); ++i)
            {
                node[name][i] >> param[i];
            }
            DictValue dict;
            if (std::numeric_limits<T>::is_integer)
            {
                dict = DictValue::arrayInt<T*>(&param[0], param.size());
            }
            else if (std::numeric_limits<T>::is_specialized)
            {
                dict = DictValue::arrayReal<T*>(&param[0], param.size());
            }
            else
                CV_Error(Error::StsNotImplemented, "");
            lp.set(name, dict);
        }
    }

    static void extractLayerParams(const ProtobufNode& layerNode,
                                   LayerParams& lp)
    {
        CV_TRACE_FUNCTION();

        if (layerNode.has("batch_norm_param"))
        {
            setParam<float>("eps", layerNode["batch_norm_param"], lp);
            lp.type = "BatchNorm";
        }
        else if (layerNode.has("concat_param"))
        {
            setParam<int32_t>("axis", layerNode["concat_param"], lp);
            lp.type = "Concat";
        }
        else if (layerNode.has("convolution_param"))
        {
            ProtobufNode param = layerNode["convolution_param"];

            setParam<uint32_t>("num_output", param, lp);
            setParam<bool>("bias_term", param, lp);
            setParam<uint32_t>("group", param, lp);

            if (param.has("dilation"))
                setParam<uint32_t>("dilation", param, lp);

            if (param.has("pad"))
                setParam<uint32_t>("pad", param, lp);
            else
            {
                setParam<uint32_t>("pad_w", param, lp);
                setParam<uint32_t>("pad_h", param, lp);
            }

            if (param.has("kernel_size"))
                setParam<uint32_t>("kernel_size", param, lp);
            else
            {
                CV_Assert(param.has("kernel_h") && param.has("kernel_w"));
                setParam<uint32_t>("kernel_w", param, lp);
                setParam<uint32_t>("kernel_h", param, lp);
            }

            if (param.has("stride"))
                setParam<uint32_t>("stride", param, lp);
            else if (param.has("stride_h") && param.has("stride_w"))
            {
                setParam<uint32_t>("stride_w", param, lp);
                setParam<uint32_t>("stride_h", param, lp);
            }

            if (lp.type == "DECONVOLUTION" || lp.type == "Deconvolution")
                lp.type = "Deconvolution";
            else
                lp.type = "Convolution";
        }
        else if (layerNode.has("crop_param"))
        {
            setParam<int32_t>("axis", layerNode["crop_param"], lp);
            setParamArray<uint32_t>("offset", layerNode["crop_param"], lp);
            lp.type = "Crop";
        }
        else if (layerNode.has("data_param"))
        {
            lp.type = "Identity";
        }
        else if (layerNode.has("detection_output_param"))
        {
            ProtobufNode param = layerNode["detection_output_param"];

            if (param.has("num_classes"))
                setParam<uint32_t>("num_classes", param, lp);

            setParam<bool>("share_location", param, lp);
            setParam<int32_t>("background_label_id", param, lp);
            setParam<bool>("variance_encoded_in_target", param, lp);
            setParam<int32_t>("keep_top_k", param, lp);
            setParam<float>("confidence_threshold", param, lp);
            setParam<std::string>("code_type", param, lp);

            if (param.has("nms_param"))
            {
                ProtobufNode subparam = param["nms_param"];

                setParam<float>("nms_threshold", subparam, lp);
                setParam<float>("eta", subparam, lp);
                if (subparam.has("top_k"))
                    setParam<int32_t>("top_k", subparam, lp);
            }
            lp.type = "DetectionOutput";
        }
        else if (layerNode.has("dropout_param"))
        {
            lp.type = "Identity";
        }
        else if (layerNode.has("eltwise_param"))
        {
            setParam<std::string>("operation", layerNode["eltwise_param"], lp);
            setParamArray<float>("coeff", layerNode["eltwise_param"], lp);
            lp.type = "Eltwise";
        }
        else if (layerNode.has("flatten_param"))
        {
            setParam<int32_t>("axis", layerNode["flatten_param"], lp);
            setParam<int32_t>("end_axis", layerNode["flatten_param"], lp);
            lp.type = "Flatten";
        }
        else if (layerNode.has("inner_product_param"))
        {
            ProtobufNode param = layerNode["inner_product_param"];

            setParam<uint32_t>("num_output", param, lp);
            setParam<bool>("bias_term", param, lp);
            setParam<int32_t>("axis", param, lp);
            lp.type = "InnerProduct";
        }
        else if (layerNode.has("lrn_param"))
        {
            ProtobufNode param = layerNode["lrn_param"];

            setParam<uint32_t>("local_size", param, lp);
            setParam<float>("alpha", param, lp);
            setParam<float>("beta", param, lp);
            setParam<std::string>("norm_region", param, lp);
            lp.type = "LRN";
        }
        else if (layerNode.has("mvn_param"))
        {
            ProtobufNode param = layerNode["mvn_param"];
            setParam<bool>("normalize_variance", param, lp);
            setParam<bool>("across_channels", param, lp);
            setParam<float>("eps", param, lp);
            lp.type = "MVN";
        }
        else if (layerNode.has("norm_param"))
        {
            ProtobufNode param = layerNode["norm_param"];
            setParam<bool>("across_spatial", param, lp);
            setParam<bool>("channel_shared", param, lp);
            setParam<float>("eps", param, lp);
            lp.type = "NormalizeBBox";
        }
        else if (layerNode.has("permute_param"))
        {
            setParamArray<uint32_t>("order", layerNode["permute_param"], lp);
            lp.type = "Permute";
        }
        else if (layerNode.has("pooling_param"))
        {
            ProtobufNode param = layerNode["pooling_param"];

            if (param.has("pad_h") && param.has("pad_w"))
            {
                setParam<uint32_t>("pad_w", param, lp);
                setParam<uint32_t>("pad_h", param, lp);
            }
            else
                setParam<uint32_t>("pad", param, lp);

            if (param.has("stride_h") && param.has("stride_w"))
            {
                setParam<uint32_t>("stride_w", param, lp);
                setParam<uint32_t>("stride_h", param, lp);
            }
            else
                setParam<uint32_t>("stride", param, lp);

            if (param.has("kernel_h") && param.has("kernel_w"))
            {
                setParam<uint32_t>("kernel_w", param, lp);
                setParam<uint32_t>("kernel_h", param, lp);
            }
            else if (param.has("kernel_size"))
            {
                setParam<uint32_t>("kernel_size", param, lp);
            }

            setParam<std::string>("pool", param, lp);
            setParam<bool>("global_pooling", param, lp);
            if (param.has("ceil_mode"))
                setParam<bool>("ceil_mode", param, lp);
            lp.type = "Pooling";
        }
        else if (layerNode.has("power_param"))
        {
            ProtobufNode param = layerNode["power_param"];
            setParam<float>("power", param, lp);
            setParam<float>("scale", param, lp);
            setParam<float>("shift", param, lp);
            lp.type = "Power";
        }
        else if (layerNode.has("prior_box_param"))
        {
            ProtobufNode param = layerNode["prior_box_param"];
            if (param.has("min_size")) setParam<float>("min_size", param, lp);
            if (param.has("max_size")) setParam<float>("max_size", param, lp);
            setParamArray<float>("aspect_ratio", param, lp);
            setParam<bool>("flip", param, lp);
            setParam<bool>("clip", param, lp);
            setParamArray<float>("variance", param, lp);

            if (param.has("step"))
                setParam<float>("step", param, lp);
            else if (param.has("step_w") && param.has("step_h"))
            {
                setParam<float>("step_h", param, lp);
                setParam<float>("step_w", param, lp);
            }
            setParam<float>("offset", param, lp);
            lp.type = "PriorBox";
        }
        else if (layerNode.has("reshape_param"))
        {
            ProtobufNode param = layerNode["reshape_param"];

            setParam<int32_t>("axis", param, lp);
            setParam<int32_t>("num_axes", param, lp);
            setParamArray<int64_t>("dim", param["shape"], lp);
            lp.type = "Reshape";
        }
        else if (layerNode.has("scale_param"))
        {
            setParam<bool>("bias_term", layerNode["scale_param"], lp);
            lp.type = "Scale";
        }
        else if (layerNode.has("softmax_param") || lp.type == "SOFTMAX_LOSS")
        {
            if (layerNode.has("softmax_param"))
            {
                setParam<int32_t>("axis", layerNode["softmax_param"], lp);
            }
            lp.type = "Softmax";
        }
        else if (layerNode.has("slice_param"))
        {
            ProtobufNode param = layerNode["slice_param"];
            setParam<int32_t>("axis", param, lp);
            setParamArray<uint32_t>("slice_point", param, lp);
            lp.type = "Slice";
        }
        else if (layerNode.has("relu_param"))
        {
            setParamArray<float>("negative_slope", layerNode["relu_param"], lp);
            lp.type = "ReLU";
        }
    }

private:
    ProtobufParser parserBin;
    ProtobufParser parserTxt;
    bool textOnly;

    typedef std::pair<int, int> LayerPin;  // Pair layer id -- blob id.
    // Map "top" names from Caffe model to blob identifiers.
    std::map<std::string, LayerPin> addedBlobs;
};

}

Ptr<Importer> createCaffeImporter(const String &prototxt, const String &caffeModel)
{
    return Ptr<Importer>(new CaffeImporter(prototxt, caffeModel));
}

Net readNetFromCaffe(const String &prototxt, const String &caffeModel /*= String()*/)
{
    Ptr<Importer> caffeImporter = createCaffeImporter(prototxt, caffeModel);
    Net net;
    if (caffeImporter)
        caffeImporter->populateNet(net);
    return net;
}

CV__DNN_EXPERIMENTAL_NS_END
}} // namespace
#endif  // HAVE_PROTOBUF
