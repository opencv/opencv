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
//                        (3-clause BSD License)
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// * Neither the names of the copyright holders nor the names of the contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/*M///////////////////////////////////////////////////////////////////////////////////////
//MIT License
//
//Copyright (c) 2017 Joseph Redmon
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.
//
//M*/

#include "../precomp.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

#include "darknet_io.hpp"

namespace cv {
    namespace dnn {
        namespace darknet {

            template<typename T>
            T getParam(const std::map<std::string, std::string> &params, const std::string param_name, T init_val)
            {
                std::map<std::string, std::string>::const_iterator it = params.find(param_name);
                if (it != params.end()) {
                    std::stringstream ss(it->second);
                    ss >> init_val;
                }
                return init_val;
            }

            class setLayersParams {

                NetParameter *net;
                int layer_id;
                std::string last_layer;
                std::vector<std::string> fused_layer_names;

            public:
                setLayersParams(NetParameter *_net, std::string _first_layer = "data") :
                    net(_net), layer_id(0), last_layer(_first_layer)
                {}

                void setLayerBlobs(int i, std::vector<cv::Mat> blobs)
                {
                    cv::dnn::LayerParams &params = net->layers[i].layerParams;
                    params.blobs = blobs;
                }

                cv::dnn::LayerParams getParamConvolution(int kernel, int pad,
                    int stride, int filters_num)
                {
                    cv::dnn::LayerParams params;
                    params.name = "Convolution-name";
                    params.type = "Convolution";

                    params.set<int>("kernel_size", kernel);
                    params.set<int>("pad", pad);
                    params.set<int>("stride", stride);

                    params.set<bool>("bias_term", false);	// true only if(BatchNorm == false)
                    params.set<int>("num_output", filters_num);

                    return params;
                }


                void setConvolution(int kernel, int pad, int stride,
                    int filters_num, int channels_num, int use_batch_normalize, int use_relu)
                {
                    cv::dnn::LayerParams conv_param =
                        getParamConvolution(kernel, pad, stride, filters_num);

                    darknet::LayerParameter lp;
                    std::string layer_name = cv::format("conv_%d", layer_id);

                    // use BIAS in any case
                    if (!use_batch_normalize) {
                        conv_param.set<bool>("bias_term", true);
                    }

                    lp.layer_name = layer_name;
                    lp.layer_type = conv_param.type;
                    lp.layerParams = conv_param;
                    lp.bottom_indexes.push_back(last_layer);
                    last_layer = layer_name;
                    net->layers.push_back(lp);

                    if (use_batch_normalize)
                    {
                        cv::dnn::LayerParams bn_param;

                        bn_param.name = "BatchNorm-name";
                        bn_param.type = "BatchNorm";
                        bn_param.set<bool>("has_weight", true);
                        bn_param.set<bool>("has_bias", true);
                        bn_param.set<float>("eps", 1E-6);	// .000001f in Darknet Yolo

                        darknet::LayerParameter lp;
                        std::string layer_name = cv::format("bn_%d", layer_id);
                        lp.layer_name = layer_name;
                        lp.layer_type = bn_param.type;
                        lp.layerParams = bn_param;
                        lp.bottom_indexes.push_back(last_layer);
                        last_layer = layer_name;
                        net->layers.push_back(lp);
                    }

                    if (use_relu)
                    {
                        cv::dnn::LayerParams activation_param;
                        activation_param.set<float>("negative_slope", 0.1f);
                        activation_param.name = "ReLU-name";
                        activation_param.type = "ReLU";

                        darknet::LayerParameter lp;
                        std::string layer_name = cv::format("relu_%d", layer_id);
                        lp.layer_name = layer_name;
                        lp.layer_type = activation_param.type;
                        lp.layerParams = activation_param;
                        lp.bottom_indexes.push_back(last_layer);
                        last_layer = layer_name;
                        net->layers.push_back(lp);
                    }

                    layer_id++;
                    fused_layer_names.push_back(last_layer);
                }

                void setMaxpool(size_t kernel, size_t pad, size_t stride)
                {
                    cv::dnn::LayerParams maxpool_param;
                    maxpool_param.set<cv::String>("pool", "max");
                    maxpool_param.set<int>("kernel_size", kernel);
                    maxpool_param.set<int>("pad", pad);
                    maxpool_param.set<int>("stride", stride);
                    maxpool_param.set<cv::String>("pad_mode", "SAME");
                    maxpool_param.name = "Pooling-name";
                    maxpool_param.type = "Pooling";
                    darknet::LayerParameter lp;

                    std::string layer_name = cv::format("pool_%d", layer_id);
                    lp.layer_name = layer_name;
                    lp.layer_type = maxpool_param.type;
                    lp.layerParams = maxpool_param;
                    lp.bottom_indexes.push_back(last_layer);
                    last_layer = layer_name;
                    net->layers.push_back(lp);
                    layer_id++;
                    fused_layer_names.push_back(last_layer);
                }

                void setConcat(int number_of_inputs, int *input_indexes)
                {
                    cv::dnn::LayerParams concat_param;
                    concat_param.name = "Concat-name";
                    concat_param.type = "Concat";
                    concat_param.set<int>("axis", 1);	// channels are in axis = 1

                    darknet::LayerParameter lp;

                    std::string layer_name = cv::format("concat_%d", layer_id);
                    lp.layer_name = layer_name;
                    lp.layer_type = concat_param.type;
                    lp.layerParams = concat_param;
                    for (int i = 0; i < number_of_inputs; ++i)
                        lp.bottom_indexes.push_back(fused_layer_names.at(input_indexes[i]));

                    last_layer = layer_name;
                    net->layers.push_back(lp);

                    layer_id++;
                    fused_layer_names.push_back(last_layer);
                }

                void setIdentity(int bottom_index)
                {
                    cv::dnn::LayerParams identity_param;
                    identity_param.name = "Identity-name";
                    identity_param.type = "Identity";

                    darknet::LayerParameter lp;

                    std::string layer_name = cv::format("identity_%d", layer_id);
                    lp.layer_name = layer_name;
                    lp.layer_type = identity_param.type;
                    lp.layerParams = identity_param;
                    lp.bottom_indexes.push_back(fused_layer_names.at(bottom_index));

                    last_layer = layer_name;
                    net->layers.push_back(lp);

                    layer_id++;
                    fused_layer_names.push_back(last_layer);
                }

                void setReorg(int stride)
                {
                    cv::dnn::LayerParams reorg_params;
                    reorg_params.name = "Reorg-name";
                    reorg_params.type = "Reorg";
                    reorg_params.set<int>("reorg_stride", stride);

                    darknet::LayerParameter lp;
                    std::string layer_name = cv::format("reorg_%d", layer_id);
                    lp.layer_name = layer_name;
                    lp.layer_type = reorg_params.type;
                    lp.layerParams = reorg_params;
                    lp.bottom_indexes.push_back(last_layer);
                    last_layer = layer_name;

                    net->layers.push_back(lp);

                    layer_id++;
                    fused_layer_names.push_back(last_layer);
                }

                void setPermute()
                {
                    cv::dnn::LayerParams permute_params;
                    permute_params.name = "Permute-name";
                    permute_params.type = "Permute";
                    int permute[] = { 0, 2, 3, 1 };
                    cv::dnn::DictValue paramOrder = cv::dnn::DictValue::arrayInt(permute, 4);

                    permute_params.set("order", paramOrder);

                    darknet::LayerParameter lp;
                    std::string layer_name = cv::format("premute_%d", layer_id);
                    lp.layer_name = layer_name;
                    lp.layer_type = permute_params.type;
                    lp.layerParams = permute_params;
                    lp.bottom_indexes.push_back(last_layer);
                    last_layer = layer_name;
                    net->layers.push_back(lp);

                    layer_id++;
                    fused_layer_names.push_back(last_layer);
                }

                void setRegion(float thresh, int coords, int classes, int anchors, int classfix, int softmax, int softmax_tree, float *biasData)
                {
                    cv::dnn::LayerParams region_param;
                    region_param.name = "Region-name";
                    region_param.type = "Region";

                    region_param.set<float>("thresh", thresh);
                    region_param.set<int>("coords", coords);
                    region_param.set<int>("classes", classes);
                    region_param.set<int>("anchors", anchors);
                    region_param.set<int>("classfix", classfix);
                    region_param.set<bool>("softmax_tree", softmax_tree);
                    region_param.set<bool>("softmax", softmax);

                    cv::Mat biasData_mat = cv::Mat(1, anchors * 2, CV_32F, biasData).clone();
                    region_param.blobs.push_back(biasData_mat);

                    darknet::LayerParameter lp;
                    std::string layer_name = "detection_out";
                    lp.layer_name = layer_name;
                    lp.layer_type = region_param.type;
                    lp.layerParams = region_param;
                    lp.bottom_indexes.push_back(last_layer);
                    last_layer = layer_name;
                    net->layers.push_back(lp);

                    layer_id++;
                    fused_layer_names.push_back(last_layer);
                }
            };

            std::string escapeString(const std::string &src)
            {
                std::string dst;
                for (size_t i = 0; i < src.size(); ++i)
                    if (src[i] > ' ' && src[i] <= 'z')
                        dst += src[i];
                return dst;
            }

            template<typename T>
            std::vector<T> getNumbers(const std::string &src)
            {
                std::vector<T> dst;
                std::stringstream ss(src);

                for (std::string str; std::getline(ss, str, ',');) {
                    std::stringstream line(str);
                    T val;
                    line >> val;
                    dst.push_back(val);
                }
                return dst;
            }

            bool ReadDarknetFromCfgFile(const char *cfgFile, NetParameter *net)
            {
                std::ifstream ifile;
                ifile.open(cfgFile);
                if (ifile.is_open())
                {
                    bool read_net = false;
                    int layers_counter = -1;
                    for (std::string line; std::getline(ifile, line);) {
                        line = escapeString(line);
                        if (line.empty()) continue;
                        switch (line[0]) {
                        case '\0': break;
                        case '#': break;
                        case ';': break;
                        case '[':
                            if (line == "[net]") {
                                read_net = true;
                            }
                            else {
                                // read section
                                read_net = false;
                                ++layers_counter;
                                const size_t layer_type_size = line.find("]") - 1;
                                CV_Assert(layer_type_size < line.size());
                                std::string layer_type = line.substr(1, layer_type_size);
                                net->layers_cfg[layers_counter]["type"] = layer_type;
                            }
                            break;
                        default:
                            // read entry
                            const size_t separator_index = line.find('=');
                            CV_Assert(separator_index < line.size());
                            if (separator_index != std::string::npos) {
                                std::string name = line.substr(0, separator_index);
                                std::string value = line.substr(separator_index + 1, line.size() - (separator_index + 1));
                                name = escapeString(name);
                                value = escapeString(value);
                                if (name.empty() || value.empty()) continue;
                                if (read_net)
                                    net->net_cfg[name] = value;
                                else
                                    net->layers_cfg[layers_counter][name] = value;
                            }
                        }
                    }

                    std::string anchors = net->layers_cfg[net->layers_cfg.size() - 1]["anchors"];
                    std::vector<float> vec = getNumbers<float>(anchors);
                    std::map<std::string, std::string> &net_params = net->net_cfg;
                    net->width = getParam(net_params, "width", 416);
                    net->height = getParam(net_params, "height", 416);
                    net->channels = getParam(net_params, "channels", 3);
                    CV_Assert(net->width > 0 && net->height > 0 && net->channels > 0);
                }
                else
                    return false;

                int current_channels = net->channels;
                net->out_channels_vec.resize(net->layers_cfg.size());

                int layers_counter = -1;

                setLayersParams setParams(net);

                typedef std::map<int, std::map<std::string, std::string> >::iterator it_type;
                for (it_type i = net->layers_cfg.begin(); i != net->layers_cfg.end(); ++i) {
                    ++layers_counter;
                    std::map<std::string, std::string> &layer_params = i->second;
                    std::string layer_type = layer_params["type"];

                    if (layer_type == "convolutional")
                    {
                        int kernel_size = getParam<int>(layer_params, "size", -1);
                        int pad = getParam<int>(layer_params, "pad", 0);
                        int stride = getParam<int>(layer_params, "stride", 1);
                        int filters = getParam<int>(layer_params, "filters", -1);
                        std::string activation = getParam<std::string>(layer_params, "activation", "linear");
                        bool batch_normalize = getParam<int>(layer_params, "batch_normalize", 0) == 1;
                        if(activation != "linear" && activation != "leaky")
                            CV_Error(cv::Error::StsParseError, "Unsupported activation: " + activation);
                        int flipped = getParam<int>(layer_params, "flipped", 0);
                        if (flipped == 1)
                            CV_Error(cv::Error::StsNotImplemented, "Transpose the convolutional weights is not implemented");

                        // correct the strange value of pad=1 for kernel_size=1 in the Darknet cfg-file
                        if (kernel_size < 3) pad = 0;

                        CV_Assert(kernel_size > 0 && filters > 0);
                        CV_Assert(current_channels > 0);

                        setParams.setConvolution(kernel_size, pad, stride, filters, current_channels,
                            batch_normalize, activation == "leaky");

                        current_channels = filters;
                    }
                    else if (layer_type == "maxpool")
                    {
                        int kernel_size = getParam<int>(layer_params, "size", 2);
                        int stride = getParam<int>(layer_params, "stride", 2);
                        int pad = getParam<int>(layer_params, "pad", 0);
                        setParams.setMaxpool(kernel_size, pad, stride);
                    }
                    else if (layer_type == "route")
                    {
                        std::string bottom_layers = getParam<std::string>(layer_params, "layers", "");
                        CV_Assert(!bottom_layers.empty());
                        std::vector<int> layers_vec = getNumbers<int>(bottom_layers);

                        current_channels = 0;
                        for (size_t k = 0; k < layers_vec.size(); ++k) {
                            layers_vec[k] += layers_counter;
                            current_channels += net->out_channels_vec[layers_vec[k]];
                        }

                        if (layers_vec.size() == 1)
                            setParams.setIdentity(layers_vec.at(0));
                        else
                            setParams.setConcat(layers_vec.size(), layers_vec.data());
                    }
                    else if (layer_type == "reorg")
                    {
                        int stride = getParam<int>(layer_params, "stride", 2);
                        current_channels = current_channels * (stride*stride);

                        setParams.setReorg(stride);
                    }
                    else if (layer_type == "region")
                    {
                        float thresh = getParam<float>(layer_params, "thresh", 0.001);
                        int coords = getParam<int>(layer_params, "coords", 4);
                        int classes = getParam<int>(layer_params, "classes", -1);
                        int num_of_anchors = getParam<int>(layer_params, "num", -1);
                        int classfix = getParam<int>(layer_params, "classfix", 0);
                        bool softmax = (getParam<int>(layer_params, "softmax", 0) == 1);
                        bool softmax_tree = (getParam<std::string>(layer_params, "tree", "").size() > 0);

                        std::string anchors_values = getParam<std::string>(layer_params, "anchors", std::string());
                        CV_Assert(!anchors_values.empty());
                        std::vector<float> anchors_vec = getNumbers<float>(anchors_values);

                        CV_Assert(classes > 0 && num_of_anchors > 0 && (num_of_anchors * 2) == anchors_vec.size());

                        setParams.setPermute();
                        setParams.setRegion(thresh, coords, classes, num_of_anchors, classfix, softmax, softmax_tree, anchors_vec.data());
                    }
                    else {
                        CV_Error(cv::Error::StsParseError, "Unknown layer type: " + layer_type);
                    }
                    net->out_channels_vec[layers_counter] = current_channels;
                }

                return true;
            }


            bool ReadDarknetFromWeightsFile(const char *darknetModel, NetParameter *net)
            {
                std::ifstream ifile;
                ifile.open(darknetModel, std::ios::binary);
                CV_Assert(ifile.is_open());

                int32_t major_ver, minor_ver, revision;
                ifile.read(reinterpret_cast<char *>(&major_ver), sizeof(int32_t));
                ifile.read(reinterpret_cast<char *>(&minor_ver), sizeof(int32_t));
                ifile.read(reinterpret_cast<char *>(&revision), sizeof(int32_t));

                uint64_t seen;
                if ((major_ver * 10 + minor_ver) >= 2) {
                    ifile.read(reinterpret_cast<char *>(&seen), sizeof(uint64_t));
                }
                else {
                    int32_t iseen = 0;
                    ifile.read(reinterpret_cast<char *>(&iseen), sizeof(int32_t));
                    seen = iseen;
                }
                bool transpose = (major_ver > 1000) || (minor_ver > 1000);
                if(transpose)
                    CV_Error(cv::Error::StsNotImplemented, "Transpose the weights (except for convolutional) is not implemented");

                int current_channels = net->channels;
                int cv_layers_counter = -1;
                int darknet_layers_counter = -1;

                setLayersParams setParams(net);

                typedef std::map<int, std::map<std::string, std::string> >::iterator it_type;
                for (it_type i = net->layers_cfg.begin(); i != net->layers_cfg.end(); ++i) {
                    ++darknet_layers_counter;
                    ++cv_layers_counter;
                    std::map<std::string, std::string> &layer_params = i->second;
                    std::string layer_type = layer_params["type"];

                    if (layer_type == "convolutional")
                    {
                        int kernel_size = getParam<int>(layer_params, "size", -1);
                        int filters = getParam<int>(layer_params, "filters", -1);
                        std::string activation = getParam<std::string>(layer_params, "activation", "linear");
                        bool use_batch_normalize = getParam<int>(layer_params, "batch_normalize", 0) == 1;

                        CV_Assert(kernel_size > 0 && filters > 0);
                        CV_Assert(current_channels > 0);

                        size_t const weights_size = filters * current_channels * kernel_size * kernel_size;
                        int sizes_weights[] = { filters, current_channels, kernel_size, kernel_size };
                        cv::Mat weightsBlob;
                        weightsBlob.create(4, sizes_weights, CV_32F);
                        CV_Assert(weightsBlob.isContinuous());

                        cv::Mat meanData_mat(1, filters, CV_32F);	// mean
                        cv::Mat stdData_mat(1, filters, CV_32F);	// variance
                        cv::Mat weightsData_mat(1, filters, CV_32F);// scale
                        cv::Mat biasData_mat(1, filters, CV_32F);	// bias

                        ifile.read(reinterpret_cast<char *>(biasData_mat.ptr<float>()), sizeof(float)*filters);
                        if (use_batch_normalize) {
                            ifile.read(reinterpret_cast<char *>(weightsData_mat.ptr<float>()), sizeof(float)*filters);
                            ifile.read(reinterpret_cast<char *>(meanData_mat.ptr<float>()), sizeof(float)*filters);
                            ifile.read(reinterpret_cast<char *>(stdData_mat.ptr<float>()), sizeof(float)*filters);
                        }
                        ifile.read(reinterpret_cast<char *>(weightsBlob.ptr<float>()), sizeof(float)*weights_size);

                        // set convolutional weights
                        std::vector<cv::Mat> conv_blobs;
                        conv_blobs.push_back(weightsBlob);
                        if (!use_batch_normalize) {
                            // use BIAS in any case
                            conv_blobs.push_back(biasData_mat);
                        }
                        setParams.setLayerBlobs(cv_layers_counter, conv_blobs);

                        // set batch normalize (mean, variance, scale, bias)
                        if (use_batch_normalize) {
                            ++cv_layers_counter;
                            std::vector<cv::Mat> bn_blobs;
                            bn_blobs.push_back(meanData_mat);
                            bn_blobs.push_back(stdData_mat);
                            bn_blobs.push_back(weightsData_mat);
                            bn_blobs.push_back(biasData_mat);
                            setParams.setLayerBlobs(cv_layers_counter, bn_blobs);
                        }

                        if(activation == "leaky")
                            ++cv_layers_counter;
                    }
                    current_channels = net->out_channels_vec[darknet_layers_counter];
                }
                return true;
            }

        }


        void ReadNetParamsFromCfgFileOrDie(const char *cfgFile, darknet::NetParameter *net)
        {
            if (!darknet::ReadDarknetFromCfgFile(cfgFile, net)) {
                CV_Error(cv::Error::StsParseError, "Failed to parse NetParameter file: " + std::string(cfgFile));
            }
        }

        void ReadNetParamsFromBinaryFileOrDie(const char *darknetModel, darknet::NetParameter *net)
        {
            if (!darknet::ReadDarknetFromWeightsFile(darknetModel, net)) {
                CV_Error(cv::Error::StsParseError, "Failed to parse NetParameter file: " + std::string(darknetModel));
            }
        }

    }
}
