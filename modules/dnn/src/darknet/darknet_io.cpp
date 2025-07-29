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
#include <opencv2/dnn/shape_utils.hpp>

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

            static const std::string kFirstLayerName = "data";

            class setLayersParams {

                NetParameter *net;
                int layer_id;
                std::string last_layer;
                std::vector<std::string> fused_layer_names;

            public:
                setLayersParams(NetParameter *_net) :
                    net(_net), layer_id(0), last_layer(kFirstLayerName)
                {}

                void setLayerBlobs(int i, std::vector<cv::Mat> blobs)
                {
                    cv::dnn::LayerParams &params = net->layers[i].layerParams;
                    params.blobs = blobs;
                }

                void setBatchNorm()
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
                    int filters_num, int channels_num, int groups, int use_batch_normalize)
                {
                    cv::dnn::LayerParams conv_param =
                        getParamConvolution(kernel, pad, stride, filters_num);

                    darknet::LayerParameter lp;
                    std::string layer_name = cv::format("conv_%d", layer_id);

                    // use BIAS in any case
                    if (!use_batch_normalize) {
                        conv_param.set<bool>("bias_term", true);
                    }

                    conv_param.set<int>("group", groups);

                    lp.layer_name = layer_name;
                    lp.layer_type = conv_param.type;
                    lp.layerParams = conv_param;
                    lp.bottom_indexes.push_back(last_layer);
                    last_layer = layer_name;
                    net->layers.push_back(lp);

                    if (use_batch_normalize)
                        setBatchNorm();

                    layer_id++;
                    fused_layer_names.push_back(last_layer);
                }

                cv::dnn::LayerParams getParamFullyConnected(int output)
                {
                    cv::dnn::LayerParams params;
                    params.name = "FullyConnected-name";
                    params.type = "InnerProduct";

                    params.set<bool>("bias_term", false);	// true only if(BatchNorm == false)
                    params.set<int>("num_output", output);

                    return params;
                }

                void setFullyConnected(int output, int use_batch_normalize)
                {
                    cv::dnn::LayerParams fullyconnected_param =
                        getParamFullyConnected(output);

                    darknet::LayerParameter lp;
                    std::string layer_name = cv::format("fullyConnected_%d", layer_id);

                    // use BIAS in any case
                    if (!use_batch_normalize) {
                        fullyconnected_param.set<bool>("bias_term", true);
                    }

                    lp.layer_name = layer_name;
                    lp.layer_type = fullyconnected_param.type;
                    lp.layerParams = fullyconnected_param;
                    lp.bottom_indexes.push_back(last_layer);
                    last_layer = layer_name;
                    net->layers.push_back(lp);

                    if (use_batch_normalize)
                        setBatchNorm();

                    layer_id++;
                    fused_layer_names.push_back(last_layer);
                }

                void setActivation(String type)
                {
                    cv::dnn::LayerParams activation_param;
                    if (type == "relu")
                    {
                        activation_param.type = "ReLU";
                    }
                    else if (type == "leaky")
                    {
                        activation_param.set<float>("negative_slope", 0.1f);
                        activation_param.type = "ReLU";
                    }
                    else if (type == "swish" || type == "silu") // swish is an extension of silu.
                    {
                        activation_param.type = "Swish";
                    }
                    else if (type == "mish")
                    {
                        activation_param.type = "Mish";
                    }
                    else if (type == "logistic")
                    {
                        activation_param.type = "Sigmoid";
                    }
                    else if (type == "tanh")
                    {
                        activation_param.type = "TanH";
                    }
                    else
                    {
                        CV_Error(cv::Error::StsParseError, "Unsupported activation: " + type);
                    }

                    std::string layer_name = cv::format("%s_%d", type.c_str(), layer_id);

                    darknet::LayerParameter lp;
                    lp.layer_name = layer_name;
                    lp.layer_type = activation_param.type;
                    lp.layerParams = activation_param;
                    lp.bottom_indexes.push_back(last_layer);
                    last_layer = layer_name;
                    net->layers.push_back(lp);

                    fused_layer_names.back() = last_layer;
                }

                void setMaxpool(int kernel, int pad, int stride)
                {
                    cv::dnn::LayerParams maxpool_param;
                    maxpool_param.set<cv::String>("pool", "max");
                    maxpool_param.set<int>("kernel_size", kernel);
                    maxpool_param.set<int>("pad_l", floor((float)pad / 2));
                    maxpool_param.set<int>("pad_r", ceil((float)pad / 2));
                    maxpool_param.set<int>("pad_t", floor((float)pad / 2));
                    maxpool_param.set<int>("pad_b", ceil((float)pad / 2));
                    maxpool_param.set<bool>("ceil_mode", false);
                    maxpool_param.set<int>("stride", stride);
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

                void setAvgpool()
                {
                    cv::dnn::LayerParams avgpool_param;
                    avgpool_param.set<cv::String>("pool", "ave");
                    avgpool_param.set<bool>("global_pooling", true);
                    avgpool_param.name = "Pooling-name";
                    avgpool_param.type = "Pooling";
                    darknet::LayerParameter lp;

                    std::string layer_name = cv::format("avgpool_%d", layer_id);
                    lp.layer_name = layer_name;
                    lp.layer_type = avgpool_param.type;
                    lp.layerParams = avgpool_param;
                    lp.bottom_indexes.push_back(last_layer);
                    last_layer = layer_name;
                    net->layers.push_back(lp);
                    layer_id++;
                    fused_layer_names.push_back(last_layer);
                }

                void setCrop(int crop_height, int crop_width, int inp_height, int inp_width, bool noadjust)
                {
                    cv::dnn::LayerParams crop_param;
                    crop_param.name = "CropLayer-name";
                    std::vector<int> begin = {0, 0, (inp_height - crop_height) / 2, (inp_width - crop_width) / 2};
                    std::vector<int> sizes = {-1, -1, crop_height, crop_width};
                    crop_param.set("begin", DictValue::arrayInt(&begin[0], begin.size()));
                    crop_param.set("size", DictValue::arrayInt(&sizes[0], sizes.size()));
                    crop_param.type = "Slice";

                    darknet::LayerParameter lp;
                    std::string layer_name = cv::format("crop_%d", layer_id);
                    lp.layer_name = layer_name;
                    lp.layer_type = crop_param.type;
                    lp.layerParams = crop_param;
                    lp.bottom_indexes.push_back(last_layer);
                    last_layer = layer_name;
                    net->layers.push_back(lp);
                    layer_id++;

                    if (!noadjust)
                    {
                        cv::dnn::LayerParams params;
                        params.set("bias_term", true);
                        params.blobs = {
                            Mat(1, 1, CV_32F, Scalar(2)),
                            Mat(1, 1, CV_32F, Scalar(-1))
                        };

                        darknet::LayerParameter lp;
                        std::string layer_name = cv::format("adjust_crop_%d", layer_id);
                        lp.layer_name = layer_name;
                        lp.layer_type = "Scale";
                        lp.layerParams = params;
                        lp.bottom_indexes.push_back(last_layer);
                        last_layer = layer_name;
                        net->layers.push_back(lp);
                        layer_id++;
                    }
                    fused_layer_names.push_back(last_layer);
                }

                void setSoftmax()
                {
                    cv::dnn::LayerParams softmax_param;
                    softmax_param.name = "Softmax-name";
                    softmax_param.type = "Softmax";
                    // set default axis to 1
                    if(!softmax_param.has("axis"))
                        softmax_param.set("axis", 1);
                    darknet::LayerParameter lp;

                    std::string layer_name = cv::format("softmax_%d", layer_id);
                    lp.layer_name = layer_name;
                    lp.layer_type = softmax_param.type;
                    lp.layerParams = softmax_param;
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

                void setSlice(int input_index, int split_size, int group_id)
                {
                    int begin[] = {0, split_size * group_id, 0, 0};
                    cv::dnn::DictValue paramBegin = cv::dnn::DictValue::arrayInt(begin, 4);

                    int end[] = {INT_MAX, begin[1] + split_size, INT_MAX, INT_MAX};
                    cv::dnn::DictValue paramEnd = cv::dnn::DictValue::arrayInt(end, 4);

                    darknet::LayerParameter lp;
                    lp.layer_name = cv::format("slice_%d", layer_id);
                    lp.layer_type = "Slice";
                    lp.layerParams.set("begin", paramBegin);
                    lp.layerParams.set("end", paramEnd);

                    lp.bottom_indexes.push_back(fused_layer_names.at(input_index));
                    net->layers.push_back(lp);

                    layer_id++;
                    last_layer = lp.layer_name;
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

                void setPermute(bool isDarknetLayer = true)
                {
                    cv::dnn::LayerParams permute_params;
                    permute_params.name = "Permute-name";
                    permute_params.type = "Permute";
                    int permute[] = { 0, 2, 3, 1 };
                    cv::dnn::DictValue paramOrder = cv::dnn::DictValue::arrayInt(permute, 4);

                    permute_params.set("order", paramOrder);

                    darknet::LayerParameter lp;
                    std::string layer_name = cv::format("permute_%d", layer_id);
                    lp.layer_name = layer_name;
                    lp.layer_type = permute_params.type;
                    lp.layerParams = permute_params;
                    lp.bottom_indexes.push_back(last_layer);
                    last_layer = layer_name;
                    net->layers.push_back(lp);

                    if (isDarknetLayer)
                    {
                        layer_id++;
                        fused_layer_names.push_back(last_layer);
                    }
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

                void setYolo(int classes, const std::vector<int>& mask, const std::vector<float>& anchors, float thresh, float nms_threshold, float scale_x_y, int new_coords)
                {
                    cv::dnn::LayerParams region_param;
                    region_param.name = "Region-name";
                    region_param.type = "Region";

                    const int numAnchors = mask.size();

                    region_param.set<int>("classes", classes);
                    region_param.set<int>("anchors", numAnchors);
                    region_param.set<bool>("logistic", true);
                    region_param.set<float>("thresh", thresh);
                    region_param.set<float>("nms_threshold", nms_threshold);
                    region_param.set<float>("scale_x_y", scale_x_y);
                    region_param.set<int>("new_coords", new_coords);

                    std::vector<float> usedAnchors(numAnchors * 2);
                    for (int i = 0; i < numAnchors; ++i)
                    {
                        usedAnchors[i * 2] = anchors[mask[i] * 2];
                        usedAnchors[i * 2 + 1] = anchors[mask[i] * 2 + 1];
                    }

                    cv::Mat biasData_mat = cv::Mat(1, numAnchors * 2, CV_32F, &usedAnchors[0]).clone();
                    region_param.blobs.push_back(biasData_mat);

                    darknet::LayerParameter lp;
                    std::string layer_name = cv::format("yolo_%d", layer_id);
                    lp.layer_name = layer_name;
                    lp.layer_type = region_param.type;
                    lp.layerParams = region_param;
                    lp.bottom_indexes.push_back(last_layer);
                    lp.bottom_indexes.push_back(kFirstLayerName);
                    last_layer = layer_name;
                    net->layers.push_back(lp);

                    layer_id++;
                    fused_layer_names.push_back(last_layer);
                }

                void setShortcut(int from, float alpha)
                {
                    cv::dnn::LayerParams shortcut_param;
                    shortcut_param.name = "Shortcut-name";
                    shortcut_param.type = "Eltwise";

                    if (alpha != 1)
                    {
                        std::vector<float> coeffs(2, 1);
                        coeffs[0] = alpha;
                        shortcut_param.set("coeff", DictValue::arrayReal<float*>(&coeffs[0], coeffs.size()));
                    }

                    shortcut_param.set<std::string>("op", "sum");
                    shortcut_param.set<std::string>("output_channels_mode", "input_0_truncate");

                    darknet::LayerParameter lp;
                    std::string layer_name = cv::format("shortcut_%d", layer_id);
                    lp.layer_name = layer_name;
                    lp.layer_type = shortcut_param.type;
                    lp.layerParams = shortcut_param;
                    lp.bottom_indexes.push_back(last_layer);
                    lp.bottom_indexes.push_back(fused_layer_names.at(from));
                    last_layer = layer_name;
                    net->layers.push_back(lp);

                    layer_id++;
                    fused_layer_names.push_back(last_layer);
                }

                void setScaleChannels(int from)
                {
                    cv::dnn::LayerParams shortcut_param;
                    shortcut_param.type = "Scale";

                    darknet::LayerParameter lp;
                    std::string layer_name = cv::format("scale_channels_%d", layer_id);
                    lp.layer_name = layer_name;
                    lp.layer_type = shortcut_param.type;
                    lp.layerParams = shortcut_param;
                    lp.bottom_indexes.push_back(fused_layer_names.at(from));
                    lp.bottom_indexes.push_back(last_layer);
                    last_layer = layer_name;
                    net->layers.push_back(lp);

                    layer_id++;
                    fused_layer_names.push_back(last_layer);
                }

                void setSAM(int from)
                {
                    cv::dnn::LayerParams eltwise_param;
                    eltwise_param.name = "SAM-name";
                    eltwise_param.type = "Eltwise";

                    eltwise_param.set<std::string>("operation", "prod");
                    eltwise_param.set<std::string>("output_channels_mode", "same");

                    darknet::LayerParameter lp;
                    std::string layer_name = cv::format("sam_%d", layer_id);
                    lp.layer_name = layer_name;
                    lp.layer_type = eltwise_param.type;
                    lp.layerParams = eltwise_param;
                    lp.bottom_indexes.push_back(last_layer);
                    lp.bottom_indexes.push_back(fused_layer_names.at(from));
                    last_layer = layer_name;
                    net->layers.push_back(lp);

                    layer_id++;
                    fused_layer_names.push_back(last_layer);
                }

                void setUpsample(int scaleFactor)
                {
                    cv::dnn::LayerParams param;
                    param.name = "Upsample-name";
                    param.type = "Resize";

                    param.set<int>("zoom_factor", scaleFactor);
                    param.set<String>("interpolation", "nearest");

                    darknet::LayerParameter lp;
                    std::string layer_name = cv::format("upsample_%d", layer_id);
                    lp.layer_name = layer_name;
                    lp.layer_type = param.type;
                    lp.layerParams = param;
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

            bool ReadDarknetFromCfgStream(std::istream &ifile, NetParameter *net)
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
                            const size_t layer_type_size = line.find(']') - 1;
                            CV_Assert(layer_type_size < line.size());
                            std::string layer_type = line.substr(1, layer_type_size);
                            net->layers_cfg[layers_counter]["layer_type"] = layer_type;
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

                MatShape tensor_shape(3);
                tensor_shape[0] = net->channels;
                tensor_shape[1] = net->height;
                tensor_shape[2] = net->width;
                net->out_channels_vec.resize(net->layers_cfg.size());

                layers_counter = -1;

                setLayersParams setParams(net);

                typedef std::map<int, std::map<std::string, std::string> >::iterator it_type;
                for (it_type i = net->layers_cfg.begin(); i != net->layers_cfg.end(); ++i) {
                    ++layers_counter;
                    std::map<std::string, std::string> &layer_params = i->second;
                    std::string layer_type = layer_params["layer_type"];

                    if (layer_type == "convolutional")
                    {
                        int kernel_size = getParam<int>(layer_params, "size", -1);
                        int pad = getParam<int>(layer_params, "pad", 0);
                        int padding = getParam<int>(layer_params, "padding", 0);
                        int stride = getParam<int>(layer_params, "stride", 1);
                        int filters = getParam<int>(layer_params, "filters", -1);
                        int groups = getParam<int>(layer_params, "groups", 1);
                        bool batch_normalize = getParam<int>(layer_params, "batch_normalize", 0) == 1;
                        int flipped = getParam<int>(layer_params, "flipped", 0);
                        if (flipped == 1)
                            CV_Error(cv::Error::StsNotImplemented, "Transpose the convolutional weights is not implemented");

                        if (pad)
                            padding = kernel_size / 2;

                        // Cannot divide 0
                        CV_Assert(stride > 0);
                        CV_Assert(kernel_size > 0 && filters > 0);
                        CV_Assert(tensor_shape[0] > 0);
                        CV_Assert(tensor_shape[0] % groups == 0);

                        setParams.setConvolution(kernel_size, padding, stride, filters, tensor_shape[0],
                            groups, batch_normalize);

                        tensor_shape[0] = filters;
                        tensor_shape[1] = (tensor_shape[1] - kernel_size + 2 * padding) / stride + 1;
                        tensor_shape[2] = (tensor_shape[2] - kernel_size + 2 * padding) / stride + 1;
                    }
                    else if (layer_type == "connected")
                    {
                        int output = getParam<int>(layer_params, "output", 1);
                        bool batch_normalize = getParam<int>(layer_params, "batch_normalize", 0) == 1;

                        CV_Assert(output > 0);

                        setParams.setFullyConnected(output, batch_normalize);

                        if(layers_counter && tensor_shape[1] > 1)
                            net->out_channels_vec[layers_counter-1] = total(tensor_shape);

                        tensor_shape[0] = output;
                        tensor_shape[1] = 1;
                        tensor_shape[2] = 1;
                    }
                    else if (layer_type == "maxpool")
                    {
                        int kernel_size = getParam<int>(layer_params, "size", 2);
                        int stride = getParam<int>(layer_params, "stride", 2);
                        int padding = getParam<int>(layer_params, "padding", kernel_size - 1);
                        // Cannot divide 0
                        CV_Assert(stride > 0);

                        setParams.setMaxpool(kernel_size, padding, stride);

                        tensor_shape[1] = (tensor_shape[1] - kernel_size + padding) / stride + 1;
                        tensor_shape[2] = (tensor_shape[2] - kernel_size + padding) / stride + 1;
                    }
                    else if (layer_type == "avgpool")
                    {
                        setParams.setAvgpool();
                        tensor_shape[1] = 1;
                        tensor_shape[2] = 1;
                    }
                    else if (layer_type == "crop")
                    {
                        int crop_height = getParam<int>(layer_params, "crop_height", 0);
                        int crop_width = getParam<int>(layer_params, "crop_width", 0);
                        bool noadjust = getParam<int>(layer_params, "noadjust", false);
                        CV_CheckGT(crop_height, 0, "");
                        CV_CheckGT(crop_width, 0, "");

                        setParams.setCrop(crop_height, crop_width, tensor_shape[1], tensor_shape[2], noadjust);

                        tensor_shape[1] = crop_height;
                        tensor_shape[2] = crop_width;
                    }
                    else if (layer_type == "softmax")
                    {
                        int groups = getParam<int>(layer_params, "groups", 1);
                        if (groups != 1)
                            CV_Error(Error::StsNotImplemented, "Softmax from Darknet with groups != 1");
                        setParams.setSoftmax();
                    }
                    else if (layer_type == "route")
                    {
                        std::string bottom_layers = getParam<std::string>(layer_params, "layers", "");
                        CV_Assert(!bottom_layers.empty());
                        int groups = getParam<int>(layer_params, "groups", 1);
                        std::vector<int> layers_vec = getNumbers<int>(bottom_layers);

                        tensor_shape[0] = 0;
                        for (size_t k = 0; k < layers_vec.size(); ++k) {
                            layers_vec[k] = layers_vec[k] >= 0 ? layers_vec[k] : (layers_vec[k] + layers_counter);
                            tensor_shape[0] += net->out_channels_vec[layers_vec[k]];
                        }

                        if (groups > 1)
                        {
                            int group_id = getParam<int>(layer_params, "group_id", 0);
                            tensor_shape[0] /= groups;
                            int split_size = tensor_shape[0] / layers_vec.size();
                            for (size_t k = 0; k < layers_vec.size(); ++k)
                                setParams.setSlice(layers_vec[k], split_size, group_id);

                            if (layers_vec.size() > 1)
                            {
                                // layer ids in layers_vec - inputs of Slice layers
                                // after adding offset to layers_vec: layer ids - outputs of Slice layers
                                for (size_t k = 0; k < layers_vec.size(); ++k)
                                    layers_vec[k] += layers_vec.size();

                                setParams.setConcat(layers_vec.size(), layers_vec.data());
                            }
                        }
                        else
                        {
                            if (layers_vec.size() == 1)
                                setParams.setIdentity(layers_vec.at(0));
                            else
                                setParams.setConcat(layers_vec.size(), layers_vec.data());
                        }
                    }
                    else if (layer_type == "dropout" || layer_type == "cost")
                    {
                        setParams.setIdentity(layers_counter-1);
                    }
                    else if (layer_type == "reorg")
                    {
                        int stride = getParam<int>(layer_params, "stride", 2);
                        // Cannot divide 0
                        CV_Assert(stride > 0);
                        tensor_shape[0] = tensor_shape[0] * (stride * stride);
                        tensor_shape[1] = tensor_shape[1] / stride;
                        tensor_shape[2] = tensor_shape[2] / stride;

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

                        setParams.setPermute(false);
                        setParams.setRegion(thresh, coords, classes, num_of_anchors, classfix, softmax, softmax_tree, anchors_vec.data());
                    }
                    else if (layer_type == "shortcut")
                    {
                        std::string bottom_layer = getParam<std::string>(layer_params, "from", "");
                        float alpha = getParam<float>(layer_params, "alpha", 1);
                        float beta = getParam<float>(layer_params, "beta", 0);
                        if (beta != 0)
                            CV_Error(Error::StsNotImplemented, "Non-zero beta");
                        CV_Assert(!bottom_layer.empty());
                        int from = std::atoi(bottom_layer.c_str());

                        from = from < 0 ? from + layers_counter : from;
                        setParams.setShortcut(from, alpha);
                    }
                    else if (layer_type == "scale_channels")
                    {
                        std::string bottom_layer = getParam<std::string>(layer_params, "from", "");
                        CV_Assert(!bottom_layer.empty());
                        int from = std::atoi(bottom_layer.c_str());
                        from = from < 0 ? from + layers_counter : from;
                        setParams.setScaleChannels(from);
                    }
                    else if (layer_type == "sam")
                    {
                        std::string bottom_layer = getParam<std::string>(layer_params, "from", "");
                        CV_Assert(!bottom_layer.empty());
                        int from = std::atoi(bottom_layer.c_str());
                        from = from < 0 ? from + layers_counter : from;
                        setParams.setSAM(from);
                    }
                    else if (layer_type == "upsample")
                    {
                        int scaleFactor = getParam<int>(layer_params, "stride", 1);
                        setParams.setUpsample(scaleFactor);
                        tensor_shape[1] = tensor_shape[1] * scaleFactor;
                        tensor_shape[2] = tensor_shape[2] * scaleFactor;
                    }
                    else if (layer_type == "yolo")
                    {
                        int classes = getParam<int>(layer_params, "classes", -1);
                        int num_of_anchors = getParam<int>(layer_params, "num", -1);
                        float thresh = getParam<float>(layer_params, "thresh", 0.2);
                        float nms_threshold = getParam<float>(layer_params, "nms_threshold", 0.0);
                        float scale_x_y = getParam<float>(layer_params, "scale_x_y", 1.0);
                        int new_coords = getParam<int>(layer_params, "new_coords", 0);

                        std::string anchors_values = getParam<std::string>(layer_params, "anchors", std::string());
                        CV_Assert(!anchors_values.empty());
                        std::vector<float> anchors_vec = getNumbers<float>(anchors_values);

                        std::string mask_values = getParam<std::string>(layer_params, "mask", std::string());
                        CV_Assert(!mask_values.empty());
                        std::vector<int> mask_vec = getNumbers<int>(mask_values);

                        CV_Assert(classes > 0 && num_of_anchors > 0 && (num_of_anchors * 2) == anchors_vec.size());

                        setParams.setPermute(false);
                        setParams.setYolo(classes, mask_vec, anchors_vec, thresh, nms_threshold, scale_x_y, new_coords);
                    }
                    else {
                        CV_Error(cv::Error::StsParseError, "Unknown layer type: " + layer_type);
                    }

                    std::string activation = getParam<std::string>(layer_params, "activation", "linear");
                    if (activation != "linear")
                        setParams.setActivation(activation);

                    net->out_channels_vec[layers_counter] = tensor_shape[0];
                }

                return true;
            }

            bool ReadDarknetFromWeightsStream(std::istream &ifile, NetParameter *net)
            {
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

                MatShape tensor_shape(3);
                tensor_shape[0] = net->channels;
                tensor_shape[1] = net->height;
                tensor_shape[2] = net->width;
                int cv_layers_counter = -1;
                int darknet_layers_counter = -1;

                setLayersParams setParams(net);

                typedef std::map<int, std::map<std::string, std::string> >::iterator it_type;
                for (it_type i = net->layers_cfg.begin(); i != net->layers_cfg.end(); ++i) {
                    ++darknet_layers_counter;
                    ++cv_layers_counter;
                    std::map<std::string, std::string> &layer_params = i->second;
                    std::string layer_type = layer_params["layer_type"];

                    if (layer_type == "convolutional" || layer_type == "connected")
                    {
                        size_t weights_size;
                        int filters;
                        bool use_batch_normalize;
                        cv::Mat weightsBlob;
                        if(layer_type == "convolutional")
                        {
                            int kernel_size = getParam<int>(layer_params, "size", -1);
                            filters = getParam<int>(layer_params, "filters", -1);
                            int groups = getParam<int>(layer_params, "groups", 1);
                            use_batch_normalize = getParam<int>(layer_params, "batch_normalize", 0) == 1;

                            CV_Assert(kernel_size > 0 && filters > 0);
                            CV_Assert(tensor_shape[0] > 0);
                            CV_Assert(tensor_shape[0] % groups == 0);

                            weights_size = filters * (tensor_shape[0] / groups) * kernel_size * kernel_size;
                            int sizes_weights[] = { filters, tensor_shape[0] / groups, kernel_size, kernel_size };
                            weightsBlob.create(4, sizes_weights, CV_32F);
                        }
                        else
                        {
                            filters = getParam<int>(layer_params, "output", 1);
                            use_batch_normalize = getParam<int>(layer_params, "batch_normalize", 0) == 1;

                            CV_Assert(filters>0);

                            weights_size = total(tensor_shape) * filters;
                            int sizes_weights[] = { filters, total(tensor_shape) };
                            weightsBlob.create(2, sizes_weights, CV_32F);
                        }
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

                        // set conv/connected weights
                        std::vector<cv::Mat> layer_blobs;
                        layer_blobs.push_back(weightsBlob);
                        if (!use_batch_normalize) {
                            // use BIAS in any case
                            layer_blobs.push_back(biasData_mat);
                        }
                        setParams.setLayerBlobs(cv_layers_counter, layer_blobs);

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
                    }
                    if (layer_type == "region" || layer_type == "yolo")
                    {
                        ++cv_layers_counter;  // For permute.
                    }

                    std::string activation = getParam<std::string>(layer_params, "activation", "linear");
                    if (activation != "linear")
                        ++cv_layers_counter;  // For ReLU, Swish, Mish, Sigmoid, etc

                    if(!darknet_layers_counter)
                        tensor_shape.resize(1);

                    tensor_shape[0] = net->out_channels_vec[darknet_layers_counter];
                }
                return true;
            }

        }


        void ReadNetParamsFromCfgStreamOrDie(std::istream &ifile, darknet::NetParameter *net)
        {
            if (!darknet::ReadDarknetFromCfgStream(ifile, net)) {
                CV_Error(cv::Error::StsParseError, "Failed to parse NetParameter stream");
            }
        }

        void ReadNetParamsFromBinaryStreamOrDie(std::istream &ifile, darknet::NetParameter *net)
        {
            if (!darknet::ReadDarknetFromWeightsStream(ifile, net)) {
                CV_Error(cv::Error::StsParseError, "Failed to parse NetParameter stream");
            }
        }
    }
}
