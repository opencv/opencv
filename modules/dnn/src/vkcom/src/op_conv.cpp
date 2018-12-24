// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../../precomp.hpp"
#include "common.hpp"
#include "internal.hpp"
#include "../include/op_conv.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

#define LOCAL_SZ_X 256
#define MAX_COMPUTE_GFLOPS 10
// TODO: query group count from vulkan device
#define MAX_GROUP_COUNT_X 65535
#define MAX_GROUP_COUNT_Y 65535
#define MAX_GROUP_COUNT_Z 65535

struct ShaderParam {
    int in_h;
    int in_w;
    int out_h;
    int out_w;
    int stride_h;
    int stride_w;
    int pad_h;
    int pad_w;
    int filter_h;
    int filter_w;
    int dilation_h;
    int dilation_w;
    int channels;
    int batch;
    int has_bias;
    int M;
    int K;
    int N;
    int basic_shader_batch_idx;
    int basic_shader_partition_idx;
    int basic_shader_partition_size;
};

OpConv::OpConv(const int out_channel, const bool has_bias,
               const int* filter_size, const int* pad,
               const int* stride, const int* dilation,
               const int activation, const int group,
               const int padding_mode)
{
    init(out_channel, has_bias, filter_size, pad,
         stride, dilation, activation, group, padding_mode);
    type_ = "Conv";
}

void OpConv::reshapeOutTensor(Tensor& in, Tensor& out)
{
    Shape in_shape = in.getShape();
    batch_ = in_shape[kShapeIdxBatch];
    in_height_ = in_shape[kShapeIdxHeight];
    in_width_ = in_shape[kShapeIdxWidth];
    computeConvOutputShapeAndPadding(padding_mode_, padding_top_, padding_left_,
                                     in_height_, in_width_,
                                     filter_height_, filter_width_,
                                     dilation_height_, dilation_width_,
                                     stride_height_, stride_width_,
                                     out_height_, out_width_);
    Shape shape = {batch_, out_channel_, out_height_, out_width_};
    out.reshape(NULL, shape);
}

bool OpConv::init(const int out_channel, const bool has_bias,
                  const int* filter_size, const int* pad,
                  const int* stride, const int* dilation,
                  const int activation, const int group,
                  const int padding_mode)
{
    out_channel_ = out_channel;
    filter_height_ = filter_size[0];
    filter_width_ = filter_size[1];
    padding_top_ = pad[0];
    padding_left_ = pad[1];
    stride_height_ = stride[0];
    stride_width_ = stride[1];
    dilation_height_ = dilation[0];
    dilation_width_ = dilation[1];
    padding_mode_ = (PaddingMode)padding_mode;
    has_bias_ = has_bias ? 1 : 0;
    activation_ = activation;
    group_ = group;

    #define BUFFER_NUM 4
    OpBase::initVulkanThing(BUFFER_NUM);
    return true;
}

bool OpConv::forward(std::vector<Tensor>& ins,
                     std::vector<Tensor>& blobs,
                     std::vector<Tensor>& outs)
{
    std::vector<int> shape = {1};
    Tensor bias(0, shape);

    if (has_bias_)
    {
        assert(blobs.size() == 2);
        bias = blobs[1];
    }

    return forward(ins[0], blobs[0], bias, outs[0]);
}

bool OpConv::forward(Tensor& in, Tensor& filter_weights, Tensor& bias, Tensor& out)
{
    Shape in_shape = in.getShape();
    Shape out_shape = out.getShape();
    batch_ = in_shape[kShapeIdxBatch];
    in_height_ = in_shape[kShapeIdxHeight];
    in_width_ = in_shape[kShapeIdxWidth];
    in_channel_= in_shape[kShapeIdxChannel];
    out_height_ = out_shape[kShapeIdxHeight];
    out_width_ = out_shape[kShapeIdxWidth];

    dwconv_ = (out_channel_ == in_channel_ && in_channel_ == group_);
    if (dwconv_ == false)
        assert(group_ == 1); // TODO: support group > 1

    if (pipeline_ == VK_NULL_HANDLE)
    {
        config_.local_size_x = LOCAL_SZ_X;
        config_.local_size_y = 1;
        config_.local_size_z = 1;
        config_.block_height = 1;
        config_.block_width  = 1;
        config_.block_depth  = 1;
        config_.shader_type  = kConvShaderTypeBasic;

        if (dwconv_)
            createShaderModule(dw_conv_spv, sizeof(dw_conv_spv));
        else
            createShaderModule(conv_spv, sizeof(conv_spv));
        createPipeline(sizeof(ShaderParam));
        computeGroupCount();
    }

    bindTensor(device_, in, 0, descriptor_set_);
    bindTensor(device_, bias, 1, descriptor_set_);
    bindTensor(device_, filter_weights, 2, descriptor_set_);
    bindTensor(device_, out, 3, descriptor_set_);

    int M = out_height_ * out_width_;
    int K = filter_height_ * filter_width_ * in_channel_;
    int N = out_channel_;
    ShaderParam param = {in_height_, in_width_,
                         out_height_, out_width_,
                         stride_height_, stride_width_,
                         padding_top_, padding_left_,
                         filter_height_, filter_width_,
                         dilation_height_, dilation_width_,
                         in_channel_, batch_, has_bias_,
                         M, K, N, 0, 0, 0};

    int partition_num = 1;
    if (!dwconv_)
    {
        param.basic_shader_partition_size = group_y_;
        partition_num = (int)ceil(1.0 * out_channel_ / group_y_);
    }

    for (int b = 0;  b < batch_; b++)
    {
        param.basic_shader_batch_idx = b;
        for (int n = 0;  n < partition_num; n++)
        {
            param.basic_shader_partition_idx = n;
            recordCommandBuffer((void *)&param, sizeof(ShaderParam));
            runCommandBuffer();
        }
    }

    return true;
}

bool OpConv::computeGroupCount()
{
    if (dwconv_)
    {
        group_x_ = alignSize(out_width_, config_.local_size_x) / config_.local_size_x;
        group_y_ = alignSize(out_height_, config_.local_size_y) / config_.local_size_y;
        group_z_ = alignSize(in_channel_, config_.local_size_z) / config_.local_size_z;
    }
    else if (config_.shader_type == kConvShaderTypeBasic)
    {

        group_x_ = alignSize(out_height_ * out_width_, config_.local_size_x) / config_.local_size_x;
        float GFLOPS = (2.0 * filter_height_ * filter_width_ * in_channel_ + 1) *
                       (out_channel_ * out_height_ * out_width_) / 1000 / 1000 / 1000;
        CV_Assert(config_.local_size_y == 1);
        group_y_ = std::min(MAX_GROUP_COUNT_Y, (int)floor(MAX_COMPUTE_GFLOPS / (GFLOPS / out_channel_)));
        group_z_ = 1;
    }
    else
        CV_Assert(0);

    CV_Assert(group_x_ <= MAX_GROUP_COUNT_X);
    CV_Assert(group_y_ <= MAX_GROUP_COUNT_Y);
    CV_Assert(group_z_ <= MAX_GROUP_COUNT_Z);

    return true;
}

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom
