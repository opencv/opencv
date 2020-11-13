// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "common.hpp"
#include "webgpu_internal.hpp"
#include "../include/webgpu_conv.hpp"

namespace cv { namespace dnn { namespace webgpu {

#ifdef HAVE_WEBGPU

#define DEFAULT_LOCAL_SZ 256
#define MAX_COMPUTE_GFLOPS 10

#define MAX_GROUP_COUNT_X 65535
#define MAX_GROUP_COUNT_Y 65535
#define MAX_GROUP_COUNT_Z 65535

struct ShaderConstant {
    int lsz_x;
    int lsz_y;
    int lsz_z;
    int in_h;
    int in_w;
    int out_w;
    int stride_h;
    int stride_w;
    int pad_h;
    int pad_w;
    int filter_h;
    int filter_w;
    int channels;
    int batch;
    int m;
    int k;
    int n;
    int tail_m;
    int dilation_h;
    int dilation_w;
};

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
    batch_ = in_shape[wShapeIdxBatch];
    in_height_ = in_shape[wShapeIdxHeight];
    in_width_ = in_shape[wShapeIdxWidth];
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
    createBindGroupLayout(4);
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
    batch_ = in_shape[wShapeIdxBatch];
    in_height_ = in_shape[wShapeIdxHeight];
    in_width_ = in_shape[wShapeIdxWidth];
    in_channel_= in_shape[wShapeIdxChannel];
    out_height_ = out_shape[wShapeIdxHeight];
    out_width_ = out_shape[wShapeIdxWidth];
    int M = out_height_ * out_width_;
    int K = filter_height_ * filter_width_ * in_channel_;
    int N = out_channel_;
    if(pipeline_ == nullptr)
    {
        config_.local_size_x = DEFAULT_LOCAL_SZ;
        config_.local_size_y = 1;
        config_.local_size_z = 1;
        config_.block_height = 1;
        config_.block_width  = 1;
        config_.block_depth  = 1;
        if ((N % 8 == 0) && (K % 4 == 0) && (M % 4) == 0)
        {
            assert(group_ == 1); // TODO: support group > 1
            config_.shader_type  = wConvShaderType48;
            config_.local_size_x = 1;
            config_.local_size_y = DEFAULT_LOCAL_SZ;
            config_.local_size_z = 1;
            config_.block_height = 4;
            config_.block_width  = 8;
            createShaderModule(conv48_spv, sizeof(conv48_spv)/sizeof(uint32_t));
            ShaderConstant shader_constant;
            shader_constant.lsz_x = config_.local_size_x;
            shader_constant.lsz_y = config_.local_size_y;
            shader_constant.lsz_z = config_.local_size_z;
            shader_constant.in_h  = in_height_;
            shader_constant.in_w  = in_width_;
            shader_constant.out_w = out_width_;
            shader_constant.stride_h = stride_height_;
            shader_constant.stride_w = stride_width_;
            shader_constant.pad_h = padding_top_;
            shader_constant.pad_w = padding_left_;
            shader_constant.filter_h = filter_height_;
            shader_constant.filter_w = filter_width_;
            shader_constant.channels = in_channel_;
            shader_constant.batch = batch_;
            shader_constant.m = M;
            shader_constant.k = K;
            shader_constant.n = N;
            shader_constant.tail_m = M % 4;
            shader_constant.dilation_h = dilation_height_;
            shader_constant.dilation_w = dilation_width_;
            if(! uniformBuffer_) uniformBuffer_ = new Buffer(&shader_constant, sizeof(ShaderConstant));
            else uniformBuffer_->setBufferData(&shader_constant, sizeof(ShaderConstant));
        }
        else if (out_channel_ == in_channel_ && in_channel_ == group_)
        {
            config_.shader_type  = wConvShaderTypeDepthWise;
            createShaderModule(dw_conv_spv, sizeof(dw_conv_spv)/sizeof(uint32_t));
        }
        else
        {
            assert(group_ == 1); // TODO: support group > 1
            config_.shader_type  = wConvShaderTypeBasic;
            createShaderModule(conv_spv, sizeof(conv_spv)/sizeof(uint32_t));
        }
        createComputePipeline();
        computeGroupCount();
    }

    bindTensor(in, 0, bgEntries);
    bindTensor(bias, 1, bgEntries);
    bindTensor(filter_weights, 2, bgEntries);
    bindTensor(out, 3, bgEntries);
    if (config_.shader_type == wConvShaderTypeBasic || config_.shader_type == wConvShaderTypeDepthWise)
    {
        ShaderParam param = {in_height_, in_width_,
                        out_height_, out_width_,
                        stride_height_, stride_width_,
                        padding_top_, padding_left_,
                        filter_height_, filter_width_,
                        dilation_height_, dilation_width_,
                        in_channel_, batch_, has_bias_,
                        M, K, N, 0, 0, 0};
        int partition_num = 1;
        if (config_.shader_type == wConvShaderTypeBasic)
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
                if(! uniformBuffer_) uniformBuffer_ = new Buffer(&param, sizeof(ShaderParam));
                else uniformBuffer_->setBufferData(&param, sizeof(ShaderParam));
                bindUniform(*uniformBuffer_, 4, bgEntries);
                createBindGroup();
                createCommandBuffer();
                runCommandBuffer();
            }
        };
    }
    else if(config_.shader_type == wConvShaderType48)
    {
        bindUniform(*uniformBuffer_, 4, bgEntries);
        createBindGroup();
        createCommandBuffer();
        runCommandBuffer();
    }
    return true;
}

bool OpConv::computeGroupCount()
{
    if (config_.shader_type == wConvShaderTypeDepthWise)
    {
        group_x_ = alignSize(out_width_, config_.local_size_x) / config_.local_size_x;
        group_y_ = alignSize(out_height_, config_.local_size_y) / config_.local_size_y;
        group_z_ = alignSize(in_channel_, config_.local_size_z) / config_.local_size_z;
        return true;
    }

    int M = out_height_ * out_width_;
    int N = out_channel_;

    if (config_.shader_type == wConvShaderTypeBasic)
    {

        group_x_ = alignSize(out_height_ * out_width_, config_.local_size_x) / config_.local_size_x;
        float GFLOPS = (2.0 * filter_height_ * filter_width_ * in_channel_ + 1) *
                       (out_channel_ * out_height_ * out_width_) / 1000 / 1000 / 1000;
        CV_Assert(config_.local_size_y == 1);
        group_y_ = std::min(MAX_GROUP_COUNT_Y, (int)floor(MAX_COMPUTE_GFLOPS / (GFLOPS / out_channel_)));
        group_z_ = 1;
    }
    else if (config_.shader_type == wConvShaderType48)
    {
        assert(config_.block_width == 8 &&
               config_.block_height == 4 &&
               config_.block_depth == 1 &&
               config_.local_size_z == 1);
        group_x_ = N / config_.block_width;
        group_y_ = alignSize(alignSize(M, 4) / 4, config_.local_size_y) / config_.local_size_y;
        group_z_ = batch_;
    }
    else
        CV_Assert(0);

    CV_Assert(group_x_ <= MAX_GROUP_COUNT_X);
    CV_Assert(group_y_ <= MAX_GROUP_COUNT_Y);
    CV_Assert(group_z_ <= MAX_GROUP_COUNT_Z);

    return true;
}

#endif  // HAVE_WEBGPU

}}} // namespace cv::dnn::webgpu
