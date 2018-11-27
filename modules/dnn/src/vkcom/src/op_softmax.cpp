// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../../precomp.hpp"
#include "common.hpp"
#include "internal.hpp"
#include "../include/op_softmax.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

#define LOCAL_SZ_X 256
#define LOCAL_SZ_Y 1
#define LOCAL_SZ_Z 1

struct SoftmaxParam {
    int channel_size;
    int outer_size;
    int channels;
    int logsoftmax;
};

OpSoftmax::OpSoftmax(const int axis, const bool log_softmax)
{
    init(axis, log_softmax);
    type_ = "Softmax";
}

OpSoftmax::~OpSoftmax()
{
    if (max_tensor_)
        delete max_tensor_;
    if (sum_tensor_)
        delete sum_tensor_;
}

void OpSoftmax::reshapeOutTensor(Tensor& in, Tensor& out)
{
    Shape shape = in.getShape();
    out.reshape(NULL, shape);
}

bool OpSoftmax::init(const int axis, const bool log_softmax)
{
    axis_ = axis;
    log_softmax_ = log_softmax;
    max_tensor_ = NULL;
    sum_tensor_ = NULL;
    OpBase::initVulkanThing(4);
    return true;
}

bool OpSoftmax::forward(std::vector<Tensor>& ins,
                        std::vector<Tensor>& blobs,
                        std::vector<Tensor>& outs)
{
    return forward(ins[0], outs[0]);
}

bool OpSoftmax::forward(Tensor& in, Tensor& out)
{
    channels_ = in.dimSize(axis_);
    channel_size_ = in.count(axis_+1);
    outer_size_ = in.count(0, axis_);

    if (pipeline_ == VK_NULL_HANDLE)
    {
        config_.local_size_x = LOCAL_SZ_X;
        config_.local_size_y = LOCAL_SZ_Y;
        config_.local_size_z = LOCAL_SZ_Z;
        config_.block_height = 1;
        config_.block_width  = 1;
        config_.block_depth  = 1;
        createShaderModule(softmax_spv, sizeof(softmax_spv));
        createPipeline(sizeof(SoftmaxParam));
        computeGroupCount();
    }

    if (max_tensor_ == NULL || sum_tensor_ == NULL)
    {
        std::vector<int> shape = {outer_size_, channel_size_};
        max_tensor_ = new Tensor(NULL, shape);
        sum_tensor_ = new Tensor(NULL, shape);
    }
    bindTensor(device_, in,  0, descriptor_set_);
    bindTensor(device_, *max_tensor_,  1, descriptor_set_);
    bindTensor(device_, *sum_tensor_,  2, descriptor_set_);
    bindTensor(device_, out, 3, descriptor_set_);
    SoftmaxParam param = {channel_size_, outer_size_, channels_, log_softmax_ == true ? 1 : 0};
    recordCommandBuffer((void *)&param, sizeof(SoftmaxParam));
    runCommandBuffer();
    return true;
}

bool OpSoftmax::computeGroupCount()
{
    group_x_ = alignSize(outer_size_, config_.local_size_x) / config_.local_size_x;
    group_y_ = 1;
    group_z_ = 1;

    return true;
}

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom
