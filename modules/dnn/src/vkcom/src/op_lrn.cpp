// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../../precomp.hpp"
#include "common.hpp"
#include "internal.hpp"
#include "../include/op_lrn.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

#define LOCAL_SZ_X 256
#define LOCAL_SZ_Y 1
#define LOCAL_SZ_Z 1

struct LRNParam {
    int thread_num;
    int channels;
    int height;
    int width;
    int filter_len;
    int radius;
    float alpha;
    float bias;
    float negative_beta;
};

OpLRN::OpLRN(const int radius, const float bias,
             const float alpha, const float beta,
             const bool norm_by_size)
{
    init(radius, bias, alpha, beta, norm_by_size);
    type_ = "LRN";
}

void OpLRN::reshapeOutTensor(Tensor& in, Tensor& out)
{
    Shape shape = in.getShape();
    out.reshape(NULL, shape);
}

bool OpLRN::init(const int radius, const float bias,
                 const float alpha, const float beta,
                 const bool norm_by_size)
{
    radius_ = radius;
    filter_len_ = 2 * radius_ + 1;
    bias_  = bias;
    alpha_ = alpha;
    beta_  = beta;
    norm_by_size_ = norm_by_size;
    OpBase::initVulkanThing(2);
    return true;
}

bool OpLRN::forward(std::vector<Tensor>& ins,
                    std::vector<Tensor>& blobs,
                    std::vector<Tensor>& outs)
{
    return forward(ins[0], outs[0]);
}

bool OpLRN::forward(Tensor& in, Tensor& out)
{
    Shape in_shape = in.getShape();
    batch_ = in_shape[kShapeIdxBatch];
    height_ = in_shape[kShapeIdxHeight];
    width_ = in_shape[kShapeIdxWidth];
    channels_= in_shape[kShapeIdxChannel];
    thread_num_ = batch_ * height_ * width_;

    if (pipeline_ == VK_NULL_HANDLE)
    {
        config_.local_size_x = LOCAL_SZ_X;
        config_.local_size_y = LOCAL_SZ_Y;
        config_.local_size_z = LOCAL_SZ_Z;
        config_.block_height = 1;
        config_.block_width  = 1;
        config_.block_depth  = 1;
        config_.shader_type  = kLRNShaderTypeBasic;
        createShaderModule(lrn_spv, sizeof(lrn_spv));
        createPipeline(sizeof(LRNParam));
        computeGroupCount();
    }

    bindTensor(device_, in, 0, descriptor_set_);
    bindTensor(device_, out,1, descriptor_set_);

    LRNParam param = {batch_ * height_ * width_,
                      channels_, height_, width_,
                      filter_len_, radius_,
                      alpha_ / (norm_by_size_ ? filter_len_ : 1),
                      bias_, -1 * beta_};
    recordCommandBuffer((void *)&param, sizeof(LRNParam));
    runCommandBuffer();
    return true;
}

bool OpLRN::computeGroupCount()
{
    group_x_ = alignSize(thread_num_, config_.local_size_x) / config_.local_size_x;
    group_y_ = 1;
    group_z_ = 1;

    return true;
}

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom
