// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../../precomp.hpp"
#include <limits>
#include "common.hpp"
#include "internal.hpp"
#include "../include/op_prior_box.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

#define LOCAL_SZ_X 256
struct PriorBoxParam {
      int global_size;
      int nthreads;
      float step_x;
      float step_y;
      int offsets_x_size;
      int width_size;
      int layer_w;
      int image_h;
      int image_w;
      int clip;
      int variance_off;
};

OpPriorBox::OpPriorBox(float step_x,
                       float step_y,
                       bool clip,
                       int num_priors,
                       std::vector<float>& variance,
                       std::vector<float>& offsets_x,
                       std::vector<float>& offsets_y,
                       std::vector<float>& box_widths,
                       std::vector<float>& box_heights)
{
    step_x_ = step_x;
    step_y_ = step_y;
    clip_ = clip;
    num_priors_ = num_priors;
    variance_ = variance;
    offsets_x_ = offsets_x;
    offsets_y_ = offsets_y;
    box_widths_ = box_widths;
    box_heights_ = box_heights;
    type_ = "PriorBox";
#define BUFFER_NUM 6
    OpBase::initVulkanThing(BUFFER_NUM);
}

void OpPriorBox::reshapeOutTensor(std::vector<Tensor *>& ins, Tensor& out)
{
    assert(!ins.empty());

    Shape in_shape = ins[0]->getShape();
    int layer_h = in_shape[kShapeIdxHeight];
    int layer_w = in_shape[kShapeIdxWidth];
    int out_num = 1;
    int out_channel = 2;
    Shape out_shape = {out_num, out_channel, layer_h * layer_w * num_priors_ * 4};
    out.reshape(NULL, out_shape);
}

bool OpPriorBox::forward(std::vector<Tensor>& ins,
                         std::vector<Tensor>& blobs,
                         std::vector<Tensor>& outs)
{
    return forward(ins, outs[0]);
}

bool OpPriorBox::forward(std::vector<Tensor>& ins, Tensor& out)
{
    assert(ins.size() == 2);
    Shape in_shape = ins[0].getShape();
    Shape img_shape = ins[1].getShape();

    in_h_ = in_shape[kShapeIdxHeight];
    in_w_ = in_shape[kShapeIdxWidth];
    img_h_ = img_shape[kShapeIdxHeight];
    img_w_ = img_shape[kShapeIdxWidth];
    out_channel_ = out.dimSize(1);
    out_channel_size_ = out.dimSize(2);
    nthreads_ = in_h_ * in_w_;
    global_size_ = alignSize(nthreads_, LOCAL_SZ_X);

    if (pipeline_ == VK_NULL_HANDLE)
    {
        createShaderModule(prior_box_spv, sizeof(prior_box_spv));
        createPipeline(sizeof(PriorBoxParam));
        computeGroupCount();
    }

    std::vector<int>shape;
    shape.push_back(offsets_x_.size());
    tensor_offsets_x_.reshape((const char*)offsets_x_.data(), shape);
    tensor_offsets_y_.reshape((const char*)offsets_y_.data(), shape);

    shape[0] = box_widths_.size();
    tensor_widths_.reshape((const char*)box_widths_.data(), shape);
    tensor_heights_.reshape((const char*)box_heights_.data(), shape);

    float variance[4] = {variance_[0], variance_[0], variance_[0], variance_[0]};
    if (variance_.size() > 1)
    {
        assert(variance_.size() == 4);
        for (int i = 1; i < variance_.size(); i++)
            variance[i] = variance_[i];
    }
    shape[0] = 4;
    tensor_variance_.reshape((const char*)variance, shape);

    bindTensor(device_, tensor_offsets_x_,  0, descriptor_set_);
    bindTensor(device_, tensor_offsets_y_, 1, descriptor_set_);
    bindTensor(device_, tensor_widths_,  2, descriptor_set_);
    bindTensor(device_, tensor_heights_, 3, descriptor_set_);
    bindTensor(device_, tensor_variance_, 4, descriptor_set_);
    bindTensor(device_, out, 5, descriptor_set_);

    PriorBoxParam param = {global_size_,
                           nthreads_,
                           step_x_,
                           step_y_,
                           (int)offsets_x_.size(),
                           (int)box_widths_.size(),
                           in_w_,
                           img_h_,
                           img_w_,
                           clip_ ? 1 : 0,
                           out_channel_size_ / 4};
    recordCommandBuffer((void *)&param, sizeof(PriorBoxParam));
    runCommandBuffer();
    return true;
}

bool OpPriorBox::computeGroupCount()
{
    group_x_ = global_size_ / LOCAL_SZ_X;
    group_y_ = 1;
    group_z_ = 1;
    return true;
}

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom
