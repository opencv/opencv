// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../../precomp.hpp"
#include "common.hpp"
#include "internal.hpp"
#include "../include/op_relu.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

#define LOCAL_SZ_X 32

struct ReLUParam {
      int total;
      float slope;
};

OpReLU::OpReLU(const float slope) : slope_(slope)
{
    OpBase::initVulkanThing(2);
    type_ = "ReLU";
}

void OpReLU::reshapeOutTensor(Tensor& in, Tensor& out)
{
    Shape shape = in.getShape();
    out.reshape(NULL, shape);
}

bool OpReLU::forward(std::vector<Tensor>& ins,
                     std::vector<Tensor>& blobs,
                     std::vector<Tensor>& outs)
{
    return forward(ins[0], outs[0]);
}

bool OpReLU::forward(Tensor& in, Tensor& out)
{
    if (pipeline_ == VK_NULL_HANDLE)
    {
        total_ = in.count();
        batch_ = in.dimSize(0);
        channel_ = in.dimSize(1);
        height_ = in.dimSize(2);
        width_ = in.dimSize(3);
#define maxComputeWorkGroupCount 65535
        computeGroupCount();
        createShaderModule(relu_spv, sizeof(relu_spv));
        createPipeline(sizeof(ReLUParam));
    }

    bindTensor(device_, in,  0, descriptor_set_);
    bindTensor(device_, out, 1, descriptor_set_);
    ReLUParam param = { total_, slope_ };
    recordCommandBuffer((void *)&param, sizeof(ReLUParam));
    runCommandBuffer();
    return true;
}

bool OpReLU::computeGroupCount()
{
    config_.local_size_x = LOCAL_SZ_X;
    group_x_ = alignSize(total_, config_.local_size_x) / config_.local_size_x;
    // FIXME: uncomment these code if online compilation is enabled
    //        for relu shader
    // while(group_x_ > maxComputeWorkGroupCount)
    // {
    //     config_.local_size_x *= 2;
    //     group_x_ = alignSize(total_, config_.local_size_x) / config_.local_size_x;
    // }

    group_y_ = 1;
    group_z_ = 1;
    return true;
}

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom
