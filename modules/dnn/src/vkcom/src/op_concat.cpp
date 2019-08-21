// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../../precomp.hpp"
#include "common.hpp"
#include "internal.hpp"
#include "../include/op_concat.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

#define LOCAL_SZ_X 256

struct ConcatParam {
    int out_concat_axis;
    int accumulated_concat_axis;
    int concat_size;
    int total_concat_size;
    int thread_num;
};

OpConcat::OpConcat(const int axis)
{
    init(axis);
    type_ = "Concat";
}

bool OpConcat::init(const int axis)
{
    axis_ = axis;
#define BUFFER_NUM 2
    OpBase::initVulkanThing(BUFFER_NUM);
    return true;
}

void OpConcat::reshapeOutTensor(std::vector<Tensor *>& in, Tensor& out)
{
    int sum_axis = 0;

    for (int i = 0; i < in.size(); ++i)
    {
        sum_axis += in[i]->dimSize(axis_);
    }

    Shape shape = in[0]->getShape();
    shape[axis_] = sum_axis;
    out.reshape(NULL, shape);
}

bool OpConcat::forward(std::vector<Tensor>& ins,
                       std::vector<Tensor>& blobs,
                       std::vector<Tensor>& outs)
{
    return forward(ins, outs[0]);
}

bool OpConcat::forward(std::vector<Tensor>& ins, Tensor& out)
{
    int input_num = ins.size();
    Tensor& first_tensor = ins[0];
    int sum_axis = first_tensor.dimSize(axis_);
    int dim_num = first_tensor.dimNum();
    for (int i = 1; i < input_num; ++i)
    {
        Tensor& tensor = ins[i];
        assert(tensor.dimNum() == dim_num);
        for (int d = 0; d < dim_num; ++d)
        {
            if (d == axis_)
            {
                sum_axis += tensor.dimSize(axis_);;
            }
            else
            {
                assert(first_tensor.dimSize(d) == tensor.dimSize(d));
            }
        }
    }

    assert(out.dimSize(axis_) == sum_axis);
    for (int d = 0; d < dim_num; ++d)
    {
        if (d != axis_)
        {
            assert(out.dimSize(d) == first_tensor.dimSize(d));
        }
    }
    out_concat_axis_ = sum_axis;
    concat_size_ = out.count(axis_ + 1);

    if (pipeline_ == VK_NULL_HANDLE)
    {
        config_.local_size_x = LOCAL_SZ_X;
        config_.block_height = 1;
        config_.block_width  = 1;
        config_.block_depth  = 1;
        createShaderModule(concat_spv, sizeof(concat_spv));
        createPipeline(sizeof(ConcatParam));
    }

    accumulated_concat_axis_ = 0;
    for (int i = 0; i < input_num; i++)
    {
        bindTensor(device_, ins[i], 0, descriptor_set_);
        bindTensor(device_, out, 1, descriptor_set_);
        total_concat_size_ = ins[i].count(axis_);
        thread_num_ = ins[i].count();
        computeGroupCount();
        ConcatParam param = {out_concat_axis_,
                             accumulated_concat_axis_,
                             concat_size_,
                             total_concat_size_,
                             thread_num_};
        recordCommandBuffer((void *)&param, sizeof(ConcatParam));
        runCommandBuffer();
        accumulated_concat_axis_ += ins[i].dimSize(axis_);
    }

    return true;
}

bool OpConcat::computeGroupCount()
{
    group_x_ = alignSize(thread_num_, config_.local_size_x) / config_.local_size_x;
    group_y_ = 1;
    group_z_ = 1;

    return true;
}

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom
