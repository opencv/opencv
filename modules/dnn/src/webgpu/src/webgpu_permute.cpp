// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include <limits>
#include "common.hpp"
#include "webgpu_internal.hpp"
#include "../include/webgpu_permute.hpp"

namespace cv { namespace dnn { namespace webgpu {

#ifdef HAVE_WEBGPU

struct PermuteParam {
      int global_size;
      int num_axes;
      int nthreads;
};

static bool needForPermutation(std::vector<int>& order)
{
    for (int i = 0; i < order.size(); ++i)
    {
        if (order[i] != i)
            return true;
    }
    return false;
}

OpPermute::OpPermute(std::vector<size_t>& order)
{
    order_.assign(order.begin(), order.end());
    dims_ = order.size();
    need_permute_ = needForPermutation(order_);
    type_ = "Permute";
    if (need_permute_)
        createBindGroupLayout(5);
}

OpPermute::~OpPermute()
{
    if(tensor_new_stride_)
        delete tensor_new_stride_;
    if(tensor_old_stride_)
        delete tensor_old_stride_;
    if(tensor_order_)
        delete tensor_order_;
}

void OpPermute::reshapeOutTensor(std::vector<Tensor *>& ins, std::vector<Tensor>& outs)
{
    assert(!ins.empty());
    assert(ins.size() == outs.size());

    if (need_permute_)
    {
        assert(dims_ == ins[0]->dimNum());

        Shape shape_before = ins[0]->getShape();
        Shape shape_after;
        for (size_t i = 0; i < dims_; i++)
        {
            shape_after.push_back(shape_before[order_[i]]);
        }

        for (size_t i = 0; i < ins.size(); i++)
        {
            assert(ins[i]->dimNum() == 4);
            assert(ins[i]->dimSize(2) == shape_before[2] && ins[i]->dimSize(3) == shape_before[3]);
            assert(ins[i]->count() == shapeCount(shape_after));
            outs[i].reshape(NULL, shape_after);
        }
    }
    else
    {
        for(int i = 0; i < ins.size(); i++)
        {
            Shape in_shape = ins[i]->getShape();
            outs[i].reshape(NULL, in_shape);
        }
    }
}

void OpPermute::prepareStrides(const Shape &shape_before, const Shape &shape_after)
{
    assert(shape_before.size() == dims_);
    assert(shape_after.size() == dims_);

    old_stride_.resize(dims_);
    new_stride_.resize(dims_);

    old_stride_[dims_ - 1] = 1;
    new_stride_[dims_ - 1] = 1;

    for(int i = dims_ - 2; i >= 0; i--)
    {
        old_stride_[i] = old_stride_[i + 1] * shape_before[i + 1];
        new_stride_[i] = new_stride_[i + 1] * shape_after[i + 1];
    }

    Shape shape(1, old_stride_.size());
    if(! tensor_old_stride_)
        tensor_old_stride_ = new Tensor(old_stride_.data(), shape, wFormatInt32);
    else
        tensor_old_stride_->reshape((const char*)old_stride_.data(), shape, wFormatInt32);
    if(! tensor_new_stride_)
        tensor_new_stride_ = new Tensor(new_stride_.data(), shape, wFormatInt32);
    else
        tensor_new_stride_->reshape((const char*)new_stride_.data(), shape, wFormatInt32);
}

bool OpPermute::forward(std::vector<Tensor>& ins,
                        std::vector<Tensor>& blobs,
                        std::vector<Tensor>& outs)
{
    return forward(ins, outs);
}

bool OpPermute::forward(std::vector<Tensor>& ins, std::vector<Tensor>& outs)
{
    int num_ins = ins.size();
    in_shape_ = ins[0].getShape();
    out_shape_ = outs[0].getShape();
    if (!need_permute_)
    {
        for (int i = 0; i < num_ins; i++)
        {
            assert(outs[i].count() == ins[i].count());
            if (outs[i].getBuffer() != ins[i].getBuffer())
                ins[i].copyTo(outs[i]);
        }
        return true;
    }

    if (pipeline_ == nullptr)
    {
        createShaderModule(permute_spv, sizeof(permute_spv)/sizeof(uint32_t));
        createComputePipeline();
    }

    prepareStrides(ins[0].getShape(), outs[0].getShape());
    std::vector<int>shape(1, order_.size());
    if(! tensor_order_)
        tensor_order_ = new Tensor(order_.data(), shape, wFormatInt32);
    else
        tensor_order_->reshape((const char*)order_.data(), shape, wFormatInt32);
    bindTensor(ins[0],  0, bgEntries);
    bindTensor(* tensor_order_, 1, bgEntries);
    bindTensor(* tensor_old_stride_, 2, bgEntries);
    bindTensor(* tensor_new_stride_, 3, bgEntries);
    bindTensor(outs[0], 4, bgEntries);

    nthreads_ = ins[0].count();
#define LOCAL_SZ_X 256
    global_size_ = alignSize(nthreads_, LOCAL_SZ_X);
    computeGroupCount();

    PermuteParam param = {global_size_, dims_, nthreads_};
    if(! uniformBuffer_) uniformBuffer_ = new Buffer(&param, sizeof(PermuteParam));
    else uniformBuffer_->setBufferData(&param, sizeof(PermuteParam));
    bindUniform(* uniformBuffer_, 5, bgEntries);

    for (int i = 0; i < num_ins; i++)
    {
        bindTensor(ins[i],  0, bgEntries);
        bindTensor(outs[i], 4, bgEntries);

        createBindGroup();
        createCommandBuffer();
        runCommandBuffer();
    }
    return true;
}

bool OpPermute::computeGroupCount()
{
    group_x_ = global_size_ / LOCAL_SZ_X;
    group_y_ = 1;
    group_z_ = 1;
    return true;
}

#endif  // HAVE_WEBGPU

}}} // namespace cv::dnn::webgpu