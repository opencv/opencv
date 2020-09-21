// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "common.hpp"
#include "webgpu_internal.hpp"
#include "../include/webgpu_lrn.hpp"

namespace cv { namespace dnn { namespace webgpu {

#ifdef HAVE_WEBGPU

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
    createBindGroupLayout(2);
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
    batch_ = in_shape[wShapeIdxBatch];
    height_ = in_shape[wShapeIdxHeight];
    width_ = in_shape[wShapeIdxWidth];
    channels_= in_shape[wShapeIdxChannel];
    thread_num_ = batch_ * height_ * width_;
    if(pipeline_ == nullptr)
    {
        config_.local_size_x = LOCAL_SZ_X;
        config_.local_size_y = LOCAL_SZ_Y;
        config_.local_size_z = LOCAL_SZ_Z;
        config_.block_height = 1;
        config_.block_width  = 1;
        config_.block_depth  = 1;
        config_.shader_type  = kLRNShaderTypeBasic;
        createShaderModule(lrn_spv, sizeof(lrn_spv)/sizeof(uint32_t));
        createComputePipeline();
        computeGroupCount();
    }

    bindTensor(in, 0, bgEntries);
    bindTensor(out,1, bgEntries);
    LRNParam param = {batch_ * height_ * width_,
                channels_, height_, width_,
                filter_len_, radius_,
                alpha_ / (norm_by_size_ ? filter_len_ : 1),
                bias_, -1 * beta_};
    if(! uniformBuffer_) uniformBuffer_ = new Buffer(&param, sizeof(LRNParam));
    else uniformBuffer_->setBufferData(&param, sizeof(LRNParam));
    bindUniform(* uniformBuffer_, 2, bgEntries);
    createBindGroup();
    createCommandBuffer();
    runCommandBuffer();;
    return true;
}

bool OpLRN::computeGroupCount()
{
    group_x_ = alignSize(thread_num_, config_.local_size_x) / config_.local_size_x;
    group_y_ = 1;
    group_z_ = 1;

    return true;
}

#endif  // HAVE_WEBGPU

}}} // namespace cv::dnn::vkcom
