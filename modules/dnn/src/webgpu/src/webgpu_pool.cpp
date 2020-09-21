// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include <limits>
#include "common.hpp"
#include "webgpu_internal.hpp"
#include "../include/webgpu_pool.hpp"

namespace cv { namespace dnn { namespace webgpu {

#ifdef HAVE_WEBGPU

#define LOCAL_SZ_X 256
#define LOCAL_SZ_Y 1
#define LOCAL_SZ_Z 1

struct PoolParam {
      int channels;
      int in_height;
      int in_width;
      int out_height;
      int out_width;
      int padding_top;
      int padding_left;
      int filter_h;
      int filter_w;
      int stride_h;
      int stride_w;
      int total;
      int mask_or_padded_area;
};

OpPool::OpPool(const int* filter_size, const int* pad, const int* stride,
               const int padding_mode, const PoolType type,
               const bool avg_pool_padded_area)
{
    init(filter_size, pad, stride, padding_mode, type, avg_pool_padded_area);
    type_ = "Pool";
}

bool OpPool::init(const int* filter_size, const int* pad, const int* stride,
                  const int padding_mode, const PoolType type, bool avg_pool_padded_area)
{
    WGPU_CHECK_BOOL_RET_VAL(padding_mode >= 0 && padding_mode < wPaddingModeNum, false);
    WGPU_CHECK_POINTER_RET_VAL(filter_size, false);
    WGPU_CHECK_POINTER_RET_VAL(pad, false);
    WGPU_CHECK_POINTER_RET_VAL(stride, false);

    filter_height_ = filter_size[0];
    filter_width_ = filter_size[1];
    padding_top_ = pad[0];
    padding_left_ = pad[1];
    padding_mode_ = (PaddingMode)padding_mode;
    stride_height_ = stride[0];
    stride_width_ = stride[1];
    pool_type_ = type;
    avg_pool_padded_area_ = avg_pool_padded_area ? 1 : 0;

    if (pool_type_ == wPoolTypeAvg)
        createBindGroupLayout(2);
    else if (pool_type_ == wPoolTypeMax)
        createBindGroupLayout(3);
    else
        assert(0);
    return true;
}

void OpPool::reshapeOutTensor(Tensor& in, Tensor& out)
{
    Shape in_shape = in.getShape();
    batch_ = in_shape[wShapeIdxBatch];
    channels_ = in_shape[wShapeIdxChannel];
    in_height_ = in_shape[wShapeIdxHeight];
    in_width_ = in_shape[wShapeIdxWidth];
    computePoolOutputShape(padding_mode_, padding_top_, padding_left_,
                           in_height_, in_width_,
                           filter_height_, filter_width_,
                           stride_height_, stride_width_,
                           out_height_, out_width_);
    Shape out_shape = {batch_, channels_, out_height_, out_width_};
    out.reshape(NULL, out_shape);
}

bool OpPool::forward(std::vector<Tensor>& ins,
                     std::vector<Tensor>& blobs,
                     std::vector<Tensor>& outs)
{
    Tensor& inpMat = ins[0];
    Tensor& outMat = outs[0];
    Tensor maskMat = outs.size() > 1 ? outs[1] : Tensor();
    return forward(inpMat, outMat, maskMat);
}

bool OpPool::forward(Tensor& in, Tensor& out, Tensor& mask)
{
    Shape in_shape = in.getShape();
    Shape out_shape = out.getShape();
    batch_ = in_shape[wShapeIdxBatch];
    channels_ = in_shape[wShapeIdxChannel];
    in_height_ = in_shape[wShapeIdxHeight];
    in_width_ = in_shape[wShapeIdxWidth];
    out_height_ = out_shape[wShapeIdxHeight];
    out_width_ = out_shape[wShapeIdxWidth];
    need_mask_ = mask.isEmpty() ? 0 : 1;

    if (pipeline_ == nullptr)
    {
        config_.local_size_x = LOCAL_SZ_X;
        config_.local_size_y = LOCAL_SZ_Y;
        config_.local_size_z = LOCAL_SZ_Z;
        config_.block_height = 1;
        config_.block_width  = 1;
        config_.block_depth  = 1;
        if (pool_type_ == wPoolTypeAvg)
            createShaderModule(avg_pool_spv, sizeof(avg_pool_spv)/sizeof(uint32_t));
        else
            createShaderModule(max_pool_spv, sizeof(max_pool_spv)/sizeof(uint32_t));
        createComputePipeline();
        computeGroupCount();
    }

    bindTensor(in,  0, bgEntries);
    bindTensor(out, 1, bgEntries);
    if (need_mask_)
        bindTensor(mask, 2, bgEntries);
    else
    {
        std::vector<int> shape = {1,1};
        mask.reshape(NULL, shape);
        bindTensor(mask, 2, bgEntries);
    }
    PoolParam param = {channels_,
                    in_height_, in_width_,
                    out_height_, out_width_,
                    padding_top_, padding_left_,
                    filter_height_, filter_width_,
                    stride_height_, stride_width_, out.count(),
                    pool_type_ == wPoolTypeAvg ? avg_pool_padded_area_ : need_mask_};
    if(! uniformBuffer_) uniformBuffer_ = new Buffer(&param, sizeof(PoolParam));
    else uniformBuffer_->setBufferData(& param, sizeof(PoolParam));
    bindUniform(*uniformBuffer_, pool_type_ == wPoolTypeAvg ? 2 : 3, bgEntries);
    createBindGroup();
    createCommandBuffer();
    runCommandBuffer();
    return true;
}

bool OpPool::computeGroupCount()
{
#define GLOBAL_SIZE (128 * 128)
    group_x_ = alignSize(GLOBAL_SIZE, config_.local_size_x) / config_.local_size_x;
    group_y_ = 1;
    group_z_ = 1;
    return true;
}

#endif  // HAVE_WEBGPU

}}} // namespace cv::dnn::webgpu
