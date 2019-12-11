// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_VKCOM_OP_POOL_HPP
#define OPENCV_DNN_VKCOM_OP_POOL_HPP

#include "vkcom.hpp"
#include "op_base.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

enum PoolType { kPoolTypeAvg, kPoolTypeMax, kPoolTypeNum };

struct PoolShaderConfig
{
    int local_size_x;
    int local_size_y;
    int local_size_z;
    int block_height;
    int block_width;
    int block_depth;
};

class OpPool: public OpBase
{
public:
    OpPool(const int* filter_size, const int* pad, const int* stride,
           const int padding_mode, const PoolType pool_type,
           const bool avg_pool_padded_area);
    bool forward(Tensor& in, Tensor& out, Tensor& mask);
    void reshapeOutTensor(Tensor& in, Tensor& out);
    virtual bool forward(std::vector<Tensor>& ins,
                         std::vector<Tensor>& blobs,
                         std::vector<Tensor>& outs) CV_OVERRIDE;
private:
    bool init(const int* filter_size, const int* pad, const int* stride,
              const int padding_mode, const PoolType type, const bool avg_pool_padded_area);
    bool computeGroupCount();

    int batch_;
    int channels_;
    int in_height_;
    int in_width_;
    int out_height_;
    int out_width_;
    int filter_height_;
    int filter_width_;
    int stride_height_;
    int stride_width_;
    int padding_left_;
    int padding_top_;
    PoolType pool_type_;
    int avg_pool_padded_area_;
    int need_mask_;
    PaddingMode padding_mode_;
    //int activation_;
    PoolShaderConfig config_;
};

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom

#endif // OPENCV_DNN_VKCOM_OP_POOL_HPP
