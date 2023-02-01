// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_VKCOM_OP_CONCAT_HPP
#define OPENCV_DNN_VKCOM_OP_CONCAT_HPP

#include "vkcom.hpp"
#include "op_base.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

struct ConcatShaderConfig
{
    int local_size_x;
    int local_size_y;
    int local_size_z;
    int block_height;
    int block_width;
    int block_depth;
};

class OpConcat: public OpBase
{
public:
    OpConcat(const int axis);
    bool forward(std::vector<Tensor>& ins, Tensor& out);
    void reshapeOutTensor(std::vector<Tensor *>& in, Tensor& out);
    virtual bool forward(std::vector<Tensor>& ins,
                         std::vector<Tensor>& blobs,
                         std::vector<Tensor>& outs) CV_OVERRIDE;

private:
    bool init(const int axis);
    bool computeGroupCount();

    ConcatShaderConfig config_;
    int axis_;
    int out_concat_axis_;
    int accumulated_concat_axis_;
    int concat_size_;
    int total_concat_size_;
    int thread_num_;
};

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom

#endif // OPENCV_DNN_VKCOM_OP_CONCAT_HPP
