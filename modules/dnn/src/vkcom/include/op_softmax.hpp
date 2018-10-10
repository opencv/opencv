// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_VKCOM_OP_SOFTMAX_HPP
#define OPENCV_DNN_VKCOM_OP_SOFTMAX_HPP

#include "vkcom.hpp"
#include "op_base.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

struct SoftmaxShaderConfig
{
    int local_size_x;
    int local_size_y;
    int local_size_z;
    int block_height;
    int block_width;
    int block_depth;
};

class OpSoftmax: public OpBase
{
public:
    OpSoftmax(const int axis, const bool log_softmax = false);
    ~OpSoftmax();
    void reshapeOutTensor(Tensor& in, Tensor& out);
    bool forward(Tensor& in, Tensor& out);
    virtual bool forward(std::vector<Tensor>& ins,
                         std::vector<Tensor>& blobs,
                         std::vector<Tensor>& outs) CV_OVERRIDE;
private:
    bool init(const int axis, const bool log_softmax);
    bool computeGroupCount();

    int axis_;
    int channels_;
    int channel_size_;
    int outer_size_;
    bool log_softmax_;
    SoftmaxShaderConfig config_;
    Tensor* max_tensor_;
    Tensor* sum_tensor_;
};

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom

#endif // OPENCV_DNN_VKCOM_OP_SOFTMAX_HPP
