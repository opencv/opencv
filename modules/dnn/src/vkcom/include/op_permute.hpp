// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_VKCOM_OP_PERMUTE_HPP
#define OPENCV_DNN_VKCOM_OP_PERMUTE_HPP

#include "vkcom.hpp"
#include "op_base.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

class OpPermute: public OpBase
{
public:
    OpPermute(std::vector<size_t>& order);
    bool forward(std::vector<Tensor>& ins, std::vector<Tensor>& outs);
    void reshapeOutTensor(std::vector<Tensor *>& in, std::vector<Tensor>& outs);
    virtual bool forward(std::vector<Tensor>& ins,
                         std::vector<Tensor>& blobs,
                         std::vector<Tensor>& outs) CV_OVERRIDE;

private:
    void prepareStrides(const Shape &shape_before, const Shape &shape_after);
    bool computeGroupCount();

    std::vector<int> order_;
    bool need_permute_;
    int global_size_;
    int nthreads_;
    int dims_;
    Tensor tensor_order_;
    Tensor tensor_old_stride_;
    Tensor tensor_new_stride_;
    std::vector<int> old_stride_;
    std::vector<int> new_stride_;
    Shape in_shape_;
    Shape out_shape_;
};

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom

#endif // OPENCV_DNN_VKCOM_OP_PERMUTE_HPP
