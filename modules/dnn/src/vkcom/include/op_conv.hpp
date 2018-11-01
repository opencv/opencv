// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_VKCOM_OP_CONV_HPP
#define OPENCV_DNN_VKCOM_OP_CONV_HPP

#include "vkcom.hpp"
#include "op_base.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

enum ConvShaderType
{
    kConvShaderTypeBasic = 0,
    kConvShaderTypeIDLF = 1,
    kConvShaderTypeNum
};

struct ConvShaderConfig
{
    int local_size_x;
    int local_size_y;
    int local_size_z;
    int block_height;
    int block_width;
    int block_depth;
    ConvShaderType shader_type;
};

class OpConv : public OpBase
{
public:
    OpConv(const int out_channel, const bool has_bias,
           const int* filter_size, const int* pad,
           const int* stride, const int* dilation,
           const int activation, const int group,
           const int padding_mode);
    void reshapeOutTensor(Tensor& in, Tensor& out);
    bool forward(Tensor& in, Tensor& filter_weights, Tensor& bias, Tensor& out);
    virtual bool forward(std::vector<Tensor>& ins,
                         std::vector<Tensor>& blobs,
                         std::vector<Tensor>& outs) CV_OVERRIDE;
private:
    bool init(const int out_channel, const bool has_bias,
              const int* filter_size, const int* pad,
              const int* stride, const int* dilation,
              const int activation, const int group,
              const int padding_mode);
    bool computeGroupCount();

    int batch_;
    int in_height_;
    int in_width_;
    int in_channel_;
    int out_height_;
    int out_width_;
    int out_channel_;
    int filter_height_;
    int filter_width_;
    int stride_height_;
    int stride_width_;
    int padding_top_;
    int padding_left_;
    int dilation_height_;
    int dilation_width_;
    int activation_;
    PaddingMode padding_mode_;
    int group_;
    int has_bias_;
    Tensor swizzled_weights;
    ConvShaderConfig config_;
    bool dwconv_;
};

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom

#endif // OPENCV_DNN_VKCOM_OP_CONV_HPP
