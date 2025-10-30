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
    kConvShaderTypeGeneric = 0,
    kConvShaderTypeDepthWise = 2, // special branch
    kConvShaderTypeWinograd = 3,
    kConvShaderTest = 4,
};

struct ConvShaderConfig
{
    int local_size_x;
    int local_size_y;
    int local_size_z;
};

// Current Vulkan Convolution layer only support Conv2D.
class OpConv : public OpBase
{
public:
    OpConv(const Mat& weightBlob, const std::vector<float>& biasvec, int activType, const int ngroups, const int K, const int C, const int Hk, const int Wk,
           const int stride_h, const int stride_w, const int dilation_h, const int dilation_w,
           const int pad_left, const int pad_top);
    ~OpConv();

    void firstForward(); // Execute only in the first forward.
    virtual bool forward(std::vector<Tensor>& ins, std::vector<Tensor>& outs) CV_OVERRIDE;

    std::vector<float> biasCopy;
    Ptr<Tensor> weightTensorPtr;
    Ptr<Tensor> biasTensorPtr;

private:
    bool computeGroupCount();

    FusedActivationType activ;
    const int ngroups;
    const int K, C, Hk, Wk; // output channel, input channel, height of kernel, width of kernel.
    const int stride_h, stride_w;
    const int dilation_h, dilation_w;
    const int pad_left, pad_top;

    int H0, W0;
    int Hi, Wi;
    int batch;
    int Kg, Cg;
    int CgHkWk, CgHkWk_aligned, ksize;

    int STRIP_LEN;
    bool fast_1x1 = false;

    ConvShaderType shaderType;
    ConvShaderConfig config;
    bool firstForwardFinsh = false;
};

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom

#endif // OPENCV_DNN_VKCOM_OP_CONV_HPP
