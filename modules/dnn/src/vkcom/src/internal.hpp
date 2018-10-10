// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_VKCOM_INTERNAL_HPP
#define OPENCV_DNN_VKCOM_INTERNAL_HPP

#include <float.h>
#include "../include/vkcom.hpp"
#include "context.hpp"

#ifdef USE_SHADERC
#include "shaderc/shaderc.hpp"
#else
typedef int shaderc_shader_kind;
#define shaderc_compute_shader 0
#endif

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

Context* getContext();
VkPhysicalDevice getPhysicalDevice();
std::vector<uint32_t> compile(const std::string& name,
                              shaderc_shader_kind kind,
                              const std::string& data);
void bindTensor(VkDevice& device, Tensor& tensor, int binding, VkDescriptorSet descriptor_set);
void computeConvOutputShapeAndPadding(const PaddingMode& padding_mode,
                                      int& padding_top, int& padding_left,
                                      const int& in_h, const int& in_w,
                                      const int& filter_h, const int& filter_w,
                                      const int& dilation_h, const int& dilation_w,
                                      const int& stride_h, const int& stride_w,
                                      int& out_h, int& out_w);
void computePoolOutputShape(const PaddingMode& padding_mode,
                            const int& padding_top, const int& padding_left,
                            const int& in_h, const int& in_w,
                            const int& filter_h, const int& filter_w,
                            const int& stride_h, const int& stride_w,
                            int& out_h, int& out_w);

inline bool checkFormat(Format fmt)
{
    return (fmt > -1 && fmt < kFormatNum) ? true : false;
}

inline size_t elementSize(Format fmt)
{
    if (fmt == kFormatFp32 || fmt == kFormatInt32)
    {
        return 4;
    }
    else if (fmt >= 0 && fmt < kFormatNum)
    {
        CV_LOG_WARNING(NULL, format("Unsupported format %d", fmt));
    }
    else
    {
        CV_Error(Error::StsError, format("Invalid format %d", fmt));
    }
    return 0;
}

inline int shapeCount(const Shape& shape, int start = -1, int end = -1)
{
    if (start == -1) start = 0;
    if (end == -1) end = (int)shape.size();

    if (shape.empty())
        return 0;

    int elems = 1;
    assert(start <= (int)shape.size() &&
           end <= (int)shape.size() &&
           start <= end);
    for(int i = start; i < end; i++)
    {
        elems *= shape[i];
    }
    return elems;
}

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom

#endif // OPENCV_DNN_VKCOM_INTERNAL_HPP
