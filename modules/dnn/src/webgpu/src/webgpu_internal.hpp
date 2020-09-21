// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_WEBGPU_INTERNAL_HPP
#define OPENCV_DNN_WEBGPU_INTERNAL_HPP

#include <float.h>
#include "../include/webgpu_common.hpp"
#include "webgpu_context.hpp"

#ifdef USE_SHADERC
#include "shaderc/shaderc.hpp"
#else
typedef int shaderc_shader_kind;
#define shaderc_compute_shader 0
#endif

namespace cv { namespace dnn { namespace webgpu {

#ifdef HAVE_WEBGPU
std::vector<uint32_t> compile(const std::string& name,
                              shaderc_shader_kind kind,
                              const std::string& data);
void bindTensor(Tensor& tensor, uint32_t binding,
                std::vector<wgpu::BindGroupEntry>& bgEntries);
void bindUniform(Buffer& buffer, uint32_t binding,
                 std::vector<wgpu::BindGroupEntry>& bgEntries);
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
    return (fmt > -1 && fmt < wFormatNum) ? true : false;
}

inline size_t elementSize(Format fmt)
{
    if (fmt == wFormatFp32 || fmt == wFormatInt32)
    {
        return 4;
    }
    else if (fmt >= 0 && fmt < wFormatNum)
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
#endif  // HAVE_WEBGPU

}}} //namespace::dnn::webgpu

#endif  //OPENCV_DNN_WEBGPU_INTERNEL_HPP