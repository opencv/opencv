/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

using namespace cv;
using namespace cv::gpu;

#if !defined (HAVE_CUDA)

void cv::gpu::cvtColor(const GpuMat&, GpuMat&, int, int, Stream&) { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

BEGIN_OPENCV_DEVICE_NAMESPACE

#define OPENCV_GPU_DECLARE_CVTCOLOR_ONE(name) \
    void name(const DevMem2Db& src, const DevMem2Db& dst, cudaStream_t stream);

#define OPENCV_GPU_DECLARE_CVTCOLOR_ALL(name) \
    OPENCV_GPU_DECLARE_CVTCOLOR_ONE(name ## _8u) \
    OPENCV_GPU_DECLARE_CVTCOLOR_ONE(name ## _16u) \
    OPENCV_GPU_DECLARE_CVTCOLOR_ONE(name ## _32f)

#define OPENCV_GPU_DECLARE_CVTCOLOR_8U32F(name) \
    OPENCV_GPU_DECLARE_CVTCOLOR_ONE(name ## _8u) \
    OPENCV_GPU_DECLARE_CVTCOLOR_ONE(name ## _32f) \
    OPENCV_GPU_DECLARE_CVTCOLOR_ONE(name ## _full_8u) \
    OPENCV_GPU_DECLARE_CVTCOLOR_ONE(name ## _full_32f)

OPENCV_GPU_DECLARE_CVTCOLOR_ALL(bgr_to_rgb)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(bgr_to_bgra)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(bgr_to_rgba)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(bgra_to_bgr)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(bgra_to_rgb)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(bgra_to_rgba)

OPENCV_GPU_DECLARE_CVTCOLOR_ONE(bgr_to_bgr555)
OPENCV_GPU_DECLARE_CVTCOLOR_ONE(bgr_to_bgr565)
OPENCV_GPU_DECLARE_CVTCOLOR_ONE(rgb_to_bgr555)
OPENCV_GPU_DECLARE_CVTCOLOR_ONE(rgb_to_bgr565)
OPENCV_GPU_DECLARE_CVTCOLOR_ONE(bgra_to_bgr555)
OPENCV_GPU_DECLARE_CVTCOLOR_ONE(bgra_to_bgr565)
OPENCV_GPU_DECLARE_CVTCOLOR_ONE(rgba_to_bgr555)
OPENCV_GPU_DECLARE_CVTCOLOR_ONE(rgba_to_bgr565)

OPENCV_GPU_DECLARE_CVTCOLOR_ONE(bgr555_to_rgb)
OPENCV_GPU_DECLARE_CVTCOLOR_ONE(bgr565_to_rgb)
OPENCV_GPU_DECLARE_CVTCOLOR_ONE(bgr555_to_bgr)
OPENCV_GPU_DECLARE_CVTCOLOR_ONE(bgr565_to_bgr)
OPENCV_GPU_DECLARE_CVTCOLOR_ONE(bgr555_to_rgba)
OPENCV_GPU_DECLARE_CVTCOLOR_ONE(bgr565_to_rgba)
OPENCV_GPU_DECLARE_CVTCOLOR_ONE(bgr555_to_bgra)
OPENCV_GPU_DECLARE_CVTCOLOR_ONE(bgr565_to_bgra)

OPENCV_GPU_DECLARE_CVTCOLOR_ALL(gray_to_bgr)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(gray_to_bgra)

OPENCV_GPU_DECLARE_CVTCOLOR_ONE(gray_to_bgr555)
OPENCV_GPU_DECLARE_CVTCOLOR_ONE(gray_to_bgr565)

OPENCV_GPU_DECLARE_CVTCOLOR_ONE(bgr555_to_gray)
OPENCV_GPU_DECLARE_CVTCOLOR_ONE(bgr565_to_gray)

OPENCV_GPU_DECLARE_CVTCOLOR_ALL(rgb_to_gray)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(bgr_to_gray)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(rgba_to_gray)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(bgra_to_gray)

OPENCV_GPU_DECLARE_CVTCOLOR_ALL(rgb_to_yuv)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(rgba_to_yuv)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(rgb_to_yuv4)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(rgba_to_yuv4)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(bgr_to_yuv)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(bgra_to_yuv)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(bgr_to_yuv4)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(bgra_to_yuv4)

OPENCV_GPU_DECLARE_CVTCOLOR_ALL(yuv_to_rgb)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(yuv_to_rgba)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(yuv4_to_rgb)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(yuv4_to_rgba)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(yuv_to_bgr)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(yuv_to_bgra)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(yuv4_to_bgr)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(yuv4_to_bgra)

OPENCV_GPU_DECLARE_CVTCOLOR_ALL(rgb_to_YCrCb)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(rgba_to_YCrCb)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(rgb_to_YCrCb4)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(rgba_to_YCrCb4)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(bgr_to_YCrCb)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(bgra_to_YCrCb)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(bgr_to_YCrCb4)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(bgra_to_YCrCb4)

OPENCV_GPU_DECLARE_CVTCOLOR_ALL(YCrCb_to_rgb)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(YCrCb_to_rgba)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(YCrCb4_to_rgb)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(YCrCb4_to_rgba)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(YCrCb_to_bgr)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(YCrCb_to_bgra)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(YCrCb4_to_bgr)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(YCrCb4_to_bgra)

OPENCV_GPU_DECLARE_CVTCOLOR_ALL(rgb_to_xyz)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(rgba_to_xyz)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(rgb_to_xyz4)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(rgba_to_xyz4)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(bgr_to_xyz)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(bgra_to_xyz)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(bgr_to_xyz4)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(bgra_to_xyz4)

OPENCV_GPU_DECLARE_CVTCOLOR_ALL(xyz_to_rgb)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(xyz4_to_rgb)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(xyz_to_rgba)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(xyz4_to_rgba)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(xyz_to_bgr)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(xyz4_to_bgr)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(xyz_to_bgra)
OPENCV_GPU_DECLARE_CVTCOLOR_ALL(xyz4_to_bgra)

OPENCV_GPU_DECLARE_CVTCOLOR_8U32F(rgb_to_hsv)
OPENCV_GPU_DECLARE_CVTCOLOR_8U32F(rgba_to_hsv)
OPENCV_GPU_DECLARE_CVTCOLOR_8U32F(rgb_to_hsv4)
OPENCV_GPU_DECLARE_CVTCOLOR_8U32F(rgba_to_hsv4)
OPENCV_GPU_DECLARE_CVTCOLOR_8U32F(bgr_to_hsv)
OPENCV_GPU_DECLARE_CVTCOLOR_8U32F(bgra_to_hsv)
OPENCV_GPU_DECLARE_CVTCOLOR_8U32F(bgr_to_hsv4)
OPENCV_GPU_DECLARE_CVTCOLOR_8U32F(bgra_to_hsv4)

OPENCV_GPU_DECLARE_CVTCOLOR_8U32F(hsv_to_rgb)
OPENCV_GPU_DECLARE_CVTCOLOR_8U32F(hsv_to_rgba)
OPENCV_GPU_DECLARE_CVTCOLOR_8U32F(hsv4_to_rgb)
OPENCV_GPU_DECLARE_CVTCOLOR_8U32F(hsv4_to_rgba)
OPENCV_GPU_DECLARE_CVTCOLOR_8U32F(hsv_to_bgr)
OPENCV_GPU_DECLARE_CVTCOLOR_8U32F(hsv_to_bgra)
OPENCV_GPU_DECLARE_CVTCOLOR_8U32F(hsv4_to_bgr)
OPENCV_GPU_DECLARE_CVTCOLOR_8U32F(hsv4_to_bgra)

OPENCV_GPU_DECLARE_CVTCOLOR_8U32F(rgb_to_hls)
OPENCV_GPU_DECLARE_CVTCOLOR_8U32F(rgba_to_hls)
OPENCV_GPU_DECLARE_CVTCOLOR_8U32F(rgb_to_hls4)
OPENCV_GPU_DECLARE_CVTCOLOR_8U32F(rgba_to_hls4)
OPENCV_GPU_DECLARE_CVTCOLOR_8U32F(bgr_to_hls)
OPENCV_GPU_DECLARE_CVTCOLOR_8U32F(bgra_to_hls)
OPENCV_GPU_DECLARE_CVTCOLOR_8U32F(bgr_to_hls4)
OPENCV_GPU_DECLARE_CVTCOLOR_8U32F(bgra_to_hls4)

OPENCV_GPU_DECLARE_CVTCOLOR_8U32F(hls_to_rgb)
OPENCV_GPU_DECLARE_CVTCOLOR_8U32F(hls_to_rgba)
OPENCV_GPU_DECLARE_CVTCOLOR_8U32F(hls4_to_rgb)
OPENCV_GPU_DECLARE_CVTCOLOR_8U32F(hls4_to_rgba)
OPENCV_GPU_DECLARE_CVTCOLOR_8U32F(hls_to_bgr)
OPENCV_GPU_DECLARE_CVTCOLOR_8U32F(hls_to_bgra)
OPENCV_GPU_DECLARE_CVTCOLOR_8U32F(hls4_to_bgr)
OPENCV_GPU_DECLARE_CVTCOLOR_8U32F(hls4_to_bgra)

#undef OPENCV_GPU_DECLARE_CVTCOLOR_ONE
#undef OPENCV_GPU_DECLARE_CVTCOLOR_ALL
#undef OPENCV_GPU_DECLARE_CVTCOLOR_8U32F

END_OPENCV_DEVICE_NAMESPACE

using namespace OPENCV_DEVICE_NAMESPACE;

namespace
{
    typedef void (*gpu_func_t)(const DevMem2Db& src, const DevMem2Db& dst, cudaStream_t stream);

    void bgr_to_rgb(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[] = {bgr_to_rgb_8u, 0, bgr_to_rgb_16u, 0, 0, bgr_to_rgb_32f};
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), 3));        

        funcs[src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgr_to_bgra(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[] = {bgr_to_bgra_8u, 0, bgr_to_bgra_16u, 0, 0, bgr_to_bgra_32f};
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), 4));        

        funcs[src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgr_to_rgba(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[] = {bgr_to_rgba_8u, 0, bgr_to_rgba_16u, 0, 0, bgr_to_rgba_32f};
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), 4));        

        funcs[src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgra_to_bgr(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[] = {bgra_to_bgr_8u, 0, bgra_to_bgr_16u, 0, 0, bgra_to_bgr_32f};
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 4);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), 3));        

        funcs[src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgra_to_rgb(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[] = {bgra_to_rgb_8u, 0, bgra_to_rgb_16u, 0, 0, bgra_to_rgb_32f};
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 4);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), 3));        

        funcs[src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgra_to_rgba(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[] = {bgra_to_rgba_8u, 0, bgra_to_rgba_16u, 0, 0, bgra_to_rgba_32f};
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 4);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), 4));        

        funcs[src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgr_to_bgr555(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {        
        CV_Assert(src.depth() == CV_8U);
        CV_Assert(src.channels() == 3);

        dst.create(src.size(), CV_8UC2);        

        device::bgr_to_bgr555(src, dst, StreamAccessor::getStream(stream));
    }

    void bgr_to_bgr565(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {        
        CV_Assert(src.depth() == CV_8U);
        CV_Assert(src.channels() == 3);

        dst.create(src.size(), CV_8UC2);        

        device::bgr_to_bgr565(src, dst, StreamAccessor::getStream(stream));
    }

    void rgb_to_bgr555(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {        
        CV_Assert(src.depth() == CV_8U);
        CV_Assert(src.channels() == 3);

        dst.create(src.size(), CV_8UC2);        

        device::rgb_to_bgr555(src, dst, StreamAccessor::getStream(stream));
    }

    void rgb_to_bgr565(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {        
        CV_Assert(src.depth() == CV_8U);
        CV_Assert(src.channels() == 3);

        dst.create(src.size(), CV_8UC2);        

        device::rgb_to_bgr565(src, dst, StreamAccessor::getStream(stream));
    }

    void bgra_to_bgr555(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {        
        CV_Assert(src.depth() == CV_8U);
        CV_Assert(src.channels() == 4);

        dst.create(src.size(), CV_8UC2);        

        device::bgra_to_bgr555(src, dst, StreamAccessor::getStream(stream));
    }

    void bgra_to_bgr565(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {        
        CV_Assert(src.depth() == CV_8U);
        CV_Assert(src.channels() == 4);

        dst.create(src.size(), CV_8UC2);        

        device::bgra_to_bgr565(src, dst, StreamAccessor::getStream(stream));
    }

    void rgba_to_bgr555(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {        
        CV_Assert(src.depth() == CV_8U);
        CV_Assert(src.channels() == 4);

        dst.create(src.size(), CV_8UC2);        

        device::rgba_to_bgr555(src, dst, StreamAccessor::getStream(stream));
    }

    void rgba_to_bgr565(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {        
        CV_Assert(src.depth() == CV_8U);
        CV_Assert(src.channels() == 4);

        dst.create(src.size(), CV_8UC2);        

        device::rgba_to_bgr565(src, dst, StreamAccessor::getStream(stream));
    }

    void bgr555_to_rgb(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {        
        CV_Assert(src.depth() == CV_8U);
        CV_Assert(src.channels() == 2);

        dst.create(src.size(), CV_8UC3);        

        device::bgr555_to_rgb(src, dst, StreamAccessor::getStream(stream));
    }

    void bgr565_to_rgb(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {        
        CV_Assert(src.depth() == CV_8U);
        CV_Assert(src.channels() == 2);

        dst.create(src.size(), CV_8UC3);        

        device::bgr565_to_rgb(src, dst, StreamAccessor::getStream(stream));
    }

    void bgr555_to_bgr(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {        
        CV_Assert(src.depth() == CV_8U);
        CV_Assert(src.channels() == 2);

        dst.create(src.size(), CV_8UC3);        

        device::bgr555_to_bgr(src, dst, StreamAccessor::getStream(stream));
    }

    void bgr565_to_bgr(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {        
        CV_Assert(src.depth() == CV_8U);
        CV_Assert(src.channels() == 2);

        dst.create(src.size(), CV_8UC3);        

        device::bgr565_to_bgr(src, dst, StreamAccessor::getStream(stream));
    }

    void bgr555_to_rgba(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {        
        CV_Assert(src.depth() == CV_8U);
        CV_Assert(src.channels() == 2);

        dst.create(src.size(), CV_8UC4);        

        device::bgr555_to_rgba(src, dst, StreamAccessor::getStream(stream));
    }

    void bgr565_to_rgba(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {        
        CV_Assert(src.depth() == CV_8U);
        CV_Assert(src.channels() == 2);

        dst.create(src.size(), CV_8UC4);        

        device::bgr565_to_rgba(src, dst, StreamAccessor::getStream(stream));
    }

    void bgr555_to_bgra(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {        
        CV_Assert(src.depth() == CV_8U);
        CV_Assert(src.channels() == 2);

        dst.create(src.size(), CV_8UC4);        

        device::bgr555_to_bgra(src, dst, StreamAccessor::getStream(stream));
    }

    void bgr565_to_bgra(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {        
        CV_Assert(src.depth() == CV_8U);
        CV_Assert(src.channels() == 2);

        dst.create(src.size(), CV_8UC4);        

        device::bgr565_to_bgra(src, dst, StreamAccessor::getStream(stream));
    }

    void gray_to_bgr(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[] = {gray_to_bgr_8u, 0, gray_to_bgr_16u, 0, 0, gray_to_bgr_32f};
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 1);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), 3));        

        funcs[src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void gray_to_bgra(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[] = {gray_to_bgra_8u, 0, gray_to_bgra_16u, 0, 0, gray_to_bgra_32f};
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 1);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), 4));        

        funcs[src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void gray_to_bgr555(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {        
        CV_Assert(src.depth() == CV_8U);
        CV_Assert(src.channels() == 1);

        dst.create(src.size(), CV_8UC2);        

        device::gray_to_bgr555(src, dst, StreamAccessor::getStream(stream));
    }

    void gray_to_bgr565(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {        
        CV_Assert(src.depth() == CV_8U);
        CV_Assert(src.channels() == 1);

        dst.create(src.size(), CV_8UC2);        

        device::gray_to_bgr565(src, dst, StreamAccessor::getStream(stream));
    }

    void bgr555_to_gray(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {        
        CV_Assert(src.depth() == CV_8U);
        CV_Assert(src.channels() == 2);

        dst.create(src.size(), CV_8UC1);        

        device::bgr555_to_gray(src, dst, StreamAccessor::getStream(stream));
    }

    void bgr565_to_gray(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {        
        CV_Assert(src.depth() == CV_8U);
        CV_Assert(src.channels() == 2);

        dst.create(src.size(), CV_8UC1);        

        device::bgr565_to_gray(src, dst, StreamAccessor::getStream(stream));
    }

    void rgb_to_gray(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[] = {rgb_to_gray_8u, 0, rgb_to_gray_16u, 0, 0, rgb_to_gray_32f};
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), 1));        

        funcs[src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgr_to_gray(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[] = {bgr_to_gray_8u, 0, bgr_to_gray_16u, 0, 0, bgr_to_gray_32f};
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), 1));        

        funcs[src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void rgba_to_gray(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[] = {rgba_to_gray_8u, 0, rgba_to_gray_16u, 0, 0, rgba_to_gray_32f};
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 4);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), 1));        

        funcs[src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgra_to_gray(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[] = {bgra_to_gray_8u, 0, bgra_to_gray_16u, 0, 0, bgra_to_gray_32f};
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 4);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), 1));        

        funcs[src.depth()](src, dst, StreamAccessor::getStream(stream));
    }
    
    void rgb_to_yuv(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] = 
        {
            {
                {rgb_to_yuv_8u, 0, rgb_to_yuv_16u, 0, 0, rgb_to_yuv_32f},
                {rgba_to_yuv_8u, 0, rgba_to_yuv_16u, 0, 0, rgba_to_yuv_32f}
            },
            {
                {rgb_to_yuv4_8u, 0, rgb_to_yuv4_16u, 0, 0, rgb_to_yuv4_32f},
                {rgba_to_yuv4_8u, 0, rgba_to_yuv4_16u, 0, 0, rgba_to_yuv4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));        

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgr_to_yuv(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] = 
        {
            {
                {bgr_to_yuv_8u, 0, bgr_to_yuv_16u, 0, 0, bgr_to_yuv_32f},
                {bgra_to_yuv_8u, 0, bgra_to_yuv_16u, 0, 0, bgra_to_yuv_32f}
            },
            {
                {bgr_to_yuv4_8u, 0, bgr_to_yuv4_16u, 0, 0, bgr_to_yuv4_32f},
                {bgra_to_yuv4_8u, 0, bgra_to_yuv4_16u, 0, 0, bgra_to_yuv4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));        

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void yuv_to_rgb(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] = 
        {
            {
                {yuv_to_rgb_8u, 0, yuv_to_rgb_16u, 0, 0, yuv_to_rgb_32f},
                {yuv4_to_rgb_8u, 0, yuv4_to_rgb_16u, 0, 0, yuv4_to_rgb_32f}
            },
            {
                {yuv_to_rgba_8u, 0, yuv_to_rgba_16u, 0, 0, yuv_to_rgba_32f},
                {yuv4_to_rgba_8u, 0, yuv4_to_rgba_16u, 0, 0, yuv4_to_rgba_32f}
            }
        };

        if (dcn <= 0) dcn = 3;
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));        

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void yuv_to_bgr(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] = 
        {
            {
                {yuv_to_bgr_8u, 0, yuv_to_bgr_16u, 0, 0, yuv_to_bgr_32f},
                {yuv4_to_bgr_8u, 0, yuv4_to_bgr_16u, 0, 0, yuv4_to_bgr_32f}
            },
            {
                {yuv_to_bgra_8u, 0, yuv_to_bgra_16u, 0, 0, yuv_to_bgra_32f},
                {yuv4_to_bgra_8u, 0, yuv4_to_bgra_16u, 0, 0, yuv4_to_bgra_32f}
            }
        };

        if (dcn <= 0) dcn = 3;
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));        

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }
    
    void rgb_to_YCrCb(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] = 
        {
            {
                {rgb_to_YCrCb_8u, 0, rgb_to_YCrCb_16u, 0, 0, rgb_to_YCrCb_32f},
                {rgba_to_YCrCb_8u, 0, rgba_to_YCrCb_16u, 0, 0, rgba_to_YCrCb_32f}
            },
            {
                {rgb_to_YCrCb4_8u, 0, rgb_to_YCrCb4_16u, 0, 0, rgb_to_YCrCb4_32f},
                {rgba_to_YCrCb4_8u, 0, rgba_to_YCrCb4_16u, 0, 0, rgba_to_YCrCb4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));        

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgr_to_YCrCb(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] = 
        {
            {
                {bgr_to_YCrCb_8u, 0, bgr_to_YCrCb_16u, 0, 0, bgr_to_YCrCb_32f},
                {bgra_to_YCrCb_8u, 0, bgra_to_YCrCb_16u, 0, 0, bgra_to_YCrCb_32f}
            },
            {
                {bgr_to_YCrCb4_8u, 0, bgr_to_YCrCb4_16u, 0, 0, bgr_to_YCrCb4_32f},
                {bgra_to_YCrCb4_8u, 0, bgra_to_YCrCb4_16u, 0, 0, bgra_to_YCrCb4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));        

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void YCrCb_to_rgb(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] = 
        {
            {
                {YCrCb_to_rgb_8u, 0, YCrCb_to_rgb_16u, 0, 0, YCrCb_to_rgb_32f},
                {YCrCb4_to_rgb_8u, 0, YCrCb4_to_rgb_16u, 0, 0, YCrCb4_to_rgb_32f}
            },
            {
                {YCrCb_to_rgba_8u, 0, YCrCb_to_rgba_16u, 0, 0, YCrCb_to_rgba_32f},
                {YCrCb4_to_rgba_8u, 0, YCrCb4_to_rgba_16u, 0, 0, YCrCb4_to_rgba_32f}
            }
        };

        if (dcn <= 0) dcn = 3;
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));        

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void YCrCb_to_bgr(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] = 
        {
            {
                {YCrCb_to_bgr_8u, 0, YCrCb_to_bgr_16u, 0, 0, YCrCb_to_bgr_32f},
                {YCrCb4_to_bgr_8u, 0, YCrCb4_to_bgr_16u, 0, 0, YCrCb4_to_bgr_32f}
            },
            {
                {YCrCb_to_bgra_8u, 0, YCrCb_to_bgra_16u, 0, 0, YCrCb_to_bgra_32f},
                {YCrCb4_to_bgra_8u, 0, YCrCb4_to_bgra_16u, 0, 0, YCrCb4_to_bgra_32f}
            }
        };

        if (dcn <= 0) dcn = 3;
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));        

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void rgb_to_xyz(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] = 
        {
            {
                {rgb_to_xyz_8u, 0, rgb_to_xyz_16u, 0, 0, rgb_to_xyz_32f},
                {rgba_to_xyz_8u, 0, rgba_to_xyz_16u, 0, 0, rgba_to_xyz_32f}
            },
            {
                {rgb_to_xyz4_8u, 0, rgb_to_xyz4_16u, 0, 0, rgb_to_xyz4_32f},
                {rgba_to_xyz4_8u, 0, rgba_to_xyz4_16u, 0, 0, rgba_to_xyz4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));        

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgr_to_xyz(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] = 
        {
            {
                {bgr_to_xyz_8u, 0, bgr_to_xyz_16u, 0, 0, bgr_to_xyz_32f},
                {bgra_to_xyz_8u, 0, bgra_to_xyz_16u, 0, 0, bgra_to_xyz_32f}
            },
            {
                {bgr_to_xyz4_8u, 0, bgr_to_xyz4_16u, 0, 0, bgr_to_xyz4_32f},
                {bgra_to_xyz4_8u, 0, bgra_to_xyz4_16u, 0, 0, bgra_to_xyz4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));        

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void xyz_to_rgb(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] = 
        {
            {
                {xyz_to_rgb_8u, 0, xyz_to_rgb_16u, 0, 0, xyz_to_rgb_32f},
                {xyz4_to_rgb_8u, 0, xyz4_to_rgb_16u, 0, 0, xyz4_to_rgb_32f}
            },
            {
                {xyz_to_rgba_8u, 0, xyz_to_rgba_16u, 0, 0, xyz_to_rgba_32f},
                {xyz4_to_rgba_8u, 0, xyz4_to_rgba_16u, 0, 0, xyz4_to_rgba_32f}
            }
        };

        if (dcn <= 0) dcn = 3;
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));        

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void xyz_to_bgr(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] = 
        {
            {
                {xyz_to_bgr_8u, 0, xyz_to_bgr_16u, 0, 0, xyz_to_bgr_32f},
                {xyz4_to_bgr_8u, 0, xyz4_to_bgr_16u, 0, 0, xyz4_to_bgr_32f}
            },
            {
                {xyz_to_bgra_8u, 0, xyz_to_bgra_16u, 0, 0, xyz_to_bgra_32f},
                {xyz4_to_bgra_8u, 0, xyz4_to_bgra_16u, 0, 0, xyz4_to_bgra_32f}
            }
        };

        if (dcn <= 0) dcn = 3;
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));        

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void rgb_to_hsv(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] = 
        {
            {
                {rgb_to_hsv_8u, 0, 0, 0, 0, rgb_to_hsv_32f},
                {rgba_to_hsv_8u, 0, 0, 0, 0, rgba_to_hsv_32f},
            },
            {
                {rgb_to_hsv4_8u, 0, 0, 0, 0, rgb_to_hsv4_32f},
                {rgba_to_hsv4_8u, 0, 0, 0, 0, rgba_to_hsv4_32f},
            }
        };

        if (dcn <= 0) dcn = 3;
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));        

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgr_to_hsv(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] = 
        {
            {
                {bgr_to_hsv_8u, 0, 0, 0, 0, bgr_to_hsv_32f},
                {bgra_to_hsv_8u, 0, 0, 0, 0, bgra_to_hsv_32f}
            },
            {
                {bgr_to_hsv4_8u, 0, 0, 0, 0, bgr_to_hsv4_32f},
                {bgra_to_hsv4_8u, 0, 0, 0, 0, bgra_to_hsv4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));        

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void hsv_to_rgb(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] = 
        {
            {
                {hsv_to_rgb_8u, 0, 0, 0, 0, hsv_to_rgb_32f},
                {hsv4_to_rgb_8u, 0, 0, 0, 0, hsv4_to_rgb_32f}
            },
            {
                {hsv_to_rgba_8u, 0, 0, 0, 0, hsv_to_rgba_32f},
                {hsv4_to_rgba_8u, 0, 0, 0, 0, hsv4_to_rgba_32f}
            }
        };

        if (dcn <= 0) dcn = 3;
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));        

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void hsv_to_bgr(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] = 
        {
            {
                {hsv_to_bgr_8u, 0, 0, 0, 0, hsv_to_bgr_32f},
                {hsv4_to_bgr_8u, 0, 0, 0, 0, hsv4_to_bgr_32f}
            },
            {
                {hsv_to_bgra_8u, 0, 0, 0, 0, hsv_to_bgra_32f},
                {hsv4_to_bgra_8u, 0, 0, 0, 0, hsv4_to_bgra_32f}
            }
        };

        if (dcn <= 0) dcn = 3;
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));        

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }    

    void rgb_to_hls(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] = 
        {
            {
                {rgb_to_hls_8u, 0, 0, 0, 0, rgb_to_hls_32f},
                {rgba_to_hls_8u, 0, 0, 0, 0, rgba_to_hls_32f},
            },
            {
                {rgb_to_hls4_8u, 0, 0, 0, 0, rgb_to_hls4_32f},
                {rgba_to_hls4_8u, 0, 0, 0, 0, rgba_to_hls4_32f},
            }
        };

        if (dcn <= 0) dcn = 3;
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));        

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgr_to_hls(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] = 
        {
            {
                {bgr_to_hls_8u, 0, 0, 0, 0, bgr_to_hls_32f},
                {bgra_to_hls_8u, 0, 0, 0, 0, bgra_to_hls_32f}
            },
            {
                {bgr_to_hls4_8u, 0, 0, 0, 0, bgr_to_hls4_32f},
                {bgra_to_hls4_8u, 0, 0, 0, 0, bgra_to_hls4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));        

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void hls_to_rgb(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] = 
        {
            {
                {hls_to_rgb_8u, 0, 0, 0, 0, hls_to_rgb_32f},
                {hls4_to_rgb_8u, 0, 0, 0, 0, hls4_to_rgb_32f}
            },
            {
                {hls_to_rgba_8u, 0, 0, 0, 0, hls_to_rgba_32f},
                {hls4_to_rgba_8u, 0, 0, 0, 0, hls4_to_rgba_32f}
            }
        };

        if (dcn <= 0) dcn = 3;
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));        

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void hls_to_bgr(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] = 
        {
            {
                {hls_to_bgr_8u, 0, 0, 0, 0, hls_to_bgr_32f},
                {hls4_to_bgr_8u, 0, 0, 0, 0, hls4_to_bgr_32f}
            },
            {
                {hls_to_bgra_8u, 0, 0, 0, 0, hls_to_bgra_32f},
                {hls4_to_bgra_8u, 0, 0, 0, 0, hls4_to_bgra_32f}
            }
        };

        if (dcn <= 0) dcn = 3;
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));        

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }   

    void rgb_to_hsv_full(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] = 
        {
            {
                {rgb_to_hsv_full_8u, 0, 0, 0, 0, rgb_to_hsv_full_32f},
                {rgba_to_hsv_full_8u, 0, 0, 0, 0, rgba_to_hsv_full_32f},
            },
            {
                {rgb_to_hsv4_full_8u, 0, 0, 0, 0, rgb_to_hsv4_full_32f},
                {rgba_to_hsv4_full_8u, 0, 0, 0, 0, rgba_to_hsv4_full_32f},
            }
        };

        if (dcn <= 0) dcn = 3;
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));        

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgr_to_hsv_full(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] = 
        {
            {
                {bgr_to_hsv_full_8u, 0, 0, 0, 0, bgr_to_hsv_full_32f},
                {bgra_to_hsv_full_8u, 0, 0, 0, 0, bgra_to_hsv_full_32f}
            },
            {
                {bgr_to_hsv4_full_8u, 0, 0, 0, 0, bgr_to_hsv4_full_32f},
                {bgra_to_hsv4_full_8u, 0, 0, 0, 0, bgra_to_hsv4_full_32f}
            }
        };

        if (dcn <= 0) dcn = 3;
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));        

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void hsv_to_rgb_full(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] = 
        {
            {
                {hsv_to_rgb_full_8u, 0, 0, 0, 0, hsv_to_rgb_full_32f},
                {hsv4_to_rgb_full_8u, 0, 0, 0, 0, hsv4_to_rgb_full_32f}
            },
            {
                {hsv_to_rgba_full_8u, 0, 0, 0, 0, hsv_to_rgba_full_32f},
                {hsv4_to_rgba_full_8u, 0, 0, 0, 0, hsv4_to_rgba_full_32f}
            }
        };

        if (dcn <= 0) dcn = 3;
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));        

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void hsv_to_bgr_full(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] = 
        {
            {
                {hsv_to_bgr_full_8u, 0, 0, 0, 0, hsv_to_bgr_full_32f},
                {hsv4_to_bgr_full_8u, 0, 0, 0, 0, hsv4_to_bgr_full_32f}
            },
            {
                {hsv_to_bgra_full_8u, 0, 0, 0, 0, hsv_to_bgra_full_32f},
                {hsv4_to_bgra_full_8u, 0, 0, 0, 0, hsv4_to_bgra_full_32f}
            }
        };

        if (dcn <= 0) dcn = 3;
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));        

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }    

    void rgb_to_hls_full(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] = 
        {
            {
                {rgb_to_hls_full_8u, 0, 0, 0, 0, rgb_to_hls_full_32f},
                {rgba_to_hls_full_8u, 0, 0, 0, 0, rgba_to_hls_full_32f},
            },
            {
                {rgb_to_hls4_full_8u, 0, 0, 0, 0, rgb_to_hls4_full_32f},
                {rgba_to_hls4_full_8u, 0, 0, 0, 0, rgba_to_hls4_full_32f},
            }
        };

        if (dcn <= 0) dcn = 3;
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));        

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgr_to_hls_full(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] = 
        {
            {
                {bgr_to_hls_full_8u, 0, 0, 0, 0, bgr_to_hls_full_32f},
                {bgra_to_hls_full_8u, 0, 0, 0, 0, bgra_to_hls_full_32f}
            },
            {
                {bgr_to_hls4_full_8u, 0, 0, 0, 0, bgr_to_hls4_full_32f},
                {bgra_to_hls4_full_8u, 0, 0, 0, 0, bgra_to_hls4_full_32f}
            }
        };

        if (dcn <= 0) dcn = 3;
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));        

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void hls_to_rgb_full(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] = 
        {
            {
                {hls_to_rgb_full_8u, 0, 0, 0, 0, hls_to_rgb_full_32f},
                {hls4_to_rgb_full_8u, 0, 0, 0, 0, hls4_to_rgb_full_32f}
            },
            {
                {hls_to_rgba_full_8u, 0, 0, 0, 0, hls_to_rgba_full_32f},
                {hls4_to_rgba_full_8u, 0, 0, 0, 0, hls4_to_rgba_full_32f}
            }
        };

        if (dcn <= 0) dcn = 3;
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));        

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void hls_to_bgr_full(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] = 
        {
            {
                {hls_to_bgr_full_8u, 0, 0, 0, 0, hls_to_bgr_full_32f},
                {hls4_to_bgr_full_8u, 0, 0, 0, 0, hls4_to_bgr_full_32f}
            },
            {
                {hls_to_bgra_full_8u, 0, 0, 0, 0, hls_to_bgra_full_32f},
                {hls4_to_bgra_full_8u, 0, 0, 0, 0, hls4_to_bgra_full_32f}
            }
        };

        if (dcn <= 0) dcn = 3;
        
        CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));        

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }
}

void cv::gpu::cvtColor(const GpuMat& src, GpuMat& dst, int code, int dcn, Stream& stream)
{
    typedef void (*func_t)(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream);
    static const func_t funcs[] = 
    {
        bgr_to_bgra,            // CV_BGR2BGRA    =0
        bgra_to_bgr,            // CV_BGRA2BGR    =1
        bgr_to_rgba,            // CV_BGR2RGBA    =2
        bgra_to_rgb,            // CV_RGBA2BGR    =3
        bgr_to_rgb,             // CV_BGR2RGB     =4
        bgra_to_rgba,           // CV_BGRA2RGBA   =5

        bgr_to_gray,            // CV_BGR2GRAY    =6
        rgb_to_gray,            // CV_RGB2GRAY    =7
        gray_to_bgr,            // CV_GRAY2BGR    =8
        gray_to_bgra,           // CV_GRAY2BGRA   =9
        bgra_to_gray,           // CV_BGRA2GRAY   =10
        rgba_to_gray,           // CV_RGBA2GRAY   =11

        bgr_to_bgr565,          // CV_BGR2BGR565  =12
        rgb_to_bgr565,          // CV_RGB2BGR565  =13
        bgr565_to_bgr,          // CV_BGR5652BGR  =14
        bgr565_to_rgb,          // CV_BGR5652RGB  =15
        bgra_to_bgr565,         // CV_BGRA2BGR565 =16
        rgba_to_bgr565,         // CV_RGBA2BGR565 =17
        bgr565_to_bgra,         // CV_BGR5652BGRA =18
        bgr565_to_rgba,         // CV_BGR5652RGBA =19

        gray_to_bgr565,         // CV_GRAY2BGR565 =20
        bgr565_to_gray,         // CV_BGR5652GRAY =21

        bgr_to_bgr555,          // CV_BGR2BGR555  =22
        rgb_to_bgr555,          // CV_RGB2BGR555  =23
        bgr555_to_bgr,          // CV_BGR5552BGR  =24
        bgr555_to_rgb,          // CV_BGR5552RGB  =25
        bgra_to_bgr555,         // CV_BGRA2BGR555 =26
        rgba_to_bgr555,         // CV_RGBA2BGR555 =27
        bgr555_to_bgra,         // CV_BGR5552BGRA =28
        bgr555_to_rgba,         // CV_BGR5552RGBA =29

        gray_to_bgr555,         // CV_GRAY2BGR555 =30
        bgr555_to_gray,         // CV_BGR5552GRAY =31

        bgr_to_xyz,             // CV_BGR2XYZ     =32
        rgb_to_xyz,             // CV_RGB2XYZ     =33
        xyz_to_bgr,             // CV_XYZ2BGR     =34
        xyz_to_rgb,             // CV_XYZ2RGB     =35

        bgr_to_YCrCb,           // CV_BGR2YCrCb   =36
        rgb_to_YCrCb,           // CV_RGB2YCrCb   =37
        YCrCb_to_bgr,           // CV_YCrCb2BGR   =38
        YCrCb_to_rgb,           // CV_YCrCb2RGB   =39

        bgr_to_hsv,             // CV_BGR2HSV     =40
        rgb_to_hsv,             // CV_RGB2HSV     =41

        0,                      //                =42
        0,                      //                =43

        0,                      // CV_BGR2Lab     =44 
        0,                      // CV_RGB2Lab     =45

        0,                      // CV_BayerBG2BGR =46
        0,                      // CV_BayerGB2BGR =47
        0,                      // CV_BayerRG2BGR =48
        0,                      // CV_BayerGR2BGR =49

        0,                      // CV_BGR2Luv     =50
        0,                      // CV_RGB2Luv     =51

        bgr_to_hls,             // CV_BGR2HLS     =52
        rgb_to_hls,             // CV_RGB2HLS     =53

        hsv_to_bgr,             // CV_HSV2BGR     =54
        hsv_to_rgb,             // CV_HSV2RGB     =55

        0,                      // CV_Lab2BGR     =56
        0,                      // CV_Lab2RGB     =57
        0,                      // CV_Luv2BGR     =58
        0,                      // CV_Luv2RGB     =59
        
        hls_to_bgr,             // CV_HLS2BGR     =60
        hls_to_rgb,             // CV_HLS2RGB     =61

        0,                      // CV_BayerBG2BGR_VNG =62
        0,                      // CV_BayerGB2BGR_VNG =63
        0,                      // CV_BayerRG2BGR_VNG =64
        0,                      // CV_BayerGR2BGR_VNG =65

        bgr_to_hsv_full,        // CV_BGR2HSV_FULL = 66
        rgb_to_hsv_full,        // CV_RGB2HSV_FULL = 67
        bgr_to_hls_full,        // CV_BGR2HLS_FULL = 68
        rgb_to_hls_full,        // CV_RGB2HLS_FULL = 69

        hsv_to_bgr_full,        // CV_HSV2BGR_FULL = 70
        hsv_to_rgb_full,        // CV_HSV2RGB_FULL = 71
        hls_to_bgr_full,        // CV_HLS2BGR_FULL = 72
        hls_to_rgb_full,        // CV_HLS2RGB_FULL = 73

        0,                      // CV_LBGR2Lab     = 74
        0,                      // CV_LRGB2Lab     = 75
        0,                      // CV_LBGR2Luv     = 76
        0,                      // CV_LRGB2Luv     = 77

        0,                      // CV_Lab2LBGR     = 78
        0,                      // CV_Lab2LRGB     = 79
        0,                      // CV_Luv2LBGR     = 80
        0,                      // CV_Luv2LRGB     = 81

        bgr_to_yuv,             // CV_BGR2YUV      = 82
        rgb_to_yuv,             // CV_RGB2YUV      = 83
        yuv_to_bgr,             // CV_YUV2BGR      = 84
        yuv_to_rgb,             // CV_YUV2RGB      = 85

        0,                      // CV_BayerBG2GRAY = 86
        0,                      // CV_BayerGB2GRAY = 87
        0,                      // CV_BayerRG2GRAY = 88
        0,                      // CV_BayerGR2GRAY = 89

        0,                      // CV_YUV420i2RGB  = 90
        0,                      // CV_YUV420i2BGR  = 91
        0,                      // CV_YUV420sp2RGB = 92
        0                       // CV_YUV420sp2BGR = 93
    };

    CV_Assert(code < 94);

    func_t func = funcs[code];

    if (func == 0)
        CV_Error( CV_StsBadFlag, "Unknown/unsupported color conversion code" );

    func(src, dst, dcn, stream);
}

#endif /* !defined (HAVE_CUDA) */
