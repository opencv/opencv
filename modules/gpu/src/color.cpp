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

#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)

void cv::gpu::cvtColor(const GpuMat&, GpuMat&, int, int, Stream&) { throw_nogpu(); }
void cv::gpu::demosaicing(const GpuMat&, GpuMat&, int, int, Stream&) { throw_nogpu(); }
void cv::gpu::swapChannels(GpuMat&, const int[], Stream&) { throw_nogpu(); }
void cv::gpu::gammaCorrection(const GpuMat&, GpuMat&, bool, Stream&) { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

#include "cvt_color_internal.h"

namespace cv { namespace gpu {
    namespace device
    {
        template <int cn>
        void Bayer2BGR_8u_gpu(PtrStepSzb src, PtrStepSzb dst, bool blue_last, bool start_with_green, cudaStream_t stream);
        template <int cn>
        void Bayer2BGR_16u_gpu(PtrStepSzb src, PtrStepSzb dst, bool blue_last, bool start_with_green, cudaStream_t stream);

        template <int cn>
        void MHCdemosaic(PtrStepSzb src, int2 sourceOffset, PtrStepSzb dst, int2 firstRed, cudaStream_t stream);
    }
}}

using namespace ::cv::gpu::device;

#ifdef OPENCV_TINY_GPU_MODULE
    #define APPEND_16U(func) 0
#else
    #define APPEND_16U(func) func ## _16u
#endif

namespace
{
    typedef void (*gpu_func_t)(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);

    void bgr_to_rgb(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[] = {bgr_to_rgb_8u, 0, APPEND_16U(bgr_to_rgb), 0, 0, bgr_to_rgb_32f};

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3);
        CV_Assert(funcs[src.depth()] != 0);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), 3));

        funcs[src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgr_to_bgra(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[] = {bgr_to_bgra_8u, 0, APPEND_16U(bgr_to_bgra), 0, 0, bgr_to_bgra_32f};

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3);
        CV_Assert(funcs[src.depth()] != 0);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), 4));

        funcs[src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgr_to_rgba(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[] = {bgr_to_rgba_8u, 0, APPEND_16U(bgr_to_rgba), 0, 0, bgr_to_rgba_32f};

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3);
        CV_Assert(funcs[src.depth()] != 0);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), 4));

        funcs[src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgra_to_bgr(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[] = {bgra_to_bgr_8u, 0, APPEND_16U(bgra_to_bgr), 0, 0, bgra_to_bgr_32f};

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 4);
        CV_Assert(funcs[src.depth()] != 0);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), 3));

        funcs[src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgra_to_rgb(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[] = {bgra_to_rgb_8u, 0, APPEND_16U(bgra_to_rgb), 0, 0, bgra_to_rgb_32f};

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 4);
        CV_Assert(funcs[src.depth()] != 0);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), 3));

        funcs[src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgra_to_rgba(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[] = {bgra_to_rgba_8u, 0, APPEND_16U(bgra_to_rgba), 0, 0, bgra_to_rgba_32f};

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 4);
        CV_Assert(funcs[src.depth()] != 0);

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
        static const gpu_func_t funcs[] = {gray_to_bgr_8u, 0, APPEND_16U(gray_to_bgr), 0, 0, gray_to_bgr_32f};

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 1);
        CV_Assert(funcs[src.depth()] != 0);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), 3));

        funcs[src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void gray_to_bgra(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[] = {gray_to_bgra_8u, 0, APPEND_16U(gray_to_bgra), 0, 0, gray_to_bgra_32f};

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 1);
        CV_Assert(funcs[src.depth()] != 0);

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
        static const gpu_func_t funcs[] = {rgb_to_gray_8u, 0, APPEND_16U(rgb_to_gray), 0, 0, rgb_to_gray_32f};

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3);
        CV_Assert(funcs[src.depth()] != 0);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), 1));

        funcs[src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgr_to_gray(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[] = {bgr_to_gray_8u, 0, APPEND_16U(bgr_to_gray), 0, 0, bgr_to_gray_32f};

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3);
        CV_Assert(funcs[src.depth()] != 0);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), 1));

        funcs[src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void rgba_to_gray(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[] = {rgba_to_gray_8u, 0, APPEND_16U(rgba_to_gray), 0, 0, rgba_to_gray_32f};

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 4);
        CV_Assert(funcs[src.depth()] != 0);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), 1));

        funcs[src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgra_to_gray(const GpuMat& src, GpuMat& dst, int, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[] = {bgra_to_gray_8u, 0, APPEND_16U(bgra_to_gray), 0, 0, bgra_to_gray_32f};

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 4);
        CV_Assert(funcs[src.depth()] != 0);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), 1));

        funcs[src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void rgb_to_yuv(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {rgb_to_yuv_8u, 0, APPEND_16U(rgb_to_yuv), 0, 0, rgb_to_yuv_32f},
                {rgba_to_yuv_8u, 0, APPEND_16U(rgba_to_yuv), 0, 0, rgba_to_yuv_32f}
            },
            {
                {rgb_to_yuv4_8u, 0, APPEND_16U(rgb_to_yuv4), 0, 0, rgb_to_yuv4_32f},
                {rgba_to_yuv4_8u, 0, APPEND_16U(rgba_to_yuv4), 0, 0, rgba_to_yuv4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);
        CV_Assert(funcs[dcn == 4][src.channels() == 4][src.depth()] != 0);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgr_to_yuv(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {bgr_to_yuv_8u, 0, APPEND_16U(bgr_to_yuv), 0, 0, bgr_to_yuv_32f},
                {bgra_to_yuv_8u, 0, APPEND_16U(bgra_to_yuv), 0, 0, bgra_to_yuv_32f}
            },
            {
                {bgr_to_yuv4_8u, 0, APPEND_16U(bgr_to_yuv4), 0, 0, bgr_to_yuv4_32f},
                {bgra_to_yuv4_8u, 0, APPEND_16U(bgra_to_yuv4), 0, 0, bgra_to_yuv4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);
        CV_Assert(funcs[dcn == 4][src.channels() == 4][src.depth()] != 0);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void yuv_to_rgb(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {yuv_to_rgb_8u, 0, APPEND_16U(yuv_to_rgb), 0, 0, yuv_to_rgb_32f},
                {yuv4_to_rgb_8u, 0, APPEND_16U(yuv4_to_rgb), 0, 0, yuv4_to_rgb_32f}
            },
            {
                {yuv_to_rgba_8u, 0, APPEND_16U(yuv_to_rgba), 0, 0, yuv_to_rgba_32f},
                {yuv4_to_rgba_8u, 0, APPEND_16U(yuv4_to_rgba), 0, 0, yuv4_to_rgba_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);
        CV_Assert(funcs[dcn == 4][src.channels() == 4][src.depth()] != 0);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void yuv_to_bgr(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {yuv_to_bgr_8u, 0, APPEND_16U(yuv_to_bgr), 0, 0, yuv_to_bgr_32f},
                {yuv4_to_bgr_8u, 0, APPEND_16U(yuv4_to_bgr), 0, 0, yuv4_to_bgr_32f}
            },
            {
                {yuv_to_bgra_8u, 0, APPEND_16U(yuv_to_bgra), 0, 0, yuv_to_bgra_32f},
                {yuv4_to_bgra_8u, 0, APPEND_16U(yuv4_to_bgra), 0, 0, yuv4_to_bgra_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);
        CV_Assert(funcs[dcn == 4][src.channels() == 4][src.depth()] != 0);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void rgb_to_YCrCb(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {rgb_to_YCrCb_8u, 0, APPEND_16U(rgb_to_YCrCb), 0, 0, rgb_to_YCrCb_32f},
                {rgba_to_YCrCb_8u, 0, APPEND_16U(rgba_to_YCrCb), 0, 0, rgba_to_YCrCb_32f}
            },
            {
                {rgb_to_YCrCb4_8u, 0, APPEND_16U(rgb_to_YCrCb4), 0, 0, rgb_to_YCrCb4_32f},
                {rgba_to_YCrCb4_8u, 0, APPEND_16U(rgba_to_YCrCb4), 0, 0, rgba_to_YCrCb4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);
        CV_Assert(funcs[dcn == 4][src.channels() == 4][src.depth()] != 0);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgr_to_YCrCb(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {bgr_to_YCrCb_8u, 0, APPEND_16U(bgr_to_YCrCb), 0, 0, bgr_to_YCrCb_32f},
                {bgra_to_YCrCb_8u, 0, APPEND_16U(bgra_to_YCrCb), 0, 0, bgra_to_YCrCb_32f}
            },
            {
                {bgr_to_YCrCb4_8u, 0, APPEND_16U(bgr_to_YCrCb4), 0, 0, bgr_to_YCrCb4_32f},
                {bgra_to_YCrCb4_8u, 0, APPEND_16U(bgra_to_YCrCb4), 0, 0, bgra_to_YCrCb4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);
        CV_Assert(funcs[dcn == 4][src.channels() == 4][src.depth()] != 0);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void YCrCb_to_rgb(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {YCrCb_to_rgb_8u, 0, APPEND_16U(YCrCb_to_rgb), 0, 0, YCrCb_to_rgb_32f},
                {YCrCb4_to_rgb_8u, 0, APPEND_16U(YCrCb4_to_rgb), 0, 0, YCrCb4_to_rgb_32f}
            },
            {
                {YCrCb_to_rgba_8u, 0, APPEND_16U(YCrCb_to_rgba), 0, 0, YCrCb_to_rgba_32f},
                {YCrCb4_to_rgba_8u, 0, APPEND_16U(YCrCb4_to_rgba), 0, 0, YCrCb4_to_rgba_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);
        CV_Assert(funcs[dcn == 4][src.channels() == 4][src.depth()] != 0);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void YCrCb_to_bgr(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {YCrCb_to_bgr_8u, 0, APPEND_16U(YCrCb_to_bgr), 0, 0, YCrCb_to_bgr_32f},
                {YCrCb4_to_bgr_8u, 0, APPEND_16U(YCrCb4_to_bgr), 0, 0, YCrCb4_to_bgr_32f}
            },
            {
                {YCrCb_to_bgra_8u, 0, APPEND_16U(YCrCb_to_bgra), 0, 0, YCrCb_to_bgra_32f},
                {YCrCb4_to_bgra_8u, 0, APPEND_16U(YCrCb4_to_bgra), 0, 0, YCrCb4_to_bgra_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);
        CV_Assert(funcs[dcn == 4][src.channels() == 4][src.depth()] != 0);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void rgb_to_xyz(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {rgb_to_xyz_8u, 0, APPEND_16U(rgb_to_xyz), 0, 0, rgb_to_xyz_32f},
                {rgba_to_xyz_8u, 0, APPEND_16U(rgba_to_xyz), 0, 0, rgba_to_xyz_32f}
            },
            {
                {rgb_to_xyz4_8u, 0, APPEND_16U(rgb_to_xyz4), 0, 0, rgb_to_xyz4_32f},
                {rgba_to_xyz4_8u, 0, APPEND_16U(rgba_to_xyz4), 0, 0, rgba_to_xyz4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);
        CV_Assert(funcs[dcn == 4][src.channels() == 4][src.depth()] != 0);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgr_to_xyz(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {bgr_to_xyz_8u, 0, APPEND_16U(bgr_to_xyz), 0, 0, bgr_to_xyz_32f},
                {bgra_to_xyz_8u, 0, APPEND_16U(bgra_to_xyz), 0, 0, bgra_to_xyz_32f}
            },
            {
                {bgr_to_xyz4_8u, 0, APPEND_16U(bgr_to_xyz4), 0, 0, bgr_to_xyz4_32f},
                {bgra_to_xyz4_8u, 0, APPEND_16U(bgra_to_xyz4), 0, 0, bgra_to_xyz4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);
        CV_Assert(funcs[dcn == 4][src.channels() == 4][src.depth()] != 0);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void xyz_to_rgb(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {xyz_to_rgb_8u, 0, APPEND_16U(xyz_to_rgb), 0, 0, xyz_to_rgb_32f},
                {xyz4_to_rgb_8u, 0, APPEND_16U(xyz4_to_rgb), 0, 0, xyz4_to_rgb_32f}
            },
            {
                {xyz_to_rgba_8u, 0, APPEND_16U(xyz_to_rgba), 0, 0, xyz_to_rgba_32f},
                {xyz4_to_rgba_8u, 0, APPEND_16U(xyz4_to_rgba), 0, 0, xyz4_to_rgba_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);
        CV_Assert(funcs[dcn == 4][src.channels() == 4][src.depth()] != 0);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void xyz_to_bgr(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {xyz_to_bgr_8u, 0, APPEND_16U(xyz_to_bgr), 0, 0, xyz_to_bgr_32f},
                {xyz4_to_bgr_8u, 0, APPEND_16U(xyz4_to_bgr), 0, 0, xyz4_to_bgr_32f}
            },
            {
                {xyz_to_bgra_8u, 0, APPEND_16U(xyz_to_bgra), 0, 0, xyz_to_bgra_32f},
                {xyz4_to_bgra_8u, 0, APPEND_16U(xyz4_to_bgra), 0, 0, xyz4_to_bgra_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);
        CV_Assert(funcs[dcn == 4][src.channels() == 4][src.depth()] != 0);

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

    void bgr_to_lab(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][2] =
        {
            {
                {bgr_to_lab_8u, bgr_to_lab_32f},
                {bgra_to_lab_8u, bgra_to_lab_32f}
            },
            {
                {bgr_to_lab4_8u, bgr_to_lab4_32f},
                {bgra_to_lab4_8u, bgra_to_lab4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, StreamAccessor::getStream(stream));
    }

    void rgb_to_lab(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][2] =
        {
            {
                {rgb_to_lab_8u, rgb_to_lab_32f},
                {rgba_to_lab_8u, rgba_to_lab_32f}
            },
            {
                {rgb_to_lab4_8u, rgb_to_lab4_32f},
                {rgba_to_lab4_8u, rgba_to_lab4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, StreamAccessor::getStream(stream));
    }

    void lbgr_to_lab(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][2] =
        {
            {
                {lbgr_to_lab_8u, lbgr_to_lab_32f},
                {lbgra_to_lab_8u, lbgra_to_lab_32f}
            },
            {
                {lbgr_to_lab4_8u, lbgr_to_lab4_32f},
                {lbgra_to_lab4_8u, lbgra_to_lab4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, StreamAccessor::getStream(stream));
    }

    void lrgb_to_lab(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][2] =
        {
            {
                {lrgb_to_lab_8u, lrgb_to_lab_32f},
                {lrgba_to_lab_8u, lrgba_to_lab_32f}
            },
            {
                {lrgb_to_lab4_8u, lrgb_to_lab4_32f},
                {lrgba_to_lab4_8u, lrgba_to_lab4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, StreamAccessor::getStream(stream));
    }

    void lab_to_bgr(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][2] =
        {
            {
                {lab_to_bgr_8u, lab_to_bgr_32f},
                {lab4_to_bgr_8u, lab4_to_bgr_32f}
            },
            {
                {lab_to_bgra_8u, lab_to_bgra_32f},
                {lab4_to_bgra_8u, lab4_to_bgra_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, StreamAccessor::getStream(stream));
    }

    void lab_to_rgb(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][2] =
        {
            {
                {lab_to_rgb_8u, lab_to_rgb_32f},
                {lab4_to_rgb_8u, lab4_to_rgb_32f}
            },
            {
                {lab_to_rgba_8u, lab_to_rgba_32f},
                {lab4_to_rgba_8u, lab4_to_rgba_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, StreamAccessor::getStream(stream));
    }

    void lab_to_lbgr(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][2] =
        {
            {
                {lab_to_lbgr_8u, lab_to_lbgr_32f},
                {lab4_to_lbgr_8u, lab4_to_lbgr_32f}
            },
            {
                {lab_to_lbgra_8u, lab_to_lbgra_32f},
                {lab4_to_lbgra_8u, lab4_to_lbgra_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, StreamAccessor::getStream(stream));
    }

    void lab_to_lrgb(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][2] =
        {
            {
                {lab_to_lrgb_8u, lab_to_lrgb_32f},
                {lab4_to_lrgb_8u, lab4_to_lrgb_32f}
            },
            {
                {lab_to_lrgba_8u, lab_to_lrgba_32f},
                {lab4_to_lrgba_8u, lab4_to_lrgba_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, StreamAccessor::getStream(stream));
    }

    void bgr_to_luv(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][2] =
        {
            {
                {bgr_to_luv_8u, bgr_to_luv_32f},
                {bgra_to_luv_8u, bgra_to_luv_32f}
            },
            {
                {bgr_to_luv4_8u, bgr_to_luv4_32f},
                {bgra_to_luv4_8u, bgra_to_luv4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, StreamAccessor::getStream(stream));
    }

    void rgb_to_luv(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][2] =
        {
            {
                {rgb_to_luv_8u, rgb_to_luv_32f},
                {rgba_to_luv_8u, rgba_to_luv_32f}
            },
            {
                {rgb_to_luv4_8u, rgb_to_luv4_32f},
                {rgba_to_luv4_8u, rgba_to_luv4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, StreamAccessor::getStream(stream));
    }

    void lbgr_to_luv(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][2] =
        {
            {
                {lbgr_to_luv_8u, lbgr_to_luv_32f},
                {lbgra_to_luv_8u, lbgra_to_luv_32f}
            },
            {
                {lbgr_to_luv4_8u, lbgr_to_luv4_32f},
                {lbgra_to_luv4_8u, lbgra_to_luv4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, StreamAccessor::getStream(stream));
    }

    void lrgb_to_luv(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][2] =
        {
            {
                {lrgb_to_luv_8u, lrgb_to_luv_32f},
                {lrgba_to_luv_8u, lrgba_to_luv_32f}
            },
            {
                {lrgb_to_luv4_8u, lrgb_to_luv4_32f},
                {lrgba_to_luv4_8u, lrgba_to_luv4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, StreamAccessor::getStream(stream));
    }

    void luv_to_bgr(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][2] =
        {
            {
                {luv_to_bgr_8u, luv_to_bgr_32f},
                {luv4_to_bgr_8u, luv4_to_bgr_32f}
            },
            {
                {luv_to_bgra_8u, luv_to_bgra_32f},
                {luv4_to_bgra_8u, luv4_to_bgra_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, StreamAccessor::getStream(stream));
    }

    void luv_to_rgb(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][2] =
        {
            {
                {luv_to_rgb_8u, luv_to_rgb_32f},
                {luv4_to_rgb_8u, luv4_to_rgb_32f}
            },
            {
                {luv_to_rgba_8u, luv_to_rgba_32f},
                {luv4_to_rgba_8u, luv4_to_rgba_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, StreamAccessor::getStream(stream));
    }

    void luv_to_lbgr(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][2] =
        {
            {
                {luv_to_lbgr_8u, luv_to_lbgr_32f},
                {luv4_to_lbgr_8u, luv4_to_lbgr_32f}
            },
            {
                {luv_to_lbgra_8u, luv_to_lbgra_32f},
                {luv4_to_lbgra_8u, luv4_to_lbgra_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, StreamAccessor::getStream(stream));
    }

    void luv_to_lrgb(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        using namespace cv::gpu::device;
        static const gpu_func_t funcs[2][2][2] =
        {
            {
                {luv_to_lrgb_8u, luv_to_lrgb_32f},
                {luv4_to_lrgb_8u, luv4_to_lrgb_32f}
            },
            {
                {luv_to_lrgba_8u, luv_to_lrgba_32f},
                {luv4_to_lrgba_8u, luv4_to_lrgba_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
        CV_Assert(src.channels() == 3 || src.channels() == 4);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, StreamAccessor::getStream(stream));
    }

    void rgba_to_mbgra(const GpuMat& src, GpuMat& dst, int, Stream& st)
    {
    #if (CUDART_VERSION < 5000)
        (void)src;
        (void)dst;
        (void)st;
        CV_Error( CV_StsBadFlag, "Unknown/unsupported color conversion code" );
    #else
        CV_Assert(src.type() == CV_8UC4 || src.type() == CV_16UC4);

        dst.create(src.size(), src.type());

        cudaStream_t stream = StreamAccessor::getStream(st);
        NppStreamHandler h(stream);

        NppiSize oSizeROI;
        oSizeROI.width = src.cols;
        oSizeROI.height = src.rows;

        if (src.depth() == CV_8U)
            nppSafeCall( nppiAlphaPremul_8u_AC4R(src.ptr<Npp8u>(), static_cast<int>(src.step), dst.ptr<Npp8u>(), static_cast<int>(dst.step), oSizeROI) );
        else
            nppSafeCall( nppiAlphaPremul_16u_AC4R(src.ptr<Npp16u>(), static_cast<int>(src.step), dst.ptr<Npp16u>(), static_cast<int>(dst.step), oSizeROI) );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    #endif
    }

    void bayer_to_bgr(const GpuMat& src, GpuMat& dst, int dcn, bool blue_last, bool start_with_green, Stream& stream)
    {
        typedef void (*func_t)(PtrStepSzb src, PtrStepSzb dst, bool blue_last, bool start_with_green, cudaStream_t stream);
        static const func_t funcs[3][4] =
        {
            {0,0,Bayer2BGR_8u_gpu<3>, Bayer2BGR_8u_gpu<4>},
            {0,0,0,0},
            {0,0,Bayer2BGR_16u_gpu<3>, Bayer2BGR_16u_gpu<4>}
        };

        if (dcn <= 0) dcn = 3;

        CV_Assert(src.type() == CV_8UC1 || src.type() == CV_16UC1);
        CV_Assert(src.rows > 2 && src.cols > 2);
        CV_Assert(dcn == 3 || dcn == 4);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), dcn));

        funcs[src.depth()][dcn - 1](src, dst, blue_last, start_with_green, StreamAccessor::getStream(stream));
    }
    void bayerBG_to_bgr(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        bayer_to_bgr(src, dst, dcn, false, false, stream);
    }
    void bayerGB_to_bgr(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        bayer_to_bgr(src, dst, dcn, false, true, stream);
    }
    void bayerRG_to_bgr(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        bayer_to_bgr(src, dst, dcn, true, false, stream);
    }
    void bayerGR_to_bgr(const GpuMat& src, GpuMat& dst, int dcn, Stream& stream)
    {
        bayer_to_bgr(src, dst, dcn, true, true, stream);
    }

    void bayer_to_gray(const GpuMat& src, GpuMat& dst, bool blue_last, bool start_with_green, Stream& stream)
    {
        typedef void (*func_t)(PtrStepSzb src, PtrStepSzb dst, bool blue_last, bool start_with_green, cudaStream_t stream);
        static const func_t funcs[3] =
        {
            Bayer2BGR_8u_gpu<1>,
            0,
            Bayer2BGR_16u_gpu<1>,
        };

        CV_Assert(src.type() == CV_8UC1 || src.type() == CV_16UC1);
        CV_Assert(src.rows > 2 && src.cols > 2);

        dst.create(src.size(), CV_MAKETYPE(src.depth(), 1));

        funcs[src.depth()](src, dst, blue_last, start_with_green, StreamAccessor::getStream(stream));
    }
    void bayerBG_to_gray(const GpuMat& src, GpuMat& dst, int /*dcn*/, Stream& stream)
    {
        bayer_to_gray(src, dst, false, false, stream);
    }
    void bayerGB_to_gray(const GpuMat& src, GpuMat& dst, int /*dcn*/, Stream& stream)
    {
        bayer_to_gray(src, dst, false, true, stream);
    }
    void bayerRG_to_gray(const GpuMat& src, GpuMat& dst, int /*dcn*/, Stream& stream)
    {
        bayer_to_gray(src, dst, true, false, stream);
    }
    void bayerGR_to_gray(const GpuMat& src, GpuMat& dst, int /*dcn*/, Stream& stream)
    {
        bayer_to_gray(src, dst, true, true, stream);
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

        bgr_to_lab,             // CV_BGR2Lab     =44
        rgb_to_lab,             // CV_RGB2Lab     =45

        bayerBG_to_bgr,         // CV_BayerBG2BGR =46
        bayerGB_to_bgr,         // CV_BayerGB2BGR =47
        bayerRG_to_bgr,         // CV_BayerRG2BGR =48
        bayerGR_to_bgr,         // CV_BayerGR2BGR =49

        bgr_to_luv,             // CV_BGR2Luv     =50
        rgb_to_luv,             // CV_RGB2Luv     =51

        bgr_to_hls,             // CV_BGR2HLS     =52
        rgb_to_hls,             // CV_RGB2HLS     =53

        hsv_to_bgr,             // CV_HSV2BGR     =54
        hsv_to_rgb,             // CV_HSV2RGB     =55

        lab_to_bgr,             // CV_Lab2BGR     =56
        lab_to_rgb,             // CV_Lab2RGB     =57
        luv_to_bgr,             // CV_Luv2BGR     =58
        luv_to_rgb,             // CV_Luv2RGB     =59

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

        lbgr_to_lab,            // CV_LBGR2Lab     = 74
        lrgb_to_lab,            // CV_LRGB2Lab     = 75
        lbgr_to_luv,            // CV_LBGR2Luv     = 76
        lrgb_to_luv,            // CV_LRGB2Luv     = 77

        lab_to_lbgr,            // CV_Lab2LBGR     = 78
        lab_to_lrgb,            // CV_Lab2LRGB     = 79
        luv_to_lbgr,            // CV_Luv2LBGR     = 80
        luv_to_lrgb,            // CV_Luv2LRGB     = 81

        bgr_to_yuv,             // CV_BGR2YUV      = 82
        rgb_to_yuv,             // CV_RGB2YUV      = 83
        yuv_to_bgr,             // CV_YUV2BGR      = 84
        yuv_to_rgb,             // CV_YUV2RGB      = 85

        bayerBG_to_gray,        // CV_BayerBG2GRAY = 86
        bayerGB_to_gray,        // CV_BayerGB2GRAY = 87
        bayerRG_to_gray,        // CV_BayerRG2GRAY = 88
        bayerGR_to_gray,        // CV_BayerGR2GRAY = 89

        //YUV 4:2:0 formats family
        0,                      // CV_YUV2RGB_NV12 = 90,
        0,                      // CV_YUV2BGR_NV12 = 91,
        0,                      // CV_YUV2RGB_NV21 = 92,
        0,                      // CV_YUV2BGR_NV21 = 93,

        0,                      // CV_YUV2RGBA_NV12 = 94,
        0,                      // CV_YUV2BGRA_NV12 = 95,
        0,                      // CV_YUV2RGBA_NV21 = 96,
        0,                      // CV_YUV2BGRA_NV21 = 97,

        0,                      // CV_YUV2RGB_YV12 = 98,
        0,                      // CV_YUV2BGR_YV12 = 99,
        0,                      // CV_YUV2RGB_IYUV = 100,
        0,                      // CV_YUV2BGR_IYUV = 101,

        0,                      // CV_YUV2RGBA_YV12 = 102,
        0,                      // CV_YUV2BGRA_YV12 = 103,
        0,                      // CV_YUV2RGBA_IYUV = 104,
        0,                      // CV_YUV2BGRA_IYUV = 105,

        0,                      // CV_YUV2GRAY_420 = 106,

        //YUV 4:2:2 formats family
        0,                      // CV_YUV2RGB_UYVY = 107,
        0,                      // CV_YUV2BGR_UYVY = 108,
        0,                      // //CV_YUV2RGB_VYUY = 109,
        0,                      // //CV_YUV2BGR_VYUY = 110,

        0,                      // CV_YUV2RGBA_UYVY = 111,
        0,                      // CV_YUV2BGRA_UYVY = 112,
        0,                      // //CV_YUV2RGBA_VYUY = 113,
        0,                      // //CV_YUV2BGRA_VYUY = 114,

        0,                      // CV_YUV2RGB_YUY2 = 115,
        0,                      // CV_YUV2BGR_YUY2 = 116,
        0,                      // CV_YUV2RGB_YVYU = 117,
        0,                      // CV_YUV2BGR_YVYU = 118,

        0,                      // CV_YUV2RGBA_YUY2 = 119,
        0,                      // CV_YUV2BGRA_YUY2 = 120,
        0,                      // CV_YUV2RGBA_YVYU = 121,
        0,                      // CV_YUV2BGRA_YVYU = 122,

        0,                      // CV_YUV2GRAY_UYVY = 123,
        0,                      // CV_YUV2GRAY_YUY2 = 124,

        // alpha premultiplication
        rgba_to_mbgra,          // CV_RGBA2mRGBA = 125,
        0,                      // CV_mRGBA2RGBA = 126,

        0,                      // CV_COLORCVT_MAX  = 127
    };

    CV_Assert(code < 128);

    func_t func = funcs[code];

    if (func == 0)
        CV_Error( CV_StsBadFlag, "Unknown/unsupported color conversion code" );

    func(src, dst, dcn, stream);
}

void cv::gpu::demosaicing(const GpuMat& src, GpuMat& dst, int code, int dcn, Stream& stream)
{
    const int depth = src.depth();

    CV_Assert( src.channels() == 1 && !src.empty() );

    switch (code)
    {
    case CV_BayerBG2GRAY: case CV_BayerGB2GRAY: case CV_BayerRG2GRAY: case CV_BayerGR2GRAY:
        bayer_to_gray(src, dst, code == CV_BayerBG2GRAY || code == CV_BayerGB2GRAY, code == CV_BayerGB2GRAY || code == CV_BayerGR2GRAY, stream);
        break;

    case CV_BayerBG2BGR: case CV_BayerGB2BGR: case CV_BayerRG2BGR: case CV_BayerGR2BGR:
        bayer_to_bgr(src, dst, dcn, code == CV_BayerBG2BGR || code == CV_BayerGB2BGR, code == CV_BayerGB2BGR || code == CV_BayerGR2BGR, stream);
        break;

    case COLOR_BayerBG2BGR_MHT: case COLOR_BayerGB2BGR_MHT: case COLOR_BayerRG2BGR_MHT: case COLOR_BayerGR2BGR_MHT:
    {
        if (dcn <= 0)
            dcn = 3;

        CV_Assert( depth == CV_8U );
        CV_Assert( dcn == 3 || dcn == 4 );

        dst.create(src.size(), CV_MAKETYPE(depth, dcn));
        dst.setTo(Scalar::all(0));

        Size wholeSize;
        Point ofs;
        src.locateROI(wholeSize, ofs);
        PtrStepSzb srcWhole(wholeSize.height, wholeSize.width, src.datastart, src.step);

        const int2 firstRed = make_int2(code == COLOR_BayerRG2BGR_MHT || code == COLOR_BayerGB2BGR_MHT ? 0 : 1,
                                        code == COLOR_BayerRG2BGR_MHT || code == COLOR_BayerGR2BGR_MHT ? 0 : 1);

        if (dcn == 3)
            device::MHCdemosaic<3>(srcWhole, make_int2(ofs.x, ofs.y), dst, firstRed, StreamAccessor::getStream(stream));
        else
            device::MHCdemosaic<4>(srcWhole, make_int2(ofs.x, ofs.y), dst, firstRed, StreamAccessor::getStream(stream));

        break;
    }

    case COLOR_BayerBG2GRAY_MHT: case COLOR_BayerGB2GRAY_MHT: case COLOR_BayerRG2GRAY_MHT: case COLOR_BayerGR2GRAY_MHT:
    {
        CV_Assert( depth == CV_8U );

        dst.create(src.size(), CV_MAKETYPE(depth, 1));
        dst.setTo(Scalar::all(0));

        Size wholeSize;
        Point ofs;
        src.locateROI(wholeSize, ofs);
        PtrStepSzb srcWhole(wholeSize.height, wholeSize.width, src.datastart, src.step);

        const int2 firstRed = make_int2(code == COLOR_BayerRG2BGR_MHT || code == COLOR_BayerGB2BGR_MHT ? 0 : 1,
                                        code == COLOR_BayerRG2BGR_MHT || code == COLOR_BayerGR2BGR_MHT ? 0 : 1);

        device::MHCdemosaic<1>(srcWhole, make_int2(ofs.x, ofs.y), dst, firstRed, StreamAccessor::getStream(stream));

        break;
    }

    default:
        CV_Error( CV_StsBadFlag, "Unknown / unsupported color conversion code" );
    }
}

void cv::gpu::swapChannels(GpuMat& image, const int dstOrder[4], Stream& s)
{
    CV_Assert(image.type() == CV_8UC4);

    cudaStream_t stream = StreamAccessor::getStream(s);

    NppStreamHandler h(stream);

    NppiSize sz;
    sz.width  = image.cols;
    sz.height = image.rows;

    nppSafeCall( nppiSwapChannels_8u_C4IR(image.ptr<Npp8u>(), static_cast<int>(image.step), sz, dstOrder) );

    if (stream == 0)
        cudaSafeCall( cudaDeviceSynchronize() );
}

void cv::gpu::gammaCorrection(const GpuMat& src, GpuMat& dst, bool forward, Stream& stream)
{
#if (CUDART_VERSION < 5000)
    (void)src;
    (void)dst;
    (void)forward;
    (void)stream;
    CV_Error( CV_StsNotImplemented, "This function works only with CUDA 5.0 or higher" );
#else
    typedef NppStatus (*func_t)(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst, int nDstStep, NppiSize oSizeROI);
    typedef NppStatus (*func_inplace_t)(Npp8u* pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

    static const func_t funcs[2][5] =
    {
        {0, 0, 0, nppiGammaInv_8u_C3R, nppiGammaInv_8u_AC4R},
        {0, 0, 0, nppiGammaFwd_8u_C3R, nppiGammaFwd_8u_AC4R}
    };
    static const func_inplace_t funcs_inplace[2][5] =
    {
        {0, 0, 0, nppiGammaInv_8u_C3IR, nppiGammaInv_8u_AC4IR},
        {0, 0, 0, nppiGammaFwd_8u_C3IR, nppiGammaFwd_8u_AC4IR}
    };

    CV_Assert(src.type() == CV_8UC3 || src.type() == CV_8UC4);

    dst.create(src.size(), src.type());

    NppStreamHandler h(StreamAccessor::getStream(stream));

    NppiSize oSizeROI;
    oSizeROI.width = src.cols;
    oSizeROI.height = src.rows;

    if (dst.data == src.data)
        funcs_inplace[forward][src.channels()](dst.ptr<Npp8u>(), static_cast<int>(src.step), oSizeROI);
    else
        funcs[forward][src.channels()](src.ptr<Npp8u>(), static_cast<int>(src.step), dst.ptr<Npp8u>(), static_cast<int>(dst.step), oSizeROI);

#endif
}

#endif /* !defined (HAVE_CUDA) */
