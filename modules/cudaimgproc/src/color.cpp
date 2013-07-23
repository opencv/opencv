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
using namespace cv::cuda;

#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)

void cv::cuda::cvtColor(InputArray, OutputArray, int, int, Stream&) { throw_no_cuda(); }

void cv::cuda::demosaicing(InputArray, OutputArray, int, int, Stream&) { throw_no_cuda(); }

void cv::cuda::swapChannels(InputOutputArray, const int[], Stream&) { throw_no_cuda(); }

void cv::cuda::gammaCorrection(InputArray, OutputArray, bool, Stream&) { throw_no_cuda(); }

void cv::cuda::alphaComp(InputArray, InputArray, OutputArray, int, Stream&) { throw_no_cuda(); }


#else /* !defined (HAVE_CUDA) */

#include "cvt_color_internal.h"

namespace cv { namespace cuda {
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

using namespace ::cv::cuda::device;

namespace
{
    typedef void (*gpu_func_t)(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);

    void bgr_to_rgb(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[] = {bgr_to_rgb_8u, 0, bgr_to_rgb_16u, 0, 0, bgr_to_rgb_32f};

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), 3));
        GpuMat dst = _dst.getGpuMat();

        funcs[src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgr_to_bgra(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[] = {bgr_to_bgra_8u, 0, bgr_to_bgra_16u, 0, 0, bgr_to_bgra_32f};

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), 4));
        GpuMat dst = _dst.getGpuMat();

        funcs[src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgr_to_rgba(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[] = {bgr_to_rgba_8u, 0, bgr_to_rgba_16u, 0, 0, bgr_to_rgba_32f};

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), 4));
        GpuMat dst = _dst.getGpuMat();

        funcs[src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgra_to_bgr(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[] = {bgra_to_bgr_8u, 0, bgra_to_bgr_16u, 0, 0, bgra_to_bgr_32f};

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), 3));
        GpuMat dst = _dst.getGpuMat();

        funcs[src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgra_to_rgb(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[] = {bgra_to_rgb_8u, 0, bgra_to_rgb_16u, 0, 0, bgra_to_rgb_32f};

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), 3));
        GpuMat dst = _dst.getGpuMat();

        funcs[src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgra_to_rgba(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[] = {bgra_to_rgba_8u, 0, bgra_to_rgba_16u, 0, 0, bgra_to_rgba_32f};

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), 4));
        GpuMat dst = _dst.getGpuMat();

        funcs[src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgr_to_bgr555(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 3 );

        _dst.create(src.size(), CV_8UC2);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::bgr_to_bgr555(src, dst, StreamAccessor::getStream(stream));
    }

    void bgr_to_bgr565(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 3 );

        _dst.create(src.size(), CV_8UC2);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::bgr_to_bgr565(src, dst, StreamAccessor::getStream(stream));
    }

    void rgb_to_bgr555(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 3 );

        _dst.create(src.size(), CV_8UC2);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::rgb_to_bgr555(src, dst, StreamAccessor::getStream(stream));
    }

    void rgb_to_bgr565(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 3 );

        _dst.create(src.size(), CV_8UC2);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::rgb_to_bgr565(src, dst, StreamAccessor::getStream(stream));
    }

    void bgra_to_bgr555(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 4 );

        _dst.create(src.size(), CV_8UC2);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::bgra_to_bgr555(src, dst, StreamAccessor::getStream(stream));
    }

    void bgra_to_bgr565(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 4 );

        _dst.create(src.size(), CV_8UC2);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::bgra_to_bgr565(src, dst, StreamAccessor::getStream(stream));
    }

    void rgba_to_bgr555(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 4 );

        _dst.create(src.size(), CV_8UC2);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::rgba_to_bgr555(src, dst, StreamAccessor::getStream(stream));
    }

    void rgba_to_bgr565(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 4 );

        _dst.create(src.size(), CV_8UC2);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::rgba_to_bgr565(src, dst, StreamAccessor::getStream(stream));
    }

    void bgr555_to_rgb(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 2 );

        _dst.create(src.size(), CV_8UC3);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::bgr555_to_rgb(src, dst, StreamAccessor::getStream(stream));
    }

    void bgr565_to_rgb(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 2 );

        _dst.create(src.size(), CV_8UC3);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::bgr565_to_rgb(src, dst, StreamAccessor::getStream(stream));
    }

    void bgr555_to_bgr(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 2 );

        _dst.create(src.size(), CV_8UC3);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::bgr555_to_bgr(src, dst, StreamAccessor::getStream(stream));
    }

    void bgr565_to_bgr(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 2 );

        _dst.create(src.size(), CV_8UC3);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::bgr565_to_bgr(src, dst, StreamAccessor::getStream(stream));
    }

    void bgr555_to_rgba(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 2 );

        _dst.create(src.size(), CV_8UC4);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::bgr555_to_rgba(src, dst, StreamAccessor::getStream(stream));
    }

    void bgr565_to_rgba(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 2 );

        _dst.create(src.size(), CV_8UC4);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::bgr565_to_rgba(src, dst, StreamAccessor::getStream(stream));
    }

    void bgr555_to_bgra(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 2 );

        _dst.create(src.size(), CV_8UC4);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::bgr555_to_bgra(src, dst, StreamAccessor::getStream(stream));
    }

    void bgr565_to_bgra(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 2 );

        _dst.create(src.size(), CV_8UC4);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::bgr565_to_bgra(src, dst, StreamAccessor::getStream(stream));
    }

    void gray_to_bgr(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[] = {gray_to_bgr_8u, 0, gray_to_bgr_16u, 0, 0, gray_to_bgr_32f};

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 1 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), 3));
        GpuMat dst = _dst.getGpuMat();

        funcs[src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void gray_to_bgra(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[] = {gray_to_bgra_8u, 0, gray_to_bgra_16u, 0, 0, gray_to_bgra_32f};

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 1 );

        _dst.create(src.size(), CV_MAKETYPE(src.depth(), 4));
        GpuMat dst = _dst.getGpuMat();

        funcs[src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void gray_to_bgr555(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 1 );

        _dst.create(src.size(), CV_8UC2);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::gray_to_bgr555(src, dst, StreamAccessor::getStream(stream));
    }

    void gray_to_bgr565(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 1 );

        _dst.create(src.size(), CV_8UC2);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::gray_to_bgr565(src, dst, StreamAccessor::getStream(stream));
    }

    void bgr555_to_gray(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 2 );

        _dst.create(src.size(), CV_8UC1);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::bgr555_to_gray(src, dst, StreamAccessor::getStream(stream));
    }

    void bgr565_to_gray(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 2 );

        _dst.create(src.size(), CV_8UC1);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::bgr565_to_gray(src, dst, StreamAccessor::getStream(stream));
    }

    void rgb_to_gray(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[] = {rgb_to_gray_8u, 0, rgb_to_gray_16u, 0, 0, rgb_to_gray_32f};

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), 1));
        GpuMat dst = _dst.getGpuMat();

        funcs[src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgr_to_gray(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[] = {bgr_to_gray_8u, 0, bgr_to_gray_16u, 0, 0, bgr_to_gray_32f};

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), 1));
        GpuMat dst = _dst.getGpuMat();

        funcs[src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void rgba_to_gray(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[] = {rgba_to_gray_8u, 0, rgba_to_gray_16u, 0, 0, rgba_to_gray_32f};

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), 1));
        GpuMat dst = _dst.getGpuMat();

        funcs[src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgra_to_gray(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[] = {bgra_to_gray_8u, 0, bgra_to_gray_16u, 0, 0, bgra_to_gray_32f};

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), 1));
        GpuMat dst = _dst.getGpuMat();

        funcs[src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void rgb_to_yuv(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgr_to_yuv(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void yuv_to_rgb(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void yuv_to_bgr(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void rgb_to_YCrCb(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgr_to_YCrCb(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void YCrCb_to_rgb(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void YCrCb_to_bgr(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void rgb_to_xyz(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgr_to_xyz(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void xyz_to_rgb(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void xyz_to_bgr(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void rgb_to_hsv(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgr_to_hsv(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void hsv_to_rgb(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void hsv_to_bgr(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void rgb_to_hls(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgr_to_hls(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void hls_to_rgb(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void hls_to_bgr(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void rgb_to_hsv_full(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgr_to_hsv_full(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void hsv_to_rgb_full(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void hsv_to_bgr_full(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void rgb_to_hls_full(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgr_to_hls_full(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void hls_to_rgb_full(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void hls_to_bgr_full(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, StreamAccessor::getStream(stream));
    }

    void bgr_to_lab(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, StreamAccessor::getStream(stream));
    }

    void rgb_to_lab(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, StreamAccessor::getStream(stream));
    }

    void lbgr_to_lab(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, StreamAccessor::getStream(stream));
    }

    void lrgb_to_lab(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, StreamAccessor::getStream(stream));
    }

    void lab_to_bgr(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, StreamAccessor::getStream(stream));
    }

    void lab_to_rgb(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, StreamAccessor::getStream(stream));
    }

    void lab_to_lbgr(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, StreamAccessor::getStream(stream));
    }

    void lab_to_lrgb(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, StreamAccessor::getStream(stream));
    }

    void bgr_to_luv(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, StreamAccessor::getStream(stream));
    }

    void rgb_to_luv(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, StreamAccessor::getStream(stream));
    }

    void lbgr_to_luv(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, StreamAccessor::getStream(stream));
    }

    void lrgb_to_luv(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, StreamAccessor::getStream(stream));
    }

    void luv_to_bgr(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, StreamAccessor::getStream(stream));
    }

    void luv_to_rgb(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, StreamAccessor::getStream(stream));
    }

    void luv_to_lbgr(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, StreamAccessor::getStream(stream));
    }

    void luv_to_lrgb(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
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

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, StreamAccessor::getStream(stream));
    }

    void rgba_to_mbgra(InputArray _src, OutputArray _dst, int, Stream& _stream)
    {
    #if (CUDA_VERSION < 5000)
        (void) _src;
        (void) _dst;
        (void) _stream;
        CV_Error( Error::StsBadFlag, "Unknown/unsupported color conversion code" );
    #else
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.type() == CV_8UC4 || src.type() == CV_16UC4 );

        _dst.create(src.size(), src.type());
        GpuMat dst = _dst.getGpuMat();

        cudaStream_t stream = StreamAccessor::getStream(_stream);
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

    void bayer_to_bgr(InputArray _src, OutputArray _dst, int dcn, bool blue_last, bool start_with_green, Stream& stream)
    {
        typedef void (*func_t)(PtrStepSzb src, PtrStepSzb dst, bool blue_last, bool start_with_green, cudaStream_t stream);
        static const func_t funcs[3][4] =
        {
            {0,0,Bayer2BGR_8u_gpu<3>, Bayer2BGR_8u_gpu<4>},
            {0,0,0,0},
            {0,0,Bayer2BGR_16u_gpu<3>, Bayer2BGR_16u_gpu<4>}
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.type() == CV_8UC1 || src.type() == CV_16UC1 );
        CV_Assert( src.rows > 2 && src.cols > 2 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[src.depth()][dcn - 1](src, dst, blue_last, start_with_green, StreamAccessor::getStream(stream));
    }
    void bayerBG_to_bgr(InputArray src, OutputArray dst, int dcn, Stream& stream)
    {
        bayer_to_bgr(src, dst, dcn, false, false, stream);
    }
    void bayerGB_to_bgr(InputArray src, OutputArray dst, int dcn, Stream& stream)
    {
        bayer_to_bgr(src, dst, dcn, false, true, stream);
    }
    void bayerRG_to_bgr(InputArray src, OutputArray dst, int dcn, Stream& stream)
    {
        bayer_to_bgr(src, dst, dcn, true, false, stream);
    }
    void bayerGR_to_bgr(InputArray src, OutputArray dst, int dcn, Stream& stream)
    {
        bayer_to_bgr(src, dst, dcn, true, true, stream);
    }

    void bayer_to_gray(InputArray _src, OutputArray _dst, bool blue_last, bool start_with_green, Stream& stream)
    {
        typedef void (*func_t)(PtrStepSzb src, PtrStepSzb dst, bool blue_last, bool start_with_green, cudaStream_t stream);
        static const func_t funcs[3] =
        {
            Bayer2BGR_8u_gpu<1>,
            0,
            Bayer2BGR_16u_gpu<1>,
        };

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.type() == CV_8UC1 || src.type() == CV_16UC1 );
        CV_Assert( src.rows > 2 && src.cols > 2 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), 1));
        GpuMat dst = _dst.getGpuMat();

        funcs[src.depth()](src, dst, blue_last, start_with_green, StreamAccessor::getStream(stream));
    }
    void bayerBG_to_gray(InputArray src, OutputArray dst, int /*dcn*/, Stream& stream)
    {
        bayer_to_gray(src, dst, false, false, stream);
    }
    void bayerGB_to_gray(InputArray src, OutputArray dst, int /*dcn*/, Stream& stream)
    {
        bayer_to_gray(src, dst, false, true, stream);
    }
    void bayerRG_to_gray(InputArray src, OutputArray dst, int /*dcn*/, Stream& stream)
    {
        bayer_to_gray(src, dst, true, false, stream);
    }
    void bayerGR_to_gray(InputArray src, OutputArray dst, int /*dcn*/, Stream& stream)
    {
        bayer_to_gray(src, dst, true, true, stream);
    }
}

////////////////////////////////////////////////////////////////////////
// cvtColor

void cv::cuda::cvtColor(InputArray src, OutputArray dst, int code, int dcn, Stream& stream)
{
    typedef void (*func_t)(InputArray src, OutputArray dst, int dcn, Stream& stream);
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

    CV_Assert( code < 128 );

    func_t func = funcs[code];

    if (func == 0)
        CV_Error(Error::StsBadFlag, "Unknown/unsupported color conversion code");

    func(src, dst, dcn, stream);
}

////////////////////////////////////////////////////////////////////////
// demosaicing

void cv::cuda::demosaicing(InputArray _src, OutputArray _dst, int code, int dcn, Stream& stream)
{
    switch (code)
    {
    case cv::COLOR_BayerBG2GRAY: case cv::COLOR_BayerGB2GRAY: case cv::COLOR_BayerRG2GRAY: case cv::COLOR_BayerGR2GRAY:
        bayer_to_gray(_src, _dst, code == cv::COLOR_BayerBG2GRAY || code == cv::COLOR_BayerGB2GRAY, code == cv::COLOR_BayerGB2GRAY || code == cv::COLOR_BayerGR2GRAY, stream);
        break;

    case cv::COLOR_BayerBG2BGR: case cv::COLOR_BayerGB2BGR: case cv::COLOR_BayerRG2BGR: case cv::COLOR_BayerGR2BGR:
        bayer_to_bgr(_src, _dst, dcn, code == cv::COLOR_BayerBG2BGR || code == cv::COLOR_BayerGB2BGR, code == cv::COLOR_BayerGB2BGR || code == cv::COLOR_BayerGR2BGR, stream);
        break;

    case COLOR_BayerBG2BGR_MHT: case COLOR_BayerGB2BGR_MHT: case COLOR_BayerRG2BGR_MHT: case COLOR_BayerGR2BGR_MHT:
    {
        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();
        const int depth = _src.depth();

        CV_Assert( depth == CV_8U );
        CV_Assert( src.channels() == 1 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(_src.size(), CV_MAKE_TYPE(depth, dcn));
        GpuMat dst = _dst.getGpuMat();

        dst.setTo(Scalar::all(0), stream);

        Size wholeSize;
        Point ofs;
        src.locateROI(wholeSize, ofs);
        PtrStepSzb srcWhole(wholeSize.height, wholeSize.width, src.datastart, src.step);

        const int2 firstRed = make_int2(code == COLOR_BayerRG2BGR_MHT || code == COLOR_BayerGB2BGR_MHT ? 0 : 1,
                                        code == COLOR_BayerRG2BGR_MHT || code == COLOR_BayerGR2BGR_MHT ? 0 : 1);

        if (dcn == 3)
            cv::cuda::device::MHCdemosaic<3>(srcWhole, make_int2(ofs.x, ofs.y), dst, firstRed, StreamAccessor::getStream(stream));
        else
            cv::cuda::device::MHCdemosaic<4>(srcWhole, make_int2(ofs.x, ofs.y), dst, firstRed, StreamAccessor::getStream(stream));

        break;
    }

    case COLOR_BayerBG2GRAY_MHT: case COLOR_BayerGB2GRAY_MHT: case COLOR_BayerRG2GRAY_MHT: case COLOR_BayerGR2GRAY_MHT:
    {
        GpuMat src = _src.getGpuMat();
        const int depth = _src.depth();

        CV_Assert( depth == CV_8U );

        _dst.create(_src.size(), CV_MAKE_TYPE(depth, 1));
        GpuMat dst = _dst.getGpuMat();

        dst.setTo(Scalar::all(0), stream);

        Size wholeSize;
        Point ofs;
        src.locateROI(wholeSize, ofs);
        PtrStepSzb srcWhole(wholeSize.height, wholeSize.width, src.datastart, src.step);

        const int2 firstRed = make_int2(code == COLOR_BayerRG2BGR_MHT || code == COLOR_BayerGB2BGR_MHT ? 0 : 1,
                                        code == COLOR_BayerRG2BGR_MHT || code == COLOR_BayerGR2BGR_MHT ? 0 : 1);

        cv::cuda::device::MHCdemosaic<1>(srcWhole, make_int2(ofs.x, ofs.y), dst, firstRed, StreamAccessor::getStream(stream));

        break;
    }

    default:
        CV_Error(Error::StsBadFlag, "Unknown / unsupported color conversion code");
    }
}

////////////////////////////////////////////////////////////////////////
// swapChannels

void cv::cuda::swapChannels(InputOutputArray _image, const int dstOrder[4], Stream& _stream)
{
    GpuMat image = _image.getGpuMat();

    CV_Assert( image.type() == CV_8UC4 );

    cudaStream_t stream = StreamAccessor::getStream(_stream);
    NppStreamHandler h(stream);

    NppiSize sz;
    sz.width  = image.cols;
    sz.height = image.rows;

    nppSafeCall( nppiSwapChannels_8u_C4IR(image.ptr<Npp8u>(), static_cast<int>(image.step), sz, dstOrder) );

    if (stream == 0)
        cudaSafeCall( cudaDeviceSynchronize() );
}

////////////////////////////////////////////////////////////////////////
// gammaCorrection

void cv::cuda::gammaCorrection(InputArray _src, OutputArray _dst, bool forward, Stream& stream)
{
#if (CUDA_VERSION < 5000)
    (void) _src;
    (void) _dst;
    (void) forward;
    (void) stream;
    CV_Error(Error::StsNotImplemented, "This function works only with CUDA 5.0 or higher");
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

    GpuMat src = _src.getGpuMat();

    CV_Assert( src.type() == CV_8UC3 || src.type() == CV_8UC4 );

    _dst.create(src.size(), src.type());
    GpuMat dst = _dst.getGpuMat();

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

////////////////////////////////////////////////////////////////////////
// alphaComp

namespace
{
    template <int DEPTH> struct NppAlphaCompFunc
    {
        typedef typename NPPTypeTraits<DEPTH>::npp_type npp_t;

        typedef NppStatus (*func_t)(const npp_t* pSrc1, int nSrc1Step, const npp_t* pSrc2, int nSrc2Step, npp_t* pDst, int nDstStep, NppiSize oSizeROI, NppiAlphaOp eAlphaOp);
    };

    template <int DEPTH, typename NppAlphaCompFunc<DEPTH>::func_t func> struct NppAlphaComp
    {
        typedef typename NPPTypeTraits<DEPTH>::npp_type npp_t;

        static void call(const GpuMat& img1, const GpuMat& img2, GpuMat& dst, NppiAlphaOp eAlphaOp, cudaStream_t stream)
        {
            NppStreamHandler h(stream);

            NppiSize oSizeROI;
            oSizeROI.width = img1.cols;
            oSizeROI.height = img2.rows;

            nppSafeCall( func(img1.ptr<npp_t>(), static_cast<int>(img1.step), img2.ptr<npp_t>(), static_cast<int>(img2.step),
                              dst.ptr<npp_t>(), static_cast<int>(dst.step), oSizeROI, eAlphaOp) );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
}

void cv::cuda::alphaComp(InputArray _img1, InputArray _img2, OutputArray _dst, int alpha_op, Stream& stream)
{
    static const NppiAlphaOp npp_alpha_ops[] = {
        NPPI_OP_ALPHA_OVER,
        NPPI_OP_ALPHA_IN,
        NPPI_OP_ALPHA_OUT,
        NPPI_OP_ALPHA_ATOP,
        NPPI_OP_ALPHA_XOR,
        NPPI_OP_ALPHA_PLUS,
        NPPI_OP_ALPHA_OVER_PREMUL,
        NPPI_OP_ALPHA_IN_PREMUL,
        NPPI_OP_ALPHA_OUT_PREMUL,
        NPPI_OP_ALPHA_ATOP_PREMUL,
        NPPI_OP_ALPHA_XOR_PREMUL,
        NPPI_OP_ALPHA_PLUS_PREMUL,
        NPPI_OP_ALPHA_PREMUL
    };

    typedef void (*func_t)(const GpuMat& img1, const GpuMat& img2, GpuMat& dst, NppiAlphaOp eAlphaOp, cudaStream_t stream);
    static const func_t funcs[] =
    {
        NppAlphaComp<CV_8U, nppiAlphaComp_8u_AC4R>::call,
        0,
        NppAlphaComp<CV_16U, nppiAlphaComp_16u_AC4R>::call,
        0,
        NppAlphaComp<CV_32S, nppiAlphaComp_32s_AC4R>::call,
        NppAlphaComp<CV_32F, nppiAlphaComp_32f_AC4R>::call
    };

    GpuMat img1 = _img1.getGpuMat();
    GpuMat img2 = _img2.getGpuMat();

    CV_Assert( img1.type() == CV_8UC4 || img1.type() == CV_16UC4 || img1.type() == CV_32SC4 || img1.type() == CV_32FC4 );
    CV_Assert( img1.size() == img2.size() && img1.type() == img2.type() );

    _dst.create(img1.size(), img1.type());
    GpuMat dst = _dst.getGpuMat();

    const func_t func = funcs[img1.depth()];

    func(img1, img2, dst, npp_alpha_ops[alpha_op], StreamAccessor::getStream(stream));
}

#endif /* !defined (HAVE_CUDA) */
