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
    typedef void (*gpu_func_t)(const GpuMat& _src, GpuMat& _dst, Stream& stream);

    void BGR_to_RGB(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[] = {BGR_to_RGB_8u, 0, BGR_to_RGB_16u, 0, 0, BGR_to_RGB_32f};

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), 3));
        GpuMat dst = _dst.getGpuMat();

        funcs[src.depth()](src, dst, stream);
    }

    void BGR_to_BGRA(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[] = {BGR_to_BGRA_8u, 0, BGR_to_BGRA_16u, 0, 0, BGR_to_BGRA_32f};

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), 4));
        GpuMat dst = _dst.getGpuMat();

        funcs[src.depth()](src, dst, stream);
    }

    void BGR_to_RGBA(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[] = {BGR_to_RGBA_8u, 0, BGR_to_RGBA_16u, 0, 0, BGR_to_RGBA_32f};

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), 4));
        GpuMat dst = _dst.getGpuMat();

        funcs[src.depth()](src, dst, stream);
    }

    void BGRA_to_BGR(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[] = {BGRA_to_BGR_8u, 0, BGRA_to_BGR_16u, 0, 0, BGRA_to_BGR_32f};

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), 3));
        GpuMat dst = _dst.getGpuMat();

        funcs[src.depth()](src, dst, stream);
    }

    void BGRA_to_RGB(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[] = {BGRA_to_RGB_8u, 0, BGRA_to_RGB_16u, 0, 0, BGRA_to_RGB_32f};

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), 3));
        GpuMat dst = _dst.getGpuMat();

        funcs[src.depth()](src, dst, stream);
    }

    void BGRA_to_RGBA(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[] = {BGRA_to_RGBA_8u, 0, BGRA_to_RGBA_16u, 0, 0, BGRA_to_RGBA_32f};

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), 4));
        GpuMat dst = _dst.getGpuMat();

        funcs[src.depth()](src, dst, stream);
    }

    void BGR_to_BGR555(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 3 );

        _dst.create(src.size(), CV_8UC2);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::BGR_to_BGR555(src, dst, stream);
    }

    void BGR_to_BGR565(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 3 );

        _dst.create(src.size(), CV_8UC2);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::BGR_to_BGR565(src, dst, stream);
    }

    void RGB_to_BGR555(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 3 );

        _dst.create(src.size(), CV_8UC2);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::RGB_to_BGR555(src, dst, stream);
    }

    void RGB_to_BGR565(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 3 );

        _dst.create(src.size(), CV_8UC2);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::RGB_to_BGR565(src, dst, stream);
    }

    void BGRA_to_BGR555(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 4 );

        _dst.create(src.size(), CV_8UC2);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::BGRA_to_BGR555(src, dst, stream);
    }

    void BGRA_to_BGR565(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 4 );

        _dst.create(src.size(), CV_8UC2);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::BGRA_to_BGR565(src, dst, stream);
    }

    void RGBA_to_BGR555(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 4 );

        _dst.create(src.size(), CV_8UC2);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::RGBA_to_BGR555(src, dst, stream);
    }

    void RGBA_to_BGR565(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 4 );

        _dst.create(src.size(), CV_8UC2);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::RGBA_to_BGR565(src, dst, stream);
    }

    void BGR555_to_RGB(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 2 );

        _dst.create(src.size(), CV_8UC3);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::BGR555_to_RGB(src, dst, stream);
    }

    void BGR565_to_RGB(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 2 );

        _dst.create(src.size(), CV_8UC3);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::BGR565_to_RGB(src, dst, stream);
    }

    void BGR555_to_BGR(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 2 );

        _dst.create(src.size(), CV_8UC3);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::BGR555_to_BGR(src, dst, stream);
    }

    void BGR565_to_BGR(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 2 );

        _dst.create(src.size(), CV_8UC3);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::BGR565_to_BGR(src, dst, stream);
    }

    void BGR555_to_RGBA(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 2 );

        _dst.create(src.size(), CV_8UC4);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::BGR555_to_RGBA(src, dst, stream);
    }

    void BGR565_to_RGBA(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 2 );

        _dst.create(src.size(), CV_8UC4);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::BGR565_to_RGBA(src, dst, stream);
    }

    void BGR555_to_BGRA(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 2 );

        _dst.create(src.size(), CV_8UC4);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::BGR555_to_BGRA(src, dst, stream);
    }

    void BGR565_to_BGRA(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 2 );

        _dst.create(src.size(), CV_8UC4);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::BGR565_to_BGRA(src, dst, stream);
    }

    void GRAY_to_BGR(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[] = {GRAY_to_BGR_8u, 0, GRAY_to_BGR_16u, 0, 0, GRAY_to_BGR_32f};

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 1 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), 3));
        GpuMat dst = _dst.getGpuMat();

        funcs[src.depth()](src, dst, stream);
    }

    void GRAY_to_BGRA(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[] = {GRAY_to_BGRA_8u, 0, GRAY_to_BGRA_16u, 0, 0, GRAY_to_BGRA_32f};

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 1 );

        _dst.create(src.size(), CV_MAKETYPE(src.depth(), 4));
        GpuMat dst = _dst.getGpuMat();

        funcs[src.depth()](src, dst, stream);
    }

    void GRAY_to_BGR555(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 1 );

        _dst.create(src.size(), CV_8UC2);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::GRAY_to_BGR555(src, dst, stream);
    }

    void GRAY_to_BGR565(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 1 );

        _dst.create(src.size(), CV_8UC2);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::GRAY_to_BGR565(src, dst, stream);
    }

    void BGR555_to_GRAY(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 2 );

        _dst.create(src.size(), CV_8UC1);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::BGR555_to_GRAY(src, dst, stream);
    }

    void BGR565_to_GRAY(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U );
        CV_Assert( src.channels() == 2 );

        _dst.create(src.size(), CV_8UC1);
        GpuMat dst = _dst.getGpuMat();

        cv::cuda::device::BGR565_to_GRAY(src, dst, stream);
    }

    void RGB_to_GRAY(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[] = {RGB_to_GRAY_8u, 0, RGB_to_GRAY_16u, 0, 0, RGB_to_GRAY_32f};

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), 1));
        GpuMat dst = _dst.getGpuMat();

        funcs[src.depth()](src, dst, stream);
    }

    void BGR_to_GRAY(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[] = {BGR_to_GRAY_8u, 0, BGR_to_GRAY_16u, 0, 0, BGR_to_GRAY_32f};

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), 1));
        GpuMat dst = _dst.getGpuMat();

        funcs[src.depth()](src, dst, stream);
    }

    void RGBA_to_GRAY(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[] = {RGBA_to_GRAY_8u, 0, RGBA_to_GRAY_16u, 0, 0, RGBA_to_GRAY_32f};

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), 1));
        GpuMat dst = _dst.getGpuMat();

        funcs[src.depth()](src, dst, stream);
    }

    void BGRA_to_GRAY(InputArray _src, OutputArray _dst, int, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[] = {BGRA_to_GRAY_8u, 0, BGRA_to_GRAY_16u, 0, 0, BGRA_to_GRAY_32f};

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), 1));
        GpuMat dst = _dst.getGpuMat();

        funcs[src.depth()](src, dst, stream);
    }

    void RGB_to_YUV(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {RGB_to_YUV_8u, 0, RGB_to_YUV_16u, 0, 0, RGB_to_YUV_32f},
                {RGBA_to_YUV_8u, 0, RGBA_to_YUV_16u, 0, 0, RGBA_to_YUV_32f}
            },
            {
                {RGB_to_YUV4_8u, 0, RGB_to_YUV4_16u, 0, 0, RGB_to_YUV4_32f},
                {RGBA_to_YUV4_8u, 0, RGBA_to_YUV4_16u, 0, 0, RGBA_to_YUV4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, stream);
    }

    void BGR_to_YUV(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {BGR_to_YUV_8u, 0, BGR_to_YUV_16u, 0, 0, BGR_to_YUV_32f},
                {BGRA_to_YUV_8u, 0, BGRA_to_YUV_16u, 0, 0, BGRA_to_YUV_32f}
            },
            {
                {BGR_to_YUV4_8u, 0, BGR_to_YUV4_16u, 0, 0, BGR_to_YUV4_32f},
                {BGRA_to_YUV4_8u, 0, BGRA_to_YUV4_16u, 0, 0, BGRA_to_YUV4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, stream);
    }

    void YUV_to_RGB(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {YUV_to_RGB_8u, 0, YUV_to_RGB_16u, 0, 0, YUV_to_RGB_32f},
                {YUV4_to_RGB_8u, 0, YUV4_to_RGB_16u, 0, 0, YUV4_to_RGB_32f}
            },
            {
                {YUV_to_RGBA_8u, 0, YUV_to_RGBA_16u, 0, 0, YUV_to_RGBA_32f},
                {YUV4_to_RGBA_8u, 0, YUV4_to_RGBA_16u, 0, 0, YUV4_to_RGBA_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, stream);
    }

    void YUV_to_BGR(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {YUV_to_BGR_8u, 0, YUV_to_BGR_16u, 0, 0, YUV_to_BGR_32f},
                {YUV4_to_BGR_8u, 0, YUV4_to_BGR_16u, 0, 0, YUV4_to_BGR_32f}
            },
            {
                {YUV_to_BGRA_8u, 0, YUV_to_BGRA_16u, 0, 0, YUV_to_BGRA_32f},
                {YUV4_to_BGRA_8u, 0, YUV4_to_BGRA_16u, 0, 0, YUV4_to_BGRA_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, stream);
    }

    void RGB_to_YCrCb(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {RGB_to_YCrCb_8u, 0, RGB_to_YCrCb_16u, 0, 0, RGB_to_YCrCb_32f},
                {RGBA_to_YCrCb_8u, 0, RGBA_to_YCrCb_16u, 0, 0, RGBA_to_YCrCb_32f}
            },
            {
                {RGB_to_YCrCb4_8u, 0, RGB_to_YCrCb4_16u, 0, 0, RGB_to_YCrCb4_32f},
                {RGBA_to_YCrCb4_8u, 0, RGBA_to_YCrCb4_16u, 0, 0, RGBA_to_YCrCb4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, stream);
    }

    void BGR_to_YCrCb(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {BGR_to_YCrCb_8u, 0, BGR_to_YCrCb_16u, 0, 0, BGR_to_YCrCb_32f},
                {BGRA_to_YCrCb_8u, 0, BGRA_to_YCrCb_16u, 0, 0, BGRA_to_YCrCb_32f}
            },
            {
                {BGR_to_YCrCb4_8u, 0, BGR_to_YCrCb4_16u, 0, 0, BGR_to_YCrCb4_32f},
                {BGRA_to_YCrCb4_8u, 0, BGRA_to_YCrCb4_16u, 0, 0, BGRA_to_YCrCb4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, stream);
    }

    void YCrCb_to_RGB(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {YCrCb_to_RGB_8u, 0, YCrCb_to_RGB_16u, 0, 0, YCrCb_to_RGB_32f},
                {YCrCb4_to_RGB_8u, 0, YCrCb4_to_RGB_16u, 0, 0, YCrCb4_to_RGB_32f}
            },
            {
                {YCrCb_to_RGBA_8u, 0, YCrCb_to_RGBA_16u, 0, 0, YCrCb_to_RGBA_32f},
                {YCrCb4_to_RGBA_8u, 0, YCrCb4_to_RGBA_16u, 0, 0, YCrCb4_to_RGBA_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, stream);
    }

    void YCrCb_to_BGR(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {YCrCb_to_BGR_8u, 0, YCrCb_to_BGR_16u, 0, 0, YCrCb_to_BGR_32f},
                {YCrCb4_to_BGR_8u, 0, YCrCb4_to_BGR_16u, 0, 0, YCrCb4_to_BGR_32f}
            },
            {
                {YCrCb_to_BGRA_8u, 0, YCrCb_to_BGRA_16u, 0, 0, YCrCb_to_BGRA_32f},
                {YCrCb4_to_BGRA_8u, 0, YCrCb4_to_BGRA_16u, 0, 0, YCrCb4_to_BGRA_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, stream);
    }

    void RGB_to_XYZ(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {RGB_to_XYZ_8u, 0, RGB_to_XYZ_16u, 0, 0, RGB_to_XYZ_32f},
                {RGBA_to_XYZ_8u, 0, RGBA_to_XYZ_16u, 0, 0, RGBA_to_XYZ_32f}
            },
            {
                {RGB_to_XYZ4_8u, 0, RGB_to_XYZ4_16u, 0, 0, RGB_to_XYZ4_32f},
                {RGBA_to_XYZ4_8u, 0, RGBA_to_XYZ4_16u, 0, 0, RGBA_to_XYZ4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, stream);
    }

    void BGR_to_XYZ(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {BGR_to_XYZ_8u, 0, BGR_to_XYZ_16u, 0, 0, BGR_to_XYZ_32f},
                {BGRA_to_XYZ_8u, 0, BGRA_to_XYZ_16u, 0, 0, BGRA_to_XYZ_32f}
            },
            {
                {BGR_to_XYZ4_8u, 0, BGR_to_XYZ4_16u, 0, 0, BGR_to_XYZ4_32f},
                {BGRA_to_XYZ4_8u, 0, BGRA_to_XYZ4_16u, 0, 0, BGRA_to_XYZ4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, stream);
    }

    void XYZ_to_RGB(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {XYZ_to_RGB_8u, 0, XYZ_to_RGB_16u, 0, 0, XYZ_to_RGB_32f},
                {XYZ4_to_RGB_8u, 0, XYZ4_to_RGB_16u, 0, 0, XYZ4_to_RGB_32f}
            },
            {
                {XYZ_to_RGBA_8u, 0, XYZ_to_RGBA_16u, 0, 0, XYZ_to_RGBA_32f},
                {XYZ4_to_RGBA_8u, 0, XYZ4_to_RGBA_16u, 0, 0, XYZ4_to_RGBA_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, stream);
    }

    void XYZ_to_BGR(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {XYZ_to_BGR_8u, 0, XYZ_to_BGR_16u, 0, 0, XYZ_to_BGR_32f},
                {XYZ4_to_BGR_8u, 0, XYZ4_to_BGR_16u, 0, 0, XYZ4_to_BGR_32f}
            },
            {
                {XYZ_to_BGRA_8u, 0, XYZ_to_BGRA_16u, 0, 0, XYZ_to_BGRA_32f},
                {XYZ4_to_BGRA_8u, 0, XYZ4_to_BGRA_16u, 0, 0, XYZ4_to_BGRA_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, stream);
    }

    void RGB_to_HSV(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {RGB_to_HSV_8u, 0, 0, 0, 0, RGB_to_HSV_32f},
                {RGBA_to_HSV_8u, 0, 0, 0, 0, RGBA_to_HSV_32f},
            },
            {
                {RGB_to_HSV4_8u, 0, 0, 0, 0, RGB_to_HSV4_32f},
                {RGBA_to_HSV4_8u, 0, 0, 0, 0, RGBA_to_HSV4_32f},
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, stream);
    }

    void BGR_to_HSV(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {BGR_to_HSV_8u, 0, 0, 0, 0, BGR_to_HSV_32f},
                {BGRA_to_HSV_8u, 0, 0, 0, 0, BGRA_to_HSV_32f}
            },
            {
                {BGR_to_HSV4_8u, 0, 0, 0, 0, BGR_to_HSV4_32f},
                {BGRA_to_HSV4_8u, 0, 0, 0, 0, BGRA_to_HSV4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, stream);
    }

    void HSV_to_RGB(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {HSV_to_RGB_8u, 0, 0, 0, 0, HSV_to_RGB_32f},
                {HSV4_to_RGB_8u, 0, 0, 0, 0, HSV4_to_RGB_32f}
            },
            {
                {HSV_to_RGBA_8u, 0, 0, 0, 0, HSV_to_RGBA_32f},
                {HSV4_to_RGBA_8u, 0, 0, 0, 0, HSV4_to_RGBA_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, stream);
    }

    void HSV_to_BGR(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {HSV_to_BGR_8u, 0, 0, 0, 0, HSV_to_BGR_32f},
                {HSV4_to_BGR_8u, 0, 0, 0, 0, HSV4_to_BGR_32f}
            },
            {
                {HSV_to_BGRA_8u, 0, 0, 0, 0, HSV_to_BGRA_32f},
                {HSV4_to_BGRA_8u, 0, 0, 0, 0, HSV4_to_BGRA_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, stream);
    }

    void RGB_to_HLS(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {RGB_to_HLS_8u, 0, 0, 0, 0, RGB_to_HLS_32f},
                {RGBA_to_HLS_8u, 0, 0, 0, 0, RGBA_to_HLS_32f},
            },
            {
                {RGB_to_HLS4_8u, 0, 0, 0, 0, RGB_to_HLS4_32f},
                {RGBA_to_HLS4_8u, 0, 0, 0, 0, RGBA_to_HLS4_32f},
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, stream);
    }

    void BGR_to_HLS(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {BGR_to_HLS_8u, 0, 0, 0, 0, BGR_to_HLS_32f},
                {BGRA_to_HLS_8u, 0, 0, 0, 0, BGRA_to_HLS_32f}
            },
            {
                {BGR_to_HLS4_8u, 0, 0, 0, 0, BGR_to_HLS4_32f},
                {BGRA_to_HLS4_8u, 0, 0, 0, 0, BGRA_to_HLS4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, stream);
    }

    void HLS_to_RGB(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {HLS_to_RGB_8u, 0, 0, 0, 0, HLS_to_RGB_32f},
                {HLS4_to_RGB_8u, 0, 0, 0, 0, HLS4_to_RGB_32f}
            },
            {
                {HLS_to_RGBA_8u, 0, 0, 0, 0, HLS_to_RGBA_32f},
                {HLS4_to_RGBA_8u, 0, 0, 0, 0, HLS4_to_RGBA_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, stream);
    }

    void HLS_to_BGR(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {HLS_to_BGR_8u, 0, 0, 0, 0, HLS_to_BGR_32f},
                {HLS4_to_BGR_8u, 0, 0, 0, 0, HLS4_to_BGR_32f}
            },
            {
                {HLS_to_BGRA_8u, 0, 0, 0, 0, HLS_to_BGRA_32f},
                {HLS4_to_BGRA_8u, 0, 0, 0, 0, HLS4_to_BGRA_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, stream);
    }

    void RGB_to_HSV_FULL(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {RGB_to_HSV_FULL_8u, 0, 0, 0, 0, RGB_to_HSV_FULL_32f},
                {RGBA_to_HSV_FULL_8u, 0, 0, 0, 0, RGBA_to_HSV_FULL_32f},
            },
            {
                {RGB_to_HSV4_FULL_8u, 0, 0, 0, 0, RGB_to_HSV4_FULL_32f},
                {RGBA_to_HSV4_FULL_8u, 0, 0, 0, 0, RGBA_to_HSV4_FULL_32f},
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, stream);
    }

    void BGR_to_HSV_FULL(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {BGR_to_HSV_FULL_8u, 0, 0, 0, 0, BGR_to_HSV_FULL_32f},
                {BGRA_to_HSV_FULL_8u, 0, 0, 0, 0, BGRA_to_HSV_FULL_32f}
            },
            {
                {BGR_to_HSV4_FULL_8u, 0, 0, 0, 0, BGR_to_HSV4_FULL_32f},
                {BGRA_to_HSV4_FULL_8u, 0, 0, 0, 0, BGRA_to_HSV4_FULL_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, stream);
    }

    void HSV_to_RGB_FULL(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {HSV_to_RGB_FULL_8u, 0, 0, 0, 0, HSV_to_RGB_FULL_32f},
                {HSV4_to_RGB_FULL_8u, 0, 0, 0, 0, HSV4_to_RGB_FULL_32f}
            },
            {
                {HSV_to_RGBA_FULL_8u, 0, 0, 0, 0, HSV_to_RGBA_FULL_32f},
                {HSV4_to_RGBA_FULL_8u, 0, 0, 0, 0, HSV4_to_RGBA_FULL_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, stream);
    }

    void HSV_to_BGR_FULL(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {HSV_to_BGR_FULL_8u, 0, 0, 0, 0, HSV_to_BGR_FULL_32f},
                {HSV4_to_BGR_FULL_8u, 0, 0, 0, 0, HSV4_to_BGR_FULL_32f}
            },
            {
                {HSV_to_BGRA_FULL_8u, 0, 0, 0, 0, HSV_to_BGRA_FULL_32f},
                {HSV4_to_BGRA_FULL_8u, 0, 0, 0, 0, HSV4_to_BGRA_FULL_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, stream);
    }

    void RGB_to_HLS_FULL(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {RGB_to_HLS_FULL_8u, 0, 0, 0, 0, RGB_to_HLS_FULL_32f},
                {RGBA_to_HLS_FULL_8u, 0, 0, 0, 0, RGBA_to_HLS_FULL_32f},
            },
            {
                {RGB_to_HLS4_FULL_8u, 0, 0, 0, 0, RGB_to_HLS4_FULL_32f},
                {RGBA_to_HLS4_FULL_8u, 0, 0, 0, 0, RGBA_to_HLS4_FULL_32f},
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, stream);
    }

    void BGR_to_HLS_FULL(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {BGR_to_HLS_FULL_8u, 0, 0, 0, 0, BGR_to_HLS_FULL_32f},
                {BGRA_to_HLS_FULL_8u, 0, 0, 0, 0, BGRA_to_HLS_FULL_32f}
            },
            {
                {BGR_to_HLS4_FULL_8u, 0, 0, 0, 0, BGR_to_HLS4_FULL_32f},
                {BGRA_to_HLS4_FULL_8u, 0, 0, 0, 0, BGRA_to_HLS4_FULL_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, stream);
    }

    void HLS_to_RGB_FULL(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {HLS_to_RGB_FULL_8u, 0, 0, 0, 0, HLS_to_RGB_FULL_32f},
                {HLS4_to_RGB_FULL_8u, 0, 0, 0, 0, HLS4_to_RGB_FULL_32f}
            },
            {
                {HLS_to_RGBA_FULL_8u, 0, 0, 0, 0, HLS_to_RGBA_FULL_32f},
                {HLS4_to_RGBA_FULL_8u, 0, 0, 0, 0, HLS4_to_RGBA_FULL_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, stream);
    }

    void HLS_to_BGR_FULL(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][6] =
        {
            {
                {HLS_to_BGR_FULL_8u, 0, 0, 0, 0, HLS_to_BGR_FULL_32f},
                {HLS4_to_BGR_FULL_8u, 0, 0, 0, 0, HLS4_to_BGR_FULL_32f}
            },
            {
                {HLS_to_BGRA_FULL_8u, 0, 0, 0, 0, HLS_to_BGRA_FULL_32f},
                {HLS4_to_BGRA_FULL_8u, 0, 0, 0, 0, HLS4_to_BGRA_FULL_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth()](src, dst, stream);
    }

    void BGR_to_Lab(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][2] =
        {
            {
                {BGR_to_Lab_8u, BGR_to_Lab_32f},
                {BGRA_to_Lab_8u, BGRA_to_Lab_32f}
            },
            {
                {BGR_to_Lab4_8u, BGR_to_Lab4_32f},
                {BGRA_to_Lab4_8u, BGRA_to_Lab4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, stream);
    }

    void RGB_to_Lab(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][2] =
        {
            {
                {RGB_to_Lab_8u, RGB_to_Lab_32f},
                {RGBA_to_Lab_8u, RGBA_to_Lab_32f}
            },
            {
                {RGB_to_Lab4_8u, RGB_to_Lab4_32f},
                {RGBA_to_Lab4_8u, RGBA_to_Lab4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, stream);
    }

    void LBGR_to_Lab(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][2] =
        {
            {
                {LBGR_to_Lab_8u, LBGR_to_Lab_32f},
                {LBGRA_to_Lab_8u, LBGRA_to_Lab_32f}
            },
            {
                {LBGR_to_Lab4_8u, LBGR_to_Lab4_32f},
                {LBGRA_to_Lab4_8u, LBGRA_to_Lab4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, stream);
    }

    void LRGB_to_Lab(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][2] =
        {
            {
                {LRGB_to_Lab_8u, LRGB_to_Lab_32f},
                {LRGBA_to_Lab_8u, LRGBA_to_Lab_32f}
            },
            {
                {LRGB_to_Lab4_8u, LRGB_to_Lab4_32f},
                {LRGBA_to_Lab4_8u, LRGBA_to_Lab4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, stream);
    }

    void Lab_to_BGR(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][2] =
        {
            {
                {Lab_to_BGR_8u, Lab_to_BGR_32f},
                {Lab4_to_BGR_8u, Lab4_to_BGR_32f}
            },
            {
                {Lab_to_BGRA_8u, Lab_to_BGRA_32f},
                {Lab4_to_BGRA_8u, Lab4_to_BGRA_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, stream);
    }

    void Lab_to_RGB(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][2] =
        {
            {
                {Lab_to_RGB_8u, Lab_to_RGB_32f},
                {Lab4_to_RGB_8u, Lab4_to_RGB_32f}
            },
            {
                {Lab_to_RGBA_8u, Lab_to_RGBA_32f},
                {Lab4_to_RGBA_8u, Lab4_to_RGBA_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, stream);
    }

    void Lab_to_LBGR(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][2] =
        {
            {
                {Lab_to_LBGR_8u, Lab_to_LBGR_32f},
                {Lab4_to_LBGR_8u, Lab4_to_LBGR_32f}
            },
            {
                {Lab_to_LBGRA_8u, Lab_to_LBGRA_32f},
                {Lab4_to_LBGRA_8u, Lab4_to_LBGRA_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, stream);
    }

    void Lab_to_LRGB(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][2] =
        {
            {
                {Lab_to_LRGB_8u, Lab_to_LRGB_32f},
                {Lab4_to_LRGB_8u, Lab4_to_LRGB_32f}
            },
            {
                {Lab_to_LRGBA_8u, Lab_to_LRGBA_32f},
                {Lab4_to_LRGBA_8u, Lab4_to_LRGBA_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, stream);
    }

    void BGR_to_Luv(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][2] =
        {
            {
                {BGR_to_Luv_8u, BGR_to_Luv_32f},
                {BGRA_to_Luv_8u, BGRA_to_Luv_32f}
            },
            {
                {BGR_to_Luv4_8u, BGR_to_Luv4_32f},
                {BGRA_to_Luv4_8u, BGRA_to_Luv4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, stream);
    }

    void RGB_to_Luv(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][2] =
        {
            {
                {RGB_to_Luv_8u, RGB_to_Luv_32f},
                {RGBA_to_Luv_8u, RGBA_to_Luv_32f}
            },
            {
                {RGB_to_Luv4_8u, RGB_to_Luv4_32f},
                {RGBA_to_Luv4_8u, RGBA_to_Luv4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, stream);
    }

    void LBGR_to_Luv(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][2] =
        {
            {
                {LBGR_to_Luv_8u, LBGR_to_Luv_32f},
                {LBGRA_to_Luv_8u, LBGRA_to_Luv_32f}
            },
            {
                {LBGR_to_Luv4_8u, LBGR_to_Luv4_32f},
                {LBGRA_to_Luv4_8u, LBGRA_to_Luv4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, stream);
    }

    void LRGB_to_Luv(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][2] =
        {
            {
                {LRGB_to_Luv_8u, LRGB_to_Luv_32f},
                {LRGBA_to_Luv_8u, LRGBA_to_Luv_32f}
            },
            {
                {LRGB_to_Luv4_8u, LRGB_to_Luv4_32f},
                {LRGBA_to_Luv4_8u, LRGBA_to_Luv4_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, stream);
    }

    void Luv_to_BGR(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][2] =
        {
            {
                {Luv_to_BGR_8u, Luv_to_BGR_32f},
                {Luv4_to_BGR_8u, Luv4_to_BGR_32f}
            },
            {
                {Luv_to_BGRA_8u, Luv_to_BGRA_32f},
                {Luv4_to_BGRA_8u, Luv4_to_BGRA_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, stream);
    }

    void Luv_to_RGB(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][2] =
        {
            {
                {Luv_to_RGB_8u, Luv_to_RGB_32f},
                {Luv4_to_RGB_8u, Luv4_to_RGB_32f}
            },
            {
                {Luv_to_RGBA_8u, Luv_to_RGBA_32f},
                {Luv4_to_RGBA_8u, Luv4_to_RGBA_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, stream);
    }

    void Luv_to_LBGR(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][2] =
        {
            {
                {Luv_to_LBGR_8u, Luv_to_LBGR_32f},
                {Luv4_to_LBGR_8u, Luv4_to_LBGR_32f}
            },
            {
                {Luv_to_LBGRA_8u, Luv_to_LBGRA_32f},
                {Luv4_to_LBGRA_8u, Luv4_to_LBGRA_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, stream);
    }

    void Luv_to_LRGB(InputArray _src, OutputArray _dst, int dcn, Stream& stream)
    {
        using namespace cv::cuda::device;
        static const gpu_func_t funcs[2][2][2] =
        {
            {
                {Luv_to_LRGB_8u, Luv_to_LRGB_32f},
                {Luv4_to_LRGB_8u, Luv4_to_LRGB_32f}
            },
            {
                {Luv_to_LRGBA_8u, Luv_to_LRGBA_32f},
                {Luv4_to_LRGBA_8u, Luv4_to_LRGBA_32f}
            }
        };

        if (dcn <= 0) dcn = 3;

        GpuMat src = _src.getGpuMat();

        CV_Assert( src.depth() == CV_8U || src.depth() == CV_32F );
        CV_Assert( src.channels() == 3 || src.channels() == 4 );
        CV_Assert( dcn == 3 || dcn == 4 );

        _dst.create(src.size(), CV_MAKE_TYPE(src.depth(), dcn));
        GpuMat dst = _dst.getGpuMat();

        funcs[dcn == 4][src.channels() == 4][src.depth() == CV_32F](src, dst, stream);
    }

    void RGBA_to_mBGRA(InputArray _src, OutputArray _dst, int, Stream& _stream)
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

    void bayer_to_BGR(InputArray _src, OutputArray _dst, int dcn, bool blue_last, bool start_with_green, Stream& stream)
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
    void bayerBG_to_BGR(InputArray src, OutputArray dst, int dcn, Stream& stream)
    {
        bayer_to_BGR(src, dst, dcn, false, false, stream);
    }
    void bayeRGB_to_BGR(InputArray src, OutputArray dst, int dcn, Stream& stream)
    {
        bayer_to_BGR(src, dst, dcn, false, true, stream);
    }
    void bayerRG_to_BGR(InputArray src, OutputArray dst, int dcn, Stream& stream)
    {
        bayer_to_BGR(src, dst, dcn, true, false, stream);
    }
    void bayerGR_to_BGR(InputArray src, OutputArray dst, int dcn, Stream& stream)
    {
        bayer_to_BGR(src, dst, dcn, true, true, stream);
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
    void bayeRGB_to_GRAY(InputArray src, OutputArray dst, int /*dcn*/, Stream& stream)
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
        BGR_to_BGRA,            // CV_BGR2BGRA    =0
        BGRA_to_BGR,            // CV_BGRA2BGR    =1
        BGR_to_RGBA,            // CV_BGR2RGBA    =2
        BGRA_to_RGB,            // CV_RGBA2BGR    =3
        BGR_to_RGB,             // CV_BGR2RGB     =4
        BGRA_to_RGBA,           // CV_BGRA2RGBA   =5

        BGR_to_GRAY,            // CV_BGR2GRAY    =6
        RGB_to_GRAY,            // CV_RGB2GRAY    =7
        GRAY_to_BGR,            // CV_GRAY2BGR    =8
        GRAY_to_BGRA,           // CV_GRAY2BGRA   =9
        BGRA_to_GRAY,           // CV_BGRA2GRAY   =10
        RGBA_to_GRAY,           // CV_RGBA2GRAY   =11

        BGR_to_BGR565,          // CV_BGR2BGR565  =12
        RGB_to_BGR565,          // CV_RGB2BGR565  =13
        BGR565_to_BGR,          // CV_BGR5652BGR  =14
        BGR565_to_RGB,          // CV_BGR5652RGB  =15
        BGRA_to_BGR565,         // CV_BGRA2BGR565 =16
        RGBA_to_BGR565,         // CV_RGBA2BGR565 =17
        BGR565_to_BGRA,         // CV_BGR5652BGRA =18
        BGR565_to_RGBA,         // CV_BGR5652RGBA =19

        GRAY_to_BGR565,         // CV_GRAY2BGR565 =20
        BGR565_to_GRAY,         // CV_BGR5652GRAY =21

        BGR_to_BGR555,          // CV_BGR2BGR555  =22
        RGB_to_BGR555,          // CV_RGB2BGR555  =23
        BGR555_to_BGR,          // CV_BGR5552BGR  =24
        BGR555_to_RGB,          // CV_BGR5552RGB  =25
        BGRA_to_BGR555,         // CV_BGRA2BGR555 =26
        RGBA_to_BGR555,         // CV_RGBA2BGR555 =27
        BGR555_to_BGRA,         // CV_BGR5552BGRA =28
        BGR555_to_RGBA,         // CV_BGR5552RGBA =29

        GRAY_to_BGR555,         // CV_GRAY2BGR555 =30
        BGR555_to_GRAY,         // CV_BGR5552GRAY =31

        BGR_to_XYZ,             // CV_BGR2XYZ     =32
        RGB_to_XYZ,             // CV_RGB2XYZ     =33
        XYZ_to_BGR,             // CV_XYZ2BGR     =34
        XYZ_to_RGB,             // CV_XYZ2RGB     =35

        BGR_to_YCrCb,           // CV_BGR2YCrCb   =36
        RGB_to_YCrCb,           // CV_RGB2YCrCb   =37
        YCrCb_to_BGR,           // CV_YCrCb2BGR   =38
        YCrCb_to_RGB,           // CV_YCrCb2RGB   =39

        BGR_to_HSV,             // CV_BGR2HSV     =40
        RGB_to_HSV,             // CV_RGB2HSV     =41

        0,                      //                =42
        0,                      //                =43

        BGR_to_Lab,             // CV_BGR2Lab     =44
        RGB_to_Lab,             // CV_RGB2Lab     =45

        bayerBG_to_BGR,         // CV_BayerBG2BGR =46
        bayeRGB_to_BGR,         // CV_BayeRGB2BGR =47
        bayerRG_to_BGR,         // CV_BayerRG2BGR =48
        bayerGR_to_BGR,         // CV_BayerGR2BGR =49

        BGR_to_Luv,             // CV_BGR2Luv     =50
        RGB_to_Luv,             // CV_RGB2Luv     =51

        BGR_to_HLS,             // CV_BGR2HLS     =52
        RGB_to_HLS,             // CV_RGB2HLS     =53

        HSV_to_BGR,             // CV_HSV2BGR     =54
        HSV_to_RGB,             // CV_HSV2RGB     =55

        Lab_to_BGR,             // CV_Lab2BGR     =56
        Lab_to_RGB,             // CV_Lab2RGB     =57
        Luv_to_BGR,             // CV_Luv2BGR     =58
        Luv_to_RGB,             // CV_Luv2RGB     =59

        HLS_to_BGR,             // CV_HLS2BGR     =60
        HLS_to_RGB,             // CV_HLS2RGB     =61

        0,                      // CV_BayerBG2BGR_VNG =62
        0,                      // CV_BayeRGB2BGR_VNG =63
        0,                      // CV_BayerRG2BGR_VNG =64
        0,                      // CV_BayerGR2BGR_VNG =65

        BGR_to_HSV_FULL,        // CV_BGR2HSV_FULL = 66
        RGB_to_HSV_FULL,        // CV_RGB2HSV_FULL = 67
        BGR_to_HLS_FULL,        // CV_BGR2HLS_FULL = 68
        RGB_to_HLS_FULL,        // CV_RGB2HLS_FULL = 69

        HSV_to_BGR_FULL,        // CV_HSV2BGR_FULL = 70
        HSV_to_RGB_FULL,        // CV_HSV2RGB_FULL = 71
        HLS_to_BGR_FULL,        // CV_HLS2BGR_FULL = 72
        HLS_to_RGB_FULL,        // CV_HLS2RGB_FULL = 73

        LBGR_to_Lab,            // CV_LBGR2Lab     = 74
        LRGB_to_Lab,            // CV_LRGB2Lab     = 75
        LBGR_to_Luv,            // CV_LBGR2Luv     = 76
        LRGB_to_Luv,            // CV_LRGB2Luv     = 77

        Lab_to_LBGR,            // CV_Lab2LBGR     = 78
        Lab_to_LRGB,            // CV_Lab2LRGB     = 79
        Luv_to_LBGR,            // CV_Luv2LBGR     = 80
        Luv_to_LRGB,            // CV_Luv2LRGB     = 81

        BGR_to_YUV,             // CV_BGR2YUV      = 82
        RGB_to_YUV,             // CV_RGB2YUV      = 83
        YUV_to_BGR,             // CV_YUV2BGR      = 84
        YUV_to_RGB,             // CV_YUV2RGB      = 85

        bayerBG_to_gray,        // CV_BayerBG2GRAY = 86
        bayeRGB_to_GRAY,        // CV_BayeRGB2GRAY = 87
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
        RGBA_to_mBGRA,          // CV_RGBA2mRGBA = 125,
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
    CV_Assert( !_src.empty() );

    switch (code)
    {
    case cv::COLOR_BayerBG2GRAY: case cv::COLOR_BayerGB2GRAY: case cv::COLOR_BayerRG2GRAY: case cv::COLOR_BayerGR2GRAY:
        bayer_to_gray(_src, _dst, code == cv::COLOR_BayerBG2GRAY || code == cv::COLOR_BayerGB2GRAY, code == cv::COLOR_BayerGB2GRAY || code == cv::COLOR_BayerGR2GRAY, stream);
        break;

    case cv::COLOR_BayerBG2BGR: case cv::COLOR_BayerGB2BGR: case cv::COLOR_BayerRG2BGR: case cv::COLOR_BayerGR2BGR:
        bayer_to_BGR(_src, _dst, dcn, code == cv::COLOR_BayerBG2BGR || code == cv::COLOR_BayerGB2BGR, code == cv::COLOR_BayerGB2BGR || code == cv::COLOR_BayerGR2BGR, stream);
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
