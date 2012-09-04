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
//     and/or other GpuMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or bpied warranties, including, but not limited to, the bpied
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

void cv::gpu::bilateralFilter(const GpuMat&, GpuMat&, int, float, float, int, Stream&) { throw_nogpu(); }
void cv::gpu::nonLocalMeans(const GpuMat&, GpuMat&, float, int, int, int, Stream&) { throw_nogpu(); }

#else


namespace cv { namespace gpu { namespace device
{
    namespace imgproc
    {
        template<typename T>
        void bilateral_filter_gpu(const PtrStepSzb& src, PtrStepSzb dst, int kernel_size, float sigma_spatial, float sigma_color, int borderMode, cudaStream_t stream);

        template<typename T>
        void nlm_bruteforce_gpu(const PtrStepSzb& src, PtrStepSzb dst, int search_radius, int block_radius, float h, int borderMode, cudaStream_t stream);
    }
}}}

void cv::gpu::bilateralFilter(const GpuMat& src, GpuMat& dst, int kernel_size, float sigma_color, float sigma_spatial, int borderMode, Stream& s)
{
    using cv::gpu::device::imgproc::bilateral_filter_gpu;

    typedef void (*func_t)(const PtrStepSzb& src, PtrStepSzb dst, int kernel_size, float sigma_spatial, float sigma_color, int borderMode, cudaStream_t s);

    static const func_t funcs[6][4] =
    {
        {bilateral_filter_gpu<uchar>      , 0 /*bilateral_filter_gpu<uchar2>*/ , bilateral_filter_gpu<uchar3>      , bilateral_filter_gpu<uchar4>      },
        {0 /*bilateral_filter_gpu<schar>*/, 0 /*bilateral_filter_gpu<schar2>*/ , 0 /*bilateral_filter_gpu<schar3>*/, 0 /*bilateral_filter_gpu<schar4>*/},
        {bilateral_filter_gpu<ushort>     , 0 /*bilateral_filter_gpu<ushort2>*/, bilateral_filter_gpu<ushort3>     , bilateral_filter_gpu<ushort4>     },
        {bilateral_filter_gpu<short>      , 0 /*bilateral_filter_gpu<short2>*/ , bilateral_filter_gpu<short3>      , bilateral_filter_gpu<short4>      },
        {0 /*bilateral_filter_gpu<int>*/  , 0 /*bilateral_filter_gpu<int2>*/   , 0 /*bilateral_filter_gpu<int3>*/  , 0 /*bilateral_filter_gpu<int4>*/  },
        {bilateral_filter_gpu<float>      , 0 /*bilateral_filter_gpu<float2>*/ , bilateral_filter_gpu<float3>      , bilateral_filter_gpu<float4>      }
    };

    sigma_color = (sigma_color <= 0 ) ? 1 : sigma_color;
    sigma_spatial = (sigma_spatial <= 0 ) ? 1 : sigma_spatial;


    int radius = (kernel_size <= 0) ? cvRound(sigma_spatial*1.5) : kernel_size/2;
    kernel_size = std::max(radius, 1)*2 + 1;

    CV_Assert(src.depth() <= CV_32F && src.channels() <= 4);
    const func_t func = funcs[src.depth()][src.channels() - 1];
    CV_Assert(func != 0);

    CV_Assert(borderMode == BORDER_REFLECT101 || borderMode == BORDER_REPLICATE || borderMode == BORDER_CONSTANT || borderMode == BORDER_REFLECT || borderMode == BORDER_WRAP);

    int gpuBorderType;
    CV_Assert(tryConvertToGpuBorderType(borderMode, gpuBorderType));

    dst.create(src.size(), src.type());
    func(src, dst, kernel_size, sigma_spatial, sigma_color, gpuBorderType, StreamAccessor::getStream(s));
}

void cv::gpu::nonLocalMeans(const GpuMat& src, GpuMat& dst, float h, int search_window_size, int block_size, int borderMode, Stream& s)
{
    using cv::gpu::device::imgproc::nlm_bruteforce_gpu;
    typedef void (*func_t)(const PtrStepSzb& src, PtrStepSzb dst, int search_radius, int block_radius, float h, int borderMode, cudaStream_t stream);

    static const func_t funcs[4] = { nlm_bruteforce_gpu<uchar>, 0 /*nlm_bruteforce_gpu<uchar2>*/ , nlm_bruteforce_gpu<uchar3>, 0/*nlm_bruteforce_gpu<uchar4>,*/ };

    CV_Assert(src.type() == CV_8U || src.type() == CV_8UC3);

    const func_t func = funcs[src.channels() - 1];
    CV_Assert(func != 0);

    int b = borderMode;
    CV_Assert(b == BORDER_REFLECT101 || b == BORDER_REPLICATE || b == BORDER_CONSTANT || b == BORDER_REFLECT || b == BORDER_WRAP);

    int gpuBorderType;
    CV_Assert(tryConvertToGpuBorderType(borderMode, gpuBorderType));

    int search_radius = search_window_size/2;
    int block_radius = block_size/2;

    dst.create(src.size(), src.type());
    func(src, dst, search_radius, block_radius, h, gpuBorderType, StreamAccessor::getStream(s));
}








#endif
