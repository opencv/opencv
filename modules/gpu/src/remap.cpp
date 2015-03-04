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

#if !defined HAVE_CUDA || defined(CUDA_DISABLER)

void cv::gpu::remap(const GpuMat&, GpuMat&, const GpuMat&, const GpuMat&, int, int, Scalar, Stream&){ throw_nogpu(); }

#else // HAVE_CUDA

namespace cv { namespace gpu { namespace device
{
    namespace imgproc
    {
        template <typename T>
        void remap_gpu(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst,
                       int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);
    }
}}}

void cv::gpu::remap(const GpuMat& src, GpuMat& dst, const GpuMat& xmap, const GpuMat& ymap, int interpolation, int borderMode, Scalar borderValue, Stream& stream)
{
    using namespace cv::gpu::device::imgproc;

    typedef void (*func_t)(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, int interpolation,
        int borderMode, const float* borderValue, cudaStream_t stream, bool cc20);

#ifdef OPENCV_TINY_GPU_MODULE
    static const func_t funcs[6][4] =
    {
        {remap_gpu<uchar>      , 0 /*remap_gpu<uchar2>*/ , remap_gpu<uchar3>     , remap_gpu<uchar4>     },
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {remap_gpu<float>      , 0 /*remap_gpu<float2>*/ , remap_gpu<float3>     , remap_gpu<float4>     }
    };
#else
    static const func_t funcs[6][4] =
    {
        {remap_gpu<uchar>      , 0 /*remap_gpu<uchar2>*/ , remap_gpu<uchar3>     , remap_gpu<uchar4>     },
        {0 /*remap_gpu<schar>*/, 0 /*remap_gpu<char2>*/  , 0 /*remap_gpu<char3>*/, 0 /*remap_gpu<char4>*/},
        {remap_gpu<ushort>     , 0 /*remap_gpu<ushort2>*/, remap_gpu<ushort3>    , remap_gpu<ushort4>    },
        {remap_gpu<short>      , 0 /*remap_gpu<short2>*/ , remap_gpu<short3>     , remap_gpu<short4>     },
        {0 /*remap_gpu<int>*/  , 0 /*remap_gpu<int2>*/   , 0 /*remap_gpu<int3>*/ , 0 /*remap_gpu<int4>*/ },
        {remap_gpu<float>      , 0 /*remap_gpu<float2>*/ , remap_gpu<float3>     , remap_gpu<float4>     }
    };
#endif

    CV_Assert(src.depth() <= CV_32F && src.channels() <= 4);
    CV_Assert(xmap.type() == CV_32F && ymap.type() == CV_32F && xmap.size() == ymap.size());
    CV_Assert(interpolation == INTER_NEAREST || interpolation == INTER_LINEAR || interpolation == INTER_CUBIC);
    CV_Assert(borderMode == BORDER_REFLECT101 || borderMode == BORDER_REPLICATE || borderMode == BORDER_CONSTANT || borderMode == BORDER_REFLECT || borderMode == BORDER_WRAP);

    const func_t func = funcs[src.depth()][src.channels() - 1];
    CV_Assert(func != 0);

    int gpuBorderType;
    CV_Assert(tryConvertToGpuBorderType(borderMode, gpuBorderType));

    dst.create(xmap.size(), src.type());

    Scalar_<float> borderValueFloat;
    borderValueFloat = borderValue;

    Size wholeSize;
    Point ofs;
    src.locateROI(wholeSize, ofs);

    func(src, PtrStepSzb(wholeSize.height, wholeSize.width, src.datastart, src.step), ofs.x, ofs.y, xmap, ymap,
        dst, interpolation, gpuBorderType, borderValueFloat.val, StreamAccessor::getStream(stream), deviceSupports(FEATURE_SET_COMPUTE_20));
}

#endif // HAVE_CUDA
