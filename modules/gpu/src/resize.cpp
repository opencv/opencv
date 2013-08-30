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

void cv::gpu::resize(const GpuMat&, GpuMat&, Size, double, double, int, Stream&) { throw_nogpu(); }

#else // HAVE_CUDA

namespace cv { namespace gpu { namespace device
{
    template <typename T>
    void resize(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, cudaStream_t stream);
}}}

void cv::gpu::resize(const GpuMat& src, GpuMat& dst, Size dsize, double fx, double fy, int interpolation, Stream& stream)
{
    typedef void (*func_t)(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, cudaStream_t stream);
    static const func_t funcs[6][4] =
    {
        {device::resize<uchar>      , 0 /*device::resize<uchar2>*/ , device::resize<uchar3>     , device::resize<uchar4>     },
        {0 /*device::resize<schar>*/, 0 /*device::resize<char2>*/  , 0 /*device::resize<char3>*/, 0 /*device::resize<char4>*/},
        {device::resize<ushort>     , 0 /*device::resize<ushort2>*/, device::resize<ushort3>    , device::resize<ushort4>    },
        {device::resize<short>      , 0 /*device::resize<short2>*/ , device::resize<short3>     , device::resize<short4>     },
        {0 /*device::resize<int>*/  , 0 /*device::resize<int2>*/   , 0 /*device::resize<int3>*/ , 0 /*device::resize<int4>*/ },
        {device::resize<float>      , 0 /*device::resize<float2>*/ , device::resize<float3>     , device::resize<float4>     }
    };

    CV_Assert( src.depth() <= CV_32F && src.channels() <= 4 );
    CV_Assert( interpolation == INTER_NEAREST || interpolation == INTER_LINEAR || interpolation == INTER_CUBIC || interpolation == INTER_AREA );
    CV_Assert( !(dsize == Size()) || (fx > 0 && fy > 0) );

    if (dsize == Size())
    {
        dsize = Size(saturate_cast<int>(src.cols * fx), saturate_cast<int>(src.rows * fy));
    }
    else
    {
        fx = static_cast<double>(dsize.width) / src.cols;
        fy = static_cast<double>(dsize.height) / src.rows;
    }

    dst.create(dsize, src.type());

    if (dsize == src.size())
    {
        if (stream)
            stream.enqueueCopy(src, dst);
        else
            src.copyTo(dst);
        return;
    }

    const func_t func = funcs[src.depth()][src.channels() - 1];

    if (!func)
        CV_Error(CV_StsUnsupportedFormat, "Unsupported combination of source and destination types");

    Size wholeSize;
    Point ofs;
    src.locateROI(wholeSize, ofs);
    PtrStepSzb wholeSrc(wholeSize.height, wholeSize.width, src.datastart, src.step);

    func(src, wholeSrc, ofs.y, ofs.x, dst, static_cast<float>(1.0 / fy), static_cast<float>(1.0 / fx), interpolation, StreamAccessor::getStream(stream));
}

#endif // HAVE_CUDA
