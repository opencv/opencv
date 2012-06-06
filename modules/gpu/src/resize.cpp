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

#ifndef HAVE_CUDA

void cv::gpu::resize(const GpuMat&, GpuMat&, Size, double, double, int, Stream&) { throw_nogpu(); }

#else // HAVE_CUDA

namespace cv { namespace gpu { namespace device
{
    namespace imgproc
    {
        template <typename T>
        void resize_gpu(DevMem2Db src, DevMem2Db srcWhole, int xoff, int yoff, float fx, float fy,
                        DevMem2Db dst, int interpolation, cudaStream_t stream);
    }
}}}

void cv::gpu::resize(const GpuMat& src, GpuMat& dst, Size dsize, double fx, double fy, int interpolation, Stream& s)
{
    CV_Assert(src.depth() <= CV_32F && src.channels() <= 4);
    CV_Assert(interpolation == INTER_NEAREST || interpolation == INTER_LINEAR
            || interpolation == INTER_CUBIC || interpolation == INTER_AREA);
    CV_Assert(!(dsize == Size()) || (fx > 0 && fy > 0));

    if (dsize == Size())
        dsize = Size(saturate_cast<int>(src.cols * fx), saturate_cast<int>(src.rows * fy));
    else
    {
        fx = static_cast<double>(dsize.width) / src.cols;
        fy = static_cast<double>(dsize.height) / src.rows;
    }

    dst.create(dsize, src.type());

    if (dsize == src.size())
    {
        if (s)
            s.enqueueCopy(src, dst);
        else
            src.copyTo(dst);
        return;
    }

    cudaStream_t stream = StreamAccessor::getStream(s);

    Size wholeSize;
    Point ofs;
    src.locateROI(wholeSize, ofs);

    bool useNpp = (src.type() == CV_8UC1 || src.type() == CV_8UC4);
    useNpp = useNpp && (interpolation == INTER_NEAREST || interpolation == INTER_LINEAR || src.type() == CV_8UC4);

    if (useNpp)
    {
        typedef NppStatus (*func_t)(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u * pDst, int nDstStep, NppiSize dstROISize,
                                    double xFactor, double yFactor, int eInterpolation);

        const func_t funcs[4] = { nppiResize_8u_C1R, 0, 0, nppiResize_8u_C4R };

        static const int npp_inter[] = {NPPI_INTER_NN, NPPI_INTER_LINEAR, NPPI_INTER_CUBIC, 0, NPPI_INTER_LANCZOS};

        NppiSize srcsz;
        srcsz.width  = wholeSize.width;
        srcsz.height = wholeSize.height;

        NppiRect srcrect;
        srcrect.x = ofs.x;
        srcrect.y = ofs.y;
        srcrect.width  = src.cols;
        srcrect.height = src.rows;

        NppiSize dstsz;
        dstsz.width  = dst.cols;
        dstsz.height = dst.rows;

        NppStreamHandler h(stream);

        nppSafeCall( funcs[src.channels() - 1](src.datastart, srcsz, static_cast<int>(src.step), srcrect,
                dst.ptr<Npp8u>(), static_cast<int>(dst.step), dstsz, fx, fy, npp_inter[interpolation]) );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
    else
    {
        using namespace ::cv::gpu::device::imgproc;

        typedef void (*func_t)(DevMem2Db src, DevMem2Db srcWhole, int xoff, int yoff, float fx, float fy, DevMem2Db dst, int interpolation, cudaStream_t stream);

        static const func_t funcs[6][4] =
        {
            {resize_gpu<uchar>      , 0 /*resize_gpu<uchar2>*/ , resize_gpu<uchar3>     , resize_gpu<uchar4>     },
            {0 /*resize_gpu<schar>*/, 0 /*resize_gpu<char2>*/  , 0 /*resize_gpu<char3>*/, 0 /*resize_gpu<char4>*/},
            {resize_gpu<ushort>     , 0 /*resize_gpu<ushort2>*/, resize_gpu<ushort3>    , resize_gpu<ushort4>    },
            {resize_gpu<short>      , 0 /*resize_gpu<short2>*/ , resize_gpu<short3>     , resize_gpu<short4>     },
            {0 /*resize_gpu<int>*/  , 0 /*resize_gpu<int2>*/   , 0 /*resize_gpu<int3>*/ , 0 /*resize_gpu<int4>*/ },
            {resize_gpu<float>      , 0 /*resize_gpu<float2>*/ , resize_gpu<float3>     , resize_gpu<float4>     }
        };

        const func_t func = funcs[src.depth()][src.channels() - 1];
        CV_Assert(func != 0);

        func(src, DevMem2Db(wholeSize.height, wholeSize.width, src.datastart, src.step), ofs.x, ofs.y,
            static_cast<float>(1.0 / fx), static_cast<float>(1.0 / fy), dst, interpolation, stream);
    }
}

#endif // HAVE_CUDA
