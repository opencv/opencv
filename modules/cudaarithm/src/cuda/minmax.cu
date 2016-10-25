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

#include "opencv2/opencv_modules.hpp"

#ifndef HAVE_OPENCV_CUDEV

#error "opencv_cudev is required"

#else

#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudev.hpp"
#include "opencv2/core/private.cuda.hpp"

using namespace cv;
using namespace cv::cuda;
using namespace cv::cudev;

namespace
{
    template <typename T, typename R>
    void minMaxImpl(const GpuMat& _src, const GpuMat& mask, GpuMat& _dst, Stream& stream)
    {
        const GpuMat_<T>& src = (const GpuMat_<T>&) _src;
        GpuMat_<R>& dst = (GpuMat_<R>&) _dst;

        if (mask.empty())
            gridFindMinMaxVal(src, dst, stream);
        else
            gridFindMinMaxVal(src, dst, globPtr<uchar>(mask), stream);
    }

    template <typename T, typename R>
    void minMaxImpl(const GpuMat& src, const GpuMat& mask, double* minVal, double* maxVal)
    {
        BufferPool pool(Stream::Null());
        GpuMat buf(pool.getBuffer(1, 2, DataType<R>::type));

        minMaxImpl<T, R>(src, mask, buf, Stream::Null());

        R data[2];
        buf.download(Mat(1, 2, buf.type(), data));

    }
}

void cv::cuda::findMinMax(InputArray _src, OutputArray _dst, InputArray _mask, Stream& stream)
{
    typedef void (*func_t)(const GpuMat& _src, const GpuMat& mask, GpuMat& _dst, Stream& stream);
    static const func_t funcs[] =
    {
        minMaxImpl<uchar, int>,
        minMaxImpl<schar, int>,
        minMaxImpl<ushort, int>,
        minMaxImpl<short, int>,
        minMaxImpl<int, int>,
        minMaxImpl<float, float>,
        minMaxImpl<double, double>
    };

    const GpuMat src = getInputMat(_src, stream);
    const GpuMat mask = getInputMat(_mask, stream);

    CV_Assert( src.channels() == 1 );
    CV_Assert( mask.empty() || (mask.size() == src.size() && mask.type() == CV_8U) );

    const int src_depth = src.depth();
    const int dst_depth = src_depth < CV_32F ? CV_32S : src_depth;

    GpuMat dst = getOutputMat(_dst, 1, 2, dst_depth, stream);

    const func_t func = funcs[src.depth()];
    func(src, mask, dst, stream);

    syncOutput(dst, _dst, stream);
}

void cv::cuda::minMax(InputArray _src, double* minVal, double* maxVal, InputArray _mask)
{
    Stream& stream = Stream::Null();

    HostMem dst;
    findMinMax(_src, dst, _mask, stream);

    stream.waitForCompletion();

    double vals[2];
    dst.createMatHeader().convertTo(Mat(1, 2, CV_64FC1, &vals[0]), CV_64F);

    if (minVal)
        *minVal = vals[0];

    if (maxVal)
        *maxVal = vals[1];
}

namespace cv { namespace cuda { namespace device {

void findMaxAbs(InputArray _src, OutputArray _dst, InputArray _mask, Stream& stream);

}}}

namespace
{
    template <typename T, typename R>
    void findMaxAbsImpl(const GpuMat& _src, const GpuMat& mask, GpuMat& _dst, Stream& stream)
    {
        const GpuMat_<T>& src = (const GpuMat_<T>&) _src;
        GpuMat_<R>& dst = (GpuMat_<R>&) _dst;

        if (mask.empty())
            gridFindMaxVal(abs_(src), dst, stream);
        else
            gridFindMaxVal(abs_(src), dst, globPtr<uchar>(mask), stream);
    }
}

void cv::cuda::device::findMaxAbs(InputArray _src, OutputArray _dst, InputArray _mask, Stream& stream)
{
    typedef void (*func_t)(const GpuMat& _src, const GpuMat& mask, GpuMat& _dst, Stream& stream);
    static const func_t funcs[] =
    {
        findMaxAbsImpl<uchar, int>,
        findMaxAbsImpl<schar, int>,
        findMaxAbsImpl<ushort, int>,
        findMaxAbsImpl<short, int>,
        findMaxAbsImpl<int, int>,
        findMaxAbsImpl<float, float>,
        findMaxAbsImpl<double, double>
    };

    const GpuMat src = getInputMat(_src, stream);
    const GpuMat mask = getInputMat(_mask, stream);

    CV_Assert( src.channels() == 1 );
    CV_Assert( mask.empty() || (mask.size() == src.size() && mask.type() == CV_8U) );

    const int src_depth = src.depth();
    const int dst_depth = src_depth < CV_32F ? CV_32S : src_depth;

    GpuMat dst = getOutputMat(_dst, 1, 1, dst_depth, stream);

    const func_t func = funcs[src.depth()];
    func(src, mask, dst, stream);

    syncOutput(dst, _dst, stream);
}

#endif
