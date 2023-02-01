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
    void normDiffInf(const GpuMat& _src1, const GpuMat& _src2, GpuMat& _dst, Stream& stream)
    {
        const GpuMat_<uchar>& src1 = (const GpuMat_<uchar>&) _src1;
        const GpuMat_<uchar>& src2 = (const GpuMat_<uchar>&) _src2;
        GpuMat_<int>& dst = (GpuMat_<int>&) _dst;

        gridFindMaxVal(abs_(cvt_<int>(src1) - cvt_<int>(src2)), dst, stream);
    }

    void normDiffL1(const GpuMat& _src1, const GpuMat& _src2, GpuMat& _dst, Stream& stream)
    {
        const GpuMat_<uchar>& src1 = (const GpuMat_<uchar>&) _src1;
        const GpuMat_<uchar>& src2 = (const GpuMat_<uchar>&) _src2;
        GpuMat_<int>& dst = (GpuMat_<int>&) _dst;

        gridCalcSum(abs_(cvt_<int>(src1) - cvt_<int>(src2)), dst, stream);
    }

    void normDiffL2(const GpuMat& _src1, const GpuMat& _src2, GpuMat& _dst, Stream& stream)
    {
        const GpuMat_<uchar>& src1 = (const GpuMat_<uchar>&) _src1;
        const GpuMat_<uchar>& src2 = (const GpuMat_<uchar>&) _src2;
        GpuMat_<double>& dst = (GpuMat_<double>&) _dst;

        BufferPool pool(stream);
        GpuMat_<double> buf(1, 1, pool.getAllocator());

        gridCalcSum(sqr_(cvt_<double>(src1) - cvt_<double>(src2)), buf, stream);
        gridTransformUnary(buf, dst, sqrt_func<double>(), stream);
    }
}

void cv::cuda::calcNormDiff(InputArray _src1, InputArray _src2, OutputArray _dst, int normType, Stream& stream)
{
    typedef void (*func_t)(const GpuMat& _src1, const GpuMat& _src2, GpuMat& _dst, Stream& stream);
    static const func_t funcs[] =
    {
        0, normDiffInf, normDiffL1, 0, normDiffL2
    };

    GpuMat src1 = getInputMat(_src1, stream);
    GpuMat src2 = getInputMat(_src2, stream);

    CV_Assert( src1.type() == CV_8UC1 );
    CV_Assert( src1.size() == src2.size() && src1.type() == src2.type() );
    CV_Assert( normType == NORM_INF || normType == NORM_L1 || normType == NORM_L2 );

    GpuMat dst = getOutputMat(_dst, 1, 1, normType == NORM_L2 ? CV_64FC1 : CV_32SC1, stream);

    const func_t func = funcs[normType];
    func(src1, src2, dst, stream);

    syncOutput(dst, _dst, stream);
}

double cv::cuda::norm(InputArray _src1, InputArray _src2, int normType)
{
    Stream& stream = Stream::Null();

    HostMem dst;
    calcNormDiff(_src1, _src2, dst, normType, stream);

    stream.waitForCompletion();

    double val;
    dst.createMatHeader().convertTo(Mat(1, 1, CV_64FC1, &val), CV_64F);

    return val;
}

namespace cv { namespace cuda { namespace device {

void normL2(cv::InputArray _src, cv::OutputArray _dst, cv::InputArray _mask, Stream& stream);

}}}

namespace
{
    template <typename T, typename R>
    void normL2Impl(const GpuMat& _src, const GpuMat& mask, GpuMat& _dst, Stream& stream)
    {
        const GpuMat_<T>& src = (const GpuMat_<T>&) _src;
        GpuMat_<R>& dst = (GpuMat_<R>&) _dst;

        BufferPool pool(stream);
        GpuMat_<double> buf(1, 1, pool.getAllocator());

        if (mask.empty())
        {
            gridCalcSum(sqr_(cvt_<double>(src)), buf, stream);
        }
        else
        {
            gridCalcSum(sqr_(cvt_<double>(src)), buf, globPtr<uchar>(mask), stream);
        }

        gridTransformUnary(buf, dst, sqrt_func<double>(), stream);
    }
}

void cv::cuda::device::normL2(InputArray _src, OutputArray _dst, InputArray _mask, Stream& stream)
{
    typedef void (*func_t)(const GpuMat& _src, const GpuMat& mask, GpuMat& _dst, Stream& stream);
    static const func_t funcs[] =
    {
        normL2Impl<uchar, double>,
        normL2Impl<schar, double>,
        normL2Impl<ushort, double>,
        normL2Impl<short, double>,
        normL2Impl<int, double>,
        normL2Impl<float, double>,
        normL2Impl<double, double>
    };

    const GpuMat src = getInputMat(_src, stream);
    const GpuMat mask = getInputMat(_mask, stream);

    CV_Assert( src.channels() == 1 );
    CV_Assert( mask.empty() || (mask.size() == src.size() && mask.type() == CV_8U) );

    GpuMat dst = getOutputMat(_dst, 1, 1, CV_64FC1, stream);

    const func_t func = funcs[src.depth()];
    func(src, mask, dst, stream);

    syncOutput(dst, _dst, stream);
}

#endif
