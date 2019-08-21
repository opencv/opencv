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
    template <typename T, typename R, int cn>
    void sumImpl(const GpuMat& _src, GpuMat& _dst, const GpuMat& mask, Stream& stream)
    {
        typedef typename MakeVec<T, cn>::type src_type;
        typedef typename MakeVec<R, cn>::type res_type;

        const GpuMat_<src_type>& src = (const GpuMat_<src_type>&) _src;
        GpuMat_<res_type>& dst = (GpuMat_<res_type>&) _dst;

        if (mask.empty())
            gridCalcSum(src, dst, stream);
        else
            gridCalcSum(src, dst, globPtr<uchar>(mask), stream);
    }

    template <typename T, typename R, int cn>
    void sumAbsImpl(const GpuMat& _src, GpuMat& _dst, const GpuMat& mask, Stream& stream)
    {
        typedef typename MakeVec<T, cn>::type src_type;
        typedef typename MakeVec<R, cn>::type res_type;

        const GpuMat_<src_type>& src = (const GpuMat_<src_type>&) _src;
        GpuMat_<res_type>& dst = (GpuMat_<res_type>&) _dst;

        if (mask.empty())
            gridCalcSum(abs_(cvt_<res_type>(src)), dst, stream);
        else
            gridCalcSum(abs_(cvt_<res_type>(src)), dst, globPtr<uchar>(mask), stream);
    }

    template <typename T, typename R, int cn>
    void sumSqrImpl(const GpuMat& _src, GpuMat& _dst, const GpuMat& mask, Stream& stream)
    {
        typedef typename MakeVec<T, cn>::type src_type;
        typedef typename MakeVec<R, cn>::type res_type;

        const GpuMat_<src_type>& src = (const GpuMat_<src_type>&) _src;
        GpuMat_<res_type>& dst = (GpuMat_<res_type>&) _dst;

        if (mask.empty())
            gridCalcSum(sqr_(cvt_<res_type>(src)), dst, stream);
        else
            gridCalcSum(sqr_(cvt_<res_type>(src)), dst, globPtr<uchar>(mask), stream);
    }
}

void cv::cuda::calcSum(InputArray _src, OutputArray _dst, InputArray _mask, Stream& stream)
{
    typedef void (*func_t)(const GpuMat& _src, GpuMat& _dst, const GpuMat& mask, Stream& stream);
    static const func_t funcs[7][4] =
    {
        {sumImpl<uchar , double, 1>, sumImpl<uchar , double, 2>, sumImpl<uchar , double, 3>, sumImpl<uchar , double, 4>},
        {sumImpl<schar , double, 1>, sumImpl<schar , double, 2>, sumImpl<schar , double, 3>, sumImpl<schar , double, 4>},
        {sumImpl<ushort, double, 1>, sumImpl<ushort, double, 2>, sumImpl<ushort, double, 3>, sumImpl<ushort, double, 4>},
        {sumImpl<short , double, 1>, sumImpl<short , double, 2>, sumImpl<short , double, 3>, sumImpl<short , double, 4>},
        {sumImpl<int   , double, 1>, sumImpl<int   , double, 2>, sumImpl<int   , double, 3>, sumImpl<int   , double, 4>},
        {sumImpl<float , double, 1>, sumImpl<float , double, 2>, sumImpl<float , double, 3>, sumImpl<float , double, 4>},
        {sumImpl<double, double, 1>, sumImpl<double, double, 2>, sumImpl<double, double, 3>, sumImpl<double, double, 4>}
    };

    const GpuMat src = getInputMat(_src, stream);
    const GpuMat mask = getInputMat(_mask, stream);

    CV_Assert( mask.empty() || (mask.type() == CV_8UC1 && mask.size() == src.size()) );

    const int src_depth = src.depth();
    const int channels = src.channels();

    GpuMat dst = getOutputMat(_dst, 1, 1, CV_64FC(channels), stream);

    const func_t func = funcs[src_depth][channels - 1];
    func(src, dst, mask, stream);

    syncOutput(dst, _dst, stream);
}

cv::Scalar cv::cuda::sum(InputArray _src, InputArray _mask)
{
    Stream& stream = Stream::Null();

    HostMem dst;
    calcSum(_src, dst, _mask, stream);

    stream.waitForCompletion();

    cv::Scalar val;
    dst.createMatHeader().convertTo(cv::Mat(dst.size(), CV_64FC(dst.channels()), val.val), CV_64F);

    return val;
}

void cv::cuda::calcAbsSum(InputArray _src, OutputArray _dst, InputArray _mask, Stream& stream)
{
    typedef void (*func_t)(const GpuMat& _src, GpuMat& _dst, const GpuMat& mask, Stream& stream);
    static const func_t funcs[7][4] =
    {
        {sumAbsImpl<uchar , double, 1>, sumAbsImpl<uchar , double, 2>, sumAbsImpl<uchar , double, 3>, sumAbsImpl<uchar , double, 4>},
        {sumAbsImpl<schar , double, 1>, sumAbsImpl<schar , double, 2>, sumAbsImpl<schar , double, 3>, sumAbsImpl<schar , double, 4>},
        {sumAbsImpl<ushort, double, 1>, sumAbsImpl<ushort, double, 2>, sumAbsImpl<ushort, double, 3>, sumAbsImpl<ushort, double, 4>},
        {sumAbsImpl<short , double, 1>, sumAbsImpl<short , double, 2>, sumAbsImpl<short , double, 3>, sumAbsImpl<short , double, 4>},
        {sumAbsImpl<int   , double, 1>, sumAbsImpl<int   , double, 2>, sumAbsImpl<int   , double, 3>, sumAbsImpl<int   , double, 4>},
        {sumAbsImpl<float , double, 1>, sumAbsImpl<float , double, 2>, sumAbsImpl<float , double, 3>, sumAbsImpl<float , double, 4>},
        {sumAbsImpl<double, double, 1>, sumAbsImpl<double, double, 2>, sumAbsImpl<double, double, 3>, sumAbsImpl<double, double, 4>}
    };

    const GpuMat src = getInputMat(_src, stream);
    const GpuMat mask = getInputMat(_mask, stream);

    CV_Assert( mask.empty() || (mask.type() == CV_8UC1 && mask.size() == src.size()) );

    const int src_depth = src.depth();
    const int channels = src.channels();

    GpuMat dst = getOutputMat(_dst, 1, 1, CV_64FC(channels), stream);

    const func_t func = funcs[src_depth][channels - 1];
    func(src, dst, mask, stream);

    syncOutput(dst, _dst, stream);
}

cv::Scalar cv::cuda::absSum(InputArray _src, InputArray _mask)
{
    Stream& stream = Stream::Null();

    HostMem dst;
    calcAbsSum(_src, dst, _mask, stream);

    stream.waitForCompletion();

    cv::Scalar val;
    dst.createMatHeader().convertTo(cv::Mat(dst.size(), CV_64FC(dst.channels()), val.val), CV_64F);

    return val;
}

void cv::cuda::calcSqrSum(InputArray _src, OutputArray _dst, InputArray _mask, Stream& stream)
{
    typedef void (*func_t)(const GpuMat& _src, GpuMat& _dst, const GpuMat& mask, Stream& stream);
    static const func_t funcs[7][4] =
    {
        {sumSqrImpl<uchar , double, 1>, sumSqrImpl<uchar , double, 2>, sumSqrImpl<uchar , double, 3>, sumSqrImpl<uchar , double, 4>},
        {sumSqrImpl<schar , double, 1>, sumSqrImpl<schar , double, 2>, sumSqrImpl<schar , double, 3>, sumSqrImpl<schar , double, 4>},
        {sumSqrImpl<ushort, double, 1>, sumSqrImpl<ushort, double, 2>, sumSqrImpl<ushort, double, 3>, sumSqrImpl<ushort, double, 4>},
        {sumSqrImpl<short , double, 1>, sumSqrImpl<short , double, 2>, sumSqrImpl<short , double, 3>, sumSqrImpl<short , double, 4>},
        {sumSqrImpl<int   , double, 1>, sumSqrImpl<int   , double, 2>, sumSqrImpl<int   , double, 3>, sumSqrImpl<int   , double, 4>},
        {sumSqrImpl<float , double, 1>, sumSqrImpl<float , double, 2>, sumSqrImpl<float , double, 3>, sumSqrImpl<float , double, 4>},
        {sumSqrImpl<double, double, 1>, sumSqrImpl<double, double, 2>, sumSqrImpl<double, double, 3>, sumSqrImpl<double, double, 4>}
    };

    const GpuMat src = getInputMat(_src, stream);
    const GpuMat mask = getInputMat(_mask, stream);

    CV_Assert( mask.empty() || (mask.type() == CV_8UC1 && mask.size() == src.size()) );

    const int src_depth = src.depth();
    const int channels = src.channels();

    GpuMat dst = getOutputMat(_dst, 1, 1, CV_64FC(channels), stream);

    const func_t func = funcs[src_depth][channels - 1];
    func(src, dst, mask, stream);

    syncOutput(dst, _dst, stream);
}

cv::Scalar cv::cuda::sqrSum(InputArray _src, InputArray _mask)
{
    Stream& stream = Stream::Null();

    HostMem dst;
    calcSqrSum(_src, dst, _mask, stream);

    stream.waitForCompletion();

    cv::Scalar val;
    dst.createMatHeader().convertTo(cv::Mat(dst.size(), CV_64FC(dst.channels()), val.val), CV_64F);

    return val;
}

#endif
