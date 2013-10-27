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

using namespace cv::cudev;

namespace
{
    template <typename T, typename R, int cn>
    cv::Scalar sumImpl(const GpuMat& _src, const GpuMat& mask, GpuMat& _buf)
    {
        typedef typename MakeVec<T, cn>::type src_type;
        typedef typename MakeVec<R, cn>::type res_type;

        const GpuMat_<src_type>& src = (const GpuMat_<src_type>&) _src;
        GpuMat_<res_type>& buf = (GpuMat_<res_type>&) _buf;

        if (mask.empty())
            gridCalcSum(src, buf);
        else
            gridCalcSum(src, buf, globPtr<uchar>(mask));

        cv::Scalar_<R> res;
        cv::Mat res_mat(buf.size(), buf.type(), res.val);
        buf.download(res_mat);

        return res;
    }

    template <typename T, typename R, int cn>
    cv::Scalar sumAbsImpl(const GpuMat& _src, const GpuMat& mask, GpuMat& _buf)
    {
        typedef typename MakeVec<T, cn>::type src_type;
        typedef typename MakeVec<R, cn>::type res_type;

        const GpuMat_<src_type>& src = (const GpuMat_<src_type>&) _src;
        GpuMat_<res_type>& buf = (GpuMat_<res_type>&) _buf;

        if (mask.empty())
            gridCalcSum(abs_(cvt_<res_type>(src)), buf);
        else
            gridCalcSum(abs_(cvt_<res_type>(src)), buf, globPtr<uchar>(mask));

        cv::Scalar_<R> res;
        cv::Mat res_mat(buf.size(), buf.type(), res.val);
        buf.download(res_mat);

        return res;
    }

    template <typename T, typename R, int cn>
    cv::Scalar sumSqrImpl(const GpuMat& _src, const GpuMat& mask, GpuMat& _buf)
    {
        typedef typename MakeVec<T, cn>::type src_type;
        typedef typename MakeVec<R, cn>::type res_type;

        const GpuMat_<src_type>& src = (const GpuMat_<src_type>&) _src;
        GpuMat_<res_type>& buf = (GpuMat_<res_type>&) _buf;

        if (mask.empty())
            gridCalcSum(sqr_(cvt_<res_type>(src)), buf);
        else
            gridCalcSum(sqr_(cvt_<res_type>(src)), buf, globPtr<uchar>(mask));

        cv::Scalar_<R> res;
        cv::Mat res_mat(buf.size(), buf.type(), res.val);
        buf.download(res_mat);

        return res;
    }
}

cv::Scalar cv::cuda::sum(InputArray _src, InputArray _mask, GpuMat& buf)
{
    typedef cv::Scalar (*func_t)(const GpuMat& _src, const GpuMat& mask, GpuMat& _buf);
    static const func_t funcs[7][4] =
    {
        {sumImpl<uchar , uint  , 1>, sumImpl<uchar , uint  , 2>, sumImpl<uchar , uint  , 3>, sumImpl<uchar , uint  , 4>},
        {sumImpl<schar , int   , 1>, sumImpl<schar , int   , 2>, sumImpl<schar , int   , 3>, sumImpl<schar , int   , 4>},
        {sumImpl<ushort, uint  , 1>, sumImpl<ushort, uint  , 2>, sumImpl<ushort, uint  , 3>, sumImpl<ushort, uint  , 4>},
        {sumImpl<short , int   , 1>, sumImpl<short , int   , 2>, sumImpl<short , int   , 3>, sumImpl<short , int   , 4>},
        {sumImpl<int   , int   , 1>, sumImpl<int   , int   , 2>, sumImpl<int   , int   , 3>, sumImpl<int   , int   , 4>},
        {sumImpl<float , float , 1>, sumImpl<float , float , 2>, sumImpl<float , float , 3>, sumImpl<float , float , 4>},
        {sumImpl<double, double, 1>, sumImpl<double, double, 2>, sumImpl<double, double, 3>, sumImpl<double, double, 4>}
    };

    GpuMat src = _src.getGpuMat();
    GpuMat mask = _mask.getGpuMat();

    CV_DbgAssert( mask.empty() || (mask.type() == CV_8UC1 && mask.size() == src.size()) );

    const func_t func = funcs[src.depth()][src.channels() - 1];

    return func(src, mask, buf);
}

cv::Scalar cv::cuda::absSum(InputArray _src, InputArray _mask, GpuMat& buf)
{
    typedef cv::Scalar (*func_t)(const GpuMat& _src, const GpuMat& mask, GpuMat& _buf);
    static const func_t funcs[7][4] =
    {
        {sumAbsImpl<uchar , uint  , 1>, sumAbsImpl<uchar , uint  , 2>, sumAbsImpl<uchar , uint  , 3>, sumAbsImpl<uchar , uint  , 4>},
        {sumAbsImpl<schar , int   , 1>, sumAbsImpl<schar , int   , 2>, sumAbsImpl<schar , int   , 3>, sumAbsImpl<schar , int   , 4>},
        {sumAbsImpl<ushort, uint  , 1>, sumAbsImpl<ushort, uint  , 2>, sumAbsImpl<ushort, uint  , 3>, sumAbsImpl<ushort, uint  , 4>},
        {sumAbsImpl<short , int   , 1>, sumAbsImpl<short , int   , 2>, sumAbsImpl<short , int   , 3>, sumAbsImpl<short , int   , 4>},
        {sumAbsImpl<int   , int   , 1>, sumAbsImpl<int   , int   , 2>, sumAbsImpl<int   , int   , 3>, sumAbsImpl<int   , int   , 4>},
        {sumAbsImpl<float , float , 1>, sumAbsImpl<float , float , 2>, sumAbsImpl<float , float , 3>, sumAbsImpl<float , float , 4>},
        {sumAbsImpl<double, double, 1>, sumAbsImpl<double, double, 2>, sumAbsImpl<double, double, 3>, sumAbsImpl<double, double, 4>}
    };

    GpuMat src = _src.getGpuMat();
    GpuMat mask = _mask.getGpuMat();

    CV_DbgAssert( mask.empty() || (mask.type() == CV_8UC1 && mask.size() == src.size()) );

    const func_t func = funcs[src.depth()][src.channels() - 1];

    return func(src, mask, buf);
}

cv::Scalar cv::cuda::sqrSum(InputArray _src, InputArray _mask, GpuMat& buf)
{
    typedef cv::Scalar (*func_t)(const GpuMat& _src, const GpuMat& mask, GpuMat& _buf);
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

    GpuMat src = _src.getGpuMat();
    GpuMat mask = _mask.getGpuMat();

    CV_DbgAssert( mask.empty() || (mask.type() == CV_8UC1 && mask.size() == src.size()) );

    const func_t func = funcs[src.depth()][src.channels() - 1];

    return func(src, mask, buf);
}

#endif
