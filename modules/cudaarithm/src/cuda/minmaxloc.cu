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
    template <typename T>
    void minMaxLocImpl(const GpuMat& _src, const GpuMat& mask, GpuMat& _valBuf, GpuMat& _locBuf, double* minVal, double* maxVal, cv::Point* minLoc, cv::Point* maxLoc)
    {
        typedef typename SelectIf<
                TypesEquals<T, double>::value,
                double,
                typename SelectIf<TypesEquals<T, float>::value, float, int>::type
                >::type work_type;

        const GpuMat_<T>& src = (const GpuMat_<T>&) _src;
        GpuMat_<work_type>& valBuf = (GpuMat_<work_type>&) _valBuf;
        GpuMat_<int>& locBuf = (GpuMat_<int>&) _locBuf;

        if (mask.empty())
            gridMinMaxLoc(src, valBuf, locBuf);
        else
            gridMinMaxLoc(src, valBuf, locBuf, globPtr<uchar>(mask));

        cv::Mat_<work_type> h_valBuf;
        cv::Mat_<int> h_locBuf;

        valBuf.download(h_valBuf);
        locBuf.download(h_locBuf);

        if (minVal)
            *minVal = h_valBuf(0, 0);

        if (maxVal)
            *maxVal = h_valBuf(1, 0);

        if (minLoc)
        {
            const int idx = h_locBuf(0, 0);
            *minLoc = cv::Point(idx % src.cols, idx / src.cols);
        }

        if (maxLoc)
        {
            const int idx = h_locBuf(1, 0);
            *maxLoc = cv::Point(idx % src.cols, idx / src.cols);
        }
    }
}

void cv::cuda::minMaxLoc(InputArray _src, double* minVal, double* maxVal, Point* minLoc, Point* maxLoc, InputArray _mask, GpuMat& valBuf, GpuMat& locBuf)
{
    typedef void (*func_t)(const GpuMat& _src, const GpuMat& mask, GpuMat& _valBuf, GpuMat& _locBuf, double* minVal, double* maxVal, cv::Point* minLoc, cv::Point* maxLoc);
    static const func_t funcs[] =
    {
        minMaxLocImpl<uchar>,
        minMaxLocImpl<schar>,
        minMaxLocImpl<ushort>,
        minMaxLocImpl<short>,
        minMaxLocImpl<int>,
        minMaxLocImpl<float>,
        minMaxLocImpl<double>
    };

    GpuMat src = _src.getGpuMat();
    GpuMat mask = _mask.getGpuMat();

    CV_Assert( src.channels() == 1 );
    CV_DbgAssert( mask.empty() || (mask.size() == src.size() && mask.type() == CV_8U) );

    const func_t func = funcs[src.depth()];

    func(src, mask, valBuf, locBuf, minVal, maxVal, minLoc, maxLoc);
}

#endif
