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
    double normDiffInf(const GpuMat& _src1, const GpuMat& _src2, GpuMat& _buf)
    {
        const GpuMat_<uchar>& src1 = (const GpuMat_<uchar>&) _src1;
        const GpuMat_<uchar>& src2 = (const GpuMat_<uchar>&) _src2;
        GpuMat_<int>& buf = (GpuMat_<int>&) _buf;

        gridFindMinMaxVal(abs_(cvt_<int>(src1) - cvt_<int>(src2)), buf);

        int data[2];
        buf.download(cv::Mat(1, 2, buf.type(), data));

        return data[1];
    }

    double normDiffL1(const GpuMat& _src1, const GpuMat& _src2, GpuMat& _buf)
    {
        const GpuMat_<uchar>& src1 = (const GpuMat_<uchar>&) _src1;
        const GpuMat_<uchar>& src2 = (const GpuMat_<uchar>&) _src2;
        GpuMat_<int>& buf = (GpuMat_<int>&) _buf;

        gridCalcSum(abs_(cvt_<int>(src1) - cvt_<int>(src2)), buf);

        int data;
        buf.download(cv::Mat(1, 1, buf.type(), &data));

        return data;
    }

    double normDiffL2(const GpuMat& _src1, const GpuMat& _src2, GpuMat& _buf)
    {
        const GpuMat_<uchar>& src1 = (const GpuMat_<uchar>&) _src1;
        const GpuMat_<uchar>& src2 = (const GpuMat_<uchar>&) _src2;
        GpuMat_<double>& buf = (GpuMat_<double>&) _buf;

        gridCalcSum(sqr_(cvt_<double>(src1) - cvt_<double>(src2)), buf);

        double data;
        buf.download(cv::Mat(1, 1, buf.type(), &data));

        return std::sqrt(data);
    }
}

double cv::cuda::norm(InputArray _src1, InputArray _src2, GpuMat& buf, int normType)
{
    typedef double (*func_t)(const GpuMat& _src1, const GpuMat& _src2, GpuMat& _buf);
    static const func_t funcs[] =
    {
        0, normDiffInf, normDiffL1, 0, normDiffL2
    };

    GpuMat src1 = _src1.getGpuMat();
    GpuMat src2 = _src2.getGpuMat();

    CV_Assert( src1.type() == CV_8UC1 );
    CV_Assert( src1.size() == src2.size() && src1.type() == src2.type() );
    CV_Assert( normType == NORM_INF || normType == NORM_L1 || normType == NORM_L2 );

    return funcs[normType](src1, src2, buf);
}

#endif
