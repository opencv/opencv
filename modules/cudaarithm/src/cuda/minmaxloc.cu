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
    void minMaxLocImpl(const GpuMat& _src, const GpuMat& mask, GpuMat& _valBuf, GpuMat& _locBuf, Stream& stream)
    {
        const GpuMat_<T>& src = (const GpuMat_<T>&) _src;
        GpuMat_<R>& valBuf = (GpuMat_<R>&) _valBuf;
        GpuMat_<int>& locBuf = (GpuMat_<int>&) _locBuf;

        if (mask.empty())
            gridMinMaxLoc(src, valBuf, locBuf, stream);
        else
            gridMinMaxLoc(src, valBuf, locBuf, globPtr<uchar>(mask), stream);
    }
}

void cv::cuda::findMinMaxLoc(InputArray _src, OutputArray _minMaxVals, OutputArray _loc, InputArray _mask, Stream& stream)
{
    typedef void (*func_t)(const GpuMat& _src, const GpuMat& mask, GpuMat& _valBuf, GpuMat& _locBuf, Stream& stream);
    static const func_t funcs[] =
    {
        minMaxLocImpl<uchar, int>,
        minMaxLocImpl<schar, int>,
        minMaxLocImpl<ushort, int>,
        minMaxLocImpl<short, int>,
        minMaxLocImpl<int, int>,
        minMaxLocImpl<float, float>,
        minMaxLocImpl<double, double>
    };

    const GpuMat src = getInputMat(_src, stream);
    const GpuMat mask = getInputMat(_mask, stream);

    CV_Assert( src.channels() == 1 );
    CV_Assert( mask.empty() || (mask.size() == src.size() && mask.type() == CV_8U) );

    const int src_depth = src.depth();

    BufferPool pool(stream);
    GpuMat valBuf(pool.getAllocator());
    GpuMat locBuf(pool.getAllocator());

    const func_t func = funcs[src_depth];
    func(src, mask, valBuf, locBuf, stream);

    GpuMat minMaxVals = valBuf.colRange(0, 1);
    GpuMat loc = locBuf.colRange(0, 1);

    if (_minMaxVals.kind() == _InputArray::CUDA_GPU_MAT)
    {
        minMaxVals.copyTo(_minMaxVals, stream);
    }
    else
    {
        minMaxVals.download(_minMaxVals, stream);
    }

    if (_loc.kind() == _InputArray::CUDA_GPU_MAT)
    {
        loc.copyTo(_loc, stream);
    }
    else
    {
        loc.download(_loc, stream);
    }
}

void cv::cuda::minMaxLoc(InputArray _src, double* minVal, double* maxVal, Point* minLoc, Point* maxLoc, InputArray _mask)
{
    Stream& stream = Stream::Null();

    HostMem minMaxVals, locVals;
    findMinMaxLoc(_src, minMaxVals, locVals, _mask, stream);

    stream.waitForCompletion();

    double vals[2];
    minMaxVals.createMatHeader().convertTo(Mat(minMaxVals.size(), CV_64FC1, &vals[0]), CV_64F);

    int locs[2];
    locVals.createMatHeader().copyTo(Mat(locVals.size(), CV_32SC1, &locs[0]));
    Size size = _src.size();
    cv::Point locs2D[] = {
        cv::Point(locs[0] % size.width, locs[0] / size.width),
        cv::Point(locs[1] % size.width, locs[1] / size.width),
    };

    if (minVal)
        *minVal = vals[0];

    if (maxVal)
        *maxVal = vals[1];

    if (minLoc)
        *minLoc = locs2D[0];

    if (maxLoc)
        *maxLoc = locs2D[1];
}

#endif
