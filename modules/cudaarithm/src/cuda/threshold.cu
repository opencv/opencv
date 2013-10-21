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

using namespace cv::cudev;

namespace
{
    template <typename ScalarDepth> struct TransformPolicy : DefaultTransformPolicy
    {
    };
    template <> struct TransformPolicy<double> : DefaultTransformPolicy
    {
        enum {
            shift = 1
        };
    };

    template <typename T>
    void thresholdImpl(const GpuMat& src, GpuMat& dst, double thresh, double maxVal, int type, Stream& stream)
    {
        const T thresh_ = static_cast<T>(thresh);
        const T maxVal_ = static_cast<T>(maxVal);

        switch (type)
        {
        case 0:
            gridTransformUnary_< TransformPolicy<T> >(globPtr<T>(src), globPtr<T>(dst), thresh_binary_func(thresh_, maxVal_), stream);
            break;
        case 1:
            gridTransformUnary_< TransformPolicy<T> >(globPtr<T>(src), globPtr<T>(dst), thresh_binary_inv_func(thresh_, maxVal_), stream);
            break;
        case 2:
            gridTransformUnary_< TransformPolicy<T> >(globPtr<T>(src), globPtr<T>(dst), thresh_trunc_func(thresh_), stream);
            break;
        case 3:
            gridTransformUnary_< TransformPolicy<T> >(globPtr<T>(src), globPtr<T>(dst), thresh_to_zero_func(thresh_), stream);
            break;
        case 4:
            gridTransformUnary_< TransformPolicy<T> >(globPtr<T>(src), globPtr<T>(dst), thresh_to_zero_inv_func(thresh_), stream);
            break;
        };
    }
}

double cv::cuda::threshold(InputArray _src, OutputArray _dst, double thresh, double maxVal, int type, Stream& stream)
{
    GpuMat src = _src.getGpuMat();

    const int depth = src.depth();

    CV_DbgAssert( src.channels() == 1 && depth <= CV_64F );
    CV_DbgAssert( type <= 4 /*THRESH_TOZERO_INV*/ );

    _dst.create(src.size(), src.type());
    GpuMat dst = _dst.getGpuMat();

    if (depth == CV_32F && type == 2 /*THRESH_TRUNC*/)
    {
        NppStreamHandler h(StreamAccessor::getStream(stream));

        NppiSize sz;
        sz.width  = src.cols;
        sz.height = src.rows;

        nppSafeCall( nppiThreshold_32f_C1R(src.ptr<Npp32f>(), static_cast<int>(src.step),
            dst.ptr<Npp32f>(), static_cast<int>(dst.step), sz, static_cast<Npp32f>(thresh), NPP_CMP_GREATER) );

        if (!stream)
            CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
    }
    else
    {
        typedef void (*func_t)(const GpuMat& src, GpuMat& dst, double thresh, double maxVal, int type, Stream& stream);
        static const func_t funcs[] =
        {
            thresholdImpl<uchar>,
            thresholdImpl<schar>,
            thresholdImpl<ushort>,
            thresholdImpl<short>,
            thresholdImpl<int>,
            thresholdImpl<float>,
            thresholdImpl<double>
        };

        if (depth != CV_32F && depth != CV_64F)
        {
            thresh = cvFloor(thresh);
            maxVal = cvRound(maxVal);
        }

        funcs[depth](src, dst, thresh, maxVal, type, stream);
    }

    return thresh;
}

#endif
