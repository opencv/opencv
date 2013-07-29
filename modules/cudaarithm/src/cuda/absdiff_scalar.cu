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

#include "opencv2/cudev.hpp"

using namespace cv::cudev;

void absDiffScalar(const GpuMat& src, cv::Scalar val, bool, GpuMat& dst, const GpuMat&, double, Stream& stream, int);

namespace
{
    template <typename T, typename S> struct AbsDiffScalarOp : unary_function<T, T>
    {
        S val;

        __device__ __forceinline__ T operator ()(T a) const
        {
            abs_func<S> f;
            return saturate_cast<T>(f(a - val));
        }
    };

    template <typename ScalarDepth> struct TransformPolicy : DefaultTransformPolicy
    {
    };
    template <> struct TransformPolicy<double> : DefaultTransformPolicy
    {
        enum {
            shift = 1
        };
    };

    template <typename SrcType, typename ScalarDepth>
    void absDiffScalarImpl(const GpuMat& src, double value, GpuMat& dst, Stream& stream)
    {
        AbsDiffScalarOp<SrcType, ScalarDepth> op;
        op.val = static_cast<ScalarDepth>(value);
        gridTransformUnary_< TransformPolicy<ScalarDepth> >(globPtr<SrcType>(src), globPtr<SrcType>(dst), op, stream);
    }
}

void absDiffScalar(const GpuMat& src, cv::Scalar val, bool, GpuMat& dst, const GpuMat&, double, Stream& stream, int)
{
    typedef void (*func_t)(const GpuMat& src, double val, GpuMat& dst, Stream& stream);
    static const func_t funcs[] =
    {
        absDiffScalarImpl<uchar, float>,
        absDiffScalarImpl<schar, float>,
        absDiffScalarImpl<ushort, float>,
        absDiffScalarImpl<short, float>,
        absDiffScalarImpl<int, float>,
        absDiffScalarImpl<float, float>,
        absDiffScalarImpl<double, double>
    };

    const int depth = src.depth();

    CV_DbgAssert( depth <= CV_64F );

    funcs[depth](src, val[0], dst, stream);
}

#endif
