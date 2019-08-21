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
    template <typename SrcType, typename ScalarType, typename DstType> struct AbsDiffScalarOp : unary_function<SrcType, DstType>
    {
        ScalarType val;

        __device__ __forceinline__ DstType operator ()(SrcType a) const
        {
            abs_func<ScalarType> f;
            return saturate_cast<DstType>(f(saturate_cast<ScalarType>(a) - val));
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
    void absDiffScalarImpl(const GpuMat& src, cv::Scalar value, GpuMat& dst, Stream& stream)
    {
        typedef typename MakeVec<ScalarDepth, VecTraits<SrcType>::cn>::type ScalarType;

        cv::Scalar_<ScalarDepth> value_ = value;

        AbsDiffScalarOp<SrcType, ScalarType, SrcType> op;
        op.val = VecTraits<ScalarType>::make(value_.val);
        gridTransformUnary_< TransformPolicy<ScalarDepth> >(globPtr<SrcType>(src), globPtr<SrcType>(dst), op, stream);
    }
}

void absDiffScalar(const GpuMat& src, cv::Scalar val, bool, GpuMat& dst, const GpuMat&, double, Stream& stream, int)
{
    typedef void (*func_t)(const GpuMat& src, cv::Scalar val, GpuMat& dst, Stream& stream);
    static const func_t funcs[7][4] =
    {
        {
            absDiffScalarImpl<uchar, float>, absDiffScalarImpl<uchar2, float>, absDiffScalarImpl<uchar3, float>, absDiffScalarImpl<uchar4, float>
        },
        {
            absDiffScalarImpl<schar, float>, absDiffScalarImpl<char2, float>, absDiffScalarImpl<char3, float>, absDiffScalarImpl<char4, float>
        },
        {
            absDiffScalarImpl<ushort, float>, absDiffScalarImpl<ushort2, float>, absDiffScalarImpl<ushort3, float>, absDiffScalarImpl<ushort4, float>
        },
        {
            absDiffScalarImpl<short, float>, absDiffScalarImpl<short2, float>, absDiffScalarImpl<short3, float>, absDiffScalarImpl<short4, float>
        },
        {
            absDiffScalarImpl<int, float>, absDiffScalarImpl<int2, float>, absDiffScalarImpl<int3, float>, absDiffScalarImpl<int4, float>
        },
        {
          absDiffScalarImpl<float, float>, absDiffScalarImpl<float2, float>, absDiffScalarImpl<float3, float>, absDiffScalarImpl<float4, float>
        },
        {
          absDiffScalarImpl<double, double>, absDiffScalarImpl<double2, double>, absDiffScalarImpl<double3, double>, absDiffScalarImpl<double4, double>
        }
    };

    const int sdepth = src.depth();
    const int cn = src.channels();

    CV_DbgAssert( sdepth <= CV_64F && cn <= 4 && src.type() == dst.type());

    const func_t func = funcs[sdepth][cn - 1];
    if (!func)
        CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported combination of source and destination types");

    func(src, val, dst, stream);
}

#endif
