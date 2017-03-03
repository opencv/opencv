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

    template <typename SrcType, typename ScalarDepth, typename DstType>
    void absDiffScalarImpl(const GpuMat& src, cv::Scalar value, GpuMat& dst, Stream& stream)
    {
        typedef typename MakeVec<ScalarDepth, VecTraits<SrcType>::cn>::type ScalarType;

        cv::Scalar_<ScalarDepth> value_ = value;

        AbsDiffScalarOp<SrcType, ScalarType, DstType> op;
        op.val = VecTraits<ScalarType>::make(value_.val);
        gridTransformUnary_< TransformPolicy<ScalarDepth> >(globPtr<SrcType>(src), globPtr<SrcType>(dst), op, stream);
    }
}

void absDiffScalar(const GpuMat& src, cv::Scalar val, bool, GpuMat& dst, const GpuMat&, double, Stream& stream, int)
{
    typedef void (*func_t)(const GpuMat& src, cv::Scalar val, GpuMat& dst, Stream& stream);
    static const func_t funcs[7][7][4] =
    {
        {
            {absDiffScalarImpl<uchar, float, uchar>, absDiffScalarImpl<uchar2, float, uchar2>, absDiffScalarImpl<uchar3, float, uchar3>, absDiffScalarImpl<uchar4, float, uchar4>},
            {absDiffScalarImpl<uchar, float, schar>, absDiffScalarImpl<uchar2, float, char2>, absDiffScalarImpl<uchar3, float, char3>, absDiffScalarImpl<uchar4, float, char4>},
            {absDiffScalarImpl<uchar, float, ushort>, absDiffScalarImpl<uchar2, float, ushort2>, absDiffScalarImpl<uchar3, float, ushort3>, absDiffScalarImpl<uchar4, float, ushort4>},
            {absDiffScalarImpl<uchar, float, short>, absDiffScalarImpl<uchar2, float, short2>, absDiffScalarImpl<uchar3, float, short3>, absDiffScalarImpl<uchar4, float, short4>},
            {absDiffScalarImpl<uchar, float, int>, absDiffScalarImpl<uchar2, float, int2>, absDiffScalarImpl<uchar3, float, int3>, absDiffScalarImpl<uchar4, float, int4>},
            {absDiffScalarImpl<uchar, float, float>, absDiffScalarImpl<uchar2, float, float2>, absDiffScalarImpl<uchar3, float, float3>, absDiffScalarImpl<uchar4, float, float4>},
            {absDiffScalarImpl<uchar, double, double>, absDiffScalarImpl<uchar2, double, double2>, absDiffScalarImpl<uchar3, double, double3>, absDiffScalarImpl<uchar4, double, double4>}
        },
        {
            {absDiffScalarImpl<schar, float, uchar>, absDiffScalarImpl<char2, float, uchar2>, absDiffScalarImpl<char3, float, uchar3>, absDiffScalarImpl<char4, float, uchar4>},
            {absDiffScalarImpl<schar, float, schar>, absDiffScalarImpl<char2, float, char2>, absDiffScalarImpl<char3, float, char3>, absDiffScalarImpl<char4, float, char4>},
            {absDiffScalarImpl<schar, float, ushort>, absDiffScalarImpl<char2, float, ushort2>, absDiffScalarImpl<char3, float, ushort3>, absDiffScalarImpl<char4, float, ushort4>},
            {absDiffScalarImpl<schar, float, short>, absDiffScalarImpl<char2, float, short2>, absDiffScalarImpl<char3, float, short3>, absDiffScalarImpl<char4, float, short4>},
            {absDiffScalarImpl<schar, float, int>, absDiffScalarImpl<char2, float, int2>, absDiffScalarImpl<char3, float, int3>, absDiffScalarImpl<char4, float, int4>},
            {absDiffScalarImpl<schar, float, float>, absDiffScalarImpl<char2, float, float2>, absDiffScalarImpl<char3, float, float3>, absDiffScalarImpl<char4, float, float4>},
            {absDiffScalarImpl<schar, double, double>, absDiffScalarImpl<char2, double, double2>, absDiffScalarImpl<char3, double, double3>, absDiffScalarImpl<char4, double, double4>}
        },
        {
            {0 /*absDiffScalarImpl<ushort, float, uchar>*/, 0 /*absDiffScalarImpl<ushort2, float, uchar2>*/, 0 /*absDiffScalarImpl<ushort3, float, uchar3>*/, 0 /*absDiffScalarImpl<ushort4, float, uchar4>*/},
            {0 /*absDiffScalarImpl<ushort, float, schar>*/, 0 /*absDiffScalarImpl<ushort2, float, char2>*/, 0 /*absDiffScalarImpl<ushort3, float, char3>*/, 0 /*absDiffScalarImpl<ushort4, float, char4>*/},
            {absDiffScalarImpl<ushort, float, ushort>, absDiffScalarImpl<ushort2, float, ushort2>, absDiffScalarImpl<ushort3, float, ushort3>, absDiffScalarImpl<ushort4, float, ushort4>},
            {absDiffScalarImpl<ushort, float, short>, absDiffScalarImpl<ushort2, float, short2>, absDiffScalarImpl<ushort3, float, short3>, absDiffScalarImpl<ushort4, float, short4>},
            {absDiffScalarImpl<ushort, float, int>, absDiffScalarImpl<ushort2, float, int2>, absDiffScalarImpl<ushort3, float, int3>, absDiffScalarImpl<ushort4, float, int4>},
            {absDiffScalarImpl<ushort, float, float>, absDiffScalarImpl<ushort2, float, float2>, absDiffScalarImpl<ushort3, float, float3>, absDiffScalarImpl<ushort4, float, float4>},
            {absDiffScalarImpl<ushort, double, double>, absDiffScalarImpl<ushort2, double, double2>, absDiffScalarImpl<ushort3, double, double3>, absDiffScalarImpl<ushort4, double, double4>}
        },
        {
            {0 /*absDiffScalarImpl<short, float, uchar>*/, 0 /*absDiffScalarImpl<short2, float, uchar2>*/, 0 /*absDiffScalarImpl<short3, float, uchar3>*/, 0 /*absDiffScalarImpl<short4, float, uchar4>*/},
            {0 /*absDiffScalarImpl<short, float, schar>*/, 0 /*absDiffScalarImpl<short2, float, char2>*/, 0 /*absDiffScalarImpl<short3, float, char3>*/, 0 /*absDiffScalarImpl<short4, float, char4>*/},
            {absDiffScalarImpl<short, float, ushort>, absDiffScalarImpl<short2, float, ushort2>, absDiffScalarImpl<short3, float, ushort3>, absDiffScalarImpl<short4, float, ushort4>},
            {absDiffScalarImpl<short, float, short>, absDiffScalarImpl<short2, float, short2>, absDiffScalarImpl<short3, float, short3>, absDiffScalarImpl<short4, float, short4>},
            {absDiffScalarImpl<short, float, int>, absDiffScalarImpl<short2, float, int2>, absDiffScalarImpl<short3, float, int3>, absDiffScalarImpl<short4, float, int4>},
            {absDiffScalarImpl<short, float, float>, absDiffScalarImpl<short2, float, float2>, absDiffScalarImpl<short3, float, float3>, absDiffScalarImpl<short4, float, float4>},
            {absDiffScalarImpl<short, double, double>, absDiffScalarImpl<short2, double, double2>, absDiffScalarImpl<short3, double, double3>, absDiffScalarImpl<short4, double, double4>}
        },
        {
            {0 /*absDiffScalarImpl<int, float, uchar>*/, 0 /*absDiffScalarImpl<int2, float, uchar2>*/, 0 /*absDiffScalarImpl<int3, float, uchar3>*/, 0 /*absDiffScalarImpl<int4, float, uchar4>*/},
            {0 /*absDiffScalarImpl<int, float, schar>*/, 0 /*absDiffScalarImpl<int2, float, char2>*/, 0 /*absDiffScalarImpl<int3, float, char3>*/, 0 /*absDiffScalarImpl<int4, float, char4>*/},
            {0 /*absDiffScalarImpl<int, float, ushort>*/, 0 /*absDiffScalarImpl<int2, float, ushort2>*/, 0 /*absDiffScalarImpl<int3, float, ushort3>*/, 0 /*absDiffScalarImpl<int4, float, ushort4>*/},
            {0 /*absDiffScalarImpl<int, float, short>*/, 0 /*absDiffScalarImpl<int2, float, short2>*/, 0 /*absDiffScalarImpl<int3, float, short3>*/, 0 /*absDiffScalarImpl<int4, float, short4>*/},
            {absDiffScalarImpl<int, float, int>, absDiffScalarImpl<int2, float, int2>, absDiffScalarImpl<int3, float, int3>, absDiffScalarImpl<int4, float, int4>},
            {absDiffScalarImpl<int, float, float>, absDiffScalarImpl<int2, float, float2>, absDiffScalarImpl<int3, float, float3>, absDiffScalarImpl<int4, float, float4>},
            {absDiffScalarImpl<int, double, double>, absDiffScalarImpl<int2, double, double2>, absDiffScalarImpl<int3, double, double3>, absDiffScalarImpl<int4, double, double4>}
        },
        {
            {0 /*absDiffScalarImpl<float, float, uchar>*/, 0 /*absDiffScalarImpl<float2, float, uchar2>*/, 0 /*absDiffScalarImpl<float3, float, uchar3>*/, 0 /*absDiffScalarImpl<float4, float, uchar4>*/},
            {0 /*absDiffScalarImpl<float, float, schar>*/, 0 /*absDiffScalarImpl<float2, float, char2>*/, 0 /*absDiffScalarImpl<float3, float, char3>*/, 0 /*absDiffScalarImpl<float4, float, char4>*/},
            {0 /*absDiffScalarImpl<float, float, ushort>*/, 0 /*absDiffScalarImpl<float2, float, ushort2>*/, 0 /*absDiffScalarImpl<float3, float, ushort3>*/, 0 /*absDiffScalarImpl<float4, float, ushort4>*/},
            {0 /*absDiffScalarImpl<float, float, short>*/, 0 /*absDiffScalarImpl<float2, float, short2>*/, 0 /*absDiffScalarImpl<float3, float, short3>*/, 0 /*absDiffScalarImpl<float4, float, short4>*/},
            {0 /*absDiffScalarImpl<float, float, int>*/, 0 /*absDiffScalarImpl<float2, float, int2>*/, 0 /*absDiffScalarImpl<float3, float, int3>*/, 0 /*absDiffScalarImpl<float4, float, int4>*/},
            {absDiffScalarImpl<float, float, float>, absDiffScalarImpl<float2, float, float2>, absDiffScalarImpl<float3, float, float3>, absDiffScalarImpl<float4, float, float4>},
            {absDiffScalarImpl<float, double, double>, absDiffScalarImpl<float2, double, double2>, absDiffScalarImpl<float3, double, double3>, absDiffScalarImpl<float4, double, double4>}
        },
        {
            {0 /*absDiffScalarImpl<double, double, uchar>*/, 0 /*absDiffScalarImpl<double2, double, uchar2>*/, 0 /*absDiffScalarImpl<double3, double, uchar3>*/, 0 /*absDiffScalarImpl<double4, double, uchar4>*/},
            {0 /*absDiffScalarImpl<double, double, schar>*/, 0 /*absDiffScalarImpl<double2, double, char2>*/, 0 /*absDiffScalarImpl<double3, double, char3>*/, 0 /*absDiffScalarImpl<double4, double, char4>*/},
            {0 /*absDiffScalarImpl<double, double, ushort>*/, 0 /*absDiffScalarImpl<double2, double, ushort2>*/, 0 /*absDiffScalarImpl<double3, double, ushort3>*/, 0 /*absDiffScalarImpl<double4, double, ushort4>*/},
            {0 /*absDiffScalarImpl<double, double, short>*/, 0 /*absDiffScalarImpl<double2, double, short2>*/, 0 /*absDiffScalarImpl<double3, double, short3>*/, 0 /*absDiffScalarImpl<double4, double, short4>*/},
            {0 /*absDiffScalarImpl<double, double, int>*/, 0 /*absDiffScalarImpl<double2, double, int2>*/, 0 /*absDiffScalarImpl<double3, double, int3>*/, 0 /*absDiffScalarImpl<double4, double, int4>*/},
            {0 /*absDiffScalarImpl<double, double, float>*/, 0 /*absDiffScalarImpl<double2, double, float2>*/, 0 /*absDiffScalarImpl<double3, double, float3>*/, 0 /*absDiffScalarImpl<double4, double, float4>*/},
            {absDiffScalarImpl<double, double, double>, absDiffScalarImpl<double2, double, double2>, absDiffScalarImpl<double3, double, double3>, absDiffScalarImpl<double4, double, double4>}
        }
    };

    const int sdepth = src.depth();
    const int ddepth = dst.depth();
    const int cn = src.channels();

    CV_DbgAssert( sdepth <= CV_64F && ddepth <= CV_64F && cn <= 4 );

    const func_t func = funcs[sdepth][ddepth][cn - 1];

    if (!func)
        CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported combination of source and destination types");

    func(src, val, dst, stream);
}

#endif
