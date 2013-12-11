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

void mulScalar(const GpuMat& src, cv::Scalar val, bool, GpuMat& dst, const GpuMat& mask, double scale, Stream& stream, int);

namespace
{
    template <typename SrcType, typename ScalarType, typename DstType> struct MulScalarOp : unary_function<SrcType, DstType>
    {
        ScalarType val;

        __device__ __forceinline__ DstType operator ()(SrcType a) const
        {
            return saturate_cast<DstType>(saturate_cast<ScalarType>(a) * val);
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
    void mulScalarImpl(const GpuMat& src, cv::Scalar value, GpuMat& dst, Stream& stream)
    {
        typedef typename MakeVec<ScalarDepth, VecTraits<SrcType>::cn>::type ScalarType;

        cv::Scalar_<ScalarDepth> value_ = value;

        MulScalarOp<SrcType, ScalarType, DstType> op;
        op.val = VecTraits<ScalarType>::make(value_.val);

        gridTransformUnary_< TransformPolicy<ScalarDepth> >(globPtr<SrcType>(src), globPtr<DstType>(dst), op, stream);
    }
}

void mulScalar(const GpuMat& src, cv::Scalar val, bool, GpuMat& dst, const GpuMat&, double scale, Stream& stream, int)
{
    typedef void (*func_t)(const GpuMat& src, cv::Scalar val, GpuMat& dst, Stream& stream);
    static const func_t funcs[7][7][4] =
    {
        {
            {mulScalarImpl<uchar, float, uchar>, mulScalarImpl<uchar2, float, uchar2>, mulScalarImpl<uchar3, float, uchar3>, mulScalarImpl<uchar4, float, uchar4>},
            {mulScalarImpl<uchar, float, schar>, mulScalarImpl<uchar2, float, char2>, mulScalarImpl<uchar3, float, char3>, mulScalarImpl<uchar4, float, char4>},
            {mulScalarImpl<uchar, float, ushort>, mulScalarImpl<uchar2, float, ushort2>, mulScalarImpl<uchar3, float, ushort3>, mulScalarImpl<uchar4, float, ushort4>},
            {mulScalarImpl<uchar, float, short>, mulScalarImpl<uchar2, float, short2>, mulScalarImpl<uchar3, float, short3>, mulScalarImpl<uchar4, float, short4>},
            {mulScalarImpl<uchar, float, int>, mulScalarImpl<uchar2, float, int2>, mulScalarImpl<uchar3, float, int3>, mulScalarImpl<uchar4, float, int4>},
            {mulScalarImpl<uchar, float, float>, mulScalarImpl<uchar2, float, float2>, mulScalarImpl<uchar3, float, float3>, mulScalarImpl<uchar4, float, float4>},
            {mulScalarImpl<uchar, double, double>, mulScalarImpl<uchar2, double, double2>, mulScalarImpl<uchar3, double, double3>, mulScalarImpl<uchar4, double, double4>}
        },
        {
            {mulScalarImpl<schar, float, uchar>, mulScalarImpl<char2, float, uchar2>, mulScalarImpl<char3, float, uchar3>, mulScalarImpl<char4, float, uchar4>},
            {mulScalarImpl<schar, float, schar>, mulScalarImpl<char2, float, char2>, mulScalarImpl<char3, float, char3>, mulScalarImpl<char4, float, char4>},
            {mulScalarImpl<schar, float, ushort>, mulScalarImpl<char2, float, ushort2>, mulScalarImpl<char3, float, ushort3>, mulScalarImpl<char4, float, ushort4>},
            {mulScalarImpl<schar, float, short>, mulScalarImpl<char2, float, short2>, mulScalarImpl<char3, float, short3>, mulScalarImpl<char4, float, short4>},
            {mulScalarImpl<schar, float, int>, mulScalarImpl<char2, float, int2>, mulScalarImpl<char3, float, int3>, mulScalarImpl<char4, float, int4>},
            {mulScalarImpl<schar, float, float>, mulScalarImpl<char2, float, float2>, mulScalarImpl<char3, float, float3>, mulScalarImpl<char4, float, float4>},
            {mulScalarImpl<schar, double, double>, mulScalarImpl<char2, double, double2>, mulScalarImpl<char3, double, double3>, mulScalarImpl<char4, double, double4>}
        },
        {
            {0 /*mulScalarImpl<ushort, float, uchar>*/, 0 /*mulScalarImpl<ushort2, float, uchar2>*/, 0 /*mulScalarImpl<ushort3, float, uchar3>*/, 0 /*mulScalarImpl<ushort4, float, uchar4>*/},
            {0 /*mulScalarImpl<ushort, float, schar>*/, 0 /*mulScalarImpl<ushort2, float, char2>*/, 0 /*mulScalarImpl<ushort3, float, char3>*/, 0 /*mulScalarImpl<ushort4, float, char4>*/},
            {mulScalarImpl<ushort, float, ushort>, mulScalarImpl<ushort2, float, ushort2>, mulScalarImpl<ushort3, float, ushort3>, mulScalarImpl<ushort4, float, ushort4>},
            {mulScalarImpl<ushort, float, short>, mulScalarImpl<ushort2, float, short2>, mulScalarImpl<ushort3, float, short3>, mulScalarImpl<ushort4, float, short4>},
            {mulScalarImpl<ushort, float, int>, mulScalarImpl<ushort2, float, int2>, mulScalarImpl<ushort3, float, int3>, mulScalarImpl<ushort4, float, int4>},
            {mulScalarImpl<ushort, float, float>, mulScalarImpl<ushort2, float, float2>, mulScalarImpl<ushort3, float, float3>, mulScalarImpl<ushort4, float, float4>},
            {mulScalarImpl<ushort, double, double>, mulScalarImpl<ushort2, double, double2>, mulScalarImpl<ushort3, double, double3>, mulScalarImpl<ushort4, double, double4>}
        },
        {
            {0 /*mulScalarImpl<short, float, uchar>*/, 0 /*mulScalarImpl<short2, float, uchar2>*/, 0 /*mulScalarImpl<short3, float, uchar3>*/, 0 /*mulScalarImpl<short4, float, uchar4>*/},
            {0 /*mulScalarImpl<short, float, schar>*/, 0 /*mulScalarImpl<short2, float, char2>*/, 0 /*mulScalarImpl<short3, float, char3>*/, 0 /*mulScalarImpl<short4, float, char4>*/},
            {mulScalarImpl<short, float, ushort>, mulScalarImpl<short2, float, ushort2>, mulScalarImpl<short3, float, ushort3>, mulScalarImpl<short4, float, ushort4>},
            {mulScalarImpl<short, float, short>, mulScalarImpl<short2, float, short2>, mulScalarImpl<short3, float, short3>, mulScalarImpl<short4, float, short4>},
            {mulScalarImpl<short, float, int>, mulScalarImpl<short2, float, int2>, mulScalarImpl<short3, float, int3>, mulScalarImpl<short4, float, int4>},
            {mulScalarImpl<short, float, float>, mulScalarImpl<short2, float, float2>, mulScalarImpl<short3, float, float3>, mulScalarImpl<short4, float, float4>},
            {mulScalarImpl<short, double, double>, mulScalarImpl<short2, double, double2>, mulScalarImpl<short3, double, double3>, mulScalarImpl<short4, double, double4>}
        },
        {
            {0 /*mulScalarImpl<int, float, uchar>*/, 0 /*mulScalarImpl<int2, float, uchar2>*/, 0 /*mulScalarImpl<int3, float, uchar3>*/, 0 /*mulScalarImpl<int4, float, uchar4>*/},
            {0 /*mulScalarImpl<int, float, schar>*/, 0 /*mulScalarImpl<int2, float, char2>*/, 0 /*mulScalarImpl<int3, float, char3>*/, 0 /*mulScalarImpl<int4, float, char4>*/},
            {0 /*mulScalarImpl<int, float, ushort>*/, 0 /*mulScalarImpl<int2, float, ushort2>*/, 0 /*mulScalarImpl<int3, float, ushort3>*/, 0 /*mulScalarImpl<int4, float, ushort4>*/},
            {0 /*mulScalarImpl<int, float, short>*/, 0 /*mulScalarImpl<int2, float, short2>*/, 0 /*mulScalarImpl<int3, float, short3>*/, 0 /*mulScalarImpl<int4, float, short4>*/},
            {mulScalarImpl<int, float, int>, mulScalarImpl<int2, float, int2>, mulScalarImpl<int3, float, int3>, mulScalarImpl<int4, float, int4>},
            {mulScalarImpl<int, float, float>, mulScalarImpl<int2, float, float2>, mulScalarImpl<int3, float, float3>, mulScalarImpl<int4, float, float4>},
            {mulScalarImpl<int, double, double>, mulScalarImpl<int2, double, double2>, mulScalarImpl<int3, double, double3>, mulScalarImpl<int4, double, double4>}
        },
        {
            {0 /*mulScalarImpl<float, float, uchar>*/, 0 /*mulScalarImpl<float2, float, uchar2>*/, 0 /*mulScalarImpl<float3, float, uchar3>*/, 0 /*mulScalarImpl<float4, float, uchar4>*/},
            {0 /*mulScalarImpl<float, float, schar>*/, 0 /*mulScalarImpl<float2, float, char2>*/, 0 /*mulScalarImpl<float3, float, char3>*/, 0 /*mulScalarImpl<float4, float, char4>*/},
            {0 /*mulScalarImpl<float, float, ushort>*/, 0 /*mulScalarImpl<float2, float, ushort2>*/, 0 /*mulScalarImpl<float3, float, ushort3>*/, 0 /*mulScalarImpl<float4, float, ushort4>*/},
            {0 /*mulScalarImpl<float, float, short>*/, 0 /*mulScalarImpl<float2, float, short2>*/, 0 /*mulScalarImpl<float3, float, short3>*/, 0 /*mulScalarImpl<float4, float, short4>*/},
            {0 /*mulScalarImpl<float, float, int>*/, 0 /*mulScalarImpl<float2, float, int2>*/, 0 /*mulScalarImpl<float3, float, int3>*/, 0 /*mulScalarImpl<float4, float, int4>*/},
            {mulScalarImpl<float, float, float>, mulScalarImpl<float2, float, float2>, mulScalarImpl<float3, float, float3>, mulScalarImpl<float4, float, float4>},
            {mulScalarImpl<float, double, double>, mulScalarImpl<float2, double, double2>, mulScalarImpl<float3, double, double3>, mulScalarImpl<float4, double, double4>}
        },
        {
            {0 /*mulScalarImpl<double, double, uchar>*/, 0 /*mulScalarImpl<double2, double, uchar2>*/, 0 /*mulScalarImpl<double3, double, uchar3>*/, 0 /*mulScalarImpl<double4, double, uchar4>*/},
            {0 /*mulScalarImpl<double, double, schar>*/, 0 /*mulScalarImpl<double2, double, char2>*/, 0 /*mulScalarImpl<double3, double, char3>*/, 0 /*mulScalarImpl<double4, double, char4>*/},
            {0 /*mulScalarImpl<double, double, ushort>*/, 0 /*mulScalarImpl<double2, double, ushort2>*/, 0 /*mulScalarImpl<double3, double, ushort3>*/, 0 /*mulScalarImpl<double4, double, ushort4>*/},
            {0 /*mulScalarImpl<double, double, short>*/, 0 /*mulScalarImpl<double2, double, short2>*/, 0 /*mulScalarImpl<double3, double, short3>*/, 0 /*mulScalarImpl<double4, double, short4>*/},
            {0 /*mulScalarImpl<double, double, int>*/, 0 /*mulScalarImpl<double2, double, int2>*/, 0 /*mulScalarImpl<double3, double, int3>*/, 0 /*mulScalarImpl<double4, double, int4>*/},
            {0 /*mulScalarImpl<double, double, float>*/, 0 /*mulScalarImpl<double2, double, float2>*/, 0 /*mulScalarImpl<double3, double, float3>*/, 0 /*mulScalarImpl<double4, double, float4>*/},
            {mulScalarImpl<double, double, double>, mulScalarImpl<double2, double, double2>, mulScalarImpl<double3, double, double3>, mulScalarImpl<double4, double, double4>}
        }
    };

    const int sdepth = src.depth();
    const int ddepth = dst.depth();
    const int cn = src.channels();

    CV_DbgAssert( sdepth <= CV_64F && ddepth <= CV_64F && cn <= 4 );

    val[0] *= scale;
    val[1] *= scale;
    val[2] *= scale;
    val[3] *= scale;

    const func_t func = funcs[sdepth][ddepth][cn - 1];

    if (!func)
        CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported combination of source and destination types");

    func(src, val, dst, stream);
}

#endif
