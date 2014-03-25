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

void addScalar(const GpuMat& src, cv::Scalar val, bool, GpuMat& dst, const GpuMat& mask, double, Stream& stream, int);

namespace
{
    template <typename SrcType, typename ScalarType, typename DstType> struct AddScalarOp : unary_function<SrcType, DstType>
    {
        ScalarType val;

        __device__ __forceinline__ DstType operator ()(SrcType a) const
        {
            return saturate_cast<DstType>(saturate_cast<ScalarType>(a) + val);
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
    void addScalarImpl(const GpuMat& src, cv::Scalar value, GpuMat& dst, const GpuMat& mask, Stream& stream)
    {
        typedef typename MakeVec<ScalarDepth, VecTraits<SrcType>::cn>::type ScalarType;

        cv::Scalar_<ScalarDepth> value_ = value;

        AddScalarOp<SrcType, ScalarType, DstType> op;
        op.val = VecTraits<ScalarType>::make(value_.val);

        if (mask.data)
            gridTransformUnary_< TransformPolicy<ScalarDepth> >(globPtr<SrcType>(src), globPtr<DstType>(dst), op, globPtr<uchar>(mask), stream);
        else
            gridTransformUnary_< TransformPolicy<ScalarDepth> >(globPtr<SrcType>(src), globPtr<DstType>(dst), op, stream);
    }
}

void addScalar(const GpuMat& src, cv::Scalar val, bool, GpuMat& dst, const GpuMat& mask, double, Stream& stream, int)
{
    typedef void (*func_t)(const GpuMat& src, cv::Scalar val, GpuMat& dst, const GpuMat& mask, Stream& stream);
    static const func_t funcs[7][7][4] =
    {
        {
            {addScalarImpl<uchar, float, uchar>, addScalarImpl<uchar2, float, uchar2>, addScalarImpl<uchar3, float, uchar3>, addScalarImpl<uchar4, float, uchar4>},
            {addScalarImpl<uchar, float, schar>, addScalarImpl<uchar2, float, char2>, addScalarImpl<uchar3, float, char3>, addScalarImpl<uchar4, float, char4>},
            {addScalarImpl<uchar, float, ushort>, addScalarImpl<uchar2, float, ushort2>, addScalarImpl<uchar3, float, ushort3>, addScalarImpl<uchar4, float, ushort4>},
            {addScalarImpl<uchar, float, short>, addScalarImpl<uchar2, float, short2>, addScalarImpl<uchar3, float, short3>, addScalarImpl<uchar4, float, short4>},
            {addScalarImpl<uchar, float, int>, addScalarImpl<uchar2, float, int2>, addScalarImpl<uchar3, float, int3>, addScalarImpl<uchar4, float, int4>},
            {addScalarImpl<uchar, float, float>, addScalarImpl<uchar2, float, float2>, addScalarImpl<uchar3, float, float3>, addScalarImpl<uchar4, float, float4>},
            {addScalarImpl<uchar, double, double>, addScalarImpl<uchar2, double, double2>, addScalarImpl<uchar3, double, double3>, addScalarImpl<uchar4, double, double4>}
        },
        {
            {addScalarImpl<schar, float, uchar>, addScalarImpl<char2, float, uchar2>, addScalarImpl<char3, float, uchar3>, addScalarImpl<char4, float, uchar4>},
            {addScalarImpl<schar, float, schar>, addScalarImpl<char2, float, char2>, addScalarImpl<char3, float, char3>, addScalarImpl<char4, float, char4>},
            {addScalarImpl<schar, float, ushort>, addScalarImpl<char2, float, ushort2>, addScalarImpl<char3, float, ushort3>, addScalarImpl<char4, float, ushort4>},
            {addScalarImpl<schar, float, short>, addScalarImpl<char2, float, short2>, addScalarImpl<char3, float, short3>, addScalarImpl<char4, float, short4>},
            {addScalarImpl<schar, float, int>, addScalarImpl<char2, float, int2>, addScalarImpl<char3, float, int3>, addScalarImpl<char4, float, int4>},
            {addScalarImpl<schar, float, float>, addScalarImpl<char2, float, float2>, addScalarImpl<char3, float, float3>, addScalarImpl<char4, float, float4>},
            {addScalarImpl<schar, double, double>, addScalarImpl<char2, double, double2>, addScalarImpl<char3, double, double3>, addScalarImpl<char4, double, double4>}
        },
        {
            {0 /*addScalarImpl<ushort, float, uchar>*/, 0 /*addScalarImpl<ushort2, float, uchar2>*/, 0 /*addScalarImpl<ushort3, float, uchar3>*/, 0 /*addScalarImpl<ushort4, float, uchar4>*/},
            {0 /*addScalarImpl<ushort, float, schar>*/, 0 /*addScalarImpl<ushort2, float, char2>*/, 0 /*addScalarImpl<ushort3, float, char3>*/, 0 /*addScalarImpl<ushort4, float, char4>*/},
            {addScalarImpl<ushort, float, ushort>, addScalarImpl<ushort2, float, ushort2>, addScalarImpl<ushort3, float, ushort3>, addScalarImpl<ushort4, float, ushort4>},
            {addScalarImpl<ushort, float, short>, addScalarImpl<ushort2, float, short2>, addScalarImpl<ushort3, float, short3>, addScalarImpl<ushort4, float, short4>},
            {addScalarImpl<ushort, float, int>, addScalarImpl<ushort2, float, int2>, addScalarImpl<ushort3, float, int3>, addScalarImpl<ushort4, float, int4>},
            {addScalarImpl<ushort, float, float>, addScalarImpl<ushort2, float, float2>, addScalarImpl<ushort3, float, float3>, addScalarImpl<ushort4, float, float4>},
            {addScalarImpl<ushort, double, double>, addScalarImpl<ushort2, double, double2>, addScalarImpl<ushort3, double, double3>, addScalarImpl<ushort4, double, double4>}
        },
        {
            {0 /*addScalarImpl<short, float, uchar>*/, 0 /*addScalarImpl<short2, float, uchar2>*/, 0 /*addScalarImpl<short3, float, uchar3>*/, 0 /*addScalarImpl<short4, float, uchar4>*/},
            {0 /*addScalarImpl<short, float, schar>*/, 0 /*addScalarImpl<short2, float, char2>*/, 0 /*addScalarImpl<short3, float, char3>*/, 0 /*addScalarImpl<short4, float, char4>*/},
            {addScalarImpl<short, float, ushort>, addScalarImpl<short2, float, ushort2>, addScalarImpl<short3, float, ushort3>, addScalarImpl<short4, float, ushort4>},
            {addScalarImpl<short, float, short>, addScalarImpl<short2, float, short2>, addScalarImpl<short3, float, short3>, addScalarImpl<short4, float, short4>},
            {addScalarImpl<short, float, int>, addScalarImpl<short2, float, int2>, addScalarImpl<short3, float, int3>, addScalarImpl<short4, float, int4>},
            {addScalarImpl<short, float, float>, addScalarImpl<short2, float, float2>, addScalarImpl<short3, float, float3>, addScalarImpl<short4, float, float4>},
            {addScalarImpl<short, double, double>, addScalarImpl<short2, double, double2>, addScalarImpl<short3, double, double3>, addScalarImpl<short4, double, double4>}
        },
        {
            {0 /*addScalarImpl<int, float, uchar>*/, 0 /*addScalarImpl<int2, float, uchar2>*/, 0 /*addScalarImpl<int3, float, uchar3>*/, 0 /*addScalarImpl<int4, float, uchar4>*/},
            {0 /*addScalarImpl<int, float, schar>*/, 0 /*addScalarImpl<int2, float, char2>*/, 0 /*addScalarImpl<int3, float, char3>*/, 0 /*addScalarImpl<int4, float, char4>*/},
            {0 /*addScalarImpl<int, float, ushort>*/, 0 /*addScalarImpl<int2, float, ushort2>*/, 0 /*addScalarImpl<int3, float, ushort3>*/, 0 /*addScalarImpl<int4, float, ushort4>*/},
            {0 /*addScalarImpl<int, float, short>*/, 0 /*addScalarImpl<int2, float, short2>*/, 0 /*addScalarImpl<int3, float, short3>*/, 0 /*addScalarImpl<int4, float, short4>*/},
            {addScalarImpl<int, float, int>, addScalarImpl<int2, float, int2>, addScalarImpl<int3, float, int3>, addScalarImpl<int4, float, int4>},
            {addScalarImpl<int, float, float>, addScalarImpl<int2, float, float2>, addScalarImpl<int3, float, float3>, addScalarImpl<int4, float, float4>},
            {addScalarImpl<int, double, double>, addScalarImpl<int2, double, double2>, addScalarImpl<int3, double, double3>, addScalarImpl<int4, double, double4>}
        },
        {
            {0 /*addScalarImpl<float, float, uchar>*/, 0 /*addScalarImpl<float2, float, uchar2>*/, 0 /*addScalarImpl<float3, float, uchar3>*/, 0 /*addScalarImpl<float4, float, uchar4>*/},
            {0 /*addScalarImpl<float, float, schar>*/, 0 /*addScalarImpl<float2, float, char2>*/, 0 /*addScalarImpl<float3, float, char3>*/, 0 /*addScalarImpl<float4, float, char4>*/},
            {0 /*addScalarImpl<float, float, ushort>*/, 0 /*addScalarImpl<float2, float, ushort2>*/, 0 /*addScalarImpl<float3, float, ushort3>*/, 0 /*addScalarImpl<float4, float, ushort4>*/},
            {0 /*addScalarImpl<float, float, short>*/, 0 /*addScalarImpl<float2, float, short2>*/, 0 /*addScalarImpl<float3, float, short3>*/, 0 /*addScalarImpl<float4, float, short4>*/},
            {0 /*addScalarImpl<float, float, int>*/, 0 /*addScalarImpl<float2, float, int2>*/, 0 /*addScalarImpl<float3, float, int3>*/, 0 /*addScalarImpl<float4, float, int4>*/},
            {addScalarImpl<float, float, float>, addScalarImpl<float2, float, float2>, addScalarImpl<float3, float, float3>, addScalarImpl<float4, float, float4>},
            {addScalarImpl<float, double, double>, addScalarImpl<float2, double, double2>, addScalarImpl<float3, double, double3>, addScalarImpl<float4, double, double4>}
        },
        {
            {0 /*addScalarImpl<double, double, uchar>*/, 0 /*addScalarImpl<double2, double, uchar2>*/, 0 /*addScalarImpl<double3, double, uchar3>*/, 0 /*addScalarImpl<double4, double, uchar4>*/},
            {0 /*addScalarImpl<double, double, schar>*/, 0 /*addScalarImpl<double2, double, char2>*/, 0 /*addScalarImpl<double3, double, char3>*/, 0 /*addScalarImpl<double4, double, char4>*/},
            {0 /*addScalarImpl<double, double, ushort>*/, 0 /*addScalarImpl<double2, double, ushort2>*/, 0 /*addScalarImpl<double3, double, ushort3>*/, 0 /*addScalarImpl<double4, double, ushort4>*/},
            {0 /*addScalarImpl<double, double, short>*/, 0 /*addScalarImpl<double2, double, short2>*/, 0 /*addScalarImpl<double3, double, short3>*/, 0 /*addScalarImpl<double4, double, short4>*/},
            {0 /*addScalarImpl<double, double, int>*/, 0 /*addScalarImpl<double2, double, int2>*/, 0 /*addScalarImpl<double3, double, int3>*/, 0 /*addScalarImpl<double4, double, int4>*/},
            {0 /*addScalarImpl<double, double, float>*/, 0 /*addScalarImpl<double2, double, float2>*/, 0 /*addScalarImpl<double3, double, float3>*/, 0 /*addScalarImpl<double4, double, float4>*/},
            {addScalarImpl<double, double, double>, addScalarImpl<double2, double, double2>, addScalarImpl<double3, double, double3>, addScalarImpl<double4, double, double4>}
        }
    };

    const int sdepth = src.depth();
    const int ddepth = dst.depth();
    const int cn = src.channels();

    CV_DbgAssert( sdepth <= CV_64F && ddepth <= CV_64F && cn <= 4 );

    const func_t func = funcs[sdepth][ddepth][cn - 1];

    if (!func)
        CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported combination of source and destination types");

    func(src, val, dst, mask, stream);
}

#endif
