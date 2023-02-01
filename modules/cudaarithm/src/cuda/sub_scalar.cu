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

void subScalar(const GpuMat& src, cv::Scalar val, bool inv, GpuMat& dst, const GpuMat& mask, double, Stream& stream, int);

namespace
{
    template <typename SrcType, typename ScalarType, typename DstType> struct SubScalarOp : unary_function<SrcType, DstType>
    {
        ScalarType val;

        __device__ __forceinline__ DstType operator ()(SrcType a) const
        {
            return saturate_cast<DstType>(saturate_cast<ScalarType>(a) - val);
        }
    };

    template <typename SrcType, typename ScalarType, typename DstType> struct SubScalarOpInv : unary_function<SrcType, DstType>
    {
        ScalarType val;

        __device__ __forceinline__ DstType operator ()(SrcType a) const
        {
            return saturate_cast<DstType>(val - saturate_cast<ScalarType>(a));
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
    void subScalarImpl(const GpuMat& src, cv::Scalar value, bool inv, GpuMat& dst, const GpuMat& mask, Stream& stream)
    {
        typedef typename MakeVec<ScalarDepth, VecTraits<SrcType>::cn>::type ScalarType;

        cv::Scalar_<ScalarDepth> value_ = value;

        if (inv)
        {
            SubScalarOpInv<SrcType, ScalarType, DstType> op;
            op.val = VecTraits<ScalarType>::make(value_.val);

            if (mask.data)
                gridTransformUnary_< TransformPolicy<ScalarDepth> >(globPtr<SrcType>(src), globPtr<DstType>(dst), op, globPtr<uchar>(mask), stream);
            else
                gridTransformUnary_< TransformPolicy<ScalarDepth> >(globPtr<SrcType>(src), globPtr<DstType>(dst), op, stream);
        }
        else
        {
            SubScalarOp<SrcType, ScalarType, DstType> op;
            op.val = VecTraits<ScalarType>::make(value_.val);

            if (mask.data)
                gridTransformUnary_< TransformPolicy<ScalarDepth> >(globPtr<SrcType>(src), globPtr<DstType>(dst), op, globPtr<uchar>(mask), stream);
            else
                gridTransformUnary_< TransformPolicy<ScalarDepth> >(globPtr<SrcType>(src), globPtr<DstType>(dst), op, stream);
        }
    }
}

void subScalar(const GpuMat& src, cv::Scalar val, bool inv, GpuMat& dst, const GpuMat& mask, double, Stream& stream, int)
{
    typedef void (*func_t)(const GpuMat& src, cv::Scalar val, bool inv, GpuMat& dst, const GpuMat& mask, Stream& stream);
    static const func_t funcs[7][7][4] =
    {
        {
            {subScalarImpl<uchar, float, uchar>, subScalarImpl<uchar2, float, uchar2>, subScalarImpl<uchar3, float, uchar3>, subScalarImpl<uchar4, float, uchar4>},
            {subScalarImpl<uchar, float, schar>, subScalarImpl<uchar2, float, char2>, subScalarImpl<uchar3, float, char3>, subScalarImpl<uchar4, float, char4>},
            {subScalarImpl<uchar, float, ushort>, subScalarImpl<uchar2, float, ushort2>, subScalarImpl<uchar3, float, ushort3>, subScalarImpl<uchar4, float, ushort4>},
            {subScalarImpl<uchar, float, short>, subScalarImpl<uchar2, float, short2>, subScalarImpl<uchar3, float, short3>, subScalarImpl<uchar4, float, short4>},
            {subScalarImpl<uchar, float, int>, subScalarImpl<uchar2, float, int2>, subScalarImpl<uchar3, float, int3>, subScalarImpl<uchar4, float, int4>},
            {subScalarImpl<uchar, float, float>, subScalarImpl<uchar2, float, float2>, subScalarImpl<uchar3, float, float3>, subScalarImpl<uchar4, float, float4>},
            {subScalarImpl<uchar, double, double>, subScalarImpl<uchar2, double, double2>, subScalarImpl<uchar3, double, double3>, subScalarImpl<uchar4, double, double4>}
        },
        {
            {subScalarImpl<schar, float, uchar>, subScalarImpl<char2, float, uchar2>, subScalarImpl<char3, float, uchar3>, subScalarImpl<char4, float, uchar4>},
            {subScalarImpl<schar, float, schar>, subScalarImpl<char2, float, char2>, subScalarImpl<char3, float, char3>, subScalarImpl<char4, float, char4>},
            {subScalarImpl<schar, float, ushort>, subScalarImpl<char2, float, ushort2>, subScalarImpl<char3, float, ushort3>, subScalarImpl<char4, float, ushort4>},
            {subScalarImpl<schar, float, short>, subScalarImpl<char2, float, short2>, subScalarImpl<char3, float, short3>, subScalarImpl<char4, float, short4>},
            {subScalarImpl<schar, float, int>, subScalarImpl<char2, float, int2>, subScalarImpl<char3, float, int3>, subScalarImpl<char4, float, int4>},
            {subScalarImpl<schar, float, float>, subScalarImpl<char2, float, float2>, subScalarImpl<char3, float, float3>, subScalarImpl<char4, float, float4>},
            {subScalarImpl<schar, double, double>, subScalarImpl<char2, double, double2>, subScalarImpl<char3, double, double3>, subScalarImpl<char4, double, double4>}
        },
        {
            {0 /*subScalarImpl<ushort, float, uchar>*/, 0 /*subScalarImpl<ushort2, float, uchar2>*/, 0 /*subScalarImpl<ushort3, float, uchar3>*/, 0 /*subScalarImpl<ushort4, float, uchar4>*/},
            {0 /*subScalarImpl<ushort, float, schar>*/, 0 /*subScalarImpl<ushort2, float, char2>*/, 0 /*subScalarImpl<ushort3, float, char3>*/, 0 /*subScalarImpl<ushort4, float, char4>*/},
            {subScalarImpl<ushort, float, ushort>, subScalarImpl<ushort2, float, ushort2>, subScalarImpl<ushort3, float, ushort3>, subScalarImpl<ushort4, float, ushort4>},
            {subScalarImpl<ushort, float, short>, subScalarImpl<ushort2, float, short2>, subScalarImpl<ushort3, float, short3>, subScalarImpl<ushort4, float, short4>},
            {subScalarImpl<ushort, float, int>, subScalarImpl<ushort2, float, int2>, subScalarImpl<ushort3, float, int3>, subScalarImpl<ushort4, float, int4>},
            {subScalarImpl<ushort, float, float>, subScalarImpl<ushort2, float, float2>, subScalarImpl<ushort3, float, float3>, subScalarImpl<ushort4, float, float4>},
            {subScalarImpl<ushort, double, double>, subScalarImpl<ushort2, double, double2>, subScalarImpl<ushort3, double, double3>, subScalarImpl<ushort4, double, double4>}
        },
        {
            {0 /*subScalarImpl<short, float, uchar>*/, 0 /*subScalarImpl<short2, float, uchar2>*/, 0 /*subScalarImpl<short3, float, uchar3>*/, 0 /*subScalarImpl<short4, float, uchar4>*/},
            {0 /*subScalarImpl<short, float, schar>*/, 0 /*subScalarImpl<short2, float, char2>*/, 0 /*subScalarImpl<short3, float, char3>*/, 0 /*subScalarImpl<short4, float, char4>*/},
            {subScalarImpl<short, float, ushort>, subScalarImpl<short2, float, ushort2>, subScalarImpl<short3, float, ushort3>, subScalarImpl<short4, float, ushort4>},
            {subScalarImpl<short, float, short>, subScalarImpl<short2, float, short2>, subScalarImpl<short3, float, short3>, subScalarImpl<short4, float, short4>},
            {subScalarImpl<short, float, int>, subScalarImpl<short2, float, int2>, subScalarImpl<short3, float, int3>, subScalarImpl<short4, float, int4>},
            {subScalarImpl<short, float, float>, subScalarImpl<short2, float, float2>, subScalarImpl<short3, float, float3>, subScalarImpl<short4, float, float4>},
            {subScalarImpl<short, double, double>, subScalarImpl<short2, double, double2>, subScalarImpl<short3, double, double3>, subScalarImpl<short4, double, double4>}
        },
        {
            {0 /*subScalarImpl<int, float, uchar>*/, 0 /*subScalarImpl<int2, float, uchar2>*/, 0 /*subScalarImpl<int3, float, uchar3>*/, 0 /*subScalarImpl<int4, float, uchar4>*/},
            {0 /*subScalarImpl<int, float, schar>*/, 0 /*subScalarImpl<int2, float, char2>*/, 0 /*subScalarImpl<int3, float, char3>*/, 0 /*subScalarImpl<int4, float, char4>*/},
            {0 /*subScalarImpl<int, float, ushort>*/, 0 /*subScalarImpl<int2, float, ushort2>*/, 0 /*subScalarImpl<int3, float, ushort3>*/, 0 /*subScalarImpl<int4, float, ushort4>*/},
            {0 /*subScalarImpl<int, float, short>*/, 0 /*subScalarImpl<int2, float, short2>*/, 0 /*subScalarImpl<int3, float, short3>*/, 0 /*subScalarImpl<int4, float, short4>*/},
            {subScalarImpl<int, float, int>, subScalarImpl<int2, float, int2>, subScalarImpl<int3, float, int3>, subScalarImpl<int4, float, int4>},
            {subScalarImpl<int, float, float>, subScalarImpl<int2, float, float2>, subScalarImpl<int3, float, float3>, subScalarImpl<int4, float, float4>},
            {subScalarImpl<int, double, double>, subScalarImpl<int2, double, double2>, subScalarImpl<int3, double, double3>, subScalarImpl<int4, double, double4>}
        },
        {
            {0 /*subScalarImpl<float, float, uchar>*/, 0 /*subScalarImpl<float2, float, uchar2>*/, 0 /*subScalarImpl<float3, float, uchar3>*/, 0 /*subScalarImpl<float4, float, uchar4>*/},
            {0 /*subScalarImpl<float, float, schar>*/, 0 /*subScalarImpl<float2, float, char2>*/, 0 /*subScalarImpl<float3, float, char3>*/, 0 /*subScalarImpl<float4, float, char4>*/},
            {0 /*subScalarImpl<float, float, ushort>*/, 0 /*subScalarImpl<float2, float, ushort2>*/, 0 /*subScalarImpl<float3, float, ushort3>*/, 0 /*subScalarImpl<float4, float, ushort4>*/},
            {0 /*subScalarImpl<float, float, short>*/, 0 /*subScalarImpl<float2, float, short2>*/, 0 /*subScalarImpl<float3, float, short3>*/, 0 /*subScalarImpl<float4, float, short4>*/},
            {0 /*subScalarImpl<float, float, int>*/, 0 /*subScalarImpl<float2, float, int2>*/, 0 /*subScalarImpl<float3, float, int3>*/, 0 /*subScalarImpl<float4, float, int4>*/},
            {subScalarImpl<float, float, float>, subScalarImpl<float2, float, float2>, subScalarImpl<float3, float, float3>, subScalarImpl<float4, float, float4>},
            {subScalarImpl<float, double, double>, subScalarImpl<float2, double, double2>, subScalarImpl<float3, double, double3>, subScalarImpl<float4, double, double4>}
        },
        {
            {0 /*subScalarImpl<double, double, uchar>*/, 0 /*subScalarImpl<double2, double, uchar2>*/, 0 /*subScalarImpl<double3, double, uchar3>*/, 0 /*subScalarImpl<double4, double, uchar4>*/},
            {0 /*subScalarImpl<double, double, schar>*/, 0 /*subScalarImpl<double2, double, char2>*/, 0 /*subScalarImpl<double3, double, char3>*/, 0 /*subScalarImpl<double4, double, char4>*/},
            {0 /*subScalarImpl<double, double, ushort>*/, 0 /*subScalarImpl<double2, double, ushort2>*/, 0 /*subScalarImpl<double3, double, ushort3>*/, 0 /*subScalarImpl<double4, double, ushort4>*/},
            {0 /*subScalarImpl<double, double, short>*/, 0 /*subScalarImpl<double2, double, short2>*/, 0 /*subScalarImpl<double3, double, short3>*/, 0 /*subScalarImpl<double4, double, short4>*/},
            {0 /*subScalarImpl<double, double, int>*/, 0 /*subScalarImpl<double2, double, int2>*/, 0 /*subScalarImpl<double3, double, int3>*/, 0 /*subScalarImpl<double4, double, int4>*/},
            {0 /*subScalarImpl<double, double, float>*/, 0 /*subScalarImpl<double2, double, float2>*/, 0 /*subScalarImpl<double3, double, float3>*/, 0 /*subScalarImpl<double4, double, float4>*/},
            {subScalarImpl<double, double, double>, subScalarImpl<double2, double, double2>, subScalarImpl<double3, double, double3>, subScalarImpl<double4, double, double4>}
        }
    };

    const int sdepth = src.depth();
    const int ddepth = dst.depth();
    const int cn = src.channels();

    CV_DbgAssert( sdepth <= CV_64F && ddepth <= CV_64F && cn <= 4 );

    const func_t func = funcs[sdepth][ddepth][cn - 1];

    if (!func)
        CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported combination of source and destination types");

    func(src, val, inv, dst, mask, stream);
}

#endif
