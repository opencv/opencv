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

void divScalar(const GpuMat& src, cv::Scalar val, bool inv, GpuMat& dst, const GpuMat& mask, double scale, Stream& stream, int);

namespace
{
    template <typename T, int cn> struct SafeDiv;
    template <typename T> struct SafeDiv<T, 1>
    {
        __device__ __forceinline__ static T op(T a, T b)
        {
            return b != 0 ? a / b : 0;
        }
    };
    template <typename T> struct SafeDiv<T, 2>
    {
        __device__ __forceinline__ static T op(const T& a, const T& b)
        {
            T res;

            res.x = b.x != 0 ? a.x / b.x : 0;
            res.y = b.y != 0 ? a.y / b.y : 0;

            return res;
        }
    };
    template <typename T> struct SafeDiv<T, 3>
    {
        __device__ __forceinline__ static T op(const T& a, const T& b)
        {
            T res;

            res.x = b.x != 0 ? a.x / b.x : 0;
            res.y = b.y != 0 ? a.y / b.y : 0;
            res.z = b.z != 0 ? a.z / b.z : 0;

            return res;
        }
    };
    template <typename T> struct SafeDiv<T, 4>
    {
        __device__ __forceinline__ static T op(const T& a, const T& b)
        {
            T res;

            res.x = b.x != 0 ? a.x / b.x : 0;
            res.y = b.y != 0 ? a.y / b.y : 0;
            res.z = b.z != 0 ? a.z / b.z : 0;
            res.w = b.w != 0 ? a.w / b.w : 0;

            return res;
        }
    };

    template <typename SrcType, typename ScalarType, typename DstType> struct DivScalarOp : unary_function<SrcType, DstType>
    {
        ScalarType val;

        __device__ __forceinline__ DstType operator ()(SrcType a) const
        {
            return saturate_cast<DstType>(SafeDiv<ScalarType, VecTraits<ScalarType>::cn>::op(saturate_cast<ScalarType>(a), val));
        }
    };

    template <typename SrcType, typename ScalarType, typename DstType> struct DivScalarOpInv : unary_function<SrcType, DstType>
    {
        ScalarType val;

        __device__ __forceinline__ DstType operator ()(SrcType a) const
        {
            return saturate_cast<DstType>(SafeDiv<ScalarType, VecTraits<ScalarType>::cn>::op(val, saturate_cast<ScalarType>(a)));
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
    void divScalarImpl(const GpuMat& src, cv::Scalar value, bool inv, GpuMat& dst, Stream& stream)
    {
        typedef typename MakeVec<ScalarDepth, VecTraits<SrcType>::cn>::type ScalarType;

        cv::Scalar_<ScalarDepth> value_ = value;

        if (inv)
        {
            DivScalarOpInv<SrcType, ScalarType, DstType> op;
            op.val = VecTraits<ScalarType>::make(value_.val);

            gridTransformUnary_< TransformPolicy<ScalarDepth> >(globPtr<SrcType>(src), globPtr<DstType>(dst), op, stream);
        }
        else
        {
            DivScalarOp<SrcType, ScalarType, DstType> op;
            op.val = VecTraits<ScalarType>::make(value_.val);

            gridTransformUnary_< TransformPolicy<ScalarDepth> >(globPtr<SrcType>(src), globPtr<DstType>(dst), op, stream);
        }
    }
}

void divScalar(const GpuMat& src, cv::Scalar val, bool inv, GpuMat& dst, const GpuMat&, double scale, Stream& stream, int)
{
    typedef void (*func_t)(const GpuMat& src, cv::Scalar val, bool inv, GpuMat& dst, Stream& stream);
    static const func_t funcs[7][7][4] =
    {
        {
            {divScalarImpl<uchar, float, uchar>, divScalarImpl<uchar2, float, uchar2>, divScalarImpl<uchar3, float, uchar3>, divScalarImpl<uchar4, float, uchar4>},
            {divScalarImpl<uchar, float, schar>, divScalarImpl<uchar2, float, char2>, divScalarImpl<uchar3, float, char3>, divScalarImpl<uchar4, float, char4>},
            {divScalarImpl<uchar, float, ushort>, divScalarImpl<uchar2, float, ushort2>, divScalarImpl<uchar3, float, ushort3>, divScalarImpl<uchar4, float, ushort4>},
            {divScalarImpl<uchar, float, short>, divScalarImpl<uchar2, float, short2>, divScalarImpl<uchar3, float, short3>, divScalarImpl<uchar4, float, short4>},
            {divScalarImpl<uchar, float, int>, divScalarImpl<uchar2, float, int2>, divScalarImpl<uchar3, float, int3>, divScalarImpl<uchar4, float, int4>},
            {divScalarImpl<uchar, float, float>, divScalarImpl<uchar2, float, float2>, divScalarImpl<uchar3, float, float3>, divScalarImpl<uchar4, float, float4>},
            {divScalarImpl<uchar, double, double>, divScalarImpl<uchar2, double, double2>, divScalarImpl<uchar3, double, double3>, divScalarImpl<uchar4, double, double4>}
        },
        {
            {divScalarImpl<schar, float, uchar>, divScalarImpl<char2, float, uchar2>, divScalarImpl<char3, float, uchar3>, divScalarImpl<char4, float, uchar4>},
            {divScalarImpl<schar, float, schar>, divScalarImpl<char2, float, char2>, divScalarImpl<char3, float, char3>, divScalarImpl<char4, float, char4>},
            {divScalarImpl<schar, float, ushort>, divScalarImpl<char2, float, ushort2>, divScalarImpl<char3, float, ushort3>, divScalarImpl<char4, float, ushort4>},
            {divScalarImpl<schar, float, short>, divScalarImpl<char2, float, short2>, divScalarImpl<char3, float, short3>, divScalarImpl<char4, float, short4>},
            {divScalarImpl<schar, float, int>, divScalarImpl<char2, float, int2>, divScalarImpl<char3, float, int3>, divScalarImpl<char4, float, int4>},
            {divScalarImpl<schar, float, float>, divScalarImpl<char2, float, float2>, divScalarImpl<char3, float, float3>, divScalarImpl<char4, float, float4>},
            {divScalarImpl<schar, double, double>, divScalarImpl<char2, double, double2>, divScalarImpl<char3, double, double3>, divScalarImpl<char4, double, double4>}
        },
        {
            {0 /*divScalarImpl<ushort, float, uchar>*/, 0 /*divScalarImpl<ushort2, float, uchar2>*/, 0 /*divScalarImpl<ushort3, float, uchar3>*/, 0 /*divScalarImpl<ushort4, float, uchar4>*/},
            {0 /*divScalarImpl<ushort, float, schar>*/, 0 /*divScalarImpl<ushort2, float, char2>*/, 0 /*divScalarImpl<ushort3, float, char3>*/, 0 /*divScalarImpl<ushort4, float, char4>*/},
            {divScalarImpl<ushort, float, ushort>, divScalarImpl<ushort2, float, ushort2>, divScalarImpl<ushort3, float, ushort3>, divScalarImpl<ushort4, float, ushort4>},
            {divScalarImpl<ushort, float, short>, divScalarImpl<ushort2, float, short2>, divScalarImpl<ushort3, float, short3>, divScalarImpl<ushort4, float, short4>},
            {divScalarImpl<ushort, float, int>, divScalarImpl<ushort2, float, int2>, divScalarImpl<ushort3, float, int3>, divScalarImpl<ushort4, float, int4>},
            {divScalarImpl<ushort, float, float>, divScalarImpl<ushort2, float, float2>, divScalarImpl<ushort3, float, float3>, divScalarImpl<ushort4, float, float4>},
            {divScalarImpl<ushort, double, double>, divScalarImpl<ushort2, double, double2>, divScalarImpl<ushort3, double, double3>, divScalarImpl<ushort4, double, double4>}
        },
        {
            {0 /*divScalarImpl<short, float, uchar>*/, 0 /*divScalarImpl<short2, float, uchar2>*/, 0 /*divScalarImpl<short3, float, uchar3>*/, 0 /*divScalarImpl<short4, float, uchar4>*/},
            {0 /*divScalarImpl<short, float, schar>*/, 0 /*divScalarImpl<short2, float, char2>*/, 0 /*divScalarImpl<short3, float, char3>*/, 0 /*divScalarImpl<short4, float, char4>*/},
            {divScalarImpl<short, float, ushort>, divScalarImpl<short2, float, ushort2>, divScalarImpl<short3, float, ushort3>, divScalarImpl<short4, float, ushort4>},
            {divScalarImpl<short, float, short>, divScalarImpl<short2, float, short2>, divScalarImpl<short3, float, short3>, divScalarImpl<short4, float, short4>},
            {divScalarImpl<short, float, int>, divScalarImpl<short2, float, int2>, divScalarImpl<short3, float, int3>, divScalarImpl<short4, float, int4>},
            {divScalarImpl<short, float, float>, divScalarImpl<short2, float, float2>, divScalarImpl<short3, float, float3>, divScalarImpl<short4, float, float4>},
            {divScalarImpl<short, double, double>, divScalarImpl<short2, double, double2>, divScalarImpl<short3, double, double3>, divScalarImpl<short4, double, double4>}
        },
        {
            {0 /*divScalarImpl<int, float, uchar>*/, 0 /*divScalarImpl<int2, float, uchar2>*/, 0 /*divScalarImpl<int3, float, uchar3>*/, 0 /*divScalarImpl<int4, float, uchar4>*/},
            {0 /*divScalarImpl<int, float, schar>*/, 0 /*divScalarImpl<int2, float, char2>*/, 0 /*divScalarImpl<int3, float, char3>*/, 0 /*divScalarImpl<int4, float, char4>*/},
            {0 /*divScalarImpl<int, float, ushort>*/, 0 /*divScalarImpl<int2, float, ushort2>*/, 0 /*divScalarImpl<int3, float, ushort3>*/, 0 /*divScalarImpl<int4, float, ushort4>*/},
            {0 /*divScalarImpl<int, float, short>*/, 0 /*divScalarImpl<int2, float, short2>*/, 0 /*divScalarImpl<int3, float, short3>*/, 0 /*divScalarImpl<int4, float, short4>*/},
            {divScalarImpl<int, float, int>, divScalarImpl<int2, float, int2>, divScalarImpl<int3, float, int3>, divScalarImpl<int4, float, int4>},
            {divScalarImpl<int, float, float>, divScalarImpl<int2, float, float2>, divScalarImpl<int3, float, float3>, divScalarImpl<int4, float, float4>},
            {divScalarImpl<int, double, double>, divScalarImpl<int2, double, double2>, divScalarImpl<int3, double, double3>, divScalarImpl<int4, double, double4>}
        },
        {
            {0 /*divScalarImpl<float, float, uchar>*/, 0 /*divScalarImpl<float2, float, uchar2>*/, 0 /*divScalarImpl<float3, float, uchar3>*/, 0 /*divScalarImpl<float4, float, uchar4>*/},
            {0 /*divScalarImpl<float, float, schar>*/, 0 /*divScalarImpl<float2, float, char2>*/, 0 /*divScalarImpl<float3, float, char3>*/, 0 /*divScalarImpl<float4, float, char4>*/},
            {0 /*divScalarImpl<float, float, ushort>*/, 0 /*divScalarImpl<float2, float, ushort2>*/, 0 /*divScalarImpl<float3, float, ushort3>*/, 0 /*divScalarImpl<float4, float, ushort4>*/},
            {0 /*divScalarImpl<float, float, short>*/, 0 /*divScalarImpl<float2, float, short2>*/, 0 /*divScalarImpl<float3, float, short3>*/, 0 /*divScalarImpl<float4, float, short4>*/},
            {0 /*divScalarImpl<float, float, int>*/, 0 /*divScalarImpl<float2, float, int2>*/, 0 /*divScalarImpl<float3, float, int3>*/, 0 /*divScalarImpl<float4, float, int4>*/},
            {divScalarImpl<float, float, float>, divScalarImpl<float2, float, float2>, divScalarImpl<float3, float, float3>, divScalarImpl<float4, float, float4>},
            {divScalarImpl<float, double, double>, divScalarImpl<float2, double, double2>, divScalarImpl<float3, double, double3>, divScalarImpl<float4, double, double4>}
        },
        {
            {0 /*divScalarImpl<double, double, uchar>*/, 0 /*divScalarImpl<double2, double, uchar2>*/, 0 /*divScalarImpl<double3, double, uchar3>*/, 0 /*divScalarImpl<double4, double, uchar4>*/},
            {0 /*divScalarImpl<double, double, schar>*/, 0 /*divScalarImpl<double2, double, char2>*/, 0 /*divScalarImpl<double3, double, char3>*/, 0 /*divScalarImpl<double4, double, char4>*/},
            {0 /*divScalarImpl<double, double, ushort>*/, 0 /*divScalarImpl<double2, double, ushort2>*/, 0 /*divScalarImpl<double3, double, ushort3>*/, 0 /*divScalarImpl<double4, double, ushort4>*/},
            {0 /*divScalarImpl<double, double, short>*/, 0 /*divScalarImpl<double2, double, short2>*/, 0 /*divScalarImpl<double3, double, short3>*/, 0 /*divScalarImpl<double4, double, short4>*/},
            {0 /*divScalarImpl<double, double, int>*/, 0 /*divScalarImpl<double2, double, int2>*/, 0 /*divScalarImpl<double3, double, int3>*/, 0 /*divScalarImpl<double4, double, int4>*/},
            {0 /*divScalarImpl<double, double, float>*/, 0 /*divScalarImpl<double2, double, float2>*/, 0 /*divScalarImpl<double3, double, float3>*/, 0 /*divScalarImpl<double4, double, float4>*/},
            {divScalarImpl<double, double, double>, divScalarImpl<double2, double, double2>, divScalarImpl<double3, double, double3>, divScalarImpl<double4, double, double4>}
        }
    };

    const int sdepth = src.depth();
    const int ddepth = dst.depth();
    const int cn = src.channels();

    CV_DbgAssert( sdepth <= CV_64F && ddepth <= CV_64F && cn <= 4 );

    if (inv)
    {
        val[0] *= scale;
        val[1] *= scale;
        val[2] *= scale;
        val[3] *= scale;
    }
    else
    {
        val[0] /= scale;
        val[1] /= scale;
        val[2] /= scale;
        val[3] /= scale;
    }

    const func_t func = funcs[sdepth][ddepth][cn - 1];

    if (!func)
        CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported combination of source and destination types");

    func(src, val, inv, dst, stream);
}

#endif
