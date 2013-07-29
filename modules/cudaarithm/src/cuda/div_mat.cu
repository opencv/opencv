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

void divMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat&, double scale, Stream& stream, int);
void divMat_8uc4_32f(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream);
void divMat_16sc4_32f(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream);

namespace
{
    template <typename T, typename D> struct DivOp : binary_function<T, T, D>
    {
        __device__ __forceinline__ D operator ()(T a, T b) const
        {
            return b != 0 ? saturate_cast<D>(a / b) : 0;
        }
    };
    template <typename T> struct DivOp<T, float> : binary_function<T, T, float>
    {
        __device__ __forceinline__ float operator ()(T a, T b) const
        {
            return b != 0 ? static_cast<float>(a) / b : 0.0f;
        }
    };
    template <typename T> struct DivOp<T, double> : binary_function<T, T, double>
    {
        __device__ __forceinline__ double operator ()(T a, T b) const
        {
            return b != 0 ? static_cast<double>(a) / b : 0.0;
        }
    };

    template <typename T, typename S, typename D> struct DivScaleOp : binary_function<T, T, D>
    {
        S scale;

        __device__ __forceinline__ D operator ()(T a, T b) const
        {
            return b != 0 ? saturate_cast<D>(scale * a / b) : 0;
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

    template <typename T, typename S, typename D>
    void divMatImpl(const GpuMat& src1, const GpuMat& src2, const GpuMat& dst, double scale, Stream& stream)
    {
        if (scale == 1)
        {
            DivOp<T, D> op;
            gridTransformBinary_< TransformPolicy<S> >(globPtr<T>(src1), globPtr<T>(src2), globPtr<D>(dst), op, stream);
        }
        else
        {
            DivScaleOp<T, S, D> op;
            op.scale = static_cast<S>(scale);
            gridTransformBinary_< TransformPolicy<S> >(globPtr<T>(src1), globPtr<T>(src2), globPtr<D>(dst), op, stream);
        }
    }
}

void divMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat&, double scale, Stream& stream, int)
{
    typedef void (*func_t)(const GpuMat& src1, const GpuMat& src2, const GpuMat& dst, double scale, Stream& stream);
    static const func_t funcs[7][7] =
    {
        {
            divMatImpl<uchar, float, uchar>,
            divMatImpl<uchar, float, schar>,
            divMatImpl<uchar, float, ushort>,
            divMatImpl<uchar, float, short>,
            divMatImpl<uchar, float, int>,
            divMatImpl<uchar, float, float>,
            divMatImpl<uchar, double, double>
        },
        {
            divMatImpl<schar, float, uchar>,
            divMatImpl<schar, float, schar>,
            divMatImpl<schar, float, ushort>,
            divMatImpl<schar, float, short>,
            divMatImpl<schar, float, int>,
            divMatImpl<schar, float, float>,
            divMatImpl<schar, double, double>
        },
        {
            0 /*divMatImpl<ushort, float, uchar>*/,
            0 /*divMatImpl<ushort, float, schar>*/,
            divMatImpl<ushort, float, ushort>,
            divMatImpl<ushort, float, short>,
            divMatImpl<ushort, float, int>,
            divMatImpl<ushort, float, float>,
            divMatImpl<ushort, double, double>
        },
        {
            0 /*divMatImpl<short, float, uchar>*/,
            0 /*divMatImpl<short, float, schar>*/,
            divMatImpl<short, float, ushort>,
            divMatImpl<short, float, short>,
            divMatImpl<short, float, int>,
            divMatImpl<short, float, float>,
            divMatImpl<short, double, double>
        },
        {
            0 /*divMatImpl<int, float, uchar>*/,
            0 /*divMatImpl<int, float, schar>*/,
            0 /*divMatImpl<int, float, ushort>*/,
            0 /*divMatImpl<int, float, short>*/,
            divMatImpl<int, float, int>,
            divMatImpl<int, float, float>,
            divMatImpl<int, double, double>
        },
        {
            0 /*divMatImpl<float, float, uchar>*/,
            0 /*divMatImpl<float, float, schar>*/,
            0 /*divMatImpl<float, float, ushort>*/,
            0 /*divMatImpl<float, float, short>*/,
            0 /*divMatImpl<float, float, int>*/,
            divMatImpl<float, float, float>,
            divMatImpl<float, double, double>
        },
        {
            0 /*divMatImpl<double, double, uchar>*/,
            0 /*divMatImpl<double, double, schar>*/,
            0 /*divMatImpl<double, double, ushort>*/,
            0 /*divMatImpl<double, double, short>*/,
            0 /*divMatImpl<double, double, int>*/,
            0 /*divMatImpl<double, double, float>*/,
            divMatImpl<double, double, double>
        }
    };

    const int sdepth = src1.depth();
    const int ddepth = dst.depth();

    CV_DbgAssert( sdepth <= CV_64F && ddepth <= CV_64F );

    GpuMat src1_ = src1.reshape(1);
    GpuMat src2_ = src2.reshape(1);
    GpuMat dst_ = dst.reshape(1);

    const func_t func = funcs[sdepth][ddepth];

    if (!func)
        CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported combination of source and destination types");

    func(src1_, src2_, dst_, scale, stream);
}

namespace
{
    template <typename T>
    struct DivOpSpecial : binary_function<T, float, T>
    {
        __device__ __forceinline__ T operator ()(const T& a, float b) const
        {
            typedef typename VecTraits<T>::elem_type elem_type;

            T res = VecTraits<T>::all(0);

            if (b != 0)
            {
                b = 1.0f / b;
                res.x = saturate_cast<elem_type>(a.x * b);
                res.y = saturate_cast<elem_type>(a.y * b);
                res.z = saturate_cast<elem_type>(a.z * b);
                res.w = saturate_cast<elem_type>(a.w * b);
            }

            return res;
        }
    };
}

void divMat_8uc4_32f(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream)
{
    gridTransformBinary(globPtr<uchar4>(src1), globPtr<float>(src2), globPtr<uchar4>(dst), DivOpSpecial<uchar4>(), stream);
}

void divMat_16sc4_32f(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream)
{
    gridTransformBinary(globPtr<short4>(src1), globPtr<float>(src2), globPtr<short4>(dst), DivOpSpecial<short4>(), stream);
}

#endif
