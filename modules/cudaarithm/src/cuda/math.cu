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
    template <typename ScalarDepth> struct TransformPolicy : DefaultTransformPolicy
    {
    };
    template <> struct TransformPolicy<double> : DefaultTransformPolicy
    {
        enum {
            shift = 1
        };
    };
}

//////////////////////////////////////////////////////////////////////////////
/// abs

namespace
{
    template <typename T>
    void absMat(const GpuMat& src, const GpuMat& dst, Stream& stream)
    {
        gridTransformUnary_< TransformPolicy<T> >(globPtr<T>(src), globPtr<T>(dst), abs_func<T>(), stream);
    }
}

void cv::cuda::abs(InputArray _src, OutputArray _dst, Stream& stream)
{
    typedef void (*func_t)(const GpuMat& src, const GpuMat& dst, Stream& stream);
    static const func_t funcs[] =
    {
        absMat<uchar>,
        absMat<schar>,
        absMat<ushort>,
        absMat<short>,
        absMat<int>,
        absMat<float>,
        absMat<double>
    };

    GpuMat src = getInputMat(_src, stream);

    CV_Assert( src.depth() <= CV_64F );

    GpuMat dst = getOutputMat(_dst, src.size(), src.type(), stream);

    funcs[src.depth()](src.reshape(1), dst.reshape(1), stream);

    syncOutput(dst, _dst, stream);
}

//////////////////////////////////////////////////////////////////////////////
/// sqr

namespace
{
    template <typename T> struct SqrOp : unary_function<T, T>
    {
        __device__ __forceinline__ T operator ()(T x) const
        {
            return cudev::saturate_cast<T>(x * x);
        }
    };

    template <typename T>
    void sqrMat(const GpuMat& src, const GpuMat& dst, Stream& stream)
    {
        gridTransformUnary_< TransformPolicy<T> >(globPtr<T>(src), globPtr<T>(dst), SqrOp<T>(), stream);
    }
}

void cv::cuda::sqr(InputArray _src, OutputArray _dst, Stream& stream)
{
    typedef void (*func_t)(const GpuMat& src, const GpuMat& dst, Stream& stream);
    static const func_t funcs[] =
    {
        sqrMat<uchar>,
        sqrMat<schar>,
        sqrMat<ushort>,
        sqrMat<short>,
        sqrMat<int>,
        sqrMat<float>,
        sqrMat<double>
    };

    GpuMat src = getInputMat(_src, stream);

    CV_Assert( src.depth() <= CV_64F );

    GpuMat dst = getOutputMat(_dst, src.size(), src.type(), stream);

    funcs[src.depth()](src.reshape(1), dst.reshape(1), stream);

    syncOutput(dst, _dst, stream);
}

//////////////////////////////////////////////////////////////////////////////
/// sqrt

namespace
{
    template <typename T>
    void sqrtMat(const GpuMat& src, const GpuMat& dst, Stream& stream)
    {
        gridTransformUnary_< TransformPolicy<T> >(globPtr<T>(src), globPtr<T>(dst), sqrt_func<T>(), stream);
    }
}

void cv::cuda::sqrt(InputArray _src, OutputArray _dst, Stream& stream)
{
    typedef void (*func_t)(const GpuMat& src, const GpuMat& dst, Stream& stream);
    static const func_t funcs[] =
    {
        sqrtMat<uchar>,
        sqrtMat<schar>,
        sqrtMat<ushort>,
        sqrtMat<short>,
        sqrtMat<int>,
        sqrtMat<float>,
        sqrtMat<double>
    };

    GpuMat src = getInputMat(_src, stream);

    CV_Assert( src.depth() <= CV_64F );

    GpuMat dst = getOutputMat(_dst, src.size(), src.type(), stream);

    funcs[src.depth()](src.reshape(1), dst.reshape(1), stream);

    syncOutput(dst, _dst, stream);
}

////////////////////////////////////////////////////////////////////////
/// exp

namespace
{
    template <typename T> struct ExpOp : unary_function<T, T>
    {
        __device__ __forceinline__ T operator ()(T x) const
        {
            exp_func<T> f;
            return cudev::saturate_cast<T>(f(x));
        }
    };

    template <typename T>
    void expMat(const GpuMat& src, const GpuMat& dst, Stream& stream)
    {
        gridTransformUnary_< TransformPolicy<T> >(globPtr<T>(src), globPtr<T>(dst), ExpOp<T>(), stream);
    }
}

void cv::cuda::exp(InputArray _src, OutputArray _dst, Stream& stream)
{
    typedef void (*func_t)(const GpuMat& src, const GpuMat& dst, Stream& stream);
    static const func_t funcs[] =
    {
        expMat<uchar>,
        expMat<schar>,
        expMat<ushort>,
        expMat<short>,
        expMat<int>,
        expMat<float>,
        expMat<double>
    };

    GpuMat src = getInputMat(_src, stream);

    CV_Assert( src.depth() <= CV_64F );

    GpuMat dst = getOutputMat(_dst, src.size(), src.type(), stream);

    funcs[src.depth()](src.reshape(1), dst.reshape(1), stream);

    syncOutput(dst, _dst, stream);
}

////////////////////////////////////////////////////////////////////////
// log

namespace
{
    template <typename T>
    void logMat(const GpuMat& src, const GpuMat& dst, Stream& stream)
    {
        gridTransformUnary_< TransformPolicy<T> >(globPtr<T>(src), globPtr<T>(dst), log_func<T>(), stream);
    }
}

void cv::cuda::log(InputArray _src, OutputArray _dst, Stream& stream)
{
    typedef void (*func_t)(const GpuMat& src, const GpuMat& dst, Stream& stream);
    static const func_t funcs[] =
    {
        logMat<uchar>,
        logMat<schar>,
        logMat<ushort>,
        logMat<short>,
        logMat<int>,
        logMat<float>,
        logMat<double>
    };

    GpuMat src = getInputMat(_src, stream);

    CV_Assert( src.depth() <= CV_64F );

    GpuMat dst = getOutputMat(_dst, src.size(), src.type(), stream);

    funcs[src.depth()](src.reshape(1), dst.reshape(1), stream);

    syncOutput(dst, _dst, stream);
}

////////////////////////////////////////////////////////////////////////
// pow

namespace
{
    template<typename T, bool Signed = numeric_limits<T>::is_signed> struct PowOp : unary_function<T, T>
    {
        float power;

        __device__ __forceinline__ T operator()(T e) const
        {
            return cudev::saturate_cast<T>(__powf((float)e, power));
        }
    };
    template<typename T> struct PowOp<T, true> : unary_function<T, T>
    {
        float power;

        __device__ __forceinline__ T operator()(T e) const
        {
            T res = cudev::saturate_cast<T>(__powf((float)e, power));

            if ((e < 0) && (1 & static_cast<int>(power)))
                res *= -1;

            return res;
        }
    };
    template<> struct PowOp<float> : unary_function<float, float>
    {
        float power;

        __device__ __forceinline__ float operator()(float e) const
        {
            return __powf(::fabs(e), power);
        }
    };
    template<> struct PowOp<double> : unary_function<double, double>
    {
        double power;

        __device__ __forceinline__ double operator()(double e) const
        {
            return ::pow(::fabs(e), power);
        }
    };

    template<typename T>
    void powMat(const GpuMat& src, double power, const GpuMat& dst, Stream& stream)
    {
        PowOp<T> op;
        op.power = static_cast<typename LargerType<T, float>::type>(power);

        gridTransformUnary_< TransformPolicy<T> >(globPtr<T>(src), globPtr<T>(dst), op, stream);
    }
}

void cv::cuda::pow(InputArray _src, double power, OutputArray _dst, Stream& stream)
{
    typedef void (*func_t)(const GpuMat& src, double power, const GpuMat& dst, Stream& stream);
    static const func_t funcs[] =
    {
        powMat<uchar>,
        powMat<schar>,
        powMat<ushort>,
        powMat<short>,
        powMat<int>,
        powMat<float>,
        powMat<double>
    };

    GpuMat src = getInputMat(_src, stream);

    CV_Assert( src.depth() <= CV_64F );

    GpuMat dst = getOutputMat(_dst, src.size(), src.type(), stream);

    funcs[src.depth()](src.reshape(1), power, dst.reshape(1), stream);

    syncOutput(dst, _dst, stream);
}

#endif
