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

#if !defined CUDA_DISABLER

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/functional.hpp"
#include "opencv2/core/cuda/transform.hpp"
#include "opencv2/core/cuda/saturate_cast.hpp"
#include "opencv2/core/cuda/simd_functions.hpp"
#include "opencv2/core/cuda/limits.hpp"
#include "opencv2/core/cuda/type_traits.hpp"

#include "arithm_func_traits.hpp"

using namespace cv::cuda;
using namespace cv::cuda::device;

//////////////////////////////////////////////////////////////////////////
// absMat

namespace cv { namespace cuda { namespace device
{
    template <typename T> struct TransformFunctorTraits< abs_func<T> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(T)>
    {
    };
}}}

namespace arithm
{
    template <typename T>
    void absMat(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream)
    {
        device::transform((PtrStepSz<T>) src, (PtrStepSz<T>) dst, abs_func<T>(), WithOutMask(), stream);
    }

    template void absMat<uchar>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void absMat<schar>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void absMat<ushort>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void absMat<short>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void absMat<int>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void absMat<float>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void absMat<double>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
}

//////////////////////////////////////////////////////////////////////////
// sqrMat

namespace arithm
{
    template <typename T> struct Sqr : unary_function<T, T>
    {
        __device__ __forceinline__ T operator ()(T x) const
        {
            return saturate_cast<T>(x * x);
        }

        __host__ __device__ __forceinline__ Sqr() {}
        __host__ __device__ __forceinline__ Sqr(const Sqr&) {}
    };
}

namespace cv { namespace cuda { namespace device
{
    template <typename T> struct TransformFunctorTraits< arithm::Sqr<T> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(T)>
    {
    };
}}}

namespace arithm
{
    template <typename T>
    void sqrMat(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream)
    {
        device::transform((PtrStepSz<T>) src, (PtrStepSz<T>) dst, Sqr<T>(), WithOutMask(), stream);
    }

    template void sqrMat<uchar>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void sqrMat<schar>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void sqrMat<ushort>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void sqrMat<short>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void sqrMat<int>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void sqrMat<float>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void sqrMat<double>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
}

//////////////////////////////////////////////////////////////////////////
// sqrtMat

namespace cv { namespace cuda { namespace device
{
    template <typename T> struct TransformFunctorTraits< sqrt_func<T> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(T)>
    {
    };
}}}

namespace arithm
{
    template <typename T>
    void sqrtMat(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream)
    {
        device::transform((PtrStepSz<T>) src, (PtrStepSz<T>) dst, sqrt_func<T>(), WithOutMask(), stream);
    }

    template void sqrtMat<uchar>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void sqrtMat<schar>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void sqrtMat<ushort>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void sqrtMat<short>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void sqrtMat<int>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void sqrtMat<float>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void sqrtMat<double>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
}

//////////////////////////////////////////////////////////////////////////
// logMat

namespace cv { namespace cuda { namespace device
{
    template <typename T> struct TransformFunctorTraits< log_func<T> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(T)>
    {
    };
}}}

namespace arithm
{
    template <typename T>
    void logMat(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream)
    {
        device::transform((PtrStepSz<T>) src, (PtrStepSz<T>) dst, log_func<T>(), WithOutMask(), stream);
    }

    template void logMat<uchar>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void logMat<schar>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void logMat<ushort>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void logMat<short>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void logMat<int>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void logMat<float>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void logMat<double>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
}

//////////////////////////////////////////////////////////////////////////
// expMat

namespace arithm
{
    template <typename T> struct Exp : unary_function<T, T>
    {
        __device__ __forceinline__ T operator ()(T x) const
        {
            exp_func<T> f;
            return saturate_cast<T>(f(x));
        }

        __host__ __device__ __forceinline__ Exp() {}
        __host__ __device__ __forceinline__ Exp(const Exp&) {}
    };
}

namespace cv { namespace cuda { namespace device
{
    template <typename T> struct TransformFunctorTraits< arithm::Exp<T> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(T)>
    {
    };
}}}

namespace arithm
{
    template <typename T>
    void expMat(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream)
    {
        device::transform((PtrStepSz<T>) src, (PtrStepSz<T>) dst, Exp<T>(), WithOutMask(), stream);
    }

    template void expMat<uchar>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void expMat<schar>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void expMat<ushort>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void expMat<short>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void expMat<int>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void expMat<float>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void expMat<double>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
}

//////////////////////////////////////////////////////////////////////////
// pow

namespace arithm
{
    template<typename T, bool Signed = numeric_limits<T>::is_signed> struct PowOp : unary_function<T, T>
    {
        float power;

        __host__ explicit PowOp(double power_) : power(static_cast<float>(power_)) {}

        __device__ __forceinline__ T operator()(T e) const
        {
            return saturate_cast<T>(__powf((float)e, power));
        }
    };
    template<typename T> struct PowOp<T, true> : unary_function<T, T>
    {
        float power;

        __host__ explicit PowOp(double power_) : power(static_cast<float>(power_)) {}

        __device__ __forceinline__ T operator()(T e) const
        {
            T res = saturate_cast<T>(__powf((float)e, power));

            if ((e < 0) && (1 & static_cast<int>(power)))
                res *= -1;

            return res;
        }
    };
    template<> struct PowOp<float> : unary_function<float, float>
    {
        float power;

        __host__ explicit PowOp(double power_) : power(static_cast<float>(power_)) {}

        __device__ __forceinline__ float operator()(float e) const
        {
            return __powf(::fabs(e), power);
        }
    };
    template<> struct PowOp<double> : unary_function<double, double>
    {
        double power;

        __host__ explicit PowOp(double power_) : power(power_) {}

        __device__ __forceinline__ double operator()(double e) const
        {
            return ::pow(::fabs(e), power);
        }
    };
}

namespace cv { namespace cuda { namespace device
{
    template <typename T> struct TransformFunctorTraits< arithm::PowOp<T> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(T)>
    {
    };
}}}

namespace arithm
{
    template<typename T>
    void pow(PtrStepSzb src, double power, PtrStepSzb dst, cudaStream_t stream)
    {
        device::transform((PtrStepSz<T>) src, (PtrStepSz<T>) dst, PowOp<T>(power), WithOutMask(), stream);
    }

    template void pow<uchar>(PtrStepSzb src, double power, PtrStepSzb dst, cudaStream_t stream);
    template void pow<schar>(PtrStepSzb src, double power, PtrStepSzb dst, cudaStream_t stream);
    template void pow<short>(PtrStepSzb src, double power, PtrStepSzb dst, cudaStream_t stream);
    template void pow<ushort>(PtrStepSzb src, double power, PtrStepSzb dst, cudaStream_t stream);
    template void pow<int>(PtrStepSzb src, double power, PtrStepSzb dst, cudaStream_t stream);
    template void pow<float>(PtrStepSzb src, double power, PtrStepSzb dst, cudaStream_t stream);
    template void pow<double>(PtrStepSzb src, double power, PtrStepSzb dst, cudaStream_t stream);
}

#endif // CUDA_DISABLER
