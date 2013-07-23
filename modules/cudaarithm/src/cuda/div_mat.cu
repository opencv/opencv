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

#include "arithm_func_traits.hpp"

using namespace cv::cuda;
using namespace cv::cuda::device;

namespace arithm
{
    struct Div_8uc4_32f : binary_function<uint, float, uint>
    {
        __device__ __forceinline__ uint operator ()(uint a, float b) const
        {
            uint res = 0;

            if (b != 0)
            {
                b = 1.0f / b;
                res |= (saturate_cast<uchar>((0xffu & (a      )) * b)      );
                res |= (saturate_cast<uchar>((0xffu & (a >>  8)) * b) <<  8);
                res |= (saturate_cast<uchar>((0xffu & (a >> 16)) * b) << 16);
                res |= (saturate_cast<uchar>((0xffu & (a >> 24)) * b) << 24);
            }

            return res;
        }
    };

    struct Div_16sc4_32f : binary_function<short4, float, short4>
    {
        __device__ __forceinline__ short4 operator ()(short4 a, float b) const
        {
            return b != 0 ? make_short4(saturate_cast<short>(a.x / b), saturate_cast<short>(a.y / b),
                                        saturate_cast<short>(a.z / b), saturate_cast<short>(a.w / b))
                          : make_short4(0,0,0,0);
        }
    };

    template <typename T, typename D> struct Div : binary_function<T, T, D>
    {
        __device__ __forceinline__ D operator ()(T a, T b) const
        {
            return b != 0 ? saturate_cast<D>(a / b) : 0;
        }

        __host__ __device__ __forceinline__ Div() {}
        __host__ __device__ __forceinline__ Div(const Div&) {}
    };
    template <typename T> struct Div<T, float> : binary_function<T, T, float>
    {
        __device__ __forceinline__ float operator ()(T a, T b) const
        {
            return b != 0 ? static_cast<float>(a) / b : 0;
        }

        __host__ __device__ __forceinline__ Div() {}
        __host__ __device__ __forceinline__ Div(const Div&) {}
    };
    template <typename T> struct Div<T, double> : binary_function<T, T, double>
    {
        __device__ __forceinline__ double operator ()(T a, T b) const
        {
            return b != 0 ? static_cast<double>(a) / b : 0;
        }

        __host__ __device__ __forceinline__ Div() {}
        __host__ __device__ __forceinline__ Div(const Div&) {}
    };

    template <typename T, typename S, typename D> struct DivScale : binary_function<T, T, D>
    {
        S scale;

        __host__ explicit DivScale(S scale_) : scale(scale_) {}

        __device__ __forceinline__ D operator ()(T a, T b) const
        {
            return b != 0 ? saturate_cast<D>(scale * a / b) : 0;
        }
    };
}

namespace cv { namespace cuda { namespace device
{
    template <> struct TransformFunctorTraits<arithm::Div_8uc4_32f> : arithm::ArithmFuncTraits<sizeof(uint), sizeof(uint)>
    {
    };

    template <typename T, typename D> struct TransformFunctorTraits< arithm::Div<T, D> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(D)>
    {
    };

    template <typename T, typename S, typename D> struct TransformFunctorTraits< arithm::DivScale<T, S, D> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(D)>
    {
    };
}}}

namespace arithm
{
    void divMat_8uc4_32f(PtrStepSz<uint> src1, PtrStepSzf src2, PtrStepSz<uint> dst, cudaStream_t stream)
    {
        device::transform(src1, src2, dst, Div_8uc4_32f(), WithOutMask(), stream);
    }

    void divMat_16sc4_32f(PtrStepSz<short4> src1, PtrStepSzf src2, PtrStepSz<short4> dst, cudaStream_t stream)
    {
        device::transform(src1, src2, dst, Div_16sc4_32f(), WithOutMask(), stream);
    }

    template <typename T, typename S, typename D>
    void divMat(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream)
    {
        if (scale == 1)
        {
            Div<T, D> op;
            device::transform((PtrStepSz<T>) src1, (PtrStepSz<T>) src2, (PtrStepSz<D>) dst, op, WithOutMask(), stream);
        }
        else
        {
            DivScale<T, S, D> op(static_cast<S>(scale));
            device::transform((PtrStepSz<T>) src1, (PtrStepSz<T>) src2, (PtrStepSz<D>) dst, op, WithOutMask(), stream);
        }
    }

    template void divMat<uchar, float, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<uchar, float, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<uchar, float, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<uchar, float, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<uchar, float, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<uchar, float, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<uchar, double, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);

    template void divMat<schar, float, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<schar, float, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<schar, float, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<schar, float, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<schar, float, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<schar, float, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<schar, double, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);

    //template void divMat<ushort, float, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void divMat<ushort, float, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<ushort, float, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<ushort, float, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<ushort, float, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<ushort, float, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<ushort, double, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);

    //template void divMat<short, float, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void divMat<short, float, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<short, float, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<short, float, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<short, float, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<short, float, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<short, double, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);

    //template void divMat<int, float, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void divMat<int, float, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void divMat<int, float, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void divMat<int, float, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<int, float, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<int, float, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<int, double, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);

    //template void divMat<float, float, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void divMat<float, float, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void divMat<float, float, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void divMat<float, float, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void divMat<float, float, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<float, float, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<float, double, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);

    //template void divMat<double, double, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void divMat<double, double, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void divMat<double, double, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void divMat<double, double, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void divMat<double, double, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void divMat<double, double, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<double, double, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
}

#endif // CUDA_DISABLER
