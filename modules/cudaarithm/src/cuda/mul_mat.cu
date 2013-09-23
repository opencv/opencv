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
    struct Mul_8uc4_32f : binary_function<uint, float, uint>
    {
        __device__ __forceinline__ uint operator ()(uint a, float b) const
        {
            uint res = 0;

            res |= (saturate_cast<uchar>((0xffu & (a      )) * b)      );
            res |= (saturate_cast<uchar>((0xffu & (a >>  8)) * b) <<  8);
            res |= (saturate_cast<uchar>((0xffu & (a >> 16)) * b) << 16);
            res |= (saturate_cast<uchar>((0xffu & (a >> 24)) * b) << 24);

            return res;
        }

        __host__ __device__ __forceinline__ Mul_8uc4_32f() {}
        __host__ __device__ __forceinline__ Mul_8uc4_32f(const Mul_8uc4_32f&) {}
    };

    struct Mul_16sc4_32f : binary_function<short4, float, short4>
    {
        __device__ __forceinline__ short4 operator ()(short4 a, float b) const
        {
            return make_short4(saturate_cast<short>(a.x * b), saturate_cast<short>(a.y * b),
                               saturate_cast<short>(a.z * b), saturate_cast<short>(a.w * b));
        }

        __host__ __device__ __forceinline__ Mul_16sc4_32f() {}
        __host__ __device__ __forceinline__ Mul_16sc4_32f(const Mul_16sc4_32f&) {}
    };

    template <typename T, typename D> struct Mul : binary_function<T, T, D>
    {
        __device__ __forceinline__ D operator ()(T a, T b) const
        {
            return saturate_cast<D>(a * b);
        }

        __host__ __device__ __forceinline__ Mul() {}
        __host__ __device__ __forceinline__ Mul(const Mul&) {}
    };

    template <typename T, typename S, typename D> struct MulScale : binary_function<T, T, D>
    {
        S scale;

        __host__ explicit MulScale(S scale_) : scale(scale_) {}

        __device__ __forceinline__ D operator ()(T a, T b) const
        {
            return saturate_cast<D>(scale * a * b);
        }
    };
}

namespace cv { namespace cuda { namespace device
{
    template <> struct TransformFunctorTraits<arithm::Mul_8uc4_32f> : arithm::ArithmFuncTraits<sizeof(uint), sizeof(uint)>
    {
    };

    template <typename T, typename D> struct TransformFunctorTraits< arithm::Mul<T, D> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(D)>
    {
    };

    template <typename T, typename S, typename D> struct TransformFunctorTraits< arithm::MulScale<T, S, D> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(D)>
    {
    };
}}}

namespace arithm
{
    void mulMat_8uc4_32f(PtrStepSz<uint> src1, PtrStepSzf src2, PtrStepSz<uint> dst, cudaStream_t stream)
    {
        device::transform(src1, src2, dst, Mul_8uc4_32f(), WithOutMask(), stream);
    }

    void mulMat_16sc4_32f(PtrStepSz<short4> src1, PtrStepSzf src2, PtrStepSz<short4> dst, cudaStream_t stream)
    {
        device::transform(src1, src2, dst, Mul_16sc4_32f(), WithOutMask(), stream);
    }

    template <typename T, typename S, typename D>
    void mulMat(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream)
    {
        if (scale == 1)
        {
            Mul<T, D> op;
            device::transform((PtrStepSz<T>) src1, (PtrStepSz<T>) src2, (PtrStepSz<D>) dst, op, WithOutMask(), stream);
        }
        else
        {
            MulScale<T, S, D> op(static_cast<S>(scale));
            device::transform((PtrStepSz<T>) src1, (PtrStepSz<T>) src2, (PtrStepSz<D>) dst, op, WithOutMask(), stream);
        }
    }

    template void mulMat<uchar, float, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<uchar, float, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<uchar, float, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<uchar, float, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<uchar, float, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<uchar, float, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<uchar, double, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);

    template void mulMat<schar, float, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<schar, float, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<schar, float, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<schar, float, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<schar, float, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<schar, float, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<schar, double, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);

    //template void mulMat<ushort, float, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void mulMat<ushort, float, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<ushort, float, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<ushort, float, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<ushort, float, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<ushort, float, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<ushort, double, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);

    //template void mulMat<short, float, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void mulMat<short, float, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<short, float, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<short, float, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<short, float, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<short, float, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<short, double, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);

    //template void mulMat<int, float, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void mulMat<int, float, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void mulMat<int, float, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void mulMat<int, float, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<int, float, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<int, float, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<int, double, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);

    //template void mulMat<float, float, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void mulMat<float, float, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void mulMat<float, float, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void mulMat<float, float, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void mulMat<float, float, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<float, float, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<float, double, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);

    //template void mulMat<double, double, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void mulMat<double, double, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void mulMat<double, double, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void mulMat<double, double, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void mulMat<double, double, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void mulMat<double, double, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<double, double, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
}

#endif // CUDA_DISABLER
