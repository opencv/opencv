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
    struct VAbsDiff4 : binary_function<uint, uint, uint>
    {
        __device__ __forceinline__ uint operator ()(uint a, uint b) const
        {
            return vabsdiff4(a, b);
        }

        __host__ __device__ __forceinline__ VAbsDiff4() {}
        __host__ __device__ __forceinline__ VAbsDiff4(const VAbsDiff4&) {}
    };

    struct VAbsDiff2 : binary_function<uint, uint, uint>
    {
        __device__ __forceinline__ uint operator ()(uint a, uint b) const
        {
            return vabsdiff2(a, b);
        }

        __host__ __device__ __forceinline__ VAbsDiff2() {}
        __host__ __device__ __forceinline__ VAbsDiff2(const VAbsDiff2&) {}
    };

    __device__ __forceinline__ int _abs(int a)
    {
        return ::abs(a);
    }
    __device__ __forceinline__ float _abs(float a)
    {
        return ::fabsf(a);
    }
    __device__ __forceinline__ double _abs(double a)
    {
        return ::fabs(a);
    }

    template <typename T> struct AbsDiffMat : binary_function<T, T, T>
    {
        __device__ __forceinline__ T operator ()(T a, T b) const
        {
            return saturate_cast<T>(_abs(a - b));
        }

        __host__ __device__ __forceinline__ AbsDiffMat() {}
        __host__ __device__ __forceinline__ AbsDiffMat(const AbsDiffMat&) {}
    };
}

namespace cv { namespace cuda { namespace device
{
    template <> struct TransformFunctorTraits< arithm::VAbsDiff4 > : arithm::ArithmFuncTraits<sizeof(uint), sizeof(uint)>
    {
    };

    template <> struct TransformFunctorTraits< arithm::VAbsDiff2 > : arithm::ArithmFuncTraits<sizeof(uint), sizeof(uint)>
    {
    };

    template <typename T> struct TransformFunctorTraits< arithm::AbsDiffMat<T> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(T)>
    {
    };
}}}

namespace arithm
{
    void absDiffMat_v4(PtrStepSz<uint> src1, PtrStepSz<uint> src2, PtrStepSz<uint> dst, cudaStream_t stream)
    {
        device::transform(src1, src2, dst, VAbsDiff4(), WithOutMask(), stream);
    }

    void absDiffMat_v2(PtrStepSz<uint> src1, PtrStepSz<uint> src2, PtrStepSz<uint> dst, cudaStream_t stream)
    {
        device::transform(src1, src2, dst, VAbsDiff2(), WithOutMask(), stream);
    }

    template <typename T>
    void absDiffMat(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream)
    {
        device::transform((PtrStepSz<T>) src1, (PtrStepSz<T>) src2, (PtrStepSz<T>) dst, AbsDiffMat<T>(), WithOutMask(), stream);
    }

    template void absDiffMat<uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void absDiffMat<schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void absDiffMat<ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void absDiffMat<short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void absDiffMat<int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void absDiffMat<float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void absDiffMat<double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
}

#endif // CUDA_DISABLER
