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

//////////////////////////////////////////////////////////////////////////
// min

namespace arithm
{
    struct VMin4 : binary_function<uint, uint, uint>
    {
        __device__ __forceinline__ uint operator ()(uint a, uint b) const
        {
            return vmin4(a, b);
        }

        __host__ __device__ __forceinline__ VMin4() {}
        __host__ __device__ __forceinline__ VMin4(const VMin4&) {}
    };

    struct VMin2 : binary_function<uint, uint, uint>
    {
        __device__ __forceinline__ uint operator ()(uint a, uint b) const
        {
            return vmin2(a, b);
        }

        __host__ __device__ __forceinline__ VMin2() {}
        __host__ __device__ __forceinline__ VMin2(const VMin2&) {}
    };
}

namespace cv { namespace cuda { namespace device
{
    template <> struct TransformFunctorTraits< arithm::VMin4 > : arithm::ArithmFuncTraits<sizeof(uint), sizeof(uint)>
    {
    };

    template <> struct TransformFunctorTraits< arithm::VMin2 > : arithm::ArithmFuncTraits<sizeof(uint), sizeof(uint)>
    {
    };

    template <typename T> struct TransformFunctorTraits< minimum<T> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(T)>
    {
    };

    template <typename T> struct TransformFunctorTraits< binder2nd< minimum<T> > > : arithm::ArithmFuncTraits<sizeof(T), sizeof(T)>
    {
    };
}}}

namespace arithm
{
    void minMat_v4(PtrStepSz<uint> src1, PtrStepSz<uint> src2, PtrStepSz<uint> dst, cudaStream_t stream)
    {
        device::transform(src1, src2, dst, VMin4(), WithOutMask(), stream);
    }

    void minMat_v2(PtrStepSz<uint> src1, PtrStepSz<uint> src2, PtrStepSz<uint> dst, cudaStream_t stream)
    {
        device::transform(src1, src2, dst, VMin2(), WithOutMask(), stream);
    }

    template <typename T> void minMat(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream)
    {
        device::transform((PtrStepSz<T>) src1, (PtrStepSz<T>) src2, (PtrStepSz<T>) dst, minimum<T>(), WithOutMask(), stream);
    }

    template void minMat<uchar >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void minMat<schar >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void minMat<ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void minMat<short >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void minMat<int   >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void minMat<float >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void minMat<double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);

    template <typename T> void minScalar(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream)
    {
        device::transform((PtrStepSz<T>) src1, (PtrStepSz<T>) dst, cv::cuda::device::bind2nd(minimum<T>(), src2), WithOutMask(), stream);
    }

    template void minScalar<uchar >(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
    template void minScalar<schar >(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
    template void minScalar<ushort>(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
    template void minScalar<short >(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
    template void minScalar<int   >(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
    template void minScalar<float >(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
    template void minScalar<double>(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
}

//////////////////////////////////////////////////////////////////////////
// max

namespace arithm
{
    struct VMax4 : binary_function<uint, uint, uint>
    {
        __device__ __forceinline__ uint operator ()(uint a, uint b) const
        {
            return vmax4(a, b);
        }

        __host__ __device__ __forceinline__ VMax4() {}
        __host__ __device__ __forceinline__ VMax4(const VMax4&) {}
    };

    struct VMax2 : binary_function<uint, uint, uint>
    {
        __device__ __forceinline__ uint operator ()(uint a, uint b) const
        {
            return vmax2(a, b);
        }

        __host__ __device__ __forceinline__ VMax2() {}
        __host__ __device__ __forceinline__ VMax2(const VMax2&) {}
    };
}

namespace cv { namespace cuda { namespace device
{
    template <> struct TransformFunctorTraits< arithm::VMax4 > : arithm::ArithmFuncTraits<sizeof(uint), sizeof(uint)>
    {
    };

    template <> struct TransformFunctorTraits< arithm::VMax2 > : arithm::ArithmFuncTraits<sizeof(uint), sizeof(uint)>
    {
    };

    template <typename T> struct TransformFunctorTraits< maximum<T> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(T)>
    {
    };

    template <typename T> struct TransformFunctorTraits< binder2nd< maximum<T> > > : arithm::ArithmFuncTraits<sizeof(T), sizeof(T)>
    {
    };
}}}

namespace arithm
{
    void maxMat_v4(PtrStepSz<uint> src1, PtrStepSz<uint> src2, PtrStepSz<uint> dst, cudaStream_t stream)
    {
        device::transform(src1, src2, dst, VMax4(), WithOutMask(), stream);
    }

    void maxMat_v2(PtrStepSz<uint> src1, PtrStepSz<uint> src2, PtrStepSz<uint> dst, cudaStream_t stream)
    {
        device::transform(src1, src2, dst, VMax2(), WithOutMask(), stream);
    }

    template <typename T> void maxMat(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream)
    {
        device::transform((PtrStepSz<T>) src1, (PtrStepSz<T>) src2, (PtrStepSz<T>) dst, maximum<T>(), WithOutMask(), stream);
    }

    template void maxMat<uchar >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void maxMat<schar >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void maxMat<ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void maxMat<short >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void maxMat<int   >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void maxMat<float >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void maxMat<double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);

    template <typename T> void maxScalar(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream)
    {
        device::transform((PtrStepSz<T>) src1, (PtrStepSz<T>) dst, cv::cuda::device::bind2nd(maximum<T>(), src2), WithOutMask(), stream);
    }

    template void maxScalar<uchar >(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
    template void maxScalar<schar >(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
    template void maxScalar<ushort>(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
    template void maxScalar<short >(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
    template void maxScalar<int   >(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
    template void maxScalar<float >(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
    template void maxScalar<double>(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
}

#endif // CUDA_DISABLER
