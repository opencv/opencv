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
    struct VAdd4 : binary_function<uint, uint, uint>
    {
        __device__ __forceinline__ uint operator ()(uint a, uint b) const
        {
            return vadd4(a, b);
        }

        __host__ __device__ __forceinline__ VAdd4() {}
        __host__ __device__ __forceinline__ VAdd4(const VAdd4&) {}
    };

    struct VAdd2 : binary_function<uint, uint, uint>
    {
        __device__ __forceinline__ uint operator ()(uint a, uint b) const
        {
            return vadd2(a, b);
        }

        __host__ __device__ __forceinline__ VAdd2() {}
        __host__ __device__ __forceinline__ VAdd2(const VAdd2&) {}
    };

    template <typename T, typename D> struct AddMat : binary_function<T, T, D>
    {
        __device__ __forceinline__ D operator ()(T a, T b) const
        {
            return saturate_cast<D>(a + b);
        }

        __host__ __device__ __forceinline__ AddMat() {}
        __host__ __device__ __forceinline__ AddMat(const AddMat&) {}
    };
}

namespace cv { namespace cuda { namespace device
{
    template <> struct TransformFunctorTraits< arithm::VAdd4 > : arithm::ArithmFuncTraits<sizeof(uint), sizeof(uint)>
    {
    };

    template <> struct TransformFunctorTraits< arithm::VAdd2 > : arithm::ArithmFuncTraits<sizeof(uint), sizeof(uint)>
    {
    };

    template <typename T, typename D> struct TransformFunctorTraits< arithm::AddMat<T, D> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(D)>
    {
    };
}}}

namespace arithm
{
    void addMat_v4(PtrStepSz<uint> src1, PtrStepSz<uint> src2, PtrStepSz<uint> dst, cudaStream_t stream)
    {
        device::transform(src1, src2, dst, VAdd4(), WithOutMask(), stream);
    }

    void addMat_v2(PtrStepSz<uint> src1, PtrStepSz<uint> src2, PtrStepSz<uint> dst, cudaStream_t stream)
    {
        device::transform(src1, src2, dst, VAdd2(), WithOutMask(), stream);
    }

    template <typename T, typename D>
    void addMat(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream)
    {
        if (mask.data)
            device::transform((PtrStepSz<T>) src1, (PtrStepSz<T>) src2, (PtrStepSz<D>) dst, AddMat<T, D>(), mask, stream);
        else
            device::transform((PtrStepSz<T>) src1, (PtrStepSz<T>) src2, (PtrStepSz<D>) dst, AddMat<T, D>(), WithOutMask(), stream);
    }

    template void addMat<uchar, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addMat<uchar, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addMat<uchar, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addMat<uchar, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addMat<uchar, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addMat<uchar, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addMat<uchar, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);

    template void addMat<schar, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addMat<schar, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addMat<schar, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addMat<schar, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addMat<schar, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addMat<schar, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addMat<schar, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);

    //template void addMat<ushort, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addMat<ushort, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addMat<ushort, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addMat<ushort, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addMat<ushort, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addMat<ushort, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addMat<ushort, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);

    //template void addMat<short, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addMat<short, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addMat<short, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addMat<short, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addMat<short, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addMat<short, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addMat<short, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);

    //template void addMat<int, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addMat<int, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addMat<int, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addMat<int, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addMat<int, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addMat<int, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addMat<int, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);

    //template void addMat<float, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addMat<float, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addMat<float, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addMat<float, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addMat<float, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addMat<float, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addMat<float, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);

    //template void addMat<double, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addMat<double, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addMat<double, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addMat<double, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addMat<double, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addMat<double, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addMat<double, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
}

#endif // CUDA_DISABLER
