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
    struct VSub4 : binary_function<uint, uint, uint>
    {
        __device__ __forceinline__ uint operator ()(uint a, uint b) const
        {
            return vsub4(a, b);
        }

        __host__ __device__ __forceinline__ VSub4() {}
        __host__ __device__ __forceinline__ VSub4(const VSub4&) {}
    };

    struct VSub2 : binary_function<uint, uint, uint>
    {
        __device__ __forceinline__ uint operator ()(uint a, uint b) const
        {
            return vsub2(a, b);
        }

        __host__ __device__ __forceinline__ VSub2() {}
        __host__ __device__ __forceinline__ VSub2(const VSub2&) {}
    };

    template <typename T, typename D> struct SubMat : binary_function<T, T, D>
    {
        __device__ __forceinline__ D operator ()(T a, T b) const
        {
            return saturate_cast<D>(a - b);
        }

        __host__ __device__ __forceinline__ SubMat() {}
        __host__ __device__ __forceinline__ SubMat(const SubMat&) {}
    };
}

namespace cv { namespace cuda { namespace device
{
    template <> struct TransformFunctorTraits< arithm::VSub4 > : arithm::ArithmFuncTraits<sizeof(uint), sizeof(uint)>
    {
    };

    template <> struct TransformFunctorTraits< arithm::VSub2 > : arithm::ArithmFuncTraits<sizeof(uint), sizeof(uint)>
    {
    };

    template <typename T, typename D> struct TransformFunctorTraits< arithm::SubMat<T, D> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(D)>
    {
    };
}}}

namespace arithm
{
    void subMat_v4(PtrStepSz<uint> src1, PtrStepSz<uint> src2, PtrStepSz<uint> dst, cudaStream_t stream)
    {
        device::transform(src1, src2, dst, VSub4(), WithOutMask(), stream);
    }

    void subMat_v2(PtrStepSz<uint> src1, PtrStepSz<uint> src2, PtrStepSz<uint> dst, cudaStream_t stream)
    {
        device::transform(src1, src2, dst, VSub2(), WithOutMask(), stream);
    }

    template <typename T, typename D>
    void subMat(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream)
    {
        if (mask.data)
            device::transform((PtrStepSz<T>) src1, (PtrStepSz<T>) src2, (PtrStepSz<D>) dst, SubMat<T, D>(), mask, stream);
        else
            device::transform((PtrStepSz<T>) src1, (PtrStepSz<T>) src2, (PtrStepSz<D>) dst, SubMat<T, D>(), WithOutMask(), stream);
    }

    template void subMat<uchar, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subMat<uchar, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subMat<uchar, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subMat<uchar, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subMat<uchar, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subMat<uchar, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subMat<uchar, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);

    template void subMat<schar, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subMat<schar, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subMat<schar, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subMat<schar, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subMat<schar, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subMat<schar, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subMat<schar, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);

    //template void subMat<ushort, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void subMat<ushort, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subMat<ushort, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subMat<ushort, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subMat<ushort, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subMat<ushort, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subMat<ushort, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);

    //template void subMat<short, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void subMat<short, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subMat<short, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subMat<short, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subMat<short, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subMat<short, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subMat<short, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);

    //template void subMat<int, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void subMat<int, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void subMat<int, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void subMat<int, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subMat<int, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subMat<int, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subMat<int, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);

    //template void subMat<float, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void subMat<float, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void subMat<float, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void subMat<float, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void subMat<float, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subMat<float, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subMat<float, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);

    //template void subMat<double, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void subMat<double, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void subMat<double, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void subMat<double, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void subMat<double, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void subMat<double, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subMat<double, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
}

#endif // CUDA_DISABLER
