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
    struct VCmpEq4 : binary_function<uint, uint, uint>
    {
        __device__ __forceinline__ uint operator ()(uint a, uint b) const
        {
            return vcmpeq4(a, b);
        }

        __host__ __device__ __forceinline__ VCmpEq4() {}
        __host__ __device__ __forceinline__ VCmpEq4(const VCmpEq4&) {}
    };
    struct VCmpNe4 : binary_function<uint, uint, uint>
    {
        __device__ __forceinline__ uint operator ()(uint a, uint b) const
        {
            return vcmpne4(a, b);
        }

        __host__ __device__ __forceinline__ VCmpNe4() {}
        __host__ __device__ __forceinline__ VCmpNe4(const VCmpNe4&) {}
    };
    struct VCmpLt4 : binary_function<uint, uint, uint>
    {
        __device__ __forceinline__ uint operator ()(uint a, uint b) const
        {
            return vcmplt4(a, b);
        }

        __host__ __device__ __forceinline__ VCmpLt4() {}
        __host__ __device__ __forceinline__ VCmpLt4(const VCmpLt4&) {}
    };
    struct VCmpLe4 : binary_function<uint, uint, uint>
    {
        __device__ __forceinline__ uint operator ()(uint a, uint b) const
        {
            return vcmple4(a, b);
        }

        __host__ __device__ __forceinline__ VCmpLe4() {}
        __host__ __device__ __forceinline__ VCmpLe4(const VCmpLe4&) {}
    };

    template <class Op, typename T>
    struct Cmp : binary_function<T, T, uchar>
    {
        __device__ __forceinline__ uchar operator()(T a, T b) const
        {
            Op op;
            return -op(a, b);
        }
    };
}

namespace cv { namespace cuda { namespace device
{
    template <> struct TransformFunctorTraits< arithm::VCmpEq4 > : arithm::ArithmFuncTraits<sizeof(uint), sizeof(uint)>
    {
    };
    template <> struct TransformFunctorTraits< arithm::VCmpNe4 > : arithm::ArithmFuncTraits<sizeof(uint), sizeof(uint)>
    {
    };
    template <> struct TransformFunctorTraits< arithm::VCmpLt4 > : arithm::ArithmFuncTraits<sizeof(uint), sizeof(uint)>
    {
    };
    template <> struct TransformFunctorTraits< arithm::VCmpLe4 > : arithm::ArithmFuncTraits<sizeof(uint), sizeof(uint)>
    {
    };

    template <class Op, typename T> struct TransformFunctorTraits< arithm::Cmp<Op, T> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(uchar)>
    {
    };
}}}

namespace arithm
{
    void cmpMatEq_v4(PtrStepSz<uint> src1, PtrStepSz<uint> src2, PtrStepSz<uint> dst, cudaStream_t stream)
    {
        device::transform(src1, src2, dst, VCmpEq4(), WithOutMask(), stream);
    }
    void cmpMatNe_v4(PtrStepSz<uint> src1, PtrStepSz<uint> src2, PtrStepSz<uint> dst, cudaStream_t stream)
    {
        device::transform(src1, src2, dst, VCmpNe4(), WithOutMask(), stream);
    }
    void cmpMatLt_v4(PtrStepSz<uint> src1, PtrStepSz<uint> src2, PtrStepSz<uint> dst, cudaStream_t stream)
    {
        device::transform(src1, src2, dst, VCmpLt4(), WithOutMask(), stream);
    }
    void cmpMatLe_v4(PtrStepSz<uint> src1, PtrStepSz<uint> src2, PtrStepSz<uint> dst, cudaStream_t stream)
    {
        device::transform(src1, src2, dst, VCmpLe4(), WithOutMask(), stream);
    }

    template <template <typename> class Op, typename T>
    void cmpMat(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream)
    {
        Cmp<Op<T>, T> op;
        device::transform((PtrStepSz<T>) src1, (PtrStepSz<T>) src2, dst, op, WithOutMask(), stream);
    }

    template <typename T> void cmpMatEq(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream)
    {
        cmpMat<equal_to, T>(src1, src2, dst, stream);
    }
    template <typename T> void cmpMatNe(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream)
    {
        cmpMat<not_equal_to, T>(src1, src2, dst, stream);
    }
    template <typename T> void cmpMatLt(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream)
    {
        cmpMat<less, T>(src1, src2, dst, stream);
    }
    template <typename T> void cmpMatLe(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream)
    {
        cmpMat<less_equal, T>(src1, src2, dst, stream);
    }

    template void cmpMatEq<uchar >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatEq<schar >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatEq<ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatEq<short >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatEq<int   >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatEq<float >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatEq<double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);

    template void cmpMatNe<uchar >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatNe<schar >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatNe<ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatNe<short >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatNe<int   >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatNe<float >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatNe<double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);

    template void cmpMatLt<uchar >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatLt<schar >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatLt<ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatLt<short >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatLt<int   >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatLt<float >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatLt<double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);

    template void cmpMatLe<uchar >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatLe<schar >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatLe<ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatLe<short >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatLe<int   >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatLe<float >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatLe<double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
}

#endif // CUDA_DISABLER
