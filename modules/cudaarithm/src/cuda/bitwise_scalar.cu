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

namespace cv { namespace cuda { namespace device
{
    template <typename T> struct TransformFunctorTraits< binder2nd< bit_and<T> > > : arithm::ArithmFuncTraits<sizeof(T), sizeof(T)>
    {
    };

    template <typename T> struct TransformFunctorTraits< binder2nd< bit_or<T> > > : arithm::ArithmFuncTraits<sizeof(T), sizeof(T)>
    {
    };

    template <typename T> struct TransformFunctorTraits< binder2nd< bit_xor<T> > > : arithm::ArithmFuncTraits<sizeof(T), sizeof(T)>
    {
    };
}}}

namespace arithm
{
    template <typename T> void bitScalarAnd(PtrStepSzb src1, uint src2, PtrStepSzb dst, cudaStream_t stream)
    {
        device::transform((PtrStepSz<T>) src1, (PtrStepSz<T>) dst, cv::cuda::device::bind2nd(bit_and<T>(), src2), WithOutMask(), stream);
    }

    template <typename T> void bitScalarOr(PtrStepSzb src1, uint src2, PtrStepSzb dst, cudaStream_t stream)
    {
        device::transform((PtrStepSz<T>) src1, (PtrStepSz<T>) dst, cv::cuda::device::bind2nd(bit_or<T>(), src2), WithOutMask(), stream);
    }

    template <typename T> void bitScalarXor(PtrStepSzb src1, uint src2, PtrStepSzb dst, cudaStream_t stream)
    {
        device::transform((PtrStepSz<T>) src1, (PtrStepSz<T>) dst, cv::cuda::device::bind2nd(bit_xor<T>(), src2), WithOutMask(), stream);
    }

    template void bitScalarAnd<uchar>(PtrStepSzb src1, uint src2, PtrStepSzb dst, cudaStream_t stream);
    template void bitScalarAnd<ushort>(PtrStepSzb src1, uint src2, PtrStepSzb dst, cudaStream_t stream);
    template void bitScalarAnd<int>(PtrStepSzb src1, uint src2, PtrStepSzb dst, cudaStream_t stream);
    template void bitScalarAnd<unsigned int>(PtrStepSzb src1, uint src2, PtrStepSzb dst, cudaStream_t stream);

    template void bitScalarOr<uchar>(PtrStepSzb src1, uint src2, PtrStepSzb dst, cudaStream_t stream);
    template void bitScalarOr<ushort>(PtrStepSzb src1, uint src2, PtrStepSzb dst, cudaStream_t stream);
    template void bitScalarOr<int>(PtrStepSzb src1, uint src2, PtrStepSzb dst, cudaStream_t stream);
    template void bitScalarOr<unsigned int>(PtrStepSzb src1, uint src2, PtrStepSzb dst, cudaStream_t stream);

    template void bitScalarXor<uchar>(PtrStepSzb src1, uint src2, PtrStepSzb dst, cudaStream_t stream);
    template void bitScalarXor<ushort>(PtrStepSzb src1, uint src2, PtrStepSzb dst, cudaStream_t stream);
    template void bitScalarXor<int>(PtrStepSzb src1, uint src2, PtrStepSzb dst, cudaStream_t stream);
    template void bitScalarXor<unsigned int>(PtrStepSzb src1, uint src2, PtrStepSzb dst, cudaStream_t stream);
}

#endif // CUDA_DISABLER
