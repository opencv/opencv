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
    template <typename T> struct TransformFunctorTraits< thresh_binary_func<T> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(T)>
    {
    };

    template <typename T> struct TransformFunctorTraits< thresh_binary_inv_func<T> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(T)>
    {
    };

    template <typename T> struct TransformFunctorTraits< thresh_trunc_func<T> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(T)>
    {
    };

    template <typename T> struct TransformFunctorTraits< thresh_to_zero_func<T> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(T)>
    {
    };

    template <typename T> struct TransformFunctorTraits< thresh_to_zero_inv_func<T> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(T)>
    {
    };
}}}

namespace arithm
{
    template <template <typename> class Op, typename T>
    void threshold_caller(PtrStepSz<T> src, PtrStepSz<T> dst, T thresh, T maxVal, cudaStream_t stream)
    {
        Op<T> op(thresh, maxVal);
        device::transform(src, dst, op, WithOutMask(), stream);
    }

    template <typename T>
    void threshold(PtrStepSzb src, PtrStepSzb dst, double thresh, double maxVal, int type, cudaStream_t stream)
    {
        typedef void (*caller_t)(PtrStepSz<T> src, PtrStepSz<T> dst, T thresh, T maxVal, cudaStream_t stream);

        static const caller_t callers[] =
        {
            threshold_caller<thresh_binary_func, T>,
            threshold_caller<thresh_binary_inv_func, T>,
            threshold_caller<thresh_trunc_func, T>,
            threshold_caller<thresh_to_zero_func, T>,
            threshold_caller<thresh_to_zero_inv_func, T>
        };

        callers[type]((PtrStepSz<T>) src, (PtrStepSz<T>) dst, static_cast<T>(thresh), static_cast<T>(maxVal), stream);
    }

    template void threshold<uchar>(PtrStepSzb src, PtrStepSzb dst, double thresh, double maxVal, int type, cudaStream_t stream);
    template void threshold<schar>(PtrStepSzb src, PtrStepSzb dst, double thresh, double maxVal, int type, cudaStream_t stream);
    template void threshold<ushort>(PtrStepSzb src, PtrStepSzb dst, double thresh, double maxVal, int type, cudaStream_t stream);
    template void threshold<short>(PtrStepSzb src, PtrStepSzb dst, double thresh, double maxVal, int type, cudaStream_t stream);
    template void threshold<int>(PtrStepSzb src, PtrStepSzb dst, double thresh, double maxVal, int type, cudaStream_t stream);
    template void threshold<float>(PtrStepSzb src, PtrStepSzb dst, double thresh, double maxVal, int type, cudaStream_t stream);
    template void threshold<double>(PtrStepSzb src, PtrStepSzb dst, double thresh, double maxVal, int type, cudaStream_t stream);
}

#endif // CUDA_DISABLER
