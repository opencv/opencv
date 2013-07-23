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
    template <typename T, typename S, typename D> struct DivScalar : unary_function<T, D>
    {
        S val;

        __host__ explicit DivScalar(S val_) : val(val_) {}

        __device__ __forceinline__ D operator ()(T a) const
        {
            return saturate_cast<D>(a / val);
        }
    };

    template <typename T, typename S, typename D> struct DivScalarInv : unary_function<T, D>
    {
        S val;

        explicit DivScalarInv(S val_) : val(val_) {}

        __device__ __forceinline__ D operator ()(T a) const
        {
            return a != 0 ? saturate_cast<D>(val / a) : 0;
        }
    };
}

namespace cv { namespace cuda { namespace device
{
    template <typename T, typename S, typename D> struct TransformFunctorTraits< arithm::DivScalar<T, S, D> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(D)>
    {
    };

    template <typename T, typename S, typename D> struct TransformFunctorTraits< arithm::DivScalarInv<T, S, D> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(D)>
    {
    };
}}}

namespace arithm
{
    template <typename T, typename S, typename D>
    void divScalar(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream)
    {
        if (inv)
        {
            DivScalarInv<T, S, D> op(static_cast<S>(val));
            device::transform((PtrStepSz<T>) src1, (PtrStepSz<D>) dst, op, WithOutMask(), stream);
        }
        else
        {
            DivScalar<T, S, D> op(static_cast<S>(val));
            device::transform((PtrStepSz<T>) src1, (PtrStepSz<D>) dst, op, WithOutMask(), stream);
        }
    }

    template void divScalar<uchar, float, uchar>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<uchar, float, schar>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<uchar, float, ushort>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<uchar, float, short>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<uchar, float, int>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<uchar, float, float>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<uchar, double, double>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);

    template void divScalar<schar, float, uchar>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<schar, float, schar>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<schar, float, ushort>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<schar, float, short>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<schar, float, int>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<schar, float, float>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<schar, double, double>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);

    //template void divScalar<ushort, float, uchar>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    //template void divScalar<ushort, float, schar>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<ushort, float, ushort>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<ushort, float, short>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<ushort, float, int>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<ushort, float, float>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<ushort, double, double>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);

    //template void divScalar<short, float, uchar>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    //template void divScalar<short, float, schar>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<short, float, ushort>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<short, float, short>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<short, float, int>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<short, float, float>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<short, double, double>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);

    //template void divScalar<int, float, uchar>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    //template void divScalar<int, float, schar>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    //template void divScalar<int, float, ushort>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    //template void divScalar<int, float, short>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<int, float, int>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<int, float, float>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<int, double, double>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);

    //template void divScalar<float, float, uchar>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    //template void divScalar<float, float, schar>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    //template void divScalar<float, float, ushort>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    //template void divScalar<float, float, short>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    //template void divScalar<float, float, int>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<float, float, float>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<float, double, double>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);

    //template void divScalar<double, double, uchar>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    //template void divScalar<double, double, schar>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    //template void divScalar<double, double, ushort>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    //template void divScalar<double, double, short>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    //template void divScalar<double, double, int>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    //template void divScalar<double, double, float>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<double, double, double>(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
}

#endif // CUDA_DISABLER
