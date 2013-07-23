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
    template <typename T, typename S, typename D> struct AddScalar : unary_function<T, D>
    {
        S val;

        __host__ explicit AddScalar(S val_) : val(val_) {}

        __device__ __forceinline__ D operator ()(T a) const
        {
            return saturate_cast<D>(a + val);
        }
    };
}

namespace cv { namespace cuda { namespace device
{
    template <typename T, typename S, typename D> struct TransformFunctorTraits< arithm::AddScalar<T, S, D> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(D)>
    {
    };
}}}

namespace arithm
{
    template <typename T, typename S, typename D>
    void addScalar(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream)
    {
        AddScalar<T, S, D> op(static_cast<S>(val));

        if (mask.data)
            device::transform((PtrStepSz<T>) src1, (PtrStepSz<D>) dst, op, mask, stream);
        else
            device::transform((PtrStepSz<T>) src1, (PtrStepSz<D>) dst, op, WithOutMask(), stream);
    }

    template void addScalar<uchar, float, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addScalar<uchar, float, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addScalar<uchar, float, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addScalar<uchar, float, short>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addScalar<uchar, float, int>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addScalar<uchar, float, float>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addScalar<uchar, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);

    template void addScalar<schar, float, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addScalar<schar, float, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addScalar<schar, float, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addScalar<schar, float, short>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addScalar<schar, float, int>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addScalar<schar, float, float>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addScalar<schar, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);

    //template void addScalar<ushort, float, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addScalar<ushort, float, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addScalar<ushort, float, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addScalar<ushort, float, short>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addScalar<ushort, float, int>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addScalar<ushort, float, float>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addScalar<ushort, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);

    //template void addScalar<short, float, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addScalar<short, float, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addScalar<short, float, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addScalar<short, float, short>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addScalar<short, float, int>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addScalar<short, float, float>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addScalar<short, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);

    //template void addScalar<int, float, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addScalar<int, float, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addScalar<int, float, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addScalar<int, float, short>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addScalar<int, float, int>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addScalar<int, float, float>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addScalar<int, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);

    //template void addScalar<float, float, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addScalar<float, float, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addScalar<float, float, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addScalar<float, float, short>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addScalar<float, float, int>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addScalar<float, float, float>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addScalar<float, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);

    //template void addScalar<double, double, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addScalar<double, double, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addScalar<double, double, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addScalar<double, double, short>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addScalar<double, double, int>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addScalar<double, double, float>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addScalar<double, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
}

#endif // CUDA_DISABLER
