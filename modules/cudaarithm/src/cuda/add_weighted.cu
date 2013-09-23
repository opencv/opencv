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

#include "arithm_func_traits.hpp"

using namespace cv::cuda;
using namespace cv::cuda::device;

namespace arithm
{
    template <typename T> struct UseDouble_
    {
        enum {value = 0};
    };
    template <> struct UseDouble_<double>
    {
        enum {value = 1};
    };
    template <typename T1, typename T2, typename D> struct UseDouble
    {
        enum {value = (UseDouble_<T1>::value || UseDouble_<T2>::value || UseDouble_<D>::value)};
    };

    template <typename T1, typename T2, typename D, bool useDouble> struct AddWeighted_;
    template <typename T1, typename T2, typename D> struct AddWeighted_<T1, T2, D, false> : binary_function<T1, T2, D>
    {
        float alpha;
        float beta;
        float gamma;

        __host__ AddWeighted_(double alpha_, double beta_, double gamma_) : alpha(static_cast<float>(alpha_)), beta(static_cast<float>(beta_)), gamma(static_cast<float>(gamma_)) {}

        __device__ __forceinline__ D operator ()(T1 a, T2 b) const
        {
            return saturate_cast<D>(a * alpha + b * beta + gamma);
        }
    };
    template <typename T1, typename T2, typename D> struct AddWeighted_<T1, T2, D, true> : binary_function<T1, T2, D>
    {
        double alpha;
        double beta;
        double gamma;

        __host__ AddWeighted_(double alpha_, double beta_, double gamma_) : alpha(alpha_), beta(beta_), gamma(gamma_) {}

        __device__ __forceinline__ D operator ()(T1 a, T2 b) const
        {
            return saturate_cast<D>(a * alpha + b * beta + gamma);
        }
    };
    template <typename T1, typename T2, typename D> struct AddWeighted : AddWeighted_<T1, T2, D, UseDouble<T1, T2, D>::value>
    {
        AddWeighted(double alpha_, double beta_, double gamma_) : AddWeighted_<T1, T2, D, UseDouble<T1, T2, D>::value>(alpha_, beta_, gamma_) {}
    };
}

namespace cv { namespace cuda { namespace device
{
    template <typename T1, typename T2, typename D, size_t src1_size, size_t src2_size, size_t dst_size> struct AddWeightedTraits : DefaultTransformFunctorTraits< arithm::AddWeighted<T1, T2, D> >
    {
    };
    template <typename T1, typename T2, typename D, size_t src_size, size_t dst_size> struct AddWeightedTraits<T1, T2, D, src_size, src_size, dst_size> : arithm::ArithmFuncTraits<src_size, dst_size>
    {
    };

    template <typename T1, typename T2, typename D> struct TransformFunctorTraits< arithm::AddWeighted<T1, T2, D> > : AddWeightedTraits<T1, T2, D, sizeof(T1), sizeof(T2), sizeof(D)>
    {
    };
}}}

namespace arithm
{
    template <typename T1, typename T2, typename D>
    void addWeighted(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream)
    {
        AddWeighted<T1, T2, D> op(alpha, beta, gamma);

        device::transform((PtrStepSz<T1>) src1, (PtrStepSz<T2>) src2, (PtrStepSz<D>) dst, op, WithOutMask(), stream);
    }

    template void addWeighted<uchar, uchar, uchar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, uchar, schar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, uchar, ushort>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, uchar, short>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, uchar, int>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, uchar, float>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, uchar, double>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);

    template void addWeighted<uchar, schar, uchar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, schar, schar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, schar, ushort>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, schar, short>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, schar, int>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, schar, float>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, schar, double>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);

    template void addWeighted<uchar, ushort, uchar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, ushort, schar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, ushort, ushort>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, ushort, short>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, ushort, int>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, ushort, float>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, ushort, double>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);

    template void addWeighted<uchar, short, uchar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, short, schar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, short, ushort>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, short, short>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, short, int>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, short, float>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, short, double>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);

    template void addWeighted<uchar, int, uchar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, int, schar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, int, ushort>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, int, short>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, int, int>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, int, float>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, int, double>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);

    template void addWeighted<uchar, float, uchar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, float, schar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, float, ushort>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, float, short>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, float, int>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, float, float>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, float, double>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);

    template void addWeighted<uchar, double, uchar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, double, schar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, double, ushort>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, double, short>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, double, int>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, double, float>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<uchar, double, double>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);



    template void addWeighted<schar, schar, uchar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, schar, schar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, schar, ushort>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, schar, short>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, schar, int>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, schar, float>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, schar, double>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);

    template void addWeighted<schar, ushort, uchar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, ushort, schar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, ushort, ushort>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, ushort, short>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, ushort, int>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, ushort, float>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, ushort, double>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);

    template void addWeighted<schar, short, uchar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, short, schar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, short, ushort>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, short, short>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, short, int>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, short, float>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, short, double>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);

    template void addWeighted<schar, int, uchar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, int, schar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, int, ushort>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, int, short>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, int, int>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, int, float>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, int, double>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);

    template void addWeighted<schar, float, uchar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, float, schar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, float, ushort>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, float, short>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, float, int>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, float, float>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, float, double>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);

    template void addWeighted<schar, double, uchar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, double, schar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, double, ushort>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, double, short>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, double, int>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, double, float>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<schar, double, double>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);



    template void addWeighted<ushort, ushort, uchar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<ushort, ushort, schar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<ushort, ushort, ushort>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<ushort, ushort, short>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<ushort, ushort, int>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<ushort, ushort, float>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<ushort, ushort, double>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);

    template void addWeighted<ushort, short, uchar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<ushort, short, schar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<ushort, short, ushort>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<ushort, short, short>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<ushort, short, int>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<ushort, short, float>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<ushort, short, double>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);

    template void addWeighted<ushort, int, uchar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<ushort, int, schar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<ushort, int, ushort>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<ushort, int, short>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<ushort, int, int>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<ushort, int, float>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<ushort, int, double>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);

    template void addWeighted<ushort, float, uchar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<ushort, float, schar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<ushort, float, ushort>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<ushort, float, short>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<ushort, float, int>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<ushort, float, float>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<ushort, float, double>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);

    template void addWeighted<ushort, double, uchar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<ushort, double, schar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<ushort, double, ushort>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<ushort, double, short>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<ushort, double, int>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<ushort, double, float>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<ushort, double, double>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);



    template void addWeighted<short, short, uchar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<short, short, schar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<short, short, ushort>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<short, short, short>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<short, short, int>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<short, short, float>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<short, short, double>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);

    template void addWeighted<short, int, uchar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<short, int, schar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<short, int, ushort>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<short, int, short>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<short, int, int>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<short, int, float>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<short, int, double>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);

    template void addWeighted<short, float, uchar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<short, float, schar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<short, float, ushort>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<short, float, short>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<short, float, int>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<short, float, float>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<short, float, double>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);

    template void addWeighted<short, double, uchar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<short, double, schar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<short, double, ushort>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<short, double, short>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<short, double, int>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<short, double, float>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<short, double, double>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);



    template void addWeighted<int, int, uchar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<int, int, schar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<int, int, ushort>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<int, int, short>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<int, int, int>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<int, int, float>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<int, int, double>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);

    template void addWeighted<int, float, uchar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<int, float, schar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<int, float, ushort>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<int, float, short>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<int, float, int>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<int, float, float>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<int, float, double>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);

    template void addWeighted<int, double, uchar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<int, double, schar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<int, double, ushort>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<int, double, short>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<int, double, int>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<int, double, float>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<int, double, double>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);



    template void addWeighted<float, float, uchar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<float, float, schar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<float, float, ushort>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<float, float, short>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<float, float, int>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<float, float, float>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<float, float, double>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);

    template void addWeighted<float, double, uchar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<float, double, schar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<float, double, ushort>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<float, double, short>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<float, double, int>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<float, double, float>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<float, double, double>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);



    template void addWeighted<double, double, uchar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<double, double, schar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<double, double, ushort>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<double, double, short>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<double, double, int>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<double, double, float>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<double, double, double>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
}

#endif /* CUDA_DISABLER */
