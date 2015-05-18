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

#include "opencv2/gpu/device/common.hpp"
#include "opencv2/gpu/device/functional.hpp"
#include "opencv2/gpu/device/vec_math.hpp"
#include "opencv2/gpu/device/transform.hpp"
#include "opencv2/gpu/device/limits.hpp"
#include "opencv2/gpu/device/saturate_cast.hpp"
#include "opencv2/gpu/device/simd_functions.hpp"

using namespace cv::gpu;
using namespace cv::gpu::device;

namespace arithm
{
    template <size_t src_size, size_t dst_size> struct ArithmFuncTraits
    {
        enum { simple_block_dim_x = 32 };
        enum { simple_block_dim_y = 8 };

        enum { smart_block_dim_x = 32 };
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 1 };
    };

    template <> struct ArithmFuncTraits<1, 1>
    {
        enum { simple_block_dim_x = 32 };
        enum { simple_block_dim_y = 8 };

        enum { smart_block_dim_x = 32 };
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct ArithmFuncTraits<1, 2>
    {
        enum { simple_block_dim_x = 32 };
        enum { simple_block_dim_y = 8 };

        enum { smart_block_dim_x = 32 };
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct ArithmFuncTraits<1, 4>
    {
        enum { simple_block_dim_x = 32 };
        enum { simple_block_dim_y = 8 };

        enum { smart_block_dim_x = 32 };
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };

    template <> struct ArithmFuncTraits<2, 1>
    {
        enum { simple_block_dim_x = 32 };
        enum { simple_block_dim_y = 8 };

        enum { smart_block_dim_x = 32 };
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct ArithmFuncTraits<2, 2>
    {
        enum { simple_block_dim_x = 32 };
        enum { simple_block_dim_y = 8 };

        enum { smart_block_dim_x = 32 };
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct ArithmFuncTraits<2, 4>
    {
        enum { simple_block_dim_x = 32 };
        enum { simple_block_dim_y = 8 };

        enum { smart_block_dim_x = 32 };
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };

    template <> struct ArithmFuncTraits<4, 1>
    {
        enum { simple_block_dim_x = 32 };
        enum { simple_block_dim_y = 8 };

        enum { smart_block_dim_x = 32 };
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct ArithmFuncTraits<4, 2>
    {
        enum { simple_block_dim_x = 32 };
        enum { simple_block_dim_y = 8 };

        enum { smart_block_dim_x = 32 };
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct ArithmFuncTraits<4, 4>
    {
        enum { simple_block_dim_x = 32 };
        enum { simple_block_dim_y = 8 };

        enum { smart_block_dim_x = 32 };
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
}

//////////////////////////////////////////////////////////////////////////
// addMat

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

    ////////////////////////////////////

    struct VAdd2 : binary_function<uint, uint, uint>
    {
        __device__ __forceinline__ uint operator ()(uint a, uint b) const
        {
            return vadd2(a, b);
        }

        __host__ __device__ __forceinline__ VAdd2() {}
        __host__ __device__ __forceinline__ VAdd2(const VAdd2&) {}
    };

    ////////////////////////////////////

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

namespace cv { namespace gpu { namespace device
{
    template <> struct TransformFunctorTraits< arithm::VAdd4 > : arithm::ArithmFuncTraits<sizeof(uint), sizeof(uint)>
    {
    };

    ////////////////////////////////////

    template <> struct TransformFunctorTraits< arithm::VAdd2 > : arithm::ArithmFuncTraits<sizeof(uint), sizeof(uint)>
    {
    };

    ////////////////////////////////////

    template <typename T, typename D> struct TransformFunctorTraits< arithm::AddMat<T, D> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(D)>
    {
    };
}}}

namespace arithm
{
    void addMat_v4(PtrStepSz<uint> src1, PtrStepSz<uint> src2, PtrStepSz<uint> dst, cudaStream_t stream)
    {
        transform(src1, src2, dst, VAdd4(), WithOutMask(), stream);
    }

    void addMat_v2(PtrStepSz<uint> src1, PtrStepSz<uint> src2, PtrStepSz<uint> dst, cudaStream_t stream)
    {
        transform(src1, src2, dst, VAdd2(), WithOutMask(), stream);
    }

    template <typename T, typename D>
    void addMat(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream)
    {
        if (mask.data)
            transform((PtrStepSz<T>) src1, (PtrStepSz<T>) src2, (PtrStepSz<D>) dst, AddMat<T, D>(), mask, stream);
        else
            transform((PtrStepSz<T>) src1, (PtrStepSz<T>) src2, (PtrStepSz<D>) dst, AddMat<T, D>(), WithOutMask(), stream);
    }

    template void addMat<uchar, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
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
#endif
    template void addMat<float, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void addMat<float, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);

    //template void addMat<double, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addMat<double, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addMat<double, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addMat<double, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addMat<double, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addMat<double, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addMat<double, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
#endif
}

//////////////////////////////////////////////////////////////////////////
// addScalar

namespace arithm
{
    template <typename T, typename S, typename D> struct AddScalar : unary_function<T, D>
    {
        S val;

        explicit AddScalar(S val_) : val(val_) {}

        __device__ __forceinline__ D operator ()(T a) const
        {
            return saturate_cast<D>(a + val);
        }
    };
}

namespace cv { namespace gpu { namespace device
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
            transform((PtrStepSz<T>) src1, (PtrStepSz<D>) dst, op, mask, stream);
        else
            transform((PtrStepSz<T>) src1, (PtrStepSz<D>) dst, op, WithOutMask(), stream);
    }

    template void addScalar<uchar, float, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
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
#endif
    template void addScalar<float, float, float>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void addScalar<float, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);

    //template void addScalar<double, double, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addScalar<double, double, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addScalar<double, double, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addScalar<double, double, short>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addScalar<double, double, int>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void addScalar<double, double, float>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void addScalar<double, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
#endif
}

//////////////////////////////////////////////////////////////////////////
// subMat

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

    ////////////////////////////////////

    struct VSub2 : binary_function<uint, uint, uint>
    {
        __device__ __forceinline__ uint operator ()(uint a, uint b) const
        {
            return vsub2(a, b);
        }

        __host__ __device__ __forceinline__ VSub2() {}
        __host__ __device__ __forceinline__ VSub2(const VSub2&) {}
    };

    ////////////////////////////////////

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

namespace cv { namespace gpu { namespace device
{
    template <> struct TransformFunctorTraits< arithm::VSub4 > : arithm::ArithmFuncTraits<sizeof(uint), sizeof(uint)>
    {
    };

    ////////////////////////////////////

    template <> struct TransformFunctorTraits< arithm::VSub2 > : arithm::ArithmFuncTraits<sizeof(uint), sizeof(uint)>
    {
    };

    ////////////////////////////////////

    template <typename T, typename D> struct TransformFunctorTraits< arithm::SubMat<T, D> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(D)>
    {
    };
}}}

namespace arithm
{
    void subMat_v4(PtrStepSz<uint> src1, PtrStepSz<uint> src2, PtrStepSz<uint> dst, cudaStream_t stream)
    {
        transform(src1, src2, dst, VSub4(), WithOutMask(), stream);
    }

    void subMat_v2(PtrStepSz<uint> src1, PtrStepSz<uint> src2, PtrStepSz<uint> dst, cudaStream_t stream)
    {
        transform(src1, src2, dst, VSub2(), WithOutMask(), stream);
    }

    template <typename T, typename D>
    void subMat(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream)
    {
        if (mask.data)
            transform((PtrStepSz<T>) src1, (PtrStepSz<T>) src2, (PtrStepSz<D>) dst, SubMat<T, D>(), mask, stream);
        else
            transform((PtrStepSz<T>) src1, (PtrStepSz<T>) src2, (PtrStepSz<D>) dst, SubMat<T, D>(), WithOutMask(), stream);
    }

    template void subMat<uchar, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
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
#endif
    template void subMat<float, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void subMat<float, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);

    //template void subMat<double, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void subMat<double, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void subMat<double, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void subMat<double, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void subMat<double, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void subMat<double, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subMat<double, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
#endif
}

//////////////////////////////////////////////////////////////////////////
// subScalar

namespace arithm
{
    template <typename T, typename S, typename D>
    void subScalar(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream)
    {
        AddScalar<T, S, D> op(-static_cast<S>(val));

        if (mask.data)
            transform((PtrStepSz<T>) src1, (PtrStepSz<D>) dst, op, mask, stream);
        else
            transform((PtrStepSz<T>) src1, (PtrStepSz<D>) dst, op, WithOutMask(), stream);
    }

    template void subScalar<uchar, float, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void subScalar<uchar, float, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subScalar<uchar, float, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subScalar<uchar, float, short>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subScalar<uchar, float, int>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subScalar<uchar, float, float>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subScalar<uchar, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);

    template void subScalar<schar, float, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subScalar<schar, float, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subScalar<schar, float, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subScalar<schar, float, short>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subScalar<schar, float, int>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subScalar<schar, float, float>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subScalar<schar, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);

    //template void subScalar<ushort, float, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void subScalar<ushort, float, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subScalar<ushort, float, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subScalar<ushort, float, short>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subScalar<ushort, float, int>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subScalar<ushort, float, float>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subScalar<ushort, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);

    //template void subScalar<short, float, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void subScalar<short, float, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subScalar<short, float, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subScalar<short, float, short>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subScalar<short, float, int>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subScalar<short, float, float>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subScalar<short, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);

    //template void subScalar<int, float, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void subScalar<int, float, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void subScalar<int, float, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void subScalar<int, float, short>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subScalar<int, float, int>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subScalar<int, float, float>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subScalar<int, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);

    //template void subScalar<float, float, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void subScalar<float, float, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void subScalar<float, float, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void subScalar<float, float, short>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void subScalar<float, float, int>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
#endif
    template void subScalar<float, float, float>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void subScalar<float, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);

    //template void subScalar<double, double, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void subScalar<double, double, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void subScalar<double, double, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void subScalar<double, double, short>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void subScalar<double, double, int>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    //template void subScalar<double, double, float>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template void subScalar<double, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
#endif
}

//////////////////////////////////////////////////////////////////////////
// mulMat

namespace arithm
{
    struct Mul_8uc4_32f : binary_function<uint, float, uint>
    {
        __device__ __forceinline__ uint operator ()(uint a, float b) const
        {
            uint res = 0;

            res |= (saturate_cast<uchar>((0xffu & (a      )) * b)      );
            res |= (saturate_cast<uchar>((0xffu & (a >>  8)) * b) <<  8);
            res |= (saturate_cast<uchar>((0xffu & (a >> 16)) * b) << 16);
            res |= (saturate_cast<uchar>((0xffu & (a >> 24)) * b) << 24);

            return res;
        }

        __host__ __device__ __forceinline__ Mul_8uc4_32f() {}
        __host__ __device__ __forceinline__ Mul_8uc4_32f(const Mul_8uc4_32f&) {}
    };

    struct Mul_16sc4_32f : binary_function<short4, float, short4>
    {
        __device__ __forceinline__ short4 operator ()(short4 a, float b) const
        {
            return make_short4(saturate_cast<short>(a.x * b), saturate_cast<short>(a.y * b),
                               saturate_cast<short>(a.z * b), saturate_cast<short>(a.w * b));
        }

        __host__ __device__ __forceinline__ Mul_16sc4_32f() {}
        __host__ __device__ __forceinline__ Mul_16sc4_32f(const Mul_16sc4_32f&) {}
    };

    template <typename T, typename D> struct Mul : binary_function<T, T, D>
    {
        __device__ __forceinline__ D operator ()(T a, T b) const
        {
            return saturate_cast<D>(a * b);
        }

        __host__ __device__ __forceinline__ Mul() {}
        __host__ __device__ __forceinline__ Mul(const Mul&) {}
    };

    template <typename T, typename S, typename D> struct MulScale : binary_function<T, T, D>
    {
        S scale;

        explicit MulScale(S scale_) : scale(scale_) {}

        __device__ __forceinline__ D operator ()(T a, T b) const
        {
            return saturate_cast<D>(scale * a * b);
        }
    };
}

namespace cv { namespace gpu { namespace device
{
    template <> struct TransformFunctorTraits<arithm::Mul_8uc4_32f> : arithm::ArithmFuncTraits<sizeof(uint), sizeof(uint)>
    {
    };

    template <typename T, typename D> struct TransformFunctorTraits< arithm::Mul<T, D> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(D)>
    {
    };

    template <typename T, typename S, typename D> struct TransformFunctorTraits< arithm::MulScale<T, S, D> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(D)>
    {
    };
}}}

namespace arithm
{
    void mulMat_8uc4_32f(PtrStepSz<uint> src1, PtrStepSzf src2, PtrStepSz<uint> dst, cudaStream_t stream)
    {
        transform(src1, src2, dst, Mul_8uc4_32f(), WithOutMask(), stream);
    }

    void mulMat_16sc4_32f(PtrStepSz<short4> src1, PtrStepSzf src2, PtrStepSz<short4> dst, cudaStream_t stream)
    {
        transform(src1, src2, dst, Mul_16sc4_32f(), WithOutMask(), stream);
    }

    template <typename T, typename S, typename D>
    void mulMat(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream)
    {
        if (scale == 1)
        {
            Mul<T, D> op;
            transform((PtrStepSz<T>) src1, (PtrStepSz<T>) src2, (PtrStepSz<D>) dst, op, WithOutMask(), stream);
        }
        else
        {
            MulScale<T, S, D> op(static_cast<S>(scale));
            transform((PtrStepSz<T>) src1, (PtrStepSz<T>) src2, (PtrStepSz<D>) dst, op, WithOutMask(), stream);
        }
    }

    template void mulMat<uchar, float, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void mulMat<uchar, float, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<uchar, float, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<uchar, float, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<uchar, float, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<uchar, float, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<uchar, double, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);

    template void mulMat<schar, float, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<schar, float, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<schar, float, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<schar, float, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<schar, float, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<schar, float, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<schar, double, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);

    //template void mulMat<ushort, float, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void mulMat<ushort, float, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<ushort, float, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<ushort, float, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<ushort, float, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<ushort, float, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<ushort, double, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);

    //template void mulMat<short, float, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void mulMat<short, float, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<short, float, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<short, float, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<short, float, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<short, float, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<short, double, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);

    //template void mulMat<int, float, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void mulMat<int, float, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void mulMat<int, float, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void mulMat<int, float, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<int, float, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<int, float, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<int, double, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);

    //template void mulMat<float, float, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void mulMat<float, float, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void mulMat<float, float, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void mulMat<float, float, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void mulMat<float, float, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
#endif
    template void mulMat<float, float, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void mulMat<float, double, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);

    //template void mulMat<double, double, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void mulMat<double, double, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void mulMat<double, double, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void mulMat<double, double, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void mulMat<double, double, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void mulMat<double, double, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void mulMat<double, double, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
#endif
}

//////////////////////////////////////////////////////////////////////////
// mulScalar

namespace arithm
{
    template <typename T, typename S, typename D> struct MulScalar : unary_function<T, D>
    {
        S val;

        explicit MulScalar(S val_) : val(val_) {}

        __device__ __forceinline__ D operator ()(T a) const
        {
            return saturate_cast<D>(a * val);
        }
    };
}

namespace cv { namespace gpu { namespace device
{
    template <typename T, typename S, typename D> struct TransformFunctorTraits< arithm::MulScalar<T, S, D> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(D)>
    {
    };
}}}

namespace arithm
{
    template <typename T, typename S, typename D>
    void mulScalar(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream)
    {
        MulScalar<T, S, D> op(static_cast<S>(val));
        transform((PtrStepSz<T>) src1, (PtrStepSz<D>) dst, op, WithOutMask(), stream);
    }

    template void mulScalar<uchar, float, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void mulScalar<uchar, float, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void mulScalar<uchar, float, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void mulScalar<uchar, float, short>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void mulScalar<uchar, float, int>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void mulScalar<uchar, float, float>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void mulScalar<uchar, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);

    template void mulScalar<schar, float, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void mulScalar<schar, float, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void mulScalar<schar, float, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void mulScalar<schar, float, short>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void mulScalar<schar, float, int>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void mulScalar<schar, float, float>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void mulScalar<schar, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);

    //template void mulScalar<ushort, float, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void mulScalar<ushort, float, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void mulScalar<ushort, float, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void mulScalar<ushort, float, short>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void mulScalar<ushort, float, int>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void mulScalar<ushort, float, float>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void mulScalar<ushort, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);

    //template void mulScalar<short, float, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void mulScalar<short, float, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void mulScalar<short, float, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void mulScalar<short, float, short>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void mulScalar<short, float, int>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void mulScalar<short, float, float>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void mulScalar<short, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);

    //template void mulScalar<int, float, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void mulScalar<int, float, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void mulScalar<int, float, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void mulScalar<int, float, short>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void mulScalar<int, float, int>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void mulScalar<int, float, float>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void mulScalar<int, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);

    //template void mulScalar<float, float, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void mulScalar<float, float, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void mulScalar<float, float, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void mulScalar<float, float, short>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void mulScalar<float, float, int>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
#endif
    template void mulScalar<float, float, float>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void mulScalar<float, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);

    //template void mulScalar<double, double, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void mulScalar<double, double, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void mulScalar<double, double, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void mulScalar<double, double, short>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void mulScalar<double, double, int>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void mulScalar<double, double, float>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void mulScalar<double, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
#endif
}

//////////////////////////////////////////////////////////////////////////
// divMat

namespace arithm
{
    struct Div_8uc4_32f : binary_function<uint, float, uint>
    {
        __device__ __forceinline__ uint operator ()(uint a, float b) const
        {
            uint res = 0;

            if (b != 0)
            {
                b = 1.0f / b;
                res |= (saturate_cast<uchar>((0xffu & (a      )) * b)      );
                res |= (saturate_cast<uchar>((0xffu & (a >>  8)) * b) <<  8);
                res |= (saturate_cast<uchar>((0xffu & (a >> 16)) * b) << 16);
                res |= (saturate_cast<uchar>((0xffu & (a >> 24)) * b) << 24);
            }

            return res;
        }
    };

    struct Div_16sc4_32f : binary_function<short4, float, short4>
    {
        __device__ __forceinline__ short4 operator ()(short4 a, float b) const
        {
            return b != 0 ? make_short4(saturate_cast<short>(a.x / b), saturate_cast<short>(a.y / b),
                                        saturate_cast<short>(a.z / b), saturate_cast<short>(a.w / b))
                          : make_short4(0,0,0,0);
        }
    };

    template <typename T, typename D> struct Div : binary_function<T, T, D>
    {
        __device__ __forceinline__ D operator ()(T a, T b) const
        {
            return b != 0 ? saturate_cast<D>(a / b) : 0;
        }

        __host__ __device__ __forceinline__ Div() {}
        __host__ __device__ __forceinline__ Div(const Div&) {}
    };
    template <typename T> struct Div<T, float> : binary_function<T, T, float>
    {
        __device__ __forceinline__ float operator ()(T a, T b) const
        {
            return b != 0 ? static_cast<float>(a) / b : 0;
        }

        __host__ __device__ __forceinline__ Div() {}
        __host__ __device__ __forceinline__ Div(const Div&) {}
    };
    template <typename T> struct Div<T, double> : binary_function<T, T, double>
    {
        __device__ __forceinline__ double operator ()(T a, T b) const
        {
            return b != 0 ? static_cast<double>(a) / b : 0;
        }

        __host__ __device__ __forceinline__ Div() {}
        __host__ __device__ __forceinline__ Div(const Div&) {}
    };

    template <typename T, typename S, typename D> struct DivScale : binary_function<T, T, D>
    {
        S scale;

        explicit DivScale(S scale_) : scale(scale_) {}

        __device__ __forceinline__ D operator ()(T a, T b) const
        {
            return b != 0 ? saturate_cast<D>(scale * a / b) : 0;
        }
    };
}

namespace cv { namespace gpu { namespace device
{
    template <> struct TransformFunctorTraits<arithm::Div_8uc4_32f> : arithm::ArithmFuncTraits<sizeof(uint), sizeof(uint)>
    {
    };

    template <typename T, typename D> struct TransformFunctorTraits< arithm::Div<T, D> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(D)>
    {
    };

    template <typename T, typename S, typename D> struct TransformFunctorTraits< arithm::DivScale<T, S, D> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(D)>
    {
    };
}}}

namespace arithm
{
    void divMat_8uc4_32f(PtrStepSz<uint> src1, PtrStepSzf src2, PtrStepSz<uint> dst, cudaStream_t stream)
    {
        transform(src1, src2, dst, Div_8uc4_32f(), WithOutMask(), stream);
    }

    void divMat_16sc4_32f(PtrStepSz<short4> src1, PtrStepSzf src2, PtrStepSz<short4> dst, cudaStream_t stream)
    {
        transform(src1, src2, dst, Div_16sc4_32f(), WithOutMask(), stream);
    }

    template <typename T, typename S, typename D>
    void divMat(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream)
    {
        if (scale == 1)
        {
            Div<T, D> op;
            transform((PtrStepSz<T>) src1, (PtrStepSz<T>) src2, (PtrStepSz<D>) dst, op, WithOutMask(), stream);
        }
        else
        {
            DivScale<T, S, D> op(static_cast<S>(scale));
            transform((PtrStepSz<T>) src1, (PtrStepSz<T>) src2, (PtrStepSz<D>) dst, op, WithOutMask(), stream);
        }
    }

    template void divMat<uchar, float, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void divMat<uchar, float, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<uchar, float, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<uchar, float, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<uchar, float, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<uchar, float, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<uchar, double, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);

    template void divMat<schar, float, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<schar, float, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<schar, float, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<schar, float, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<schar, float, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<schar, float, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<schar, double, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);

    //template void divMat<ushort, float, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void divMat<ushort, float, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<ushort, float, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<ushort, float, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<ushort, float, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<ushort, float, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<ushort, double, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);

    //template void divMat<short, float, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void divMat<short, float, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<short, float, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<short, float, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<short, float, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<short, float, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<short, double, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);

    //template void divMat<int, float, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void divMat<int, float, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void divMat<int, float, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void divMat<int, float, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<int, float, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<int, float, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<int, double, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);

    //template void divMat<float, float, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void divMat<float, float, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void divMat<float, float, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void divMat<float, float, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void divMat<float, float, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
#endif
    template void divMat<float, float, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void divMat<float, double, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);

    //template void divMat<double, double, uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void divMat<double, double, schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void divMat<double, double, ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void divMat<double, double, short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void divMat<double, double, int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    //template void divMat<double, double, float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    template void divMat<double, double, double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
#endif
}

//////////////////////////////////////////////////////////////////////////
// divScalar

namespace arithm
{
    template <typename T, typename S, typename D>
    void divScalar(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream)
    {
        MulScalar<T, S, D> op(static_cast<S>(1.0 / val));
        transform((PtrStepSz<T>) src1, (PtrStepSz<D>) dst, op, WithOutMask(), stream);
    }

    template void divScalar<uchar, float, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void divScalar<uchar, float, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<uchar, float, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<uchar, float, short>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<uchar, float, int>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<uchar, float, float>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<uchar, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);

    template void divScalar<schar, float, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<schar, float, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<schar, float, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<schar, float, short>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<schar, float, int>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<schar, float, float>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<schar, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);

    //template void divScalar<ushort, float, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void divScalar<ushort, float, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<ushort, float, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<ushort, float, short>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<ushort, float, int>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<ushort, float, float>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<ushort, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);

    //template void divScalar<short, float, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void divScalar<short, float, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<short, float, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<short, float, short>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<short, float, int>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<short, float, float>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<short, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);

    //template void divScalar<int, float, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void divScalar<int, float, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void divScalar<int, float, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void divScalar<int, float, short>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<int, float, int>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<int, float, float>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<int, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);

    //template void divScalar<float, float, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void divScalar<float, float, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void divScalar<float, float, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void divScalar<float, float, short>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void divScalar<float, float, int>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
#endif
    template void divScalar<float, float, float>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void divScalar<float, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);

    //template void divScalar<double, double, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void divScalar<double, double, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void divScalar<double, double, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void divScalar<double, double, short>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void divScalar<double, double, int>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void divScalar<double, double, float>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divScalar<double, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
#endif
}

//////////////////////////////////////////////////////////////////////////
// divInv

namespace arithm
{
    template <typename T, typename S, typename D> struct DivInv : unary_function<T, D>
    {
        S val;

        explicit DivInv(S val_) : val(val_) {}

        __device__ __forceinline__ D operator ()(T a) const
        {
            return a != 0 ? saturate_cast<D>(val / a) : 0;
        }
    };
}

namespace cv { namespace gpu { namespace device
{
    template <typename T, typename S, typename D> struct TransformFunctorTraits< arithm::DivInv<T, S, D> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(D)>
    {
    };
}}}

namespace arithm
{
    template <typename T, typename S, typename D>
    void divInv(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream)
    {
        DivInv<T, S, D> op(static_cast<S>(val));
        transform((PtrStepSz<T>) src1, (PtrStepSz<D>) dst, op, WithOutMask(), stream);
    }

    template void divInv<uchar, float, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void divInv<uchar, float, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divInv<uchar, float, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divInv<uchar, float, short>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divInv<uchar, float, int>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divInv<uchar, float, float>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divInv<uchar, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);

    template void divInv<schar, float, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divInv<schar, float, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divInv<schar, float, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divInv<schar, float, short>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divInv<schar, float, int>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divInv<schar, float, float>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divInv<schar, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);

    //template void divInv<ushort, float, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void divInv<ushort, float, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divInv<ushort, float, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divInv<ushort, float, short>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divInv<ushort, float, int>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divInv<ushort, float, float>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divInv<ushort, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);

    //template void divInv<short, float, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void divInv<short, float, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divInv<short, float, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divInv<short, float, short>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divInv<short, float, int>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divInv<short, float, float>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divInv<short, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);

    //template void divInv<int, float, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void divInv<int, float, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void divInv<int, float, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void divInv<int, float, short>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divInv<int, float, int>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divInv<int, float, float>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divInv<int, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);

    //template void divInv<float, float, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void divInv<float, float, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void divInv<float, float, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void divInv<float, float, short>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void divInv<float, float, int>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
#endif
    template void divInv<float, float, float>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void divInv<float, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);

    //template void divInv<double, double, uchar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void divInv<double, double, schar>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void divInv<double, double, ushort>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void divInv<double, double, short>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void divInv<double, double, int>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    //template void divInv<double, double, float>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    template void divInv<double, double, double>(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
#endif
}

//////////////////////////////////////////////////////////////////////////
// absDiffMat

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

    ////////////////////////////////////

    struct VAbsDiff2 : binary_function<uint, uint, uint>
    {
        __device__ __forceinline__ uint operator ()(uint a, uint b) const
        {
            return vabsdiff2(a, b);
        }

        __host__ __device__ __forceinline__ VAbsDiff2() {}
        __host__ __device__ __forceinline__ VAbsDiff2(const VAbsDiff2&) {}
    };

    ////////////////////////////////////

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

namespace cv { namespace gpu { namespace device
{
    template <> struct TransformFunctorTraits< arithm::VAbsDiff4 > : arithm::ArithmFuncTraits<sizeof(uint), sizeof(uint)>
    {
    };

    ////////////////////////////////////

    template <> struct TransformFunctorTraits< arithm::VAbsDiff2 > : arithm::ArithmFuncTraits<sizeof(uint), sizeof(uint)>
    {
    };

    ////////////////////////////////////

    template <typename T> struct TransformFunctorTraits< arithm::AbsDiffMat<T> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(T)>
    {
    };
}}}

namespace arithm
{
    void absDiffMat_v4(PtrStepSz<uint> src1, PtrStepSz<uint> src2, PtrStepSz<uint> dst, cudaStream_t stream)
    {
        transform(src1, src2, dst, VAbsDiff4(), WithOutMask(), stream);
    }

    void absDiffMat_v2(PtrStepSz<uint> src1, PtrStepSz<uint> src2, PtrStepSz<uint> dst, cudaStream_t stream)
    {
        transform(src1, src2, dst, VAbsDiff2(), WithOutMask(), stream);
    }

    template <typename T>
    void absDiffMat(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream)
    {
        transform((PtrStepSz<T>) src1, (PtrStepSz<T>) src2, (PtrStepSz<T>) dst, AbsDiffMat<T>(), WithOutMask(), stream);
    }

    template void absDiffMat<uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void absDiffMat<schar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void absDiffMat<ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void absDiffMat<short>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void absDiffMat<int>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
#endif
    template void absDiffMat<float>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void absDiffMat<double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
#endif
}

//////////////////////////////////////////////////////////////////////////
// absDiffScalar

namespace arithm
{
    template <typename T, typename S> struct AbsDiffScalar : unary_function<T, T>
    {
        S val;

        explicit AbsDiffScalar(S val_) : val(val_) {}

        __device__ __forceinline__ T operator ()(T a) const
        {
            abs_func<S> f;
            return saturate_cast<T>(f(a - val));
        }
    };
}

namespace cv { namespace gpu { namespace device
{
    template <typename T, typename S> struct TransformFunctorTraits< arithm::AbsDiffScalar<T, S> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(T)>
    {
    };
}}}

namespace arithm
{
    template <typename T, typename S>
    void absDiffScalar(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream)
    {
        AbsDiffScalar<T, S> op(static_cast<S>(val));

        transform((PtrStepSz<T>) src1, (PtrStepSz<T>) dst, op, WithOutMask(), stream);
    }

    template void absDiffScalar<uchar, float>(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void absDiffScalar<schar, float>(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
    template void absDiffScalar<ushort, float>(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
    template void absDiffScalar<short, float>(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
    template void absDiffScalar<int, float>(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
#endif
    template void absDiffScalar<float, float>(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void absDiffScalar<double, double>(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
#endif
}

//////////////////////////////////////////////////////////////////////////
// absMat

namespace cv { namespace gpu { namespace device
{
    template <typename T> struct TransformFunctorTraits< abs_func<T> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(T)>
    {
    };
}}}

namespace arithm
{
    template <typename T>
    void absMat(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream)
    {
        transform((PtrStepSz<T>) src, (PtrStepSz<T>) dst, abs_func<T>(), WithOutMask(), stream);
    }

#ifndef OPENCV_TINY_GPU_MODULE
    template void absMat<uchar>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void absMat<schar>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void absMat<ushort>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void absMat<short>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void absMat<int>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
#endif
    template void absMat<float>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void absMat<double>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
#endif
}

//////////////////////////////////////////////////////////////////////////
// sqrMat

namespace arithm
{
    template <typename T> struct Sqr : unary_function<T, T>
    {
        __device__ __forceinline__ T operator ()(T x) const
        {
            return saturate_cast<T>(x * x);
        }

        __host__ __device__ __forceinline__ Sqr() {}
        __host__ __device__ __forceinline__ Sqr(const Sqr&) {}
    };
}

namespace cv { namespace gpu { namespace device
{
    template <typename T> struct TransformFunctorTraits< arithm::Sqr<T> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(T)>
    {
    };
}}}

namespace arithm
{
    template <typename T>
    void sqrMat(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream)
    {
        transform((PtrStepSz<T>) src, (PtrStepSz<T>) dst, Sqr<T>(), WithOutMask(), stream);
    }

#ifndef OPENCV_TINY_GPU_MODULE
    template void sqrMat<uchar>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void sqrMat<schar>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void sqrMat<ushort>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void sqrMat<short>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void sqrMat<int>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
#endif
    template void sqrMat<float>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void sqrMat<double>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
#endif
}

//////////////////////////////////////////////////////////////////////////
// sqrtMat

namespace cv { namespace gpu { namespace device
{
    template <typename T> struct TransformFunctorTraits< sqrt_func<T> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(T)>
    {
    };
}}}

namespace arithm
{
    template <typename T>
    void sqrtMat(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream)
    {
        transform((PtrStepSz<T>) src, (PtrStepSz<T>) dst, sqrt_func<T>(), WithOutMask(), stream);
    }

#ifndef OPENCV_TINY_GPU_MODULE
    template void sqrtMat<uchar>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void sqrtMat<schar>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void sqrtMat<ushort>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void sqrtMat<short>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void sqrtMat<int>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
#endif
    template void sqrtMat<float>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void sqrtMat<double>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
#endif
}

//////////////////////////////////////////////////////////////////////////
// logMat

namespace cv { namespace gpu { namespace device
{
    template <typename T> struct TransformFunctorTraits< log_func<T> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(T)>
    {
    };
}}}

namespace arithm
{
    template <typename T>
    void logMat(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream)
    {
        transform((PtrStepSz<T>) src, (PtrStepSz<T>) dst, log_func<T>(), WithOutMask(), stream);
    }

#ifndef OPENCV_TINY_GPU_MODULE
    template void logMat<uchar>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void logMat<schar>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void logMat<ushort>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void logMat<short>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void logMat<int>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
#endif
    template void logMat<float>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void logMat<double>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
#endif
}

//////////////////////////////////////////////////////////////////////////
// expMat

namespace arithm
{
    template <typename T> struct Exp : unary_function<T, T>
    {
        __device__ __forceinline__ T operator ()(T x) const
        {
            exp_func<T> f;
            return saturate_cast<T>(f(x));
        }

        __host__ __device__ __forceinline__ Exp() {}
        __host__ __device__ __forceinline__ Exp(const Exp&) {}
    };
}

namespace cv { namespace gpu { namespace device
{
    template <typename T> struct TransformFunctorTraits< arithm::Exp<T> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(T)>
    {
    };
}}}

namespace arithm
{
    template <typename T>
    void expMat(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream)
    {
        transform((PtrStepSz<T>) src, (PtrStepSz<T>) dst, Exp<T>(), WithOutMask(), stream);
    }

#ifndef OPENCV_TINY_GPU_MODULE
    template void expMat<uchar>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void expMat<schar>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void expMat<ushort>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void expMat<short>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    template void expMat<int>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
#endif
    template void expMat<float>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void expMat<double>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
#endif
}

//////////////////////////////////////////////////////////////////////////////////////
// cmpMat

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

    ////////////////////////////////////

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

namespace cv { namespace gpu { namespace device
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

    ////////////////////////////////////

    template <class Op, typename T> struct TransformFunctorTraits< arithm::Cmp<Op, T> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(uchar)>
    {
    };
}}}

namespace arithm
{
    void cmpMatEq_v4(PtrStepSz<uint> src1, PtrStepSz<uint> src2, PtrStepSz<uint> dst, cudaStream_t stream)
    {
        transform(src1, src2, dst, VCmpEq4(), WithOutMask(), stream);
    }
    void cmpMatNe_v4(PtrStepSz<uint> src1, PtrStepSz<uint> src2, PtrStepSz<uint> dst, cudaStream_t stream)
    {
        transform(src1, src2, dst, VCmpNe4(), WithOutMask(), stream);
    }
    void cmpMatLt_v4(PtrStepSz<uint> src1, PtrStepSz<uint> src2, PtrStepSz<uint> dst, cudaStream_t stream)
    {
        transform(src1, src2, dst, VCmpLt4(), WithOutMask(), stream);
    }
    void cmpMatLe_v4(PtrStepSz<uint> src1, PtrStepSz<uint> src2, PtrStepSz<uint> dst, cudaStream_t stream)
    {
        transform(src1, src2, dst, VCmpLe4(), WithOutMask(), stream);
    }

    template <template <typename> class Op, typename T>
    void cmpMat(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream)
    {
        Cmp<Op<T>, T> op;
        transform((PtrStepSz<T>) src1, (PtrStepSz<T>) src2, dst, op, WithOutMask(), stream);
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
#ifndef OPENCV_TINY_GPU_MODULE
    template void cmpMatEq<schar >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatEq<ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatEq<short >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatEq<int   >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
#endif
    template void cmpMatEq<float >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void cmpMatEq<double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
#endif

    template void cmpMatNe<uchar >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void cmpMatNe<schar >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatNe<ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatNe<short >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatNe<int   >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
#endif
    template void cmpMatNe<float >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void cmpMatNe<double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
#endif

    template void cmpMatLt<uchar >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void cmpMatLt<schar >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatLt<ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatLt<short >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatLt<int   >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
#endif
    template void cmpMatLt<float >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void cmpMatLt<double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
#endif

    template void cmpMatLe<uchar >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void cmpMatLe<schar >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatLe<ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatLe<short >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void cmpMatLe<int   >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
#endif
    template void cmpMatLe<float >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void cmpMatLe<double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
#endif
}

//////////////////////////////////////////////////////////////////////////////////////
// cmpScalar

namespace arithm
{
#define TYPE_VEC(type, cn) typename TypeVec<type, cn>::vec_type

    template <class Op, typename T, int cn> struct CmpScalar;
    template <class Op, typename T>
    struct CmpScalar<Op, T, 1> : unary_function<T, uchar>
    {
        const T val;

        __host__ explicit CmpScalar(T val_) : val(val_) {}

        __device__ __forceinline__ uchar operator()(T src) const
        {
            Cmp<Op, T> op;
            return op(src, val);
        }
    };
    template <class Op, typename T>
    struct CmpScalar<Op, T, 2> : unary_function<TYPE_VEC(T, 2), TYPE_VEC(uchar, 2)>
    {
        const TYPE_VEC(T, 2) val;

        __host__ explicit CmpScalar(TYPE_VEC(T, 2) val_) : val(val_) {}

        __device__ __forceinline__ TYPE_VEC(uchar, 2) operator()(const TYPE_VEC(T, 2) & src) const
        {
            Cmp<Op, T> op;
            return VecTraits<TYPE_VEC(uchar, 2)>::make(op(src.x, val.x), op(src.y, val.y));
        }
    };
    template <class Op, typename T>
    struct CmpScalar<Op, T, 3> : unary_function<TYPE_VEC(T, 3), TYPE_VEC(uchar, 3)>
    {
        const TYPE_VEC(T, 3) val;

        __host__ explicit CmpScalar(TYPE_VEC(T, 3) val_) : val(val_) {}

        __device__ __forceinline__ TYPE_VEC(uchar, 3) operator()(const TYPE_VEC(T, 3) & src) const
        {
            Cmp<Op, T> op;
            return VecTraits<TYPE_VEC(uchar, 3)>::make(op(src.x, val.x), op(src.y, val.y), op(src.z, val.z));
        }
    };
    template <class Op, typename T>
    struct CmpScalar<Op, T, 4> : unary_function<TYPE_VEC(T, 4), TYPE_VEC(uchar, 4)>
    {
        const TYPE_VEC(T, 4) val;

        __host__ explicit CmpScalar(TYPE_VEC(T, 4) val_) : val(val_) {}

        __device__ __forceinline__ TYPE_VEC(uchar, 4) operator()(const TYPE_VEC(T, 4) & src) const
        {
            Cmp<Op, T> op;
            return VecTraits<TYPE_VEC(uchar, 4)>::make(op(src.x, val.x), op(src.y, val.y), op(src.z, val.z), op(src.w, val.w));
        }
    };

#undef TYPE_VEC
}

namespace cv { namespace gpu { namespace device
{
    template <class Op, typename T> struct TransformFunctorTraits< arithm::CmpScalar<Op, T, 1> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(uchar)>
    {
    };
}}}

namespace arithm
{
    template <template <typename> class Op, typename T, int cn>
    void cmpScalar(PtrStepSzb src, double val[4], PtrStepSzb dst, cudaStream_t stream)
    {
        typedef typename TypeVec<T, cn>::vec_type src_t;
        typedef typename TypeVec<uchar, cn>::vec_type dst_t;

        T sval[] = {static_cast<T>(val[0]), static_cast<T>(val[1]), static_cast<T>(val[2]), static_cast<T>(val[3])};
        src_t val1 = VecTraits<src_t>::make(sval);

        CmpScalar<Op<T>, T, cn> op(val1);
        transform((PtrStepSz<src_t>) src, (PtrStepSz<dst_t>) dst, op, WithOutMask(), stream);
    }

    template <typename T> void cmpScalarEq(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream)
    {
        typedef void (*func_t)(PtrStepSzb src, double val[4], PtrStepSzb dst, cudaStream_t stream);
        static const func_t funcs[] =
        {
            0,
            cmpScalar<equal_to, T, 1>,
            cmpScalar<equal_to, T, 2>,
            cmpScalar<equal_to, T, 3>,
            cmpScalar<equal_to, T, 4>
        };

        funcs[cn](src, val, dst, stream);
    }
    template <typename T> void cmpScalarNe(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream)
    {
        typedef void (*func_t)(PtrStepSzb src, double val[4], PtrStepSzb dst, cudaStream_t stream);
        static const func_t funcs[] =
        {
            0,
            cmpScalar<not_equal_to, T, 1>,
            cmpScalar<not_equal_to, T, 2>,
            cmpScalar<not_equal_to, T, 3>,
            cmpScalar<not_equal_to, T, 4>
        };

        funcs[cn](src, val, dst, stream);
    }
    template <typename T> void cmpScalarLt(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream)
    {
        typedef void (*func_t)(PtrStepSzb src, double val[4], PtrStepSzb dst, cudaStream_t stream);
        static const func_t funcs[] =
        {
            0,
            cmpScalar<less, T, 1>,
            cmpScalar<less, T, 2>,
            cmpScalar<less, T, 3>,
            cmpScalar<less, T, 4>
        };

        funcs[cn](src, val, dst, stream);
    }
    template <typename T> void cmpScalarLe(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream)
    {
        typedef void (*func_t)(PtrStepSzb src, double val[4], PtrStepSzb dst, cudaStream_t stream);
        static const func_t funcs[] =
        {
            0,
            cmpScalar<less_equal, T, 1>,
            cmpScalar<less_equal, T, 2>,
            cmpScalar<less_equal, T, 3>,
            cmpScalar<less_equal, T, 4>
        };

        funcs[cn](src, val, dst, stream);
    }
    template <typename T> void cmpScalarGt(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream)
    {
        typedef void (*func_t)(PtrStepSzb src, double val[4], PtrStepSzb dst, cudaStream_t stream);
        static const func_t funcs[] =
        {
            0,
            cmpScalar<greater, T, 1>,
            cmpScalar<greater, T, 2>,
            cmpScalar<greater, T, 3>,
            cmpScalar<greater, T, 4>
        };

        funcs[cn](src, val, dst, stream);
    }
    template <typename T> void cmpScalarGe(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream)
    {
        typedef void (*func_t)(PtrStepSzb src, double val[4], PtrStepSzb dst, cudaStream_t stream);
        static const func_t funcs[] =
        {
            0,
            cmpScalar<greater_equal, T, 1>,
            cmpScalar<greater_equal, T, 2>,
            cmpScalar<greater_equal, T, 3>,
            cmpScalar<greater_equal, T, 4>
        };

        funcs[cn](src, val, dst, stream);
    }

    template void cmpScalarEq<uchar >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void cmpScalarEq<schar >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarEq<ushort>(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarEq<short >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarEq<int   >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
#endif
    template void cmpScalarEq<float >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void cmpScalarEq<double>(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
#endif

    template void cmpScalarNe<uchar >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void cmpScalarNe<schar >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarNe<ushort>(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarNe<short >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarNe<int   >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
#endif
    template void cmpScalarNe<float >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void cmpScalarNe<double>(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
#endif

    template void cmpScalarLt<uchar >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void cmpScalarLt<schar >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarLt<ushort>(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarLt<short >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarLt<int   >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
#endif
    template void cmpScalarLt<float >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void cmpScalarLt<double>(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
#endif

    template void cmpScalarLe<uchar >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void cmpScalarLe<schar >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarLe<ushort>(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarLe<short >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarLe<int   >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
#endif
    template void cmpScalarLe<float >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void cmpScalarLe<double>(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
#endif

    template void cmpScalarGt<uchar >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void cmpScalarGt<schar >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarGt<ushort>(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarGt<short >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarGt<int   >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
#endif
    template void cmpScalarGt<float >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void cmpScalarGt<double>(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
#endif

    template void cmpScalarGe<uchar >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void cmpScalarGe<schar >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarGe<ushort>(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarGe<short >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template void cmpScalarGe<int   >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
#endif
    template void cmpScalarGe<float >(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void cmpScalarGe<double>(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
#endif
}

//////////////////////////////////////////////////////////////////////////////////////
// bitMat

namespace cv { namespace gpu { namespace device
{
    template <typename T> struct TransformFunctorTraits< bit_not<T> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(T)>
    {
    };

    template <typename T> struct TransformFunctorTraits< bit_and<T> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(T)>
    {
    };

    template <typename T> struct TransformFunctorTraits< bit_or<T> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(T)>
    {
    };

    template <typename T> struct TransformFunctorTraits< bit_xor<T> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(T)>
    {
    };
}}}

namespace arithm
{
    template <typename T> void bitMatNot(PtrStepSzb src, PtrStepSzb dst, PtrStepb mask, int num_channels, cudaStream_t stream)
    {
        if (mask.data)
            transform((PtrStepSz<T>) src, (PtrStepSz<T>) dst, bit_not<T>(), SingleMaskChannels(mask, num_channels), stream);
        else
            transform((PtrStepSz<T>) src, (PtrStepSz<T>) dst, bit_not<T>(), WithOutMask(), stream);
    }

    template <typename T> void bitMatAnd(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, int num_channels, cudaStream_t stream)
    {
        if (mask.data)
            transform((PtrStepSz<T>) src1, (PtrStepSz<T>) src2, (PtrStepSz<T>) dst, bit_and<T>(), SingleMaskChannels(mask, num_channels), stream);
        else
            transform((PtrStepSz<T>) src1, (PtrStepSz<T>) src2, (PtrStepSz<T>) dst, bit_and<T>(), WithOutMask(), stream);
    }

    template <typename T> void bitMatOr(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, int num_channels, cudaStream_t stream)
    {
        if (mask.data)
            transform((PtrStepSz<T>) src1, (PtrStepSz<T>) src2, (PtrStepSz<T>) dst, bit_or<T>(), SingleMaskChannels(mask, num_channels), stream);
        else
            transform((PtrStepSz<T>) src1, (PtrStepSz<T>) src2, (PtrStepSz<T>) dst, bit_or<T>(), WithOutMask(), stream);
    }

    template <typename T> void bitMatXor(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, int num_channels, cudaStream_t stream)
    {
        if (mask.data)
            transform((PtrStepSz<T>) src1, (PtrStepSz<T>) src2, (PtrStepSz<T>) dst, bit_xor<T>(), SingleMaskChannels(mask, num_channels), stream);
        else
            transform((PtrStepSz<T>) src1, (PtrStepSz<T>) src2, (PtrStepSz<T>) dst, bit_xor<T>(), WithOutMask(), stream);
    }

    template void bitMatNot<uchar>(PtrStepSzb src, PtrStepSzb dst, PtrStepb mask, int num_channels, cudaStream_t stream);
    template void bitMatNot<ushort>(PtrStepSzb src, PtrStepSzb dst, PtrStepb mask, int num_channels, cudaStream_t stream);
    template void bitMatNot<uint>(PtrStepSzb src, PtrStepSzb dst, PtrStepb mask, int num_channels, cudaStream_t stream);

    template void bitMatAnd<uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, int num_channels, cudaStream_t stream);
    template void bitMatAnd<ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, int num_channels, cudaStream_t stream);
    template void bitMatAnd<uint>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, int num_channels, cudaStream_t stream);

    template void bitMatOr<uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, int num_channels, cudaStream_t stream);
    template void bitMatOr<ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, int num_channels, cudaStream_t stream);
    template void bitMatOr<uint>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, int num_channels, cudaStream_t stream);

    template void bitMatXor<uchar>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, int num_channels, cudaStream_t stream);
    template void bitMatXor<ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, int num_channels, cudaStream_t stream);
    template void bitMatXor<uint>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, int num_channels, cudaStream_t stream);
}

//////////////////////////////////////////////////////////////////////////////////////
// bitScalar

namespace cv { namespace gpu { namespace device
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
        transform((PtrStepSz<T>) src1, (PtrStepSz<T>) dst, cv::gpu::device::bind2nd(bit_and<T>(), src2), WithOutMask(), stream);
    }

    template <typename T> void bitScalarOr(PtrStepSzb src1, uint src2, PtrStepSzb dst, cudaStream_t stream)
    {
        transform((PtrStepSz<T>) src1, (PtrStepSz<T>) dst, cv::gpu::device::bind2nd(bit_or<T>(), src2), WithOutMask(), stream);
    }

    template <typename T> void bitScalarXor(PtrStepSzb src1, uint src2, PtrStepSzb dst, cudaStream_t stream)
    {
        transform((PtrStepSz<T>) src1, (PtrStepSz<T>) dst, cv::gpu::device::bind2nd(bit_xor<T>(), src2), WithOutMask(), stream);
    }

    template void bitScalarAnd<uchar>(PtrStepSzb src1, uint src2, PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void bitScalarAnd<ushort>(PtrStepSzb src1, uint src2, PtrStepSzb dst, cudaStream_t stream);
    template void bitScalarAnd<int>(PtrStepSzb src1, uint src2, PtrStepSzb dst, cudaStream_t stream);
    template void bitScalarAnd<unsigned int>(PtrStepSzb src1, uint src2, PtrStepSzb dst, cudaStream_t stream);
#endif

    template void bitScalarOr<uchar>(PtrStepSzb src1, uint src2, PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void bitScalarOr<ushort>(PtrStepSzb src1, uint src2, PtrStepSzb dst, cudaStream_t stream);
    template void bitScalarOr<int>(PtrStepSzb src1, uint src2, PtrStepSzb dst, cudaStream_t stream);
    template void bitScalarOr<unsigned int>(PtrStepSzb src1, uint src2, PtrStepSzb dst, cudaStream_t stream);
#endif

    template void bitScalarXor<uchar>(PtrStepSzb src1, uint src2, PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void bitScalarXor<ushort>(PtrStepSzb src1, uint src2, PtrStepSzb dst, cudaStream_t stream);
    template void bitScalarXor<int>(PtrStepSzb src1, uint src2, PtrStepSzb dst, cudaStream_t stream);
    template void bitScalarXor<unsigned int>(PtrStepSzb src1, uint src2, PtrStepSzb dst, cudaStream_t stream);
#endif
}

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

    ////////////////////////////////////

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

namespace cv { namespace gpu { namespace device
{
    template <> struct TransformFunctorTraits< arithm::VMin4 > : arithm::ArithmFuncTraits<sizeof(uint), sizeof(uint)>
    {
    };

    ////////////////////////////////////

    template <> struct TransformFunctorTraits< arithm::VMin2 > : arithm::ArithmFuncTraits<sizeof(uint), sizeof(uint)>
    {
    };

    ////////////////////////////////////

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
        transform(src1, src2, dst, VMin4(), WithOutMask(), stream);
    }

    void minMat_v2(PtrStepSz<uint> src1, PtrStepSz<uint> src2, PtrStepSz<uint> dst, cudaStream_t stream)
    {
        transform(src1, src2, dst, VMin2(), WithOutMask(), stream);
    }

    template <typename T> void minMat(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream)
    {
        transform((PtrStepSz<T>) src1, (PtrStepSz<T>) src2, (PtrStepSz<T>) dst, minimum<T>(), WithOutMask(), stream);
    }

    template void minMat<uchar >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void minMat<schar >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void minMat<ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void minMat<short >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void minMat<int   >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
#endif
    template void minMat<float >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void minMat<double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
#endif

    template <typename T> void minScalar(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream)
    {
        transform((PtrStepSz<T>) src1, (PtrStepSz<T>) dst, cv::gpu::device::bind2nd(minimum<T>(), src2), WithOutMask(), stream);
    }

#ifdef OPENCV_TINY_GPU_MODULE
    template void minScalar<uchar >(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
    template void minScalar<int   >(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
    template void minScalar<float >(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
#else
    template void minScalar<uchar >(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
    template void minScalar<schar >(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
    template void minScalar<ushort>(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
    template void minScalar<short >(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
    template void minScalar<int   >(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
    template void minScalar<float >(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
    template void minScalar<double>(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
#endif
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

    ////////////////////////////////////

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

namespace cv { namespace gpu { namespace device
{
    template <> struct TransformFunctorTraits< arithm::VMax4 > : arithm::ArithmFuncTraits<sizeof(uint), sizeof(uint)>
    {
    };

    ////////////////////////////////////

    template <> struct TransformFunctorTraits< arithm::VMax2 > : arithm::ArithmFuncTraits<sizeof(uint), sizeof(uint)>
    {
    };

    ////////////////////////////////////

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
        transform(src1, src2, dst, VMax4(), WithOutMask(), stream);
    }

    void maxMat_v2(PtrStepSz<uint> src1, PtrStepSz<uint> src2, PtrStepSz<uint> dst, cudaStream_t stream)
    {
        transform(src1, src2, dst, VMax2(), WithOutMask(), stream);
    }

    template <typename T> void maxMat(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream)
    {
        transform((PtrStepSz<T>) src1, (PtrStepSz<T>) src2, (PtrStepSz<T>) dst, maximum<T>(), WithOutMask(), stream);
    }

    template void maxMat<uchar >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void maxMat<schar >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void maxMat<ushort>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void maxMat<short >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template void maxMat<int   >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
#endif
    template void maxMat<float >(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void maxMat<double>(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
#endif

    template <typename T> void maxScalar(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream)
    {
        transform((PtrStepSz<T>) src1, (PtrStepSz<T>) dst, cv::gpu::device::bind2nd(maximum<T>(), src2), WithOutMask(), stream);
    }

    template void maxScalar<uchar >(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void maxScalar<schar >(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
    template void maxScalar<ushort>(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
    template void maxScalar<short >(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
    template void maxScalar<int   >(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
#endif
    template void maxScalar<float >(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void maxScalar<double>(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
#endif
}

//////////////////////////////////////////////////////////////////////////
// threshold

namespace cv { namespace gpu { namespace device
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
        transform(src, dst, op, WithOutMask(), stream);
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
#ifndef OPENCV_TINY_GPU_MODULE
    template void threshold<schar>(PtrStepSzb src, PtrStepSzb dst, double thresh, double maxVal, int type, cudaStream_t stream);
    template void threshold<ushort>(PtrStepSzb src, PtrStepSzb dst, double thresh, double maxVal, int type, cudaStream_t stream);
    template void threshold<short>(PtrStepSzb src, PtrStepSzb dst, double thresh, double maxVal, int type, cudaStream_t stream);
    template void threshold<int>(PtrStepSzb src, PtrStepSzb dst, double thresh, double maxVal, int type, cudaStream_t stream);
#endif
    template void threshold<float>(PtrStepSzb src, PtrStepSzb dst, double thresh, double maxVal, int type, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void threshold<double>(PtrStepSzb src, PtrStepSzb dst, double thresh, double maxVal, int type, cudaStream_t stream);
#endif
}

//////////////////////////////////////////////////////////////////////////
// pow

namespace arithm
{
    template<typename T, bool Signed = numeric_limits<T>::is_signed> struct PowOp : unary_function<T, T>
    {
        float power;

        PowOp(double power_) : power(static_cast<float>(power_)) {}

        __device__ __forceinline__ T operator()(T e) const
        {
            return saturate_cast<T>(__powf((float)e, power));
        }
    };
    template<typename T> struct PowOp<T, true> : unary_function<T, T>
    {
        float power;

        PowOp(double power_) : power(static_cast<float>(power_)) {}

        __device__ __forceinline__ T operator()(T e) const
        {
            T res = saturate_cast<T>(__powf((float)e, power));

            if ((e < 0) && (1 & static_cast<int>(power)))
                res *= -1;

            return res;
        }
    };
    template<> struct PowOp<float> : unary_function<float, float>
    {
        const float power;

        PowOp(double power_) : power(static_cast<float>(power_)) {}

        __device__ __forceinline__ float operator()(float e) const
        {
            return __powf(::fabs(e), power);
        }
    };
    template<> struct PowOp<double> : unary_function<double, double>
    {
        double power;

        PowOp(double power_) : power(power_) {}

        __device__ __forceinline__ double operator()(double e) const
        {
            return ::pow(::fabs(e), power);
        }
    };
}

namespace cv { namespace gpu { namespace device
{
    template <typename T> struct TransformFunctorTraits< arithm::PowOp<T> > : arithm::ArithmFuncTraits<sizeof(T), sizeof(T)>
    {
    };
}}}

namespace arithm
{
    template<typename T>
    void pow(PtrStepSzb src, double power, PtrStepSzb dst, cudaStream_t stream)
    {
        transform((PtrStepSz<T>) src, (PtrStepSz<T>) dst, PowOp<T>(power), WithOutMask(), stream);
    }

#ifndef OPENCV_TINY_GPU_MODULE
    template void pow<uchar>(PtrStepSzb src, double power, PtrStepSzb dst, cudaStream_t stream);
    template void pow<schar>(PtrStepSzb src, double power, PtrStepSzb dst, cudaStream_t stream);
    template void pow<short>(PtrStepSzb src, double power, PtrStepSzb dst, cudaStream_t stream);
    template void pow<ushort>(PtrStepSzb src, double power, PtrStepSzb dst, cudaStream_t stream);
    template void pow<int>(PtrStepSzb src, double power, PtrStepSzb dst, cudaStream_t stream);
#endif
    template void pow<float>(PtrStepSzb src, double power, PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void pow<double>(PtrStepSzb src, double power, PtrStepSzb dst, cudaStream_t stream);
#endif
}

//////////////////////////////////////////////////////////////////////////
// addWeighted

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

        AddWeighted_(double alpha_, double beta_, double gamma_) : alpha(static_cast<float>(alpha_)), beta(static_cast<float>(beta_)), gamma(static_cast<float>(gamma_)) {}

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

        AddWeighted_(double alpha_, double beta_, double gamma_) : alpha(alpha_), beta(beta_), gamma(gamma_) {}

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

namespace cv { namespace gpu { namespace device
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

        transform((PtrStepSz<T1>) src1, (PtrStepSz<T2>) src2, (PtrStepSz<D>) dst, op, WithOutMask(), stream);
    }

    template void addWeighted<uchar, uchar, uchar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
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
#endif


#ifndef OPENCV_TINY_GPU_MODULE
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
#endif


#ifndef OPENCV_TINY_GPU_MODULE
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
#endif


#ifndef OPENCV_TINY_GPU_MODULE
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
#endif


#ifndef OPENCV_TINY_GPU_MODULE
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
#endif


#ifndef OPENCV_TINY_GPU_MODULE
    template void addWeighted<float, float, uchar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<float, float, schar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<float, float, ushort>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<float, float, short>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<float, float, int>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
#endif
    template void addWeighted<float, float, float>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
    template void addWeighted<float, float, double>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);

    template void addWeighted<float, double, uchar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<float, double, schar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<float, double, ushort>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<float, double, short>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<float, double, int>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<float, double, float>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<float, double, double>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
#endif



#ifndef OPENCV_TINY_GPU_MODULE
    template void addWeighted<double, double, uchar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<double, double, schar>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<double, double, ushort>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<double, double, short>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<double, double, int>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<double, double, float>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    template void addWeighted<double, double, double>(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
#endif
}

#endif /* CUDA_DISABLER */
