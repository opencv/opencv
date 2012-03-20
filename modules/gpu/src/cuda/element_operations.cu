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

#include "internal_shared.hpp"
#include "opencv2/gpu/device/functional.hpp"
#include "opencv2/gpu/device/vec_math.hpp"
#include "opencv2/gpu/device/transform.hpp"
#include "opencv2/gpu/device/limits.hpp"
#include "opencv2/gpu/device/saturate_cast.hpp"

namespace cv { namespace gpu { namespace device
{
    //////////////////////////////////////////////////////////////////////////
    // add

    template <typename T, typename D> struct Add : binary_function<T, T, D>
    {
        __device__ __forceinline__ D operator ()(T a, T b) const
        {
            return saturate_cast<D>(a + b);
        }
    };

    template <> struct TransformFunctorTraits< Add<ushort, ushort> > : DefaultTransformFunctorTraits< Add<ushort, ushort> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< Add<short, short> > : DefaultTransformFunctorTraits< Add<short, short> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< Add<int, int> > : DefaultTransformFunctorTraits< Add<int, int> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< Add<float, float> > : DefaultTransformFunctorTraits< Add<float, float> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };

    template <typename T, typename D> void add_gpu(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream)
    {
        if (mask.data)
            cv::gpu::device::transform((DevMem2D_<T>)src1, (DevMem2D_<T>)src2, (DevMem2D_<D>)dst, Add<T, D>(), SingleMask(mask), stream);
        else
            cv::gpu::device::transform((DevMem2D_<T>)src1, (DevMem2D_<T>)src2, (DevMem2D_<D>)dst, Add<T, D>(), WithOutMask(), stream);
    }

    template void add_gpu<uchar, uchar>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<uchar, schar>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<uchar, ushort>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<uchar, short>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<uchar, int>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<uchar, float>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<uchar, double>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);

    //template void add_gpu<schar, uchar>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<schar, schar>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<schar, ushort>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<schar, short>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<schar, int>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<schar, float>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<schar, double>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);

    //template void add_gpu<ushort, uchar>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<ushort, schar>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<ushort, ushort>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<ushort, short>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<ushort, int>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<ushort, float>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<ushort, double>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);

    //template void add_gpu<short, uchar>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<short, schar>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<short, ushort>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<short, short>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<short, int>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<short, float>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<short, double>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);

    //template void add_gpu<int, uchar>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<int, schar>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<int, ushort>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<int, short>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<int, int>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<int, float>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<int, double>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);

    //template void add_gpu<float, uchar>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<float, schar>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<float, ushort>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<float, short>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<float, int>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<float, float>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<float, double>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);

    //template void add_gpu<double, uchar>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<double, schar>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<double, ushort>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<double, short>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<double, int>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<double, float>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<double, double>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);

    template <typename T, typename D> struct AddScalar : unary_function<T, D>
    {
        AddScalar(double val_) : val(val_) {}
        __device__ __forceinline__ D operator ()(T a) const
        {
            return saturate_cast<D>(a + val);
        }
        const double val;
    };

    template <> struct TransformFunctorTraits< AddScalar<ushort, ushort> > : DefaultTransformFunctorTraits< AddScalar<ushort, ushort>  >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< AddScalar<short, short> > : DefaultTransformFunctorTraits< AddScalar<short, short> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< AddScalar<int, int> > : DefaultTransformFunctorTraits< AddScalar<int, int> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< AddScalar<float, float> > : DefaultTransformFunctorTraits< AddScalar<float, float> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };

    template <typename T, typename D> void add_gpu(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream)
    {
        cudaSafeCall( cudaSetDoubleForDevice(&val) );
        AddScalar<T, D> op(val);
        if (mask.data)
            cv::gpu::device::transform((DevMem2D_<T>)src1, (DevMem2D_<D>)dst, op, SingleMask(mask), stream);
        else
            cv::gpu::device::transform((DevMem2D_<T>)src1, (DevMem2D_<D>)dst, op, WithOutMask(), stream);
    }

    template void add_gpu<uchar, uchar>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<uchar, schar>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<uchar, ushort>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<uchar, short >(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<uchar, int   >(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<uchar, float >(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<uchar, double>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);

    //template void add_gpu<schar, uchar>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<schar, schar>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<schar, ushort>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<schar, short>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<schar, int>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<schar, float>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<schar, double>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);

    //template void add_gpu<ushort, uchar>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<ushort, schar>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<ushort, ushort>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<ushort, short>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<ushort, int>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<ushort, float>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<ushort, double>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);

    //template void add_gpu<short, uchar>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<short, schar>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<short, ushort>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<short, short>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<short, int>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<short, float>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<short, double>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);

    //template void add_gpu<int, uchar>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<int, schar>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<int, ushort>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<int, short>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<int, int>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<int, float>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<int, double>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);

    //template void add_gpu<float, uchar>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<float, schar>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<float, ushort>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<float, short>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<float, int>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<float, float>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<float, double>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);

    //template void add_gpu<double, uchar>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<double, schar>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<double, ushort>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<double, short>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<double, int>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void add_gpu<double, float>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void add_gpu<double, double>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);

    //////////////////////////////////////////////////////////////////////////
    // subtract

    template <typename T, typename D> struct Subtract : binary_function<T, T, D>
    {
        __device__ __forceinline__ D operator ()(T a, T b) const
        {
            return saturate_cast<D>(a - b);
        }
    };

    template <> struct TransformFunctorTraits< Subtract<ushort, ushort> > : DefaultTransformFunctorTraits< Subtract<ushort, ushort> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< Subtract<short, short> > : DefaultTransformFunctorTraits< Subtract<short, short> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< Subtract<int, int> > : DefaultTransformFunctorTraits< Subtract<int, int> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< Subtract<float, float> > : DefaultTransformFunctorTraits< Subtract<float, float> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };

    template <typename T, typename D> void subtract_gpu(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream)
    {
        if (mask.data)
            cv::gpu::device::transform((DevMem2D_<T>)src1, (DevMem2D_<T>)src2, (DevMem2D_<D>)dst, Subtract<T, D>(), SingleMask(mask), stream);
        else
            cv::gpu::device::transform((DevMem2D_<T>)src1, (DevMem2D_<T>)src2, (DevMem2D_<D>)dst, Subtract<T, D>(), WithOutMask(), stream);
    }

    template void subtract_gpu<uchar, uchar>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<uchar, schar>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<uchar, ushort>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<uchar, short>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<uchar, int>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<uchar, float>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<uchar, double>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);

    //template void subtract_gpu<schar, uchar>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<schar, schar>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<schar, ushort>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<schar, short>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<schar, int>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<schar, float>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<schar, double>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);

    //template void subtract_gpu<ushort, uchar>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<ushort, schar>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<ushort, ushort>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<ushort, short>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<ushort, int>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<ushort, float>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<ushort, double>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);

    //template void subtract_gpu<short, uchar>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<short, schar>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<short, ushort>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<short, short>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<short, int>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<short, float>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<short, double>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);

    //template void subtract_gpu<int, uchar>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<int, schar>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<int, ushort>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<int, short>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<int, int>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<int, float>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<int, double>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);

    //template void subtract_gpu<float, uchar>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<float, schar>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<float, ushort>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<float, short>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<float, int>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<float, float>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<float, double>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);

    //template void subtract_gpu<double, uchar>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<double, schar>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<double, ushort>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<double, short>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<double, int>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<double, float>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<double, double>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);

    template <typename T, typename D> struct SubtractScalar : unary_function<T, D>
    {
        SubtractScalar(double val_) : val(val_) {}
        __device__ __forceinline__ D operator ()(T a) const
        {
            return saturate_cast<D>(a - val);
        }
        const double val;
    };

    template <> struct TransformFunctorTraits< SubtractScalar<ushort, ushort> > : DefaultTransformFunctorTraits< SubtractScalar<ushort, ushort>  >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< SubtractScalar<short, short> > : DefaultTransformFunctorTraits< SubtractScalar<short, short> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< SubtractScalar<int, int> > : DefaultTransformFunctorTraits< SubtractScalar<int, int> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< SubtractScalar<float, float> > : DefaultTransformFunctorTraits< SubtractScalar<float, float> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };

    template <typename T, typename D> void subtract_gpu(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream)
    {
        cudaSafeCall( cudaSetDoubleForDevice(&val) );
        SubtractScalar<T, D> op(val);
        if (mask.data)
            cv::gpu::device::transform((DevMem2D_<T>)src1, (DevMem2D_<D>)dst, op, SingleMask(mask), stream);
        else
            cv::gpu::device::transform((DevMem2D_<T>)src1, (DevMem2D_<D>)dst, op, WithOutMask(), stream);
    }

    template void subtract_gpu<uchar, uchar>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<uchar, schar>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<uchar, ushort>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<uchar, short >(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<uchar, int   >(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<uchar, float >(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<uchar, double>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);

    //template void subtract_gpu<schar, uchar>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<schar, schar>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<schar, ushort>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<schar, short>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<schar, int>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<schar, float>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<schar, double>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);

    //template void subtract_gpu<ushort, uchar>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<ushort, schar>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<ushort, ushort>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<ushort, short>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<ushort, int>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<ushort, float>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<ushort, double>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);

    //template void subtract_gpu<short, uchar>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<short, schar>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<short, ushort>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<short, short>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<short, int>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<short, float>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<short, double>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);

    //template void subtract_gpu<int, uchar>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<int, schar>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<int, ushort>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<int, short>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<int, int>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<int, float>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<int, double>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);

    //template void subtract_gpu<float, uchar>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<float, schar>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<float, ushort>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<float, short>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<float, int>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<float, float>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<float, double>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);

    //template void subtract_gpu<double, uchar>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<double, schar>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<double, ushort>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<double, short>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<double, int>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    //template void subtract_gpu<double, float>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
    template void subtract_gpu<double, double>(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);

    //////////////////////////////////////////////////////////////////////////
    // multiply

    struct multiply_8uc4_32f : binary_function<uint, float, uint>
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
    };

    OPENCV_GPU_TRANSFORM_FUNCTOR_TRAITS(multiply_8uc4_32f)
    {
        enum { smart_block_dim_x = 8 };
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 8 };
    };

    void multiply_gpu(const DevMem2D_<uchar4>& src1, const DevMem2Df& src2, const DevMem2D_<uchar4>& dst, cudaStream_t stream)
    {
        cv::gpu::device::transform(static_cast< DevMem2D_<uint> >(src1), src2, static_cast< DevMem2D_<uint> >(dst), multiply_8uc4_32f(), WithOutMask(), stream);
    }

    struct multiply_16sc4_32f : binary_function<short4, float, short4>
    {
        __device__ __forceinline__ short4 operator ()(short4 a, float b) const
        {
            return make_short4(saturate_cast<short>(a.x * b), saturate_cast<short>(a.y * b),
                               saturate_cast<short>(a.z * b), saturate_cast<short>(a.w * b));
        }
    };

    OPENCV_GPU_TRANSFORM_FUNCTOR_TRAITS(multiply_16sc4_32f)
    {
        enum { smart_block_dim_x = 8 };
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 8 };
    };

    void multiply_gpu(const DevMem2D_<short4>& src1, const DevMem2Df& src2, const DevMem2D_<short4>& dst, cudaStream_t stream)
    {
        cv::gpu::device::transform(static_cast< DevMem2D_<short4> >(src1), src2, static_cast< DevMem2D_<short4> >(dst), multiply_16sc4_32f(), WithOutMask(), stream);
    }

    template <typename T, typename D> struct Multiply : binary_function<T, T, D>
    {
        Multiply(double scale_) : scale(scale_) {}
        __device__ __forceinline__ D operator ()(T a, T b) const
        {
            return saturate_cast<D>(scale * a * b);
        }
        const double scale;
    };

    template <> struct TransformFunctorTraits< Multiply<ushort, ushort> > : DefaultTransformFunctorTraits< Multiply<ushort, ushort> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< Multiply<short, short> > : DefaultTransformFunctorTraits< Multiply<short, short> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< Multiply<int, int> > : DefaultTransformFunctorTraits< Multiply<int, int> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< Multiply<float, float> > : DefaultTransformFunctorTraits< Multiply<float, float> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };

    template <typename T, typename D> void multiply_gpu(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream)
    {
        cudaSafeCall( cudaSetDoubleForDevice(&scale) );
        Multiply<T, D> op(scale);
        cv::gpu::device::transform((DevMem2D_<T>)src1, (DevMem2D_<T>)src2, (DevMem2D_<D>)dst, op, WithOutMask(), stream);
    }

    template void multiply_gpu<uchar, uchar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<uchar, schar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<uchar, ushort>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<uchar, short >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<uchar, int   >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<uchar, float >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<uchar, double>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);

    //template void multiply_gpu<schar, uchar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<schar, schar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<schar, ushort>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<schar, short >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<schar, int   >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<schar, float >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<schar, double>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);

    //template void multiply_gpu<ushort, uchar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<ushort, schar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<ushort, ushort>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<ushort, short >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<ushort, int   >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<ushort, float >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<ushort, double>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);

    //template void multiply_gpu<short, uchar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<short, schar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<short, ushort>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<short, short >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<short, int   >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<short, float >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<short, double>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);

    //template void multiply_gpu<int, uchar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<int, schar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<int, ushort>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<int, short >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<int, int   >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<int, float >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<int, double>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);

    //template void multiply_gpu<float, uchar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<float, schar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<float, ushort>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<float, short >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<float, int   >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<float, float >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<float, double>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);

    //template void multiply_gpu<double, uchar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<double, schar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<double, ushort>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<double, short >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<double, int   >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<double, float >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<double, double>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);

    template <typename T, typename D> struct MultiplyScalar : unary_function<T, D>
    {
        MultiplyScalar(double val_, double scale_) : val(val_), scale(scale_) {}
        __device__ __forceinline__ D operator ()(T a) const
        {
            return saturate_cast<D>(scale * a * val);
        }
        const double val;
        const double scale;
    };

    template <> struct TransformFunctorTraits< MultiplyScalar<ushort, ushort> > : DefaultTransformFunctorTraits< MultiplyScalar<ushort, ushort> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< MultiplyScalar<short, short> > : DefaultTransformFunctorTraits< MultiplyScalar<short, short> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< MultiplyScalar<int, int> > : DefaultTransformFunctorTraits< MultiplyScalar<int, int> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< MultiplyScalar<float, float> > : DefaultTransformFunctorTraits< MultiplyScalar<float, float> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };

    template <typename T, typename D> void multiply_gpu(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream)
    {
        cudaSafeCall( cudaSetDoubleForDevice(&val) );
        cudaSafeCall( cudaSetDoubleForDevice(&scale) );
        MultiplyScalar<T, D> op(val, scale);
        cv::gpu::device::transform((DevMem2D_<T>)src1, (DevMem2D_<D>)dst, op, WithOutMask(), stream);
    }

    template void multiply_gpu<uchar, uchar >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<uchar, schar >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<uchar, ushort>(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<uchar, short >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<uchar, int   >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<uchar, float >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<uchar, double>(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);

    //template void multiply_gpu<schar, uchar >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<schar, schar >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<schar, ushort>(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<schar, short >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<schar, int   >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<schar, float >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<schar, double>(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);

    //template void multiply_gpu<ushort, uchar >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<ushort, schar >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<ushort, ushort>(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<ushort, short >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<ushort, int   >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<ushort, float >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<ushort, double>(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);

    //template void multiply_gpu<short, uchar >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<short, schar >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<short, ushort>(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<short, short >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<short, int   >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<short, float >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<short, double>(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);

    //template void multiply_gpu<int, uchar >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<int, schar >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<int, ushort>(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<int, short >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<int, int   >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<int, float >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<int, double>(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);

    //template void multiply_gpu<float, uchar >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<float, schar >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<float, ushort>(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<float, short >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<float, int   >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<float, float >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<float, double>(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);

    //template void multiply_gpu<double, uchar >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<double, schar >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<double, ushort>(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<double, short >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<double, int   >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void multiply_gpu<double, float >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void multiply_gpu<double, double>(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);

    //////////////////////////////////////////////////////////////////////////
    // divide

    struct divide_8uc4_32f : binary_function<uchar4, float, uchar4>
    {
        __device__ __forceinline__ uchar4 operator ()(uchar4 a, float b) const
        {
            return b != 0 ? make_uchar4(saturate_cast<uchar>(a.x / b), saturate_cast<uchar>(a.y / b),
                                        saturate_cast<uchar>(a.z / b), saturate_cast<uchar>(a.w / b))
                          : make_uchar4(0,0,0,0);
        }
    };

    OPENCV_GPU_TRANSFORM_FUNCTOR_TRAITS(divide_8uc4_32f)
    {
        enum { smart_block_dim_x = 8 };
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 8 };
    };

    void divide_gpu(const DevMem2D_<uchar4>& src1, const DevMem2Df& src2, const DevMem2D_<uchar4>& dst, cudaStream_t stream)
    {
        cv::gpu::device::transform(static_cast< DevMem2D_<uchar4> >(src1), src2, static_cast< DevMem2D_<uchar4> >(dst), divide_8uc4_32f(), WithOutMask(), stream);
    }


    struct divide_16sc4_32f : binary_function<short4, float, short4>
    {
        __device__ __forceinline__ short4 operator ()(short4 a, float b) const
        {
            return b != 0 ? make_short4(saturate_cast<short>(a.x / b), saturate_cast<short>(a.y / b),
                                        saturate_cast<short>(a.z / b), saturate_cast<short>(a.w / b))
                          : make_short4(0,0,0,0);
        }
    };

    OPENCV_GPU_TRANSFORM_FUNCTOR_TRAITS(divide_16sc4_32f)
    {
        enum { smart_block_dim_x = 8 };
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 8 };
    };

    void divide_gpu(const DevMem2D_<short4>& src1, const DevMem2Df& src2, const DevMem2D_<short4>& dst, cudaStream_t stream)
    {
        cv::gpu::device::transform(static_cast< DevMem2D_<short4> >(src1), src2, static_cast< DevMem2D_<short4> >(dst), divide_16sc4_32f(), WithOutMask(), stream);
    }

    template <typename T, typename D> struct Divide : binary_function<T, T, D>
    {
        Divide(double scale_) : scale(scale_) {}
        __device__ __forceinline__ D operator ()(T a, T b) const
        {
            return b != 0 ? saturate_cast<D>(scale * a / b) : 0;
        }
        const double scale;
    };

    template <> struct TransformFunctorTraits< Divide<ushort, ushort> > : DefaultTransformFunctorTraits< Divide<ushort, ushort> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< Divide<short, short> > : DefaultTransformFunctorTraits< Divide<short, short> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< Divide<int, int> > : DefaultTransformFunctorTraits< Divide<int, int> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< Divide<float, float> > : DefaultTransformFunctorTraits< Divide<float, float> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };

    template <typename T, typename D> void divide_gpu(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream)
    {
        cudaSafeCall( cudaSetDoubleForDevice(&scale) );
        Divide<T, D> op(scale);
        cv::gpu::device::transform((DevMem2D_<T>)src1, (DevMem2D_<T>)src2, (DevMem2D_<D>)dst, op, WithOutMask(), stream);
    }

    template void divide_gpu<uchar, uchar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<uchar, schar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<uchar, ushort>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<uchar, short >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<uchar, int   >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<uchar, float >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<uchar, double>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);

    //template void divide_gpu<schar, uchar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<schar, schar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<schar, ushort>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<schar, short >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<schar, int   >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<schar, float >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<schar, double>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);

    //template void divide_gpu<ushort, uchar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<ushort, schar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<ushort, ushort>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<ushort, short >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<ushort, int   >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<ushort, float >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<ushort, double>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);

    //template void divide_gpu<short, uchar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<short, schar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<short, ushort>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<short, short >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<short, int   >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<short, float >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<short, double>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);

    //template void divide_gpu<int, uchar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<int, schar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<int, ushort>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<int, short >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<int, int   >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<int, float >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<int, double>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);

    //template void divide_gpu<float, uchar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<float, schar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<float, ushort>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<float, short >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<float, int   >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<float, float >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<float, double>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);

    //template void divide_gpu<double, uchar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<double, schar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<double, ushort>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<double, short >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<double, int   >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<double, float >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<double, double>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);

    template <typename T, typename D> struct DivideScalar : unary_function<T, D>
    {
        DivideScalar(double val_, double scale_) : val(val_), scale(scale_) {}
        __device__ __forceinline__ D operator ()(T a) const
        {
            return saturate_cast<D>(scale * a / val);
        }
        const double val;
        const double scale;
    };

    template <> struct TransformFunctorTraits< DivideScalar<ushort, ushort> > : DefaultTransformFunctorTraits< DivideScalar<ushort, ushort> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< DivideScalar<short, short> > : DefaultTransformFunctorTraits< DivideScalar<short, short> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< DivideScalar<int, int> > : DefaultTransformFunctorTraits< DivideScalar<int, int> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< DivideScalar<float, float> > : DefaultTransformFunctorTraits< DivideScalar<float, float> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };

    template <typename T, typename D> void divide_gpu(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream)
    {
        cudaSafeCall( cudaSetDoubleForDevice(&val) );
        cudaSafeCall( cudaSetDoubleForDevice(&scale) );
        DivideScalar<T, D> op(val, scale);
        cv::gpu::device::transform((DevMem2D_<T>)src1, (DevMem2D_<D>)dst, op, WithOutMask(), stream);
    }

    template void divide_gpu<uchar, uchar >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<uchar, schar >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<uchar, ushort>(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<uchar, short >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<uchar, int   >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<uchar, float >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<uchar, double>(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);

    //template void divide_gpu<schar, uchar >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<schar, schar >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<schar, ushort>(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<schar, short >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<schar, int   >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<schar, float >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<schar, double>(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);

    //template void divide_gpu<ushort, uchar >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<ushort, schar >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<ushort, ushort>(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<ushort, short >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<ushort, int   >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<ushort, float >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<ushort, double>(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);

    //template void divide_gpu<short, uchar >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<short, schar >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<short, ushort>(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<short, short >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<short, int   >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<short, float >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<short, double>(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);

    //template void divide_gpu<int, uchar >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<int, schar >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<int, ushort>(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<int, short >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<int, int   >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<int, float >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<int, double>(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);

    //template void divide_gpu<float, uchar >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<float, schar >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<float, ushort>(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<float, short >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<float, int   >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<float, float >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<float, double>(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);

    //template void divide_gpu<double, uchar >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<double, schar >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<double, ushort>(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<double, short >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<double, int   >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    //template void divide_gpu<double, float >(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
    template void divide_gpu<double, double>(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);

    template <typename T, typename D> struct Reciprocal : unary_function<T, D>
    {
        Reciprocal(double scale_) : scale(scale_) {}
        __device__ __forceinline__ D operator ()(T a) const
        {
            return a != 0 ? saturate_cast<D>(scale / a) : 0;
        }
        const double scale;
    };

    template <> struct TransformFunctorTraits< Reciprocal<ushort, ushort> > : DefaultTransformFunctorTraits< Reciprocal<ushort, ushort> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< Reciprocal<short, short> > : DefaultTransformFunctorTraits< Reciprocal<short, short> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< Reciprocal<int, int> > : DefaultTransformFunctorTraits< Reciprocal<int, int> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< Reciprocal<float, float> > : DefaultTransformFunctorTraits< Reciprocal<float, float> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };

    template <typename T, typename D> void divide_gpu(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream)
    {
        cudaSafeCall( cudaSetDoubleForDevice(&scalar) );
        Reciprocal<T, D> op(scalar);
        cv::gpu::device::transform((DevMem2D_<T>)src2, (DevMem2D_<D>)dst, op, WithOutMask(), stream);
    }

    template void divide_gpu<uchar, uchar >(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    //template void divide_gpu<uchar, schar >(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void divide_gpu<uchar, ushort>(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void divide_gpu<uchar, short >(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void divide_gpu<uchar, int   >(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void divide_gpu<uchar, float >(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void divide_gpu<uchar, double>(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);

    //template void divide_gpu<schar, uchar >(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    //template void divide_gpu<schar, schar >(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    //template void divide_gpu<schar, ushort>(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    //template void divide_gpu<schar, short >(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    //template void divide_gpu<schar, int   >(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    //template void divide_gpu<schar, float >(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    //template void divide_gpu<schar, double>(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);

    //template void divide_gpu<ushort, uchar >(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    //template void divide_gpu<ushort, schar >(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void divide_gpu<ushort, ushort>(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    //template void divide_gpu<ushort, short >(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void divide_gpu<ushort, int   >(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void divide_gpu<ushort, float >(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void divide_gpu<ushort, double>(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);

    //template void divide_gpu<short, uchar >(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    //template void divide_gpu<short, schar >(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    //template void divide_gpu<short, ushort>(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void divide_gpu<short, short >(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void divide_gpu<short, int   >(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void divide_gpu<short, float >(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void divide_gpu<short, double>(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);

    //template void divide_gpu<int, uchar >(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    //template void divide_gpu<int, schar >(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    //template void divide_gpu<int, ushort>(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    //template void divide_gpu<int, short >(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void divide_gpu<int, int   >(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void divide_gpu<int, float >(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void divide_gpu<int, double>(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);

    //template void divide_gpu<float, uchar >(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    //template void divide_gpu<float, schar >(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    //template void divide_gpu<float, ushort>(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    //template void divide_gpu<float, short >(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    //template void divide_gpu<float, int   >(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void divide_gpu<float, float >(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void divide_gpu<float, double>(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);

    //template void divide_gpu<double, uchar >(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    //template void divide_gpu<double, schar >(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    //template void divide_gpu<double, ushort>(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    //template void divide_gpu<double, short >(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    //template void divide_gpu<double, int   >(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    //template void divide_gpu<double, float >(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void divide_gpu<double, double>(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);

    //////////////////////////////////////////////////////////////////////////
    // absdiff

    template <typename T> struct Absdiff : binary_function<T, T, T>
    {
        static __device__ __forceinline__ int abs(int a)
        {
            return ::abs(a);
        }
        static __device__ __forceinline__ float abs(float a)
        {
            return ::fabsf(a);
        }
        static __device__ __forceinline__ double abs(double a)
        {
            return ::fabs(a);
        }

        __device__ __forceinline__ T operator ()(T a, T b) const
        {
            return saturate_cast<T>(::abs(a - b));
        }
    };

    template <> struct TransformFunctorTraits< Absdiff<ushort> > : DefaultTransformFunctorTraits< Absdiff<ushort> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< Absdiff<short> > : DefaultTransformFunctorTraits< Absdiff<short> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< Absdiff<int> > : DefaultTransformFunctorTraits< Absdiff<int> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< Absdiff<float> > : DefaultTransformFunctorTraits< Absdiff<float> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };

    template <typename T> void absdiff_gpu(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream)
    {
        cv::gpu::device::transform((DevMem2D_<T>)src1, (DevMem2D_<T>)src2, (DevMem2D_<T>)dst, Absdiff<T>(), WithOutMask(), stream);
    }

    template void absdiff_gpu<uchar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void absdiff_gpu<schar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void absdiff_gpu<ushort>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void absdiff_gpu<short >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void absdiff_gpu<int   >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void absdiff_gpu<float >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void absdiff_gpu<double>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);

    template <typename T> struct AbsdiffScalar : unary_function<T, T>
    {
        AbsdiffScalar(double val_) : val(val_) {}
        __device__ __forceinline__ T operator ()(T a) const
        {
            return saturate_cast<T>(::fabs(a - val));
        }
        double val;
    };

    template <> struct TransformFunctorTraits< AbsdiffScalar<ushort> > : DefaultTransformFunctorTraits< AbsdiffScalar<ushort> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< AbsdiffScalar<short> > : DefaultTransformFunctorTraits< AbsdiffScalar<short> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< AbsdiffScalar<int> > : DefaultTransformFunctorTraits< AbsdiffScalar<int> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< AbsdiffScalar<float> > : DefaultTransformFunctorTraits< AbsdiffScalar<float> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };

    template <typename T> void absdiff_gpu(const DevMem2Db& src1, double val, const DevMem2Db& dst, cudaStream_t stream)
    {
        cudaSafeCall( cudaSetDoubleForDevice(&val) );
        AbsdiffScalar<T> op(val);
        cv::gpu::device::transform((DevMem2D_<T>)src1, (DevMem2D_<T>)dst, op, WithOutMask(), stream);
    }

    //template void absdiff_gpu<uchar >(const DevMem2Db& src1, double src2, const DevMem2Db& dst, cudaStream_t stream);
    template void absdiff_gpu<schar >(const DevMem2Db& src1, double src2, const DevMem2Db& dst, cudaStream_t stream);
    //template void absdiff_gpu<ushort>(const DevMem2Db& src1, double src2, const DevMem2Db& dst, cudaStream_t stream);
    template void absdiff_gpu<short >(const DevMem2Db& src1, double src2, const DevMem2Db& dst, cudaStream_t stream);
    template void absdiff_gpu<int   >(const DevMem2Db& src1, double src2, const DevMem2Db& dst, cudaStream_t stream);
    //template void absdiff_gpu<float >(const DevMem2Db& src1, double src2, const DevMem2Db& dst, cudaStream_t stream);
    template void absdiff_gpu<double>(const DevMem2Db& src1, double src2, const DevMem2Db& dst, cudaStream_t stream);

    //////////////////////////////////////////////////////////////////////////////////////
    // Compare

    template <typename T> struct Equal : binary_function<T, T, uchar>
    {
        __device__ __forceinline__ uchar operator()(T src1, T src2) const
        {
            return static_cast<uchar>((src1 == src2) * 255);
        }
    };
    template <typename T> struct NotEqual : binary_function<T, T, uchar>
    {
        __device__ __forceinline__ uchar operator()(T src1, T src2) const
        {
            return static_cast<uchar>((src1 != src2) * 255);
        }
    };
    template <typename T> struct Less : binary_function<T, T, uchar>
    {
        __device__ __forceinline__ uchar operator()(T src1, T src2) const
        {
            return static_cast<uchar>((src1 < src2) * 255);
        }
    };
    template <typename T> struct LessEqual : binary_function<T, T, uchar>
    {
        __device__ __forceinline__ uchar operator()(T src1, T src2) const
        {
            return static_cast<uchar>((src1 <= src2) * 255);
        }
    };

    template <> struct TransformFunctorTraits< Equal<int> > : DefaultTransformFunctorTraits< Equal<int> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< Equal<float> > : DefaultTransformFunctorTraits< Equal<float> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< NotEqual<int> > : DefaultTransformFunctorTraits< NotEqual<int> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< NotEqual<float> > : DefaultTransformFunctorTraits< NotEqual<float> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< Less<int> > : DefaultTransformFunctorTraits< Less<int> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< Less<float> > : DefaultTransformFunctorTraits< Less<float> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< LessEqual<int> > : DefaultTransformFunctorTraits< LessEqual<int> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< LessEqual<float> > : DefaultTransformFunctorTraits< LessEqual<float> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };

    template <template <typename> class Op, typename T> void compare(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream)
    {
        Op<T> op;
        cv::gpu::device::transform(static_cast< DevMem2D_<T> >(src1), static_cast< DevMem2D_<T> >(src2), dst, op, WithOutMask(), stream);
    }

    template <typename T> void compare_eq(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream)
    {
        compare<Equal, T>(src1, src2, dst, stream);
    }
    template <typename T> void compare_ne(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream)
    {
        compare<NotEqual, T>(src1, src2, dst, stream);
    }
    template <typename T> void compare_lt(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream)
    {
        compare<Less, T>(src1, src2, dst, stream);
    }
    template <typename T> void compare_le(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream)
    {
        compare<LessEqual, T>(src1, src2, dst, stream);
    }

    template void compare_eq<uchar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void compare_eq<schar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void compare_eq<ushort>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void compare_eq<short >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void compare_eq<int   >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void compare_eq<float >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void compare_eq<double>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);

    template void compare_ne<uchar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void compare_ne<schar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void compare_ne<ushort>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void compare_ne<short >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void compare_ne<int   >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void compare_ne<float >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void compare_ne<double>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);

    template void compare_lt<uchar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void compare_lt<schar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void compare_lt<ushort>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void compare_lt<short >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void compare_lt<int   >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void compare_lt<float >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void compare_lt<double>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);

    template void compare_le<uchar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void compare_le<schar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void compare_le<ushort>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void compare_le<short >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void compare_le<int   >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void compare_le<float >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void compare_le<double>(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);


    //////////////////////////////////////////////////////////////////////////
    // Unary bitwise logical matrix operations

    enum { UN_OP_NOT };

    template <typename T, int opid>
    struct UnOp;

    template <typename T>
    struct UnOp<T, UN_OP_NOT>
    {
        static __device__ __forceinline__ T call(T v) { return ~v; }
    };


    template <int opid>
    __global__ void bitwiseUnOpKernel(int rows, int width, const PtrStepb src, PtrStepb dst)
    {
        const int x = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
        const int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (y < rows)
        {
            uchar* dst_ptr = dst.ptr(y) + x;
            const uchar* src_ptr = src.ptr(y) + x;
            if (x + sizeof(uint) - 1 < width)
            {
                *(uint*)dst_ptr = UnOp<uint, opid>::call(*(uint*)src_ptr);
            }
            else
            {
                const uchar* src_end = src.ptr(y) + width;
                while (src_ptr < src_end)
                {
                    *dst_ptr++ = UnOp<uchar, opid>::call(*src_ptr++);
                }
            }
        }
    }


    template <int opid>
    void bitwiseUnOp(int rows, int width, const PtrStepb src, PtrStepb dst,
                     cudaStream_t stream)
    {
        dim3 threads(16, 16);
        dim3 grid(divUp(width, threads.x * sizeof(uint)),
                  divUp(rows, threads.y));

        bitwiseUnOpKernel<opid><<<grid, threads>>>(rows, width, src, dst);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }


    template <typename T, int opid>
    __global__ void bitwiseUnOpKernel(int rows, int cols, int cn, const PtrStepb src,
                                      const PtrStepb mask, PtrStepb dst)
    {
        const int x = blockDim.x * blockIdx.x + threadIdx.x;
        const int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (x < cols && y < rows && mask.ptr(y)[x / cn])
        {
            T* dst_row = (T*)dst.ptr(y);
            const T* src_row = (const T*)src.ptr(y);

            dst_row[x] = UnOp<T, opid>::call(src_row[x]);
        }
    }


    template <typename T, int opid>
    void bitwiseUnOp(int rows, int cols, int cn, const PtrStepb src,
                     const PtrStepb mask, PtrStepb dst, cudaStream_t stream)
    {
        dim3 threads(16, 16);
        dim3 grid(divUp(cols, threads.x), divUp(rows, threads.y));

        bitwiseUnOpKernel<T, opid><<<grid, threads>>>(rows, cols, cn, src, mask, dst);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }


    void bitwiseNotCaller(int rows, int cols, size_t elem_size1, int cn,
                          const PtrStepb src, PtrStepb dst, cudaStream_t stream)
    {
        bitwiseUnOp<UN_OP_NOT>(rows, static_cast<int>(cols * elem_size1 * cn), src, dst, stream);
    }


    template <typename T>
    void bitwiseMaskNotCaller(int rows, int cols, int cn, const PtrStepb src,
                              const PtrStepb mask, PtrStepb dst, cudaStream_t stream)
    {
        bitwiseUnOp<T, UN_OP_NOT>(rows, cols * cn, cn, src, mask, dst, stream);
    }

    template void bitwiseMaskNotCaller<uchar>(int, int, int, const PtrStepb, const PtrStepb, PtrStepb, cudaStream_t);
    template void bitwiseMaskNotCaller<ushort>(int, int, int, const PtrStepb, const PtrStepb, PtrStepb, cudaStream_t);
    template void bitwiseMaskNotCaller<uint>(int, int, int, const PtrStepb, const PtrStepb, PtrStepb, cudaStream_t);


    //////////////////////////////////////////////////////////////////////////
    // Binary bitwise logical matrix operations

    enum { BIN_OP_OR, BIN_OP_AND, BIN_OP_XOR };

    template <typename T, int opid>
    struct BinOp;

    template <typename T>
    struct BinOp<T, BIN_OP_OR>
    {
        static __device__ __forceinline__ T call(T a, T b) { return a | b; }
    };


    template <typename T>
    struct BinOp<T, BIN_OP_AND>
    {
        static __device__ __forceinline__ T call(T a, T b) { return a & b; }
    };

    template <typename T>
    struct BinOp<T, BIN_OP_XOR>
    {
        static __device__ __forceinline__ T call(T a, T b) { return a ^ b; }
    };


    template <int opid>
    __global__ void bitwiseBinOpKernel(int rows, int width, const PtrStepb src1,
                                       const PtrStepb src2, PtrStepb dst)
    {
        const int x = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
        const int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (y < rows)
        {
            uchar* dst_ptr = dst.ptr(y) + x;
            const uchar* src1_ptr = src1.ptr(y) + x;
            const uchar* src2_ptr = src2.ptr(y) + x;

            if (x + sizeof(uint) - 1 < width)
            {
                *(uint*)dst_ptr = BinOp<uint, opid>::call(*(uint*)src1_ptr, *(uint*)src2_ptr);
            }
            else
            {
                const uchar* src1_end = src1.ptr(y) + width;
                while (src1_ptr < src1_end)
                {
                    *dst_ptr++ = BinOp<uchar, opid>::call(*src1_ptr++, *src2_ptr++);
                }
            }
        }
    }


    template <int opid>
    void bitwiseBinOp(int rows, int width, const PtrStepb src1, const PtrStepb src2,
                      PtrStepb dst, cudaStream_t stream)
    {
        dim3 threads(16, 16);
        dim3 grid(divUp(width, threads.x * sizeof(uint)), divUp(rows, threads.y));

        bitwiseBinOpKernel<opid><<<grid, threads>>>(rows, width, src1, src2, dst);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }


    template <typename T, int opid>
    __global__ void bitwiseBinOpKernel(
            int rows, int cols, int cn, const PtrStepb src1, const PtrStepb src2,
            const PtrStepb mask, PtrStepb dst)
    {
        const int x = blockDim.x * blockIdx.x + threadIdx.x;
        const int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (x < cols && y < rows && mask.ptr(y)[x / cn])
        {
            T* dst_row = (T*)dst.ptr(y);
            const T* src1_row = (const T*)src1.ptr(y);
            const T* src2_row = (const T*)src2.ptr(y);

            dst_row[x] = BinOp<T, opid>::call(src1_row[x], src2_row[x]);
        }
    }


    template <typename T, int opid>
    void bitwiseBinOp(int rows, int cols, int cn, const PtrStepb src1, const PtrStepb src2,
                        const PtrStepb mask, PtrStepb dst, cudaStream_t stream)
    {
        dim3 threads(16, 16);
        dim3 grid(divUp(cols, threads.x), divUp(rows, threads.y));

        bitwiseBinOpKernel<T, opid><<<grid, threads>>>(rows, cols, cn, src1, src2, mask, dst);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }


    void bitwiseOrCaller(int rows, int cols, size_t elem_size1, int cn, const PtrStepb src1,
                         const PtrStepb src2, PtrStepb dst, cudaStream_t stream)
    {
        bitwiseBinOp<BIN_OP_OR>(rows, static_cast<int>(cols * elem_size1 * cn), src1, src2, dst, stream);
    }


    template <typename T>
    void bitwiseMaskOrCaller(int rows, int cols, int cn, const PtrStepb src1, const PtrStepb src2,
                             const PtrStepb mask, PtrStepb dst, cudaStream_t stream)
    {
        bitwiseBinOp<T, BIN_OP_OR>(rows, cols * cn, cn, src1, src2, mask, dst, stream);
    }

    template void bitwiseMaskOrCaller<uchar>(int, int, int, const PtrStepb, const PtrStepb, const PtrStepb, PtrStepb, cudaStream_t);
    template void bitwiseMaskOrCaller<ushort>(int, int, int, const PtrStepb, const PtrStepb, const PtrStepb, PtrStepb, cudaStream_t);
    template void bitwiseMaskOrCaller<uint>(int, int, int, const PtrStepb, const PtrStepb, const PtrStepb, PtrStepb, cudaStream_t);


    void bitwiseAndCaller(int rows, int cols, size_t elem_size1, int cn, const PtrStepb src1,
                          const PtrStepb src2, PtrStepb dst, cudaStream_t stream)
    {
        bitwiseBinOp<BIN_OP_AND>(rows, static_cast<int>(cols * elem_size1 * cn), src1, src2, dst, stream);
    }


    template <typename T>
    void bitwiseMaskAndCaller(int rows, int cols, int cn, const PtrStepb src1, const PtrStepb src2,
                              const PtrStepb mask, PtrStepb dst, cudaStream_t stream)
    {
        bitwiseBinOp<T, BIN_OP_AND>(rows, cols * cn, cn, src1, src2, mask, dst, stream);
    }

    template void bitwiseMaskAndCaller<uchar>(int, int, int, const PtrStepb, const PtrStepb, const PtrStepb, PtrStepb, cudaStream_t);
    template void bitwiseMaskAndCaller<ushort>(int, int, int, const PtrStepb, const PtrStepb, const PtrStepb, PtrStepb, cudaStream_t);
    template void bitwiseMaskAndCaller<uint>(int, int, int, const PtrStepb, const PtrStepb, const PtrStepb, PtrStepb, cudaStream_t);


    void bitwiseXorCaller(int rows, int cols, size_t elem_size1, int cn, const PtrStepb src1,
                          const PtrStepb src2, PtrStepb dst, cudaStream_t stream)
    {
        bitwiseBinOp<BIN_OP_XOR>(rows, static_cast<int>(cols * elem_size1 * cn), src1, src2, dst, stream);
    }


    template <typename T>
    void bitwiseMaskXorCaller(int rows, int cols, int cn, const PtrStepb src1, const PtrStepb src2,
                              const PtrStepb mask, PtrStepb dst, cudaStream_t stream)
    {
        bitwiseBinOp<T, BIN_OP_XOR>(rows, cols * cn, cn, src1, src2, mask, dst, stream);
    }

    template void bitwiseMaskXorCaller<uchar>(int, int, int, const PtrStepb, const PtrStepb, const PtrStepb, PtrStepb, cudaStream_t);
    template void bitwiseMaskXorCaller<ushort>(int, int, int, const PtrStepb, const PtrStepb, const PtrStepb, PtrStepb, cudaStream_t);
    template void bitwiseMaskXorCaller<uint>(int, int, int, const PtrStepb, const PtrStepb, const PtrStepb, PtrStepb, cudaStream_t);

    //////////////////////////////////////////////////////////////////////////
    // min/max

    namespace detail
    {
        template <size_t size, typename F> struct MinMaxTraits : DefaultTransformFunctorTraits<F>
        {
        };
        template <typename F> struct MinMaxTraits<2, F> : DefaultTransformFunctorTraits<F>
        {
            enum { smart_shift = 4 };
        };
        template <typename F> struct MinMaxTraits<4, F> : DefaultTransformFunctorTraits<F>
        {
            enum { smart_block_dim_y = 4 };
            enum { smart_shift = 4 };
        };
    }

    template <typename T> struct TransformFunctorTraits< minimum<T> > : detail::MinMaxTraits< sizeof(T), minimum<T> >
    {
    };
    template <typename T> struct TransformFunctorTraits< maximum<T> > : detail::MinMaxTraits< sizeof(T), maximum<T> >
    {
    };
    template <typename T> struct TransformFunctorTraits< binder2nd< minimum<T> > > : detail::MinMaxTraits< sizeof(T), binder2nd< minimum<T> > >
    {
    };
    template <typename T> struct TransformFunctorTraits< binder2nd< maximum<T> > > : detail::MinMaxTraits< sizeof(T), binder2nd< maximum<T> > >
    {
    };

    template <typename T>
    void min_gpu(const DevMem2D_<T>& src1, const DevMem2D_<T>& src2, const DevMem2D_<T>& dst, cudaStream_t stream)
    {
        cv::gpu::device::transform(src1, src2, dst, minimum<T>(), WithOutMask(), stream);
    }

    template void min_gpu<uchar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void min_gpu<schar >(const DevMem2D_<schar>& src1, const DevMem2D_<schar>& src2, const DevMem2D_<schar>& dst, cudaStream_t stream);
    template void min_gpu<ushort>(const DevMem2D_<ushort>& src1, const DevMem2D_<ushort>& src2, const DevMem2D_<ushort>& dst, cudaStream_t stream);
    template void min_gpu<short >(const DevMem2D_<short>& src1, const DevMem2D_<short>& src2, const DevMem2D_<short>& dst, cudaStream_t stream);
    template void min_gpu<int   >(const DevMem2D_<int>& src1, const DevMem2D_<int>& src2, const DevMem2D_<int>& dst, cudaStream_t stream);
    template void min_gpu<float >(const DevMem2D_<float>& src1, const DevMem2D_<float>& src2, const DevMem2D_<float>& dst, cudaStream_t stream);
    template void min_gpu<double>(const DevMem2D_<double>& src1, const DevMem2D_<double>& src2, const DevMem2D_<double>& dst, cudaStream_t stream);

    template <typename T>
    void max_gpu(const DevMem2D_<T>& src1, const DevMem2D_<T>& src2, const DevMem2D_<T>& dst, cudaStream_t stream)
    {
        cv::gpu::device::transform(src1, src2, dst, maximum<T>(), WithOutMask(), stream);
    }

    template void max_gpu<uchar >(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template void max_gpu<schar >(const DevMem2D_<schar>& src1, const DevMem2D_<schar>& src2, const DevMem2D_<schar>& dst, cudaStream_t stream);
    template void max_gpu<ushort>(const DevMem2D_<ushort>& src1, const DevMem2D_<ushort>& src2, const DevMem2D_<ushort>& dst, cudaStream_t stream);
    template void max_gpu<short >(const DevMem2D_<short>& src1, const DevMem2D_<short>& src2, const DevMem2D_<short>& dst, cudaStream_t stream);
    template void max_gpu<int   >(const DevMem2D_<int>& src1, const DevMem2D_<int>& src2, const DevMem2D_<int>& dst, cudaStream_t stream);
    template void max_gpu<float >(const DevMem2D_<float>& src1, const DevMem2D_<float>& src2, const DevMem2D_<float>& dst, cudaStream_t stream);
    template void max_gpu<double>(const DevMem2D_<double>& src1, const DevMem2D_<double>& src2, const DevMem2D_<double>& dst, cudaStream_t stream);

    template <typename T>
    void min_gpu(const DevMem2D_<T>& src1, T src2, const DevMem2D_<T>& dst, cudaStream_t stream)
    {
        cv::gpu::device::transform(src1, dst, device::bind2nd(minimum<T>(), src2), WithOutMask(), stream);
    }

    template void min_gpu<uchar >(const DevMem2Db& src1, uchar src2, const DevMem2Db& dst, cudaStream_t stream);
    template void min_gpu<schar >(const DevMem2D_<schar>& src1, schar src2, const DevMem2D_<schar>& dst, cudaStream_t stream);
    template void min_gpu<ushort>(const DevMem2D_<ushort>& src1, ushort src2, const DevMem2D_<ushort>& dst, cudaStream_t stream);
    template void min_gpu<short >(const DevMem2D_<short>& src1, short src2, const DevMem2D_<short>& dst, cudaStream_t stream);
    template void min_gpu<int   >(const DevMem2D_<int>& src1, int src2, const DevMem2D_<int>& dst, cudaStream_t stream);
    template void min_gpu<float >(const DevMem2D_<float>& src1, float src2, const DevMem2D_<float>& dst, cudaStream_t stream);
    template void min_gpu<double>(const DevMem2D_<double>& src1, double src2, const DevMem2D_<double>& dst, cudaStream_t stream);

    template <typename T>
    void max_gpu(const DevMem2D_<T>& src1, T src2, const DevMem2D_<T>& dst, cudaStream_t stream)
    {
        cv::gpu::device::transform(src1, dst, device::bind2nd(maximum<T>(), src2), WithOutMask(), stream);
    }

    template void max_gpu<uchar >(const DevMem2Db& src1, uchar src2, const DevMem2Db& dst, cudaStream_t stream);
    template void max_gpu<schar >(const DevMem2D_<schar>& src1, schar src2, const DevMem2D_<schar>& dst, cudaStream_t stream);
    template void max_gpu<ushort>(const DevMem2D_<ushort>& src1, ushort src2, const DevMem2D_<ushort>& dst, cudaStream_t stream);
    template void max_gpu<short >(const DevMem2D_<short>& src1, short src2, const DevMem2D_<short>& dst, cudaStream_t stream);
    template void max_gpu<int   >(const DevMem2D_<int>& src1, int src2, const DevMem2D_<int>& dst, cudaStream_t stream);
    template void max_gpu<float >(const DevMem2D_<float>& src1, float src2, const DevMem2D_<float>& dst, cudaStream_t stream);
    template void max_gpu<double>(const DevMem2D_<double>& src1, double src2, const DevMem2D_<double>& dst, cudaStream_t stream);

    //////////////////////////////////////////////////////////////////////////
    // threshold

    namespace detail
    {
        template <size_t size, typename F> struct ThresholdTraits : DefaultTransformFunctorTraits<F>
        {
        };
        template <typename F> struct ThresholdTraits<2, F> : DefaultTransformFunctorTraits<F>
        {
            enum { smart_shift = 4 };
        };
        template <typename F> struct ThresholdTraits<4, F> : DefaultTransformFunctorTraits<F>
        {
            enum { smart_block_dim_y = 4 };
            enum { smart_shift = 4 };
        };
    }

    template <typename T> struct TransformFunctorTraits< thresh_binary_func<T> > : detail::ThresholdTraits< sizeof(T), thresh_binary_func<T> >
    {
    };
    template <typename T> struct TransformFunctorTraits< thresh_binary_inv_func<T> > : detail::ThresholdTraits< sizeof(T), thresh_binary_inv_func<T> >
    {
    };
    template <typename T> struct TransformFunctorTraits< thresh_trunc_func<T> > : detail::ThresholdTraits< sizeof(T), thresh_trunc_func<T> >
    {
    };
    template <typename T> struct TransformFunctorTraits< thresh_to_zero_func<T> > : detail::ThresholdTraits< sizeof(T), thresh_to_zero_func<T> >
    {
    };
    template <typename T> struct TransformFunctorTraits< thresh_to_zero_inv_func<T> > : detail::ThresholdTraits< sizeof(T), thresh_to_zero_inv_func<T> >
    {
    };

    template <template <typename> class Op, typename T>
    void threshold_caller(const DevMem2D_<T>& src, const DevMem2D_<T>& dst, T thresh, T maxVal, cudaStream_t stream)
    {
        Op<T> op(thresh, maxVal);
        cv::gpu::device::transform(src, dst, op, WithOutMask(), stream);
    }

    template <typename T>
    void threshold_gpu(const DevMem2Db& src, const DevMem2Db& dst, T thresh, T maxVal, int type,
        cudaStream_t stream)
    {
        typedef void (*caller_t)(const DevMem2D_<T>& src, const DevMem2D_<T>& dst, T thresh, T maxVal, cudaStream_t stream);

        static const caller_t callers[] =
        {
            threshold_caller<thresh_binary_func, T>,
            threshold_caller<thresh_binary_inv_func, T>,
            threshold_caller<thresh_trunc_func, T>,
            threshold_caller<thresh_to_zero_func, T>,
            threshold_caller<thresh_to_zero_inv_func, T>
        };

        callers[type]((DevMem2D_<T>)src, (DevMem2D_<T>)dst, thresh, maxVal, stream);
    }

    template void threshold_gpu<uchar>(const DevMem2Db& src, const DevMem2Db& dst, uchar thresh, uchar maxVal, int type, cudaStream_t stream);
    template void threshold_gpu<schar>(const DevMem2Db& src, const DevMem2Db& dst, schar thresh, schar maxVal, int type, cudaStream_t stream);
    template void threshold_gpu<ushort>(const DevMem2Db& src, const DevMem2Db& dst, ushort thresh, ushort maxVal, int type, cudaStream_t stream);
    template void threshold_gpu<short>(const DevMem2Db& src, const DevMem2Db& dst, short thresh, short maxVal, int type, cudaStream_t stream);
    template void threshold_gpu<int>(const DevMem2Db& src, const DevMem2Db& dst, int thresh, int maxVal, int type, cudaStream_t stream);
    template void threshold_gpu<float>(const DevMem2Db& src, const DevMem2Db& dst, float thresh, float maxVal, int type, cudaStream_t stream);
    template void threshold_gpu<double>(const DevMem2Db& src, const DevMem2Db& dst, double thresh, double maxVal, int type, cudaStream_t stream);

    //////////////////////////////////////////////////////////////////////////
    // pow

    template<typename T, bool Signed = device::numeric_limits<T>::is_signed> struct PowOp : unary_function<T, T>
    {
        const float power;

        PowOp(double power_) : power(static_cast<float>(power_)) {}

        __device__ __forceinline__ T operator()(T e) const
        {
            return saturate_cast<T>(__powf((float)e, power));
        }
    };
    template<typename T> struct PowOp<T, true> : unary_function<T, T>
    {
        const float power;

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
        const double power;

        PowOp(double power_) : power(power_) {}

        __device__ __forceinline__ double operator()(double e) const
        {
            return ::pow(::fabs(e), power);
        }
    };

    namespace detail
    {
        template <size_t size, typename T> struct PowOpTraits : DefaultTransformFunctorTraits< PowOp<T> >
        {
        };
        template <typename T> struct PowOpTraits<1, T> : DefaultTransformFunctorTraits< PowOp<T> >
        {
            enum { smart_block_dim_y = 8 };
            enum { smart_shift = 8 };
        };
        template <typename T> struct PowOpTraits<2, T> : DefaultTransformFunctorTraits< PowOp<T> >
        {
            enum { smart_shift = 4 };
        };
        template <typename T> struct PowOpTraits<4, T> : DefaultTransformFunctorTraits< PowOp<T> >
        {
            enum { smart_block_dim_y = 4 };
            enum { smart_shift = 4 };
        };
    }

    template <typename T> struct TransformFunctorTraits< PowOp<T> > : detail::PowOpTraits<sizeof(T), T>
    {
    };

    template<typename T>
    void pow_caller(DevMem2Db src, double power, DevMem2Db dst, cudaStream_t stream)
    {
        cv::gpu::device::transform((DevMem2D_<T>)src, (DevMem2D_<T>)dst, PowOp<T>(power), WithOutMask(), stream);
    }

    template void pow_caller<uchar>(DevMem2Db src, double power, DevMem2Db dst, cudaStream_t stream);
    template void pow_caller<schar>(DevMem2Db src, double power, DevMem2Db dst, cudaStream_t stream);
    template void pow_caller<short>(DevMem2Db src, double power, DevMem2Db dst, cudaStream_t stream);
    template void pow_caller<ushort>(DevMem2Db src, double power, DevMem2Db dst, cudaStream_t stream);
    template void pow_caller<int>(DevMem2Db src, double power, DevMem2Db dst, cudaStream_t stream);
    template void pow_caller<float>(DevMem2Db src, double power, DevMem2Db dst, cudaStream_t stream);
    template void pow_caller<double>(DevMem2Db src, double power, DevMem2Db dst, cudaStream_t stream);

    //////////////////////////////////////////////////////////////////////////
    // addWeighted

    template <typename T1, typename T2, typename D> struct AddWeighted : binary_function<T1, T2, D>
    {
        __host__ __device__ __forceinline__ AddWeighted(double alpha_, double beta_, double gamma_) : alpha(alpha_), beta(beta_), gamma(gamma_) {}

        __device__ __forceinline__ D operator ()(typename TypeTraits<T1>::ParameterType a, typename TypeTraits<T2>::ParameterType b) const
        {
            return saturate_cast<D>(alpha * a + beta * b + gamma);
        }

        const double alpha;
        const double beta;
        const double gamma;
    };

    template <> struct TransformFunctorTraits< AddWeighted<ushort, ushort, ushort> > : DefaultTransformFunctorTraits< AddWeighted<ushort, ushort, ushort> >
    {
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< AddWeighted<ushort, ushort, short> > : DefaultTransformFunctorTraits< AddWeighted<ushort, ushort, short> >
    {
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< AddWeighted<ushort, short, ushort> > : DefaultTransformFunctorTraits< AddWeighted<ushort, short, ushort> >
    {
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< AddWeighted<ushort, short, short> > : DefaultTransformFunctorTraits< AddWeighted<ushort, short, short> >
    {
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< AddWeighted<short, short, ushort> > : DefaultTransformFunctorTraits< AddWeighted<short, short, ushort> >
    {
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< AddWeighted<short, short, short> > : DefaultTransformFunctorTraits< AddWeighted<short, short, short> >
    {
        enum { smart_shift = 4 };
    };

    template <> struct TransformFunctorTraits< AddWeighted<int, int, int> > : DefaultTransformFunctorTraits< AddWeighted<int, int, int> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< AddWeighted<int, int, float> > : DefaultTransformFunctorTraits< AddWeighted<int, int, float> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< AddWeighted<int, float, int> > : DefaultTransformFunctorTraits< AddWeighted<int, float, int> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< AddWeighted<int, float, float> > : DefaultTransformFunctorTraits< AddWeighted<int, float, float> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< AddWeighted<float, float, int> > : DefaultTransformFunctorTraits< AddWeighted<float, float, float> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
    template <> struct TransformFunctorTraits< AddWeighted<float, float, float> > : DefaultTransformFunctorTraits< AddWeighted<float, float, float> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };

    template <typename T1, typename T2, typename D>
    void addWeighted_gpu(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream)
    {
        cudaSafeCall( cudaSetDoubleForDevice(&alpha) );
        cudaSafeCall( cudaSetDoubleForDevice(&beta) );
        cudaSafeCall( cudaSetDoubleForDevice(&gamma) );

        AddWeighted<T1, T2, D> op(alpha, beta, gamma);

        cv::gpu::device::transform(static_cast< DevMem2D_<T1> >(src1), static_cast< DevMem2D_<T2> >(src2), static_cast< DevMem2D_<D> >(dst), op, WithOutMask(), stream);
    }

    template void addWeighted_gpu<uchar, uchar, uchar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, uchar, schar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, uchar, ushort>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, uchar, short>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, uchar, int>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, uchar, float>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, uchar, double>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);

    template void addWeighted_gpu<uchar, schar, uchar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, schar, schar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, schar, ushort>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, schar, short>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, schar, int>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, schar, float>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, schar, double>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);

    template void addWeighted_gpu<uchar, ushort, uchar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, ushort, schar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, ushort, ushort>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, ushort, short>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, ushort, int>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, ushort, float>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, ushort, double>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);

    template void addWeighted_gpu<uchar, short, uchar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, short, schar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, short, ushort>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, short, short>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, short, int>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, short, float>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, short, double>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);

    template void addWeighted_gpu<uchar, int, uchar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, int, schar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, int, ushort>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, int, short>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, int, int>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, int, float>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, int, double>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);

    template void addWeighted_gpu<uchar, float, uchar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, float, schar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, float, ushort>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, float, short>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, float, int>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, float, float>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, float, double>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);

    template void addWeighted_gpu<uchar, double, uchar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, double, schar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, double, ushort>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, double, short>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, double, int>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, double, float>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<uchar, double, double>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);



    template void addWeighted_gpu<schar, schar, uchar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, schar, schar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, schar, ushort>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, schar, short>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, schar, int>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, schar, float>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, schar, double>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);

    template void addWeighted_gpu<schar, ushort, uchar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, ushort, schar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, ushort, ushort>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, ushort, short>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, ushort, int>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, ushort, float>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, ushort, double>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);

    template void addWeighted_gpu<schar, short, uchar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, short, schar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, short, ushort>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, short, short>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, short, int>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, short, float>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, short, double>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);

    template void addWeighted_gpu<schar, int, uchar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, int, schar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, int, ushort>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, int, short>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, int, int>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, int, float>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, int, double>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);

    template void addWeighted_gpu<schar, float, uchar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, float, schar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, float, ushort>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, float, short>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, float, int>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, float, float>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, float, double>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);

    template void addWeighted_gpu<schar, double, uchar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, double, schar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, double, ushort>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, double, short>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, double, int>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, double, float>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<schar, double, double>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);



    template void addWeighted_gpu<ushort, ushort, uchar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<ushort, ushort, schar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<ushort, ushort, ushort>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<ushort, ushort, short>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<ushort, ushort, int>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<ushort, ushort, float>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<ushort, ushort, double>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);

    template void addWeighted_gpu<ushort, short, uchar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<ushort, short, schar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<ushort, short, ushort>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<ushort, short, short>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<ushort, short, int>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<ushort, short, float>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<ushort, short, double>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);

    template void addWeighted_gpu<ushort, int, uchar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<ushort, int, schar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<ushort, int, ushort>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<ushort, int, short>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<ushort, int, int>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<ushort, int, float>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<ushort, int, double>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);

    template void addWeighted_gpu<ushort, float, uchar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<ushort, float, schar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<ushort, float, ushort>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<ushort, float, short>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<ushort, float, int>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<ushort, float, float>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<ushort, float, double>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);

    template void addWeighted_gpu<ushort, double, uchar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<ushort, double, schar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<ushort, double, ushort>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<ushort, double, short>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<ushort, double, int>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<ushort, double, float>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<ushort, double, double>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);



    template void addWeighted_gpu<short, short, uchar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<short, short, schar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<short, short, ushort>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<short, short, short>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<short, short, int>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<short, short, float>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<short, short, double>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);

    template void addWeighted_gpu<short, int, uchar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<short, int, schar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<short, int, ushort>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<short, int, short>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<short, int, int>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<short, int, float>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<short, int, double>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);

    template void addWeighted_gpu<short, float, uchar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<short, float, schar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<short, float, ushort>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<short, float, short>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<short, float, int>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<short, float, float>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<short, float, double>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);

    template void addWeighted_gpu<short, double, uchar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<short, double, schar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<short, double, ushort>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<short, double, short>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<short, double, int>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<short, double, float>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<short, double, double>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);



    template void addWeighted_gpu<int, int, uchar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<int, int, schar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<int, int, ushort>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<int, int, short>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<int, int, int>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<int, int, float>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<int, int, double>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);

    template void addWeighted_gpu<int, float, uchar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<int, float, schar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<int, float, ushort>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<int, float, short>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<int, float, int>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<int, float, float>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<int, float, double>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);

    template void addWeighted_gpu<int, double, uchar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<int, double, schar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<int, double, ushort>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<int, double, short>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<int, double, int>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<int, double, float>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<int, double, double>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);



    template void addWeighted_gpu<float, float, uchar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<float, float, schar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<float, float, ushort>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<float, float, short>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<float, float, int>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<float, float, float>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<float, float, double>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);

    template void addWeighted_gpu<float, double, uchar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<float, double, schar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<float, double, ushort>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<float, double, short>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<float, double, int>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<float, double, float>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<float, double, double>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);



    template void addWeighted_gpu<double, double, uchar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<double, double, schar>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<double, double, ushort>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<double, double, short>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<double, double, int>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<double, double, float>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
    template void addWeighted_gpu<double, double, double>(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
}}} // namespace cv { namespace gpu { namespace device
