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

#include "opencv2/opencv_modules.hpp"

#ifndef HAVE_OPENCV_CUDEV

#error "opencv_cudev is required"

#else

#include "opencv2/cudev.hpp"

using namespace cv::cudev;

void absDiffMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat&, double, Stream& stream, int);

namespace
{
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

    template <typename T> struct AbsDiffOp1 : binary_function<T, T, T>
    {
        __device__ __forceinline__ T operator ()(T a, T b) const
        {
            return saturate_cast<T>(_abs(a - b));
        }
    };

    template <typename ScalarDepth> struct TransformPolicy : DefaultTransformPolicy
    {
    };
    template <> struct TransformPolicy<double> : DefaultTransformPolicy
    {
        enum {
            shift = 1
        };
    };

    template <typename T>
    void absDiffMat_v1(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream)
    {
        gridTransformBinary_< TransformPolicy<T> >(globPtr<T>(src1), globPtr<T>(src2), globPtr<T>(dst), AbsDiffOp1<T>(), stream);
    }

    struct AbsDiffOp2 : binary_function<uint, uint, uint>
    {
        __device__ __forceinline__ uint operator ()(uint a, uint b) const
        {
            return vabsdiff2(a, b);
        }
    };

    void absDiffMat_v2(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream)
    {
        const int vcols = src1.cols >> 1;

        GlobPtrSz<uint> src1_ = globPtr((uint*) src1.data, src1.step, src1.rows, vcols);
        GlobPtrSz<uint> src2_ = globPtr((uint*) src2.data, src2.step, src1.rows, vcols);
        GlobPtrSz<uint> dst_ = globPtr((uint*) dst.data, dst.step, src1.rows, vcols);

        gridTransformBinary(src1_, src2_, dst_, AbsDiffOp2(), stream);
    }

    struct AbsDiffOp4 : binary_function<uint, uint, uint>
    {
        __device__ __forceinline__ uint operator ()(uint a, uint b) const
        {
            return vabsdiff4(a, b);
        }
    };

    void absDiffMat_v4(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream)
    {
        const int vcols = src1.cols >> 2;

        GlobPtrSz<uint> src1_ = globPtr((uint*) src1.data, src1.step, src1.rows, vcols);
        GlobPtrSz<uint> src2_ = globPtr((uint*) src2.data, src2.step, src1.rows, vcols);
        GlobPtrSz<uint> dst_ = globPtr((uint*) dst.data, dst.step, src1.rows, vcols);

        gridTransformBinary(src1_, src2_, dst_, AbsDiffOp4(), stream);
    }
}

void absDiffMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat&, double, Stream& stream, int)
{
    typedef void (*func_t)(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream);
    static const func_t funcs[] =
    {
        absDiffMat_v1<uchar>,
        absDiffMat_v1<schar>,
        absDiffMat_v1<ushort>,
        absDiffMat_v1<short>,
        absDiffMat_v1<int>,
        absDiffMat_v1<float>,
        absDiffMat_v1<double>
    };

    const int depth = src1.depth();

    CV_DbgAssert( depth <= CV_64F );

    GpuMat src1_ = src1.reshape(1);
    GpuMat src2_ = src2.reshape(1);
    GpuMat dst_ = dst.reshape(1);

    if (depth == CV_8U || depth == CV_16U)
    {
        const intptr_t src1ptr = reinterpret_cast<intptr_t>(src1_.data);
        const intptr_t src2ptr = reinterpret_cast<intptr_t>(src2_.data);
        const intptr_t dstptr = reinterpret_cast<intptr_t>(dst_.data);

        const bool isAllAligned = (src1ptr & 31) == 0 && (src2ptr & 31) == 0 && (dstptr & 31) == 0;

        if (isAllAligned)
        {
            if (depth == CV_8U && (src1_.cols & 3) == 0)
            {
                absDiffMat_v4(src1_, src2_, dst_, stream);
                return;
            }
            else if (depth == CV_16U && (src1_.cols & 1) == 0)
            {
                absDiffMat_v2(src1_, src2_, dst_, stream);
                return;
            }
        }
    }

    const func_t func = funcs[depth];

    if (!func)
        CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported combination of source and destination types");

    func(src1_, src2_, dst_, stream);
}

#endif
