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

void minMaxMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat&, double, Stream& stream, int op);

void minMaxScalar(const GpuMat& src, cv::Scalar value, bool, GpuMat& dst, const GpuMat&, double, Stream& stream, int op);

///////////////////////////////////////////////////////////////////////
/// minMaxMat

namespace
{
    template <template <typename> class Op, typename T>
    void minMaxMat_v1(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream)
    {
        gridTransformBinary(globPtr<T>(src1), globPtr<T>(src2), globPtr<T>(dst), Op<T>(), stream);
    }

    struct MinOp2 : binary_function<uint, uint, uint>
    {
        __device__ __forceinline__ uint operator ()(uint a, uint b) const
        {
            return vmin2(a, b);
        }
    };

    struct MaxOp2 : binary_function<uint, uint, uint>
    {
        __device__ __forceinline__ uint operator ()(uint a, uint b) const
        {
            return vmax2(a, b);
        }
    };

    template <class Op2>
    void minMaxMat_v2(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream)
    {
        const int vcols = src1.cols >> 1;

        GlobPtrSz<uint> src1_ = globPtr((uint*) src1.data, src1.step, src1.rows, vcols);
        GlobPtrSz<uint> src2_ = globPtr((uint*) src2.data, src2.step, src1.rows, vcols);
        GlobPtrSz<uint> dst_ = globPtr((uint*) dst.data, dst.step, src1.rows, vcols);

        gridTransformBinary(src1_, src2_, dst_, Op2(), stream);
    }

    struct MinOp4 : binary_function<uint, uint, uint>
    {
        __device__ __forceinline__ uint operator ()(uint a, uint b) const
        {
            return vmin4(a, b);
        }
    };

    struct MaxOp4 : binary_function<uint, uint, uint>
    {
        __device__ __forceinline__ uint operator ()(uint a, uint b) const
        {
            return vmax4(a, b);
        }
    };

    template <class Op4>
    void minMaxMat_v4(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream)
    {
        const int vcols = src1.cols >> 2;

        GlobPtrSz<uint> src1_ = globPtr((uint*) src1.data, src1.step, src1.rows, vcols);
        GlobPtrSz<uint> src2_ = globPtr((uint*) src2.data, src2.step, src1.rows, vcols);
        GlobPtrSz<uint> dst_ = globPtr((uint*) dst.data, dst.step, src1.rows, vcols);

        gridTransformBinary(src1_, src2_, dst_, Op4(), stream);
    }
}

void minMaxMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat&, double, Stream& stream, int op)
{
    typedef void (*func_t)(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream);
    static const func_t funcs_v1[2][7] =
    {
        {
            minMaxMat_v1<minimum, uchar>,
            minMaxMat_v1<minimum, schar>,
            minMaxMat_v1<minimum, ushort>,
            minMaxMat_v1<minimum, short>,
            minMaxMat_v1<minimum, int>,
            minMaxMat_v1<minimum, float>,
            minMaxMat_v1<minimum, double>
        },
        {
            minMaxMat_v1<maximum, uchar>,
            minMaxMat_v1<maximum, schar>,
            minMaxMat_v1<maximum, ushort>,
            minMaxMat_v1<maximum, short>,
            minMaxMat_v1<maximum, int>,
            minMaxMat_v1<maximum, float>,
            minMaxMat_v1<maximum, double>
        }
    };

    static const func_t funcs_v2[2] =
    {
        minMaxMat_v2<MinOp2>, minMaxMat_v2<MaxOp2>
    };

    static const func_t funcs_v4[2] =
    {
        minMaxMat_v4<MinOp4>, minMaxMat_v4<MaxOp4>
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
                funcs_v4[op](src1_, src2_, dst_, stream);
                return;
            }
            else if (depth == CV_16U && (src1_.cols & 1) == 0)
            {
                funcs_v2[op](src1_, src2_, dst_, stream);
                return;
            }
        }
    }

    const func_t func = funcs_v1[op][depth];

    func(src1_, src2_, dst_, stream);
}

///////////////////////////////////////////////////////////////////////
/// minMaxScalar

namespace
{
    template <template <typename> class Op, typename T>
    void minMaxScalar(const GpuMat& src, double value, GpuMat& dst, Stream& stream)
    {
        gridTransformUnary(globPtr<T>(src), globPtr<T>(dst), bind2nd(Op<T>(), cv::saturate_cast<T>(value)), stream);
    }
}

void minMaxScalar(const GpuMat& src, cv::Scalar value, bool, GpuMat& dst, const GpuMat&, double, Stream& stream, int op)
{
    typedef void (*func_t)(const GpuMat& src, double value, GpuMat& dst, Stream& stream);
    static const func_t funcs[2][7] =
    {
        {
            minMaxScalar<minimum, uchar>,
            minMaxScalar<minimum, schar>,
            minMaxScalar<minimum, ushort>,
            minMaxScalar<minimum, short>,
            minMaxScalar<minimum, int>,
            minMaxScalar<minimum, float>,
            minMaxScalar<minimum, double>
        },
        {
            minMaxScalar<maximum, uchar>,
            minMaxScalar<maximum, schar>,
            minMaxScalar<maximum, ushort>,
            minMaxScalar<maximum, short>,
            minMaxScalar<maximum, int>,
            minMaxScalar<maximum, float>,
            minMaxScalar<maximum, double>
        }
    };

    const int depth = src.depth();

    CV_DbgAssert( depth <= CV_64F );
    CV_DbgAssert( src.channels() == 1 );

    funcs[op][depth](src, value[0], dst, stream);
}

#endif
