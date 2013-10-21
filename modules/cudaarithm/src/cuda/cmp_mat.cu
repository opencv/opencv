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

void cmpMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat&, double, Stream& stream, int cmpop);

namespace
{
    template <class Op, typename T> struct CmpOp : binary_function<T, T, uchar>
    {
        __device__ __forceinline__ uchar operator()(T a, T b) const
        {
            Op op;
            return -op(a, b);
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

    template <template <typename> class Op, typename T>
    void cmpMat_v1(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream)
    {
        CmpOp<Op<T>, T> op;
        gridTransformBinary_< TransformPolicy<T> >(globPtr<T>(src1), globPtr<T>(src2), globPtr<uchar>(dst), op, stream);
    }

    struct VCmpEq4 : binary_function<uint, uint, uint>
    {
        __device__ __forceinline__ uint operator ()(uint a, uint b) const
        {
            return vcmpeq4(a, b);
        }
    };
    struct VCmpNe4 : binary_function<uint, uint, uint>
    {
        __device__ __forceinline__ uint operator ()(uint a, uint b) const
        {
            return vcmpne4(a, b);
        }
    };
    struct VCmpLt4 : binary_function<uint, uint, uint>
    {
        __device__ __forceinline__ uint operator ()(uint a, uint b) const
        {
            return vcmplt4(a, b);
        }
    };
    struct VCmpLe4 : binary_function<uint, uint, uint>
    {
        __device__ __forceinline__ uint operator ()(uint a, uint b) const
        {
            return vcmple4(a, b);
        }
    };

    void cmpMatEq_v4(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream)
    {
        const int vcols = src1.cols >> 2;

        GlobPtrSz<uint> src1_ = globPtr((uint*) src1.data, src1.step, src1.rows, vcols);
        GlobPtrSz<uint> src2_ = globPtr((uint*) src2.data, src2.step, src1.rows, vcols);
        GlobPtrSz<uint> dst_ = globPtr((uint*) dst.data, dst.step, src1.rows, vcols);

        gridTransformBinary(src1_, src2_, dst_, VCmpEq4(), stream);
    }
    void cmpMatNe_v4(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream)
    {
        const int vcols = src1.cols >> 2;

        GlobPtrSz<uint> src1_ = globPtr((uint*) src1.data, src1.step, src1.rows, vcols);
        GlobPtrSz<uint> src2_ = globPtr((uint*) src2.data, src2.step, src1.rows, vcols);
        GlobPtrSz<uint> dst_ = globPtr((uint*) dst.data, dst.step, src1.rows, vcols);

        gridTransformBinary(src1_, src2_, dst_, VCmpNe4(), stream);
    }
    void cmpMatLt_v4(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream)
    {
        const int vcols = src1.cols >> 2;

        GlobPtrSz<uint> src1_ = globPtr((uint*) src1.data, src1.step, src1.rows, vcols);
        GlobPtrSz<uint> src2_ = globPtr((uint*) src2.data, src2.step, src1.rows, vcols);
        GlobPtrSz<uint> dst_ = globPtr((uint*) dst.data, dst.step, src1.rows, vcols);

        gridTransformBinary(src1_, src2_, dst_, VCmpLt4(), stream);
    }
    void cmpMatLe_v4(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream)
    {
        const int vcols = src1.cols >> 2;

        GlobPtrSz<uint> src1_ = globPtr((uint*) src1.data, src1.step, src1.rows, vcols);
        GlobPtrSz<uint> src2_ = globPtr((uint*) src2.data, src2.step, src1.rows, vcols);
        GlobPtrSz<uint> dst_ = globPtr((uint*) dst.data, dst.step, src1.rows, vcols);

        gridTransformBinary(src1_, src2_, dst_, VCmpLe4(), stream);
    }
}

void cmpMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat&, double, Stream& stream, int cmpop)
{
    typedef void (*func_t)(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream);
    static const func_t funcs[7][4] =
    {
        {cmpMat_v1<equal_to, uchar> , cmpMat_v1<not_equal_to, uchar> , cmpMat_v1<less, uchar> , cmpMat_v1<less_equal, uchar> },
        {cmpMat_v1<equal_to, schar> , cmpMat_v1<not_equal_to, schar> , cmpMat_v1<less, schar> , cmpMat_v1<less_equal, schar> },
        {cmpMat_v1<equal_to, ushort>, cmpMat_v1<not_equal_to, ushort>, cmpMat_v1<less, ushort>, cmpMat_v1<less_equal, ushort>},
        {cmpMat_v1<equal_to, short> , cmpMat_v1<not_equal_to, short> , cmpMat_v1<less, short> , cmpMat_v1<less_equal, short> },
        {cmpMat_v1<equal_to, int>   , cmpMat_v1<not_equal_to, int>   , cmpMat_v1<less, int>   , cmpMat_v1<less_equal, int>   },
        {cmpMat_v1<equal_to, float> , cmpMat_v1<not_equal_to, float> , cmpMat_v1<less, float> , cmpMat_v1<less_equal, float> },
        {cmpMat_v1<equal_to, double>, cmpMat_v1<not_equal_to, double>, cmpMat_v1<less, double>, cmpMat_v1<less_equal, double>}
    };

    typedef void (*func_v4_t)(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream);
    static const func_v4_t funcs_v4[] =
    {
        cmpMatEq_v4, cmpMatNe_v4, cmpMatLt_v4, cmpMatLe_v4
    };

    const int depth = src1.depth();

    CV_DbgAssert( depth <= CV_64F );

    static const int codes[] =
    {
        0, 2, 3, 2, 3, 1
    };
    const GpuMat* psrc1[] =
    {
        &src1, &src2, &src2, &src1, &src1, &src1
    };
    const GpuMat* psrc2[] =
    {
        &src2, &src1, &src1, &src2, &src2, &src2
    };

    const int code = codes[cmpop];

    GpuMat src1_ = psrc1[cmpop]->reshape(1);
    GpuMat src2_ = psrc2[cmpop]->reshape(1);
    GpuMat dst_ = dst.reshape(1);

    if (depth == CV_8U && (src1_.cols & 3) == 0)
    {
        const intptr_t src1ptr = reinterpret_cast<intptr_t>(src1_.data);
        const intptr_t src2ptr = reinterpret_cast<intptr_t>(src2_.data);
        const intptr_t dstptr = reinterpret_cast<intptr_t>(dst_.data);

        const bool isAllAligned = (src1ptr & 31) == 0 && (src2ptr & 31) == 0 && (dstptr & 31) == 0;

        if (isAllAligned)
        {
            funcs_v4[code](src1_, src2_, dst_, stream);
            return;
        }
    }

    const func_t func = funcs[depth][code];

    func(src1_, src2_, dst_, stream);
}

#endif
