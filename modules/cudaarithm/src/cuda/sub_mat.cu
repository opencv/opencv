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

void subMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, double, Stream& _stream, int);

namespace
{
    template <typename T, typename D> struct SubOp1 : binary_function<T, T, D>
    {
        __device__ __forceinline__ D operator ()(T a, T b) const
        {
            return saturate_cast<D>(a - b);
        }
    };

    template <typename T, typename D>
    void subMat_v1(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, Stream& stream)
    {
        if (mask.data)
            gridTransformBinary(globPtr<T>(src1), globPtr<T>(src2), globPtr<D>(dst), SubOp1<T, D>(), globPtr<uchar>(mask), stream);
        else
            gridTransformBinary(globPtr<T>(src1), globPtr<T>(src2), globPtr<D>(dst), SubOp1<T, D>(), stream);
    }

    struct SubOp2 : binary_function<uint, uint, uint>
    {
        __device__ __forceinline__ uint operator ()(uint a, uint b) const
        {
            return vsub2(a, b);
        }
    };

    void subMat_v2(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream)
    {
        const int vcols = src1.cols >> 1;

        GlobPtrSz<uint> src1_ = globPtr((uint*) src1.data, src1.step, src1.rows, vcols);
        GlobPtrSz<uint> src2_ = globPtr((uint*) src2.data, src2.step, src1.rows, vcols);
        GlobPtrSz<uint> dst_ = globPtr((uint*) dst.data, dst.step, src1.rows, vcols);

        gridTransformBinary(src1_, src2_, dst_, SubOp2(), stream);
    }

    struct SubOp4 : binary_function<uint, uint, uint>
    {
        __device__ __forceinline__ uint operator ()(uint a, uint b) const
        {
            return vsub4(a, b);
        }
    };

    void subMat_v4(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream)
    {
        const int vcols = src1.cols >> 2;

        GlobPtrSz<uint> src1_ = globPtr((uint*) src1.data, src1.step, src1.rows, vcols);
        GlobPtrSz<uint> src2_ = globPtr((uint*) src2.data, src2.step, src1.rows, vcols);
        GlobPtrSz<uint> dst_ = globPtr((uint*) dst.data, dst.step, src1.rows, vcols);

        gridTransformBinary(src1_, src2_, dst_, SubOp4(), stream);
    }
}

void subMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, double, Stream& stream, int)
{
    typedef void (*func_t)(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, Stream& stream);
    static const func_t funcs[7][7] =
    {
        {
            subMat_v1<uchar, uchar>,
            subMat_v1<uchar, schar>,
            subMat_v1<uchar, ushort>,
            subMat_v1<uchar, short>,
            subMat_v1<uchar, int>,
            subMat_v1<uchar, float>,
            subMat_v1<uchar, double>
        },
        {
            subMat_v1<schar, uchar>,
            subMat_v1<schar, schar>,
            subMat_v1<schar, ushort>,
            subMat_v1<schar, short>,
            subMat_v1<schar, int>,
            subMat_v1<schar, float>,
            subMat_v1<schar, double>
        },
        {
            0 /*subMat_v1<ushort, uchar>*/,
            0 /*subMat_v1<ushort, schar>*/,
            subMat_v1<ushort, ushort>,
            subMat_v1<ushort, short>,
            subMat_v1<ushort, int>,
            subMat_v1<ushort, float>,
            subMat_v1<ushort, double>
        },
        {
            0 /*subMat_v1<short, uchar>*/,
            0 /*subMat_v1<short, schar>*/,
            subMat_v1<short, ushort>,
            subMat_v1<short, short>,
            subMat_v1<short, int>,
            subMat_v1<short, float>,
            subMat_v1<short, double>
        },
        {
            0 /*subMat_v1<int, uchar>*/,
            0 /*subMat_v1<int, schar>*/,
            0 /*subMat_v1<int, ushort>*/,
            0 /*subMat_v1<int, short>*/,
            subMat_v1<int, int>,
            subMat_v1<int, float>,
            subMat_v1<int, double>
        },
        {
            0 /*subMat_v1<float, uchar>*/,
            0 /*subMat_v1<float, schar>*/,
            0 /*subMat_v1<float, ushort>*/,
            0 /*subMat_v1<float, short>*/,
            0 /*subMat_v1<float, int>*/,
            subMat_v1<float, float>,
            subMat_v1<float, double>
        },
        {
            0 /*subMat_v1<double, uchar>*/,
            0 /*subMat_v1<double, schar>*/,
            0 /*subMat_v1<double, ushort>*/,
            0 /*subMat_v1<double, short>*/,
            0 /*subMat_v1<double, int>*/,
            0 /*subMat_v1<double, float>*/,
            subMat_v1<double, double>
        }
    };

    const int sdepth = src1.depth();
    const int ddepth = dst.depth();

    CV_DbgAssert( sdepth <= CV_64F && ddepth <= CV_64F );

    GpuMat src1_ = src1.reshape(1);
    GpuMat src2_ = src2.reshape(1);
    GpuMat dst_ = dst.reshape(1);

    if (mask.empty() && (sdepth == CV_8U || sdepth == CV_16U) && ddepth == sdepth)
    {
        const intptr_t src1ptr = reinterpret_cast<intptr_t>(src1_.data);
        const intptr_t src2ptr = reinterpret_cast<intptr_t>(src2_.data);
        const intptr_t dstptr = reinterpret_cast<intptr_t>(dst_.data);

        const bool isAllAligned = (src1ptr & 31) == 0 && (src2ptr & 31) == 0 && (dstptr & 31) == 0;

        if (isAllAligned)
        {
            if (sdepth == CV_8U && (src1_.cols & 3) == 0)
            {
                subMat_v4(src1_, src2_, dst_, stream);
                return;
            }
            else if (sdepth == CV_16U && (src1_.cols & 1) == 0)
            {
                subMat_v2(src1_, src2_, dst_, stream);
                return;
            }
        }
    }

    const func_t func = funcs[sdepth][ddepth];

    if (!func)
        CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported combination of source and destination types");

    func(src1_, src2_, dst_, mask, stream);
}

#endif
