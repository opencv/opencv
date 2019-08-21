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

#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudev.hpp"
#include "opencv2/core/private.cuda.hpp"

using namespace cv;
using namespace cv::cuda;
using namespace cv::cudev;

void bitMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, double, Stream& stream, int op);

//////////////////////////////////////////////////////////////////////////////
/// bitwise_not

void cv::cuda::bitwise_not(InputArray _src, OutputArray _dst, InputArray _mask, Stream& stream)
{
    GpuMat src = getInputMat(_src, stream);
    GpuMat mask = getInputMat(_mask, stream);

    const int depth = src.depth();

    CV_DbgAssert( depth <= CV_32F );
    CV_DbgAssert( mask.empty() || (mask.type() == CV_8UC1 && mask.size() == src.size()) );

    GpuMat dst = getOutputMat(_dst, src.size(), src.type(), stream);

    if (mask.empty())
    {
        const int bcols = (int) (src.cols * src.elemSize());

        if ((bcols & 3) == 0)
        {
            const int vcols = bcols >> 2;

            GlobPtrSz<uint> vsrc = globPtr((uint*) src.data, src.step, src.rows, vcols);
            GlobPtrSz<uint> vdst = globPtr((uint*) dst.data, dst.step, src.rows, vcols);

            gridTransformUnary(vsrc, vdst, bit_not<uint>(), stream);
        }
        else if ((bcols & 1) == 0)
        {
            const int vcols = bcols >> 1;

            GlobPtrSz<ushort> vsrc = globPtr((ushort*) src.data, src.step, src.rows, vcols);
            GlobPtrSz<ushort> vdst = globPtr((ushort*) dst.data, dst.step, src.rows, vcols);

            gridTransformUnary(vsrc, vdst, bit_not<ushort>(), stream);
        }
        else
        {
            GlobPtrSz<uchar> vsrc = globPtr((uchar*) src.data, src.step, src.rows, bcols);
            GlobPtrSz<uchar> vdst = globPtr((uchar*) dst.data, dst.step, src.rows, bcols);

            gridTransformUnary(vsrc, vdst, bit_not<uchar>(), stream);
        }
    }
    else
    {
        if (depth == CV_32F || depth == CV_32S)
        {
            GlobPtrSz<uint> vsrc = globPtr((uint*) src.data, src.step, src.rows, src.cols * src.channels());
            GlobPtrSz<uint> vdst = globPtr((uint*) dst.data, dst.step, src.rows, src.cols * src.channels());

            gridTransformUnary(vsrc, vdst, bit_not<uint>(), singleMaskChannels(globPtr<uchar>(mask), src.channels()), stream);
        }
        else if (depth == CV_16S || depth == CV_16U)
        {
            GlobPtrSz<ushort> vsrc = globPtr((ushort*) src.data, src.step, src.rows, src.cols * src.channels());
            GlobPtrSz<ushort> vdst = globPtr((ushort*) dst.data, dst.step, src.rows, src.cols * src.channels());

            gridTransformUnary(vsrc, vdst, bit_not<ushort>(), singleMaskChannels(globPtr<uchar>(mask), src.channels()), stream);
        }
        else
        {
            GlobPtrSz<uchar> vsrc = globPtr((uchar*) src.data, src.step, src.rows, src.cols * src.channels());
            GlobPtrSz<uchar> vdst = globPtr((uchar*) dst.data, dst.step, src.rows, src.cols * src.channels());

            gridTransformUnary(vsrc, vdst, bit_not<uchar>(), singleMaskChannels(globPtr<uchar>(mask), src.channels()), stream);
        }
    }

    syncOutput(dst, _dst, stream);
}

//////////////////////////////////////////////////////////////////////////////
/// Binary bitwise logical operations

namespace
{
    template <template <typename> class Op, typename T>
    void bitMatOp(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, Stream& stream)
    {
        GlobPtrSz<T> vsrc1 = globPtr((T*) src1.data, src1.step, src1.rows, src1.cols * src1.channels());
        GlobPtrSz<T> vsrc2 = globPtr((T*) src2.data, src2.step, src1.rows, src1.cols * src1.channels());
        GlobPtrSz<T> vdst = globPtr((T*) dst.data, dst.step, src1.rows, src1.cols * src1.channels());

        if (mask.data)
            gridTransformBinary(vsrc1, vsrc2, vdst, Op<T>(), singleMaskChannels(globPtr<uchar>(mask), src1.channels()), stream);
        else
            gridTransformBinary(vsrc1, vsrc2, vdst, Op<T>(), stream);
    }
}

void bitMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, double, Stream& stream, int op)
{
    typedef void (*func_t)(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, Stream& stream);
    static const func_t funcs32[] =
    {
        bitMatOp<bit_and, uint>,
        bitMatOp<bit_or, uint>,
        bitMatOp<bit_xor, uint>
    };
    static const func_t funcs16[] =
    {
        bitMatOp<bit_and, ushort>,
        bitMatOp<bit_or, ushort>,
        bitMatOp<bit_xor, ushort>
    };
    static const func_t funcs8[] =
    {
        bitMatOp<bit_and, uchar>,
        bitMatOp<bit_or, uchar>,
        bitMatOp<bit_xor, uchar>
    };

    const int depth = src1.depth();

    CV_DbgAssert( depth <= CV_32F );
    CV_DbgAssert( op >= 0 && op < 3 );

    if (mask.empty())
    {
        const int bcols = (int) (src1.cols * src1.elemSize());

        if ((bcols & 3) == 0)
        {
            const int vcols = bcols >> 2;

            GpuMat vsrc1(src1.rows, vcols, CV_32SC1, src1.data, src1.step);
            GpuMat vsrc2(src1.rows, vcols, CV_32SC1, src2.data, src2.step);
            GpuMat vdst(src1.rows, vcols, CV_32SC1, dst.data, dst.step);

            funcs32[op](vsrc1, vsrc2, vdst, GpuMat(), stream);
        }
        else if ((bcols & 1) == 0)
        {
            const int vcols = bcols >> 1;

            GpuMat vsrc1(src1.rows, vcols, CV_16UC1, src1.data, src1.step);
            GpuMat vsrc2(src1.rows, vcols, CV_16UC1, src2.data, src2.step);
            GpuMat vdst(src1.rows, vcols, CV_16UC1, dst.data, dst.step);

            funcs16[op](vsrc1, vsrc2, vdst, GpuMat(), stream);
        }
        else
        {
            GpuMat vsrc1(src1.rows, bcols, CV_8UC1, src1.data, src1.step);
            GpuMat vsrc2(src1.rows, bcols, CV_8UC1, src2.data, src2.step);
            GpuMat vdst(src1.rows, bcols, CV_8UC1, dst.data, dst.step);

            funcs8[op](vsrc1, vsrc2, vdst, GpuMat(), stream);
        }
    }
    else
    {
        if (depth == CV_32F || depth == CV_32S)
        {
            funcs32[op](src1, src2, dst, mask, stream);
        }
        else if (depth == CV_16S || depth == CV_16U)
        {
            funcs16[op](src1, src2, dst, mask, stream);
        }
        else
        {
            funcs8[op](src1, src2, dst, mask, stream);
        }
    }
}

#endif
