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

namespace
{
    template <typename T, typename S, typename D>
    void reduceToRowImpl(const GpuMat& _src, GpuMat& _dst, int reduceOp, Stream& stream)
    {
        const GpuMat_<T>& src = (const GpuMat_<T>&) _src;
        GpuMat_<D>& dst = (GpuMat_<D>&) _dst;

        switch (reduceOp)
        {
        case cv::REDUCE_SUM:
            gridReduceToRow< Sum<S> >(src, dst, stream);
            break;

        case cv::REDUCE_AVG:
            gridReduceToRow< Avg<S> >(src, dst, stream);
            break;

        case cv::REDUCE_MIN:
            gridReduceToRow< Min<S> >(src, dst, stream);
            break;

        case cv::REDUCE_MAX:
            gridReduceToRow< Max<S> >(src, dst, stream);
            break;
        };
    }

    template <typename T, typename S, typename D>
    void reduceToColumnImpl_(const GpuMat& _src, GpuMat& _dst, int reduceOp, Stream& stream)
    {
        const GpuMat_<T>& src = (const GpuMat_<T>&) _src;
        GpuMat_<D>& dst = (GpuMat_<D>&) _dst;

        switch (reduceOp)
        {
        case cv::REDUCE_SUM:
            gridReduceToColumn< Sum<S> >(src, dst, stream);
            break;

        case cv::REDUCE_AVG:
            gridReduceToColumn< Avg<S> >(src, dst, stream);
            break;

        case cv::REDUCE_MIN:
            gridReduceToColumn< Min<S> >(src, dst, stream);
            break;

        case cv::REDUCE_MAX:
            gridReduceToColumn< Max<S> >(src, dst, stream);
            break;
        };
    }

    template <typename T, typename S, typename D>
    void reduceToColumnImpl(const GpuMat& src, GpuMat& dst, int reduceOp, Stream& stream)
    {
        typedef void (*func_t)(const GpuMat& src, GpuMat& dst, int reduceOp, Stream& stream);
        static const func_t funcs[4] =
        {
            reduceToColumnImpl_<T, S, D>,
            reduceToColumnImpl_<typename MakeVec<T, 2>::type, typename MakeVec<S, 2>::type, typename MakeVec<D, 2>::type>,
            reduceToColumnImpl_<typename MakeVec<T, 3>::type, typename MakeVec<S, 3>::type, typename MakeVec<D, 3>::type>,
            reduceToColumnImpl_<typename MakeVec<T, 4>::type, typename MakeVec<S, 4>::type, typename MakeVec<D, 4>::type>
        };

        funcs[src.channels() - 1](src, dst, reduceOp, stream);
    }
}

void cv::cuda::reduce(InputArray _src, OutputArray _dst, int dim, int reduceOp, int dtype, Stream& stream)
{
    GpuMat src = getInputMat(_src, stream);

    CV_Assert( src.channels() <= 4 );
    CV_Assert( dim == 0 || dim == 1 );
    CV_Assert( reduceOp == REDUCE_SUM || reduceOp == REDUCE_AVG || reduceOp == REDUCE_MAX || reduceOp == REDUCE_MIN );

    if (dtype < 0)
        dtype = src.depth();

    GpuMat dst = getOutputMat(_dst, 1, dim == 0 ? src.cols : src.rows, CV_MAKE_TYPE(CV_MAT_DEPTH(dtype), src.channels()), stream);

    if (dim == 0)
    {
        typedef void (*func_t)(const GpuMat& _src, GpuMat& _dst, int reduceOp, Stream& stream);
        static const func_t funcs[7][7] =
        {
            {
                reduceToRowImpl<uchar, int, uchar>,
                0 /*reduceToRowImpl<uchar, int, schar>*/,
                0 /*reduceToRowImpl<uchar, int, ushort>*/,
                0 /*reduceToRowImpl<uchar, int, short>*/,
                reduceToRowImpl<uchar, int, int>,
                reduceToRowImpl<uchar, float, float>,
                reduceToRowImpl<uchar, double, double>
            },
            {
                0 /*reduceToRowImpl<schar, int, uchar>*/,
                0 /*reduceToRowImpl<schar, int, schar>*/,
                0 /*reduceToRowImpl<schar, int, ushort>*/,
                0 /*reduceToRowImpl<schar, int, short>*/,
                0 /*reduceToRowImpl<schar, int, int>*/,
                0 /*reduceToRowImpl<schar, float, float>*/,
                0 /*reduceToRowImpl<schar, double, double>*/
            },
            {
                0 /*reduceToRowImpl<ushort, int, uchar>*/,
                0 /*reduceToRowImpl<ushort, int, schar>*/,
                reduceToRowImpl<ushort, int, ushort>,
                0 /*reduceToRowImpl<ushort, int, short>*/,
                reduceToRowImpl<ushort, int, int>,
                reduceToRowImpl<ushort, float, float>,
                reduceToRowImpl<ushort, double, double>
            },
            {
                0 /*reduceToRowImpl<short, int, uchar>*/,
                0 /*reduceToRowImpl<short, int, schar>*/,
                0 /*reduceToRowImpl<short, int, ushort>*/,
                reduceToRowImpl<short, int, short>,
                reduceToRowImpl<short, int, int>,
                reduceToRowImpl<short, float, float>,
                reduceToRowImpl<short, double, double>
            },
            {
                0 /*reduceToRowImpl<int, int, uchar>*/,
                0 /*reduceToRowImpl<int, int, schar>*/,
                0 /*reduceToRowImpl<int, int, ushort>*/,
                0 /*reduceToRowImpl<int, int, short>*/,
                reduceToRowImpl<int, int, int>,
                reduceToRowImpl<int, float, float>,
                reduceToRowImpl<int, double, double>
            },
            {
                0 /*reduceToRowImpl<float, float, uchar>*/,
                0 /*reduceToRowImpl<float, float, schar>*/,
                0 /*reduceToRowImpl<float, float, ushort>*/,
                0 /*reduceToRowImpl<float, float, short>*/,
                0 /*reduceToRowImpl<float, float, int>*/,
                reduceToRowImpl<float, float, float>,
                reduceToRowImpl<float, double, double>
            },
            {
                0 /*reduceToRowImpl<double, double, uchar>*/,
                0 /*reduceToRowImpl<double, double, schar>*/,
                0 /*reduceToRowImpl<double, double, ushort>*/,
                0 /*reduceToRowImpl<double, double, short>*/,
                0 /*reduceToRowImpl<double, double, int>*/,
                0 /*reduceToRowImpl<double, double, float>*/,
                reduceToRowImpl<double, double, double>
            }
        };

        const func_t func = funcs[src.depth()][dst.depth()];

        if (!func)
            CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported combination of input and output array formats");

        GpuMat dst_cont = dst.reshape(1);
        func(src.reshape(1), dst_cont, reduceOp, stream);
    }
    else
    {
        typedef void (*func_t)(const GpuMat& _src, GpuMat& _dst, int reduceOp, Stream& stream);
        static const func_t funcs[7][7] =
        {
            {
                reduceToColumnImpl<uchar, int, uchar>,
                0 /*reduceToColumnImpl<uchar, int, schar>*/,
                0 /*reduceToColumnImpl<uchar, int, ushort>*/,
                0 /*reduceToColumnImpl<uchar, int, short>*/,
                reduceToColumnImpl<uchar, int, int>,
                reduceToColumnImpl<uchar, float, float>,
                reduceToColumnImpl<uchar, double, double>
            },
            {
                0 /*reduceToColumnImpl<schar, int, uchar>*/,
                0 /*reduceToColumnImpl<schar, int, schar>*/,
                0 /*reduceToColumnImpl<schar, int, ushort>*/,
                0 /*reduceToColumnImpl<schar, int, short>*/,
                0 /*reduceToColumnImpl<schar, int, int>*/,
                0 /*reduceToColumnImpl<schar, float, float>*/,
                0 /*reduceToColumnImpl<schar, double, double>*/
            },
            {
                0 /*reduceToColumnImpl<ushort, int, uchar>*/,
                0 /*reduceToColumnImpl<ushort, int, schar>*/,
                reduceToColumnImpl<ushort, int, ushort>,
                0 /*reduceToColumnImpl<ushort, int, short>*/,
                reduceToColumnImpl<ushort, int, int>,
                reduceToColumnImpl<ushort, float, float>,
                reduceToColumnImpl<ushort, double, double>
            },
            {
                0 /*reduceToColumnImpl<short, int, uchar>*/,
                0 /*reduceToColumnImpl<short, int, schar>*/,
                0 /*reduceToColumnImpl<short, int, ushort>*/,
                reduceToColumnImpl<short, int, short>,
                reduceToColumnImpl<short, int, int>,
                reduceToColumnImpl<short, float, float>,
                reduceToColumnImpl<short, double, double>
            },
            {
                0 /*reduceToColumnImpl<int, int, uchar>*/,
                0 /*reduceToColumnImpl<int, int, schar>*/,
                0 /*reduceToColumnImpl<int, int, ushort>*/,
                0 /*reduceToColumnImpl<int, int, short>*/,
                reduceToColumnImpl<int, int, int>,
                reduceToColumnImpl<int, float, float>,
                reduceToColumnImpl<int, double, double>
            },
            {
                0 /*reduceToColumnImpl<float, float, uchar>*/,
                0 /*reduceToColumnImpl<float, float, schar>*/,
                0 /*reduceToColumnImpl<float, float, ushort>*/,
                0 /*reduceToColumnImpl<float, float, short>*/,
                0 /*reduceToColumnImpl<float, float, int>*/,
                reduceToColumnImpl<float, float, float>,
                reduceToColumnImpl<float, double, double>
            },
            {
                0 /*reduceToColumnImpl<double, double, uchar>*/,
                0 /*reduceToColumnImpl<double, double, schar>*/,
                0 /*reduceToColumnImpl<double, double, ushort>*/,
                0 /*reduceToColumnImpl<double, double, short>*/,
                0 /*reduceToColumnImpl<double, double, int>*/,
                0 /*reduceToColumnImpl<double, double, float>*/,
                reduceToColumnImpl<double, double, double>
            }
        };

        const func_t func = funcs[src.depth()][dst.depth()];

        if (!func)
            CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported combination of input and output array formats");

        func(src, dst, reduceOp, stream);
    }

    syncOutput(dst, _dst, stream);
}

#endif
