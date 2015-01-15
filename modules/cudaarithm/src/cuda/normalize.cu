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

namespace {

template <typename T, typename R, typename I>
struct ConvertorMinMax : unary_function<T, R>
{
    typedef typename LargerType<T, R>::type larger_type1;
    typedef typename LargerType<larger_type1, I>::type larger_type2;
    typedef typename LargerType<larger_type2, float>::type scalar_type;

    scalar_type dmin, dmax;
    const I* minMaxVals;

    __device__ R operator ()(typename TypeTraits<T>::parameter_type src) const
    {
        const scalar_type smin = minMaxVals[0];
        const scalar_type smax = minMaxVals[1];

        const scalar_type scale = (dmax - dmin) * (smax - smin > numeric_limits<scalar_type>::epsilon() ? 1.0 / (smax - smin) : 0.0);
        const scalar_type shift = dmin - smin * scale;

        return cudev::saturate_cast<R>(scale * src + shift);
    }
};

template <typename T, typename R, typename I>
void normalizeMinMax(const GpuMat& _src, GpuMat& _dst, double a, double b, const GpuMat& mask, Stream& stream)
{
    const GpuMat_<T>& src = (const GpuMat_<T>&)_src;
    GpuMat_<R>& dst = (GpuMat_<R>&)_dst;

    BufferPool pool(stream);
    GpuMat_<I> minMaxVals(1, 2, pool.getAllocator());

    if (mask.empty())
    {
        gridFindMinMaxVal(src, minMaxVals, stream);
    }
    else
    {
        gridFindMinMaxVal(src, minMaxVals, globPtr<uchar>(mask), stream);
    }

    ConvertorMinMax<T, R, I> cvt;
    cvt.dmin = std::min(a, b);
    cvt.dmax = std::max(a, b);
    cvt.minMaxVals = minMaxVals[0];

    if (mask.empty())
    {
        gridTransformUnary(src, dst, cvt, stream);
    }
    else
    {
        dst.setTo(Scalar::all(0), stream);
        gridTransformUnary(src, dst, cvt, globPtr<uchar>(mask), stream);
    }
}

template <typename T, typename R, typename I, bool normL2>
struct ConvertorNorm : unary_function<T, R>
{
    typedef typename LargerType<T, R>::type larger_type1;
    typedef typename LargerType<larger_type1, I>::type larger_type2;
    typedef typename LargerType<larger_type2, float>::type scalar_type;

    scalar_type a;
    const I* normVal;

    __device__ R operator ()(typename TypeTraits<T>::parameter_type src) const
    {
        sqrt_func<scalar_type> sqrt;

        scalar_type scale = normL2 ? sqrt(*normVal) : *normVal;
        scale = scale > numeric_limits<scalar_type>::epsilon() ? a / scale : 0.0;

        return cudev::saturate_cast<R>(scale * src);
    }
};

template <typename T, typename R, typename I>
void normalizeNorm(const GpuMat& _src, GpuMat& _dst, double a, int normType, const GpuMat& mask, Stream& stream)
{
    const GpuMat_<T>& src = (const GpuMat_<T>&)_src;
    GpuMat_<R>& dst = (GpuMat_<R>&)_dst;

    BufferPool pool(stream);
    GpuMat_<I> normVal(1, 1, pool.getAllocator());

    if (normType == NORM_L1)
    {
        if (mask.empty())
        {
            gridCalcSum(abs_(cvt_<I>(src)), normVal, stream);
        }
        else
        {
            gridCalcSum(abs_(cvt_<I>(src)), normVal, globPtr<uchar>(mask), stream);
        }
    }
    else if (normType == NORM_L2)
    {
        if (mask.empty())
        {
            gridCalcSum(sqr_(cvt_<I>(src)), normVal, stream);
        }
        else
        {
            gridCalcSum(sqr_(cvt_<I>(src)), normVal, globPtr<uchar>(mask), stream);
        }
    }
    else // NORM_INF
    {
        if (mask.empty())
        {
            gridFindMaxVal(abs_(cvt_<I>(src)), normVal, stream);
        }
        else
        {
            gridFindMaxVal(abs_(cvt_<I>(src)), normVal, globPtr<uchar>(mask), stream);
        }
    }

    if (normType == NORM_L2)
    {
        ConvertorNorm<T, R, I, true> cvt;
        cvt.a = a;
        cvt.normVal = normVal[0];

        if (mask.empty())
        {
            gridTransformUnary(src, dst, cvt, stream);
        }
        else
        {
            dst.setTo(Scalar::all(0), stream);
            gridTransformUnary(src, dst, cvt, globPtr<uchar>(mask), stream);
        }
    }
    else
    {
        ConvertorNorm<T, R, I, false> cvt;
        cvt.a = a;
        cvt.normVal = normVal[0];

        if (mask.empty())
        {
            gridTransformUnary(src, dst, cvt, stream);
        }
        else
        {
            dst.setTo(Scalar::all(0), stream);
            gridTransformUnary(src, dst, cvt, globPtr<uchar>(mask), stream);
        }
    }
}

} // namespace

void cv::cuda::normalize(InputArray _src, OutputArray _dst, double a, double b, int normType, int dtype, InputArray _mask, Stream& stream)
{
    typedef void (*func_minmax_t)(const GpuMat& _src, GpuMat& _dst, double a, double b, const GpuMat& mask, Stream& stream);
    typedef void (*func_norm_t)(const GpuMat& _src, GpuMat& _dst, double a, int normType, const GpuMat& mask, Stream& stream);

    static const func_minmax_t funcs_minmax[] =
    {
        normalizeMinMax<uchar, float, float>,
        normalizeMinMax<schar, float, float>,
        normalizeMinMax<ushort, float, float>,
        normalizeMinMax<short, float, float>,
        normalizeMinMax<int, float, float>,
        normalizeMinMax<float, float, float>,
        normalizeMinMax<double, double, double>
    };

    static const func_norm_t funcs_norm[] =
    {
        normalizeNorm<uchar, float, float>,
        normalizeNorm<schar, float, float>,
        normalizeNorm<ushort, float, float>,
        normalizeNorm<short, float, float>,
        normalizeNorm<int, float, float>,
        normalizeNorm<float, float, float>,
        normalizeNorm<double, double, double>
    };

    CV_Assert( normType == NORM_INF || normType == NORM_L1 || normType == NORM_L2 || normType == NORM_MINMAX );

    const GpuMat src = getInputMat(_src, stream);
    const GpuMat mask = getInputMat(_mask, stream);

    CV_Assert( src.channels() == 1 );
    CV_Assert( mask.empty() || (mask.size() == src.size() && mask.type() == CV_8U) );

    dtype = CV_MAT_DEPTH(dtype);

    const int src_depth = src.depth();
    const int tmp_depth = src_depth <= CV_32F ? CV_32F : src_depth;

    GpuMat dst;
    if (dtype == tmp_depth)
    {
        _dst.create(src.size(), tmp_depth);
        dst = getOutputMat(_dst, src.size(), tmp_depth, stream);
    }
    else
    {
        BufferPool pool(stream);
        dst = pool.getBuffer(src.size(), tmp_depth);
    }

    if (normType == NORM_MINMAX)
    {
        const func_minmax_t func = funcs_minmax[src_depth];
        func(src, dst, a, b, mask, stream);
    }
    else
    {
        const func_norm_t func = funcs_norm[src_depth];
        func(src, dst, a, normType, mask, stream);
    }

    if (dtype == tmp_depth)
    {
        syncOutput(dst, _dst, stream);
    }
    else
    {
        dst.convertTo(_dst, dtype, stream);
    }
}

#endif
