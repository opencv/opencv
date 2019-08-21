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

void cv::cuda::magnitude(InputArray _x, InputArray _y, OutputArray _dst, Stream& stream)
{
    GpuMat x = getInputMat(_x, stream);
    GpuMat y = getInputMat(_y, stream);

    CV_Assert( x.depth() == CV_32F );
    CV_Assert( y.type() == x.type() && y.size() == x.size() );

    GpuMat dst = getOutputMat(_dst, x.size(), CV_32FC1, stream);

    GpuMat_<float> xc(x.reshape(1));
    GpuMat_<float> yc(y.reshape(1));
    GpuMat_<float> magc(dst.reshape(1));

    gridTransformBinary(xc, yc, magc, magnitude_func<float>(), stream);

    syncOutput(dst, _dst, stream);
}

void cv::cuda::magnitudeSqr(InputArray _x, InputArray _y, OutputArray _dst, Stream& stream)
{
    GpuMat x = getInputMat(_x, stream);
    GpuMat y = getInputMat(_y, stream);

    CV_Assert( x.depth() == CV_32F );
    CV_Assert( y.type() == x.type() && y.size() == x.size() );

    GpuMat dst = getOutputMat(_dst, x.size(), CV_32FC1, stream);

    GpuMat_<float> xc(x.reshape(1));
    GpuMat_<float> yc(y.reshape(1));
    GpuMat_<float> magc(dst.reshape(1));

    gridTransformBinary(xc, yc, magc, magnitude_sqr_func<float>(), stream);

    syncOutput(dst, _dst, stream);
}

void cv::cuda::phase(InputArray _x, InputArray _y, OutputArray _dst, bool angleInDegrees, Stream& stream)
{
    GpuMat x = getInputMat(_x, stream);
    GpuMat y = getInputMat(_y, stream);

    CV_Assert( x.depth() == CV_32F );
    CV_Assert( y.type() == x.type() && y.size() == x.size() );

    GpuMat dst = getOutputMat(_dst, x.size(), CV_32FC1, stream);

    GpuMat_<float> xc(x.reshape(1));
    GpuMat_<float> yc(y.reshape(1));
    GpuMat_<float> anglec(dst.reshape(1));

    if (angleInDegrees)
        gridTransformBinary(xc, yc, anglec, direction_func<float, true>(), stream);
    else
        gridTransformBinary(xc, yc, anglec, direction_func<float, false>(), stream);

    syncOutput(dst, _dst, stream);
}

void cv::cuda::cartToPolar(InputArray _x, InputArray _y, OutputArray _mag, OutputArray _angle, bool angleInDegrees, Stream& stream)
{
    GpuMat x = getInputMat(_x, stream);
    GpuMat y = getInputMat(_y, stream);

    CV_Assert( x.depth() == CV_32F );
    CV_Assert( y.type() == x.type() && y.size() == x.size() );

    GpuMat mag = getOutputMat(_mag, x.size(), CV_32FC1, stream);
    GpuMat angle = getOutputMat(_angle, x.size(), CV_32FC1, stream);

    GpuMat_<float> xc(x.reshape(1));
    GpuMat_<float> yc(y.reshape(1));
    GpuMat_<float> magc(mag.reshape(1));
    GpuMat_<float> anglec(angle.reshape(1));

    if (angleInDegrees)
    {
        gridTransformTuple(zipPtr(xc, yc),
                           tie(magc, anglec),
                           make_tuple(
                               binaryTupleAdapter<0, 1>(magnitude_func<float>()),
                               binaryTupleAdapter<0, 1>(direction_func<float, true>())),
                           stream);
    }
    else
    {
        gridTransformTuple(zipPtr(xc, yc),
                           tie(magc, anglec),
                           make_tuple(
                               binaryTupleAdapter<0, 1>(magnitude_func<float>()),
                               binaryTupleAdapter<0, 1>(direction_func<float, false>())),
                           stream);
    }

    syncOutput(mag, _mag, stream);
    syncOutput(angle, _angle, stream);
}

namespace
{
    template <typename T> struct sincos_op
    {
        __device__ __forceinline__ void operator()(T a, T *sptr, T *cptr) const
        {
            ::sincos(a, sptr, cptr);
        }
    };
    template <> struct sincos_op<float>
    {
        __device__ __forceinline__ void operator()(float a, float *sptr, float *cptr) const
        {
            ::sincosf(a, sptr, cptr);
        }
    };

    template <typename T, bool useMag>
    __global__ void polarToCartImpl_(const GlobPtr<T> mag, const GlobPtr<T> angle, GlobPtr<T> xmat, GlobPtr<T> ymat, const T scale, const int rows, const int cols)
    {
        const int x = blockDim.x * blockIdx.x + threadIdx.x;
        const int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (x >= cols || y >= rows)
            return;

        const T mag_val = useMag ? mag(y, x) : static_cast<T>(1.0);
        const T angle_val = angle(y, x);

        T sin_a, cos_a;
        sincos_op<T> op;
        op(scale * angle_val, &sin_a, &cos_a);

        xmat(y, x) = mag_val * cos_a;
        ymat(y, x) = mag_val * sin_a;
    }

    template <typename T>
    void polarToCartImpl(const GpuMat& mag, const GpuMat& angle, GpuMat& x, GpuMat& y, bool angleInDegrees, cudaStream_t& stream)
    {
        GpuMat_<T> xc(x.reshape(1));
        GpuMat_<T> yc(y.reshape(1));
        GpuMat_<T> magc(mag.reshape(1));
        GpuMat_<T> anglec(angle.reshape(1));

        const dim3 block(32, 8);
        const dim3 grid(divUp(anglec.cols, block.x), divUp(anglec.rows, block.y));

        const T scale = angleInDegrees ? static_cast<T>(CV_PI / 180.0) : static_cast<T>(1.0);

        if (magc.empty())
            polarToCartImpl_<T, false> << <grid, block, 0, stream >> >(shrinkPtr(magc), shrinkPtr(anglec), shrinkPtr(xc), shrinkPtr(yc), scale, anglec.rows, anglec.cols);
        else
            polarToCartImpl_<T, true> << <grid, block, 0, stream >> >(shrinkPtr(magc), shrinkPtr(anglec), shrinkPtr(xc), shrinkPtr(yc), scale, anglec.rows, anglec.cols);
    }
}

void cv::cuda::polarToCart(InputArray _mag, InputArray _angle, OutputArray _x, OutputArray _y, bool angleInDegrees, Stream& _stream)
{
    typedef void(*func_t)(const GpuMat& mag, const GpuMat& angle, GpuMat& x, GpuMat& y, bool angleInDegrees, cudaStream_t& stream);
    static const func_t funcs[7] = { 0, 0, 0, 0, 0, polarToCartImpl<float>, polarToCartImpl<double> };

    GpuMat mag = getInputMat(_mag, _stream);
    GpuMat angle = getInputMat(_angle, _stream);

    CV_Assert(angle.depth() == CV_32F || angle.depth() == CV_64F);
    CV_Assert( mag.empty() || (mag.type() == angle.type() && mag.size() == angle.size()) );

    GpuMat x = getOutputMat(_x, angle.size(), CV_MAKETYPE(angle.depth(), 1), _stream);
    GpuMat y = getOutputMat(_y, angle.size(), CV_MAKETYPE(angle.depth(), 1), _stream);

    cudaStream_t stream = StreamAccessor::getStream(_stream);
    funcs[angle.depth()](mag, angle, x, y, angleInDegrees, stream);
    CV_CUDEV_SAFE_CALL( cudaGetLastError() );

    syncOutput(x, _x, _stream);
    syncOutput(y, _y, _stream);

    if (stream == 0)
        CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
}

#endif
