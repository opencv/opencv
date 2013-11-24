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

using namespace cv::cudev;

void cv::cuda::magnitude(InputArray _x, InputArray _y, OutputArray _dst, Stream& stream)
{
    GpuMat x = _x.getGpuMat();
    GpuMat y = _y.getGpuMat();

    CV_DbgAssert( x.depth() == CV_32F );
    CV_DbgAssert( y.type() == x.type() && y.size() == x.size() );

    _dst.create(x.size(), CV_32FC1);
    GpuMat dst = _dst.getGpuMat();

    GpuMat_<float> xc(x.reshape(1));
    GpuMat_<float> yc(y.reshape(1));
    GpuMat_<float> magc(dst.reshape(1));

    gridTransformBinary(xc, yc, magc, magnitude_func<float>(), stream);
}

void cv::cuda::magnitudeSqr(InputArray _x, InputArray _y, OutputArray _dst, Stream& stream)
{
    GpuMat x = _x.getGpuMat();
    GpuMat y = _y.getGpuMat();

    CV_DbgAssert( x.depth() == CV_32F );
    CV_DbgAssert( y.type() == x.type() && y.size() == x.size() );

    _dst.create(x.size(), CV_32FC1);
    GpuMat dst = _dst.getGpuMat();

    GpuMat_<float> xc(x.reshape(1));
    GpuMat_<float> yc(y.reshape(1));
    GpuMat_<float> magc(dst.reshape(1));

    gridTransformBinary(xc, yc, magc, magnitude_sqr_func<float>(), stream);
}

void cv::cuda::phase(InputArray _x, InputArray _y, OutputArray _dst, bool angleInDegrees, Stream& stream)
{
    GpuMat x = _x.getGpuMat();
    GpuMat y = _y.getGpuMat();

    CV_DbgAssert( x.depth() == CV_32F );
    CV_DbgAssert( y.type() == x.type() && y.size() == x.size() );

    _dst.create(x.size(), CV_32FC1);
    GpuMat dst = _dst.getGpuMat();

    GpuMat_<float> xc(x.reshape(1));
    GpuMat_<float> yc(y.reshape(1));
    GpuMat_<float> anglec(dst.reshape(1));

    if (angleInDegrees)
        gridTransformBinary(xc, yc, anglec, direction_func<float, true>(), stream);
    else
        gridTransformBinary(xc, yc, anglec, direction_func<float, false>(), stream);
}

void cv::cuda::cartToPolar(InputArray _x, InputArray _y, OutputArray _mag, OutputArray _angle, bool angleInDegrees, Stream& stream)
{
    GpuMat x = _x.getGpuMat();
    GpuMat y = _y.getGpuMat();

    CV_DbgAssert( x.depth() == CV_32F );
    CV_DbgAssert( y.type() == x.type() && y.size() == x.size() );

    _mag.create(x.size(), CV_32FC1);
    GpuMat mag = _mag.getGpuMat();

    _angle.create(x.size(), CV_32FC1);
    GpuMat angle = _angle.getGpuMat();

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
}

namespace
{
    template <bool useMag>
    __global__ void polarToCartImpl(const GlobPtr<float> mag, const GlobPtr<float> angle, GlobPtr<float> xmat, GlobPtr<float> ymat, const float scale, const int rows, const int cols)
    {
        const int x = blockDim.x * blockIdx.x + threadIdx.x;
        const int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (x >= cols || y >= rows)
            return;

        const float mag_val = useMag ? mag(y, x) : 1.0f;
        const float angle_val = angle(y, x);

        float sin_a, cos_a;
        ::sincosf(scale * angle_val, &sin_a, &cos_a);

        xmat(y, x) = mag_val * cos_a;
        ymat(y, x) = mag_val * sin_a;
    }
}

void cv::cuda::polarToCart(InputArray _mag, InputArray _angle, OutputArray _x, OutputArray _y, bool angleInDegrees, Stream& _stream)
{
    GpuMat mag = _mag.getGpuMat();
    GpuMat angle = _angle.getGpuMat();

    CV_DbgAssert( angle.depth() == CV_32F );
    CV_DbgAssert( mag.empty() || (mag.type() == angle.type() && mag.size() == angle.size()) );

    _x.create(angle.size(), CV_32FC1);
    GpuMat x = _x.getGpuMat();

    _y.create(angle.size(), CV_32FC1);
    GpuMat y = _y.getGpuMat();

    GpuMat_<float> xc(x.reshape(1));
    GpuMat_<float> yc(y.reshape(1));
    GpuMat_<float> magc(mag.reshape(1));
    GpuMat_<float> anglec(angle.reshape(1));

    const dim3 block(32, 8);
    const dim3 grid(divUp(anglec.cols, block.x), divUp(anglec.rows, block.y));

    const float scale = angleInDegrees ? (CV_PI_F / 180.0f) : 1.0f;

    cudaStream_t stream = StreamAccessor::getStream(_stream);

    if (magc.empty())
        polarToCartImpl<false><<<grid, block, 0, stream>>>(shrinkPtr(magc), shrinkPtr(anglec), shrinkPtr(xc), shrinkPtr(yc), scale, anglec.rows, anglec.cols);
    else
        polarToCartImpl<true><<<grid, block, 0, stream>>>(shrinkPtr(magc), shrinkPtr(anglec), shrinkPtr(xc), shrinkPtr(yc), scale, anglec.rows, anglec.cols);

    CV_CUDEV_SAFE_CALL( cudaGetLastError() );

    if (stream == 0)
        CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
}

#endif
