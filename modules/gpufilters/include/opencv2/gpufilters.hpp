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

#ifndef __OPENCV_GPUFILTERS_HPP__
#define __OPENCV_GPUFILTERS_HPP__

#ifndef __cplusplus
#  error gpufilters.hpp header must be compiled as C++
#endif

#include "opencv2/core/gpu.hpp"
#include "opencv2/imgproc.hpp"

#if defined __GNUC__
    #define __OPENCV_GPUFILTERS_DEPR_BEFORE__
    #define __OPENCV_GPUFILTERS_DEPR_AFTER__ __attribute__ ((deprecated))
#elif (defined WIN32 || defined _WIN32)
    #define __OPENCV_GPUFILTERS_DEPR_BEFORE__ __declspec(deprecated)
    #define __OPENCV_GPUFILTERS_DEPR_AFTER__
#else
    #define __OPENCV_GPUFILTERS_DEPR_BEFORE__
    #define __OPENCV_GPUFILTERS_DEPR_AFTER__
#endif

namespace cv { namespace gpu {

class CV_EXPORTS Filter : public Algorithm
{
public:
    virtual void apply(InputArray src, OutputArray dst, Stream& stream = Stream::Null()) = 0;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Box Filter

//! smooths the image using the normalized box filter
//! supports CV_8UC1, CV_8UC4 types
CV_EXPORTS Ptr<Filter> createBoxFilter(int srcType, int dstType, Size ksize, Point anchor = Point(-1,-1),
                                       int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0));

__OPENCV_GPUFILTERS_DEPR_BEFORE__ void boxFilter(InputArray src, OutputArray dst, int dstType,
                                                 Size ksize, Point anchor = Point(-1,-1),
                                                 Stream& stream = Stream::Null()) __OPENCV_GPUFILTERS_DEPR_AFTER__;

inline void boxFilter(InputArray src, OutputArray dst, int dstType, Size ksize, Point anchor, Stream& stream)
{
    Ptr<gpu::Filter> f = gpu::createBoxFilter(src.type(), dstType, ksize, anchor);
    f->apply(src, dst, stream);
}

__OPENCV_GPUFILTERS_DEPR_BEFORE__ void blur(InputArray src, OutputArray dst, Size ksize,
                                            Point anchor = Point(-1,-1),
                                            Stream& stream = Stream::Null()) __OPENCV_GPUFILTERS_DEPR_AFTER__;

inline void blur(InputArray src, OutputArray dst, Size ksize, Point anchor, Stream& stream)
{
    Ptr<gpu::Filter> f = gpu::createBoxFilter(src.type(), -1, ksize, anchor);
    f->apply(src, dst, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Linear Filter

//! non-separable linear 2D filter
CV_EXPORTS Ptr<Filter> createLinearFilter(int srcType, int dstType, InputArray kernel, Point anchor = Point(-1,-1),
                                            int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0));

__OPENCV_GPUFILTERS_DEPR_BEFORE__ void filter2D(InputArray src, OutputArray dst, int ddepth, InputArray kernel,
                                                Point anchor = Point(-1,-1), int borderType = BORDER_DEFAULT,
                                                Stream& stream = Stream::Null()) __OPENCV_GPUFILTERS_DEPR_AFTER__;

inline void filter2D(InputArray src, OutputArray dst, int ddepth, InputArray kernel, Point anchor, int borderType, Stream& stream)
{
    Ptr<gpu::Filter> f = gpu::createLinearFilter(src.type(), ddepth, kernel, anchor, borderType);
    f->apply(src, dst, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Laplacian Filter

//! applies Laplacian operator to the image
//! supports only ksize = 1 and ksize = 3
CV_EXPORTS Ptr<Filter> createLaplacianFilter(int srcType, int dstType, int ksize = 1, double scale = 1,
                                            int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0));

__OPENCV_GPUFILTERS_DEPR_BEFORE__ void Laplacian(InputArray src, OutputArray dst, int ddepth,
                                                 int ksize = 1, double scale = 1, int borderType = BORDER_DEFAULT,
                                                 Stream& stream = Stream::Null()) __OPENCV_GPUFILTERS_DEPR_AFTER__;

inline void Laplacian(InputArray src, OutputArray dst, int ddepth, int ksize, double scale, int borderType, Stream& stream)
{
    Ptr<gpu::Filter> f = gpu::createLaplacianFilter(src.type(), ddepth, ksize, scale, borderType);
    f->apply(src, dst, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Separable Linear Filter

//! separable linear 2D filter
CV_EXPORTS Ptr<Filter> createSeparableLinearFilter(int srcType, int dstType, InputArray rowKernel, InputArray columnKernel,
                                                   Point anchor = Point(-1,-1), int rowBorderMode = BORDER_DEFAULT, int columnBorderMode = -1);

__OPENCV_GPUFILTERS_DEPR_BEFORE__ void sepFilter2D(InputArray src, OutputArray dst, int ddepth, InputArray kernelX, InputArray kernelY,
                            Point anchor = Point(-1,-1), int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1,
                                                   Stream& stream = Stream::Null()) __OPENCV_GPUFILTERS_DEPR_AFTER__;

inline void sepFilter2D(InputArray src, OutputArray dst, int ddepth, InputArray kernelX, InputArray kernelY, Point anchor, int rowBorderType, int columnBorderType, Stream& stream)
{
    Ptr<gpu::Filter> f = gpu::createSeparableLinearFilter(src.type(), ddepth, kernelX, kernelY, anchor, rowBorderType, columnBorderType);
    f->apply(src, dst, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Deriv Filter

//! the generalized Deriv operator
CV_EXPORTS Ptr<Filter> createDerivFilter(int srcType, int dstType, int dx, int dy,
                                         int ksize, bool normalize = false, double scale = 1,
                                         int rowBorderMode = BORDER_DEFAULT, int columnBorderMode = -1);

//! the Sobel operator
CV_EXPORTS Ptr<Filter> createSobelFilter(int srcType, int dstType, int dx, int dy, int ksize = 3,
                                         double scale = 1, int rowBorderMode = BORDER_DEFAULT, int columnBorderMode = -1);

//! the vertical or horizontal Scharr operator
CV_EXPORTS Ptr<Filter> createScharrFilter(int srcType, int dstType, int dx, int dy,
                                          double scale = 1, int rowBorderMode = BORDER_DEFAULT, int columnBorderMode = -1);

__OPENCV_GPUFILTERS_DEPR_BEFORE__ void Sobel(InputArray src, OutputArray dst, int ddepth, int dx, int dy, int ksize = 3, double scale = 1,
                                             int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1,
                                             Stream& stream = Stream::Null()) __OPENCV_GPUFILTERS_DEPR_AFTER__;

inline void Sobel(InputArray src, OutputArray dst, int ddepth, int dx, int dy, int ksize, double scale, int rowBorderType, int columnBorderType, Stream& stream)
{
    Ptr<gpu::Filter> f = gpu::createSobelFilter(src.type(), ddepth, dx, dy, ksize, scale, rowBorderType, columnBorderType);
    f->apply(src, dst, stream);
}

__OPENCV_GPUFILTERS_DEPR_BEFORE__ void Scharr(InputArray src, OutputArray dst, int ddepth, int dx, int dy, double scale = 1,
                                              int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1,
                                              Stream& stream = Stream::Null()) __OPENCV_GPUFILTERS_DEPR_AFTER__;

inline void Scharr(InputArray src, OutputArray dst, int ddepth, int dx, int dy, double scale, int rowBorderType, int columnBorderType, Stream& stream)
{
    Ptr<gpu::Filter> f = gpu::createScharrFilter(src.type(), ddepth, dx, dy, scale, rowBorderType, columnBorderType);
    f->apply(src, dst, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Gaussian Filter

//! smooths the image using Gaussian filter
CV_EXPORTS Ptr<Filter> createGaussianFilter(int srcType, int dstType, Size ksize,
                                            double sigma1, double sigma2 = 0,
                                            int rowBorderMode = BORDER_DEFAULT, int columnBorderMode = -1);

__OPENCV_GPUFILTERS_DEPR_BEFORE__ void GaussianBlur(InputArray src, OutputArray dst, Size ksize,
                                                    double sigma1, double sigma2 = 0,
                                                    int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1,
                                                    Stream& stream = Stream::Null()) __OPENCV_GPUFILTERS_DEPR_AFTER__;

inline void GaussianBlur(InputArray src, OutputArray dst, Size ksize, double sigma1, double sigma2, int rowBorderType, int columnBorderType, Stream& stream)
{
    Ptr<gpu::Filter> f = gpu::createGaussianFilter(src.type(), -1, ksize, sigma1, sigma2, rowBorderType, columnBorderType);
    f->apply(src, dst, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Morphology Filter

//! returns 2D morphological filter
//! supports CV_8UC1 and CV_8UC4 types
CV_EXPORTS Ptr<Filter> createMorphologyFilter(int op, int srcType, InputArray kernel, Point anchor = Point(-1, -1), int iterations = 1);

__OPENCV_GPUFILTERS_DEPR_BEFORE__ void erode(InputArray src, OutputArray dst, InputArray kernel,
                                             Point anchor = Point(-1, -1), int iterations = 1,
                                             Stream& stream = Stream::Null()) __OPENCV_GPUFILTERS_DEPR_AFTER__;

inline void erode(InputArray src, OutputArray dst, InputArray kernel, Point anchor, int iterations, Stream& stream)
{
    Ptr<gpu::Filter> f = gpu::createMorphologyFilter(MORPH_ERODE, src.type(), kernel, anchor, iterations);
    f->apply(src, dst, stream);
}

__OPENCV_GPUFILTERS_DEPR_BEFORE__ void dilate(InputArray src, OutputArray dst, InputArray kernel,
                                              Point anchor = Point(-1, -1), int iterations = 1,
                                              Stream& stream = Stream::Null()) __OPENCV_GPUFILTERS_DEPR_AFTER__;

inline void dilate(InputArray src, OutputArray dst, InputArray kernel, Point anchor, int iterations, Stream& stream)
{
    Ptr<gpu::Filter> f = gpu::createMorphologyFilter(MORPH_DILATE, src.type(), kernel, anchor, iterations);
    f->apply(src, dst, stream);
}

__OPENCV_GPUFILTERS_DEPR_BEFORE__ void morphologyEx(InputArray src, OutputArray dst, int op,
                                                    InputArray kernel, Point anchor = Point(-1, -1), int iterations = 1,
                                                    Stream& stream = Stream::Null()) __OPENCV_GPUFILTERS_DEPR_AFTER__;

inline void morphologyEx(InputArray src, OutputArray dst, int op, InputArray kernel, Point anchor, int iterations, Stream& stream)
{
    Ptr<gpu::Filter> f = gpu::createMorphologyFilter(op, src.type(), kernel, anchor, iterations);
    f->apply(src, dst, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Image Rank Filter

//! Result pixel value is the maximum of pixel values under the rectangular mask region
CV_EXPORTS Ptr<Filter> createBoxMaxFilter(int srcType, Size ksize,
                                          Point anchor = Point(-1, -1),
                                          int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0));

//! Result pixel value is the maximum of pixel values under the rectangular mask region
CV_EXPORTS Ptr<Filter> createBoxMinFilter(int srcType, Size ksize,
                                          Point anchor = Point(-1, -1),
                                          int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0));

////////////////////////////////////////////////////////////////////////////////////////////////////
// 1D Sum Filter

//! creates a horizontal 1D box filter
//! supports only CV_8UC1 source type and CV_32FC1 sum type
CV_EXPORTS Ptr<Filter> createRowSumFilter(int srcType, int dstType, int ksize, int anchor = -1, int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0));

//! creates a vertical 1D box filter
//! supports only CV_8UC1 sum type and CV_32FC1 dst type
CV_EXPORTS Ptr<Filter> createColumnSumFilter(int srcType, int dstType, int ksize, int anchor = -1, int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0));

}} // namespace cv { namespace gpu {

#undef __OPENCV_GPUFILTERS_DEPR_BEFORE__
#undef __OPENCV_GPUFILTERS_DEPR_AFTER__

#endif /* __OPENCV_GPUFILTERS_HPP__ */
