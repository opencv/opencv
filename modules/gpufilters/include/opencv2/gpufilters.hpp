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

#include "opencv2/core/gpumat.hpp"
#include "opencv2/imgproc.hpp"

namespace cv { namespace gpu {

/*!
The Base Class for 1D or Row-wise Filters

This is the base class for linear or non-linear filters that process 1D data.
In particular, such filters are used for the "horizontal" filtering parts in separable filters.
*/
class CV_EXPORTS BaseRowFilter_GPU
{
public:
    BaseRowFilter_GPU(int ksize_, int anchor_) : ksize(ksize_), anchor(anchor_) {}
    virtual ~BaseRowFilter_GPU() {}
    virtual void operator()(const GpuMat& src, GpuMat& dst, Stream& stream = Stream::Null()) = 0;
    int ksize, anchor;
};

/*!
The Base Class for Column-wise Filters

This is the base class for linear or non-linear filters that process columns of 2D arrays.
Such filters are used for the "vertical" filtering parts in separable filters.
*/
class CV_EXPORTS BaseColumnFilter_GPU
{
public:
    BaseColumnFilter_GPU(int ksize_, int anchor_) : ksize(ksize_), anchor(anchor_) {}
    virtual ~BaseColumnFilter_GPU() {}
    virtual void operator()(const GpuMat& src, GpuMat& dst, Stream& stream = Stream::Null()) = 0;
    int ksize, anchor;
};

/*!
The Base Class for Non-Separable 2D Filters.

This is the base class for linear or non-linear 2D filters.
*/
class CV_EXPORTS BaseFilter_GPU
{
public:
    BaseFilter_GPU(const Size& ksize_, const Point& anchor_) : ksize(ksize_), anchor(anchor_) {}
    virtual ~BaseFilter_GPU() {}
    virtual void operator()(const GpuMat& src, GpuMat& dst, Stream& stream = Stream::Null()) = 0;
    Size ksize;
    Point anchor;
};

/*!
The Base Class for Filter Engine.

The class can be used to apply an arbitrary filtering operation to an image.
It contains all the necessary intermediate buffers.
*/
class CV_EXPORTS FilterEngine_GPU
{
public:
    virtual ~FilterEngine_GPU() {}

    virtual void apply(const GpuMat& src, GpuMat& dst, Rect roi = Rect(0,0,-1,-1), Stream& stream = Stream::Null()) = 0;
};

//! returns the non-separable filter engine with the specified filter
CV_EXPORTS Ptr<FilterEngine_GPU> createFilter2D_GPU(const Ptr<BaseFilter_GPU>& filter2D, int srcType, int dstType);

//! returns the separable filter engine with the specified filters
CV_EXPORTS Ptr<FilterEngine_GPU> createSeparableFilter_GPU(const Ptr<BaseRowFilter_GPU>& rowFilter,
    const Ptr<BaseColumnFilter_GPU>& columnFilter, int srcType, int bufType, int dstType);
CV_EXPORTS Ptr<FilterEngine_GPU> createSeparableFilter_GPU(const Ptr<BaseRowFilter_GPU>& rowFilter,
    const Ptr<BaseColumnFilter_GPU>& columnFilter, int srcType, int bufType, int dstType, GpuMat& buf);

//! returns horizontal 1D box filter
//! supports only CV_8UC1 source type and CV_32FC1 sum type
CV_EXPORTS Ptr<BaseRowFilter_GPU> getRowSumFilter_GPU(int srcType, int sumType, int ksize, int anchor = -1);

//! returns vertical 1D box filter
//! supports only CV_8UC1 sum type and CV_32FC1 dst type
CV_EXPORTS Ptr<BaseColumnFilter_GPU> getColumnSumFilter_GPU(int sumType, int dstType, int ksize, int anchor = -1);

//! returns 2D box filter
//! supports CV_8UC1 and CV_8UC4 source type, dst type must be the same as source type
CV_EXPORTS Ptr<BaseFilter_GPU> getBoxFilter_GPU(int srcType, int dstType, const Size& ksize, Point anchor = Point(-1, -1));

//! returns box filter engine
CV_EXPORTS Ptr<FilterEngine_GPU> createBoxFilter_GPU(int srcType, int dstType, const Size& ksize,
    const Point& anchor = Point(-1,-1));

//! returns 2D morphological filter
//! only MORPH_ERODE and MORPH_DILATE are supported
//! supports CV_8UC1 and CV_8UC4 types
//! kernel must have CV_8UC1 type, one rows and cols == ksize.width * ksize.height
CV_EXPORTS Ptr<BaseFilter_GPU> getMorphologyFilter_GPU(int op, int type, const Mat& kernel, const Size& ksize,
    Point anchor=Point(-1,-1));

//! returns morphological filter engine. Only MORPH_ERODE and MORPH_DILATE are supported.
CV_EXPORTS Ptr<FilterEngine_GPU> createMorphologyFilter_GPU(int op, int type, const Mat& kernel,
    const Point& anchor = Point(-1,-1), int iterations = 1);
CV_EXPORTS Ptr<FilterEngine_GPU> createMorphologyFilter_GPU(int op, int type, const Mat& kernel, GpuMat& buf,
    const Point& anchor = Point(-1,-1), int iterations = 1);

//! returns 2D filter with the specified kernel
//! supports CV_8U, CV_16U and CV_32F one and four channel image
CV_EXPORTS Ptr<BaseFilter_GPU> getLinearFilter_GPU(int srcType, int dstType, const Mat& kernel, Point anchor = Point(-1, -1), int borderType = BORDER_DEFAULT);

//! returns the non-separable linear filter engine
CV_EXPORTS Ptr<FilterEngine_GPU> createLinearFilter_GPU(int srcType, int dstType, const Mat& kernel,
    Point anchor = Point(-1,-1), int borderType = BORDER_DEFAULT);

//! returns the primitive row filter with the specified kernel.
//! supports only CV_8UC1, CV_8UC4, CV_16SC1, CV_16SC2, CV_32SC1, CV_32FC1 source type.
//! there are two version of algorithm: NPP and OpenCV.
//! NPP calls when srcType == CV_8UC1 or srcType == CV_8UC4 and bufType == srcType,
//! otherwise calls OpenCV version.
//! NPP supports only BORDER_CONSTANT border type.
//! OpenCV version supports only CV_32F as buffer depth and
//! BORDER_REFLECT101, BORDER_REPLICATE and BORDER_CONSTANT border types.
CV_EXPORTS Ptr<BaseRowFilter_GPU> getLinearRowFilter_GPU(int srcType, int bufType, const Mat& rowKernel,
    int anchor = -1, int borderType = BORDER_DEFAULT);

//! returns the primitive column filter with the specified kernel.
//! supports only CV_8UC1, CV_8UC4, CV_16SC1, CV_16SC2, CV_32SC1, CV_32FC1 dst type.
//! there are two version of algorithm: NPP and OpenCV.
//! NPP calls when dstType == CV_8UC1 or dstType == CV_8UC4 and bufType == dstType,
//! otherwise calls OpenCV version.
//! NPP supports only BORDER_CONSTANT border type.
//! OpenCV version supports only CV_32F as buffer depth and
//! BORDER_REFLECT101, BORDER_REPLICATE and BORDER_CONSTANT border types.
CV_EXPORTS Ptr<BaseColumnFilter_GPU> getLinearColumnFilter_GPU(int bufType, int dstType, const Mat& columnKernel,
    int anchor = -1, int borderType = BORDER_DEFAULT);

//! returns the separable linear filter engine
CV_EXPORTS Ptr<FilterEngine_GPU> createSeparableLinearFilter_GPU(int srcType, int dstType, const Mat& rowKernel,
    const Mat& columnKernel, const Point& anchor = Point(-1,-1), int rowBorderType = BORDER_DEFAULT,
    int columnBorderType = -1);
CV_EXPORTS Ptr<FilterEngine_GPU> createSeparableLinearFilter_GPU(int srcType, int dstType, const Mat& rowKernel,
    const Mat& columnKernel, GpuMat& buf, const Point& anchor = Point(-1,-1), int rowBorderType = BORDER_DEFAULT,
    int columnBorderType = -1);

//! returns filter engine for the generalized Sobel operator
CV_EXPORTS Ptr<FilterEngine_GPU> createDerivFilter_GPU(int srcType, int dstType, int dx, int dy, int ksize,
                                                       int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1);
CV_EXPORTS Ptr<FilterEngine_GPU> createDerivFilter_GPU(int srcType, int dstType, int dx, int dy, int ksize, GpuMat& buf,
                                                       int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1);

//! returns the Gaussian filter engine
CV_EXPORTS Ptr<FilterEngine_GPU> createGaussianFilter_GPU(int type, Size ksize, double sigma1, double sigma2 = 0,
                                                          int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1);
CV_EXPORTS Ptr<FilterEngine_GPU> createGaussianFilter_GPU(int type, Size ksize, GpuMat& buf, double sigma1, double sigma2 = 0,
                                                          int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1);

//! returns maximum filter
CV_EXPORTS Ptr<BaseFilter_GPU> getMaxFilter_GPU(int srcType, int dstType, const Size& ksize, Point anchor = Point(-1,-1));

//! returns minimum filter
CV_EXPORTS Ptr<BaseFilter_GPU> getMinFilter_GPU(int srcType, int dstType, const Size& ksize, Point anchor = Point(-1,-1));

//! smooths the image using the normalized box filter
//! supports CV_8UC1, CV_8UC4 types
CV_EXPORTS void boxFilter(const GpuMat& src, GpuMat& dst, int ddepth, Size ksize, Point anchor = Point(-1,-1), Stream& stream = Stream::Null());

//! a synonym for normalized box filter
static inline void blur(const GpuMat& src, GpuMat& dst, Size ksize, Point anchor = Point(-1,-1), Stream& stream = Stream::Null())
{
    boxFilter(src, dst, -1, ksize, anchor, stream);
}

//! erodes the image (applies the local minimum operator)
CV_EXPORTS void erode(const GpuMat& src, GpuMat& dst, const Mat& kernel, Point anchor = Point(-1, -1), int iterations = 1);
CV_EXPORTS void erode(const GpuMat& src, GpuMat& dst, const Mat& kernel, GpuMat& buf,
                      Point anchor = Point(-1, -1), int iterations = 1,
                      Stream& stream = Stream::Null());

//! dilates the image (applies the local maximum operator)
CV_EXPORTS void dilate(const GpuMat& src, GpuMat& dst, const Mat& kernel, Point anchor = Point(-1, -1), int iterations = 1);
CV_EXPORTS void dilate(const GpuMat& src, GpuMat& dst, const Mat& kernel, GpuMat& buf,
                       Point anchor = Point(-1, -1), int iterations = 1,
                       Stream& stream = Stream::Null());

//! applies an advanced morphological operation to the image
CV_EXPORTS void morphologyEx(const GpuMat& src, GpuMat& dst, int op, const Mat& kernel, Point anchor = Point(-1, -1), int iterations = 1);
CV_EXPORTS void morphologyEx(const GpuMat& src, GpuMat& dst, int op, const Mat& kernel, GpuMat& buf1, GpuMat& buf2,
                             Point anchor = Point(-1, -1), int iterations = 1, Stream& stream = Stream::Null());

//! applies non-separable 2D linear filter to the image
CV_EXPORTS void filter2D(const GpuMat& src, GpuMat& dst, int ddepth, const Mat& kernel, Point anchor=Point(-1,-1), int borderType = BORDER_DEFAULT, Stream& stream = Stream::Null());

//! applies separable 2D linear filter to the image
CV_EXPORTS void sepFilter2D(const GpuMat& src, GpuMat& dst, int ddepth, const Mat& kernelX, const Mat& kernelY,
                            Point anchor = Point(-1,-1), int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1);
CV_EXPORTS void sepFilter2D(const GpuMat& src, GpuMat& dst, int ddepth, const Mat& kernelX, const Mat& kernelY, GpuMat& buf,
                            Point anchor = Point(-1,-1), int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1,
                            Stream& stream = Stream::Null());

//! applies generalized Sobel operator to the image
CV_EXPORTS void Sobel(const GpuMat& src, GpuMat& dst, int ddepth, int dx, int dy, int ksize = 3, double scale = 1,
                      int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1);
CV_EXPORTS void Sobel(const GpuMat& src, GpuMat& dst, int ddepth, int dx, int dy, GpuMat& buf, int ksize = 3, double scale = 1,
                      int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1, Stream& stream = Stream::Null());

//! applies the vertical or horizontal Scharr operator to the image
CV_EXPORTS void Scharr(const GpuMat& src, GpuMat& dst, int ddepth, int dx, int dy, double scale = 1,
                       int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1);
CV_EXPORTS void Scharr(const GpuMat& src, GpuMat& dst, int ddepth, int dx, int dy, GpuMat& buf, double scale = 1,
                       int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1, Stream& stream = Stream::Null());

//! smooths the image using Gaussian filter.
CV_EXPORTS void GaussianBlur(const GpuMat& src, GpuMat& dst, Size ksize, double sigma1, double sigma2 = 0,
                             int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1);
CV_EXPORTS void GaussianBlur(const GpuMat& src, GpuMat& dst, Size ksize, GpuMat& buf, double sigma1, double sigma2 = 0,
                             int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1, Stream& stream = Stream::Null());

//! applies Laplacian operator to the image
//! supports only ksize = 1 and ksize = 3
CV_EXPORTS void Laplacian(const GpuMat& src, GpuMat& dst, int ddepth, int ksize = 1, double scale = 1, int borderType = BORDER_DEFAULT, Stream& stream = Stream::Null());

}} // namespace cv { namespace gpu {

#endif /* __OPENCV_GPUFILTERS_HPP__ */
