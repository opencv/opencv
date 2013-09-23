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

#ifndef __OPENCV_CUDAFILTERS_HPP__
#define __OPENCV_CUDAFILTERS_HPP__

#ifndef __cplusplus
#  error cudafilters.hpp header must be compiled as C++
#endif

#include "opencv2/core/cuda.hpp"
#include "opencv2/imgproc.hpp"

namespace cv { namespace cuda {

class CV_EXPORTS Filter : public Algorithm
{
public:
    virtual void apply(InputArray src, OutputArray dst, Stream& stream = Stream::Null()) = 0;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Box Filter

//! creates a normalized 2D box filter
//! supports CV_8UC1, CV_8UC4 types
CV_EXPORTS Ptr<Filter> createBoxFilter(int srcType, int dstType, Size ksize, Point anchor = Point(-1,-1),
                                       int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0));

////////////////////////////////////////////////////////////////////////////////////////////////////
// Linear Filter

//! Creates a non-separable linear 2D filter
//! supports 1 and 4 channel CV_8U, CV_16U and CV_32F input
CV_EXPORTS Ptr<Filter> createLinearFilter(int srcType, int dstType, InputArray kernel, Point anchor = Point(-1,-1),
                                          int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0));

////////////////////////////////////////////////////////////////////////////////////////////////////
// Laplacian Filter

//! creates a Laplacian operator
//! supports only ksize = 1 and ksize = 3
CV_EXPORTS Ptr<Filter> createLaplacianFilter(int srcType, int dstType, int ksize = 1, double scale = 1,
                                             int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0));

////////////////////////////////////////////////////////////////////////////////////////////////////
// Separable Linear Filter

//! creates a separable linear filter
CV_EXPORTS Ptr<Filter> createSeparableLinearFilter(int srcType, int dstType, InputArray rowKernel, InputArray columnKernel,
                                                   Point anchor = Point(-1,-1), int rowBorderMode = BORDER_DEFAULT, int columnBorderMode = -1);

////////////////////////////////////////////////////////////////////////////////////////////////////
// Deriv Filter

//! creates a generalized Deriv operator
CV_EXPORTS Ptr<Filter> createDerivFilter(int srcType, int dstType, int dx, int dy,
                                         int ksize, bool normalize = false, double scale = 1,
                                         int rowBorderMode = BORDER_DEFAULT, int columnBorderMode = -1);

//! creates a Sobel operator
CV_EXPORTS Ptr<Filter> createSobelFilter(int srcType, int dstType, int dx, int dy, int ksize = 3,
                                         double scale = 1, int rowBorderMode = BORDER_DEFAULT, int columnBorderMode = -1);

//! creates a vertical or horizontal Scharr operator
CV_EXPORTS Ptr<Filter> createScharrFilter(int srcType, int dstType, int dx, int dy,
                                          double scale = 1, int rowBorderMode = BORDER_DEFAULT, int columnBorderMode = -1);

////////////////////////////////////////////////////////////////////////////////////////////////////
// Gaussian Filter

//! creates a Gaussian filter
CV_EXPORTS Ptr<Filter> createGaussianFilter(int srcType, int dstType, Size ksize,
                                            double sigma1, double sigma2 = 0,
                                            int rowBorderMode = BORDER_DEFAULT, int columnBorderMode = -1);

////////////////////////////////////////////////////////////////////////////////////////////////////
// Morphology Filter

//! creates a 2D morphological filter
//! supports CV_8UC1 and CV_8UC4 types
CV_EXPORTS Ptr<Filter> createMorphologyFilter(int op, int srcType, InputArray kernel, Point anchor = Point(-1, -1), int iterations = 1);

////////////////////////////////////////////////////////////////////////////////////////////////////
// Image Rank Filter

//! result pixel value is the maximum of pixel values under the rectangular mask region
CV_EXPORTS Ptr<Filter> createBoxMaxFilter(int srcType, Size ksize,
                                          Point anchor = Point(-1, -1),
                                          int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0));

//! result pixel value is the maximum of pixel values under the rectangular mask region
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

}} // namespace cv { namespace cuda {

#endif /* __OPENCV_CUDAFILTERS_HPP__ */
