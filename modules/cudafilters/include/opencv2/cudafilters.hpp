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

/**
  @addtogroup cuda
  @{
    @defgroup cudafilters Image Filtering

Functions and classes described in this section are used to perform various linear or non-linear
filtering operations on 2D images.

@note
   -   An example containing all basic morphology operators like erode and dilate can be found at
        opencv_source_code/samples/gpu/morphology.cpp

  @}
 */

namespace cv { namespace cuda {

//! @addtogroup cudafilters
//! @{

/** @brief Common interface for all CUDA filters :
 */
class CV_EXPORTS Filter : public Algorithm
{
public:
    /** @brief Applies the specified filter to the image.

    @param src Input image.
    @param dst Output image.
    @param stream Stream for the asynchronous version.
     */
    virtual void apply(InputArray src, OutputArray dst, Stream& stream = Stream::Null()) = 0;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Box Filter

/** @brief Creates a normalized 2D box filter.

@param srcType Input image type. Only CV_8UC1, CV_8UC4 and CV_32FC1 are supported for now.
@param dstType Output image type. Only the same type as src is supported for now.
@param ksize Kernel size.
@param anchor Anchor point. The default value Point(-1, -1) means that the anchor is at the kernel
center.
@param borderMode Pixel extrapolation method. For details, see borderInterpolate .
@param borderVal Default border value.

@sa boxFilter
 */
CV_EXPORTS Ptr<Filter> createBoxFilter(int srcType, int dstType, Size ksize, Point anchor = Point(-1,-1),
                                       int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0));

////////////////////////////////////////////////////////////////////////////////////////////////////
// Linear Filter

/** @brief Creates a non-separable linear 2D filter.

@param srcType Input image type. Supports CV_8U , CV_16U and CV_32F one and four channel image.
@param dstType Output image type. Only the same type as src is supported for now.
@param kernel 2D array of filter coefficients.
@param anchor Anchor point. The default value Point(-1, -1) means that the anchor is at the kernel
center.
@param borderMode Pixel extrapolation method. For details, see borderInterpolate .
@param borderVal Default border value.

@sa filter2D
 */
CV_EXPORTS Ptr<Filter> createLinearFilter(int srcType, int dstType, InputArray kernel, Point anchor = Point(-1,-1),
                                          int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0));

////////////////////////////////////////////////////////////////////////////////////////////////////
// Laplacian Filter

/** @brief Creates a Laplacian operator.

@param srcType Input image type. Supports CV_8U , CV_16U and CV_32F one and four channel image.
@param dstType Output image type. Only the same type as src is supported for now.
@param ksize Aperture size used to compute the second-derivative filters (see getDerivKernels). It
must be positive and odd. Only ksize = 1 and ksize = 3 are supported.
@param scale Optional scale factor for the computed Laplacian values. By default, no scaling is
applied (see getDerivKernels ).
@param borderMode Pixel extrapolation method. For details, see borderInterpolate .
@param borderVal Default border value.

@sa Laplacian
 */
CV_EXPORTS Ptr<Filter> createLaplacianFilter(int srcType, int dstType, int ksize = 1, double scale = 1,
                                             int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0));

////////////////////////////////////////////////////////////////////////////////////////////////////
// Separable Linear Filter

/** @brief Creates a separable linear filter.

@param srcType Source array type.
@param dstType Destination array type.
@param rowKernel Horizontal filter coefficients. Support kernels with size \<= 32 .
@param columnKernel Vertical filter coefficients. Support kernels with size \<= 32 .
@param anchor Anchor position within the kernel. Negative values mean that anchor is positioned at
the aperture center.
@param rowBorderMode Pixel extrapolation method in the vertical direction For details, see
borderInterpolate.
@param columnBorderMode Pixel extrapolation method in the horizontal direction.

@sa sepFilter2D
 */
CV_EXPORTS Ptr<Filter> createSeparableLinearFilter(int srcType, int dstType, InputArray rowKernel, InputArray columnKernel,
                                                   Point anchor = Point(-1,-1), int rowBorderMode = BORDER_DEFAULT, int columnBorderMode = -1);

////////////////////////////////////////////////////////////////////////////////////////////////////
// Deriv Filter

/** @brief Creates a generalized Deriv operator.

@param srcType Source image type.
@param dstType Destination array type.
@param dx Derivative order in respect of x.
@param dy Derivative order in respect of y.
@param ksize Aperture size. See getDerivKernels for details.
@param normalize Flag indicating whether to normalize (scale down) the filter coefficients or not.
See getDerivKernels for details.
@param scale Optional scale factor for the computed derivative values. By default, no scaling is
applied. For details, see getDerivKernels .
@param rowBorderMode Pixel extrapolation method in the vertical direction. For details, see
borderInterpolate.
@param columnBorderMode Pixel extrapolation method in the horizontal direction.
 */
CV_EXPORTS Ptr<Filter> createDerivFilter(int srcType, int dstType, int dx, int dy,
                                         int ksize, bool normalize = false, double scale = 1,
                                         int rowBorderMode = BORDER_DEFAULT, int columnBorderMode = -1);

/** @brief Creates a Sobel operator.

@param srcType Source image type.
@param dstType Destination array type.
@param dx Derivative order in respect of x.
@param dy Derivative order in respect of y.
@param ksize Size of the extended Sobel kernel. Possible values are 1, 3, 5 or 7.
@param scale Optional scale factor for the computed derivative values. By default, no scaling is
applied. For details, see getDerivKernels .
@param rowBorderMode Pixel extrapolation method in the vertical direction. For details, see
borderInterpolate.
@param columnBorderMode Pixel extrapolation method in the horizontal direction.

@sa Sobel
 */
CV_EXPORTS Ptr<Filter> createSobelFilter(int srcType, int dstType, int dx, int dy, int ksize = 3,
                                         double scale = 1, int rowBorderMode = BORDER_DEFAULT, int columnBorderMode = -1);

/** @brief Creates a vertical or horizontal Scharr operator.

@param srcType Source image type.
@param dstType Destination array type.
@param dx Order of the derivative x.
@param dy Order of the derivative y.
@param scale Optional scale factor for the computed derivative values. By default, no scaling is
applied. See getDerivKernels for details.
@param rowBorderMode Pixel extrapolation method in the vertical direction. For details, see
borderInterpolate.
@param columnBorderMode Pixel extrapolation method in the horizontal direction.

@sa Scharr
 */
CV_EXPORTS Ptr<Filter> createScharrFilter(int srcType, int dstType, int dx, int dy,
                                          double scale = 1, int rowBorderMode = BORDER_DEFAULT, int columnBorderMode = -1);

////////////////////////////////////////////////////////////////////////////////////////////////////
// Gaussian Filter

/** @brief Creates a Gaussian filter.

@param srcType Source image type.
@param dstType Destination array type.
@param ksize Aperture size. See getGaussianKernel for details.
@param sigma1 Gaussian sigma in the horizontal direction. See getGaussianKernel for details.
@param sigma2 Gaussian sigma in the vertical direction. If 0, then
\f$\texttt{sigma2}\leftarrow\texttt{sigma1}\f$ .
@param rowBorderMode Pixel extrapolation method in the vertical direction. For details, see
borderInterpolate.
@param columnBorderMode Pixel extrapolation method in the horizontal direction.

@sa GaussianBlur
 */
CV_EXPORTS Ptr<Filter> createGaussianFilter(int srcType, int dstType, Size ksize,
                                            double sigma1, double sigma2 = 0,
                                            int rowBorderMode = BORDER_DEFAULT, int columnBorderMode = -1);

////////////////////////////////////////////////////////////////////////////////////////////////////
// Morphology Filter

/** @brief Creates a 2D morphological filter.

@param op Type of morphological operation. The following types are possible:
-   **MORPH_ERODE** erode
-   **MORPH_DILATE** dilate
-   **MORPH_OPEN** opening
-   **MORPH_CLOSE** closing
-   **MORPH_GRADIENT** morphological gradient
-   **MORPH_TOPHAT** "top hat"
-   **MORPH_BLACKHAT** "black hat"
@param srcType Input/output image type. Only CV_8UC1 and CV_8UC4 are supported.
@param kernel 2D 8-bit structuring element for the morphological operation.
@param anchor Anchor position within the structuring element. Negative values mean that the anchor
is at the center.
@param iterations Number of times erosion and dilation to be applied.

@sa morphologyEx
 */
CV_EXPORTS Ptr<Filter> createMorphologyFilter(int op, int srcType, InputArray kernel, Point anchor = Point(-1, -1), int iterations = 1);

////////////////////////////////////////////////////////////////////////////////////////////////////
// Image Rank Filter

/** @brief Creates the maximum filter.

@param srcType Input/output image type. Only CV_8UC1 and CV_8UC4 are supported.
@param ksize Kernel size.
@param anchor Anchor point. The default value (-1) means that the anchor is at the kernel center.
@param borderMode Pixel extrapolation method. For details, see borderInterpolate .
@param borderVal Default border value.
 */
CV_EXPORTS Ptr<Filter> createBoxMaxFilter(int srcType, Size ksize,
                                          Point anchor = Point(-1, -1),
                                          int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0));

/** @brief Creates the minimum filter.

@param srcType Input/output image type. Only CV_8UC1 and CV_8UC4 are supported.
@param ksize Kernel size.
@param anchor Anchor point. The default value (-1) means that the anchor is at the kernel center.
@param borderMode Pixel extrapolation method. For details, see borderInterpolate .
@param borderVal Default border value.
 */
CV_EXPORTS Ptr<Filter> createBoxMinFilter(int srcType, Size ksize,
                                          Point anchor = Point(-1, -1),
                                          int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0));

////////////////////////////////////////////////////////////////////////////////////////////////////
// 1D Sum Filter

/** @brief Creates a horizontal 1D box filter.

@param srcType Input image type. Only CV_8UC1 type is supported for now.
@param dstType Output image type. Only CV_32FC1 type is supported for now.
@param ksize Kernel size.
@param anchor Anchor point. The default value (-1) means that the anchor is at the kernel center.
@param borderMode Pixel extrapolation method. For details, see borderInterpolate .
@param borderVal Default border value.
 */
CV_EXPORTS Ptr<Filter> createRowSumFilter(int srcType, int dstType, int ksize, int anchor = -1, int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0));

/** @brief Creates a vertical 1D box filter.

@param srcType Input image type. Only CV_8UC1 type is supported for now.
@param dstType Output image type. Only CV_32FC1 type is supported for now.
@param ksize Kernel size.
@param anchor Anchor point. The default value (-1) means that the anchor is at the kernel center.
@param borderMode Pixel extrapolation method. For details, see borderInterpolate .
@param borderVal Default border value.
 */
CV_EXPORTS Ptr<Filter> createColumnSumFilter(int srcType, int dstType, int ksize, int anchor = -1, int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0));

//! @}

///////////////////////////// Median Filtering //////////////////////////////

/** @brief Performs median filtering for each point of the source image.

@param srcType type of of source image. Only CV_8UC1 images are supported for now.
@param windowSize Size of the kernerl used for the filtering. Uses a (windowSize x windowSize) filter.
@param partition Specifies the parallel granularity of the workload. This parameter should be used GPU experts when optimizing performance.

Outputs an image that has been filtered using median-filtering formulation.
 */
CV_EXPORTS Ptr<Filter> createMedianFilter(int srcType, int windowSize, int partition=128);

}} // namespace cv { namespace cuda {

#endif /* __OPENCV_CUDAFILTERS_HPP__ */
