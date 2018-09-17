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

#ifndef OPENCV_CUDAFILTERS_HPP
#define OPENCV_CUDAFILTERS_HPP

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
class CV_EXPORTS_W Filter : public Algorithm
{
public:
    /** @brief Applies the specified filter to the image.

    @param src Input image.
    @param dst Output image.
    @param stream Stream for the asynchronous version.
     */
    CV_WRAP virtual void apply(InputArray src, OutputArray dst, Stream& stream = Stream::Null()) = 0;
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
CV_EXPORTS_W Ptr<Filter> createBoxFilter(ElemType srcType, ElemType dstType, Size ksize, Point anchor = Point(-1, -1),
                                       int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0));
#ifdef CV_TYPE_COMPATIBLE_API
#  ifndef OPENCV_DISABLE_DEPRECATED_WARNING_INT_ELEMTYPE_OVERLOAD
CV_DEPRECATED_MSG(CV_DEPRECATED_PARAM(int, dstType, ElemType, dstType) ". Similarly, " CV_DEPRECATED_PARAM(int, srcType, ElemType, srcType))
#  endif
static inline Ptr<Filter> createBoxFilter(int srcType, int dstType, Size ksize, Point anchor = Point(-1, -1),
                                       int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0))
{
    return createBoxFilter(static_cast<ElemType>(srcType), static_cast<ElemType>(dstType), ksize, anchor, borderMode, borderVal);
}
#  ifdef OPENCV_ENABLE_DEPRECATED_WARNING_ELEMDEPTH_ELEMTYPE_OVERLOAD
CV_DEPRECATED_MSG(CV_DEPRECATED_PARAM(ElemDepth, dstType, ElemType, dstType) ". Similarly, " CV_DEPRECATED_PARAM(ElemDepth, srcType, ElemType, srcType))
#  endif
static inline Ptr<Filter> createBoxFilter(ElemDepth srcType, ElemDepth dstType, Size ksize, Point anchor = Point(-1, -1),
                                       int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0))
{
    return createBoxFilter(CV_MAKETYPE(srcType, 1), CV_MAKETYPE(dstType, 1), ksize, anchor, borderMode, borderVal);
}
#endif // CV_TYPE_COMPATIBLE_API

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
CV_EXPORTS_W Ptr<Filter> createLinearFilter(ElemType srcType, ElemType dstType, InputArray kernel, Point anchor = Point(-1, -1),
                                          int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0));
#ifdef CV_TYPE_COMPATIBLE_API
#  ifndef OPENCV_DISABLE_DEPRECATED_WARNING_INT_ELEMTYPE_OVERLOAD
CV_DEPRECATED_MSG(CV_DEPRECATED_PARAM(int, dstType, ElemType, dstType) ". Similarly, " CV_DEPRECATED_PARAM(int, srcType, ElemType, srcType))
#  endif
static inline Ptr<Filter> createLinearFilter(int srcType, int dstType, InputArray kernel, Point anchor = Point(-1, -1),
                                          int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0))
{
    return createLinearFilter(static_cast<ElemType>(srcType), static_cast<ElemType>(dstType), kernel, anchor, borderMode, borderVal);
}
#  ifdef OPENCV_ENABLE_DEPRECATED_WARNING_ELEMDEPTH_ELEMTYPE_OVERLOAD
CV_DEPRECATED_MSG(CV_DEPRECATED_PARAM(ElemDepth, dstType, ElemType, dstType) ". Similarly, " CV_DEPRECATED_PARAM(ElemDepth, srcType, ElemType, srcType))
#  endif
static inline Ptr<Filter> createLinearFilter(ElemDepth srcType, ElemDepth dstType, InputArray kernel, Point anchor = Point(-1, -1),
                                          int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0))
{
    return createLinearFilter(CV_MAKETYPE(srcType, 1), CV_MAKETYPE(dstType, 1), kernel, anchor, borderMode, borderVal);
}
#endif // CV_TYPE_COMPATIBLE_API

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
CV_EXPORTS_W Ptr<Filter> createLaplacianFilter(ElemType srcType, ElemType dstType, int ksize = 1, double scale = 1,
                                             int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0));
#ifdef CV_TYPE_COMPATIBLE_API
#  ifndef OPENCV_DISABLE_DEPRECATED_WARNING_INT_ELEMTYPE_OVERLOAD
CV_DEPRECATED_MSG(CV_DEPRECATED_PARAM(int, dstType, ElemType, dstType) ". Similarly, " CV_DEPRECATED_PARAM(int, srcType, ElemType, srcType))
#  endif
static inline Ptr<Filter> createLaplacianFilter(int srcType, int dstType, int ksize = 1, double scale = 1,
                                             int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0))
{
    return createLaplacianFilter(static_cast<ElemType>(srcType), static_cast<ElemType>(dstType), ksize, scale, borderMode, borderVal);
}
#  ifdef OPENCV_ENABLE_DEPRECATED_WARNING_ELEMDEPTH_ELEMTYPE_OVERLOAD
CV_DEPRECATED_MSG(CV_DEPRECATED_PARAM(ElemDepth, dstType, ElemType, dstType) ". Similarly, " CV_DEPRECATED_PARAM(ElemDepth, srcType, ElemType, srcType))
#  endif
static inline Ptr<Filter> createLaplacianFilter(ElemDepth srcType, ElemDepth dstType, int ksize = 1, double scale = 1,
                                             int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0))
{
    return createLaplacianFilter(CV_MAKETYPE(srcType, 1), CV_MAKETYPE(dstType, 1), ksize, scale, borderMode, borderVal);
}
#endif // CV_TYPE_COMPATIBLE_API

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
CV_EXPORTS_W Ptr<Filter> createSeparableLinearFilter(ElemType srcType, ElemType dstType, InputArray rowKernel, InputArray columnKernel,
                                                   Point anchor = Point(-1,-1), int rowBorderMode = BORDER_DEFAULT, int columnBorderMode = -1);
#ifdef CV_TYPE_COMPATIBLE_API
#  ifndef OPENCV_DISABLE_DEPRECATED_WARNING_INT_ELEMTYPE_OVERLOAD
CV_DEPRECATED_MSG(CV_DEPRECATED_PARAM(int, dstType, ElemType, dstType) ". Similarly, " CV_DEPRECATED_PARAM(int, srcType, ElemType, srcType))
#  endif
static inline Ptr<Filter> createSeparableLinearFilter(int srcType, int dstType, InputArray rowKernel, InputArray columnKernel,
                                                   Point anchor = Point(-1,-1), int rowBorderMode = BORDER_DEFAULT, int columnBorderMode = -1)
{
    return createSeparableLinearFilter(static_cast<ElemType>(srcType), static_cast<ElemType>(dstType), rowKernel, columnKernel, anchor, rowBorderMode, columnBorderMode);
}
#  ifdef OPENCV_ENABLE_DEPRECATED_WARNING_ELEMDEPTH_ELEMTYPE_OVERLOAD
CV_DEPRECATED_MSG(CV_DEPRECATED_PARAM(ElemDepth, dstType, ElemType, dstType) ". Similarly, " CV_DEPRECATED_PARAM(ElemDepth, srcType, ElemType, srcType))
#  endif
static inline Ptr<Filter> createSeparableLinearFilter(ElemDepth srcType, ElemDepth dstType, InputArray rowKernel, InputArray columnKernel,
                                                   Point anchor = Point(-1,-1), int rowBorderMode = BORDER_DEFAULT, int columnBorderMode = -1)
{
    return createSeparableLinearFilter(CV_MAKETYPE(srcType, 1), CV_MAKETYPE(dstType, 1), rowKernel, columnKernel, anchor, rowBorderMode, columnBorderMode);
}
#endif // CV_TYPE_COMPATIBLE_API

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
CV_EXPORTS_W Ptr<Filter> createDerivFilter(ElemType srcType, ElemType dstType, int dx, int dy,
                                         int ksize, bool normalize = false, double scale = 1,
                                         int rowBorderMode = BORDER_DEFAULT, int columnBorderMode = -1);
#ifdef CV_TYPE_COMPATIBLE_API
#  ifndef OPENCV_DISABLE_DEPRECATED_WARNING_INT_ELEMTYPE_OVERLOAD
CV_DEPRECATED_MSG(CV_DEPRECATED_PARAM(int, srcType, ElemType, srcType) ". Similarly, " CV_DEPRECATED_PARAM(int, dstType, ElemType, dstType))
#  endif
static inline Ptr<Filter> createDerivFilter(int srcType, int dstType, int dx, int dy,
                                         int ksize, bool normalize = false, double scale = 1,
                                         int rowBorderMode = BORDER_DEFAULT, int columnBorderMode = -1)
{
    return createDerivFilter(static_cast<ElemType>(srcType), static_cast<ElemType>(dstType), dx, dy, ksize, normalize, scale, rowBorderMode, columnBorderMode);
}
#  ifdef OPENCV_ENABLE_DEPRECATED_WARNING_ELEMDEPTH_ELEMTYPE_OVERLOAD
CV_DEPRECATED_MSG(CV_DEPRECATED_PARAM(ElemDepth, srcType, ElemType, srcType) ". Similarly, " CV_DEPRECATED_PARAM(ElemDepth, dstType, ElemType, dstType))
#  endif
static inline Ptr<Filter> createDerivFilter(ElemDepth srcType, ElemDepth dstType, int dx, int dy,
                                         int ksize, bool normalize = false, double scale = 1,
                                         int rowBorderMode = BORDER_DEFAULT, int columnBorderMode = -1)
{
    return createDerivFilter(CV_MAKETYPE(srcType, 1), CV_MAKETYPE(dstType, 1), dx, dy, ksize, normalize, scale, rowBorderMode, columnBorderMode);
}
#endif // CV_TYPE_COMPATIBLE_API

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
CV_EXPORTS_W Ptr<Filter> createSobelFilter(ElemType srcType, ElemType dstType, int dx, int dy, int ksize = 3,
                                         double scale = 1, int rowBorderMode = BORDER_DEFAULT, int columnBorderMode = -1);
#ifdef CV_TYPE_COMPATIBLE_API
#  ifndef OPENCV_DISABLE_DEPRECATED_WARNING_INT_ELEMTYPE_OVERLOAD
CV_DEPRECATED_MSG(CV_DEPRECATED_PARAM(int, dstType, ElemType, dstType) ". Similarly, " CV_DEPRECATED_PARAM(int, srcType, ElemType, srcType))
#  endif
static inline Ptr<Filter> createSobelFilter(int srcType, int dstType, int dx, int dy, int ksize = 3,
                                         double scale = 1, int rowBorderMode = BORDER_DEFAULT, int columnBorderMode = -1)
{
    return createSobelFilter(static_cast<ElemType>(srcType), static_cast<ElemType>(dstType), dx, dy, ksize, scale, rowBorderMode, columnBorderMode);
}
#  ifdef OPENCV_ENABLE_DEPRECATED_WARNING_ELEMDEPTH_ELEMTYPE_OVERLOAD
CV_DEPRECATED_MSG(CV_DEPRECATED_PARAM(ElemDepth, dstType, ElemType, dstType) ". Similarly, " CV_DEPRECATED_PARAM(ElemDepth, srcType, ElemType, srcType))
#  endif
static inline Ptr<Filter> createSobelFilter(ElemDepth srcType, ElemDepth dstType, int dx, int dy, int ksize = 3,
                                         double scale = 1, int rowBorderMode = BORDER_DEFAULT, int columnBorderMode = -1)
{
    return createSobelFilter(CV_MAKETYPE(srcType, 1), CV_MAKETYPE(dstType, 1), dx, dy, ksize, scale, rowBorderMode, columnBorderMode);
}
#endif // CV_TYPE_COMPATIBLE_API

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
CV_EXPORTS_W Ptr<Filter> createScharrFilter(ElemType srcType, ElemType dstType, int dx, int dy,
                                          double scale = 1, int rowBorderMode = BORDER_DEFAULT, int columnBorderMode = -1);
#ifdef CV_TYPE_COMPATIBLE_API
#  ifndef OPENCV_DISABLE_DEPRECATED_WARNING_INT_ELEMTYPE_OVERLOAD
CV_DEPRECATED_MSG(CV_DEPRECATED_PARAM(int, dstType, ElemType, dstType) ". Similarly, " CV_DEPRECATED_PARAM(int, srcType, ElemType, srcType))
#  endif
static inline Ptr<Filter> createScharrFilter(int srcType, int dstType, int dx, int dy,
                                          double scale = 1, int rowBorderMode = BORDER_DEFAULT, int columnBorderMode = -1)
{
    return createScharrFilter(static_cast<ElemType>(srcType), static_cast<ElemType>(dstType), dx, dy, scale, rowBorderMode, columnBorderMode);
}
#  ifdef OPENCV_ENABLE_DEPRECATED_WARNING_ELEMDEPTH_ELEMTYPE_OVERLOAD
CV_DEPRECATED_MSG(CV_DEPRECATED_PARAM(ElemDepth, dstType, ElemType, dstType) ". Similarly, " CV_DEPRECATED_PARAM(ElemDepth, srcType, ElemType, srcType))
#  endif
static inline Ptr<Filter> createScharrFilter(ElemDepth srcType, ElemDepth dstType, int dx, int dy,
                                          double scale = 1, int rowBorderMode = BORDER_DEFAULT, int columnBorderMode = -1)
{
    return createScharrFilter(CV_MAKETYPE(srcType, 1), CV_MAKETYPE(dstType, 1), dx, dy, scale, rowBorderMode, columnBorderMode);
}
#endif // CV_TYPE_COMPATIBLE_API

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
CV_EXPORTS_W Ptr<Filter> createGaussianFilter(ElemType srcType, ElemType dstType, Size ksize,
                                            double sigma1, double sigma2 = 0,
                                            int rowBorderMode = BORDER_DEFAULT, int columnBorderMode = -1);
#ifdef CV_TYPE_COMPATIBLE_API
#  ifndef OPENCV_DISABLE_DEPRECATED_WARNING_INT_ELEMTYPE_OVERLOAD
CV_DEPRECATED_MSG(CV_DEPRECATED_PARAM(int, srcType, ElemType, srcType) ". Similarly, " CV_DEPRECATED_PARAM(int, dstType, ElemType, dstType))
#  endif
static inline Ptr<Filter> createGaussianFilter(int srcType, int dstType, Size ksize,
                                            double sigma1, double sigma2 = 0,
                                            int rowBorderMode = BORDER_DEFAULT, int columnBorderMode = -1)
{
    return createGaussianFilter(static_cast<ElemType>(srcType), static_cast<ElemType>(dstType), ksize, sigma1, sigma2, rowBorderMode, columnBorderMode);
}
#  ifdef OPENCV_ENABLE_DEPRECATED_WARNING_ELEMDEPTH_ELEMTYPE_OVERLOAD
CV_DEPRECATED_MSG(CV_DEPRECATED_PARAM(ElemDepth, srcType, ElemType, srcType) ". Similarly, " CV_DEPRECATED_PARAM(ElemDepth, dstType, ElemType, dstType))
#  endif
static inline Ptr<Filter> createGaussianFilter(ElemDepth srcType, ElemDepth dstType, Size ksize,
                                            double sigma1, double sigma2 = 0,
                                            int rowBorderMode = BORDER_DEFAULT, int columnBorderMode = -1)
{
    return createGaussianFilter(CV_MAKETYPE(srcType, 1), CV_MAKETYPE(dstType, 1), ksize, sigma1, sigma2, rowBorderMode, columnBorderMode);
}
#endif // CV_TYPE_COMPATIBLE_API

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
@param srcType Input/output image type. Only CV_8UC1, CV_8UC4, CV_32FC1 and CV_32FC4 are supported.
@param kernel 2D 8-bit structuring element for the morphological operation.
@param anchor Anchor position within the structuring element. Negative values mean that the anchor
is at the center.
@param iterations Number of times erosion and dilation to be applied.

@sa morphologyEx
 */
CV_EXPORTS_W Ptr<Filter> createMorphologyFilter(int op, ElemType srcType, InputArray kernel, Point anchor = Point(-1, -1), int iterations = 1);
#ifdef CV_TYPE_COMPATIBLE_API
CV_DEPRECATED_INT_TO_ELEMTYPE_ATTR(srcType, srcType)
static inline Ptr<Filter> createMorphologyFilter(int op, int srcType, InputArray kernel, Point anchor = Point(-1, -1), int iterations = 1)
{
    return createMorphologyFilter(op, static_cast<ElemType>(srcType), kernel, anchor, iterations);
}
CV_DEPRECATED_ELEMDEPTH_TO_ELEMTYPE_ATTR(srcType, srcType)
static inline Ptr<Filter> createMorphologyFilter(int op, ElemDepth srcType, InputArray kernel, Point anchor = Point(-1, -1), int iterations = 1)
{
    return createMorphologyFilter(op, CV_MAKETYPE(srcType, 1), kernel, anchor, iterations);
}
#endif // CV_TYPE_COMPATIBLE_API

////////////////////////////////////////////////////////////////////////////////////////////////////
// Image Rank Filter

/** @brief Creates the maximum filter.

@param srcType Input/output image type. Only CV_8UC1 and CV_8UC4 are supported.
@param ksize Kernel size.
@param anchor Anchor point. The default value (-1) means that the anchor is at the kernel center.
@param borderMode Pixel extrapolation method. For details, see borderInterpolate .
@param borderVal Default border value.
 */
CV_EXPORTS_W Ptr<Filter> createBoxMaxFilter(ElemType srcType, Size ksize,
                                          Point anchor = Point(-1, -1),
                                          int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0));
#ifdef CV_TYPE_COMPATIBLE_API
CV_DEPRECATED_INT_TO_ELEMTYPE_ATTR(srcType, srcType)
static inline Ptr<Filter> createBoxMaxFilter(int srcType, Size ksize,
                                          Point anchor = Point(-1, -1),
                                          int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0))
{
    return createBoxMaxFilter(static_cast<ElemType>(srcType), ksize, anchor, borderMode, borderVal);
}
CV_DEPRECATED_ELEMDEPTH_TO_ELEMTYPE_ATTR(srcType, srcType)
static inline Ptr<Filter> createBoxMaxFilter(ElemDepth srcType, Size ksize,
                                          Point anchor = Point(-1, -1),
                                          int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0))
{
    return createBoxMaxFilter(CV_MAKETYPE(srcType, 1), ksize, anchor, borderMode, borderVal);
}
#endif // CV_TYPE_COMPATIBLE_API

/** @brief Creates the minimum filter.

@param srcType Input/output image type. Only CV_8UC1 and CV_8UC4 are supported.
@param ksize Kernel size.
@param anchor Anchor point. The default value (-1) means that the anchor is at the kernel center.
@param borderMode Pixel extrapolation method. For details, see borderInterpolate .
@param borderVal Default border value.
 */
CV_EXPORTS_W Ptr<Filter> createBoxMinFilter(ElemType srcType, Size ksize,
                                          Point anchor = Point(-1, -1),
                                          int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0));
#ifdef CV_TYPE_COMPATIBLE_API
CV_DEPRECATED_INT_TO_ELEMTYPE_ATTR(srcType, srcType)
static inline Ptr<Filter> createBoxMinFilter(int srcType, Size ksize,
                                          Point anchor = Point(-1, -1),
                                          int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0))
{
    return createBoxMinFilter(static_cast<ElemType>(srcType), ksize, anchor, borderMode, borderVal);
}
CV_DEPRECATED_ELEMDEPTH_TO_ELEMTYPE_ATTR(srcType, srcType)
static inline Ptr<Filter> createBoxMinFilter(ElemDepth srcType, Size ksize,
                                          Point anchor = Point(-1, -1),
                                          int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0))
{
    return createBoxMinFilter(CV_MAKETYPE(srcType, 1), ksize, anchor, borderMode, borderVal);
}
#endif // CV_TYPE_COMPATIBLE_API

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
CV_EXPORTS_W Ptr<Filter> createRowSumFilter(ElemType srcType, ElemType dstType, int ksize, int anchor = -1, int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0));
#ifdef CV_TYPE_COMPATIBLE_API
#  ifndef OPENCV_DISABLE_DEPRECATED_WARNING_INT_ELEMTYPE_OVERLOAD
CV_DEPRECATED_MSG(CV_DEPRECATED_PARAM(int, srcType, ElemType, srcType) ". Similarly, " CV_DEPRECATED_PARAM(int, dstType, ElemType, dstType))
#  endif
static inline Ptr<Filter> createRowSumFilter(int srcType, int dstType, int ksize, int anchor = -1, int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0))
{
    return createRowSumFilter(static_cast<ElemType>(srcType), static_cast<ElemType>(dstType), ksize, anchor, borderMode, borderVal);
}
#  ifdef OPENCV_ENABLE_DEPRECATED_WARNING_ELEMDEPTH_ELEMTYPE_OVERLOAD
CV_DEPRECATED_MSG(CV_DEPRECATED_PARAM(ElemDepth, srcType, ElemType, srcType) ". Similarly, " CV_DEPRECATED_PARAM(ElemDepth, dstType, ElemType, dstType))
#  endif
static inline Ptr<Filter> createRowSumFilter(ElemDepth srcType, ElemDepth dstType, int ksize, int anchor = -1, int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0))
{
    return createRowSumFilter(CV_MAKETYPE(srcType, 1), CV_MAKETYPE(dstType, 1), ksize, anchor, borderMode, borderVal);
}
#endif // CV_TYPE_COMPATIBLE_API

/** @brief Creates a vertical 1D box filter.

@param srcType Input image type. Only CV_8UC1 type is supported for now.
@param dstType Output image type. Only CV_32FC1 type is supported for now.
@param ksize Kernel size.
@param anchor Anchor point. The default value (-1) means that the anchor is at the kernel center.
@param borderMode Pixel extrapolation method. For details, see borderInterpolate .
@param borderVal Default border value.
 */
CV_EXPORTS_W Ptr<Filter> createColumnSumFilter(ElemType srcType, ElemType dstType, int ksize, int anchor = -1, int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0));
#ifdef CV_TYPE_COMPATIBLE_API
#  ifndef OPENCV_DISABLE_DEPRECATED_WARNING_INT_ELEMTYPE_OVERLOAD
CV_DEPRECATED_MSG(CV_DEPRECATED_PARAM(int, srcType, ElemType, srcType) ". Similarly, " CV_DEPRECATED_PARAM(int, dstType, ElemType, dstType))
#  endif
static inline Ptr<Filter> createColumnSumFilter(int srcType, int dstType, int ksize, int anchor = -1, int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0))
{
    return createColumnSumFilter(static_cast<ElemType>(srcType), static_cast<ElemType>(dstType), ksize, anchor, borderMode, borderVal);
}
#  ifdef OPENCV_ENABLE_DEPRECATED_WARNING_ELEMDEPTH_ELEMTYPE_OVERLOAD
CV_DEPRECATED_MSG(CV_DEPRECATED_PARAM(ElemDepth, srcType, ElemType, srcType) ". Similarly, " CV_DEPRECATED_PARAM(ElemDepth, dstType, ElemType, dstType))
#  endif
static inline Ptr<Filter> createColumnSumFilter(ElemDepth srcType, ElemDepth dstType, int ksize, int anchor = -1, int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0))
{
    return createColumnSumFilter(CV_MAKETYPE(srcType, 1), CV_MAKETYPE(dstType, 1), ksize, anchor, borderMode, borderVal);
}
#endif // CV_TYPE_COMPATIBLE_API

//! @}

///////////////////////////// Median Filtering //////////////////////////////

/** @brief Performs median filtering for each point of the source image.

@param srcType type of of source image. Only CV_8UC1 images are supported for now.
@param windowSize Size of the kernerl used for the filtering. Uses a (windowSize x windowSize) filter.
@param partition Specifies the parallel granularity of the workload. This parameter should be used GPU experts when optimizing performance.

Outputs an image that has been filtered using median-filtering formulation.
 */
CV_EXPORTS_W Ptr<Filter> createMedianFilter(ElemType srcType, int windowSize, int partition = 128);
#ifdef CV_TYPE_COMPATIBLE_API
CV_DEPRECATED_INT_TO_ELEMTYPE_ATTR(srcType, srcType)
static inline Ptr<Filter> createMedianFilter(int srcType, int windowSize, int partition = 128)
{
    return createMedianFilter(static_cast<ElemType>(srcType), windowSize, partition);
}
CV_DEPRECATED_ELEMDEPTH_TO_ELEMTYPE_ATTR(srcType, srcType)
static inline Ptr<Filter> createMedianFilter(ElemDepth srcType, int windowSize, int partition = 128)
{
    return createMedianFilter(CV_MAKETYPE(srcType, 1), windowSize, partition);
}
#endif // CV_TYPE_COMPATIBLE_API

}} // namespace cv { namespace cuda {

#endif /* OPENCV_CUDAFILTERS_HPP */
