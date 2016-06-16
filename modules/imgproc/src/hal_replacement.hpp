/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015, Itseez Inc., all rights reserved.
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

#ifndef OPENCV_IMGPROC_HAL_REPLACEMENT_HPP
#define OPENCV_IMGPROC_HAL_REPLACEMENT_HPP

#include "opencv2/core/hal/interface.h"

#if defined __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wunused-parameter"
#elif defined _MSC_VER
#  pragma warning( push )
#  pragma warning( disable: 4100 )
#endif

//! @addtogroup imgproc_hal_interface
//! @note Define your functions to override default implementations:
//! @code
//! #undef hal_add8u
//! #define hal_add8u my_add8u
//! @endcode
//! @{

/**
@brief Dummy structure storing filtering context

Users can convert this pointer to any type they want. Initialisation and destruction should be made in Init and Free function implementations correspondingly.
Example:
@code{.cpp}
int my_hal_filterInit(cvhalFilter2D **context, ...) {
    context = static_cast<cvhalFilter2D*>(new MyFilterData());
    //... init
}

int my_hal_filterFree(cvhalFilter2D *context) {
    MyFilterData *c = static_cast<MyFilterData*>(context);
    delete c;
}
@endcode
 */
struct cvhalFilter2D {};

/**
   @brief hal_filterInit
   @param context double pointer to user-defined context
   @param kernel_data pointer to kernel data
   @param kernel_step kernel step
   @param kernel_type kernel type (CV_8U, ...)
   @param kernel_width kernel width
   @param kernel_height kernel height
   @param max_width max possible image width, can be used to allocate working buffers
   @param max_height max possible image height
   @param src_type source image type
   @param dst_type destination image type
   @param borderType border processing mode (CV_HAL_BORDER_REFLECT, ...)
   @param delta added to pixel values
   @param anchor_x relative X position of center point within the kernel
   @param anchor_y relative Y position of center point within the kernel
   @param allowSubmatrix indicates whether the submatrices will be allowed as source image
   @param allowInplace indicates whether the inplace operation will be possible
   @sa cv::filter2D, cv::hal::Filter2D
 */
inline int hal_ni_filterInit(cvhalFilter2D **context, uchar *kernel_data, size_t kernel_step, int kernel_type, int kernel_width, int kernel_height, int max_width, int max_height, int src_type, int dst_type, int borderType, double delta, int anchor_x, int anchor_y, bool allowSubmatrix, bool allowInplace) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
/**
   @brief hal_filter
   @param context pointer to user-defined context
   @param src_data source image data
   @param src_step source image step
   @param dst_data destination image data
   @param dst_step destination image step
   @param width images width
   @param height images height
   @param full_width full width of source image (outside the ROI)
   @param full_height full height of source image (outside the ROI)
   @param offset_x source image ROI offset X
   @param offset_y source image ROI offset Y
   @sa cv::filter2D, cv::hal::Filter2D
 */
inline int hal_ni_filter(cvhalFilter2D *context, uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step, int width, int height, int full_width, int full_height, int offset_x, int offset_y) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
/**
   @brief hal_filterFree
   @param context pointer to user-defined context
   @sa cv::filter2D, cv::hal::Filter2D
 */
inline int hal_ni_filterFree(cvhalFilter2D *context) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }

//! @cond IGNORED
#define cv_hal_filterInit hal_ni_filterInit
#define cv_hal_filter hal_ni_filter
#define cv_hal_filterFree hal_ni_filterFree
//! @endcond

/**
   @brief hal_sepFilterInit
   @param context double pointer to user-defined context
   @param src_type source image type
   @param dst_type destination image type
   @param kernel_type kernels type
   @param kernelx_data pointer to x-kernel data
   @param kernelx_step x-kernel step
   @param kernelx_width x-kernel width
   @param kernelx_height x-kernel height
   @param kernely_data pointer to y-kernel data
   @param kernely_step y-kernel step
   @param kernely_width y-kernel width
   @param kernely_height y-kernel height
   @param anchor_x relative X position of center point within the kernel
   @param anchor_y relative Y position of center point within the kernel
   @param delta added to pixel values
   @param borderType border processing mode (CV_HAL_BORDER_REFLECT, ...)
   @sa cv::sepFilter2D, cv::hal::SepFilter2D
 */
inline int hal_ni_sepFilterInit(cvhalFilter2D **context, int src_type, int dst_type, int kernel_type, uchar *kernelx_data, size_t kernelx_step, int kernelx_width, int kernelx_height, uchar *kernely_data, size_t kernely_step, int kernely_width, int kernely_height, int anchor_x, int anchor_y, double delta, int borderType) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
/**
   @brief hal_sepFilter
   @param context pointer to user-defined context
   @param src_data source image data
   @param src_step source image step
   @param dst_data destination image data
   @param dst_step destination image step
   @param width images width
   @param height images height
   @param full_width full width of source image (outside the ROI)
   @param full_height full height of source image (outside the ROI)
   @param offset_x source image ROI offset X
   @param offset_y source image ROI offset Y
   @sa cv::sepFilter2D, cv::hal::SepFilter2D
 */
inline int hal_ni_sepFilter(cvhalFilter2D *context, uchar *src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height, int full_width, int full_height, int offset_x, int offset_y) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
/**
   @brief hal_sepFilterFree
   @param context pointer to user-defined context
   @sa cv::sepFilter2D, cv::hal::SepFilter2D
 */
inline int hal_ni_sepFilterFree(cvhalFilter2D *context) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }

//! @cond IGNORED
#define cv_hal_sepFilterInit hal_ni_sepFilterInit
#define cv_hal_sepFilter hal_ni_sepFilter
#define cv_hal_sepFilterFree hal_ni_sepFilterFree
//! @endcond

/**
   @brief hal_morphInit
   @param context double pointer to user-defined context
   @param operation morphology operation CV_HAL_MORPH_ERODE or CV_HAL_MORPH_DILATE
   @param src_type source image type
   @param dst_type destination image type
   @param max_width max possible image width, can be used to allocate working buffers
   @param max_height max possible image height
   @param kernel_type kernel type (CV_8U, ...)
   @param kernel_data pointer to kernel data
   @param kernel_step kernel step
   @param kernel_width kernel width
   @param kernel_height kernel height
   @param anchor_x relative X position of center point within the kernel
   @param anchor_y relative Y position of center point within the kernel
   @param borderType border processing mode (CV_HAL_BORDER_REFLECT, ...)
   @param borderValue values to use for CV_HAL_BORDER_CONSTANT mode
   @param iterations number of iterations
   @param allowSubmatrix indicates whether the submatrices will be allowed as source image
   @param allowInplace indicates whether the inplace operation will be possible
   @sa cv::erode, cv::dilate, cv::morphologyEx, cv::hal::Morph
 */
inline int hal_ni_morphInit(cvhalFilter2D **context, int operation, int src_type, int dst_type, int max_width, int max_height, int kernel_type, uchar *kernel_data, size_t kernel_step, int kernel_width, int kernel_height, int anchor_x, int anchor_y, int borderType, const double borderValue[4], int iterations, bool allowSubmatrix, bool allowInplace) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
/**
   @brief hal_morph
   @param context pointer to user-defined context
   @param src_data source image data
   @param src_step source image step
   @param dst_data destination image data
   @param dst_step destination image step
   @param width images width
   @param height images height
   @param src_full_width full width of source image (outside the ROI)
   @param src_full_height full height of source image (outside the ROI)
   @param src_roi_x source image ROI X offset
   @param src_roi_y source image ROI Y offset
   @param dst_full_width full width of destination image
   @param dst_full_height full height of destination image
   @param dst_roi_x destination image ROI X offset
   @param dst_roi_y destination image ROI Y offset
   @sa cv::erode, cv::dilate, cv::morphologyEx, cv::hal::Morph
 */
inline int hal_ni_morph(cvhalFilter2D *context, uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step, int width, int height, int src_full_width, int src_full_height, int src_roi_x, int src_roi_y, int dst_full_width, int dst_full_height, int dst_roi_x, int dst_roi_y) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
/**
   @brief hal_morphFree
   @param context pointer to user-defined context
   @sa cv::erode, cv::dilate, cv::morphologyEx, cv::hal::Morph
 */
inline int hal_ni_morphFree(cvhalFilter2D *context) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }

//! @cond IGNORED
#define cv_hal_morphInit hal_ni_morphInit
#define cv_hal_morph hal_ni_morph
#define cv_hal_morphFree hal_ni_morphFree
//! @endcond

/**
   @brief hal_resize
   @param src_type source and destination image type
   @param src_data source image data
   @param src_step source image step
   @param src_width source image width
   @param src_height source image height
   @param dst_data destination image data
   @param dst_step destination image step
   @param dst_width destination image width
   @param dst_height destination image height
   @param inv_scale_x inversed scale X coefficient
   @param inv_scale_y inversed scale Y coefficient
   @param interpolation interpolation mode (CV_HAL_INTER_NEAREST, ...)
   @sa cv::resize, cv::hal::resize
 */
inline int hal_ni_resize(int src_type, const uchar *src_data, size_t src_step, int src_width, int src_height, uchar *dst_data, size_t dst_step, int dst_width, int dst_height, double inv_scale_x, double inv_scale_y, int interpolation) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
/**
   @brief hal_warpAffine
   @param src_type source and destination image type
   @param src_data source image data
   @param src_step source image step
   @param src_width source image width
   @param src_height source image height
   @param dst_data destination image data
   @param dst_step destination image step
   @param dst_width destination image width
   @param dst_height destination image height
   @param M 3x2 matrix with transform coefficients
   @param interpolation interpolation mode (CV_HAL_INTER_NEAREST, ...)
   @param borderType border processing mode (CV_HAL_BORDER_REFLECT, ...)
   @param borderValue values to use for CV_HAL_BORDER_CONSTANT mode
   @sa cv::warpAffine, cv::hal::warpAffine
 */
inline int hal_ni_warpAffine(int src_type, const uchar *src_data, size_t src_step, int src_width, int src_height, uchar *dst_data, size_t dst_step, int dst_width, int dst_height, const double M[6], int interpolation, int borderType, const double borderValue[4]) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
/**
   @brief hal_warpPerspectve
   @param src_type source and destination image type
   @param src_data source image data
   @param src_step source image step
   @param src_width source image width
   @param src_height source image height
   @param dst_data destination image data
   @param dst_step destination image step
   @param dst_width destination image width
   @param dst_height destination image height
   @param M 3x3 matrix with transform coefficients
   @param interpolation interpolation mode (CV_HAL_INTER_NEAREST, ...)
   @param borderType border processing mode (CV_HAL_BORDER_REFLECT, ...)
   @param borderValue values to use for CV_HAL_BORDER_CONSTANT mode
   @sa cv::warpPerspective, cv::hal::warpPerspective
 */
inline int hal_ni_warpPerspectve(int src_type, const uchar *src_data, size_t src_step, int src_width, int src_height, uchar *dst_data, size_t dst_step, int dst_width, int dst_height, const double M[9], int interpolation, int borderType, const double borderValue[4]) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }

//! @cond IGNORED
#define cv_hal_resize hal_ni_resize
#define cv_hal_warpAffine hal_ni_warpAffine
#define cv_hal_warpPerspective hal_ni_warpPerspectve
//! @endcond

//! @}

#if defined __GNUC__
#  pragma GCC diagnostic pop
#elif defined _MSC_VER
#  pragma warning( pop )
#endif


#include "custom_hal.hpp"

//! @cond IGNORED
#define CALL_HAL_RET(name, fun, retval, ...) \
    int res = fun(__VA_ARGS__, &retval); \
    if (res == CV_HAL_ERROR_OK) \
        return retval; \
    else if (res != CV_HAL_ERROR_NOT_IMPLEMENTED) \
        CV_Error_(cv::Error::StsInternal, \
            ("HAL implementation " CVAUX_STR(name) " ==> " CVAUX_STR(fun) " returned %d (0x%08x)", res, res));


#define CALL_HAL(name, fun, ...) \
    int res = fun(__VA_ARGS__); \
    if (res == CV_HAL_ERROR_OK) \
        return; \
    else if (res != CV_HAL_ERROR_NOT_IMPLEMENTED) \
        CV_Error_(cv::Error::StsInternal, \
            ("HAL implementation " CVAUX_STR(name) " ==> " CVAUX_STR(fun) " returned %d (0x%08x)", res, res));
//! @endcond

#endif
