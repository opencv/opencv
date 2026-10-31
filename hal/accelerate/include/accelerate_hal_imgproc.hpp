// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef __ACCELERATE_HAL_IMGPROC_HPP__
#define __ACCELERATE_HAL_IMGPROC_HPP__

#include <opencv2/core/base.hpp>


#if ((defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && defined(__MAC_11_0) && __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_11_0) || \
     (defined(__IPHONE_OS_VERSION_MAX_ALLOWED) && defined(__IPHONE_14_0) && __IPHONE_OS_VERSION_MAX_ALLOWED >= __IPHONE_14_0) || \
     (defined(__WATCH_OS_VERSION_MAX_ALLOWED) && defined(__WATCHOS_7_0) && __WATCH_OS_VERSION_MAX_ALLOWED >= __WATCHOS_7_0) || \
     (defined(__TV_OS_VERSION_MAX_ALLOWED) && defined(__TVOS_14_0) && __TV_OS_VERSION_MAX_ALLOWED >= __TVOS_14_0))

int accelerate_hal_sepFilter_stateless(const uchar* src_data, size_t src_step, int src_type,
                                       uchar* dst_data, size_t dst_step, int dst_type,
                                       int width, int height, int full_width, int full_height, int offset_x, int offset_y,
                                       const uchar* kernelx_data, int kernelx_len,
                                       const uchar* kernely_data, int kernely_len,
                                       int kernel_type, int anchor_x, int anchor_y, double delta, int borderType);

#undef cv_hal_sepFilter_stateless
#define cv_hal_sepFilter_stateless accelerate_hal_sepFilter_stateless

#endif

#endif
