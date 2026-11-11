// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef __ACCELERATE_HAL_IMGPROC_HPP__
#define __ACCELERATE_HAL_IMGPROC_HPP__

#include <opencv2/core/base.hpp>

// sepFilter2D on top of Apple's vImage (vImageSepConvolve_*).
//
// Handled: 8U->8U, 8U->16U, 32F->32F and 16F->16F (16F needs macOS 13 / iOS 16 at run time),
// with a CV_32F kernel, default anchor and BORDER_CONSTANT or BORDER_REPLICATE.
// Everything else returns CV_HAL_ERROR_NOT_IMPLEMENTED, and OpenCV's own code is used.
//
// Threading: the function keeps no state, so any number of threads can call it at once.
// The image is split into stripes of rows with cv::parallel_for_, and each stripe is filtered
// on one thread (kvImageDoNotTile), so vImage starts no threads of its own. cv::setNumThreads()
// and nested parallel regions work as in the rest of OpenCV.

int accelerate_hal_sepFilter_stateless(const uchar* src_data, size_t src_step, int src_type,
                                       uchar* dst_data, size_t dst_step, int dst_type,
                                       int width, int height, int full_width, int full_height, int offset_x, int offset_y,
                                       const uchar* kernelx_data, int kernelx_len,
                                       const uchar* kernely_data, int kernely_len,
                                       int kernel_type, int anchor_x, int anchor_y, double delta, int borderType);

#undef cv_hal_sepFilter_stateless
#define cv_hal_sepFilter_stateless accelerate_hal_sepFilter_stateless

#endif
