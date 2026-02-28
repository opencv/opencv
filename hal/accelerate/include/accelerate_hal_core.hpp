// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef __ACCELERATE_HAL_CORE_HPP__
#define __ACCELERATE_HAL_CORE_HPP__

#include <opencv2/core/base.hpp>

int accelerate_hal_meanStdDev(const uchar* src_data, size_t src_step, int width, int height, int src_type,
                              double* mean_val, double* stddev_val, uchar* mask, size_t mask_step);

#undef cv_hal_meanStdDev
#define cv_hal_meanStdDev accelerate_hal_meanStdDev

#endif
