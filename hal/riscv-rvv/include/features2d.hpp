// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_RVV_HAL_FEATURES2D_HPP
#define OPENCV_RVV_HAL_FEATURES2D_HPP

struct cvhalFilter2D;

namespace cv { namespace rvv_hal { namespace features2d {

#if CV_HAL_RVV_1P0_ENABLED

int FAST(const uchar* src_data, size_t src_step, int width, int height,
          void** keypoints_data, size_t* keypoints_count,
          int threshold, bool nonmax_suppression, int detector_type, void *realloc_func(void*, size_t));

#undef cv_hal_FAST
#define cv_hal_FAST cv::rvv_hal::features2d::FAST

#endif // CV_HAL_RVV_1P0_ENABLED


}}} // cv::rvv_hal::features2d

#endif // OPENCV_RVV_HAL_IMGPROC_HPP
