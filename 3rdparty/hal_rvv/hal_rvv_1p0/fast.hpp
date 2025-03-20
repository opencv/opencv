// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.
#pragma once

#undef cv_hal_FAST_dense
#define cv_hal_FAST_dense cv::cv_hal_rvv::fast_dense

#undef cv_hal_FAST_NMS
#define cv_hal_FAST_NMS cv::cv_hal_rvv::fast_nms

#include <riscv_vector.h>
#include "../../../modules/features2d/include/opencv2/features2d.hpp"

#include <cfloat>

namespace cv::cv_hal_rvv {

inline int fast_dense(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, 
                        int width, int height, FastFeatureDetector::DetectorType type) {
    return CV_HAL_ERROR_OK;
}

inline int fast_nms(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height) {
    return CV_HAL_ERROR_OK;
}

} // namespace cv::cv_hal_rvv
