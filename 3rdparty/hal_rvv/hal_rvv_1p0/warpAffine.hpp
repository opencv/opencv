// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_WARPAFFINE_HPP_INCLUDED
#define OPENCV_HAL_RVV_WARPAFFINE_HPP_INCLUDED

#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv {
#undef cv_hal_warpAffine
#define cv_hal_warpAffine cv::cv_hal_rvv::warpAffine
#undef cv_hal_warpAffineBlocklineNN
#define cv_hal_warpAffineBlocklineNN cv::cv_hal_rvv::warpAffineBlocklineNN
#undef cv_hal_warpAffineBlockline
#define cv_hal_warpAffineBlockline cv::cv_hal_rvv::warpAffineBlockline

void warpAffine(int src_type,
                const uchar * src_data, size_t src_step, int src_width, int src_height,
                uchar * dst_data, size_t dst_step, int dst_width, int dst_height,
                const double M[6], int interpolation, int borderType, const double borderValue[4]) {
// return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

void warpAffineBlocklineNN(int *adelta, int *bdelta, short* xy, int X0, int Y0, int bw) {
// return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

void warpAffineBlockline(int *adelta, int *bdelta, short* xy, short* alpha, int X0, int Y0, int bw) {
// return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

} // cv_hal_rvv::
} // cv::

#endif