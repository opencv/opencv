// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_WARPAFFINE_HPP_INCLUDED
#define OPENCV_HAL_RVV_WARPAFFINE_HPP_INCLUDED

#include <riscv_vector.h>
namespace cv { namespace cv_hal_rvv {

#undef cv_hal_remap32f
#define cv_hal_remap32f cv::cv_hal_rvv::remap32f

static int remap32f(int src_type, const uchar* src_data, size_t src_step, int src_width, int src_height,
    uchar* dst_data, size_t dst_step, int dst_width, int dst_height, float* mapx, size_t mapx_step,
    float* mapy, size_t mapy_step, int interpolation, int border_type, const double border_value[4])

} // cv_hal_rvv::
} // cv::

#endif