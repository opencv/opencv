// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef __IPP_HAL_IMGPROC_HPP__
#define __IPP_HAL_IMGPROC_HPP__

#include <opencv2/core/base.hpp>
#include "ipp_utils.hpp"

#ifdef HAVE_IPP_IW

int ipp_hal_warpAffine(int src_type, const uchar *src_data, size_t src_step, int src_width, int src_height, uchar *dst_data, size_t dst_step, int dst_width,
                       int dst_height, const double M[6], int interpolation, int borderType, const double borderValue[4]);

// Does not pass tests in 5.x branch
//#undef cv_hal_warpAffine
//#define cv_hal_warpAffine ipp_hal_warpAffine
#endif

#if IPP_VERSION_X100 >= 810
int ipp_hal_warpPerspective(int src_type, const uchar *src_data, size_t src_step, int src_width, int src_height, uchar *dst_data, size_t dst_step, int dst_width,
                            int dst_height, const double M[9], int interpolation, int borderType, const double borderValue[4]);

// Does not pass tests in 5.x branch
//#undef cv_hal_warpPerspective
//#define cv_hal_warpPerspective ipp_hal_warpPerspective
#endif


int ipp_hal_remap32f(int src_type, const uchar *src_data, size_t src_step, int src_width, int src_height,
    uchar *dst_data, size_t dst_step, int dst_width, int dst_height,
    float* mapx, size_t mapx_step, float* mapy, size_t mapy_step,
    int interpolation, int border_type, const double border_value[4]);
#undef cv_hal_remap32f
#define cv_hal_remap32f ipp_hal_remap32f


#endif //__IPP_HAL_IMGPROC_HPP__
