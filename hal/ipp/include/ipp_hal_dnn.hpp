// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, Intel Corporation, all rights reserved.

#ifndef __IPP_HAL_DNN_HPP__
#define __IPP_HAL_DNN_HPP__

#include <opencv2/core/base.hpp>
#include "ipp_utils.hpp"

// Older ICV packages (2026.0.0, Mac/ia32 pins) miss some of the required primitives (e.g. ippiCopy_*_C3R_L)
#if IPP_VERSION_X100 >= 202610

int ipp_hal_blobFromImages(const uchar* const* src_data, const size_t* src_step, int nimages,
                           int width, int height, int src_type, int ddepth, int swapRB,
                           int dst_width, int dst_height, int paddingmode, const double borderValue[4],
                           const double mean[4], const double scalefactor[4],
                           uchar* dst_data, const size_t* dst_step, int layout, int interpolation);

#undef cv_hal_dnn_blobFromImages
#define cv_hal_dnn_blobFromImages ipp_hal_blobFromImages

#endif // IPP_VERSION_X100 >= 202610

#endif // __IPP_HAL_DNN_HPP__
