// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef __IPP_HAL_DNN_HPP__
#define __IPP_HAL_DNN_HPP__

#include <opencv2/core/hal/interface.h>

// Note: ipp_utils.hpp is not included here since this header only declares
// function interfaces that use standard C types. The implementation in
// hal/ipp/src/blob_from_images_ipp.cpp will include ipp_utils.hpp as needed.

// IPP_HAL_DNN is defined by the build system when IPP HAL is available
// Do not define it here unconditionally

int ipp_hal_blobFromImages(const uchar* const* src_data, const size_t* src_step,
                               int nimages, int width, int height, int depth, int channels,
                               int ddepth, int swapRB,
                               int dst_width, int dst_height, int paddingmode,
                               const double borderValue[4],
                               const float mean[4], const float scalefactor[4],
                               uchar* dst_data, size_t* dst_step, int layout,
                               int interpolation);

#endif // __IPP_HAL_DNN_HPP__
