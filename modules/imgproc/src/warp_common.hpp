// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_IMGPROC_WARP_COMMON_HPP__
#define __OPENCV_IMGPROC_WARP_COMMON_HPP__

#include "warp_common.vector.hpp"
#include "warp_common.scalar.hpp"

namespace cv {

typedef void (*ImgWarpFunc)(const float* x, const float* y, int len,
                            const void* src, size_t srcstep, Size size,
                            void* dst, const float* params,
                            int borderType, const void* borderVal);

ImgWarpFunc getImgWarpFunc(int type, int interpolation);

}

#endif // __OPENCV_IMGPROC_WARP_COMMON_HPP__
