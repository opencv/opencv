// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_SRC_METAL_HPP
#define OPENCV_CORE_SRC_METAL_HPP

#include "opencv2/core/mat.hpp"

namespace cv {
namespace metal {

bool haveMetal();
MatAllocator* getMetalAllocator();
bool copyToMask(const UMat& src, const UMat& mask, UMat& dst, bool haveDstUninit);

} // namespace metal
} // namespace cv

#endif // OPENCV_CORE_SRC_METAL_HPP
