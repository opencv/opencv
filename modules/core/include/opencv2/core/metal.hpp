// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_METAL_HPP
#define OPENCV_CORE_METAL_HPP

#include "opencv2/core/cvdef.h"
#include "opencv2/core/mat.hpp"

namespace cv {
/** @addtogroup core_metal
@{
*/
namespace metal {

/** @brief Returns true if the Metal UMat backend is available at runtime.

The result is true only when OpenCV was built with Metal support and a default
Metal device and command queue can be created.
*/
CV_EXPORTS bool haveMetal();

//! @cond INTERNAL
CV_EXPORTS bool threshold(const UMat& src, UMat& dst, double thresh, double maxval, int thresholdType);
//! @endcond

} // namespace metal
/** @} */
} // namespace cv

#endif // OPENCV_CORE_METAL_HPP
