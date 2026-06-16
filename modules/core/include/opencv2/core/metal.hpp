// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_METAL_HPP
#define OPENCV_CORE_METAL_HPP

#include "opencv2/core/cvdef.h"
#include "opencv2/core/mat.hpp"

namespace cv {
namespace metal {

CV_EXPORTS bool haveMetal();
CV_EXPORTS bool threshold(const UMat& src, UMat& dst, double thresh, double maxval, int thresholdType);

} // namespace metal
} // namespace cv

#endif // OPENCV_CORE_METAL_HPP
