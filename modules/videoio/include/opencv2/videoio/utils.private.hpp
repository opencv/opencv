// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_VIDEOIO_UTILS_PRIVATE_HPP
#define OPENCV_VIDEOIO_UTILS_PRIVATE_HPP

#include "opencv2/core/cvdef.h"
#include <string>

namespace cv {
CV_EXPORTS std::string icvExtractPattern(const std::string& filename, unsigned *offset);
}

#endif // OPENCV_VIDEOIO_UTILS_PRIVATE_HPP
