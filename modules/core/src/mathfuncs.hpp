// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_SRC_MATHFUNCS_HPP
#define OPENCV_CORE_SRC_MATHFUNCS_HPP

namespace cv { namespace details {
const double* getExpTab64f();
const float*  getExpTab32f();
const double* getLogTab64f();
const float*  getLogTab32f();
}} // namespace

#endif // OPENCV_CORE_SRC_MATHFUNCS_HPP
