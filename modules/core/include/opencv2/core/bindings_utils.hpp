// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_BINDINGS_UTILS_HPP
#define OPENCV_CORE_BINDINGS_UTILS_HPP

namespace cv { namespace utils {
//! @addtogroup core_utils
//! @{

CV_EXPORTS_W String dumpInputArray(InputArray argument);

CV_EXPORTS_W String dumpInputArrayOfArrays(InputArrayOfArrays argument);

CV_EXPORTS_W String dumpInputOutputArray(InputOutputArray argument);

CV_EXPORTS_W String dumpInputOutputArrayOfArrays(InputOutputArrayOfArrays argument);

//! @}
}} // namespace

#endif // OPENCV_CORE_BINDINGS_UTILS_HPP
