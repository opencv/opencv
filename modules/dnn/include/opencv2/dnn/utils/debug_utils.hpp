// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_UTILS_DEBUG_UTILS_HPP
#define OPENCV_DNN_UTILS_DEBUG_UTILS_HPP

#include "../dnn.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

/**
 * @brief Skip model import after diagnostic run in readNet() functions.
 * @param[in] skip Indicates whether to skip the import.
 *
 * This is an internal OpenCV function not intended for users.
 */
CV_EXPORTS void skipModelImport(bool skip);

CV__DNN_INLINE_NS_END
}} // namespace

#endif // OPENCV_DNN_UTILS_DEBUG_UTILS_HPP
