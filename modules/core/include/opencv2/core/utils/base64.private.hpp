// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_UTILS_BASE64_HPP
#define OPENCV_UTILS_BASE64_HPP

#include <opencv2/core/cvstd.hpp>

namespace cv { namespace utils {

/** @brief Encodes a byte buffer as base64 (RFC 4648), with padding.

Not part of the public API: this is the encoder FileStorage already uses for its base64
rawdata, exposed so that other modules do not have to carry their own copy.

@param data Bytes to encode. May be NULL only when @p size is 0.
@param size Number of bytes to encode.
@return The encoded text, or an empty string when @p size is 0.
*/
CV_EXPORTS String base64Encode(const uchar* data, size_t size);

}} // namespace cv::utils

#endif // OPENCV_UTILS_BASE64_HPP
