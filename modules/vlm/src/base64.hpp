// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_VLM_BASE64_HPP
#define OPENCV_VLM_BASE64_HPP

#include <cstddef>
#include <string>

namespace cv { namespace vlm {

std::string base64Encode(const unsigned char* data, size_t size);

}} // namespace cv::vlm

#endif // OPENCV_VLM_BASE64_HPP
