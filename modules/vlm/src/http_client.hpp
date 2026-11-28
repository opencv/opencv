// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_VLM_HTTP_CLIENT_HPP
#define OPENCV_VLM_HTTP_CLIENT_HPP

#include <string>
#include <vector>

namespace cv { namespace vlm {

struct HttpResponse
{
    long statusCode;
    std::string body;
};

HttpResponse httpPostJson(const std::string& url, const std::string& jsonBody,
                          const std::vector<std::string>& headers);

}} // namespace cv::vlm

#endif // OPENCV_VLM_HTTP_CLIENT_HPP
