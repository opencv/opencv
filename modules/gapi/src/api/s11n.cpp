// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include <opencv2/gapi/s11n.hpp>

#include "backends/common/serialization.hpp"

std::vector<char> cv::gapi::serialize(const cv::GComputation &c) {
    cv::gimpl::s11n::ByteMemoryOutStream os;
    c.serialize(os);
    return os.data();
}

cv::GComputation cv::gapi::detail::getGraph(const std::vector<char> &p) {
    cv::gimpl::s11n::ByteMemoryInStream is(p);
    return cv::GComputation(is);
}
