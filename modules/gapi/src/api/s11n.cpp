// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include <opencv2/gapi/s11n.hpp>

#include "backends/common/serialization.hpp"

std::vector<char> cv::gapi::serialize(const cv::GComputation &c) {
    cv::gimpl::s11n::SerializationStream os;
    c.serialize(os);

    // FIXME: This stream API needs a fix-up
    std::vector<char> result;
    result.resize(os.getSize());
    std::copy_n(os.getData(), os.getSize(), result.begin());
    return result;
}

cv::GComputation cv::gapi::detail::getGraph(const std::vector<char> &p) {
    cv::gimpl::s11n::DeSerializationStream is(p.data(), p.size());
    return cv::GComputation(is);
}
