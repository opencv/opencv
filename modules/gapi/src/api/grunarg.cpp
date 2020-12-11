// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include "precomp.hpp"
#include <opencv2/gapi/garg.hpp>

cv::GRunArg::GRunArg() {
}

cv::GRunArg::GRunArg(const cv::GRunArg &arg)
    : cv::GRunArgBase(static_cast<const cv::GRunArgBase&>(arg))
    , meta(arg.meta) {
}

cv::GRunArg::GRunArg(cv::GRunArg &&arg)
    : cv::GRunArgBase(std::move(static_cast<const cv::GRunArgBase&>(arg)))
    , meta(std::move(arg.meta)) {
}

cv::GRunArg& cv::GRunArg::operator= (const cv::GRunArg &arg) {
    cv::GRunArgBase::operator=(static_cast<const cv::GRunArgBase&>(arg));
    meta = arg.meta;
    return *this;
}

cv::GRunArg& cv::GRunArg::operator= (cv::GRunArg &&arg) {
    cv::GRunArgBase::operator=(std::move(static_cast<const cv::GRunArgBase&>(arg)));
    meta = std::move(arg.meta);
    return *this;
}
