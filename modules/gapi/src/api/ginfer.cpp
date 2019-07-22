// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2019 Intel Corporation


#include "precomp.hpp"

#include <functional> // hash
#include <numeric> // accumulate
#include <unordered_set>
#include <iterator>

#include <ade/util/algorithm.hpp>

#include <opencv2/gapi/infer.hpp>

cv::gapi::GNetPackage::GNetPackage(std::initializer_list<GNetParam> &&ii)
    : networks(std::move(ii)) {
}

std::vector<cv::gapi::GBackend> cv::gapi::GNetPackage::backends() const {
    std::unordered_set<cv::gapi::GBackend> unique_set;
    for (const auto &nn : networks) unique_set.insert(nn.backend);
    return std::vector<cv::gapi::GBackend>(unique_set.begin(), unique_set.end());
}
