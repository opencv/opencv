// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_S11N_HPP
#define OPENCV_GAPI_S11N_HPP

#include <vector>
#include <opencv2/gapi/gcomputation.hpp>

namespace cv {
namespace gapi {

namespace detail {
    GAPI_EXPORTS cv::GComputation getGraph(const std::vector<char> &p);
} // namespace detail

GAPI_EXPORTS std::vector<char> serialize(const cv::GComputation &c);
//namespace{

template<typename T> static inline
T deserialize(const std::vector<char> &p);

//} //ananymous namespace

template<> inline
cv::GComputation deserialize(const std::vector<char> &p) {
    return detail::getGraph(p);
}



} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_S11N_HPP
