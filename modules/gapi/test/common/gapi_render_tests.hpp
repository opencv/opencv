// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_RENDER_TESTS_HPP
#define OPENCV_GAPI_RENDER_TESTS_HPP

#include "gapi_tests_common.hpp"

namespace opencv_test
{

using Points = std::vector<cv::Point>;
using Rects  = std::vector<cv::Rect>;

struct RenderTextTest : public TestParams <std::tuple<cv::Size,std::string,Points,int,double,cv::Scalar,int,int,bool>> {};

} // opencv_test

#endif //OPENCV_GAPI_RENDER_TESTS_HPP

