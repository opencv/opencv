// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#ifndef OPENCV_GAPI_RENDER_PERF_TESTS_HPP
#define OPENCV_GAPI_RENDER_PERF_TESTS_HPP


#include "../../test/common/gapi_tests_common.hpp"
#include <opencv2/gapi/render/render.hpp>

namespace opencv_test
{

using namespace perf;

class RenderTestFTexts : public TestPerfParams<tuple<std::wstring, cv::Size, cv::Point,
                                                     int, cv::Scalar, cv::GCompileArgs>> {};
class RenderTestTexts : public TestPerfParams<tuple<std::string, cv::Size, cv::Point,
                                                    int, cv::Scalar, int, int,
                                                    bool, cv::GCompileArgs>> {};
class RenderTestRects : public TestPerfParams<tuple<cv::Size, cv::Rect, cv::Scalar,
                                                    int, int, int, cv::GCompileArgs>> {};
class RenderTestCircles : public TestPerfParams<tuple<cv::Size, cv::Point, int,
                                                      cv::Scalar, int, int, int,
                                                      cv::GCompileArgs>> {};
class RenderTestLines : public TestPerfParams<tuple<cv::Size, cv::Point, cv::Point,
                                                    cv::Scalar, int, int, int,
                                                    cv::GCompileArgs>> {};
class RenderTestMosaics : public TestPerfParams<tuple<cv::Size, cv::Rect, int, int,
                                                      cv::GCompileArgs>> {};
class RenderTestImages : public TestPerfParams<tuple<cv::Size, cv::Rect, cv::Scalar, double,
                                                     cv::GCompileArgs>> {};
class RenderTestPolylines : public TestPerfParams<tuple<cv::Size, std::vector<cv::Point>,
                                                        cv::Scalar, int, int, int,
                                                        cv::GCompileArgs>> {};
class RenderTestPolyItems : public TestPerfParams<tuple<cv::Size, int, int, int, cv::GCompileArgs>> {};

}
#endif //OPENCV_GAPI_RENDER_PERF_TESTS_HPP
