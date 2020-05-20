// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 20120 Intel Corporation


#ifndef OPENCV_GAPI_RENDER_PERF_TESTS_HPP
#define OPENCV_GAPI_RENDER_PERF_TESTS_HPP


#include <codecvt>

#include "../../test/common/gapi_tests_common.hpp"
#include <opencv2/gapi/imgproc.hpp>

namespace opencv_test
{
  using namespace perf;

  class RenderTestFTexts : public TestPerfParams<tuple<std::wstring, cv::Size, cv::Point,
                                                       int, cv::Scalar, MatType, cv::GCompileArgs>> {};
  class RenderTestTexts : public TestPerfParams<tuple<std::string, cv::Size, cv::Point,
                                                      int, cv::Scalar, int, int,
                                                      bool, MatType, cv::GCompileArgs>> {};
  class RenderTestRects : public TestPerfParams<tuple<cv::Size, cv::Rect, cv::Scalar,
                                                      int, int, int, MatType, cv::GCompileArgs>> {};
  class RenderTestCircles : public TestPerfParams<tuple<cv::Size, cv::Point, int,
                                                        cv::Scalar, int, int, int,
                                                        MatType, cv::GCompileArgs>> {};
  class RenderTestLines : public TestPerfParams<tuple<cv::Size, cv::Point, cv::Point,
                                                      cv::Scalar, int, int, int, MatType,
                                                      cv::GCompileArgs>> {};
  class RenderTestMosaics : public TestPerfParams<tuple<cv::Size, cv::Rect, int, int,
                                                        MatType, cv::GCompileArgs>> {};
  class RenderTestImages : public TestPerfParams<tuple<cv::Size, cv::Rect, cv::Scalar, double,
                                                       MatType, cv::GCompileArgs>> {};
  class RenderTestPolylines : public TestPerfParams<tuple<cv::Size, std::vector<cv::Point>,
                                                          cv::Scalar, int, int, int, MatType,
                                                          cv::GCompileArgs>> {};
  class RenderTestPolyItems : public TestPerfParams<tuple<cv::Size, int, int, int, cv::GCompileArgs>> {};

}
#endif //OPENCV_GAPI_RENDER_PERF_TESTS_HPP
