// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#ifndef OPENCV_GAPI_RENDER_PERF_TESTS_HPP
#define OPENCV_GAPI_RENDER_PERF_TESTS_HPP



#include "../../test/common/gapi_tests_common.hpp"
#include <opencv2/gapi/imgproc.hpp>

#include <string>

namespace opencv_test
{

  using namespace perf;

  #ifdef HAVE_FREETYPE
  class RenderTestFTexts : public TestPerfParams<tuple<std::wstring, cv::Size, cv::Point,
                                                       int, cv::Scalar, MatType>> {};
  #endif // HAVE_FREETYPE

  class RenderTestTexts : public TestPerfParams<tuple<std::string, cv::Size, cv::Point,
                                                      int, double, cv::Scalar, int, int,
                                                      bool, MatType>> {};

  class RenderTestRects : public TestPerfParams<tuple<cv::Size, cv::Rect, cv::Scalar,
                                                      int, int, int, MatType,
                                                      compare_f>> {};

  class RenderTestCircles : public TestPerfParams<tuple<cv::Size, cv::Point, int,
                                                        cv::Scalar, int, int, int,
                                                        MatType>> {};

  class RenderTestLines : public TestPerfParams<tuple<cv::Size, cv::Point, cv::Point,
                                                      cv::Scalar, int, int, int, MatType>> {};

  class RenderTestMosaics : public TestPerfParams<tuple<cv::Size, cv::Rect, int, int,
                                                        MatType>> {};

  class RenderTestImages : public TestPerfParams<tuple<cv::Size, cv::Rect, cv::Scalar, double,
                                                       MatType>> {};

  class RenderTestPolylines : public TestPerfParams<tuple<cv::Size, std::vector<cv::Point>,
                                                          cv::Scalar, int, int, int, MatType>> {};

  class RenderTestPolyItems : public TestPerfParams<tuple<cv::Size, int, int, int>> {};

} // opencv_test

#endif // OPENCV_GAPI_RENDER_PERF_TESTS_HPP
