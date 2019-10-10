// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../test_precomp.hpp"
#include "../common/gapi_render_tests.hpp"

namespace opencv_test
{

// FIXME avoid code duplicate for NV12 and BGR cases
INSTANTIATE_TEST_CASE_P(RenderBGRTestRectsImpl, RenderBGRTestRects,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(cv::Rect(100, 100, 200, 200)),
                                Values(cv::Scalar(100, 50, 150)),
                                Values(2)));

INSTANTIATE_TEST_CASE_P(RenderNV12TestRectsImpl, RenderNV12TestRects,
                        Combine(Values(cv::Size(1280, 720)),
                                       Values(cv::Rect(100, 100, 200, 200)),
                                       Values(cv::Scalar(100, 50, 150)),
                                       Values(2)));

INSTANTIATE_TEST_CASE_P(RenderBGRTestCirclesImpl, RenderBGRTestCircles,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(cv::Point(100, 100)),
                                Values(10),
                                Values(cv::Scalar(100, 50, 150)),
                                Values(2)));

INSTANTIATE_TEST_CASE_P(RenderNV12TestCirclesImpl, RenderNV12TestCircles,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(cv::Point(100, 100)),
                                Values(10),
                                Values(cv::Scalar(100, 50, 150)),
                                Values(2)));

INSTANTIATE_TEST_CASE_P(RenderBGRTestLinesImpl, RenderBGRTestLines,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(cv::Point(100, 100)),
                                Values(cv::Point(200, 200)),
                                Values(cv::Scalar(100, 50, 150)),
                                Values(2)));

INSTANTIATE_TEST_CASE_P(RenderNV12TestLinesImpl, RenderNV12TestLines,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(cv::Point(100, 100)),
                                Values(cv::Point(200, 200)),
                                Values(cv::Scalar(100, 50, 150)),
                                Values(2)));

INSTANTIATE_TEST_CASE_P(RenderBGRTestTextsImpl, RenderBGRTestTexts,
                        Combine(Values(cv::Size(1280, 720)),
                                Values("SomeText"),
                                Values(cv::Point(200, 200)),
                                Values(2.0),
                                Values(cv::Scalar(0, 255, 0))));

INSTANTIATE_TEST_CASE_P(RenderNV12TestTextsImpl, RenderNV12TestTexts,
                        Combine(Values(cv::Size(1280, 720)),
                                Values("SomeText"),
                                Values(cv::Point(200, 200)),
                                Values(2.0),
                                Values(cv::Scalar(0, 255, 0))));

INSTANTIATE_TEST_CASE_P(RenderBGRTestMosaicsImpl, RenderBGRTestMosaics,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(cv::Rect(100, 100, 200, 200)),
                                Values(25),
                                Values(0)));

INSTANTIATE_TEST_CASE_P(RenderNV12TestMosaicsImpl, RenderNV12TestMosaics,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(cv::Rect(100, 100, 200, 200)),
                                Values(25),
                                Values(0)));

INSTANTIATE_TEST_CASE_P(RenderBGRTestImagesImpl, RenderBGRTestImages,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(cv::Rect(100, 100, 200, 200)),
                                Values(cv::Scalar(100, 150, 60)),
                                Values(1.0)));

INSTANTIATE_TEST_CASE_P(RenderNV12TestImagesImpl, RenderNV12TestImages,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(cv::Rect(100, 100, 200, 200)),
                                Values(cv::Scalar(100, 150, 60)),
                                Values(1.0)));

INSTANTIATE_TEST_CASE_P(RenderBGRTestPolylinesImpl, RenderBGRTestPolylines,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(std::vector<cv::Point>{{100, 100}, {200, 200}, {150, 300}, {400, 150}}),
                                Values(cv::Scalar(100, 150, 60)),
                                Values(3)));

INSTANTIATE_TEST_CASE_P(RenderNV12TestPolylinesImpl, RenderNV12TestPolylines,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(std::vector<cv::Point>{{100, 100}, {200, 200}, {150, 300}, {400, 150}}),
                                Values(cv::Scalar(100, 150, 60)),
                                Values(3)));
}
