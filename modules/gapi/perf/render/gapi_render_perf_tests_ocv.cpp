// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#include "../perf_precomp.hpp"
#include "../common/gapi_render_perf_tests.hpp"

#define RENDER_OCV cv::gapi::render::ocv::kernels()

namespace opencv_test
{

#ifdef HAVE_FREETYPE
INSTANTIATE_TEST_CASE_P(RenderTestFTexts, RenderTestFTexts,
                        Combine(Values(L"\xe4\xbd\xa0\xe5\xa5\xbd"),
                                Values(szVGA, sz720p, sz1080p),
                                Values(cv::Point(50, 50)),
                                Values(60),
                                Values(cv::Scalar(200, 100, 25)),
                                Values(cv::compile_args(RENDER_OCV))));
#endif // HAVE_FREETYPE

INSTANTIATE_TEST_CASE_P(RenderTestTexts, RenderTestTexts,
                        Combine(Values(std::string("Some text")),
                                Values(szVGA, sz720p, sz1080p),
                                Values(cv::Point(200, 200)),
                                Values(FONT_HERSHEY_SIMPLEX),
                                Values(cv::Scalar(0, 255, 0)),
                                Values(2),
                                Values(LINE_8),
                                Values(false),
                                Values(cv::compile_args(RENDER_OCV))));

INSTANTIATE_TEST_CASE_P(RenderTestRects, RenderTestRects,
                        Combine(Values(szVGA, sz720p, sz1080p),
                                Values(cv::Rect(100, 100, 200, 200)),
                                Values(cv::Scalar(100, 50, 150)),
                                Values(2),
                                Values(LINE_8),
                                Values(0),
                                Values(cv::compile_args(RENDER_OCV))));

INSTANTIATE_TEST_CASE_P(RenderTestCircles, RenderTestCircles,
                        Combine(Values(szVGA, sz720p, sz1080p),
                                Values(cv::Point(100, 100)),
                                Values(10),
                                Values(cv::Scalar(100, 50, 150)),
                                Values(2),
                                Values(LINE_8),
                                Values(0),
                                Values(cv::compile_args(RENDER_OCV))));

INSTANTIATE_TEST_CASE_P(RenderTestLines, RenderTestLines,
                        Combine(Values(szVGA, sz720p, sz1080p),
                                Values(cv::Point(100, 100)),
                                Values(cv::Point(200, 200)),
                                Values(cv::Scalar(100, 50, 150)),
                                Values(2),
                                Values(LINE_8),
                                Values(0),
                                Values(cv::compile_args(RENDER_OCV))));

INSTANTIATE_TEST_CASE_P(RenderTestMosaics, RenderTestMosaics,
                        Combine(Values(szVGA, sz720p, sz1080p),
                                Values(cv::Rect(100, 100, 200, 200)),
                                Values(25),
                                Values(0),
                                Values(cv::compile_args(RENDER_OCV))));

INSTANTIATE_TEST_CASE_P(RenderTestImages, RenderTestImages,
                        Combine(Values(szVGA, sz720p, sz1080p),
                                Values(cv::Rect(50, 50, 100, 100)),
                                Values(cv::Scalar(100, 150, 60)),
                                Values(1.0),
                                Values(cv::compile_args(RENDER_OCV))));

INSTANTIATE_TEST_CASE_P(RenderTestPolylines, RenderTestPolylines,
                        Combine(Values(szVGA, sz720p, sz1080p),
                                Values(std::vector<cv::Point>{{100, 100}, {200, 200}, {150, 300}, {400, 150}}),
                                Values(cv::Scalar(100, 150, 60)),
                                Values(2),
                                Values(LINE_8),
                                Values(0),
                                Values(cv::compile_args(RENDER_OCV))));

INSTANTIATE_TEST_CASE_P(RenderTestPolyItems, RenderTestPolyItems,
                        Combine(Values(szVGA, sz720p, sz1080p),
                                Values(50),
                                Values(50),
                                Values(50),
                                Values(cv::compile_args(RENDER_OCV))));
}
