// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../test_precomp.hpp"
#include "../common/gapi_render_tests.hpp"

namespace opencv_test
{

#define OCV cv::gapi::ocv::kernels()

/* NV12 test cases */
INSTANTIATE_TEST_CASE_P(RenderNV12OCVRects, RenderNV12,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(rects),
                                Values(OCV)));

INSTANTIATE_TEST_CASE_P(RenderNV12OCVCircles, RenderNV12,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(circles),
                                Values(OCV)));

INSTANTIATE_TEST_CASE_P(RenderNV12OCVLines, RenderNV12,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(lines),
                                Values(OCV)));

INSTANTIATE_TEST_CASE_P(RenderNV12OCVMosaics, RenderNV12,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(mosaics),
                                Values(OCV)));

// FIXME difference in color
INSTANTIATE_TEST_CASE_P(RenderNV12OCVImages, RenderNV12,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(images),
                                Values(OCV)));

INSTANTIATE_TEST_CASE_P(RenderNV12OCVPolygons, RenderNV12,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(polygons),
                                Values(OCV)));

INSTANTIATE_TEST_CASE_P(RenderNV12OCVTexts, RenderNV12,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(texts),
                                Values(OCV)));

/* BGR test cases */
INSTANTIATE_TEST_CASE_P(RenderBGROCVRects, RenderBGR,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(rects),
                                Values(OCV)));

INSTANTIATE_TEST_CASE_P(RenderBGROCVCircles, RenderBGR,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(circles),
                                Values(OCV)));

INSTANTIATE_TEST_CASE_P(RenderBGROCVLines, RenderBGR,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(lines),
                                Values(OCV)));

INSTANTIATE_TEST_CASE_P(RenderBGROCVMosaics, RenderBGR,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(mosaics),
                                Values(OCV)));

INSTANTIATE_TEST_CASE_P(RenderBGROCVImages, RenderBGR,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(images),
                                Values(OCV)));

INSTANTIATE_TEST_CASE_P(RenderBGROCVPolygons, RenderBGR,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(polygons),
                                Values(OCV)));

INSTANTIATE_TEST_CASE_P(RenderBGROCVTexts, RenderBGR,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(texts),
                                Values(OCV)));
}
