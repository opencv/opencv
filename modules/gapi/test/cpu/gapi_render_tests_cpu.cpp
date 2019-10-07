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
                                Values(PrimsTestParam::RECTS),
                                Values(OCV)));

INSTANTIATE_TEST_CASE_P(RenderNV12OCVCircles, RenderNV12,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(PrimsTestParam::CIRCLES),
                                Values(OCV)));

INSTANTIATE_TEST_CASE_P(RenderNV12OCVLines, RenderNV12,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(PrimsTestParam::LINES),
                                Values(OCV)));

INSTANTIATE_TEST_CASE_P(RenderNV12OCVMosaics, RenderNV12,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(PrimsTestParam::MOSAICS),
                                Values(OCV)));

// FIXME difference in color
INSTANTIATE_TEST_CASE_P(RenderNV12OCVImages, RenderNV12,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(PrimsTestParam::IMAGES),
                                Values(OCV)));

INSTANTIATE_TEST_CASE_P(RenderNV12OCVPolygons, RenderNV12,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(PrimsTestParam::POLYGONS),
                                Values(OCV)));

INSTANTIATE_TEST_CASE_P(RenderNV12OCVTexts, RenderNV12,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(PrimsTestParam::TEXTS),
                                Values(OCV)));

/* BGR test cases */
INSTANTIATE_TEST_CASE_P(RenderBGROCVRects, RenderBGR,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(PrimsTestParam::RECTS),
                                Values(OCV)));

INSTANTIATE_TEST_CASE_P(RenderBGROCVCircles, RenderBGR,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(PrimsTestParam::CIRCLES),
                                Values(OCV)));

INSTANTIATE_TEST_CASE_P(RenderBGROCVLines, RenderBGR,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(PrimsTestParam::LINES),
                                Values(OCV)));

INSTANTIATE_TEST_CASE_P(RenderBGROCVMosaics, RenderBGR,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(PrimsTestParam::MOSAICS),
                                Values(OCV)));

INSTANTIATE_TEST_CASE_P(RenderBGROCVImages, RenderBGR,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(PrimsTestParam::IMAGES),
                                Values(OCV)));

INSTANTIATE_TEST_CASE_P(RenderBGROCVPolygons, RenderBGR,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(PrimsTestParam::POLYGONS),
                                Values(OCV)));

INSTANTIATE_TEST_CASE_P(RenderBGROCVTexts, RenderBGR,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(PrimsTestParam::TEXTS),
                                Values(OCV)));

}  // namespace
