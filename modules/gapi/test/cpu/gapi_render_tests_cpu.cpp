// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../test_precomp.hpp"
#include "../common/gapi_render_tests.hpp"

namespace opencv_test
{

INSTANTIATE_TEST_CASE_P(RenderTextTestCPU, RenderTextTest,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values("text"),
                                Values(Points{Point(30,150)}),
                                Values(5),
                                Values(5.5),
                                Values(cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0)),
                                Values(5),
                                Values(5),
                                testing::Bool()));
}
