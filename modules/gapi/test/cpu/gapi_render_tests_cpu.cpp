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
                                Values(Points{Point(5,30), Point(40, 70), Point(75, 110)}),
/* Font face  */                Values(FONT_HERSHEY_SIMPLEX),
/* Font scale */                Values(2),
/* Color      */                Values(cv::Scalar(255, 0, 0), cv::Scalar(0, 0, 255)),
/* Thickness  */                Values(1, 0),
                                Values(LINE_4, LINE_8, LINE_AA),
/* Bottom left origin */        testing::Bool(),
/* NV12 format        */        testing::Bool()));
}
