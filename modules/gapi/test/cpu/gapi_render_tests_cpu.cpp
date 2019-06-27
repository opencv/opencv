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
                                Values(Points{Point(5, 30), Point(40, 70), Point(-1, -1)}),
/* Font face          */        Values(FONT_HERSHEY_SIMPLEX),
/* Font scale         */        Values(2),
/* Color              */        Values(cv::Scalar(255, 0, 0)),
/* Thickness          */        Values(1),
/* Line type          */        Values(LINE_8),
/* Bottom left origin */        testing::Bool(),
/* NV12 format or not */        testing::Bool()));

INSTANTIATE_TEST_CASE_P(RenderRectTestCPU, RenderRectTest,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(Rects{Rect(5, 30, 40, 50),
                                             Rect(40, 70, 40, 50),
/* Edge case, rectangle will not be drawn */ Rect(75, 110, -40, 50),
/* Edge case, rectangle will not be drawn */ Rect(70, 100, 0, 50)}),
/* Color              */        Values(cv::Scalar(255, 0, 0)),
/* Thickness          */        Values(1),
/* Line type          */        Values(LINE_8),
/* Shift              */        Values(0),
/* NV12 format or not */        testing::Bool()));

INSTANTIATE_TEST_CASE_P(RenderCircleTestCPU, RenderCircleTest,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(Points{Point(5, 30), Point(40, 70), Point(75, 110)}),
/* Radius             */        Values(5),
/* Color              */        Values(cv::Scalar(255, 0, 0)),
/* Thickness          */        Values(1),
/* Line type          */        Values(LINE_8),
/* Shift              */        Values(0),
/* NV12 format or not */        testing::Bool()));

INSTANTIATE_TEST_CASE_P(RenderLineTestCPU, RenderLineTest,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(VecOfPairOfPoints{ {Point(5, 30)  , Point(5, 40)   },
                                                          {Point(40, 70) , Point(50, 70)  },
                                                          {Point(75, 110), Point(100, 115)} }),
/* Color              */        Values(cv::Scalar(255, 0, 0)),
/* Thickness          */        Values(1),
/* Line type          */        Values(LINE_8),
/* Shift              */        Values(0),
/* NV12 format or not */        testing::Bool()));
}
