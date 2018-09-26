// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../common/gapi_imgproc_perf_tests.hpp"

namespace opencv_test
{

  INSTANTIATE_TEST_CASE_P(SepFilterPerfTestCPU_8U, SepFilterPerfTest,
                          Combine(Values(CV_8UC1, CV_8UC3),
                                  Values(3),
                                  Values(szVGA, sz720p, sz1080p),
                                  Values(-1, CV_16S, CV_32F)));

  INSTANTIATE_TEST_CASE_P(SepFilterPerfTestCPU_other, SepFilterPerfTest,
                          Combine(Values(CV_16UC1, CV_16SC1, CV_32FC1),
                                  Values(3),
                                  Values(szVGA, sz720p, sz1080p),
                                  Values(-1, CV_32F)));

  INSTANTIATE_TEST_CASE_P(Filter2DPerfTestCPU, Filter2DPerfTest,
                          Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                  Values(3, 4, 5, 7),
                                  Values(szVGA, sz720p, sz1080p),
                                  Values(cv::BORDER_DEFAULT),
                                  Values(-1, CV_32F)));

  INSTANTIATE_TEST_CASE_P(BoxFilterPerfTestCPU, BoxFilterPerfTest,
                          Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                  Values(3,5),
                                  Values(szVGA, sz720p, sz1080p),
                                  Values(cv::BORDER_DEFAULT),
                                  Values(-1, CV_32F),
                                  Values(0.0)));

  INSTANTIATE_TEST_CASE_P(BlurPerfTestCPU, BlurPerfTest,
                          Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                  Values(3, 5),
                                  Values(szVGA, sz720p, sz1080p),
                                  Values(cv::BORDER_DEFAULT),
                                  Values(0.0)));

  INSTANTIATE_TEST_CASE_P(GaussianBlurPerfTestCPU, GaussianBlurPerfTest,
                          Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                  Values(3, 5),
                                  Values(szVGA, sz720p, sz1080p)));

   INSTANTIATE_TEST_CASE_P(MedianBlurPerfTestCPU, MedianBlurPerfTest,
                           Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                   Values(3, 5),
                                   Values(szVGA, sz720p, sz1080p)));

  INSTANTIATE_TEST_CASE_P(ErodePerfTestCPU, ErodePerfTest,
                          Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                  Values(3, 5),
                                  Values(szVGA, sz720p, sz1080p),
                                  Values(cv::MorphShapes::MORPH_RECT,
                                         cv::MorphShapes::MORPH_CROSS,
                                         cv::MorphShapes::MORPH_ELLIPSE)));

   INSTANTIATE_TEST_CASE_P(Erode3x3PerfTestCPU, Erode3x3PerfTest,
                           Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                   Values(szVGA, sz720p, sz1080p),
                                   Values(1,2,4)));

   INSTANTIATE_TEST_CASE_P(DilatePerfTestCPU, DilatePerfTest,
                           Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                   Values(3, 5),
                                   Values(szVGA, sz720p, sz1080p),
                                   Values(cv::MorphShapes::MORPH_RECT,
                                          cv::MorphShapes::MORPH_CROSS,
                                          cv::MorphShapes::MORPH_ELLIPSE)));

    INSTANTIATE_TEST_CASE_P(Dilate3x3PerfTestCPU, Dilate3x3PerfTest,
                            Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                    Values(szVGA, sz720p, sz1080p),
                                    Values(1,2,4)));

    INSTANTIATE_TEST_CASE_P(SobelPerfTestCPU, SobelPerfTest,
                            Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                    Values(3, 5),
                                    Values(szVGA, sz720p, sz1080p),
                                    Values(-1, CV_32F),
                                    Values(0, 1),
                                    Values(1, 2)));

    INSTANTIATE_TEST_CASE_P(CannyPerfTestCPU, CannyPerfTest,
                            Combine(Values(CV_8UC1, CV_8UC3),
                                    Values(szVGA, sz720p, sz1080p),
                                    Values(3.0, 120.0),
                                    Values(125.0, 240.0),
                                    Values(3, 5),
                                    Values(true, false)));

    INSTANTIATE_TEST_CASE_P(EqHistPerfTestCPU, EqHistPerfTest,  Values(szVGA, sz720p, sz1080p));

    INSTANTIATE_TEST_CASE_P(RGB2GrayPerfTestCPU, RGB2GrayPerfTest,  Values(szVGA, sz720p, sz1080p));

    INSTANTIATE_TEST_CASE_P(BGR2GrayPerfTestCPU, BGR2GrayPerfTest,  Values(szVGA, sz720p, sz1080p));

    INSTANTIATE_TEST_CASE_P(RGB2YUVPerfTestCPU, RGB2YUVPerfTest,  Values(szVGA, sz720p, sz1080p));

    INSTANTIATE_TEST_CASE_P(YUV2RGBPerfTestCPU, YUV2RGBPerfTest,  Values(szVGA, sz720p, sz1080p));

    INSTANTIATE_TEST_CASE_P(RGB2LabPerfTestCPU, RGB2LabPerfTest,  Values(szVGA, sz720p, sz1080p));

    INSTANTIATE_TEST_CASE_P(BGR2LUVPerfTestCPU, BGR2LUVPerfTest,  Values(szVGA, sz720p, sz1080p));

    INSTANTIATE_TEST_CASE_P(LUV2BGRPerfTestCPU, LUV2BGRPerfTest,  Values(szVGA, sz720p, sz1080p));

    INSTANTIATE_TEST_CASE_P(BGR2YUVPerfTestCPU, BGR2YUVPerfTest, Values(szVGA, sz720p, sz1080p));

    INSTANTIATE_TEST_CASE_P(YUV2BGRPerfTestCPU, YUV2BGRPerfTest, Values(szVGA, sz720p, sz1080p));

}
