// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2019 Intel Corporation


#include "../test_precomp.hpp"

#include "../common/gapi_imgproc_tests.hpp"
#include <opencv2/gapi/cpu/imgproc.hpp>

namespace
{
#define IMGPROC_CPU [] () { return cv::compile_args(cv::gapi::imgproc::cpu::kernels()); }
}  // anonymous namespace

namespace opencv_test
{

INSTANTIATE_TEST_CASE_P(Filter2DTestCPU, Filter2DTest,
                        Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1280, 720),
                                        cv::Size(640, 480),
                                        cv::Size(128, 128)),
                                Values(-1, CV_32F),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj()),
                                Values(cv::Size(3, 3),
                                       cv::Size(4, 4),
                                       cv::Size(5, 5),
                                       cv::Size(7, 7)),
                                Values(cv::BORDER_DEFAULT)));

INSTANTIATE_TEST_CASE_P(BoxFilterTestCPU, BoxFilterTest,
                        Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(-1, CV_32F),
                                Values(IMGPROC_CPU),
                                Values(AbsTolerance(0).to_compare_obj()),
                                Values(3,5),
                                Values(cv::BORDER_DEFAULT)));

INSTANTIATE_TEST_CASE_P(SepFilterTestCPU_8U, SepFilterTest,
                        Combine(Values(CV_8UC1, CV_8UC3),
                                Values(cv::Size(1280, 720),
                                        cv::Size(640, 480)),
                                Values(-1, CV_16S, CV_32F),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj()),
                                Values(3)));

INSTANTIATE_TEST_CASE_P(SepFilterTestCPU_other, SepFilterTest,
                        Combine(Values(CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(-1, CV_32F),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj()),
                                Values(3)));

INSTANTIATE_TEST_CASE_P(BlurTestCPU, BlurTest,
                        Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(-1),
                                Values(IMGPROC_CPU),
                                Values(AbsTolerance(0.0).to_compare_obj()),
                                Values(3,5),
                                Values(cv::BORDER_DEFAULT)));

INSTANTIATE_TEST_CASE_P(gaussBlurTestCPU, GaussianBlurTest,
                        Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(-1),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj()),
                                Values(3, 5)));

INSTANTIATE_TEST_CASE_P(MedianBlurTestCPU, MedianBlurTest,
                        Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(-1),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj()),
                                Values(3, 5)));

INSTANTIATE_TEST_CASE_P(ErodeTestCPU, ErodeTest,
                        Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(-1),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj()),
                                Values(3, 5),
                                Values(cv::MorphShapes::MORPH_RECT,
                                       cv::MorphShapes::MORPH_CROSS,
                                       cv::MorphShapes::MORPH_ELLIPSE)));

INSTANTIATE_TEST_CASE_P(Erode3x3TestCPU, Erode3x3Test,
                        Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(-1),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj()),
                                Values(1,2,4)));

INSTANTIATE_TEST_CASE_P(DilateTestCPU, DilateTest,
                        Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(-1),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj()),
                                Values(3, 5),
                                Values(cv::MorphShapes::MORPH_RECT,
                                       cv::MorphShapes::MORPH_CROSS,
                                       cv::MorphShapes::MORPH_ELLIPSE)));

INSTANTIATE_TEST_CASE_P(Dilate3x3TestCPU, Dilate3x3Test,
                        Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(-1),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj()),
                                Values(1,2,4)));

INSTANTIATE_TEST_CASE_P(SobelTestCPU, SobelTest,
                        Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(-1, CV_16S, CV_32F),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj()),
                                Values(3, 5),
                                Values(0, 1),
                                Values(1, 2)));

INSTANTIATE_TEST_CASE_P(SobelTestCPU32F, SobelTest,
                        Combine(Values(CV_32FC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(CV_32F),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj()),
                                Values(3, 5),
                                Values(0, 1),
                                Values(1, 2)));

INSTANTIATE_TEST_CASE_P(SobelXYTestCPU, SobelXYTest,
                        Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(-1, CV_16S, CV_32F),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj()),
                                Values(3, 5),
                                Values(1, 2),
                                Values(BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT),
                                Values(0, 1, 255)));

INSTANTIATE_TEST_CASE_P(SobelXYTestCPU32F, SobelXYTest,
                        Combine(Values(CV_32FC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(CV_32F),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj()),
                                Values(3, 5),
                                Values(1, 2),
                                Values(BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT),
                                Values(0, 1, 255)));

INSTANTIATE_TEST_CASE_P(EqHistTestCPU, EqHistTest,
                        Combine(Values(CV_8UC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(CV_8UC1),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(CannyTestCPU, CannyTest,
                        Combine(Values(CV_8UC1, CV_8UC3),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(CV_8UC1),
                                Values(IMGPROC_CPU),
                                Values(AbsSimilarPoints(0, 0.05).to_compare_obj()),
                                Values(3.0, 120.0),
                                Values(125.0, 240.0),
                                Values(3, 5),
                                testing::Bool()));

INSTANTIATE_TEST_CASE_P(RGB2GrayTestCPU, RGB2GrayTest,
                        Combine(Values(CV_8UC3),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(CV_8UC1),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(BGR2GrayTestCPU, BGR2GrayTest,
                        Combine(Values(CV_8UC3),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(CV_8UC1),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(RGB2YUVTestCPU, RGB2YUVTest,
                        Combine(Values(CV_8UC3),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(CV_8UC3),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(YUV2RGBTestCPU, YUV2RGBTest,
                        Combine(Values(CV_8UC3),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(CV_8UC3),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(NV12toRGBTestCPU, NV12toRGBTest,
                        Combine(Values(CV_8UC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(CV_8UC3),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(NV12toBGRTestCPU, NV12toBGRTest,
                        Combine(Values(CV_8UC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(CV_8UC3),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(NV12toGrayTestCPU, NV12toGrayTest,
                        Combine(Values(CV_8UC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(CV_8UC1),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(NV12toRGBpTestCPU, NV12toRGBpTest,
                        Combine(Values(CV_8UC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(CV_8UC1),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(NV12toBGRpTestCPU, NV12toBGRpTest,
                        Combine(Values(CV_8UC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(CV_8UC3),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(RGB2LabTestCPU, RGB2LabTest,
                        Combine(Values(CV_8UC3),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(CV_8UC3),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(BGR2LUVTestCPU, BGR2LUVTest,
                        Combine(Values(CV_8UC3),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(CV_8UC3),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(LUV2BGRTestCPU, LUV2BGRTest,
                        Combine(Values(CV_8UC3),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(CV_8UC3),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(BGR2YUVTestCPU, BGR2YUVTest,
                        Combine(Values(CV_8UC3),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(CV_8UC3),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(YUV2BGRTestCPU, YUV2BGRTest,
                        Combine(Values(CV_8UC3),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(CV_8UC3),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(RGB2HSVTestCPU, RGB2HSVTest,
                        Combine(Values(CV_8UC3),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(CV_8UC3),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(BayerGR2RGBTestCPU, BayerGR2RGBTest,
                        Combine(Values(CV_8UC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(CV_8UC3),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(RGB2YUV422TestCPU, RGB2YUV422Test,
                        Combine(Values(CV_8UC3),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(CV_8UC2),
                                Values(IMGPROC_CPU),
                                Values(AbsTolerance(1).to_compare_obj())));
} // opencv_test
