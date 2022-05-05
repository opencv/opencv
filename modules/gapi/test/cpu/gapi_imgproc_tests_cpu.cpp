// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#include "../test_precomp.hpp"

#include "../common/gapi_imgproc_tests.hpp"
#include <opencv2/gapi/cpu/imgproc.hpp>

namespace
{
#define IMGPROC_CPU [] () { return cv::compile_args(cv::gapi::use_only{cv::gapi::imgproc::cpu::kernels()}); }
    const std::vector <cv::Size> in_sizes{ cv::Size(1280, 720), cv::Size(128, 128) };
}  // anonymous namespace

namespace opencv_test
{

INSTANTIATE_TEST_CASE_P(ResizeTestCPU, ResizeTest,
                        Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                ValuesIn(in_sizes),
                                Values(-1),
                                Values(IMGPROC_CPU),
                                Values(AbsSimilarPoints(2, 0.05).to_compare_obj()),
                                Values(cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_AREA),
                                Values(cv::Size(64,64),
                                       cv::Size(30,30))));

INSTANTIATE_TEST_CASE_P(ResizePTestCPU, ResizePTest,
                        Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                ValuesIn(in_sizes),
                                Values(-1),
                                Values(IMGPROC_CPU),
                                Values(AbsSimilarPoints(2, 0.05).to_compare_obj()),
                                Values(cv::INTER_LINEAR),
                                Values(cv::Size(64,64),
                                       cv::Size(30,30))));

INSTANTIATE_TEST_CASE_P(ResizeTestCPU, ResizeTestFxFy,
                        Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                ValuesIn(in_sizes),
                                Values(-1),
                                Values(IMGPROC_CPU),
                                Values(AbsSimilarPoints(2, 0.05).to_compare_obj()),
                                Values(cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_AREA),
                                Values(0.5, 0.1),
                                Values(0.5, 0.1)));

INSTANTIATE_TEST_CASE_P(Filter2DTestCPU, Filter2DTest,
                        Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                ValuesIn(in_sizes),
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
                                Values(cv::Size(1280, 720)),
                                Values(-1, CV_32F),
                                Values(IMGPROC_CPU),
                                Values(AbsTolerance(0).to_compare_obj()),
                                Values(3,5),
                                Values(cv::BORDER_DEFAULT)));

INSTANTIATE_TEST_CASE_P(SepFilterTestCPU_8U, SepFilterTest,
                        Combine(Values(CV_8UC1, CV_8UC3),
                                Values(cv::Size(1280, 720)),
                                Values(-1, CV_16S, CV_32F),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj()),
                                Values(3)));

INSTANTIATE_TEST_CASE_P(SepFilterTestCPU_other, SepFilterTest,
                        Combine(Values(CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1280, 720)),
                                Values(-1, CV_32F),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj()),
                                Values(3)));

INSTANTIATE_TEST_CASE_P(BlurTestCPU, BlurTest,
                        Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1280, 720)),
                                Values(-1),
                                Values(IMGPROC_CPU),
                                Values(AbsTolerance(0.0).to_compare_obj()),
                                Values(3,5),
                                Values(cv::BORDER_DEFAULT)));

INSTANTIATE_TEST_CASE_P(gaussBlurTestCPU, GaussianBlurTest,
                        Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1280, 720)),
                                Values(-1),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj()),
                                Values(3, 5)));

INSTANTIATE_TEST_CASE_P(MedianBlurTestCPU, MedianBlurTest,
                        Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1280, 720)),
                                Values(-1),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj()),
                                Values(3, 5)));

INSTANTIATE_TEST_CASE_P(ErodeTestCPU, ErodeTest,
                        Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1280, 720)),
                                Values(-1),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj()),
                                Values(3, 5),
                                Values(cv::MorphShapes::MORPH_RECT,
                                       cv::MorphShapes::MORPH_CROSS,
                                       cv::MorphShapes::MORPH_ELLIPSE)));

INSTANTIATE_TEST_CASE_P(Erode3x3TestCPU, Erode3x3Test,
                        Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1280, 720)),
                                Values(-1),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj()),
                                Values(1,2,4)));

INSTANTIATE_TEST_CASE_P(DilateTestCPU, DilateTest,
                        Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1280, 720)),
                                Values(-1),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj()),
                                Values(3, 5),
                                Values(cv::MorphShapes::MORPH_RECT,
                                       cv::MorphShapes::MORPH_CROSS,
                                       cv::MorphShapes::MORPH_ELLIPSE)));

INSTANTIATE_TEST_CASE_P(Dilate3x3TestCPU, Dilate3x3Test,
                        Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1280, 720)),
                                Values(-1),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj()),
                                Values(1,2,4)));

INSTANTIATE_TEST_CASE_P(MorphologyExTestCPU, MorphologyExTest,
                        Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1280, 720)),
                                Values(-1),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj()),
                                Values(cv::MorphTypes::MORPH_ERODE,
                                       cv::MorphTypes::MORPH_DILATE,
                                       cv::MorphTypes::MORPH_OPEN,
                                       cv::MorphTypes::MORPH_CLOSE,
                                       cv::MorphTypes::MORPH_GRADIENT,
                                       cv::MorphTypes::MORPH_TOPHAT,
                                       cv::MorphTypes::MORPH_BLACKHAT)));

INSTANTIATE_TEST_CASE_P(MorphologyExHitMissTestCPU, MorphologyExTest,
                        Combine(Values(CV_8UC1),
                                Values(cv::Size(1280, 720)),
                                Values(-1),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj()),
                                Values(cv::MorphTypes::MORPH_HITMISS)));

INSTANTIATE_TEST_CASE_P(SobelTestCPU, SobelTest,
                        Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720)),
                                Values(-1, CV_16S, CV_32F),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj()),
                                Values(3, 5),
                                Values(0, 1),
                                Values(1, 2)));

INSTANTIATE_TEST_CASE_P(SobelTestCPU32F, SobelTest,
                        Combine(Values(CV_32FC1),
                                Values(cv::Size(1280, 720)),
                                Values(CV_32F),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj()),
                                Values(3, 5),
                                Values(0, 1),
                                Values(1, 2)));

INSTANTIATE_TEST_CASE_P(SobelXYTestCPU, SobelXYTest,
                        Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720)),
                                Values(-1, CV_16S, CV_32F),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj()),
                                Values(3, 5),
                                Values(1, 2),
                                Values(BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT),
                                Values(0, 1, 255)));

INSTANTIATE_TEST_CASE_P(SobelXYTestCPU32F, SobelXYTest,
                        Combine(Values(CV_32FC1),
                                Values(cv::Size(1280, 720)),
                                Values(CV_32F),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj()),
                                Values(3, 5),
                                Values(1, 2),
                                Values(BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT),
                                Values(0, 1, 255)));

INSTANTIATE_TEST_CASE_P(LaplacianTestCPU, LaplacianTest,
                        Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720)),
                                Values(-1),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj()),
                                Values(1, 3),
                                Values(0.2, 1.0),
                                Values(BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT)));

INSTANTIATE_TEST_CASE_P(BilateralFilterTestCPU, BilateralFilterTest,
                        Combine(Values(CV_32FC1, CV_32FC3, CV_8UC1, CV_8UC3),
                                Values(cv::Size(1280, 720)),
                                Values(-1),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj()),
                                Values(3, 5),
                                Values(20),
                                Values(10),
                                Values(BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT)));

INSTANTIATE_TEST_CASE_P(EqHistTestCPU, EqHistTest,
                        Combine(Values(CV_8UC1),
                                Values(cv::Size(1280, 720)),
                                Values(CV_8UC1),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(CannyTestCPU, CannyTest,
                        Combine(Values(CV_8UC1, CV_8UC3),
                                Values(cv::Size(1280, 720)),
                                Values(CV_8UC1),
                                Values(IMGPROC_CPU),
                                Values(AbsSimilarPoints(0, 0.05).to_compare_obj()),
                                Values(3.0, 120.0),
                                Values(125.0, 240.0),
                                Values(3, 5),
                                testing::Bool()));

INSTANTIATE_TEST_CASE_P(GoodFeaturesTestCPU, GoodFeaturesTest,
                        Combine(Values(IMGPROC_CPU),
                                Values(AbsExactVector<cv::Point2f>().to_compare_obj()),
                                Values("cv/shared/fruits.png"),
                                Values(CV_32FC1, CV_8UC1),
                                Values(50, 100),
                                Values(0.01),
                                Values(10.0),
                                Values(3),
                                testing::Bool()));

INSTANTIATE_TEST_CASE_P(GoodFeaturesInternalTestCPU, GoodFeaturesTest,
                        Combine(Values(IMGPROC_CPU),
                                Values(AbsExactVector<cv::Point2f>().to_compare_obj()),
                                Values("cv/cascadeandhog/images/audrybt1.png"),
                                Values(CV_32FC1, CV_8UC1),
                                Values(100),
                                Values(0.0000001),
                                Values(5.0),
                                Values(3),
                                Values(true)));

INSTANTIATE_TEST_CASE_P(FindContoursNoOffsetTestCPU, FindContoursNoOffsetTest,
                        Combine(Values(IMGPROC_CPU),
                                Values(cv::Size(1280, 720)),
                                Values(CV_8UC1),
                                Values(RETR_EXTERNAL),
                                Values(CHAIN_APPROX_NONE),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(FindContoursOffsetTestCPU, FindContoursOffsetTest,
                        Values(IMGPROC_CPU));

INSTANTIATE_TEST_CASE_P(FindContoursHNoOffsetTestCPU, FindContoursHNoOffsetTest,
                        Combine(Values(IMGPROC_CPU),
                                Values(cv::Size(1280, 720)),
                                Values(CV_8UC1),
                                Values(RETR_EXTERNAL, RETR_LIST, RETR_CCOMP, RETR_TREE),
                                Values(CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE,
                                       CHAIN_APPROX_TC89_L1, CHAIN_APPROX_TC89_KCOS),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(FindContoursHNoOffset32STestCPU, FindContoursHNoOffsetTest,
                        Combine(Values(IMGPROC_CPU),
                                Values(cv::Size(1280, 720)),
                                Values(CV_32SC1),
                                Values(RETR_CCOMP, RETR_FLOODFILL),
                                Values(CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE,
                                       CHAIN_APPROX_TC89_L1, CHAIN_APPROX_TC89_KCOS),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(FindContoursHOffsetTestCPU, FindContoursHOffsetTest,
                        Values(IMGPROC_CPU));

INSTANTIATE_TEST_CASE_P(BoundingRectMatTestCPU, BoundingRectMatTest,
                        Combine(Values( CV_8UC1 ),
                                ValuesIn(in_sizes),
                                Values(-1),
                                Values(IMGPROC_CPU),
                                Values(IoUToleranceRect(0).to_compare_obj()),
                                Values(false)));

INSTANTIATE_TEST_CASE_P(BoundingRectMatVectorTestCPU, BoundingRectMatTest,
                        Combine(Values(CV_32S, CV_32F),
                                Values(cv::Size(1280, 1),
                                       cv::Size(128, 1)),
                                Values(-1),
                                Values(IMGPROC_CPU),
                                Values(IoUToleranceRect(1e-5).to_compare_obj()),
                                Values(true)));

INSTANTIATE_TEST_CASE_P(BoundingRectVector32STestCPU, BoundingRectVector32STest,
                        Combine(Values(-1),
                                Values(cv::Size(1280, 1),
                                       cv::Size(128, 1)),
                                Values(-1),
                                Values(IMGPROC_CPU),
                                Values(IoUToleranceRect(0).to_compare_obj())));

 INSTANTIATE_TEST_CASE_P(BoundingRectVector32FTestCPU, BoundingRectVector32FTest,
                         Combine(Values(-1),
                                 Values(cv::Size(1280, 1),
                                        cv::Size(128, 1)),
                                 Values(-1),
                                 Values(IMGPROC_CPU),
                                 Values(IoUToleranceRect(1e-5).to_compare_obj())));

INSTANTIATE_TEST_CASE_P(FitLine2DMatVectorTestCPU, FitLine2DMatVectorTest,
                        Combine(Values(CV_8U, CV_8S, CV_16U, CV_16S,
                                       CV_32S, CV_32F, CV_64F),
                                Values(cv::Size(8, 0), cv::Size(1024, 0)),
                                Values(-1),
                                Values(IMGPROC_CPU),
                                Values(RelDiffToleranceVec<float, 4>(0.01).to_compare_obj()),
                                Values(DIST_L1, DIST_L2, DIST_L12, DIST_FAIR,
                                       DIST_WELSCH, DIST_HUBER)));

INSTANTIATE_TEST_CASE_P(FitLine2DVector32STestCPU, FitLine2DVector32STest,
                        Combine(Values(-1),
                                Values(cv::Size(8, 0)),
                                Values(-1),
                                Values(IMGPROC_CPU),
                                Values(RelDiffToleranceVec<float, 4>(0.01).to_compare_obj()),
                                Values(DIST_L1)));

INSTANTIATE_TEST_CASE_P(FitLine2DVector32FTestCPU, FitLine2DVector32FTest,
                        Combine(Values(-1),
                                Values(cv::Size(8, 0)),
                                Values(-1),
                                Values(IMGPROC_CPU),
                                Values(RelDiffToleranceVec<float, 4>(0.01).to_compare_obj()),
                                Values(DIST_L1)));

INSTANTIATE_TEST_CASE_P(FitLine2DVector64FTestCPU, FitLine2DVector64FTest,
                        Combine(Values(-1),
                                Values(cv::Size(8, 0)),
                                Values(-1),
                                Values(IMGPROC_CPU),
                                Values(RelDiffToleranceVec<float, 4>(0.01).to_compare_obj()),
                                Values(DIST_L1)));

INSTANTIATE_TEST_CASE_P(FitLine3DMatVectorTestCPU, FitLine3DMatVectorTest,
                        Combine(Values(CV_8UC1, CV_8SC1, CV_16UC1, CV_16SC1,
                                       CV_32SC1, CV_32FC1, CV_64FC1),
                                Values(cv::Size(8, 0), cv::Size(1024, 0)),
                                Values(-1),
                                Values(IMGPROC_CPU),
                                Values(RelDiffToleranceVec<float, 6>(0.01).to_compare_obj()),
                                Values(DIST_L1, DIST_L2, DIST_L12, DIST_FAIR,
                                       DIST_WELSCH, DIST_HUBER)));

INSTANTIATE_TEST_CASE_P(FitLine3DVector32STestCPU, FitLine3DVector32STest,
                        Combine(Values(-1),
                                Values(cv::Size(8, 0)),
                                Values(-1),
                                Values(IMGPROC_CPU),
                                Values(RelDiffToleranceVec<float, 6>(0.01).to_compare_obj()),
                                Values(DIST_L1)));

INSTANTIATE_TEST_CASE_P(FitLine3DVector32FTestCPU, FitLine3DVector32FTest,
                        Combine(Values(-1),
                                Values(cv::Size(8, 0)),
                                Values(-1),
                                Values(IMGPROC_CPU),
                                Values(RelDiffToleranceVec<float, 6>(0.01).to_compare_obj()),
                                Values(DIST_L1)));

INSTANTIATE_TEST_CASE_P(FitLine3DVector64FTestCPU, FitLine3DVector64FTest,
                        Combine(Values(-1),
                                Values(cv::Size(8, 0)),
                                Values(-1),
                                Values(IMGPROC_CPU),
                                Values(RelDiffToleranceVec<float, 6>(0.01).to_compare_obj()),
                                Values(DIST_L1)));

INSTANTIATE_TEST_CASE_P(BGR2RGBTestCPU, BGR2RGBTest,
                        Combine(Values(CV_8UC3),
                                Values(cv::Size(1280, 720)),
                                Values(CV_8UC3),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(RGB2GrayTestCPU, RGB2GrayTest,
                        Combine(Values(CV_8UC3),
                                Values(cv::Size(1280, 720)),
                                Values(CV_8UC1),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(BGR2GrayTestCPU, BGR2GrayTest,
                        Combine(Values(CV_8UC3),
                                Values(cv::Size(1280, 720)),
                                Values(CV_8UC1),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(RGB2YUVTestCPU, RGB2YUVTest,
                        Combine(Values(CV_8UC3),
                                Values(cv::Size(1280, 720)),
                                Values(CV_8UC3),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(YUV2RGBTestCPU, YUV2RGBTest,
                        Combine(Values(CV_8UC3),
                                Values(cv::Size(1280, 720)),
                                Values(CV_8UC3),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(BGR2I420TestCPU, BGR2I420Test,
                        Combine(Values(CV_8UC3),
                                Values(cv::Size(1280, 720)),
                                Values(CV_8UC1),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(RGB2I420TestCPU, RGB2I420Test,
                        Combine(Values(CV_8UC3),
                                Values(cv::Size(1280, 720)),
                                Values(CV_8UC1),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(I4202BGRTestCPU, I4202BGRTest,
                        Combine(Values(CV_8UC1),
                                Values(cv::Size(1280, 720)),
                                Values(CV_8UC3),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(I4202RGBTestCPU, I4202RGBTest,
                        Combine(Values(CV_8UC1),
                                Values(cv::Size(1280, 720)),
                                Values(CV_8UC3),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(NV12toRGBTestCPU, NV12toRGBTest,
                        Combine(Values(CV_8UC1),
                                Values(cv::Size(1280, 720)),
                                Values(CV_8UC3),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(NV12toBGRTestCPU, NV12toBGRTest,
                        Combine(Values(CV_8UC1),
                                Values(cv::Size(1280, 720)),
                                Values(CV_8UC3),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(NV12toGrayTestCPU, NV12toGrayTest,
                        Combine(Values(CV_8UC1),
                                Values(cv::Size(1280, 720)),
                                Values(CV_8UC1),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(NV12toRGBpTestCPU, NV12toRGBpTest,
                        Combine(Values(CV_8UC1),
                                Values(cv::Size(1280, 720)),
                                Values(CV_8UC1),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(NV12toBGRpTestCPU, NV12toBGRpTest,
                        Combine(Values(CV_8UC1),
                                Values(cv::Size(1280, 720)),
                                Values(CV_8UC3),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(RGB2LabTestCPU, RGB2LabTest,
                        Combine(Values(CV_8UC3),
                                Values(cv::Size(1280, 720)),
                                Values(CV_8UC3),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(BGR2LUVTestCPU, BGR2LUVTest,
                        Combine(Values(CV_8UC3),
                                Values(cv::Size(1280, 720)),
                                Values(CV_8UC3),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(LUV2BGRTestCPU, LUV2BGRTest,
                        Combine(Values(CV_8UC3),
                                Values(cv::Size(1280, 720)),
                                Values(CV_8UC3),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(BGR2YUVTestCPU, BGR2YUVTest,
                        Combine(Values(CV_8UC3),
                                Values(cv::Size(1280, 720)),
                                Values(CV_8UC3),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(YUV2BGRTestCPU, YUV2BGRTest,
                        Combine(Values(CV_8UC3),
                                Values(cv::Size(1280, 720)),
                                Values(CV_8UC3),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(RGB2HSVTestCPU, RGB2HSVTest,
                        Combine(Values(CV_8UC3),
                                Values(cv::Size(1280, 720)),
                                Values(CV_8UC3),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(BayerGR2RGBTestCPU, BayerGR2RGBTest,
                        Combine(Values(CV_8UC1),
                                Values(cv::Size(1280, 720)),
                                Values(CV_8UC3),
                                Values(IMGPROC_CPU),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(RGB2YUV422TestCPU, RGB2YUV422Test,
                        Combine(Values(CV_8UC3),
                                Values(cv::Size(1280, 720)),
                                Values(CV_8UC2),
                                Values(IMGPROC_CPU),
                                Values(AbsTolerance(1).to_compare_obj())));
} // opencv_test
