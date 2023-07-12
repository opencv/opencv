// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#include "../perf_precomp.hpp"
#include "../common/gapi_imgproc_perf_tests.hpp"
#include <opencv2/gapi/cpu/imgproc.hpp>

#define IMGPROC_CPU cv::gapi::imgproc::cpu::kernels()

namespace opencv_test
{

INSTANTIATE_TEST_CASE_P(SepFilterPerfTestCPU_8U, SepFilterPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
        Values(CV_8UC1, CV_8UC3),
        Values(3),
        Values(szVGA, sz720p, sz1080p),
        Values(-1, CV_16S, CV_32F),
        Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(SepFilterPerfTestCPU_other, SepFilterPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
        Values(CV_16UC1, CV_16SC1, CV_32FC1),
        Values(3),
        Values(szVGA, sz720p, sz1080p),
        Values(-1, CV_32F),
        Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(Filter2DPerfTestCPU, Filter2DPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(3, 4, 5, 7),
        Values(szVGA, sz720p, sz1080p),
        Values(cv::BORDER_DEFAULT),
        Values(-1, CV_32F),
        Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(BoxFilterPerfTestCPU, BoxFilterPerfTest,
    Combine(Values(AbsTolerance(0).to_compare_f()),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(3, 5),
        Values(szVGA, sz720p, sz1080p),
        Values(cv::BORDER_DEFAULT),
        Values(-1, CV_32F),
        Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(BlurPerfTestCPU, BlurPerfTest,
    Combine(Values(AbsTolerance(0).to_compare_f()),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(3, 5),
        Values(szVGA, sz720p, sz1080p),
        Values(cv::BORDER_DEFAULT),
        Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(GaussianBlurPerfTestCPU, GaussianBlurPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(3, 5),
        Values(szVGA, sz720p, sz1080p),
        Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(MedianBlurPerfTestCPU, MedianBlurPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(3, 5),
        Values(szVGA, sz720p, sz1080p),
        Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(ErodePerfTestCPU, ErodePerfTest,
    Combine(Values(AbsExact().to_compare_f()),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(3, 5),
        Values(szVGA, sz720p, sz1080p),
        Values(cv::MorphShapes::MORPH_RECT,
            cv::MorphShapes::MORPH_CROSS,
            cv::MorphShapes::MORPH_ELLIPSE),
        Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(Erode3x3PerfTestCPU, Erode3x3PerfTest,
    Combine(Values(AbsExact().to_compare_f()),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(szVGA, sz720p, sz1080p),
        Values(1, 2, 4),
        Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(DilatePerfTestCPU, DilatePerfTest,
    Combine(Values(AbsExact().to_compare_f()),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(3, 5),
        Values(szVGA, sz720p, sz1080p),
        Values(cv::MorphShapes::MORPH_RECT,
            cv::MorphShapes::MORPH_CROSS,
            cv::MorphShapes::MORPH_ELLIPSE),
        Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(Dilate3x3PerfTestCPU, Dilate3x3PerfTest,
    Combine(Values(AbsExact().to_compare_f()),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(szVGA, sz720p, sz1080p),
        Values(1, 2, 4),
        Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(MorphologyExPerfTestCPU, MorphologyExPerfTest,
                        Combine(Values(AbsExact().to_compare_f()),
                                Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(szVGA, sz720p, sz1080p),
                                Values(cv::MorphTypes::MORPH_ERODE,
                                       cv::MorphTypes::MORPH_DILATE,
                                       cv::MorphTypes::MORPH_OPEN,
                                       cv::MorphTypes::MORPH_CLOSE,
                                       cv::MorphTypes::MORPH_GRADIENT,
                                       cv::MorphTypes::MORPH_TOPHAT,
                                       cv::MorphTypes::MORPH_BLACKHAT),
                                Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(MorphologyExHitMissPerfTestCPU, MorphologyExPerfTest,
                        Combine(Values(AbsExact().to_compare_f()),
                                Values(CV_8UC1),
                                Values(szVGA, sz720p, sz1080p),
                                Values(cv::MorphTypes::MORPH_HITMISS),
                                Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(SobelPerfTestCPU, SobelPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1),
        Values(3, 5),
        Values(szVGA, sz720p, sz1080p),
        Values(-1, CV_16S, CV_32F),
        Values(0, 1),
        Values(1, 2),
        Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(SobelPerfTestCPU32F, SobelPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
        Values(CV_32FC1),
        Values(3, 5),
        Values(szVGA, sz720p, sz1080p),
        Values(CV_32F),
        Values(0, 1),
        Values(1, 2),
        Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(LaplacianPerfTestCPU, LaplacianPerfTest,
                        Combine(Values(AbsExact().to_compare_f()),
                                Values(CV_8UC1, CV_8UC3),
                                Values(3),
                                Values(szVGA, sz720p, sz1080p),
                                Values(-1),
                                Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(BilateralFilterPerfTestCPU, BilateralFilterPerfTest,
                        Combine(Values(AbsExact().to_compare_f()),
                                Values(CV_32FC1, CV_32FC3),
                                Values(-1),
                                Values(szVGA, sz720p, sz1080p),
                                Values(3),
                                Values(20),
                                Values(10),
                                Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(CannyPerfTestCPU, CannyPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
        Values(CV_8UC1, CV_8UC3),
        Values(szVGA, sz720p, sz1080p),
        Values(3.0, 120.0),
        Values(125.0, 240.0),
        Values(3, 5),
        Values(true, false),
        Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(GoodFeaturesPerfTestCPU, GoodFeaturesPerfTest,
    Combine(Values(AbsExactVector<cv::Point2f>().to_compare_f()),
            Values("cv/shared/pic5.png", "stitching/a1.png"),
            Values(CV_32FC1, CV_8UC1),
            Values(100, 500),
            Values(0.1, 0.01),
            Values(1.0),
            Values(3, 5),
            Values(true, false),
            Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(GoodFeaturesInternalPerfTestCPU, GoodFeaturesPerfTest,
    Combine(Values(AbsExactVector<cv::Point2f>().to_compare_f()),
            Values("cv/cascadeandhog/images/audrybt1.png"),
            Values(CV_32FC1, CV_8UC1),
            Values(100),
            Values(0.0000001),
            Values(5.0),
            Values(3),
            Values(true),
            Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(FindContoursPerfTestCPU, FindContoursPerfTest,
                        Combine(Values(AbsExact().to_compare_obj()),
                                Values(CV_8UC1),
                                Values(szVGA, sz720p, sz1080p),
                                Values(RETR_EXTERNAL, RETR_LIST, RETR_CCOMP, RETR_TREE),
                                Values(CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE,
                                       CHAIN_APPROX_TC89_L1, CHAIN_APPROX_TC89_KCOS),
                                Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(FindContours32SPerfTestCPU, FindContoursPerfTest,
                        Combine(Values(AbsExact().to_compare_obj()),
                                Values(CV_32SC1),
                                Values(szVGA, sz720p, sz1080p),
                                Values(RETR_CCOMP, RETR_FLOODFILL),
                                Values(CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE,
                                       CHAIN_APPROX_TC89_L1, CHAIN_APPROX_TC89_KCOS),
                                Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(FindContoursHPerfTestCPU, FindContoursHPerfTest,
                        Combine(Values(AbsExact().to_compare_obj()),
                                Values(CV_8UC1),
                                Values(szVGA, sz720p, sz1080p),
                                Values(RETR_EXTERNAL, RETR_LIST, RETR_CCOMP, RETR_TREE),
                                Values(CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE,
                                       CHAIN_APPROX_TC89_L1, CHAIN_APPROX_TC89_KCOS),
                                Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(FindContoursH32SPerfTestCPU, FindContoursHPerfTest,
                        Combine(Values(AbsExact().to_compare_obj()),
                                Values(CV_32SC1),
                                Values(szVGA, sz720p, sz1080p),
                                Values(RETR_CCOMP, RETR_FLOODFILL),
                                Values(CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE,
                                       CHAIN_APPROX_TC89_L1, CHAIN_APPROX_TC89_KCOS),
                                Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(BoundingRectMatPerfTestCPU, BoundingRectMatPerfTest,
                        Combine(Values(IoUToleranceRect(0).to_compare_obj()),
                                Values(CV_8UC1),
                                Values(szVGA, sz720p, sz1080p),
                                Values(false),
                                Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(BoundingRectMatVectorPerfTestCPU, BoundingRectMatPerfTest,
                        Combine(Values(IoUToleranceRect(1e-5).to_compare_obj()),
                                Values(CV_32S, CV_32F),
                                Values(szVGA, sz720p, sz1080p),
                                Values(true),
                                Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(BoundingRectVector32SPerfTestCPU, BoundingRectVector32SPerfTest,
                        Combine(Values(IoUToleranceRect(0).to_compare_obj()),
                                Values(szVGA, sz720p, sz1080p),
                                Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(BoundingRectVector32FPerfTestCPU, BoundingRectVector32FPerfTest,
                        Combine(Values(IoUToleranceRect(1e-5).to_compare_obj()),
                                Values(szVGA, sz720p, sz1080p),
                                Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(FitLine2DMatVectorPerfTestCPU, FitLine2DMatVectorPerfTest,
                        Combine(Values(RelDiffToleranceVec<float, 4>(0.01).to_compare_obj()),
                                Values(CV_8U, CV_8S, CV_16U, CV_16S,
                                       CV_32S, CV_32F, CV_64F),
                                Values(cv::Size(8, 0), cv::Size(1024, 0)),
                                Values(DIST_L1, DIST_L2, DIST_L12, DIST_FAIR,
                                       DIST_WELSCH, DIST_HUBER),
                                Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(FitLine2DVector32SPerfTestCPU, FitLine2DVector32SPerfTest,
                        Combine(Values(RelDiffToleranceVec<float, 4>(0.01).to_compare_obj()),
                                Values(cv::Size(8, 0), cv::Size(1024, 0)),
                                Values(DIST_L1, DIST_L2, DIST_L12, DIST_FAIR,
                                       DIST_WELSCH, DIST_HUBER),
                                Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(FitLine2DVector32FPerfTestCPU, FitLine2DVector32FPerfTest,
                        Combine(Values(RelDiffToleranceVec<float, 4>(0.01).to_compare_obj()),
                                Values(cv::Size(8, 0), cv::Size(1024, 0)),
                                Values(DIST_L1, DIST_L2, DIST_L12, DIST_FAIR,
                                       DIST_WELSCH, DIST_HUBER),
                                Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(FitLine2DVector64FPerfTestCPU, FitLine2DVector64FPerfTest,
                        Combine(Values(RelDiffToleranceVec<float, 4>(0.01).to_compare_obj()),
                                Values(cv::Size(8, 0), cv::Size(1024, 0)),
                                Values(DIST_L1, DIST_L2, DIST_L12, DIST_FAIR,
                                       DIST_WELSCH, DIST_HUBER),
                                Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(FitLine3DMatVectorPerfTestCPU, FitLine3DMatVectorPerfTest,
                        Combine(Values(RelDiffToleranceVec<float, 6>(0.01).to_compare_obj()),
                                Values(CV_8U, CV_8S, CV_16U, CV_16S,
                                       CV_32S, CV_32F, CV_64F),
                                Values(cv::Size(8, 0), cv::Size(1024, 0)),
                                Values(DIST_L1, DIST_L2, DIST_L12, DIST_FAIR,
                                       DIST_WELSCH, DIST_HUBER),
                                Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(FitLine3DVector32SPerfTestCPU, FitLine3DVector32SPerfTest,
                        Combine(Values(RelDiffToleranceVec<float, 6>(0.01).to_compare_obj()),
                                Values(cv::Size(8, 0), cv::Size(1024, 0)),
                                Values(DIST_L1, DIST_L2, DIST_L12, DIST_FAIR,
                                       DIST_WELSCH, DIST_HUBER),
                                Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(FitLine3DVector32FPerfTestCPU, FitLine3DVector32FPerfTest,
                        Combine(Values(RelDiffToleranceVec<float, 6>(0.01).to_compare_obj()),
                                Values(cv::Size(8, 0), cv::Size(1024, 0)),
                                Values(DIST_L1, DIST_L2, DIST_L12, DIST_FAIR,
                                       DIST_WELSCH, DIST_HUBER),
                                Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(FitLine3DVector64FPerfTestCPU, FitLine3DVector64FPerfTest,
                        Combine(Values(RelDiffToleranceVec<float, 6>(0.01).to_compare_obj()),
                                Values(cv::Size(8, 0), cv::Size(1024, 0)),
                                Values(DIST_L1, DIST_L2, DIST_L12, DIST_FAIR,
                                       DIST_WELSCH, DIST_HUBER),
                                Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(EqHistPerfTestCPU, EqHistPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
        Values(szVGA, sz720p, sz1080p),
        Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(BGR2RGBPerfTestCPU, BGR2RGBPerfTest,
                        Combine(Values(AbsExact().to_compare_f()),
                                Values(szVGA, sz720p, sz1080p),
                                Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(RGB2GrayPerfTestCPU, RGB2GrayPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
        Values(szVGA, sz720p, sz1080p),
        Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(BGR2GrayPerfTestCPU, BGR2GrayPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
        Values(szVGA, sz720p, sz1080p),
        Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(RGB2YUVPerfTestCPU, RGB2YUVPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
        Values(szVGA, sz720p, sz1080p),
        Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(YUV2RGBPerfTestCPU, YUV2RGBPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
        Values(szVGA, sz720p, sz1080p),
        Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(BGR2I420PerfTestCPU, BGR2I420PerfTest,
                        Combine(Values(AbsExact().to_compare_f()),
                                Values(szVGA, sz720p, sz1080p),
                                Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(RGB2I420PerfTestCPU, RGB2I420PerfTest,
                        Combine(Values(AbsExact().to_compare_f()),
                                Values(szVGA, sz720p, sz1080p),
                                Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(I4202BGRPerfTestCPU, I4202BGRPerfTest,
                        Combine(Values(AbsExact().to_compare_f()),
                                Values(szVGA, sz720p, sz1080p),
                                Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(I4202RGBPerfTestCPU, I4202RGBPerfTest,
                        Combine(Values(AbsExact().to_compare_f()),
                                Values(szVGA, sz720p, sz1080p),
                                Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(RGB2LabPerfTestCPU, RGB2LabPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
        Values(szVGA, sz720p, sz1080p),
        Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(BGR2LUVPerfTestCPU, BGR2LUVPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
        Values(szVGA, sz720p, sz1080p),
        Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(LUV2BGRPerfTestCPU, LUV2BGRPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
        Values(szVGA, sz720p, sz1080p),
        Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(BGR2YUVPerfTestCPU, BGR2YUVPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
        Values(szVGA, sz720p, sz1080p),
        Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(YUV2BGRPerfTestCPU, YUV2BGRPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
        Values(szVGA, sz720p, sz1080p),
        Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(RGB2HSVPerfTestCPU, RGB2HSVPerfTest,
        Combine(Values(AbsExact().to_compare_f()),
            Values(szVGA, sz720p, sz1080p),
            Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(BayerGR2RGBPerfTestCPU, BayerGR2RGBPerfTest,
        Combine(Values(AbsExact().to_compare_f()),
            Values(szVGA, sz720p, sz1080p),
            Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(RGB2YUV422PerfTestCPU, RGB2YUV422PerfTest,
        Combine(Values(ToleranceColor(1e-3).to_compare_f()),
            Values(szVGA, sz720p, sz1080p),
            Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(ResizePerfTestCPU, ResizePerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1),
            Values(cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_AREA),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values( cv::Size(64, 64),
                    cv::Size(32, 32)),
            Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(BottleneckKernelsPerfTestCPU, BottleneckKernelsConstInputPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values( "cv/optflow/frames/1080p_00.png", "cv/optflow/frames/720p_00.png",
                    "cv/optflow/frames/VGA_00.png", "cv/dnn_face/recognition/Aaron_Tippin_0001.jpg"),
            Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(ResizeInSimpleGraphPerfTestCPU, ResizeInSimpleGraphPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(CV_8UC3),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(0.5),
            Values(0.5),
            Values(cv::compile_args(IMGPROC_CPU))));

INSTANTIATE_TEST_CASE_P(ResizeFxFyPerfTestCPU, ResizeFxFyPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1),
            Values(cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_AREA),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(0.5, 0.1),
            Values(0.5, 0.1),
            Values(cv::compile_args(IMGPROC_CPU))));
} // opencv_test
