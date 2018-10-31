// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../test_precomp.hpp"

#include "../common/gapi_imgproc_tests.hpp"
#include "opencv2/gapi/gpu/imgproc.hpp"

#define IMGPROC_GPU cv::gapi::imgproc::gpu::kernels()

namespace opencv_test
{

class AbsExactGPU : public Wrappable<AbsExactGPU>
{
public:
    AbsExactGPU() {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const { return cv::countNonZero(in1 != in2) == 0; }
private:
};

class AbsToleranceGPU : public Wrappable<AbsToleranceGPU>
{
public:
    AbsToleranceGPU(double tol) : _tol(tol) {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
    {
        cv::Mat absDiff; cv::absdiff(in1, in2, absDiff);
        return cv::countNonZero(absDiff > _tol) == 0;
    }
private:
    double _tol;
};

class AbsTolerance32FGPU : public Wrappable<AbsTolerance32FGPU>
{
public:
    AbsTolerance32FGPU(double tol, double tol8u) : _tol(tol), _tol8u(tol8u) {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
    {
        if (CV_MAT_DEPTH(in1.type()) == CV_32F)
            return ((cv::countNonZero(cv::abs(in1 - in2) > (_tol)*cv::abs(in2))) ? false : true);
        else
            return ((cv::countNonZero(in1 != in2) <= (_tol8u) * in2.total()) ? true : false);
    }
private:
    double _tol;
    double _tol8u;
};

class AbsToleranceSepFilterGPU : public Wrappable<AbsToleranceSepFilterGPU>
{
public:
    AbsToleranceSepFilterGPU(double tol) : _tol(tol) {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
    {
        return ((cv::countNonZero(cv::abs(in1 - in2) > (_tol)* cv::abs(in2)) <= 0.01 * in2.total()) ? true : false);
    }
private:
    double _tol;
};

class AbsToleranceGaussianBlurGPU : public Wrappable<AbsToleranceGaussianBlurGPU>
{
public:
    AbsToleranceGaussianBlurGPU(double tol, double tol8u) : _tol(tol), _tol8u(tol8u) {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
    {
        if (CV_MAT_DEPTH(in1.type()) == CV_32F || CV_MAT_DEPTH(in1.type()) == CV_64F)
        {
            return ((cv::countNonZero(cv::abs(in1 - in2) > (_tol)*cv::abs(in2))) ? false : true);
        }
        else
        {
            if (CV_MAT_DEPTH(in1.type()) == CV_8U)
            {
                bool a = (cv::countNonZero(cv::abs(in1 - in2) > 1) <= _tol8u * in2.total());
                return ((a == 1 ? 0 : 1) && ((cv::countNonZero(cv::abs(in1 - in2) > 2) <= 0) == 1 ? 0 : 1)) == 1 ? false : true;
            }
            else return cv::countNonZero(in1 != in2) == 0;
        }
    }
private:
    double _tol;
    double _tol8u;
};

class ToleranceTripleGPU : public Wrappable<ToleranceTripleGPU>
{
public:
    ToleranceTripleGPU(double tol1, double tol2, double tol3) : _tol1(tol1), _tol2(tol2), _tol3(tol3) {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
    {
        bool a = (cv::countNonZero((in1 - in2) > 0) <= _tol1 * in2.total());
        return (((a == 1 ? 0 : 1) &&
            ((cv::countNonZero((in1 - in2) > 1) <= _tol2 * in2.total()) == 1 ? 0 : 1) &&
            ((cv::countNonZero((in1 - in2) > 2) <= _tol3 * in2.total()) == 1 ? 0 : 1))) == 1 ? false : true;
    }
private:
    double _tol1, _tol2, _tol3;
};


INSTANTIATE_TEST_CASE_P(Filter2DTestGPU, Filter2DTest,
                        Combine(Values(AbsTolerance32FGPU(1e-5, 1e-3).to_compare_f()),
                                Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(3, 4, 5, 7),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::BORDER_DEFAULT),
                                Values(-1, CV_32F),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(BoxFilterTestGPU, BoxFilterTest,
                        Combine(Values(AbsTolerance32FGPU(1e-5, 1e-3).to_compare_f()),
                                Values(/*CV_8UC1,*/ CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(3,5),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(cv::BORDER_DEFAULT),
                                Values(-1, CV_32F),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));  //TODO: 8UC1 doesn't work

INSTANTIATE_TEST_CASE_P(SepFilterTestGPU_8U, SepFilterTest,
                        Combine(Values(AbsToleranceSepFilterGPU(1e-4f).to_compare_f()),
                                Values(CV_8UC1, CV_8UC3),
                                Values(3),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(-1, CV_16S, CV_32F),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(SepFilterTestGPU_other, SepFilterTest,
                        Combine(Values(AbsToleranceSepFilterGPU(1e-4f).to_compare_f()),
                                Values(CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(3),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(-1, CV_32F),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(BlurTestGPU, BlurTest,
                        Combine(Values(AbsToleranceGPU(1e-4f).to_compare_f()),
                                Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(3,5),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(cv::BORDER_DEFAULT),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(gaussBlurTestGPU, GaussianBlurTest,
                        Combine(Values(AbsToleranceGaussianBlurGPU(1e-5, 0.05).to_compare_f()), //TODO: too relaxed?
                                Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(3, 5),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(MedianBlurTestGPU, MedianBlurTest,
                        Combine(Values(AbsExactGPU().to_compare_f()),
                                Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(3, 5),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(ErodeTestGPU, ErodeTest,
                        Combine(Values(AbsExactGPU().to_compare_f()),
                                Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(3, 5),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(cv::MorphShapes::MORPH_RECT,
                                       cv::MorphShapes::MORPH_CROSS,
                                       cv::MorphShapes::MORPH_ELLIPSE),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(Erode3x3TestGPU, Erode3x3Test,
                        Combine(Values(AbsExactGPU().to_compare_f()),
                                Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
/*init output matrices or not*/ testing::Bool(),
                                Values(1,2,4),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(DilateTestGPU, DilateTest,
                        Combine(Values(AbsExactGPU().to_compare_f()),
                                Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(3, 5),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(cv::MorphShapes::MORPH_RECT,
                                       cv::MorphShapes::MORPH_CROSS,
                                       cv::MorphShapes::MORPH_ELLIPSE),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(Dilate3x3TestGPU, Dilate3x3Test,
                        Combine(Values(AbsExactGPU().to_compare_f()),
                                Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
/*init output matrices or not*/ testing::Bool(),
                                Values(1,2,4),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(SobelTestGPU, SobelTest,
                        Combine(Values(AbsTolerance32FGPU(1e-4, 1e-4).to_compare_f()),
                                Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1/*, CV_32FC1*/), //TODO: CV_32FC1 fails accuracy
                                Values(3, 5),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(-1, CV_32F),
                                Values(0, 1),
                                Values(1, 2),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(EqHistTestGPU, EqHistTest,
                        Combine(Values(AbsExactGPU().to_compare_f()),
                                Values(cv::Size(1280, 720),
                                cv::Size(640, 480)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(CannyTestGPU, CannyTest,
                        Combine(Values(AbsTolerance32FGPU(1e-4, 1e-2).to_compare_f()),
                                Values(CV_8UC1, CV_8UC3),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(3.0, 120.0),
                                Values(125.0, 240.0),
                                Values(3, 5),
                                testing::Bool(),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(RGB2GrayTestGPU, RGB2GrayTest,
                        Combine(Values(AbsExactGPU().to_compare_f()),
                                Values(cv::Size(1280, 720),
                                cv::Size(640, 480)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(BGR2GrayTestGPU, BGR2GrayTest,
                        Combine(Values(AbsExactGPU().to_compare_f()),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(RGB2YUVTestGPU, RGB2YUVTest,
                        Combine(Values(AbsExactGPU().to_compare_f()),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(YUV2RGBTestGPU, YUV2RGBTest,
                        Combine(Values(AbsExactGPU().to_compare_f()),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(RGB2LabTestGPU, RGB2LabTest,
                        Combine(Values(AbsExactGPU().to_compare_f()),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(BGR2LUVTestGPU, BGR2LUVTest,
                        Combine(Values(ToleranceTripleGPU(0.25 * 3, 0.01 * 3, 0.0001 * 3).to_compare_f()),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(LUV2BGRTestGPU, LUV2BGRTest,
                        Combine(Values(AbsExactGPU().to_compare_f()),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(BGR2YUVTestGPU, BGR2YUVTest,
                        Combine(Values(AbsExactGPU().to_compare_f()),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(YUV2BGRTestGPU, YUV2BGRTest,
                        Combine(Values(AbsExactGPU().to_compare_f()),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));


} // opencv_test
