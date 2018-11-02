// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../test_precomp.hpp"
#include "../common/gapi_imgproc_tests.hpp"
#include "backends/fluid/gfluidimgproc.hpp"

#define IMGPROC_FLUID cv::gapi::imgproc::fluid::kernels()

namespace opencv_test
{

class AbsExactFluid : public Wrappable<AbsExactFluid>
{
public:
    AbsExactFluid() {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const { return cv::countNonZero(in1 != in2) == 0; }
private:
};


class AbsToleranceFluid : public Wrappable<AbsToleranceFluid>
{
public:
    AbsToleranceFluid(double tol) : _tol(tol) {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
    {
        cv::Mat absDiff; cv::absdiff(in1, in2, absDiff);
        return cv::countNonZero(absDiff > _tol) == 0;
    }
private:
    double _tol;
};

class AbsTolerance32FFluid : public Wrappable<AbsTolerance32FFluid>
{
public:
    AbsTolerance32FFluid(double tol) : _tol(tol) {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
    {
        if (CV_MAT_DEPTH(in1.type()) == CV_32F)
            return ((cv::countNonZero(cv::abs(in1 - in2) > (_tol)*cv::abs(in2))) ? false : true);
        else
            return ((cv::countNonZero(in1 != in2) <= (_tol * 100) * in2.total()) ? true : false);
    }
private:
    double _tol;
};

class AbsToleranceSepFilterFluid : public Wrappable<AbsToleranceSepFilterFluid>
{
public:
    AbsToleranceSepFilterFluid(double tol) : _tol(tol) {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
    {
        return ((cv::countNonZero(cv::abs(in1 - in2) > (_tol)* cv::abs(in2)) <= 0.01 * in2.total()) ? true : false);
    }
private:
    double _tol;
};

class AbsToleranceGaussianBlurFluid : public Wrappable<AbsToleranceGaussianBlurFluid>
{
public:
    AbsToleranceGaussianBlurFluid(double tol) : _tol(tol) {}
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
                bool a = (cv::countNonZero(cv::abs(in1 - in2) > 1) <= _tol * in2.total());
                return ((a == 1 ? 0 : 1) && ((cv::countNonZero(cv::abs(in1 - in2) > 2) <= 0) == 1 ? 0 : 1)) == 1 ? false : true;
            }
            else return cv::countNonZero(in1 != in2)==0;
        }
    }
private:
    double _tol;
};

class AbsToleranceRGBBGRFluid : public Wrappable<AbsToleranceRGBBGRFluid>
{
public:
    AbsToleranceRGBBGRFluid(double tol) : _tol(tol) {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
    {
        bool a = (cv::countNonZero((in1 - in2) > 0) <= _tol * in2.total());
        return ((a == 1 ? 0 : 1) && ((cv::countNonZero((in1 - in2) > 1) <= 0) == 1 ? 0 : 1)) == 1 ? false : true;
    }
private:
    double _tol;
};

class ToleranceTripleFluid : public Wrappable<ToleranceTripleFluid>
{
public:
    ToleranceTripleFluid(double tol1, double tol2, double tol3) : _tol1(tol1), _tol2(tol2), _tol3(tol3) {}
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

INSTANTIATE_TEST_CASE_P(RGB2GrayTestFluid, RGB2GrayTest,
                        Combine(Values(AbsToleranceRGBBGRFluid(0.001).to_compare_f()),
                                Values(cv::Size(1280, 720),
                                cv::Size(640, 480)),
                                Values(true, false),
                                Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(BGR2GrayTestFluid, BGR2GrayTest,
                        Combine(Values(AbsToleranceRGBBGRFluid(0.001).to_compare_f()),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(true, false),
                                Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(RGB2YUVTestFluid, RGB2YUVTest,
                        Combine(Values(AbsToleranceRGBBGRFluid(0.15*3).to_compare_f()),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(true, false),
                                Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(YUV2RGBTestFluid, YUV2RGBTest,
                        Combine(Values(ToleranceTripleFluid(0.25 * 3, 0.01 * 3, 1e-5 * 3).to_compare_f()),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(true, false),
                                Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(RGB2LabTestFluid, RGB2LabTest,
                        Combine(Values(ToleranceTripleFluid(0.25 * 3, 0.0, 1e-5 * 3).to_compare_f()),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(true, false),
                                Values(cv::compile_args(IMGPROC_FLUID))));

// FIXME: Not supported by Fluid yet (no kernel implemented)
INSTANTIATE_TEST_CASE_P(BGR2LUVTestFluid, BGR2LUVTest,
                        Combine(Values(ToleranceTripleFluid(0.25 * 3, 0.01 * 3, 0.0001 * 3).to_compare_f()),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(true, false),
                                Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(blurTestFluid, BlurTest,
                        Combine(Values(AbsToleranceFluid(0.0).to_compare_f()),
                                Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(3), // add kernel size=5 when implementation is ready
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(cv::BORDER_DEFAULT),
                                Values(true, false),
                                Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(gaussBlurTestFluid, GaussianBlurTest,
                        Combine(Values(AbsToleranceGaussianBlurFluid(1e-6).to_compare_f()),
                                Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(3), // add kernel size=5 when implementation is ready
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(true, false),
                                Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(medianBlurTestFluid, MedianBlurTest,
                        Combine(Values(AbsExactFluid().to_compare_f()),
                                Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(3), // add kernel size=5 when implementation is ready
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(true, false),
                                Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(erodeTestFluid, ErodeTest,
                        Combine(Values(AbsExactFluid().to_compare_f()),
                                Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(3), // add kernel size=5 when implementation is ready
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(cv::MorphShapes::MORPH_RECT,
                                       cv::MorphShapes::MORPH_CROSS,
                                       cv::MorphShapes::MORPH_ELLIPSE),
                                Values(true, false),
                                Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(dilateTestFluid, DilateTest,
                        Combine(Values(AbsExactFluid().to_compare_f()),
                                Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(3), // add kernel size=5 when implementation is ready
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(cv::MorphShapes::MORPH_RECT,
                                       cv::MorphShapes::MORPH_CROSS,
                                       cv::MorphShapes::MORPH_ELLIPSE),
                                Values(true, false),
                                Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(SobelTestFluid, SobelTest,
                        Combine(Values(AbsExactFluid().to_compare_f()),
                                Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(3), // add kernel size=5 when implementation is ready
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(-1, CV_32F),
                                Values(0, 1),
                                Values(1, 2),
                                Values(true, false),
                                Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(boxFilterTestFluid32, BoxFilterTest,
                        Combine(Values(AbsTolerance32FFluid(1e-6).to_compare_f()),
                                Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(3), // add kernel size=5 when implementation is ready
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(cv::BORDER_DEFAULT),
                                Values(-1, CV_32F),
                                Values(true, false),
                                Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(sepFilterTestFluid, SepFilterTest,
                        Combine(Values(AbsToleranceSepFilterFluid(1e-5f).to_compare_f()),
                                Values(CV_32FC1),
                                Values(3), // add kernel size=5 when implementation is ready
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(-1, CV_32F),
                                Values(true, false),
                                Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(filter2DTestFluid, Filter2DTest,
                        Combine(Values(AbsTolerance32FFluid(1e-6).to_compare_f()),
                                Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(3), // add kernel size=4,5,7 when implementation ready
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::BORDER_DEFAULT),
                                Values(-1, CV_32F),
                                Values(true, false),
                                Values(cv::compile_args(IMGPROC_FLUID))));

} // opencv_test
