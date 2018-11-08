// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../perf_precomp.hpp"
#include "../common/gapi_imgproc_perf_tests.hpp"
#include "../../src/backends/fluid/gfluidimgproc.hpp"


#define IMGPROC_FLUID cv::gapi::imgproc::fluid::kernels()

namespace opencv_test
{

    class AbsExact : public Wrappable<AbsExact>
    {
    public:
        AbsExact() {}
        bool operator() (const cv::Mat& in1, const cv::Mat& in2) const { return cv::countNonZero(in1 != in2) == 0; }
    private:
    };

    class AbsTolerance : public Wrappable<AbsTolerance>
    {
    public:
        AbsTolerance(double tol) : _tol(tol) {}
        bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
        {
            cv::Mat absDiff; cv::absdiff(in1, in2, absDiff);
            return cv::countNonZero(absDiff > _tol) == 0;
        }
    private:
        double _tol;
    };

    class AbsToleranceSobelFluid : public Wrappable<AbsToleranceSobelFluid>
    {
    public:
        AbsToleranceSobelFluid(double tol) : tolerance(tol) {}
        bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
        {
            cv::Mat diff, a1, a2, b, base;
            cv::absdiff(in1, in2, diff);
            a1 = cv::abs(in1);
            a2 = cv::abs(in2);
            cv::max(a1, a2, b);
            cv::max(1, b, base);  // base = max{1, |in1|, |in2|}
            return cv::countNonZero(diff > tolerance*base) == 0;
        }
    private:
        double tolerance;
    };


    INSTANTIATE_TEST_CASE_P(SobelPerfTestFluid, SobelPerfTest,
        Combine(Values(AbsExact().to_compare_f()),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1),  // add CV_32FC1 when ready
            Values(3),                                     // add 5x5 once supported
            Values(szVGA, sz720p, sz1080p),
            Values(-1, CV_16S, CV_32F),
            Values(0, 1),
            Values(1, 2),
            Values(cv::compile_args(IMGPROC_FLUID))));

    INSTANTIATE_TEST_CASE_P(SobelPerfTestFluid32F, SobelPerfTest,
        Combine(Values(AbsToleranceSobelFluid(1e-3).to_compare_f()),
            Values(CV_32FC1),
            Values(3),                                     // add 5x5 once supported
            Values(szVGA, sz720p, sz1080p),
            Values(CV_32F),
            Values(0, 1),
            Values(1, 2),
            Values(cv::compile_args(IMGPROC_FLUID))));

}
