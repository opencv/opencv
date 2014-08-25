// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Itseez, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

namespace cvtest {
namespace ocl {

struct Vec2fComparator
{
    bool operator()(const cv::Vec2f& a, const cv::Vec2f b) const
    {
        if(a[0] != b[0]) return a[0] < b[0];
        else return a[1] < b[1];
    }
};

PARAM_TEST_CASE(HoughLinesTestBase, double, double, int)
{
    double rhoStep;
    double thetaStep;
    int threshold;

    Size src_size;
    Mat src, dst;
    UMat usrc, udst;

    virtual void SetUp()
    {
        rhoStep = GET_PARAM(0);
        thetaStep = GET_PARAM(1);
        threshold = GET_PARAM(2);
    }

    virtual void generateTestData()
    {
        src_size = randomSize(500, 1000);
        src.create(src_size, CV_8UC1);
        src.setTo(Scalar::all(0));
        line(src, Point(0, 100), Point(100, 100), Scalar::all(255), 1);
        line(src, Point(0, 200), Point(100, 200), Scalar::all(255), 1);
        line(src, Point(0, 400), Point(100, 400), Scalar::all(255), 1);
        line(src, Point(100, 0), Point(100, 200), Scalar::all(255), 1);
        line(src, Point(200, 0), Point(200, 200), Scalar::all(255), 1);
        line(src, Point(400, 0), Point(400, 200), Scalar::all(255), 1);
        
        src.copyTo(usrc);
    }
};

typedef HoughLinesTestBase HoughLines;

OCL_TEST_P(HoughLines, GeneratedImage)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::HoughLines(src, dst, rhoStep, thetaStep, threshold));
        OCL_ON(cv::HoughLines(usrc, udst, rhoStep, thetaStep, threshold));

        //Near(1e-5);
    }
}

OCL_INSTANTIATE_TEST_CASE_P(Imgproc, HoughLines, Combine(Values(1, 0.5),                        // rhoStep
                                                         Values(CV_PI / 180.0, CV_PI / 360.0),  // thetaStep
                                                         Values(80, 150)));                     // threshold

} } // namespace cvtest::ocl

#endif // HAVE_OPENCL