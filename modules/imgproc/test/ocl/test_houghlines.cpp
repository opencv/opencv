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

PARAM_TEST_CASE(HoughLinesTestBase, bool)
{
    double rhoStep;
    double thetaStep;
    int threshold;
    bool useRoi;

    Mat src, dst;
    UMat usrc, udst;

    virtual void SetUp()
    {
        rhoStep = 10;
        thetaStep = 0.1;
        threshold = 80;
        useRoi = false;
    }

    virtual void generateTestData()
    {
        //Mat image = readImage("shared/pic1.png", IMREAD_GRAYSCALE);
        
        Mat image = randomMat(Size(100, 100), CV_8UC1, 0, 255, false);
        
        cv::threshold(image, src, 127, 255, THRESH_BINARY);
        //Canny(image, src, 100, 150, 3);
        src.copyTo(usrc);
    }
};

typedef HoughLinesTestBase HoughLines;

OCL_TEST_P(HoughLines, RealImage)
{
    generateTestData();

    //std::cout << src << std::endl;

    OCL_OFF(cv::HoughLines(src, dst, rhoStep, thetaStep, threshold, 0, 0));
    OCL_ON(cv::HoughLines(usrc, udst, rhoStep, thetaStep, threshold, 0, 0));
}

OCL_INSTANTIATE_TEST_CASE_P(Imgproc, HoughLines, Values(true, false));

} } // namespace cvtest::ocl

#endif // HAVE_OPENCL