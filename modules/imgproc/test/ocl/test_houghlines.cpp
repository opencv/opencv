// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Itseez, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

namespace opencv_test {
namespace ocl {

struct Vec2fComparator
{
    bool operator()(const Vec2f& a, const Vec2f b) const
    {
        if(a[0] != b[0]) return a[0] < b[0];
        else return a[1] < b[1];
    }
};

/////////////////////////////// HoughLines ////////////////////////////////////

PARAM_TEST_CASE(HoughLines, double, double, int)
{
    double rhoStep, thetaStep;
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

    void generateTestData()
    {
        src_size = randomSize(500, 1920);
        src.create(src_size, CV_8UC1);
        src.setTo(Scalar::all(0));

        // Horizontal lines (theta ≈ π, ~180°) - should be filtered when max_theta < π
        line(src, Point(0, 100), Point(200, 100), Scalar::all(255), 1);
        line(src, Point(0, 200), Point(200, 200), Scalar::all(255), 1);
        line(src, Point(0, 400), Point(200, 400), Scalar::all(255), 1);

        // Vertical lines (theta ≈ π/2, ~90°) - should be filtered when max_theta < π/2
        line(src, Point(100, 50), Point(100, 250), Scalar::all(255), 1);
        line(src, Point(200, 50), Point(200, 250), Scalar::all(255), 1);
        line(src, Point(400, 50), Point(400, 250), Scalar::all(255), 1);

        // Diagonal lines at ~30° (theta ≈ π/6, should pass when max_theta > π/6)
        line(src, Point(50, 50), Point(200, 137), Scalar::all(255), 1);
        line(src, Point(250, 150), Point(400, 237), Scalar::all(255), 1);

        // Diagonal lines at ~60° (theta ≈ π/3, should pass when min_theta < π/3 and max_theta > π/3)
        line(src, Point(50, 250), Point(108, 350), Scalar::all(255), 1);
        line(src, Point(300, 150), Point(358, 250), Scalar::all(255), 1);

        src.copyTo(usrc);
    }

    void readRealTestData()
    {
        Mat img = readImage("shared/pic5.png", IMREAD_GRAYSCALE);
        Canny(img, src, 100, 150, 3);

        src.copyTo(usrc);
    }

    void Near(double eps = 0.)
    {
        EXPECT_EQ(dst.size(), udst.size());

        if (dst.total() > 0)
        {
            Mat lines_cpu, lines_gpu;
            dst.copyTo(lines_cpu);
            udst.copyTo(lines_gpu);

            std::sort(lines_cpu.begin<Vec2f>(), lines_cpu.end<Vec2f>(), Vec2fComparator());
            std::sort(lines_gpu.begin<Vec2f>(), lines_gpu.end<Vec2f>(), Vec2fComparator());

            EXPECT_LE(TestUtils::checkNorm2(lines_cpu, lines_gpu), eps);
        }
    }
};

OCL_TEST_P(HoughLines, RealImage)
{
    readRealTestData();

    OCL_OFF(cv::HoughLines(src, dst, rhoStep, thetaStep, threshold));
    OCL_ON(cv::HoughLines(usrc, udst, rhoStep, thetaStep, threshold));

    Near(1e-5);
}

OCL_TEST_P(HoughLines, GeneratedImage)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::HoughLines(src, dst, rhoStep, thetaStep, threshold));
        OCL_ON(cv::HoughLines(usrc, udst, rhoStep, thetaStep, threshold));

        Near(1e-5);
    }
}

/////////////////////////////// HoughLinesP ///////////////////////////////////

PARAM_TEST_CASE(HoughLinesP, int, double, double)
{
    double rhoStep, thetaStep, minLineLength, maxGap;
    int threshold;

    Size src_size;
    Mat src, dst;
    UMat usrc, udst;

    virtual void SetUp()
    {
        rhoStep = 1.0;
        thetaStep = CV_PI / 180;
        threshold = GET_PARAM(0);
        minLineLength = GET_PARAM(1);
        maxGap = GET_PARAM(2);
    }

    void readRealTestData()
    {
        Mat img = readImage("shared/pic5.png", IMREAD_GRAYSCALE);
        Canny(img, src, 50, 200, 3);

        src.copyTo(usrc);
    }

    void Near(double eps = 0.)
    {
        Mat lines_gpu = udst.getMat(ACCESS_READ);

        if (dst.total() > 0 && lines_gpu.total() > 0)
        {
            Mat result_cpu(src.size(), CV_8UC1, Scalar::all(0));
            Mat result_gpu(src.size(), CV_8UC1, Scalar::all(0));

            MatConstIterator_<Vec4i> it = dst.begin<Vec4i>(), end = dst.end<Vec4i>();
            for ( ; it != end; it++)
            {
                Vec4i p = *it;
                line(result_cpu, Point(p[0], p[1]), Point(p[2], p[3]), Scalar(255));
            }

            it = lines_gpu.begin<Vec4i>(), end = lines_gpu.end<Vec4i>();
            for ( ; it != end; it++)
            {
                Vec4i p = *it;
                line(result_gpu, Point(p[0], p[1]), Point(p[2], p[3]), Scalar(255));
            }

            EXPECT_MAT_SIMILAR(result_cpu, result_gpu, eps);
        }
    }
};


OCL_TEST_P(HoughLinesP, RealImage)
{
    readRealTestData();

    OCL_OFF(cv::HoughLinesP(src, dst, rhoStep, thetaStep, threshold, minLineLength, maxGap));
    OCL_ON(cv::HoughLinesP(usrc, udst, rhoStep, thetaStep, threshold, minLineLength, maxGap));

    Near(0.25);
}

OCL_TEST_P(HoughLines, ThetaRange)
{
    // Test that min_theta and max_theta parameters are correctly used
    generateTestData();

    // Test with restricted theta range (only near horizontal lines)
    double min_theta = 0.0;
    double max_theta = CV_PI / 4;  // 0 to 45 degrees

    OCL_OFF(cv::HoughLines(src, dst, rhoStep, thetaStep, threshold, 0, min_theta, max_theta));
    OCL_ON(cv::HoughLines(usrc, udst, rhoStep, thetaStep, threshold, 0, min_theta, max_theta));

    // Verify that all detected lines have theta within the specified range
    Mat lines_gpu;
    udst.copyTo(lines_gpu);
    for (int i = 0; i < lines_gpu.rows; i++)
    {
        Vec2f line = lines_gpu.at<Vec2f>(i);
        double theta = line[1];
        EXPECT_GE(theta, min_theta) << "Line " << i << " has theta " << theta << " which is less than min_theta " << min_theta;
        EXPECT_LE(theta, max_theta) << "Line " << i << " has theta " << theta << " which is greater than max_theta " << max_theta;
    }

    Near(1e-5);

    // Test with different theta range (near vertical lines)
    min_theta = CV_PI / 3;  // 60 degrees
    max_theta = 2 * CV_PI / 3;  // 120 degrees

    OCL_OFF(cv::HoughLines(src, dst, rhoStep, thetaStep, threshold, 0, min_theta, max_theta));
    OCL_ON(cv::HoughLines(usrc, udst, rhoStep, thetaStep, threshold, 0, min_theta, max_theta));

    // Verify that all detected lines have theta within the specified range
    udst.copyTo(lines_gpu);
    for (int i = 0; i < lines_gpu.rows; i++)
    {
        Vec2f line = lines_gpu.at<Vec2f>(i);
        double theta = line[1];
        EXPECT_GE(theta, min_theta) << "Line " << i << " has theta " << theta << " which is less than min_theta " << min_theta;
        EXPECT_LE(theta, max_theta) << "Line " << i << " has theta " << theta << " which is greater than max_theta " << max_theta;
    }

    Near(1e-5);
}

OCL_INSTANTIATE_TEST_CASE_P(Imgproc, HoughLines, Combine(Values(1, 0.5),                        // rhoStep
                                                         Values(CV_PI / 180.0, CV_PI / 360.0),  // thetaStep
                                                         Values(85, 150)));                     // threshold

OCL_INSTANTIATE_TEST_CASE_P(Imgproc, HoughLinesP, Combine(Values(100, 150),                     // threshold
                                                          Values(50, 100),                      // minLineLength
                                                          Values(5, 10)));                      // maxLineGap

} } // namespace opencv_test::ocl

#endif // HAVE_OPENCL
