/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#include "test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"

namespace opencv_test { namespace {

class CV_SmallContourMomentTest : public cvtest::BaseTest
{
public:
    CV_SmallContourMomentTest() {}
    ~CV_SmallContourMomentTest() {}
protected:
    void run(int)
    {
        try
        {
            vector<Point> points;
            points.push_back(Point(50, 56));
            points.push_back(Point(53, 53));
            points.push_back(Point(46, 54));
            points.push_back(Point(49, 51));

            Moments m = moments(points, false);
            double area = contourArea(points);

            CV_Assert( m.m00 == 0 && m.m01 == 0 && m.m10 == 0 && area == 0 );
        }
        catch(...)
        {
            ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
        }
    }
};

TEST(Imgproc_ContourMoment, small) { CV_SmallContourMomentTest test; test.safe_run(); }

TEST(Imgproc_Moments, degenerateContours)
{
    std::vector<cv::Point> c1;
    c1.push_back(cv::Point(10,10));
    cv::Moments m1 = cv::moments(c1, false);
    EXPECT_EQ(m1.m00, 0);

    std::vector<cv::Point> c2;
    c2.push_back(cv::Point(0,0));
    c2.push_back(cv::Point(5,5));
    c2.push_back(cv::Point(10,10));
    cv::Moments m2 = cv::moments(c2, false);
    EXPECT_EQ(m2.m00, 0);
}

// Evaluate the defining polynomials separately from HuMoments' factored form.
static void referenceHuMoments(const Moments& m, double hu[7])
{
    const long double a = m.nu20, b = m.nu11, c = m.nu02;
    const long double d = m.nu30, e = m.nu21, f = m.nu12, g = m.nu03;
    hu[0] = static_cast<double>(a + c);
    hu[1] = static_cast<double>((a - c)*(a - c) + 4*b*b);
    hu[2] = static_cast<double>((d - 3*f)*(d - 3*f) + (3*e - g)*(3*e - g));
    hu[3] = static_cast<double>((d + f)*(d + f) + (e + g)*(e + g));
    hu[4] = static_cast<double>((d - 3*f)*(d + f)*((d + f)*(d + f) - 3*(e + g)*(e + g)) +
                               (3*e - g)*(e + g)*(3*(d + f)*(d + f) - (e + g)*(e + g)));
    hu[5] = static_cast<double>((a - c)*((d + f)*(d + f) - (e + g)*(e + g)) +
                               4*b*(d + f)*(e + g));
    hu[6] = static_cast<double>((3*e - g)*(d + f)*((d + f)*(d + f) - 3*(e + g)*(e + g)) -
                               (d - 3*f)*(e + g)*(3*(d + f)*(d + f) - (e + g)*(e + g)));
}

TEST(Imgproc_HuMoments, accuracy)
{
    // Restore the polynomial accuracy coverage removed with the C API (#24957).
    RNG rng(0x24957);
    for (int sample = 0; sample < 256; ++sample)
    {
        SCOPED_TRACE(sample);
        Moments m;
        double* normalized[] = { &m.nu20, &m.nu11, &m.nu02, &m.nu30,
                                 &m.nu21, &m.nu12, &m.nu03 };
        for (int i = 0; i < 7; ++i)
        {
            // Zero input and individual basis inputs also exercise sparse cases.
            if (sample <= 7)
                *normalized[i] = sample == i + 1 ? 1. : 0.;
            else
                *normalized[i] = rng.uniform(-1., 1.);
        }

        double expected[7], actual[7];
        referenceHuMoments(m, expected);
        HuMoments(m, actual);
        Mat output;
        HuMoments(m, output);
        ASSERT_EQ(Size(1, 7), output.size());
        ASSERT_EQ(CV_64FC1, output.type());
        for (int i = 0; i < 7; ++i)
        {
            SCOPED_TRACE(i);
            const double tolerance = 1e-12 * std::max(1., std::abs(expected[i]));
            EXPECT_NEAR(expected[i], actual[i], tolerance);
            EXPECT_NEAR(expected[i], output.at<double>(i), tolerance);
        }
    }
}

TEST(Imgproc_HuMoments, similarityInvariance)
{
    const Point2f vertices[] = { Point2f(0, 0), Point2f(7, 1), Point2f(6, 5),
                                 Point2f(2, 8), Point2f(-1, 3) };
    const std::vector<Point2f> contour(vertices, vertices + 5);
    double expected[7];
    HuMoments(moments(contour), expected);
    // An asymmetric contour is needed to detect a missing reflection sign.
    ASSERT_GT(std::abs(expected[6]), 1e-12);

    for (int transform = 0; transform < 4; ++transform)
    {
        SCOPED_TRACE(transform);
        std::vector<Point2f> transformed;
        for (size_t i = 0; i < contour.size(); ++i)
        {
            const Point2f& p = contour[i];
            if (transform == 0)
                transformed.push_back(p + Point2f(16, -8));
            else if (transform == 1)
                transformed.push_back(p * 4.f);
            else if (transform == 2)
                transformed.push_back(Point2f(-p.y, p.x));
            else
                transformed.push_back(Point2f(-p.x, p.y));
        }
        double actual[7];
        HuMoments(moments(transformed), actual);
        for (int i = 0; i < 7; ++i)
        {
            SCOPED_TRACE(i);
            const double value = transform == 3 && i == 6 ? -expected[i] : expected[i];
            EXPECT_NEAR(value, actual[i], 1e-9 * std::max(std::abs(value), 1e-15));
        }
    }
}

}} // namespace
