/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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

using namespace cv;
using namespace std;

class CV_solvePnPRansac_Test : public cvtest::BaseTest
{
public:
    CV_solvePnPRansac_Test() {}
    ~CV_solvePnPRansac_Test() {}
protected:
    void generate3DPointCloud(vector<Point3f>& points, Point3f pmin = Point3f(-1,
                    -1, 5), Point3f pmax = Point3f(1, 1, 10))
    {
            const Point3f delta = pmax - pmin;
            for (size_t i = 0; i < points.size(); i++)
            {
                    Point3f p(float(rand()) / RAND_MAX, float(rand()) / RAND_MAX,
                                    float(rand()) / RAND_MAX);
                    p.x *= delta.x;
                    p.y *= delta.y;
                    p.z *= delta.z;
                    p = p + pmin;
                    points[i] = p;
            }
    }

    void run(int)
    {
        cvtest::TS& ts = *this->ts;
        ts.set_failed_test_info(cvtest::TS::OK);
        Mat intrinsics = Mat::eye(3, 3, CV_32FC1);
        intrinsics.at<float> (0, 0) = 400.0;
        intrinsics.at<float> (1, 1) = 400.0;
        intrinsics.at<float> (0, 2) = 640 / 2;
        intrinsics.at<float> (1, 2) = 480 / 2;
        Mat dist_coeffs = Mat::zeros(5, 1, CV_32FC1);
        Mat rvec1 = Mat::zeros(3, 1, CV_64FC1);
        Mat tvec1 = Mat::zeros(3, 1, CV_64FC1);
        rvec1.at<double> (0, 0) = 1.0f;
        tvec1.at<double> (0, 0) = 1.0f;
        tvec1.at<double> (1, 0) = 1.0f;
        vector<Point3f> points;
        points.resize(500);
        generate3DPointCloud(points);

        vector<Point2f> points1;
        points1.resize(points.size());
        projectPoints(Mat(points), rvec1, tvec1, intrinsics, dist_coeffs, points1);
        for (size_t i = 0; i < points1.size(); i++)
        {
            if (i % 20 == 0)
            {
                    points1[i] = points1[rand() % points.size()];
            }
        }
        double eps = 1.0e-7;
        for (int testIndex = 0; testIndex<  10; testIndex++)
        {
            try
            {
                Mat rvec, tvec;
                vector<int> inliers;

                solvePnPRansac(Mat(points), Mat(points1), intrinsics, dist_coeffs, rvec, tvec,
                                false, 1000, 2.0, -1, &inliers);

                bool isTestSuccess = inliers.size() == 475;

                isTestSuccess = isTestSuccess
                                && (abs(rvec.at<double> (0, 0) - 1) < eps);
                isTestSuccess = isTestSuccess && (abs(rvec.at<double> (1, 0)) < eps);
                isTestSuccess = isTestSuccess && (abs(rvec.at<double> (2, 0)) < eps);
                isTestSuccess = isTestSuccess
                                && (abs(tvec.at<double> (0, 0) - 1) < eps);
                isTestSuccess = isTestSuccess
                                && (abs(tvec.at<double> (1, 0) - 1) < eps);
                isTestSuccess = isTestSuccess && (abs(tvec.at<double> (2, 0)) < eps);
                if (!isTestSuccess)
                {
                    ts.printf( cvtest::TS::LOG, "Invalid accuracy, inliers.size = %d\n", inliers.size());
                    ts.set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                    break;
                }

            }
            catch(...)
            {
                    ts.printf(cvtest::TS::LOG, "Exception in solvePnPRansac\n");
                    ts.set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
            }
        }
    }
};

TEST(Calib3d_SolvePnPRansac, accuracy) { CV_solvePnPRansac_Test test; test.safe_run(); }

