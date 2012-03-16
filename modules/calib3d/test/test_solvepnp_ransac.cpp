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
    CV_solvePnPRansac_Test()
    {
        eps[CV_ITERATIVE] = 1.0e-2;
        eps[CV_EPNP] = 1.0e-2;
        eps[CV_P3P] = 1.0e-2;
        totalTestsCount = 10;
    }
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

    void generateCameraMatrix(Mat& cameraMatrix, RNG& rng)
    {
        const double fcMinVal = 1e-3;
        const double fcMaxVal = 100;
        cameraMatrix.create(3, 3, CV_64FC1);
        cameraMatrix.setTo(Scalar(0));
        cameraMatrix.at<double>(0,0) = rng.uniform(fcMinVal, fcMaxVal);
        cameraMatrix.at<double>(1,1) = rng.uniform(fcMinVal, fcMaxVal);
        cameraMatrix.at<double>(0,2) = rng.uniform(fcMinVal, fcMaxVal);
        cameraMatrix.at<double>(1,2) = rng.uniform(fcMinVal, fcMaxVal);
        cameraMatrix.at<double>(2,2) = 1;
    }

    void generateDistCoeffs(Mat& distCoeffs, RNG& rng)
    {
        distCoeffs = Mat::zeros(4, 1, CV_64FC1);
        for (int i = 0; i < 3; i++)
            distCoeffs.at<double>(i,0) = rng.uniform(0.0, 1.0e-6);
    }

    void generatePose(Mat& rvec, Mat& tvec, RNG& rng)
    {
        const double minVal = 1.0e-3;
        const double maxVal = 1.0;
        rvec.create(3, 1, CV_64FC1);
        tvec.create(3, 1, CV_64FC1);
        for (int i = 0; i < 3; i++)
        {
            rvec.at<double>(i,0) = rng.uniform(minVal, maxVal);
            tvec.at<double>(i,0) = rng.uniform(minVal, maxVal/10);
        }
    }

    virtual bool runTest(RNG& rng, int mode, int method, const vector<Point3f>& points, const double* eps, double& maxError)
    {
        Mat rvec, tvec;
        vector<int> inliers;
        Mat trueRvec, trueTvec;
        Mat intrinsics, distCoeffs;
        generateCameraMatrix(intrinsics, rng);
        if (mode == 0)
            distCoeffs = Mat::zeros(4, 1, CV_64FC1);
        else
            generateDistCoeffs(distCoeffs, rng);
        generatePose(trueRvec, trueTvec, rng);

        vector<Point2f> projectedPoints;
        projectedPoints.resize(points.size());
        projectPoints(Mat(points), trueRvec, trueTvec, intrinsics, distCoeffs, projectedPoints);
        for (size_t i = 0; i < projectedPoints.size(); i++)
        {
            if (i % 20 == 0)
            {
                projectedPoints[i] = projectedPoints[rng.uniform(0,(int)points.size()-1)];
            }
        }

        solvePnPRansac(points, projectedPoints, intrinsics, distCoeffs, rvec, tvec,
            false, 500, 0.5, -1, inliers, method);

        bool isTestSuccess = inliers.size() >= points.size()*0.95;

        double rvecDiff = norm(rvec-trueRvec), tvecDiff = norm(tvec-trueTvec);
        isTestSuccess = isTestSuccess && rvecDiff < eps[method] && tvecDiff < eps[method];
        double error = rvecDiff > tvecDiff ? rvecDiff : tvecDiff;
        //cout << error << " " << inliers.size() << " " << eps[method] << endl;
        if (error > maxError)
            maxError = error;

        return isTestSuccess;
    }

    void run(int)
    {
        cvtest::TS& ts = *this->ts;
        ts.set_failed_test_info(cvtest::TS::OK);

        vector<Point3f> points;
        const int pointsCount = 500;
        points.resize(pointsCount);
        generate3DPointCloud(points);


        const int methodsCount = 3;
        RNG rng = ts.get_rng();


        for (int mode = 0; mode < 2; mode++)
        {
            for (int method = 0; method < methodsCount; method++)
            {
                double maxError = 0;
                int successfulTestsCount = 0;
                for (int testIndex = 0; testIndex < totalTestsCount; testIndex++)
                {
                    if (runTest(rng, mode, method, points, eps, maxError))
                        successfulTestsCount++;
                }
                //cout <<  maxError << " " << successfulTestsCount << endl;
                if (successfulTestsCount < 0.7*totalTestsCount)
                {
                    ts.printf( cvtest::TS::LOG, "Invalid accuracy for method %d, failed %d tests from %d, maximum error equals %f, distortion mode equals %d\n",
                        method, totalTestsCount - successfulTestsCount, totalTestsCount, maxError, mode);
                    ts.set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                }
            }
        }
    }
    double eps[3];
    int totalTestsCount;
};

class CV_solvePnP_Test : public CV_solvePnPRansac_Test
{
public:
    CV_solvePnP_Test()
    {
        eps[CV_ITERATIVE] = 1.0e-6;
        eps[CV_EPNP] = 1.0e-6;
        eps[CV_P3P] = 1.0e-4;
        totalTestsCount = 1000;
    }

    ~CV_solvePnP_Test() {}
protected:
    virtual bool runTest(RNG& rng, int mode, int method, const vector<Point3f>& points, const double* eps, double& maxError)
    {
        Mat rvec, tvec;
        Mat trueRvec, trueTvec;
        Mat intrinsics, distCoeffs;
        generateCameraMatrix(intrinsics, rng);
        if (mode == 0)
            distCoeffs = Mat::zeros(4, 1, CV_64FC1);
        else
            generateDistCoeffs(distCoeffs, rng);
        generatePose(trueRvec, trueTvec, rng);

        std::vector<Point3f> opoints;
        if (method == 2)
        {
            opoints = std::vector<Point3f>(points.begin(), points.begin()+4);
        }
        else
            opoints = points;

        vector<Point2f> projectedPoints;
        projectedPoints.resize(opoints.size());
        projectPoints(Mat(opoints), trueRvec, trueTvec, intrinsics, distCoeffs, projectedPoints);

        solvePnP(opoints, projectedPoints, intrinsics, distCoeffs, rvec, tvec,
            false, method);

        double rvecDiff = norm(rvec-trueRvec), tvecDiff = norm(tvec-trueTvec);
        bool isTestSuccess = rvecDiff < eps[method] && tvecDiff < eps[method];

        double error = rvecDiff > tvecDiff ? rvecDiff : tvecDiff;
        if (error > maxError)
            maxError = error;

        return isTestSuccess;
    }
};

TEST(Calib3d_SolvePnPRansac, accuracy) { CV_solvePnPRansac_Test test; test.safe_run(); }
TEST(Calib3d_SolvePnP, accuracy) { CV_solvePnP_Test test; test.safe_run(); }

