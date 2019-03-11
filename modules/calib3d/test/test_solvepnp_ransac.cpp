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

namespace opencv_test { namespace {

class CV_solvePnPRansac_Test : public cvtest::BaseTest
{
public:
    CV_solvePnPRansac_Test()
    {
        eps[SOLVEPNP_ITERATIVE] = 1.0e-2;
        eps[SOLVEPNP_EPNP] = 1.0e-2;
        eps[SOLVEPNP_P3P] = 1.0e-2;
        eps[SOLVEPNP_AP3P] = 1.0e-2;
        eps[SOLVEPNP_DLS] = 1.0e-2;
        eps[SOLVEPNP_UPNP] = 1.0e-2;
        totalTestsCount = 10;
        pointsCount = 500;
    }
    ~CV_solvePnPRansac_Test() {}
protected:
    void generate3DPointCloud(vector<Point3f>& points,
        Point3f pmin = Point3f(-1, -1, 5),
        Point3f pmax = Point3f(1, 1, 10))
    {
        RNG rng = cv::theRNG(); // fix the seed to use "fixed" input 3D points

        for (size_t i = 0; i < points.size(); i++)
        {
            float _x = rng.uniform(pmin.x, pmax.x);
            float _y = rng.uniform(pmin.y, pmax.y);
            float _z = rng.uniform(pmin.z, pmax.z);
            points[i] = Point3f(_x, _y, _z);
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

    virtual bool runTest(RNG& rng, int mode, int method, const vector<Point3f>& points, const double* epsilon, double& maxError)
    {
        Mat rvec, tvec;
        vector<int> inliers;
        Mat trueRvec, trueTvec;
        Mat intrinsics, distCoeffs;
        generateCameraMatrix(intrinsics, rng);
        if (method == 4) intrinsics.at<double>(1,1) = intrinsics.at<double>(0,0);
        if (mode == 0)
            distCoeffs = Mat::zeros(4, 1, CV_64FC1);
        else
            generateDistCoeffs(distCoeffs, rng);
        generatePose(trueRvec, trueTvec, rng);

        vector<Point2f> projectedPoints;
        projectedPoints.resize(points.size());
        projectPoints(points, trueRvec, trueTvec, intrinsics, distCoeffs, projectedPoints);
        for (size_t i = 0; i < projectedPoints.size(); i++)
        {
            if (i % 20 == 0)
            {
                projectedPoints[i] = projectedPoints[rng.uniform(0,(int)points.size()-1)];
            }
        }

        solvePnPRansac(points, projectedPoints, intrinsics, distCoeffs, rvec, tvec, false, pointsCount, 0.5f, 0.99, inliers, method);

        bool isTestSuccess = inliers.size() >= points.size()*0.95;

        double rvecDiff = cvtest::norm(rvec, trueRvec, NORM_L2), tvecDiff = cvtest::norm(tvec, trueTvec, NORM_L2);
        isTestSuccess = isTestSuccess && rvecDiff < epsilon[method] && tvecDiff < epsilon[method];
        double error = rvecDiff > tvecDiff ? rvecDiff : tvecDiff;
        //cout << error << " " << inliers.size() << " " << eps[method] << endl;
        if (error > maxError)
            maxError = error;

        return isTestSuccess;
    }

    virtual void run(int)
    {
        ts->set_failed_test_info(cvtest::TS::OK);

        vector<Point3f> points, points_dls;
        points.resize(pointsCount);
        generate3DPointCloud(points);

        RNG rng = ts->get_rng();


        for (int mode = 0; mode < 2; mode++)
        {
            for (int method = 0; method < SOLVEPNP_MAX_COUNT; method++)
            {
                double maxError = 0;
                int successfulTestsCount = 0;
                for (int testIndex = 0; testIndex < totalTestsCount; testIndex++)
                {
                    if (runTest(rng, mode, method, points, eps, maxError))
                    {
                        successfulTestsCount++;
                    }
                }
                if (successfulTestsCount < 0.7*totalTestsCount)
                {
                    ts->printf( cvtest::TS::LOG, "Invalid accuracy for method %d, failed %d tests from %d, maximum error equals %f, distortion mode equals %d\n",
                        method, totalTestsCount - successfulTestsCount, totalTestsCount, maxError, mode);
                    ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                }
                cout << "mode: " << mode << ", method: " << method << " -> "
                     << ((double)successfulTestsCount / totalTestsCount) * 100 << "%"
                     << " (err < " << maxError << ")" << endl;
            }
        }
    }
    double eps[SOLVEPNP_MAX_COUNT];
    int totalTestsCount;
    int pointsCount;
};

class CV_solvePnP_Test : public CV_solvePnPRansac_Test
{
public:
    CV_solvePnP_Test()
    {
        eps[SOLVEPNP_ITERATIVE] = 1.0e-6;
        eps[SOLVEPNP_EPNP] = 1.0e-6;
        eps[SOLVEPNP_P3P] = 2.0e-4;
        eps[SOLVEPNP_AP3P] = 1.0e-4;
        eps[SOLVEPNP_DLS] = 1.0e-4;
        eps[SOLVEPNP_UPNP] = 1.0e-4;
        totalTestsCount = 1000;
    }

    ~CV_solvePnP_Test() {}
protected:
    virtual bool runTest(RNG& rng, int mode, int method, const vector<Point3f>& points, const double* epsilon, double& maxError)
    {
        Mat rvec, tvec;
        Mat trueRvec, trueTvec;
        Mat intrinsics, distCoeffs;
        generateCameraMatrix(intrinsics, rng);
        if (method == SOLVEPNP_DLS)
        {
            intrinsics.at<double>(1,1) = intrinsics.at<double>(0,0);
        }
        if (mode == 0)
        {
            distCoeffs = Mat::zeros(4, 1, CV_64FC1);
        }
        else
        {
            generateDistCoeffs(distCoeffs, rng);
        }
        generatePose(trueRvec, trueTvec, rng);

        std::vector<Point3f> opoints;
        switch(method)
        {
            case SOLVEPNP_P3P:
            case SOLVEPNP_AP3P:
                opoints = std::vector<Point3f>(points.begin(), points.begin()+4);
                break;
            case SOLVEPNP_UPNP:
                opoints = std::vector<Point3f>(points.begin(), points.begin()+50);
                break;
            default:
                opoints = points;
                break;
        }

        vector<Point2f> projectedPoints;
        projectedPoints.resize(opoints.size());
        projectPoints(opoints, trueRvec, trueTvec, intrinsics, distCoeffs, projectedPoints);

        bool isEstimateSuccess = solvePnP(opoints, projectedPoints, intrinsics, distCoeffs, rvec, tvec, false, method);
        if (isEstimateSuccess == false)
        {
            return isEstimateSuccess;
        }

        double rvecDiff = cvtest::norm(rvec, trueRvec, NORM_L2), tvecDiff = cvtest::norm(tvec, trueTvec, NORM_L2);
        bool isTestSuccess = rvecDiff < epsilon[method] && tvecDiff < epsilon[method];

        double error = rvecDiff > tvecDiff ? rvecDiff : tvecDiff;
        if (error > maxError)
        {
            maxError = error;
        }

        return isTestSuccess;
    }
};

class CV_solveP3P_Test : public CV_solvePnPRansac_Test
{
 public:
  CV_solveP3P_Test()
  {
    eps[SOLVEPNP_P3P] = 2.0e-4;
    eps[SOLVEPNP_AP3P] = 1.0e-4;
    totalTestsCount = 1000;
  }

  ~CV_solveP3P_Test() {}
 protected:
  virtual bool runTest(RNG& rng, int mode, int method, const vector<Point3f>& points, const double* epsilon, double& maxError)
  {
    std::vector<Mat> rvecs, tvecs;
    Mat trueRvec, trueTvec;
    Mat intrinsics, distCoeffs;
    generateCameraMatrix(intrinsics, rng);
    if (mode == 0)
      distCoeffs = Mat::zeros(4, 1, CV_64FC1);
    else
      generateDistCoeffs(distCoeffs, rng);
    generatePose(trueRvec, trueTvec, rng);

    std::vector<Point3f> opoints;
    opoints = std::vector<Point3f>(points.begin(), points.begin()+3);

    vector<Point2f> projectedPoints;
    projectedPoints.resize(opoints.size());
    projectPoints(opoints, trueRvec, trueTvec, intrinsics, distCoeffs, projectedPoints);

    int num_of_solutions = solveP3P(opoints, projectedPoints, intrinsics, distCoeffs, rvecs, tvecs, method);
    if (num_of_solutions != (int) rvecs.size() || num_of_solutions != (int) tvecs.size() || num_of_solutions == 0)
      return false;

    bool isTestSuccess = false;
    double error = DBL_MAX;
    for (unsigned int i = 0; i < rvecs.size() && !isTestSuccess; ++i) {
      double rvecDiff = cvtest::norm(rvecs[i], trueRvec, NORM_L2);
      double tvecDiff = cvtest::norm(tvecs[i], trueTvec, NORM_L2);
      isTestSuccess = rvecDiff < epsilon[method] && tvecDiff < epsilon[method];
      error = std::min(error, std::max(rvecDiff, tvecDiff));
    }

    if (error > maxError)
      maxError = error;

    return isTestSuccess;
  }

  virtual void run(int)
  {
    ts->set_failed_test_info(cvtest::TS::OK);

    vector<Point3f> points;
    points.resize(pointsCount);
    generate3DPointCloud(points);

    const int methodsCount = 2;
    int methods[] = {SOLVEPNP_P3P, SOLVEPNP_AP3P};
    RNG rng = ts->get_rng();

    for (int mode = 0; mode < 2; mode++)
    {
      for (int method = 0; method < methodsCount; method++)
      {
        double maxError = 0;
        int successfulTestsCount = 0;
        for (int testIndex = 0; testIndex < totalTestsCount; testIndex++)
        {
          if (runTest(rng, mode, methods[method], points, eps, maxError))
            successfulTestsCount++;
        }
        if (successfulTestsCount < 0.7*totalTestsCount)
        {
          ts->printf( cvtest::TS::LOG, "Invalid accuracy for method %d, failed %d tests from %d, maximum error equals %f, distortion mode equals %d\n",
                      method, totalTestsCount - successfulTestsCount, totalTestsCount, maxError, mode);
          ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
        }
        cout << "mode: " << mode << ", method: " << method << " -> "
             << ((double)successfulTestsCount / totalTestsCount) * 100 << "%"
             << " (err < " << maxError << ")" << endl;
      }
    }
  }
};


TEST(Calib3d_SolveP3P, accuracy) { CV_solveP3P_Test test; test.safe_run();}
TEST(Calib3d_SolvePnPRansac, accuracy) { CV_solvePnPRansac_Test test; test.safe_run(); }
TEST(Calib3d_SolvePnP, accuracy) { CV_solvePnP_Test test; test.safe_run(); }

TEST(Calib3d_SolvePnPRansac, concurrency)
{
    int count = 7*13;

    Mat object(1, count, CV_32FC3);
    randu(object, -100, 100);

    Mat camera_mat(3, 3, CV_32FC1);
    randu(camera_mat, 0.5, 1);
    camera_mat.at<float>(0, 1) = 0.f;
    camera_mat.at<float>(1, 0) = 0.f;
    camera_mat.at<float>(2, 0) = 0.f;
    camera_mat.at<float>(2, 1) = 0.f;

    Mat dist_coef(1, 8, CV_32F, cv::Scalar::all(0));

    vector<cv::Point2f> image_vec;
    Mat rvec_gold(1, 3, CV_32FC1);
    randu(rvec_gold, 0, 1);
    Mat tvec_gold(1, 3, CV_32FC1);
    randu(tvec_gold, 0, 1);
    projectPoints(object, rvec_gold, tvec_gold, camera_mat, dist_coef, image_vec);

    Mat image(1, count, CV_32FC2, &image_vec[0]);

    Mat rvec1, rvec2;
    Mat tvec1, tvec2;

    int threads = getNumThreads();
    {
        // limit concurrency to get deterministic result
        theRNG().state = 20121010;
        setNumThreads(1);
        solvePnPRansac(object, image, camera_mat, dist_coef, rvec1, tvec1);
    }

    {
        setNumThreads(threads);
        Mat rvec;
        Mat tvec;
        // parallel executions
        for(int i = 0; i < 10; ++i)
        {
            cv::theRNG().state = 20121010;
            solvePnPRansac(object, image, camera_mat, dist_coef, rvec, tvec);
        }
    }

    {
        // single thread again
        theRNG().state = 20121010;
        setNumThreads(1);
        solvePnPRansac(object, image, camera_mat, dist_coef, rvec2, tvec2);
    }

    double rnorm = cvtest::norm(rvec1, rvec2, NORM_INF);
    double tnorm = cvtest::norm(tvec1, tvec2, NORM_INF);

    EXPECT_LT(rnorm, 1e-6);
    EXPECT_LT(tnorm, 1e-6);
}

TEST(Calib3d_SolvePnPRansac, input_type)
{
    const int numPoints = 10;
    Matx33d intrinsics(5.4794130238156129e+002, 0., 2.9835545700043139e+002, 0.,
        5.4817724002728005e+002, 2.3062194051986233e+002, 0., 0., 1.);

    std::vector<cv::Point3f> points3d;
    std::vector<cv::Point2f> points2d;
    for (int i = 0; i < numPoints; i+=2)
    {
        points3d.push_back(cv::Point3i(5+i, 3, 2));
        points3d.push_back(cv::Point3i(5+i, 3+i, 2+i));
        points2d.push_back(cv::Point2i(0, i));
        points2d.push_back(cv::Point2i(-i, i));
    }
    Mat R1, t1, R2, t2, R3, t3, R4, t4;

    EXPECT_TRUE(solvePnPRansac(points3d, points2d, intrinsics, cv::Mat(), R1, t1));

    Mat points3dMat(points3d);
    Mat points2dMat(points2d);
    EXPECT_TRUE(solvePnPRansac(points3dMat, points2dMat, intrinsics, cv::Mat(), R2, t2));

    points3dMat = points3dMat.reshape(3, 1);
    points2dMat = points2dMat.reshape(2, 1);
    EXPECT_TRUE(solvePnPRansac(points3dMat, points2dMat, intrinsics, cv::Mat(), R3, t3));

    points3dMat = points3dMat.reshape(1, numPoints);
    points2dMat = points2dMat.reshape(1, numPoints);
    EXPECT_TRUE(solvePnPRansac(points3dMat, points2dMat, intrinsics, cv::Mat(), R4, t4));

    EXPECT_LE(cvtest::norm(R1, R2, NORM_INF), 1e-6);
    EXPECT_LE(cvtest::norm(t1, t2, NORM_INF), 1e-6);
    EXPECT_LE(cvtest::norm(R1, R3, NORM_INF), 1e-6);
    EXPECT_LE(cvtest::norm(t1, t3, NORM_INF), 1e-6);
    EXPECT_LE(cvtest::norm(R1, R4, NORM_INF), 1e-6);
    EXPECT_LE(cvtest::norm(t1, t4, NORM_INF), 1e-6);
}

TEST(Calib3d_SolvePnP, double_support)
{
    Matx33d intrinsics(5.4794130238156129e+002, 0., 2.9835545700043139e+002, 0.,
                       5.4817724002728005e+002, 2.3062194051986233e+002, 0., 0., 1.);
    std::vector<cv::Point3d> points3d;
    std::vector<cv::Point2d> points2d;
    std::vector<cv::Point3f> points3dF;
    std::vector<cv::Point2f> points2dF;
    for (int i = 0; i < 10 ; i+=2)
    {
        points3d.push_back(cv::Point3d(5+i, 3, 2));
        points3dF.push_back(cv::Point3d(5+i, 3, 2));
        points3d.push_back(cv::Point3d(5+i, 3+i, 2+i));
        points3dF.push_back(cv::Point3d(5+i, 3+i, 2+i));
        points2d.push_back(cv::Point2d(0, i));
        points2dF.push_back(cv::Point2d(0, i));
        points2d.push_back(cv::Point2d(-i, i));
        points2dF.push_back(cv::Point2d(-i, i));
    }
    Mat R,t, RF, tF;
    vector<int> inliers;

    solvePnPRansac(points3dF, points2dF, intrinsics, cv::Mat(), RF, tF, true, 100, 8.f, 0.999, inliers, cv::SOLVEPNP_P3P);
    solvePnPRansac(points3d, points2d, intrinsics, cv::Mat(), R, t, true, 100, 8.f, 0.999, inliers, cv::SOLVEPNP_P3P);

    EXPECT_LE(cvtest::norm(R, Mat_<double>(RF), NORM_INF), 1e-3);
    EXPECT_LE(cvtest::norm(t, Mat_<double>(tF), NORM_INF), 1e-3);
}

TEST(Calib3d_SolvePnP, translation)
{
    Mat cameraIntrinsic = Mat::eye(3,3, CV_32FC1);
    vector<float> crvec;
    crvec.push_back(0.f);
    crvec.push_back(0.f);
    crvec.push_back(0.f);
    vector<float> ctvec;
    ctvec.push_back(100.f);
    ctvec.push_back(100.f);
    ctvec.push_back(0.f);
    vector<Point3f> p3d;
    p3d.push_back(Point3f(0,0,0));
    p3d.push_back(Point3f(0,0,10));
    p3d.push_back(Point3f(0,10,10));
    p3d.push_back(Point3f(10,10,10));
    p3d.push_back(Point3f(2,5,5));

    vector<Point2f> p2d;
    projectPoints(p3d, crvec, ctvec, cameraIntrinsic, noArray(), p2d);
    Mat rvec;
    Mat tvec;
    rvec =(Mat_<float>(3,1) << 0, 0, 0);
    tvec = (Mat_<float>(3,1) << 100, 100, 0);

    solvePnP(p3d, p2d, cameraIntrinsic, noArray(), rvec, tvec, true);
    EXPECT_TRUE(checkRange(rvec));
    EXPECT_TRUE(checkRange(tvec));

    rvec =(Mat_<double>(3,1) << 0, 0, 0);
    tvec = (Mat_<double>(3,1) << 100, 100, 0);
    solvePnP(p3d, p2d, cameraIntrinsic, noArray(), rvec, tvec, true);
    EXPECT_TRUE(checkRange(rvec));
    EXPECT_TRUE(checkRange(tvec));

    solvePnP(p3d, p2d, cameraIntrinsic, noArray(), rvec, tvec, false);
    EXPECT_TRUE(checkRange(rvec));
    EXPECT_TRUE(checkRange(tvec));
}

TEST(Calib3d_SolvePnP, iterativeInitialGuess3pts)
{
    {
        Matx33d intrinsics(605.4, 0.0, 317.35,
                           0.0, 601.2, 242.63,
                           0.0, 0.0, 1.0);

        double L = 0.1;
        vector<Point3d> p3d;
        p3d.push_back(Point3d(-L, -L, 0.0));
        p3d.push_back(Point3d(L, -L, 0.0));
        p3d.push_back(Point3d(L, L, 0.0));

        Mat rvec_ground_truth = (Mat_<double>(3,1) << 0.3, -0.2, 0.75);
        Mat tvec_ground_truth = (Mat_<double>(3,1) << 0.15, -0.2, 1.5);

        vector<Point2d> p2d;
        projectPoints(p3d, rvec_ground_truth, tvec_ground_truth, intrinsics, noArray(), p2d);

        Mat rvec_est = (Mat_<double>(3,1) << 0.2, -0.1, 0.6);
        Mat tvec_est = (Mat_<double>(3,1) << 0.05, -0.05, 1.0);

        solvePnP(p3d, p2d, intrinsics, noArray(), rvec_est, tvec_est, true, SOLVEPNP_ITERATIVE);

        std::cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
        std::cout << "rvec_est: " << rvec_est.t() << std::endl;
        std::cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
        std::cout << "tvec_est: " << tvec_est.t() << std::endl;

        EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-6);
        EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-6);
    }

    {
        Matx33f intrinsics(605.4f, 0.0f, 317.35f,
                           0.0f, 601.2f, 242.63f,
                           0.0f, 0.0f, 1.0f);

        float L = 0.1f;
        vector<Point3f> p3d;
        p3d.push_back(Point3f(-L, -L, 0.0f));
        p3d.push_back(Point3f(L, -L, 0.0f));
        p3d.push_back(Point3f(L, L, 0.0f));

        Mat rvec_ground_truth = (Mat_<float>(3,1) << -0.75f, 0.4f, 0.34f);
        Mat tvec_ground_truth = (Mat_<float>(3,1) << -0.15f, 0.35f, 1.58f);

        vector<Point2f> p2d;
        projectPoints(p3d, rvec_ground_truth, tvec_ground_truth, intrinsics, noArray(), p2d);

        Mat rvec_est = (Mat_<float>(3,1) << -0.5f, 0.2f, 0.2f);
        Mat tvec_est = (Mat_<float>(3,1) << 0.0f, 0.2f, 1.0f);

        solvePnP(p3d, p2d, intrinsics, noArray(), rvec_est, tvec_est, true, SOLVEPNP_ITERATIVE);

        std::cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
        std::cout << "rvec_est: " << rvec_est.t() << std::endl;
        std::cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
        std::cout << "tvec_est: " << tvec_est.t() << std::endl;

        EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-6);
        EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-6);
    }
}

}} // namespace
