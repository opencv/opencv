// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

class CV_UndistortTest : public cvtest::BaseTest
{
public:
    CV_UndistortTest();
    ~CV_UndistortTest();
protected:
    void run(int);
private:
    void generate3DPointCloud(vector<Point3f>& points, Point3f pmin = Point3f(-1,
    -1, 5), Point3f pmax = Point3f(1, 1, 10));
    void generateCameraMatrix(Mat& cameraMatrix);
    void generateDistCoeffs(Mat& distCoeffs, int count);

    double thresh;
    RNG rng;
};

CV_UndistortTest::CV_UndistortTest()
{
    thresh = 1.0e-2;
}
CV_UndistortTest::~CV_UndistortTest() {}

void CV_UndistortTest::generate3DPointCloud(vector<Point3f>& points, Point3f pmin, Point3f pmax)
{
    RNG rng_Point = cv::theRNG(); // fix the seed to use "fixed" input 3D points
    for (size_t i = 0; i < points.size(); i++)
    {
        float _x = rng_Point.uniform(pmin.x, pmax.x);
        float _y = rng_Point.uniform(pmin.y, pmax.y);
        float _z = rng_Point.uniform(pmin.z, pmax.z);
        points[i] = Point3f(_x, _y, _z);
    }
}
void CV_UndistortTest::generateCameraMatrix(Mat& cameraMatrix)
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
void CV_UndistortTest::generateDistCoeffs(Mat& distCoeffs, int count)
{
    distCoeffs = Mat::zeros(count, 1, CV_64FC1);
    for (int i = 0; i < count; i++)
        distCoeffs.at<double>(i,0) = rng.uniform(0.0, 1.0e-3);
}

void CV_UndistortTest::run(int /* start_from */)
{
    Mat intrinsics, distCoeffs;
    generateCameraMatrix(intrinsics);
    vector<Point3f> points(500);
    generate3DPointCloud(points);
    vector<Point2f> projectedPoints;
    projectedPoints.resize(points.size());

    int modelMembersCount[] = {4,5,8};
    for (int idx = 0; idx < 3; idx++)
    {
        generateDistCoeffs(distCoeffs, modelMembersCount[idx]);
        projectPoints(Mat(points), Mat::zeros(3,1,CV_64FC1), Mat::zeros(3,1,CV_64FC1), intrinsics, distCoeffs, projectedPoints);

        vector<Point2f> realUndistortedPoints;
        projectPoints(Mat(points), Mat::zeros(3,1,CV_64FC1), Mat::zeros(3,1,CV_64FC1), intrinsics,  Mat::zeros(4,1,CV_64FC1), realUndistortedPoints);

        Mat undistortedPoints;
        undistortPoints(Mat(projectedPoints), undistortedPoints, intrinsics, distCoeffs);

        Mat p;
        perspectiveTransform(undistortedPoints, p, intrinsics);
        undistortedPoints = p;
        double diff = cvtest::norm(Mat(realUndistortedPoints), undistortedPoints, NORM_L2);
        if (diff > thresh)
        {
            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
            return;
        }
        ts->set_failed_test_info(cvtest::TS::OK);
    }
}

TEST(Calib3d_Undistort, accuracy) { CV_UndistortTest test; test.safe_run(); }

TEST(Calib3d_Undistort, stop_criteria)
{
    Mat cameraMatrix = (Mat_<double>(3,3,CV_64F) << 857.48296979, 0, 968.06224829,
                                                        0, 876.71824265, 556.37145899,
                                                        0, 0, 1);
    Mat distCoeffs = (Mat_<double>(5,1,CV_64F) <<
                      -2.57614020e-01, 8.77086999e-02, -2.56970803e-04, -5.93390389e-04, -1.52194091e-02);
    RNG rng(2);
    Point2d pt_distorted(rng.uniform(0.0, 1920.0), rng.uniform(0.0, 1080.0));
    std::vector<Point2d> pt_distorted_vec;
    pt_distorted_vec.push_back(pt_distorted);
    const double maxError = 1e-6;
    TermCriteria criteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 100, maxError);
    std::vector<Point2d> pt_undist_vec;
    undistortPoints(pt_distorted_vec, pt_undist_vec, cameraMatrix, distCoeffs, noArray(), noArray(), criteria);

    std::vector<Point2d> pt_redistorted_vec;
    std::vector<Point3d> pt_undist_vec_homogeneous;
    pt_undist_vec_homogeneous.push_back( Point3d(pt_undist_vec[0].x, pt_undist_vec[0].y, 1.0) );
    projectPoints(pt_undist_vec_homogeneous, Mat::zeros(3,1,CV_64F), Mat::zeros(3,1,CV_64F), cameraMatrix, distCoeffs, pt_redistorted_vec);
    const double obtainedError = sqrt( pow(pt_distorted.x - pt_redistorted_vec[0].x, 2) + pow(pt_distorted.y - pt_redistorted_vec[0].y, 2) );

    ASSERT_LE(obtainedError, maxError);
}

}} // namespace
