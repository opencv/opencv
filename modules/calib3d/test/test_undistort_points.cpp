// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <opencv2/ts/cuda_test.hpp> // EXPECT_MAT_NEAR
#include "opencv2/core/types.hpp"
#include "test_precomp.hpp"

namespace opencv_test { namespace {

class UndistortPointsTest : public ::testing::Test
{
protected:
    void generate3DPointCloud(vector<Point3f>& points, Point3f pmin = Point3f(-1,
    -1, 5), Point3f pmax = Point3f(1, 1, 10));
    void generateCameraMatrix(Mat& cameraMatrix);
    void generateDistCoeffs(Mat& distCoeffs, int count);
    cv::Mat generateRotationVector();
    std::vector<cv::Point2d> distortPoints(const cv::Mat &cameraMatrix, const cv::Mat &dist, const std::vector<cv::Point2d> &points);

    double thresh = 1.0e-2;
};

void UndistortPointsTest::generate3DPointCloud(vector<Point3f>& points, Point3f pmin, Point3f pmax)
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

void UndistortPointsTest::generateCameraMatrix(Mat& cameraMatrix)
{
    const double fcMinVal = 1e-3;
    const double fcMaxVal = 100;
    cameraMatrix.create(3, 3, CV_64FC1);
    cameraMatrix.setTo(Scalar(0));
    cameraMatrix.at<double>(0,0) = theRNG().uniform(fcMinVal, fcMaxVal);
    cameraMatrix.at<double>(1,1) = theRNG().uniform(fcMinVal, fcMaxVal);
    cameraMatrix.at<double>(0,2) = theRNG().uniform(fcMinVal, fcMaxVal);
    cameraMatrix.at<double>(1,2) = theRNG().uniform(fcMinVal, fcMaxVal);
    cameraMatrix.at<double>(2,2) = 1;
}

void UndistortPointsTest::generateDistCoeffs(Mat& distCoeffs, int count)
{
    distCoeffs = Mat::zeros(count, 1, CV_64FC1);
    for (int i = 0; i < count; i++)
        distCoeffs.at<double>(i,0) = theRNG().uniform(-0.1, 0.1);
}

cv::Mat UndistortPointsTest::generateRotationVector()
{
    Mat rvec(1, 3, CV_64F);
    theRNG().fill(rvec, RNG::UNIFORM, -0.2, 0.2);

    return rvec;
}

std::vector<cv::Point2d> UndistortPointsTest::distortPoints(const cv::Mat &cameraMatrix, const cv::Mat &dist, const std::vector<cv::Point2d> &points)
{
    CV_Assert(cameraMatrix.rows == 3 && cameraMatrix.cols == 3);
    CV_Assert(cameraMatrix.type() == CV_64F);
    CV_Assert(dist.rows * dist.cols == 12);
    CV_Assert(dist.type() == CV_64F);
    double *k = reinterpret_cast<double *>(dist.data);
    double fx = cameraMatrix.at<double>(0, 0);
    double fy = cameraMatrix.at<double>(1, 1);
    double cx = cameraMatrix.at<double>(0, 2);
    double cy = cameraMatrix.at<double>(1, 2);
    std::vector<cv::Point2d> distortedPoints;
    distortedPoints.reserve(points.size());

    for (const cv::Point2d p : points) {
        double x = (p.x - cx) / fx;
        double y = (p.y - cy) / fy;
        double r2 = x*x + y*y;
        double cdist = (1 + ((k[4]*r2 + k[1])*r2 + k[0])*r2)/(1 + ((k[7]*r2 + k[6])*r2 + k[5])*r2);
        CV_Assert(cdist >= 0);
        double deltaX = 2*k[2]*x*y + k[3]*(r2 + 2*x*x)+ k[8]*r2+k[9]*r2*r2;
        double deltaY = k[2]*(r2 + 2*y*y) + 2*k[3]*x*y+ k[10]*r2+k[11]*r2*r2;
        distortedPoints.push_back(cv::Point2d((x * cdist + deltaX) * fx + cx, (y * cdist + deltaY) * fy + cy));
    }

    return distortedPoints;
}

TEST_F(UndistortPointsTest, accuracy)
{
    Mat intrinsics, distCoeffs;
    generateCameraMatrix(intrinsics);

    vector<Point3f> points(500);
    generate3DPointCloud(points);

    Mat rvec = generateRotationVector();
    Mat R;
    cv::Rodrigues(rvec, R);


    int modelMembersCount[] = {4,5,8};
    for (int idx = 0; idx < 3; idx++)
    {
        generateDistCoeffs(distCoeffs, modelMembersCount[idx]);

        /* Project points with distortion */
        vector<Point2f> projectedPoints;
        projectPoints(Mat(points), Mat::zeros(3,1,CV_64FC1),
                      Mat::zeros(3,1,CV_64FC1), intrinsics,
                      distCoeffs, projectedPoints);

        /* Project points without distortion */
        vector<Point2f> realUndistortedPoints;
        projectPoints(Mat(points), rvec,
                      Mat::zeros(3,1,CV_64FC1), intrinsics,
                      Mat::zeros(4,1,CV_64FC1), realUndistortedPoints);

        /* Undistort points */
        Mat undistortedPoints;
        undistortPoints(Mat(projectedPoints), undistortedPoints, intrinsics, distCoeffs, R, intrinsics);

        EXPECT_MAT_NEAR(realUndistortedPoints, undistortedPoints.t(), thresh);
    }
}

TEST_F(UndistortPointsTest, undistortImagePointsAccuracy)
{
    Mat intrinsics, distCoeffs;
    generateCameraMatrix(intrinsics);

    vector<Point3f> points(500);
    generate3DPointCloud(points);


    int modelMembersCount[] = {4,5,8};
    for (int idx = 0; idx < 3; idx++)
    {
        generateDistCoeffs(distCoeffs, modelMembersCount[idx]);

        /* Project points with distortion */
        vector<Point2f> projectedPoints;
        projectPoints(Mat(points), Mat::zeros(3,1,CV_64FC1),
                      Mat::zeros(3,1,CV_64FC1), intrinsics,
                      distCoeffs, projectedPoints);

        /* Project points without distortion */
        vector<Point2f> realUndistortedPoints;
        projectPoints(Mat(points), Mat::zeros(3, 1, CV_64FC1),
                      Mat::zeros(3,1,CV_64FC1), intrinsics,
                      Mat::zeros(4,1,CV_64FC1), realUndistortedPoints);

        /* Undistort points */
        Mat undistortedPoints;
        TermCriteria termCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 5, thresh / 2);
        undistortImagePoints(Mat(projectedPoints), undistortedPoints, intrinsics, distCoeffs,
                             termCriteria);

        EXPECT_MAT_NEAR(realUndistortedPoints, undistortedPoints.t(), thresh);
    }
}


TEST_F(UndistortPointsTest, stop_criteria)
{
    Mat cameraMatrix = (Mat_<double>(3,3,CV_64F) << 857.48296979, 0, 968.06224829,
                                                        0, 876.71824265, 556.37145899,
                                                        0, 0, 1);
    Mat distCoeffs = (Mat_<double>(5,1,CV_64F) <<
                      -2.57614020e-01, 8.77086999e-02, -2.56970803e-04, -5.93390389e-04, -1.52194091e-02);

    Point2d pt_distorted(theRNG().uniform(0.0, 1920.0), theRNG().uniform(0.0, 1080.0));

    std::vector<Point2d> pt_distorted_vec;
    pt_distorted_vec.push_back(pt_distorted);

    const double maxError = 1e-6;
    TermCriteria criteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 100, maxError);

    std::vector<Point2d> pt_undist_vec;
    Mat rVec = Mat(Matx31d(0.1, -0.2, 0.2));
    Mat R;
    cv::Rodrigues(rVec, R);

    undistortPoints(pt_distorted_vec, pt_undist_vec, cameraMatrix, distCoeffs, R, noArray(), criteria);

    std::vector<Point3d> pt_undist_vec_homogeneous;
    pt_undist_vec_homogeneous.emplace_back(pt_undist_vec[0].x, pt_undist_vec[0].y, 1.0 );

    std::vector<Point2d> pt_redistorted_vec;
    projectPoints(pt_undist_vec_homogeneous, -rVec,
                  Mat::zeros(3,1,CV_64F), cameraMatrix, distCoeffs, pt_redistorted_vec);

    const double obtainedError = sqrt( pow(pt_distorted.x - pt_redistorted_vec[0].x, 2) + pow(pt_distorted.y - pt_redistorted_vec[0].y, 2) );

    ASSERT_LE(obtainedError, maxError);
}

TEST_F(UndistortPointsTest, regression_14583)
{
    const int col = 720;
    // const int row = 540;
    float camera_matrix_value[] = {
        437.8995f, 0.0f, 342.9241f,
        0.0f, 438.8216f, 273.7163f,
        0.0f, 0.0f,      1.0f
    };
    cv::Mat camera_interior(3, 3, CV_32F, camera_matrix_value);

    float camera_distort_value[] = {-0.34329f, 0.11431f, 0.0f, 0.0f, -0.017375f};
    cv::Mat camera_distort(1, 5, CV_32F, camera_distort_value);

    float distort_points_value[] = {col, 0.};
    cv::Mat distort_pt(1, 1, CV_32FC2, distort_points_value);

    cv::Mat undistort_pt;
    cv::undistortPoints(distort_pt, undistort_pt, camera_interior,
                        camera_distort, cv::Mat(), camera_interior);

    EXPECT_NEAR(distort_pt.at<Vec2f>(0)[0], undistort_pt.at<Vec2f>(0)[0], col / 2)
        << "distort point: " << distort_pt << std::endl
        << "undistort point: " << undistort_pt;
}

TEST_F(UndistortPointsTest, regression_27916)
{
    cv::Mat K = (cv::Mat_<double>(3, 3) <<
        1570.8956145992222, 0., 744.87337646727406, 0.,
        1570.3494207432338, 575.55087456337526, 0., 0., 1.);
    cv::Mat dist = (cv::Mat_<double>(1, 12) <<
        -2.8247717583453804, -0.80078070764368037,
        -0.014595359484103326, 0.0018820998949700702, 1.9827795585249783,
        -2.7306773773930897, -1.217725820479524, 2.4052243546080136,
        -0.0020670359760441713, 3.4660880793174063e-05,
        0.014100351510458799, -3.0935329736207612e-05);

    const cv::TermCriteria termCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 100, thresh / 2);
    std::vector<cv::Point2d> distortedPoints, distortedPoints2;
    std::vector<cv::Point2d> undistortedPoints;

    for (int i = 0; i < 50; i++)
    {
        for (int j = 0; j < 50; j++)
        {
            distortedPoints.push_back(cv::Point2d(i, j));
        }
    }

    cv::undistortPoints(distortedPoints, undistortedPoints, K, dist, cv::noArray(), K, termCriteria);
    distortedPoints2 = distortPoints(K, dist, undistortedPoints);
    EXPECT_MAT_NEAR(distortedPoints2, distortedPoints, thresh);
}

}} // namespace
