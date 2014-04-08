#include "test_precomp.hpp"
#include <string>

using namespace cv;
using namespace std;

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
