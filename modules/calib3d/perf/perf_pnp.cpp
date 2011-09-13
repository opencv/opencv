#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;


typedef std::tr1::tuple<std::string, cv::Size> String_Size_t;
typedef perf::TestBaseWithParam<int> PointsNumber;

PERF_TEST_P(PointsNumber, solvePnP, testing::Values(4, 3*9, 7*13)
             )
{
    int pointsNum = GetParam();

    vector<Point2f> points2d(pointsNum);
    vector<Point3f> points3d(pointsNum);
    Mat rvec = Mat::zeros(3, 1, CV_32FC1);
    Mat tvec = Mat::zeros(3, 1, CV_32FC1);

    Mat distortion = Mat::zeros(5, 1, CV_32FC1);
    Mat intrinsics = Mat::eye(3, 3, CV_32FC1);
    intrinsics.at<float> (0, 0) = 400.0;
    intrinsics.at<float> (1, 1) = 400.0;
    intrinsics.at<float> (0, 2) = 640 / 2;
    intrinsics.at<float> (1, 2) = 480 / 2;

    warmup(points3d, WARMUP_RNG);
    warmup(rvec, WARMUP_RNG);
    warmup(tvec, WARMUP_RNG);

    projectPoints(points3d, rvec, tvec, intrinsics, distortion, points2d);

    //add noise
    Mat noise(1, points2d.size(), CV_32FC2);
    randu(noise, 0, 0.01);
    add(points2d, noise, points2d);

    declare.in(points3d, points2d);

    TEST_CYCLE(1000)
    {
        solvePnP(points3d, points2d, intrinsics, distortion, rvec, tvec, false);
    }

    SANITY_CHECK(rvec);
    SANITY_CHECK(tvec);
}
