#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

CV_ENUM(pnpAlgo, SOLVEPNP_ITERATIVE, SOLVEPNP_EPNP, SOLVEPNP_P3P, SOLVEPNP_DLS, SOLVEPNP_UPNP)

typedef tuple<int, pnpAlgo> PointsNum_Algo_t;
typedef perf::TestBaseWithParam<PointsNum_Algo_t> PointsNum_Algo;

typedef perf::TestBaseWithParam<int> PointsNum;

PERF_TEST_P(PointsNum_Algo, solvePnP,
            testing::Combine(
                testing::Values(5, 3*9, 7*13), //TODO: find why results on 4 points are too unstable
                testing::Values((int)SOLVEPNP_ITERATIVE, (int)SOLVEPNP_EPNP, (int)SOLVEPNP_UPNP, (int)SOLVEPNP_DLS)
                )
            )
{
    int pointsNum = get<0>(GetParam());
    pnpAlgo algo = get<1>(GetParam());

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
    Mat noise(1, (int)points2d.size(), CV_32FC2);
    randu(noise, 0, 0.01);
    cv::add(points2d, noise, points2d);

    declare.in(points3d, points2d);
    declare.time(100);

    TEST_CYCLE_N(1000)
    {
        cv::solvePnP(points3d, points2d, intrinsics, distortion, rvec, tvec, false, algo);
    }

    SANITY_CHECK(rvec, 1e-6);
    SANITY_CHECK(tvec, 1e-6);
}

PERF_TEST_P(PointsNum_Algo, solvePnPSmallPoints,
            testing::Combine(
                testing::Values(5),
                testing::Values((int)SOLVEPNP_P3P, (int)SOLVEPNP_EPNP, (int)SOLVEPNP_DLS, (int)SOLVEPNP_UPNP)
                )
            )
{
    int pointsNum = get<0>(GetParam());
    pnpAlgo algo = get<1>(GetParam());
    if( algo == SOLVEPNP_P3P )
        pointsNum = 4;

    vector<Point2f> points2d(pointsNum);
    vector<Point3f> points3d(pointsNum);
    Mat rvec = Mat::zeros(3, 1, CV_32FC1);
    Mat tvec = Mat::zeros(3, 1, CV_32FC1);

    Mat distortion = Mat::zeros(5, 1, CV_32FC1);
    Mat intrinsics = Mat::eye(3, 3, CV_32FC1);
    intrinsics.at<float> (0, 0) = 400.0f;
    intrinsics.at<float> (1, 1) = 400.0f;
    intrinsics.at<float> (0, 2) = 640 / 2;
    intrinsics.at<float> (1, 2) = 480 / 2;

    warmup(points3d, WARMUP_RNG);
    warmup(rvec, WARMUP_RNG);
    warmup(tvec, WARMUP_RNG);

    // normalize Rodrigues vector
    Mat rvec_tmp = Mat::eye(3, 3, CV_32F);
    cv::Rodrigues(rvec, rvec_tmp);
    cv::Rodrigues(rvec_tmp, rvec);

    cv::projectPoints(points3d, rvec, tvec, intrinsics, distortion, points2d);

    //add noise
    Mat noise(1, (int)points2d.size(), CV_32FC2);
    randu(noise, -0.001, 0.001);
    cv::add(points2d, noise, points2d);

    declare.in(points3d, points2d);
    declare.time(100);

    TEST_CYCLE_N(1000)
    {
        cv::solvePnP(points3d, points2d, intrinsics, distortion, rvec, tvec, false, algo);
    }

    SANITY_CHECK(rvec, 1e-1);
    SANITY_CHECK(tvec, 1e-2);
}

PERF_TEST_P(PointsNum, DISABLED_SolvePnPRansac, testing::Values(5, 3*9, 7*13))
{
    int count = GetParam();

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

    Mat rvec;
    Mat tvec;

    TEST_CYCLE()
    {
        cv::solvePnPRansac(object, image, camera_mat, dist_coef, rvec, tvec);
    }

    SANITY_CHECK(rvec, 1e-6);
    SANITY_CHECK(tvec, 1e-6);
}

} // namespace
