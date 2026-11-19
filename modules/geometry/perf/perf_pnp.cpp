#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

CV_ENUM(pnpAlgo, SOLVEPNP_ITERATIVE, SOLVEPNP_EPNP, SOLVEPNP_P3P)

typedef tuple<int, pnpAlgo> PointsNum_Algo_t;
typedef perf::TestBaseWithParam<PointsNum_Algo_t> PointsNum_Algo;

typedef perf::TestBaseWithParam<int> PointsNum;

PERF_TEST_P(PointsNum_Algo, solvePnP,
            testing::Combine( //When non planar, DLT needs at least 6 points for SOLVEPNP_ITERATIVE flag
                testing::Values(6, 3*9, 7*13), //TODO: find why results on 4 points are too unstable
                testing::Values((int)SOLVEPNP_ITERATIVE, (int)SOLVEPNP_EPNP)
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
    int sz = (int)points2d.size();
    Mat noise(1, &sz, CV_32FC2);
    randu(noise, 0, 0.01);
    cv::add(points2d, noise, points2d);

    declare.in(points3d, points2d);
    declare.time(100);

    TEST_CYCLE_N(1000)
    {
        cv::solvePnP(points3d, points2d, intrinsics, distortion, rvec, tvec, false, algo);
    }

    SANITY_CHECK(rvec, 1e-4);
    // the check is relaxed from 1e-4 to 2e-2 after LevMarq replacement (#21018),
    // then to 3e-2 after the cv::Mat broadcasting element-wise engine rewrite (#29426)
    SANITY_CHECK(tvec, 3e-2);
}

PERF_TEST_P(PointsNum_Algo, solvePnPSmallPoints,
            testing::Combine(
                testing::Values(5),
                testing::Values((int)SOLVEPNP_P3P, (int)SOLVEPNP_EPNP)
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
    int npoints = (int)points2d.size();
    Mat noise(1, &npoints, CV_32FC2);
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

PERF_TEST_P(PointsNum, SolvePnPRansac, testing::Values(5, 3*9, 7*13))
{
    int count = GetParam();

    Mat object(1, count, CV_32FC3);
    randu(object, -100, 100);

    // same camera as solvePnP above; a random one with a focal length below 1 px lets any pose pass the RANSAC threshold
    Mat camera_mat = Mat_<float>({3, 3}, {400.f, 0.f, 320.f, 0.f, 400.f, 240.f, 0.f, 0.f, 1.f});

    Mat dist_coef(1, 8, CV_32F, cv::Scalar::all(0));

    vector<cv::Point2f> image_vec;

    Mat rvec_gold(1, 3, CV_32FC1);
    randu(rvec_gold, 0, 1);

    Mat tvec_gold(1, 3, CV_32FC1);
    randu(tvec_gold, 0, 1);
    tvec_gold.at<float>(2) += 500.f;  // keep the whole object in front of the camera
    projectPoints(object, rvec_gold, tvec_gold, camera_mat, dist_coef, image_vec);

    Mat image(1, count, CV_32FC2, &image_vec[0]);

    Mat rvec;
    Mat tvec;
    bool found = false;

    TEST_CYCLE()
    {
        found = cv::solvePnPRansac(object, image, camera_mat, dist_coef, rvec, tvec);
    }

    // the projections are noise free, so RANSAC must find the generating pose again
    EXPECT_TRUE(found);
    Mat rvec_gold64, tvec_gold64;
    rvec_gold.reshape(1, 3).convertTo(rvec_gold64, CV_64F);
    tvec_gold.reshape(1, 3).convertTo(tvec_gold64, CV_64F);
    EXPECT_LE(cv::norm(rvec, rvec_gold64, NORM_INF), 1e-4);
    EXPECT_LE(cv::norm(tvec, tvec_gold64, NORM_INF), 1e-2);

    SANITY_CHECK_NOTHING();
}

} // namespace
