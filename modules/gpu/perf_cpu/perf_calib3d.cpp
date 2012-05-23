#include "perf_cpu_precomp.hpp"

#ifdef HAVE_CUDA

//////////////////////////////////////////////////////////////////////
// StereoBM

GPU_PERF_TEST_1(StereoBM, cv::gpu::DeviceInfo)
{
    cv::Mat img_l = readImage("gpu/perf/aloe.jpg", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img_l.empty());

    cv::Mat img_r = readImage("gpu/perf/aloeR.jpg", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img_r.empty());

    cv::StereoBM bm(0, 256);

    cv::Mat dst;

    bm(img_l, img_r, dst);

    declare.time(5.0);

    TEST_CYCLE()
    {
        bm(img_l, img_r, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Calib3D, StereoBM, ALL_DEVICES);

//////////////////////////////////////////////////////////////////////
// ProjectPoints

IMPLEMENT_PARAM_CLASS(Count, int)

GPU_PERF_TEST(ProjectPoints, cv::gpu::DeviceInfo, Count)
{
    int count = GET_PARAM(1);

    cv::Mat src(1, count, CV_32FC3);
    fill(src, -100, 100);

    cv::Mat rvec = cv::Mat::ones(1, 3, CV_32FC1);
    cv::Mat tvec = cv::Mat::ones(1, 3, CV_32FC1);
    cv::Mat camera_mat = cv::Mat::ones(3, 3, CV_32FC1);
    cv::Mat dst;

    cv::projectPoints(src, rvec, tvec, camera_mat, cv::noArray(), dst);

    TEST_CYCLE()
    {
        cv::projectPoints(src, rvec, tvec, camera_mat, cv::noArray(), dst);
    }
}

INSTANTIATE_TEST_CASE_P(Calib3D, ProjectPoints, testing::Combine(
    ALL_DEVICES,
    testing::Values<Count>(5000, 10000, 20000)));

//////////////////////////////////////////////////////////////////////
// SolvePnPRansac

GPU_PERF_TEST(SolvePnPRansac, cv::gpu::DeviceInfo, Count)
{
    int count = GET_PARAM(1);

    cv::Mat object(1, count, CV_32FC3);
    fill(object, -100, 100);

    cv::Mat camera_mat(3, 3, CV_32FC1);
    fill(camera_mat, 0.5, 1);
    camera_mat.at<float>(0, 1) = 0.f;
    camera_mat.at<float>(1, 0) = 0.f;
    camera_mat.at<float>(2, 0) = 0.f;
    camera_mat.at<float>(2, 1) = 0.f;

    cv::Mat dist_coef(1, 8, CV_32F, cv::Scalar::all(0));

    std::vector<cv::Point2f> image_vec;
    cv::Mat rvec_gold(1, 3, CV_32FC1);
    fill(rvec_gold, 0, 1);
    cv::Mat tvec_gold(1, 3, CV_32FC1);
    fill(tvec_gold, 0, 1);
    cv::projectPoints(object, rvec_gold, tvec_gold, camera_mat, dist_coef, image_vec);

    cv::Mat image(1, count, CV_32FC2, &image_vec[0]);

    cv::Mat rvec;
    cv::Mat tvec;

    cv::solvePnPRansac(object, image, camera_mat, dist_coef, rvec, tvec);

    declare.time(10.0);

    TEST_CYCLE()
    {
        cv::solvePnPRansac(object, image, camera_mat, dist_coef, rvec, tvec);
    }
}

INSTANTIATE_TEST_CASE_P(Calib3D, SolvePnPRansac, testing::Combine(
    ALL_DEVICES,
    testing::Values<Count>(5000, 10000, 20000)));

//////////////////////////////////////////////////////////////////////
// ReprojectImageTo3D

GPU_PERF_TEST(ReprojectImageTo3D, cv::gpu::DeviceInfo, cv::Size, MatDepth)
{
    cv::Size size = GET_PARAM(1);
    int depth = GET_PARAM(2);

    cv::Mat src(size, depth);
    fill(src, 5.0, 30.0);

    cv::Mat Q(4, 4, CV_32FC1);
    fill(Q, 0.1, 1.0);

    cv::Mat dst;

    cv::reprojectImageTo3D(src, dst, Q);

    TEST_CYCLE()
    {
        cv::reprojectImageTo3D(src, dst, Q);
    }
}

INSTANTIATE_TEST_CASE_P(Calib3D, ReprojectImageTo3D, testing::Combine(
    ALL_DEVICES,
    GPU_TYPICAL_MAT_SIZES,
    testing::Values<MatDepth>(CV_8U, CV_16S)));

#endif

