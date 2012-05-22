#include "perf_cpu_precomp.hpp"

#ifdef HAVE_CUDA

//////////////////////////////////////////////////////////////////////
// ProjectPoints

GPU_PERF_TEST_1(ProjectPoints, cv::gpu::DeviceInfo)
{
    cv::Mat src(1, 10000, CV_32FC3);

    declare.in(src, WARMUP_RNG);

    cv::Mat dst;

    TEST_CYCLE()
    {
        cv::projectPoints(src, cv::Mat::ones(1, 3, CV_32FC1), cv::Mat::ones(1, 3, CV_32FC1), cv::Mat::ones(3, 3, CV_32FC1), cv::Mat(), dst);
    }
}

INSTANTIATE_TEST_CASE_P(Calib3D, ProjectPoints, ALL_DEVICES);

//////////////////////////////////////////////////////////////////////
// SolvePnPRansac

GPU_PERF_TEST_1(SolvePnPRansac, cv::gpu::DeviceInfo)
{
    cv::Mat object(1, 10000, CV_32FC3);
    cv::Mat image(1, 10000, CV_32FC2);

    declare.in(object, image, WARMUP_RNG);

    cv::Mat rvec, tvec;

    declare.time(3.0);

    TEST_CYCLE()
    {
        cv::solvePnPRansac(object, image, cv::Mat::ones(3, 3, CV_32FC1), cv::Mat(1, 8, CV_32F, cv::Scalar::all(0)), rvec, tvec);
    }
}

INSTANTIATE_TEST_CASE_P(Calib3D, SolvePnPRansac, ALL_DEVICES);

//////////////////////////////////////////////////////////////////////
// StereoBM

GPU_PERF_TEST_1(StereoBM, cv::gpu::DeviceInfo)
{
    cv::Mat img_l = readImage("gpu/perf/aloe.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat img_r = readImage("gpu/perf/aloeR.jpg", cv::IMREAD_GRAYSCALE);

    ASSERT_FALSE(img_l.empty());
    ASSERT_FALSE(img_r.empty());

    cv::Mat dst;

    cv::StereoBM bm(0, 256);

    declare.time(5.0);

    TEST_CYCLE()
    {
        bm(img_l, img_r, dst);
    }
}

INSTANTIATE_TEST_CASE_P(Calib3D, StereoBM, ALL_DEVICES);

#endif

