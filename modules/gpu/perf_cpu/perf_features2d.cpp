#include "perf_cpu_precomp.hpp"

#ifdef HAVE_CUDA

//////////////////////////////////////////////////////////////////////
// BruteForceMatcher_match

GPU_PERF_TEST(BruteForceMatcher_match, cv::gpu::DeviceInfo, int)
{
    int desc_size = GET_PARAM(1);

    cv::Mat query(3000, desc_size, CV_32FC1);
    cv::Mat train(3000, desc_size, CV_32FC1);

    declare.in(query, train, WARMUP_RNG);

    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<cv::DMatch> matches;

    declare.time(10.0);

    TEST_CYCLE()
    {
        matcher.match(query, train, matches);
    }
}

INSTANTIATE_TEST_CASE_P(Features2D, BruteForceMatcher_match, testing::Combine(
                        ALL_DEVICES,
                        testing::Values(64, 128, 256)));

//////////////////////////////////////////////////////////////////////
// BruteForceMatcher_knnMatch

GPU_PERF_TEST(BruteForceMatcher_knnMatch, cv::gpu::DeviceInfo, int, int)
{
    int desc_size = GET_PARAM(1);
    int k = GET_PARAM(2);

    cv::Mat query(3000, desc_size, CV_32FC1);
    cv::Mat train(3000, desc_size, CV_32FC1);

    declare.in(query, train, WARMUP_RNG);

    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector< std::vector<cv::DMatch> > matches;

    declare.time(10.0);

    TEST_CYCLE()
    {
        matcher.knnMatch(query, train, matches, k);
    }
}

INSTANTIATE_TEST_CASE_P(Features2D, BruteForceMatcher_knnMatch, testing::Combine(
                        ALL_DEVICES,
                        testing::Values(64, 128, 256),
                        testing::Values(2, 3)));

//////////////////////////////////////////////////////////////////////
// BruteForceMatcher_radiusMatch

GPU_PERF_TEST(BruteForceMatcher_radiusMatch, cv::gpu::DeviceInfo, int)
{
    int desc_size = GET_PARAM(1);

    cv::Mat query(3000, desc_size, CV_32FC1);
    cv::Mat train(3000, desc_size, CV_32FC1);

    fill(query, 0, 1);
    fill(train, 0, 1);

    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector< std::vector<cv::DMatch> > matches;

    declare.time(10.0);

    TEST_CYCLE()
    {
        matcher.radiusMatch(query, train, matches, 2.0);
    }
}

INSTANTIATE_TEST_CASE_P(Features2D, BruteForceMatcher_radiusMatch, testing::Combine(
                        ALL_DEVICES,
                        testing::Values(64, 128, 256)));

//////////////////////////////////////////////////////////////////////
// SURF

GPU_PERF_TEST_1(SURF, cv::gpu::DeviceInfo)
{
    cv::Mat img = readImage("gpu/perf/aloe.jpg", cv::IMREAD_GRAYSCALE);

    ASSERT_FALSE(img.empty());

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    cv::SURF surf;

    declare.time(30.0);

    TEST_CYCLE()
    {
        surf(img, cv::noArray(), keypoints, descriptors);
    }
}

INSTANTIATE_TEST_CASE_P(Features2D, SURF, DEVICES(cv::gpu::GLOBAL_ATOMICS));

//////////////////////////////////////////////////////////////////////
// FAST

GPU_PERF_TEST_1(FAST, cv::gpu::DeviceInfo)
{
    cv::Mat img = readImage("gpu/perf/aloe.jpg", cv::IMREAD_GRAYSCALE);

    ASSERT_FALSE(img.empty());

    std::vector<cv::KeyPoint> keypoints;

    TEST_CYCLE()
    {
        cv::FAST(img, keypoints, 20);
    }
}

INSTANTIATE_TEST_CASE_P(Features2D, FAST, DEVICES(cv::gpu::GLOBAL_ATOMICS));

//////////////////////////////////////////////////////////////////////
// ORB

GPU_PERF_TEST_1(ORB, cv::gpu::DeviceInfo)
{
    cv::Mat img = readImage("gpu/perf/aloe.jpg", cv::IMREAD_GRAYSCALE);

    ASSERT_FALSE(img.empty());

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    cv::ORB orb(4000);

    TEST_CYCLE()
    {
        orb(img, cv::noArray(), keypoints, descriptors);
    }
}

INSTANTIATE_TEST_CASE_P(Features2D, ORB, DEVICES(cv::gpu::GLOBAL_ATOMICS));

#endif
