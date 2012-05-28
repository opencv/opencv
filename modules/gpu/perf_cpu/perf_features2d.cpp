#include "perf_cpu_precomp.hpp"

#ifdef HAVE_CUDA

//////////////////////////////////////////////////////////////////////
// SURF

GPU_PERF_TEST_1(SURF, cv::gpu::DeviceInfo)
{
    cv::Mat img = readImage("gpu/perf/aloe.jpg", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    cv::SURF surf;

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    surf(img, cv::noArray(), keypoints, descriptors);

    declare.time(50.0);

    TEST_CYCLE()
    {
        keypoints.clear();
        surf(img, cv::noArray(), keypoints, descriptors);
    }
}

INSTANTIATE_TEST_CASE_P(Features2D, SURF, ALL_DEVICES);

//////////////////////////////////////////////////////////////////////
// FAST

GPU_PERF_TEST_1(FAST, cv::gpu::DeviceInfo)
{
    cv::Mat img = readImage("gpu/perf/aloe.jpg", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    std::vector<cv::KeyPoint> keypoints;

    cv::FAST(img, keypoints, 20);

    TEST_CYCLE()
    {
        keypoints.clear();
        cv::FAST(img, keypoints, 20);
    }
}

INSTANTIATE_TEST_CASE_P(Features2D, FAST, ALL_DEVICES);

//////////////////////////////////////////////////////////////////////
// ORB

GPU_PERF_TEST_1(ORB, cv::gpu::DeviceInfo)
{
    cv::Mat img = readImage("gpu/perf/aloe.jpg", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    cv::ORB orb(4000);

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    orb(img, cv::noArray(), keypoints, descriptors);

    TEST_CYCLE()
    {
        keypoints.clear();
        orb(img, cv::noArray(), keypoints, descriptors);
    }
}

INSTANTIATE_TEST_CASE_P(Features2D, ORB, ALL_DEVICES);

//////////////////////////////////////////////////////////////////////
// BruteForceMatcher_match

IMPLEMENT_PARAM_CLASS(DescriptorSize, int)

GPU_PERF_TEST(BruteForceMatcher_match, cv::gpu::DeviceInfo, DescriptorSize, NormType)
{
    int desc_size = GET_PARAM(1);
    int normType = GET_PARAM(2);

    int type = normType == cv::NORM_HAMMING ? CV_8U : CV_32F;

    cv::Mat query(3000, desc_size, type);
    fill(query, 0.0, 10.0);

    cv::Mat train(3000, desc_size, type);
    fill(train, 0.0, 10.0);

    cv::BFMatcher matcher(normType);

    std::vector<cv::DMatch> matches;

    matcher.match(query, train, matches);

    declare.time(20.0);

    TEST_CYCLE()
    {
        matcher.match(query, train, matches);
    }
}

INSTANTIATE_TEST_CASE_P(Features2D, BruteForceMatcher_match, testing::Combine(
    ALL_DEVICES,
    testing::Values(DescriptorSize(64), DescriptorSize(128), DescriptorSize(256)),
    testing::Values(NormType(cv::NORM_L1), NormType(cv::NORM_L2), NormType(cv::NORM_HAMMING))));

//////////////////////////////////////////////////////////////////////
// BruteForceMatcher_knnMatch

IMPLEMENT_PARAM_CLASS(K, int)

GPU_PERF_TEST(BruteForceMatcher_knnMatch, cv::gpu::DeviceInfo, DescriptorSize, K, NormType)
{
    int desc_size = GET_PARAM(1);
    int k = GET_PARAM(2);
    int normType = GET_PARAM(3);

    int type = normType == cv::NORM_HAMMING ? CV_8U : CV_32F;

    cv::Mat query(3000, desc_size, type);
    fill(query, 0.0, 10.0);

    cv::Mat train(3000, desc_size, type);
    fill(train, 0.0, 10.0);

    cv::BFMatcher matcher(normType);

    std::vector< std::vector<cv::DMatch> > matches;

    matcher.knnMatch(query, train, matches, k);

    declare.time(30.0);

    TEST_CYCLE()
    {
        matcher.knnMatch(query, train, matches, k);
    }
}

INSTANTIATE_TEST_CASE_P(Features2D, BruteForceMatcher_knnMatch, testing::Combine(
    ALL_DEVICES,
    testing::Values(DescriptorSize(64), DescriptorSize(128), DescriptorSize(256)),
    testing::Values(K(2), K(3)),
    testing::Values(NormType(cv::NORM_L1), NormType(cv::NORM_L2), NormType(cv::NORM_HAMMING))));

//////////////////////////////////////////////////////////////////////
// BruteForceMatcher_radiusMatch

GPU_PERF_TEST(BruteForceMatcher_radiusMatch, cv::gpu::DeviceInfo, DescriptorSize, NormType)
{
    int desc_size = GET_PARAM(1);
    int normType = GET_PARAM(2);

    int type = normType == cv::NORM_HAMMING ? CV_8U : CV_32F;

    cv::Mat query(3000, desc_size, type);
    fill(query, 0.0, 1.0);

    cv::Mat train(3000, desc_size, type);
    fill(train, 0.0, 1.0);

    cv::BFMatcher matcher(normType);

    std::vector< std::vector<cv::DMatch> > matches;

    matcher.radiusMatch(query, train, matches, 2.0);

    declare.time(30.0);

    TEST_CYCLE()
    {
        matcher.radiusMatch(query, train, matches, 2.0);
    }
}

INSTANTIATE_TEST_CASE_P(Features2D, BruteForceMatcher_radiusMatch, testing::Combine(
    ALL_DEVICES,
    testing::Values(DescriptorSize(64), DescriptorSize(128), DescriptorSize(256)),
    testing::Values(NormType(cv::NORM_L1), NormType(cv::NORM_L2), NormType(cv::NORM_HAMMING))));

#endif
