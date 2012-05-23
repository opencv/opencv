#include "perf_precomp.hpp"

#ifdef HAVE_CUDA

//////////////////////////////////////////////////////////////////////
// SURF

GPU_PERF_TEST_1(SURF, cv::gpu::DeviceInfo)
{
    cv::gpu::DeviceInfo devInfo = GetParam();
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat img_host = readImage("gpu/perf/aloe.jpg", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img_host.empty());

    cv::gpu::SURF_GPU surf;

    cv::gpu::GpuMat img(img_host);
    cv::gpu::GpuMat keypoints, descriptors;

    surf(img, cv::gpu::GpuMat(), keypoints, descriptors);

    declare.time(2.0);

    TEST_CYCLE()
    {
        surf(img, cv::gpu::GpuMat(), keypoints, descriptors);
    }
}

INSTANTIATE_TEST_CASE_P(Features2D, SURF, ALL_DEVICES);

//////////////////////////////////////////////////////////////////////
// FAST

GPU_PERF_TEST_1(FAST, cv::gpu::DeviceInfo)
{
    cv::gpu::DeviceInfo devInfo = GetParam();
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat img_host = readImage("gpu/perf/aloe.jpg", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img_host.empty());

    cv::gpu::FAST_GPU fast(20);

    cv::gpu::GpuMat img(img_host);
    cv::gpu::GpuMat keypoints;

    fast(img, cv::gpu::GpuMat(), keypoints);

    TEST_CYCLE()
    {
        fast(img, cv::gpu::GpuMat(), keypoints);
    }
}

INSTANTIATE_TEST_CASE_P(Features2D, FAST, ALL_DEVICES);

//////////////////////////////////////////////////////////////////////
// ORB

GPU_PERF_TEST_1(ORB, cv::gpu::DeviceInfo)
{
    cv::gpu::DeviceInfo devInfo = GetParam();
    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat img_host = readImage("gpu/perf/aloe.jpg", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img_host.empty());

    cv::gpu::ORB_GPU orb(4000);

    cv::gpu::GpuMat img(img_host);
    cv::gpu::GpuMat keypoints, descriptors;

    TEST_CYCLE()
    {
        orb(img, cv::gpu::GpuMat(), keypoints, descriptors);
    }
}

INSTANTIATE_TEST_CASE_P(Features2D, ORB, ALL_DEVICES);

//////////////////////////////////////////////////////////////////////
// BruteForceMatcher_match

IMPLEMENT_PARAM_CLASS(DescriptorSize, int)

GPU_PERF_TEST(BruteForceMatcher_match, cv::gpu::DeviceInfo, DescriptorSize, NormType)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    int desc_size = GET_PARAM(1);
    int normType = GET_PARAM(2);

    int type = normType == cv::NORM_HAMMING ? CV_8U : CV_32F;

    cv::Mat query_host(3000, desc_size, type);
    fill(query_host, 0.0, 10.0);

    cv::Mat train_host(3000, desc_size, type);
    fill(train_host, 0.0, 10.0);

    cv::gpu::BFMatcher_GPU matcher(normType);

    cv::gpu::GpuMat query(query_host);
    cv::gpu::GpuMat train(train_host);
    cv::gpu::GpuMat trainIdx, distance;

    matcher.matchSingle(query, train, trainIdx, distance);

    declare.time(3.0);

    TEST_CYCLE()
    {
        matcher.matchSingle(query, train, trainIdx, distance);
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
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    int desc_size = GET_PARAM(1);
    int k = GET_PARAM(2);
    int normType = GET_PARAM(3);

    int type = normType == cv::NORM_HAMMING ? CV_8U : CV_32F;

    cv::Mat query_host(3000, desc_size, type);
    fill(query_host, 0.0, 10.0);

    cv::Mat train_host(3000, desc_size, type);
    fill(train_host, 0.0, 10.0);

    cv::gpu::BFMatcher_GPU matcher(normType);

    cv::gpu::GpuMat query(query_host);
    cv::gpu::GpuMat train(train_host);
    cv::gpu::GpuMat trainIdx, distance, allDist;

    matcher.knnMatchSingle(query, train, trainIdx, distance, allDist, k);

    declare.time(3.0);

    TEST_CYCLE()
    {
        matcher.knnMatchSingle(query, train, trainIdx, distance, allDist, k);
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
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());

    int desc_size = GET_PARAM(1);
    int normType = GET_PARAM(2);

    int type = normType == cv::NORM_HAMMING ? CV_8U : CV_32F;

    cv::Mat query_host(3000, desc_size, type);
    fill(query_host, 0.0, 1.0);

    cv::Mat train_host(3000, desc_size, type);
    fill(train_host, 0.0, 1.0);

    cv::gpu::BFMatcher_GPU matcher(normType);

    cv::gpu::GpuMat query(query_host);
    cv::gpu::GpuMat train(train_host);
    cv::gpu::GpuMat trainIdx, nMatches, distance;

    matcher.radiusMatchSingle(query, train, trainIdx, distance, nMatches, 2.0);

    declare.time(3.0);

    TEST_CYCLE()
    {
        matcher.radiusMatchSingle(query, train, trainIdx, distance, nMatches, 2.0);
    }
}

INSTANTIATE_TEST_CASE_P(Features2D, BruteForceMatcher_radiusMatch, testing::Combine(
    ALL_DEVICES,
    testing::Values(DescriptorSize(64), DescriptorSize(128), DescriptorSize(256)),
    testing::Values(NormType(cv::NORM_L1), NormType(cv::NORM_L2), NormType(cv::NORM_HAMMING))));

#endif
