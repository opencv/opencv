#include "perf_precomp.hpp"

#ifdef HAVE_CUDA

//////////////////////////////////////////////////////////////////////
// BruteForceMatcher_match

GPU_PERF_TEST(BruteForceMatcher_match, cv::gpu::DeviceInfo, int)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    int desc_size = GET_PARAM(1);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat query_host(3000, desc_size, CV_32FC1);
    cv::Mat train_host(3000, desc_size, CV_32FC1);

    declare.in(query_host, train_host, WARMUP_RNG);

    cv::gpu::GpuMat query(query_host);
    cv::gpu::GpuMat train(train_host);
    cv::gpu::GpuMat trainIdx, distance;

    cv::gpu::BFMatcher_GPU matcher(cv::NORM_L2);

    declare.time(3.0);

    TEST_CYCLE()
    {
        matcher.matchSingle(query, train, trainIdx, distance);
    }
}

INSTANTIATE_TEST_CASE_P(Features2D, BruteForceMatcher_match, testing::Combine(
                        ALL_DEVICES, 
                        testing::Values(64, 128, 256)));

//////////////////////////////////////////////////////////////////////
// BruteForceMatcher_knnMatch

GPU_PERF_TEST(BruteForceMatcher_knnMatch, cv::gpu::DeviceInfo, int, int)
{
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    int desc_size = GET_PARAM(1);
    int k = GET_PARAM(2);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat query_host(3000, desc_size, CV_32FC1);
    cv::Mat train_host(3000, desc_size, CV_32FC1);

    declare.in(query_host, train_host, WARMUP_RNG);

    cv::gpu::GpuMat query(query_host);
    cv::gpu::GpuMat train(train_host);
    cv::gpu::GpuMat trainIdx, distance, allDist;

    cv::gpu::BFMatcher_GPU matcher(cv::NORM_L2);

    declare.time(3.0);

    TEST_CYCLE()
    {
        matcher.knnMatchSingle(query, train, trainIdx, distance, allDist, k);
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
    cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    int desc_size = GET_PARAM(1);

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat query_host(3000, desc_size, CV_32FC1);
    cv::Mat train_host(3000, desc_size, CV_32FC1);

    fill(query_host, 0, 1);
    fill(train_host, 0, 1);

    cv::gpu::GpuMat query(query_host);
    cv::gpu::GpuMat train(train_host);
    cv::gpu::GpuMat trainIdx, nMatches, distance;

    cv::gpu::BFMatcher_GPU matcher(cv::NORM_L2);

    declare.time(3.0);

    TEST_CYCLE()
    {
        matcher.radiusMatchSingle(query, train, trainIdx, distance, nMatches, 2.0);
    }
}

INSTANTIATE_TEST_CASE_P(Features2D, BruteForceMatcher_radiusMatch, testing::Combine(
                        ALL_DEVICES, 
                        testing::Values(64, 128, 256)));

//////////////////////////////////////////////////////////////////////
// SURF

GPU_PERF_TEST_1(SURF, cv::gpu::DeviceInfo)
{
    cv::gpu::DeviceInfo devInfo = GetParam();

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat img_host = readImage("gpu/perf/aloe.jpg", cv::IMREAD_GRAYSCALE);

    ASSERT_FALSE(img_host.empty());

    cv::gpu::GpuMat img(img_host);
    cv::gpu::GpuMat keypoints, descriptors;

    cv::gpu::SURF_GPU surf;

    declare.time(2.0);

    TEST_CYCLE()
    {
        surf(img, cv::gpu::GpuMat(), keypoints, descriptors);
    }
}

INSTANTIATE_TEST_CASE_P(Features2D, SURF, DEVICES(cv::gpu::GLOBAL_ATOMICS));

//////////////////////////////////////////////////////////////////////
// FAST

GPU_PERF_TEST_1(FAST, cv::gpu::DeviceInfo)
{
    cv::gpu::DeviceInfo devInfo = GetParam();

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat img_host = readImage("gpu/perf/aloe.jpg", cv::IMREAD_GRAYSCALE);

    ASSERT_FALSE(img_host.empty());

    cv::gpu::GpuMat img(img_host);
    cv::gpu::GpuMat keypoints, descriptors;

    cv::gpu::FAST_GPU fastGPU(20);

    TEST_CYCLE()
    {
        fastGPU(img, cv::gpu::GpuMat(), keypoints);
    }
}

INSTANTIATE_TEST_CASE_P(Features2D, FAST, DEVICES(cv::gpu::GLOBAL_ATOMICS));

//////////////////////////////////////////////////////////////////////
// ORB

GPU_PERF_TEST_1(ORB, cv::gpu::DeviceInfo)
{
    cv::gpu::DeviceInfo devInfo = GetParam();

    cv::gpu::setDevice(devInfo.deviceID());

    cv::Mat img_host = readImage("gpu/perf/aloe.jpg", cv::IMREAD_GRAYSCALE);

    ASSERT_FALSE(img_host.empty());

    cv::gpu::GpuMat img(img_host);
    cv::gpu::GpuMat keypoints, descriptors;

    cv::gpu::ORB_GPU orbGPU(4000);

    TEST_CYCLE()
    {
        orbGPU(img, cv::gpu::GpuMat(), keypoints, descriptors);
    }
}

INSTANTIATE_TEST_CASE_P(Features2D, ORB, DEVICES(cv::gpu::GLOBAL_ATOMICS));

#endif
