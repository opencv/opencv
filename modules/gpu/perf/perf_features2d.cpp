#include "perf_precomp.hpp"

PERF_TEST_P(DevInfo_DescSize, BruteForceMatcher_match, testing::Combine(testing::ValuesIn(devices()),
                                                                        testing::Values(64, 128, 256)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    int desc_size = std::tr1::get<1>(GetParam());

    setDevice(devInfo.deviceID());

    Mat query_host(3000, desc_size, CV_32FC1);
    Mat train_host(3000, desc_size, CV_32FC1);

    declare.in(query_host, train_host, WARMUP_RNG);

    GpuMat query(query_host);
    GpuMat train(train_host);
    GpuMat trainIdx, distance;

    BruteForceMatcher_GPU< L2<float> > matcher;

    declare.time(3.0);

    TEST_CYCLE(100)
    {
        matcher.matchSingle(query, train, trainIdx, distance);
    }

    Mat trainIdx_host(trainIdx);
    Mat distance_host(distance);

    SANITY_CHECK(trainIdx_host);
    SANITY_CHECK(distance_host);
}

PERF_TEST_P(DevInfo_K_DescSize, BruteForceMatcher_knnMatch, testing::Combine(testing::ValuesIn(devices()),
                                                                             testing::Values(2, 3),
                                                                             testing::Values(64, 128, 256)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    int k = std::tr1::get<1>(GetParam());
    int desc_size = std::tr1::get<2>(GetParam());

    setDevice(devInfo.deviceID());

    Mat query_host(3000, desc_size, CV_32FC1);
    Mat train_host(3000, desc_size, CV_32FC1);

    declare.in(query_host, train_host, WARMUP_RNG);

    GpuMat query(query_host);
    GpuMat train(train_host);
    GpuMat trainIdx, distance, allDist;

    BruteForceMatcher_GPU< L2<float> > matcher;

    declare.time(3.0);

    TEST_CYCLE(100)
    {
        matcher.knnMatchSingle(query, train, trainIdx, distance, allDist, k);
    }

    Mat trainIdx_host(trainIdx);
    Mat distance_host(distance);

    SANITY_CHECK(trainIdx_host);
    SANITY_CHECK(distance_host);
}

PERF_TEST_P(DevInfo_DescSize, BruteForceMatcher_radiusMatch, testing::Combine(testing::ValuesIn(devices(SHARED_ATOMICS)),
                                                                        testing::Values(64, 128, 256)))
{
    DeviceInfo devInfo = std::tr1::get<0>(GetParam());
    int desc_size = std::tr1::get<1>(GetParam());

    setDevice(devInfo.deviceID());

    Mat query_host = cvtest::randomMat(theRNG(), Size(desc_size, 3000), CV_32FC1, 0, 1, false);
    Mat train_host = cvtest::randomMat(theRNG(), Size(desc_size, 3000), CV_32FC1, 0, 1, false);

    GpuMat query(query_host);
    GpuMat train(train_host);
    GpuMat trainIdx, nMatches, distance;

    BruteForceMatcher_GPU< L2<float> > matcher;

    declare.time(3.0);

    TEST_CYCLE(100)
    {
        matcher.radiusMatchSingle(query, train, trainIdx, distance, nMatches, 2.0);
    }

    Mat trainIdx_host(trainIdx);
    Mat nMatches_host(nMatches);
    Mat distance_host(distance);

    SANITY_CHECK(trainIdx_host);
    SANITY_CHECK(nMatches_host);
    SANITY_CHECK(distance_host);
}

PERF_TEST_P(DevInfo, SURF, testing::ValuesIn(devices()))
{
    DeviceInfo devInfo = GetParam();

    setDevice(devInfo.deviceID());

    Mat img_host = readImage("gpu/perf/aloe.jpg", CV_LOAD_IMAGE_GRAYSCALE);

    ASSERT_FALSE(img_host.empty());

    GpuMat img(img_host);
    GpuMat keypoints, descriptors;

    SURF_GPU surf;

    declare.time(2.0);

    TEST_CYCLE(100)
    {
        surf(img, GpuMat(), keypoints, descriptors);
    }
}

PERF_TEST_P(DevInfo, FAST, testing::ValuesIn(devices()))
{
    DeviceInfo devInfo = GetParam();

    setDevice(devInfo.deviceID());

    Mat img_host = readImage("gpu/perf/aloe.jpg", CV_LOAD_IMAGE_GRAYSCALE);

    ASSERT_FALSE(img_host.empty());

    GpuMat img(img_host);
    GpuMat keypoints;

    FAST_GPU fastGPU(20);

    declare.time(2.0);

    TEST_CYCLE(100)
    {
        fastGPU(img, GpuMat(), keypoints);
    }
}

PERF_TEST_P(DevInfo, ORB, testing::ValuesIn(devices()))
{
    DeviceInfo devInfo = GetParam();

    setDevice(devInfo.deviceID());

    Mat img_host = readImage("gpu/perf/aloe.jpg", CV_LOAD_IMAGE_GRAYSCALE);

    ASSERT_FALSE(img_host.empty());

    GpuMat img(img_host);
    GpuMat keypoints, descriptors;

    ORB_GPU orbGPU(4000);

    declare.time(2.0);

    TEST_CYCLE(100)
    {
        orbGPU(img, GpuMat(), keypoints, descriptors);
    }
}
