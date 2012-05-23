#include "perf_cpu_precomp.hpp"

#ifdef HAVE_CUDA

//////////////////////////////////////////////////////
// GoodFeaturesToTrack

IMPLEMENT_PARAM_CLASS(MinDistance, double)

GPU_PERF_TEST(GoodFeaturesToTrack, cv::gpu::DeviceInfo, MinDistance)
{
    double minDistance = GET_PARAM(1);

    cv::Mat image = readImage("gpu/perf/aloe.jpg", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(image.empty());

    cv::Mat corners;

    cv::goodFeaturesToTrack(image, corners, 8000, 0.01, minDistance);

    TEST_CYCLE()
    {
        cv::goodFeaturesToTrack(image, corners, 8000, 0.01, minDistance);
    }
}

INSTANTIATE_TEST_CASE_P(Video, GoodFeaturesToTrack, testing::Combine(
    ALL_DEVICES,
    testing::Values(MinDistance(0.0), MinDistance(3.0))));

//////////////////////////////////////////////////////
// PyrLKOpticalFlowSparse

IMPLEMENT_PARAM_CLASS(GraySource, bool)
IMPLEMENT_PARAM_CLASS(Points, int)
IMPLEMENT_PARAM_CLASS(WinSize, int)

GPU_PERF_TEST(PyrLKOpticalFlowSparse, cv::gpu::DeviceInfo, GraySource, Points, WinSize)
{
    bool useGray = GET_PARAM(1);
    int points = GET_PARAM(2);
    int win_size = GET_PARAM(3);

    cv::Mat frame0 = readImage("gpu/opticalflow/frame0.png", useGray ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
    ASSERT_FALSE(frame0.empty());

    cv::Mat frame1 = readImage("gpu/opticalflow/frame1.png", useGray ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
    ASSERT_FALSE(frame1.empty());

    cv::Mat gray_frame;
    if (useGray)
        gray_frame = frame0;
    else
        cv::cvtColor(frame0, gray_frame, cv::COLOR_BGR2GRAY);

    cv::Mat pts;
    cv::goodFeaturesToTrack(gray_frame, pts, points, 0.01, 0.0);

    cv::Mat nextPts;
    cv::Mat status;

    cv::calcOpticalFlowPyrLK(frame0, frame1, pts, nextPts, status, cv::noArray(), cv::Size(win_size, win_size));

    TEST_CYCLE()
    {
        cv::calcOpticalFlowPyrLK(frame0, frame1, pts, nextPts, status, cv::noArray(), cv::Size(win_size, win_size));
    }
}

INSTANTIATE_TEST_CASE_P(Video, PyrLKOpticalFlowSparse, testing::Combine(
    ALL_DEVICES,
    testing::Values(GraySource(true), GraySource(false)),
    testing::Values(Points(1000), Points(2000), Points(4000), Points(8000)),
    testing::Values(WinSize(17), WinSize(21))));

//////////////////////////////////////////////////////
// FarnebackOpticalFlowTest

GPU_PERF_TEST_1(FarnebackOpticalFlowTest, cv::gpu::DeviceInfo)
{
    cv::Mat frame0 = readImage("gpu/opticalflow/frame0.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame0.empty());

    cv::Mat frame1 = readImage("gpu/opticalflow/frame1.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame1.empty());

    cv::Mat flow;

    int numLevels = 5;
    double pyrScale = 0.5;
    int winSize = 13;
    int numIters = 10;
    int polyN = 5;
    double polySigma = 1.1;
    int flags = 0;

    cv::calcOpticalFlowFarneback(frame0, frame1, flow, pyrScale, numLevels, winSize, numIters, polyN, polySigma, flags);

    declare.time(10);

    TEST_CYCLE()
    {
    }
}

INSTANTIATE_TEST_CASE_P(Video, FarnebackOpticalFlowTest, ALL_DEVICES);

#endif
