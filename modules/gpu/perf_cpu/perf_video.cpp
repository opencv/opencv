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

//////////////////////////////////////////////////////
// FGDStatModel

namespace cv
{
    template<> void Ptr<CvBGStatModel>::delete_obj()
    {
        cvReleaseBGStatModel(&obj);
    }
}

GPU_PERF_TEST(FGDStatModel, cv::gpu::DeviceInfo, std::string)
{
    std::string inputFile = perf::TestBase::getDataPath(std::string("gpu/video/") + GET_PARAM(1));

    cv::VideoCapture cap(inputFile);
    ASSERT_TRUE(cap.isOpened());

    cv::Mat frame;
    cap >> frame;
    ASSERT_FALSE(frame.empty());

    IplImage ipl_frame = frame;
    cv::Ptr<CvBGStatModel> model(cvCreateFGDStatModel(&ipl_frame));

    declare.time(60);

    for (int i = 0; i < 10; ++i)
    {
        cap >> frame;
        ASSERT_FALSE(frame.empty());

        ipl_frame = frame;

        startTimer();
        next();

        cvUpdateBGStatModel(&ipl_frame, model);

        stopTimer();
    }
}

INSTANTIATE_TEST_CASE_P(Video, FGDStatModel, testing::Combine(
    ALL_DEVICES,
    testing::Values(std::string("768x576.avi"), std::string("1920x1080.avi"))));

//////////////////////////////////////////////////////
// VideoWriter

#ifdef WIN32

GPU_PERF_TEST(VideoWriter, cv::gpu::DeviceInfo, std::string)
{
    const double FPS = 25.0;

    std::string inputFile = perf::TestBase::getDataPath(std::string("gpu/video/") + GET_PARAM(1));
    std::string outputFile = inputFile.substr(0, inputFile.find('.')) + "_test.avi";

    cv::VideoCapture reader(inputFile);
    ASSERT_TRUE( reader.isOpened() );

    cv::VideoWriter writer;

    cv::Mat frame;

    declare.time(30);

    for (int i = 0; i < 10; ++i)
    {
        reader >> frame;
        ASSERT_FALSE(frame.empty());

        if (!writer.isOpened())
            writer.open(outputFile, CV_FOURCC('H', '2', '6', '4'), frame.size(), FPS);

        startTimer(); next();
        writer.write(frame);
        stopTimer();
    }
}

INSTANTIATE_TEST_CASE_P(Video, VideoWriter, testing::Combine(
    ALL_DEVICES,
    testing::Values(std::string("768x576.avi"), std::string("1920x1080.avi"))));

#endif // WIN32

//////////////////////////////////////////////////////
// VideoReader

GPU_PERF_TEST(VideoReader, cv::gpu::DeviceInfo, std::string)
{
    std::string inputFile = perf::TestBase::getDataPath(std::string("gpu/video/") + GET_PARAM(1));

    cv::VideoCapture reader(inputFile);
    ASSERT_TRUE( reader.isOpened() );

    cv::Mat frame;

    reader >> frame;

    declare.time(20);

    TEST_CYCLE_N(10)
    {
        reader >> frame;
    }
}

INSTANTIATE_TEST_CASE_P(Video, VideoReader, testing::Combine(
    ALL_DEVICES,
    testing::Values(std::string("768x576.avi"), std::string("1920x1080.avi"))));

#endif
