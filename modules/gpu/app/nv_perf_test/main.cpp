#include <cstdio>
#define HAVE_CUDA 1
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/ts/ts.hpp>
#include <opencv2/ts/ts_perf.hpp>

static void printOsInfo()
{
#if defined _WIN32
#   if defined _WIN64
        printf("[----------]\n[ GPU INFO ] \tRun on OS Windows x64.\n[----------]\n"); fflush(stdout);
#   else
        printf("[----------]\n[ GPU INFO ] \tRun on OS Windows x32.\n[----------]\n"); fflush(stdout);
#   endif
#elif defined linux
#   if defined _LP64
        printf("[----------]\n[ GPU INFO ] \tRun on OS Linux x64.\n[----------]\n"); fflush(stdout);
#   else
        printf("[----------]\n[ GPU INFO ] \tRun on OS Linux x32.\n[----------]\n"); fflush(stdout);
#   endif
#elif defined __APPLE__
#   if defined _LP64
        printf("[----------]\n[ GPU INFO ] \tRun on OS Apple x64.\n[----------]\n"); fflush(stdout);
#   else
        printf("[----------]\n[ GPU INFO ] \tRun on OS Apple x32.\n[----------]\n"); fflush(stdout);
#   endif
#endif
}

static void printCudaInfo()
{
    const int deviceCount = cv::gpu::getCudaEnabledDeviceCount();

    printf("[----------]\n"); fflush(stdout);
    printf("[ GPU INFO ] \tCUDA device count:: %d.\n", deviceCount); fflush(stdout);
    printf("[----------]\n"); fflush(stdout);

    for (int i = 0; i < deviceCount; ++i)
    {
        cv::gpu::DeviceInfo info(i);

        printf("[----------]\n"); fflush(stdout);
        printf("[ DEVICE   ] \t# %d %s.\n", i, info.name().c_str()); fflush(stdout);
        printf("[          ] \tCompute capability: %d.%d\n", info.majorVersion(), info.minorVersion()); fflush(stdout);
        printf("[          ] \tMulti Processor Count:  %d\n", info.multiProcessorCount()); fflush(stdout);
        printf("[          ] \tTotal memory: %d Mb\n", static_cast<int>(static_cast<int>(info.totalMemory() / 1024.0) / 1024.0)); fflush(stdout);
        printf("[          ] \tFree  memory: %d Mb\n", static_cast<int>(static_cast<int>(info.freeMemory()  / 1024.0) / 1024.0)); fflush(stdout);
        if (!info.isCompatible())
            printf("[ GPU INFO ] \tThis device is NOT compatible with current GPU module build\n");
        printf("[----------]\n"); fflush(stdout);
    }
}

int main(int argc, char* argv[])
{
    printOsInfo();
    printCudaInfo();

    perf::Regression::Init("nv_perf_test");
    perf::TestBase::Init(argc, argv);
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

//////////////////////////////////////////////////////////
// Tests

#define DEF_PARAM_TEST(name, ...) typedef ::perf::TestBaseWithParam< std::tr1::tuple< __VA_ARGS__ > > name
#define DEF_PARAM_TEST_1(name, param_type) typedef ::perf::TestBaseWithParam< param_type > name

DEF_PARAM_TEST_1(Depth, perf::MatDepth);

PERF_TEST_P(Depth, GoodFeaturesToTrack, testing::Values(CV_8U, CV_16U))
{
    declare.time(60);

    const int depth = GetParam();
    const int maxCorners = 5000;
    const double qualityLevel = 0.05;
    const int minDistance = 5;
    const int blockSize = 3;
    const bool useHarrisDetector = true;
    const double k = 0.05;

    const std::string fileName = "im1_1280x800.jpg";

    cv::Mat src = cv::imread(fileName, cv::IMREAD_GRAYSCALE);
    if (src.empty())
        FAIL() << "Unable to load source image [" << fileName << "]";

    if (depth != CV_8U)
        src.convertTo(src, depth);

    cv::Mat mask(src.size(), CV_8UC1, cv::Scalar::all(1));
    mask(cv::Rect(0, 0, 100, 100)).setTo(cv::Scalar::all(0));

    if (PERF_RUN_GPU())
    {
        cv::gpu::GoodFeaturesToTrackDetector_GPU d_detector(maxCorners, qualityLevel, minDistance, blockSize, useHarrisDetector, k);

        cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat d_mask(mask);
        cv::gpu::GpuMat d_pts;

        d_detector(d_src, d_pts, d_mask);

        TEST_CYCLE()
        {
            d_detector(d_src, d_pts, d_mask);
        }
    }
    else
    {
        cv::Mat pts;

        cv::goodFeaturesToTrack(src, pts, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k);

        TEST_CYCLE()
        {
            cv::goodFeaturesToTrack(src, pts, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k);
        }
    }

    SANITY_CHECK(0);
}

DEF_PARAM_TEST(Depth_GraySource, perf::MatDepth, bool);

PERF_TEST_P(Depth_GraySource, PyrLKOpticalFlowSparse, testing::Combine(testing::Values(CV_8U, CV_16U), testing::Bool()))
{
    declare.time(60);

    const int depth = std::tr1::get<0>(GetParam());
    const bool graySource = std::tr1::get<1>(GetParam());

    // PyrLK params
    const cv::Size winSize(15, 15);
    const int maxLevel = 5;
    const cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);

    // GoodFeaturesToTrack params
    const int maxCorners = 5000;
    const double qualityLevel = 0.05;
    const int minDistance = 5;
    const int blockSize = 3;
    const bool useHarrisDetector = true;
    const double k = 0.05;

    const std::string fileName1 = "im1_1280x800.jpg";
    const std::string fileName2 = "im2_1280x800.jpg";

    cv::Mat src1 = cv::imread(fileName1, graySource ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
    if (src1.empty())
        FAIL() << "Unable to load source image [" << fileName1 << "]";

    cv::Mat src2 = cv::imread(fileName2, graySource ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
    if (src2.empty())
        FAIL() << "Unable to load source image [" << fileName2 << "]";

    cv::Mat gray_src;
    if (graySource)
        gray_src = src1;
    else
        cv::cvtColor(src1, gray_src, cv::COLOR_BGR2GRAY);

    cv::Mat pts;
    cv::goodFeaturesToTrack(gray_src, pts, maxCorners, qualityLevel, minDistance, cv::noArray(), blockSize, useHarrisDetector, k);

    if (depth != CV_8U)
    {
        src1.convertTo(src1, depth);
        src2.convertTo(src2, depth);
    }

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_pts(pts.reshape(2, 1));
        cv::gpu::GpuMat d_nextPts;
        cv::gpu::GpuMat d_status;

        cv::gpu::PyrLKOpticalFlow d_pyrLK;
        d_pyrLK.winSize = winSize;
        d_pyrLK.maxLevel = maxLevel;
        d_pyrLK.iters = criteria.maxCount;
        d_pyrLK.useInitialFlow = false;

        d_pyrLK.sparse(d_src1, d_src2, d_pts, d_nextPts, d_status);

        TEST_CYCLE()
        {
            d_pyrLK.sparse(d_src1, d_src2, d_pts, d_nextPts, d_status);
        }
    }
    else
    {
        cv::Mat nextPts;
        cv::Mat status;

        cv::calcOpticalFlowPyrLK(src1, src2, pts, nextPts, status, cv::noArray(), winSize, maxLevel, criteria);

        TEST_CYCLE()
        {
            cv::calcOpticalFlowPyrLK(src1, src2, pts, nextPts, status, cv::noArray(), winSize, maxLevel, criteria);
        }
    }

    SANITY_CHECK(0);
}

DEF_PARAM_TEST_1(Depth, perf::MatDepth);

PERF_TEST_P(Depth, FarnebackOpticalFlow, testing::Values(CV_8U, CV_16U))
{
    declare.time(60);

    const int depth = GetParam();

    const double pyrScale = 0.5;
    const int numLevels = 6;
    const int winSize = 7;
    const int numIters = 15;
    const int polyN = 7;
    const double polySigma = 1.5;
    const int flags = cv::OPTFLOW_USE_INITIAL_FLOW;

    const std::string fileName1 = "im1_1280x800.jpg";
    const std::string fileName2 = "im2_1280x800.jpg";

    cv::Mat src1 = cv::imread(fileName1, cv::IMREAD_GRAYSCALE);
    if (src1.empty())
        FAIL() << "Unable to load source image [" << fileName1 << "]";

    cv::Mat src2 = cv::imread(fileName2, cv::IMREAD_GRAYSCALE);
    if (src2.empty())
        FAIL() << "Unable to load source image [" << fileName2 << "]";

    if (depth != CV_8U)
    {
        src1.convertTo(src1, depth);
        src2.convertTo(src2, depth);
    }

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_u(src1.size(), CV_32FC1, cv::Scalar::all(0));
        cv::gpu::GpuMat d_v(src1.size(), CV_32FC1, cv::Scalar::all(0));

        cv::gpu::FarnebackOpticalFlow d_farneback;
        d_farneback.pyrScale = pyrScale;
        d_farneback.numLevels = numLevels;
        d_farneback.winSize = winSize;
        d_farneback.numIters = numIters;
        d_farneback.polyN = polyN;
        d_farneback.polySigma = polySigma;
        d_farneback.flags = flags;

        d_farneback(d_src1, d_src2, d_u, d_v);

        TEST_CYCLE()
        {
            d_farneback(d_src1, d_src2, d_u, d_v);
        }
    }
    else
    {
        cv::Mat flow(src1.size(), CV_32FC2, cv::Scalar::all(0));

        cv::calcOpticalFlowFarneback(src1, src2, flow, pyrScale, numLevels, winSize, numIters, polyN, polySigma, flags);

        TEST_CYCLE()
        {
            cv::calcOpticalFlowFarneback(src1, src2, flow, pyrScale, numLevels, winSize, numIters, polyN, polySigma, flags);
        }
    }

    SANITY_CHECK(0);
}
