#include <cstdio>
#ifdef HAVE_CVCONFIG_H
#include "cvconfig.h"
#endif
#include "opencv2/core/core.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/ts/ts.hpp"
#include "opencv2/ts/ts_perf.hpp"

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

#define DEF_PARAM_TEST(name, ...) typedef ::perf::TestBaseWithParam< std::tr1::tuple< __VA_ARGS__ > > name
#define DEF_PARAM_TEST_1(name, param_type) typedef ::perf::TestBaseWithParam< param_type > name

//////////////////////////////////////////////////////////
// HoughLinesP

DEF_PARAM_TEST_1(Image, std::string);

PERF_TEST_P(Image, HoughLinesP, testing::Values(std::string("im1_1280x800.jpg")))
{
    declare.time(30.0);

    std::string fileName = GetParam();

    const float rho = 1.f;
    const float theta = 1.f;
    const int threshold = 40;
    const int minLineLenght = 20;
    const int maxLineGap = 5;

    cv::Mat image = cv::imread(fileName, cv::IMREAD_GRAYSCALE);

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_image(image);
        cv::gpu::GpuMat d_lines;
        cv::gpu::HoughLinesBuf d_buf;

        cv::gpu::HoughLinesP(d_image, d_lines, d_buf, rho, theta, minLineLenght, maxLineGap);

        TEST_CYCLE()
        {
            cv::gpu::HoughLinesP(d_image, d_lines, d_buf, rho, theta, minLineLenght, maxLineGap);
        }
    }
    else
    {
        cv::Mat mask;
        cv::Canny(image, mask, 50, 100);

        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(mask, lines, rho, theta, threshold, minLineLenght, maxLineGap);

        TEST_CYCLE()
        {
            cv::HoughLinesP(mask, lines, rho, theta, threshold, minLineLenght, maxLineGap);
        }
    }

    SANITY_CHECK(0);
}

//////////////////////////////////////////////////////////
// GoodFeaturesToTrack

DEF_PARAM_TEST(Image_Depth, std::string, perf::MatDepth);

PERF_TEST_P(Image_Depth, GoodFeaturesToTrack,
                testing::Combine(
                testing::Values(std::string("im1_1280x800.jpg")),
                testing::Values(CV_8U, CV_16U)
                ))
{
    declare.time(60);

    const std::string fileName = std::tr1::get<0>(GetParam());
    const int depth = std::tr1::get<1>(GetParam());

    const int maxCorners = 5000;
    const double qualityLevel = 0.05;
    const int minDistance = 5;
    const int blockSize = 3;
    const bool useHarrisDetector = true;
    const double k = 0.05;

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
        if (depth != CV_8U)
            FAIL() << "Unsupported depth";

        cv::Mat pts;

        cv::goodFeaturesToTrack(src, pts, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k);

        TEST_CYCLE()
        {
            cv::goodFeaturesToTrack(src, pts, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k);
        }
    }

    SANITY_CHECK(0);
}

//////////////////////////////////////////////////////////
// OpticalFlowPyrLKSparse

typedef std::pair<std::string, std::string> string_pair;

DEF_PARAM_TEST(ImagePair_Depth_GraySource, string_pair, perf::MatDepth, bool);

PERF_TEST_P(ImagePair_Depth_GraySource, OpticalFlowPyrLKSparse,
                testing::Combine(
                    testing::Values(string_pair("im1_1280x800.jpg", "im2_1280x800.jpg")),
                    testing::Values(CV_8U, CV_16U),
                    testing::Bool()
                    ))
{
    declare.time(60);

    const string_pair fileNames = std::tr1::get<0>(GetParam());
    const int depth = std::tr1::get<1>(GetParam());
    const bool graySource = std::tr1::get<2>(GetParam());

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

    cv::Mat src1 = cv::imread(fileNames.first, graySource ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
    if (src1.empty())
        FAIL() << "Unable to load source image [" << fileNames.first << "]";

    cv::Mat src2 = cv::imread(fileNames.second, graySource ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
    if (src2.empty())
        FAIL() << "Unable to load source image [" << fileNames.second << "]";

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
        if (depth != CV_8U)
            FAIL() << "Unsupported depth";

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

//////////////////////////////////////////////////////////
// OpticalFlowFarneback

DEF_PARAM_TEST(ImagePair_Depth, string_pair, perf::MatDepth);

PERF_TEST_P(ImagePair_Depth, OpticalFlowFarneback,
                testing::Combine(
                    testing::Values(string_pair("im1_1280x800.jpg", "im2_1280x800.jpg")),
                    testing::Values(CV_8U, CV_16U)
                    ))
{
    declare.time(500);

    const string_pair fileNames = std::tr1::get<0>(GetParam());
    const int depth = std::tr1::get<1>(GetParam());

    const double pyrScale = 0.5;
    const int numLevels = 6;
    const int winSize = 7;
    const int numIters = 15;
    const int polyN = 7;
    const double polySigma = 1.5;
    const int flags = cv::OPTFLOW_USE_INITIAL_FLOW;

    cv::Mat src1 = cv::imread(fileNames.first, cv::IMREAD_GRAYSCALE);
    if (src1.empty())
        FAIL() << "Unable to load source image [" << fileNames.first << "]";

    cv::Mat src2 = cv::imread(fileNames.second, cv::IMREAD_GRAYSCALE);
    if (src2.empty())
        FAIL() << "Unable to load source image [" << fileNames.second << "]";

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

        TEST_CYCLE_N(10)
        {
            d_farneback(d_src1, d_src2, d_u, d_v);
        }
    }
    else
    {
        if (depth != CV_8U)
            FAIL() << "Unsupported depth";

        cv::Mat flow(src1.size(), CV_32FC2, cv::Scalar::all(0));

        cv::calcOpticalFlowFarneback(src1, src2, flow, pyrScale, numLevels, winSize, numIters, polyN, polySigma, flags);

        TEST_CYCLE_N(10)
        {
            cv::calcOpticalFlowFarneback(src1, src2, flow, pyrScale, numLevels, winSize, numIters, polyN, polySigma, flags);
        }
    }

    SANITY_CHECK(0);
}

//////////////////////////////////////////////////////////
// OpticalFlowBM

void calcOpticalFlowBM(const cv::Mat& prev, const cv::Mat& curr,
                       cv::Size bSize, cv::Size shiftSize, cv::Size maxRange, int usePrevious,
                       cv::Mat& velx, cv::Mat& vely)
{
    cv::Size sz((curr.cols - bSize.width + shiftSize.width)/shiftSize.width, (curr.rows - bSize.height + shiftSize.height)/shiftSize.height);

    velx.create(sz, CV_32FC1);
    vely.create(sz, CV_32FC1);

    CvMat cvprev = prev;
    CvMat cvcurr = curr;

    CvMat cvvelx = velx;
    CvMat cvvely = vely;

    cvCalcOpticalFlowBM(&cvprev, &cvcurr, bSize, shiftSize, maxRange, usePrevious, &cvvelx, &cvvely);
}

DEF_PARAM_TEST(ImagePair_BlockSize_ShiftSize_MaxRange, string_pair, cv::Size, cv::Size, cv::Size);

PERF_TEST_P(ImagePair_BlockSize_ShiftSize_MaxRange, OpticalFlowBM,
                testing::Combine(
                    testing::Values(string_pair("im1_1280x800.jpg", "im2_1280x800.jpg")),
                    testing::Values(cv::Size(16, 16)),
                    testing::Values(cv::Size(2, 2)),
                    testing::Values(cv::Size(16, 16))
                    ))
{
    declare.time(3000);

    const string_pair fileNames = std::tr1::get<0>(GetParam());
    const cv::Size block_size = std::tr1::get<1>(GetParam());
    const cv::Size shift_size = std::tr1::get<2>(GetParam());
    const cv::Size max_range = std::tr1::get<3>(GetParam());

    cv::Mat src1 = cv::imread(fileNames.first, cv::IMREAD_GRAYSCALE);
    if (src1.empty())
        FAIL() << "Unable to load source image [" << fileNames.first << "]";

    cv::Mat src2 = cv::imread(fileNames.second, cv::IMREAD_GRAYSCALE);
    if (src2.empty())
        FAIL() << "Unable to load source image [" << fileNames.second << "]";

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_velx, d_vely, buf;

        cv::gpu::calcOpticalFlowBM(d_src1, d_src2, block_size, shift_size, max_range, false, d_velx, d_vely, buf);

        TEST_CYCLE_N(10)
        {
            cv::gpu::calcOpticalFlowBM(d_src1, d_src2, block_size, shift_size, max_range, false, d_velx, d_vely, buf);
        }
    }
    else
    {
        cv::Mat velx, vely;

        calcOpticalFlowBM(src1, src2, block_size, shift_size, max_range, false, velx, vely);

        TEST_CYCLE_N(10)
        {
            calcOpticalFlowBM(src1, src2, block_size, shift_size, max_range, false, velx, vely);
        }
    }

    SANITY_CHECK(0);
}

PERF_TEST_P(ImagePair_BlockSize_ShiftSize_MaxRange, FastOpticalFlowBM,
                testing::Combine(
                    testing::Values(string_pair("im1_1280x800.jpg", "im2_1280x800.jpg")),
                    testing::Values(cv::Size(16, 16)),
                    testing::Values(cv::Size(1, 1)),
                    testing::Values(cv::Size(16, 16))
                    ))
{
    declare.time(3000);

    const string_pair fileNames = std::tr1::get<0>(GetParam());
    const cv::Size block_size = std::tr1::get<1>(GetParam());
    const cv::Size shift_size = std::tr1::get<2>(GetParam());
    const cv::Size max_range = std::tr1::get<3>(GetParam());

    cv::Mat src1 = cv::imread(fileNames.first, cv::IMREAD_GRAYSCALE);
    if (src1.empty())
        FAIL() << "Unable to load source image [" << fileNames.first << "]";

    cv::Mat src2 = cv::imread(fileNames.second, cv::IMREAD_GRAYSCALE);
    if (src2.empty())
        FAIL() << "Unable to load source image [" << fileNames.second << "]";

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_src1(src1);
        cv::gpu::GpuMat d_src2(src2);
        cv::gpu::GpuMat d_velx, d_vely;

        cv::gpu::FastOpticalFlowBM fastBM;

        fastBM(d_src1, d_src2, d_velx, d_vely, max_range.width, block_size.width);

        TEST_CYCLE_N(10)
        {
            fastBM(d_src1, d_src2, d_velx, d_vely, max_range.width, block_size.width);
        }
    }
    else
    {
        cv::Mat velx, vely;

        calcOpticalFlowBM(src1, src2, block_size, shift_size, max_range, false, velx, vely);

        TEST_CYCLE_N(10)
        {
            calcOpticalFlowBM(src1, src2, block_size, shift_size, max_range, false, velx, vely);
        }
    }

    SANITY_CHECK(0);
}
