// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"
#include "opencv2/imgcodecs.hpp"

namespace opencv_test { namespace {

const int SEQUENCE_LENGTH = 10;

static std::vector<Mat> makeSequence(Size size)
{
    Mat base(size.height * 2, size.width * 2, CV_8U);
    RNG(0x5eed).fill(base, RNG::UNIFORM, 0, 256);
    GaussianBlur(base, base, Size(9, 9), 3, 3);

    std::vector<Mat> frames;
    for( int i = 0; i < SEQUENCE_LENGTH; i++ )
    {
        Mat shift = Mat::eye(2, 3, CV_64F);
        shift.at<double>(0, 2) = 1.7 * i;
        shift.at<double>(1, 2) = -1.1 * i;
        Mat warped;
        warpAffine(base, warped, shift, base.size(), INTER_LINEAR);
        frames.push_back(warped(Rect(size.width / 2, size.height / 2,
                                     size.width, size.height)).clone());
    }
    return frames;
}

typedef tuple<Size, bool> FarnebackParams;
typedef TestBaseWithParam<FarnebackParams> DenseOpticalFlow_Farneback;

PERF_TEST_P(DenseOpticalFlow_Farneback, perf,
            Combine(Values(szVGA, sz720p), testing::Bool()))
{
    const Size size = get<0>(GetParam());
    const bool reuseExpansion = get<1>(GetParam());

    const std::vector<Mat> frames = makeSequence(size);
    Mat flow;

    Ptr<FarnebackOpticalFlow> algo = FarnebackOpticalFlow::create();
    algo->setFlags(reuseExpansion ? OPTFLOW_FARNEBACK_REUSE_EXPANSION : 0);

    TEST_CYCLE()
    {
        for( int i = 0; i + 1 < SEQUENCE_LENGTH; i++ )
            algo->calc(frames[i], frames[i + 1], flow);
    }

    SANITY_CHECK_NOTHING();
}

CV_ENUM(FarnebackGaussian, 0, OPTFLOW_FARNEBACK_GAUSSIAN)
typedef tuple<int, FarnebackGaussian, bool> FarnebackCPUParams;
typedef TestBaseWithParam<FarnebackCPUParams> FarnebackCPU;

PERF_TEST_P(FarnebackCPU, calcOpticalFlowFarneback,
            Combine(Values(5, 7), FarnebackGaussian::all(), testing::Bool()))
{
    const Mat prev = imread(getDataPath("cv/optflow/RubberWhale1.png"), IMREAD_GRAYSCALE);
    const Mat next = imread(getDataPath("cv/optflow/RubberWhale2.png"), IMREAD_GRAYSCALE);
    ASSERT_FALSE(prev.empty());
    ASSERT_FALSE(next.empty());

    const int polyN = get<0>(GetParam());
    const double polySigma = polyN == 5 ? 1.1 : 1.5;
    int flags = get<1>(GetParam());
    const bool useInitialFlow = get<2>(GetParam());
    const double pyrScale = 0.5;
    const int levels = 5, winSize = 13, iterations = 10;
    Mat flow(prev.size(), CV_32FC2);
    Mat initialFlow;
    if (useInitialFlow)
    {
        cv::calcOpticalFlowFarneback(prev, next, initialFlow, pyrScale, levels, winSize,
                                     iterations, polyN, polySigma, flags);
        flags |= OPTFLOW_USE_INITIAL_FLOW;
    }

    declare.in(prev, next).out(flow);
    while (this->next())
    {
        if (useInitialFlow)
            initialFlow.copyTo(flow);
        startTimer();
        cv::calcOpticalFlowFarneback(prev, next, flow, pyrScale, levels, winSize,
                                     iterations, polyN, polySigma, flags);
        stopTimer();
    }

    EXPECT_EQ(prev.size(), flow.size());
    EXPECT_EQ(CV_32FC2, flow.type());
    EXPECT_TRUE(checkRange(flow));
    SANITY_CHECK_NOTHING();
}

}} // namespace
