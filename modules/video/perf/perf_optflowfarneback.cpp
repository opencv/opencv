// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

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
typedef tuple<Size, int, FarnebackGaussian> FarnebackSequenceParams;
typedef TestBaseWithParam<FarnebackSequenceParams> FarnebackSequence;

PERF_TEST_P(FarnebackSequence, calcOpticalFlowFarneback,
            Combine(Values(sz720p, sz1080p), Values(5, 7), FarnebackGaussian::all()))
{
    const Size size = get<0>(GetParam());
    const int polyN = get<1>(GetParam());
    const int flags = get<2>(GetParam());
    const double polySigma = polyN == 5 ? 1.1 : 1.5;
    const std::vector<Mat> frames = makeSequence(size);
    Mat flow(size, CV_32FC2);
    int pair = 0;

    // Nine samples cover all frame pairs. Wrap for the framework's default tenth sample.
    declare.iterations(SEQUENCE_LENGTH).time(900);
    TEST_CYCLE()
    {
        cv::calcOpticalFlowFarneback(frames[pair], frames[pair + 1], flow,
                                    0.5, 5, 13, 3, polyN, polySigma, flags);
        pair = (pair + 1) % (SEQUENCE_LENGTH - 1);
    }

    EXPECT_EQ(size, flow.size());
    EXPECT_EQ(CV_32FC2, flow.type());
    EXPECT_TRUE(checkRange(flow));
    SANITY_CHECK_NOTHING();
}

}} // namespace
