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
        Mat shift = (Mat_<double>(2, 3) << 1, 0, 1.7 * i, 0, 1, -1.1 * i);
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
    algo->setReuseExpansion(reuseExpansion);

    TEST_CYCLE()
    {
        for( int i = 0; i + 1 < SEQUENCE_LENGTH; i++ )
            algo->calc(frames[i], frames[i + 1], flow);
    }

    SANITY_CHECK_NOTHING();
}

}} // namespace
