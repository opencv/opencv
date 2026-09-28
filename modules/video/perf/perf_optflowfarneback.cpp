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

enum { REUSE_OFF, REUSE_CHAINED, REUSE_UNCHAINED };
CV_ENUM(ReuseMode, REUSE_OFF, REUSE_CHAINED, REUSE_UNCHAINED)

typedef tuple<Size, ReuseMode> FarnebackParams;
typedef TestBaseWithParam<FarnebackParams> DenseOpticalFlow_Farneback;

// Every arm computes the same frame pairs. REUSE_UNCHAINED visits them last to first, so no
// call's first image is the previous call's second and every call misses.
PERF_TEST_P(DenseOpticalFlow_Farneback, perf,
            Combine(Values(szVGA, sz720p), ReuseMode::all()))
{
    const Size size = get<0>(GetParam());
    const int mode = get<1>(GetParam());

    const std::vector<Mat> frames = makeSequence(size);
    Mat flow;

    Ptr<FarnebackOpticalFlow> algo = FarnebackOpticalFlow::create();
    algo->setReuseExpansion(mode != REUSE_OFF);

    TEST_CYCLE()
    {
        for( int j = 0; j + 1 < SEQUENCE_LENGTH; j++ )
        {
            const int i = mode == REUSE_UNCHAINED ? SEQUENCE_LENGTH - 2 - j : j;
            algo->calc(frames[i], frames[i + 1], flow);
        }
    }

    SANITY_CHECK_NOTHING();
}

}} // namespace
