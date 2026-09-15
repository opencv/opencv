// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test { namespace {

// One iteration walks a short sequence, which is how dense flow is normally used, so the
// one cold call at the head of a sequence dilutes the saving a little.
//
// sharedInstance keeps one object across the sequence and so reuses; sharedInstanceNoReuse
// is the same object with collectGarbage() before every calc(), which drops the held
// expansion and nothing else on this path, so the gap between the two is the saving by
// itself; calcOpticalFlowFarneback is how most callers reach this code and creates an
// instance per pair, differing in its allocation as well.

const int SEQUENCE_LENGTH = 10;

// create()'s defaults, passed to the free function too, so every arm does the same work.
const double PYR_SCALE = 0.5;
const int NUM_LEVELS = 5;
const int WIN_SIZE = 13;
const int NUM_ITERS = 10;
const int POLY_N = 5;
const double POLY_SIGMA = 1.1;

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

typedef tuple<Size, std::string> FarnebackParams;
typedef TestBaseWithParam<FarnebackParams> DenseOpticalFlow_Farneback;

// Two sizes: an iteration is a whole sequence, so six cases already take about a minute at
// 20 samples, and the saving is a ratio that does not need a third size.
PERF_TEST_P(DenseOpticalFlow_Farneback, perf,
            Combine(Values(szQVGA, szVGA),
                    Values("calcOpticalFlowFarneback", "sharedInstance",
                           "sharedInstanceNoReuse")))
{
    const Size size = get<0>(GetParam());
    const std::string mode = get<1>(GetParam());
    const bool shared = mode != "calcOpticalFlowFarneback";
    const bool dropExpansion = mode == "sharedInstanceNoReuse";

    const std::vector<Mat> frames = makeSequence(size);
    Mat flow;

    // Built once, since what is measured is where the expansion comes from rather than the
    // allocation. Every cycle still starts cold: the frame held from the end of the last
    // one is not frames[0].
    Ptr<FarnebackOpticalFlow> instance = FarnebackOpticalFlow::create(
        NUM_LEVELS, PYR_SCALE, false, WIN_SIZE, NUM_ITERS, POLY_N, POLY_SIGMA, 0);

    TEST_CYCLE()
    {
        for( int i = 0; i + 1 < SEQUENCE_LENGTH; i++ )
        {
            if( !shared )
            {
                calcOpticalFlowFarneback(frames[i], frames[i + 1], flow, PYR_SCALE, NUM_LEVELS,
                                         WIN_SIZE, NUM_ITERS, POLY_N, POLY_SIGMA, 0);
                continue;
            }
            if( dropExpansion )
                instance->collectGarbage();
            instance->calc(frames[i], frames[i + 1], flow);
        }
    }

    SANITY_CHECK_NOTHING();
}

}} // namespace
