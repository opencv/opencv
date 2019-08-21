#include "perf_precomp.hpp"

namespace opencv_test { namespace {
using namespace perf;

typedef TestBaseWithParam< std::pair<string, string> > ImagePair;

std::pair<string, string> impair(const char* im1, const char* im2)
{
    return std::make_pair(string(im1), string(im2));
}

PERF_TEST_P(ImagePair, OpticalFlowDual_TVL1, testing::Values(impair("cv/optflow/RubberWhale1.png", "cv/optflow/RubberWhale2.png")))
{
    declare.time(260);

    Mat frame1 = imread(getDataPath(GetParam().first), IMREAD_GRAYSCALE);
    Mat frame2 = imread(getDataPath(GetParam().second), IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame1.empty());
    ASSERT_FALSE(frame2.empty());

    Mat flow;

    Ptr<DenseOpticalFlow> tvl1 = createOptFlow_DualTVL1();

    TEST_CYCLE() tvl1->calc(frame1, frame2, flow);

    SANITY_CHECK_NOTHING();
}

}} // namespace
