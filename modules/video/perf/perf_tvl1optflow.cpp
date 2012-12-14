#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;

typedef TestBaseWithParam< pair<string, string> > ImagePair;

pair<string, string> impair(const char* im1, const char* im2)
{
    return make_pair(string(im1), string(im2));
}

PERF_TEST_P(ImagePair, OpticalFlowDual_TVL1, testing::Values(impair("cv/optflow/RubberWhale1.png", "cv/optflow/RubberWhale2.png")))
{
    declare.time(40);

    Mat frame1 = imread(getDataPath(GetParam().first), IMREAD_GRAYSCALE);
    Mat frame2 = imread(getDataPath(GetParam().second), IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame1.empty());
    ASSERT_FALSE(frame2.empty());

    Mat flow;

    OpticalFlowDual_TVL1 tvl1;

    TEST_CYCLE()
    {
        tvl1(frame1, frame2, flow);
    }

    SANITY_CHECK(flow, 0.02);
}
