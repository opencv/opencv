#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;

typedef perf::TestBaseWithParam<size_t> VectorLength;

PERF_TEST_P(VectorLength, phase32f, testing::Values(128, 1000, 128*1024, 512*1024, 1024*1024))
{
    int length = GetParam();
    vector<float> X(length);
    vector<float> Y(length);
    vector<float> angle(length);

    declare.in(X, Y, WARMUP_RNG).out(angle);

    TEST_CYCLE(200) cv::phase(X, Y, angle, true);

    SANITY_CHECK(angle);
}
