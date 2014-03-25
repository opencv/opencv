#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using namespace testing;
using std::tr1::make_tuple;
using std::tr1::get;

typedef TestBaseWithParam<Size > CreateHanningWindowFixture;

PERF_TEST_P( CreateHanningWindowFixture, CreateHanningWindow, Values(szVGA, sz1080p))
{
    const Size size = GetParam();
    Mat dst(size, CV_32FC1);

    declare.in(dst, WARMUP_RNG).out(dst);

    TEST_CYCLE() cv::createHanningWindow(dst, size, CV_32FC1);

    SANITY_CHECK(dst, 1e-6, ERROR_RELATIVE);
}
