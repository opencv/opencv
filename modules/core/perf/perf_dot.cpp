#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

typedef tuple<MatType, int> MatType_Length_t;
typedef TestBaseWithParam<MatType_Length_t> MatType_Length;

PERF_TEST_P( MatType_Length, dot,
             testing::Combine(
                 testing::Values( CV_8UC1, CV_32SC1, CV_32FC1 ),
                 testing::Values( 32, 64, 128, 256, 512, 1024 )
                 ))
{
    int type = get<0>(GetParam());
    int size = get<1>(GetParam());
    Mat a(size, size, type);
    Mat b(size, size, type);

    declare.in(a, b, WARMUP_RNG);
    declare.time(100);

    double product;

    TEST_CYCLE_N(1000) product = a.dot(b);

    SANITY_CHECK(product, 1e-6, ERROR_RELATIVE);
}

} // namespace
