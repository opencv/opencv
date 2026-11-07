#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

typedef tuple<MatType, int> MatType_Length_t;
typedef TestBaseWithParam<MatType_Length_t> MatType_Length;

PERF_TEST_P( MatType_Length, dot,
             testing::Combine(
                 testing::Values( CV_8UC1, CV_8SC1, CV_16SC1, CV_16UC1, CV_32SC1, CV_32FC1 ),
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

PERF_TEST_P( MatType_Length, dot_new_types,
             testing::Combine(
                 testing::Values( CV_16FC1, CV_16BFC1, CV_BoolC1, CV_32UC1, CV_64UC1, CV_64SC1,
                                  CV_8F_E4M3FNC1, CV_8F_E4M3FNUZC1 ),
                 testing::Values( 32, 64, 128, 256, 512, 1024 )
                 ))
{
    int type = get<0>(GetParam());
    int size = get<1>(GetParam());
    Mat a(size, size, type);
    Mat b(size, size, type);

    // small integers are exact in every depth
    randu(a, -8, 8);
    randu(b, -8, 8);
    declare.in(a, b, WARMUP_NONE); // the default warmup calls cv::sum, which has no fp8 kernel
    declare.time(100);

    TEST_CYCLE_N(1000) a.dot(b);

    SANITY_CHECK_NOTHING();
}

} // namespace
