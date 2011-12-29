#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

#define TYPICAL_MAT_SIZES_BITW_ARITHM  TYPICAL_MAT_SIZES
#define TYPICAL_MAT_TYPES_BITW_ARITHM  CV_8UC1, CV_8SC1, CV_8UC4, CV_32SC1, CV_32SC4
#define TYPICAL_MATS_BITW_ARITHM       testing::Combine(testing::Values(TYPICAL_MAT_SIZES_BITW_ARITHM), testing::Values(TYPICAL_MAT_TYPES_BITW_ARITHM))

PERF_TEST_P(Size_MatType, bitwise_not, TYPICAL_MATS_BITW_ARITHM)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());

    cv::Mat a = Mat(sz, type);
    cv::Mat c = Mat(sz, type);

    declare.in(a, WARMUP_RNG).out(c);

    TEST_CYCLE() cv::bitwise_not(a, c);

    SANITY_CHECK(c);
}

PERF_TEST_P(Size_MatType, bitwise_and, TYPICAL_MATS_BITW_ARITHM)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Mat b = Mat(sz, type);
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() bitwise_and(a, b, c);

    SANITY_CHECK(c);
}

PERF_TEST_P(Size_MatType, bitwise_or, TYPICAL_MATS_BITW_ARITHM)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Mat b = Mat(sz, type);
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() bitwise_or(a, b, c);

    SANITY_CHECK(c);
}

PERF_TEST_P(Size_MatType, bitwise_xor, TYPICAL_MATS_BITW_ARITHM)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Mat b = Mat(sz, type);
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() bitwise_xor(a, b, c);

    SANITY_CHECK(c);
}

