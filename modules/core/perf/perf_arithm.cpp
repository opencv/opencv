#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

#define TYPICAL_MAT_SIZES_CORE_ARITHM   TYPICAL_MAT_SIZES 
#define TYPICAL_MAT_TYPES_CORE_ARITHM   CV_8UC1, CV_8SC1, CV_16SC1, CV_16SC2, CV_16SC3, CV_16SC4, CV_8UC4, CV_32SC1, CV_32FC1
#define TYPICAL_MATS_CORE_ARITHM        testing::Combine( testing::Values( TYPICAL_MAT_SIZES_CORE_ARITHM ), testing::Values( TYPICAL_MAT_TYPES_CORE_ARITHM ) )

#ifdef ANDROID
PERF_TEST(convert, cvRound)
{
    double number = theRNG().uniform(-100, 100);

    int result = 0;

    TEST_CYCLE_N(1000)
    {
        for (int i = 0; i < 500000; ++i)
            result += cvRound(number);
    }

    SANITY_CHECK(result);
}
#endif

PERF_TEST_P(Size_MatType, min, TYPICAL_MATS_CORE_ARITHM)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Mat b = Mat(sz, type);
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() min(a, b, c);

    SANITY_CHECK(c);
}

PERF_TEST_P(Size_MatType, minScalar, TYPICAL_MATS_CORE_ARITHM)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Scalar b;
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() min(a, b, c);

    SANITY_CHECK(c);
}

PERF_TEST_P(Size_MatType, max, TYPICAL_MATS_CORE_ARITHM)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Mat b = Mat(sz, type);
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() max(a, b, c);

    SANITY_CHECK(c);
}

PERF_TEST_P(Size_MatType, maxScalar, TYPICAL_MATS_CORE_ARITHM)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Scalar b;
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() max(a, b, c);

    SANITY_CHECK(c);
}

PERF_TEST_P(Size_MatType, absdiff, TYPICAL_MATS_CORE_ARITHM)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Mat b = Mat(sz, type);
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() absdiff(a, b, c);

    //see ticket 1529: absdiff can be without saturation on 32S
    if (CV_MAT_DEPTH(type) != CV_32S)
        SANITY_CHECK(c, 1e-8);
}

PERF_TEST_P(Size_MatType, absdiffScalar, TYPICAL_MATS_CORE_ARITHM)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Scalar b;
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() absdiff(a, b, c);

    //see ticket 1529: absdiff can be without saturation on 32S
    if (CV_MAT_DEPTH(type) != CV_32S)
        SANITY_CHECK(c, 1e-8);
}

PERF_TEST_P(Size_MatType, add, TYPICAL_MATS_CORE_ARITHM)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Mat b = Mat(sz, type);
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() add(a, b, c);

    //see ticket 1529: add can be without saturation on 32S
    if (CV_MAT_DEPTH(type) != CV_32S)
        SANITY_CHECK(c, 1e-8);
}

PERF_TEST_P(Size_MatType, addScalar, TYPICAL_MATS_CORE_ARITHM)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Scalar b;
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() add(a, b, c);

    //see ticket 1529: add can be without saturation on 32S
    if (CV_MAT_DEPTH(type) != CV_32S)
        SANITY_CHECK(c, 1e-8);
}

PERF_TEST_P(Size_MatType, subtract, TYPICAL_MATS_CORE_ARITHM)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Mat b = Mat(sz, type);
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() subtract(a, b, c);

    //see ticket 1529: subtract can be without saturation on 32S
    if (CV_MAT_DEPTH(type) != CV_32S)
        SANITY_CHECK(c, 1e-8);
}

PERF_TEST_P(Size_MatType, subtractScalar, TYPICAL_MATS_CORE_ARITHM)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Scalar b;
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() subtract(a, b, c);

    //see ticket 1529: subtract can be without saturation on 32S
    if (CV_MAT_DEPTH(type) != CV_32S)
        SANITY_CHECK(c, 1e-8);
}
