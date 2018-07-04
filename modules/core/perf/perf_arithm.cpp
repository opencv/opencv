#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

#define TYPICAL_MAT_SIZES_CORE_ARITHM   szVGA, sz720p, sz1080p
#define TYPICAL_MAT_TYPES_CORE_ARITHM   CV_8UC1, CV_8SC1, CV_16SC1, CV_16SC2, CV_16SC3, CV_16SC4, CV_8UC4, CV_32SC1, CV_32FC1
#define TYPICAL_MATS_CORE_ARITHM        testing::Combine( testing::Values( TYPICAL_MAT_SIZES_CORE_ARITHM ), testing::Values( TYPICAL_MAT_TYPES_CORE_ARITHM ) )

PERF_TEST_P(Size_MatType, min, TYPICAL_MATS_CORE_ARITHM)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Mat b = Mat(sz, type);
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() cv::min(a, b, c);

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

    TEST_CYCLE() cv::min(a, b, c);

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

    TEST_CYCLE() cv::max(a, b, c);

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

    TEST_CYCLE() cv::max(a, b, c);

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

    double eps = 1e-8;
    if (CV_MAT_DEPTH(type) == CV_32S)
    {
        //see ticket 1529: absdiff can be without saturation on 32S
        a /= 2;
        b /= 2;
        eps = 1;
    }

    TEST_CYCLE() cv::absdiff(a, b, c);

    SANITY_CHECK(c, eps);
}

PERF_TEST_P(Size_MatType, absdiffScalar, TYPICAL_MATS_CORE_ARITHM)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Scalar b;
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    double eps = 1e-8;
    if (CV_MAT_DEPTH(type) == CV_32S)
    {
        //see ticket 1529: absdiff can be without saturation on 32S
        a /= 2;
        b /= 2;
        eps = 1;
    }

    TEST_CYCLE() cv::absdiff(a, b, c);

    SANITY_CHECK(c, eps);
}

PERF_TEST_P(Size_MatType, add, TYPICAL_MATS_CORE_ARITHM)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Mat b = Mat(sz, type);
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);
    declare.time(50);

    double eps = 1e-8;
    if (CV_MAT_DEPTH(type) == CV_32S)
    {
        //see ticket 1529: add can be without saturation on 32S
        a /= 2;
        b /= 2;
        eps = 1;
    }

    TEST_CYCLE() cv::add(a, b, c);

    SANITY_CHECK(c, eps);
}

PERF_TEST_P(Size_MatType, addScalar, TYPICAL_MATS_CORE_ARITHM)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Scalar b;
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    double eps = 1e-8;
    if (CV_MAT_DEPTH(type) == CV_32S)
    {
        //see ticket 1529: add can be without saturation on 32S
        a /= 2;
        b /= 2;
        eps = 1;
    }

    TEST_CYCLE() cv::add(a, b, c);

    SANITY_CHECK(c, eps);
}

PERF_TEST_P(Size_MatType, subtract, TYPICAL_MATS_CORE_ARITHM)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Mat b = Mat(sz, type);
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    double eps = 1e-8;
    if (CV_MAT_DEPTH(type) == CV_32S)
    {
        //see ticket 1529: subtract can be without saturation on 32S
        a /= 2;
        b /= 2;
        eps = 1;
    }

    TEST_CYCLE() cv::subtract(a, b, c);

    SANITY_CHECK(c, eps);
}

PERF_TEST_P(Size_MatType, subtractScalar, TYPICAL_MATS_CORE_ARITHM)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Scalar b;
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    double eps = 1e-8;
    if (CV_MAT_DEPTH(type) == CV_32S)
    {
        //see ticket 1529: subtract can be without saturation on 32S
        a /= 2;
        b /= 2;
        eps = 1;
    }

    TEST_CYCLE() cv::subtract(a, b, c);

    SANITY_CHECK(c, eps);
}

PERF_TEST_P(Size_MatType, multiply, TYPICAL_MATS_CORE_ARITHM)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a(sz, type), b(sz, type), c(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);
    if (CV_MAT_DEPTH(type) == CV_32S)
    {
        //According to docs, saturation is not applied when result is 32bit integer
        a /= (2 << 16);
        b /= (2 << 16);
    }

    TEST_CYCLE() cv::multiply(a, b, c);

    SANITY_CHECK(c, 1e-8);
}

PERF_TEST_P(Size_MatType, multiplyScale, TYPICAL_MATS_CORE_ARITHM)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a(sz, type), b(sz, type), c(sz, type);
    double scale = 0.5;

    declare.in(a, b, WARMUP_RNG).out(c);

    if (CV_MAT_DEPTH(type) == CV_32S)
    {
        //According to docs, saturation is not applied when result is 32bit integer
        a /= (2 << 16);
        b /= (2 << 16);
    }

    TEST_CYCLE() cv::multiply(a, b, c, scale);

    SANITY_CHECK(c, 1e-8);
}

PERF_TEST_P(Size_MatType, divide, TYPICAL_MATS_CORE_ARITHM)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a(sz, type), b(sz, type), c(sz, type);
    double scale = 0.5;

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() cv::divide(a, b, c, scale);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Size_MatType, reciprocal, TYPICAL_MATS_CORE_ARITHM)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat b(sz, type), c(sz, type);
    double scale = 0.5;

    declare.in(b, WARMUP_RNG).out(c);

    TEST_CYCLE() cv::divide(scale, b, c);

    SANITY_CHECK_NOTHING();
}

} // namespace
