#include "perf_precomp.hpp"
#include <numeric>

namespace opencv_test
{
using namespace perf;

typedef Size_MatType BinaryOpTest;

PERF_TEST_P_(BinaryOpTest, min)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Mat b = Mat(sz, type);
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() cv::min(a, b, c);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, minScalarDouble)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Scalar b;
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() cv::min(a, b, c);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, minScalarSameType)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Scalar b;
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    if (CV_MAT_DEPTH(type) < CV_32S)
    {
        b = Scalar(1, 0, 3, 4); // don't pass non-integer values for 8U/8S/16U/16S processing
    }
    else if (CV_MAT_DEPTH(type) == CV_32S)
    {
        b = Scalar(1, 0, -3, 4); // don't pass non-integer values for 32S processing
    }

    TEST_CYCLE() cv::min(a, b, c);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, max)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Mat b = Mat(sz, type);
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() cv::max(a, b, c);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, maxScalarDouble)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Scalar b;
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() cv::max(a, b, c);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, maxScalarSameType)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Scalar b;
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    if (CV_MAT_DEPTH(type) < CV_32S)
    {
        b = Scalar(1, 0, 3, 4); // don't pass non-integer values for 8U/8S/16U/16S processing
    }
    else if (CV_MAT_DEPTH(type) == CV_32S)
    {
        b = Scalar(1, 0, -3, 4); // don't pass non-integer values for 32S processing
    }

    TEST_CYCLE() cv::max(a, b, c);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, absdiff)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Mat b = Mat(sz, type);
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    if (CV_MAT_DEPTH(type) == CV_32S)
    {
        //see ticket 1529: absdiff can be without saturation on 32S
        a /= 2;
        b /= 2;
    }

    TEST_CYCLE() cv::absdiff(a, b, c);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, absdiffScalarDouble)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Scalar b;
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    if (CV_MAT_DEPTH(type) == CV_32S)
    {
        //see ticket 1529: absdiff can be without saturation on 32S
        a /= 2;
        b /= 2;
    }

    TEST_CYCLE() cv::absdiff(a, b, c);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, absdiffScalarSameType)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Scalar b;
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    if (CV_MAT_DEPTH(type) < CV_32S)
    {
        b = Scalar(1, 0, 3, 4); // don't pass non-integer values for 8U/8S/16U/16S processing
    }
    else if (CV_MAT_DEPTH(type) == CV_32S)
    {
        //see ticket 1529: absdiff can be without saturation on 32S
        a /= 2;
        b = Scalar(1, 0, -3, 4); // don't pass non-integer values for 32S processing
    }

    TEST_CYCLE() cv::absdiff(a, b, c);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, add)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Mat b = Mat(sz, type);
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);
    declare.time(50);

    if (CV_MAT_DEPTH(type) == CV_32S)
    {
        //see ticket 1529: add can be without saturation on 32S
        a /= 2;
        b /= 2;
    }

    TEST_CYCLE() cv::add(a, b, c);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, addScalarDouble)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Scalar b;
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    if (CV_MAT_DEPTH(type) == CV_32S)
    {
        //see ticket 1529: add can be without saturation on 32S
        a /= 2;
        b /= 2;
    }

    TEST_CYCLE() cv::add(a, b, c);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, addScalarSameType)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Scalar b;
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    if (CV_MAT_DEPTH(type) < CV_32S)
    {
        b = Scalar(1, 0, 3, 4); // don't pass non-integer values for 8U/8S/16U/16S processing
    }
    else if (CV_MAT_DEPTH(type) == CV_32S)
    {
        //see ticket 1529: add can be without saturation on 32S
        a /= 2;
        b = Scalar(1, 0, -3, 4); // don't pass non-integer values for 32S processing
    }

    TEST_CYCLE() cv::add(a, b, c, noArray(), type);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, subtract)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Mat b = Mat(sz, type);
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    if (CV_MAT_DEPTH(type) == CV_32S)
    {
        //see ticket 1529: subtract can be without saturation on 32S
        a /= 2;
        b /= 2;
    }

    TEST_CYCLE() cv::subtract(a, b, c);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, subtractScalarDouble)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Scalar b;
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    if (CV_MAT_DEPTH(type) == CV_32S)
    {
        //see ticket 1529: subtract can be without saturation on 32S
        a /= 2;
        b /= 2;
    }

    TEST_CYCLE() cv::subtract(a, b, c);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, subtractScalarSameType)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Scalar b;
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    if (CV_MAT_DEPTH(type) < CV_32S)
    {
        b = Scalar(1, 0, 3, 4); // don't pass non-integer values for 8U/8S/16U/16S processing
    }
    else if (CV_MAT_DEPTH(type) == CV_32S)
    {
        //see ticket 1529: subtract can be without saturation on 32S
        a /= 2;
        b = Scalar(1, 0, -3, 4); // don't pass non-integer values for 32S processing
    }

    TEST_CYCLE() cv::subtract(a, b, c, noArray(), type);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, multiply)
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

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, multiplyScale)
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

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, divide)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a(sz, type), b(sz, type), c(sz, type);
    double scale = 0.5;

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() cv::divide(a, b, c, scale);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, reciprocal)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat b(sz, type), c(sz, type);
    double scale = 0.5;

    declare.in(b, WARMUP_RNG).out(c);

    TEST_CYCLE() cv::divide(scale, b, c);

    SANITY_CHECK_NOTHING();
}


PERF_TEST_P_(BinaryOpTest, transposeND)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type).reshape(1);

    std::vector<int> order(a.dims);
    std::iota(order.begin(), order.end(), 0);
    std::reverse(order.begin(), order.end());

    std::vector<int> new_sz(a.dims);
    std::copy(a.size.p, a.size.p + a.dims, new_sz.begin());
    std::reverse(new_sz.begin(), new_sz.end());
    cv::Mat b = Mat(new_sz, type);

    declare.in(a,WARMUP_RNG).out(b);

    TEST_CYCLE() cv::transposeND(a, order, b);

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/*nothing*/ , BinaryOpTest,
    testing::Combine(
        testing::Values(szVGA, sz720p, sz1080p),
        testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_8SC1, CV_16SC1, CV_16SC2, CV_16SC3, CV_16SC4, CV_32SC1, CV_32FC1)
    )
);

} // namespace
