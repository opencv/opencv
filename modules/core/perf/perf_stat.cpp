#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

PERF_TEST_P(Size_MatType, sum, TYPICAL_MATS)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());

    Mat arr(sz, type);
    Scalar s;

    declare.in(arr, WARMUP_RNG).out(s);

    TEST_CYCLE() s = sum(arr);

    SANITY_CHECK(s, 1e-6, ERROR_RELATIVE);
}

PERF_TEST_P(Size_MatType, mean, TYPICAL_MATS)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());

    Mat src(sz, type);
    Scalar s;

    declare.in(src, WARMUP_RNG).out(s);

    TEST_CYCLE() s = mean(src);

    SANITY_CHECK(s, 1e-5);
}

PERF_TEST_P(Size_MatType, mean_mask, TYPICAL_MATS)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());

    Mat src(sz, type);
    Mat mask = Mat::ones(src.size(), CV_8U);
    Scalar s;

    declare.in(src, WARMUP_RNG).in(mask).out(s);

    TEST_CYCLE() s = mean(src, mask);

    SANITY_CHECK(s, 5e-5);
}

PERF_TEST_P(Size_MatType, meanStdDev, TYPICAL_MATS)
{
    Size sz = get<0>(GetParam());
    int matType = get<1>(GetParam());

    Mat src(sz, matType);
    Scalar mean;
    Scalar dev;

    declare.in(src, WARMUP_RNG).out(mean, dev);

    TEST_CYCLE() meanStdDev(src, mean, dev);

    SANITY_CHECK(mean, 1e-5, ERROR_RELATIVE);
    SANITY_CHECK(dev, 1e-5, ERROR_RELATIVE);
}

PERF_TEST_P(Size_MatType, meanStdDev_mask, TYPICAL_MATS)
{
    Size sz = get<0>(GetParam());
    int matType = get<1>(GetParam());

    Mat src(sz, matType);
    Mat mask = Mat::ones(sz, CV_8U);
    Scalar mean;
    Scalar dev;

    declare.in(src, WARMUP_RNG).in(mask).out(mean, dev);

    TEST_CYCLE() meanStdDev(src, mean, dev, mask);

    SANITY_CHECK(mean, 1e-5);
    SANITY_CHECK(dev, 1e-5);
}

PERF_TEST_P(Size_MatType, countNonZero, testing::Combine( testing::Values( TYPICAL_MAT_SIZES ), testing::Values( CV_8UC1, CV_8SC1, CV_16UC1, CV_16SC1, CV_32SC1, CV_32FC1, CV_64FC1 ) ))
{
    Size sz = get<0>(GetParam());
    int matType = get<1>(GetParam());

    Mat src(sz, matType);
    int cnt = 0;

    declare.in(src, WARMUP_RNG);

    int runs = (sz.width <= 640) ? 8 : 1;
    TEST_CYCLE_MULTIRUN(runs) cnt = countNonZero(src);

    SANITY_CHECK(cnt);
}
