#include "perf_precomp.hpp"

namespace opencv_test { namespace {
using namespace perf;

typedef perf::TestBaseWithParam<Size> SizePrm;

PERF_TEST_P( SizePrm, LUT,
             testing::Values(szQVGA, szVGA, sz1080p)
           )
{
    Size sz = GetParam();

    int maxValue = 255;

    Mat src(sz, CV_8UC1);
    randu(src, 0, maxValue);
    Mat lut(1, 256, CV_8UC1);
    randu(lut, 0, maxValue);
    Mat dst(sz, CV_8UC1);

    TEST_CYCLE() LUT(src, lut, dst);

    SANITY_CHECK(dst, 0.1);
}

PERF_TEST_P( SizePrm, LUT_multi,
             testing::Values(szQVGA, szVGA, sz1080p)
           )
{
    Size sz = GetParam();

    int maxValue = 255;

    Mat src(sz, CV_8UC3);
    randu(src, 0, maxValue);
    Mat lut(1, 256, CV_8UC1);
    randu(lut, 0, maxValue);
    Mat dst(sz, CV_8UC3);

    TEST_CYCLE() LUT(src, lut, dst);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P( SizePrm, LUT_multi2,
             testing::Values(szQVGA, szVGA, sz1080p)
           )
{
    Size sz = GetParam();

    int maxValue = 255;

    Mat src(sz, CV_8UC3);
    randu(src, 0, maxValue);
    Mat lut(1, 256, CV_8UC3);
    randu(lut, 0, maxValue);
    Mat dst(sz, CV_8UC3);

    TEST_CYCLE() LUT(src, lut, dst);

    SANITY_CHECK_NOTHING();
}

}} // namespace
