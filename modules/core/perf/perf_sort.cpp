#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

#define TYPICAL_MAT_SIZES_SORT  TYPICAL_MAT_SIZES
#define TYPICAL_MAT_TYPES_SORT  CV_8UC1, CV_16UC1, CV_32FC1
#define SORT_TYPES              SORT_EVERY_ROW | SORT_ASCENDING, SORT_EVERY_ROW | SORT_DESCENDING
#define TYPICAL_MATS_SORT       testing::Combine( testing::Values(TYPICAL_MAT_SIZES_SORT), testing::Values(TYPICAL_MAT_TYPES_SORT), testing::Values(SORT_TYPES) )

typedef tuple<Size, MatType, int> sortParams;
typedef TestBaseWithParam<sortParams> sortFixture;

PERF_TEST_P(sortFixture, sort, TYPICAL_MATS_SORT)
{
    const sortParams params = GetParam();
    const Size sz = get<0>(params);
    const int type = get<1>(params), flags = get<2>(params);

    cv::Mat a(sz, type), b(sz, type);

    declare.in(a, WARMUP_RNG).out(b);

    TEST_CYCLE() cv::sort(a, b, flags);

    SANITY_CHECK(b);
}

typedef sortFixture sortIdxFixture;

#undef SORT_TYPES
#define SORT_TYPES SORT_EVERY_COLUMN | SORT_ASCENDING, SORT_EVERY_COLUMN | SORT_DESCENDING

PERF_TEST_P(sortIdxFixture, sorIdx, TYPICAL_MATS_SORT)
{
    const sortParams params = GetParam();
    const Size sz = get<0>(params);
    const int type = get<1>(params), flags = get<2>(params);

    cv::Mat a(sz, type), b(sz, type);

    declare.in(a, WARMUP_RNG).out(b);

    TEST_CYCLE() cv::sortIdx(a, b, flags);

    SANITY_CHECK_NOTHING();
}

} // namespace
