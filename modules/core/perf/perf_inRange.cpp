#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

#define TYPICAL_MAT_TYPES_INRANGE  CV_8UC1, CV_8UC4, CV_8SC1, CV_16UC1, CV_16SC1, CV_32SC1, CV_32FC1, CV_32FC4
#define TYPICAL_MATS_INRANGE       testing::Combine(testing::Values(szVGA, sz720p, sz1080p), testing::Values(TYPICAL_MAT_TYPES_INRANGE))

PERF_TEST_P(Size_MatType, inRange, TYPICAL_MATS_INRANGE)
{
    Size size = get<0>(GetParam());
    int type = get<1>(GetParam());
    Mat src1(size, type);
    Mat src2(size, type);
    Mat src3(size, type);
    Mat dst(size, type);

    declare.in(src1, src2, src3, WARMUP_RNG).out(dst);

    TEST_CYCLE() inRange( src1, src2, src3, dst );

    SANITY_CHECK(dst);
}

} // namespace
