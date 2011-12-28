#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;

#define TYPICAL_MAT_SIZES_MINMAX  TYPICAL_MAT_SIZES 
#define TYPICAL_MAT_TYPES_MINMAX  CV_8SC1, CV_8SC4, CV_32SC1, CV_32FC1
#define TYPICAL_MATS_MINMAX       testing::Combine( testing::Values( TYPICAL_MAT_SIZES_MINMAX), testing::Values( TYPICAL_MAT_TYPES_MINMAX) )

PERF_TEST_P(Size_MatType, min_double, TYPICAL_MATS_MINMAX)
{
    Size sz = std::tr1::get<0>(GetParam());
    int type = std::tr1::get<1>(GetParam());

    cv::Mat a = Mat(sz, type);
    cv::Mat c = Mat(sz, type);

    declare.in(a, ::perf::TestBase::WARMUP_RNG).out(c);

    TEST_CYCLE(100) cv::min(a, 10, c);

    SANITY_CHECK(c);
}

PERF_TEST_P(Size_MatType, max_double, TYPICAL_MATS_MINMAX)
{
    Size sz = std::tr1::get<0>(GetParam());
    int type = std::tr1::get<1>(GetParam());

    cv::Mat a = Mat(sz, type);
    cv::Mat c = Mat(sz, type);

    declare.in(a, ::perf::TestBase::WARMUP_RNG).out(c);

    TEST_CYCLE(100) cv::max(a, 10, c);

    SANITY_CHECK(c);
}
