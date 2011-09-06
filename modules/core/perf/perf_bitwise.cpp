#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;

#define TYPICAL_MAT_SIZES_BITWNOT  TYPICAL_MAT_SIZES
#define TYPICAL_MAT_TYPES_BITWNOT  CV_8SC1, CV_8SC4, CV_32SC1, CV_32SC4
#define TYPICAL_MATS_BITWNOT       testing::Combine( testing::Values( TYPICAL_MAT_SIZES_BITWNOT), testing::Values( TYPICAL_MAT_TYPES_BITWNOT) )

PERF_TEST_P(Size_MatType, bitwise_not, TYPICAL_MATS_BITWNOT)
{
    Size sz = std::tr1::get<0>(GetParam());
    int type = std::tr1::get<1>(GetParam());

    cv::Mat a = Mat(sz, type);
    cv::Mat c = Mat(sz, type);

    declare.in(a, WARMUP_RNG).out(c);

    TEST_CYCLE(100) cv::bitwise_not(a, c);

    SANITY_CHECK(c);
}

