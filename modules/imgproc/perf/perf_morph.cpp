#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;

#define TYPICAL_MAT_TYPES_MORPH  CV_8UC1, CV_8UC4
#define TYPICAL_MATS_MORPH       testing::Combine( SZ_ALL_GA, testing::Values( TYPICAL_MAT_TYPES_MORPH) )

PERF_TEST_P(Size_MatType, erode, TYPICAL_MATS_MORPH)
{
    Size sz = std::tr1::get<0>(GetParam());
    int type = std::tr1::get<1>(GetParam());

    Mat src(sz, type);
    Mat dst(sz, type);

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE(100) 
    {
        erode(src, dst, noArray());
    }

    SANITY_CHECK(dst);
}

PERF_TEST_P(Size_MatType, dilate, TYPICAL_MATS_MORPH)
{
    Size sz = std::tr1::get<0>(GetParam());
    int type = std::tr1::get<1>(GetParam());

    Mat src(sz, type);
    Mat dst(sz, type);

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE(100)
    {
        dilate(src, dst, noArray());
    }

    SANITY_CHECK(dst);
}
