#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

PERF_TEST_P(Size_MatType, pyrDown, testing::Combine(
                testing::Values(sz1080p, sz720p, szVGA, szQVGA, szODD),
                testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16SC1, CV_16SC3, CV_16SC4)
                )
            )
{
    Size sz = get<0>(GetParam());
    int matType = get<1>(GetParam());

    Mat src(sz, matType);
    Mat dst((sz.height + 1)/2, (sz.width + 1)/2, matType);

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE() pyrDown(src, dst);

    SANITY_CHECK(dst);
}

PERF_TEST_P(Size_MatType, pyrUp, testing::Combine(
                testing::Values(sz720p, szVGA, szQVGA, szODD),
                testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16SC1, CV_16SC3, CV_16SC4)
                )
            )
{
    Size sz = get<0>(GetParam());
    int matType = get<1>(GetParam());

    Mat src(sz, matType);
    Mat dst(sz.height*2, sz.width*2, matType);

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE() pyrUp(src, dst);

    SANITY_CHECK(dst);
}
