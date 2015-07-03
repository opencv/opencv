#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using namespace testing;
using std::tr1::make_tuple;
using std::tr1::get;

typedef std::tr1::tuple<Size, float> Size_Hval_t;
typedef perf::TestBaseWithParam<Size_Hval_t> Size_HvalType;

PERF_TEST_P( Size_HvalType, fastNlMeansDenoisingColored,
    Combine(
        SZ_ALL_HD,
        Values( 10.f )
    )
)
{
    Size size = std::tr1::get<0>(GetParam());
    int h = std::tr1::get<1>(GetParam());

    Mat src(size, CV_8UC3);
    Mat dst(size, CV_8UC3);

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE() fastNlMeansDenoisingColored(src, dst, h, h, 7, 21);

    SANITY_CHECK(dst);
}

PERF_TEST_P( Size_HvalType, halNlMeansDenoising,
    Combine(
        SZ_ALL_HD,
        Values( 10.f )
    )
)
{
    Size size = std::tr1::get<0>(GetParam());
    int h = std::tr1::get<1>(GetParam());

    Mat src(size, CV_8UC3);
    Mat dst(size, CV_8UC3);

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE() halNlMeansDenoising(src, dst, h);

    SANITY_CHECK(dst);
}
