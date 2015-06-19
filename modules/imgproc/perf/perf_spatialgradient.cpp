#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using namespace testing;
using std::tr1::make_tuple;
using std::tr1::get;

typedef std::tr1::tuple<Size, int> Size_Ksize_t;
typedef perf::TestBaseWithParam<Size_Ksize_t> Size_Ksize;

PERF_TEST_P( Size_Ksize, spatialGradient,
    Combine(
        SZ_ALL_HD,
        Values( 3 )
    )
)
{
    Size size = std::tr1::get<0>(GetParam());
    int ksize = std::tr1::get<1>(GetParam());

    Mat src(size, CV_8UC1);
    Mat dx(size, CV_16SC1);
    Mat dy(size, CV_16SC1);

    declare.in(src, WARMUP_RNG).out(dx, dy);

    TEST_CYCLE() spatialGradient(src, dx, dy, ksize);

    SANITY_CHECK(dx);
    SANITY_CHECK(dy);
}

