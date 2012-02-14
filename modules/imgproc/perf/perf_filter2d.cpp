#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using namespace testing;
using std::tr1::make_tuple;
using std::tr1::get;


CV_ENUM(BorderMode, BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT_101);

typedef TestBaseWithParam< tr1::tuple<Size, int, BorderMode> > TestFilter2d;


PERF_TEST_P( TestFilter2d, Filter2d,
             Combine(
                Values( Size(320, 240), szVGA, sz720p, sz1080p ),
                Values( 3, 5 ),
                ValuesIn( BorderMode::all() )
             )
)
{
    Size sz;
    int borderMode, kSize;
    sz         = get<0>(GetParam());
    kSize      = get<1>(GetParam());
    borderMode = get<2>(GetParam());

    Mat src(sz, CV_8UC4);
    Mat dst(sz, CV_8UC4);

    Mat kernel(kSize, kSize, CV_32FC1);
    randu(kernel, -3, 10);
    float s = (float)fabs( sum(kernel)[0] );
    if(s > 1e-3) kernel /= s;

    declare.in(src, WARMUP_RNG).out(dst).time(20);

    TEST_CYCLE() filter2D(src, dst, CV_8UC4, kernel, Point(1, 1), 0., borderMode);

    SANITY_CHECK(dst);
}


