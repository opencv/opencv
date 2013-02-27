#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

typedef std::tr1::tuple<Size, MatType, int> Size_Depth_Channels_t;
typedef perf::TestBaseWithParam<Size_Depth_Channels_t> Size_Depth_Channels;

PERF_TEST_P( Size_Depth_Channels, split,
             testing::Combine
             (
                 testing::Values(TYPICAL_MAT_SIZES),
                 testing::Values(CV_8U, CV_16S, CV_32F, CV_64F),
                 testing::Values(2, 3, 4)
             )
           )
{
    Size sz = get<0>(GetParam());
    int depth = get<1>(GetParam());
    int channels = get<2>(GetParam());

    Mat m(sz, CV_MAKETYPE(depth, channels));
    randu(m, 0, 255);

    vector<Mat> mv;
    int runs = (sz.width <= 640) ? 8 : 1;
    TEST_CYCLE_MULTIRUN(runs) split(m, (vector<Mat>&)mv);

    SANITY_CHECK(mv, 1e-12);
}
