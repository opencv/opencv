#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

typedef std::tr1::tuple<Size, MatType, int> Size_SrcDepth_DstChannels_t;
typedef perf::TestBaseWithParam<Size_SrcDepth_DstChannels_t> Size_SrcDepth_DstChannels;

PERF_TEST_P( Size_SrcDepth_DstChannels, merge,
             testing::Combine
             (
                 testing::Values(TYPICAL_MAT_SIZES),
                 testing::Values(CV_8U, CV_16S, CV_32S, CV_32F, CV_64F),
                 testing::Values(2, 3, 4)
             )
           )
{
    Size sz = get<0>(GetParam());
    int srcDepth = get<1>(GetParam());
    int dstChannels = get<2>(GetParam());

    vector<Mat> mv;
    for( int i = 0; i < dstChannels; ++i )
    {
        mv.push_back( Mat(sz, CV_MAKETYPE(srcDepth, 1)) );
        randu(mv[i], 0, 255);
    }

    Mat dst;
    int runs = (sz.width <= 640) ? 8 : 1;
    TEST_CYCLE_MULTIRUN(runs) merge( (vector<Mat> &)mv, dst );

    SANITY_CHECK(dst, 1e-12);
}
