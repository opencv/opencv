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

#ifdef __aarch64__
    // looks like random generator produces a little bit
    // different source data on aarch64 platform and
    // eps should be increased to allow the tests pass
    SANITY_CHECK(dst, (srcDepth == CV_32F ? 1.55e-5 : 1e-12));
#else
    SANITY_CHECK(dst, 1e-12);
#endif
}
