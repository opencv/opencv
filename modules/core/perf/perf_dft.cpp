#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

#define MAT_TYPES_DFT  CV_32FC1, CV_32FC2, CV_64FC1
#define MAT_SIZES_DFT  cv::Size(320, 480), cv::Size(800, 600), cv::Size(1280, 1024), sz1080p, sz2K
CV_ENUM(FlagsType, 0, DFT_INVERSE, DFT_SCALE, DFT_COMPLEX_OUTPUT, DFT_ROWS, DFT_INVERSE|DFT_COMPLEX_OUTPUT)
#define TEST_MATS_DFT  testing::Combine(testing::Values(MAT_SIZES_DFT), testing::Values(MAT_TYPES_DFT), FlagsType::all())

typedef std::tr1::tuple<Size, MatType, FlagsType> Size_MatType_FlagsType_t;
typedef perf::TestBaseWithParam<Size_MatType_FlagsType_t> Size_MatType_FlagsType;

PERF_TEST_P(Size_MatType_FlagsType, dft, TEST_MATS_DFT)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    int flags = get<2>(GetParam());

    Mat src(sz, type);
    Mat dst(sz, type);

    declare.in(src, WARMUP_RNG).time(60);

    TEST_CYCLE() dft(src, dst, flags);

    SANITY_CHECK(dst, 1e-5, ERROR_RELATIVE);
}