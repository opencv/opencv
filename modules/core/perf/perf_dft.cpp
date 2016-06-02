#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

///////////////////////////////////////////////////////dft//////////////////////////////////////////////////////////////

#define MAT_TYPES_DFT  CV_32FC1, CV_32FC2, CV_64FC1
#define MAT_SIZES_DFT  cv::Size(320, 480), cv::Size(800, 600), cv::Size(1280, 1024), sz1080p, sz2K
CV_ENUM(FlagsType, 0, DFT_INVERSE, DFT_SCALE, DFT_COMPLEX_OUTPUT, DFT_ROWS, DFT_INVERSE|DFT_COMPLEX_OUTPUT)
#define TEST_MATS_DFT  testing::Combine(testing::Values(MAT_SIZES_DFT), testing::Values(MAT_TYPES_DFT), FlagsType::all(), testing::Values(true, false))

typedef std::tr1::tuple<Size, MatType, FlagsType, bool> Size_MatType_FlagsType_NzeroRows_t;
typedef perf::TestBaseWithParam<Size_MatType_FlagsType_NzeroRows_t> Size_MatType_FlagsType_NzeroRows;

PERF_TEST_P(Size_MatType_FlagsType_NzeroRows, dft, TEST_MATS_DFT)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    int flags = get<2>(GetParam());
    bool isNzeroRows = get<3>(GetParam());

    int nonzero_rows = 0;

    Mat src(sz, type);
    Mat dst(sz, type);

    declare.in(src, WARMUP_RNG).time(60);

    if (isNzeroRows)
        nonzero_rows = sz.height/2;

    TEST_CYCLE() dft(src, dst, flags, nonzero_rows);

    SANITY_CHECK(dst, 1e-5, ERROR_RELATIVE);
}

///////////////////////////////////////////////////////dct//////////////////////////////////////////////////////

CV_ENUM(DCT_FlagsType, 0, DCT_INVERSE , DCT_ROWS, DCT_INVERSE|DCT_ROWS)

typedef std::tr1::tuple<Size, MatType, DCT_FlagsType> Size_MatType_Flag_t;
typedef perf::TestBaseWithParam<Size_MatType_Flag_t> Size_MatType_Flag;

PERF_TEST_P(Size_MatType_Flag, dct, testing::Combine(
                                    testing::Values(cv::Size(320, 240),cv::Size(800, 600),
                                                    cv::Size(1024, 768), cv::Size(1280, 1024),
                                                    sz1080p, sz2K),
                                    testing::Values(CV_32FC1, CV_64FC1), DCT_FlagsType::all()))
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    int flags = get<2>(GetParam());

    Mat src(sz, type);
    Mat dst(sz, type);

    declare
        .in(src, WARMUP_RNG)
        .out(dst)
        .time(60);

    TEST_CYCLE() dct(src, dst, flags);

    SANITY_CHECK(dst, 1e-5, ERROR_RELATIVE);
}