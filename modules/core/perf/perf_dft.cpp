#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

///////////////////////////////////////////////////////dft//////////////////////////////////////////////////////////////

#define MAT_TYPES_DFT  CV_32FC1, CV_32FC2, CV_64FC1
#define MAT_SIZES_DFT  cv::Size(320, 480), cv::Size(800, 600), cv::Size(1280, 1024), sz1080p, sz2K
CV_ENUM(FlagsType, 0, DFT_INVERSE, DFT_SCALE, DFT_COMPLEX_OUTPUT, DFT_ROWS, DFT_INVERSE|DFT_COMPLEX_OUTPUT)
#define TEST_MATS_DFT  testing::Combine(testing::Values(MAT_SIZES_DFT), testing::Values(MAT_TYPES_DFT), FlagsType::all(), testing::Values(true, false))

typedef tuple<Size, MatType, FlagsType, bool> Size_MatType_FlagsType_NzeroRows_t;
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

typedef tuple<Size, MatType, DCT_FlagsType> Size_MatType_Flag_t;
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

///////////////////////////////////////////// 1D transforms /////////////////////////////////////////////

CV_ENUM(DFT1D_FlagsType, 0, DFT_INVERSE, DFT_SCALE, DFT_COMPLEX_OUTPUT)

typedef tuple<int, MatType, DFT1D_FlagsType, bool> Len_MatType_Flags_Col_t;
typedef perf::TestBaseWithParam<Len_MatType_Flags_Col_t> Len_MatType_Flags_Col;

// single row (1 x N) or single column (N x 1) vectors, N covers the 2^k, 5-smooth and odd plans
PERF_TEST_P(Len_MatType_Flags_Col, dft1d, testing::Combine(
                                    testing::Values(256, 1000, 1024, 4096, 4725, 65536),
                                    testing::Values(CV_32FC1, CV_32FC2, CV_64FC1, CV_64FC2),
                                    DFT1D_FlagsType::all(), testing::Bool()))
{
    int len = get<0>(GetParam());
    int type = get<1>(GetParam());
    int flags = get<2>(GetParam());
    bool column = get<3>(GetParam());

    Mat src(column ? len : 1, column ? 1 : len, type);
    Mat dst;

    declare.in(src, WARMUP_RNG).time(20);

    TEST_CYCLE() dft(src, dst, flags);

    SANITY_CHECK_NOTHING();
}

typedef tuple<int, MatType, DCT_FlagsType> Len_MatType_Flag_t;
typedef perf::TestBaseWithParam<Len_MatType_Flag_t> Len_MatType_Flag;

PERF_TEST_P(Len_MatType_Flag, dct1d, testing::Combine(
                                    testing::Values(256, 1000, 1024, 4096, 65536),
                                    testing::Values(CV_32FC1, CV_64FC1),
                                    testing::Values(0, DCT_INVERSE)))
{
    int len = get<0>(GetParam());
    int type = get<1>(GetParam());
    int flags = get<2>(GetParam());

    Mat src(1, len, type);
    Mat dst;

    declare.in(src, WARMUP_RNG).time(20);

    TEST_CYCLE() dct(src, dst, flags);

    SANITY_CHECK_NOTHING();
}

} // namespace
