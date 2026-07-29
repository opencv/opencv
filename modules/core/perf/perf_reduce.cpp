// Copyright (C) 2026, Intel Corporation, all rights reserved.
#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

CV_ENUM(ROp, REDUCE_SUM, REDUCE_AVG, REDUCE_MAX, REDUCE_MIN, REDUCE_SUM2)
typedef tuple<Size, MatType, ROp> Size_MatType_ROp_t;
typedef perf::TestBaseWithParam<Size_MatType_ROp_t> Size_MatType_ROp;

// C1/C3/C4 over 8U/16U/16S/32F (widened from TYPICAL_MAT_TYPES to match IPP MAX/MIN coverage).
#define TYPICAL_MAT_TYPES_REDUCE CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, \
                                 CV_16SC1, CV_16SC3, CV_16SC4, CV_32FC1, CV_32FC3, CV_32FC4

PERF_TEST_P(Size_MatType_ROp, reduceR,
            testing::Combine(
                testing::Values(TYPICAL_MAT_SIZES),
                testing::Values(TYPICAL_MAT_TYPES_REDUCE),
                ROp::all()
                )
            )
{
    Size sz = get<0>(GetParam());
    int matType = get<1>(GetParam());
    int reduceOp = get<2>(GetParam());

    int sdepth = CV_MAT_DEPTH(matType);
    bool accumulate = (reduceOp == REDUCE_SUM || reduceOp == REDUCE_AVG || reduceOp == REDUCE_SUM2);

    // Accumulate ops (SUM/AVG/SUM2) over 16U/16S need a floating-point output;
    // that depth-changing coverage is exercised by the reduce_diff test below.
    // Here we keep the same-depth path: MAX/MIN for all types, and SUM/AVG for
    // 8U (->32S) and float types.
    if( accumulate && (sdepth == CV_16U || sdepth == CV_16S) )
        throw ::perf::TestBase::PerfSkipTestException();

    int ddepth = -1;
    if( accumulate && sdepth < CV_32S )
        ddepth = CV_32S;

    Mat src(sz, matType);
    Mat vec(1, sz.width, ddepth < 0 ? matType : ddepth);

    declare.in(src, WARMUP_RNG).out(vec);
    declare.time(100);

    int runs = 15;
    TEST_CYCLE_MULTIRUN(runs) reduce(src, vec, 0, reduceOp, ddepth);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Size_MatType_ROp, reduceC,
            testing::Combine(
                testing::Values(TYPICAL_MAT_SIZES),
                testing::Values(TYPICAL_MAT_TYPES_REDUCE),
                ROp::all()
                )
            )
{
    Size sz = get<0>(GetParam());
    int matType = get<1>(GetParam());
    int reduceOp = get<2>(GetParam());

    int sdepth = CV_MAT_DEPTH(matType);
    bool accumulate = (reduceOp == REDUCE_SUM || reduceOp == REDUCE_AVG || reduceOp == REDUCE_SUM2);

    // See reduceR: accumulate ops over 16U/16S are covered by reduce_diff.
    if( accumulate && (sdepth == CV_16U || sdepth == CV_16S) )
        throw ::perf::TestBase::PerfSkipTestException();

    int ddepth = -1;
    if( accumulate && sdepth < CV_32S )
        ddepth = CV_32S;

    Mat src(sz, matType);
    Mat vec(sz.height, 1, ddepth < 0 ? matType : ddepth);

    declare.in(src, WARMUP_RNG).out(vec);
    declare.time(100);

    TEST_CYCLE() reduce(src, vec, 1, reduceOp, ddepth);

    SANITY_CHECK_NOTHING();
}

CV_ENUM(ROpDiff, REDUCE_SUM, REDUCE_AVG)
typedef tuple<Size, tuple<MatType, MatDepth>, ROpDiff, int> Size_SrcType_DstDepth_ROp_Dim_t;
typedef perf::TestBaseWithParam<Size_SrcType_DstDepth_ROp_Dim_t> Size_SrcType_DstDepth_ROp_Dim;

PERF_TEST_P(Size_SrcType_DstDepth_ROp_Dim, reduce_diff,
            testing::Combine(
                testing::Values(sz1080p),
                testing::Values(
                    make_tuple((MatType)CV_8UC1,  (MatDepth)CV_32S), make_tuple((MatType)CV_8UC3,  (MatDepth)CV_32S), make_tuple((MatType)CV_8UC4,  (MatDepth)CV_32S),
                    make_tuple((MatType)CV_8UC1,  (MatDepth)CV_32F), make_tuple((MatType)CV_8UC3,  (MatDepth)CV_32F), make_tuple((MatType)CV_8UC4,  (MatDepth)CV_32F),
                    make_tuple((MatType)CV_8UC1,  (MatDepth)CV_64F), make_tuple((MatType)CV_8UC3,  (MatDepth)CV_64F), make_tuple((MatType)CV_8UC4,  (MatDepth)CV_64F),
                    make_tuple((MatType)CV_16UC1, (MatDepth)CV_32F), make_tuple((MatType)CV_16UC3, (MatDepth)CV_32F), make_tuple((MatType)CV_16UC4, (MatDepth)CV_32F),
                    make_tuple((MatType)CV_16UC1, (MatDepth)CV_64F), make_tuple((MatType)CV_16UC3, (MatDepth)CV_64F), make_tuple((MatType)CV_16UC4, (MatDepth)CV_64F),
                    make_tuple((MatType)CV_16SC1, (MatDepth)CV_32F), make_tuple((MatType)CV_16SC3, (MatDepth)CV_32F), make_tuple((MatType)CV_16SC4, (MatDepth)CV_32F),
                    make_tuple((MatType)CV_16SC1, (MatDepth)CV_64F), make_tuple((MatType)CV_16SC3, (MatDepth)CV_64F), make_tuple((MatType)CV_16SC4, (MatDepth)CV_64F),
                    make_tuple((MatType)CV_32FC1, (MatDepth)CV_32F), make_tuple((MatType)CV_32FC3, (MatDepth)CV_32F), make_tuple((MatType)CV_32FC4, (MatDepth)CV_32F),
                    make_tuple((MatType)CV_32FC1, (MatDepth)CV_64F), make_tuple((MatType)CV_32FC3, (MatDepth)CV_64F), make_tuple((MatType)CV_32FC4, (MatDepth)CV_64F),
                    make_tuple((MatType)CV_64FC1, (MatDepth)CV_64F), make_tuple((MatType)CV_64FC3, (MatDepth)CV_64F), make_tuple((MatType)CV_64FC4, (MatDepth)CV_64F)),
                ROpDiff::all(),
                testing::Values(0, 1)   // reduce dim: 0 = rows, 1 = columns
                )
            )
{
    Size sz = get<0>(GetParam());
    int srcType = get<0>(get<1>(GetParam()));
    int dstDepth = get<1>(get<1>(GetParam()));
    int reduceOp = get<2>(GetParam());
    int dim = get<3>(GetParam());

    int cn = CV_MAT_CN(srcType);
    Size dstSize = (dim == 0) ? Size(sz.width, 1) : Size(1, sz.height);

    Mat src(sz, srcType);
    Mat vec(dstSize, CV_MAKETYPE(dstDepth, cn));

    declare.in(src, WARMUP_RNG).out(vec);
    declare.time(100);

    TEST_CYCLE_MULTIRUN(15) reduce(src, vec, dim, reduceOp, dstDepth);

    SANITY_CHECK_NOTHING();
}

typedef tuple<Size, MatType, int> Size_MatType_RMode_t;
typedef perf::TestBaseWithParam<Size_MatType_RMode_t> Size_MatType_RMode;

PERF_TEST_P(Size_MatType_RMode, DISABLED_reduceArgMinMax, testing::Combine(
        testing::Values(TYPICAL_MAT_SIZES),
        testing::Values(CV_8U, CV_32F),
        testing::Values(0, 1)
)
)
{
    Size srcSize = get<0>(GetParam());
    int matType = get<1>(GetParam());
    int axis = get<2>(GetParam());

    Mat src(srcSize, matType);

    std::vector<int> dstSize(src.dims);
    std::copy(src.size.p, src.size.p + src.dims, dstSize.begin());
    dstSize[axis] = 1;

    Mat dst(dstSize, CV_32S, 0.);

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE() cv::reduceArgMin(src, dst, axis, true);

    SANITY_CHECK_NOTHING();
}

} // namespace
