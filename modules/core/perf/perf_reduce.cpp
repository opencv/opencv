#include "perf_precomp.hpp"

namespace opencv_test {
using namespace perf;

///////////// reduce() performance tests /////////////

typedef std::tuple<cv::Size, MatType, MatDepth> Size_MatType_OutDepth_t;
typedef TestBaseWithParam<Size_MatType_OutDepth_t> Size_MatType_OutDepth;

// Test: reduce SUM with float input -> float output (most common case)
PERF_TEST_P(Size_MatType_OutDepth, reduceSum,
    testing::Combine(
        testing::Values(TYPICAL_MAT_SIZES),
        testing::Values(CV_32FC1),
        testing::Values(CV_32F)
    )
)
{
    Size sz = std::get<0>(GetParam());
    int srcType = std::get<1>(GetParam());
    int dstDepth = std::get<2>(GetParam());

    Mat src(sz, srcType);
    Mat dst(1, sz.width, CV_MAKETYPE(dstDepth, 1));

    declare.in(src, WARMUP_RNG).out(dst).iterations(100);

    TEST_CYCLE()
    {
        cv::reduce(src, dst, 0, REDUCE_SUM, dstDepth);
    }

    SANITY_CHECK_NOTHING();
}

// Test: reduce SUM with uchar input -> int output (tests SIMD widening path)
PERF_TEST_P(Size_MatType_OutDepth, reduceSum8u32s,
    testing::Combine(
        testing::Values(TYPICAL_MAT_SIZES),
        testing::Values(CV_8UC1, CV_8UC3),
        testing::Values(CV_32S)
    )
)
{
    Size sz = std::get<0>(GetParam());
    int srcType = std::get<1>(GetParam());
    int dstDepth = std::get<2>(GetParam());

    Mat src(sz, srcType);
    int cn = CV_MAT_CN(srcType);
    Mat dst(1, sz.width * cn, CV_MAKETYPE(dstDepth, 1));

    declare.in(src, WARMUP_RNG).out(dst).iterations(100);

    TEST_CYCLE()
    {
        cv::reduce(src, dst, 0, REDUCE_SUM, dstDepth);
    }

    SANITY_CHECK_NOTHING();
}

// Test: reduce MAX with float input
PERF_TEST_P(Size_MatType_OutDepth, reduceMax,
    testing::Combine(
        testing::Values(TYPICAL_MAT_SIZES),
        testing::Values(CV_32FC1),
        testing::Values(CV_32F)
    )
)
{
    Size sz = std::get<0>(GetParam());
    int srcType = std::get<1>(GetParam());
    int dstDepth = std::get<2>(GetParam());

    Mat src(sz, srcType);
    Mat dst(1, sz.width, CV_MAKETYPE(dstDepth, 1));

    declare.in(src, WARMUP_RNG).out(dst).iterations(100);

    TEST_CYCLE()
    {
        cv::reduce(src, dst, 0, REDUCE_MAX, dstDepth);
    }

    SANITY_CHECK_NOTHING();
}

// Test: reduce MIN with float input
PERF_TEST_P(Size_MatType_OutDepth, reduceMin,
    testing::Combine(
        testing::Values(TYPICAL_MAT_SIZES),
        testing::Values(CV_32FC1),
        testing::Values(CV_32F)
    )
)
{
    Size sz = std::get<0>(GetParam());
    int srcType = std::get<1>(GetParam());
    int dstDepth = std::get<2>(GetParam());

    Mat src(sz, srcType);
    Mat dst(1, sz.width, CV_MAKETYPE(dstDepth, 1));

    declare.in(src, WARMUP_RNG).out(dst).iterations(100);

    TEST_CYCLE()
    {
        cv::reduce(src, dst, 0, REDUCE_MIN, dstDepth);
    }

    SANITY_CHECK_NOTHING();
}

// Test: column-wise reduce SUM (reduce to single column)
PERF_TEST_P(Size_MatType_OutDepth, reduceSumCol,
    testing::Combine(
        testing::Values(TYPICAL_MAT_SIZES),
        testing::Values(CV_32FC1),
        testing::Values(CV_32F)
    )
)
{
    Size sz = std::get<0>(GetParam());
    int srcType = std::get<1>(GetParam());
    int dstDepth = std::get<2>(GetParam());

    Mat src(sz, srcType);
    Mat dst(sz.height, 1, CV_MAKETYPE(dstDepth, 1));

    declare.in(src, WARMUP_RNG).out(dst).iterations(100);

    TEST_CYCLE()
    {
        cv::reduce(src, dst, 1, REDUCE_SUM, dstDepth);
    }

    SANITY_CHECK_NOTHING();
}

} // namespace opencv_test

namespace opencv_test
{
using namespace perf;

CV_ENUM(ROp, REDUCE_SUM, REDUCE_AVG, REDUCE_MAX, REDUCE_MIN, REDUCE_SUM2)
typedef tuple<Size, MatType, ROp> Size_MatType_ROp_t;
typedef perf::TestBaseWithParam<Size_MatType_ROp_t> Size_MatType_ROp;


PERF_TEST_P(Size_MatType_ROp, reduceR,
            testing::Combine(
                testing::Values(TYPICAL_MAT_SIZES),
                testing::Values(TYPICAL_MAT_TYPES),
                ROp::all()
                )
            )
{
    Size sz = get<0>(GetParam());
    int matType = get<1>(GetParam());
    int reduceOp = get<2>(GetParam());

    int ddepth = -1;
    if( CV_MAT_DEPTH(matType) < CV_32S && (reduceOp == REDUCE_SUM || reduceOp == REDUCE_AVG || reduceOp == REDUCE_SUM2) )
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
                testing::Values(TYPICAL_MAT_TYPES),
                ROp::all()
                )
            )
{
    Size sz = get<0>(GetParam());
    int matType = get<1>(GetParam());
    int reduceOp = get<2>(GetParam());

    int ddepth = -1;
    if( CV_MAT_DEPTH(matType)< CV_32S && (reduceOp == REDUCE_SUM || reduceOp == REDUCE_AVG || reduceOp == REDUCE_SUM2) )
        ddepth = CV_32S;

    Mat src(sz, matType);
    Mat vec(sz.height, 1, ddepth < 0 ? matType : ddepth);

    declare.in(src, WARMUP_RNG).out(vec);
    declare.time(100);

    TEST_CYCLE() reduce(src, vec, 1, reduceOp, ddepth);

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
