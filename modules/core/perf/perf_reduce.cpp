#include "perf_precomp.hpp"
#include "opencv2/core/core_c.h"

namespace opencv_test
{
using namespace perf;

CV_ENUM(ROp, CV_REDUCE_SUM, CV_REDUCE_AVG, CV_REDUCE_MAX, CV_REDUCE_MIN)
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
    if( CV_MAT_DEPTH(matType) < CV_32S && (reduceOp == CV_REDUCE_SUM || reduceOp == CV_REDUCE_AVG) )
        ddepth = CV_32S;

    Mat src(sz, matType);
    Mat vec(1, sz.width, ddepth < 0 ? matType : ddepth);

    declare.in(src, WARMUP_RNG).out(vec);
    declare.time(100);

    int runs = 15;
    TEST_CYCLE_MULTIRUN(runs) reduce(src, vec, 0, reduceOp, ddepth);

    SANITY_CHECK(vec, 1);
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
    if( CV_MAT_DEPTH(matType)< CV_32S && (reduceOp == CV_REDUCE_SUM || reduceOp == CV_REDUCE_AVG) )
        ddepth = CV_32S;

    Mat src(sz, matType);
    Mat vec(sz.height, 1, ddepth < 0 ? matType : ddepth);

    declare.in(src, WARMUP_RNG).out(vec);
    declare.time(100);

    TEST_CYCLE() reduce(src, vec, 1, reduceOp, ddepth);

    SANITY_CHECK(vec, 1);
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
