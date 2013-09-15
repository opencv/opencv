#include "perf_precomp.hpp"
#include "opencv2/core/core_c.h"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

CV_ENUM(ROp, CV_REDUCE_SUM, CV_REDUCE_AVG, CV_REDUCE_MAX, CV_REDUCE_MIN)
typedef std::tr1::tuple<Size, MatType, ROp> Size_MatType_ROp_t;
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
