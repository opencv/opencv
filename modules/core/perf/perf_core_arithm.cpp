#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;

#define TYPICAL_MAT_SIZES_CORE_ARITHM   TYPICAL_MAT_SIZES 
#define TYPICAL_MAT_TYPES_CORE_ARITHM   CV_8UC1, CV_8SC1, CV_8UC4, CV_32SC1, CV_32FC1
#define TYPICAL_MATS_CORE_ARITHM        testing::Combine( testing::Values( TYPICAL_MAT_SIZES_CORE_ARITHM ), testing::Values( TYPICAL_MAT_TYPES_CORE_ARITHM ) )

#define TYPICAL_MAT_TYPES_BITW_ARITHM   CV_8UC1, CV_8SC1, CV_8UC4, CV_32SC1, CV_32SC4
#define TYPICAL_MATS_BITW_ARITHM        testing::Combine( testing::Values( TYPICAL_MAT_SIZES_CORE_ARITHM ), testing::Values( TYPICAL_MAT_TYPES_BITW_ARITHM ) )

#define PERF_TEST_P__CORE_ARITHM(__f, __testset) \
PERF_TEST_P(Size_MatType, core_arithm__ ## __f, __testset)     \
{                                                              \
    Size sz = std::tr1::get<0>(GetParam());                    \
    int type = std::tr1::get<1>(GetParam());                   \
    cv::Mat a = Mat(sz, type);                                 \
    cv::Mat b = Mat(sz, type);                                 \
    cv::Mat c = Mat(sz, type);                                 \
                                                               \
    declare.in(a, b, WARMUP_RNG)                               \
        .out(c);                                               \
                                                               \
    TEST_CYCLE(100) __f(a,b, c);                               \
                                                               \
    SANITY_CHECK(c);                                           \
}

#define PERF_TEST_P__CORE_ARITHM_SCALAR(__f, __testset) \
PERF_TEST_P(Size_MatType, core_arithm__ ## __f ##__Scalar, __testset)     \
{                                                                         \
    Size sz = std::tr1::get<0>(GetParam());                               \
    int type = std::tr1::get<1>(GetParam());                              \
    cv::Mat a = Mat(sz, type);                                            \
    cv::Scalar b;                                                         \
    cv::Mat c = Mat(sz, type);                                            \
                                                                          \
    declare.in(a, b, WARMUP_RNG)                                          \
        .out(c);                                                          \
                                                                          \
    TEST_CYCLE(100) __f(a,b, c);                                          \
                                                                          \
    SANITY_CHECK(c);                                                      \
}

PERF_TEST_P__CORE_ARITHM(bitwise_and, TYPICAL_MATS_BITW_ARITHM)
PERF_TEST_P__CORE_ARITHM(bitwise_or, TYPICAL_MATS_BITW_ARITHM)
PERF_TEST_P__CORE_ARITHM(bitwise_xor, TYPICAL_MATS_BITW_ARITHM)
PERF_TEST_P__CORE_ARITHM(add, TYPICAL_MATS_CORE_ARITHM)
PERF_TEST_P__CORE_ARITHM(subtract, TYPICAL_MATS_CORE_ARITHM)
PERF_TEST_P__CORE_ARITHM(min, TYPICAL_MATS_CORE_ARITHM)
PERF_TEST_P__CORE_ARITHM(max, TYPICAL_MATS_CORE_ARITHM)
PERF_TEST_P__CORE_ARITHM(absdiff, TYPICAL_MATS_CORE_ARITHM)

PERF_TEST_P__CORE_ARITHM_SCALAR(bitwise_and, TYPICAL_MATS_BITW_ARITHM)
PERF_TEST_P__CORE_ARITHM_SCALAR(bitwise_or, TYPICAL_MATS_BITW_ARITHM)
PERF_TEST_P__CORE_ARITHM_SCALAR(bitwise_xor, TYPICAL_MATS_BITW_ARITHM)
PERF_TEST_P__CORE_ARITHM_SCALAR(add, TYPICAL_MATS_CORE_ARITHM)
PERF_TEST_P__CORE_ARITHM_SCALAR(subtract, TYPICAL_MATS_CORE_ARITHM)
PERF_TEST_P__CORE_ARITHM_SCALAR(absdiff, TYPICAL_MATS_CORE_ARITHM)


