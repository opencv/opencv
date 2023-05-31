// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test {

typedef Size_MatType Accumulate;

#define MAT_TYPES_ACCUMLATE CV_8UC1, CV_16UC1, CV_32FC1
#define MAT_TYPES_ACCUMLATE_C MAT_TYPES_ACCUMLATE, CV_8UC3, CV_16UC3, CV_32FC3
#define MAT_TYPES_ACCUMLATE_D MAT_TYPES_ACCUMLATE, CV_64FC1
#define MAT_TYPES_ACCUMLATE_D_C MAT_TYPES_ACCUMLATE_C, CV_64FC1, CV_64FC1

#define PERF_ACCUMULATE_INIT(_FLTC)                    \
    const Size srcSize = get<0>(GetParam());           \
    const int srcType = get<1>(GetParam());            \
    const int dstType = _FLTC(CV_MAT_CN(srcType));     \
    Mat src1(srcSize, srcType), dst(srcSize, dstType); \
    declare.in(src1, dst, WARMUP_RNG).out(dst);

#define PERF_ACCUMULATE_MASK_INIT(_FLTC) \
    PERF_ACCUMULATE_INIT(_FLTC)          \
    Mat mask(srcSize, CV_8UC1);          \
    declare.in(mask, WARMUP_RNG);

#define PERF_TEST_P_ACCUMULATE(_NAME, _TYPES, _INIT, _FUN)           \
    PERF_TEST_P(Accumulate, _NAME,                                   \
        testing::Combine(                                            \
            testing::Values(sz1080p, sz720p, szVGA, szQVGA, szODD),  \
            testing::Values(_TYPES)                                  \
        )                                                            \
    )                                                                \
    {                                                                \
        _INIT                                                        \
        TEST_CYCLE() _FUN;                                           \
        SANITY_CHECK_NOTHING();                                      \
    }

/////////////////////////////////// Accumulate ///////////////////////////////////

PERF_TEST_P_ACCUMULATE(Accumulate, MAT_TYPES_ACCUMLATE,
        PERF_ACCUMULATE_INIT(CV_32FC), accumulate(src1, dst))

PERF_TEST_P_ACCUMULATE(AccumulateMask, MAT_TYPES_ACCUMLATE_C,
    PERF_ACCUMULATE_MASK_INIT(CV_32FC), accumulate(src1, dst, mask))

PERF_TEST_P_ACCUMULATE(AccumulateDouble, MAT_TYPES_ACCUMLATE_D,
    PERF_ACCUMULATE_INIT(CV_64FC), accumulate(src1, dst))

PERF_TEST_P_ACCUMULATE(AccumulateDoubleMask, MAT_TYPES_ACCUMLATE_D_C,
    PERF_ACCUMULATE_MASK_INIT(CV_64FC), accumulate(src1, dst, mask))

///////////////////////////// AccumulateSquare ///////////////////////////////////

PERF_TEST_P_ACCUMULATE(Square, MAT_TYPES_ACCUMLATE,
    PERF_ACCUMULATE_INIT(CV_32FC), accumulateSquare(src1, dst))

PERF_TEST_P_ACCUMULATE(SquareMask, MAT_TYPES_ACCUMLATE_C,
    PERF_ACCUMULATE_MASK_INIT(CV_32FC), accumulateSquare(src1, dst, mask))

PERF_TEST_P_ACCUMULATE(SquareDouble, MAT_TYPES_ACCUMLATE_D,
    PERF_ACCUMULATE_INIT(CV_64FC), accumulateSquare(src1, dst))

PERF_TEST_P_ACCUMULATE(SquareDoubleMask, MAT_TYPES_ACCUMLATE_D_C,
    PERF_ACCUMULATE_MASK_INIT(CV_64FC), accumulateSquare(src1, dst, mask))

///////////////////////////// AccumulateProduct ///////////////////////////////////

#define PERF_ACCUMULATE_INIT_2(_FLTC) \
    PERF_ACCUMULATE_INIT(_FLTC)       \
    Mat src2(srcSize, srcType);       \
    declare.in(src2);

#define PERF_ACCUMULATE_MASK_INIT_2(_FLTC) \
    PERF_ACCUMULATE_MASK_INIT(_FLTC)       \
    Mat src2(srcSize, srcType);            \
    declare.in(src2);

PERF_TEST_P_ACCUMULATE(Product, MAT_TYPES_ACCUMLATE,
    PERF_ACCUMULATE_INIT_2(CV_32FC), accumulateProduct(src1, src2, dst))

PERF_TEST_P_ACCUMULATE(ProductMask, MAT_TYPES_ACCUMLATE_C,
    PERF_ACCUMULATE_MASK_INIT_2(CV_32FC), accumulateProduct(src1, src2, dst, mask))

PERF_TEST_P_ACCUMULATE(ProductDouble, MAT_TYPES_ACCUMLATE_D,
    PERF_ACCUMULATE_INIT_2(CV_64FC), accumulateProduct(src1, src2, dst))

PERF_TEST_P_ACCUMULATE(ProductDoubleMask, MAT_TYPES_ACCUMLATE_D_C,
    PERF_ACCUMULATE_MASK_INIT_2(CV_64FC), accumulateProduct(src1, src2, dst, mask))

///////////////////////////// AccumulateWeighted ///////////////////////////////////

PERF_TEST_P_ACCUMULATE(Weighted, MAT_TYPES_ACCUMLATE,
    PERF_ACCUMULATE_INIT(CV_32FC), accumulateWeighted(src1, dst, 0.123))

PERF_TEST_P_ACCUMULATE(WeightedMask, MAT_TYPES_ACCUMLATE_C,
    PERF_ACCUMULATE_MASK_INIT(CV_32FC), accumulateWeighted(src1, dst, 0.123, mask))

PERF_TEST_P_ACCUMULATE(WeightedDouble, MAT_TYPES_ACCUMLATE_D,
    PERF_ACCUMULATE_INIT(CV_64FC), accumulateWeighted(src1, dst, 0.123456))

PERF_TEST_P_ACCUMULATE(WeightedDoubleMask, MAT_TYPES_ACCUMLATE_D_C,
    PERF_ACCUMULATE_MASK_INIT(CV_64FC), accumulateWeighted(src1, dst, 0.123456, mask))

} // namespace
