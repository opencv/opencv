#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

#define TYPICAL_MAT_SIZES_MAGNITUDE  TYPICAL_MAT_SIZES
#define TYPICAL_MAT_TYPES_MAGNITUDE  CV_32FC1, CV_64FC1
#define TYPICAL_MATS_MAGNITUDE       testing::Combine(testing::Values(TYPICAL_MAT_SIZES_MAGNITUDE), testing::Values(TYPICAL_MAT_TYPES_MAGNITUDE))

PERF_TEST_P(Size_MatType, magnitude, TYPICAL_MATS_MAGNITUDE)
{
    Size size = get<0>(GetParam());
    int type = get<1>(GetParam());
    Mat src1(size, type);
    Mat src2(size, type);
    Mat dst(size, type);

    declare.in(src1, src2, WARMUP_RNG).out(dst);

    TEST_CYCLE() magnitude(src1, src2, dst);

    SANITY_CHECK_NOTHING();
}

#define TYPICAL_MAT_SIZES_MAGNITUDECOMPLEX  TYPICAL_MAT_SIZES
#define TYPICAL_MAT_TYPES_MAGNITUDECOMPLEX  CV_32FC2, CV_64FC2
#define TYPICAL_MATS_MAGNITUDECOMPLEX       testing::Combine(testing::Values(TYPICAL_MAT_SIZES_MAGNITUDECOMPLEX), testing::Values(TYPICAL_MAT_TYPES_MAGNITUDECOMPLEX))

PERF_TEST_P(Size_MatType, magnitudeComplex, TYPICAL_MATS_MAGNITUDECOMPLEX)
{
    Size size = get<0>(GetParam());
    int type = get<1>(GetParam());
    Mat src1(size, type);
    Mat dst(size, CV_MAKETYPE(CV_MAT_DEPTH(type), 1));

    declare.in(src1, WARMUP_RNG).out(dst);

    TEST_CYCLE() magnitudeComplex(src1, dst);

    SANITY_CHECK_NOTHING();
}

} // namespace
