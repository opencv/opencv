// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2021 Intel Corporation


#ifndef OPENCV_GAPI_CORE_TESTS_HPP
#define OPENCV_GAPI_CORE_TESTS_HPP

#include <iostream>

#include "gapi_tests_common.hpp"
#include "gapi_parsers_tests_common.hpp"

namespace opencv_test
{
enum mathOp
{
    ADD = 0,
    SUB = 1,
    MUL = 2,
    DIV = 3
};

enum bitwiseOp
{
    AND = 0,
    OR = 1,
    XOR = 2,
    NOT = 3
};

// Note: namespace must match the namespace of the type of the printed object
inline std::ostream& operator<<(std::ostream& os, mathOp op)
{
#define CASE(v) case mathOp::v: os << #v; break
    switch (op)
    {
        CASE(ADD);
        CASE(SUB);
        CASE(MUL);
        CASE(DIV);
        default: GAPI_Error("unknown mathOp value");
    }
#undef CASE
    return os;
}

// Note: namespace must match the namespace of the type of the printed object
inline std::ostream& operator<<(std::ostream& os, bitwiseOp op)
{
#define CASE(v) case bitwiseOp::v: os << #v; break
    switch (op)
    {
        CASE(AND);
        CASE(OR);
        CASE(XOR);
        CASE(NOT);
        default: GAPI_Error("unknown bitwiseOp value");
    }
#undef CASE
    return os;
}

// Create new value-parameterized test fixture:
// MathOpTest - fixture name
// initMatsRandU - function that is used to initialize input/output data
// FIXTURE_API(mathOp,bool,double,bool) - test-specific parameters (types)
// 4 - number of test-specific parameters
// opType, testWithScalar, scale, doReverseOp - test-specific parameters (names)
//
// We get:
// 1. Default parameters: int type, cv::Size sz, int dtype, getCompileArgs() function
//      - available in test body
// 2. Input/output matrices will be initialized by initMatsRandU (in this fixture)
// 3. Specific parameters: opType, testWithScalar, scale, doReverseOp of corresponding types
//      - created (and initialized) automatically
//      - available in test body
// Note: all parameter _values_ (e.g. type CV_8UC3) are set via INSTANTIATE_TEST_CASE_P macro
GAPI_TEST_FIXTURE(MathOpTest, initMatsRandU, FIXTURE_API(mathOp,bool,double,bool), 4,
    opType, testWithScalar, scale, doReverseOp)
// No specific parameters for MulDoubleTest, so "fixture API" is empty - <>
GAPI_TEST_FIXTURE(MulDoubleTest, initMatrixRandU, <>, 0)
GAPI_TEST_FIXTURE(DivTest, initMatrixRandU, <>, 0)
GAPI_TEST_FIXTURE(DivCTest, initMatrixRandU, <>, 0)
GAPI_TEST_FIXTURE(MeanTest, initMatrixRandU, <>, 0)
GAPI_TEST_FIXTURE(MaskTest, initMatrixRandU, <>, 0)
GAPI_TEST_FIXTURE(Polar2CartTest, initMatsRandU, <>, 0)
GAPI_TEST_FIXTURE(Cart2PolarTest, initMatsRandU, <>, 0)
GAPI_TEST_FIXTURE(CmpTest, initMatsRandU, FIXTURE_API(CmpTypes,bool,CompareMats), 3, opType, testWithScalar, cmpF)
GAPI_TEST_FIXTURE(BitwiseTest, initMatsRandU, FIXTURE_API(bitwiseOp,bool), 2, opType, testWithScalar)
GAPI_TEST_FIXTURE(NotTest, initMatrixRandU, <>, 0)
GAPI_TEST_FIXTURE(SelectTest, initMatsRandU, <>, 0)
GAPI_TEST_FIXTURE(MinTest, initMatsRandU, <>, 0)
GAPI_TEST_FIXTURE(MaxTest, initMatsRandU, <>, 0)
GAPI_TEST_FIXTURE(AbsDiffTest, initMatsRandU, <>, 0)
GAPI_TEST_FIXTURE(AbsDiffCTest, initMatsRandU, <>, 0)
GAPI_TEST_FIXTURE(SumTest, initMatrixRandU, FIXTURE_API(CompareScalars), 1, cmpF)
GAPI_TEST_FIXTURE(CountNonZeroTest, initMatrixRandU, FIXTURE_API(CompareScalars), 1, cmpF)
GAPI_TEST_FIXTURE(AddWeightedTest, initMatsRandU, FIXTURE_API(CompareMats), 1, cmpF)
GAPI_TEST_FIXTURE(NormTest, initMatrixRandU, FIXTURE_API(CompareScalars,NormTypes), 2,
    cmpF, opType)
GAPI_TEST_FIXTURE(IntegralTest, initNothing, <>, 0)
GAPI_TEST_FIXTURE(ThresholdTest, initMatrixRandU, FIXTURE_API(int, cv::Scalar), 2, tt, maxval)
GAPI_TEST_FIXTURE(ThresholdOTTest, initMatrixRandU, FIXTURE_API(int), 1, tt)
GAPI_TEST_FIXTURE(InRangeTest, initMatrixRandU, <>, 0)
GAPI_TEST_FIXTURE(Split3Test, initMatrixRandU, <>, 0)
GAPI_TEST_FIXTURE(Split4Test, initMatrixRandU, <>, 0)
GAPI_TEST_FIXTURE(Merge3Test, initMatsRandU, <>, 0)
GAPI_TEST_FIXTURE(Merge4Test, initMatsRandU, <>, 0)
GAPI_TEST_FIXTURE(RemapTest, initMatrixRandU, <>, 0)
GAPI_TEST_FIXTURE(FlipTest, initMatrixRandU, FIXTURE_API(int), 1, flipCode)
GAPI_TEST_FIXTURE(CropTest, initMatrixRandU, FIXTURE_API(cv::Rect), 1, rect_to)
GAPI_TEST_FIXTURE(CopyTest, initMatrixRandU, <>, 0)
GAPI_TEST_FIXTURE(ConcatHorTest, initNothing, <>, 0)
GAPI_TEST_FIXTURE(ConcatVertTest, initNothing, <>, 0)
GAPI_TEST_FIXTURE(ConcatVertVecTest, initNothing, <>, 0)
GAPI_TEST_FIXTURE(ConcatHorVecTest, initNothing, <>, 0)
GAPI_TEST_FIXTURE(LUTTest, initNothing, <>, 0)
GAPI_TEST_FIXTURE(ConvertToTest, initNothing, FIXTURE_API(CompareMats, double, double), 3,
    cmpF, alpha, beta)
GAPI_TEST_FIXTURE(PhaseTest, initMatsRandU, FIXTURE_API(bool), 1, angle_in_degrees)
GAPI_TEST_FIXTURE(SqrtTest, initMatrixRandU, <>, 0)
GAPI_TEST_FIXTURE(NormalizeTest, initNothing, FIXTURE_API(CompareMats,double,double,int,MatType2), 5,
    cmpF, a, b, norm_type, ddepth)
struct BackendOutputAllocationTest : TestWithParams<>
{
    BackendOutputAllocationTest()
    {
        in_mat1 = cv::Mat(sz, type);
        in_mat2 = cv::Mat(sz, type);
        cv::randu(in_mat1, cv::Scalar::all(1), cv::Scalar::all(15));
        cv::randu(in_mat2, cv::Scalar::all(1), cv::Scalar::all(15));
    }
};
// FIXME: move all tests from this fixture to the base class once all issues are resolved
struct BackendOutputAllocationLargeSizeWithCorrectSubmatrixTest : BackendOutputAllocationTest {};
GAPI_TEST_FIXTURE(ReInitOutTest, initNothing, <cv::Size>, 1, out_sz)

GAPI_TEST_FIXTURE(WarpPerspectiveTest, initMatrixRandU,
        FIXTURE_API(CompareMats, double , double, int, int, cv::Scalar),
        6, cmpF, angle, scale, flags, border_mode, border_value)

GAPI_TEST_FIXTURE(WarpAffineTest, initMatrixRandU,
        FIXTURE_API(CompareMats, double , double, int, int, cv::Scalar),
        6, cmpF, angle, scale, flags, border_mode, border_value)
GAPI_TEST_FIXTURE(KMeansNDTest, initMatrixRandU, FIXTURE_API(CompareMats, int, cv::KmeansFlags), 3, cmpF, K, flags)
GAPI_TEST_FIXTURE(KMeans2DTest, initNothing,     FIXTURE_API(int, cv::KmeansFlags), 2, K, flags)
GAPI_TEST_FIXTURE(KMeans3DTest, initNothing,     FIXTURE_API(int, cv::KmeansFlags), 2, K, flags)
GAPI_TEST_FIXTURE(TransposeTest, initMatrixRandU, FIXTURE_API(CompareMats), 1, cmpF)


GAPI_TEST_EXT_BASE_FIXTURE(ParseSSDBLTest, ParserSSDTest, initNothing,
    FIXTURE_API(float, int), 2, confidence_threshold, filter_label)
GAPI_TEST_EXT_BASE_FIXTURE(ParseSSDTest, ParserSSDTest, initNothing,
    FIXTURE_API(float, bool, bool), 3, confidence_threshold, alignment_to_square, filter_out_of_bounds)
GAPI_TEST_EXT_BASE_FIXTURE(ParseYoloTest, ParserYoloTest, initNothing,
    FIXTURE_API(float, float, int, std::pair<bool,int>), 4, confidence_threshold, nms_threshold, num_classes, dims_config)
GAPI_TEST_FIXTURE(SizeTest, initMatrixRandU, <>, 0)
GAPI_TEST_FIXTURE(SizeRTest, initNothing, <>, 0)
GAPI_TEST_FIXTURE(SizeMFTest, initNothing, <>, 0)
} // opencv_test

#endif //OPENCV_GAPI_CORE_TESTS_HPP
