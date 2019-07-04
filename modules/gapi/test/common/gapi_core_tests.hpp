// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2019 Intel Corporation


#ifndef OPENCV_GAPI_CORE_TESTS_HPP
#define OPENCV_GAPI_CORE_TESTS_HPP

#include <iostream>

#include "gapi_tests_common.hpp"

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

namespace
{
const char *MathOperations[] = {"ADD", "SUB", "MUL", "DIV"};
const char *BitwiseOperations[] = {"And", "Or", "Xor"};
const char *CompareOperations[] = {"CMP_EQ", "CMP_GT", "CMP_GE", "CMP_LT", "CMP_LE", "CMP_NE"};
//corresponds to OpenCV
const char *NormOperations[] = {"", "NORM_INF", "NORM_L1", "","NORM_L2"};
}


struct PrintMathOpCoreParams
{
    template <class TestParams>
    std::string operator()(const ::testing::TestParamInfo<TestParams>& info) const
    {
        std::stringstream ss;
        using AllParams = Params<mathOp,bool,double,bool>;
        const AllParams::params_t& params = info.param;
        cv::Size sz = AllParams::getCommon<1>(params);  // size
        ss<<MathOperations[AllParams::getSpecific<0>(params)]  // mathOp
                    <<"_"<<AllParams::getSpecific<1>(params)  // testWithScalar
                    <<"_"<<AllParams::getCommon<0>(params)  // type
                    <<"_"<<(int)AllParams::getSpecific<2>(params)  // scale
                    <<"_"<<sz.width
                    <<"x"<<sz.height
                    <<"_"<<(AllParams::getCommon<2>(params)+1)  // dtype
                    <<"_"<<AllParams::getCommon<3>(params)  // createOutputMatrices
                    <<"_"<<AllParams::getSpecific<3>(params);  // doReverseOp
        return ss.str();
   }
};

struct PrintCmpCoreParams
{
    template <class TestParams>
    std::string operator()(const ::testing::TestParamInfo<TestParams>& info) const
    {
        std::stringstream ss;
        using AllParams = Params<CmpTypes,bool>;
        const AllParams::params_t& params = info.param;
        cv::Size sz = AllParams::getCommon<1>(params);  // size
        ss<<CompareOperations[AllParams::getSpecific<0>(params)]  // CmpType
                    <<"_"<<AllParams::getSpecific<1>(params)  // testWithScalar
                    <<"_"<<AllParams::getCommon<0>(params)  // type
                    <<"_"<<sz.width
                    <<"x"<<sz.height
                    <<"_"<<AllParams::getCommon<3>(params);  // createOutputMatrices
        return ss.str();
   }
};

struct PrintBWCoreParams
{
    template <class TestParams>
    std::string operator()(const ::testing::TestParamInfo<TestParams>& info) const
    {
        std::stringstream ss;
        using AllParams = Params<bitwiseOp>;
        const AllParams::params_t& params = info.param;
        cv::Size sz = AllParams::getCommon<1>(params);  // size
        ss<<BitwiseOperations[AllParams::getSpecific<0>(params)]  // bitwiseOp
                    <<"_"<<AllParams::getCommon<0>(params)  // type
                    <<"_"<<sz.width
                    <<"x"<<sz.height
                    <<"_"<<AllParams::getCommon<3>(params);  // createOutputMatrices
        return ss.str();
   }
};

struct PrintNormCoreParams
{
    template <class TestParams>
    std::string operator()(const ::testing::TestParamInfo<TestParams>& info) const
    {
        std::stringstream ss;
        using AllParams = Params<compare_scalar_f,NormTypes>;
        const AllParams::params_t& params = info.param;
        cv::Size sz = AllParams::getCommon<1>(params);  // size
        ss<<NormOperations[AllParams::getSpecific<1>(params)]  // NormTypes
                    <<"_"<<AllParams::getCommon<0>(params)  // type
                    <<"_"<<sz.width
                    <<"x"<<sz.height;
        return ss.str();
   }
};

GAPI_TEST_FIXTURE(MathOpTest, initMatsRandU, FIXTURE_API(mathOp,bool,double,bool), 4,
    opType, testWithScalar, scale, doReverseOp)
GAPI_TEST_FIXTURE(MulDoubleTest, initMatrixRandU, <>, 0)
GAPI_TEST_FIXTURE(DivTest, initMatrixRandU, <>, 0)
GAPI_TEST_FIXTURE(DivCTest, initMatrixRandU, <>, 0)
GAPI_TEST_FIXTURE(MeanTest, initMatrixRandU, <>, 0)
GAPI_TEST_FIXTURE(MaskTest, initMatrixRandU, <>, 0)
GAPI_TEST_FIXTURE(Polar2CartTest, initMatsRandU, <>, 0)
GAPI_TEST_FIXTURE(Cart2PolarTest, initMatsRandU, <>, 0)
GAPI_TEST_FIXTURE(CmpTest, initMatsRandU, FIXTURE_API(CmpTypes,bool), 2, opType, testWithScalar)
GAPI_TEST_FIXTURE(BitwiseTest, initMatsRandU, FIXTURE_API(bitwiseOp), 1, opType)
GAPI_TEST_FIXTURE(NotTest, initMatrixRandU, <>, 0)
GAPI_TEST_FIXTURE(SelectTest, initMatsRandU, <>, 0)
GAPI_TEST_FIXTURE(MinTest, initMatsRandU, <>, 0)
GAPI_TEST_FIXTURE(MaxTest, initMatsRandU, <>, 0)
GAPI_TEST_FIXTURE(AbsDiffTest, initMatsRandU, <>, 0)
GAPI_TEST_FIXTURE(AbsDiffCTest, initMatsRandU, <>, 0)
GAPI_TEST_FIXTURE(SumTest, initMatrixRandU, FIXTURE_API(compare_scalar_f), 1, cmpF)
GAPI_TEST_FIXTURE(AddWeightedTest, initMatsRandU, FIXTURE_API(compare_f), 1, cmpF)
GAPI_TEST_FIXTURE(NormTest, initMatrixRandU, FIXTURE_API(compare_scalar_f,NormTypes), 2,
    cmpF, opType)
GAPI_TEST_FIXTURE(IntegralTest, initNothing, <>, 0)
GAPI_TEST_FIXTURE(ThresholdTest, initMatrixRandU, FIXTURE_API(int), 1, tt)
GAPI_TEST_FIXTURE(ThresholdOTTest, initMatrixRandU, FIXTURE_API(int), 1, tt)
GAPI_TEST_FIXTURE(InRangeTest, initMatrixRandU, <>, 0)
GAPI_TEST_FIXTURE(Split3Test, initMatrixRandU, <>, 0)
GAPI_TEST_FIXTURE(Split4Test, initMatrixRandU, <>, 0)
GAPI_TEST_FIXTURE(ResizeTest, initNothing, FIXTURE_API(compare_f,int,cv::Size), 3,
    cmpF, interp, sz_out)
GAPI_TEST_FIXTURE(ResizePTest, initNothing, FIXTURE_API(compare_f,int,cv::Size), 3,
    cmpF, interp, sz_out)
GAPI_TEST_FIXTURE(ResizeTestFxFy, initNothing, FIXTURE_API(compare_f,int,double,double), 4,
    cmpF, interp, fx, fy)
GAPI_TEST_FIXTURE(Merge3Test, initMatsRandU, <>, 0)
GAPI_TEST_FIXTURE(Merge4Test, initMatsRandU, <>, 0)
GAPI_TEST_FIXTURE(RemapTest, initMatrixRandU, <>, 0)
GAPI_TEST_FIXTURE(FlipTest, initMatrixRandU, FIXTURE_API(int), 1, flipCode)
GAPI_TEST_FIXTURE(CropTest, initMatrixRandU, FIXTURE_API(cv::Rect), 1, rect_to)
GAPI_TEST_FIXTURE(ConcatHorTest, initNothing, <>, 0)
GAPI_TEST_FIXTURE(ConcatVertTest, initNothing, <>, 0)
GAPI_TEST_FIXTURE(ConcatVertVecTest, initNothing, <>, 0)
GAPI_TEST_FIXTURE(ConcatHorVecTest, initNothing, <>, 0)
GAPI_TEST_FIXTURE(LUTTest, initNothing, <>, 0)
GAPI_TEST_FIXTURE(ConvertToTest, initNothing, FIXTURE_API(compare_f, double, double), 3,
    cmpF, alpha, beta)
GAPI_TEST_FIXTURE(PhaseTest, initMatsRandU, FIXTURE_API(bool), 1, angle_in_degrees)
GAPI_TEST_FIXTURE(SqrtTest, initMatrixRandU, <>, 0)
GAPI_TEST_FIXTURE(NormalizeTest, initNothing, FIXTURE_API(compare_f,double,double,int,MatType), 5,
    cmpF, a, b, norm_type, ddepth)
} // opencv_test

#endif //OPENCV_GAPI_CORE_TESTS_HPP
