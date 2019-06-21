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
        // TODO: simplify this
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

struct MathOpTest        : public TestWithParamBase<mathOp,bool,double,bool>
{
    DEFINE_SPECIFIC_PARAMS_4(opType, testWithScalar, scale, doReverseOp);
    USE_UNIFORM_INIT(MathOpTest);
};
struct MulDoubleTest     : public TestWithParamBase<> { USE_UNIFORM_INIT_ONE_MAT(MulDoubleTest); };
struct DivTest           : public TestWithParamBase<> { USE_UNIFORM_INIT_ONE_MAT(DivTest); };
struct DivCTest          : public TestWithParamBase<> { USE_UNIFORM_INIT_ONE_MAT(DivCTest); };
struct MeanTest          : public TestWithParamBase<> { USE_UNIFORM_INIT_ONE_MAT(MeanTest); };
struct MaskTest          : public TestWithParamBase<> { USE_UNIFORM_INIT_ONE_MAT(MaskTest); };
struct Polar2CartTest    : public TestWithParamBase<> { USE_UNIFORM_INIT(Polar2CartTest); };
struct Cart2PolarTest    : public TestWithParamBase<> { USE_UNIFORM_INIT(Cart2PolarTest); };
struct CmpTest           : public TestWithParamBase<CmpTypes,bool>
{
    DEFINE_SPECIFIC_PARAMS_2(opType, testWithScalar);
    USE_UNIFORM_INIT(CmpTest);
};
struct BitwiseTest       : public TestWithParamBase<bitwiseOp>
{
    DEFINE_SPECIFIC_PARAMS_1(opType);
    USE_UNIFORM_INIT(BitwiseTest);
};
struct NotTest           : public TestWithParamBase<> { USE_UNIFORM_INIT_ONE_MAT(NotTest); };
struct SelectTest        : public TestWithParamBase<> { USE_UNIFORM_INIT(SelectTest); };
struct MinTest           : public TestWithParamBase<> { USE_UNIFORM_INIT(MinTest); };
struct MaxTest           : public TestWithParamBase<> { USE_UNIFORM_INIT(MaxTest); };
struct AbsDiffTest       : public TestWithParamBase<> { USE_UNIFORM_INIT(AbsDiffTest); };
struct AbsDiffCTest      : public TestWithParamBase<> { USE_UNIFORM_INIT(AbsDiffCTest); };
struct SumTest           : public TestWithParamBase<compare_scalar_f>
{
    DEFINE_SPECIFIC_PARAMS_1(cmpF);
    USE_UNIFORM_INIT_ONE_MAT(SumTest);
};
struct AddWeightedTest   : public TestWithParamBase<compare_f>
{
    DEFINE_SPECIFIC_PARAMS_1(cmpF);
    USE_UNIFORM_INIT(AddWeightedTest);
};
struct NormTest          : public TestWithParamBase<compare_scalar_f,NormTypes>
{
    DEFINE_SPECIFIC_PARAMS_2(cmpF, opType);
    USE_UNIFORM_INIT_ONE_MAT(NormTest);
};
struct IntegralTest      : public TestWithParamBase<> { USE_UNIFORM_INIT(IntegralTest); };
struct ThresholdTest     : public TestWithParamBase<int>
{
    DEFINE_SPECIFIC_PARAMS_1(tt);
    USE_UNIFORM_INIT_ONE_MAT(ThresholdTest);
};
struct ThresholdOTTest   : public TestWithParamBase<int>
{
    DEFINE_SPECIFIC_PARAMS_1(tt);
    USE_UNIFORM_INIT_ONE_MAT(ThresholdOTTest);
};
struct InRangeTest       : public TestWithParamBase<> { USE_UNIFORM_INIT_ONE_MAT(InRangeTest); };
struct Split3Test        : public TestWithParamBase<> { USE_UNIFORM_INIT_ONE_MAT(Split3Test); };
struct Split4Test        : public TestWithParamBase<> { USE_UNIFORM_INIT_ONE_MAT(Split4Test); };
struct ResizeTest        : public TestWithParamBase<compare_f,int,cv::Size>
{
    DEFINE_SPECIFIC_PARAMS_3(cmpF, interp, sz_out);
    USE_UNIFORM_INIT(ResizeTest);
};
struct ResizeTestFxFy    : public TestWithParamBase<compare_f,int,double,double>
{
    DEFINE_SPECIFIC_PARAMS_4(cmpF, interp, fx, fy);
    USE_UNIFORM_INIT(ResizeTestFxFy);
};
struct Merge3Test        : public TestWithParamBase<> { USE_UNIFORM_INIT(Merge3Test); };
struct Merge4Test        : public TestWithParamBase<> { USE_UNIFORM_INIT(Merge4Test); };
struct RemapTest         : public TestWithParamBase<> { USE_UNIFORM_INIT_ONE_MAT(RemapTest); };
struct FlipTest          : public TestWithParamBase<int>
{
    DEFINE_SPECIFIC_PARAMS_1(flipCode);
    USE_UNIFORM_INIT_ONE_MAT(FlipTest);
};
struct CropTest          : public TestWithParamBase<cv::Rect>
{
    DEFINE_SPECIFIC_PARAMS_1(rect_to);
    USE_UNIFORM_INIT_ONE_MAT(CropTest);
};
struct ConcatHorTest     : public TestWithParamBase<> { USE_UNIFORM_INIT(ConcatHorTest); };
struct ConcatVertTest    : public TestWithParamBase<> { USE_UNIFORM_INIT(ConcatVertTest); };
struct ConcatVertVecTest : public TestWithParamBase<> { USE_UNIFORM_INIT(ConcatVertVecTest); };
struct ConcatHorVecTest  : public TestWithParamBase<> { USE_UNIFORM_INIT(ConcatHorVecTest); };
struct LUTTest           : public TestWithParamBase<> { USE_UNIFORM_INIT_ONE_MAT(LUTTest); };
struct ConvertToTest     : public TestWithParamBase<compare_f, double, double>
{
    DEFINE_SPECIFIC_PARAMS_3(cmpF, alpha, beta);
    USE_UNIFORM_INIT_ONE_MAT(ConvertToTest);
};
struct PhaseTest         : public TestWithParamBase<bool>
{
    DEFINE_SPECIFIC_PARAMS_1(angle_in_degrees);
    USE_UNIFORM_INIT(PhaseTest);
};
struct SqrtTest          : public TestWithParamBase<> { USE_UNIFORM_INIT_ONE_MAT(SqrtTest); };
struct NormalizeTest : public TestWithParamBase<compare_f,double,double,int,MatType>
{
    DEFINE_SPECIFIC_PARAMS_5(cmpF, a, b, norm_type, ddepth);
    USE_NORMAL_INIT(NormalizeTest);
};
} // opencv_test

#endif //OPENCV_GAPI_CORE_TESTS_HPP
