// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


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
        Params<mathOp,bool,double,bool> params = info.param;
        const auto& common_params = params.commonParams();
        const auto& specific_params = params.specificParams();
        cv::Size sz = std::get<1>(common_params);  // size
        ss<<MathOperations[std::get<0>(specific_params)]  // mathOp
                    <<"_"<<std::get<1>(specific_params)  // testWithScalar
                    <<"_"<<std::get<0>(common_params)  // type
                    <<"_"<<(int)std::get<2>(specific_params)  // scale
                    <<"_"<<sz.width
                    <<"x"<<sz.height
                    <<"_"<<(std::get<2>(common_params)+1)  // dtype
                    <<"_"<<std::get<3>(common_params)  // createOutputMatrices
                    <<"_"<<std::get<3>(specific_params);  // doReverseOp
        return ss.str();
   }
};

struct PrintCmpCoreParams
{
    template <class TestParams>
    std::string operator()(const ::testing::TestParamInfo<TestParams>& info) const
    {
        std::stringstream ss;
        Params<CmpTypes,bool> params = info.param;
        const auto& common_params = params.commonParams();
        const auto& specific_params = params.specificParams();
        cv::Size sz = std::get<1>(common_params);  // size
        ss<<CompareOperations[std::get<0>(specific_params)]  // CmpType
                    <<"_"<<std::get<1>(specific_params)  // testWithScalar
                    <<"_"<<std::get<0>(common_params)  // type
                    <<"_"<<sz.width
                    <<"x"<<sz.height
                    <<"_"<<std::get<3>(common_params);  // createOutputMatrices
        return ss.str();
   }
};

struct PrintBWCoreParams
{
    template <class TestParams>
    std::string operator()(const ::testing::TestParamInfo<TestParams>& info) const
    {
        std::stringstream ss;
        Params<bitwiseOp> params = info.param;
        const auto& common_params = params.commonParams();
        const auto& specific_params = params.specificParams();
        cv::Size sz = std::get<1>(common_params);  // size
        ss<<BitwiseOperations[std::get<0>(specific_params)]  // bitwiseOp
                    <<"_"<<std::get<0>(common_params)  // type
                    <<"_"<<sz.width
                    <<"x"<<sz.height
                    <<"_"<<std::get<3>(common_params);  // createOutputMatrices
        return ss.str();
   }
};

struct PrintNormCoreParams
{
    template <class TestParams>
    std::string operator()(const ::testing::TestParamInfo<TestParams>& info) const
    {
        std::stringstream ss;
        Params<compare_scalar_f,NormTypes> params = info.param;
        const auto& common_params = params.commonParams();
        const auto& specific_params = params.specificParams();
        cv::Size sz = std::get<1>(common_params);  // size
        ss<<NormOperations[std::get<1>(specific_params)]  // NormTypes
                    <<"_"<<std::get<0>(common_params)  // type
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
struct MulDoubleTest     : public TestWithParamBase<>
{
    USE_UNIFORM_INIT(MulDoubleTest);
};
struct DivTest           : public TestWithParamBase<>
{
    USE_UNIFORM_INIT(DivTest);
};
struct DivCTest          : public TestWithParamBase<>
{
    USE_UNIFORM_INIT(DivCTest);
};
struct MeanTest          : public TestWithParamBase<>
{
    USE_UNIFORM_INIT(MeanTest);
};
struct MaskTest          : public TestWithParamBase<>
{
    USE_UNIFORM_INIT(MaskTest);
};
struct Polar2CartTest    : public TestWithParamBase<>
{
    USE_UNIFORM_INIT(Polar2CartTest);
};
struct Cart2PolarTest    : public TestWithParamBase<>
{
    USE_UNIFORM_INIT(Cart2PolarTest);
};
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
struct NotTest           : public TestWithParamBase<>
{
    USE_UNIFORM_INIT(NotTest);
};
struct SelectTest        : public TestWithParamBase<>
{
    USE_UNIFORM_INIT(SelectTest);
};
struct MinTest           : public TestWithParamBase<>
{
    USE_UNIFORM_INIT(MinTest);
};
struct MaxTest           : public TestWithParamBase<>
{
    USE_UNIFORM_INIT(MaxTest);
};
struct AbsDiffTest       : public TestWithParamBase<>
{
    USE_UNIFORM_INIT(AbsDiffTest);
};
struct AbsDiffCTest      : public TestWithParamBase<>
{
    USE_UNIFORM_INIT(AbsDiffCTest);
};
struct SumTest           : public TestWithParamBase<compare_scalar_f>
{
    DEFINE_SPECIFIC_PARAMS_1(cmpF);
    USE_UNIFORM_INIT(SumTest);
};
struct AddWeightedTest   : public TestWithParamBase<compare_f>
{
    DEFINE_SPECIFIC_PARAMS_1(cmpF);
    USE_UNIFORM_INIT(AddWeightedTest);
};
struct NormTest          : public TestWithParamBase<compare_scalar_f,NormTypes>
{
    DEFINE_SPECIFIC_PARAMS_2(cmpF, opType);
    USE_UNIFORM_INIT(NormTest);
};
struct IntegralTest      : public TestWithParamBase<>
{
    USE_UNIFORM_INIT(IntegralTest);
};
struct ThresholdTest     : public TestWithParamBase<int>
{
    DEFINE_SPECIFIC_PARAMS_1(tt);
    USE_UNIFORM_INIT(ThresholdTest);
};
struct ThresholdOTTest   : public TestWithParamBase<int>
{
    DEFINE_SPECIFIC_PARAMS_1(tt);
    USE_UNIFORM_INIT(ThresholdOTTest);
};
struct InRangeTest       : public TestWithParamBase<>
{
    USE_UNIFORM_INIT(InRangeTest);
};
struct Split3Test        : public TestWithParamBase<>
{
    USE_UNIFORM_INIT(Split3Test);
};
struct Split4Test        : public TestWithParamBase<>
{
    USE_UNIFORM_INIT(Split4Test);
};
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
struct Merge3Test        : public TestWithParamBase<>
{
    USE_UNIFORM_INIT(Merge3Test);
};
struct Merge4Test        : public TestWithParamBase<>
{
    USE_UNIFORM_INIT(Merge4Test);
};
struct RemapTest         : public TestWithParamBase<>
{
    USE_UNIFORM_INIT(RemapTest);
};
struct FlipTest          : public TestWithParamBase<int>
{
    DEFINE_SPECIFIC_PARAMS_1(flipCode);
    USE_UNIFORM_INIT(FlipTest);
};
struct CropTest          : public TestWithParamBase<cv::Rect>
{
    DEFINE_SPECIFIC_PARAMS_1(rect_to);
    USE_UNIFORM_INIT(CropTest);
};
struct ConcatHorTest     : public TestWithParamBase<>
{
    USE_UNIFORM_INIT(ConcatHorTest);
};
struct ConcatVertTest    : public TestWithParamBase<>
{
    USE_UNIFORM_INIT(ConcatVertTest);
};
struct ConcatVertVecTest : public TestWithParamBase<>
{
    USE_UNIFORM_INIT(ConcatVertVecTest);
};
struct ConcatHorVecTest  : public TestWithParamBase<>
{
    USE_UNIFORM_INIT(ConcatHorVecTest);
};
struct LUTTest           : public TestWithParamBase<>
{
    USE_UNIFORM_INIT(LUTTest);
};
struct ConvertToTest     : public TestWithParamBase<compare_f, double, double>
{
    DEFINE_SPECIFIC_PARAMS_3(cmpF, alpha, beta);
    USE_UNIFORM_INIT(ConvertToTest);
};
struct PhaseTest         : public TestWithParamBase<bool>
{
    DEFINE_SPECIFIC_PARAMS_1(angle_in_degrees);
    USE_UNIFORM_INIT(PhaseTest);
};
struct SqrtTest          : public TestWithParamBase<>
{
    USE_UNIFORM_INIT(SqrtTest);
};
struct NormalizeTest : public TestWithParamBase<compare_f,double,double,int,MatType>
{
    DEFINE_SPECIFIC_PARAMS_5(cmpF, a, b, norm_type, ddepth);
    USE_NORMAL_INIT(NormalizeTest);
};
} // opencv_test

#endif //OPENCV_GAPI_CORE_TESTS_HPP
