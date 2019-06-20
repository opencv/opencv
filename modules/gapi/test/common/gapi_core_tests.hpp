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
    USE_UNIFORM_INIT(MathOpTest, false);
};
struct MulDoubleTest     : public TestWithParamBase<> { USE_UNIFORM_INIT(MulDoubleTest, true); };
struct DivTest           : public TestWithParamBase<> { USE_UNIFORM_INIT(DivTest, true); };
struct DivCTest          : public TestWithParamBase<> { USE_UNIFORM_INIT(DivCTest, true); };
struct MeanTest          : public TestWithParamBase<> { USE_UNIFORM_INIT(MeanTest, true); };
struct MaskTest          : public TestWithParamBase<> { USE_UNIFORM_INIT(MaskTest, true); };
struct Polar2CartTest    : public TestWithParamBase<> { USE_UNIFORM_INIT(Polar2CartTest, false); };
struct Cart2PolarTest    : public TestWithParamBase<> { USE_UNIFORM_INIT(Cart2PolarTest, false); };
struct CmpTest           : public TestWithParamBase<CmpTypes,bool>
{
    DEFINE_SPECIFIC_PARAMS_2(opType, testWithScalar);
    USE_UNIFORM_INIT(CmpTest, false);
};
struct BitwiseTest       : public TestWithParamBase<bitwiseOp>
{
    DEFINE_SPECIFIC_PARAMS_1(opType);
    USE_UNIFORM_INIT(BitwiseTest, false);
};
struct NotTest           : public TestWithParamBase<> { USE_UNIFORM_INIT(NotTest, true); };
struct SelectTest        : public TestWithParamBase<> { USE_UNIFORM_INIT(SelectTest, false); };
struct MinTest           : public TestWithParamBase<> { USE_UNIFORM_INIT(MinTest, false); };
struct MaxTest           : public TestWithParamBase<> { USE_UNIFORM_INIT(MaxTest, false); };
struct AbsDiffTest       : public TestWithParamBase<> { USE_UNIFORM_INIT(AbsDiffTest, false); };
struct AbsDiffCTest      : public TestWithParamBase<> { USE_UNIFORM_INIT(AbsDiffCTest, false); };
struct SumTest           : public TestWithParamBase<compare_scalar_f>
{
    DEFINE_SPECIFIC_PARAMS_1(cmpF);
    USE_UNIFORM_INIT(SumTest, true);
};
struct AddWeightedTest   : public TestWithParamBase<compare_f>
{
    DEFINE_SPECIFIC_PARAMS_1(cmpF);
    USE_UNIFORM_INIT(AddWeightedTest, false);
};
struct NormTest          : public TestWithParamBase<compare_scalar_f,NormTypes>
{
    DEFINE_SPECIFIC_PARAMS_2(cmpF, opType);
    USE_UNIFORM_INIT(NormTest, true);
};
struct IntegralTest      : public TestWithParamBase<> { USE_UNIFORM_INIT(IntegralTest, false); };
struct ThresholdTest     : public TestWithParamBase<int>
{
    DEFINE_SPECIFIC_PARAMS_1(tt);
    USE_UNIFORM_INIT(ThresholdTest, true);
};
struct ThresholdOTTest   : public TestWithParamBase<int>
{
    DEFINE_SPECIFIC_PARAMS_1(tt);
    USE_UNIFORM_INIT(ThresholdOTTest, true);
};
struct InRangeTest       : public TestWithParamBase<> { USE_UNIFORM_INIT(InRangeTest, true); };
struct Split3Test        : public TestWithParamBase<> { USE_UNIFORM_INIT(Split3Test, true); };
struct Split4Test        : public TestWithParamBase<> { USE_UNIFORM_INIT(Split4Test, true); };
struct ResizeTest        : public TestWithParamBase<compare_f,int,cv::Size>
{
    DEFINE_SPECIFIC_PARAMS_3(cmpF, interp, sz_out);
    USE_UNIFORM_INIT(ResizeTest, false);
};
struct ResizeTestFxFy    : public TestWithParamBase<compare_f,int,double,double>
{
    DEFINE_SPECIFIC_PARAMS_4(cmpF, interp, fx, fy);
    USE_UNIFORM_INIT(ResizeTestFxFy, false);
};
struct Merge3Test        : public TestWithParamBase<> { USE_UNIFORM_INIT(Merge3Test, false); };
struct Merge4Test        : public TestWithParamBase<> { USE_UNIFORM_INIT(Merge4Test, false); };
struct RemapTest         : public TestWithParamBase<> { USE_UNIFORM_INIT(RemapTest, true); };
struct FlipTest          : public TestWithParamBase<int>
{
    DEFINE_SPECIFIC_PARAMS_1(flipCode);
    USE_UNIFORM_INIT(FlipTest, true);
};
struct CropTest          : public TestWithParamBase<cv::Rect>
{
    DEFINE_SPECIFIC_PARAMS_1(rect_to);
    USE_UNIFORM_INIT(CropTest, true);
};
struct ConcatHorTest     : public TestWithParamBase<> { USE_UNIFORM_INIT(ConcatHorTest, false); };
struct ConcatVertTest    : public TestWithParamBase<> { USE_UNIFORM_INIT(ConcatVertTest, false); };
struct ConcatVertVecTest : public TestWithParamBase<> { USE_UNIFORM_INIT(ConcatVertVecTest, false); };
struct ConcatHorVecTest  : public TestWithParamBase<> { USE_UNIFORM_INIT(ConcatHorVecTest, false); };
struct LUTTest           : public TestWithParamBase<> { USE_UNIFORM_INIT(LUTTest, true); };
struct ConvertToTest     : public TestWithParamBase<compare_f, double, double>
{
    DEFINE_SPECIFIC_PARAMS_3(cmpF, alpha, beta);
    USE_UNIFORM_INIT(ConvertToTest, true);
};
struct PhaseTest         : public TestWithParamBase<bool>
{
    DEFINE_SPECIFIC_PARAMS_1(angle_in_degrees);
    USE_UNIFORM_INIT(PhaseTest, false);
};
struct SqrtTest          : public TestWithParamBase<> { USE_UNIFORM_INIT(SqrtTest, true); };
struct NormalizeTest : public TestWithParamBase<compare_f,double,double,int,MatType>
{
    DEFINE_SPECIFIC_PARAMS_5(cmpF, a, b, norm_type, ddepth);
    USE_NORMAL_INIT(NormalizeTest);
};
} // opencv_test

#endif //OPENCV_GAPI_CORE_TESTS_HPP
