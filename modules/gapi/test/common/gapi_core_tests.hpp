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
        cv::Size sz = std::get<4>(info.param);
        ss<<MathOperations[std::get<0>(info.param)]
                    <<"_"<<std::get<1>(info.param)
                    <<"_"<<std::get<2>(info.param)
                    <<"_"<<(int)std::get<3>(info.param)
                    <<"_"<<sz.width
                    <<"x"<<sz.height
                    <<"_"<<(std::get<5>(info.param)+1)
                    <<"_"<<std::get<6>(info.param)
                    <<"_"<<std::get<7>(info.param);
        return ss.str();
   }
};

struct PrintCmpCoreParams
{
    template <class TestParams>
    std::string operator()(const ::testing::TestParamInfo<TestParams>& info) const
    {
        std::stringstream ss;
        cv::Size sz = std::get<3>(info.param);
        ss<<CompareOperations[std::get<0>(info.param)]
                    <<"_"<<std::get<1>(info.param)
                    <<"_"<<std::get<2>(info.param)
                    <<"_"<<sz.width
                    <<"x"<<sz.height
                    <<"_"<<std::get<4>(info.param);
        return ss.str();
   }
};

struct PrintBWCoreParams
{
    template <class TestParams>
    std::string operator()(const ::testing::TestParamInfo<TestParams>& info) const
    {
        std::stringstream ss;
        cv::Size sz = std::get<2>(info.param);
        ss<<BitwiseOperations[std::get<0>(info.param)]
                    <<"_"<<std::get<1>(info.param)
                    <<"_"<<sz.width
                    <<"x"<<sz.height
                    <<"_"<<std::get<3>(info.param);
        return ss.str();
   }
};

struct PrintNormCoreParams
{
    template <class TestParams>
    std::string operator()(const ::testing::TestParamInfo<TestParams>& info) const
    {
        std::stringstream ss;
        cv::Size sz = std::get<2>(info.param);
        ss<<NormOperations[std::get<0>(info.param)]
                    <<"_"<<std::get<1>(info.param)
                    <<"_"<<sz.width
                    <<"x"<<sz.height;
        return ss.str();
   }
};

struct MathOpTest        : public TestParams<std::tuple<mathOp,bool,int,double,cv::Size,int,bool,bool,cv::GCompileArgs>>{};
struct MulDoubleTest     : public TestParams<std::tuple<int,cv::Size,int,bool,cv::GCompileArgs>>{};
struct DivTest           : public TestParams<std::tuple<int,cv::Size,int,bool, cv::GCompileArgs>>{};
struct DivCTest          : public TestParams<std::tuple<int,cv::Size,int,bool, cv::GCompileArgs>>{};
struct MeanTest          : public TestParams<std::tuple<int,cv::Size,bool, cv::GCompileArgs>> {};
struct MaskTest          : public TestParams<std::tuple<int,cv::Size,bool, cv::GCompileArgs>> {};
struct Polar2CartTest    : public TestParams<std::tuple<cv::Size,bool, cv::GCompileArgs>> {};
struct Cart2PolarTest    : public TestParams<std::tuple<cv::Size,bool, cv::GCompileArgs>> {};
struct CmpTest           : public TestParams<std::tuple<CmpTypes,bool,int,cv::Size,bool, cv::GCompileArgs>>{};
struct BitwiseTest       : public TestParams<std::tuple<bitwiseOp,int,cv::Size,bool, cv::GCompileArgs>>{};
struct NotTest           : public TestParams<std::tuple<int,cv::Size,bool, cv::GCompileArgs>> {};
struct SelectTest        : public TestParams<std::tuple<int,cv::Size,bool, cv::GCompileArgs>> {};
struct MinTest           : public TestParams<std::tuple<int,cv::Size,bool, cv::GCompileArgs>>{};
struct MaxTest           : public TestParams<std::tuple<int,cv::Size,bool, cv::GCompileArgs>>{};
struct AbsDiffTest       : public TestParams<std::tuple<int,cv::Size,bool, cv::GCompileArgs>>{};
struct AbsDiffCTest      : public TestParams<std::tuple<int,cv::Size,bool, cv::GCompileArgs>> {};
struct SumTest           : public TestParams<std::tuple<int, cv::Size,bool,double,cv::GCompileArgs>> {};
struct AddWeightedTest   : public TestParams<std::tuple<int,cv::Size,int,bool,double,cv::GCompileArgs>>{};
struct NormTest          : public TestParams<std::tuple<NormTypes,int,cv::Size, double, cv::GCompileArgs>>{};
struct IntegralTest      : public TestWithParam<std::tuple<int,cv::Size, cv::GCompileArgs>> {};
struct ThresholdTest     : public TestParams<std::tuple<int,cv::Size,int,bool, cv::GCompileArgs>> {};
struct ThresholdOTTest   : public TestParams<std::tuple<int,cv::Size,int,bool, cv::GCompileArgs>> {};
struct InRangeTest       : public TestParams<std::tuple<int,cv::Size,bool, cv::GCompileArgs>> {};
struct Split3Test        : public TestParams<std::tuple<cv::Size, cv::GCompileArgs>> {};
struct Split4Test        : public TestParams<std::tuple<cv::Size, cv::GCompileArgs>> {};
struct ResizeTest        : public TestWithParam<std::tuple<compare_f, int, int, cv::Size, cv::Size, cv::GCompileArgs>> {};
struct ResizeTestFxFy    : public TestWithParam<std::tuple<compare_f, int, int, cv::Size, double, double, cv::GCompileArgs>> {};
struct Merge3Test        : public TestParams<std::tuple<cv::Size, cv::GCompileArgs>> {};
struct Merge4Test        : public TestParams<std::tuple<cv::Size, cv::GCompileArgs>> {};
struct RemapTest         : public TestParams<std::tuple<int,cv::Size,bool, cv::GCompileArgs>> {};
struct FlipTest          : public TestParams<std::tuple<int, int, cv::Size,bool, cv::GCompileArgs>> {};
struct CropTest          : public TestParams<std::tuple<int,cv::Rect,cv::Size,bool, cv::GCompileArgs>> {};
struct ConcatHorTest     : public TestWithParam<std::tuple<int, cv::Size, cv::GCompileArgs>> {};
struct ConcatVertTest    : public TestWithParam<std::tuple<int, cv::Size, cv::GCompileArgs>> {};
struct ConcatVertVecTest : public TestWithParam<std::tuple<int, cv::Size, cv::GCompileArgs>> {};
struct ConcatHorVecTest  : public TestWithParam<std::tuple<int, cv::Size, cv::GCompileArgs>> {};
struct LUTTest           : public TestParams<std::tuple<int, int, cv::Size,bool, cv::GCompileArgs>> {};
struct ConvertToTest     : public TestParams<std::tuple<int, int, cv::Size, cv::GCompileArgs>> {};
} // opencv_test

#endif //OPENCV_GAPI_CORE_TESTS_HPP
