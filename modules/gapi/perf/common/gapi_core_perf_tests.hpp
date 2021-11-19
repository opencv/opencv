// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2021 Intel Corporation


#ifndef OPENCV_GAPI_CORE_PERF_TESTS_HPP
#define OPENCV_GAPI_CORE_PERF_TESTS_HPP


#include "../../test/common/gapi_tests_common.hpp"
#include "../../test/common/gapi_parsers_tests_common.hpp"
#include <opencv2/gapi/core.hpp>

namespace opencv_test
{
  using namespace perf;

  enum bitwiseOp
  {
      AND = 0,
      OR = 1,
      XOR = 2,
      NOT = 3
  };

//------------------------------------------------------------------------------

    class AddPerfTest : public TestPerfParams<tuple<cv::Size, MatType, int, cv::GCompileArgs>> {};
    class AddCPerfTest : public TestPerfParams<tuple<cv::Size, MatType, int, cv::GCompileArgs>> {};
    class SubPerfTest : public TestPerfParams<tuple<cv::Size, MatType, int, cv::GCompileArgs>> {};
    class SubCPerfTest : public TestPerfParams<tuple<cv::Size, MatType, int, cv::GCompileArgs>> {};
    class SubRCPerfTest : public TestPerfParams<tuple<cv::Size, MatType, int, cv::GCompileArgs>> {};
    class MulPerfTest : public TestPerfParams<tuple<compare_f, cv::Size, MatType, int, double, cv::GCompileArgs>> {};
    class MulDoublePerfTest : public TestPerfParams<tuple<cv::Size, MatType, int, cv::GCompileArgs>> {};
    class MulCPerfTest : public TestPerfParams<tuple<cv::Size, MatType, int, cv::GCompileArgs>> {};
    class DivPerfTest : public TestPerfParams<tuple<compare_f, cv::Size, MatType, int, double, cv::GCompileArgs>> {};
    class DivCPerfTest : public TestPerfParams<tuple<cv::Size, MatType, int, cv::GCompileArgs>> {};
    class DivRCPerfTest : public TestPerfParams<tuple<compare_f,cv::Size, MatType, int, cv::GCompileArgs>> {};
    class MaskPerfTest : public TestPerfParams<tuple<cv::Size, MatType, cv::GCompileArgs>> {};
    class MeanPerfTest : public TestPerfParams<tuple<cv::Size, MatType, cv::GCompileArgs>> {};
    class Polar2CartPerfTest : public TestPerfParams<tuple<compare_f, cv::Size, cv::GCompileArgs>> {};
    class Cart2PolarPerfTest : public TestPerfParams<tuple<compare_f, cv::Size, cv::GCompileArgs>> {};
    class CmpPerfTest : public TestPerfParams<tuple<CmpTypes, cv::Size, MatType, cv::GCompileArgs>> {};
    class CmpWithScalarPerfTest : public TestPerfParams<tuple<compare_f, CmpTypes, cv::Size, MatType, cv::GCompileArgs>> {};
    class BitwisePerfTest : public TestPerfParams<tuple<bitwiseOp, bool, cv::Size, MatType, cv::GCompileArgs>> {};
    class BitwiseNotPerfTest : public TestPerfParams<tuple<cv::Size, MatType, cv::GCompileArgs>> {};
    class SelectPerfTest : public TestPerfParams<tuple<cv::Size, MatType, cv::GCompileArgs>> {};
    class MinPerfTest : public TestPerfParams<tuple<cv::Size, MatType, cv::GCompileArgs>> {};
    class MaxPerfTest : public TestPerfParams<tuple<cv::Size, MatType, cv::GCompileArgs>> {};
    class AbsDiffPerfTest : public TestPerfParams<tuple<cv::Size, MatType, cv::GCompileArgs>> {};
    class AbsDiffCPerfTest : public TestPerfParams<tuple<cv::Size, MatType, cv::GCompileArgs>> {};
    class SumPerfTest : public TestPerfParams<tuple<compare_scalar_f, cv::Size, MatType, cv::GCompileArgs>> {};
    class CountNonZeroPerfTest : public TestPerfParams<tuple<compare_scalar_f, cv::Size, MatType, cv::GCompileArgs>> {};
    class AddWeightedPerfTest : public TestPerfParams<tuple<compare_f, cv::Size, MatType, int, cv::GCompileArgs>> {};
    class NormPerfTest : public TestPerfParams<tuple<compare_scalar_f, NormTypes, cv::Size, MatType, cv::GCompileArgs>> {};
    class IntegralPerfTest : public TestPerfParams<tuple<cv::Size, MatType, cv::GCompileArgs>> {};
    class ThresholdPerfTest : public TestPerfParams<tuple<cv::Size, MatType, int, cv::GCompileArgs>> {};
    class ThresholdOTPerfTest : public TestPerfParams<tuple<cv::Size, MatType, int, cv::GCompileArgs>> {};
    class InRangePerfTest : public TestPerfParams<tuple<cv::Size, MatType, cv::GCompileArgs>> {};
    class Split3PerfTest : public TestPerfParams<tuple<cv::Size, cv::GCompileArgs>> {};
    class Split4PerfTest : public TestPerfParams<tuple<cv::Size, cv::GCompileArgs>> {};
    class Merge3PerfTest : public TestPerfParams<tuple<cv::Size, cv::GCompileArgs>> {};
    class Merge4PerfTest : public TestPerfParams<tuple<cv::Size, cv::GCompileArgs>> {};
    class RemapPerfTest : public TestPerfParams<tuple<cv::Size, MatType, cv::GCompileArgs>> {};
    class FlipPerfTest : public TestPerfParams<tuple<cv::Size, MatType, int, cv::GCompileArgs>> {};
    class CropPerfTest : public TestPerfParams<tuple<cv::Size, MatType, cv::Rect, cv::GCompileArgs>> {};
    class CopyPerfTest : public TestPerfParams<tuple<cv::Size, MatType, cv::GCompileArgs>> {};
    class ConcatHorPerfTest : public TestPerfParams<tuple<cv::Size, MatType, cv::GCompileArgs>> {};
    class ConcatHorVecPerfTest : public TestPerfParams<tuple<cv::Size, MatType, cv::GCompileArgs>> {};
    class ConcatVertPerfTest : public TestPerfParams<tuple<cv::Size, MatType, cv::GCompileArgs>> {};
    class ConcatVertVecPerfTest : public TestPerfParams<tuple<cv::Size, MatType, cv::GCompileArgs>> {};
    class LUTPerfTest : public TestPerfParams<tuple<MatType, MatType, cv::Size, cv::GCompileArgs>> {};
    class ConvertToPerfTest : public TestPerfParams<tuple<compare_f, MatType, int, cv::Size, double, double, cv::GCompileArgs>> {};
    class KMeansNDPerfTest : public TestPerfParams<tuple<cv::Size, CompareMats, int,
                                                         cv::KmeansFlags, cv::GCompileArgs>> {};
    class KMeans2DPerfTest : public TestPerfParams<tuple<int, int, cv::KmeansFlags,
                                                         cv::GCompileArgs>> {};
    class KMeans3DPerfTest : public TestPerfParams<tuple<int, int, cv::KmeansFlags,
                                                         cv::GCompileArgs>> {};
    class TransposePerfTest : public TestPerfParams<tuple<compare_f, cv::Size, MatType, cv::GCompileArgs>> {};
    class ResizePerfTest : public TestPerfParams<tuple<compare_f, MatType, int, cv::Size, cv::Size, cv::GCompileArgs>> {};
    class BottleneckKernelsConstInputPerfTest : public TestPerfParams<tuple<compare_f, std::string, cv::GCompileArgs>> {};
    class ResizeFxFyPerfTest : public TestPerfParams<tuple<compare_f, MatType, int, cv::Size, double, double, cv::GCompileArgs>> {};
    class ResizeInSimpleGraphPerfTest : public TestPerfParams<tuple<compare_f, MatType, cv::Size, cv::GCompileArgs>> {};
    class ParseSSDBLPerfTest : public TestPerfParams<tuple<cv::Size, float, int, cv::GCompileArgs>>, public ParserSSDTest {};
    class ParseSSDPerfTest   : public TestPerfParams<tuple<cv::Size, float, bool, bool, cv::GCompileArgs>>, public ParserSSDTest {};
    class ParseYoloPerfTest  : public TestPerfParams<tuple<cv::Size, float, float, int, cv::GCompileArgs>>, public ParserYoloTest {};
    class SizePerfTest       : public TestPerfParams<tuple<MatType, cv::Size, cv::GCompileArgs>> {};
    class SizeRPerfTest      : public TestPerfParams<tuple<cv::Size, cv::GCompileArgs>> {};
}
#endif // OPENCV_GAPI_CORE_PERF_TESTS_HPP
