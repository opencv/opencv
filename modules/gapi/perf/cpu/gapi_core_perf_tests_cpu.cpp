// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../common/gapi_core_perf_tests.hpp"

namespace opencv_test
{

  INSTANTIATE_TEST_CASE_P(AddPerfTestCPU, AddPerfTest,
                          Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                  Values( -1, CV_8U, CV_16U, CV_32F )));

  INSTANTIATE_TEST_CASE_P(AddCPerfTestCPU, AddCPerfTest,
                          Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                  Values( -1, CV_8U, CV_16U, CV_32F )));

  INSTANTIATE_TEST_CASE_P(SubPerfTestCPU, SubPerfTest,
                          Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                  Values( -1, CV_8U, CV_16U, CV_32F )));

  INSTANTIATE_TEST_CASE_P(SubCPerfTestCPU, SubCPerfTest,
                          Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                  Values( -1, CV_8U, CV_16U, CV_32F )));

  INSTANTIATE_TEST_CASE_P(SubRCPerfTestCPU, SubRCPerfTest,
                          Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                  Values( -1, CV_8U, CV_16U, CV_32F )));

  INSTANTIATE_TEST_CASE_P(MulPerfTestCPU, MulPerfTest,
                          Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                  Values( -1, CV_8U, CV_16U, CV_32F )));

  INSTANTIATE_TEST_CASE_P(MulDoublePerfTestCPU, MulDoublePerfTest,
                          Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                  Values( -1, CV_8U, CV_16U, CV_32F )));

  INSTANTIATE_TEST_CASE_P(MulCPerfTestCPU, MulCPerfTest,
                          Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                  Values( -1, CV_8U, CV_16U, CV_32F )));

  INSTANTIATE_TEST_CASE_P(DivPerfTestCPU, DivPerfTest,
                          Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                  Values( -1, CV_8U, CV_16U, CV_32F )));

  INSTANTIATE_TEST_CASE_P(DivCPerfTestCPU, DivCPerfTest,
                          Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                  Values( -1, CV_8U, CV_16U, CV_32F )));

  INSTANTIATE_TEST_CASE_P(DivRCPerfTestCPU, DivRCPerfTest,
                          Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                  Values( -1, CV_8U, CV_16U, CV_32F )));

  INSTANTIATE_TEST_CASE_P(MaskPerfTestCPU, MaskPerfTest,
                          Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1, CV_16UC1, CV_16SC1)));

  INSTANTIATE_TEST_CASE_P(MeanPerfTestCPU, MeanPerfTest,
                          Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 )));

  INSTANTIATE_TEST_CASE_P(Polar2CartPerfTestCPU, Polar2CartPerfTest, Values( szSmall128, szVGA, sz720p, sz1080p ));

  INSTANTIATE_TEST_CASE_P(Cart2PolarPerfTestCPU, Cart2PolarPerfTest, Values( szSmall128, szVGA, sz720p, sz1080p ));

  INSTANTIATE_TEST_CASE_P(CmpPerfTestCPU, CmpPerfTest,
                          Combine(Values(CMP_EQ, CMP_GE, CMP_NE, CMP_GT, CMP_LT, CMP_LE),
                                  Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 )));

  INSTANTIATE_TEST_CASE_P(CmpWithScalarPerfTestCPU, CmpWithScalarPerfTest,
                          Combine(Values(CMP_EQ, CMP_GE, CMP_NE, CMP_GT, CMP_LT, CMP_LE),
                                  Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 )));

  INSTANTIATE_TEST_CASE_P(BitwisePerfTestCPU, BitwisePerfTest,
                          Combine(Values(AND, OR, XOR),
                                  Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1)));

  INSTANTIATE_TEST_CASE_P(BitwiseNotPerfTestCPU, BitwiseNotPerfTest,
                          Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 )));

  INSTANTIATE_TEST_CASE_P(SelectPerfTestCPU, SelectPerfTest,
                          Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 )));

  INSTANTIATE_TEST_CASE_P(MinPerfTestCPU, MinPerfTest,
                          Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 )));

  INSTANTIATE_TEST_CASE_P(MaxPerfTestCPU, MaxPerfTest,
                          Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 )));

  INSTANTIATE_TEST_CASE_P(AbsDiffPerfTestCPU, AbsDiffPerfTest,
                          Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 )));

  INSTANTIATE_TEST_CASE_P(AbsDiffCPerfTestCPU, AbsDiffCPerfTest,
                          Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 )));

  INSTANTIATE_TEST_CASE_P(SumPerfTestCPU, SumPerfTest,
                          Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 )));

  INSTANTIATE_TEST_CASE_P(AddWeightedPerfTestCPU, AddWeightedPerfTest,
                          Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                  Values( -1, CV_8U, CV_16U, CV_32F )));

  INSTANTIATE_TEST_CASE_P(NormPerfTestCPU, NormPerfTest,
                          Combine(Values(NORM_INF, NORM_L1, NORM_L2),
                                  Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 )));

  INSTANTIATE_TEST_CASE_P(IntegralPerfTestCPU, IntegralPerfTest,
                          Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1, CV_16UC1, CV_16SC1, CV_32FC1 )));

  INSTANTIATE_TEST_CASE_P(ThresholdPerfTestCPU, ThresholdPerfTest,
                          Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                  Values(cv::THRESH_BINARY, cv::THRESH_BINARY_INV, cv::THRESH_TRUNC, cv::THRESH_TOZERO, cv::THRESH_TOZERO_INV)));

  INSTANTIATE_TEST_CASE_P(ThresholdPerfTestCPU, ThresholdOTPerfTest,
                          Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1 ),
                                  Values(cv::THRESH_OTSU, cv::THRESH_TRIANGLE)));

  INSTANTIATE_TEST_CASE_P(InRangePerfTestCPU, InRangePerfTest,
                          Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1 )));

  INSTANTIATE_TEST_CASE_P(Split3PerfTestCPU, Split3PerfTest, Values( szSmall128, szVGA, sz720p, sz1080p ));

  INSTANTIATE_TEST_CASE_P(Split4PerfTestCPU, Split4PerfTest, Values( szSmall128, szVGA, sz720p, sz1080p ));

  INSTANTIATE_TEST_CASE_P(Merge3PerfTestCPU, Merge3PerfTest, Values( szSmall128, szVGA, sz720p, sz1080p ));

  INSTANTIATE_TEST_CASE_P(Merge4PerfTestCPU, Merge4PerfTest, Values( szSmall128, szVGA, sz720p, sz1080p ));

  INSTANTIATE_TEST_CASE_P(RemapPerfTestCPU, RemapPerfTest,
                          Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 )));

  INSTANTIATE_TEST_CASE_P(FlipPerfTestCPU, FlipPerfTest,
                          Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                  Values(0,1,-1)));

  INSTANTIATE_TEST_CASE_P(CropPerfTestCPU, CropPerfTest,
                          Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                  Values(cv::Rect(10, 8, 20, 35), cv::Rect(4, 10, 37, 50))));

  INSTANTIATE_TEST_CASE_P(ConcatHorPerfTestCPU, ConcatHorPerfTest,
                          Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 )));

  INSTANTIATE_TEST_CASE_P(ConcatHorVecPerfTestCPU, ConcatHorVecPerfTest,
                          Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 )));

  INSTANTIATE_TEST_CASE_P(ConcatVertPerfTestCPU, ConcatVertPerfTest,
                          Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 )));

  INSTANTIATE_TEST_CASE_P(ConcatVertVecPerfTestCPU, ConcatVertVecPerfTest,
                          Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 )));

  INSTANTIATE_TEST_CASE_P(LUTPerfTestCPU, LUTPerfTest,
                          Combine(Values(CV_8UC1, CV_8UC3),
                                  Values(CV_8UC1),
                                  Values( szSmall128, szVGA, sz720p, sz1080p )));

  INSTANTIATE_TEST_CASE_P(LUTPerfTestCustomCPU, LUTPerfTest,
                          Combine(Values(CV_8UC3),
                                  Values(CV_8UC3),
                                  Values( szSmall128, szVGA, sz720p, sz1080p )));

  INSTANTIATE_TEST_CASE_P(ConvertToPerfTestCPU, ConvertToPerfTest,
                          Combine(Values(CV_8UC3, CV_8UC1, CV_16UC1, CV_32FC1),
                                  Values(CV_8U, CV_16U, CV_16S, CV_32F),
                                  Values( szSmall128, szVGA, sz720p, sz1080p )));

  INSTANTIATE_TEST_CASE_P(ResizePerfTestCPU, ResizePerfTest,
                          Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                  Values(cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_AREA),
                                  Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values(cv::Size(64,64),
                                         cv::Size(30,30)),
                                  Values(0.0)));

  INSTANTIATE_TEST_CASE_P(ResizeFxFyPerfTestCPU, ResizeFxFyPerfTest,
                          Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                  Values(cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_AREA),
                                  Values( szSmall128, szVGA, sz720p, sz1080p ),
                                  Values(0.5, 0.1),
                                  Values(0.5, 0.1),
                                  Values(0.0)));
}
