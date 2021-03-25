// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../perf_precomp.hpp"
#include "../../test/common/gapi_tests_common.hpp"

namespace opencv_test
{
using namespace perf;

class CompilerPerfTest : public TestPerfParams<tuple<cv::Size, MatType>> {};
PERF_TEST_P_(CompilerPerfTest, TestPerformance)
{
  const auto params = GetParam();
  Size sz = get<0>(params);
  MatType type = get<1>(params);

  initMatsRandU(type, sz, type, false);

  // G-API code ////////////////////////////////////////////////////////////
  cv::GMat in;
  auto splitted = cv::gapi::split3(in);
  auto add1 = cv::gapi::addC({1}, std::get<0>(splitted));
  auto add2 = cv::gapi::addC({2}, std::get<1>(splitted));
  auto add3 = cv::gapi::addC({3}, std::get<2>(splitted));
  auto out = cv::gapi::merge3(add1, add2, add3);

  TEST_CYCLE()
  {
      cv::GComputation c(in, out);
      c.apply(in_mat1, out_mat_gapi, cv::compile_args(cv::gapi::core::fluid::kernels()));
  }

  SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(CompilerPerfTest, CompilerPerfTest,
                        Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
                                Values(CV_8UC3)));

} // namespace opencv_test
