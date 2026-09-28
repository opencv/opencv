// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "perf_precomp.hpp"

namespace opencv_test {
using namespace perf;

CV_ENUM(ReduceNDOp, REDUCE_SUM, REDUCE_AVG, REDUCE_MAX, REDUCE_L2, REDUCE_LOG_SUM_EXP)

typedef tuple<std::vector<int>, std::vector<int>, ReduceNDOp, perf::MatType> ReduceNDParams;
typedef TestBaseWithParam<ReduceNDParams> ReduceNDPerf;

PERF_TEST_P_(ReduceNDPerf, reduceND)
{
    const std::vector<int> shape = get<0>(GetParam()), axes = get<1>(GetParam());
    const int op = get<2>(GetParam()), type = get<3>(GetParam());
    Mat src(shape, type), dst;
    Mat bytes(1, (int)(src.total()*src.elemSize()), CV_8U, src.data);
    randu(bytes, 0, 100);   // declare.in() only handles 2D arrays
    if (CV_MAT_DEPTH(type) == CV_32F)
        randu(Mat(1, (int)src.total(), CV_32F, src.data), -1, 1);
    cv::reduceND(src, dst, axes, op);

    TEST_CYCLE() cv::reduceND(src, dst, axes, op);

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, ReduceNDPerf, testing::Combine(
    testing::Values(std::vector<int>{1, 256, 80, 80}),
    testing::Values(std::vector<int>{1}, std::vector<int>{2, 3}, std::vector<int>{}),
    ReduceNDOp::all(),
    testing::Values(perf::MatType(CV_32FC1), CV_8UC1)
));

INSTANTIATE_TEST_CASE_P(Rows, ReduceNDPerf, testing::Combine(
    testing::Values(std::vector<int>{4096, 768}),
    testing::Values(std::vector<int>{0}, std::vector<int>{1}),
    testing::Values(ReduceNDOp(REDUCE_SUM), ReduceNDOp(REDUCE_MAX)),
    testing::Values(perf::MatType(CV_32FC1), CV_16FC1)
));

} // namespace
