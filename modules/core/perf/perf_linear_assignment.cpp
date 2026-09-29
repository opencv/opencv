// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

// The threshold decides which of the two paths runs. Above every cost nothing is forbidden and
// the dummy columns are skipped, so the solver sees an n-column matrix. Inside the cost range it
// forbids pairs, and the padded n + rows matrix is used instead.
typedef TestBaseWithParam< tuple<Size, double> > Size_Threshold_LinearAssignment;

PERF_TEST_P_(Size_Threshold_LinearAssignment, solve)
{
    const Size sz = get<0>(GetParam());
    const double costThreshold = get<1>(GetParam());

    Mat cost(sz.height, sz.width, CV_64F);
    declare.in(cost, WARMUP_RNG);

    std::vector<int> assignment;
    TEST_CYCLE() cv::linearAssignment(cost, assignment, costThreshold);

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, Size_Threshold_LinearAssignment,
                        testing::Combine(
                            testing::Values(Size(64, 64), Size(256, 256), Size(512, 512),
                                            Size(512, 128)),
                            testing::Values(DBL_MAX, 0.5)));

} // namespace opencv_test
