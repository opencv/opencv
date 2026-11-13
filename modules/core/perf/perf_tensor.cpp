// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

// a CPU Tensor should cost the same as a Mat in a cv:: call
PERF_TEST_P(Size_MatType, Tensor_add_mat,
            testing::Combine(testing::Values(szVGA, sz1080p), testing::Values(CV_8UC3, CV_32FC1)))
{
    const Size sz = get<0>(GetParam());
    const int type = get<1>(GetParam());
    Mat a(sz, type), b(sz, type), dst(sz, type);
    declare.in(a, b, WARMUP_RNG).out(dst);

    TEST_CYCLE() cv::add(a, b, dst);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Size_MatType, Tensor_add_tensor,
            testing::Combine(testing::Values(szVGA, sz1080p), testing::Values(CV_8UC3, CV_32FC1)))
{
    const Size sz = get<0>(GetParam());
    const int type = get<1>(GetParam());
    Mat a(sz, type), b(sz, type), dst(sz, type);
    declare.in(a, b, WARMUP_RNG).out(dst);
    const Tensor ta(a), tb(b);
    Tensor tdst(dst);

    TEST_CYCLE() cv::add(ta, tb, tdst);

    SANITY_CHECK_NOTHING();
}

} // namespace
