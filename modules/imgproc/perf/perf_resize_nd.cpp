// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "perf_precomp.hpp"

namespace opencv_test {

CV_ENUM(ResizeNDInter, INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_AREA)

typedef tuple<std::vector<int>, Size, ResizeNDInter> ResizeNDParams;
typedef TestBaseWithParam<ResizeNDParams> ResizeNDPerf;

// n-dimensional (NCHW) input: every HxW plane is resized
PERF_TEST_P_(ResizeNDPerf, resize)
{
    const std::vector<int> shape = get<0>(GetParam());
    const Size dsize = get<1>(GetParam());
    const int interp = get<2>(GetParam());
    Mat src(shape, CV_32F), dst;
    randu(Mat(1, (int)src.total(), CV_32F, src.data), 0, 1);   // declare.in() only handles 2D arrays
    cv::resize(src, dst, dsize, 0, 0, interp);

    TEST_CYCLE() cv::resize(src, dst, dsize, 0, 0, interp);

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, ResizeNDPerf, testing::Combine(
    testing::Values(std::vector<int>{1, 64, 80, 80}),
    testing::Values(Size(160, 160), Size(40, 40)),
    ResizeNDInter::all()
));

INSTANTIATE_TEST_CASE_P(Image, ResizeNDPerf, testing::Combine(
    testing::Values(std::vector<int>{1, 3, 224, 224}),
    testing::Values(Size(448, 448), Size(112, 112)),
    testing::Values(ResizeNDInter(INTER_LINEAR), ResizeNDInter(INTER_CUBIC))
));

} // namespace
