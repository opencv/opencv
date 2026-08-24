// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#include "perf_precomp.hpp"

namespace opencv_test {

typedef tuple<Size, int> Size_BgChannels_t;
typedef perf::TestBaseWithParam<Size_BgChannels_t> Size_BgChannels;

PERF_TEST_P(Size_BgChannels, alphaComposite,
            testing::Combine(
                testing::Values(szVGA, sz720p, sz1080p),
                testing::Values(3, 4)
                )
            )
{
    Size size = get<0>(GetParam());
    int bgChannels = get<1>(GetParam());

    Mat overlay(size, CV_8UC4);
    Mat background(size, CV_8UC(bgChannels));
    Mat dst(size, CV_8UC(bgChannels));

    declare.in(overlay, background, WARMUP_RNG).out(dst);

    TEST_CYCLE() alphaComposite(overlay, background, dst);

    SANITY_CHECK_NOTHING();
}

} // namespace opencv_test
