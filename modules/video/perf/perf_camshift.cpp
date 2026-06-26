// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, Advanced Micro Devices, Inc., all rights reserved.

#include "perf_precomp.hpp"
#include "opencv2/video/tracking.hpp"

namespace opencv_test {

typedef perf::TestBaseWithParam< tuple<Size, int> > MeanShiftFixture;
typedef perf::TestBaseWithParam< tuple<Size, int> > CamShiftFixture;

static void initProbImage(const Size& size, Mat& prob, Rect& window)
{
    prob.create(size, CV_32FC1);
    prob.setTo(0);

    const Point center(size.width / 2, size.height / 2);
    const int radius = std::max(8, std::min(size.width, size.height) / 8);
    circle(prob, center, radius, Scalar(1.f), -1);

    const int winSize = std::max(16, std::min(size.width, size.height) / 4);
    window = Rect(center.x - winSize / 2, center.y - winSize / 2, winSize, winSize);
    window &= Rect(0, 0, size.width, size.height);
}

PERF_TEST_P(MeanShiftFixture, meanShift,
            testing::Combine(
                testing::Values(szVGA, sz720p, sz1080p),
                testing::Values(5, 10, 20)))
{
    const Size size = get<0>(GetParam());
    const int maxCount = get<1>(GetParam());

    Mat prob;
    Rect window, initWindow;
    initProbImage(size, prob, initWindow);

    declare.in(prob);

    const TermCriteria criteria(TermCriteria::MAX_ITER | TermCriteria::EPS, maxCount, 1.0);
    int iters = 0;

    TEST_CYCLE()
    {
        window = initWindow;
        iters = cv::meanShift(prob, window, criteria);
    }

    SANITY_CHECK(iters);
}

PERF_TEST_P(CamShiftFixture, CamShift,
            testing::Combine(
                testing::Values(szVGA, sz720p, sz1080p),
                testing::Values(5, 10, 20)))
{
    const Size size = get<0>(GetParam());
    const int maxCount = get<1>(GetParam());

    Mat prob;
    Rect window, initWindow;
    initProbImage(size, prob, initWindow);

    declare.in(prob);

    const TermCriteria criteria(TermCriteria::MAX_ITER | TermCriteria::EPS, maxCount, 1.0);
    RotatedRect box;

    TEST_CYCLE()
    {
        window = initWindow;
        box = cv::CamShift(prob, window, criteria);
    }

    const float boxWidth = box.size.width;
    const float boxHeight = box.size.height;
    SANITY_CHECK(boxWidth);
    SANITY_CHECK(boxHeight);
}

} // namespace opencv_test
