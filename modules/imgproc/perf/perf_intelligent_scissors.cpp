// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test { namespace {


typedef perf::TestBaseWithParam<int> TestIntelligentScissorsMB;

PERF_TEST_P(TestIntelligentScissorsMB, buildMap, testing::Values( IMREAD_GRAYSCALE, IMREAD_COLOR ))
{
    const int flags = GetParam();

    const Mat image = imread(samples::findFile("HappyFish.jpg"), flags);
    ASSERT_TRUE(!image.empty());

    const Point source_point(140, 20);

    segmentation::IntelligentScissorsMB tool;
    tool.applyImage(image);

    PERF_SAMPLE_BEGIN()
    for (size_t idx = 1; idx <= 100; ++idx)
    {
        tool.buildMap(source_point);
    }
    PERF_SAMPLE_END()

    SANITY_CHECK_NOTHING();
}


}} // namespace
