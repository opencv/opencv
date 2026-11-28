// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test { namespace {

enum { CCL_CLUSTERED, CCL_SCATTERED, CCL_DENSE, CCL_EMPTY };
CV_ENUM(ConnectedComponentsMask, CCL_CLUSTERED, CCL_SCATTERED, CCL_DENSE, CCL_EMPTY)

typedef TestBaseWithParam<tuple<Size, ConnectedComponentsMask> > ConnectedComponents;

static Mat makeConnectedComponentsMask(Size size, int pattern)
{
    Mat mask = Mat::zeros(size, CV_8U);
    RNG rng(56051);
    if (pattern == CCL_CLUSTERED)
    {
        // One 12x4 rectangle per 40x40 tile gives 3% foreground in long runs.
        for (int y = 0; y + 40 <= size.height; y += 40)
        {
            for (int x = 0; x + 40 <= size.width; x += 40)
            {
                const int left = x + rng.uniform(0, 29);
                const int top = y + rng.uniform(0, 37);
                mask(Rect(left, top, 12, 4)).setTo(255);
            }
        }
    }
    else if (pattern != CCL_EMPTY)
    {
        const unsigned threshold = pattern == CCL_SCATTERED ? 7 : 128;
        for (int y = 0; y < size.height; ++y)
        {
            uchar* row = mask.ptr<uchar>(y);
            for (int x = 0; x < size.width; ++x)
                row[x] = (rng.next() >> 24) < threshold ? 255 : 0;
        }
    }
    return mask;
}

PERF_TEST_P(ConnectedComponents, connectedComponents_8u,
            Combine(Values(szVGA, sz1080p), ConnectedComponentsMask::all()))
{
    const Mat mask = makeConnectedComponentsMask(get<0>(GetParam()), get<1>(GetParam()));
    Mat labels;

    TEST_CYCLE() connectedComponents(mask, labels, 8, CV_32S, CCL_DEFAULT);

    EXPECT_EQ(countNonZero(mask), countNonZero(labels));
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(ConnectedComponents, connectedComponents_8u_with_stats,
            Combine(Values(szVGA, sz1080p), ConnectedComponentsMask::all()))
{
    const Mat mask = makeConnectedComponentsMask(get<0>(GetParam()), get<1>(GetParam()));
    Mat labels, stats, centroids;
    int nLabels = 0;

    TEST_CYCLE() nLabels = connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S, CCL_DEFAULT);

    ASSERT_GT(nLabels, 0);
    ASSERT_EQ(nLabels, stats.rows);
    EXPECT_EQ(nLabels, centroids.rows);
    const int foreground = countNonZero(mask);
    EXPECT_EQ(foreground, countNonZero(labels));
    EXPECT_EQ(static_cast<int>(mask.total()) - foreground, stats.at<int>(0, CC_STAT_AREA));
    SANITY_CHECK_NOTHING();
}

}} // namespace
