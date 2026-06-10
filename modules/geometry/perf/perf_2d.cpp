// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"
#include "opencv2/ts.hpp"
#include "opencv2/ts/ts_perf.hpp"

namespace opencv_test { namespace {

using namespace perf;

typedef TestBaseWithParam< tuple<MatDepth, int> > TestBoundingRect;

PERF_TEST_P(TestBoundingRect, BoundingRect,
            Combine(
                testing::Values(CV_32S, CV_32F), // points type
                    Values(400, 511, 1000, 10000, 100000) // points count
            )
)

{
    int ptType = get<0>(GetParam());
    int n = get<1>(GetParam());

    Mat pts(n, 2, ptType);
    declare.in(pts, WARMUP_RNG);

    cv::Rect rect;
    TEST_CYCLE() rect = boundingRect(pts);

    SANITY_CHECK_NOTHING();
}

typedef TestBaseWithParam< tuple<MatDepth, int> > TestMinEnclosingCircle;
PERF_TEST_P(TestMinEnclosingCircle, minEnclosingCircle,
            Combine(
                testing::Values(CV_32S, CV_32F),
                    Values(400, 1000, 10000, 100000)
            ))
{
    int ptType = get<0>(GetParam());
    int n = get<1>(GetParam());
    Mat pts(n, 2, ptType);
    declare.in(pts, WARMUP_RNG);

    Point2f center;
    float radius;
    TEST_CYCLE() minEnclosingCircle(pts, center, radius);
    SANITY_CHECK_NOTHING();
}

typedef TestBaseWithParam<int> TestMinEnclosingCircleWorstCase;
PERF_TEST_P(TestMinEnclosingCircleWorstCase, minEnclosingCircle_sequential,
            Values(400, 1000, 5000, 10000))
{
    int n = GetParam();
    vector<Point2f> contour;
    for(int i = 0; i < n; ++i) {
        float angle = (float)(i * 2 * CV_PI / n);
        contour.push_back(Point2f(cos(angle) * 100, sin(angle) * 100));
    }

    Point2f center;
    float radius;
    TEST_CYCLE() minEnclosingCircle(contour, center, radius);
    SANITY_CHECK_NOTHING();
}

}} // namespace
