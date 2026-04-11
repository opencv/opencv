// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test { namespace {

CV_ENUM(RetrMode, RETR_EXTERNAL, RETR_LIST, RETR_CCOMP, RETR_TREE)
CV_ENUM(ApproxMode, CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE, CHAIN_APPROX_TC89_L1, CHAIN_APPROX_TC89_KCOS)

typedef TestBaseWithParam< tuple<Size, RetrMode, ApproxMode, int> > TestFindContours;

PERF_TEST_P(TestFindContours, findContours,
            Combine(
               Values( szVGA, sz1080p ), // image size
               RetrMode::all(), // retrieval mode
               ApproxMode::all(), // approximation method
               Values( 32, 128 ) // blob count
            )
           )
{
    Size img_size = get<0>(GetParam());
    int retr_mode = get<1>(GetParam());
    int approx_method = get<2>(GetParam());
    int blob_count = get<3>(GetParam());

    RNG rng;
    Mat img = Mat::zeros(img_size, CV_8UC1);
    for(int i = 0; i < blob_count; i++ )
    {
        Point center;
        center.x = (unsigned)rng % (img.cols-2);
        center.y = (unsigned)rng % (img.rows-2);
        Size  axes;
        axes.width = ((unsigned)rng % 49 + 2)/2;
        axes.height = ((unsigned)rng % 49 + 2)/2;
        double angle = (unsigned)rng % 180;
        int brightness = (unsigned)rng % 2;

        // keep the border clear
        ellipse( img(Rect(1,1,img.cols-2,img.rows-2)), Point(center), Size(axes), angle, 0., 360., Scalar(brightness), -1);
    }
    vector< vector<Point> > contours;

    TEST_CYCLE() findContours( img, contours, retr_mode, approx_method );

    SANITY_CHECK_NOTHING();
}

typedef TestBaseWithParam< tuple<Size, ApproxMode, int> > TestFindContoursFF;

PERF_TEST_P(TestFindContoursFF, findContours,
    Combine(
        Values(szVGA, sz1080p), // image size
        ApproxMode::all(), // approximation method
        Values(32, 128) // blob count
    )
)
{
    Size img_size = get<0>(GetParam());
    int approx_method = get<1>(GetParam());
    int blob_count = get<2>(GetParam());

    RNG rng;
    Mat img = Mat::zeros(img_size, CV_32SC1);
    for (int i = 0; i < blob_count; i++)
    {
        Point center;
        center.x = (unsigned)rng % (img.cols - 2);
        center.y = (unsigned)rng % (img.rows - 2);
        Size  axes;
        axes.width = ((unsigned)rng % 49 + 2) / 2;
        axes.height = ((unsigned)rng % 49 + 2) / 2;
        double angle = (unsigned)rng % 180;
        int brightness = (unsigned)rng % 2;

        // keep the border clear
        ellipse(img(Rect(1, 1, img.cols - 2, img.rows - 2)), Point(center), Size(axes), angle, 0., 360., Scalar(brightness), -1);
    }
    vector< vector<Point> > contours;

    TEST_CYCLE() findContours(img, contours, RETR_FLOODFILL, approx_method);

    SANITY_CHECK_NOTHING();
}

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

// ============================================================
// findTRUContours performance tests
// ============================================================

typedef TestBaseWithParam< tuple<Size, int, int> > TestFindTRUContours;

PERF_TEST_P(TestFindTRUContours, findTRUContours,
    Combine(
        Values(sz1080p, sz2160p),   // image size
        Values(128, 512, 2048),     // circle count
        Values(1, 0)                // nthreads: 1=single-thread baseline, 0=all available
    )
)
{
    Size img_size  = get<0>(GetParam());
    int num_circles = get<1>(GetParam());
    int nthreads   = get<2>(GetParam());

    RNG rng(12345);
    Mat img = Mat::zeros(img_size, CV_8UC1);
    for (int i = 0; i < num_circles; ++i)
    {
        Point center(rng.uniform(50, img_size.width  - 50),
                     rng.uniform(50, img_size.height - 50));
        int radius = rng.uniform(10, 200);
        circle(img, center, radius, Scalar::all(255), FILLED);
    }

    Mat binary;
    adaptiveThreshold(img, binary, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 11, 0);

    vector<vector<Point>> contours;

    TEST_CYCLE() findTRUContours(binary, contours, 0, nthreads);

    SANITY_CHECK_NOTHING();
}

// Baseline: same image, findContours(RETR_LIST, CHAIN_APPROX_NONE) for direct comparison
typedef TestBaseWithParam< tuple<Size, int> > TestFindContoursBaseline;

PERF_TEST_P(TestFindContoursBaseline, findContours_baseline_for_TRUCO,
    Combine(
        Values(sz1080p, sz2160p),
        Values(128, 512, 2048)
    )
)
{
    Size img_size   = get<0>(GetParam());
    int num_circles = get<1>(GetParam());

    RNG rng(12345);
    Mat img = Mat::zeros(img_size, CV_8UC1);
    for (int i = 0; i < num_circles; ++i)
    {
        Point center(rng.uniform(50, img_size.width  - 50),
                     rng.uniform(50, img_size.height - 50));
        int radius = rng.uniform(10, 200);
        circle(img, center, radius, Scalar::all(255), FILLED);
    }

    Mat binary;
    adaptiveThreshold(img, binary, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 11, 0);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    TEST_CYCLE() findContours(binary, contours, hierarchy, RETR_LIST, CHAIN_APPROX_NONE);

    SANITY_CHECK_NOTHING();
}

} } // namespace
