// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <queue>

namespace opencv_test { namespace {

// --------------------------------------------------------------------------------------
// Independent brute-force reference implementation.
//
// This purposefully does NOT reuse any of the internal machinery in
// modules/imgproc/src/lambda_connected.cpp: it recomputes the bottleneck-connectivity
// value B(seed, p) = max over all paths of the min pairwise degree along the path by
// repeatedly relaxing every pixel against its neighbors until nothing changes anymore
// (an O(V*E) fixed-point iteration, correct but far too slow for production use -- it
// exists purely so the tests below have an independent ground truth to compare the real
// O(N log N) heap-based implementation against).
// --------------------------------------------------------------------------------------

static inline float degreeGrayRef(const Mat& img, Point p, Point q, double maxDist)
{
    int d = std::abs((int)img.at<uchar>(p) - (int)img.at<uchar>(q));
    double deg = 1.0 - d / maxDist;
    return deg > 0.0 ? (float)deg : 0.f;
}

static inline float degreeColorRef(const Mat& img, Point p, Point q, double maxDist)
{
    Vec3b a = img.at<Vec3b>(p), b = img.at<Vec3b>(q);
    double d0 = (double)a[0] - b[0], d1 = (double)a[1] - b[1], d2 = (double)a[2] - b[2];
    double dist = std::sqrt(d0 * d0 + d1 * d1 + d2 * d2);
    double deg = 1.0 - dist / maxDist;
    return deg > 0.0 ? (float)deg : 0.f;
}

static void neighborsOf(Point p, int rows, int cols, int connectivity, std::vector<Point>& out)
{
    out.clear();
    static const int dx4[4] = { 1, -1, 0, 0 };
    static const int dy4[4] = { 0, 0, 1, -1 };
    static const int dx8[8] = { 1, -1, 0, 0, 1, 1, -1, -1 };
    static const int dy8[8] = { 0, 0, 1, -1, 1, -1, 1, -1 };
    const int* dx = connectivity == 8 ? dx8 : dx4;
    const int* dy = connectivity == 8 ? dy8 : dy4;
    for (int k = 0; k < connectivity; k++)
    {
        int nx = p.x + dx[k], ny = p.y + dy[k];
        if (nx >= 0 && nx < cols && ny >= 0 && ny < rows)
            out.push_back(Point(nx, ny));
    }
}

template<typename DegreeFn>
static Mat bruteForceBottleneck(const Mat& img, Point seed, int connectivity, double maxDist,
                                 DegreeFn degreeFn)
{
    const int rows = img.rows, cols = img.cols;
    Mat B = Mat::zeros(rows, cols, CV_32F);
    B.at<float>(seed) = 1.f;

    std::vector<Point> nbrs;
    bool changed = true;
    int guard = 0;
    while (changed)
    {
        changed = false;
        CV_Assert(guard++ < 50 * rows * cols && "brute-force reference failed to converge");
        for (int y = 0; y < rows; y++)
            for (int x = 0; x < cols; x++)
            {
                Point p(x, y);
                if (p == seed)
                    continue;
                float best = B.at<float>(p);
                neighborsOf(p, rows, cols, connectivity, nbrs);
                for (size_t i = 0; i < nbrs.size(); i++)
                {
                    float cand = std::min(B.at<float>(nbrs[i]), degreeFn(img, nbrs[i], p, maxDist));
                    if (cand > best)
                        best = cand;
                }
                if (best > B.at<float>(p) + 1e-6f)
                {
                    B.at<float>(p) = best;
                    changed = true;
                }
            }
    }
    return B;
}

// --------------------------------------------------------------------------------------
// Correctness vs. brute-force reference
// --------------------------------------------------------------------------------------

TEST(Imgproc_LambdaConnected, matchesBruteForceReference_gray_conn4)
{
    RNG& rng = theRNG();
    for (int trial = 0; trial < 8; trial++)
    {
        Mat img(9, 11, CV_8UC1);
        rng.fill(img, RNG::UNIFORM, 0, 256);
        Point seed(rng.uniform(0, img.cols), rng.uniform(0, img.rows));
        const double maxDist = 80.0;

        Mat refMap = bruteForceBottleneck(img, seed, 4, maxDist, degreeGrayRef);

        Mat mask, map;
        lambdaConnected(img, mask, seed, 0.5, 4, maxDist, map);

        ASSERT_EQ(map.type(), CV_32F);
        ASSERT_EQ(map.size(), img.size());
        double maxErr = 0;
        for (int y = 0; y < img.rows; y++)
            for (int x = 0; x < img.cols; x++)
                maxErr = std::max(maxErr, (double)std::abs(map.at<float>(y, x) - refMap.at<float>(y, x)));
        EXPECT_LT(maxErr, 1e-4) << "trial " << trial << " connectivity=4";
    }
}

TEST(Imgproc_LambdaConnected, matchesBruteForceReference_gray_conn8)
{
    RNG& rng = theRNG();
    for (int trial = 0; trial < 8; trial++)
    {
        Mat img(9, 11, CV_8UC1);
        rng.fill(img, RNG::UNIFORM, 0, 256);
        Point seed(rng.uniform(0, img.cols), rng.uniform(0, img.rows));
        const double maxDist = 80.0;

        Mat refMap = bruteForceBottleneck(img, seed, 8, maxDist, degreeGrayRef);

        Mat mask, map;
        lambdaConnected(img, mask, seed, 0.5, 8, maxDist, map);

        double maxErr = 0;
        for (int y = 0; y < img.rows; y++)
            for (int x = 0; x < img.cols; x++)
                maxErr = std::max(maxErr, (double)std::abs(map.at<float>(y, x) - refMap.at<float>(y, x)));
        EXPECT_LT(maxErr, 1e-4) << "trial " << trial << " connectivity=8";
    }
}

TEST(Imgproc_LambdaConnected, matchesBruteForceReference_color)
{
    RNG& rng = theRNG();
    for (int trial = 0; trial < 5; trial++)
    {
        Mat img(8, 9, CV_8UC3);
        rng.fill(img, RNG::UNIFORM, 0, 256);
        Point seed(rng.uniform(0, img.cols), rng.uniform(0, img.rows));
        const double maxDist = 120.0;

        Mat refMap = bruteForceBottleneck(img, seed, 4, maxDist, degreeColorRef);

        Mat mask, map;
        lambdaConnected(img, mask, seed, 0.5, 4, maxDist, map);

        ASSERT_EQ(map.type(), CV_32F);
        double maxErr = 0;
        for (int y = 0; y < img.rows; y++)
            for (int x = 0; x < img.cols; x++)
                maxErr = std::max(maxErr, (double)std::abs(map.at<float>(y, x) - refMap.at<float>(y, x)));
        EXPECT_LT(maxErr, 1e-3) << "trial " << trial;
    }
}

// --------------------------------------------------------------------------------------
// mask == threshold(connectivityMap, lambda), and mask convention is 0/255
// --------------------------------------------------------------------------------------

TEST(Imgproc_LambdaConnected, maskMatchesThresholdedMap)
{
    RNG& rng = theRNG();
    Mat img(40, 40, CV_8UC1);
    rng.fill(img, RNG::UNIFORM, 0, 256);
    Point seed(20, 20);

    Mat mask, map;
    lambdaConnected(img, mask, seed, 0.6, 8, 60.0, map);

    ASSERT_EQ(mask.type(), CV_8UC1);
    ASSERT_EQ(mask.size(), img.size());
    for (int y = 0; y < img.rows; y++)
        for (int x = 0; x < img.cols; x++)
        {
            uchar expected = (map.at<float>(y, x) >= 0.6f) ? 255 : 0;
            ASSERT_EQ((int)mask.at<uchar>(y, x), (int)expected) << "at (" << x << "," << y << ")";
        }

    // Seed pixel is always fully connected to itself.
    EXPECT_FLOAT_EQ(map.at<float>(seed), 1.0f);
    EXPECT_EQ((int)mask.at<uchar>(seed), 255);
}

// --------------------------------------------------------------------------------------
// lambda = 1 degenerates to requiring an exact-match path at every step, i.e. the
// region is exactly the flood-filled connected component of pixels with the identical
// value as the seed (zero-tolerance flood fill).
// --------------------------------------------------------------------------------------

TEST(Imgproc_LambdaConnected, lambdaOneMatchesExactFloodFill)
{
    Mat img(30, 30, CV_8UC1, Scalar(50));
    // an island of a genuinely different value: with lambda=1 (exact-match-only
    // paths), it must never be reachable from the seed's 50-valued plateau
    rectangle(img, Rect(20, 20, 5, 5), Scalar(120), FILLED);
    circle(img, Point(10, 10), 3, Scalar(51), FILLED); // single-value bump breaks exactness there

    Point seed(7, 7);

    Mat mask, map;
    lambdaConnected(img, mask, seed, 1.0, 4, 255.0, map);

    Mat ffMask;
    Mat imgCopy = img.clone();
    floodFill(imgCopy, ffMask, seed, Scalar(200), 0, Scalar(0), Scalar(0), 4 | FLOODFILL_MASK_ONLY | (255 << 8));
    Mat ffMaskInner = ffMask(Rect(1, 1, img.cols, img.rows));

    ASSERT_EQ(mask.size(), ffMaskInner.size());
    for (int y = 0; y < img.rows; y++)
        for (int x = 0; x < img.cols; x++)
            ASSERT_EQ((int)mask.at<uchar>(y, x), (int)ffMaskInner.at<uchar>(y, x))
                << "mismatch at (" << x << "," << y << ")";

    // sanity: the disconnected same-valued island must NOT be included
    EXPECT_EQ((int)mask.at<uchar>(22, 22), 0);
}

// --------------------------------------------------------------------------------------
// connectivity=4 vs connectivity=8 give different results on a diagonal-only path
// --------------------------------------------------------------------------------------

TEST(Imgproc_LambdaConnected, connectivity4Vs8Differ)
{
    // Build a diagonal staircase of identical bright pixels on a dark background: pixels
    // only touch corner-to-corner, never edge-to-edge, so 8-connectivity reaches along
    // the whole staircase while 4-connectivity cannot leave the seed pixel's component.
    Mat img(10, 10, CV_8UC1, Scalar(10));
    for (int i = 0; i < 8; i++)
        img.at<uchar>(i, i) = 250;

    Point seed(0, 0);
    Mat mask4, mask8;
    lambdaConnected(img, mask4, seed, 0.9, 4, 255.0);
    lambdaConnected(img, mask8, seed, 0.9, 8, 255.0);

    int area4 = countNonZero(mask4);
    int area8 = countNonZero(mask8);
    EXPECT_EQ(area4, 1) << "4-connectivity must not jump the diagonal gap";
    EXPECT_EQ(area8, 8) << "8-connectivity should follow the whole diagonal staircase";
}

// --------------------------------------------------------------------------------------
// At any single fixed lambda, lambdaConnected's mask must agree exactly with plain
// BFS/flood-fill region growing using the same per-edge test (bottleneck/widest-path
// equivalence -- see the PR description for the full discussion). We verify this by
// building a flood-fill-equivalent per-edge test with floodFill's floating-range mode
// and an equivalent tolerance, derived from the same degree formula.
// --------------------------------------------------------------------------------------

TEST(Imgproc_LambdaConnected, singleLambdaMatchesFloodFillFloatingRange)
{
    // Smooth 1-D ramp: 4-connected, so the only edges that matter are horizontal.
    Mat img(5, 50, CV_8UC1);
    for (int y = 0; y < img.rows; y++)
        for (int x = 0; x < img.cols; x++)
            img.at<uchar>(y, x) = saturate_cast<uchar>(x * 4); // step of 4 per pixel

    Point seed(0, 2);
    const double maxDist = 40.0;
    const double lambda = 0.8; // degree(step=4) = 1 - 4/40 = 0.9 >= 0.8 -> every step passes

    Mat mask, map;
    lambdaConnected(img, mask, seed, lambda, 4, maxDist, map);

    // equivalent floodFill tolerance: degree >= lambda  <=>  |diff| <= maxDist*(1-lambda)
    double tol = maxDist * (1.0 - lambda);
    Mat ffImg = img.clone();
    Mat ffMask;
    floodFill(ffImg, ffMask, seed, Scalar(1), 0, Scalar(tol), Scalar(tol), 4 | FLOODFILL_MASK_ONLY | (255 << 8));
    Mat ffMaskInner = ffMask(Rect(1, 1, img.cols, img.rows));

    for (int y = 0; y < img.rows; y++)
        for (int x = 0; x < img.cols; x++)
            ASSERT_EQ((int)mask.at<uchar>(y, x), (int)ffMaskInner.at<uchar>(y, x))
                << "mismatch at (" << x << "," << y << ") -- bottleneck/widest-path "
                << "equivalence with floodFill's floating-range mode should hold exactly";

    // and since every single 4-unit step clears the lambda=0.8 bar, the whole ramp
    // should be included -- demonstrating the classical gradient-leakage behaviour that
    // a single fixed lambda cannot avoid any better than floodFill can (see PR notes).
    EXPECT_EQ(countNonZero(mask), img.rows * img.cols);
}

// --------------------------------------------------------------------------------------
// Gradient-leakage demonstration: an object with smooth internal shading, bounded by a
// genuine discontinuity (not a seamless ramp) from an unrelated background. A single
// lambdaConnected computation's connectivityMap exposes the full lambda-vs-recall curve
// (every lambda row below reuses the ONE map that was already computed), which is the
// practical "compute once, threshold many" benefit over floodFill highlighted in the PR.
// --------------------------------------------------------------------------------------

TEST(Imgproc_LambdaConnected, gradientLeakageObjectRecallCurve)
{
    const int n = 41;
    Mat img(n, n, CV_8UC1, Scalar(40));
    Point center(12, 20);
    const int radius = 8;
    Mat objMask = Mat::zeros(n, n, CV_8U);
    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++)
        {
            double dist = std::sqrt((double)(x - center.x) * (x - center.x) +
                                     (y - center.y) * (y - center.y));
            if (dist <= radius)
            {
                img.at<uchar>(y, x) = saturate_cast<uchar>(150 - 40 * (dist / radius));
                objMask.at<uchar>(y, x) = 255;
            }
        }

    Mat mask, map;
    lambdaConnected(img, mask, center, 0.5, 4, 25.0, map);

    // one computation (`map`) suffices to evaluate every lambda below -- no re-run
    int objArea = countNonZero(objMask);
    double recallLow = (double)countNonZero((map >= 0.3f) & (objMask > 0)) / objArea;
    double recallMid = (double)countNonZero((map >= 0.7f) & (objMask > 0)) / objArea;
    double recallHigh = (double)countNonZero((map >= 0.95f) & (objMask > 0)) / objArea;

    EXPECT_GE(recallLow, 0.99);
    EXPECT_GE(recallMid, 0.99);
    EXPECT_LT(recallHigh, 0.5) << "a near-1.0 lambda should NOT recover the whole shaded disc";

    // the hard rim (object to background jump ~ >= 40 vs maxDist=25) must block any
    // leakage into the background at any reasonable lambda > 0
    EXPECT_EQ((int)mask.at<uchar>(0, 0), 0);
    EXPECT_EQ((int)mask.at<uchar>(n - 1, n - 1), 0);
}

// --------------------------------------------------------------------------------------
// Error handling
// --------------------------------------------------------------------------------------

TEST(Imgproc_LambdaConnected, rejectsInvalidLambda)
{
    Mat img(10, 10, CV_8UC1, Scalar(0));
    Mat mask;
    EXPECT_THROW(lambdaConnected(img, mask, Point(0, 0), -0.1), cv::Exception);
    EXPECT_THROW(lambdaConnected(img, mask, Point(0, 0), 1.1), cv::Exception);
}

TEST(Imgproc_LambdaConnected, rejectsInvalidConnectivity)
{
    Mat img(10, 10, CV_8UC1, Scalar(0));
    Mat mask;
    EXPECT_THROW(lambdaConnected(img, mask, Point(0, 0), 0.5, 5), cv::Exception);
}

TEST(Imgproc_LambdaConnected, rejectsSeedOutsideImage)
{
    Mat img(10, 10, CV_8UC1, Scalar(0));
    Mat mask;
    EXPECT_THROW(lambdaConnected(img, mask, Point(100, 100), 0.5), cv::Exception);
    EXPECT_THROW(lambdaConnected(img, mask, Point(-1, 0), 0.5), cv::Exception);
}

TEST(Imgproc_LambdaConnected, rejectsUnsupportedType)
{
    Mat img32f(10, 10, CV_32FC1, Scalar(0));
    Mat img8u4c(10, 10, CV_8UC4, Scalar(0));
    Mat mask;
    EXPECT_THROW(lambdaConnected(img32f, mask, Point(0, 0), 0.5), cv::Exception);
    EXPECT_THROW(lambdaConnected(img8u4c, mask, Point(0, 0), 0.5), cv::Exception);
}

TEST(Imgproc_LambdaConnected, rejectsNonPositiveMaxDist)
{
    Mat img(10, 10, CV_8UC1, Scalar(0));
    Mat mask;
    EXPECT_THROW(lambdaConnected(img, mask, Point(0, 0), 0.5, 4, 0.0), cv::Exception);
    EXPECT_THROW(lambdaConnected(img, mask, Point(0, 0), 0.5, 4, -5.0), cv::Exception);
}

}} // namespace
