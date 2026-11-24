// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

// --------------------------------------------------------------------------------------
// Independent brute-force reference implementation.
//
// This purposefully does NOT reuse any of the internal machinery in
// modules/imgproc/src/lambda_connected.cpp: it recomputes the bottleneck/widest-path
// distance D(seed, p) = min over all paths of the max pairwise distance along the path by
// repeatedly relaxing every pixel against its neighbors until nothing changes anymore
// (an O(V*E) fixed-point iteration, correct but far too slow for production use -- it
// exists purely so the tests below have an independent ground truth to compare the real
// bucket-queue implementation against).
// --------------------------------------------------------------------------------------

static inline double distGrayRef(const Mat& img, Point p, Point q)
{
    return std::abs((int)img.at<uchar>(p) - (int)img.at<uchar>(q));
}

static inline double distColorRef(const Mat& img, Point p, Point q)
{
    Vec3b a = img.at<Vec3b>(p), b = img.at<Vec3b>(q);
    double d0 = (double)a[0] - b[0], d1 = (double)a[1] - b[1], d2 = (double)a[2] - b[2];
    return std::sqrt(d0 * d0 + d1 * d1 + d2 * d2);
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

template<typename DistFn>
static Mat bruteForceBottleneck(const Mat& img, Point seed, int connectivity, DistFn distFn)
{
    const int rows = img.rows, cols = img.cols;
    const double INF = 1e18;
    Mat D(rows, cols, CV_64F, Scalar(INF));
    D.at<double>(seed) = 0.0;

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
                double best = D.at<double>(p);
                neighborsOf(p, rows, cols, connectivity, nbrs);
                for (size_t i = 0; i < nbrs.size(); i++)
                {
                    double cand = std::max(D.at<double>(nbrs[i]), distFn(img, nbrs[i], p));
                    if (cand < best)
                        best = cand;
                }
                if (best < D.at<double>(p) - 1e-9)
                {
                    D.at<double>(p) = best;
                    changed = true;
                }
            }
    }
    return D;
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

        Mat refD = bruteForceBottleneck(img, seed, 4, distGrayRef);

        Mat map;
        lambdaConnected(img, map, seed, 4);

        ASSERT_EQ(map.type(), CV_32F);
        ASSERT_EQ(map.size(), img.size());
        double maxErr = 0;
        for (int y = 0; y < img.rows; y++)
            for (int x = 0; x < img.cols; x++)
                maxErr = std::max(maxErr, std::abs((double)map.at<float>(y, x) - refD.at<double>(y, x)));
        // grayscale distances are exact non-negative integers on both sides
        EXPECT_LT(maxErr, 1e-6) << "trial " << trial << " connectivity=4";
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

        Mat refD = bruteForceBottleneck(img, seed, 8, distGrayRef);

        Mat map;
        lambdaConnected(img, map, seed, 8);

        double maxErr = 0;
        for (int y = 0; y < img.rows; y++)
            for (int x = 0; x < img.cols; x++)
                maxErr = std::max(maxErr, std::abs((double)map.at<float>(y, x) - refD.at<double>(y, x)));
        EXPECT_LT(maxErr, 1e-6) << "trial " << trial << " connectivity=8";
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

        Mat refD = bruteForceBottleneck(img, seed, 4, distColorRef);

        Mat map;
        lambdaConnected(img, map, seed, 4);

        ASSERT_EQ(map.type(), CV_32F);
        double maxErr = 0;
        for (int y = 0; y < img.rows; y++)
            for (int x = 0; x < img.cols; x++)
                maxErr = std::max(maxErr, std::abs((double)map.at<float>(y, x) - refD.at<double>(y, x)));
        // color path goes through one sqrt of an integer squared-distance key per pixel,
        // so allow a small float-rounding tolerance
        EXPECT_LT(maxErr, 1e-3) << "trial " << trial;
    }
}

// --------------------------------------------------------------------------------------
// Larger stress check: a bigger grid means many more pixels get pushed into the bucket
// queue more than once at a strictly decreasing sequence of keys before their best entry
// is popped (the scenario the QNode free-list / lazy-deletion logic exists to handle), and
// 8-connectivity on a larger grid exercises deeper bucket-queue traffic than the small
// grids above. Still cross-checked bit-for-bit (gray) / near-exactly (color) against the
// same independent brute-force reference.
// --------------------------------------------------------------------------------------

TEST(Imgproc_LambdaConnected, matchesBruteForceReference_largerGrid)
{
    RNG& rng = theRNG();
    for (int trial = 0; trial < 4; trial++)
    {
        Mat imgGray(25, 23, CV_8UC1);
        rng.fill(imgGray, RNG::UNIFORM, 0, 256);
        Point seedGray(rng.uniform(0, imgGray.cols), rng.uniform(0, imgGray.rows));

        Mat refGray = bruteForceBottleneck(imgGray, seedGray, 8, distGrayRef);
        Mat mapGray;
        lambdaConnected(imgGray, mapGray, seedGray, 8);

        double maxErrGray = 0;
        for (int y = 0; y < imgGray.rows; y++)
            for (int x = 0; x < imgGray.cols; x++)
                maxErrGray = std::max(maxErrGray, std::abs((double)mapGray.at<float>(y, x) - refGray.at<double>(y, x)));
        EXPECT_LT(maxErrGray, 1e-6) << "gray trial " << trial;

        Mat imgColor(20, 18, CV_8UC3);
        rng.fill(imgColor, RNG::UNIFORM, 0, 256);
        Point seedColor(rng.uniform(0, imgColor.cols), rng.uniform(0, imgColor.rows));

        Mat refColor = bruteForceBottleneck(imgColor, seedColor, 8, distColorRef);
        Mat mapColor;
        lambdaConnected(imgColor, mapColor, seedColor, 8);

        double maxErrColor = 0;
        for (int y = 0; y < imgColor.rows; y++)
            for (int x = 0; x < imgColor.cols; x++)
                maxErrColor = std::max(maxErrColor, std::abs((double)mapColor.at<float>(y, x) - refColor.at<double>(y, x)));
        EXPECT_LT(maxErrColor, 1e-3) << "color trial " << trial;
    }
}

// --------------------------------------------------------------------------------------
// Basic shape/range sanity: seed distance is 0, values are non-negative and bounded by
// the per-type maximum possible pairwise distance.
// --------------------------------------------------------------------------------------

TEST(Imgproc_LambdaConnected, mapRangeAndSeedValue)
{
    RNG& rng = theRNG();
    Mat img(40, 40, CV_8UC1);
    rng.fill(img, RNG::UNIFORM, 0, 256);
    Point seed(20, 20);

    Mat map;
    lambdaConnected(img, map, seed, 8);

    ASSERT_EQ(map.type(), CV_32F);
    ASSERT_EQ(map.size(), img.size());
    EXPECT_FLOAT_EQ(map.at<float>(seed), 0.0f);

    double minVal, maxVal;
    minMaxLoc(map, &minVal, &maxVal);
    EXPECT_GE(minVal, 0.0);
    EXPECT_LE(maxVal, 255.0 + 1e-3);

    Mat imgColor(30, 30, CV_8UC3);
    rng.fill(imgColor, RNG::UNIFORM, 0, 256);
    Mat mapColor;
    lambdaConnected(imgColor, mapColor, Point(15, 15), 4);
    minMaxLoc(mapColor, &minVal, &maxVal);
    EXPECT_GE(minVal, 0.0);
    EXPECT_LE(maxVal, 255.0 * std::sqrt(3.0) + 1e-2);
}

// --------------------------------------------------------------------------------------
// mask-at-lambda == compare(connectivityMap, maxDist*(1-lambda), CMP_LE), the externally
// composed equivalent of the old built-in thresholding this PR removed.
// --------------------------------------------------------------------------------------

TEST(Imgproc_LambdaConnected, maskViaCompareMatchesManualThreshold)
{
    RNG& rng = theRNG();
    Mat img(40, 40, CV_8UC1);
    rng.fill(img, RNG::UNIFORM, 0, 256);
    Point seed(20, 20);
    const double maxDist = 60.0, lambda = 0.6;
    const double tol = maxDist * (1.0 - lambda);

    Mat map;
    lambdaConnected(img, map, seed, 8);

    Mat mask;
    cv::compare(map, tol, mask, CMP_LE);

    ASSERT_EQ(mask.type(), CV_8UC1);
    ASSERT_EQ(mask.size(), img.size());
    for (int y = 0; y < img.rows; y++)
        for (int x = 0; x < img.cols; x++)
        {
            uchar expected = (map.at<float>(y, x) <= (float)tol) ? 255 : 0;
            ASSERT_EQ((int)mask.at<uchar>(y, x), (int)expected) << "at (" << x << "," << y << ")";
        }

    // Seed pixel is always at distance 0, so always included at any lambda > 0.
    EXPECT_EQ((int)mask.at<uchar>(seed), 255);
}

// --------------------------------------------------------------------------------------
// lambda = 1 degenerates to requiring an exact-match path at every step, i.e. the
// region is exactly the flood-filled connected component of pixels with the identical
// value as the seed (zero-tolerance flood fill). In distance terms, this is simply
// thresholding the map at distance 0.
// --------------------------------------------------------------------------------------

TEST(Imgproc_LambdaConnected, zeroToleranceMatchesExactFloodFill)
{
    Mat img(30, 30, CV_8UC1, Scalar(50));
    // an island of a genuinely different value: with a zero-tolerance (exact-match-only
    // paths), it must never be reachable from the seed's 50-valued plateau
    rectangle(img, Rect(20, 20, 5, 5), Scalar(120), FILLED);
    circle(img, Point(10, 10), 3, Scalar(51), FILLED); // single-value bump breaks exactness there

    Point seed(7, 7);

    Mat map;
    lambdaConnected(img, map, seed, 4);
    Mat mask;
    cv::compare(map, 0.0, mask, CMP_LE); // tol = maxDist*(1 - 1.0) = 0

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
    const double maxDist = 255.0, lambda = 0.9;
    const double tol = maxDist * (1.0 - lambda);

    Mat map4, map8, mask4, mask8;
    lambdaConnected(img, map4, seed, 4);
    lambdaConnected(img, map8, seed, 8);
    cv::compare(map4, tol, mask4, CMP_LE);
    cv::compare(map8, tol, mask8, CMP_LE);

    int area4 = countNonZero(mask4);
    int area8 = countNonZero(mask8);
    EXPECT_EQ(area4, 1) << "4-connectivity must not jump the diagonal gap";
    EXPECT_EQ(area8, 8) << "8-connectivity should follow the whole diagonal staircase";
}

// --------------------------------------------------------------------------------------
// At any single fixed tolerance, lambdaConnected's distance map, thresholded with
// compare(), must agree exactly with plain BFS/flood-fill region growing using the same
// per-edge test (bottleneck/widest-path equivalence -- see the Doxygen note on
// cv::lambdaConnected for the full discussion). We verify this by building a
// flood-fill-equivalent per-edge test with floodFill's floating-range mode and the same
// tolerance.
// --------------------------------------------------------------------------------------

TEST(Imgproc_LambdaConnected, singleToleranceMatchesFloodFillFloatingRange)
{
    // Smooth 1-D ramp: 4-connected, so the only edges that matter are horizontal.
    Mat img(5, 50, CV_8UC1);
    for (int y = 0; y < img.rows; y++)
        for (int x = 0; x < img.cols; x++)
            img.at<uchar>(y, x) = saturate_cast<uchar>(x * 4); // step of 4 per pixel

    Point seed(0, 2);
    const double maxDist = 40.0, lambda = 0.8; // degree(step=4) = 1 - 4/40 = 0.9 >= 0.8 -> every step passes
    const double tol = maxDist * (1.0 - lambda);

    Mat map, mask;
    lambdaConnected(img, map, seed, 4);
    cv::compare(map, tol, mask, CMP_LE);

    Mat ffImg = img.clone();
    Mat ffMask;
    floodFill(ffImg, ffMask, seed, Scalar(1), 0, Scalar(tol), Scalar(tol), 4 | FLOODFILL_MASK_ONLY | (255 << 8));
    Mat ffMaskInner = ffMask(Rect(1, 1, img.cols, img.rows));

    for (int y = 0; y < img.rows; y++)
        for (int x = 0; x < img.cols; x++)
            ASSERT_EQ((int)mask.at<uchar>(y, x), (int)ffMaskInner.at<uchar>(y, x))
                << "mismatch at (" << x << "," << y << ") -- bottleneck/widest-path "
                << "equivalence with floodFill's floating-range mode should hold exactly";

    // and since every single 4-unit step clears the tolerance, the whole ramp should be
    // included -- demonstrating the classical gradient-leakage behaviour that a single
    // fixed tolerance cannot avoid any better than floodFill can (see the PR discussion).
    EXPECT_EQ(countNonZero(mask), img.rows * img.cols);
}

// --------------------------------------------------------------------------------------
// Gradient-leakage demonstration: an object with smooth internal shading, bounded by a
// genuine discontinuity (not a seamless ramp) from an unrelated background. A single
// lambdaConnected computation's connectivityMap exposes the full recall-vs-tolerance
// curve (every tolerance row below reuses the ONE map that was already computed), which
// is the practical "compute once, threshold many" benefit over floodFill.
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

    Mat map;
    lambdaConnected(img, map, center, 4);

    const double maxDist = 25.0;
    // one computation (`map`) suffices to evaluate every tolerance below -- no re-run
    int objArea = countNonZero(objMask);
    double recallLow = (double)countNonZero((map <= (float)(maxDist * (1.0 - 0.3))) & (objMask > 0)) / objArea;
    double recallMid = (double)countNonZero((map <= (float)(maxDist * (1.0 - 0.7))) & (objMask > 0)) / objArea;
    double recallHigh = (double)countNonZero((map <= (float)(maxDist * (1.0 - 0.95))) & (objMask > 0)) / objArea;

    EXPECT_GE(recallLow, 0.99);
    EXPECT_GE(recallMid, 0.99);
    EXPECT_LT(recallHigh, 0.5) << "a near-zero tolerance should NOT recover the whole shaded disc";

    // the hard rim (object to background jump ~ >= 40 vs maxDist=25) must block any
    // leakage into the background at any reasonable tolerance > 0
    EXPECT_GT(map.at<float>(0, 0), (float)maxDist);
    EXPECT_GT(map.at<float>(n - 1, n - 1), (float)maxDist);
}

// --------------------------------------------------------------------------------------
// Error handling
// --------------------------------------------------------------------------------------

TEST(Imgproc_LambdaConnected, rejectsInvalidConnectivity)
{
    Mat img(10, 10, CV_8UC1, Scalar(0));
    Mat map;
    EXPECT_THROW(lambdaConnected(img, map, Point(0, 0), 5), cv::Exception);
}

TEST(Imgproc_LambdaConnected, rejectsSeedOutsideImage)
{
    Mat img(10, 10, CV_8UC1, Scalar(0));
    Mat map;
    EXPECT_THROW(lambdaConnected(img, map, Point(100, 100), 4), cv::Exception);
    EXPECT_THROW(lambdaConnected(img, map, Point(-1, 0), 4), cv::Exception);
}

TEST(Imgproc_LambdaConnected, rejectsUnsupportedType)
{
    Mat img32f(10, 10, CV_32FC1, Scalar(0));
    Mat img8u4c(10, 10, CV_8UC4, Scalar(0));
    Mat map;
    EXPECT_THROW(lambdaConnected(img32f, map, Point(0, 0), 4), cv::Exception);
    EXPECT_THROW(lambdaConnected(img8u4c, map, Point(0, 0), 4), cv::Exception);
}

}} // namespace
