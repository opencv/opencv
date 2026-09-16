// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"

#include <limits>

namespace opencv_test { namespace {

const double INF_D = std::numeric_limits<double>::infinity();
const double NAN_D = std::numeric_limits<double>::quiet_NaN();

static bool isForbidden(double c, double thr)
{
    return cvIsNaN(c) || cvIsInf(c) || c > thr;
}

// Counts the pairs and sums their cost from an assignment vector.
static void summarise(const Mat& cost, const std::vector<int>& a, int& pairs, double& total)
{
    pairs = 0;
    total = 0.0;
    for (size_t i = 0; i < a.size(); i++)
    {
        if (a[i] < 0)
            continue;
        pairs++;
        total += cost.at<double>((int)i, a[i]);
    }
}

// True when (k1, c1) beats (k2, c2) under the padded objective. A threshold this large means the
// penalty already dominates every rearrangement of the real costs, so it degenerates to "as many
// pairs as possible, then cheapest".
static bool better(int k1, double c1, int k2, double c2, double thr, int cap)
{
    if (thr < 1e100)
        return c1 + thr * (cap - k1) < c2 + thr * (cap - k2) - 1e-9;
    if (k1 != k2)
        return k1 > k2;
    return c1 < c2 - 1e-9;
}

static void bruteRec(const Mat& cost, double thr, int row, std::vector<char>& usedCol,
                     int k, double acc, int& bestK, double& bestCost)
{
    const int M = cost.rows, N = cost.cols;
    if (row == M)
    {
        if (bestK < 0 || better(k, acc, bestK, bestCost, thr, std::min(M, N)))
        {
            bestK = k;
            bestCost = acc;
        }
        return;
    }
    bruteRec(cost, thr, row + 1, usedCol, k, acc, bestK, bestCost);   // leave row unmatched
    for (int j = 0; j < N; j++)
    {
        const double c = cost.at<double>(row, j);
        if (usedCol[j] || isForbidden(c, thr))
            continue;
        usedCol[j] = 1;
        bruteRec(cost, thr, row + 1, usedCol, k + 1, acc + c, bestK, bestCost);
        usedCol[j] = 0;
    }
}

static void bruteForce(const Mat& cost, double thr, int& pairs, double& total)
{
    std::vector<char> usedCol((size_t)cost.cols, 0);
    pairs = -1;
    total = 0.0;
    bruteRec(cost, thr, 0, usedCol, 0, 0.0, pairs, total);
}

// Every pair the solver reports must be legal and each column used at most once.
static void checkWellFormed(const Mat& cost, const std::vector<int>& a, double thr)
{
    ASSERT_EQ((size_t)cost.rows, a.size());
    std::vector<char> seen((size_t)cost.cols, 0);
    for (size_t i = 0; i < a.size(); i++)
    {
        if (a[i] < 0)
        {
            EXPECT_EQ(-1, a[i]);
            continue;
        }
        ASSERT_LT(a[i], cost.cols);
        EXPECT_FALSE(seen[a[i]]) << "column " << a[i] << " used twice";
        seen[a[i]] = 1;
        EXPECT_FALSE(isForbidden(cost.at<double>((int)i, a[i]), thr))
            << "forbidden pair (" << i << ", " << a[i] << ") was matched";
    }
}

TEST(Core_LinearAssignment, regression_basic)
{
    Mat cost = Mat_<double>({3, 3}, {4, 1, 3,
                                     2, 0, 5,
                                     3, 2, 2});
    std::vector<int> a;
    const double total = cv::linearAssignment(cost, a);

    EXPECT_NEAR(5.0, total, 1e-12);
    ASSERT_EQ(3u, a.size());
    EXPECT_EQ(1, a[0]);
    EXPECT_EQ(0, a[1]);
    EXPECT_EQ(2, a[2]);
}

TEST(Core_LinearAssignment, non_square_wide)
{
    Mat cost = Mat_<double>({2, 4}, {7, 1, 9, 8,
                                     6, 5, 2, 4});
    std::vector<int> a;
    const double total = cv::linearAssignment(cost, a);

    ASSERT_EQ(2u, a.size());
    EXPECT_GE(a[0], 0);
    EXPECT_GE(a[1], 0);
    EXPECT_NEAR(3.0, total, 1e-12);
}

TEST(Core_LinearAssignment, non_square_tall)
{
    Mat cost = Mat_<double>({4, 2}, {7, 6,
                                     1, 5,
                                     9, 2,
                                     8, 4});
    std::vector<int> a;
    const double total = cv::linearAssignment(cost, a);

    ASSERT_EQ(4u, a.size());
    int pairs;
    double sum;
    summarise(cost, a, pairs, sum);
    EXPECT_EQ(2, pairs) << "M > N must assign exactly N rows";
    EXPECT_NEAR(3.0, total, 1e-12);
}

TEST(Core_LinearAssignment, empty)
{
    std::vector<int> a;

    EXPECT_NEAR(0.0, cv::linearAssignment(Mat(0, 0, CV_64F), a), 1e-12);
    EXPECT_TRUE(a.empty());

    EXPECT_NEAR(0.0, cv::linearAssignment(Mat(0, 5, CV_64F), a), 1e-12);
    EXPECT_TRUE(a.empty());

    EXPECT_NEAR(0.0, cv::linearAssignment(Mat(5, 0, CV_64F), a), 1e-12);
    ASSERT_EQ(5u, a.size());
    for (size_t i = 0; i < a.size(); i++)
        EXPECT_EQ(-1, a[i]);
}

TEST(Core_LinearAssignment, all_infeasible)
{
    Mat cost = Mat_<double>({2, 3}, {INF_D, INF_D, INF_D,
                                     INF_D, INF_D, INF_D});
    std::vector<int> a;
    const double total = cv::linearAssignment(cost, a);

    EXPECT_NEAR(0.0, total, 1e-12);
    ASSERT_EQ(2u, a.size());
    EXPECT_EQ(-1, a[0]);
    EXPECT_EQ(-1, a[1]);
}

// Guards the two formulations that were rejected during review.
TEST(Core_LinearAssignment, threshold_limited)
{
    // A post-filter threshold returns 3.44 here. Same cardinality, 6.13 worse.
    Mat c1 = Mat_<double>({3, 3}, {16.2,  19.59,  3.09,
                                   14.67, 13.02, -3.04,
                                    0.35, 17.36, 15.09});
    std::vector<int> a;
    const double t1 = cv::linearAssignment(c1, a, 10.0);
    checkWellFormed(c1, a, 10.0);
    int pairs;
    double sum;
    summarise(c1, a, pairs, sum);
    EXPECT_EQ(2, pairs);
    EXPECT_NEAR(-2.69, t1, 1e-9);

    // Max-cardinality-then-min-cost takes both pairs for 20.0, throwing away a free perfect
    // match to manufacture a marginal second pair.
    Mat c2 = Mat_<double>({2, 2}, {0, 10,
                                  10, 100});
    const double t2 = cv::linearAssignment(c2, a, 10.0);
    checkWellFormed(c2, a, 10.0);
    summarise(c2, a, pairs, sum);
    EXPECT_EQ(1, pairs);
    EXPECT_NEAR(0.0, t2, 1e-12);

    // A threshold above every cost recovers the max-cardinality answer on the same matrix.
    const double t3 = cv::linearAssignment(c2, a, 50.0);
    summarise(c2, a, pairs, sum);
    EXPECT_EQ(2, pairs);
    EXPECT_NEAR(20.0, t3, 1e-12);
}

// Skipping rows whose augmenting search fails returns 10.0 on the second matrix.
TEST(Core_LinearAssignment, partial_feasibility)
{
    std::vector<int> a;

    Mat c1 = Mat_<double>({2, 2}, {1, INF_D,
                                   2, INF_D});
    EXPECT_NEAR(1.0, cv::linearAssignment(c1, a), 1e-12);
    EXPECT_EQ(0, a[0]);
    EXPECT_EQ(-1, a[1]);

    Mat c2 = Mat_<double>({2, 2}, {10, INF_D,
                                    1, INF_D});
    EXPECT_NEAR(1.0, cv::linearAssignment(c2, a), 1e-12);
    EXPECT_EQ(-1, a[0]);
    EXPECT_EQ(0, a[1]);
}

TEST(Core_LinearAssignment, nan_is_forbidden)
{
    Mat cost = Mat_<double>({2, 2}, {NAN_D, 5,
                                         3, NAN_D});
    std::vector<int> a;
    const double total = cv::linearAssignment(cost, a);

    EXPECT_FALSE(cvIsNaN(total)) << "a NaN cell leaked into the total";
    EXPECT_NEAR(8.0, total, 1e-12);
    EXPECT_EQ(1, a[0]);
    EXPECT_EQ(0, a[1]);
}

TEST(Core_LinearAssignment, transpose_invariance)
{
    RNG rng(0x5EED1234);
    for (int iter = 0; iter < 300; iter++)
    {
        const int M = rng.uniform(1, 6);
        const int N = rng.uniform(1, 6);
        Mat cost(M, N, CV_64F);
        rng.fill(cost, RNG::UNIFORM, -5.0, 20.0);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                if (rng.uniform(0, 100) < 25)
                    cost.at<double>(i, j) = INF_D;

        const double thr = rng.uniform(0, 4) < 2 ? 10.0 : DBL_MAX;

        std::vector<int> a, at;
        const double t = cv::linearAssignment(cost, a, thr);
        Mat costT;
        cv::transpose(cost, costT);
        const double tt = cv::linearAssignment(costT, at, thr);

        int pairs, pairsT;
        double sum, sumT;
        summarise(cost, a, pairs, sum);
        summarise(costT, at, pairsT, sumT);

        EXPECT_EQ(pairs, pairsT) << "iteration " << iter;
        EXPECT_NEAR(t, tt, 1e-9) << "iteration " << iter;
    }
}

TEST(Core_LinearAssignment, random_vs_bruteforce)
{
    RNG rng(0xB00B1E5);
    const double thresholds[] = { 2.0, 5.0, 10.0, 15.0, DBL_MAX };

    for (int iter = 0; iter < 2000; iter++)
    {
        const int M = rng.uniform(1, 6);
        const int N = rng.uniform(1, 6);
        Mat cost(M, N, CV_64F);
        rng.fill(cost, RNG::UNIFORM, -5.0, 20.0);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                if (rng.uniform(0, 100) < 30)
                    cost.at<double>(i, j) = INF_D;

        const double thr = thresholds[rng.uniform(0, 5)];

        std::vector<int> a;
        const double total = cv::linearAssignment(cost, a, thr);
        checkWellFormed(cost, a, thr);

        int pairs;
        double sum;
        summarise(cost, a, pairs, sum);
        EXPECT_NEAR(total, sum, 1e-9) << "returned total disagrees with the pairs reported";

        int refPairs;
        double refCost;
        bruteForce(cost, thr, refPairs, refCost);

        EXPECT_EQ(refPairs, pairs) << "iteration " << iter << ", threshold " << thr
                                   << "\n" << cost;
        EXPECT_NEAR(refCost, total, 1e-9) << "iteration " << iter << ", threshold " << thr
                                          << "\n" << cost;
    }
}

// Independent oracle that scales: plant a cheap permutation among expensive cells, so the optimum
// is that permutation by construction. Any other matching gives up at least two cheap cells for
// two expensive ones.
TEST(Core_LinearAssignment, known_optimum_large)
{
    RNG rng(0xC0FFEE);
    const int sizes[] = { 17, 64, 150 };

    for (int s = 0; s < 3; s++)
    {
        const int n = sizes[s];
        Mat cost(n, n, CV_64F);
        rng.fill(cost, RNG::UNIFORM, 10.0, 20.0);

        std::vector<int> perm((size_t)n);
        for (int i = 0; i < n; i++)
            perm[i] = i;
        for (int i = n - 1; i > 0; i--)
            std::swap(perm[i], perm[rng.uniform(0, i + 1)]);

        double expected = 0.0;
        for (int i = 0; i < n; i++)
        {
            const double v = rng.uniform(0.0, 1.0);
            cost.at<double>(i, perm[i]) = v;
            expected += v;
        }

        std::vector<int> a;
        const double total = cv::linearAssignment(cost, a);

        ASSERT_EQ((size_t)n, a.size());
        EXPECT_NEAR(expected, total, 1e-9) << "n = " << n;
        for (int i = 0; i < n; i++)
            EXPECT_EQ(perm[i], a[i]) << "n = " << n << ", row " << i;
    }
}

// The dummy columns are skipped when nothing is forbidden and every cost is cheaper than the
// price of not pairing. That is a different matrix shape reaching the solver, so sweep it on its
// own rather than relying on a random matrix happening to contain no inf.
TEST(Core_LinearAssignment, unpadded_path_vs_bruteforce)
{
    RNG rng(0xFEEDBEEF);
    // 25.5 and 1e6 sit above the cost range, so no cell is forbidden and no cell can tie with
    // the threshold; DBL_MAX is the default. All three take the skip.
    const double thresholds[] = { 25.5, 1e6, DBL_MAX };

    for (int iter = 0; iter < 2000; iter++)
    {
        const int M = rng.uniform(1, 7);
        const int N = rng.uniform(1, 7);
        Mat cost(M, N, CV_64F);
        rng.fill(cost, RNG::UNIFORM, -5.0, 20.0);   // no inf, no NaN
        const double thr = thresholds[rng.uniform(0, 3)];

        std::vector<int> a;
        const double total = cv::linearAssignment(cost, a, thr);
        checkWellFormed(cost, a, thr);

        int pairs;
        double sum;
        summarise(cost, a, pairs, sum);
        EXPECT_EQ(std::min(M, N), pairs)
            << "nothing is forbidden, so every row that can pair must pair\n" << cost;
        EXPECT_NEAR(total, sum, 1e-9);

        int refPairs;
        double refCost;
        bruteForce(cost, thr, refPairs, refCost);
        EXPECT_EQ(refPairs, pairs) << "iteration " << iter << ", threshold " << thr;
        EXPECT_NEAR(refCost, total, 1e-9) << "iteration " << iter << ", threshold " << thr;
    }
}

// cost == costThreshold is a genuine tie: pairing scores the same as not pairing, so the
// cardinality is not determined. Only the objective is, so that is all this asserts.
TEST(Core_LinearAssignment, threshold_tie_objective)
{
    Mat cost = Mat_<double>({2, 2}, {5, 5,
                                     5, 5});
    std::vector<int> a;
    const double total = cv::linearAssignment(cost, a, 5.0);
    checkWellFormed(cost, a, 5.0);

    int pairs;
    double sum;
    summarise(cost, a, pairs, sum);

    // Every pair costs exactly the price of leaving one unmade, so the objective is 10 whatever
    // the solver picks.
    const double objective = total + 5.0 * (2 - pairs);
    EXPECT_NEAR(10.0, objective, 1e-9) << "pairs = " << pairs << ", total = " << total;
}

TEST(Core_LinearAssignment, types_and_errors)
{
    Mat cost64 = Mat_<double>({3, 4}, {5, 2, 8, 1,
                                       3, 9, 4, 7,
                                       6, 1, 2, 5});
    Mat cost32;
    cost64.convertTo(cost32, CV_32F);

    std::vector<int> a64, a32;
    const double t64 = cv::linearAssignment(cost64, a64);
    const double t32 = cv::linearAssignment(cost32, a32);

    EXPECT_NEAR(t64, t32, 1e-6);
    EXPECT_EQ(a64, a32);

    EXPECT_ANY_THROW(cv::linearAssignment(Mat::zeros(3, 3, CV_8U), a64));
    EXPECT_ANY_THROW(cv::linearAssignment(Mat::zeros(3, 3, CV_32S), a64));
    EXPECT_ANY_THROW(cv::linearAssignment(Mat::zeros(3, 3, CV_32FC2), a64));

    // A NaN threshold has no meaning: every comparison against it is false, so it would silently
    // forbid everything rather than fail.
    EXPECT_ANY_THROW(cv::linearAssignment(cost64, a64, NAN_D));

    const int sizes[] = { 2, 2, 2 };
    EXPECT_ANY_THROW(cv::linearAssignment(Mat(3, sizes, CV_64F, Scalar(0)), a64));
}

}} // namespace opencv_test::<anonymous>
