// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/**
 * Performance tests for KDTreeIndex.
 *
 * Setup: N=10 000 random points in [-1000, 1000]^D, K=10 neighbours,
 * Q=500 queries.  Radius is scaled as 50*sqrt(D/3) so the search ball
 * covers a comparable data fraction at each dimension.
 *
 * Run all KDTree perf tests:
 *   ./opencv_perf_flann --gtest_filter="*KDTree*"
 *
 * Compare two builds (e.g. patched vs vanilla) with the perf regression tool:
 *   opencv_regression_db  --help
 */

#include "perf_precomp.hpp"

namespace opencv_test {
using namespace perf;

// ─── shared constants ────────────────────────────────────────────────────────

static const int KDTREE_N        = 10000;
static const int KDTREE_K        = 10;
static const int KDTREE_Q        = 500;
static const float KDTREE_R_3D   = 50.f;   // radius at dim=3; scaled for other dims

// Dimensions to sweep: low-dim shows the biggest speedup.
static const int kDims[] = { 2, 3, 8, 32, 128 };

// ─── helper: fill a CV_32F matrix with seeded uniform random data ─────────────

static void fill_rng(cv::Mat& m, float lo, float hi, uint64_t seed)
{
    cv::RNG rng(seed);
    rng.fill(m, cv::RNG::UNIFORM, lo, hi);
}

// ─── build ────────────────────────────────────────────────────────────────────

typedef perf::TestBaseWithParam<int> Flann_KDTree_Build;

PERF_TEST_P(Flann_KDTree_Build, dim,
            testing::ValuesIn(kDims))
{
    const int dim = GetParam();

    cv::Mat data(KDTREE_N, dim, CV_32F);
    fill_rng(data, -1000.f, 1000.f, /*seed*/42);

    declare.in(data);

    TEST_CYCLE()
    {
        cv::flann::Index idx(data, cv::flann::KDTreeIndexParams(1),
                             cvflann::FLANN_DIST_L2);
        (void)idx;
    }

    SANITY_CHECK_NOTHING();
}

// ─── KNN approximate ─────────────────────────────────────────────────────────

typedef perf::TestBaseWithParam<int> Flann_KDTree_KNN_Approx;

PERF_TEST_P(Flann_KDTree_KNN_Approx, dim,
            testing::ValuesIn(kDims))
{
    const int dim = GetParam();

    cv::Mat data(KDTREE_N, dim, CV_32F);
    cv::Mat queries(KDTREE_Q, dim, CV_32F);
    fill_rng(data,    -1000.f, 1000.f, /*seed*/42);
    fill_rng(queries, -1000.f, 1000.f, /*seed*/43);

    cv::flann::Index idx(data, cv::flann::KDTreeIndexParams(1), cvflann::FLANN_DIST_L2);

    cv::Mat idx_out(KDTREE_Q, KDTREE_K, CV_32S);
    cv::Mat dist_out(KDTREE_Q, KDTREE_K, CV_32F);

    declare.in(queries).out(idx_out, dist_out);

    TEST_CYCLE()
    {
        idx.knnSearch(queries, idx_out, dist_out, KDTREE_K,
                      cv::flann::SearchParams(32));
    }

    SANITY_CHECK_NOTHING();
}

// ─── KNN exact ───────────────────────────────────────────────────────────────

typedef perf::TestBaseWithParam<int> Flann_KDTree_KNN_Exact;

PERF_TEST_P(Flann_KDTree_KNN_Exact, dim,
            testing::ValuesIn(kDims))
{
    const int dim = GetParam();

    cv::Mat data(KDTREE_N, dim, CV_32F);
    cv::Mat queries(KDTREE_Q, dim, CV_32F);
    fill_rng(data,    -1000.f, 1000.f, /*seed*/42);
    fill_rng(queries, -1000.f, 1000.f, /*seed*/43);

    cv::flann::Index idx(data, cv::flann::KDTreeIndexParams(1), cvflann::FLANN_DIST_L2);

    cv::Mat idx_out(KDTREE_Q, KDTREE_K, CV_32S);
    cv::Mat dist_out(KDTREE_Q, KDTREE_K, CV_32F);

    declare.in(queries).out(idx_out, dist_out);

    TEST_CYCLE()
    {
        idx.knnSearch(queries, idx_out, dist_out, KDTREE_K,
                      cv::flann::SearchParams(cvflann::FLANN_CHECKS_UNLIMITED));
    }

    SANITY_CHECK_NOTHING();
}

// ─── radius search approximate ────────────────────────────────────────────────

typedef perf::TestBaseWithParam<int> Flann_KDTree_Radius_Approx;

PERF_TEST_P(Flann_KDTree_Radius_Approx, dim,
            testing::ValuesIn(kDims))
{
    const int dim = GetParam();
    const float radius    = KDTREE_R_3D * std::sqrt(float(dim) / 3.f);
    const float radius_sq = radius * radius;

    cv::Mat data(KDTREE_N, dim, CV_32F);
    cv::Mat queries(KDTREE_Q, dim, CV_32F);
    fill_rng(data,    -1000.f, 1000.f, /*seed*/42);
    fill_rng(queries, -1000.f, 1000.f, /*seed*/43);

    cv::flann::Index idx(data, cv::flann::KDTreeIndexParams(1), cvflann::FLANN_DIST_L2);

    declare.in(queries);

    TEST_CYCLE()
    {
        for (int qi = 0; qi < KDTREE_Q; ++qi)
        {
            cv::Mat ri, rd;
            idx.radiusSearch(queries.row(qi), ri, rd, radius_sq, KDTREE_N,
                             cv::flann::SearchParams(32));
        }
    }

    SANITY_CHECK_NOTHING();
}

// ─── radius search exact ──────────────────────────────────────────────────────

typedef perf::TestBaseWithParam<int> Flann_KDTree_Radius_Exact;

PERF_TEST_P(Flann_KDTree_Radius_Exact, dim,
            testing::ValuesIn(kDims))
{
    const int dim = GetParam();
    const float radius    = KDTREE_R_3D * std::sqrt(float(dim) / 3.f);
    const float radius_sq = radius * radius;

    cv::Mat data(KDTREE_N, dim, CV_32F);
    cv::Mat queries(KDTREE_Q, dim, CV_32F);
    fill_rng(data,    -1000.f, 1000.f, /*seed*/42);
    fill_rng(queries, -1000.f, 1000.f, /*seed*/43);

    cv::flann::Index idx(data, cv::flann::KDTreeIndexParams(1), cvflann::FLANN_DIST_L2);

    declare.in(queries);

    TEST_CYCLE()
    {
        for (int qi = 0; qi < KDTREE_Q; ++qi)
        {
            cv::Mat ri, rd;
            idx.radiusSearch(queries.row(qi), ri, rd, radius_sq, KDTREE_N,
                             cv::flann::SearchParams(cvflann::FLANN_CHECKS_UNLIMITED));
        }
    }

    SANITY_CHECK_NOTHING();
}

} // namespace opencv_test
