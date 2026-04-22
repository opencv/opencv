/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"
#include <chrono>
#include "../src/convex_hull_bucket_sort.hpp"

namespace opencv_test { namespace {

TEST(Imgproc_ConvexHull, dense_columns_correct_hull)
{
    std::vector<Point> points{
        Point(0, 0), Point(0, 2), Point(0, 5),
        Point(1, 1), Point(1, 3), Point(1, 6),
        Point(2, 0), Point(2, 2), Point(2, 7),
        Point(3, 1), Point(3, 4), Point(3, 6),
        Point(2, 7), // duplicate extreme point
        Point(1, 3)  // duplicate inner point
    };

    std::vector<Point> hull_pts;
    std::vector<int> hull_idx;

    convexHull(points, hull_pts, false, true);
    convexHull(points, hull_idx, false, false);

    ASSERT_EQ(hull_pts.size(), hull_idx.size());
    ASSERT_GE(hull_pts.size(), 3u);

    for (size_t i = 0; i < hull_idx.size(); ++i)
    {
        ASSERT_GE(hull_idx[i], 0);
        ASSERT_LT(hull_idx[i], (int)points.size());
        EXPECT_EQ(hull_pts[i], points[hull_idx[i]]);
    }

    std::vector<Point> expected{
        Point(0, 0),
        Point(2, 0),
        Point(3, 1),
        Point(3, 6),
        Point(2, 7),
        Point(0, 5)
    };

    ASSERT_EQ(expected.size(), hull_pts.size());

    // find rotation offset
    int shift = -1;
    for (size_t i = 0; i < hull_pts.size(); ++i)
    {
        if (hull_pts[i] == expected[0])
        {
            shift = (int)i;
            break;
        }
    }

    ASSERT_NE(shift, -1);

    // compare with rotation
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_EQ(expected[i], hull_pts[(i + shift) % hull_pts.size()])
            << "Mismatch at hull vertex " << i;
    }
}

struct BucketSortCmpPoints
{
    // Sort point pointers by x, then by y.
    bool operator()(const Point* p1, const Point* p2) const
    {
        if (p1->x != p2->x)
            return p1->x < p2->x;
        if (p1->y != p2->y)
            return p1->y < p2->y;
        return p1 < p2;
    }
};


// Build the pointer array expected by convex_hull_bucket_sort.
static std::vector<Point*> makePointerArray(std::vector<Point>& points)
{
    std::vector<Point*> ptrs(points.size());
    for (size_t i = 0; i < points.size(); ++i)
        ptrs[i] = &points[i];
    return ptrs;
}

static void findMinMaxYIndices(const std::vector<Point*>& ptrs,
                                           int& miny_ind,
                                           int& maxy_ind)
{
    ASSERT_FALSE(ptrs.empty());

    miny_ind = 0;
    maxy_ind = 0;

    for (int i = 1; i < (int)ptrs.size(); ++i)
    {
        const int y = ptrs[i]->y;
        if (ptrs[miny_ind]->y > y)
            miny_ind = i;
        if (ptrs[maxy_ind]->y < y)
            maxy_ind = i;
    }
}

// Given a sorted pointer array, keeps only the lowest-y and highest-y point for each unique x value
static std::vector<Point*> keepExtremeYPerX(std::vector<Point*>& sorted_ptrs)
{
    std::vector<Point*> compressed;

    if (sorted_ptrs.empty())
        return compressed;

    int i = 0;
    while (i < (int)sorted_ptrs.size())
    {
        int j = i;
        const int x = sorted_ptrs[i]->x;

        while (j < (int)sorted_ptrs.size() && sorted_ptrs[j]->x == x)
            ++j;

        Point* pmin = sorted_ptrs[i];
        Point* pmax = sorted_ptrs[j - 1];

        compressed.push_back(pmin);

        if (pmax != pmin)
            compressed.push_back(pmax);

        i = j;
    }

    return compressed;
}

// Generate random points with many repeated x values.
static std::vector<Point> generateDenseColumnsPoints(int total_points,
                                                     int range_x,
                                                     int y_min,
                                                     int y_max,
                                                     uint64 seed)
{
    CV_Assert(total_points > 0);
    CV_Assert(range_x > 0);

    RNG rng((uint64)seed);
    std::vector<Point> points;
    points.reserve(total_points);

    for (int i = 0; i < total_points; ++i)
    {
        int x = rng.uniform(0, range_x);
        int y = rng.uniform(y_min, y_max);

        points.push_back(Point(x, y));
    }

    return points;
}

static bool runBucketSort(std::vector<Point>& points,
                          std::vector<Point*>& out_ptrs,
                          int& total,
                          int& miny_ind,
                          int& maxy_ind)
{
    out_ptrs = makePointerArray(points);
    total = (int)points.size();

    bool ok = convex_hull_bucket_sort(points.data(),
                                      out_ptrs.data(),
                                      total,
                                      miny_ind,
                                      maxy_ind);
    if (ok)
        out_ptrs.resize(total);

    return ok;
}

// Sorts points with std::sort, extracts y-extremes per x, and finds min/max y indices.
// Produces the ground-truth output that tests compare against.
static void runReferenceSort(std::vector<Point>& points,
                                       std::vector<Point*>& out_ptrs,
                                       int& miny_ind,
                                       int& maxy_ind)
{
    std::vector<Point*> sorted_ptrs = makePointerArray(points);

    std::sort(sorted_ptrs.begin(), sorted_ptrs.end(), BucketSortCmpPoints());
    out_ptrs = keepExtremeYPerX(sorted_ptrs);
    findMinMaxYIndices(out_ptrs, miny_ind, maxy_ind);
}

// Check bucket sort against the reference on a small dense-column case.
TEST(Imgproc_ConvexHullBucketSort, dense_columns_matches_reference)
{
    std::vector<Point> points{
        Point(0, 0), Point(0, 2), Point(0, 5),
        Point(1, 1), Point(1, 3), Point(1, 6),
        Point(2, 0), Point(2, 2), Point(2, 7),
        Point(3, 1), Point(3, 4), Point(3, 6),
        Point(2, 7), // duplicate extreme point
        Point(1, 3)  // duplicate inner point
    };


    std::vector<Point> bucket_points = points;
    std::vector<Point> ref_points = points;

    std::vector<Point*> bucket_out;
    std::vector<Point*> ref_out;

    int bucket_total = 0;
    int bucket_miny = -1;
    int bucket_maxy = -1;

    int ref_miny = -1;
    int ref_maxy = -1;

    ASSERT_TRUE(runBucketSort(bucket_points, bucket_out, bucket_total, bucket_miny, bucket_maxy));

    runReferenceSort(ref_points, ref_out, ref_miny, ref_maxy);

    ASSERT_EQ(bucket_out.size(), ref_out.size());

    for (size_t i = 0; i < bucket_out.size(); ++i)
    {
        EXPECT_EQ(*bucket_out[i], *ref_out[i])
            << "Mismatch in compressed output at index " << i;
    }

    ASSERT_GE(bucket_miny, 0);
    ASSERT_GE(bucket_maxy, 0);
    ASSERT_LT(bucket_miny, (int)bucket_out.size());
    ASSERT_LT(bucket_maxy, (int)bucket_out.size());

    EXPECT_EQ(bucket_out[bucket_miny]->y, ref_out[ref_miny]->y);
    EXPECT_EQ(bucket_out[bucket_maxy]->y, ref_out[ref_maxy]->y);
}

// Repeat the same check on many reproducible random dense-column inputs.
TEST(Imgproc_ConvexHullBucketSort, random_dense_columns_match_reference)
{
    const int kIterations = 50;
    for (int iter = 0; iter < kIterations; ++iter)
    {
        SCOPED_TRACE(cv::format("iteration=%d", iter));

        const int total_points = 200 + iter * 10;
        const int range_x = 16;

        std::vector<Point> points = generateDenseColumnsPoints(total_points, range_x, -1000, 1000, 12345 + iter);
        std::vector<Point> bucket_points = points;
        std::vector<Point> ref_points = points;

        std::vector<Point*> bucket_out;
        std::vector<Point*> ref_out;

        int bucket_total = 0;
        int bucket_miny = -1;
        int bucket_maxy = -1;

        int ref_miny = -1;
        int ref_maxy = -1;

        ASSERT_TRUE(runBucketSort(bucket_points, bucket_out, bucket_total, bucket_miny, bucket_maxy));
        runReferenceSort(ref_points, ref_out, ref_miny, ref_maxy);

        ASSERT_EQ(bucket_out.size(), ref_out.size());

        for (size_t i = 0; i < bucket_out.size(); ++i)
        {
            EXPECT_EQ(*bucket_out[i], *ref_out[i])
                << "Mismatch at output index " << i;
        }

        ASSERT_FALSE(bucket_out.empty());
        ASSERT_GE(bucket_miny, 0);
        ASSERT_GE(bucket_maxy, 0);
        ASSERT_LT(bucket_miny, (int)bucket_out.size());
        ASSERT_LT(bucket_maxy, (int)bucket_out.size());

        EXPECT_EQ(bucket_out[bucket_miny]->y, ref_out[ref_miny]->y);
        EXPECT_EQ(bucket_out[bucket_maxy]->y, ref_out[ref_maxy]->y);
    }
}

// Measure average runtime of bucket sort only, in microseconds.
static double benchmarkBucketSortOnly(const std::vector<Point>& input_points,
                                      int iterations)
{
    using namespace std::chrono;

    double total_us = 0.0;

    for (int iter = 0; iter < iterations; ++iter)
    {
        std::vector<Point> points = input_points;
        std::vector<Point*> ptrs = makePointerArray(points);

        int total = (int)points.size();
        int miny_ind = 0;
        int maxy_ind = 0;

        auto t0 = high_resolution_clock::now();
        bool ok = convex_hull_bucket_sort(points.data(),
                                          ptrs.data(),
                                          total,
                                          miny_ind,
                                          maxy_ind);
        auto t1 = high_resolution_clock::now();

        EXPECT_TRUE(ok);
        if (!ok)
            return -1.0;

        total_us += duration_cast<duration<double, std::micro>>(t1 - t0).count();
    }

    return total_us / iterations;
}

// Measure the reference sort-compress-minmax pipeline, in microseconds.
// Provides the baseline timing for the perf test.
static double benchmarkReferenceSortOnly(const std::vector<Point>& input_points,
                                               int iterations)
{
    using namespace std::chrono;

    double total_us = 0.0;

    for (int iter = 0; iter < iterations; ++iter)
    {
        std::vector<Point> points = input_points;
        std::vector<Point*> ptrs = makePointerArray(points);

        auto t0 = high_resolution_clock::now();

        std::sort(ptrs.begin(), ptrs.end(), BucketSortCmpPoints());
        std::vector<Point*> compressed = keepExtremeYPerX(ptrs);
        int miny_ind = 0;
        int maxy_ind = 0;
        findMinMaxYIndices(compressed, miny_ind, maxy_ind);

        auto t1 = high_resolution_clock::now();

        total_us += duration_cast<duration<double, std::micro>>(t1 - t0).count();
    }

    return total_us / iterations;
}

// Manual benchmark for bucket sort versus the reference path.
TEST(Imgproc_ConvexHullBucketSortPerf, DISABLED_dense_columns_bucket_vs_std_sort)
{
    struct Case
    {
        int total_points;
        int range_x;
        const char* name;
    };

    std::vector<Case> cases{
        {1000,  8,  "small_range_8"},
        {1000, 16,  "small_range_16"},
        {5000, 16,  "medium_range_16"},
        {5000, 32,  "medium_range_32"},
        {10000, 32, "large_range_32"}
    };

    const int kWarmupIterations = 5;
    const int kMeasureIterations = 50;

    for (const auto& tc : cases)
    {
        SCOPED_TRACE(cv::format("case=%s total=%d range_x=%d",
                                tc.name, tc.total_points, tc.range_x));

        std::vector<Point> points = generateDenseColumnsPoints(tc.total_points,
                                                               tc.range_x,
                                                               -100000,
                                                               100000,
                                                               777);

        (void)benchmarkBucketSortOnly(points, kWarmupIterations);
        (void)benchmarkReferenceSortOnly(points, kWarmupIterations);

        const double bucket_us = benchmarkBucketSortOnly(points, kMeasureIterations);
        const double sort_us   = benchmarkReferenceSortOnly(points, kMeasureIterations);
        std::cout
            << "\n[BucketSortPerf] case=" << tc.name
            << " total_points=" << tc.total_points
            << " range_x=" << tc.range_x
            << " bucket_us=" << bucket_us
            << " ref_sort_us=" << sort_us
            << " speedup=" << (sort_us / bucket_us)
            << "x\n";

        EXPECT_GT(bucket_us, 0.0);
        EXPECT_GT(sort_us, 0.0);
    }
}

// Ensure bucket sort rejects inputs with an excessively large x range.
TEST(Imgproc_ConvexHullBucketSort, rejects_huge_rangeX)
{
    std::vector<Point> points{
        Point(0, 0),
        Point(200000, 1),
        Point(400000, 2)
    };

    std::vector<Point*> ptrs = makePointerArray(points);
    int total = (int)points.size();
    int miny_ind = 0;
    int maxy_ind = 0;

    bool ok = convex_hull_bucket_sort(points.data(),
                                      ptrs.data(),
                                      total,
                                      miny_ind,
                                      maxy_ind);

    EXPECT_FALSE(ok);
}

}} // namespace