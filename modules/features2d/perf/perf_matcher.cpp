// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2026, Advanced Micro Devices, Inc., all rights reserved.

// Brute-force descriptor matching perf tests. These exercise the core distance
// kernels (hal::normL2Sqr_ / normL1_ / normHamming via cv::batchDistance) that
// back BFMatcher, which is how float descriptors (SIFT 128-d, SURF 64-d) and
// binary descriptors (ORB 32-byte) are matched.

#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

// (descriptor dimension, query/train descriptor count)
typedef tuple<int, int> Dim_Count_t;
typedef TestBaseWithParam<Dim_Count_t> DescriptorMatcherFixture;

// Float descriptors matched with L2 (SIFT=128, SURF=64) — uses hal::normL2Sqr_.
PERF_TEST_P(DescriptorMatcherFixture, bfmatch_knn_L2_float,
            testing::Combine(
                testing::Values(64, 128),       // SURF, SIFT descriptor sizes
                testing::Values(512, 1000)
            ))
{
    const int dim   = get<0>(GetParam());
    const int count = get<1>(GetParam());

    Mat query(count, dim, CV_32F);
    Mat train(count, dim, CV_32F);
    declare.in(query, train, WARMUP_RNG);
    declare.time(60);

    BFMatcher matcher(NORM_L2, false);
    std::vector<std::vector<DMatch> > matches;

    TEST_CYCLE() matcher.knnMatch(query, train, matches, 2);

    SANITY_CHECK_NOTHING();
}

// Float descriptors matched with L1 — uses hal::normL1_.
PERF_TEST_P(DescriptorMatcherFixture, bfmatch_knn_L1_float,
            testing::Combine(
                testing::Values(64, 128),
                testing::Values(512, 1000)
            ))
{
    const int dim   = get<0>(GetParam());
    const int count = get<1>(GetParam());

    Mat query(count, dim, CV_32F);
    Mat train(count, dim, CV_32F);
    declare.in(query, train, WARMUP_RNG);
    declare.time(60);

    BFMatcher matcher(NORM_L1, false);
    std::vector<std::vector<DMatch> > matches;

    TEST_CYCLE() matcher.knnMatch(query, train, matches, 2);

    SANITY_CHECK_NOTHING();
}

// Binary descriptors matched with Hamming (ORB/BRISK=32 bytes) — uses hal::normHamming.
PERF_TEST_P(DescriptorMatcherFixture, bfmatch_knn_Hamming_binary,
            testing::Combine(
                testing::Values(32, 64),        // ORB (32), BRISK/FREAK (64) byte sizes
                testing::Values(512, 1000)
            ))
{
    const int bytes = get<0>(GetParam());
    const int count = get<1>(GetParam());

    Mat query(count, bytes, CV_8U);
    Mat train(count, bytes, CV_8U);
    declare.in(query, train, WARMUP_RNG);
    declare.time(60);

    BFMatcher matcher(NORM_HAMMING, false);
    std::vector<std::vector<DMatch> > matches;

    TEST_CYCLE() matcher.knnMatch(query, train, matches, 2);

    SANITY_CHECK_NOTHING();
}

} // namespace
