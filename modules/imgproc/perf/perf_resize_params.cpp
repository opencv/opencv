// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the
// top-level directory of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "perf_precomp.hpp"

namespace opencv_test {

// ResizeCoord as an int, so the report lists the mode next to its timing:
// 0 PIXEL_CENTER, 1 HALF_PIXEL, 2 PYTORCH_HALF_PIXEL, 3 ALIGN_CORNERS,
// 4 ASYMMETRIC, 5 TF_HALF_PIXEL_FOR_NN, 6 HALF_PIXEL_SYMMETRIC.
#define RESIZE_ALL_COORD_MODES testing::Values(0, 1, 2, 3, 4, 5, 6)

typedef tuple<MatType, int, int> Type_Interp_Coord_t;
typedef TestBaseWithParam<Type_Interp_Coord_t> ResizeParams_CoordMode;

// Every mode resolves into the tables and then runs the same kernels, so the whole sweep has to
// come out at one speed. A mode that drifted onto a slower path shows up as an outlier column.
PERF_TEST_P(ResizeParams_CoordMode, coord_modes_1080p_to_VGA,
            testing::Combine(
                testing::Values(CV_8UC3, CV_32FC1),
                testing::Values(INTER_NEAREST, INTER_LINEAR, INTER_CUBIC),
                RESIZE_ALL_COORD_MODES))
{
    const int matType = get<0>(GetParam());
    const int interp = get<1>(GetParam());
    const int coord = get<2>(GetParam());

    Mat src(sz1080p, matType), dst(szVGA, matType);
    RNG(0x5eed).fill(src, RNG::UNIFORM, 0, 255);
    declare.in(src).out(dst);

    ResizeParams params(szVGA, 0, 0, interp);
    params.coordMode = (ResizeCoord)coord;

    TEST_CYCLE() resize(src, dst, params);

    SANITY_CHECK_NOTHING();
}

typedef TestBaseWithParam<tuple<MatType, int> > ResizeParams_NearestMode;

// Same for nearest_mode: it picks the rounding rule while the table is built, so all four tie.
PERF_TEST_P(ResizeParams_NearestMode, nearest_modes_1080p_to_VGA,
            testing::Combine(
                testing::Values(CV_8UC3, CV_16UC1),
                testing::Values(0, 1, 2, 3)))   // FLOOR, CEIL, ROUND_PREFER_CEIL, ROUND_PREFER_FLOOR
{
    const int matType = get<0>(GetParam());
    const int round = get<1>(GetParam());

    Mat src(sz1080p, matType), dst(szVGA, matType);
    RNG(0x5eed).fill(src, RNG::UNIFORM, 0, 255);
    declare.in(src).out(dst);

    ResizeParams params(szVGA, 0, 0, INTER_NEAREST);
    params.coordMode = ResizeCoord::HALF_PIXEL;
    params.nearestMode = (ResizeNearest)round;

    TEST_CYCLE() resize(src, dst, params);

    SANITY_CHECK_NOTHING();
}

typedef TestBaseWithParam<tuple<int, bool> > ResizeParams_ExcludeOutside;

// exclude_outside only zeroes and renormalises weights in the table, so it is free at run time.
PERF_TEST_P(ResizeParams_ExcludeOutside, exclude_outside_1080p_to_VGA,
            testing::Combine(
                testing::Values(INTER_CUBIC, INTER_LANCZOS4),
                testing::Bool()))
{
    const int interp = get<0>(GetParam());
    const bool exclude = get<1>(GetParam());

    Mat src(sz1080p, CV_8UC3), dst(szVGA, CV_8UC3);
    RNG(0x5eed).fill(src, RNG::UNIFORM, 0, 255);
    declare.in(src).out(dst);

    ResizeParams params(szVGA, 0, 0, interp);
    params.coordMode = ResizeCoord::HALF_PIXEL;
    params.excludeOutside = exclude;

    TEST_CYCLE() resize(src, dst, params);

    SANITY_CHECK_NOTHING();
}

// Each case runs twice: today's per-image loop, then the batched cv::resize.
typedef tuple<MatType, int, Size, Size, int> BatchCase_t;

#define RESIZE_BATCH_CASES                                                           \
    testing::Values(                                                                 \
        BatchCase_t(CV_8UC3,  4096, Size(16, 16),   Size(8, 8),     INTER_LINEAR),   \
        BatchCase_t(CV_8UC3,  1024, Size(64, 64),   Size(32, 32),   INTER_LINEAR),   \
        BatchCase_t(CV_8UC3,  256,  Size(224, 224), Size(112, 112), INTER_LINEAR),   \
        BatchCase_t(CV_8UC3,  16,   sz1080p,        szVGA,          INTER_LINEAR),   \
        BatchCase_t(CV_32FC1, 512,  Size(128, 128), Size(64, 64),   INTER_LINEAR),   \
        BatchCase_t(CV_32FC1, 8,    sz1080p,        szVGA,          INTER_LINEAR),   \
        BatchCase_t(CV_8UC3,  256,  Size(224, 224), Size(112, 112), INTER_NEAREST),  \
        BatchCase_t(CV_8UC3,  256,  Size(224, 224), Size(112, 112), INTER_CUBIC),    \
        BatchCase_t(CV_8UC3,  256,  Size(224, 224), Size(112, 112), INTER_AREA)      \
    )

static void fillBatch(std::vector<Mat>& planes, int matType, Size from)
{
    RNG rng(0x5eed);
    for (size_t i = 0; i < planes.size(); i++)
    {
        planes[i].create(from, matType);
        rng.fill(planes[i], RNG::UNIFORM, 0, 255);
    }
}

typedef TestBaseWithParam<BatchCase_t> ResizeBatch;

// Baseline: what callers write today -- one cv::resize per image, spread over the threads.
PERF_TEST_P(ResizeBatch, parallel_for_single_resize, RESIZE_BATCH_CASES)
{
    int matType = get<0>(GetParam());
    int n = get<1>(GetParam());
    Size from = get<2>(GetParam());
    Size to = get<3>(GetParam());
    int interp = get<4>(GetParam());

    std::vector<Mat> src(n), dst(n);
    fillBatch(src, matType, from);
    for (int i = 0; i < n; i++)
        dst[i].create(to, matType);

    TEST_CYCLE()
    {
        parallel_for_(Range(0, n), [&](const Range& r) {
            for (int i = r.start; i < r.end; i++)
                resize(src[i], dst[i], to, 0, 0, interp);
        });
    }

    SANITY_CHECK_NOTHING();
}

// The batched overload on a std::vector<Mat>: one plan per distinct geometry, one parallel loop.
PERF_TEST_P(ResizeBatch, batched_vector, RESIZE_BATCH_CASES)
{
    int matType = get<0>(GetParam());
    int n = get<1>(GetParam());
    Size from = get<2>(GetParam());
    Size to = get<3>(GetParam());
    int interp = get<4>(GetParam());

    std::vector<Mat> src(n), dst;
    fillBatch(src, matType, from);

    ResizeParams params(to, 0, 0, interp);

    TEST_CYCLE() resize(src, dst, params);

    SANITY_CHECK_NOTHING();
}

// The same batch as one N-D tensor, the shape a DNN pipeline holds.
PERF_TEST_P(ResizeBatch, batched_tensor, RESIZE_BATCH_CASES)
{
    int matType = get<0>(GetParam());
    int n = get<1>(GetParam());
    Size from = get<2>(GetParam());
    Size to = get<3>(GetParam());
    int interp = get<4>(GetParam());

    int srcSizes[] = { n, from.height, from.width };
    Mat src(3, srcSizes, matType);
    RNG(0x5eed).fill(src, RNG::UNIFORM, 0, 255);

    ResizeParams params(to, 0, 0, interp);

    Mat dst;
    TEST_CYCLE() resize(src, dst, params);

    SANITY_CHECK_NOTHING();
}

} // namespace
