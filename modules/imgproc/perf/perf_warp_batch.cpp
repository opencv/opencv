// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the
// top-level directory of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "perf_precomp.hpp"

namespace opencv_test {

// 0 is warpAffine, 1 is warpPerspective.
typedef tuple<int, MatType, int, Size, Size, int> WarpBatchCase_t;

#define WARP_BATCH_CASES                                                                        \
    testing::Values(                                                                            \
        WarpBatchCase_t(0, CV_8UC3,  4096, Size(16, 16),   Size(16, 16),   INTER_LINEAR),       \
        WarpBatchCase_t(0, CV_8UC3,  1024, Size(64, 64),   Size(64, 64),   INTER_LINEAR),       \
        WarpBatchCase_t(0, CV_8UC3,  256,  Size(224, 224), Size(224, 224), INTER_LINEAR),       \
        WarpBatchCase_t(0, CV_8UC3,  16,   sz1080p,        sz1080p,        INTER_LINEAR),       \
        WarpBatchCase_t(0, CV_8UC3,  4,    sz1080p,        sz1080p,        INTER_LINEAR),       \
        WarpBatchCase_t(0, CV_8UC3,  4,    sz1080p,        sz1080p,        INTER_LANCZOS4),     \
        WarpBatchCase_t(0, CV_32FC1, 512,  Size(128, 128), Size(128, 128), INTER_LINEAR),       \
        WarpBatchCase_t(0, CV_8UC3,  256,  Size(224, 224), Size(224, 224), INTER_NEAREST),      \
        WarpBatchCase_t(0, CV_8UC3,  256,  Size(224, 224), Size(224, 224), INTER_CUBIC),        \
        WarpBatchCase_t(0, CV_8UC1,  256,  Size(224, 224), Size(224, 224), INTER_LANCZOS4),     \
        WarpBatchCase_t(1, CV_8UC3,  1024, Size(64, 64),   Size(64, 64),   INTER_LINEAR),       \
        WarpBatchCase_t(1, CV_8UC3,  256,  Size(224, 224), Size(224, 224), INTER_LINEAR),       \
        WarpBatchCase_t(1, CV_32FC1, 4,    sz1080p,        sz1080p,        INTER_LINEAR)        \
    )

static Mat makeBatchTransform(int op, Size sz, int i)
{
    RNG rng(0x5eed + i);
    Mat M = getRotationMatrix2D(Point2f(sz.width*0.5f, sz.height*0.5f), rng.uniform(-30., 30.), rng.uniform(0.8, 1.2));
    if (op == 0)
        return M;
    Mat P = Mat::eye(3, 3, CV_64F);
    M.copyTo(P.rowRange(0, 2));
    P.at<double>(2, 0) = rng.uniform(-1e-4, 1e-4);
    P.at<double>(2, 1) = rng.uniform(-1e-4, 1e-4);
    return P;
}

static void warpOneImage(int op, InputArray src, OutputArray dst, InputArray M, Size to, int interp)
{
    if (op == 0)
        warpAffine(src, dst, M, to, interp);
    else
        warpPerspective(src, dst, M, to, interp);
}

typedef TestBaseWithParam<WarpBatchCase_t> WarpBatch;

// Baseline: what callers write today, one warp per image spread over the threads.
PERF_TEST_P(WarpBatch, parallel_for_single_warp, WARP_BATCH_CASES)
{
    const int op = get<0>(GetParam()), matType = get<1>(GetParam()), n = get<2>(GetParam());
    const Size from = get<3>(GetParam()), to = get<4>(GetParam());
    const int interp = get<5>(GetParam());

    std::vector<Mat> src(n), dst(n), Ms(n);
    RNG rng(0x5eed);
    for (int i = 0; i < n; i++)
    {
        src[i].create(from, matType);
        rng.fill(src[i], RNG::UNIFORM, 0, 255);
        dst[i].create(to, matType);
        Ms[i] = makeBatchTransform(op, from, i);
    }

    TEST_CYCLE()
    {
        parallel_for_(Range(0, n), [&](const Range& r) {
            for (int i = r.start; i < r.end; i++)
                warpOneImage(op, src[i], dst[i], Ms[i], to, interp);
        });
    }

    SANITY_CHECK_NOTHING();
}

// The batched call on a std::vector<Mat>, one matrix per image.
PERF_TEST_P(WarpBatch, batched_vector, WARP_BATCH_CASES)
{
    const int op = get<0>(GetParam()), matType = get<1>(GetParam()), n = get<2>(GetParam());
    const Size from = get<3>(GetParam()), to = get<4>(GetParam());
    const int interp = get<5>(GetParam());

    std::vector<Mat> src(n), dst, Ms(n);
    RNG rng(0x5eed);
    for (int i = 0; i < n; i++)
    {
        src[i].create(from, matType);
        rng.fill(src[i], RNG::UNIFORM, 0, 255);
        Ms[i] = makeBatchTransform(op, from, i);
    }

    TEST_CYCLE() warpOneImage(op, src, dst, Ms, to, interp);

    SANITY_CHECK_NOTHING();
}

// The same batch as one N-D tensor, the shape a DNN pipeline holds.
PERF_TEST_P(WarpBatch, batched_tensor, WARP_BATCH_CASES)
{
    const int op = get<0>(GetParam()), matType = get<1>(GetParam()), n = get<2>(GetParam());
    const Size from = get<3>(GetParam()), to = get<4>(GetParam());
    const int interp = get<5>(GetParam());

    int srcSizes[] = { n, from.height, from.width };
    Mat src(3, srcSizes, matType), dst;
    RNG(0x5eed).fill(src, RNG::UNIFORM, 0, 255);
    std::vector<Mat> Ms(n);
    for (int i = 0; i < n; i++)
        Ms[i] = makeBatchTransform(op, from, i);

    TEST_CYCLE() warpOneImage(op, src, dst, Ms, to, interp);

    SANITY_CHECK_NOTHING();
}

} // namespace
