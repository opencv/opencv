// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the
// top-level directory of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "perf_precomp.hpp"

namespace opencv_test {

typedef tuple<int, Size, Size> Interp_Size_Size_t;
typedef TestBaseWithParam<Interp_Size_Size_t> ResizeParams_CoordMode;

PERF_TEST_P(ResizeParams_CoordMode, resize_HalfPixel,
            testing::Values(
                Interp_Size_Size_t(INTER_LINEAR, sz1080p, szVGA),
                Interp_Size_Size_t(INTER_LINEAR, szVGA, sz1080p),
                Interp_Size_Size_t(INTER_NEAREST, sz1080p, szVGA),
                Interp_Size_Size_t(INTER_NEAREST, szVGA, sz1080p),
                Interp_Size_Size_t(INTER_CUBIC, sz1080p, szVGA),
                Interp_Size_Size_t(INTER_CUBIC, szVGA, sz1080p)
                )
            )
{
    int interp = get<0>(GetParam());
    Size from = get<1>(GetParam());
    Size to = get<2>(GetParam());

    Mat src(from, CV_32FC1), dst(to, CV_32FC1);
    cvtest::fillGradient<float>(src);
    declare.in(src).out(dst);

    ResizeParams params(to, 0, 0, interp);
    params.coordMode = ResizeCoord::HALF_PIXEL;

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
