// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, Intel Corporation, all rights reserved.
#include "perf_precomp.hpp"

namespace opencv_test {

CV_ENUM(BorderMode, BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT_101)
CV_ENUM(TargetDepth, CV_8U, CV_16S, CV_32F)
CV_ENUM(SrcDepth, CV_8U, CV_32F)

typedef tuple<Size, int, TargetDepth, BorderMode> LaplacianParams;
typedef perf::TestBaseWithParam<LaplacianParams> Perf_Laplacian;

PERF_TEST_P(Perf_Laplacian, Laplacian,
            testing::Combine(
                testing::Values(szVGA, sz720p, sz1080p),
                testing::Values(1, 3, 5),                // ksize: 1, 3, 5
                TargetDepth::all(),                      // CV_8U, CV_16S and CV_32F
                BorderMode::all()
            ))
{
    Size sz        = get<0>(GetParam());
    int ksize      = get<1>(GetParam());
    int ddepth     = get<2>(GetParam());
    int borderMode = get<3>(GetParam());

    Mat src(sz, CV_8UC1);
    Mat dst(sz, ddepth == CV_16S ? CV_16SC1 : ddepth == CV_32F ? CV_32FC1 : CV_8UC1);

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE()
    {
        cv::Laplacian(src, dst, ddepth, ksize, 1.0, 0.0, borderMode);
    }

    SANITY_CHECK(dst);
}

// Float source + scale/delta sweep (matches IPP perf coverage: 32F src -> 32F dst,
// scaled∈{0,1}: scale=1.333, delta=2). cv::Laplacian requires ddepth >= sdepth,
// so a 32F source cannot produce a 16S output.
typedef tuple<Size, int, TargetDepth, BorderMode, bool> LaplacianParams32f;
typedef perf::TestBaseWithParam<LaplacianParams32f> Perf_Laplacian32f;

PERF_TEST_P(Perf_Laplacian32f, Laplacian,
            testing::Combine(
                testing::Values(szVGA, sz720p, sz1080p),
                testing::Values(1, 3, 5),
                testing::Values((int)CV_32F),
                BorderMode::all(),
                testing::Bool()
            ))
{
    Size sz        = get<0>(GetParam());
    int ksize      = get<1>(GetParam());
    int ddepth     = get<2>(GetParam());
    int borderMode = get<3>(GetParam());
    bool scaled    = get<4>(GetParam());
    double scale   = scaled ? 1.333 : 1.0;
    double delta   = scaled ? 2.0 : 0.0;

    Mat src(sz, CV_32FC1);
    Mat dst(sz, ddepth);

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE()
    {
        cv::Laplacian(src, dst, ddepth, ksize, scale, delta, borderMode);
    }

    SANITY_CHECK_NOTHING();
}

// 8U source + scale/delta sweep (matches IPP: 8U src -> 8U/16S/32F dst, scaled).
// The base Perf_Laplacian test above keeps its SANITY_CHECK baseline (scale=1);
// this variant adds the scaled path without disturbing that baseline.
typedef tuple<Size, int, TargetDepth, BorderMode> LaplacianParamsScaled;
typedef perf::TestBaseWithParam<LaplacianParamsScaled> Perf_Laplacian_scaled;

PERF_TEST_P(Perf_Laplacian_scaled, Laplacian,
            testing::Combine(
                testing::Values(szVGA, sz720p, sz1080p),
                testing::Values(1, 3, 5),
                TargetDepth::all(),                      // CV_8U, CV_16S, CV_32F
                BorderMode::all()
            ))
{
    Size sz        = get<0>(GetParam());
    int ksize      = get<1>(GetParam());
    int ddepth     = get<2>(GetParam());
    int borderMode = get<3>(GetParam());

    Mat src(sz, CV_8UC1);
    Mat dst(sz, ddepth == CV_16S ? CV_16SC1 : ddepth == CV_32F ? CV_32FC1 : CV_8UC1);

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE()
    {
        cv::Laplacian(src, dst, ddepth, ksize, 1.333, 2.0, borderMode);
    }

    SANITY_CHECK_NOTHING();
}

} // namespace opencv_test