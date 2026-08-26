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

    ResizeParams params;
    params.interpolation = interp;
    params.coordMode = ResizeCoordMode::HALF_PIXEL;

    TEST_CYCLE() resize(src, dst, to, params);

    SANITY_CHECK_NOTHING();
}

typedef tuple<MatType, int, Size, Size> MatInfo_N_Size_Size_t;
typedef TestBaseWithParam<MatInfo_N_Size_Size_t> ResizeParams_Batch;

// cv::resize's vector<Mat> batch kind: N single-channel planes to one common size, DNN-style.
PERF_TEST_P(ResizeParams_Batch, resize_VectorMat_Linear,
            testing::Values(
                MatInfo_N_Size_Size_t(CV_32FC1, 8, sz720p, Size(224, 224)),
                MatInfo_N_Size_Size_t(CV_32FC1, 32, sz720p, Size(224, 224))
                )
            )
{
    int matType = get<0>(GetParam());
    int n = get<1>(GetParam());
    Size from = get<2>(GetParam());
    Size to = get<3>(GetParam());

    std::vector<Mat> src(n), dst;
    for (int i = 0; i < n; i++)
    {
        src[i] = Mat(from, matType);
        cvtest::fillGradient<float>(src[i]);
    }

    ResizeParams params;
    params.interpolation = INTER_LINEAR;

    TEST_CYCLE() resize(src, dst, to, params);

    SANITY_CHECK_NOTHING();
}

typedef TestBaseWithParam<MatInfo_N_Size_Size_t> ResizeParams_NDTensor;

// cv::resize's other batch kind: one NCHW tensor, one shared table. Compare against
// resize_VectorMat_Linear above.
PERF_TEST_P(ResizeParams_NDTensor, resize_NCHW_Linear,
            testing::Values(
                MatInfo_N_Size_Size_t(CV_32FC1, 8, sz720p, Size(224, 224)),
                MatInfo_N_Size_Size_t(CV_32FC1, 32, sz720p, Size(224, 224)),
                MatInfo_N_Size_Size_t(CV_32FC1, 128, Size(16, 16), Size(8, 8))
                )
            )
{
    int matType = get<0>(GetParam());
    int n = get<1>(GetParam());
    Size from = get<2>(GetParam());
    Size to = get<3>(GetParam());

    int srcSizes[] = { n, 1, from.height, from.width };
    Mat src(4, srcSizes, matType);
    Mat srcView = src.reshape(1, n * from.height);
    cvtest::fillGradient<float>(srcView);

    ResizeParams params;
    params.interpolation = INTER_LINEAR;

    Mat dst;
    TEST_CYCLE() resize(src, dst, to, params);

    SANITY_CHECK_NOTHING();
}

} // namespace
