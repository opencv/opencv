// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, Intel Corporation, all rights reserved.
#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

CV_ENUM(BorderTypeCopy, BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT_101)

// 8U/16U/16S/32S/32F x C1/C3/C4.
#define COPYMAKEBORDER_TYPES CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, \
                             CV_16SC1, CV_16SC3, CV_16SC4, CV_32SC1, CV_32SC3, CV_32SC4, \
                             CV_32FC1, CV_32FC3, CV_32FC4

typedef tuple<Size, MatType, BorderTypeCopy, int> Size_MatType_Border_BorderSize_t;
typedef perf::TestBaseWithParam<Size_MatType_Border_BorderSize_t> Size_MatType_Border_BorderSize;

PERF_TEST_P(Size_MatType_Border_BorderSize, copyMakeBorder,
            testing::Combine(
                testing::Values(szVGA, sz720p, sz1080p),
                testing::Values(COPYMAKEBORDER_TYPES),
                BorderTypeCopy::all(),
                testing::Values(1, 2, 6)
                ))
{
    Size sz         = get<0>(GetParam());
    int  type       = get<1>(GetParam());
    int  borderType = get<2>(GetParam());
    int  borderSize = get<3>(GetParam());

    Mat src(sz, type);
    Mat dst(sz + Size(borderSize * 2, borderSize * 2), type);

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE_MULTIRUN(10)
        cv::copyMakeBorder(src, dst, borderSize, borderSize, borderSize, borderSize,
                           borderType | BORDER_ISOLATED, Scalar(-50));

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Size_MatType_Border_BorderSize, copyMakeBorder_inplace,
            testing::Combine(
                testing::Values(szVGA, sz720p, sz1080p),
                testing::Values(COPYMAKEBORDER_TYPES),
                BorderTypeCopy::all(),
                testing::Values(1, 2, 6)
                ))
{
    Size sz         = get<0>(GetParam());
    int  type       = get<1>(GetParam());
    int  borderType = get<2>(GetParam());
    int  borderSize = get<3>(GetParam());

    Mat dst(sz + Size(borderSize * 2, borderSize * 2), type);
    Mat src = Mat(dst, Rect(borderSize, borderSize, sz.width, sz.height));

    declare.in(src, WARMUP_RNG);

    TEST_CYCLE_MULTIRUN(100)
        cv::copyMakeBorder(src, dst, borderSize, borderSize, borderSize, borderSize,
                           borderType | BORDER_ISOLATED, Scalar(-50));

    SANITY_CHECK_NOTHING();
}

} // namespace
