// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, Intel Corporation, all rights reserved.
#include "perf_precomp.hpp"

namespace opencv_test {

// 8U x C1/C3/C4 — depth/channel base shared by the type sweeps below.
#define BLUR_8U_TYPES CV_8UC1, CV_8UC3, CV_8UC4
// 8U/16U/16S x C1/C3/C4 — integer coverage shared by medianBlur and boxFilter.
#define BLUR_INT_TYPES BLUR_8U_TYPES, CV_16UC1, CV_16UC3, CV_16UC4, CV_16SC1, CV_16SC3, CV_16SC4
#define MEDIAN_BLUR_TYPES BLUR_INT_TYPES, CV_32FC1
// cv::medianBlur supports aperture > 5 for CV_8U only.
#define MEDIAN_BLUR_LARGE_TYPES BLUR_8U_TYPES
// Full boxFilter type sweep.
#define BOX_FILTER_TYPES BLUR_INT_TYPES, CV_32SC1, CV_32FC1, CV_32FC3, CV_32FC4
// 3x3 / 5x5 GaussianBlur type coverage.
#define GAUSSIAN_BLUR_TYPES BLUR_8U_TYPES, CV_16UC1, CV_16UC3, CV_16SC1, CV_16SC3, CV_32FC1, CV_32FC3
// Large Gaussian kernels (7/21) drop the C4 variants.
#define GAUSSIAN_BLUR_LARGE_TYPES CV_8UC1, CV_8UC3, CV_16UC1, CV_16UC3, CV_16SC1, CV_16SC3, CV_32FC1, CV_32FC3

typedef tuple<Size, MatType, int> Size_MatType_kSize_t;
typedef perf::TestBaseWithParam<Size_MatType_kSize_t> Size_MatType_kSize;

PERF_TEST_P(Size_MatType_kSize, medianBlur,
            testing::Combine(
                testing::Values(szODD, szQVGA, szVGA, sz720p),
                testing::Values(MEDIAN_BLUR_TYPES),
                testing::Values(3, 5)
                )
            )
{
    Size size = get<0>(GetParam());
    int type = get<1>(GetParam());
    int ksize = get<2>(GetParam());

    Mat src(size, type);
    Mat dst(size, type);

    declare.in(src, WARMUP_RNG).out(dst);

    if (CV_MAT_DEPTH(type) > CV_16S || CV_MAT_CN(type) > 1)
        declare.time(15);

    TEST_CYCLE() medianBlur(src, dst, ksize);

    SANITY_CHECK(dst);
}

PERF_TEST_P(Size_MatType_kSize, medianBlur_large,
            testing::Combine(
                testing::Values(szODD, szQVGA, szVGA, sz720p),
                testing::Values(MEDIAN_BLUR_LARGE_TYPES),
                testing::Values(7, 21)
                )
            )
{
    Size size = get<0>(GetParam());
    int type = get<1>(GetParam());
    int ksize = get<2>(GetParam());

    Mat src(size, type);
    Mat dst(size, type);

    declare.in(src, WARMUP_RNG).out(dst);
    declare.time(15);

    TEST_CYCLE() medianBlur(src, dst, ksize);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Size_MatType_kSize, medianBlur_inplace,
            testing::Combine(
                testing::Values(szODD, szQVGA, szVGA, sz720p),
                testing::Values(MEDIAN_BLUR_TYPES),
                testing::Values(3, 5)
                )
            )
{
    Size size = get<0>(GetParam());
    int type = get<1>(GetParam());
    int ksize = get<2>(GetParam());

    Mat src(size, type);
    Mat dst(size, type);

    declare.in(src, WARMUP_RNG).out(dst);
    if (CV_MAT_DEPTH(type) > CV_16S || CV_MAT_CN(type) > 1)
        declare.time(15);

    while (next())
    {
        src.copyTo(dst);
        startTimer();
        cv::medianBlur(dst, dst, ksize);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

CV_ENUM(BorderType3x3, BORDER_REPLICATE, BORDER_CONSTANT)
CV_ENUM(BorderType, BORDER_REPLICATE, BORDER_CONSTANT, BORDER_REFLECT, BORDER_REFLECT101)

typedef tuple<Size, MatType, BorderType3x3> Size_MatType_BorderType3x3_t;
typedef perf::TestBaseWithParam<Size_MatType_BorderType3x3_t> Size_MatType_BorderType3x3;

typedef tuple<Size, MatType, BorderType> Size_MatType_BorderType_t;
typedef perf::TestBaseWithParam<Size_MatType_BorderType_t> Size_MatType_BorderType;

typedef tuple<Size, int, BorderType3x3> Size_ksize_BorderType_t;
typedef perf::TestBaseWithParam<Size_ksize_BorderType_t> Size_ksize_BorderType;

typedef tuple<Size, MatType, BorderType, int> Size_MatType_BorderType_ksize_t;
typedef perf::TestBaseWithParam<Size_MatType_BorderType_ksize_t> Size_MatType_BorderType_ksize;


PERF_TEST_P(Size_MatType_BorderType3x3, gaussianBlur3x3,
            testing::Combine(
                testing::Values(szODD, szQVGA, szVGA, sz720p),
                testing::Values(GAUSSIAN_BLUR_TYPES),
                BorderType3x3::all()
                )
            )
{
    Size size = get<0>(GetParam());
    int type = get<1>(GetParam());
    BorderType3x3 btype = get<2>(GetParam());

    Mat src(size, type);
    Mat dst(size, type);

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE() GaussianBlur(src, dst, Size(3,3), 0, 0, btype);

    SANITY_CHECK(dst, 1);
}

PERF_TEST_P(Size_MatType_BorderType3x3, blur3x3,
            testing::Combine(
                testing::Values(szODD, szQVGA, szVGA, sz720p),
                testing::Values(CV_8UC1, CV_8UC4, CV_16UC1, CV_16SC1, CV_32FC1),
                BorderType3x3::all()
                )
            )
{
    Size size = get<0>(GetParam());
    int type = get<1>(GetParam());
    BorderType3x3 btype = get<2>(GetParam());

    Mat src(size, type);
    Mat dst(size, type);

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE() blur(src, dst, Size(3,3), Point(-1,-1), btype);

    SANITY_CHECK(dst, 1);
}

PERF_TEST_P(Size_MatType_BorderType, blur16x16,
            testing::Combine(
                testing::Values(szVGA, sz720p),
                testing::Values(CV_8UC1, CV_8UC4, CV_16UC1, CV_16SC1, CV_32FC1),
                BorderType::all()
                )
            )
{
    Size size = get<0>(GetParam());
    int type = get<1>(GetParam());
    BorderType btype = get<2>(GetParam());
    double eps = 1.25e-3;

    eps = CV_MAT_DEPTH(type) <= CV_32S ? 1 : eps;

    Mat src(size, type);
    Mat dst(size, type);

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE() blur(src, dst, Size(16,16), Point(-1,-1), btype);

    SANITY_CHECK(dst, eps);
}

PERF_TEST_P(Size_MatType_BorderType_ksize, box,
            testing::Combine(
                testing::Values(szODD, szQVGA, szVGA, sz720p),
                testing::Values(BOX_FILTER_TYPES),
                BorderType::all(),
                testing::Values(3, 5, 7, 21)
                )
            )
{
    auto p = GetParam();
    Size       size  = get<0>(p);
    int        type  = get<1>(p);
    BorderType btype = get<2>(p);
    int        ksize = get<3>(p);

    Mat src(size, type);
    Mat dst(size, type);

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE() boxFilter(src, dst, -1, Size(ksize, ksize), Point(-1,-1), false, btype);

    SANITY_CHECK(dst, 1e-6, ERROR_RELATIVE);
}

PERF_TEST_P(Size_ksize_BorderType, box_CV8U_CV16U,
            testing::Combine(
                    testing::Values(szODD, szQVGA, szVGA, sz720p),
                    testing::Values(3, 5, 15),
                    BorderType3x3::all()
                    )
            )
{
    Size size = get<0>(GetParam());
    int ksize = get<1>(GetParam());
    BorderType3x3 btype = get<2>(GetParam());

    Mat src(size, CV_8UC1);
    Mat dst(size, CV_16UC1);

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE() boxFilter(src, dst, CV_16UC1, Size(ksize, ksize), Point(-1,-1), false, btype);

    SANITY_CHECK(dst, 1e-6, ERROR_RELATIVE);
}

PERF_TEST_P(Size_MatType_BorderType_ksize, box_inplace,
            testing::Combine(
                testing::Values(szODD, szQVGA, szVGA, sz720p),
                testing::Values(CV_8UC1, CV_16SC1, CV_32SC1, CV_32FC1, CV_32FC3),
                BorderType::all(),
                testing::Values(3, 5)
                )
            )
{
    auto p = GetParam();
    Size       size  = get<0>(p);
    int        type  = get<1>(p);
    BorderType btype = get<2>(p);
    int        ksize = get<3>(p);

    Mat src(size, type);
    Mat dst(size, type);

    declare.in(src, WARMUP_RNG).out(dst);

    while(next())
    {
        src.copyTo(dst);
        startTimer();
        boxFilter(dst, dst, -1, Size(ksize, ksize), Point(-1,-1), false, btype);
        stopTimer();
    }

    SANITY_CHECK(dst, 1e-6, ERROR_RELATIVE);
}

PERF_TEST_P(Size_MatType_BorderType, gaussianBlur5x5,
            testing::Combine(
                testing::Values(szODD, szQVGA, szVGA, sz720p),
                testing::Values(GAUSSIAN_BLUR_TYPES),
                BorderType::all()
                )
            )
{
    Size size = get<0>(GetParam());
    int type = get<1>(GetParam());
    BorderType btype = get<2>(GetParam());

    Mat src(size, type);
    Mat dst(size, type);

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE() GaussianBlur(src, dst, Size(5,5), 0, 0, btype);

    SANITY_CHECK(dst, 1);
}

PERF_TEST_P(Size_MatType_BorderType_ksize, gaussianBlur,
            testing::Combine(
                testing::Values(szVGA, sz720p),
                testing::Values(GAUSSIAN_BLUR_LARGE_TYPES),
                BorderType::all(),
                testing::Values(7, 21)
                )
            )
{
    auto p = GetParam();
    Size       size  = get<0>(p);
    int        type  = get<1>(p);
    BorderType btype = get<2>(p);
    int        ksize = get<3>(p);

    Mat src(size, type);
    Mat dst(size, type);

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE() GaussianBlur(src, dst, Size(ksize, ksize), 0, 0, btype);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Size_MatType_BorderType, blur5x5,
            testing::Combine(
                testing::Values(szVGA, sz720p),
                testing::Values(CV_8UC1, CV_8UC4, CV_16UC1, CV_16SC1, CV_32FC1, CV_32FC3),
                BorderType::all()
                )
            )
{
    Size size = get<0>(GetParam());
    int type = get<1>(GetParam());
    BorderType btype = get<2>(GetParam());

    Mat src(size, type);
    Mat dst(size, type);

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE() blur(src, dst, Size(5,5), Point(-1,-1), btype);

    SANITY_CHECK(dst, 1);
}

///////////// BlendLinear ////////////////////////
PERF_TEST_P(Size_MatType, BlendLinear,
            testing::Combine(
                testing::Values(szVGA, sz720p, sz1080p, sz2160p),
                testing::Values(CV_8UC1, CV_32FC1, CV_8UC3, CV_32FC3, CV_8UC4, CV_32FC4)
                )
           )
{
    const Size srcSize = get<0>(GetParam());
    const int srcType = get<1>(GetParam());

    Mat src1(srcSize, srcType), src2(srcSize, srcType), dst(srcSize, srcType);
    Mat weights1(srcSize, CV_32FC1), weights2(srcSize, CV_32FC1);

    declare.in(src1, src2, WARMUP_RNG).in(weights1, weights2, WARMUP_READ).out(dst);
    randu(weights1, 0, 1);
    randu(weights2, 0, 1);

    TEST_CYCLE() blendLinear(src1, src2, weights1, weights2, dst);

    SANITY_CHECK_NOTHING();
}

///////////// Stackblur ////////////////////////
PERF_TEST_P(Size_MatType, stackblur3x3,
            testing::Combine(
                    testing::Values(sz720p, sz1080p, sz2160p),
                    testing::Values(CV_8UC1, CV_8UC4, CV_16UC1, CV_16SC1, CV_32FC1)
            )
)
{
    Size size = get<0>(GetParam());
    int type = get<1>(GetParam());
    double eps = 1e-3;

    eps = CV_MAT_DEPTH(type) <= CV_32S ? 1 : eps;

    Mat src(size, type);
    Mat dst(size, type);

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE() stackBlur(src, dst, Size(3,3));

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Size_MatType, stackblur101x101,
            testing::Combine(
                    testing::Values(sz720p, sz1080p, sz2160p),
                    testing::Values(CV_8UC1, CV_8UC4, CV_16UC1, CV_16SC1, CV_32FC1)
            )
)
{
    Size size = get<0>(GetParam());
    int type = get<1>(GetParam());
    double eps = 1e-3;

    eps = CV_MAT_DEPTH(type) <= CV_32S ? 1 : eps;

    Mat src(size, type);
    Mat dst(size, type);

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE() stackBlur(src, dst, Size(101,101));

    SANITY_CHECK_NOTHING();
}


} // namespace
