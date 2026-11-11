// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "perf_precomp.hpp"

namespace opencv_test {

// CV_32F is the baseline: the half depths run scalar kernels, so the gap
// against it is the current cost of storage-only support
typedef tuple<Size, MatType> Size_MatType_HalfFloat_t;
typedef perf::TestBaseWithParam<Size_MatType_HalfFloat_t> Size_MatType_HalfFloat;

#define HALF_FLOAT_TYPES testing::Values(CV_16FC1, CV_16FC3, CV_16BFC1, CV_32FC1)
#define HALF_FLOAT_SIZES testing::Values(szVGA, sz720p)

static Mat halfFloatSrc(Size size, int type)
{
    Mat src32(size, CV_MAKETYPE(CV_32F, CV_MAT_CN(type))), src;
    cv::randu(src32, Scalar::all(0), Scalar::all(1));
    src32.convertTo(src, type);
    return src;
}

PERF_TEST_P(Size_MatType_HalfFloat, resize_linear,
            testing::Combine(HALF_FLOAT_SIZES, HALF_FLOAT_TYPES))
{
    const Size size = get<0>(GetParam());
    const int type = get<1>(GetParam());

    Mat src = halfFloatSrc(size, type);
    // non-integer scale: an exact 2x INTER_LINEAR downscale is rerouted to INTER_AREA
    Mat dst(Size(size.width * 2 / 3, size.height * 2 / 3), type);
    declare.in(src).out(dst);

    TEST_CYCLE() resize(src, dst, dst.size(), 0, 0, INTER_LINEAR);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Size_MatType_HalfFloat, warpAffine_linear,
            testing::Combine(HALF_FLOAT_SIZES, HALF_FLOAT_TYPES))
{
    const Size size = get<0>(GetParam());
    const int type = get<1>(GetParam());

    Mat src = halfFloatSrc(size, type);
    Mat dst(size, type);
    Matx23d M(0.94, -0.08, 3.5,
              0.06,  0.97, -2.0);
    declare.in(src).out(dst);

    TEST_CYCLE() warpAffine(src, dst, Mat(M), size, INTER_LINEAR);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Size_MatType_HalfFloat, remap_linear,
            testing::Combine(HALF_FLOAT_SIZES, HALF_FLOAT_TYPES))
{
    const Size size = get<0>(GetParam());
    const int type = get<1>(GetParam());

    Mat src = halfFloatSrc(size, type);
    Mat dst(size, type);
    Mat mx(size, CV_32FC1), my(size, CV_32FC1);
    for (int y = 0; y < size.height; y++)
        for (int x = 0; x < size.width; x++)
        {
            mx.at<float>(y, x) = x + 0.375f;
            my.at<float>(y, x) = y + 0.375f;
        }
    declare.in(src).out(dst);

    TEST_CYCLE() remap(src, dst, mx, my, INTER_LINEAR);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Size_MatType_HalfFloat, GaussianBlur_5x5,
            testing::Combine(HALF_FLOAT_SIZES, HALF_FLOAT_TYPES))
{
    const Size size = get<0>(GetParam());
    const int type = get<1>(GetParam());

    Mat src = halfFloatSrc(size, type);
    Mat dst(size, type);
    declare.in(src).out(dst);
    declare.time(20);

    TEST_CYCLE() GaussianBlur(src, dst, Size(5, 5), 1.1, 1.1);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Size_MatType_HalfFloat, Sobel_3x3,
            testing::Combine(HALF_FLOAT_SIZES, HALF_FLOAT_TYPES))
{
    const Size size = get<0>(GetParam());
    const int type = get<1>(GetParam());

    Mat src = halfFloatSrc(size, type);
    Mat dst(size, type);
    declare.in(src).out(dst);
    declare.time(20);

    TEST_CYCLE() Sobel(src, dst, -1, 1, 0, 3);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Size_MatType_HalfFloat, filter2D_3x3,
            testing::Combine(HALF_FLOAT_SIZES, HALF_FLOAT_TYPES))
{
    const Size size = get<0>(GetParam());
    const int type = get<1>(GetParam());

    Mat src = halfFloatSrc(size, type);
    Mat dst(size, type);
    Mat kernel = Mat::ones(3, 3, CV_32F) / 9.f;
    declare.in(src).out(dst);
    declare.time(20);

    TEST_CYCLE() cv::filter2D(src, dst, -1, kernel);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Size_MatType_HalfFloat, blur_5x5,
            testing::Combine(HALF_FLOAT_SIZES, HALF_FLOAT_TYPES))
{
    const Size size = get<0>(GetParam());
    const int type = get<1>(GetParam());

    Mat src = halfFloatSrc(size, type);
    Mat dst(size, type);
    declare.in(src).out(dst);
    declare.time(20);

    TEST_CYCLE() blur(src, dst, Size(5, 5));

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Size_MatType_HalfFloat, erode_3x3,
            testing::Combine(HALF_FLOAT_SIZES, HALF_FLOAT_TYPES))
{
    const Size size = get<0>(GetParam());
    const int type = get<1>(GetParam());

    Mat src = halfFloatSrc(size, type);
    Mat dst(size, type);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    declare.in(src).out(dst);

    TEST_CYCLE() cv::erode(src, dst, kernel);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Size_MatType_HalfFloat, pyrDown,
            testing::Combine(HALF_FLOAT_SIZES, HALF_FLOAT_TYPES))
{
    const Size size = get<0>(GetParam());
    const int type = get<1>(GetParam());

    Mat src = halfFloatSrc(size, type);
    Mat dst(Size((size.width + 1) / 2, (size.height + 1) / 2), type);
    declare.in(src).out(dst);

    TEST_CYCLE() pyrDown(src, dst);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Size_MatType_HalfFloat, stackBlur_5x5,
            testing::Combine(HALF_FLOAT_SIZES, HALF_FLOAT_TYPES))
{
    const Size size = get<0>(GetParam());
    const int type = get<1>(GetParam());

    Mat src = halfFloatSrc(size, type);
    Mat dst(size, type);
    declare.in(src).out(dst);
    declare.time(20);

    TEST_CYCLE() stackBlur(src, dst, Size(5, 5));

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Size_MatType_HalfFloat, threshold_binary,
            testing::Combine(HALF_FLOAT_SIZES, HALF_FLOAT_TYPES))
{
    const Size size = get<0>(GetParam());
    const int type = get<1>(GetParam());

    Mat src = halfFloatSrc(size, type);
    Mat dst(size, type);
    declare.in(src).out(dst);

    TEST_CYCLE() cv::threshold(src, dst, 0.5, 1.0, THRESH_BINARY);

    SANITY_CHECK_NOTHING();
}

} // namespace opencv_test
