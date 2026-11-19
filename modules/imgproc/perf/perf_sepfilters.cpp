// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test {

#define FILTER_SRC_SIZES szODD, szQVGA, szVGA

CV_ENUM(BorderType3x3, BORDER_REPLICATE, BORDER_CONSTANT)
CV_ENUM(BorderType3x3ROI, BORDER_DEFAULT, BORDER_REPLICATE|BORDER_ISOLATED, BORDER_CONSTANT|BORDER_ISOLATED)

CV_ENUM(BorderType, BORDER_REPLICATE, BORDER_CONSTANT, BORDER_REFLECT, BORDER_REFLECT101)
CV_ENUM(BorderTypeROI, BORDER_DEFAULT, BORDER_REPLICATE|BORDER_ISOLATED, BORDER_CONSTANT|BORDER_ISOLATED, BORDER_REFLECT|BORDER_ISOLATED, BORDER_REFLECT101|BORDER_ISOLATED)

typedef tuple<Size, MatType, tuple<int, int>, BorderType3x3> Size_MatType_dx_dy_Border3x3_t;
typedef perf::TestBaseWithParam<Size_MatType_dx_dy_Border3x3_t> Size_MatType_dx_dy_Border3x3;

typedef tuple<Size, MatType, tuple<int, int>, BorderType3x3ROI> Size_MatType_dx_dy_Border3x3ROI_t;
typedef perf::TestBaseWithParam<Size_MatType_dx_dy_Border3x3ROI_t> Size_MatType_dx_dy_Border3x3ROI;

typedef tuple<Size, MatType, tuple<int, int>, BorderType> Size_MatType_dx_dy_Border5x5_t;
typedef perf::TestBaseWithParam<Size_MatType_dx_dy_Border5x5_t> Size_MatType_dx_dy_Border5x5;

typedef tuple<Size, MatType, tuple<int, int>, BorderTypeROI> Size_MatType_dx_dy_Border5x5ROI_t;
typedef perf::TestBaseWithParam<Size_MatType_dx_dy_Border5x5ROI_t> Size_MatType_dx_dy_Border5x5ROI;


/**************** Sobel ********************/

PERF_TEST_P(Size_MatType_dx_dy_Border3x3, sobelFilter,
            testing::Combine(
                testing::Values(FILTER_SRC_SIZES),
                testing::Values(CV_16S, CV_32F),
                testing::Values(make_tuple(0, 1), make_tuple(1, 0), make_tuple(1, 1), make_tuple(0, 2), make_tuple(2, 0), make_tuple(2, 2)),
                BorderType3x3::all()
            )
          )
{
    Size size = get<0>(GetParam());
    int ddepth = get<1>(GetParam());
    int dx = get<0>(get<2>(GetParam()));
    int dy = get<1>(get<2>(GetParam()));
    BorderType3x3 border = get<3>(GetParam());

    Mat src(size, CV_8U);
    Mat dst(size, ddepth);

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE() Sobel(src, dst, ddepth, dx, dy, 3, 1, 0, border);

    SANITY_CHECK(dst);
}

PERF_TEST_P(Size_MatType_dx_dy_Border3x3ROI, sobelFilter,
            testing::Combine(
                testing::Values(FILTER_SRC_SIZES),
                testing::Values(CV_16S, CV_32F),
                testing::Values(make_tuple(0, 1), make_tuple(1, 0), make_tuple(1, 1), make_tuple(0, 2), make_tuple(2, 0), make_tuple(2, 2)),
                BorderType3x3ROI::all()
            )
          )
{
    Size size = get<0>(GetParam());
    int ddepth = get<1>(GetParam());
    int dx = get<0>(get<2>(GetParam()));
    int dy = get<1>(get<2>(GetParam()));
    BorderType3x3ROI border = get<3>(GetParam());

    Mat src(size.height + 10, size.width + 10, CV_8U);
    Mat dst(size, ddepth);

    warmup(src, WARMUP_RNG);
    src = src(Range(5, 5 + size.height), Range(5, 5 + size.width));

    declare.in(src).out(dst);

    TEST_CYCLE() Sobel(src, dst, ddepth, dx, dy, 3, 1, 0, border);

    SANITY_CHECK(dst);
}

PERF_TEST_P(Size_MatType_dx_dy_Border5x5, sobelFilter,
            testing::Combine(
                testing::Values(FILTER_SRC_SIZES),
                testing::Values(CV_16S, CV_32F),
                testing::Values(make_tuple(0, 1), make_tuple(1, 0), make_tuple(1, 1), make_tuple(0, 2), make_tuple(2, 0)),
                BorderType::all()
            )
          )
{
    Size size = get<0>(GetParam());
    int ddepth = get<1>(GetParam());
    int dx = get<0>(get<2>(GetParam()));
    int dy = get<1>(get<2>(GetParam()));
    BorderType border = get<3>(GetParam());

    Mat src(size, CV_8U);
    Mat dst(size, ddepth);

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE() Sobel(src, dst, ddepth, dx, dy, 5, 1, 0, border);

    SANITY_CHECK(dst);
}

PERF_TEST_P(Size_MatType_dx_dy_Border5x5ROI, sobelFilter,
            testing::Combine(
                testing::Values(FILTER_SRC_SIZES),
                testing::Values(CV_16S, CV_32F),
                testing::Values(make_tuple(0, 1), make_tuple(1, 0), make_tuple(1, 1), make_tuple(0, 2), make_tuple(2, 0)),
                BorderTypeROI::all()
            )
          )
{
    Size size = get<0>(GetParam());
    int ddepth = get<1>(GetParam());
    int dx = get<0>(get<2>(GetParam()));
    int dy = get<1>(get<2>(GetParam()));
    BorderTypeROI border = get<3>(GetParam());

    Mat src(size.height + 10, size.width + 10, CV_8U);
    Mat dst(size, ddepth);

    warmup(src, WARMUP_RNG);
    src = src(Range(5, 5 + size.height), Range(5, 5 + size.width));

    declare.in(src).out(dst);

    TEST_CYCLE() Sobel(src, dst, ddepth, dx, dy, 5, 1, 0, border);

    SANITY_CHECK(dst);
}

/**************** Scharr ********************/

PERF_TEST_P(Size_MatType_dx_dy_Border3x3, scharrFilter,
            testing::Combine(
                testing::Values(FILTER_SRC_SIZES),
                testing::Values(CV_16S, CV_32F),
                testing::Values(make_tuple(0, 1), make_tuple(1, 0)),
                BorderType3x3::all()
            )
          )
{
    Size size = get<0>(GetParam());
    int ddepth = get<1>(GetParam());
    int dx = get<0>(get<2>(GetParam()));
    int dy = get<1>(get<2>(GetParam()));
    BorderType3x3 border = get<3>(GetParam());

    Mat src(size, CV_8U);
    Mat dst(size, ddepth);

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE() Scharr(src, dst, ddepth, dx, dy, 1, 0, border);

    SANITY_CHECK(dst);
}

PERF_TEST_P(Size_MatType_dx_dy_Border3x3ROI, scharrFilter,
            testing::Combine(
                testing::Values(FILTER_SRC_SIZES),
                testing::Values(CV_16S, CV_32F),
                testing::Values(make_tuple(0, 1), make_tuple(1, 0)),
                BorderType3x3ROI::all()
            )
          )
{
    Size size = get<0>(GetParam());
    int ddepth = get<1>(GetParam());
    int dx = get<0>(get<2>(GetParam()));
    int dy = get<1>(get<2>(GetParam()));
    BorderType3x3ROI border = get<3>(GetParam());

    Mat src(size.height + 10, size.width + 10, CV_8U);
    Mat dst(size, ddepth);

    warmup(src, WARMUP_RNG);
    src = src(Range(5, 5 + size.height), Range(5, 5 + size.width));

    declare.in(src).out(dst);

    TEST_CYCLE() Scharr(src, dst, ddepth, dx, dy, 1, 0, border);

    SANITY_CHECK(dst);
}

PERF_TEST_P(Size_MatType_dx_dy_Border3x3, scharrViaSobelFilter,
            testing::Combine(
                testing::Values(FILTER_SRC_SIZES),
                testing::Values(CV_16S, CV_32F),
                testing::Values(make_tuple(0, 1), make_tuple(1, 0)),
                BorderType3x3::all()
            )
          )
{
    Size size = get<0>(GetParam());
    int ddepth = get<1>(GetParam());
    int dx = get<0>(get<2>(GetParam()));
    int dy = get<1>(get<2>(GetParam()));
    BorderType3x3 border = get<3>(GetParam());

    Mat src(size, CV_8U);
    Mat dst(size, ddepth);

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE() Sobel(src, dst, ddepth, dx, dy, -1, 1, 0, border);

    SANITY_CHECK(dst);
}

PERF_TEST_P(Size_MatType_dx_dy_Border3x3ROI, scharrViaSobelFilter,
            testing::Combine(
                testing::Values(FILTER_SRC_SIZES),
                testing::Values(CV_16S, CV_32F),
                testing::Values(make_tuple(0, 1), make_tuple(1, 0)),
                BorderType3x3ROI::all()
            )
          )
{
    Size size = get<0>(GetParam());
    int ddepth = get<1>(GetParam());
    int dx = get<0>(get<2>(GetParam()));
    int dy = get<1>(get<2>(GetParam()));
    BorderType3x3ROI border = get<3>(GetParam());

    Mat src(size.height + 10, size.width + 10, CV_8U);
    Mat dst(size, ddepth);

    warmup(src, WARMUP_RNG);
    src = src(Range(5, 5 + size.height), Range(5, 5 + size.width));

    declare.in(src).out(dst);

    TEST_CYCLE() Sobel(src, dst, ddepth, dx, dy, -1, 1, 0, border);

    SANITY_CHECK(dst);
}

typedef tuple<Size, MatType, int> Filter2D_separable_t;
typedef perf::TestBaseWithParam<Filter2D_separable_t> Filter2D_separable;

PERF_TEST_P(Filter2D_separable, filter2D,
            testing::Combine(
                testing::Values(szVGA, sz1080p),
                testing::Values(CV_8U, CV_32F),
                testing::Values(9, 11, 15)
            ))
{
    Size size = get<0>(GetParam());
    int depth = get<1>(GetParam());
    int k = get<2>(GetParam());

    Mat src(size, depth);
    Mat dst(size, depth);
    warmup(src, WARMUP_RNG);

    Mat kx = cv::getGaussianKernel(k, k * 0.25, CV_32F);
    Mat ky = cv::getGaussianKernel(k, k * 0.25, CV_32F).t();
    Mat K = kx * ky;

    declare.in(src).out(dst);

    TEST_CYCLE() cv::filter2D(src, dst, -1, K, Point(-1, -1), 0, BORDER_REPLICATE);

    SANITY_CHECK(dst);
}

PERF_TEST_P(Filter2D_separable, sepFilter2D,
            testing::Combine(
                testing::Values(szVGA, sz1080p),
                testing::Values(CV_8U, CV_32F),
                testing::Values(9, 11, 15)
            ))
{
    Size size = get<0>(GetParam());
    int depth = get<1>(GetParam());
    int k = get<2>(GetParam());

    Mat src(size, depth);
    Mat dst(size, depth);
    warmup(src, WARMUP_RNG);

    Mat kx = cv::getGaussianKernel(k, k * 0.25, CV_32F);
    Mat ky = cv::getGaussianKernel(k, k * 0.25, CV_32F).t();

    declare.in(src).out(dst);

    TEST_CYCLE() cv::sepFilter2D(src, dst, -1, kx, ky, Point(-1, -1), 0, BORDER_REPLICATE);

    SANITY_CHECK(dst);
}

} // namespace
