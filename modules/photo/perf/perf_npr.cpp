// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

namespace opencv_test
{

typedef perf::TestBaseWithParam<Size> Photo_NPR;

static Mat nprSource(const Size& size)
{
    Mat img = imread(perf::TestBase::getDataPath("cv/npr/test1.png"));
    CV_Assert(!img.empty());
    Mat src;
    resize(img, src, size, 0, 0, INTER_LINEAR);
    return src;
}

#define NPR_SIZES testing::Values(perf::szVGA, perf::sz720p, perf::sz1080p)

PERF_TEST_P(Photo_NPR, edgePreservingFilter_recursive, NPR_SIZES)
{
    Mat src = nprSource(GetParam()), dst(src.size(), CV_8UC3);
    declare.in(src).out(dst);

    TEST_CYCLE() edgePreservingFilter(src, dst, RECURS_FILTER);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Photo_NPR, edgePreservingFilter_normconv, NPR_SIZES)
{
    Mat src = nprSource(GetParam()), dst(src.size(), CV_8UC3);
    declare.in(src).out(dst);

    TEST_CYCLE() edgePreservingFilter(src, dst, NORMCONV_FILTER);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Photo_NPR, detailEnhance, NPR_SIZES)
{
    Mat src = nprSource(GetParam()), dst(src.size(), CV_8UC3);
    declare.in(src).out(dst);

    TEST_CYCLE() detailEnhance(src, dst);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Photo_NPR, pencilSketch, NPR_SIZES)
{
    Mat src = nprSource(GetParam()), gray(src.size(), CV_8UC1), color(src.size(), CV_8UC3);
    declare.in(src).out(gray, color);

    TEST_CYCLE() pencilSketch(src, gray, color);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Photo_NPR, stylization, NPR_SIZES)
{
    Mat src = nprSource(GetParam()), dst(src.size(), CV_8UC3);
    declare.in(src).out(dst);

    TEST_CYCLE() stylization(src, dst);

    SANITY_CHECK_NOTHING();
}

} // namespace
