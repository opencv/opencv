// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test {

CV_ENUM(MethodType, TM_SQDIFF, TM_SQDIFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_CCOEFF, TM_CCOEFF_NORMED)

typedef tuple<Size, Size, MethodType> ImgSize_TmplSize_Method_t;
typedef perf::TestBaseWithParam<ImgSize_TmplSize_Method_t> ImgSize_TmplSize_Method;

PERF_TEST_P(ImgSize_TmplSize_Method, matchTemplateSmall,
            testing::Combine(
                testing::Values(szSmall128, cv::Size(320, 240),
                                cv::Size(640, 480), cv::Size(800, 600),
                                cv::Size(1024, 768), cv::Size(1280, 1024)),
                testing::Values(cv::Size(12, 12), cv::Size(28, 9),
                                cv::Size(8, 30), cv::Size(16, 16)),
                MethodType::all()
                )
            )
{
    Size imgSz = get<0>(GetParam());
    Size tmplSz = get<1>(GetParam());
    int method = get<2>(GetParam());

    Mat img(imgSz, CV_8UC1);
    Mat tmpl(tmplSz, CV_8UC1);
    Mat result(imgSz - tmplSz + Size(1,1), CV_32F);

    declare
        .in(img, WARMUP_RNG)
        .in(tmpl, WARMUP_RNG)
        .out(result)
        .time(30);

    TEST_CYCLE() matchTemplate(img, tmpl, result, method);

    bool isNormed =
        method == TM_CCORR_NORMED ||
        method == TM_SQDIFF_NORMED ||
        method == TM_CCOEFF_NORMED;
    double eps = isNormed ? 1e-5
        : 255 * 255 * tmpl.total() * 1e-6;

    SANITY_CHECK(result, eps);
}

PERF_TEST_P(ImgSize_TmplSize_Method, matchTemplateBig,
            testing::Combine(
                testing::Values(cv::Size(1280, 1024)),
                testing::Values(cv::Size(1260, 1000), cv::Size(1261, 1013)),
                MethodType::all()
                )
    )
{
    Size imgSz = get<0>(GetParam());
    Size tmplSz = get<1>(GetParam());
    int method = get<2>(GetParam());

    Mat img(imgSz, CV_8UC1);
    Mat tmpl(tmplSz, CV_8UC1);
    Mat result(imgSz - tmplSz + Size(1,1), CV_32F);

    declare
        .in(img, WARMUP_RNG)
        .in(tmpl, WARMUP_RNG)
        .out(result)
        .time(30);

    TEST_CYCLE() matchTemplate(img, tmpl, result, method);

    bool isNormed =
        method == TM_CCORR_NORMED ||
        method == TM_SQDIFF_NORMED ||
        method == TM_CCOEFF_NORMED;
    double eps = isNormed ? 1e-6
        : 255.0 * 255.0 * (double)tmpl.total() * 1e-6;

    SANITY_CHECK(result, eps);
}

} // namespace
