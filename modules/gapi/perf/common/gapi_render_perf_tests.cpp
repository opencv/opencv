// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include "../perf_precomp.hpp"

#include <iostream>

#include "./gapi_render_perf_tests.hpp"

#include "../../src/api/render_priv.hpp"

#ifdef HAVE_FREETYPE
#include <codecvt>
#endif // HAVE_FREETYPE

namespace opencv_test
{
  using namespace perf;

namespace {
void create_rand_mats(const cv::Size &size, MatType type, cv::Mat &ref_mat, cv::Mat &gapi_mat)
{
    ref_mat.create(size, type);
    cv::randu(ref_mat, cv::Scalar::all(0), cv::Scalar::all(255));
    ref_mat.copyTo(gapi_mat);
};

// FIXME: remove this code duplication with /gapi/test/common/gapi_render_tests.hpp
cv::Scalar cvtBGRToYUVC(const cv::Scalar &bgr)
{
    double y = bgr[2] * 0.299000 + bgr[1] * 0.587000 + bgr[0] * 0.114000;
    double u = bgr[2] * -0.168736 + bgr[1] * -0.331264 + bgr[0] * 0.500000 + 128;
    double v = bgr[2] * 0.500000 + bgr[1] * -0.418688 + bgr[0] * -0.081312 + 128;
    return {y, u, v};
}

// FIXME: remove this code duplication with /gapi/test/common/gapi_render_tests.hpp
void drawMosaicRef(const cv::Mat &mat, const cv::Rect &rect, int cellSz)
{
    cv::Rect mat_rect(0, 0, mat.cols, mat.rows);
    auto intersection = mat_rect & rect;

    cv::Mat msc_roi = mat(intersection);

    bool has_crop_x = false;
    bool has_crop_y = false;

    int cols = msc_roi.cols;
    int rows = msc_roi.rows;

    if (msc_roi.cols % cellSz != 0)
    {
        has_crop_x = true;
        cols -= msc_roi.cols % cellSz;
    }

    if (msc_roi.rows % cellSz != 0)
    {
        has_crop_y = true;
        rows -= msc_roi.rows % cellSz;
    }

    cv::Mat cell_roi;
    for (int i = 0; i < rows; i += cellSz)
    {
        for (int j = 0; j < cols; j += cellSz)
        {
            cell_roi = msc_roi(cv::Rect(j, i, cellSz, cellSz));
            cell_roi = cv::mean(cell_roi);
        }
        if (has_crop_x)
        {
            cell_roi = msc_roi(cv::Rect(cols, i, msc_roi.cols - cols, cellSz));
            cell_roi = cv::mean(cell_roi);
        }
    }

    if (has_crop_y)
    {
        for (int j = 0; j < cols; j += cellSz)
        {
            cell_roi = msc_roi(cv::Rect(j, rows, cellSz, msc_roi.rows - rows));
            cell_roi = cv::mean(cell_roi);
        }
        if (has_crop_x)
        {
            cell_roi = msc_roi(cv::Rect(cols, rows, msc_roi.cols - cols, msc_roi.rows - rows));
            cell_roi = cv::mean(cell_roi);
        }
    }
}

// FIXME: remove this code duplication with /gapi/test/common/gapi_render_tests.hpp
void blendImageRef(cv::Mat &mat, const cv::Point &org, const cv::Mat &img, const cv::Mat &alpha)
{
    auto roi = mat(cv::Rect(org, img.size()));
    cv::Mat img32f_w;
    cv::merge(std::vector<cv::Mat>(3, alpha), img32f_w);

    cv::Mat roi32f_w(roi.size(), CV_32FC3, cv::Scalar::all(1.0));
    roi32f_w -= img32f_w;

    cv::Mat img32f, roi32f;
    img.convertTo(img32f, CV_32F, 1.0 / 255);
    roi.convertTo(roi32f, CV_32F, 1.0 / 255);

    cv::multiply(img32f, img32f_w, img32f);
    cv::multiply(roi32f, roi32f_w, roi32f);
    roi32f += img32f;

    roi32f.convertTo(roi, CV_8U, 255.0);
}

#ifdef HAVE_FREETYPE
static std::wstring to_wstring(const char* bytes)
{
    std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
    return converter.from_bytes(bytes);
}
#endif // HAVE_FREETYPE

} // namespace

#ifdef HAVE_FREETYPE
INSTANTIATE_TEST_CASE_P(RenderTestFTexts, RenderTestFTexts,
                        Combine(Values(to_wstring("\xe4\xbd\xa0\xe5\xa5\xbd")),
                                Values(szVGA, sz720p, sz1080p),
                                Values(cv::Point(50, 50)),
                                Values(60),
                                Values(cv::Scalar(200, 100, 25)),
                                Values(CV_8UC3)));
#endif // HAVE_FREETYPE

INSTANTIATE_TEST_CASE_P(RenderTestTexts, RenderTestTexts,
                        Combine(Values(std::string("Some text")),
                                Values(szVGA, sz720p, sz1080p),
                                Values(cv::Point(200, 200)),
                                Values(FONT_HERSHEY_SIMPLEX),
                                Values(2.0),
                                Values(cv::Scalar(0, 255, 0)),
                                Values(2),
                                Values(LINE_8),
                                Values(false),
                                Values(CV_8UC3)));

INSTANTIATE_TEST_CASE_P(RenderTestRects, RenderTestRects,
                        Combine(Values(szVGA, sz720p, sz1080p),
                                Values(cv::Rect(100, 100, 200, 200)),
                                Values(cv::Scalar(100, 50, 150)),
                                Values(2),
                                Values(LINE_8),
                                Values(0),
                                Values(CV_8UC3),
                                Values(AbsExact().to_compare_f())));

INSTANTIATE_TEST_CASE_P(RenderTestCircles, RenderTestCircles,
                        Combine(Values(szVGA, sz720p, sz1080p),
                                Values(cv::Point(100, 100)),
                                Values(10),
                                Values(cv::Scalar(100, 50, 150)),
                                Values(2),
                                Values(LINE_8),
                                Values(0),
                                Values(CV_8UC3)));

INSTANTIATE_TEST_CASE_P(RenderTestLines, RenderTestLines,
                        Combine(Values(szVGA, sz720p, sz1080p),
                                Values(cv::Point(100, 100)),
                                Values(cv::Point(200, 200)),
                                Values(cv::Scalar(100, 50, 150)),
                                Values(2),
                                Values(LINE_8),
                                Values(0),
                                Values(CV_8UC3)));

INSTANTIATE_TEST_CASE_P(RenderTestMosaics, RenderTestMosaics,
                        Combine(Values(szVGA, sz720p, sz1080p),
                                Values(cv::Rect(100, 100, 200, 200)),
                                Values(25),
                                Values(0),
                                Values(CV_8UC3)));

INSTANTIATE_TEST_CASE_P(RenderTestImages, RenderTestImages,
                        Combine(Values(szVGA, sz720p, sz1080p),
                                Values(cv::Rect(50, 50, 100, 100)),
                                Values(cv::Scalar(100, 150, 60)),
                                Values(1.0),
                                Values(CV_8UC3)));

INSTANTIATE_TEST_CASE_P(RenderTestPolylines, RenderTestPolylines,
                        Combine(Values(szVGA, sz720p, sz1080p),
                                Values(std::vector<cv::Point>{{100, 100}, {200, 200}, {150, 300}, {400, 150}}),
                                Values(cv::Scalar(100, 150, 60)),
                                Values(2),
                                Values(LINE_8),
                                Values(0),
                                Values(CV_8UC3)));

INSTANTIATE_TEST_CASE_P(RenderTestPolyItems, RenderTestPolyItems,
                        Combine(Values(szVGA, sz720p, sz1080p),
                                Values(50),
                                Values(50),
                                Values(50)));


#ifdef HAVE_FREETYPE
PERF_TEST_P_(RenderTestFTexts, RenderFTextsPerformanceBGROCVTest)
{
    std::wstring text;
    cv::Size sz;
    cv::Point org;
    int fh;
    cv::Scalar color;
    MatType type;

    std::tie(text ,sz ,org ,fh ,color, type) = GetParam();

    cv::Mat gapi_mat = cv::Mat(sz, type);
    cv::Mat ref_mat(gapi_mat);

    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::FText{text, org, fh, color});

    cv::GMat in;
    cv::GArray<cv::gapi::wip::draw::Prim> arr;

    cv::GComputation comp(cv::GIn(in, arr),
                          cv::GOut(cv::gapi::wip::draw::render3ch(in, arr)));

    // Warm-up graph engine:
    comp.apply(gin(gapi_mat, prims), gout(gapi_mat));

    TEST_CYCLE()
    {
        comp.apply(gin(gapi_mat, prims), gout(gapi_mat));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestFTexts, RenderFTextsPerformanceNV12OCVTest)
{
    std::wstring text;
    cv::Size sz;
    cv::Point org;
    int fh;
    cv::Scalar color;
    MatType type;

    std::tie(text ,sz ,org ,fh ,color, type) = GetParam();

    cv::Mat y_ref_mat, uv_ref_mat;

    cv::Mat y_in_gapi_mat(sz, CV_8UC1), uv_in_gapi_mat(sz / 2, CV_8UC2),
        y_out_gapi_mat(sz, CV_8UC1), uv_out_gapi_mat(sz / 2, CV_8UC2);

    create_rand_mats(sz, CV_8UC1, y_ref_mat, y_in_gapi_mat);
    create_rand_mats(sz / 2, CV_8UC2, uv_ref_mat, uv_in_gapi_mat);

    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::FText{text, org, fh, color});

    cv::GMat y_in, uv_in, y_out, uv_out;
    cv::GArray<cv::gapi::wip::draw::Prim> arr;
    std::tie(y_out, uv_out) = cv::gapi::wip::draw::renderNV12(y_in, uv_in, arr);

    cv::GComputation comp(cv::GIn(y_in, uv_in, arr), cv::GOut(y_out, uv_out));

    // Warm-up graph engine:
    comp.apply(cv::gin(y_in_gapi_mat, uv_in_gapi_mat, prims),
               cv::gout(y_out_gapi_mat, uv_out_gapi_mat));

    TEST_CYCLE()
    {
        comp.apply(cv::gin(y_in_gapi_mat, uv_in_gapi_mat, prims),
                   cv::gout(y_out_gapi_mat, uv_out_gapi_mat));
    }

    SANITY_CHECK_NOTHING();
}
#endif // HAVE_FREETYPE

PERF_TEST_P_(RenderTestTexts, RenderTextsPerformanceBGROCVTest)
{
    cv::Point org;
    int ff, thick, lt;
    double fs;
    cv::Scalar color;
    bool blo;
    MatType type;
    std::string text;
    cv::Size sz;
    std::tie(text, sz, org, ff, fs, color, thick, lt, blo, type) = GetParam();

    cv::Mat gapi_mat = cv::Mat(sz, type);
    cv::Mat ref_mat(gapi_mat);

    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Text{text, org, ff, fs, color, thick, lt, blo});

    cv::GMat in;
    cv::GArray<cv::gapi::wip::draw::Prim> arr;

    cv::GComputation comp(cv::GIn(in, arr),
                          cv::GOut(cv::gapi::wip::draw::render3ch(in, arr)));

    // Warm-up graph engine:
    comp.apply(gin(gapi_mat, prims), gout(gapi_mat));

    TEST_CYCLE()
    {
        comp.apply(gin(gapi_mat, prims), gout(gapi_mat));
    }

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        cv::putText(ref_mat, text, org, ff, fs, color, thick, lt, blo);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::norm(gapi_mat, ref_mat, NORM_INF));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestTexts, RenderTextsPerformanceNV12OCVTest)
{
    cv::Point org;
    int ff, thick, lt;
    double fs;
    cv::Scalar color;
    bool blo;
    MatType type;
    std::string text;
    cv::Size sz;
    std::tie(text, sz, org, ff, fs, color, thick, lt, blo, type) = GetParam();

    cv::Mat y_ref_mat, uv_ref_mat;

    cv::Mat y_in_gapi_mat(sz, CV_8UC1), uv_in_gapi_mat(sz / 2, CV_8UC2),
        y_out_gapi_mat(sz, CV_8UC1), uv_out_gapi_mat(sz / 2, CV_8UC2);

    create_rand_mats(sz, CV_8UC1, y_ref_mat, y_in_gapi_mat);
    create_rand_mats(sz / 2, CV_8UC2, uv_ref_mat, uv_in_gapi_mat);

    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Text{text, org, ff, fs, color, thick, lt, blo});

    cv::GMat y_in, uv_in, y_out, uv_out;
    cv::GArray<cv::gapi::wip::draw::Prim> arr;
    std::tie(y_out, uv_out) = cv::gapi::wip::draw::renderNV12(y_in, uv_in, arr);

    cv::GComputation comp(cv::GIn(y_in, uv_in, arr), cv::GOut(y_out, uv_out));

    // Warm-up graph engine:
    comp.apply(cv::gin(y_in_gapi_mat, uv_in_gapi_mat, prims),
               cv::gout(y_out_gapi_mat, uv_out_gapi_mat));

    TEST_CYCLE()
    {
        comp.apply(cv::gin(y_in_gapi_mat, uv_in_gapi_mat, prims),
                   cv::gout(y_out_gapi_mat, uv_out_gapi_mat));
    }

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        // NV12 -> YUV
        cv::Mat yuv;
        cv::gapi::wip::draw::cvtNV12ToYUV(y_ref_mat, uv_ref_mat, yuv);

        cv::putText(yuv, text, org, ff, fs, cvtBGRToYUVC(color), thick, lt, blo);

        // YUV -> NV12
        cv::gapi::wip::draw::cvtYUVToNV12(yuv, y_ref_mat, uv_ref_mat);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::norm(y_out_gapi_mat, y_ref_mat, NORM_INF));
        EXPECT_EQ(0, cv::norm(uv_out_gapi_mat, uv_ref_mat, NORM_INF));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestRects, RenderRectsPerformanceBGROCVTest)
{
    cv::Rect rect;
    cv::Scalar color;
    int thick, lt, shift;
    MatType type;

    cv::Size sz;
    compare_f cmpF;
    std::tie(sz, rect, color, thick, lt, shift, type, cmpF) = GetParam();

    cv::Mat gapi_mat = cv::Mat(sz, type);
    cv::Mat ref_mat(gapi_mat);

    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Rect{rect, color, thick, lt, shift});

    cv::GMat in;
    cv::GArray<cv::gapi::wip::draw::Prim> arr;

    cv::GComputation comp(cv::GIn(in, arr),
                          cv::GOut(cv::gapi::wip::draw::render3ch(in, arr)));

    // Warm-up graph engine:
    comp.apply(gin(gapi_mat, prims), gout(gapi_mat));

    TEST_CYCLE()
    {
        comp.apply(gin(gapi_mat, prims), gout(gapi_mat));
    }

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        cv::rectangle(ref_mat, rect, color, thick, lt, shift);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::norm(gapi_mat, ref_mat, NORM_INF));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestRects, RenderRectsPerformanceNV12OCVTest)
{
    cv::Rect rect;
    cv::Scalar color;
    int thick, lt, shift;
    MatType type;

    cv::Size sz;
    compare_f cmpF;
    std::tie(sz, rect, color, thick, lt, shift, type, cmpF) = GetParam();

    cv::Mat y_ref_mat, uv_ref_mat;

    cv::Mat y_in_gapi_mat(sz, CV_8UC1), uv_in_gapi_mat(sz / 2, CV_8UC2),
        y_out_gapi_mat(sz, CV_8UC1), uv_out_gapi_mat(sz / 2, CV_8UC2);

    create_rand_mats(sz, CV_8UC1, y_ref_mat, y_in_gapi_mat);
    create_rand_mats(sz / 2, CV_8UC2, uv_ref_mat, uv_in_gapi_mat);

    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Rect{rect, color, thick, lt, shift});

    cv::GMat y_in, uv_in, y_out, uv_out;
    cv::GArray<cv::gapi::wip::draw::Prim> arr;
    std::tie(y_out, uv_out) = cv::gapi::wip::draw::renderNV12(y_in, uv_in, arr);

    cv::GComputation comp(cv::GIn(y_in, uv_in, arr), cv::GOut(y_out, uv_out));

    // Warm-up graph engine:
    comp.apply(cv::gin(y_in_gapi_mat, uv_in_gapi_mat, prims),
               cv::gout(y_out_gapi_mat, uv_out_gapi_mat));

    TEST_CYCLE()
    {
        comp.apply(cv::gin(y_in_gapi_mat, uv_in_gapi_mat, prims),
                   cv::gout(y_out_gapi_mat, uv_out_gapi_mat));
    }

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        // NV12 -> YUV
        cv::Mat yuv;
        cv::gapi::wip::draw::cvtNV12ToYUV(y_ref_mat, uv_ref_mat, yuv);

        cv::rectangle(yuv, rect, cvtBGRToYUVC(color), thick, lt, shift);

        // YUV -> NV12
        cv::gapi::wip::draw::cvtYUVToNV12(yuv, y_ref_mat, uv_ref_mat);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::norm(y_out_gapi_mat, y_ref_mat, NORM_INF));
        EXPECT_EQ(0, cv::norm(uv_out_gapi_mat, uv_ref_mat, NORM_INF));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestCircles, RenderCirclesPerformanceBGROCVTest)
{
    cv::Point center;
    int radius;
    cv::Scalar color;
    int thick;
    int lt;
    int shift;
    MatType type;

    cv::Size sz;
    std::tie(sz, center, radius, color, thick, lt, shift, type) = GetParam();

    cv::Mat gapi_mat = cv::Mat(sz, type);
    cv::Mat ref_mat(gapi_mat);

    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Circle{center, radius, color, thick, lt, shift});

    cv::GMat in;
    cv::GArray<cv::gapi::wip::draw::Prim> arr;

    cv::GComputation comp(cv::GIn(in, arr),
                          cv::GOut(cv::gapi::wip::draw::render3ch(in, arr)));

    // Warm-up graph engine:
    comp.apply(gin(gapi_mat, prims), gout(gapi_mat));

    TEST_CYCLE()
    {
        comp.apply(gin(gapi_mat, prims), gout(gapi_mat));
    }

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        cv::circle(ref_mat, center, radius, color, thick, lt, shift);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::norm(gapi_mat, ref_mat, NORM_INF));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestCircles, RenderCirclesPerformanceNV12OCVTest)
{
    cv::Point center;
    int radius;
    cv::Scalar color;
    int thick;
    int lt;
    int shift;
    MatType type;

    cv::Size sz;
    std::tie(sz, center, radius, color, thick, lt, shift, type) = GetParam();

    cv::Mat y_ref_mat, uv_ref_mat;

    cv::Mat y_in_gapi_mat(sz, CV_8UC1), uv_in_gapi_mat(sz / 2, CV_8UC2),
        y_out_gapi_mat(sz, CV_8UC1), uv_out_gapi_mat(sz / 2, CV_8UC2);

    create_rand_mats(sz, CV_8UC1, y_ref_mat, y_in_gapi_mat);
    create_rand_mats(sz / 2, CV_8UC2, uv_ref_mat, uv_in_gapi_mat);

    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Circle{center, radius, color, thick, lt, shift});

    cv::GMat y_in, uv_in, y_out, uv_out;
    cv::GArray<cv::gapi::wip::draw::Prim> arr;
    std::tie(y_out, uv_out) = cv::gapi::wip::draw::renderNV12(y_in, uv_in, arr);

    cv::GComputation comp(cv::GIn(y_in, uv_in, arr), cv::GOut(y_out, uv_out));

    // Warm-up graph engine:
    comp.apply(cv::gin(y_in_gapi_mat, uv_in_gapi_mat, prims),
               cv::gout(y_out_gapi_mat, uv_out_gapi_mat));

    TEST_CYCLE()
    {
        comp.apply(cv::gin(y_in_gapi_mat, uv_in_gapi_mat, prims),
                   cv::gout(y_out_gapi_mat, uv_out_gapi_mat));
    }

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        // NV12 -> YUV
        cv::Mat yuv;
        cv::gapi::wip::draw::cvtNV12ToYUV(y_ref_mat, uv_ref_mat, yuv);

        cv::circle(yuv, center, radius, cvtBGRToYUVC(color), thick, lt, shift);

        // YUV -> NV12
        cv::gapi::wip::draw::cvtYUVToNV12(yuv, y_ref_mat, uv_ref_mat);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::norm(y_out_gapi_mat, y_ref_mat, NORM_INF));
        EXPECT_EQ(0, cv::norm(uv_out_gapi_mat, uv_ref_mat, NORM_INF));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestLines, RenderLinesPerformanceBGROCVTest)
{
    cv::Point pt1;
    cv::Point pt2;
    cv::Scalar color;
    int thick;
    int lt;
    int shift;
    MatType type;

    compare_f cmpF;
    cv::Size sz;
    std::tie(sz, pt1, pt2, color, thick, lt, shift, type) = GetParam();

    cv::Mat gapi_mat = cv::Mat(sz, type);
    cv::Mat ref_mat(gapi_mat);

    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Line{pt1, pt2, color, thick, lt, shift});

    cv::GMat in;
    cv::GArray<cv::gapi::wip::draw::Prim> arr;

    cv::GComputation comp(cv::GIn(in, arr),
                          cv::GOut(cv::gapi::wip::draw::render3ch(in, arr)));

    // Warm-up graph engine:
    comp.apply(gin(gapi_mat, prims), gout(gapi_mat));

    TEST_CYCLE()
    {
        comp.apply(gin(gapi_mat, prims), gout(gapi_mat));
    }

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        cv::line(ref_mat, pt1, pt2, color, thick, lt, shift);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::norm(gapi_mat, ref_mat, NORM_INF));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestLines, RenderLinesPerformanceNV12OCVTest)
{
    cv::Point pt1;
    cv::Point pt2;
    cv::Scalar color;
    int thick;
    int lt;
    int shift;
    MatType type;

    compare_f cmpF;
    cv::Size sz;
    std::tie(sz, pt1, pt2, color, thick, lt, shift, type) = GetParam();

    cv::Mat y_ref_mat, uv_ref_mat;

    cv::Mat y_in_gapi_mat(sz, CV_8UC1), uv_in_gapi_mat(sz / 2, CV_8UC2),
        y_out_gapi_mat(sz, CV_8UC1), uv_out_gapi_mat(sz / 2, CV_8UC2);

    create_rand_mats(sz, CV_8UC1, y_ref_mat, y_in_gapi_mat);
    create_rand_mats(sz / 2, CV_8UC2, uv_ref_mat, uv_in_gapi_mat);

    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Line{pt1, pt2, color, thick, lt, shift});

    cv::GMat y_in, uv_in, y_out, uv_out;
    cv::GArray<cv::gapi::wip::draw::Prim> arr;
    std::tie(y_out, uv_out) = cv::gapi::wip::draw::renderNV12(y_in, uv_in, arr);

    cv::GComputation comp(cv::GIn(y_in, uv_in, arr), cv::GOut(y_out, uv_out));

    // Warm-up graph engine:
    comp.apply(cv::gin(y_in_gapi_mat, uv_in_gapi_mat, prims),
               cv::gout(y_out_gapi_mat, uv_out_gapi_mat));

    TEST_CYCLE()
    {
        comp.apply(cv::gin(y_in_gapi_mat, uv_in_gapi_mat, prims),
                   cv::gout(y_out_gapi_mat, uv_out_gapi_mat));
    }

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        // NV12 -> YUV
        cv::Mat yuv;
        cv::gapi::wip::draw::cvtNV12ToYUV(y_ref_mat, uv_ref_mat, yuv);

        cv::line(yuv, pt1, pt2, cvtBGRToYUVC(color), thick, lt, shift);

        // YUV -> NV12
        cv::gapi::wip::draw::cvtYUVToNV12(yuv, y_ref_mat, uv_ref_mat);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::norm(y_out_gapi_mat, y_ref_mat, NORM_INF));
        EXPECT_EQ(0, cv::norm(uv_out_gapi_mat, uv_ref_mat, NORM_INF));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestMosaics, RenderMosaicsPerformanceBGROCVTest)
{
    cv::Rect mos;
    int cellsz;
    int decim;
    MatType type;

    cv::Size sz;
    std::tie(sz, mos, cellsz, decim, type) = GetParam();

    cv::Mat gapi_mat = cv::Mat(sz, type);
    cv::Mat ref_mat(gapi_mat);

    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Mosaic{mos, cellsz, decim});

    cv::GMat in;
    cv::GArray<cv::gapi::wip::draw::Prim> arr;

    cv::GComputation comp(cv::GIn(in, arr),
                          cv::GOut(cv::gapi::wip::draw::render3ch(in, arr)));

    // Warm-up graph engine:
    comp.apply(gin(gapi_mat, prims), gout(gapi_mat));

    TEST_CYCLE()
    {
        comp.apply(gin(gapi_mat, prims), gout(gapi_mat));
    }

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        drawMosaicRef(ref_mat, mos, cellsz);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::norm(gapi_mat, ref_mat, NORM_INF));
    }

    SANITY_CHECK_NOTHING();
}


PERF_TEST_P_(RenderTestMosaics, RenderMosaicsPerformanceNV12OCVTest)
{
    cv::Rect mos;
    int cellsz;
    int decim;
    MatType type;

    cv::Size sz;
    std::tie(sz, mos, cellsz, decim, type) = GetParam();

    cv::Mat y_ref_mat, uv_ref_mat;

    cv::Mat y_in_gapi_mat(sz, CV_8UC1), uv_in_gapi_mat(sz / 2, CV_8UC2),
        y_out_gapi_mat(sz, CV_8UC1), uv_out_gapi_mat(sz / 2, CV_8UC2);

    create_rand_mats(sz, CV_8UC1, y_ref_mat, y_in_gapi_mat);
    create_rand_mats(sz / 2, CV_8UC2, uv_ref_mat, uv_in_gapi_mat);

    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Mosaic{mos, cellsz, decim});

    cv::GMat y_in, uv_in, y_out, uv_out;
    cv::GArray<cv::gapi::wip::draw::Prim> arr;
    std::tie(y_out, uv_out) = cv::gapi::wip::draw::renderNV12(y_in, uv_in, arr);

    cv::GComputation comp(cv::GIn(y_in, uv_in, arr), cv::GOut(y_out, uv_out));

    // Warm-up graph engine:
    comp.apply(cv::gin(y_in_gapi_mat, uv_in_gapi_mat, prims),
               cv::gout(y_out_gapi_mat, uv_out_gapi_mat));

    TEST_CYCLE()
    {
        comp.apply(cv::gin(y_in_gapi_mat, uv_in_gapi_mat, prims),
                   cv::gout(y_out_gapi_mat, uv_out_gapi_mat));
    }

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        // NV12 -> YUV
        cv::Mat yuv;
        cv::gapi::wip::draw::cvtNV12ToYUV(y_ref_mat, uv_ref_mat, yuv);

        drawMosaicRef(yuv, mos, cellsz);

        // YUV -> NV12
        cv::gapi::wip::draw::cvtYUVToNV12(yuv, y_ref_mat, uv_ref_mat);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::norm(y_out_gapi_mat, y_ref_mat, NORM_INF));
        EXPECT_EQ(0, cv::norm(uv_out_gapi_mat, uv_ref_mat, NORM_INF));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestImages, RenderImagesPerformanceBGROCVTest)
{
    cv::Rect rect;
    cv::Scalar color;
    double transparency;
    MatType type;

    cv::Size sz;
    std::tie(sz, rect, color, transparency, type) = GetParam();

    cv::Mat gapi_mat = cv::Mat(sz, type);
    cv::Mat ref_mat(gapi_mat);

    cv::Mat img(rect.size(), CV_8UC3, color);
    cv::Mat alpha(rect.size(), CV_32FC1, transparency);
    auto tl = rect.tl();
    cv::Point org = {tl.x, tl.y + rect.size().height};

    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Image{org, img, alpha});
    cv::gapi::wip::draw::render(gapi_mat, prims);

    cv::GMat in;
    cv::GArray<cv::gapi::wip::draw::Prim> arr;

    cv::GComputation comp(cv::GIn(in, arr),
                          cv::GOut(cv::gapi::wip::draw::render3ch(in, arr)));

    // Warm-up graph engine:
    comp.apply(gin(gapi_mat, prims), gout(gapi_mat));

    TEST_CYCLE()
    {
        comp.apply(gin(gapi_mat, prims), gout(gapi_mat));
    }

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        blendImageRef(ref_mat, org, img, alpha);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::norm(gapi_mat, ref_mat, NORM_INF));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestImages, RenderImagesPerformanceNV12OCVTest)
{
    cv::Rect rect;
    cv::Scalar color;
    double transparency;
    MatType type;

    cv::Size sz;
    std::tie(sz, rect, color, transparency, type) = GetParam();

    cv::Mat y_ref_mat, uv_ref_mat;

    cv::Mat y_in_gapi_mat(sz, CV_8UC1), uv_in_gapi_mat(sz / 2, CV_8UC2),
        y_out_gapi_mat(sz, CV_8UC1), uv_out_gapi_mat(sz / 2, CV_8UC2);

    create_rand_mats(sz, CV_8UC1, y_ref_mat, y_in_gapi_mat);
    create_rand_mats(sz / 2, CV_8UC2, uv_ref_mat, uv_in_gapi_mat);

    cv::Mat img(rect.size(), CV_8UC3, color);
    cv::Mat alpha(rect.size(), CV_32FC1, transparency);
    auto tl = rect.tl();
    cv::Point org = {tl.x, tl.y + rect.size().height};

    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Image{org, img, alpha});

    cv::GMat y_in, uv_in, y_out, uv_out;
    cv::GArray<cv::gapi::wip::draw::Prim> arr;
    std::tie(y_out, uv_out) = cv::gapi::wip::draw::renderNV12(y_in, uv_in, arr);

    cv::GComputation comp(cv::GIn(y_in, uv_in, arr), cv::GOut(y_out, uv_out));

    // Warm-up graph engine:
    comp.apply(cv::gin(y_in_gapi_mat, uv_in_gapi_mat, prims),
               cv::gout(y_out_gapi_mat, uv_out_gapi_mat));

    TEST_CYCLE()
    {
        comp.apply(cv::gin(y_in_gapi_mat, uv_in_gapi_mat, prims),
                   cv::gout(y_out_gapi_mat, uv_out_gapi_mat));
    }

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        // NV12 -> YUV
        cv::Mat yuv;
        cv::gapi::wip::draw::cvtNV12ToYUV(y_ref_mat, uv_ref_mat, yuv);

        cv::Mat yuv_img;
        cv::cvtColor(img, yuv_img, cv::COLOR_BGR2YUV);
        blendImageRef(yuv, org, yuv_img, alpha);

        // YUV -> NV12
        cv::gapi::wip::draw::cvtYUVToNV12(yuv, y_ref_mat, uv_ref_mat);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::norm(y_out_gapi_mat, y_ref_mat, NORM_INF));
        EXPECT_EQ(0, cv::norm(uv_out_gapi_mat, uv_ref_mat, NORM_INF));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestPolylines, RenderPolylinesPerformanceBGROCVTest)
{
    cv::Size sz;
    std::vector<cv::Point> points;
    cv::Scalar color;
    int thick;
    int lt;
    int shift;
    MatType type;
    std::tie(sz, points, color, thick, lt, shift, type) = GetParam();

    cv::Mat gapi_mat = cv::Mat(sz, type);
    cv::Mat ref_mat(gapi_mat);

    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Poly{points, color, thick, lt, shift});

    cv::GMat in;
    cv::GArray<cv::gapi::wip::draw::Prim> arr;

    cv::GComputation comp(cv::GIn(in, arr),
                          cv::GOut(cv::gapi::wip::draw::render3ch(in, arr)));

    // Warm-up graph engine:
    comp.apply(gin(gapi_mat, prims), gout(gapi_mat));

    TEST_CYCLE()
    {
        comp.apply(gin(gapi_mat, prims), gout(gapi_mat));
    }

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        std::vector<std::vector<cv::Point>> array_points{points};
        cv::fillPoly(ref_mat, array_points, color, lt, shift);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::norm(gapi_mat, ref_mat, NORM_INF));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestPolylines, RenderPolylinesPerformanceNV12OCVTest)
{
    std::vector<cv::Point> points;
    cv::Scalar color;
    int thick;
    int lt;
    int shift;
    MatType type;

    cv::Size sz;
    std::tie(sz, points, color, thick, lt, shift, type) = GetParam();

    cv::Mat y_ref_mat, uv_ref_mat;

    cv::Mat y_in_gapi_mat(sz, CV_8UC1), uv_in_gapi_mat(sz / 2, CV_8UC2),
        y_out_gapi_mat(sz, CV_8UC1), uv_out_gapi_mat(sz / 2, CV_8UC2);

    create_rand_mats(sz, CV_8UC1, y_ref_mat, y_in_gapi_mat);
    create_rand_mats(sz / 2, CV_8UC2, uv_ref_mat, uv_in_gapi_mat);

    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Poly{points, color, thick, lt, shift});

    cv::GMat y_in, uv_in, y_out, uv_out;
    cv::GArray<cv::gapi::wip::draw::Prim> arr;
    std::tie(y_out, uv_out) = cv::gapi::wip::draw::renderNV12(y_in, uv_in, arr);

    cv::GComputation comp(cv::GIn(y_in, uv_in, arr), cv::GOut(y_out, uv_out));

    // Warm-up graph engine:
    comp.apply(cv::gin(y_in_gapi_mat, uv_in_gapi_mat, prims),
               cv::gout(y_out_gapi_mat, uv_out_gapi_mat));

    TEST_CYCLE()
    {
        comp.apply(cv::gin(y_in_gapi_mat, uv_in_gapi_mat, prims),
                   cv::gout(y_out_gapi_mat, uv_out_gapi_mat));
    }

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        // NV12 -> YUV
        cv::Mat yuv;
        cv::gapi::wip::draw::cvtNV12ToYUV(y_ref_mat, uv_ref_mat, yuv);

        std::vector<std::vector<cv::Point>> pp{points};
        cv::fillPoly(yuv, pp, cvtBGRToYUVC(color), lt, shift);

        // YUV -> NV12
        cv::gapi::wip::draw::cvtYUVToNV12(yuv, y_ref_mat, uv_ref_mat);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::norm(y_out_gapi_mat, y_ref_mat, NORM_INF));
        EXPECT_EQ(0, cv::norm(uv_out_gapi_mat, uv_ref_mat, NORM_INF));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestPolyItems, RenderPolylinesPerformanceBGROCVTest)
{
    int thick = 2;
    int lt = LINE_8;
    cv::Scalar color(100, 50, 150);
    MatType type = CV_8UC3;

    cv::Size sz;
    int rects_num, text_num, image_num;
    std::tie(sz, rects_num, text_num, image_num) = GetParam();

    cv::Mat gapi_mat = cv::Mat(sz, type);
    cv::Mat ref_mat(gapi_mat);

    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;

    // Rects
    cv::Rect rect(200, 200, 200, 200);
    int shift = 0;

    for (int i = 0; i < rects_num; ++i)
        prims.emplace_back(cv::gapi::wip::draw::Rect(rect, color, thick, lt, shift));

    // Mosaic
    cv::Rect mos(200, 200, 200, 200);
    int cellsz = 25;
    int decim = 0;

    for (int i = 0; i < rects_num; ++i)
        prims.emplace_back(cv::gapi::wip::draw::Mosaic(mos, cellsz, decim));

    // Text
    std::string text = "Some text";
    cv::Point org(200, 200);
    int ff = FONT_HERSHEY_SIMPLEX;
    double fs = 2.0;
    bool blo = false;

    for (int i = 0; i < text_num; ++i)
        prims.emplace_back(cv::gapi::wip::draw::Text(text, org, ff, fs, color, thick, lt, blo));

    // Image
    cv::Rect rect_img(100, 100, 100, 100);
    double transparency = 1.0;

    cv::Mat img(rect_img.size(), CV_8UC3, color);
    cv::Mat alpha(rect_img.size(), CV_32FC1, transparency);
    auto tl = rect_img.tl();
    cv::Point org_img = {tl.x, tl.y + rect_img.size().height};

    for (int i = 0; i < image_num; ++i)
        prims.emplace_back(cv::gapi::wip::draw::Image(org_img, img, alpha));

    cv::GMat in;
    cv::GArray<cv::gapi::wip::draw::Prim> arr;

    cv::GComputation comp(cv::GIn(in, arr),
                          cv::GOut(cv::gapi::wip::draw::render3ch(in, arr)));

    // Warm-up graph engine:
    comp.apply(gin(gapi_mat, prims), gout(gapi_mat));

    TEST_CYCLE()
    {
        comp.apply(gin(gapi_mat, prims), gout(gapi_mat));
    }

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        cv::rectangle(ref_mat, rect, color, thick, lt, shift);

        drawMosaicRef(ref_mat, mos, cellsz);

        cv::putText(ref_mat, text, org, ff, fs, color, thick, lt, blo);

        blendImageRef(ref_mat, org_img, img, alpha);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::norm(gapi_mat, ref_mat, NORM_INF));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestPolyItems, RenderPolyItemsPerformanceNV12OCVTest)
{
    int thick = 2;
    int lt = LINE_8;
    cv::Scalar color(100, 50, 150);

    cv::Size sz;
    int rects_num, text_num, image_num;
    std::tie(sz, rects_num, text_num, image_num) = GetParam();

    cv::Mat y_ref_mat, uv_ref_mat;

    cv::Mat y_in_gapi_mat(sz, CV_8UC1), uv_in_gapi_mat(sz / 2, CV_8UC2),
        y_out_gapi_mat(sz, CV_8UC1), uv_out_gapi_mat(sz / 2, CV_8UC2);

    create_rand_mats(sz, CV_8UC1, y_ref_mat, y_in_gapi_mat);
    create_rand_mats(sz / 2, CV_8UC2, uv_ref_mat, uv_in_gapi_mat);

    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;

    // Rects
    cv::Rect rect(200, 200, 200, 200);
    int shift = 0;

    for (int i = 0; i < rects_num; ++i)
        prims.emplace_back(cv::gapi::wip::draw::Rect(rect, color, thick, lt, shift));

    // Mosaic
    cv::Rect mos(200, 200, 200, 200);
    int cellsz = 25;
    int decim = 0;

    for (int i = 0; i < rects_num; ++i)
        prims.emplace_back(cv::gapi::wip::draw::Mosaic(mos, cellsz, decim));

    // Text
    std::string text = "Some text";
    cv::Point org(200, 200);
    int ff = FONT_HERSHEY_SIMPLEX;
    double fs = 2.0;
    bool blo = false;

    for (int i = 0; i < text_num; ++i)
        prims.emplace_back(cv::gapi::wip::draw::Text(text, org, ff, fs, color, thick, lt, blo));

    // Image
    cv::Rect rect_img(0, 0, 50, 50);
    double transparency = 1.0;

    cv::Mat img(rect_img.size(), CV_8UC3, color);
    cv::Mat alpha(rect_img.size(), CV_32FC1, transparency);
    auto tl = rect_img.tl();
    cv::Point org_img = {tl.x, tl.y + rect_img.size().height};

    for (int i = 0; i < image_num; ++i)
        prims.emplace_back(cv::gapi::wip::draw::Image({org_img, img, alpha}));

    cv::GMat y_in, uv_in, y_out, uv_out;
    cv::GArray<cv::gapi::wip::draw::Prim> arr;
    std::tie(y_out, uv_out) = cv::gapi::wip::draw::renderNV12(y_in, uv_in, arr);

    cv::GComputation comp(cv::GIn(y_in, uv_in, arr), cv::GOut(y_out, uv_out));

    // Warm-up graph engine:
    comp.apply(cv::gin(y_in_gapi_mat, uv_in_gapi_mat, prims),
               cv::gout(y_out_gapi_mat, uv_out_gapi_mat));

    TEST_CYCLE()
    {
        comp.apply(cv::gin(y_in_gapi_mat, uv_in_gapi_mat, prims),
                   cv::gout(y_out_gapi_mat, uv_out_gapi_mat));
    }

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        // NV12 -> YUV
        cv::Mat yuv;
        cv::gapi::wip::draw::cvtNV12ToYUV(y_ref_mat, uv_ref_mat, yuv);

        cv::rectangle(yuv, rect, cvtBGRToYUVC(color), thick, lt, shift);

        drawMosaicRef(yuv, mos, cellsz);

        cv::putText(yuv, text, org, ff, fs, cvtBGRToYUVC(color), thick, lt, blo);

        cv::Mat yuv_img;
        cv::cvtColor(img, yuv_img, cv::COLOR_BGR2YUV);
        blendImageRef(yuv, org_img, yuv_img, alpha);

        // YUV -> NV12
        cv::gapi::wip::draw::cvtYUVToNV12(yuv, y_ref_mat, uv_ref_mat);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::norm(y_out_gapi_mat, y_ref_mat, NORM_INF));
        EXPECT_EQ(0, cv::norm(uv_out_gapi_mat, uv_ref_mat, NORM_INF));
    }

    SANITY_CHECK_NOTHING();
}

} // namespace opencv_test
