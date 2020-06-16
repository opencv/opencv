// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#include "gapi_render_perf_tests.hpp"

namespace opencv_test
{

namespace {
void create_rand_mats(const cv::Size &size, MatType type, cv::Mat &ref_mat, cv::Mat &gapi_mat)
{
    ref_mat.create(size, type);
    cv::randu(ref_mat, cv::Scalar::all(0), cv::Scalar::all(255));
    ref_mat.copyTo(gapi_mat);
};

} // namespace

PERF_TEST_P_(RenderTestFTexts, RenderFTextsPerformanceBGROCVTest)
{
    std::wstring text;
    cv::Size sz;
    cv::Point org;
    int fh = 0;
    cv::Scalar color;
    cv::GCompileArgs comp_args;
    std::tie(text ,sz ,org ,fh ,color, comp_args) = GetParam();

    MatType type =  CV_8UC3;
    cv::Mat gapi_mat, ref_mat;
    create_rand_mats(sz, type, ref_mat, gapi_mat);

    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::FText{text, org, fh, color});

    cv::GMat in;
    cv::GArray<cv::gapi::wip::draw::Prim> arr;

    cv::GComputation comp(cv::GIn(in, arr),
                          cv::GOut(cv::gapi::wip::draw::render3ch(in, arr)));

    // Warm-up graph engine:
    comp.apply(gin(gapi_mat, prims), gout(gapi_mat), std::move(comp_args));

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
    int fh = 0;
    cv::Scalar color;
    cv::GCompileArgs comp_args;
    std::tie(text ,sz ,org ,fh ,color, comp_args) = GetParam();

    cv::Mat y_ref_mat, uv_ref_mat;

    cv::Mat y_in_gapi_mat, uv_in_gapi_mat,
            y_out_gapi_mat, uv_out_gapi_mat;

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
               cv::gout(y_out_gapi_mat, uv_out_gapi_mat), std::move(comp_args));

    TEST_CYCLE()
    {
        comp.apply(cv::gin(y_in_gapi_mat, uv_in_gapi_mat, prims),
                   cv::gout(y_out_gapi_mat, uv_out_gapi_mat));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestTexts, RenderTextsPerformanceBGROCVTest)
{
    cv::Point org;
    int ff = 0;
    int thick = 0;
    int lt = 0;
    double fs = 2.0;
    cv::Scalar color;
    bool blo = false;
    std::string text;
    cv::Size sz;
    cv::GCompileArgs comp_args;
    std::tie(text, sz, org, ff, color, thick, lt, blo, comp_args) = GetParam();

    MatType type =  CV_8UC3;
    cv::Mat gapi_mat, ref_mat;
    create_rand_mats(sz, type, ref_mat, gapi_mat);

    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Text{text, org, ff, fs, color, thick, lt, blo});

    cv::GMat in;
    cv::GArray<cv::gapi::wip::draw::Prim> arr;

    cv::GComputation comp(cv::GIn(in, arr),
                          cv::GOut(cv::gapi::wip::draw::render3ch(in, arr)));

    // Warm-up graph engine:
    comp.apply(gin(gapi_mat, prims), gout(gapi_mat), std::move(comp_args));

    TEST_CYCLE()
    {
        comp.apply(gin(gapi_mat, prims), gout(gapi_mat));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestTexts, RenderTextsPerformanceNV12OCVTest)
{
    cv::Point org;
    int ff = 0;
    int thick = 0;
    int lt = 0;
    double fs = 2.0;
    cv::Scalar color;
    bool blo = false;
    std::string text;
    cv::Size sz;
    cv::GCompileArgs comp_args;
    std::tie(text, sz, org, ff, color, thick, lt, blo, comp_args) = GetParam();

    cv::Mat y_ref_mat, uv_ref_mat;

    cv::Mat y_in_gapi_mat, uv_in_gapi_mat,
            y_out_gapi_mat, uv_out_gapi_mat;

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
               cv::gout(y_out_gapi_mat, uv_out_gapi_mat), std::move(comp_args));

    TEST_CYCLE()
    {
        comp.apply(cv::gin(y_in_gapi_mat, uv_in_gapi_mat, prims),
                   cv::gout(y_out_gapi_mat, uv_out_gapi_mat));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestRects, RenderRectsPerformanceBGROCVTest)
{
    cv::Rect rect;
    cv::Scalar color;
    int thick = 0;
    int lt = 0;
    int shift = 0;
    cv::Size sz;
    cv::GCompileArgs comp_args;
    std::tie(sz, rect, color, thick, lt, shift, comp_args) = GetParam();

    MatType type =  CV_8UC3;
    cv::Mat gapi_mat, ref_mat;
    create_rand_mats(sz, type, ref_mat, gapi_mat);

    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Rect{rect, color, thick, lt, shift});

    cv::GMat in;
    cv::GArray<cv::gapi::wip::draw::Prim> arr;

    cv::GComputation comp(cv::GIn(in, arr),
                          cv::GOut(cv::gapi::wip::draw::render3ch(in, arr)));

    // Warm-up graph engine:
    comp.apply(gin(gapi_mat, prims), gout(gapi_mat), std::move(comp_args));

    TEST_CYCLE()
    {
        comp.apply(gin(gapi_mat, prims), gout(gapi_mat));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestRects, RenderRectsPerformanceNV12OCVTest)
{
    cv::Rect rect;
    cv::Scalar color;
    int thick = 0;
    int lt = 0;
    int shift = 0;
    cv::Size sz;
    cv::GCompileArgs comp_args;
    std::tie(sz, rect, color, thick, lt, shift, comp_args) = GetParam();

    cv::Mat y_ref_mat, uv_ref_mat;

    cv::Mat y_in_gapi_mat, uv_in_gapi_mat,
            y_out_gapi_mat, uv_out_gapi_mat;

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
               cv::gout(y_out_gapi_mat, uv_out_gapi_mat), std::move(comp_args));

    TEST_CYCLE()
    {
        comp.apply(cv::gin(y_in_gapi_mat, uv_in_gapi_mat, prims),
                   cv::gout(y_out_gapi_mat, uv_out_gapi_mat));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestCircles, RenderCirclesPerformanceBGROCVTest)
{
    cv::Point center;
    int radius;
    cv::Scalar color;
    int thick = 0;
    int lt = 0;
    int shift = 0;

    cv::Size sz;
    cv::GCompileArgs comp_args;
    std::tie(sz, center, radius, color, thick, lt, shift, comp_args) = GetParam();

    MatType type =  CV_8UC3;
    cv::Mat gapi_mat, ref_mat;
    create_rand_mats(sz, type, ref_mat, gapi_mat);

    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Circle{center, radius, color, thick, lt, shift});

    cv::GMat in;
    cv::GArray<cv::gapi::wip::draw::Prim> arr;

    cv::GComputation comp(cv::GIn(in, arr),
                          cv::GOut(cv::gapi::wip::draw::render3ch(in, arr)));

    // Warm-up graph engine:
    comp.apply(gin(gapi_mat, prims), gout(gapi_mat), std::move(comp_args));

    TEST_CYCLE()
    {
        comp.apply(gin(gapi_mat, prims), gout(gapi_mat));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestCircles, RenderCirclesPerformanceNV12OCVTest)
{
    cv::Point center;
    int radius;
    cv::Scalar color;
    int thick = 0;
    int lt = 0;
    int shift = 0;

    cv::Size sz;
    cv::GCompileArgs comp_args;
    std::tie(sz, center, radius, color, thick, lt, shift, comp_args) = GetParam();

    cv::Mat y_ref_mat, uv_ref_mat;

    cv::Mat y_in_gapi_mat, uv_in_gapi_mat,
            y_out_gapi_mat, uv_out_gapi_mat;

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
               cv::gout(y_out_gapi_mat, uv_out_gapi_mat), std::move(comp_args));

    TEST_CYCLE()
    {
        comp.apply(cv::gin(y_in_gapi_mat, uv_in_gapi_mat, prims),
                   cv::gout(y_out_gapi_mat, uv_out_gapi_mat));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestLines, RenderLinesPerformanceBGROCVTest)
{
    cv::Point pt1;
    cv::Point pt2;
    cv::Scalar color;
    int thick = 0;
    int lt = 0;
    int shift = 0;

    compare_f cmpF;
    cv::Size sz;
    cv::GCompileArgs comp_args;
    std::tie(sz, pt1, pt2, color, thick, lt, shift, comp_args) = GetParam();

    MatType type =  CV_8UC3;
    cv::Mat gapi_mat, ref_mat;
    create_rand_mats(sz, type, ref_mat, gapi_mat);

    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Line{pt1, pt2, color, thick, lt, shift});

    cv::GMat in;
    cv::GArray<cv::gapi::wip::draw::Prim> arr;

    cv::GComputation comp(cv::GIn(in, arr),
                          cv::GOut(cv::gapi::wip::draw::render3ch(in, arr)));

    // Warm-up graph engine:
    comp.apply(gin(gapi_mat, prims), gout(gapi_mat), std::move(comp_args));

    TEST_CYCLE()
    {
        comp.apply(gin(gapi_mat, prims), gout(gapi_mat));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestLines, RenderLinesPerformanceNV12OCVTest)
{
    cv::Point pt1;
    cv::Point pt2;
    cv::Scalar color;
    int thick = 0;
    int lt = 0;
    int shift = 0;

    compare_f cmpF;
    cv::Size sz;
    cv::GCompileArgs comp_args;
    std::tie(sz, pt1, pt2, color, thick, lt, shift, comp_args) = GetParam();

    cv::Mat y_ref_mat, uv_ref_mat;

    cv::Mat y_in_gapi_mat, uv_in_gapi_mat,
            y_out_gapi_mat, uv_out_gapi_mat;

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
               cv::gout(y_out_gapi_mat, uv_out_gapi_mat), std::move(comp_args));

    TEST_CYCLE()
    {
        comp.apply(cv::gin(y_in_gapi_mat, uv_in_gapi_mat, prims),
                   cv::gout(y_out_gapi_mat, uv_out_gapi_mat));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestMosaics, RenderMosaicsPerformanceBGROCVTest)
{
    cv::Rect mos;
    int cellsz = 0;
    int decim = 0;

    cv::Size sz;
    cv::GCompileArgs comp_args;
    std::tie(sz, mos, cellsz, decim, comp_args) = GetParam();

    MatType type =  CV_8UC3;
    cv::Mat gapi_mat, ref_mat;
    create_rand_mats(sz, type, ref_mat, gapi_mat);

    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Mosaic{mos, cellsz, decim});

    cv::GMat in;
    cv::GArray<cv::gapi::wip::draw::Prim> arr;

    cv::GComputation comp(cv::GIn(in, arr),
                          cv::GOut(cv::gapi::wip::draw::render3ch(in, arr)));

    // Warm-up graph engine:
    comp.apply(gin(gapi_mat, prims), gout(gapi_mat), std::move(comp_args));

    TEST_CYCLE()
    {
        comp.apply(gin(gapi_mat, prims), gout(gapi_mat));
    }

    SANITY_CHECK_NOTHING();
}


PERF_TEST_P_(RenderTestMosaics, RenderMosaicsPerformanceNV12OCVTest)
{
    cv::Rect mos;
    int cellsz = 0;
    int decim = 0;

    cv::Size sz;
    cv::GCompileArgs comp_args;
    std::tie(sz, mos, cellsz, decim, comp_args) = GetParam();

    cv::Mat y_ref_mat, uv_ref_mat;

    cv::Mat y_in_gapi_mat, uv_in_gapi_mat,
            y_out_gapi_mat, uv_out_gapi_mat;

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
               cv::gout(y_out_gapi_mat, uv_out_gapi_mat), std::move(comp_args));

    TEST_CYCLE()
    {
        comp.apply(cv::gin(y_in_gapi_mat, uv_in_gapi_mat, prims),
                   cv::gout(y_out_gapi_mat, uv_out_gapi_mat));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestImages, RenderImagesPerformanceBGROCVTest)
{
    cv::Rect rect;
    cv::Scalar color;
    double transparency = 0.0;

    cv::Size sz;
    cv::GCompileArgs comp_args;
    std::tie(sz, rect, color, transparency, comp_args) = GetParam();

    MatType type =  CV_8UC3;
    cv::Mat gapi_mat, ref_mat;
    create_rand_mats(sz, type, ref_mat, gapi_mat);

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
    comp.apply(gin(gapi_mat, prims), gout(gapi_mat), std::move(comp_args));

    TEST_CYCLE()
    {
        comp.apply(gin(gapi_mat, prims), gout(gapi_mat));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestImages, RenderImagesPerformanceNV12OCVTest)
{
    cv::Rect rect;
    cv::Scalar color;
    double transparency = 0.0;

    cv::Size sz;
    cv::GCompileArgs comp_args;
    std::tie(sz, rect, color, transparency, comp_args) = GetParam();

    cv::Mat y_ref_mat, uv_ref_mat;

    cv::Mat y_in_gapi_mat, uv_in_gapi_mat,
            y_out_gapi_mat, uv_out_gapi_mat;

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
               cv::gout(y_out_gapi_mat, uv_out_gapi_mat), std::move(comp_args));

    TEST_CYCLE()
    {
        comp.apply(cv::gin(y_in_gapi_mat, uv_in_gapi_mat, prims),
                   cv::gout(y_out_gapi_mat, uv_out_gapi_mat));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestPolylines, RenderPolylinesPerformanceBGROCVTest)
{
    std::vector<cv::Point> points;
    cv::Scalar color;
    int thick = 0;
    int lt = 0;
    int shift = 0;

    cv::Size sz;
    cv::GCompileArgs comp_args;
    std::tie(sz, points, color, thick, lt, shift, comp_args) = GetParam();

    MatType type =  CV_8UC3;
    cv::Mat gapi_mat, ref_mat;
    create_rand_mats(sz, type, ref_mat, gapi_mat);

    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Poly{points, color, thick, lt, shift});

    cv::GMat in;
    cv::GArray<cv::gapi::wip::draw::Prim> arr;

    cv::GComputation comp(cv::GIn(in, arr),
                          cv::GOut(cv::gapi::wip::draw::render3ch(in, arr)));

    // Warm-up graph engine:
    comp.apply(gin(gapi_mat, prims), gout(gapi_mat), std::move(comp_args));

    TEST_CYCLE()
    {
        comp.apply(gin(gapi_mat, prims), gout(gapi_mat));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestPolylines, RenderPolylinesPerformanceNV12OCVTest)
{
    std::vector<cv::Point> points;
    cv::Scalar color;
    int thick = 0;
    int lt = 0;
    int shift = 0;

    cv::Size sz;
    cv::GCompileArgs comp_args;
    std::tie(sz, points, color, thick, lt, shift, comp_args) = GetParam();

    cv::Mat y_ref_mat, uv_ref_mat;

    cv::Mat y_in_gapi_mat, uv_in_gapi_mat,
            y_out_gapi_mat, uv_out_gapi_mat;

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
               cv::gout(y_out_gapi_mat, uv_out_gapi_mat), std::move(comp_args));

    TEST_CYCLE()
    {
        comp.apply(cv::gin(y_in_gapi_mat, uv_in_gapi_mat, prims),
                   cv::gout(y_out_gapi_mat, uv_out_gapi_mat));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestPolyItems, RenderPolyItemsPerformanceBGROCVTest)
{
    cv::Size sz;
    int rects_num = 0;
    int text_num = 0;
    int image_num = 0;
    cv::GCompileArgs comp_args;
    std::tie(sz, rects_num, text_num, image_num, comp_args) = GetParam();

    int thick = 2;
    int lt = LINE_8;
    cv::Scalar color(100, 50, 150);

    MatType type = CV_8UC3;
    cv::Mat gapi_mat, ref_mat;
    create_rand_mats(sz, type, ref_mat, gapi_mat);
    cv::Mat gapi_out_mat(sz, type);
    gapi_mat.copyTo(gapi_out_mat);

    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;

    // Rects
    int shift = 0;
    for (int i = 0; i < rects_num; ++i) {
        cv::Rect rect(200 + i, 200 + i, 200, 200);
        prims.emplace_back(cv::gapi::wip::draw::Rect(rect, color, thick, lt, shift));
    }

    // Mosaic
    int cellsz = 25;
    int decim = 0;
    for (int i = 0; i < rects_num; ++i) {
        cv::Rect mos(200 + i, 200 + i, 200, 200);
        prims.emplace_back(cv::gapi::wip::draw::Mosaic(mos, cellsz, decim));
    }

    // Text
    std::string text = "Some text";
    int ff = FONT_HERSHEY_SIMPLEX;
    double fs = 2.0;
    bool blo = false;
    for (int i = 0; i < text_num; ++i) {
        cv::Point org(200 + i, 200 + i);
        prims.emplace_back(cv::gapi::wip::draw::Text(text, org, ff, fs, color, thick, lt, blo));
    }

    // Image
    double transparency = 1.0;
    cv::Rect rect_img(0 ,0 , 50, 50);
    cv::Mat img(rect_img.size(), CV_8UC3, color);
    cv::Mat alpha(rect_img.size(), CV_32FC1, transparency);
    auto tl = rect_img.tl();
    for (int i = 0; i < image_num; ++i) {
        cv::Point org_img = {tl.x + i, tl.y + rect_img.size().height + i};

        prims.emplace_back(cv::gapi::wip::draw::Image({org_img, img, alpha}));
    }

    cv::GMat in;
    cv::GArray<cv::gapi::wip::draw::Prim> arr;

    cv::GComputation comp(cv::GIn(in, arr),
                          cv::GOut(cv::gapi::wip::draw::render3ch(in, arr)));

    // Warm-up graph engine:
    comp.apply(gin(gapi_mat, prims), gout(gapi_out_mat), std::move(comp_args));

    TEST_CYCLE()
    {
        comp.apply(gin(gapi_mat, prims), gout(gapi_out_mat));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestPolyItems, RenderPolyItemsPerformanceNV12OCVTest)
{
    cv::Size sz;
    int rects_num = 0;
    int text_num = 0;
    int image_num = 0;
    cv::GCompileArgs comp_args;
    std::tie(sz, rects_num, text_num, image_num, comp_args) = GetParam();

    int thick = 2;
    int lt = LINE_8;
    cv::Scalar color(100, 50, 150);

    cv::Mat y_ref_mat, uv_ref_mat;

    cv::Mat y_in_gapi_mat, uv_in_gapi_mat,
            y_out_gapi_mat, uv_out_gapi_mat;

    create_rand_mats(sz, CV_8UC1, y_ref_mat, y_in_gapi_mat);
    create_rand_mats(sz / 2, CV_8UC2, uv_ref_mat, uv_in_gapi_mat);

    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;

    // Rects
    int shift = 0;
    for (int i = 0; i < rects_num; ++i) {
        cv::Rect rect(200 + i, 200 + i, 200, 200);
        prims.emplace_back(cv::gapi::wip::draw::Rect(rect, color, thick, lt, shift));
    }

    // Mosaic
    int cellsz = 25;
    int decim = 0;
    for (int i = 0; i < rects_num; ++i) {
        cv::Rect mos(200 + i, 200 + i, 200, 200);
        prims.emplace_back(cv::gapi::wip::draw::Mosaic(mos, cellsz, decim));
    }

    // Text
    std::string text = "Some text";
    int ff = FONT_HERSHEY_SIMPLEX;
    double fs = 2.0;
    bool blo = false;
    for (int i = 0; i < text_num; ++i) {
        cv::Point org(200 + i, 200 + i);
        prims.emplace_back(cv::gapi::wip::draw::Text(text, org, ff, fs, color, thick, lt, blo));
    }

    // Image
    double transparency = 1.0;
    cv::Rect rect_img(0 ,0 , 50, 50);
    cv::Mat img(rect_img.size(), CV_8UC3, color);
    cv::Mat alpha(rect_img.size(), CV_32FC1, transparency);
    auto tl = rect_img.tl();
    for (int i = 0; i < image_num; ++i) {
        cv::Point org_img = {tl.x + i, tl.y + rect_img.size().height + i};

        prims.emplace_back(cv::gapi::wip::draw::Image({org_img, img, alpha}));
    }

    cv::GMat y_in, uv_in, y_out, uv_out;
    cv::GArray<cv::gapi::wip::draw::Prim> arr;
    std::tie(y_out, uv_out) = cv::gapi::wip::draw::renderNV12(y_in, uv_in, arr);

    cv::GComputation comp(cv::GIn(y_in, uv_in, arr), cv::GOut(y_out, uv_out));

    // Warm-up graph engine:
    comp.apply(cv::gin(y_in_gapi_mat, uv_in_gapi_mat, prims),
               cv::gout(y_out_gapi_mat, uv_out_gapi_mat), std::move(comp_args));

    TEST_CYCLE()
    {
        comp.apply(cv::gin(y_in_gapi_mat, uv_in_gapi_mat, prims),
                   cv::gout(y_out_gapi_mat, uv_out_gapi_mat));
    }

    SANITY_CHECK_NOTHING();
}

} // namespace opencv_test
