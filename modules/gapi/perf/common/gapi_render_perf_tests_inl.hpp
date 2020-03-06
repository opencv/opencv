#ifndef OPENCV_GAPI_RENDER_PERF_TESTS_INL_HPP
#define OPENCV_GAPI_RENDER_PERF_TESTS_INL_HPP

#include <iostream>

#include "gapi_render_perf_tests.hpp"

namespace opencv_test
{
  using namespace perf;

cv::Scalar cvtBGRToYUVC(const cv::Scalar& bgr)
{
    double y = bgr[2] *  0.299000 + bgr[1] *  0.587000 + bgr[0] *  0.114000;
    double u = bgr[2] * -0.168736 + bgr[1] * -0.331264 + bgr[0] *  0.500000 + 128;
    double v = bgr[2] *  0.500000 + bgr[1] * -0.418688 + bgr[0] * -0.081312 + 128;
    return {y, u, v};
}

void drawMosaicRef(const cv::Mat& mat, const cv::Rect &rect, int cellSz)
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
    for(int i = 0; i < rows; i += cellSz )
    {
        for(int j = 0; j < cols; j += cellSz)
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
        for(int j = 0; j < cols; j += cellSz)
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

void blendImageRef(cv::Mat& mat, const cv::Point& org, const cv::Mat& img, const cv::Mat& alpha)
{
    auto roi = mat(cv::Rect(org, img.size()));
    cv::Mat img32f_w;
    cv::merge(std::vector<cv::Mat>(3, alpha), img32f_w);

    cv::Mat roi32f_w(roi.size(), CV_32FC3, cv::Scalar::all(1.0));
    roi32f_w -= img32f_w;

    cv::Mat img32f, roi32f;
    img.convertTo(img32f, CV_32F, 1.0/255);
    roi.convertTo(roi32f, CV_32F, 1.0/255);

    cv::multiply(img32f, img32f_w, img32f);
    cv::multiply(roi32f, roi32f_w, roi32f);
    roi32f += img32f;

    roi32f.convertTo(roi, CV_8U, 255.0);
}

PERF_TEST_P_(RenderTestTexts, RenderTextsPerformanceBGROCVTest)
{
    std::string text("SomeText");
    cv::Point org(200, 200);
    int ff = FONT_HERSHEY_SIMPLEX;
    double fs = 2.0;
    cv::Scalar color(0, 255, 0);
    int thick = 2;
    int lt = LINE_8;
    bool blo = false;

    compare_f cmpF;
    MatType type = 0;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, compile_args) = GetParam();

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
    comp.apply(gin(gapi_mat, prims), gout(gapi_mat), std::move(compile_args));

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
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestTexts, RenderTextsPerformanceNV12OCVTest)
{
    std::string text("SomeText");
    cv::Point org(200, 200);
    int ff = FONT_HERSHEY_SIMPLEX;
    double fs = 2.0;
    cv::Scalar color(0, 255, 0);
    int thick = 2;
    int lt = LINE_8;
    bool blo = false;

    compare_f cmpF;
    MatType type = 0;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, compile_args) = GetParam();

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
    comp.apply(gin(gapi_mat, prims), gout(gapi_mat), std::move(compile_args));

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
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestRects, RenderRectsPerformanceBGROCVTest)
{
    cv::Rect rect(100, 100, 200, 200);
    cv::Scalar color(100, 50, 150);
    int thick = 2;
    int lt = LINE_8;
    int shift = 0;

    compare_f cmpF;
    MatType type = 0;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, compile_args) = GetParam();

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
    comp.apply(gin(gapi_mat, prims), gout(gapi_mat), std::move(compile_args));

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
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}


PERF_TEST_P_(RenderTestRects, RenderRectsPerformanceTest)
{
    cv::Rect rect(100, 100, 200, 200);
    cv::Scalar color(100, 50, 150);
    int thick = 2;
    int lt = LINE_8;
    int shift = 0;

    compare_f cmpF;
    MatType type = 0;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, compile_args) = GetParam();

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
    comp.apply(gin(gapi_mat, prims), gout(gapi_mat), std::move(compile_args));

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
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestCircles, RenderCirclesPerformanceBGROCVTest)
{
    cv::Point center(cv::Point(100, 100));
    int radius = 10;
    cv::Scalar color(100, 50, 150);
    int thick = 2;
    int lt = LINE_8;
    int shift = 0;

    compare_f cmpF;
    MatType type = 0;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, compile_args) = GetParam();

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
    comp.apply(gin(gapi_mat, prims), gout(gapi_mat), std::move(compile_args));

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
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}


PERF_TEST_P_(RenderTestCircles, RenderCirclesPerformanceTest)
{
    cv::Point center(cv::Point(100, 100));
    int radius = 10;
    cv::Scalar color(100, 50, 150);
    int thick = 2;
    int lt = LINE_8;
    int shift = 0;

    compare_f cmpF;
    MatType type = 0;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, compile_args) = GetParam();

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
    comp.apply(gin(gapi_mat, prims), gout(gapi_mat), std::move(compile_args));

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
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestLines, RenderLinesPerformanceBGROCVTest)
{
    cv::Point pt1(100, 100);
    cv::Point pt2(200, 200);
    cv::Scalar color(100, 50, 150);
    int thick = 2;
    int lt = LINE_8;
    int shift = 0;

    compare_f cmpF;
    MatType type = 0;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, compile_args) = GetParam();

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
    comp.apply(gin(gapi_mat, prims), gout(gapi_mat), std::move(compile_args));

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
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}


PERF_TEST_P_(RenderTestLines, RenderLinesPerformanceTest)
{
    cv::Point pt1(100, 100);
    cv::Point pt2(200, 200);
    cv::Scalar color(100, 50, 150);
    int thick = 2;
    int lt = LINE_8;
    int shift = 0;

    compare_f cmpF;
    MatType type = 0;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, compile_args) = GetParam();

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
    comp.apply(gin(gapi_mat, prims), gout(gapi_mat), std::move(compile_args));

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
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestMosaics, RenderMosaicsPerformanceBGROCVTest)
{
    cv::Rect mos(100, 100, 200, 200);
    int cellsz = 25;
    int decim = 0;

    compare_f cmpF;
    MatType type = 0;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, compile_args) = GetParam();

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
    comp.apply(gin(gapi_mat, prims), gout(gapi_mat), std::move(compile_args));

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
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestMosaics, RenderMosaicsPerformanceTest)
{
    cv::Rect mos(100, 100, 200, 200);
    int cellsz = 25;
    int decim = 0;

    compare_f cmpF;
    MatType type = 0;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, compile_args) = GetParam();

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
    comp.apply(gin(gapi_mat, prims), gout(gapi_mat), std::move(compile_args));

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
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestImages, RenderImagesPerformanceBGROCVTest)
{
    cv::Rect rect(100, 100, 200, 200);
    cv::Scalar color(100, 150, 60);
    double  transparency = 1.0;

    compare_f cmpF;
    MatType type = 0;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, compile_args) = GetParam();

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
    comp.apply(gin(gapi_mat, prims), gout(gapi_mat), std::move(compile_args));

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
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestImages, RenderImagesPerformanceTest)
{
    cv::Rect rect(100, 100, 200, 200);
    cv::Scalar color(100, 150, 60);
    double  transparency = 1.0;

    compare_f cmpF;
    MatType type = 0;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, compile_args) = GetParam();

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
    comp.apply(gin(gapi_mat, prims), gout(gapi_mat), std::move(compile_args));

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
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RenderTestPolylines, RenderPolylinesPerformanceBGROCVTest)
{
    std::vector<cv::Point> points{{100, 100}, {200, 200}, {150, 300}, {400, 150}};
    cv::Scalar color(100, 150, 60);
    int thick = 2;
    int lt = LINE_8;
    int shift = 0;

    compare_f cmpF;
    MatType type = 0;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, compile_args) = GetParam();

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
    comp.apply(gin(gapi_mat, prims), gout(gapi_mat), std::move(compile_args));

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
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}


PERF_TEST_P_(RenderTestPolylines, RenderPolylinesPerformanceTest)
{
    std::vector<cv::Point> points{{100, 100}, {200, 200}, {150, 300}, {400, 150}};
    cv::Scalar color(100, 150, 60);
    int thick = 2;
    int lt = LINE_8;
    int shift = 0;

    compare_f cmpF;
    MatType type = 0;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, compile_args) = GetParam();

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
    comp.apply(gin(gapi_mat, prims), gout(gapi_mat), std::move(compile_args));

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
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

}

#endif // OPENCV_GAPI_RENDER_PERF_TESTS_INL_HPP
