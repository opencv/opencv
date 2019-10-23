// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../test_precomp.hpp"
#include "../common/gapi_render_tests.hpp"

#include "api/render_priv.hpp"

namespace opencv_test
{

GAPI_RENDER_TEST_FIXTURES(TestTexts,     FIXTURE_API(std::string, cv::Point, double, cv::Scalar), 4, text, org, fs, color)
GAPI_RENDER_TEST_FIXTURES(TestRects,     FIXTURE_API(cv::Rect, cv::Scalar, int),                  3, rect, color, thick)
GAPI_RENDER_TEST_FIXTURES(TestCircles,   FIXTURE_API(cv::Point, int, cv::Scalar, int),            4, center, radius, color, thick)
GAPI_RENDER_TEST_FIXTURES(TestLines,     FIXTURE_API(cv::Point, cv::Point, cv::Scalar, int),      4, pt1, pt2, color, thick)
GAPI_RENDER_TEST_FIXTURES(TestMosaics,   FIXTURE_API(cv::Rect, int, int),                         3, mos, cellsz, decim)
GAPI_RENDER_TEST_FIXTURES(TestImages,    FIXTURE_API(cv::Rect, cv::Scalar, double),               3, rect, color, transparency)
GAPI_RENDER_TEST_FIXTURES(TestPolylines, FIXTURE_API(Points, cv::Scalar, int),                    3, points, color, thick)

TEST_P(RenderBGRTestTexts, AccuracyTest)
{
    auto ff = FONT_HERSHEY_SIMPLEX;
    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Text{text, org, fs, color});
    cv::gapi::wip::draw::render(gapi_mat, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        cv::putText(ref_mat, text, org, ff, fs, color);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::norm(gapi_mat, ref_mat));
    }
}

TEST_P(RenderNV12TestTexts, AccuracyTest)
{
    auto ff = FONT_HERSHEY_SIMPLEX;
    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Text{text, org, fs, color});
    cv::gapi::wip::draw::render(y_gapi_mat, uv_gapi_mat, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        // NV12 -> YUV
        cv::Mat yuv;
        cv::gapi::wip::draw::cvtNV12ToYUV(y_ref_mat, uv_ref_mat, yuv);

        cv::putText(yuv, text, org, ff, fs, cvtBGRToYUVC(color));

        // YUV -> NV12
        cv::gapi::wip::draw::cvtYUVToNV12(yuv, y_ref_mat, uv_ref_mat);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::norm(y_gapi_mat,  y_ref_mat));
        EXPECT_EQ(0, cv::norm(uv_gapi_mat, uv_ref_mat));
    }
}

TEST_P(RenderBGRTestRects, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Rect{rect, color, thick});
    cv::gapi::wip::draw::render(gapi_mat, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        cv::rectangle(ref_mat, rect, color, thick);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::norm(gapi_mat, ref_mat));
    }
}

TEST_P(RenderNV12TestRects, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Rect{rect, color, thick});
    cv::gapi::wip::draw::render(y_gapi_mat, uv_gapi_mat, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        // NV12 -> YUV
        cv::Mat yuv;
        cv::gapi::wip::draw::cvtNV12ToYUV(y_ref_mat, uv_ref_mat, yuv);

        cv::rectangle(yuv, rect, cvtBGRToYUVC(color), thick);

        // YUV -> NV12
        cv::gapi::wip::draw::cvtYUVToNV12(yuv, y_ref_mat, uv_ref_mat);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::norm(y_gapi_mat,  y_ref_mat));
        EXPECT_EQ(0, cv::norm(uv_gapi_mat, uv_ref_mat));
    }
}

TEST_P(RenderBGRTestCircles, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Circle{center, radius, color, thick});
    cv::gapi::wip::draw::render(gapi_mat, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        cv::circle(ref_mat, center, radius, color, thick);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::norm(gapi_mat, ref_mat));
    }
}

TEST_P(RenderNV12TestCircles, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Circle{center, radius, color, thick});
    cv::gapi::wip::draw::render(y_gapi_mat, uv_gapi_mat, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        // NV12 -> YUV
        cv::Mat yuv;
        cv::gapi::wip::draw::cvtNV12ToYUV(y_ref_mat, uv_ref_mat, yuv);

        cv::circle(yuv, center, radius, cvtBGRToYUVC(color), thick);

        // YUV -> NV12
        cv::gapi::wip::draw::cvtYUVToNV12(yuv, y_ref_mat, uv_ref_mat);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::norm(y_gapi_mat,  y_ref_mat));
        EXPECT_EQ(0, cv::norm(uv_gapi_mat, uv_ref_mat));
    }
}

TEST_P(RenderBGRTestLines, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Line{pt1, pt2, color, thick});
    cv::gapi::wip::draw::render(gapi_mat, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        cv::line(ref_mat, pt1, pt2, color, thick);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::norm(gapi_mat, ref_mat));
    }
}

TEST_P(RenderNV12TestLines, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Line{pt1, pt2, color, thick});
    cv::gapi::wip::draw::render(y_gapi_mat, uv_gapi_mat, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        // NV12 -> YUV
        cv::Mat yuv;
        cv::gapi::wip::draw::cvtNV12ToYUV(y_ref_mat, uv_ref_mat, yuv);

        cv::line(yuv, pt1, pt2, cvtBGRToYUVC(color), thick);

        // YUV -> NV12
        cv::gapi::wip::draw::cvtYUVToNV12(yuv, y_ref_mat, uv_ref_mat);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::norm(y_gapi_mat,  y_ref_mat));
        EXPECT_EQ(0, cv::norm(uv_gapi_mat, uv_ref_mat));
    }
}

TEST_P(RenderBGRTestMosaics, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Mosaic{mos, cellsz, decim});
    cv::gapi::wip::draw::render(gapi_mat, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        drawMosaicRef(ref_mat, mos, cellsz);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::norm(gapi_mat, ref_mat));
    }
}

TEST_P(RenderNV12TestMosaics, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Mosaic{mos, cellsz, decim});
    cv::gapi::wip::draw::render(y_gapi_mat, uv_gapi_mat, prims);

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
        EXPECT_EQ(0, cv::norm(y_gapi_mat,  y_ref_mat));
        EXPECT_EQ(0, cv::norm(uv_gapi_mat, uv_ref_mat));
    }
}

TEST_P(RenderBGRTestImages, AccuracyTest)
{
    cv::Mat img(rect.size(), CV_8UC3, color);
    cv::Mat alpha(rect.size(), CV_32FC1, transparency);
    auto tl = rect.tl();
    cv::Point org = {tl.x, tl.y + rect.size().height};

    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Image{org, img, alpha});
    cv::gapi::wip::draw::render(gapi_mat, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        blendImageRef(ref_mat, org, img, alpha);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::norm(gapi_mat, ref_mat));
    }
}

TEST_P(RenderNV12TestImages, AccuracyTest)
{
    cv::Mat img(rect.size(), CV_8UC3, color);
    cv::Mat alpha(rect.size(), CV_32FC1, transparency);
    auto tl = rect.tl();
    cv::Point org = {tl.x, tl.y + rect.size().height};

    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Image{org, img, alpha});
    cv::gapi::wip::draw::render(y_gapi_mat, uv_gapi_mat, prims);

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
        EXPECT_EQ(0, cv::norm(y_gapi_mat,  y_ref_mat));
        EXPECT_EQ(0, cv::norm(uv_gapi_mat, uv_ref_mat));
    }
}

TEST_P(RenderBGRTestPolylines, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Poly{points, color, thick});
    cv::gapi::wip::draw::render(gapi_mat, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        std::vector<std::vector<cv::Point>> pp{points};
        cv::fillPoly(ref_mat, pp, color);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::norm(gapi_mat, ref_mat));
    }
}

TEST_P(RenderNV12TestPolylines, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Poly{points, color, thick});
    cv::gapi::wip::draw::render(y_gapi_mat, uv_gapi_mat, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        // NV12 -> YUV
        cv::Mat yuv;
        cv::gapi::wip::draw::cvtNV12ToYUV(y_ref_mat, uv_ref_mat, yuv);

        std::vector<std::vector<cv::Point>> pp{points};
        cv::fillPoly(yuv, pp, cvtBGRToYUVC(color));

        // YUV -> NV12
        cv::gapi::wip::draw::cvtYUVToNV12(yuv, y_ref_mat, uv_ref_mat);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::norm(y_gapi_mat,  y_ref_mat));
        EXPECT_EQ(0, cv::norm(uv_gapi_mat, uv_ref_mat));
    }
}

// FIXME avoid code duplicate for NV12 and BGR cases
INSTANTIATE_TEST_CASE_P(RenderBGRTestRectsImpl, RenderBGRTestRects,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(cv::Rect(100, 100, 200, 200)),
                                Values(cv::Scalar(100, 50, 150)),
                                Values(2)));

INSTANTIATE_TEST_CASE_P(RenderNV12TestRectsImpl, RenderNV12TestRects,
                        Combine(Values(cv::Size(1280, 720)),
                                       Values(cv::Rect(100, 100, 200, 200)),
                                       Values(cv::Scalar(100, 50, 150)),
                                       Values(2)));

INSTANTIATE_TEST_CASE_P(RenderBGRTestCirclesImpl, RenderBGRTestCircles,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(cv::Point(100, 100)),
                                Values(10),
                                Values(cv::Scalar(100, 50, 150)),
                                Values(2)));

INSTANTIATE_TEST_CASE_P(RenderNV12TestCirclesImpl, RenderNV12TestCircles,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(cv::Point(100, 100)),
                                Values(10),
                                Values(cv::Scalar(100, 50, 150)),
                                Values(2)));

INSTANTIATE_TEST_CASE_P(RenderBGRTestLinesImpl, RenderBGRTestLines,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(cv::Point(100, 100)),
                                Values(cv::Point(200, 200)),
                                Values(cv::Scalar(100, 50, 150)),
                                Values(2)));

INSTANTIATE_TEST_CASE_P(RenderNV12TestLinesImpl, RenderNV12TestLines,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(cv::Point(100, 100)),
                                Values(cv::Point(200, 200)),
                                Values(cv::Scalar(100, 50, 150)),
                                Values(2)));

INSTANTIATE_TEST_CASE_P(RenderBGRTestTextsImpl, RenderBGRTestTexts,
                        Combine(Values(cv::Size(1280, 720)),
                                Values("SomeText"),
                                Values(cv::Point(200, 200)),
                                Values(2.0),
                                Values(cv::Scalar(0, 255, 0))));

INSTANTIATE_TEST_CASE_P(RenderNV12TestTextsImpl, RenderNV12TestTexts,
                        Combine(Values(cv::Size(1280, 720)),
                                Values("SomeText"),
                                Values(cv::Point(200, 200)),
                                Values(2.0),
                                Values(cv::Scalar(0, 255, 0))));

INSTANTIATE_TEST_CASE_P(RenderBGRTestMosaicsImpl, RenderBGRTestMosaics,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(cv::Rect(100, 100, 200, 200)),
                                Values(25),
                                Values(0)));

INSTANTIATE_TEST_CASE_P(RenderNV12TestMosaicsImpl, RenderNV12TestMosaics,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(cv::Rect(100, 100, 200, 200)),
                                Values(25),
                                Values(0)));

INSTANTIATE_TEST_CASE_P(RenderBGRTestImagesImpl, RenderBGRTestImages,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(cv::Rect(100, 100, 200, 200)),
                                Values(cv::Scalar(100, 150, 60)),
                                Values(1.0)));

INSTANTIATE_TEST_CASE_P(RenderNV12TestImagesImpl, RenderNV12TestImages,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(cv::Rect(100, 100, 200, 200)),
                                Values(cv::Scalar(100, 150, 60)),
                                Values(1.0)));

INSTANTIATE_TEST_CASE_P(RenderBGRTestPolylinesImpl, RenderBGRTestPolylines,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(std::vector<cv::Point>{{100, 100}, {200, 200}, {150, 300}, {400, 150}}),
                                Values(cv::Scalar(100, 150, 60)),
                                Values(3)));

INSTANTIATE_TEST_CASE_P(RenderNV12TestPolylinesImpl, RenderNV12TestPolylines,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(std::vector<cv::Point>{{100, 100}, {200, 200}, {150, 300}, {400, 150}}),
                                Values(cv::Scalar(100, 150, 60)),
                                Values(3)));
}
