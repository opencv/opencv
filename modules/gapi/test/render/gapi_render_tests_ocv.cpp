// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifdef HAVE_FREETYPE
#include <codecvt>
#endif // HAVE_FREETYPE

#include "../test_precomp.hpp"
#include "../common/gapi_render_tests.hpp"

#include "api/render_priv.hpp"

namespace opencv_test
{

#ifdef HAVE_FREETYPE
GAPI_RENDER_TEST_FIXTURES(OCVTestFTexts,    FIXTURE_API(std::wstring, cv::Point, int, cv::Scalar),                        4, text, org, fh, color)
#endif // HAVE_FREETYPE

GAPI_RENDER_TEST_FIXTURES(OCVTestTexts,     FIXTURE_API(std::string, cv::Point, int, double, cv::Scalar, int, int, bool), 8, text, org, ff, fs, color, thick, lt, blo)
GAPI_RENDER_TEST_FIXTURES(OCVTestRects,     FIXTURE_API(cv::Rect, cv::Scalar, int, int, int),                             5, rect, color, thick, lt, shift)
GAPI_RENDER_TEST_FIXTURES(OCVTestCircles,   FIXTURE_API(cv::Point, int, cv::Scalar, int, int, int),                       6, center, radius, color, thick, lt, shift)
GAPI_RENDER_TEST_FIXTURES(OCVTestLines,     FIXTURE_API(cv::Point, cv::Point, cv::Scalar, int, int, int),                 6, pt1, pt2, color, thick, lt, shift)
GAPI_RENDER_TEST_FIXTURES(OCVTestMosaics,   FIXTURE_API(cv::Rect, int, int),                                              3, mos, cellsz, decim)
GAPI_RENDER_TEST_FIXTURES(OCVTestImages,    FIXTURE_API(cv::Rect, cv::Scalar, double),                                    3, rect, color, transparency)
GAPI_RENDER_TEST_FIXTURES(OCVTestPolylines, FIXTURE_API(Points, cv::Scalar, int, int, int),                               5, points, color, thick, lt, shift)

TEST_P(RenderBGROCVTestTexts, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Text{text, org, ff, fs, color, thick, lt, blo});
    cv::gapi::wip::draw::render(gapi_mat, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        cv::putText(ref_mat, text, org, ff, fs, color, thick, lt, blo);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::norm(gapi_mat, ref_mat));
    }
}

TEST_P(RenderNV12OCVTestTexts, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Text{text, org, ff, fs, color, thick, lt, blo});
    cv::gapi::wip::draw::render(y_gapi_mat, uv_gapi_mat, prims);

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
        EXPECT_EQ(0, cv::norm(y_gapi_mat,  y_ref_mat));
        EXPECT_EQ(0, cv::norm(uv_gapi_mat, uv_ref_mat));
    }
}

class TestMediaNV12 final : public cv::MediaFrame::IAdapter {
    cv::Mat m_y;
    cv::Mat m_uv;
public:
    TestMediaNV12(cv::Mat y, cv::Mat uv) : m_y(y), m_uv(uv) {
    }
    cv::GFrameDesc meta() const override {
        return cv::GFrameDesc{ cv::MediaFormat::NV12, cv::Size(m_y.cols, m_y.rows) };
    }
    cv::MediaFrame::View access(cv::MediaFrame::Access) override {
        cv::MediaFrame::View::Ptrs pp = {
            m_y.ptr(), m_uv.ptr(), nullptr, nullptr
        };
        cv::MediaFrame::View::Strides ss = {
            m_y.step, m_uv.step, 0u, 0u
        };
        return cv::MediaFrame::View(std::move(pp), std::move(ss));
    }
};

TEST_P(RenderMFrameOCVTestTexts, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Text{ text, org, ff, fs, color, thick, lt, blo });
    cv::MediaFrame nv12 = cv::MediaFrame::Create<TestMediaNV12>(y_gapi_mat, uv_gapi_mat);
    cv::gapi::wip::draw::render(nv12, prims);

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
        EXPECT_EQ(0, cv::norm(y_gapi_mat, y_ref_mat));
        EXPECT_EQ(0, cv::norm(uv_gapi_mat, uv_ref_mat));
    }
}


# ifdef HAVE_FREETYPE

TEST_P(RenderBGROCVTestFTexts, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::FText{text, org, fh, color});
    EXPECT_NO_THROW(cv::gapi::wip::draw::render(gapi_mat, prims,
                                cv::compile_args(cv::gapi::wip::draw::freetype_font{
                                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
                                })));
}

TEST_P(RenderNV12OCVTestFTexts, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::FText{text, org, fh, color});
    EXPECT_NO_THROW(cv::gapi::wip::draw::render(y_gapi_mat, uv_gapi_mat, prims,
                                cv::compile_args(cv::gapi::wip::draw::freetype_font{
                                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
                                })));
}

TEST_P(RenderMFrameOCVTestFTexts, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::Mat y_copy_mat = y_gapi_mat.clone();
    cv::Mat uv_copy_mat = uv_gapi_mat.clone();
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::FText{ text, org, fh, color });
    cv::MediaFrame nv12 = cv::MediaFrame::Create<TestMediaNV12>(y_gapi_mat, uv_gapi_mat);
    EXPECT_NO_THROW(cv::gapi::wip::draw::render(nv12, prims,
        cv::compile_args(cv::gapi::wip::draw::freetype_font{
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
            })));
    EXPECT_NE(0, cv::norm(y_gapi_mat, y_copy_mat));
    EXPECT_NE(0, cv::norm(uv_gapi_mat, uv_copy_mat));
}


static std::wstring to_wstring(const char* bytes)
{
    std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
    return converter.from_bytes(bytes);
}

TEST(RenderFText, FontsNotPassedToCompileArgs)
{
    cv::Mat in_mat(640, 480, CV_8UC3, cv::Scalar::all(0));

    std::wstring text = to_wstring("\xe4\xbd\xa0\xe5\xa5\xbd");
    cv::Point org(100, 100);
    int fh = 60;
    cv::Scalar color(200, 100, 25);

    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::FText{text, org, fh, color});

    EXPECT_ANY_THROW(cv::gapi::wip::draw::render(in_mat, prims));
}

#endif // HAVE_FREETYPE

TEST_P(RenderBGROCVTestRects, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Rect{rect, color, thick, lt, shift});
    cv::gapi::wip::draw::render(gapi_mat, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        cv::rectangle(ref_mat, rect, color, thick, lt, shift);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::norm(gapi_mat, ref_mat));
    }
}

TEST_P(RenderNV12OCVTestRects, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Rect{rect, color, thick, lt, shift});
    cv::gapi::wip::draw::render(y_gapi_mat, uv_gapi_mat, prims);

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
        EXPECT_EQ(0, cv::norm(y_gapi_mat,  y_ref_mat));
        EXPECT_EQ(0, cv::norm(uv_gapi_mat, uv_ref_mat));
    }
}

TEST_P(RenderMFrameOCVTestRects, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Rect{ rect, color, thick, lt, shift });
    cv::MediaFrame nv12 = cv::MediaFrame::Create<TestMediaNV12>(y_gapi_mat, uv_gapi_mat);
    cv::gapi::wip::draw::render(nv12, prims);

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
        EXPECT_EQ(0, cv::norm(y_gapi_mat, y_ref_mat));
        EXPECT_EQ(0, cv::norm(uv_gapi_mat, uv_ref_mat));
    }
}

TEST_P(RenderBGROCVTestCircles, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Circle{center, radius, color, thick, lt, shift});
    cv::gapi::wip::draw::render(gapi_mat, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        cv::circle(ref_mat, center, radius, color, thick, lt, shift);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::norm(gapi_mat, ref_mat));
    }
}

TEST_P(RenderNV12OCVTestCircles, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Circle{center, radius, color, thick, lt, shift});
    cv::gapi::wip::draw::render(y_gapi_mat, uv_gapi_mat, prims);

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
        EXPECT_EQ(0, cv::norm(y_gapi_mat,  y_ref_mat));
        EXPECT_EQ(0, cv::norm(uv_gapi_mat, uv_ref_mat));
    }
}

TEST_P(RenderMFrameOCVTestCircles, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Circle{ center, radius, color, thick, lt, shift });
    cv::MediaFrame nv12 = cv::MediaFrame::Create<TestMediaNV12>(y_gapi_mat, uv_gapi_mat);
    cv::gapi::wip::draw::render(nv12, prims);

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
        EXPECT_EQ(0, cv::norm(y_gapi_mat, y_ref_mat));
        EXPECT_EQ(0, cv::norm(uv_gapi_mat, uv_ref_mat));
    }
}

TEST_P(RenderBGROCVTestLines, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Line{pt1, pt2, color, thick, lt, shift});
    cv::gapi::wip::draw::render(gapi_mat, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        cv::line(ref_mat, pt1, pt2, color, thick, lt, shift);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::norm(gapi_mat, ref_mat));
    }
}

TEST_P(RenderNV12OCVTestLines, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Line{pt1, pt2, color, thick, lt, shift});
    cv::gapi::wip::draw::render(y_gapi_mat, uv_gapi_mat, prims);

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
        EXPECT_EQ(0, cv::norm(y_gapi_mat,  y_ref_mat));
        EXPECT_EQ(0, cv::norm(uv_gapi_mat, uv_ref_mat));
    }
}

TEST_P(RenderMFrameOCVTestLines, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Line{ pt1, pt2, color, thick, lt, shift });
    cv::MediaFrame nv12 = cv::MediaFrame::Create<TestMediaNV12>(y_gapi_mat, uv_gapi_mat);
    cv::gapi::wip::draw::render(nv12, prims);

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
        EXPECT_EQ(0, cv::norm(y_gapi_mat, y_ref_mat));
        EXPECT_EQ(0, cv::norm(uv_gapi_mat, uv_ref_mat));
    }
}

TEST_P(RenderBGROCVTestMosaics, AccuracyTest)
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

TEST_P(RenderNV12OCVTestMosaics, AccuracyTest)
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

TEST_P(RenderMFrameOCVTestMosaics, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Mosaic{ mos, cellsz, decim });
    cv::MediaFrame nv12 = cv::MediaFrame::Create<TestMediaNV12>(y_gapi_mat, uv_gapi_mat);
    cv::gapi::wip::draw::render(nv12, prims);

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
        EXPECT_EQ(0, cv::norm(y_gapi_mat, y_ref_mat));
        EXPECT_EQ(0, cv::norm(uv_gapi_mat, uv_ref_mat));
    }
}

TEST_P(RenderBGROCVTestImages, AccuracyTest)
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

TEST_P(RenderNV12OCVTestImages, AccuracyTest)
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

TEST_P(RenderMFrameOCVTestImages, AccuracyTest)
{
    cv::Mat img(rect.size(), CV_8UC3, color);
    cv::Mat alpha(rect.size(), CV_32FC1, transparency);
    auto tl = rect.tl();
    cv::Point org = { tl.x, tl.y + rect.size().height };

    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Image{ org, img, alpha });
    cv::MediaFrame nv12 = cv::MediaFrame::Create<TestMediaNV12>(y_gapi_mat, uv_gapi_mat);
    cv::gapi::wip::draw::render(nv12, prims);

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
        EXPECT_EQ(0, cv::norm(y_gapi_mat, y_ref_mat));
        EXPECT_EQ(0, cv::norm(uv_gapi_mat, uv_ref_mat));
    }
}

TEST_P(RenderBGROCVTestPolylines, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Poly{points, color, thick, lt, shift});
    cv::gapi::wip::draw::render(gapi_mat, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        std::vector<std::vector<cv::Point>> array_points{points};
        cv::fillPoly(ref_mat, array_points, color, lt, shift);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::norm(gapi_mat, ref_mat));
    }
}

TEST_P(RenderNV12OCVTestPolylines, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Poly{points, color, thick, lt, shift});
    cv::gapi::wip::draw::render(y_gapi_mat, uv_gapi_mat, prims);

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
        EXPECT_EQ(0, cv::norm(y_gapi_mat,  y_ref_mat));
        EXPECT_EQ(0, cv::norm(uv_gapi_mat, uv_ref_mat));
    }
}

TEST_P(RenderMFrameOCVTestPolylines, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;
    prims.emplace_back(cv::gapi::wip::draw::Poly{ points, color, thick, lt, shift });
    cv::MediaFrame nv12 = cv::MediaFrame::Create<TestMediaNV12>(y_gapi_mat, uv_gapi_mat);
    cv::gapi::wip::draw::render(nv12, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        // NV12 -> YUV
        cv::Mat yuv;
        cv::gapi::wip::draw::cvtNV12ToYUV(y_ref_mat, uv_ref_mat, yuv);

        std::vector<std::vector<cv::Point>> pp{ points };
        cv::fillPoly(yuv, pp, cvtBGRToYUVC(color), lt, shift);

        // YUV -> NV12
        cv::gapi::wip::draw::cvtYUVToNV12(yuv, y_ref_mat, uv_ref_mat);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::norm(y_gapi_mat, y_ref_mat));
        EXPECT_EQ(0, cv::norm(uv_gapi_mat, uv_ref_mat));
    }
}

// FIXME avoid code duplicate for NV12 and BGR cases
INSTANTIATE_TEST_CASE_P(RenderBGROCVTestRectsImpl, RenderBGROCVTestRects,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(cv::Rect(100, 100, 200, 200)),
                                Values(cv::Scalar(100, 50, 150)),
                                Values(2),
                                Values(LINE_8, LINE_4),
                                Values(0, 1)));

INSTANTIATE_TEST_CASE_P(RenderNV12OCVTestRectsImpl, RenderNV12OCVTestRects,
                        Combine(Values(cv::Size(1280, 720)),
                                       Values(cv::Rect(100, 100, 200, 200)),
                                       Values(cv::Scalar(100, 50, 150)),
                                       Values(2),
                                       Values(LINE_8),
                                       Values(0)));

INSTANTIATE_TEST_CASE_P(RenderMFrameOCVTestRectsImpl, RenderMFrameOCVTestRects,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(cv::Rect(100, 100, 200, 200)),
                                Values(cv::Scalar(100, 50, 150)),
                                Values(2),
                                Values(LINE_8),
                                Values(0)));

INSTANTIATE_TEST_CASE_P(RenderBGROCVTestCirclesImpl, RenderBGROCVTestCircles,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(cv::Point(100, 100)),
                                Values(10),
                                Values(cv::Scalar(100, 50, 150)),
                                Values(2),
                                Values(LINE_8),
                                Values(0)));

INSTANTIATE_TEST_CASE_P(RenderNV12OCVTestCirclesImpl, RenderNV12OCVTestCircles,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(cv::Point(100, 100)),
                                Values(10),
                                Values(cv::Scalar(100, 50, 150)),
                                Values(2),
                                Values(LINE_8, LINE_4),
                                Values(0, 1)));

INSTANTIATE_TEST_CASE_P(RenderMFrameOCVTestCirclesImpl, RenderMFrameOCVTestCircles,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(cv::Point(100, 100)),
                                Values(10),
                                Values(cv::Scalar(100, 50, 150)),
                                Values(2),
                                Values(LINE_8),
                                Values(0)));

INSTANTIATE_TEST_CASE_P(RenderBGROCVTestLinesImpl, RenderBGROCVTestLines,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(cv::Point(100, 100)),
                                Values(cv::Point(200, 200)),
                                Values(cv::Scalar(100, 50, 150)),
                                Values(2),
                                Values(LINE_8),
                                Values(0)));

INSTANTIATE_TEST_CASE_P(RenderNV12OCVTestLinesImpl, RenderNV12OCVTestLines,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(cv::Point(100, 100)),
                                Values(cv::Point(200, 200)),
                                Values(cv::Scalar(100, 50, 150)),
                                Values(2),
                                Values(LINE_8),
                                Values(0)));

INSTANTIATE_TEST_CASE_P(RenderMFrameOCVTestLinesImpl, RenderMFrameOCVTestLines,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(cv::Point(100, 100)),
                                Values(cv::Point(200, 200)),
                                Values(cv::Scalar(100, 50, 150)),
                                Values(2),
                                Values(LINE_8),
                                Values(0)));

INSTANTIATE_TEST_CASE_P(RenderBGROCVTestTextsImpl, RenderBGROCVTestTexts,
                        Combine(Values(cv::Size(1280, 720)),
                                Values("SomeText"),
                                Values(cv::Point(200, 200)),
                                Values(FONT_HERSHEY_SIMPLEX),
                                Values(2.0),
                                Values(cv::Scalar(0, 255, 0)),
                                Values(2),
                                Values(LINE_8),
                                Values(false)));

INSTANTIATE_TEST_CASE_P(RenderNV12OCVTestTextsImpl, RenderNV12OCVTestTexts,
                        Combine(Values(cv::Size(1280, 720)),
                                Values("SomeText"),
                                Values(cv::Point(200, 200)),
                                Values(FONT_HERSHEY_SIMPLEX),
                                Values(2.0),
                                Values(cv::Scalar(0, 255, 0)),
                                Values(2),
                                Values(LINE_8),
                                Values(false)));

INSTANTIATE_TEST_CASE_P(RenderMFrameOCVTestTextsImpl, RenderMFrameOCVTestTexts,
                        Combine(Values(cv::Size(1280, 720)),
                                Values("SomeText"),
                                Values(cv::Point(200, 200)),
                                Values(FONT_HERSHEY_SIMPLEX),
                                Values(2.0),
                                Values(cv::Scalar(0, 255, 0)),
                                Values(2),
                                Values(LINE_8),
                                Values(false)));


#ifdef HAVE_FREETYPE

INSTANTIATE_TEST_CASE_P(RenderBGROCVTestFTextsImpl, RenderBGROCVTestFTexts,
                        Combine(Values(cv::Size(1280, 720)),
                            Values(to_wstring("\xe4\xbd\xa0\xe5\xa5\xbd\xef\xbc\x8c\xe4\xb8\x96\xe7\x95\x8c"),
                                   to_wstring("\xe3\x80\xa4\xe3\x80\xa5\xe3\x80\xa6\xe3\x80\xa7\xe3\x80\xa8\xe3\x80\x85\xe3\x80\x86")),
                            Values(cv::Point(200, 200)),
                            Values(64),
                            Values(cv::Scalar(0, 255, 0))));

INSTANTIATE_TEST_CASE_P(RenderNV12OCVTestFTextsImpl, RenderNV12OCVTestFTexts,
                        Combine(Values(cv::Size(1280, 720)),
                            Values(to_wstring("\xe4\xbd\xa0\xe5\xa5\xbd\xef\xbc\x8c\xe4\xb8\x96\xe7\x95\x8c"),
                                   to_wstring("\xe3\x80\xa4\xe3\x80\xa5\xe3\x80\xa6\xe3\x80\xa7\xe3\x80\xa8\xe3\x80\x85\xe3\x80\x86")),
                            Values(cv::Point(200, 200)),
                            Values(64),
                            Values(cv::Scalar(0, 255, 0))));

INSTANTIATE_TEST_CASE_P(RenderMFrameOCVTestFTextsImpl, RenderMFrameOCVTestFTexts,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(to_wstring("\xe4\xbd\xa0\xe5\xa5\xbd\xef\xbc\x8c\xe4\xb8\x96\xe7\x95\x8c"),
                                to_wstring("\xe3\x80\xa4\xe3\x80\xa5\xe3\x80\xa6\xe3\x80\xa7\xe3\x80\xa8\xe3\x80\x85\xe3\x80\x86")),
                                Values(cv::Point(200, 200)),
                                Values(64),
                                Values(cv::Scalar(0, 255, 0))));

#endif // HAVE_FREETYPE

// FIXME Implement a macros to instantiate the tests because BGR and NV12 have the same parameters

INSTANTIATE_TEST_CASE_P(RenderBGROCVTestMosaicsImpl, RenderBGROCVTestMosaics,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(cv::Rect(100, 100, 200, 200),      // Normal case
                                       cv::Rect(-50, -50, 200, 200),      // Intersection with left-top corner
                                       cv::Rect(-50, 100, 200, 200),      // Intersection with left side
                                       cv::Rect(-50, 600, 200, 200),      // Intersection with left-bottom corner
                                       cv::Rect(100, 600, 200, 200),      // Intersection with bottom side
                                       cv::Rect(1200, 700, 200, 200),     // Intersection with right-bottom corner
                                       cv::Rect(1200, 400, 200, 200),     // Intersection with right side
                                       cv::Rect(1200, -50, 200, 200),     // Intersection with right-top corner
                                       cv::Rect(500, -50, 200, 200),      // Intersection with top side
                                       cv::Rect(-100, 300, 1480, 300),    // From left to right side with intersection
                                       cv::Rect(5000, 2000, 100, 100),    // Outside image
                                       cv::Rect(-300, -300, 3000, 3000),  // Cover all image
                                       cv::Rect(100, 100, -500, -500)),   // Negative width and height
                                Values(25),
                                Values(0)));

INSTANTIATE_TEST_CASE_P(RenderNV12OCVTestMosaicsImpl, RenderNV12OCVTestMosaics,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(cv::Rect(100, 100, 200, 200),      // Normal case
                                       cv::Rect(-50, -50, 200, 200),      // Intersection with left-top corner
                                       cv::Rect(-50, 100, 200, 200),      // Intersection with left side
                                       cv::Rect(-50, 600, 200, 200),      // Intersection with left-bottom corner
                                       cv::Rect(100, 600, 200, 200),      // Intersection with bottom side
                                       cv::Rect(1200, 700, 200, 200),     // Intersection with right-bottom corner
                                       cv::Rect(1200, 400, 200, 200),     // Intersection with right side
                                       cv::Rect(1200, -50, 200, 200),     // Intersection with right-top corner
                                       cv::Rect(500, -50, 200, 200),      // Intersection with top side
                                       cv::Rect(-100, 300, 1480, 300),    // From left to right side with intersection
                                       cv::Rect(5000, 2000, 100, 100),    // Outside image
                                       cv::Rect(-300, -300, 3000, 3000),  // Cover all image
                                       cv::Rect(100, 100, -500, -500)),   // Negative width and height
                                Values(25),
                                Values(0)));

INSTANTIATE_TEST_CASE_P(RenderMFrameOCVTestMosaicsImpl, RenderMFrameOCVTestMosaics,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(cv::Rect(100, 100, 200, 200),      // Normal case
                                       cv::Rect(-50, -50, 200, 200),      // Intersection with left-top corner
                                       cv::Rect(-50, 100, 200, 200),      // Intersection with left side
                                       cv::Rect(-50, 600, 200, 200),      // Intersection with left-bottom corner
                                       cv::Rect(100, 600, 200, 200),      // Intersection with bottom side
                                       cv::Rect(1200, 700, 200, 200),     // Intersection with right-bottom corner
                                       cv::Rect(1200, 400, 200, 200),     // Intersection with right side
                                       cv::Rect(1200, -50, 200, 200),     // Intersection with right-top corner
                                       cv::Rect(500, -50, 200, 200),      // Intersection with top side
                                       cv::Rect(-100, 300, 1480, 300),    // From left to right side with intersection
                                       cv::Rect(5000, 2000, 100, 100),    // Outside image
                                       cv::Rect(-300, -300, 3000, 3000),  // Cover all image
                                       cv::Rect(100, 100, -500, -500)),   // Negative width and height
                                Values(25),
                                Values(0)));

INSTANTIATE_TEST_CASE_P(RenderBGROCVTestImagesImpl, RenderBGROCVTestImages,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(cv::Rect(100, 100, 200, 200)),
                                Values(cv::Scalar(100, 150, 60)),
                                Values(1.0)));

INSTANTIATE_TEST_CASE_P(RenderNV12OCVTestImagesImpl, RenderNV12OCVTestImages,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(cv::Rect(100, 100, 200, 200)),
                                Values(cv::Scalar(100, 150, 60)),
                                Values(1.0)));

INSTANTIATE_TEST_CASE_P(RenderMFrameOCVTestImagesImpl, RenderMFrameOCVTestImages,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(cv::Rect(100, 100, 200, 200)),
                                Values(cv::Scalar(100, 150, 60)),
                                Values(1.0)));

INSTANTIATE_TEST_CASE_P(RenderBGROCVTestPolylinesImpl, RenderBGROCVTestPolylines,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(std::vector<cv::Point>{{100, 100}, {200, 200}, {150, 300}, {400, 150}}),
                                Values(cv::Scalar(100, 150, 60)),
                                Values(2),
                                Values(LINE_8),
                                Values(0)));

INSTANTIATE_TEST_CASE_P(RenderNV12OCVTestPolylinesImpl, RenderNV12OCVTestPolylines,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(std::vector<cv::Point>{{100, 100}, {200, 200}, {150, 300}, {400, 150}}),
                                Values(cv::Scalar(100, 150, 60)),
                                Values(2),
                                Values(LINE_8),
                                Values(0)));

INSTANTIATE_TEST_CASE_P(RenderMFrameOCVTestPolylinesImpl, RenderMFrameOCVTestPolylines,
                        Combine(Values(cv::Size(1280, 720)),
                                Values(std::vector<cv::Point>{ {100, 100}, { 200, 200 }, { 150, 300 }, { 400, 150 }}),
                                Values(cv::Scalar(100, 150, 60)),
                                Values(2),
                                Values(LINE_8),
                                Values(0)));
}
