// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2019 Intel Corporation


#ifndef OPENCV_GAPI_RENDER_TESTS_INL_HPP
#define OPENCV_GAPI_RENDER_TESTS_INL_HPP

#include <opencv2/gapi/render/render.hpp>
#include "gapi_render_tests.hpp"

namespace opencv_test
{

inline cv::Scalar cvtBGRToYUVC(const cv::Scalar& bgr)
{
    double y = bgr[2] *  0.299000 + bgr[1] *  0.587000 + bgr[0] *  0.114000;
    double u = bgr[2] * -0.168736 + bgr[1] * -0.331264 + bgr[0] *  0.500000 + 128;
    double v = bgr[2] *  0.500000 + bgr[1] * -0.418688 + bgr[0] * -0.081312 + 128;
    return {y, u, v};
}

inline void drawMosaicRef(const cv::Mat& mat, const cv::Rect &rect, int cellSz)
{
    cv::Mat msc_roi = mat(rect);
    int crop_x = msc_roi.cols - msc_roi.cols % cellSz;
    int crop_y = msc_roi.rows - msc_roi.rows % cellSz;

    for(int i = 0; i < crop_y; i += cellSz ) {
        for(int j = 0; j < crop_x; j += cellSz) {
            auto cell_roi = msc_roi(cv::Rect(j, i, cellSz, cellSz));
            cell_roi = cv::mean(cell_roi);
        }
    }
}

inline void blendImageRef(cv::Mat& mat, const cv::Point& org, const cv::Mat& img, const cv::Mat& alpha)
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
};

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

//TEST_P(RenderTextTestBGR, AccuracyTest)
//{
    //cv::Size size;
    //cv::gapi::wip::draw::Prim prim;
    //std::tie(size, prim) = GetParam();
    //const auto& text_p = util::get<cv::gapi::wip::draw::Text>(prim);

    //cv::Mat ref_mat(size, CV_8UC3, cv::Scalar::all(255));
    //cv::Mat gapi_mat;
    //ref_mat.copyTo(gapi_mat);

    //// G-API code //////////////////////////////////////////////////////////////
    //cv::gapi::wip::draw::render(gapi_mat, {prim});

    //// OpenCV code //////////////////////////////////////////////////////////////
    //{
        //cv::putText(ref_mat, text_p.text, text_p.org, text_p.ff,
                    //text_p.fs, text_p.color, text_p.thick, text_p.lt);
    //}

    //// Comparison //////////////////////////////////////////////////////////////
    //EXPECT_EQ(0, cv::norm(ref_mat, gapi_mat));
//}

//TEST_P(RenderTextTestNV12, AccuracyTest)
//{
    //cv::Size size;
    //cv::gapi::wip::draw::Prim prim;
    //std::tie(size, prim) = GetParam();
    //const auto& text_p = util::get<cv::gapi::wip::draw::Text>(prim);

    //cv::Mat ref_mat(size, CV_8UC3, cv::Scalar::all(255));
    //cv::Mat gapi_mat;
    //ref_mat.copyTo(gapi_mat);
    //cv::Mat y_gapi, uv_gapi, y_ref, uv_ref;

    //cv::gapi::wip::draw::BGR2NV12(gapi_mat, y_gapi, uv_gapi);
    //cv::gapi::wip::draw::BGR2NV12(ref_mat, y_ref, uv_ref);

    //// G-API code //////////////////////////////////////////////////////////////
    //cv::gapi::wip::draw::render(y_gapi, uv_gapi, {prim});

    //// OpenCV code //////////////////////////////////////////////////////////////
    //{
        //// NV12 -> YUV
        //cv::Mat upsample_uv, yuv;
        //cv::resize(uv_ref, upsample_uv, uv_ref.size() * 2, cv::INTER_LINEAR);
        //cv::merge(std::vector<cv::Mat>{y_ref, upsample_uv}, yuv);
        //double y = text_p.color[2] *  0.299000 + text_p.color[1] *  0.587000 + text_p.color[0] *  0.114000;
        //double u = text_p.color[2] * -0.168736 + text_p.color[1] * -0.331264 + text_p.color[0] *  0.500000 + 128;
        //double v = text_p.color[2] *  0.500000 + text_p.color[1] * -0.418688 + text_p.color[0] * -0.081312 + 128;
        //cv::Scalar yuv_color{y, u, v};

        //cv::putText(yuv, text_p.text, text_p.org, text_p.ff, text_p.fs, yuv_color, text_p.thick, text_p.lt);

        //// YUV -> NV12
        //std::vector<cv::Mat> chs(3);
        //cv::split(yuv, chs);
        //cv::merge(std::vector<cv::Mat>{chs[1], chs[2]}, uv_ref);
        //y_ref = chs[0];
        //cv::resize(uv_ref, uv_ref, uv_ref.size() / 2, cv::INTER_LINEAR);
    //}

    //// Comparison //////////////////////////////////////////////////////////////
    //EXPECT_EQ(0, cv::norm(y_ref, y_gapi));
    //EXPECT_EQ(0, cv::norm(uv_ref, uv_gapi));
//}

} // opencv_test

#endif //OPENCV_GAPI_RENDER_TESTS_INL_HPP
