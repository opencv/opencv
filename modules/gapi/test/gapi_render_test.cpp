// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation

#include "test_precomp.hpp"

#include "api/render_priv.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/gapi/render.hpp>
#include <opencv2/gapi/own/scalar.hpp>

namespace opencv_test
{

namespace
{
    struct RenderTestFixture : public ::testing::Test
    {
        cv::Size size  = {30, 40};
        int      thick = 2;
        int      ff    = cv::FONT_HERSHEY_SIMPLEX;
        int      lt    = LINE_8;
        double   fs    = 1;

        cv::Mat     ref_mat {320, 480, CV_8UC3, cv::Scalar::all(255)};
        cv::Mat     out_mat {320, 480, CV_8UC3, cv::Scalar::all(255)};
        cv::Scalar  color   {0, 255, 0};
        std::string text    {"some text"};

    };
} // namespace

TEST(BGR2NV12Test, CorrectConversion)
{
    cv::Mat in_mat(320, 240, CV_8UC3);
    cv::Mat out_y, out_uv, ref_y, yuv;
    cv::randu(in_mat, cv::Scalar::all(0), cv::Scalar::all(255));

    cv::cvtColor(in_mat, yuv, cv::COLOR_BGR2YUV);
    cv::Mat channels[3];
    cv::split(yuv, channels);

    ref_y = channels[0];
    cv::Mat ref_uv(in_mat.size() / 2, CV_8UC2);
    cv::resize(channels[1], channels[1], channels[1].size() / 2, 0, 0, cv::INTER_NEAREST);
    cv::resize(channels[2], channels[2], channels[2].size() / 2, 0, 0, cv::INTER_NEAREST);
    cv::merge(channels + 1, 2, ref_uv);

    cv::gapi::wip::draw::BGR2NV12(in_mat, out_y, out_uv);

    EXPECT_EQ(0, cv::countNonZero(out_y  != ref_y));
    EXPECT_EQ(0, cv::countNonZero(out_uv != ref_uv));
}

TEST_F(RenderTestFixture, PutText)
{
    std::vector<cv::gapi::wip::draw::Prim> prims;

    for (int i = 0; i < 5; ++i)
    {
        cv::Point point {30 + i * 60, 40 + i * 50};

        cv::putText(ref_mat, text, point, ff, fs, color, thick);
        prims.emplace_back(cv::gapi::wip::draw::Text{text, point, ff, fs, color, thick, lt, false});
    }

    cv::gapi::wip::draw::render(out_mat, prims);

    EXPECT_EQ(0, cv::countNonZero(out_mat != ref_mat));
}

TEST_F(RenderTestFixture, Rectangle)
{
    std::vector<cv::gapi::wip::draw::Prim> prims;

    for (int i = 0; i < 5; ++i)
    {
        cv::Rect rect {30 + i * 60, 40 + i * 50, size.width, size.height};
        cv::rectangle(ref_mat, rect, color, thick);
        prims.emplace_back(cv::gapi::wip::draw::Rect{rect, color, thick, lt, 0});
    }

    cv::gapi::wip::draw::render(out_mat, prims);

    EXPECT_EQ(0, cv::countNonZero(out_mat != ref_mat));
}

TEST_F(RenderTestFixture, PutTextAndRectangle)
{
    std::vector<cv::gapi::wip::draw::Prim> prims;

    for (int i = 0; i < 5; ++i)
    {
        cv::Point point {30 + i * 60, 40 + i * 50};
        cv::Rect  rect {point, size};

        cv::rectangle(ref_mat, rect, color, thick);
        cv::putText(ref_mat, text, point, ff, fs, color, thick);

        prims.emplace_back(cv::gapi::wip::draw::Rect{rect, color, thick, lt, 0});
        prims.emplace_back(cv::gapi::wip::draw::Text{text, point, ff, fs, color, thick, lt, false});
    }

    cv::gapi::wip::draw::render(out_mat, prims);

    EXPECT_EQ(0, cv::countNonZero(out_mat != ref_mat));
}

TEST_F(RenderTestFixture, PutTextAndRectangleNV12)
{
    cv::Mat y;
    cv::Mat uv;
    cv::gapi::wip::draw::BGR2NV12(out_mat, y, uv);

    std::vector<cv::gapi::wip::draw::Prim> prims;

    for (int i = 0; i < 5; ++i)
    {
        cv::Point point {30 + i * 60, 40 + i * 50};
        cv::Rect  rect {point, size};

        cv::rectangle(ref_mat, rect, color, thick);
        cv::putText(ref_mat, text, point, ff, fs, color, thick);

        prims.emplace_back(cv::gapi::wip::draw::Rect{rect, color, thick, lt, 0});
        prims.emplace_back(cv::gapi::wip::draw::Text{text, point, ff, fs, color, thick, lt, false});
    }

    cv::gapi::wip::draw::render(y, uv, prims);
    cv::cvtColorTwoPlane(y, uv, out_mat, cv::COLOR_YUV2BGR_NV12);

    cv::gapi::wip::draw::BGR2NV12(ref_mat, y, uv);
    cv::cvtColorTwoPlane(y, uv, ref_mat, cv::COLOR_YUV2BGR_NV12);

    EXPECT_EQ(0, cv::countNonZero(out_mat != ref_mat));
}

} // opencv_test
