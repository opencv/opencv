// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation

#include "test_precomp.hpp"
#include "opencv2/gapi/render.hpp"

namespace opencv_test
{

namespace
{
    struct RenderTestFixture : public ::testing::Test
    {
        int w     = 30;
        int h     = 40;
        int thick = 2;
        int fs    = 1;
        int ff    = cv::FONT_HERSHEY_SIMPLEX;
        int lt    = LINE_8;

        cv::Mat     ref_mat {320, 480, CV_8UC3, cv::Scalar::all(0)};
        cv::Mat     out_mat {320, 480, CV_8UC3, cv::Scalar::all(0)};
        cv::Scalar  color   {0, 255, 0};
        std::string text    {"some text"};
    };
} // namespace

TEST_F(RenderTestFixture, PutText)
{
    std::vector<cv::gapi::DrawEvent> events;

    for (int i = 0; i < 5; ++i)
    {
        int pos_x = 30 + i * 60;
        int pos_y = 40 + i * 50;

        cv::putText(ref_mat, text, cv::Point(pos_x, pos_y), ff, fs, color, thick);
        events.emplace_back(cv::gapi::TextEvent{text, pos_x, pos_y, ff, fs, color, thick, lt, false});
    }

    cv::gapi::render(out_mat, events);

    EXPECT_EQ(0, cv::countNonZero(out_mat != ref_mat));
}

TEST_F(RenderTestFixture, Rectangle)
{
    std::vector<cv::gapi::DrawEvent> events;

    for (int i = 0; i < 5; ++i)
    {
        int pos_x = 30 + i * 60;
        int pos_y = 40 + i * 50;

        cv::rectangle(ref_mat, cv::Rect(pos_x, pos_y, w, h), color, thick);
        events.emplace_back(cv::gapi::RectEvent{pos_x, pos_y, w, h, color, thick, lt, 0});
    }

    cv::gapi::render(out_mat, events);

    EXPECT_EQ(0, cv::countNonZero(out_mat != ref_mat));
}

}  // opencv_test
