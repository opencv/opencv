// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_RENDER_TESTS_INL_HPP
#define OPENCV_GAPI_RENDER_TESTS_INL_HPP

#include "gapi_render_tests.hpp"

#include <opencv2/gapi/render.hpp>

namespace opencv_test
{

TEST_P(RenderTextTest, AccuracyTest)
{
    std::vector<cv::Point> points;
    std::string text;
    int         ff;
    double      fs;
    bool        blo;

    std::tie(sz, text, points, ff, fs, color, thick, lt, blo, isNV12Format) = GetParam();
    Init();

    for (const auto& p : points) {
        cv::putText(out_mat_ocv, text, p, ff, fs, color, thick, lt, blo);
        prims.emplace_back(cv::gapi::wip::draw::Text{text, p, ff, fs, color, thick, lt, blo});
    }

    Run();

    EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
}

TEST_P(RenderRectTest, AccuracyTest)
{
    std::vector<cv::Rect> rects;
    int shift;

    std::tie(sz, rects, color, thick, lt, shift, isNV12Format) = GetParam();
    Init();

    for (const auto& r : rects) {
        cv::rectangle(out_mat_ocv, r, color, thick, lt, shift);
        prims.emplace_back(cv::gapi::wip::draw::Rect{r, color, thick, lt, shift});
    }

    Run();

    EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
}

TEST_P(RenderCircleTest, AccuracyTest)
{
    std::vector<cv::Point> points;
    int radius;
    int shift;

    std::tie(sz, points, radius, color, thick, lt, shift, isNV12Format) = GetParam();
    Init();

    for (const auto& p : points) {
        cv::circle(out_mat_ocv, p, radius, color, thick, lt, shift);
        prims.emplace_back(cv::gapi::wip::draw::Circle{p, radius, color, thick, lt, shift});
    }

    Run();

    EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
}

TEST_P(RenderLineTest, AccuracyTest)
{
    std::vector<std::pair<cv::Point, cv::Point>> points;
    int shift;

    std::tie(sz, points, color, thick, lt, shift, isNV12Format) = GetParam();
    Init();

    for (const auto& p : points) {
        cv::line(out_mat_ocv, p.first, p.second, color, thick, lt, shift);
        prims.emplace_back(cv::gapi::wip::draw::Line{p.first, p.second, color, thick, lt, shift});
    }

    Run();

    EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
}

} // opencv_test

#endif //OPENCV_GAPI_RENDER_TESTS_INL_HPP
