// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_RENDER_TESTS_INL_HPP
#define OPENCV_GAPI_RENDER_TESTS_INL_HPP

// FIXME
#include <iostream>
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
    cv::Scalar  color;
    int         thick;
    int         lt;
    bool        blo;

    std::tie(text, points, ff, fs, color, thick, lt, blo) = GetParam();

    cv::Mat ref_mat(300, 300, CV_8UC3);
    cv::Mat out_mat(300, 300, CV_8UC3);
    std::vector<cv::gapi::wip::draw::Prim> prims;

    for (const auto& p : points) {
        cv::putText(ref_mat, text, p, ff, fs, color, thick);
        prims.emplace_back(cv::gapi::wip::draw::Text{text, p, ff, fs, color, thick, lt, false});
    }
    cv::imshow("ref_mat", ref_mat);
    cv::waitKey(0);

    cv::gapi::wip::draw::render(out_mat, prims);

    // EXPECT_EQ(0, cv::countNonZero(out_mat != ref_mat));
}

} // opencv_test

#endif //OPENCV_GAPI_RENDER_TESTS_INL_HPP

