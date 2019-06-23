// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_RENDER_TESTS_INL_HPP
#define OPENCV_GAPI_RENDER_TESTS_INL_HPP

#include "gapi_render_tests.hpp"
#include "api/render_priv.hpp"

#include <opencv2/gapi/render.hpp>

namespace opencv_test
{

TEST_P(RenderTextTest, AccuracyTest)
{
    MatType type = CV_8UC3;
    cv::Size sz;
    bool initOut = true;
    std::string text;
    std::vector<cv::Point> points;
    int         ff;
    double      fs;
    cv::Scalar  color;
    int         thick;
    int         lt;
    bool        blo;

    std::tie(sz, text, points, ff, fs, color, thick, lt, blo) = GetParam();
    initMatsRandU(type, sz, type, initOut);

    std::vector<cv::gapi::wip::draw::Prim> prims;

    for (const auto& p : points) {
        cv::putText(out_mat_ocv, text, p, ff, fs, color, thick);
        prims.emplace_back(cv::gapi::wip::draw::Text{text, p, ff, fs, color, thick, lt, false});
    }

    if (true) {
        cv::Mat y, uv;
        cv::gapi::wip::draw::BGR2NV12(out_mat_gapi, y, uv);

        cv::gapi::wip::draw::render(y, uv, prims);
        cv::cvtColorTwoPlane(y, uv, out_mat_gapi, cv::COLOR_YUV2BGR_NV12);

        cv::gapi::wip::draw::BGR2NV12(out_mat_ocv, y, uv);
        cv::cvtColorTwoPlane(y, uv, out_mat_ocv, cv::COLOR_YUV2BGR_NV12);
    } else {
        cv::gapi::wip::draw::render(out_mat_gapi, prims);
    }

    EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
}

} // opencv_test

#endif //OPENCV_GAPI_RENDER_TESTS_INL_HPP
