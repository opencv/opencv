// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2019 Intel Corporation


#ifndef OPENCV_GAPI_RENDER_TESTS_INL_HPP
#define OPENCV_GAPI_RENDER_TESTS_INL_HPP

#include <opencv2/gapi/render.hpp>
#include "gapi_render_tests.hpp"

namespace opencv_test
{

TEST_P(RenderNV12, AccuracyTest)
{
    std::tie(sz, prims, pkg) = GetParam();
    Init();

    cv::gapi::wip::draw::BGR2NV12(mat_gapi, y_mat_gapi, uv_mat_gapi);
    cv::gapi::wip::draw::render(y_mat_gapi, uv_mat_gapi, prims, pkg);

    ComputeRef();

    EXPECT_EQ(0, cv::norm(y_mat_gapi, y_mat_ocv));
    EXPECT_EQ(0, cv::norm(uv_mat_gapi, uv_mat_ocv));
}

TEST_P(RenderBGR, AccuracyTest)
{
    std::tie(sz, prims, pkg) = GetParam();
    Init();

    cv::gapi::wip::draw::render(mat_gapi, prims, pkg);
    ComputeRef();

    EXPECT_EQ(0, cv::norm(mat_gapi, mat_ocv));
}

TEST_P(RenderTextTestBGR, AccuracyTest)
{
    cv::Size size;
    cv::gapi::wip::draw::Prim prim;
    std::tie(size, prim) = GetParam();
    const auto& text_p = util::get<cv::gapi::wip::draw::Text>(prim);

    cv::Mat ref_mat(size, CV_8UC3, cv::Scalar::all(255));
    cv::Mat gapi_mat;
    ref_mat.copyTo(gapi_mat);

    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::render(gapi_mat, {prim});

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        cv::putText(ref_mat, text_p.text, text_p.org, text_p.ff,
                    text_p.fs, text_p.color, text_p.thick, text_p.lt);
    }

    // Comparison //////////////////////////////////////////////////////////////
    EXPECT_EQ(0, cv::norm(ref_mat, gapi_mat));
}

TEST_P(RenderTextTestNV12, AccuracyTest)
{
    cv::Size size;
    cv::gapi::wip::draw::Prim prim;
    std::tie(size, prim) = GetParam();
    const auto& text_p = util::get<cv::gapi::wip::draw::Text>(prim);

    cv::Mat ref_mat(size, CV_8UC3, cv::Scalar::all(255));
    cv::Mat gapi_mat;
    ref_mat.copyTo(gapi_mat);
    cv::Mat y_gapi, uv_gapi, y_ref, uv_ref;

    cv::gapi::wip::draw::BGR2NV12(gapi_mat, y_gapi, uv_gapi);
    cv::gapi::wip::draw::BGR2NV12(ref_mat, y_ref, uv_ref);

    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::render(y_gapi, uv_gapi, {prim});

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        // NV12 -> YUV
        cv::Mat upsample_uv, yuv;
        cv::resize(uv_ref, upsample_uv, uv_ref.size() * 2, cv::INTER_LINEAR);
        cv::merge(std::vector<cv::Mat>{y_ref, upsample_uv}, yuv);
        double y = text_p.color[2] *  0.299000 + text_p.color[1] *  0.587000 + text_p.color[0] *  0.114000;
        double u = text_p.color[2] * -0.168736 + text_p.color[1] * -0.331264 + text_p.color[0] *  0.500000 + 128;
        double v = text_p.color[2] *  0.500000 + text_p.color[1] * -0.418688 + text_p.color[0] * -0.081312 + 128;
        cv::Scalar yuv_color{y, u, v};

        cv::putText(yuv, text_p.text, text_p.org, text_p.ff, text_p.fs, yuv_color, text_p.thick, text_p.lt);

        // YUV -> NV12
        std::vector<cv::Mat> chs(3);
        cv::split(yuv, chs);
        cv::merge(std::vector<cv::Mat>{chs[1], chs[2]}, uv_ref);
        y_ref = chs[0];
        cv::resize(uv_ref, uv_ref, uv_ref.size() / 2, cv::INTER_LINEAR);
    }

    // Comparison //////////////////////////////////////////////////////////////
    EXPECT_EQ(0, cv::norm(y_ref, y_gapi));
    EXPECT_EQ(0, cv::norm(uv_ref, uv_gapi));
}

} // opencv_test

#endif //OPENCV_GAPI_RENDER_TESTS_INL_HPP
