// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_RENDER_TESTS_HPP
#define OPENCV_GAPI_RENDER_TESTS_HPP

#include "gapi_tests_common.hpp"
#include "api/render_priv.hpp"

namespace opencv_test
{

using Points            = std::vector<cv::Point>;
using Rects             = std::vector<cv::Rect>;
using PairOfPoints      = std::pair<cv::Point, cv::Point>;
using VecOfPairOfPoints = std::vector<PairOfPoints>;

template<class T>
class RenderWithParam : public TestWithParam<T>
{
protected:
    void Init(const cv::Size& size, bool isNV12)
    {
        MatType type = CV_8UC3;

        out_mat_ocv  = cv::Mat(size, type, cv::Scalar(0));
        out_mat_gapi = cv::Mat(size, type, cv::Scalar(0));

        if (isNV12) {
            cv::gapi::wip::draw::BGR2NV12(out_mat_ocv, y, uv);
            cv::cvtColorTwoPlane(y, uv, out_mat_ocv, cv::COLOR_YUV2BGR_NV12);
        }
    }

    void Run(cv::Mat& out_mat, cv::Mat& ref_mat, bool isNV12)
    {
        if (isNV12) {
            cv::gapi::wip::draw::BGR2NV12(out_mat, y, uv);
            cv::gapi::wip::draw::render(y, uv, prims);
            cv::cvtColorTwoPlane(y, uv, out_mat, cv::COLOR_YUV2BGR_NV12);

            cv::gapi::wip::draw::BGR2NV12(ref_mat, y, uv);
            cv::cvtColorTwoPlane(y, uv, ref_mat, cv::COLOR_YUV2BGR_NV12);
        } else {
            cv::gapi::wip::draw::render(out_mat, prims);
        }
    }

    cv::Size sz;
    cv::Scalar color;
    int thick;
    int lt;
    bool isNV12Format;
    std::vector<cv::gapi::wip::draw::Prim> prims;
    cv::Mat y, uv;
    cv::Mat out_mat_ocv, out_mat_gapi;
};

struct RenderTextTest   : public RenderWithParam <std::tuple<cv::Size,std::string,Points,int,double,cv::Scalar,int,int,bool,bool>> {};
struct RenderRectTest   : public RenderWithParam <std::tuple<cv::Size,Rects,cv::Scalar,int,int,int,bool>>                          {};
struct RenderCircleTest : public RenderWithParam <std::tuple<cv::Size,Points,int,cv::Scalar,int,int,int,bool>>                     {};
struct RenderLineTest   : public RenderWithParam <std::tuple<cv::Size,VecOfPairOfPoints,cv::Scalar,int,int,int,bool>>              {};

} // opencv_test

#endif //OPENCV_GAPI_RENDER_TESTS_HPP

