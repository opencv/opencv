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
    void Init()
    {
        MatType type = CV_8UC3;
        out_mat_ocv  = cv::Mat(sz, type, cv::Scalar(255));
        out_mat_gapi = cv::Mat(sz, type, cv::Scalar(255));

        if (isNV12Format) {
            /* NB: When converting data from BGR to NV12, data loss occurs,
             * so the reference data is subjected to the same transformation
             * for correct comparison of the test results */
            cv::gapi::wip::draw::BGR2NV12(out_mat_ocv, y, uv);
            cv::cvtColorTwoPlane(y, uv, out_mat_ocv, cv::COLOR_YUV2BGR_NV12);
        }
    }

    void Run()
    {
        if (isNV12Format) {
            cv::gapi::wip::draw::BGR2NV12(out_mat_gapi, y, uv);
            cv::gapi::wip::draw::render(y, uv, prims);
            cv::cvtColorTwoPlane(y, uv, out_mat_gapi, cv::COLOR_YUV2BGR_NV12);

            // NB: Also due to data loss
            cv::gapi::wip::draw::BGR2NV12(out_mat_ocv, y, uv);
            cv::cvtColorTwoPlane(y, uv, out_mat_ocv, cv::COLOR_YUV2BGR_NV12);
        } else {
            cv::gapi::wip::draw::render(out_mat_gapi, prims);
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
