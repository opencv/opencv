// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_IMGPROC_TESTS_COMMON_HPP
#define OPENCV_GAPI_IMGPROC_TESTS_COMMON_HPP

#include "gapi_tests_common.hpp"
#include "../../include/opencv2/gapi/imgproc.hpp"

#include <opencv2/imgproc.hpp>

namespace opencv_test
{
template<typename G_In, typename In>
static cv::GComputation boundingRectTestGAPI(const In& in, cv::GCompileArgs&& args,
                                             cv::Rect& out_rect_gapi)
{
    G_In g_in;
    auto out = cv::gapi::boundingRect(g_in);
    cv::GComputation c(cv::GIn(g_in), cv::GOut(out));
    c.apply(cv::gin(in), cv::gout(out_rect_gapi), std::move(args));
    return c;
}

template<typename In>
static void boundingRectTestOpenCVCompare(const In& in, const cv::Rect& out_rect_gapi,
                                          const compare_rect_f& cmpF)
{
    // OpenCV code /////////////////////////////////////////////////////////////
    cv::Rect out_rect_ocv = cv::boundingRect(in);
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_rect_gapi, out_rect_ocv));
    }
}

template<typename G_In, typename In>
static void boundingRectTestBody(const In& in, const CompareRects& cmpF_, cv::GCompileArgs&& args)
{
    cv::Rect out_rect_gapi;
    boundingRectTestGAPI<G_In>(in, std::move(args), out_rect_gapi);

    compare_rect_f cmpF = [cmpF_](const cv::Rect& a, const cv::Rect& b){ return cmpF_(a, b); };
    boundingRectTestOpenCVCompare(in, out_rect_gapi, cmpF);
}

} // namespace opencv_test

#endif // OPENCV_GAPI_VIDEO_TESTS_COMMON_HPP
