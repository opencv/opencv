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
// Draw random ellipses on given cv::Mat of given size and type
static void initMatForFindingContours(cv::Mat& mat, const cv::Size& sz, const int type)
{
    cv::RNG& rng = theRNG();
    mat = cv::Mat(sz, type, cv::Scalar::all(0));
    const size_t numEllipses = rng.uniform(1, 10);

    for( size_t i = 0; i < numEllipses; i++ )
    {
        cv::Point center;
        cv::Size  axes;
        center.x    = rng.uniform(0, sz.width);
        center.y    = rng.uniform(0, sz.height);
        axes.width  = rng.uniform(2, sz.width);
        axes.height = rng.uniform(2, sz.height);
        const int    color = rng.uniform(1, 256);
        const double angle = rng.uniform(0., 180.);
        cv::ellipse(mat, center, axes, angle, 0., 360., color, 1, FILLED);
    }
}

enum OptionalFindContoursOutput {NONE, HIERARCHY};

template<OptionalFindContoursOutput optional = NONE>
cv::GComputation findContoursTestGAPI(const cv::Mat& in, const cv::RetrievalModes mode,
                                      const cv::ContourApproximationModes method,
                                      cv::GCompileArgs&& args,
                                      std::vector<std::vector<cv::Point>>& out_cnts_gapi,
                                      std::vector<cv::Vec4i>& /*out_hier_gapi*/,
                                      const cv::Point& offset = cv::Point())
{
    cv::GMat g_in;
    cv::GOpaque<cv::Point> gOffset;
    cv::GArray<cv::GArray<cv::Point>> outCts;
    outCts = cv::gapi::findContours(g_in, mode, method, gOffset);
    cv::GComputation c(GIn(g_in, gOffset), GOut(outCts));
    c.apply(gin(in, offset), gout(out_cnts_gapi), std::move(args));
    return c;
}

template<> cv::GComputation findContoursTestGAPI<HIERARCHY> (
    const cv::Mat& in, const cv::RetrievalModes mode, const cv::ContourApproximationModes method,
    cv::GCompileArgs&& args, std::vector<std::vector<cv::Point>>& out_cnts_gapi,
    std::vector<cv::Vec4i>& out_hier_gapi, const cv::Point& offset)
{
    cv::GMat g_in;
    cv::GOpaque<cv::Point> gOffset;
    cv::GArray<cv::GArray<cv::Point>> outCts;
    cv::GArray<cv::Vec4i> outHier;
    std::tie(outCts, outHier) = cv::gapi::findContoursH(g_in, mode, method, gOffset);
    cv::GComputation c(GIn(g_in, gOffset), GOut(outCts, outHier));
    c.apply(gin(in, offset), gout(out_cnts_gapi, out_hier_gapi), std::move(args));
    return c;
}

template<OptionalFindContoursOutput optional = NONE>
void findContoursTestOpenCVCompare(const cv::Mat& in, const cv::RetrievalModes mode,
                                   const cv::ContourApproximationModes method,
                                   const std::vector<std::vector<cv::Point>>& out_cnts_gapi,
                                   const std::vector<cv::Vec4i>&              out_hier_gapi,
                                   const CompareMats& cmpF, const cv::Point& offset = cv::Point())
{
    // OpenCV code /////////////////////////////////////////////////////////////
    std::vector<std::vector<cv::Point>> out_cnts_ocv;
    std::vector<cv::Vec4i>              out_hier_ocv;
    cv::findContours(in, out_cnts_ocv, out_hier_ocv, mode, method, offset);
    // Comparison //////////////////////////////////////////////////////////////
    EXPECT_TRUE(out_cnts_gapi.size() == out_cnts_ocv.size());

    cv::Mat out_mat_ocv  = cv::Mat(cv::Size{ in.cols, in.rows }, in.type(), cv::Scalar::all(0));
    cv::Mat out_mat_gapi = cv::Mat(cv::Size{ in.cols, in.rows }, in.type(), cv::Scalar::all(0));
    cv::fillPoly(out_mat_ocv,  out_cnts_ocv,  cv::Scalar::all(1));
    cv::fillPoly(out_mat_gapi, out_cnts_gapi, cv::Scalar::all(1));
    EXPECT_TRUE(cmpF(out_mat_ocv, out_mat_gapi));
    if (optional == HIERARCHY)
    {
        EXPECT_TRUE(out_hier_ocv.size() == out_hier_gapi.size());
        EXPECT_TRUE(AbsExactVector<cv::Vec4i>().to_compare_f()(out_hier_ocv, out_hier_gapi));
    }
}

template<OptionalFindContoursOutput optional = NONE>
void findContoursTestBody(const cv::Size& sz, const MatType2& type, const cv::RetrievalModes mode,
                          const cv::ContourApproximationModes method, const CompareMats& cmpF,
                          cv::GCompileArgs&& args, const cv::Point& offset = cv::Point())
{
    cv::Mat in;
    initMatForFindingContours(in, sz, type);

    std::vector<std::vector<cv::Point>> out_cnts_gapi;
    std::vector<cv::Vec4i>              out_hier_gapi;
    findContoursTestGAPI<optional>(in, mode, method, std::move(args), out_cnts_gapi, out_hier_gapi,
                                   offset);
    findContoursTestOpenCVCompare<optional>(in, mode, method, out_cnts_gapi, out_hier_gapi, cmpF,
                                            offset);
}

//-------------------------------------------------------------------------------------------------

template<typename In>
static cv::GComputation boundingRectTestGAPI(const In& in, cv::GCompileArgs&& args,
                                             cv::Rect& out_rect_gapi)
{
    cv::detail::g_type_of_t<In> g_in;
    auto out = cv::gapi::boundingRect(g_in);
    cv::GComputation c(cv::GIn(g_in), cv::GOut(out));
    c.apply(cv::gin(in), cv::gout(out_rect_gapi), std::move(args));
    return c;
}

template<typename In>
static void boundingRectTestOpenCVCompare(const In& in, const cv::Rect& out_rect_gapi,
                                          const CompareRects& cmpF)
{
    // OpenCV code /////////////////////////////////////////////////////////////
    cv::Rect out_rect_ocv = cv::boundingRect(in);
    // Comparison //////////////////////////////////////////////////////////////
    EXPECT_TRUE(cmpF(out_rect_gapi, out_rect_ocv));
}

template<typename In>
static void boundingRectTestBody(const In& in, const CompareRects& cmpF, cv::GCompileArgs&& args)
{
    cv::Rect out_rect_gapi;
    boundingRectTestGAPI(in, std::move(args), out_rect_gapi);
    boundingRectTestOpenCVCompare(in, out_rect_gapi, cmpF);
}

//-------------------------------------------------------------------------------------------------

template<typename In>
static cv::GComputation fitLineTestGAPI(const In& in, const cv::DistanceTypes distType,
                                        cv::GCompileArgs&& args, cv::Vec4f& out_vec_gapi)
{
    const double paramDefault = 0., repsDefault = 0., aepsDefault = 0.;

    cv::detail::g_type_of_t<In> g_in;
    auto out = cv::gapi::fitLine2D(g_in, distType, paramDefault, repsDefault, aepsDefault);
    cv::GComputation c(cv::GIn(g_in), cv::GOut(out));
    c.apply(cv::gin(in), cv::gout(out_vec_gapi), std::move(args));
    return c;
}

template<typename In>
static cv::GComputation fitLineTestGAPI(const In& in, const cv::DistanceTypes distType,
                                        cv::GCompileArgs&& args, cv::Vec6f& out_vec_gapi)
{
    const double paramDefault = 0., repsDefault = 0., aepsDefault = 0.;

    cv::detail::g_type_of_t<In> g_in;
    auto out = cv::gapi::fitLine3D(g_in, distType, paramDefault, repsDefault, aepsDefault);
    cv::GComputation c(cv::GIn(g_in), cv::GOut(out));
    c.apply(cv::gin(in), cv::gout(out_vec_gapi), std::move(args));
    return c;
}

template<typename In, int dim>
static void fitLineTestOpenCVCompare(const In& in, const cv::DistanceTypes distType,
                                     const cv::Vec<float, dim>& out_vec_gapi,
                                     const CompareVecs<float, dim>& cmpF)
{
    const double paramDefault = 0., repsDefault = 0., aepsDefault = 0.;

    // OpenCV code /////////////////////////////////////////////////////////////
    cv::Vec<float, dim> out_vec_ocv;
    cv::fitLine(in, out_vec_ocv, distType, paramDefault, repsDefault, aepsDefault);
    // Comparison //////////////////////////////////////////////////////////////
    EXPECT_TRUE(cmpF(out_vec_gapi, out_vec_ocv));
}

template<typename In, int dim>
static void fitLineTestBody(const In& in, const cv::DistanceTypes distType,
                            const CompareVecs<float, dim>& cmpF, cv::GCompileArgs&& args)
{
    cv::Vec<float, dim> out_vec_gapi;
    fitLineTestGAPI(in, distType, std::move(args), out_vec_gapi);
    fitLineTestOpenCVCompare(in, distType, out_vec_gapi, cmpF);
}
} // namespace opencv_test

#endif // OPENCV_GAPI_IMGPROC_TESTS_COMMON_HPP
