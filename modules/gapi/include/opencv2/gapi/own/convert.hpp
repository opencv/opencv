// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_OWN_CONVERT_HPP
#define OPENCV_GAPI_OWN_CONVERT_HPP

#if !defined(GAPI_STANDALONE)

#include <opencv2/gapi/opencv_includes.hpp>
#include <opencv2/gapi/own/types.hpp>
#include <opencv2/gapi/own/mat.hpp>
#include <opencv2/gapi/own/scalar.hpp>

namespace cv
{
           cv::gapi::own::Mat to_own(Mat&&) = delete;
    inline cv::gapi::own::Mat to_own(Mat const& mat)
    {
        cv::gapi::own::Mat own(mat.rows, mat.cols, mat.type(), mat.data, mat.step);
        own.datastart = mat.datastart;
        own.dataend = mat.dataend;
        own.datalimit = mat.datalimit;
        return own;
    }

    inline cv::gapi::own::Scalar to_own(const cv::Scalar& s) { return {s[0], s[1], s[2], s[3]}; }

    inline cv::gapi::own::Size to_own (const Size& s) { return {s.width, s.height}; }

    inline cv::gapi::own::Rect to_own (const Rect& r) { return {r.x, r.y, r.width, r.height}; }



namespace gapi
{
namespace own
{
           cv::Mat to_ocv(Mat&&)    = delete;
    inline cv::Mat to_ocv(Mat const& own)
    {
        cv::Mat mat(own.rows, own.cols, own.type(), own.data, own.step);
        mat.datastart = own.datastart;
        mat.dataend = own.dataend;
        mat.datalimit = own.datalimit;
        return mat;
    }

    inline cv::Scalar to_ocv(const Scalar& s) { return {s[0], s[1], s[2], s[3]}; }

    inline cv::Size to_ocv (const Size& s) { return cv::Size(s.width, s.height); }

    inline cv::Rect to_ocv (const Rect& r) { return cv::Rect(r.x, r.y, r.width, r.height); }

} // namespace own
} // namespace gapi
} // namespace cv

#endif // !defined(GAPI_STANDALONE)

#endif // OPENCV_GAPI_OWN_CONVERT_HPP
