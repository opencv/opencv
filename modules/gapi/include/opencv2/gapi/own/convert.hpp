// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_OWN_CONVERT_HPP
#define OPENCV_GAPI_OWN_CONVERT_HPP

#if !defined(GAPI_STANDALONE)

#include <opencv2/gapi/opencv_includes.hpp>
#include <opencv2/gapi/own/mat.hpp>

namespace cv
{
    template<typename T>
    std::vector<T> to_own(const cv::MatSize &sz) {
        std::vector<T> result(sz.dims());
        for (int i = 0; i < sz.dims(); i++) {
            // Note: cv::MatSize is not iterable
            result[i] = static_cast<T>(sz[i]);
        }
        return result;
    }

    cv::gapi::own::Mat to_own(Mat&&) = delete;

    inline cv::gapi::own::Mat to_own(Mat const& m) {
        return (m.dims == 2)
            ?  cv::gapi::own::Mat{m.rows, m.cols, m.type(), m.data, m.step}
            :  cv::gapi::own::Mat{to_own<int>(m.size), m.type(), m.data};
    };

namespace gapi
{
namespace own
{

    inline cv::Mat to_ocv(Mat const& m) {
        return m.dims.empty()
            ? cv::Mat{m.rows, m.cols, m.type(), m.data, m.step}
            : cv::Mat{m.dims, m.type(), m.data};
    }
    inline cv::Mat to_ocv(cv::Mat const&) = delete;

    inline cv::Rect to_ocv(cv::gapi::own::Rect const& r) {
        return cv::Rect{r.x, r.y, r.width, r.height};
    }
    inline cv::Rect to_ocv(cv::Rect const&) = delete;

    inline cv::Size to_ocv(cv::gapi::own::Size const& s) {
        return cv::Size{s.width, s.height};
    }
    inline cv::Size to_ocv(cv::Size const&) = delete;

    inline cv::Point to_ocv(cv::gapi::own::Point const& p) {
        return cv::Point{p.x, p.y};
    }
    inline cv::Point to_ocv(cv::Point const&) = delete;

    inline cv::Point2f to_ocv(cv::gapi::own::Point2f const& p2f) {
        return cv::Point2f{p2f.x, p2f.y};
    }
    inline cv::Point2f to_ocv(cv::Point2f const&) = delete;

    inline cv::Scalar to_ocv(cv::gapi::own::Scalar const& s) {
        return cv::Scalar{s[0], s[1], s[2], s[3]};
    }
    inline cv::Scalar to_ocv(cv::Scalar const&) = delete;
} // namespace own
} // namespace gapi
} // namespace cv

#endif // !defined(GAPI_STANDALONE)

#endif // OPENCV_GAPI_OWN_CONVERT_HPP
