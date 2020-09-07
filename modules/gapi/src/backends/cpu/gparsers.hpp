// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include <opencv2/gapi/core.hpp>

#ifndef OPENCV_PARSERS_OCV_HPP
#define OPENCV_PARSERS_OCV_HPP

namespace cv
{
GAPI_EXPORTS void parseSSDWL(const cv::Mat&  in_ssd_result,
                             const cv::Size& in_size,
                             const float&    confidence_threshold,
                             const int&      filter_label,
                             std::vector<cv::Rect>& out_boxes,
                             std::vector<int>&      out_labels);

GAPI_EXPORTS void parseSSD(const cv::Mat&  in_ssd_result,
                           const cv::Size& in_size,
                           const float&    confidence_threshold,
                           const bool&     filter_out_of_bounds,
                           std::vector<cv::Rect>& out_boxes);

GAPI_EXPORTS void parseYolo(const cv::Mat&  in_yolo_result,
                            const cv::Size& in_size,
                            const float&    confidence_threshold,
                            const float&    nms_threshold,
                            const cv::gapi::core::YoloAnchors& anchors,
                            std::vector<cv::Rect>& out_boxes,
                            std::vector<int>&      out_labels);
}
#endif // OPENCV_PARSERS_OCV_HPP
