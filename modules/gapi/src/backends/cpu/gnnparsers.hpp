// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include <opencv2/gapi/infer/parsers.hpp>

#ifndef OPENCV_NNPARSERS_OCV_HPP
#define OPENCV_NNPARSERS_OCV_HPP

namespace cv
{
void parseSSDBL(const cv::Mat&  in_ssd_result,
                const cv::Size& in_size,
                const float     confidence_threshold,
                const int       filter_label,
                std::vector<cv::Rect>& out_boxes,
                std::vector<int>&      out_labels);

void parseSSD(const cv::Mat&  in_ssd_result,
              const cv::Size& in_size,
              const float     confidence_threshold,
              const bool      alignment_to_square,
              const bool      filter_out_of_bounds,
              std::vector<cv::Rect>& out_boxes);

void parseYolo(const cv::Mat&  in_yolo_result,
               const cv::Size& in_size,
               const float     confidence_threshold,
               const float     nms_threshold,
               const std::vector<float>& anchors,
               std::vector<cv::Rect>& out_boxes,
               std::vector<int>&      out_labels);
}
#endif // OPENCV_NNPARSERS_OCV_HPP
