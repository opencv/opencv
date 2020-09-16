// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#ifndef OPENCV_GAPI_NNPARSERS_HPP
#define OPENCV_GAPI_NNPARSERS_HPP

#include <utility> // std::tuple

#include <opencv2/gapi/gmat.hpp>
#include <opencv2/gapi/gkernel.hpp>

namespace cv { namespace gapi {
namespace nn {
    using GRects      = cv::GArray<cv::Rect>;
    using GDetections = std::tuple<GArray<Rect>, GArray<int>>;

    G_TYPED_KERNEL(GParseSSDBL, <GDetections(GMat, GOpaque<Size>, float, int)>, "org.opencv.core.parseSSD_BL") {
        static std::tuple<GArrayDesc,GArrayDesc> outMeta(const GMatDesc&, const GOpaqueDesc&, float, int) {
            return std::make_tuple(empty_array_desc(), empty_array_desc());
        }
    };

    G_TYPED_KERNEL(GParseSSD, <GRects(cv::GMat, cv::GOpaque<cv::Size>, float, bool, bool)>, "org.opencv.core.parseSSD") {
        static cv::GArrayDesc outMeta(const cv::GMatDesc&, const cv::GOpaqueDesc&, float, bool, bool) {
            return cv::empty_array_desc();
        }
    };

    G_TYPED_KERNEL(GParseYolo, <GDetections(GMat, GOpaque<Size>, float, float, std::vector<float>)>, "org.opencv.core.parseYolo") {
        static std::tuple<GArrayDesc, GArrayDesc> outMeta(const GMatDesc&, const GOpaqueDesc&, float, float, const std::vector<float>&) {
            return std::make_tuple(empty_array_desc(), empty_array_desc());
        }
        static const std::vector<float>& defaultAnchors() {
            static std::vector<float> anchors {
                0.57273f, 0.677385f, 1.87446f, 2.06253f, 3.33843f, 5.47434f, 7.88282f, 3.52778f, 9.77052f, 9.16828f
            };
            return anchors;
        }
    };
} // namespace nn

/** @brief Parses output of SSD network.
Extracts detection information (box, confidence, label) from SSD output and
filters it by given confidence and label.

@note Function textual ID is "org.opencv.core.parseSSD_BL"

@param in Input CV_32F tensor with {1,1,N,7} dimensions.
@param in_sz Size to project detected boxes to (size of the input image).
@param confidence_threshold If confidence of the
detection is smaller than confidence threshold, detection is rejected.
@param filter_label If provided (!= -1), only detections with
given label will get to the output.
@return a tuple with a vector of detected boxes and a vector of appropriate labels.
*/
GAPI_EXPORTS std::tuple<GArray<Rect>, GArray<int>> parseSSD(const GMat& in,
                                                            const GOpaque<Size>& in_sz,
                                                            const float confidence_threshold = 0.5f,
                                                            const int   filter_label = -1);

/** @overload
Extracts detection information (box, confidence) from SSD output and
filters it by given confidence and by going out of bounds.

@note Function textual ID is "org.opencv.core.parseSSD"

@param in Input CV_32F tensor with {1,1,N,7} dimensions.
@param in_sz Size to project detected boxes to (size of the input image).
@param confidence_threshold If confidence of the
detection is smaller than confidence threshold, detection is rejected.
@param alignment_to_square If provided true, bounding boxes are converted to squares.
@param filter_out_of_bounds If provided true, out-of-bounds boxes are filtered.
@return a vector of detected bounding boxes.
*/
GAPI_EXPORTS cv::GArray<cv::Rect> parseSSD(const GMat& in,
                                           const GOpaque<Size>& in_sz,
                                           const float confidence_threshold = 0.5f,
                                           const bool alignment_to_square = false,
                                           const bool filter_out_of_bounds = false);

/** @brief Parses output of Yolo network.
Extracts detection information (box, confidence, label) from Yolo output,
filters it by given confidence and performs non-maximum supression for overlapping boxes.

@note Function textual ID is "org.opencv.core.parseYolo"

@param in Input CV_32F tensor with {1,13,13,N} dimensions, N should satisfy:
\f[\texttt{N} = (\texttt{num_classes} + \texttt{5}) * \texttt{5},\f]
where num_classes - a number of classes Yolo network was trained with.
@param in_sz Size to project detected boxes to (size of the input image).
@param confidence_threshold If confidence of the
detection is smaller than confidence threshold, detection is rejected.
@param nms_threshold Non-maximum supression threshold which controls minimum
relative box intersection area required for rejecting the box with a smaller confidence.
If 1.f, nms is not performed and no boxes are rejected.
@param anchors Anchors Yolo network was trained with.
@note The default anchor values are taken from openvinotoolkit docs:
https://docs.openvinotoolkit.org/latest/omz_models_intel_yolo_v2_tiny_vehicle_detection_0001_description_yolo_v2_tiny_vehicle_detection_0001.html#output.
@return a tuple with a vector of detected boxes and a vector of appropriate labels.
*/
GAPI_EXPORTS std::tuple<GArray<Rect>, GArray<int>> parseYolo(const GMat& in,
                                                             const GOpaque<Size>& in_sz,
                                                             const float confidence_threshold = 0.5f,
                                                             const float nms_threshold = 0.5f,
                                                             const std::vector<float>& anchors = nn::GParseYolo::defaultAnchors());

} // namespace gapi
} // namespace cv

#endif
