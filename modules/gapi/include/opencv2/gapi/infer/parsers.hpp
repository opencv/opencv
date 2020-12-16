// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#ifndef OPENCV_GAPI_PARSERS_HPP
#define OPENCV_GAPI_PARSERS_HPP

#include <utility> // std::tuple

#include <opencv2/gapi/gmat.hpp>
#include <opencv2/gapi/gkernel.hpp>

namespace cv { namespace gapi {
namespace nn {
namespace parsers {
    using GRects      = GArray<Rect>;
    using GDetections = std::tuple<GArray<Rect>, GArray<int>>;

    G_TYPED_KERNEL(GParseSSDBL, <GDetections(GMat, GOpaque<Size>, float, int)>,
                   "org.opencv.nn.parsers.parseSSD_BL") {
        static std::tuple<GArrayDesc,GArrayDesc> outMeta(const GMatDesc&, const GOpaqueDesc&, float, int) {
            return std::make_tuple(empty_array_desc(), empty_array_desc());
        }
    };

    G_TYPED_KERNEL(GParseSSD, <GRects(GMat, GOpaque<Size>, float, bool, bool)>,
                   "org.opencv.nn.parsers.parseSSD") {
        static GArrayDesc outMeta(const GMatDesc&, const GOpaqueDesc&, float, bool, bool) {
            return empty_array_desc();
        }
    };

    G_TYPED_KERNEL(GParseYolo, <GDetections(GMat, GOpaque<Size>, float, float, std::vector<float>)>,
                   "org.opencv.nn.parsers.parseYolo") {
        static std::tuple<GArrayDesc, GArrayDesc> outMeta(const GMatDesc&, const GOpaqueDesc&,
                                                          float, float, const std::vector<float>&) {
            return std::make_tuple(empty_array_desc(), empty_array_desc());
        }
        static const std::vector<float>& defaultAnchors() {
            static std::vector<float> anchors {
                0.57273f, 0.677385f, 1.87446f, 2.06253f, 3.33843f, 5.47434f, 7.88282f, 3.52778f, 9.77052f, 9.16828f
            };
            return anchors;
        }
    };
} // namespace parsers
} // namespace nn

/** @brief Parses output of SSD network.

Extracts detection information (box, confidence, label) from SSD output and
filters it by given confidence and label.

@note Function textual ID is "org.opencv.nn.parsers.parseSSD_BL"

@param in Input CV_32F tensor with {1,1,N,7} dimensions.
@param inSz Size to project detected boxes to (size of the input image).
@param confidenceThreshold If confidence of the
detection is smaller than confidence threshold, detection is rejected.
@param filterLabel If provided (!= -1), only detections with
given label will get to the output.
@return a tuple with a vector of detected boxes and a vector of appropriate labels.
*/
GAPI_EXPORTS std::tuple<GArray<Rect>, GArray<int>> parseSSD(const GMat& in,
                                                            const GOpaque<Size>& inSz,
                                                            const float confidenceThreshold = 0.5f,
                                                            const int   filterLabel = -1);

/** @overload
Extracts detection information (box, confidence) from SSD output and
filters it by given confidence and by going out of bounds.

@note Function textual ID is "org.opencv.nn.parsers.parseSSD"

@param in Input CV_32F tensor with {1,1,N,7} dimensions.
@param inSz Size to project detected boxes to (size of the input image).
@param confidenceThreshold If confidence of the
detection is smaller than confidence threshold, detection is rejected.
@param alignmentToSquare If provided true, bounding boxes are extended to squares.
The center of the rectangle remains unchanged, the side of the square is
the larger side of the rectangle.
@param filterOutOfBounds If provided true, out-of-frame boxes are filtered.
@return a vector of detected bounding boxes.
*/
GAPI_EXPORTS GArray<Rect> parseSSD(const GMat& in,
                                   const GOpaque<Size>& inSz,
                                   const float confidenceThreshold = 0.5f,
                                   const bool alignmentToSquare = false,
                                   const bool filterOutOfBounds = false);

/** @brief Parses output of Yolo network.

Extracts detection information (box, confidence, label) from Yolo output,
filters it by given confidence and performs non-maximum supression for overlapping boxes.

@note Function textual ID is "org.opencv.nn.parsers.parseYolo"

@param in Input CV_32F tensor with {1,13,13,N} dimensions, N should satisfy:
\f[\texttt{N} = (\texttt{num_classes} + \texttt{5}) * \texttt{5},\f]
where num_classes - a number of classes Yolo network was trained with.
@param inSz Size to project detected boxes to (size of the input image).
@param confidenceThreshold If confidence of the
detection is smaller than confidence threshold, detection is rejected.
@param nmsThreshold Non-maximum supression threshold which controls minimum
relative box intersection area required for rejecting the box with a smaller confidence.
If 1.f, nms is not performed and no boxes are rejected.
@param anchors Anchors Yolo network was trained with.
@note The default anchor values are specified for YOLO v2 Tiny as described in Intel Open Model Zoo
<a href="https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/yolo-v2-tiny-tf/yolo-v2-tiny-tf.md">documentation</a>.
@return a tuple with a vector of detected boxes and a vector of appropriate labels.
*/
GAPI_EXPORTS std::tuple<GArray<Rect>, GArray<int>> parseYolo(const GMat& in,
                                                             const GOpaque<Size>& inSz,
                                                             const float confidenceThreshold = 0.5f,
                                                             const float nmsThreshold = 0.5f,
                                                             const std::vector<float>& anchors
                                                                 = nn::parsers::GParseYolo::defaultAnchors());

} // namespace gapi
} // namespace cv

// Reimport parseSSD & parseYolo under their initial namespace
namespace cv {
namespace gapi {
namespace streaming {

using cv::gapi::parseSSD;
using cv::gapi::parseYolo;

} // namespace streaming
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_PARSERS_HPP
