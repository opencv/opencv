// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_GSTREAMING_FORMAT_HPP
#define OPENCV_GAPI_GSTREAMING_FORMAT_HPP

#include <opencv2/gapi/gkernel.hpp> // GKernelPackage

namespace cv {
namespace gapi {
namespace streaming {

GAPI_EXPORTS cv::gapi::GKernelPackage kernels();

G_API_OP(GBGR, <GMat(GFrame)>, "org.opencv.streaming.BGR")
{
    static GMatDesc outMeta(const GFrameDesc& in) { return GMatDesc{CV_8U, 3, in.size}; }
};

G_API_OP(GY, <GMat(GFrame)>, "org.opencv.streaming.Y") {
    static GMatDesc outMeta(const GFrameDesc& frameDesc) {
        return GMatDesc { CV_8U, 1, frameDesc.size , false };
    }
};

G_API_OP(GUV, <GMat(GFrame)>, "org.opencv.streaming.UV") {
    static GMatDesc outMeta(const GFrameDesc& frameDesc) {
        return GMatDesc { CV_8U, 2, cv::Size(frameDesc.size.width / 2, frameDesc.size.height / 2),
                          false };
    }
};

/** @brief Gets bgr plane from input frame

@note Function textual ID is "org.opencv.streaming.BGR"

@param in Input frame
@return Image in BGR format
*/
GAPI_EXPORTS cv::GMat BGR(const cv::GFrame& in);

/** @brief Extracts Y plane from media frame.

Output image is 8-bit 1-channel image of @ref CV_8UC1.

@note Function textual ID is "org.opencv.streaming.Y"

@param frame input media frame.
*/
GAPI_EXPORTS GMat Y(const cv::GFrame& frame);

/** @brief Extracts UV plane from media frame.

Output image is 8-bit 2-channel image of @ref CV_8UC2.

@note Function textual ID is "org.opencv.streaming.UV"

@param frame input media frame.
*/
GAPI_EXPORTS GMat UV(const cv::GFrame& frame);
} // namespace streaming

//! @addtogroup gapi_transform
//! @{
/** @brief Makes a copy of the input image. Note that this copy may be not real
(no actual data copied). Use this function to maintain graph contracts,
e.g when graph's input needs to be passed directly to output, like in Streaming mode.

@note Function textual ID is "org.opencv.streaming.copy"

@param in Input image
@return Copy of the input
*/
GAPI_EXPORTS GMat copy(const GMat& in);

/** @brief Makes a copy of the input frame. Note that this copy may be not real
(no actual data copied). Use this function to maintain graph contracts,
e.g when graph's input needs to be passed directly to output, like in Streaming mode.

@note Function textual ID is "org.opencv.streaming.copy"

@param in Input frame
@return Copy of the input
*/
GAPI_EXPORTS GFrame copy(const GFrame& in);
//! @} gapi_transform

} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_GSTREAMING_FORMAT_HPP
