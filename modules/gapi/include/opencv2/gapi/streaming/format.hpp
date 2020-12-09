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

cv::gapi::GKernelPackage kernels();

// FIXME: Make a generic kernel
G_API_OP(GCopy, <GFrame(GFrame)>, "org.opencv.streaming.copy")
{
    static GFrameDesc outMeta(const GFrameDesc& in) { return in; }
};

G_API_OP(GBGR, <GMat(GFrame)>, "org.opencv.streaming.BGR")
{
    static GMatDesc outMeta(const GFrameDesc& in) { return GMatDesc{CV_8U, 3, in.size}; }
};

/** @brief Gets copy from the input frame

@note Function textual ID is "org.opencv.streaming.copy"

@param in Input frame
@return Copy of the input frame
*/
GAPI_EXPORTS cv::GFrame copy(const cv::GFrame& in);

/** @brief Gets bgr plane from input frame

@note Function textual ID is "org.opencv.streaming.BGR"

@param in Input frame
@return Image in BGR format
*/
GAPI_EXPORTS cv::GMat BGR (const cv::GFrame& in);

} // namespace streaming
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_GSTREAMING_FORMAT_HPP
