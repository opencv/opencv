//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you ("License"). Unless the License provides otherwise,
// you may not use, modify, copy, publish, distribute, disclose or transmit
// this software or the related documents without Intel's prior written
// permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#ifndef OPENCV_GAPI_STREAMING_STREAMING_HPP
#define OPENCV_GAPI_STREAMING_STREAMING_HPP

#include <opencv2/gapi/gkernel.hpp> // GKernelPackage

namespace cv {
namespace gapi {
namespace streaming {

GAPI_EXPORTS cv::gapi::GBackend backend();

G_API_OP(GCopy, <GFrame(GFrame)>, "com.intel.streaming.copy")
{
    static GFrameDesc outMeta(const GFrameDesc& in) { return in; }
};

G_API_OP(GBGR, <GMat(GFrame)>, "com.intel.streaming.BGR")
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

}
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_COPY_HPP
