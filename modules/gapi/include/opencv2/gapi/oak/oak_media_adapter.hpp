// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_OAK_MEDIA_ADAPTER_HPP
#define OPENCV_GAPI_OAK_MEDIA_ADAPTER_HPP

#include <memory>

#include <opencv2/gapi/media.hpp>

namespace cv {
namespace gapi {
namespace oak {

enum class OAKFrameFormat{
    BGR = 0,
};

class OAKMediaBGR final : public cv::MediaFrame::IAdapter {
    class Priv;
    std::unique_ptr<Priv> m_priv;
public:
    OAKMediaBGR() = default;
    // FIXME: Used by the internal backend only
    explicit OAKMediaBGR(cv::Size sz, OAKFrameFormat fmt, char* data_ptr);
    cv::GFrameDesc meta() const override;
    cv::MediaFrame::View access(cv::MediaFrame::Access);
};

} // namespace oak
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_OAK_MEDIA_ADAPTER_HPP
