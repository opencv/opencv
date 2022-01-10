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

// FIXME: introduce a proper adapter
class GAPI_EXPORTS OAKMediaAdapter final : public cv::MediaFrame::IAdapter {
public:
    OAKMediaAdapter();
    OAKMediaAdapter(cv::Size sz, cv::MediaFormat fmt, std::vector<uint8_t>&& buffer);
    cv::GFrameDesc meta() const override;
    cv::MediaFrame::View access(cv::MediaFrame::Access) override;
    ~OAKMediaAdapter();
private:
    class Priv;
    std::unique_ptr<Priv> m_priv;
};

} // namespace oak
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_OAK_MEDIA_ADAPTER_HPP
