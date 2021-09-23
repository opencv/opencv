// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifdef WITH_OAK_BACKEND
#include <opencv2/gapi/oak/oak_media_adapter.hpp>

namespace cv {
namespace gapi {
namespace oak {

class OAKMediaBGR::Priv final {
public:
    Priv() = default;
    void setParams(cv::Size sz, OAKFrameFormat fmt, unsigned char* data_ptr);

    MediaFrame::View access(MediaFrame::Access access);
    cv::GFrameDesc meta() const;

private:
    cv::Size m_sz;
    OAKFrameFormat m_fmt;
    unsigned char* m_data_ptr;
};

void OAKMediaBGR::Priv::setParams(cv::Size sz, OAKFrameFormat fmt, unsigned char* data_ptr){
    m_sz = sz;
    m_fmt = fmt;
    m_data_ptr = data_ptr;
}
MediaFrame::View OAKMediaBGR::Priv::access(MediaFrame::Access) {
    return MediaFrame::View{{m_data_ptr}, {}};
}
cv::GFrameDesc OAKMediaBGR::Priv::meta() const { return {}; }

void OAKMediaBGR::setParams(cv::Size sz, OAKFrameFormat fmt, unsigned char* data_ptr) {
    m_priv->setParams(sz, fmt, data_ptr);
}
MediaFrame::View OAKMediaBGR::access(MediaFrame::Access access) {
    return m_priv->access(access);
}
cv::GFrameDesc OAKMediaBGR::meta() const { return m_priv->meta(); }

} // namespace oak
} // namespace gapi
} // namespace cv

#else

// fixme: add proper impls with asserts inside
#error 42

#endif // WITH_OAK_BACKEND
