// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include <opencv2/gapi/oak/oak_media_adapter.hpp>

#ifdef WITH_OAK_BACKEND

namespace cv {
namespace gapi {
namespace oak {

class OAKMediaAdapter::Priv final {
public:
    Priv() = default;
    void setParams(cv::Size sz, OAKFrameFormat fmt, const unsigned char* data_ptr);

    MediaFrame::View access(MediaFrame::Access access);
    cv::GFrameDesc meta() const;

    ~Priv() = default;

private:
    cv::Size m_sz;
    OAKFrameFormat m_fmt;
    const unsigned char* m_data_ptr;
};

void OAKMediaAdapter::Priv::setParams(cv::Size sz, OAKFrameFormat fmt, const unsigned char* data_ptr){
    m_sz = sz;
    m_fmt = fmt;
    m_data_ptr = data_ptr;
}
MediaFrame::View OAKMediaAdapter::Priv::access(MediaFrame::Access) {
    GAPI_Assert(m_fmt == OAKFrameFormat::BGR && "OAKMediaAdapter only supports BGR format for now");
    return MediaFrame::View{cv::MediaFrame::View::Ptrs{const_cast<unsigned char*>(m_data_ptr)},
                            cv::MediaFrame::View::Strides{}};
}
cv::GFrameDesc OAKMediaAdapter::Priv::meta() const { return {}; }

void OAKMediaAdapter::setParams(cv::Size sz, OAKFrameFormat fmt, const unsigned char* data_ptr) {
    m_priv->setParams(sz, fmt, data_ptr);
}
MediaFrame::View OAKMediaAdapter::access(MediaFrame::Access access) {
    return m_priv->access(access);
}
cv::GFrameDesc OAKMediaAdapter::meta() const { return m_priv->meta(); }

OAKMediaAdapter::OAKMediaAdapter() : m_priv(new OAKMediaAdapter::Priv()) {};
OAKMediaAdapter::~OAKMediaAdapter() = default;

} // namespace oak
} // namespace gapi
} // namespace cv

#else

namespace cv {
namespace gapi {
namespace oak {
OAKMediaAdapter::OAKMediaAdapter() {
    GAPI_Assert(false && "Built without depthai library support");
}
void OAKMediaAdapter::setParams(cv::Size sz, OAKFrameFormat fmt, const unsigned char* data_ptr) {
    GAPI_Assert(false && "Built without depthai library support");
}
cv::GFrameDesc OAKMediaAdapter::meta() const {
    GAPI_Assert(false && "Built without depthai library support");
}
cv::MediaFrame::View OAKMediaAdapter::access(cv::MediaFrame::Access) {
    GAPI_Assert(false && "Built without depthai library support");
}
OAKMediaAdapter::~OAKMediaAdapter() {
    GAPI_Assert(false && "Built without depthai library support");
}
} // namespace oak
} // namespace gapi
} // namespace cv

#endif // WITH_OAK_BACKEND
