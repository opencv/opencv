// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include "oak_media_adapter.hpp"

namespace cv {
namespace gapi {
namespace oak {

class OAKMediaAdapter::Priv final {
public:
    Priv() = default;
    Priv(cv::Size sz, OAKFrameFormat fmt, uint8_t* data_ptr);

    MediaFrame::View access(MediaFrame::Access access);
    cv::GFrameDesc meta() const;

    ~Priv() = default;

private:
    cv::Size m_sz;
    OAKFrameFormat m_fmt;
    uint8_t* m_data_ptr;
};

OAKMediaAdapter::Priv::Priv(cv::Size sz, OAKFrameFormat fmt, uint8_t* data_ptr) {
    GAPI_Assert(fmt == OAKFrameFormat::BGR && "OAKMediaAdapter only supports BGR format for now");
    m_sz = sz;
    m_fmt = fmt;
    m_data_ptr = data_ptr;
}

// FIXME: handle strides
MediaFrame::View OAKMediaAdapter::Priv::access(MediaFrame::Access) {
    return MediaFrame::View{cv::MediaFrame::View::Ptrs{m_data_ptr},
                            cv::MediaFrame::View::Strides{}};
}

cv::GFrameDesc OAKMediaAdapter::Priv::meta() const { return {MediaFormat::BGR, m_sz}; }

OAKMediaAdapter::OAKMediaAdapter(cv::Size sz, OAKFrameFormat fmt, uint8_t* data_ptr) :
    m_priv(new OAKMediaAdapter::Priv(sz, fmt, data_ptr)) {};

MediaFrame::View OAKMediaAdapter::access(MediaFrame::Access access) {
    return m_priv->access(access);
}

cv::GFrameDesc OAKMediaAdapter::meta() const { return m_priv->meta(); }

OAKMediaAdapter::OAKMediaAdapter() : m_priv(new OAKMediaAdapter::Priv()) {}
OAKMediaAdapter::~OAKMediaAdapter() = default;

} // namespace oak
} // namespace gapi
} // namespace cv
