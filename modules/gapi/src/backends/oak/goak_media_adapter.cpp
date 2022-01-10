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
    Priv(cv::Size sz, cv::MediaFormat fmt, std::vector<uint8_t>&& buffer);

    MediaFrame::View access(MediaFrame::Access access);
    cv::GFrameDesc meta() const;

    ~Priv() = default;

private:
    cv::Size m_sz;
    cv::MediaFormat m_fmt;
    std::vector<uint8_t> m_buffer;
};

OAKMediaAdapter::Priv::Priv(cv::Size sz, cv::MediaFormat fmt, std::vector<uint8_t>&& buffer) {
    GAPI_Assert(fmt == cv::MediaFormat::NV12 && "OAKMediaAdapter only supports NV12 format for now");
    m_sz = sz;
    m_fmt = fmt;
    m_buffer = buffer;
}

MediaFrame::View OAKMediaAdapter::Priv::access(MediaFrame::Access) {
    uint8_t* y_ptr = m_buffer.data();
    uint8_t* uv_ptr = m_buffer.data() + static_cast<long>(m_buffer.size() / 3 * 2);
    return MediaFrame::View{cv::MediaFrame::View::Ptrs{y_ptr, uv_ptr},
                            cv::MediaFrame::View::Strides{static_cast<long unsigned int>(m_sz.width),
                                                          static_cast<long unsigned int>(m_sz.height)}};
}

cv::GFrameDesc OAKMediaAdapter::Priv::meta() const { return {m_fmt, m_sz}; }

OAKMediaAdapter::OAKMediaAdapter(cv::Size sz, cv::MediaFormat fmt, std::vector<uint8_t>&& buffer) :
    m_priv(new OAKMediaAdapter::Priv(sz, fmt, std::move(buffer))) {};

MediaFrame::View OAKMediaAdapter::access(MediaFrame::Access access) {
    return m_priv->access(access);
}

cv::GFrameDesc OAKMediaAdapter::meta() const { return m_priv->meta(); }

OAKMediaAdapter::OAKMediaAdapter() : m_priv(new OAKMediaAdapter::Priv()) {}
OAKMediaAdapter::~OAKMediaAdapter() = default;

} // namespace oak
} // namespace gapi
} // namespace cv
