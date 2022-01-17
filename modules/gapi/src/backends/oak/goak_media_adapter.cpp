// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include "oak_media_adapter.hpp"

namespace cv {
namespace gapi {
namespace oak {

OAKMediaAdapter::OAKMediaAdapter(cv::Size sz, cv::MediaFormat fmt, std::vector<uint8_t>&& buffer) {
    GAPI_Assert(fmt == cv::MediaFormat::NV12 && "OAKMediaAdapter only supports NV12 format for now");
    m_sz = sz;
    m_fmt = fmt;
    m_buffer = buffer;
}

MediaFrame::View OAKMediaAdapter::OAKMediaAdapter::access(MediaFrame::Access) {
    uint8_t* y_ptr = m_buffer.data();
    uint8_t* uv_ptr = m_buffer.data() + static_cast<long>(m_buffer.size() / 3 * 2);
    return MediaFrame::View{cv::MediaFrame::View::Ptrs{y_ptr, uv_ptr},
                            cv::MediaFrame::View::Strides{static_cast<long unsigned int>(m_sz.width),
                                                          static_cast<long unsigned int>(m_sz.width)}};
}

cv::GFrameDesc OAKMediaAdapter::OAKMediaAdapter::meta() const { return {m_fmt, m_sz}; }

} // namespace oak
} // namespace gapi
} // namespace cv
