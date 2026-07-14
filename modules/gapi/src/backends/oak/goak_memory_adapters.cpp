// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include "oak_memory_adapters.hpp"

namespace cv {
namespace gapi {
namespace oak {

OAKMediaAdapter::OAKMediaAdapter(cv::Size sz, cv::MediaFormat fmt, std::vector<uint8_t>&& buffer)
: m_sz(sz), m_fmt(fmt), m_buffer(buffer) {
    GAPI_Assert(fmt == cv::MediaFormat::NV12 && "OAKMediaAdapter only supports NV12 format for now");
}

MediaFrame::View OAKMediaAdapter::OAKMediaAdapter::access(MediaFrame::Access) {
    uint8_t* y_ptr = m_buffer.data();
    uint8_t* uv_ptr = m_buffer.data() + static_cast<long>(m_buffer.size() / 3 * 2);
    return MediaFrame::View{cv::MediaFrame::View::Ptrs{y_ptr, uv_ptr},
                            cv::MediaFrame::View::Strides{static_cast<long unsigned int>(m_sz.width),
                                                          static_cast<long unsigned int>(m_sz.width)}};
}

cv::GFrameDesc OAKMediaAdapter::OAKMediaAdapter::meta() const { return {m_fmt, m_sz}; }

OAKRMatAdapter::OAKRMatAdapter(const cv::Size& size,
                               int precision,
                               std::vector<float>&& buffer)
    : m_size(size), m_precision(precision), m_buffer(buffer) {
    GAPI_Assert(m_precision == CV_16F);

    std::vector<int> wrapped_dims{1, 1, m_size.width, m_size.height};

    // FIXME: check layout and add strides
    m_desc = cv::GMatDesc(m_precision, wrapped_dims);
    m_mat = cv::Mat(static_cast<int>(wrapped_dims.size()),
                    wrapped_dims.data(),
                    CV_16FC1, // FIXME: cover other precisions
                    m_buffer.data());
}

cv::GMatDesc OAKRMatAdapter::desc() const {
    return m_desc;
}

cv::RMat::View OAKRMatAdapter::access(cv::RMat::Access) {
    return cv::RMat::View{m_desc, m_mat.data};
}

} // namespace oak
} // namespace gapi
} // namespace cv
