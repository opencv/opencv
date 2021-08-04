// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include <opencv2/gapi/streaming/onevpl_cap.hpp>

#include "streaming/onevpl_priv_interface.hpp"

namespace cv {
namespace gapi {
namespace wip {

OneVPLCapture::OneVPLCapture(std::unique_ptr<IPriv>&& impl) : IStreamSource(),
    m_priv(std::move(impl))
{
}

OneVPLCapture::~OneVPLCapture()
{
}

bool OneVPLCapture::pull(cv::gapi::wip::Data& data)
{
    return m_priv->pull(data);
}

GMetaArg OneVPLCapture::descr_of() const
{
    return m_priv->descr_of();
}

} // namespace wip
} // namespace gapi
} // namespace cv
