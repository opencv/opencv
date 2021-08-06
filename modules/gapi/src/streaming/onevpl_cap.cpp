// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include <opencv2/gapi/streaming/onevpl_cap.hpp>

#include "streaming/onevpl_priv_interface.hpp"
#include "streaming/vpl/vpl_source_impl.hpp"

namespace cv {
namespace gapi {
namespace wip {
    
#ifdef HAVE_ONEVPL
OneVPLCapture::OneVPLCapture(const std::string& filePath, const onevpl_params_container_t& cfg_params) :
    OneVPLCapture(std::unique_ptr<IPriv>(new VPLSourceImpl(filePath, cfg_params))) {

    if (filePath.empty()) {
        util::throw_error(std::logic_error("Cannot create 'OneVPLCapture' on empty source file name"));
    }
}
#else
OneVPLCapture::OneVPLCapture(const std::string& filePath, const onevpl_params_container_t& cfg_params) {
    GAPI_Assert(false && "Unsupported: G-API compiled without `WITH_ONEVPL=ON`")
}

#endif
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
