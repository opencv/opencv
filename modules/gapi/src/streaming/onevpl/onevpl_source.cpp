// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include <opencv2/gapi/streaming/onevpl/onevpl_source.hpp>

#include "streaming/onevpl/onevpl_source_priv.hpp"
#include "streaming/onevpl/onevpl_file_data_provider.hpp"
namespace cv {
namespace gapi {
namespace wip {

#ifdef HAVE_ONEVPL
OneVPLSource::OneVPLSource(const std::string& filePath, const onevpl_params_container_t& cfg_params) :
    OneVPLSource(std::unique_ptr<Priv>(new OneVPLSource::Priv(std::make_shared<FileDataProvider>(filePath),
                                                              cfg_params))) {

    if (filePath.empty()) {
        util::throw_error(std::logic_error("Cannot create 'OneVPLSource' on empty source file name"));
    }
}

OneVPLSource::OneVPLSource(std::shared_ptr<IDataProvider> source, const onevpl_params_container_t& cfg_params) :
     OneVPLSource(std::unique_ptr<Priv>(new OneVPLSource::Priv(source, cfg_params))) {
}
#else
OneVPLSource::OneVPLSource(const std::string&, const onevpl_params_container_t&) {
    GAPI_Assert(false && "Unsupported: G-API compiled without `WITH_GAPI_ONEVPL=ON`");
}

OneVPLSource::OneVPLSource(std::shared_ptr<IDataProvider>, const onevpl_params_container_t&) {
    GAPI_Assert(false && "Unsupported: G-API compiled without `WITH_GAPI_ONEVPL=ON`");
}
#endif

OneVPLSource::OneVPLSource(std::unique_ptr<Priv>&& impl) :
    IStreamSource(),
    m_priv(std::move(impl)) {
}

OneVPLSource::~OneVPLSource() = default;

bool OneVPLSource::pull(cv::gapi::wip::Data& data)
{
    return m_priv->pull(data);
}

GMetaArg OneVPLSource::descr_of() const
{
    return m_priv->descr_of();
}
} // namespace wip
} // namespace gapi
} // namespace cv
