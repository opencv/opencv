// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include <opencv2/gapi/streaming/onevpl/source.hpp>

#include "streaming/onevpl/source_priv.hpp"
#include "streaming/onevpl/file_data_provider.hpp"
namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

#ifdef HAVE_ONEVPL
GSource::GSource(const std::string& filePath, const CfgParams& cfg_params) :
    GSource(std::unique_ptr<Priv>(new GSource::Priv(std::make_shared<FileDataProvider>(filePath),
                                                    cfg_params))) {

    if (filePath.empty()) {
        util::throw_error(std::logic_error("Cannot create 'GSource' on empty source file name"));
    }
}

GSource::GSource(std::shared_ptr<IDataProvider> source, const CfgParams& cfg_params) :
     GSource(std::unique_ptr<Priv>(new GSource::Priv(source, cfg_params))) {
}
#else
GSource::GSource(const std::string&, const CfgParams&) {
    GAPI_Assert(false && "Unsupported: G-API compiled without `WITH_GAPI_ONEVPL=ON`");
}

GSource::GSource(std::shared_ptr<IDataProvider>, const CfgParams&) {
    GAPI_Assert(false && "Unsupported: G-API compiled without `WITH_GAPI_ONEVPL=ON`");
}
#endif

GSource::GSource(std::unique_ptr<Priv>&& impl) :
    IStreamSource(),
    m_priv(std::move(impl)) {
}

GSource::~GSource() = default;

bool GSource::pull(cv::gapi::wip::Data& data)
{
    return m_priv->pull(data);
}

GMetaArg GSource::descr_of() const
{
    return m_priv->descr_of();
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
