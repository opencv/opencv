// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include <opencv2/gapi/streaming/onevpl/source.hpp>

#include "streaming/onevpl/source_priv.hpp"
#include "streaming/onevpl/data_provider_dispatcher.hpp"
#include "streaming/onevpl/cfg_param_device_selector.hpp"

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

#ifdef HAVE_ONEVPL
GSource::GSource(const std::string& filePath, const CfgParams& cfg_params) :
    GSource(filePath, cfg_params, std::make_shared<CfgParamDeviceSelector>(cfg_params)) {
    if (filePath.empty()) {
        util::throw_error(std::logic_error("Cannot create 'GSource' on empty source file name"));
    }
}

GSource::GSource(const std::string& filePath,
                 const CfgParams& cfg_params,
                 const std::string& device_id,
                 void* accel_device_ptr,
                 void* accel_ctx_ptr) :
    GSource(filePath, cfg_params,
            std::make_shared<CfgParamDeviceSelector>(accel_device_ptr, device_id,
                                                     accel_ctx_ptr, cfg_params)) {
}

GSource::GSource(const std::string& filePath,
                 const CfgParams& cfg_params,
                 const Device &device, const Context &ctx) :
    GSource(filePath, cfg_params,
            std::make_shared<CfgParamDeviceSelector>(device, ctx, cfg_params)) {
}

GSource::GSource(const std::string& filePath,
                 const CfgParams& cfg_params,
                 std::shared_ptr<IDeviceSelector> selector) :
    GSource(DataProviderDispatcher::create(filePath, cfg_params), cfg_params, selector) {
    if (filePath.empty()) {
        util::throw_error(std::logic_error("Cannot create 'GSource' on empty source file name"));
    }
}

GSource::GSource(std::shared_ptr<IDataProvider> source, const CfgParams& cfg_params) :
    GSource(source, cfg_params,
            std::make_shared<CfgParamDeviceSelector>(cfg_params)) {
}

GSource::GSource(std::shared_ptr<IDataProvider> source,
                 const CfgParams& cfg_params,
                 const std::string& device_id,
                 void* accel_device_ptr,
                 void* accel_ctx_ptr) :
    GSource(source, cfg_params,
            std::make_shared<CfgParamDeviceSelector>(accel_device_ptr, device_id,
                                                     accel_ctx_ptr, cfg_params)) {
}

// common delegating parameters c-tor
GSource::GSource(std::shared_ptr<IDataProvider> source,
                 const CfgParams& cfg_params,
                 std::shared_ptr<IDeviceSelector> selector) :
    GSource(std::unique_ptr<Priv>(new GSource::Priv(source, cfg_params, selector))) {
}

#else
GSource::GSource(const std::string&, const CfgParams&) {
    GAPI_Assert(false && "Unsupported: G-API compiled without `WITH_GAPI_ONEVPL=ON`");
}

GSource::GSource(const std::string&, const CfgParams&, const std::string&,
                 void*, void*) {
    GAPI_Assert(false && "Unsupported: G-API compiled without `WITH_GAPI_ONEVPL=ON`");
}

GSource::GSource(const std::string&, const CfgParams&, const Device &, const Context &) {
    GAPI_Assert(false && "Unsupported: G-API compiled without `WITH_GAPI_ONEVPL=ON`");
}

GSource::GSource(const std::string&, const CfgParams&, std::shared_ptr<IDeviceSelector>) {
    GAPI_Assert(false && "Unsupported: G-API compiled without `WITH_GAPI_ONEVPL=ON`");
}

GSource::GSource(std::shared_ptr<IDataProvider>, const CfgParams&) {
    GAPI_Assert(false && "Unsupported: G-API compiled without `WITH_GAPI_ONEVPL=ON`");
}

GSource::GSource(std::shared_ptr<IDataProvider>, const CfgParams&,
                 const std::string&, void*, void*) {
    GAPI_Assert(false && "Unsupported: G-API compiled without `WITH_GAPI_ONEVPL=ON`");
}

GSource::GSource(std::shared_ptr<IDataProvider>, const CfgParams&, std::shared_ptr<IDeviceSelector>) {
    GAPI_Assert(false && "Unsupported: G-API compiled without `WITH_GAPI_ONEVPL=ON`");
}
#endif

// final delegating c-tor
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
