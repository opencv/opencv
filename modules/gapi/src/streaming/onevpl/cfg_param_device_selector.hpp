// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_CFG_PARAM_DEVICE_SELECTOR_HPP
#define GAPI_STREAMING_ONEVPL_CFG_PARAM_DEVICE_SELECTOR_HPP

#include <opencv2/gapi/streaming/onevpl/device_selector_interface.hpp>
#include <opencv2/gapi/streaming/onevpl/cfg_params.hpp>
#include <opencv2/gapi/streaming/onevpl/source.hpp>

#include "opencv2/gapi/own/exports.hpp" // GAPI_EXPORTS

#ifdef HAVE_ONEVPL

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

struct PlatformSpecificParams;
std::vector<CfgParam> update_param_with_accel_type(std::vector<CfgParam> &&param_array, AccelType type);

struct GAPI_EXPORTS CfgParamDeviceSelector final: public IDeviceSelector {
    CfgParamDeviceSelector(const CfgParams& params = {});
    CfgParamDeviceSelector(Device::Ptr device_ptr,
                           const std::string& device_id,
                           Context::Ptr ctx_ptr,
                           const CfgParams& params);
    CfgParamDeviceSelector(const Device &device_ptr,
                           const Context &ctx_ptr,
                           CfgParams params);
    ~CfgParamDeviceSelector();

    DeviceScoreTable select_devices() const override;
    DeviceContexts select_context() override;

private:
    Device suggested_device;
    Context suggested_context;
    std::unique_ptr<PlatformSpecificParams> platform_specific_data;
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#endif //HAVE_ONEVPL
#endif // GAPI_STREAMING_ONEVPL_CFG_PARAM_DEVICE_SELECTOR_HPP
