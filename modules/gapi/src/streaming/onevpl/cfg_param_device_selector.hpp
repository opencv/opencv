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

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

struct GAPI_EXPORTS CfgParamDeviceSelector final: public IDeviceSelector {
    CfgParamDeviceSelector(const CfgParams& params = {});
    CfgParamDeviceSelector(Device::Ptr device_ptr,
                           Context::Ptr ctx_ptr,
                           const CfgParams& params);
    ~CfgParamDeviceSelector();

    DeviceScoreTable select_devices() const override;
    DeviceScoreTable select_spare_devices() const override;
    Context select_context(const DeviceScoreTable& selected_devices) override;
    Context get_last_context() const override;
private:
    Device suggested_device;
    Context suggested_context;
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // GAPI_STREAMING_ONEVPL_ONEVPL_FILE_DATA_PROVIDER_HPP
