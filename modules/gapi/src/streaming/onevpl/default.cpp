// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2022 Intel Corporation

#include <iostream>
#include <memory>

#include <opencv2/gapi/streaming/onevpl/default.hpp>
#include <opencv2/gapi/streaming/onevpl/cfg_params.hpp>
#include <opencv2/gapi/streaming/onevpl/device_selector_interface.hpp>

#include "cfg_param_device_selector.hpp"

#ifdef HAVE_ONEVPL

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

std::shared_ptr<IDeviceSelector> getDefaultDeviceSelector(const std::vector<CfgParam>& cfg_params) {
    std::shared_ptr<CfgParamDeviceSelector> default_accel_contex(new CfgParamDeviceSelector(cfg_params));

    return default_accel_contex;
}

} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#else // HAVE_ONEVPL

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

std::shared_ptr<IDeviceSelector> getDefaultDeviceSelector(const std::vector<CfgParam>&) {
    std::cerr << "Cannot utilize getDefaultVPLDeviceAndCtx without HAVE_ONEVPL enabled" << std::endl;
    util::throw_error(std::logic_error("Cannot utilize getDefaultVPLDeviceAndCtx without HAVE_ONEVPL enabled"));
}

} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_ONEVPL
