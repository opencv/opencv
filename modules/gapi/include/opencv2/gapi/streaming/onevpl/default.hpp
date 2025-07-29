// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2022 Intel Corporation

#ifndef OPENCV_GAPI_STREAMING_ONEVPL_UTILS_HPP
#define OPENCV_GAPI_STREAMING_ONEVPL_UTILS_HPP

#include <opencv2/gapi/own/exports.hpp> // GAPI_EXPORTS
#include <opencv2/gapi/streaming/onevpl/cfg_params.hpp>
#include <opencv2/gapi/streaming/onevpl/device_selector_interface.hpp>

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

/**
 * @brief Provides default device selector based on config.
 */
GAPI_EXPORTS std::shared_ptr<IDeviceSelector> getDefaultDeviceSelector(const std::vector<CfgParam>& cfg_params);

} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_STREAMING_ONEVPL_UTILS_HPP
