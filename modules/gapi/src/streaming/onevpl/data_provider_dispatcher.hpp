// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_DATA_PROVIDER_DISPATCHER_HPP
#define GAPI_STREAMING_ONEVPL_DATA_PROVIDER_DISPATCHER_HPP

#ifdef HAVE_ONEVPL
#include <opencv2/gapi/streaming/onevpl/data_provider_interface.hpp>
#include <opencv2/gapi/streaming/onevpl/cfg_params.hpp>

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

struct GAPI_EXPORTS DataProviderDispatcher {

    static IDataProvider::Ptr create(const std::string& file_path,
                                     const std::vector<CfgParam> &codec_params = {});
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
#endif // GAPI_STREAMING_ONEVPL_DATA_PROVIDER_DISPATCHER_HPP
