// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2022 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_ENGINE_PREPROC_DEFINES_HPP
#define GAPI_STREAMING_ONEVPL_ENGINE_PREPROC_DEFINES_HPP

#ifdef HAVE_ONEVPL
#include "streaming/onevpl/engine/preproc/vpp_preproc_defines.hpp"
#endif // HAVE_ONEVPL


namespace cv {
namespace gapi {
namespace wip {

#ifdef VPP_PREPROC_ENGINE
#define GAPI_BACKEND_PP_PARAMS      cv::gapi::wip::onevpl::vpp_pp_params
#else
struct empty_pp_params {};
#define GAPI_BACKEND_PP_PARAMS      cv::gapi::wip::empty_pp_params;
#endif

struct pp_params {
    using value_type = cv::util::variant<GAPI_BACKEND_PP_PARAMS>;
    value_type value;
};
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // GAPI_STREAMING_ONEVPL_ENGINE_PREPROC_DEFINES_HPP
