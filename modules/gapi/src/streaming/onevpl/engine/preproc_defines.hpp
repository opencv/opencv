// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2022 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_ENGINE_PREPROC_DEFINES_HPP
#define GAPI_STREAMING_ONEVPL_ENGINE_PREPROC_DEFINES_HPP

#ifdef HAVE_ONEVPL
#include "streaming/onevpl/utils.hpp"
#include "streaming/onevpl/engine/preproc/vpp_preproc_defines.hpp"
#endif // HAVE_ONEVPL


namespace cv {
namespace gapi {
namespace wip {

#ifdef VPP_PREPROC_ENGINE
#define GAPI_BACKEND_PP_PARAMS          cv::gapi::wip::onevpl::vpp_pp_params
#define GAPI_BACKEND_PP_SESSIONS        cv::gapi::wip::onevpl::vpp_pp_session
#else // VPP_PREPROC_ENGINE
struct empty_pp_params {};
struct empty_pp_session {};
#define GAPI_BACKEND_PP_PARAMS          cv::gapi::wip::empty_pp_params
#define GAPI_BACKEND_PP_SESSIONS        cv::gapi::wip::empty_pp_session
#endif // VPP_PREPROC_ENGINE

struct pp_params {
    using value_type = cv::util::variant<GAPI_BACKEND_PP_PARAMS>;

    template<typename BackendSpecificParamType, typename ...Args>
    static pp_params create(Args&& ...args) {
        static_assert(cv::detail::contains<BackendSpecificParamType, GAPI_BACKEND_PP_PARAMS>::value,
                      "Invalid BackendSpecificParamType requested");
        pp_params ret;
        ret.value = BackendSpecificParamType{std::forward<Args>(args)...};
        return ret;
    }

    template<typename BackendSpecificParamType>
    BackendSpecificParamType& get() {
        static_assert(cv::detail::contains<BackendSpecificParamType, GAPI_BACKEND_PP_PARAMS>::value,
                      "Invalid BackendSpecificParamType requested");
        return cv::util::get<BackendSpecificParamType>(value);
    }

    template<typename BackendSpecificParamType>
    const BackendSpecificParamType& get() const {
        return static_cast<const BackendSpecificParamType&>(const_cast<pp_params*>(this)->get<BackendSpecificParamType>());
    }
private:
    value_type value;
};

struct pp_session {
    using value_type = cv::util::variant<GAPI_BACKEND_PP_SESSIONS>;

    template<typename BackendSpecificSesionType, typename ...Args>
    static pp_session create(Args&& ...args) {
        static_assert(cv::detail::contains<BackendSpecificSesionType,
                                           GAPI_BACKEND_PP_SESSIONS>::value,
                      "Invalid BackendSpecificSesionType requested");
        pp_session ret;
        ret.value = BackendSpecificSesionType{std::forward<Args>(args)...};;
        return ret;
    }

    template<typename BackendSpecificSesionType>
    BackendSpecificSesionType &get() {
        static_assert(cv::detail::contains<BackendSpecificSesionType, GAPI_BACKEND_PP_SESSIONS>::value,
                      "Invalid BackendSpecificSesionType requested");
        return cv::util::get<BackendSpecificSesionType>(value);
    }

    template<typename BackendSpecificSesionType>
    const BackendSpecificSesionType &get() const {
        return const_cast<pp_session*>(this)->get<BackendSpecificSesionType>();
    }
private:
    value_type value;
};
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // GAPI_STREAMING_ONEVPL_ENGINE_PREPROC_DEFINES_HPP
