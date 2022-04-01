// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2022 Intel Corporation

#include <algorithm>
#include <exception>

#include <opencv2/gapi/streaming/onevpl/data_provider_interface.hpp>
#include "streaming/onevpl/engine/preproc/preproc_dispatcher.hpp"

#ifdef HAVE_ONEVPL
#include "streaming/onevpl/onevpl_export.hpp"
#include "streaming/onevpl/engine/preproc/preproc_engine.hpp"
#include "streaming/onevpl/engine/preproc/preproc_session.hpp"

#include "streaming/onevpl/accelerators/accel_policy_interface.hpp"
#include "streaming/onevpl/accelerators/surface/surface.hpp"
#include "streaming/onevpl/cfg_params_parser.hpp"
#endif // HAVE_ONEVPL

#include "logger.hpp"

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
#ifdef HAVE_ONEVPL
cv::util::optional<pp_params> VPPPreprocDispatcher::is_applicable(const cv::MediaFrame& in_frame) {
    cv::util::optional<pp_params> param;
    GAPI_LOG_DEBUG(nullptr, "workers: " << workers.size());
    bool worker_found = false;
    for (const auto &w : workers) {
        param = w->is_applicable(in_frame);
        if (param.has_value()) {
            auto &vpp_param = param.value().get<vpp_pp_params>();
            BaseFrameAdapter* adapter = reinterpret_cast<BaseFrameAdapter*>(vpp_param.reserved);
            const IDeviceSelector::DeviceScoreTable &devs =
                            (std::static_pointer_cast<VPPPreprocEngine>(w))->get_accel()->get_device_selector()->select_devices();
            GAPI_DbgAssert(devs.size() >= 1 && "Invalid device selector");
            auto worker_accel_type = std::get<1>(*devs.begin()).get_type();
            GAPI_LOG_DEBUG(nullptr, "acceleration types for frame: " << to_cstring(adapter->accel_type()) <<
                           ", for worker: " << to_cstring(worker_accel_type));
            if (worker_accel_type == adapter->accel_type()){
                vpp_param.reserved = reinterpret_cast<void *>(w.get());
                GAPI_LOG_DEBUG(nullptr, "selected worker: " << vpp_param.reserved);
                worker_found = true;
                break;
            }
        }
    }
    return worker_found ? param : cv::util::optional<pp_params>{};
}

pp_session VPPPreprocDispatcher::initialize_preproc(const pp_params& initial_frame_param,
                                                    const GFrameDesc& required_frame_descr) {
    const auto &vpp_param = initial_frame_param.get<vpp_pp_params>();
    GAPI_LOG_DEBUG(nullptr, "workers: " << workers.size());
    for (auto &w : workers) {
        if (reinterpret_cast<void*>(w.get()) == vpp_param.reserved) {
            pp_session sess = w->initialize_preproc(initial_frame_param, required_frame_descr);
            vpp_pp_session &vpp_sess = sess.get<vpp_pp_session>();
            vpp_sess.reserved = reinterpret_cast<void *>(w.get());
            GAPI_LOG_DEBUG(nullptr, "initialized session preproc for worker: " << vpp_sess.reserved);
            return sess;
        }
    }
    GAPI_Assert(false && "Cannot initialize VPP preproc in dispatcher, no suitable worker");
}

cv::MediaFrame VPPPreprocDispatcher::run_sync(const pp_session &session_handle,
                                              const cv::MediaFrame& in_frame,
                                              const cv::util::optional<cv::Rect> &opt_roi) {
    const auto &vpp_sess = session_handle.get<vpp_pp_session>();
    GAPI_LOG_DEBUG(nullptr, "workers: " << workers.size());
    for (auto &w : workers) {
        if (reinterpret_cast<void*>(w.get()) == vpp_sess.reserved) {
            GAPI_LOG_DEBUG(nullptr, "trigger execution on worker: " << vpp_sess.reserved);
            return w->run_sync(session_handle, in_frame, opt_roi);
        }
    }
    GAPI_Assert(false && "Cannot invoke VPP preproc in dispatcher, no suitable worker");
}

#else // HAVE_ONEVPL
cv::util::optional<pp_params> VPPPreprocDispatcher::is_applicable(const cv::MediaFrame&) {
    return cv::util::optional<pp_params>{};
}

pp_session VPPPreprocDispatcher::initialize_preproc(const pp_params&,
                                                    const GFrameDesc&) {
    GAPI_Assert(false && "Unsupported: G-API compiled without `WITH_GAPI_ONEVPL=ON`");
}

cv::MediaFrame VPPPreprocDispatcher::run_sync(const pp_session &,
                                              const cv::MediaFrame&,
                                              const cv::util::optional<cv::Rect> &) {
    GAPI_Assert(false && "Unsupported: G-API compiled without `WITH_GAPI_ONEVPL=ON`");
}
#endif // HAVE_ONEVPL
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
