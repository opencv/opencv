// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_ENGINE_ENGINE_SESSION_HPP
#define GAPI_STREAMING_ONEVPL_ENGINE_ENGINE_SESSION_HPP

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "opencv2/gapi/util/optional.hpp"
#include "opencv2/gapi/own/exports.hpp" // GAPI_EXPORTS
#include <opencv2/gapi/streaming/onevpl/data_provider_interface.hpp>
#include "streaming/onevpl/data_provider_defines.hpp"
#include "streaming/onevpl/accelerators/accel_policy_interface.hpp"

#ifdef HAVE_ONEVPL
#include "streaming/onevpl/onevpl_export.hpp"

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

// GAPI_EXPORTS for tests
struct GAPI_EXPORTS DecoderParams {
    std::shared_ptr<IDataProvider::mfx_bitstream> stream;
    mfxVideoParam param;
    cv::util::optional<size_t> preallocated_frames_count;
};

struct GAPI_EXPORTS TranscoderParams {
    mfxVideoParam param;
};

struct GAPI_EXPORTS EngineSession {
    mfxSession session;
    mfxStatus last_status;

    EngineSession(mfxSession sess);
    std::string error_code_to_str() const;
    virtual ~EngineSession();

    virtual const mfxFrameInfo& get_video_param() const = 0;

    static void request_free_surface(mfxSession session,
                                     VPLAccelerationPolicy::pool_key_t key,
                                     VPLAccelerationPolicy &acceleration_policy,
                                     std::weak_ptr<Surface> &surface_to_exchange,
                                     bool reset_if_not_found = false);
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_ONEVPL
#endif // GAPI_STREAMING_ONEVPL_ENGINE_ENGINE_SESSION_HPP
