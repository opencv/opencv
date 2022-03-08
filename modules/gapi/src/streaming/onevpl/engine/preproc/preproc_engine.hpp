// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2022 Intel Corporation

#ifndef GAPI_STREAMING_ONVPL_PREPROC_ENGINE_HPP
#define GAPI_STREAMING_ONVPL_PREPROC_ENGINE_HPP
#include <stdio.h>
#include <memory>
#include <unordered_map>

#include "streaming/onevpl/engine/processing_engine_base.hpp"
#include "streaming/onevpl/accelerators/utils/shared_lock.hpp"

#include "streaming/onevpl/engine/preproc_engine_interface.hpp"

#ifdef HAVE_ONEVPL
#include "streaming/onevpl/onevpl_export.hpp"

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
// GAPI_EXPORTS for tests
struct GAPI_EXPORTS FrameInfoComparator {
    bool operator()(const mfxFrameInfo& lhs, const mfxFrameInfo& rhs) const;
    static bool equal_to(const mfxFrameInfo& lhs, const mfxFrameInfo& rhs);
};

class VPPPreprocSession;
struct IDataProvider;
struct VPLAccelerationPolicy;

// GAPI_EXPORTS for tests
class GAPI_EXPORTS VPPPreprocEngine final : public ProcessingEngineBase,
                                            public cv::gapi::wip::IPreprocEngine {
public:
    using session_type     = VPPPreprocSession;
    using session_ptr_type = std::shared_ptr<session_type>;

    VPPPreprocEngine(std::unique_ptr<VPLAccelerationPolicy>&& accel);

    cv::util::optional<pp_params> is_applicable(const cv::MediaFrame& in_frame) override;

    pp_session initialize_preproc(const pp_params& initial_frame_param,
                                  const GFrameDesc& required_frame_descr) override;

    cv::MediaFrame run_sync(const pp_session &session_handle,
                            const cv::MediaFrame& in_frame,
                            const cv::util::optional<cv::Rect> &opt_roi) override;

private:
    std::map<mfxFrameInfo, session_ptr_type, FrameInfoComparator> preproc_session_map;
    void on_frame_ready(session_type& sess,
                        mfxFrameSurface1* ready_surface);
    ExecutionStatus process_error(mfxStatus status, session_type& sess);
    session_ptr initialize_session(mfxSession mfx_session,
                                   const std::vector<CfgParam>& cfg_params,
                                   std::shared_ptr<IDataProvider> provider) override;
    size_t preprocessed_frames_count;
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
#endif // GAPI_STREAMING_ONVPL_PREPROC_ENGINE_HPP
