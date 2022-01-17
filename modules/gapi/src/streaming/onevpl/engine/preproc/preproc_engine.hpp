// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONVPL_PREPROC_ENGINE_HPP
#define GAPI_STREAMING_ONVPL_PREPROC_ENGINE_HPP
#include <stdio.h>
#include <memory>
#include <unordered_map>

#include "streaming/onevpl/engine/processing_engine_base.hpp"
#include "streaming/onevpl/engine/preproc/preproc_session.hpp"
#include "streaming/onevpl/accelerators/utils/shared_lock.hpp"

#ifdef HAVE_ONEVPL
#include "streaming/onevpl/onevpl_export.hpp"

#ifdef HAVE_INF_ENGINE
#include <inference_engine.hpp>

bool operator< (const mfxFrameInfo &lhs, const mfxFrameInfo &rhs);

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

class PreprocSession;
struct IDataProvider;
struct VPLAccelerationPolicy;

class GAPI_EXPORTS VPPPreprocEngine : public ProcessingEngineBase {
public:
    VPPPreprocEngine(std::unique_ptr<VPLAccelerationPolicy>&& accel);

    static cv::util::optional<PreprocParams> is_applicable(const cv::MediaFrame& in_frame);

    std::shared_ptr<PreprocSession>
            initialize_preproc(const PreprocParams& params,
                               const InferenceEngine::InputInfo::CPtr& net_input);

    session_ptr initialize_session(mfxSession mfx_session,
                                   const std::vector<CfgParam>& cfg_params,
                                   std::shared_ptr<IDataProvider> provider) override;

    cv::MediaFrame run_sync(std::shared_ptr<PreprocSession> s, const cv::MediaFrame& in_frame);
private:
    std::map<mfxFrameInfo, std::shared_ptr<PreprocSession>> preproc_session_map;
    void on_frame_ready(PreprocSession& sess,
                        mfxFrameSurface1* ready_surface);
    ExecutionStatus process_error(mfxStatus status, PreprocSession& sess);
    size_t preprocessed_frames_count;

    // NB: no nee to protect by mutex at now
    using decoded_frame_key_t = void*;
    std::unordered_map<decoded_frame_key_t, cv::MediaFrame> pending_decoded_frames_sync;

    void abandon_decode_frame(decoded_frame_key_t key);
    void remember_decode_frame(decoded_frame_key_t key, const cv::MediaFrame& in_frame);
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
#endif // HAVE_INF_ENGINE
#endif // GAPI_STREAMING_ONVPL_PREPROC_ENGINE_HPP
