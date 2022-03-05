// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2022 Intel Corporation

#ifndef GAPI_STREAMING_ONVPL_PREPROC_SESSION_HPP
#define GAPI_STREAMING_ONVPL_PREPROC_SESSION_HPP
#include <memory>
#include <queue>

#include <opencv2/gapi/streaming/meta.hpp>
#include "streaming/onevpl/engine/engine_session.hpp"
#include "streaming/onevpl/accelerators/accel_policy_interface.hpp"
#include "streaming/onevpl/engine/preproc/vpp_preproc_defines.hpp"

#ifdef HAVE_ONEVPL

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
class VPPPreprocEngine;

class VPPPreprocSession : public EngineSession {
public:
    friend class VPPPreprocEngine;
    VPPPreprocSession(mfxSession sess, const mfxVideoParam &vpp_out_param);
    ~VPPPreprocSession();

    Data::Meta generate_frame_meta();
    void swap_surface(VPPPreprocEngine& engine);
    void init_surface_pool(VPLAccelerationPolicy::pool_key_t key);

    virtual const mfxFrameInfo& get_video_param() const override;
private:
    mfxVideoParam mfx_vpp_out_param;
    VPLAccelerationPolicy::pool_key_t vpp_pool_id;
    std::weak_ptr<Surface> processing_surface_ptr;

    struct incoming_task {
        mfxSyncPoint sync_handle;
        mfxFrameSurface1* decoded_surface_ptr;
        Surface::info_t decoded_frame_info;
        cv::MediaFrame decoded_frame_copy;
        cv::util::optional<cv::Rect> roi;
    };

    struct outgoing_task {
        outgoing_task() = default;
        outgoing_task(mfxSyncPoint acquired_sync_handle,
                      mfxFrameSurface1* acquired_surface_ptr,
                      incoming_task &&in);
        mfxSyncPoint sync_handle;
        mfxFrameSurface1* vpp_surface_ptr;

        mfxFrameSurface1* original_surface_ptr;
        void release_frame();
    private:
        Surface::info_t original_frame_info;
        cv::MediaFrame original_frame;
    };

    std::queue<incoming_task> sync_in_queue;
    std::queue<outgoing_task> vpp_out_queue;
    int64_t preprocessed_frames_count;
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
#endif // GAPI_STREAMING_ONVPL_PREPROC_SESSION_HPP
