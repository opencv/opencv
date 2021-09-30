// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONVPL_ENGINE_DECODE_DECODE_SESSION_HPP
#define GAPI_STREAMING_ONVPL_ENGINE_DECODE_DECODE_SESSION_HPP
#include <stdio.h>
#include <memory>
#include <queue>

#include <opencv2/gapi/streaming/meta.hpp>

#include "streaming/onevpl/engine/engine_session.hpp"
#include "streaming/onevpl/accelerators/accel_policy_interface.hpp"
#ifdef HAVE_ONEVPL
#if (MFX_VERSION >= 2000)
    #include <vpl/mfxdispatcher.h>
#endif
#include <vpl/mfx.h>

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

struct IDataProvider;
class Surface;
struct VPLAccelerationPolicy;

class LegacyDecodeSession : public EngineSession {
public:
    friend class VPLLegacyDecodeEngine;

    LegacyDecodeSession(mfxSession sess, DecoderParams&& decoder_param, std::shared_ptr<IDataProvider> provider);
    ~LegacyDecodeSession();
    using EngineSession::EngineSession;

    void swap_surface(VPLLegacyDecodeEngine& engine);
    void init_surface_pool(VPLAccelerationPolicy::pool_key_t key);

    mfxVideoParam mfx_decoder_param;
    std::shared_ptr<IDataProvider> data_provider;

    Data::Meta generate_frame_meta();
private:
    VPLAccelerationPolicy::pool_key_t decoder_pool_id;
    mfxFrameAllocRequest request;

    std::weak_ptr<Surface> procesing_surface_ptr;

    using op_handle_t = std::pair<mfxSyncPoint, mfxFrameSurface1*>;
    std::queue<op_handle_t> sync_queue;

    int64_t decoded_frames_count;
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
#endif // GAPI_STREAMING_ONVPL_ENGINE_DECODE_DECODE_SESSION_HPP
