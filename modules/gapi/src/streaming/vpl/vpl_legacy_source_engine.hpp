// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_STREAMING_VPL_LEGACY_ENGINE_HPP
#define OPENCV_GAPI_STREAMING_VPL_LEGACY_ENGINE_HPP
#include <stdio.h>
#include <memory>

#include "streaming/engine/base_engine.hpp"
#include "streaming/vpl/vpl_accel_policy.hpp"
#ifdef HAVE_ONEVPL
#if (MFX_VERSION >= 2000)
    #include <vpl/mfxdispatcher.h>
#endif
#include <vpl/mfx.h>

namespace cv {
namespace gapi {
namespace wip {


class LegacyDecodeSession;
struct DecoderParams;
struct IDataProvider;

class VPLLegacyDecodeEngine : public VPLProcessingEngine {
public:

    VPLLegacyDecodeEngine(std::unique_ptr<VPLAccelerationPolicy>&& accel);
    void initialize_session(mfxSession mfx_session, DecoderParams&& decoder_param,
                            std::shared_ptr<IDataProvider> provider) override;

private:
    ExecutionStatus execute_op(operation_t& op, EngineSession& sess) override;
    ExecutionStatus process_error(mfxStatus status, LegacyDecodeSession& sess);

    void on_frame_ready(LegacyDecodeSession& sess);
};

class LegacyDecodeSession : public EngineSession {
public:
    friend class VPLLegacyDecodeEngine;
    
    LegacyDecodeSession(mfxSession sess, DecoderParams&& decoder_param, std::shared_ptr<IDataProvider> provider);
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
    mfxFrameSurface1* output_surface_ptr;

    int64_t decoded_frames_count;
};
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
#endif // OPENCV_GAPI_STREAMING_VPL_LEGACY_ENGINE_HPP
