#ifndef OPENCV_GAPI_STREAMING_VPL_LEGACY_ENGINE_HPP
#define OPENCV_GAPI_STREAMING_VPL_LEGACY_ENGINE_HPP
#include <stdio.h>
#include <memory>

#include "streaming/engine/base_engine.hpp"
#include "streaming/vpl/vpl_accel_policy.hpp"
#ifdef HAVE_ONEVPL
#include <vpl/mfxvideo.h>

namespace cv {
namespace gapi {
namespace wip {


class LegacyDecodeSession;
struct DecoderParams;

class VPLLegacyDecodeEngine : public VPLProcessingEngine {
public:

    VPLLegacyDecodeEngine();
    void initialize_session(mfxSession mfx_session, DecoderParams&& decoder_param,
                            file_ptr&& source_handle,
                            std::unique_ptr<VPLAccelerationPolicy>&& acceleration_policy);

private:
    ExecutionStatus execute_op(operation_t& op, EngineSession& sess) override;
    ExecutionStatus process_error(mfxStatus status, LegacyDecodeSession& sess);

    void on_frame_ready(LegacyDecodeSession& sess);
};

class LegacyDecodeSession : public EngineSession {
public:
    friend class VPLLegacyDecodeEngine;
    using file_ptr = std::unique_ptr<FILE, decltype(&fclose)>;
    
    LegacyDecodeSession(mfxSession sess, DecoderParams&& decoder_param, file_ptr&& source);
    using EngineSession::EngineSession;

    void swap_surface();
    void init_surface_pool(VPLAccelerationPolicy::pool_key_t key);
    
    mfxVideoParam mfx_decoder_param;
    VPLLegacyDecodeEngine::file_ptr source_handle;
    bool stop_processing;
private:
    std::vector<std::shared_ptr<mfxFrameSurface1>> decoder_surf_pool;
    std::unique_ptr<VPLAccelerationPolicy> acceleration_policy;
    VPLAccelerationPolicy::pool_key_t decoder_pool_id;
    mfxFrameAllocRequest request;

    VPLAccelerationPolicy::surface_weak_ptr_t procesing_surface_ptr;
    VPLAccelerationPolicy::surface_raw_ptr_t output_surface_ptr;
};
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
#endif // OPENCV_GAPI_STREAMING_VPL_LEGACY_ENGINE_HPP
