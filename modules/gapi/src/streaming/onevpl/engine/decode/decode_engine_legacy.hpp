// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONVPL_DECODE_ENGINE_LEGACY_HPP
#define GAPI_STREAMING_ONVPL_DECODE_ENGINE_LEGACY_HPP
#include <stdio.h>
#include <memory>

#include "streaming/onevpl/engine/processing_engine_base.hpp"

#ifdef HAVE_ONEVPL
#include "streaming/onevpl/onevpl_export.hpp"

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

class LegacyDecodeSession;
struct DecoderParams;
struct IDataProvider;
struct VPLAccelerationPolicy;

class GAPI_EXPORTS VPLLegacyDecodeEngine : public ProcessingEngineBase {
public:

    VPLLegacyDecodeEngine(std::unique_ptr<VPLAccelerationPolicy>&& accel);
    virtual session_ptr initialize_session(mfxSession mfx_session,
                                           const std::vector<CfgParam>& cfg_params,
                                           std::shared_ptr<IDataProvider> provider) override;
protected:
    struct SessionParam {
        void* decode_pool_key;
        DecoderParams decoder_params;
    };

    SessionParam prepare_session_param(mfxSession mfx_session,
                                       const std::vector<CfgParam>& cfg_params,
                                       std::shared_ptr<IDataProvider> provider);

    ExecutionStatus execute_op(operation_t& op, EngineSession& sess) override;
    ExecutionStatus process_error(mfxStatus status, LegacyDecodeSession& sess);

    void on_frame_ready(LegacyDecodeSession& sess,
                        mfxFrameSurface1* ready_surface);
    static void try_modify_pool_size_request_param(const char* param_name,
                                                   size_t new_frames_count,
                                                   mfxFrameAllocRequest& request);
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
#endif // GAPI_STREAMING_ONVPL_DECODE_ENGINE_LEGACY_HPP
