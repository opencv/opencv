// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONVPL_ENGINE_TRANSCODE_SESSION_HPP
#define GAPI_STREAMING_ONVPL_ENGINE_TRANSCODE_SESSION_HPP

#ifdef HAVE_ONEVPL
#include "streaming/onevpl/engine/decode/decode_session.hpp"

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
class Surface;
class VPLLegacyTranscodeEngine;
class GAPI_EXPORTS LegacyTranscodeSession : public LegacyDecodeSession {
public:
    friend class VPLLegacyTranscodeEngine;

    LegacyTranscodeSession(mfxSession sess, DecoderParams&& decoder_param,
                           TranscoderParams&& transcoder_param,
                           std::shared_ptr<IDataProvider> provider);
    ~LegacyTranscodeSession();

    void init_transcode_surface_pool(VPLAccelerationPolicy::pool_key_t key);
    void swap_transcode_surface(VPLLegacyTranscodeEngine& engine);
    const mfxFrameInfo& get_video_param() const override;
private:
    mfxVideoParam mfx_transcoder_param;
    VPLAccelerationPolicy::pool_key_t vpp_out_pool_id;

    std::weak_ptr<Surface> vpp_surface_ptr;
    std::queue<op_handle_t> vpp_queue;
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
#endif // GAPI_STREAMING_ONVPL_ENGINE_TRANSCODE_SESSION_HPP
