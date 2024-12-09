// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONVPL_TRANSCODE_ENGINE_LEGACY_HPP
#define GAPI_STREAMING_ONVPL_TRANSCODE_ENGINE_LEGACY_HPP
#include <stdio.h>
#include <memory>

#include "streaming/onevpl/engine/decode/decode_engine_legacy.hpp"

#ifdef HAVE_ONEVPL
#include "streaming/onevpl/onevpl_export.hpp"

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

class LegacyTranscodeSession;
struct IDataProvider;
struct VPLAccelerationPolicy;

class GAPI_EXPORTS VPLLegacyTranscodeEngine : public VPLLegacyDecodeEngine {
public:

    VPLLegacyTranscodeEngine(std::unique_ptr<VPLAccelerationPolicy>&& accel);
    session_ptr initialize_session(mfxSession mfx_session,
                                   const std::vector<CfgParam>& cfg_params,
                                   std::shared_ptr<IDataProvider> provider) override;

    static std::map<std::string, mfxVariant> get_vpp_params(const std::vector<CfgParam> &cfg_params);
private:
    void on_frame_ready(LegacyTranscodeSession& sess,
                        mfxFrameSurface1* ready_surface);
    void validate_vpp_param(const mfxVideoParam& mfxVPPParams);
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
#endif // GAPI_STREAMING_ONVPL_DECODE_ENGINE_LEGACY_HPP
